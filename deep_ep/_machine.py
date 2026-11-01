"""Select a Buffer backend once, before SHMEM configuration or allocation.

Topology files describe connectivity, not health or achieved bandwidth. Unknown
products do not silently select an algorithm. Physical IDs are never EP ranks.
"""

import ctypes
import hashlib
import json
import inspect
import subprocess
from pathlib import Path
import sys

if __package__:
    from ._backend_lock import LOCK
else:
    # Standalone DCMI helper: do not import the package or Torch.
    from threading import RLock

    LOCK = RLock()

KINDS = ("auto", "pod", "clos_only", "card_4pmesh")
# Only the migrated implementation takes an engine argument internally. POD
# retains its own initialization contract and mixed-engine configuration.
_TRANSPORT_BY_MACHINE = {"clos_only": "udma", "card_4pmesh": "mte"}


def read_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("topology exceeds 16 MiB")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def classify_root(root, topology, physical):
    matches = [r for r in root.get("rank_list", []) if str(r.get("device_id")) == str(physical)]
    if len(matches) != 1:
        raise ValueError("rootinfo must contain exactly one current physical device")
    rank = matches[0]
    local_id = str(rank["local_id"])
    levels = rank.get("level_list", [])
    if not levels or any(not level.get("rank_addr_list") for level in levels):
        raise ValueError("rootinfo has no usable endpoint records")
    device_levels = []
    for level in levels:
        if level.get("net_type") not in ("MESH", "CLOS", "TOPO_FILE_DESC"):
            raise ValueError("unsupported rootinfo network type")
        device_endpoints = []
        for endpoint in level["rank_addr_list"]:
            # Card rootinfo can also describe a host NIC with an IP/d2h port.
            # It is not evidence of an NPU Clos link or MTE reachability.
            if endpoint.get("ports") == ["d2h"]:
                continue
            addr = endpoint.get("addr", "")
            if endpoint.get("addr_type") != "EID" or len(addr) != 32 or len(bytes.fromhex(addr)) != 16:
                raise ValueError("rootinfo endpoint must contain a 128-bit EID")
            if not endpoint.get("ports"):
                raise ValueError("rootinfo endpoint has no ports")
            device_endpoints.append(endpoint)
        if device_endpoints:
            device_levels.append(level)
    if not device_levels:
        raise ValueError("rootinfo has no usable NPU endpoint records")
    edges = topology.get("edge_list", [])
    relevant = [e for e in edges if local_id in (str(e.get("local_a")), str(e.get("local_b")))]
    # Exclude host PCIe declarations. They are not NPU-to-NPU Clos links.
    relevant = [e for e in relevant if any("/" in p for p in e.get("local_a_ports", []))]
    if not relevant:
        raise ValueError("topology does not describe current device NPU ports")
    mesh = any("local_b" in e and e.get("topo_type") != "CLOS" for e in relevant)
    clos = any(e.get("topo_type") == "CLOS" or "local_b" not in e for e in relevant)
    declared_mesh = any(level.get("net_type") == "MESH" for level in device_levels)
    declared_clos = any(level.get("net_type") == "CLOS" for level in device_levels)
    descriptor = any(level.get("net_type") == "TOPO_FILE_DESC" for level in device_levels)
    if (
        (declared_mesh and not mesh)
        or (declared_clos and not clos)
        or (not descriptor and (mesh != declared_mesh or clos != declared_clos))
    ):
        raise ValueError("rootinfo and topology disagree about current-device fabrics")
    if clos and not mesh:
        return "clos_only"
    # A mesh+Clos declaration alone is insufficient to identify a supported POD.
    if clos and mesh:
        return "mesh_and_clos"
    if mesh:
        # A four-card product must describe a complete four-vertex Mesh domain.
        # Never infer the domain from physical_id // 4 or EP rank numbering.
        neighbors = {}
        for edge in edges:
            if "local_b" not in edge or edge.get("topo_type") == "CLOS":
                continue
            if not any("/" in p for p in edge.get("local_a_ports", [])):
                continue
            a, b = str(edge["local_a"]), str(edge["local_b"])
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
        domain = {local_id} | neighbors.get(local_id, set())
        if len(domain) == 4 and all(neighbors.get(node) == domain - {node} for node in domain):
            return "mesh4"
    raise ValueError("unsupported or incomplete NPU topology")


def physical_device():
    from ._native import npu_api

    device = npu_api().current_device()
    runtime = ctypes.CDLL("libascendcl.so")
    physical = ctypes.c_int32()
    for name in ("aclrtGetPhyDevIdByUserDevId", "aclrtGetPhyDevIdByLogicDevId"):
        fn = getattr(runtime, name, None)
        if fn is not None:
            fn.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
            fn.restype = ctypes.c_int32
            code = fn(device, ctypes.byref(physical))
            if code != 0:
                raise RuntimeError(f"{name} returned {code}")
            return physical.value
    raise RuntimeError("physical device mapping unavailable; specify machine_type explicitly")


def _board_id(physical):
    dcmi = ctypes.CDLL("libdcmi.so")
    if dcmi.dcmiv2_init() != 0:
        raise RuntimeError("DCMI initialization failed")
    logical, board = ctypes.c_uint32(), ctypes.c_uint32()
    mapping = getattr(dcmi, "dcmiv2_get_dev_id_by_chip_phy_id", None)
    if mapping is None:
        mapping = dcmi.dcmiv2_get_dev_id_from_chip_phyid
    mapping.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    mapping.restype = ctypes.c_int
    fn = dcmi.dcmiv2_get_mainboard_id
    fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
    fn.restype = ctypes.c_int
    if mapping(physical, ctypes.byref(logical)) != 0 or fn(logical.value, ctypes.byref(board)) != 0:
        raise RuntimeError("DCMI board query failed")
    return board.value


def board_id(physical):
    # Driver calls can block. A spawned CPU-only helper bounds the query without
    # forking an initialized Torch/NPU process or changing its device context.
    query = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--board", str(physical)],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return int(query.stdout.strip())


def detect(rootinfo_path=None):
    return detect_physical(physical_device(), rootinfo_path)


def detect_physical(physical, rootinfo_path=None):
    """Use the runtime classification rules without selecting/initializing an NPU.

    The caller supplies a physical chip ID, not a framework-visible device index
    or EP rank. File reads and bounded DCMI management queries are the only probes.
    """
    if isinstance(physical, bool) or not isinstance(physical, int) or not 0 <= physical < 2**32:
        raise ValueError("physical device must be an unsigned 32-bit chip ID")
    path = Path(rootinfo_path or "/etc/hccl_rootinfo.json")
    if path.exists():
        root, root_hash = read_json(path)
        topo_path = Path(root["topo_file_path"])
        topo, topo_hash = read_json(topo_path)
        kind = classify_root(root, topo, physical)
        if kind == "mesh_and_clos":
            if board_id(physical) not in (3, 7):
                raise ValueError("mesh+Clos product is not a validated POD; specify machine_type")
            kind = "pod"
        elif kind == "mesh4":
            if board_id(physical) != 0x6C:
                raise ValueError("four-card Mesh requires CARD_4PMESH board 0x6c")
            kind = "card_4pmesh"
        return {
            "machine_type": kind,
            "source": "rootinfo",
            "physical_device": physical,
            "rootinfo_sha256": root_hash,
            "topology_sha256": topo_hash,
        }
    if rootinfo_path is not None:
        raise ValueError(f"explicit rootinfo is missing: {path}")
    board = board_id(physical)
    if board not in (3, 7, 0x6C):
        raise ValueError(f"no rootinfo and no validated rule for board 0x{board:02x}; specify machine_type")
    return {
        "machine_type": "card_4pmesh" if board == 0x6C else "pod",
        "source": "dcmi_board_rule",
        "physical_device": physical,
        "board": board,
    }


def gather_records(group, record):
    """Object collective scoped to the supplied EP group, including MPI."""
    import torch.distributed as dist

    if isinstance(group, dist.ProcessGroup):
        records = [None] * dist.get_world_size(group)
        dist.all_gather_object(records, record, group=group)
        return records
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is None or not isinstance(group, mpi.Comm):
        raise TypeError("machine selection requires an EP ProcessGroup or MPI communicator")
    return group.allgather(record)


def select(machine_type="auto", group=None, rootinfo_path=None, *, transport_provided=False):
    error, result = None, None
    try:
        if transport_provided:
            raise TypeError("transport is selected by machine_type; remove the transport argument")
        if machine_type not in KINDS:
            raise ValueError(f"machine_type must be one of {KINDS}")
        result = (
            detect(rootinfo_path) if machine_type == "auto" else {"machine_type": machine_type, "source": "explicit"}
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    record = {"selection": result, "error": error}
    if group is not None:
        records = gather_records(group, record)
        failed = [(i, r["error"]) for i, r in enumerate(records) if r["error"]]
        if failed:
            raise RuntimeError(f"machine selection failed collectively: {failed}")
        if len({r["selection"]["machine_type"] for r in records}) != 1:
            raise ValueError("EP ranks selected different machine backends")
    if error:
        raise RuntimeError(error)
    return result


class MachineSelector(type):
    """Keep the existing POD class and methods; select Clos before construction."""

    def __call__(cls, *args, **kwargs):
        sys.modules[cls.__module__].require_native(f"{cls.__name__}.__init__")
        options = dict(kwargs)
        machine_type = options.pop("machine_type", "auto")
        rootinfo = options.pop("rootinfo_path", None)
        group = args[0] if args else options.get("group")
        if group is None:
            group = options.get("comm")
        # The target's explicit-rank MPI path has no ProcessGroup. Native
        # SHMEM bootstrap remains responsible for its collective configuration.
        # Reject the removed public option in the same collective as topology
        # selection, so a rank-local old argument cannot strand the other ranks
        # inside SHMEM initialization. Even None/auto/matching values are errors.
        selection = select(machine_type, group, rootinfo, transport_provided="transport" in options)
        with LOCK:
            if selection["machine_type"] in ("clos_only", "card_4pmesh"):
                from ._clos_adapter import buffer_class

                if selection["machine_type"] == "card_4pmesh":
                    from ._card_adapter import buffer_class as card_class, prepare

                    prepare(group, options)
                    implementation = card_class(cls.__name__)
                else:
                    implementation = buffer_class(cls.__name__)
                # Explicit internal values override legacy environment defaults.
                options["transport"] = _TRANSPORT_BY_MACHINE[selection["machine_type"]]
                result = implementation(*args, **options)
            else:
                # The constructor's compatibility wrapper handles both naming
                # styles; selection does not reinterpret values or defaults.
                result = super().__call__(*args, **options)
        result.machine_type = selection["machine_type"]
        result.machine_selection = selection
        return result

    @property
    def __signature__(cls):
        signature = inspect.signature(cls.__init__)
        return signature.replace(parameters=list(signature.parameters.values())[1:])

    def __instancecheck__(cls, instance):
        for name in ("deep_ep._clos_adapter", "deep_ep._card_adapter"):
            module = sys.modules.get(name)
            if module is not None and type(instance) is module.buffer_class(cls.__name__):
                return True
        return super().__instancecheck__(instance)


class HandleSelector(type):
    def __instancecheck__(cls, instance):
        module = sys.modules.get("deep_ep.backends.clos.elastic")
        if module is not None and isinstance(instance, module.EPHandle):
            return True
        return super().__instancecheck__(instance)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--board":
        raise SystemExit("expected --board PHYSICAL_ID")
    print(_board_id(int(sys.argv[2])))
