# Copyright (c) 2026, Lu Lu

"""Shared V1/V2 resource owner; no EP algorithms are enabled by this module."""

import hashlib
import ipaddress
import os
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path
from uuid import UUID

from ._native import npu_api, require_native
from ._store import StoreGroup

from ._backend_lock import LOCK as _LOCK

_OWNER = None


def integer(name, value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def endpoint():
    value = os.environ.get("DEEPEP_SHMEM_ENDPOINT", "")
    if not value.startswith("tcp://") or len(value) >= 64:
        raise ValueError("Set DEEPEP_SHMEM_ENDPOINT=tcp://<IPv4>:<unused-port>")
    host, separator, port = value[6:].partition(":")
    if not separator or not port.isdecimal() or not 1024 <= int(port) <= 65535:
        raise ValueError("DEEPEP_SHMEM_ENDPOINT requires an unprivileged TCP port")
    ipaddress.IPv4Address(host)
    return value


def _host_boot_id():
    # hostname 可能在不同机器上重复；Linux boot_id 在本机各进程间一致。
    try:
        return UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()).hex
    except (OSError, ValueError):
        # 先参加配置交换，让所有 rank 一起失败，不能退回不可靠的 hostname。
        return ""


def rank_and_world(group, rank=None, world_size=None):
    """Resolve an explicit SHMEM identity or an existing framework group."""
    if group is None:
        from .buffers._elastic_layout import WORLD_SIZES

        integer("world_size", world_size, 2, 512)
        if world_size not in WORLD_SIZES:
            raise ValueError("Runtime supports EP2/4/8/16/32/64/128/256/512")
        integer("rank", rank, 0, world_size - 1)
        return rank, world_size
    if rank is not None or world_size is not None:
        raise ValueError("Pass either group or rank/world_size, not both")
    import torch.distributed as dist

    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be an initialized torch.distributed ProcessGroup")
    return dist.get_rank(group), dist.get_world_size(group)


class Runtime:
    """One explicit collective lifetime per process; a live owner is retained."""

    _megamoe = False

    def __init__(
        self,
        group,
        capacity,
        timeout=120,
        *,
        udma=False,
        rank=None,
        world_size=None,
        bootstrap="default",
        combine_only=False,
        use_1g_huge_pages=False,
        debug_collectives=False,
    ):
        global _OWNER
        native = require_native("Runtime.__init__")
        self.ll_backend = os.environ.get("DEEPEP_LL_BACKEND", "mte")
        if self.ll_backend not in ("mte", "rdma"):
            raise ValueError("DEEPEP_LL_BACKEND must be mte or rdma")
        integer("capacity", capacity, 2 << 20, (1 << 30) if self._megamoe else (1 << 63) - 1)
        if capacity % (2 << 20):
            raise ValueError("capacity must be aligned to 2 MiB")
        integer("timeout", timeout, 1, 3600)
        if bootstrap not in ("default", "mpi"):
            raise ValueError("bootstrap must be default or mpi")
        if bootstrap == "mpi":
            if group is not None or not udma or self._megamoe:
                raise ValueError("MPI bootstrap requires explicit rank/world_size and the Elastic UDMA runtime")
            if not callable(getattr(native, "create_mpi_runtime", None)):
                raise RuntimeError("Rebuild/install DeepEP with MPI bootstrap support")
            try:
                mpi_rank, mpi_size = (int(os.environ[name]) for name in ("PMI_RANK", "PMI_SIZE"))
            except (KeyError, ValueError) as exc:
                raise ValueError("MPI bootstrap requires an MPICH launcher with PMI_RANK/PMI_SIZE") from exc
            if (rank, world_size) != (mpi_rank, mpi_size):
                raise ValueError("Explicit rank/world_size disagree with MPI_COMM_WORLD")
        address = "" if bootstrap == "mpi" else endpoint()
        if type(combine_only) is not bool or (combine_only and not udma):
            raise ValueError("combine_only requires a UDMA runtime and a bool value")
        self.combine_only = combine_only
        if type(use_1g_huge_pages) is not bool or (use_1g_huge_pages and (not udma or self._megamoe)):
            raise ValueError("use_1g_huge_pages requires an Elastic UDMA runtime and a bool value")
        self.use_1g_huge_pages = use_1g_huge_pages
        if type(debug_collectives) is not bool:
            raise TypeError("debug_collectives must be bool")
        if debug_collectives and group is None:
            raise ValueError("debug_collectives=True requires a ProcessGroup")
        self.debug_collectives = debug_collectives
        dense_qps = os.environ.get("DEEPEP_SHMEM_DENSE_QPS", "0")
        if dense_qps not in ("0", "1"):
            raise ValueError("DEEPEP_SHMEM_DENSE_QPS must be 0 or 1")
        self.bootstrap = bootstrap
        store_group = self._megamoe and isinstance(group, StoreGroup)
        if self._megamoe and (group is None or udma):
            raise ValueError("MegaMoE requires a ProcessGroup or StoreGroup and the MTE runtime")
        if store_group:
            if rank is not None or world_size is not None:
                raise ValueError("Pass either group or rank/world_size, not both")
            rank, size = group.rank(), group.size()
        else:
            rank, size = rank_and_world(group, rank, world_size)
        factory = native.Runtime
        if self._megamoe:
            integer("world_size", size, 2, 128)
            factory = getattr(native, "create_megamoe_runtime", None)
            if factory is None:
                raise RuntimeError("Rebuild with DEEPEP_BUILD_MEGAMOE=ON")
        else:
            from .buffers._elastic_layout import WORLD_SIZES

            if size not in WORLD_SIZES:
                raise ValueError("Runtime supports EP2/4/8/16/32/64/128/256/512")
        if group is not None and not store_group:
            import torch
            import torch.distributed as dist

            backend = str(dist.get_backend(group))
            if backend not in {"gloo", "hccl"}:
                raise NotImplementedError("Runtime bootstrap supports Gloo or HCCL process groups")
        npu = npu_api()
        self.device = npu.current_device()
        self.group = group
        self.host_coordinated = self.ll_backend == "rdma"
        self.coordination_failed = False
        if self.host_coordinated and (group is None or store_group or backend != "hccl"):
            raise ValueError("NDR requires the existing HCCL ProcessGroup for Host coordination")
        self.low_latency_comm_args = None
        self.closed = False
        with _LOCK:
            busy = _OWNER is not None or native.runtime_busy()

            if group is None:
                if busy:
                    raise RuntimeError("This process already owns a SHMEM runtime")
            else:

                def digest(value):
                    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:7], "little")

                local_values = [capacity, timeout, digest(address), int(busy)]
                if self._megamoe:
                    host_id = _host_boot_id()
                    local_values.extend(
                        [
                            digest(host_id) if host_id else 0,
                            digest(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")),
                            self.device,
                        ]
                    )
                local_values.append(
                    int(udma)
                    + 2 * int(combine_only)
                    + 4 * int(dense_qps)
                    + 8 * int(use_1g_huge_pages)
                    + 16 * int(debug_collectives)
                    + 32 * int(self.ll_backend == "rdma")
                )
                if store_group:
                    values = group.all_gather(local_values, "runtime configuration")
                else:
                    with torch.inference_mode(False), torch.no_grad():
                        local = torch.tensor(
                            local_values, dtype=torch.int64, device=f"npu:{self.device}" if backend == "hccl" else "cpu"
                        )
                        peers = [torch.empty_like(local) for _ in range(size)]
                    if os.environ.get("DEEPEP_LL_INIT_TRACE") == "1":
                        print(f"[DeepEP init] rank={rank} runtime configuration all_gather enter", flush=True)
                    dist.all_gather(peers, local, group=group)
                    values = [value.cpu().tolist() for value in peers]
                    if os.environ.get("DEEPEP_LL_INIT_TRACE") == "1":
                        print(f"[DeepEP init] rank={rank} runtime configuration all_gather return", flush=True)
                if any(value[3] for value in values):
                    raise RuntimeError("A rank already owns a SHMEM runtime; destroy it collectively first")
                if any(value[:3] != values[0][:3] for value in values):
                    raise ValueError("Ranks disagree on workspace capacity, timeout or bootstrap endpoint")
                if any(value[-1] != values[0][-1] for value in values):
                    raise ValueError(
                        "Ranks disagree on MTE/URMA/RDMA runtime engines, QP configuration, page size or debug_collectives"
                    )
                if self._megamoe:
                    invalid_hosts = [i for i, value in enumerate(values) if value[4] == 0]
                    if invalid_hosts:
                        raise RuntimeError(
                            f"Cannot read a valid Linux /proc/sys/kernel/random/boot_id on group ranks {invalid_hosts}; "
                            "host identity is required before SHMEM initialization"
                        )
                    hosts = {value[4] for value in values}
                    for host in hosts:
                        local = [value for value in values if value[4] == host]
                        if len({value[5] for value in local}) != 1:
                            raise ValueError("Ranks on one host must use a common visible-device mapping")
                    devices = {}
                    for peer_rank, value in enumerate(values):
                        key = (value[4], value[6])
                        if key in devices:
                            raise ValueError(
                                f"Each rank must use a distinct local NPU: group ranks {devices[key]} and "
                                f"{peer_rank} use local device {value[6]} on the same Linux host"
                            )
                        devices[key] = peer_rank
                if self.host_coordinated:
                    topology = [None] * size
                    mapping = tuple(os.environ.get(key, "") for key in (
                        "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
                        "CUDA_VISIBLE_DEVICES", "PROBE_VISIBLE_DEVICES"))
                    dist.all_gather_object(topology, (_host_boot_id(), mapping, self.device), group=group)
                    validate_ndr_topology(topology)
            self.rank, self.world_size = rank, size
            self.capacity = capacity
            self.notify_lock = threading.RLock()
            self.notify_failed = False
            self.notify_inflight = None
            self.stream = npu.Stream(device=self.device)
            if self.host_coordinated:
                # Keep a failed collective constructor reserved until process exit.
                _OWNER = self
                self.coordination_failed = True
            self.native = (
                native.create_combine_runtime(
                    rank, size, capacity, address, timeout, bootstrap == "mpi", use_1g_huge_pages
                )
                if combine_only
                else native.create_mpi_runtime(rank, size, capacity, timeout, use_1g_huge_pages)
                if bootstrap == "mpi"
                else (
                    native.Runtime(rank, size, capacity, address, timeout, udma, use_1g_huge_pages)
                    if udma
                    else factory(rank, size, capacity, address, timeout)
                )
            )
            _OWNER = self
            self.coordination_failed = False
            if capacity == 8 << 30 and 16 <= size <= 256:
                # LL peers must finish initialization before any payload writes.
                if self.host_coordinated:
                    self._host_collective("barrier")
                else:
                    self.native.barrier()

    def ensure_alive(self):
        if self.closed:
            raise RuntimeError("Buffer runtime has been destroyed")
        if getattr(self, "coordination_failed", False):
            raise RuntimeError("NDR coordination failed; restart the distributed job")
        if npu_api().current_device() != self.device:
            raise RuntimeError("Use Buffer on its creating NPU device")

    def close(self):
        global _OWNER
        with getattr(self, "notify_lock", nullcontext()), _LOCK:
            if self.closed:
                return
            if getattr(self, "host_coordinated", False):
                self._host_collective("close")
            else:
                self.ensure_alive()
                self.stream.synchronize()
                self.native.close()
            self.notify_inflight = None
            self.low_latency_comm_args = None
            self.closed = True
            _OWNER = None

    def barrier(self):
        if getattr(self, "host_coordinated", False):
            with self.notify_lock:
                self._host_collective("barrier")
            return
        self.ensure_alive()
        npu_api().current_stream(self.device).synchronize()
        self.stream.synchronize()
        self.native.barrier()

    def _host_collective(self, operation, prepare=None):
        # No new submissions may race this collective. Device completion covers
        # all streams; LL kernels quiet their owned QPs before returning.
        if self.coordination_failed:
            raise RuntimeError("NDR coordination failed; restart the distributed job")
        import torch.distributed as dist
        try:
            error = None
            try:
                self.ensure_alive()
                npu_api().synchronize(self.device)
                if prepare is not None:
                    prepare()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            peers = [None] * self.world_size
            if os.environ.get("DEEPEP_LL_INIT_TRACE") == "1":
                print(f"[DeepEP init] rank={self.rank} HCCL {operation} completion exchange enter", flush=True)
            dist.all_gather_object(peers, (operation, error), group=self.group)
            if any(peer[0] != operation for peer in peers):
                raise RuntimeError(f"NDR collective order mismatch: {peers}")
            failures = [(rank, peer[1]) for rank, peer in enumerate(peers) if peer[1] is not None]
            if failures:
                raise RuntimeError(f"NDR device completion failed: {failures}")
            # Combine reuse is a distinct collective phase, but needs only the
            # native barrier acknowledgement, never native resource cleanup.
            native_operation = "barrier" if operation == "combine_reuse" else operation
            getattr(self.native, native_operation)(host_coordinated=True)
            if os.environ.get("DEEPEP_LL_INIT_TRACE") == "1":
                print(f"[DeepEP init] rank={self.rank} HCCL {operation} complete", flush=True)
        except Exception:
            self.coordination_failed = True
            raise

    def _prepare_combine_windows(self, b, h, k, e):
        layout = (b, h, k, e)
        if getattr(self, "combine_window_layout", None) == layout:
            return
        native = require_native("NDR combine window preparation")
        if not callable(getattr(native, "clear_low_latency_combine_windows", None)):
            raise RuntimeError("Reinstall the matching NDR wheel with combine window preparation")
        native.clear_low_latency_combine_windows(self.native, b, h, k, e, self.stream.npu_stream)
        self.combine_window_layout = layout


def validate_ndr_topology(topology):
    if not topology or len(topology) % 16:
        raise ValueError("NDR requires 16 contiguous ranks per physical host")
    hosts = {}
    devices = set()
    mappings = {}
    for rank, (host, mapping, device) in enumerate(topology):
        if not host:
            raise ValueError("NDR cannot establish Linux host identity")
        hosts.setdefault(host, []).append(rank)
        if host in mappings and mappings[host] != mapping:
            raise ValueError("NDR visible-device mappings differ within a host")
        mappings[host] = mapping
        if (host, device) in devices:
            raise ValueError("NDR ranks must use distinct local devices")
        devices.add((host, device))
    expected = [list(range(start, start + 16)) for start in range(0, len(topology), 16)]
    if sorted(hosts.values()) != expected:
        raise ValueError("NDR requires 16 contiguous ranks per physical host")


def runtime_of(buffer):
    runtime = getattr(buffer, "_runtime", None)
    if runtime is None:
        raise RuntimeError("Buffer runtime was not initialized")
    return runtime


def warn_unclosed(buffer):
    runtime = getattr(buffer, "_runtime", None)
    if runtime is not None and not runtime.closed:
        warnings.warn(
            "Buffer.destroy() must be called collectively before process-group teardown; "
            "resources are retained to avoid a collective in garbage collection",
            ResourceWarning,
            stacklevel=2,
        )
