# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026
"""Host-only search space and wire format for the compiled Ascend950 kernels.

Like CatCCOS tests/dynamic_tiling, search enumerates compiled MMAD variants and
runtime communication parameters. No torch, NPU or YAML dependency is needed.
The native launchers repeat the checks with the actual device's resources.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Mapping


ALL_TO_ALLV = "all_to_allv_grouped_mat_mul"
GROUPED_MATMUL = "grouped_mat_mul_all_to_allv"
TILING_FIELDS = {
    ALL_TO_ALLV: (
        "m0",
        "n0",
        "k0",
        "comm_block_m",
        "stage_rows",
        "workspace_stages",
        "local_copy_tile_rows",
        "m_split",
        "export_core_count",
        "self_copy_cores",
    ),
    GROUPED_MATMUL: (
        "m0",
        "n0",
        "k0",
        "comm_interval",
        "local_copy_tile_rows",
        "stage_rows",
        "workspace_stages",
        "self_copy_cores",
    ),
}
MMAD_SHAPES = {
    ALL_TO_ALLV: ((128, 256, 256), (256, 128, 256), (256, 256, 256)),
    GROUPED_MATMUL: ((128, 256, 256), (256, 128, 256), (256, 256, 256)),
}
# CATLASS Ascend950 LocalTensorBuffer allocation, smaller than physical UB.
UB_BYTES = 248 * 1024
WORKSPACE_BYTES = 400 * 1024 * 1024
UDMA_SCRATCH_BYTES = 256
UINT32_MAX = (1 << 32) - 1
INT32_MAX = (1 << 31) - 1
MAX_SPACE_SIZE = 1_000_000


def _integer(name, value, minimum=1, maximum=UINT32_MAX):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(name))
    if not minimum <= value <= maximum:
        raise ValueError("{} must be in [{}, {}]".format(name, minimum, maximum))
    return value


def _aligned(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def validate_tiling(
    operator,
    tiling,
    *,
    output_all_to_allv=False,
    m=None,
    n=None,
    k=None,
    world_size=None,
    aic_cores=None,
    ub_bytes=UB_BYTES,
):
    """Validate a complete profile; return a copy in the native field order.

    Missing hardware/shape arguments defer those checks to the native launcher.
    Local row counts are deliberately not selection keys: they can differ by rank.
    """
    if operator not in TILING_FIELDS:
        raise ValueError("unsupported operator: {}".format(operator))
    if not isinstance(tiling, Mapping):
        raise ValueError("tiling must be a mapping")
    if operator == GROUPED_MATMUL:
        # Saved five-field overrides keep the legacy staging and worker pool,
        # independently of the shape-specific defaults selected by the host.
        tiling = dict(tiling)
        for name, value in (("stage_rows", 0), ("workspace_stages", 2), ("self_copy_cores", 1)):
            tiling.setdefault(name, value)
    fields = TILING_FIELDS[operator]
    missing = set(fields) - set(tiling)
    unknown = set(tiling) - set(fields)
    if missing or unknown:
        raise ValueError("tiling fields: missing={}, unknown={}".format(sorted(missing), sorted(unknown, key=str)))
    result = {
        name: _integer(
            name,
            tiling[name],
            0 if name == "export_core_count" or (operator == GROUPED_MATMUL and name == "stage_rows") else 1,
            INT32_MAX,
        )
        for name in fields
    }
    shape = tuple(result[name] for name in ("m0", "n0", "k0"))
    if shape not in MMAD_SHAPES[operator]:
        raise ValueError("MMAD shape {} has no compiled {} kernel".format(shape, operator))
    if not isinstance(output_all_to_allv, bool):
        raise ValueError("output_all_to_allv must be a boolean")
    for name, value in (("n", n), ("k", k), ("world_size", world_size), ("aic_cores", aic_cores)):
        if value is not None:
            _integer(name, value)
    if m is not None:
        _integer("m", m, 0 if operator == GROUPED_MATMUL else 1)
    if world_size is not None and world_size > 8:
        raise ValueError("world_size must not exceed 8")
    if aic_cores is not None and world_size is not None and aic_cores < world_size:
        raise ValueError("at least one AI core per rank is required")
    ub_capacity = min(_integer("ub_bytes", ub_bytes), UB_BYTES)
    local_rows = result["local_copy_tile_rows"]

    if operator == ALL_TO_ALLV:
        stages = result["workspace_stages"]
        if stages not in (2, 3) or result["m_split"] not in (1, 2, 4):
            raise ValueError("compiled export kernels require workspace_stages 2/3 and m_split 1/2/4")
        if not output_all_to_allv and (stages != 2 or result["m_split"] != 1):
            raise ValueError("non-export kernels require workspace_stages=2 and m_split=1")
        if not output_all_to_allv and result["export_core_count"] != 0:
            raise ValueError("export_core_count must be 0 for a non-export kernel")
        if result["stage_rows"] % result["m0"]:
            raise ValueError("stage_rows must be divisible by m0")
        if k is not None and k > UINT32_MAX - result["k0"] + 1:
            raise ValueError("K exceeds the uint32 padded communication shape")
        if m is not None:
            if m > UINT32_MAX - result["stage_rows"] + 1:
                raise ValueError("M exceeds the uint32 communication schedule")
            if world_size is not None:
                blocks = (result["stage_rows"] + result["comm_block_m"] - 1) // result["comm_block_m"]
                rounds = (m + result["stage_rows"] - 1) // result["stage_rows"]
                if blocks * world_size * rounds > INT32_MAX:
                    raise ValueError("communication schedule exceeds the int32 completion counter")
        if k is not None and world_size is not None:
            # DistRowMajor aligns BF16 staging rows to 512 bytes (256 elements).
            workspace = stages * result["stage_rows"] * world_size * _aligned(k, 256) * 2
            if workspace > WORKSPACE_BYTES:
                raise ValueError("tiling exceeds the 400 MiB symmetric workspace")
        if aic_cores is not None:
            if result["export_core_count"] > aic_cores:
                raise ValueError("export_core_count exceeds the AI-core count")
            if world_size is not None and world_size + result["self_copy_cores"] - 1 > aic_cores:
                raise ValueError("self_copy_cores exceeds the available AIV worker pool")
            if n is not None and world_size is not None:
                tasks = (result["stage_rows"] // result["m0"]) * world_size * ((n + result["n0"] - 1) // result["n0"])
                if tasks > UINT32_MAX - aic_cores:
                    raise ValueError("MMAD task grid exceeds the uint32 scheduler range")
        copy_columns = result["k0"]
    else:
        if output_all_to_allv:
            raise ValueError("GroupedMatMul->AllToAllV has no export mode")
        if local_rows > INT32_MAX // 2:
            raise ValueError("local_copy_tile_rows exceeds the native commTileM range")
        stage_rows = result["stage_rows"]
        stages = result["workspace_stages"]
        if stage_rows % result["m0"]:
            raise ValueError("stage_rows must be zero (derived) or divisible by m0")
        if stages not in (2, 3):
            raise ValueError("workspace_stages must be 2 or 3")
        if m is not None and m > UINT32_MAX - result["m0"] + 1:
            raise ValueError("input rows overflow the target-order tiling scheduler")
        if aic_cores is not None and world_size is not None:
            if world_size + result["self_copy_cores"] - 1 > aic_cores:
                raise ValueError("self_copy_cores exceeds the available AIV worker pool")
            if stage_rows == 0:
                blocks = aic_cores * result["comm_interval"]
                if blocks > UINT32_MAX:
                    raise ValueError("AI-core count * comm_interval exceeds uint32")
                if blocks % world_size:
                    raise ValueError("AI-core count * comm_interval must be divisible by world_size")
                if n is not None:
                    n_loops = (n + result["n0"] - 1) // result["n0"]
                    rank_tiles = blocks // world_size
                    if rank_tiles < n_loops:
                        raise ValueError("comm_interval is too small to cover one full N row of tiles")
                    stage_rows = (rank_tiles // n_loops) * result["m0"]
        if world_size is not None and stage_rows:
            if stage_rows * world_size > UINT32_MAX:
                raise ValueError("stage_rows * world_size exceeds uint32")
            if n is not None:
                if stages * stage_rows * world_size * n * 2 > WORKSPACE_BYTES:
                    raise ValueError("tiling exceeds the 400 MiB symmetric workspace")
                n_loops = (n + result["n0"] - 1) // result["n0"]
                tasks = (stage_rows // result["m0"]) * world_size * n_loops
                if aic_cores is not None and tasks > UINT32_MAX - aic_cores:
                    raise ValueError("MMAD task grid exceeds the uint32 scheduler range")
        copy_columns = _aligned(n, 16) if n is not None else None
    if n is not None and n > UINT32_MAX - result["n0"] + 1:
        raise ValueError("N exceeds the uint32 MMAD scheduler range")
    if copy_columns is not None and 2 * local_rows * copy_columns * 2 + UDMA_SCRATCH_BYTES > ub_capacity:
        raise ValueError("local_copy_tile_rows exceeds the local MTE UB capacity")
    return result


def default_tiling(case):
    """Express the existing native shape-specific fallback in searchable fields."""
    operator = case["operator"]
    m, n, k = (case["shape"][axis] for axis in ("m", "n", "k"))
    business = case["world_size"] == 8 and case["expert_num"] == 256
    export = case.get("output_all_to_allv", False)
    if operator == ALL_TO_ALLV:
        values = [128, 256, 256, 64, 1280, 2, 32, 1, 0, 1]
        if business and m == 131072 and k == 2048:
            if n == 1024:
                values = (
                    [256, 256, 256, 32, 1280, 3, 16, 1, 24, 8] if export else [128, 256, 256, 128, 1536, 2, 64, 1, 0, 1]
                )
            elif n == 512:
                values = (
                    [256, 128, 256, 64, 1280, 2, 8, 1, 0, 8] if export else [256, 128, 256, 128, 1536, 2, 64, 1, 0, 1]
                )
        elif (
            export
            and case["world_size"] == 4
            and case["expert_num"] == 28
            and m == 131072
            and k == 3584
            and n in (6144, 3072)
        ):
            values = [256, 256, 256, 32, 2048, 2, 16, 1, 8, 4 if n == 6144 else 8]
    elif operator == GROUPED_MATMUL:
        values = [128, 256, 256, 10, 32, 0, 2, 1]
        if business and n == 2048:
            if k == 1024:
                values = [128, 256, 256, 10, 4, 0, 2, 1]
            elif k == 512:
                values = [128, 256, 256, 16, 16, 0, 2, 1]
        elif case["world_size"] == 4 and case["expert_num"] == 28 and n == 3584 and k in (3072, 6144):
            values = [256, 256, 256, 8, 8, 2048, 3 if k == 3072 else 2, 8]
        # Native fallback caps the requested copy rows to the compiled UB size.
        capacity = (UB_BYTES - UDMA_SCRATCH_BYTES) // (4 * _aligned(n, 16))
        values[4] = min(values[4], capacity)
    else:
        raise ValueError("unsupported operator: {}".format(operator))
    return dict(zip(TILING_FIELDS[operator], values))


def candidate_id(tiling):
    payload = json.dumps(tiling, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_search_space(case):
    operator = case["operator"]
    if operator not in TILING_FIELDS:
        raise ValueError("unsupported operator: {}".format(operator))
    result = {"mmad": [list(shape) for shape in MMAD_SHAPES[operator]]}
    if operator == ALL_TO_ALLV:
        export = case.get("output_all_to_allv", False)
        result.update(
            comm_block_m=[32, 64, 128],
            stage_rows=[512, 1024, 1280, 1536, 2048, 3072, 4096],
            workspace_stages=[2, 3] if export else [2],
            local_copy_tile_rows=[8, 16, 32, 64],
            m_split=[1, 2, 4] if export else [1],
            export_core_count=[0, 8, 16, 24, 32] if export else [0],
            self_copy_cores=[1, 2, 4, 8],
        )
    else:
        # Only explicit search spaces expand the added physical dimensions.
        baseline = default_tiling(case)
        result.update(
            comm_interval=[1, 2, 4, 6, 8, 10, 12, 14, 16],
            local_copy_tile_rows=[1, 2, 4, 8, 16, 32, 64],
            **{name: [baseline[name]] for name in ("stage_rows", "workspace_stages", "self_copy_cores")},
        )
    return result


def _space_for_case(case, space):
    defaults = default_search_space(case)
    if space is None:
        return defaults
    if not isinstance(space, Mapping):
        raise ValueError("search space must be a mapping")
    if "operators" in space:
        if set(space) - {"schema_version", "operators"} or space.get("schema_version") != 1:
            raise ValueError("search-space document requires schema_version=1 and operators")
        operators = space["operators"]
        if not isinstance(operators, Mapping) or set(operators) - set(TILING_FIELDS):
            raise ValueError("search-space operators must name supported operators")
        space = operators.get(case["operator"], {})
    if not isinstance(space, Mapping) or set(space) - set(defaults):
        raise ValueError("unknown or inactive search dimensions; use 'mmad' for compiled M0/N0/K0")
    defaults.update(space)
    size = 1
    for name, values in defaults.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("search dimension {} must be a non-empty list".format(name))
        if name == "mmad":
            for shape in values:
                if not isinstance(shape, (list, tuple)) or len(shape) != 3:
                    raise ValueError("mmad entries must be [m0, n0, k0]")
                for value in shape:
                    _integer("mmad", value)
                if tuple(shape) not in MMAD_SHAPES[case["operator"]]:
                    raise ValueError("mmad shape has no compiled kernel: {}".format(shape))
        else:
            for value in values:
                _integer(
                    name,
                    value,
                    0
                    if name == "export_core_count" or (case["operator"] == GROUPED_MATMUL and name == "stage_rows")
                    else 1,
                )
        size *= len(values)
    if size > MAX_SPACE_SIZE:
        raise ValueError("search space exceeds {} combinations; narrow the dimensions".format(MAX_SPACE_SIZE))
    return defaults


def candidates_for_case(
    case,
    space=None,
    *,
    strategy="random",
    max_candidates=64,
    seed=2026,
    aic_cores=None,
    ub_bytes=UB_BYTES,
):
    """Return distinct valid candidates, keeping the old fallback first if valid.

    The budget includes that fallback even when a custom grid excludes it. Random
    search is reproducible sampling without replacement; exhaustive preserves grid
    order. None means no budget cap. Only the tested space can claim an optimum.
    """
    if strategy not in ("random", "exhaustive", "grid"):
        raise ValueError("strategy must be random or exhaustive")
    if max_candidates is not None:
        _integer("max_candidates", max_candidates)
    _integer("seed", seed, 0, (1 << 63) - 1)
    dimensions = _space_for_case(case, space)
    keys = list(dimensions)
    baseline = default_tiling(case)
    seen = set()
    candidates = []

    def append(tiling):
        try:
            normalized = validate_tiling(
                case["operator"],
                tiling,
                output_all_to_allv=case.get("output_all_to_allv", False),
                m=case["shape"]["m"],
                n=case["shape"]["n"],
                k=case["shape"]["k"],
                world_size=case["world_size"],
                aic_cores=aic_cores,
                ub_bytes=ub_bytes,
            )
        except ValueError:
            return
        effective = dict(normalized)
        if case["operator"] == ALL_TO_ALLV:
            if aic_cores is not None:
                effective["export_core_count"] = effective["export_core_count"] or aic_cores
        else:
            if effective["stage_rows"] == 0 and aic_cores is not None:
                # Only complete M groups reach the target-order schedulers.
                n_loops = (case["shape"]["n"] + effective["n0"] - 1) // effective["n0"]
                effective["stage_rows"] = effective["m0"] * (
                    aic_cores * effective["comm_interval"] // case["world_size"] // n_loops
                )
            if effective["stage_rows"]:
                effective.pop("comm_interval")  # Inactive when physical staging is known.
        identity = candidate_id(effective)
        if identity not in seen:
            seen.add(identity)
            candidates.append(normalized)

    append(baseline)
    baseline_valid = bool(candidates)
    for values in itertools.product(*(dimensions[key] for key in keys)):
        tiling = dict(zip(keys, values))
        tiling.update(zip(("m0", "n0", "k0"), tiling.pop("mmad")))
        append(tiling)
    if not candidates:
        raise ValueError("no tiling candidates satisfy the shape and hardware constraints")
    if strategy == "random":
        offset = 1 if baseline_valid else 0
        rest = candidates[offset:]
        random.Random(seed).shuffle(rest)
        candidates[offset:] = rest
    return candidates[:max_candidates] if max_candidates is not None else candidates
