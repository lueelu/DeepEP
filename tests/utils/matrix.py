# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Layered EP matrices for CPU reference and NPU qualification.

The reference matrix isolates dimensions and remains practical on CPU. The
device matrix contains only upstream-valid V2 combine shapes (H % 256 == 0)
and is intended to be executed once for each torchrun world size.
"""

from .cases import make_matrix_case


REFERENCE_MATRIX = (
    dict(
        name="ep1_edge",
        world_size=1,
        experts=8,
        hidden=128,
        topk=1,
        tokens_per_rank=(1,),
        dtype="bfloat16",
        distribution="small",
    ),
    dict(
        name="ep2_non_aligned",
        world_size=2,
        experts=8,
        hidden=127,
        topk=2,
        tokens_per_rank=(1, 17),
        dtype="bfloat16",
        distribution="uniform",
    ),
    dict(
        name="ep4_boundaries",
        world_size=4,
        experts=64,
        hidden=256,
        topk=4,
        tokens_per_rank=(15, 16, 17, 33),
        dtype="bfloat16",
        distribution="normal",
    ),
    dict(
        name="ep8_upstream_bf16",
        world_size=8,
        experts=256,
        hidden=7168,
        topk=6,
        tokens_per_rank=(9, 8, 7, 6, 5, 4, 3, 2),
        dtype="bfloat16",
        distribution="outlier",
    ),
    dict(
        name="ep8_upstream_fp8",
        world_size=8,
        experts=256,
        hidden=7168,
        topk=6,
        tokens_per_rank=(9, 8, 7, 6, 5, 4, 3, 2),
        dtype="float8_e4m3fn",
        distribution="uniform",
    ),
    dict(
        name="ep8_token_volume",
        world_size=8,
        experts=256,
        hidden=16,
        topk=8,
        tokens_per_rank=(4096, 4095, 4094, 4093, 4092, 4091, 4090, 4089),
        dtype="float8_e4m3fn",
        distribution="normal",
    ),
)

# Backward-compatible name used by the checked-in golden builder.
MATRIX = REFERENCE_MATRIX


DEVICE_MATRIX = (
    dict(
        name="device_ep1_bf16",
        world_size=1,
        experts=8,
        hidden=256,
        topk=1,
        tokens_per_rank=(1,),
        dtype="bfloat16",
        expert_alignment=1,
        with_weights=True,
        expanded=False,
        operation="roundtrip",
        distribution="small",
    ),
    dict(
        name="device_ep2_bf16",
        world_size=2,
        experts=8,
        hidden=256,
        topk=2,
        tokens_per_rank=(1, 17),
        dtype="bfloat16",
        expert_alignment=1,
        with_weights=True,
        expanded=False,
        operation="combine",
        distribution="uniform",
    ),
    dict(
        name="device_ep2_fp8_aligned",
        world_size=2,
        experts=8,
        hidden=512,
        topk=2,
        tokens_per_rank=(15, 17),
        dtype="float8_e4m3fn",
        expert_alignment=128,
        with_weights=True,
        expanded=True,
        operation="roundtrip",
        distribution="outlier",
    ),
    dict(
        name="device_ep4_bf16_aligned",
        world_size=4,
        experts=64,
        hidden=1024,
        topk=4,
        tokens_per_rank=(15, 16, 17, 33),
        dtype="bfloat16",
        expert_alignment=128,
        with_weights=True,
        expanded=True,
        operation="roundtrip",
        distribution="normal",
    ),
    dict(
        name="device_ep4_fp8_topk8",
        world_size=4,
        experts=64,
        hidden=256,
        topk=8,
        tokens_per_rank=(33, 17, 16, 15),
        dtype="float8_e4m3fn",
        expert_alignment=1,
        with_weights=False,
        expanded=False,
        operation="dispatch",
        distribution="uniform",
    ),
    dict(
        name="device_ep8_upstream_bf16",
        world_size=8,
        experts=256,
        hidden=7168,
        topk=6,
        tokens_per_rank=(9, 8, 7, 6, 5, 4, 3, 2),
        dtype="bfloat16",
        expert_alignment=1,
        with_weights=True,
        expanded=False,
        operation="roundtrip",
        distribution="outlier",
    ),
    dict(
        name="device_ep8_upstream_fp8",
        world_size=8,
        experts=256,
        hidden=7168,
        topk=6,
        tokens_per_rank=(9, 8, 7, 6, 5, 4, 3, 2),
        dtype="float8_e4m3fn",
        expert_alignment=128,
        with_weights=True,
        expanded=True,
        operation="roundtrip",
        distribution="normal",
    ),
    dict(
        name="device_ep8_token_boundary",
        world_size=8,
        experts=256,
        hidden=256,
        topk=8,
        tokens_per_rank=(4096, 257, 256, 17, 16, 15, 1, 0),
        dtype="bfloat16",
        expert_alignment=128,
        with_weights=False,
        expanded=True,
        operation="dispatch",
        distribution="uniform",
    ),
)


_CONTROL_FIELDS = {"expanded", "operation"}


def case_from_spec(spec):
    return make_matrix_case(**{key: value for key, value in spec.items() if key not in _CONTROL_FIELDS})


def iter_matrix_cases():
    for spec in REFERENCE_MATRIX:
        yield make_matrix_case(**spec)


def device_specs_for_world_size(world_size):
    return tuple(spec for spec in DEVICE_MATRIX if spec["world_size"] == world_size)


def validate_device_spec(spec):
    case = case_from_spec(spec)
    if case.hidden % 256:
        raise ValueError("V2 device qualification requires hidden to be divisible by 256")
    if case.expert_alignment not in (1, 128):
        raise ValueError("device qualification covers expert_alignment 1 or 128")
    if spec["operation"] not in ("dispatch", "combine", "roundtrip"):
        raise ValueError("unknown qualification operation")
    return case
