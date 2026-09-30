# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Small V1 forward workflow and independent, source-side numerical answers."""

from dataclasses import replace
from math import fsum, isfinite

from tests.utils.cases import make_matrix_case
from tests.utils.checks import assert_close, assert_dispatch
from tests.utils.precision import _float32 as fp32, compare_dual, quantize_bfloat16 as bf16


def smoke_cases():
    """Six named scenarios, not a Cartesian product or performance sweep."""
    cases = []
    for index, name in enumerate(("mixed_k1", "cross_rank", "same_rank", "hotspot", "masked_zero", "fresh_iterations")):
        case = make_matrix_case(
            name,
            world_size=2,
            experts=8,
            hidden=256 if index % 2 == 0 else 512,
            topk=1 if index == 0 else 2,
            tokens_per_rank=(16, 16) if index % 2 == 0 else (17, 17),
            distribution="uniform",
        )
        routes, weights = [], []
        for source, tokens in enumerate(case.x):
            rank_routes, rank_weights = [], []
            for token in range(len(tokens)):
                owner = (source + token) % 2
                if name == "mixed_k1":
                    route = (owner * 4 + token % 4,)
                elif name == "same_rank":
                    route = (owner * 4 + token % 4, owner * 4 + (token + 1) % 4)
                elif name == "hotspot":
                    route = (0, 1)
                else:
                    route = (token % 4, 4 + (token + 1) % 4)
                gates = case.weights[source][token]
                if name == "masked_zero":
                    route = tuple(-1 if (token + slot) % 3 == 0 or token == 0 else e for slot, e in enumerate(route))
                    gates = tuple(0.0 if (token + slot) % 4 == 0 else w for slot, w in enumerate(gates))
                rank_routes.append(route)
                rank_weights.append(gates)
            routes.append(tuple(rank_routes))
            weights.append(tuple(rank_weights))
        cases.append(replace(case, routes=tuple(routes), weights=tuple(weights)))
    return tuple(cases)


def next_iteration(case):
    return replace(
        case,
        x=tuple(tuple(tuple(-v + 0.0625 for v in row) for row in rank) for rank in case.x),
        routes=tuple(
            tuple(tuple((e + case.experts // 2) % case.experts if e >= 0 else -1 for e in row) for row in rank)
            for rank in case.routes
        ),
    )


def check_dispatch(case, dispatched):
    assert len(dispatched.ranks) == case.world_size, "Dispatch rank count differs"
    transported = replace(
        case,
        x=tuple(tuple(tuple(bf16(v) for v in row) for row in rank) for rank in case.x),
        weights=None
        if case.weights is None
        else tuple(tuple(tuple(fp32(v) for v in row) for row in rank) for rank in case.weights),
    )
    for rank, received in enumerate(dispatched.ranks):
        assert_dispatch(transported, rank, False, received.x, received.topk_idx, received.topk_weights, received.counts)


def expert_step(dispatched, *, experts):
    """Consume received rows only; never regenerate data from original inputs.

    BF16 expert output -> weighted FP32 local sum -> BF16 combine input.
    The independent FP32 value channel is deliberately NOT a gate multiplier.
    """
    local_experts = experts // len(dispatched.ranks)
    payloads, values = [], []
    has_weights = dispatched.ranks[0].topk_weights is not None
    for rank, received in enumerate(dispatched.ranks):
        rank_x, rank_values = [], []
        for row, x in enumerate(received.x):
            accumulated = [0.0] * len(x)
            # Poison undefined/nonlocal values so combine must mask them.
            returned = [777.0] * len(received.topk_idx[row])
            for slot, local in enumerate(received.topk_idx[row]):
                if local < 0:
                    continue
                expert = rank * local_experts + local
                a, b = (expert + 1) / 8.0, (expert - 3) / 64.0
                gate = received.topk_weights[row][slot] if has_weights else 1.0
                for column, value in enumerate(x):
                    activation = bf16(fp32(fp32(value * a) + b))
                    accumulated[column] = fp32(accumulated[column] + fp32(activation * gate))
                if has_weights:
                    returned[slot] = fp32(0.25 + fp32(gate * 0.5))
            rank_x.append(tuple(bf16(v) for v in accumulated))
            rank_values.append(tuple(returned))
        payloads.append(tuple(rank_x))
        values.append(tuple(rank_values))
    return tuple(payloads), tuple(values) if has_weights else None


def source_answers(case):
    """Direct original-input answers, independent of dispatch and its handle.

    Returns FP64 mathematical golden, explicitly simulated BF16/FP32 path,
    and the separately returned value channel. No device capture is implied.
    """
    golden, benchmark, returned = [], [], []
    per_rank = case.experts // case.world_size
    for source, tokens in enumerate(case.x):
        rank_golden, rank_benchmark, rank_values = [], [], []
        for token, x in enumerate(tokens):
            route = case.routes[source][token]
            gates = (1.0,) * case.topk if case.weights is None else case.weights[source][token]
            reference_row, dtype_row = [], []
            for value in x:
                reference_row.append(
                    fsum(gates[s] * ((e + 1) / 8.0 * value + (e - 3) / 64.0) for s, e in enumerate(route) if e >= 0)
                )
                total = 0.0
                for destination in range(case.world_size):
                    partial = 0.0
                    for slot, expert in enumerate(route):
                        if expert < 0 or expert // per_rank != destination:
                            continue
                        transformed = bf16(fp32(fp32(bf16(value) * ((expert + 1) / 8.0)) + (expert - 3) / 64.0))
                        partial = fp32(partial + fp32(transformed * fp32(gates[slot])))
                    total = fp32(total + bf16(partial))
                dtype_row.append(bf16(total))
            rank_golden.append(tuple(reference_row))
            rank_benchmark.append(tuple(dtype_row))
            rank_values.append(
                tuple(fp32(0.25 + fp32(fp32(gates[s]) * 0.5)) if e >= 0 else 0.0 for s, e in enumerate(route))
            )
        golden.append(tuple(rank_golden))
        benchmark.append(tuple(rank_benchmark))
        returned.append(tuple(rank_values))
    return tuple(golden), tuple(benchmark), None if case.weights is None else tuple(returned)


def _check_output_shape(actual, expected):
    if isinstance(expected, tuple):
        assert isinstance(actual, (tuple, list)) and len(actual) == len(expected), "Combine output shape differs"
        for got, wanted in zip(actual, expected):
            _check_output_shape(got, wanted)
    else:
        assert type(actual) in (float, int) and isfinite(actual), "Combine output must be finite"
        assert bf16(actual) == actual, "Combine output is not BF16-representable"


def check_combine(case, actual, values):
    golden, benchmark, expected_values = source_answers(case)
    _check_output_shape(actual, golden)
    assert_close(values, expected_values, label="combine value channel")
    # Each rank must pass independently; a good rank cannot dilute a bad one.
    reports = tuple(
        compare_dual(actual[r], benchmark[r], golden[r], dtype="bfloat16", level="L2") for r in range(case.world_size)
    )
    for rank, report in enumerate(reports):
        assert report.passed, f"rank {rank}: {report.to_dict()}"
    return reports


def run_roundtrip(case, backend):
    dispatched = backend.dispatch(case)
    check_dispatch(case, dispatched)
    payloads, values = expert_step(dispatched, experts=case.experts)
    actual, returned = backend.combine(payloads, dispatched.handle, topk_weights=values)
    reports = check_combine(case, actual, returned)
    return {
        "name": case.name,
        "tokens_per_rank": list(map(len, case.x)),
        "hidden": case.hidden,
        "experts": case.experts,
        "topk": case.topk,
        "dispatch_checked": True,
        "value_channel_checked": case.weights is not None,
        "precision_per_rank": [report.to_dict() for report in reports],
        "passed": True,
    }
