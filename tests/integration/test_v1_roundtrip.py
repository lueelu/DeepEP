# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Functional success, boundary checks and deliberately corrupted workflows."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.utils.cases import make_case, make_matrix_case
from tests.utils.precision import quantize_bfloat16 as bf16
from .functional_stub import create_backend
from .run_roundtrip import run_suite
from .workflow import (
    check_combine,
    check_dispatch,
    expert_step,
    next_iteration,
    run_roundtrip,
    smoke_cases,
    source_answers,
)


@pytest.fixture
def backend():
    worker = create_backend("functional-stub")
    try:
        yield worker
    finally:
        worker.destroy()


@pytest.fixture(scope="module")
def smoke_report():
    # Run the full matrix once; reporting checks reuse the same evidence.
    return run_suite(backend="functional-stub")


@pytest.mark.parametrize("case", smoke_cases(), ids=lambda case: case.name)
def test_smoke(smoke_report, case):
    result = next(result for result in smoke_report["results"] if result["name"] == case.name)
    assert result["passed"]
    assert all(report["passed"] for report in result["precision_per_rank"])
    if case.name == "fresh_iterations":
        assert result["second_iteration"]["passed"]


@pytest.mark.parametrize(
    "name", ("normal", "same_rank", "repeated_slot", "masked", "empty_expert", "skew", "zero_token_rank", "all_masked")
)
@pytest.mark.parametrize("weighted", (False, True))
def test_small_semantic_edges(backend, name, weighted):
    assert run_roundtrip(make_case(name, with_weights=weighted), backend)["passed"]


@pytest.mark.parametrize("distribution", ("uniform", "normal", "small", "outlier"))
def test_numeric_distributions(backend, distribution):
    case = make_matrix_case(
        "numeric", world_size=2, experts=8, hidden=16, topk=2, tokens_per_rank=(3, 4), distribution=distribution
    )
    assert run_roundtrip(case, backend)["passed"]


def test_rank_dedup_is_not_expert_dedup(backend):
    case = make_case("same_rank")
    result = backend.dispatch(case)
    assert sum(len(rank.x) for rank in result.ranks) == sum(map(len, case.x))
    assert sum(sum(rank.counts) for rank in result.ranks) == 2 * sum(map(len, case.x))
    check_dispatch(case, result)


def test_combine_only_and_weights_are_an_independent_channel(backend):
    case = make_case("normal")
    dispatched = backend.dispatch(case)
    payload = tuple(tuple((1.0,) * case.hidden for _ in rank.x) for rank in dispatched.ranks)
    values = tuple(
        tuple(tuple(3.0 if index >= 0 else 777.0 for index in row) for row in rank.topk_idx)
        for rank in dispatched.ranks
    )
    actual, returned = backend.combine(payload, dispatched.handle, topk_weights=values)
    # Both destinations receive each token: two unit rows, NOT gate * x.
    assert all(row == (2.0,) * case.hidden for rank in actual for row in rank)
    assert all(row == (3.0, 3.0) for rank in returned for row in rank)


def test_combine_can_omit_value_channel(backend):
    case = make_case("normal")
    result = backend.dispatch(case)
    payloads, _ = expert_step(result, experts=case.experts)
    _, values = backend.combine(payloads, result.handle)
    assert values is None


def test_hand_computed_weighted_answer(backend):
    case = replace(
        make_case("normal", hidden=1),
        x=(((2.0,),), ((4.0,),)),
        routes=(((0, 2),), ((3, -1),)),
        weights=(((0.5, 0.25),), ((0.0, 777.0),)),
    )
    result = backend.dispatch(case)
    payloads, values = expert_step(result, experts=case.experts)
    actual, returned = backend.combine(payloads, result.handle, topk_weights=values)
    # e0(2)=13/64, e2(2)=47/64; 1/2*13/64+1/4*47/64=73/256.
    assert actual == (((73 / 256,),), ((0.0,),))
    assert returned == (((0.5, 0.375),), ((0.25, 0.0),))
    assert source_answers(case)[0] == actual


@pytest.mark.parametrize("fault", ("payload", "route", "gate", "count", "missing_row", "missing_rank"))
def test_dispatch_checker_rejects_corruption(backend, fault):
    case = make_case("normal")
    result = backend.dispatch(case)
    first = result.ranks[0]
    if fault == "payload":
        first = replace(first, x=((99.0,) * case.hidden,) + first.x[1:])
    elif fault == "route":
        first = replace(first, topk_idx=((1, -1),) + first.topk_idx[1:])
    elif fault == "gate":
        first = replace(first, topk_weights=((9.0, 9.0),) + first.topk_weights[1:])
    elif fault == "count":
        first = replace(first, counts=(999, 999))
    elif fault == "missing_row":
        first = replace(first, x=first.x[1:])
    result = replace(result, ranks=(first,) if fault == "missing_rank" else (first,) + result.ranks[1:])
    with pytest.raises(AssertionError):
        check_dispatch(case, result)


@pytest.mark.parametrize(
    "fault", ("missing_contribution", "double_weight", "source_order", "value_channel", "shape", "nan")
)
def test_combine_checker_rejects_corruption(backend, fault):
    case = make_case("normal")
    result = backend.dispatch(case)
    payloads, values = expert_step(result, experts=case.experts)
    if fault == "missing_contribution":
        payloads = (tuple((0.0,) * case.hidden for _ in payloads[0]),) + payloads[1:]
    elif fault == "double_weight":
        payloads = tuple(
            tuple(
                tuple(
                    bf16(v * rank.topk_weights[i][next(s for s, e in enumerate(rank.topk_idx[i]) if e >= 0)])
                    for v in row
                )
                for i, row in enumerate(rows)
            )
            for rows, rank in zip(payloads, result.ranks)
        )
    actual, returned = backend.combine(payloads, result.handle, topk_weights=values)
    if fault == "source_order":
        actual = (tuple(reversed(actual[0])), actual[1])
    elif fault == "value_channel":
        returned = (((999.0, 999.0),) + returned[0][1:], returned[1])
    elif fault == "shape":
        actual = (actual[0][1:], actual[1])
    elif fault == "nan":
        actual = (((float("nan"),) * case.hidden,) + actual[0][1:], actual[1])
    with pytest.raises(AssertionError):
        check_combine(case, actual, returned)


def test_losing_second_local_expert_is_detected(backend):
    case = make_case("same_rank")
    result = backend.dispatch(case)
    broken = replace(
        result,
        ranks=tuple(replace(rank, topk_idx=tuple((row[0], -1) for row in rank.topk_idx)) for rank in result.ranks),
    )
    payloads, values = expert_step(broken, experts=case.experts)
    actual, returned = backend.combine(payloads, result.handle, topk_weights=values)
    with pytest.raises(AssertionError):
        check_combine(case, actual, returned)


def test_expert_step_consumes_received_data(backend):
    case = make_case("normal")
    result = backend.dispatch(case)
    expected, _ = expert_step(result, experts=case.experts)
    first = replace(result.ranks[0], x=((0.0,) * case.hidden,) + result.ranks[0].x[1:])
    changed, _ = expert_step(replace(result, ranks=(first, result.ranks[1])), experts=case.experts)
    assert changed != expected


def test_handles_are_owned_fresh_and_single_use(backend):
    case = make_case("normal")
    first = backend.dispatch(case)
    snapshot = first.ranks
    current = backend.dispatch(next_iteration(case))
    payloads, values = expert_step(current, experts=case.experts)
    with pytest.raises(ValueError, match="stale"):
        backend.combine(payloads, first.handle)
    other = create_backend("functional-stub")
    try:
        with pytest.raises(ValueError, match="foreign"):
            other.combine(payloads, current.handle)
    finally:
        other.destroy()
    backend.combine(payloads, current.handle, topk_weights=values)
    with pytest.raises(ValueError, match="consumed"):
        backend.combine(payloads, current.handle)
    assert first.ranks == snapshot
    check_dispatch(case, first)


def test_destroy_rejects_use(backend):
    case = make_case("normal")
    result = backend.dispatch(case)
    backend.destroy()
    backend.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        backend.dispatch(case)
    with pytest.raises(RuntimeError, match="destroyed"):
        backend.combine((), result.handle)


@pytest.mark.parametrize(
    "changes",
    (
        {"dtype": "float8_e4m3fn"},
        {"expert_alignment": 128},
        {"hidden": 513},
        {"experts": 128},
        {"topk": 3},
        {"routes": (((99, 2), (1, 3), (2, 1)), ((3, 0), (2, 1)))},
    ),
)
def test_unsupported_or_invalid_case_rejected(backend, changes):
    with pytest.raises(ValueError):
        backend.dispatch(replace(make_case("normal"), **changes))


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), 1e40, True, "1"))
@pytest.mark.parametrize("channel", ("x", "weights"))
def test_nonfinite_and_nonnumeric_inputs_rejected(backend, bad, channel):
    case = make_case("normal")
    rows = getattr(case, channel)
    altered = (((bad,) + rows[0][0][1:],) + rows[0][1:], rows[1])
    with pytest.raises(ValueError):
        backend.dispatch(replace(case, **{channel: altered}))


@pytest.mark.parametrize(
    "fault", ("ranks", "rows", "hidden", "unquantized", "nan", "value_ranks", "value_rows", "value_width", "value_nan")
)
def test_combine_validates_before_consuming_handle(backend, fault):
    case = make_case("normal")
    result = backend.dispatch(case)
    payloads, values = expert_step(result, experts=case.experts)
    broken_x, broken_v = payloads, values
    if fault == "ranks":
        broken_x = payloads[:1]
    elif fault == "rows":
        broken_x = (payloads[0][1:], payloads[1])
    elif fault in ("hidden", "unquantized", "nan"):
        row = (
            payloads[0][0][1:]
            if fault == "hidden"
            else ((1.001 if fault == "unquantized" else float("nan")),) + payloads[0][0][1:]
        )
        broken_x = ((row,) + payloads[0][1:], payloads[1])
    elif fault == "value_ranks":
        broken_v = values[:1]
    elif fault == "value_rows":
        broken_v = (values[0][1:], values[1])
    else:
        row = values[0][0][1:] if fault == "value_width" else (float("nan"),) + values[0][0][1:]
        broken_v = ((row,) + values[0][1:], values[1])
    with pytest.raises(ValueError):
        backend.combine(broken_x, result.handle, topk_weights=broken_v)
    actual, returned = backend.combine(payloads, result.handle, topk_weights=values)
    check_combine(case, actual, returned)


def test_report_never_claims_device_or_wheel_qualification(smoke_report):
    report = smoke_report
    assert report["case_count"] == 6 and report["passed"]
    assert report["execution"] == "single-process-cpu-simulation"
    for field in ("device_communication_verified", "wheel_api_verified", "device_reference_captured"):
        assert report[field] is False
    assert report["second_reference"] == "cpu-bf16-fp32-path-simulation"
    assert "bandwidth" not in report and "latency_us" not in report
    assert report["results"][-1]["second_iteration"]["passed"]


def test_no_native_or_low_latency_fallback():
    for name in ("native", "low-latency", "auto", ""):
        with pytest.raises(ValueError, match="no native fallback"):
            create_backend(name)


@pytest.mark.parametrize("field", ("hidden", "experts", "topk", "expert_alignment"))
def test_shape_fields_reject_boolean(backend, field):
    with pytest.raises(ValueError, match="integers"):
        backend.dispatch(replace(make_case("normal"), **{field: True}))


def test_failed_dispatch_does_not_invalidate_previous_handle(backend):
    case = make_case("normal")
    result = backend.dispatch(case)
    with pytest.raises(ValueError):
        backend.dispatch(replace(case, dtype="float8_e4m3fn"))
    payloads, values = expert_step(result, experts=case.experts)
    actual, returned = backend.combine(payloads, result.handle, topk_weights=values)
    check_combine(case, actual, returned)


def test_small_value_errors_fail_per_rank(backend):
    case = replace(
        make_case("normal"),
        x=(((0.0,) * 4,), ((0.0,) * 4,)),
        routes=(((3, -1),), ((3, -1),)),
        weights=(((1.0, 0.0),), ((1.0, 0.0),)),
    )
    result = backend.dispatch(case)
    payloads, values = expert_step(result, experts=case.experts)
    actual, returned = backend.combine(payloads, result.handle, topk_weights=values)
    reports = check_combine(case, actual, returned)
    assert all(report.actual.small_count == 4 for report in reports)
    corrupt = (((bf16(0.01),) * 4,), actual[1])
    with pytest.raises(AssertionError, match="rank 0"):
        check_combine(case, corrupt, returned)


def test_wrong_world_size_and_oversized_inputs_rejected(backend):
    for world_size, tokens in ((1, (2,)), (2, (65, 1))):
        case = make_matrix_case(
            "invalid_size", world_size=world_size, experts=8, hidden=4, topk=2, tokens_per_rank=tokens
        )
        with pytest.raises(ValueError):
            backend.dispatch(case)


def test_cli_rejects_disabled_assertions():
    result = subprocess.run(
        [sys.executable, "-O", "-m", "tests.integration.run_roundtrip", "--backend", "functional-stub"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "requires Python assertions" in result.stderr
    assert not result.stdout


def test_backend_exceptions_propagate_and_cleanup(monkeypatch):
    worker = create_backend("functional-stub")

    def broken_dispatch(case):
        raise NotImplementedError("required operation unavailable")

    monkeypatch.setattr(worker, "dispatch", broken_dispatch)
    monkeypatch.setattr("tests.integration.run_roundtrip.create_backend", lambda name: worker)
    with pytest.raises(NotImplementedError, match="unavailable"):
        run_suite(backend="functional-stub")
    assert worker._closed


@pytest.mark.parametrize(
    "arguments, status", (([], 2), (["--backend", "native"], 2), (["--backend", "functional-stub"], 0))
)
def test_cli_requires_explicit_backend(arguments, status):
    result = subprocess.run(
        [sys.executable, "-m", "tests.integration.run_roundtrip", *arguments],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == status, result.stderr
    if status == 0:
        assert json.loads(result.stdout)["passed"]
    else:
        assert not result.stdout
