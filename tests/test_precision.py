# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Precision policy, dtype emulation, matrix and golden regression tests."""

import json
from fractions import Fraction
from math import sqrt
from pathlib import Path

import pytest

from tests.utils.golden import build_manifest
from tests.utils.matrix import (
    DEVICE_MATRIX,
    MATRIX,
    case_from_spec,
    validate_device_spec,
)
from tests.utils.precision import (
    bootstrap_median_recheck,
    compare_dual,
    emulate_upstream_row,
    error_metrics,
    quantize_bfloat16,
    quantize_e4m3fn,
)


def test_matrix_covers_ep_shapes_topk_tokens_and_upstream_dtypes():
    assert {spec["world_size"] for spec in MATRIX} >= {1, 2, 4, 8}
    assert {spec["hidden"] for spec in MATRIX} >= {16, 127, 128, 256, 7168}
    assert {spec["topk"] for spec in MATRIX} >= {1, 2, 4, 6, 8}
    assert max(max(spec["tokens_per_rank"]) for spec in MATRIX) == 4096
    assert {spec["dtype"] for spec in MATRIX} == {"bfloat16", "float8_e4m3fn"}
    assert {spec["distribution"] for spec in MATRIX} >= {
        "uniform",
        "normal",
        "small",
        "outlier",
    }


def test_device_matrix_is_upstream_valid_and_covers_alignment():
    cases = [validate_device_spec(spec) for spec in DEVICE_MATRIX]
    assert {case.world_size for case in cases} == {1, 2, 4, 8}
    assert {case.dtype for case in cases} == {"bfloat16", "float8_e4m3fn"}
    assert {case.expert_alignment for case in cases} == {1, 128}
    assert all(case.hidden % 256 == 0 for case in cases)
    assert max(max(map(len, case.x)) for case in cases) == 4096


def test_device_matrix_rejects_invalid_v2_hidden():
    invalid = dict(DEVICE_MATRIX[0], hidden=128)
    with pytest.raises(ValueError, match="divisible by 256"):
        validate_device_spec(invalid)


def test_outlier_distribution_is_bounded_and_injected():
    spec = dict(
        name="outlier_anchor",
        world_size=1,
        experts=8,
        hidden=1000,
        topk=1,
        tokens_per_rank=(1,),
        dtype="bfloat16",
        distribution="outlier",
    )
    case = case_from_spec(spec)
    values = case.x[0][0]
    assert max(map(abs, values)) <= 5.0
    assert sum(abs(value) > 0.005 for value in values) == 1


def test_dtype_rounding_anchors():
    assert quantize_bfloat16(1.00390625) == 1.0  # exact halfway, ties to even
    assert quantize_bfloat16(1.0078125) == 1.0078125
    assert quantize_e4m3fn(500.0) == 448.0
    assert quantize_e4m3fn(-500.0) == -448.0
    assert quantize_e4m3fn(2**-9) == 2**-9


def test_upstream_fp8_emulation_scales_each_128_columns():
    row = tuple([1.0] * 128 + [224.0, 448.0])
    restored = emulate_upstream_row(row, "float8_e4m3fn")
    assert len(restored) == len(row)
    assert restored[128:] == (224.0, 448.0)
    assert restored[0] == pytest.approx(1.0, abs=0.02)


def test_dual_reference_l2_accepts_baseline_and_rejects_regression():
    golden = (1.0, 2.0, 4.0, 0.0)
    benchmark = (1.01, 1.98, 4.02, 0.0)
    assert compare_dual(benchmark, benchmark, golden, dtype="bfloat16").passed
    regression = (1.1, 1.8, 4.2, 0.01)
    report = compare_dual(regression, benchmark, golden, dtype="bfloat16")
    assert not report.passed
    assert max(report.ratios) > 2


def test_dual_reference_rejects_small_domain_error_count_ratio():
    golden = (0.0, 0.0, 0.0)
    benchmark = (0.0, 0.0, 0.0)
    actual = (0.01, 0.01, 0.01)
    report = compare_dual(actual, benchmark, golden, dtype="bfloat16")
    assert not report.passed
    assert report.small_error_ratio == 3.0


def test_small_domain_is_not_hidden_by_relative_error():
    metrics = error_metrics((0.0, 2**-7), (2**-9, 2**-7), dtype="bfloat16")
    assert metrics.small_count == 1
    assert metrics.small_error_count == 1
    assert metrics.normal_count == 1


def test_metric_reductions_preserve_small_terms_across_python_versions():
    # Exact rational accumulation is an independent oracle for the reduction.
    # On Python 3.11, builtin float sum loses the small squared errors after 1.
    actual = (2.0,) + (1.0 + 2**-27,) * 128
    golden = (1.0,) * len(actual)
    metrics = error_metrics(actual, golden, dtype="bfloat16")
    errors = [abs(a - b) for a, b in zip(actual, golden)]
    relative = [error / (abs(b) + 1e-7) for error, b in zip(errors, golden)]
    exact_relative = sum(Fraction.from_float(value) for value in relative)
    exact_squares = sum(Fraction.from_float(value * value) for value in errors)
    assert metrics.mere == float(exact_relative) / len(actual)
    assert metrics.rmse == sqrt(float(exact_squares) / len(actual))


def test_bootstrap_recheck_enforces_sample_fuse_and_ci_rule():
    small = bootstrap_median_recheck([0.9] * 199)
    assert not small.passed and small.reason == "sample_count_below_200"
    degraded = bootstrap_median_recheck([1.01] * 200)
    assert not degraded.passed and degraded.ci_lower > 1.0
    acceptable = bootstrap_median_recheck([0.99] * 200)
    assert acceptable.passed and acceptable.ci_upper < 1.0


def test_checked_in_golden_is_current():
    path = Path(__file__).parent / "golden" / "deepep_reference_v2.json"
    assert json.loads(path.read_text(encoding="utf-8")) == build_manifest()
