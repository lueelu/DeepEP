# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""CPU and NPU cascade error budgets for BF16 MegaMoE outputs.

Adapted from ascend_deepep st/reference/benchmark_accuracy.py, branch
pod-dev-timer-0922, revision 47ce7911abff84255fd0ab3879bfc4f5e7f52ad0:
https://gitcode.com/SimpleBright_Man/ascend_deepep
The CPU Golden retains this package's existing quantization/rounding contract.
"""

import numpy as np

from .golden import compare

TRIANGLE_RATIOS = dict(
    max_relative=5.0,
    mean_relative=1.5,
    sum_relative=2.0,
    rmse=1.5,
    small_bad_elements=2.0,
)


def triangle_metrics(actual, expected):
    """Compute BF16-output error metrics in FP64 without changing the inputs."""
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    absolute = np.abs(actual - expected)
    big = np.abs(expected) >= 0.001
    relative = absolute[big] / np.abs(expected[big])
    return dict(
        max_relative=float(relative.max(initial=0)),
        mean_relative=float(relative.mean()) if relative.size else 0.0,
        sum_relative=float(relative.sum()),
        rmse=float(np.sqrt(np.mean(absolute**2))) if absolute.size else 0.0,
        small_bad_elements=int(np.count_nonzero(absolute[~big] > 0.00001)),
        small_elements=int(np.count_nonzero(~big)),
    )


def compare_triangle(actual, cascade, expected, **tolerances):
    """Compare fused/CPU errors against cascade/CPU errors for the same Golden.

    The CPU mixed-error verdict remains diagnostic. Triangle acceptance uses
    only the five cascade budgets, with exact zero budgets when the cascade is
    exact. Shapes and finite values must match before computing the budgets.
    Returns the report and one row per budget; disabled rows have passed=None.
    """
    actual, cascade, expected = map(np.asarray, (actual, cascade, expected))
    result = compare(actual, expected, **tolerances)
    result.update(cpu_mixed_passed=result["passed"], passed=False, policy="cascade_error_budget")
    if actual.shape != expected.shape or cascade.shape != expected.shape:
        result["reason"] = f"triangle shapes: MegaMoE={actual.shape}, cascade={cascade.shape}, Golden={expected.shape}"
        return result, []
    if any(not np.isfinite(value).all() for value in (actual, cascade, expected)):
        result["reason"] = "non-finite MegaMoE, cascade or CPU Golden"
        return result, []
    fused_metrics, cascade_metrics = triangle_metrics(actual, expected), triangle_metrics(cascade, expected)
    rows = []
    for metric, ratio in TRIANGLE_RATIOS.items():
        enabled = metric != "small_bad_elements" or fused_metrics["small_elements"] >= 100000
        value, reference = fused_metrics[metric], cascade_metrics[metric]
        limit = reference * ratio
        rows.append(
            dict(
                metric=metric,
                mega_vs_cpu=value,
                cascade_vs_cpu=reference,
                ratio=ratio,
                limit=limit,
                enabled=enabled,
                passed=value <= limit if enabled else None,
            )
        )
    failed = [row["metric"] for row in rows if row["enabled"] and not row["passed"]]
    result["passed"] = not failed
    result["reason"] = "triangle: " + (
        ", ".join(failed) + " exceeds cascade error budget" if failed else "within cascade error budget"
    )
    return result, rows
