# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""URMA dual-reference acceptance, independent of device/profiler execution.

Policy and cascade are imported from local DeepEP revision
5a9d6ffad7474137fa134cd7a60aebb2c8ee3089. CPU fixed tolerances are diagnostic.
"""

import numpy as np

from .accuracy import compare_triangle


def compare_outputs(actual, cascade, expected, actual_counts, cascade_counts, expected_counts, selected=None):
    """Check full-output finiteness and counts even when Golden rows are sampled."""
    actual, cascade, expected = map(np.asarray, (actual, cascade, expected))
    actual_counts, cascade_counts, expected_counts = map(np.asarray, (actual_counts, cascade_counts, expected_counts))

    def matching_counts(value):
        return bool(
            value.ndim == expected_counts.ndim == 1
            and np.issubdtype(value.dtype, np.integer)
            and np.issubdtype(expected_counts.dtype, np.integer)
            and np.all(value >= 0)
            and np.all(expected_counts >= 0)
            and np.array_equal(value, expected_counts)
        )

    report = dict(
        passed=False,
        policy="cascade_error_budget",
        finite=bool(all(np.isfinite(value).all() for value in (actual, cascade, expected))),
        counts_equal=matching_counts(actual_counts),
        cascade_counts_equal=matching_counts(cascade_counts),
        triangle={},
        triangle_metrics=[],
        reason="",
        counts_source="URMA NPU route histogram summed across sources; not a post-kernel receive counter",
    )
    if actual.ndim != 2 or actual.shape != cascade.shape:
        return dict(report, reason=f"full output shapes: URMA={actual.shape}, cascade={cascade.shape}")
    if selected is None:
        selected = np.arange(actual.shape[0])
    selected = np.asarray(selected)
    if (
        selected.ndim != 1
        or not selected.size
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any(selected < 0)
        or np.any(selected >= actual.shape[0])
        or np.unique(selected).size != selected.size
    ):
        return dict(report, reason="invalid or empty selected token indices")
    triangle, rows = compare_triangle(actual[selected], cascade[selected], expected)
    report.update(triangle=triangle, triangle_metrics=rows, checked_tokens=int(selected.size))
    failures = [key for key in ("finite", "counts_equal", "cascade_counts_equal") if not report[key]]
    if not triangle["passed"]:
        failures.append(triangle["reason"])
    report.update(passed=not failures, reason="; ".join(failures) if failures else "within cascade error budget")
    return report
