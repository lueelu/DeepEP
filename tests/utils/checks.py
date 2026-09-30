# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Semantic comparisons, independent of native handle/metadata layouts."""

from collections import Counter
from math import isclose, isfinite

from .reference import dispatch


def assert_dispatch(case, rank, expanded, x, indices, weights, counts):
    expected = dispatch(case, expanded=expanded)
    rows = expected.ranks[rank]
    assert tuple(counts) == expected.counts[rank], "Per-expert counts differ"
    assert len(x) == len(rows), "Received row count differs"
    assert (indices is None) == expanded, "Dispatch index layout differs"
    assert (weights is None) == (case.weights is None), "Dispatch weight presence differs"
    if indices is not None:
        assert len(indices) == len(rows), "Index row count differs"
    if weights is not None:
        assert len(weights) == len(rows), "Weight row count differs"

    actual_keys, expected_keys = [], []
    experts = [
        rank * (case.experts // case.world_size) + expert for expert, count in enumerate(counts) for _ in range(count)
    ]
    for i, row in enumerate(rows):
        assert len(x[i]) == case.hidden, "Received hidden dimension differs"
        if expanded:
            key = (experts[i], tuple(x[i]), None if weights is None else weights[i])
            wanted = (row.expert, row.x, None if row.weights is None else row.weights[0])
        else:
            assert len(indices[i]) == case.topk, "Index width differs"
            if weights is not None:
                assert len(weights[i]) == case.topk, "Weight width differs"
            # Nonlocal slots have unspecified dispatch weight values upstream.
            # Check local weights and all indices; never rely on masked values.
            values = (
                None
                if weights is None
                else tuple(value if indices[i][slot] >= 0 else None for slot, value in enumerate(weights[i]))
            )
            expected_values = (
                None
                if row.weights is None
                else tuple(value if row.indices[slot] >= 0 else None for slot, value in enumerate(row.weights))
            )
            key = (tuple(x[i]), tuple(indices[i]), values)
            wanted = (row.x, row.indices, expected_values)
        actual_keys.append(key)
        expected_keys.append(wanted)
    # Allow intra-expert/source scheduling differences, preserve multiplicity.
    assert Counter(actual_keys) == Counter(expected_keys), "Dispatch routes/data/weights differ"


def assert_close(actual, expected, *, label="output", atol=0.0, rtol=0.0):
    """Reject wrong shape, nonfinite values, missing channels and wrong numbers."""
    if expected is None:
        assert actual is None, f"{label}: expected no value channel"
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)), f"{label}: expected a sequence"
        assert len(actual) == len(expected), f"{label}: shape differs"
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            assert_close(got, wanted, label=f"{label}[{index}]", atol=atol, rtol=rtol)
    else:
        assert isfinite(actual), f"{label}: nonfinite value"
        assert isclose(actual, expected, abs_tol=atol, rel_tol=rtol), f"{label}: {actual} != {expected}"
