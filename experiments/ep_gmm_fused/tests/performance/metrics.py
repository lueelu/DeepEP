# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

import math
import statistics


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(rank_samples, *, sample_semantics="max_rank_batch_mean_us"):
    """Match sample indices first, then reduce MAX across ranks before quantiles."""
    if not rank_samples or not rank_samples[0] or len({len(x) for x in rank_samples}) != 1:
        raise ValueError("rank sample arrays must be nonempty and have equal length")
    if any(not math.isfinite(v) or v <= 0 for rank in rank_samples for v in rank):
        raise ValueError("timings must be finite and positive")
    samples = [max(sample) for sample in zip(*rank_samples)]
    return {
        "sample_semantics": sample_semantics,
        "samples_us": samples,
        "mean_us": statistics.mean(samples),
        "p50_us": quantile(samples, 0.5),
        "p90_us": quantile(samples, 0.9),
        "p99_us": quantile(samples, 0.99),
        "min_us": min(samples),
        "max_us": max(samples),
    }
