# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Precision policies, upstream dtype emulation and dual-reference comparison.

The acceptance ratios follow ``昇腾算子精度标准2.1``.  CPU float arithmetic is
the high-precision golden.  The second reference emulates the dtype path used
by the pinned DeepSeek DeepEP tests: BF16 activations, or per-token/per-128
E4M3FN values and FP32 scales converted back to BF16 before combine.
"""

from bisect import bisect_left
from dataclasses import asdict, dataclass
from math import fsum, isfinite, sqrt
from random import Random
import struct


LEVEL_LIMITS = {
    "L0": (10.0, 2.0, 2.0),
    "L1": (5.0, 1.5, 1.5),
    "L2": (2.0, 1.2, 1.2),
}

DTYPE_POLICY = {
    "bfloat16": {"small_threshold": 2**-8, "small_error": 2**-16},
    "float8_e4m3fn": {"small_threshold": 2**-4, "small_error": 2**-6},
}


@dataclass(frozen=True)
class ErrorMetrics:
    mare: float
    mere: float
    rmse: float
    small_error_count: int
    normal_count: int
    small_count: int


@dataclass(frozen=True)
class AccuracyReport:
    dtype: str
    level: str
    actual: ErrorMetrics
    benchmark: ErrorMetrics
    ratios: tuple
    small_error_ratio: float
    passed: bool

    def to_dict(self):
        result = asdict(self)
        result["ratios"] = dict(zip(("mare", "mere", "rmse"), self.ratios))
        return result


@dataclass(frozen=True)
class RecheckReport:
    sample_count: int
    bootstrap_samples: int
    median: float
    ci_lower: float
    ci_upper: float
    passed: bool
    reason: str


def _float32(value):
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def quantize_bfloat16(value):
    """Round a finite Python value to BF16, using round-to-nearest-even."""
    bits = struct.unpack(">I", struct.pack(">f", _float32(value)))[0]
    upper = bits >> 16
    lower = bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper = (upper + 1) & 0xFFFF
    return struct.unpack(">f", struct.pack(">I", upper << 16))[0]


def _decode_e4m3fn(bits):
    sign = -1.0 if bits & 0x80 else 1.0
    exponent, mantissa = (bits >> 3) & 0xF, bits & 0x7
    if exponent == 0:
        value = mantissa * 2**-9
    elif exponent == 0xF:
        if mantissa == 0x7:
            return None
        value = (1.0 + mantissa / 8.0) * 2**8
    else:
        value = (1.0 + mantissa / 8.0) * 2 ** (exponent - 7)
    return sign * value


_E4M3_POSITIVE = sorted((value, bits) for bits in range(0x80) for value in (_decode_e4m3fn(bits),) if value is not None)
_E4M3_VALUES = [item[0] for item in _E4M3_POSITIVE]


def quantize_e4m3fn(value):
    """Round to the finite E4M3FN value set, saturating at +/-448."""
    if not isfinite(value):
        raise ValueError("E4M3FN golden generation requires finite inputs")
    sign = -1.0 if value < 0 else 1.0
    magnitude = min(abs(float(value)), 448.0)
    index = bisect_left(_E4M3_VALUES, magnitude)
    choices = _E4M3_POSITIVE[max(0, index - 1) : min(len(_E4M3_POSITIVE), index + 1)]
    best_value, best_bits = min(
        choices,
        key=lambda item: (abs(item[0] - magnitude), item[1] & 1, item[1]),
    )
    return sign * best_value


def encode_upstream_fp8_row(row):
    """Return the pinned upstream E4M3FN payload values and FP32 scales."""
    # The pinned test creates x as BF16 before applying per-token FP8 cast.
    row = tuple(quantize_bfloat16(value) for value in row)
    encoded, scales = [], []
    for start in range(0, len(row), 128):
        group = row[start : start + 128]
        amax = max((abs(value) for value in group), default=0.0)
        amax = max(amax, 1e-4)
        scale = _float32(amax / 448.0)
        scales.append(scale)
        for value in group:
            encoded.append(quantize_e4m3fn(value * (448.0 / amax)))
    return tuple(encoded), tuple(scales)


def emulate_upstream_row(row, dtype):
    if dtype == "bfloat16":
        return tuple(quantize_bfloat16(value) for value in row)
    if dtype != "float8_e4m3fn":
        raise ValueError(f"Unsupported upstream dtype: {dtype}")
    encoded, scales = encode_upstream_fp8_row(row)
    restored = []
    for start in range(0, len(encoded), 128):
        scale = scales[start // 128]
        for quantized in encoded[start : start + 128]:
            restored.append(quantize_bfloat16(_float32(quantized * scale)))
    return tuple(restored)


def emulate_upstream_case(case):
    return tuple(tuple(emulate_upstream_row(row, case.dtype) for row in rank) for rank in case.x)


def _flatten(values):
    if isinstance(values, (tuple, list)):
        result = []
        for value in values:
            result.extend(_flatten(value))
        return result
    return [float(values)]


def error_metrics(actual, golden, *, dtype):
    actual_values, golden_values = _flatten(actual), _flatten(golden)
    if len(actual_values) != len(golden_values):
        raise ValueError("actual and golden shapes differ")
    if not all(isfinite(value) for value in actual_values + golden_values):
        raise ValueError("precision comparison requires finite values")
    policy = DTYPE_POLICY[dtype]
    normal_errors, relative_errors, small_error_count, small_count = [], [], 0, 0
    for got, wanted in zip(actual_values, golden_values):
        error = abs(got - wanted)
        if abs(wanted) < policy["small_threshold"]:
            small_count += 1
            small_error_count += error > policy["small_error"]
        else:
            normal_errors.append(error)
            relative_errors.append(error / (abs(wanted) + 1e-7))
    return ErrorMetrics(
        max(relative_errors, default=0.0),
        # Built-in float sum changed in Python 3.12. Use an explicit accurate
        # reduction so the checked-in metrics also reproduce on Python 3.11.
        fsum(relative_errors) / len(relative_errors) if relative_errors else 0.0,
        (sqrt(fsum(error * error for error in normal_errors) / len(normal_errors)) if normal_errors else 0.0),
        small_error_count,
        len(normal_errors),
        small_count,
    )


def compare_dual(actual, benchmark, golden, *, dtype, level="L2"):
    """Compare NPU output to CPU golden relative to the upstream benchmark."""
    if dtype not in DTYPE_POLICY or level not in LEVEL_LIMITS:
        raise ValueError("unknown dtype or precision level")
    got = error_metrics(actual, golden, dtype=dtype)
    ref = error_metrics(benchmark, golden, dtype=dtype)
    floor = DTYPE_POLICY[dtype]["small_error"]
    ratios = tuple(getattr(got, name) / max(getattr(ref, name), floor) for name in ("mare", "mere", "rmse"))
    small_ratio = got.small_error_count / max(ref.small_error_count, 1)
    limits = LEVEL_LIMITS[level]
    return AccuracyReport(
        dtype,
        level,
        got,
        ref,
        ratios,
        small_ratio,
        all(value <= limit for value, limit in zip(ratios, limits)) and small_ratio <= 2.0,
    )


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    if fraction == 0:
        return ordered[lower]
    return ordered[lower] * (1.0 - fraction) + ordered[lower + 1] * fraction


def bootstrap_median_recheck(ratios, *, bootstrap_samples=2000, seed=0):
    """Apply the standard's small-sample fuse and median 95% bootstrap CI."""
    values = tuple(float(value) for value in ratios)
    if not values or not all(isfinite(value) and value >= 0 for value in values):
        raise ValueError("recheck ratios must be nonempty, finite and nonnegative")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    median = _median(values)
    if len(values) < 200:
        return RecheckReport(
            len(values),
            bootstrap_samples,
            median,
            float("nan"),
            float("nan"),
            False,
            "sample_count_below_200",
        )
    random = Random(seed)
    medians = sorted(_median([values[random.randrange(len(values))] for _ in values]) for _ in range(bootstrap_samples))
    lower, upper = _quantile(medians, 0.025), _quantile(medians, 0.975)
    passed = lower <= 1.0
    return RecheckReport(
        len(values),
        bootstrap_samples,
        median,
        lower,
        upper,
        passed,
        "ci_includes_or_below_1" if passed else "ci_lower_above_1",
    )
