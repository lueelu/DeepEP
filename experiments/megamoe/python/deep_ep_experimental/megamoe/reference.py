# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Small NumPy oracle; no custom operators and no communication implementation.

Finite test data only. BF16/GMM output rounding and MXFP8 intermediate rounding
are modeled; this is not a bitwise emulator of Cube accumulation or exp.
"""

import numpy as np

FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
_code = np.arange(127)
FP8_POSITIVE = np.where(_code < 8, _code * 2.0**-9, (1 + (_code % 8) / 8) * np.exp2(_code // 8 - 7)).astype(np.float32)


def bf16(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def fp8_decode(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    magnitude = codes & 127
    if np.any(magnitude == 127):
        raise ValueError("FP8 NaN is outside the finite test contract")
    return FP8_POSITIVE[magnitude] * np.where(codes & 128, -1.0, 1.0).astype(np.float32)


def fp8_encode(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("FP8 conversion requires finite values")
    value = np.minimum(np.abs(x), 448)
    upper = np.searchsorted(FP8_POSITIVE, value).clip(0, 126)
    lower = np.maximum(upper - 1, 0)
    dl, du = value - FP8_POSITIVE[lower], FP8_POSITIVE[upper] - value
    choose_low = (dl < du) | ((dl == du) & ((lower % 2) == 0))
    code = np.where(choose_low, lower, upper).astype(np.uint8)
    return code | (np.signbit(x).astype(np.uint8) << 7)


def quantize_input(x):
    """ComputeMaxExp/ComputeScale: floor(log2(block_amax))-8, block size 32."""
    x = bf16(x)
    if x.shape[-1] % 32 or not np.isfinite(x).all():
        raise ValueError("Finite inputs and K divisible by 32 are required")
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    maximum = np.abs(blocks).max(axis=-1)
    exponent = np.maximum(np.floor(np.log2(np.maximum(maximum, 2.0**-119))) - 8, -127)
    scale = np.exp2(exponent).astype(np.float32)
    codes = fp8_encode((blocks / scale[..., None]).reshape(x.shape))
    return codes, (exponent + 127).astype(np.uint8)


def dequantize_input(codes, scales):
    scale = np.exp2(np.asarray(scales, dtype=np.int16) - 127).astype(np.float32)
    return fp8_decode(codes) * np.repeat(scale, 32, axis=-1)


def unpack_weight(pair):
    packed, scales = pair
    packed = np.asarray(packed, dtype=np.uint8)
    values = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.float32)
    values[..., ::2], values[..., 1::2] = FP4[packed & 15], FP4[packed >> 4]
    shape = (*values.shape[:-1], values.shape[-1] // 32)
    scale = np.exp2(np.asarray(scales, dtype=np.int16).reshape(shape) - 127).astype(np.float32)
    return values * np.repeat(scale, 32, axis=-1)


def make_weights(experts, hidden, intermediate, seed):
    """Generate packed FP4 and non-uniform E8M0 scales; never quantize weights on the hot path."""
    rng = np.random.default_rng(seed)
    result = []
    for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
        packed = rng.integers(0, 256, (experts, n, k // 2), dtype=np.uint8)
        exponent = int(np.floor(-0.5 * np.log2(k))) - 2
        scales = rng.integers(127 + exponent - 1, 127 + exponent + 2, (experts, n, k // 64, 2), dtype=np.uint8)
        result.append((packed, scales))
    return tuple(result)


def expert_forward(x, w1, w2):
    first = bf16(x @ w1.T)
    gate, up = np.split(first, 2, axis=-1)
    activated = bf16(gate / (1 + np.exp(np.clip(-gate, -80, 80))) * up)
    quantized = dequantize_input(*quantize_input(activated))
    return bf16(quantized @ w2.T)


def local_reference(x, indices, weights, l1, l2, first_expert):
    """One rank's contribution to one source rank. l1/l2 contain local experts only."""
    result = np.zeros_like(x, dtype=np.float32)
    for local in range(len(l1)):
        token, slot = np.where(indices == first_expert + local)
        if token.size:
            contribution = expert_forward(x[token], l1[local], l2[local])
            np.add.at(result, token, contribution * weights[token, slot, None])
    return result


def error_metrics(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("Reference and result shapes differ")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"finite": False, "relative_rmse": None, "max_abs": None}
    diff = actual.astype(np.float64) - expected
    denominator = max(float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), 1e-12)
    return {
        "finite": True,
        "relative_rmse": float(np.sqrt(np.mean(diff**2))) / denominator,
        "max_abs": float(np.abs(diff).max(initial=0)),
    }
