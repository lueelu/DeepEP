"""Shared scalar numerical contract; no Torch/device operations."""
import math
import struct

FP32_MAX = float.fromhex('0x1.fffffep+127')


def normalize_clamp(value=None):
    if value is None:
        return FP32_MAX
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('activation_clamp must be None or a nonnegative number')
    if math.isnan(value) or value < 0:
        raise ValueError('activation_clamp must be nonnegative and not NaN')
    return struct.unpack('f', struct.pack('f', min(float(value), FP32_MAX)))[0]
