# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Host activation configuration shared by the public API and CPU reference."""

import math
import struct


def _positive_fp32(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite FP32 number")
    try:
        value = struct.unpack("f", struct.pack("f", float(value)))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise ValueError(f"{name} must be a positive finite FP32 number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite FP32 number")
    return value


def normalize_activation(activation="swiglu", activation_params=None):
    """Return (name, beta, optional linear_beta) in the native FP32 representation."""
    if activation not in ("swiglu", "situglu"):
        raise ValueError("activation must be 'swiglu' or 'situglu'")
    if activation_params is None:
        activation_params = {}
    if not isinstance(activation_params, dict):
        raise TypeError("activation_params must be a dictionary or None")
    allowed = {"beta", "linear_beta"} if activation == "situglu" else set()
    unknown = set(activation_params) - allowed
    if unknown:
        raise ValueError(f"Unsupported activation_params for {activation}: {list(unknown)!r}")
    if activation == "swiglu":
        return activation, 1.0, None
    beta = _positive_fp32("beta", activation_params.get("beta", 1.0))
    linear_beta = (
        _positive_fp32("linear_beta", activation_params["linear_beta"]) if "linear_beta" in activation_params else None
    )
    return activation, beta, linear_beta
