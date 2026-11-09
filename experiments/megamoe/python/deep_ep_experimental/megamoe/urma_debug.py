"""Bounded CPU diagnostics for snapshots of the real URMA device buffers.

This module never launches device work. Inputs must be completed CPU byte
snapshots; FP8 statistics describe unscaled E4M3 codes, and scale statistics
describe E8M0 powers of two. Unused self rows in gmm2_remote are excluded from
the segment checks. This is diagnostic evidence, not another accuracy policy.
"""

import json
from pathlib import Path

import numpy as np


FRAGMENT_DTYPE = np.dtype(
    [
        ("expert", "<u4"),
        ("segment_begin", "<u4"),
        ("row_begin", "<u8"),
        ("rows", "<u4"),
        ("segment_count", "<u4"),
        ("gmm1_tile_begin", "<u8"),
        ("gmm2_tile_begin", "<u8"),
    ]
)
SEGMENT_DTYPE = np.dtype(
    [
        ("source_rank", "<u4"),
        ("cv", "<u4"),
        ("receive_row", "<u8"),
        ("return_row", "<u8"),
        ("tile_row", "<u4"),
        ("rows", "<u4"),
        ("input_ready_index", "<u4"),
        ("reserved", "<u4"),
    ]
)
assert FRAGMENT_DTYPE.itemsize == SEGMENT_DTYPE.itemsize == 40

_CODE = np.arange(256, dtype=np.uint16)
_MAGNITUDE = _CODE & 127
_FP8 = np.where(
    _MAGNITUDE < 8,
    _MAGNITUDE * 2.0**-9,
    (1 + (_MAGNITUDE % 8) / 8) * np.exp2(_MAGNITUDE // 8 - 7.0),
).astype(np.float32)
_FP8 *= np.where(_CODE & 128, -1, 1)
_FP8[_MAGNITUDE == 127] = np.nan
_SCALE = np.concatenate((np.exp2(np.arange(255, dtype=np.float32) - 127), np.array([np.nan], dtype=np.float32)))
_FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
_DTYPES = {"bf16": "<u2", "fp8_e4m3": "u1", "e8m0": "u1", "fp32": "<f4", "int64": "<i8", "packed_fp4_e2m1": "u1"}


def _bytes(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    value = np.asarray(value)
    if value.dtype != np.uint8:
        raise ValueError(f"snapshot must contain uint8 bytes, got {value.dtype}")
    return np.ascontiguousarray(value).reshape(-1)


def source_weight_prefixes(raw, limit=1048576):
    """Keep bounded original CPU ND bytes, never copy prepared NPU NZ weights."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("weight prefix limit must be a positive integer")
    result = {}
    for layer, (weight, scale) in enumerate(raw, start=1):
        for name, value in ((f"source_weight{layer}_prefix", weight), (f"source_weight_scale{layer}_prefix", scale)):
            if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or not value.flags.c_contiguous:
                raise ValueError("weight sources must be contiguous CPU uint8 arrays")
            # A view of the prefix alone would retain the entire raw weight allocation.
            result[name] = value.reshape(-1)[:limit].copy()
    return result


def _decode(value, kind):
    if kind == "bf16":
        return (value.astype(np.uint32) << 16).view(np.float32)
    if kind == "fp8_e4m3":
        return _FP8[value]
    if kind == "e8m0":
        return _SCALE[value]
    if kind == "packed_fp4_e2m1":
        decoded = np.empty(value.shape + (2,), dtype=np.float32)
        decoded[..., 0], decoded[..., 1] = _FP4[value & 15], _FP4[value >> 4]
        return decoded.reshape(-1)
    return value


def _statistics(parts, kind):
    count = nonzero = finite = raw_nonzero = 0
    minimum = maximum = None
    absolute_sum = 0.0
    for part in parts:
        flat = np.asarray(part).reshape(-1)
        for offset in range(0, flat.size, 262144):
            raw = flat[offset : offset + 262144]
            values = _decode(raw, kind)
            valid = np.isfinite(values)
            good = values[valid]
            count += int(values.size)
            finite += int(np.count_nonzero(valid))
            nonzero += int(np.count_nonzero(good))
            raw_nonzero += int(np.count_nonzero(raw))
            if good.size:
                low, high = float(good.min()), float(good.max())
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
                absolute_sum += float(np.abs(good.astype(np.float64)).sum())
    return dict(
        elements=count,
        finite_elements=finite,
        nonfinite_elements=count - finite,
        nonzero_elements=nonzero,
        raw_nonzero_elements=raw_nonzero,
        all_zero=bool(count and finite == count and nonzero == 0),
        min=minimum,
        max=maximum,
        mean_abs=absolute_sum / finite if finite else None,
    )


def _difference(actual, expected):
    finite = np.isfinite(actual) & np.isfinite(expected)
    delta = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
    return dict(
        elements=int(actual.size),
        nonfinite_elements=int(actual.size - np.count_nonzero(finite)),
        different_elements=int(np.count_nonzero(actual != expected)),
        max_abs_error=float(delta.max(initial=0)),
        sum_abs_error=float(delta.sum()),
    )


def _bf16_round(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def _check_activation(arrays, *, fused_gate_up=False):
    """Condition the Golden activation on real GMM1; not hardware emulation."""
    from .reference import bf16, dequantize_input, quantize_input

    first = arrays["gmm1"]
    rows, intermediate = arrays["activation"].shape
    code_different = scale_different = 0
    difference = dict(elements=0, nonfinite_elements=0, different_elements=0, max_abs_error=0.0, sum_abs_error=0.0)
    examples = []
    for begin in range(0, rows, 32):
        end = min(rows, begin + 32)
        gmm1 = _decode(first[begin:end], "bf16")
        actual_codes = arrays["activation"][begin:end]
        actual_scales = arrays["activation_scale"][begin:end]
        if not np.isfinite(gmm1).all():
            raise ValueError(f"non-finite GMM1 in rows {begin}:{end}; conditional activation comparison skipped")
        if np.any((actual_codes & 127) == 127) or np.any(actual_scales == 255):
            raise ValueError(f"non-finite activation code/scale in rows {begin}:{end}; comparison skipped")
        if fused_gate_up:
            if intermediate % 128:
                raise ValueError("fused GMM1 snapshot requires I divisible by128")
            pairs = gmm1.reshape(end - begin, intermediate // 128, 2, 128)
            gate = pairs[:, :, 0, :].reshape(end - begin, intermediate)
            up = pairs[:, :, 1, :].reshape(end - begin, intermediate)
        else:
            gate, up = np.split(gmm1, 2, axis=-1)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # Keep this expression and the BF16/MX conversion identical to
            # reference.expert_forward; no independent rounding policy here.
            activated = bf16(gate / (1 + np.exp(np.clip(-gate, -80, 80))) * up)
            expected_codes, expected_scales = quantize_input(activated)
            expected = dequantize_input(expected_codes, expected_scales)
            actual = dequantize_input(actual_codes, actual_scales)
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError(f"non-finite dequantized activation in rows {begin}:{end}; comparison skipped")
        code_mask = actual_codes != expected_codes
        scale_mask = actual_scales != expected_scales
        code_different += int(np.count_nonzero(code_mask))
        scale_different += int(np.count_nonzero(scale_mask))
        current = _difference(actual, expected)
        for key in ("elements", "nonfinite_elements", "different_elements", "sum_abs_error"):
            difference[key] += current[key]
        difference["max_abs_error"] = max(difference["max_abs_error"], current["max_abs_error"])
        if len(examples) < 8:
            positions = np.flatnonzero(code_mask | np.repeat(scale_mask, 32, axis=-1))[: 8 - len(examples)]
            for position in positions:
                row, channel = divmod(int(position), intermediate)
                scale_channel = channel // 32
                examples.append(
                    dict(
                        row=begin + row,
                        channel=channel,
                        gate=float(gate[row, channel]),
                        up=float(up[row, channel]),
                        expected_code=int(expected_codes[row, channel]),
                        actual_code=int(actual_codes[row, channel]),
                        expected_scale=int(expected_scales[row, scale_channel]),
                        actual_scale=int(actual_scales[row, scale_channel]),
                        expected_value=float(expected[row, channel]),
                        actual_value=float(actual[row, channel]),
                    )
                )
    difference["mean_abs_error"] = (
        difference["sum_abs_error"] / difference["elements"] if difference["elements"] else 0.0
    )
    return dict(
        source="actual_gmm1_conditioned_cpu_golden",
        diagnostic_only=True,
        hardware_bitwise_emulation=False,
        rows=rows,
        chunk_rows=32,
        code_elements=rows * intermediate,
        scale_elements=rows * (intermediate // 32),
        code_different_elements=code_different,
        scale_different_elements=scale_different,
        exact_byte_match=code_different == 0 and scale_different == 0,
        dequantized_difference=difference,
        first_mismatches=examples,
    )


def save_snapshot(
    raw,
    directory,
    *,
    bs,
    hidden,
    intermediate,
    topk,
    received_rows,
    rank,
    receive_capacity_rows=None,
    fused_gate_up=False,
):
    """Save JSON statistics and small numeric samples; return the JSON report.

    Reduce is reconstructed in blocks of at most 64 tokens, one TopK term at
    a time. No full expanded FP32 matrix or copied expert-major input is made.
    Missing/malformed optional buffers are reported rather than hiding other
    available diagnostics. The caller decides whether to run another kernel.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dimensions = dict(
        bs=bs, hidden=hidden, intermediate=intermediate, topk=topk, received_rows=received_rows, rank=rank
    )
    if any(
        type(dimensions[key]) is not int or dimensions[key] <= 0 for key in ("bs", "hidden", "intermediate", "topk")
    ):
        raise ValueError("snapshot dimensions must be positive Python integers")
    if (
        type(received_rows) is not int
        or type(rank) is not int
        or received_rows < 0
        or rank < 0
        or hidden % 32
        or intermediate % 32
    ):
        raise ValueError("invalid receive count/rank or scale widths")
    report = dict(schema=1, dimensions=dimensions, stages={}, checks={}, flags={}, errors=[], findings=[])
    report["notes"] = [
        "FP8 data statistics are unscaled code values; E8M0 statistics are scale values.",
        "Weight prefixes decode two unscaled E2M1 values per stored byte; their reported shape is the byte storage shape.",
        "source_weight prefixes are original CPU ND inputs, not snapshots of the prepared device NZ weight storage.",
        "gmm2_remote has unused self rows; use remote_segment_output rather than its whole-buffer statistics.",
        "A local snapshot cannot compare remote received rows with the original other-rank inputs.",
        "CPU reduction reproduces route order and BF16 final rounding, not hardware bitwise arithmetic.",
        "CPU activation is conditioned on actual GMM1 in 32-row chunks using Golden BF16/MXFP8 semantics; "
        "it is not hardware bitwise emulation and never changes the accuracy verdict.",
    ]
    arrays, samples, byte_arrays = {}, {}, {}
    for key, value in raw.items():
        try:
            byte_arrays[key] = _bytes(value)
        except (TypeError, ValueError, RuntimeError) as error:
            report["errors"].append(f"{key}: {error}")
    receive_capacity_rows = received_rows if receive_capacity_rows is None else receive_capacity_rows
    shapes = {
        "x": ("fp8_e4m3", (bs, hidden)),
        "x_scale": ("e8m0", (bs, hidden // 32)),
        "received_x": ("fp8_e4m3", (receive_capacity_rows, hidden)),
        "received_scale": ("e8m0", (receive_capacity_rows, hidden // 32)),
        "gathered_input_scale": ("e8m0", (received_rows, hidden // 32)),
        "activation": ("fp8_e4m3", (received_rows, intermediate)),
        "activation_scale": ("e8m0", (received_rows, intermediate // 32)),
        "gmm2_remote": ("bf16", (receive_capacity_rows, hidden)),
        "returned": ("bf16", (bs * topk, hidden)),
        "output": ("bf16", (bs, hidden)),
        "weights": ("fp32", (bs, topk)),
        "reduce_index": ("int64", (bs, topk)),
        "order": ("int64", (bs * topk,)),
    }
    if "gmm1" in byte_arrays:
        shapes["gmm1"] = ("bf16", (received_rows, intermediate * 2))
    else:
        report["unavailable_stages"] = ["gmm1"]
        report["notes"].append(
            "Routed GMM1 snapshot is unavailable (FIX-to-UB has no GM shadow); conditioned activation comparison is skipped, not passed."
        )
    if "gmm2_contiguous" in byte_arrays:
        shapes["gmm2_contiguous"] = ("bf16", (received_rows, hidden))
    for name in (
        "weight1_prefix",
        "weight2_prefix",
        "weight_scale1_prefix",
        "weight_scale2_prefix",
        "source_weight1_prefix",
        "source_weight2_prefix",
        "source_weight_scale1_prefix",
        "source_weight_scale2_prefix",
    ):
        if name in byte_arrays:
            shapes[name] = ("e8m0" if "scale" in name else "packed_fp4_e2m1", (byte_arrays[name].size,))
    for name, (kind, shape) in shapes.items():
        try:
            data = byte_arrays[name]
            dtype = np.dtype(_DTYPES[kind])
            size = int(np.prod(shape))
            if data.size < size * dtype.itemsize:
                raise ValueError(f"need {size * dtype.itemsize} bytes, have {data.size}")
            value = data[: size * dtype.itemsize].view(dtype).reshape(shape)
            arrays[name] = value
            report["stages"][name] = dict(dtype=kind, shape=list(shape), **_statistics([value], kind))
            if name.startswith("source_weight"):
                report["stages"][name]["source"] = "original_cpu_nd"
            sample = value[: min(4, len(value))]
            samples[name] = _decode(sample, kind).copy()
        except (KeyError, ValueError) as error:
            report["errors"].append(f"{name}: {error}")

    if all(name in arrays for name in ("gmm1", "activation", "activation_scale")):
        try:
            report["stages"]["gmm1"]["layout"] = "gate128_up128" if fused_gate_up else "planar"
            report["checks"]["cpu_activation_from_gmm1"] = _check_activation(arrays, fused_gate_up=fused_gate_up)
        except (ValueError, FloatingPointError) as error:
            report["errors"].append(f"cpu_activation_from_gmm1: {error}")

    for name in ("gmm1_done", "input_scale_done", "activation_done", "gmm2_done", "gmm2_matmul_done", "local_flags"):
        if name not in byte_arrays:
            continue
        data = byte_arrays[name]
        if data.size % 128:
            report["errors"].append(f"{name}: flags must use 128-byte slots")
            continue
        counters = data.reshape(-1, 128)[:, :4].copy().view("<i4").reshape(-1)
        report["flags"][name] = dict(**_statistics([counters], "int64"), first_values=counters[:32].tolist())
        samples[name] = counters[:64].copy()
    if "local_flags" in report["flags"]:
        values = report["flags"]["local_flags"]["first_values"]
        report["flags"]["local_named"] = dict(zip(("dispatch_done", "aic_done", "incoming_done"), values[:3]))
        if "gmm2_matmul_done" in byte_arrays and len(values) >= 5:
            report["flags"]["local_named"].update(pack_done=values[3], reorder_done=values[4])

    if "order" in arrays:
        order = arrays["order"]
        report["checks"]["order_permutation"] = bool(np.array_equal(np.sort(order), np.arange(bs * topk)))
    if "reduce_index" in arrays:
        inverse = arrays["reduce_index"]
        valid_inverse = bool(np.all((inverse >= 0) & (inverse < bs * topk)))
        report["checks"]["reduce_indices_in_range"] = valid_inverse
        if valid_inverse and "order" in arrays:
            report["checks"]["order_inverse_match"] = bool(
                np.array_equal(arrays["order"][inverse.reshape(-1)], np.arange(bs * topk))
            )
    else:
        valid_inverse = False

    if valid_inverse and all(name in arrays for name in ("returned", "weights", "output")):
        error = dict(elements=0, nonfinite_elements=0, different_elements=0, max_abs_error=0.0, sum_abs_error=0.0)
        cpu_nonzero = 0
        for begin in range(0, bs, 64):
            end = min(bs, begin + 64)
            accumulator = np.zeros((end - begin, hidden), dtype=np.float32)
            for slot in range(topk):
                term = _decode(arrays["returned"][inverse[begin:end, slot]], "bf16")
                accumulator += term * arrays["weights"][begin:end, slot, None]
            expected = _bf16_round(accumulator)
            actual = _decode(arrays["output"][begin:end], "bf16")
            current = _difference(actual, expected)
            for key in ("elements", "nonfinite_elements", "different_elements", "sum_abs_error"):
                error[key] += current[key]
            error["max_abs_error"] = max(error["max_abs_error"], current["max_abs_error"])
            cpu_nonzero += int(np.count_nonzero(expected[np.isfinite(expected)]))
            if begin == 0:
                samples["cpu_reduce"] = expected[:4].copy()
        error.update(
            cpu_nonzero_elements=cpu_nonzero,
            output_nonzero_elements=report["stages"]["output"]["nonzero_elements"],
            exact_match=error["different_elements"] == 0 and error["nonfinite_elements"] == 0,
            mean_abs_error=error["sum_abs_error"] / error["elements"],
        )
        report["checks"]["cpu_reduce_vs_output"] = error
        if cpu_nonzero and report["stages"]["output"]["all_zero"]:
            report["findings"].append(
                "RETURN_NONZERO_OUTPUT_ZERO: returned rows and weights produce nonzero CPU Reduce; inspect device Reduce."
            )

    if "fragments" in byte_arrays and "segments" in byte_arrays:
        try:
            fragments = byte_arrays["fragments"].view(FRAGMENT_DTYPE)
            segments = byte_arrays["segments"].view(SEGMENT_DTYPE)
            samples["fragments"] = fragments[:8].copy()
            samples["segments"] = segments[:16].copy()
            _check_segments(report, samples, arrays, fragments, segments, bs, topk, received_rows, rank)
        except (ValueError, IndexError, OverflowError) as error:
            report["errors"].append(f"segments: {error}")

    stages = report["stages"]
    for previous, current in (("received_x", "gmm1"), ("gmm1", "activation"), ("activation", "returned")):
        if (
            previous in stages
            and current in stages
            and stages[previous]["nonzero_elements"]
            and stages[current]["all_zero"]
        ):
            report["findings"].append(
                f"ZERO_TRANSITION: {previous} is nonzero, {current} is all zero; inspect the intervening stages."
            )
    if stages.get("returned", {}).get("all_zero"):
        report["findings"].append(
            "RETURN_ALL_ZERO: Reduce inputs are already zero; output zero alone does not implicate Reduce."
        )
    report["sample_path"] = str(directory / "debug_samples.npz")
    np.savez_compressed(report["sample_path"], **samples)
    (directory / "debug_intermediates.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _check_segments(report, samples, arrays, fragments, segments, bs, topk, received_rows, rank):
    expert_coverage = np.zeros(received_rows, dtype=np.uint16)
    receive_capacity = (
        arrays["received_x"].shape[0]
        if "received_x" in arrays
        else max((int(s["receive_row"]) + int(s["rows"]) for s in segments), default=0)
    )
    receive_coverage = np.zeros(receive_capacity, dtype=np.uint16)
    self_rows = scale_mismatch = self_x_mismatch = self_scale_mismatch = 0
    remote_parts, gathered_x_parts = [], []
    for fragment in fragments:
        begin, count = int(fragment["segment_begin"]), int(fragment["segment_count"])
        row_begin, group_rows = int(fragment["row_begin"]), int(fragment["rows"])
        if begin + count > len(segments) or row_begin + group_rows > received_rows:
            raise ValueError("fragment exceeds segment or compute row allocation")
        for segment in segments[begin : begin + count]:
            source, receive, returned, tile, rows = map(
                int,
                (
                    segment["source_rank"],
                    segment["receive_row"],
                    segment["return_row"],
                    segment["tile_row"],
                    segment["rows"],
                ),
            )
            if tile + rows > group_rows or receive + rows > receive_capacity or returned + rows > bs * topk:
                raise ValueError("segment exceeds compute/receive/return row allocation")
            target = row_begin + tile
            expert_coverage[target : target + rows] += 1
            receive_coverage[receive : receive + rows] += 1
            if "received_x" in arrays:
                gathered_x_parts.append(arrays["received_x"][receive : receive + rows])
            if all(name in arrays for name in ("received_scale", "gathered_input_scale")):
                scale_mismatch += int(
                    np.count_nonzero(
                        arrays["received_scale"][receive : receive + rows]
                        != arrays["gathered_input_scale"][target : target + rows]
                    )
                )
            if source == rank:
                self_rows += rows
                if "order" in arrays:
                    token = arrays["order"][returned : returned + rows] // topk
                    if np.any((token < 0) | (token >= bs)):
                        raise ValueError("self Pack order refers to an invalid input token")
                    if "x" in arrays and "received_x" in arrays:
                        self_x_mismatch += int(
                            np.count_nonzero(arrays["received_x"][receive : receive + rows] != arrays["x"][token])
                        )
                    if "x_scale" in arrays and "received_scale" in arrays:
                        self_scale_mismatch += int(
                            np.count_nonzero(
                                arrays["received_scale"][receive : receive + rows] != arrays["x_scale"][token]
                            )
                        )
            elif "gmm2_remote" in arrays:
                remote_parts.append(arrays["gmm2_remote"][receive : receive + rows])
    report["checks"]["segments"] = dict(
        fragment_count=len(fragments),
        segment_count=len(segments),
        expert_rows_covered_once=bool(np.all(expert_coverage == 1)),
        receive_rows_covered_once=bool(
            np.count_nonzero(receive_coverage) == received_rows and np.all(receive_coverage <= 1)
        ),
        self_rows=self_rows,
        self_pack_x_mismatched_bytes=self_x_mismatch
        if all(k in arrays for k in ("order", "x", "received_x"))
        else None,
        self_pack_scale_mismatched_bytes=self_scale_mismatch
        if all(k in arrays for k in ("order", "x_scale", "received_scale"))
        else None,
        gathered_scale_mismatched_bytes=scale_mismatch
        if all(k in arrays for k in ("received_scale", "gathered_input_scale"))
        else None,
    )
    if gathered_x_parts:
        report["checks"]["expert_major_received_x"] = _statistics(gathered_x_parts, "fp8_e4m3")
        samples["expert_major_received_x"] = _decode(gathered_x_parts[0][:4], "fp8_e4m3").copy()
    if remote_parts:
        report["checks"]["remote_segment_output"] = _statistics(remote_parts, "bf16")
