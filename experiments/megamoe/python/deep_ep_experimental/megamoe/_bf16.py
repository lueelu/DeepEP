# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""BF16 validation, collective execution and per-call autograd state."""

from ctypes import c_float
from dataclasses import dataclass
import math

import torch
from ._autograd import BF16Function

from deep_ep._native import npu_api, require_native
from deep_ep._runtime import integer


@dataclass(frozen=True)
class Call:
    max_recv: int
    limit: float
    save_type: str


def validate_buffer_inputs(b, bs):
    for name, tensor, dtype, width in (
        ("x", b.x, torch.bfloat16, b.hidden),
        ("topk_idx", b.topk_idx, torch.int32, b.num_topk),
        ("topk_weights", b.topk_weights, None, b.num_topk),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[0] < bs
            or tensor.shape[0] > b.num_max_tokens_per_rank
            or tensor.shape[1] != width
            or tensor.device != b._device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous [rows,{width}] on the buffer device, within source capacity")
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f"{name} has unsupported dtype")
    if b.topk_weights.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("topk_weights must be bfloat16 or float32")


def prepare_plan(b, bs, limit, save_type):
    from .api import _extension

    mode = b.gmm1_save_type if save_type is None else save_type
    if not b.need_grad:
        mode = "none"
    key = (
        bs,
        b.hidden,
        b.intermediate_hidden,
        b.num_topk,
        b.num_experts,
        b.max_recv_tokens,
        b.topk_weights.dtype,
        b.need_grad,
        mode,
        limit,
    )
    if b._plan_key != key:
        native = require_native("BF16 MegaMoE")
        address = native.megamoe_workspace_address(b._runtime.native)
        plan = _extension().BF16Plan(
            address, b._runtime.native.capacity, b._runtime.rank, b._runtime.world_size, b.x[:bs], b.need_grad
        )
        plan.prepare(
            bs,
            b.hidden,
            b.intermediate_hidden,
            b.num_topk,
            b.num_experts // b.group.size(),
            b.max_recv_tokens,
            limit,
            mode,
            b.topk_weights.dtype == torch.float32,
        )
        b._plan, b._plan_key, b._plan_bs = plan, key, bs
    return b._plan


def run(y, w1, w2, b, shared_w1, shared_w2, stats, activation, clamp, fast_math):
    from .api import SymmBuffer

    if not isinstance(b, SymmBuffer) or b.mma_type != "bf16xbf16":
        raise TypeError("bf16_mega_moe requires a BF16 SymmBuffer")
    if type(b.need_grad) is not bool:
        raise TypeError("need_grad must be bool")
    if b.gmm1_save_type not in ("bf16", "mxfp8", "none"):
        raise ValueError("gmm1_save_type must be bf16, mxfp8 or none")
    integer("max_recv_tokens", b.max_recv_tokens, 1, (1 << 31) - 1)
    if shared_w1 is not None or shared_w2 is not None:
        raise NotImplementedError("BF16 shared experts are not implemented")
    if activation != "swiglu" or fast_math is not True:
        raise NotImplementedError("BF16 supports SwiGLU and the default math configuration")
    limit = 0.0 if clamp is None else float(clamp)
    if limit < 0 or not math.isfinite(limit) or not math.isfinite(c_float(limit).value):
        raise ValueError("activation_clamp must be a finite nonnegative float or None")
    if not isinstance(y, torch.Tensor) or y.ndim != 2 or y.dtype != torch.bfloat16 or y.device != b._device:
        raise ValueError("y must be BF16 [M,H] on the buffer device")
    bs = y.shape[0]
    if not 0 < bs <= b.num_max_tokens_per_rank or y.shape[1] != b.hidden or not y.is_contiguous():
        raise ValueError("y must be contiguous [M,H] within the source capacity")
    if y.requires_grad:
        raise ValueError("y must not require gradients before this call")
    validate_buffer_inputs(b, bs)
    x, ids, p = b.x[:bs], b.topk_idx[:bs], b.topk_weights[:bs]
    e, f, h = b.num_experts // b.group.size(), b.intermediate_hidden, b.hidden
    canonical = tuple(w1.shape) == (e, 2 * f, h) and tuple(w2.shape) == (e, h, f)
    legacy = tuple(w1.shape) == (e, h, 2 * f) and tuple(w2.shape) == (e, f, h)
    if not (canonical or legacy):
        raise ValueError("BF16 weights must match [E,2F,H]/[E,H,F] or [E,H,2F]/[E,F,H]")
    if any(t.dtype != torch.bfloat16 or t.device != b._device or not t.is_contiguous() for t in (w1, w2)):
        raise ValueError("BF16 weights must be contiguous on the buffer device")
    if any(y.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in (x, ids, p, w1, w2)):
        raise ValueError("y must not overlap BF16 inputs")
    if stats is not None:
        if (
            stats.dtype != torch.int32
            or stats.device != b._device
            or tuple(stats.shape) != (e,)
            or not stats.is_contiguous()
        ):
            raise ValueError("Cumulative stats must be contiguous local-expert int32 on the buffer device")
        if any(stats.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in (x, ids, p, w1, w2, y)):
            raise ValueError("Cumulative stats must not overlap inputs or output")
    gradients = torch.is_grad_enabled() and any(t.requires_grad for t in (x, p, w1, w2))
    if gradients and not b.need_grad:
        raise ValueError("Use a SymmBuffer with need_grad=True for differentiable BF16 calls")
    mode = b.gmm1_save_type if gradients else "none"
    config = Call(b.max_recv_tokens, limit, mode)
    with b._lock:
        runtime = b._runtime
        runtime.ensure_alive()
        caller = npu_api().current_stream(runtime.device)
        runtime.stream.wait_stream(caller)
        with npu_api().stream(runtime.stream):
            plan = prepare_plan(b, bs, limit, mode)
            if gradients:
                result_y, counts = BF16Function.apply(b, plan, config, x, ids, p, w1, w2, y)
            else:
                capture = getattr(b, "_timer_capture", None)
                if capture is not None:
                    capture.claim()
                result = plan.forward(y, x, ids, p, w1, w2, config.max_recv, limit, mode, capture is not None, False)
                if capture is not None:
                    capture.forward = result.timer
                result_y, counts = result.y, result.counts
            if stats is not None:
                stats.add_(counts)
                stats.record_stream(runtime.stream)
        caller.wait_stream(runtime.stream)
        if caller != runtime.stream:
            result_y.record_stream(caller)
