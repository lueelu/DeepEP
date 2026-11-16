# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""DeepGEMM-style BF16 training entry with PyTorch autograd."""

from ctypes import c_float
import math

import torch

from deep_ep._runtime import integer

from .bf16 import MegaMoeBF16Train, _MegaMoeBF16TrainFunction

# Both entry styles share one Runtime/Session and the same native executor.
SymmBuffer = MegaMoeBF16Train


def bf16_mega_moe_train(
    y: torch.Tensor,
    l1_weights: torch.Tensor,
    l2_weights: torch.Tensor,
    sym_buffer: SymmBuffer,
    shared_l1_weights: torch.Tensor | None = None,
    shared_l2_weights: torch.Tensor | None = None,
    cumulative_local_expert_recv_stats: torch.Tensor | None = None,
    activation: str = "swiglu",
    activation_clamp: float | None = None,
    fast_math: bool = True,
    *,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    max_recv_tokens: int,
    swiglu_limit: float = 0.0,
    save_gmm1_output: str = "bf16",
) -> None:
    """Write BF16 ``y`` in place and attach its training graph; return None.

    The positional arguments follow DeepGEMM's BF16 MegaMoE API. Training
    inputs are explicit keyword arguments so gradients reach the original
    ``x`` and ``topk_weights`` tensors. The latter may be BF16 or FP32;
    its gradient has the same dtype. Expert weights use the training layouts
    [E,2F,H]/[E,H,F], or [E,H,2F]/[E,F,H].

    ``sym_buffer`` uses the existing constructor (group, local_mem_size,
    timeout=120, debug_collectives=False) and collective ``close()``.
    Use a fresh, contiguous BF16 output per forward until its graph is no
    longer needed. Backward reads per-call saved state, not the output.

    ``activation_clamp`` is an alias for ``swiglu_limit``. None/0 disables
    clipping; positive L upper-clamps Gate and clamps Up to [-L,L]. Shared
    experts and fast_math=False are not supported. Counts, if requested,
    accumulate into local-expert int32 ``cumulative_local_expert_recv_stats``.
    """
    if not isinstance(sym_buffer, MegaMoeBF16Train):
        raise TypeError("sym_buffer must be a training SymmBuffer or MegaMoeBF16Train")
    sym_buffer._check_ready()
    if shared_l1_weights is not None or shared_l2_weights is not None:
        raise NotImplementedError("BF16 training supports routed experts only")
    if activation != "swiglu" or fast_math is not True:
        raise NotImplementedError("BF16 training supports activation='swiglu' and fast_math=True")
    integer("max_recv_tokens", max_recv_tokens, 1, (1 << 31) - 1)
    limit = float(swiglu_limit)
    if activation_clamp is not None:
        clamp = float(activation_clamp)
        if limit != 0.0 and limit != clamp:
            raise ValueError("activation_clamp and swiglu_limit must agree when both are supplied")
        limit = clamp
    if not math.isfinite(limit) or not math.isfinite(c_float(limit).value):
        raise ValueError("swiglu_limit must be finite and representable as float32")
    if save_gmm1_output not in ("bf16", "none", "mxfp8", "auto"):
        raise ValueError("save_gmm1_output must be bf16, none, mxfp8 or auto")
    if y.dtype != torch.bfloat16 or y.shape != x.shape or y.device != x.device or not y.is_contiguous():
        raise ValueError("y must be contiguous BF16 with the same shape and device as x")
    if y.requires_grad:
        raise ValueError("y must be a fresh output without an existing autograd graph")
    inputs = (x, topk_idx, topk_weights, l1_weights, l2_weights)
    if any(y.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() for value in inputs):
        raise ValueError("y must not share storage with any input")
    stats = cumulative_local_expert_recv_stats
    if stats is not None:
        if (
            stats.dtype != torch.int32
            or tuple(stats.shape) != (l1_weights.shape[0],)
            or stats.device != y.device
            or not stats.is_contiguous()
        ):
            raise ValueError("Cumulative stats must be contiguous local-expert int32 on the output device")
        if any(stats.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() for value in (y, *inputs)):
            raise ValueError("Cumulative stats must not share storage with inputs or output")
    _, counts = _MegaMoeBF16TrainFunction.apply(
        sym_buffer, x, topk_idx, topk_weights, l1_weights, l2_weights, max_recv_tokens, limit, save_gmm1_output, y
    )
    if stats is not None:
        stats.add_(counts)
