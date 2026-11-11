# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026
# Modified by xixing 2026

"""BF16 autograd bridge and explicit lifetime for the shared SHMEM owner."""

from ctypes import c_float
import math

import torch
from torch.autograd.function import once_differentiable

from deep_ep._native import npu_api, require_native
from deep_ep._runtime import Runtime, integer
from deep_ep._store import StoreGroup


class _TrainRuntime(Runtime):
    _megamoe_train = True

    def validate_topk_weight_type(self, dtype, stage):
        """Check dtype locally, with optional cross-rank dtype and phase checks."""
        code = {torch.bfloat16: 0, torch.float32: 1}.get(dtype, -1)
        if not self.debug_collectives:
            if code < 0:
                raise ValueError("topk_weights must be bfloat16 or float32")
            return
        local_values = [stage, code]
        if isinstance(self.group, StoreGroup):
            values = self.group.all_gather(local_values, "training TopK weight configuration")
        else:
            import torch.distributed as dist

            backend = str(dist.get_backend(self.group))
            with torch.inference_mode(False), torch.no_grad():
                local = torch.tensor(
                    local_values, dtype=torch.int64, device=f"npu:{self.device}" if backend == "hccl" else "cpu"
                )
                peers = [torch.empty_like(local) for _ in range(self.world_size)]
            dist.all_gather(peers, local, group=self.group)
            values = [value.cpu().tolist() for value in peers]
        if any(value != values[0] for value in values):
            raise ValueError("EP ranks must use the same forward/backward order and topk_weights dtype")
        if code < 0:
            raise ValueError("topk_weights must be bfloat16 or float32")


class _MegaMoeBF16TrainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, ids, weights, w1, w2, max_recv, limit, mode, out):
        op._check_ready()
        op._runtime.validate_topk_weight_type(weights.dtype, 0)
        if out is None:
            result = op._executor.forward(x, ids, weights, w1, w2, max_recv, limit, mode)
        else:
            result = op._executor.forward_out(out, x, ids, weights, w1, w2, max_recv, limit, mode)
            ctx.mark_dirty(out)
        ctx.op = op
        ctx.max_recv, ctx.limit, ctx.mode = max_recv, limit, mode
        ctx.save_for_backward(weights, w1, w2, result.counts, result.saved_state, result.dispatch_output)
        ctx.mark_non_differentiable(result.counts)
        return result.y, result.counts

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y, grad_counts=None):
        weights, w1, w2, counts, saved, dispatch = ctx.saved_tensors
        op = ctx.op
        op._check_ready()
        op._runtime.validate_topk_weight_type(weights.dtype, 1)
        result = op._executor.backward(
            grad_y.contiguous(),
            weights,
            w1,
            w2,
            counts,
            saved,
            dispatch,
            ctx.max_recv,
            ctx.limit,
            ctx.mode,
        )
        return None, result.dx, None, result.dtopk_weights, result.dw1, result.dw2, None, None, None, None


class MegaMoeBF16Train(torch.nn.Module):
    """Fused Ascend950 BF16 forward/backward, derived from source b8cff148.

    Select the NPU before construction. Pass an existing EP ProcessGroup or
    StoreGroup. All ranks construct and close in the same order; DEEPEP_SHMEM_ENDPOINT names a separate bootstrap endpoint.
    This instance owns the root Session's symmetric window. There is one SHMEM
    owner per process, shared with DeepEP's exclusion checks.

    Calls use the construction stream and must be serialized in identical rank
    order. Separate forwards retain separate saved tensors. Close only after
    backward (or discard unused graphs). Higher-order derivatives, concurrent
    streams, and simultaneous inference/communication owners are unsupported.
    """

    def __init__(self, ep_group, local_mem_size, *, timeout=120, debug_collectives=False):
        super().__init__()
        if ep_group is None:
            raise ValueError("Pass an existing EP ProcessGroup or StoreGroup")
        self._npu = npu_api()
        self._stream = self._npu.current_stream()
        native = require_native("MegaMoeBF16Train")
        from . import _C

        self._runtime = _TrainRuntime(ep_group, local_mem_size, timeout, debug_collectives=debug_collectives)
        # Root Session memset must finish on every peer before any generation signal.
        self._runtime.barrier()
        address = native.megamoe_train_workspace_address(self._runtime.native)
        like = torch.empty((), device=f"npu:{self._runtime.device}", dtype=torch.bfloat16)
        self._executor = _C.BF16(address, local_mem_size, self._runtime.rank, self._runtime.world_size, like)
        self._closed = False

    def _check_ready(self):
        if self._closed:
            raise RuntimeError("MegaMoeBF16Train has been closed")
        self._runtime.ensure_alive()
        if self._npu.current_stream() != self._stream:
            raise RuntimeError("MegaMoeBF16Train requires its construction stream")

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
        w1,
        w2,
        max_recv_tokens,
        *,
        swiglu_limit=0.0,
        save_gmm1_output="bf16",
    ):
        """Return (y, counts); autograd provides dX, dTopK, dW1 and dW2.

        Inputs are contiguous tensors on the construction NPU. X and expert
        matrices are BF16; TopK weights may be BF16 or FP32 and their gradient
        has the same dtype. All EP ranks must agree on that dtype per call.
        Expert IDs are int32. Weights use [E,2F,H]/[E,H,F], with legacy layouts
        [E,H,2F]/[E,F,H] retained. Counts are non-differentiable.
        """
        integer("max_recv_tokens", max_recv_tokens, 1, (1 << 31) - 1)
        limit = float(swiglu_limit)
        if not math.isfinite(limit) or not math.isfinite(c_float(limit).value):
            raise ValueError("swiglu_limit must be finite and representable as float32")
        if save_gmm1_output not in ("bf16", "none", "mxfp8", "auto"):
            raise ValueError("save_gmm1_output must be bf16, none, mxfp8 or auto")
        return _MegaMoeBF16TrainFunction.apply(
            self, x, topk_ids, topk_weights, w1, w2, max_recv_tokens, limit, save_gmm1_output, None
        )

    def close(self):
        """Collectively release the window; callers must finish backward first."""
        if self._closed:
            return
        self._runtime.ensure_alive()
        self._stream.synchronize()
        self._runtime.close()
        self._executor.close()
        self._closed = True
