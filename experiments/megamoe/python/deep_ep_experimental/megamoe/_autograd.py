# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""Per-call BF16 state retained by PyTorch until backward completes."""

import torch
from torch.autograd.function import once_differentiable
from deep_ep._native import npu_api


class BF16Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, plan, config, x, ids, p, w1, w2, out):
        capture = getattr(owner, "_timer_capture", None)
        ctx.capture = capture
        ctx.rebuild_dispatch = None if owner.save_dispatched_x else owner.dispatch_rebuilder
        if not owner.save_dispatched_x and not callable(ctx.rebuild_dispatch):
            raise ValueError("Set sym_buffer.dispatch_rebuilder for this microbatch when save_dispatched_x=False")
        if capture is not None:
            capture.claim()
        result = plan.forward(
            out,
            x,
            ids,
            p,
            w1,
            w2,
            config.max_recv,
            config.limit,
            config.save_type,
            capture is not None,
            owner.save_dispatched_x,
        )
        if capture is not None:
            capture.forward = result.timer
        saved = result.saved_state
        saved.gmm1_save_type = config.save_type
        ctx.owner, ctx.plan, ctx.config = owner, plan, config
        ctx.save_for_backward(p, w1, w2, result.counts, saved, result.dispatch_output)
        ctx.mark_dirty(out)
        ctx.mark_non_differentiable(result.counts)
        return result.y, result.counts

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, _counts=None):
        p, w1, w2, counts, saved, dispatched = ctx.saved_tensors
        b, config, capture = ctx.owner, ctx.config, ctx.capture
        runtime = b._runtime
        with b._lock:
            runtime.ensure_alive()
            caller = npu_api().current_stream(runtime.device)
            if dispatched is None:
                dispatched = ctx.rebuild_dispatch()
            runtime.stream.wait_stream(caller)
            with npu_api().stream(runtime.stream):
                if capture is not None:
                    capture.begin_backward()
                contiguous_dy = dy.contiguous()
                if contiguous_dy is not dy:
                    dy.record_stream(runtime.stream)
                result = ctx.plan.backward(
                    contiguous_dy,
                    p,
                    w1,
                    w2,
                    counts,
                    saved,
                    dispatched,
                    config.max_recv,
                    config.limit,
                    config.save_type,
                    capture is not None,
                )
                if capture is not None:
                    capture.backward = result.timer
            caller.wait_stream(runtime.stream)
            if caller != runtime.stream:
                for t in (result.dx, result.dtopk_weights, result.dw1, result.dw2):
                    t.record_stream(caller)
        return None, None, None, result.dx, None, result.dtopk_weights, result.dw1, result.dw2, None
