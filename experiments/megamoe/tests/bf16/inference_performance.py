# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026

"""Steady-state BF16 inference on the postweighted public interface."""

import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist

from deep_ep_experimental.megamoe import bf16_mega_moe
from deep_ep_experimental.megamoe.performance import collect_profile


@torch.no_grad()
def measure_inference(op, tensors, args, profile_directory):
    x, ids, p, w1, w2, _ = tensors
    op.x, op.topk_idx, op.topk_weights = x, ids, p
    y = torch.empty_like(x)
    counts = torch.zeros(args.expert_per_rank, dtype=torch.int32, device=x.device)

    def call(timer=None):
        bf16_mega_moe(y, w1, w2, op, activation_clamp=args.swiglu_limit)

    bf16_mega_moe(y, w1, w2, op, cumulative_local_expert_recv_stats=counts, activation_clamp=args.swiglu_limit)
    torch.npu.synchronize()
    dist.barrier()
    for _ in range(args.warmup):
        call()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(args.loops):
        call()
    end.record()
    end.synchronize()
    batch_ms = start.elapsed_time(end) / args.loops

    # Match W4A8 batch scheduling and kernel selection, with 32 MiB of prefix scratch.
    alignment = dict(rows=2048, cols=4096, repeats=20)
    prefix = torch.zeros((alignment["rows"], alignment["cols"]), device=x.device, dtype=torch.float32)
    torch.exp(prefix, out=prefix)
    group = SimpleNamespace(barrier=lambda stage: dist.barrier())
    samples = collect_profile(
        call, group, torch, {"alignment": alignment, "iters": args.loops}, prefix, profile_directory, len(x)
    )
    durations = [sample["duration_us"] / 1000 for sample in samples[len(samples) // 2 :]]
    return {
        "operator": "megamoe_bf16_inference",
        "scope": "Continuous public forward; profiler kernel durations from the tail half of a separate batch",
        "batch_ms_per_call": batch_ms,
        "forward_ms": {"samples": durations, "median": statistics.median(durations)},
        "kernel_samples": samples,
        "baseline": {"status": "NOT_TIMED"},
    }, counts
