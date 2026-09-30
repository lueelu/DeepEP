# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Minimal runnable public API example; launch with torchrun on at least two NPUs."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--timer-dir", type=Path, help="Optional separate trace, rank 0 only; use a new directory")
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401 - Registers the NPU backend.
    from . import StoreGroup, SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe
    from .validation import collective_check

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    group = StoreGroup.from_env()
    rank, world = group.rank(), group.size()
    buffer = None
    try:
        bs, h, intermediate, local_experts, topk = 17, 1024, 256, 2, 2
        buffer = SymmBuffer(
            group, world * local_experts, bs, topk, h, intermediate, ranks_per_node=int(os.environ["LOCAL_WORLD_SIZE"])
        )
        if args.timer_dir is not None and not buffer.timer_enabled:
            raise RuntimeError("Example tracing requires DEEPEP_MEGAMOE_TIMER=ON")
        torch.manual_seed(20260922 + rank)
        raw = []
        for n, k in ((2 * intermediate, h), (h, intermediate)):
            packed = torch.randint(0, 256, (local_experts, n, k // 2), dtype=torch.uint8).npu()
            scales = torch.full((local_experts, n, k // 64, 2), 120, dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)
            raw.append((packed, scales))
        w1, w2 = transform_weights_for_mega_moe(*raw)
        x = torch.randn((bs, h)).npu().to(torch.bfloat16)
        buffer.topk_idx.copy_((torch.arange(bs)[:, None] + torch.arange(topk)) % (world * local_experts))
        buffer.topk_weights.fill_(1 / topk)
        if args.input == "fp8":
            buffer.x.copy_(x.to(torch.float8_e4m3fn))
            # 此小示例使用 scale=1；真实模型应提供每 32 元素的 E8M0 scale。
            buffer.x_sf.view(torch.uint8).fill_(127)
            x = None
        buffer.validate_inputs(bs, x=x)
        buffer.prepare(bs)
        y = torch.empty((bs, h), dtype=torch.bfloat16, device="npu")
        stats = torch.zeros(local_experts, dtype=torch.int32, device="npu")
        fp8_fp4_mega_moe(y, w1, w2, buffer, cumulative_local_expert_recv_stats=stats, x=x)
        torch.npu.synchronize()
        error = None if bool(torch.isfinite(y).all().item()) else "non-finite MegaMoE output"
        collective_check(group, "example result", error)
        print(f"rank={rank} output={tuple(y.shape)} local_counts={stats.cpu().tolist()}", flush=True)
        if args.timer_dir is not None:
            timer, error = None, None
            try:
                timer = buffer.allocate_timer() if rank == 0 else None
            except Exception as failure:
                error = str(failure)
            collective_check(group, "example timer allocation", error)
            fp8_fp4_mega_moe(y, w1, w2, buffer, x=x, timer=timer)
            torch.npu.synchronize()
            if rank == 0:
                args.timer_dir.mkdir(parents=True, exist_ok=False)
                timer.cpu().numpy().astype("<i8", copy=False).tofile(args.timer_dir / "rank_0_npu_time.bin")
        group.barrier("before SHMEM teardown")
        buffer.destroy()
        group.close()
    except BaseException as error:
        group.abort(f"example: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
