# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""Two-rank real-device lifetime/stream/queued-forward regression.

Launch with torchrun --nproc-per-node=2 after installing the training wheel.
This compares repeated calls and gradients, not an independent golden.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from deep_ep_experimental.megamoe import SymmBuffer, bf16_mega_moe


def expect_error(action, message):
    try:
        action()
    except (RuntimeError, ValueError) as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"Expected rejection: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-iterations", type=int, default=0, help="Additional sequential reuse iterations")
    args = parser.parse_args()
    if args.stress_iterations < 0:
        parser.error("--stress-iterations must be nonnegative")
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    group = dist.group.WORLD
    size = dist.get_world_size()
    torch.manual_seed(20260825)
    device = f"npu:{torch.npu.current_device()}"
    m, h, f, e, topk = 129, 256, 512, 3, 4
    ids = (torch.arange(m * topk, dtype=torch.int32).reshape(m, topk) % (size * e)).to(device)
    # Use the same expert weights throughout the sequential and queued runs.
    w1 = (torch.randn(e, 2 * f, h) * 0.02).to(device=device, dtype=torch.bfloat16)
    w2 = (torch.randn(e, h, f) * 0.02).to(device=device, dtype=torch.bfloat16)
    modes = [
        (mode, limit, dtype)
        for mode, limit in (("bf16", 0.0), ("none", 0.05), ("mxfp8", 0.05))
        for dtype in (torch.bfloat16, torch.float32)
    ]

    # The first two microbatches share a plan but carry different data.
    modes.insert(1, modes[0])
    generator = torch.Generator().manual_seed(20260825 + rank)
    batches = []
    for index, (_, _, dtype) in enumerate(modes):
        x = (torch.randn(m, h, generator=generator) * 0.1).to(device=device, dtype=torch.bfloat16)
        dy = (torch.randn(m, h, generator=generator) * 0.1).to(device=device, dtype=torch.bfloat16)
        p = torch.rand(m, topk, generator=generator).softmax(dim=-1).to(device=device, dtype=dtype)
        batches.append((x, ids.roll(index, dims=1), p, dy))

    def forward(op, index=0, *, legacy_layout=False):
        x, routes, weights, _ = batches[index]
        mode, limit, _ = modes[index]
        w1_in, w2_in = w1, w2
        if legacy_layout:
            w1_in, w2_in = (w.transpose(1, 2).contiguous() for w in (w1, w2))
        values = [v.detach().clone().requires_grad_() for v in (x, weights, w1_in, w2_in)]
        x, weights, w1_in, w2_in = values
        op.x, op.topk_idx, op.topk_weights = x, routes, weights
        op.gmm1_save_type = mode
        y = torch.empty_like(x, requires_grad=False)
        counts = torch.zeros(e, dtype=torch.int32, device=device)
        pointer = y.data_ptr()
        result = bf16_mega_moe(y, w1_in, w2_in, op, cumulative_local_expert_recv_stats=counts, activation_clamp=limit)
        assert result is None and y.data_ptr() == pointer
        return values, y, counts

    def backward(index, graph):
        values, y, counts = graph
        y.backward(batches[index][3])
        assert values[1].grad.dtype == values[1].dtype
        dx, dp, dw1, dw2 = (v.grad.cpu() for v in values)
        return dict(y=y.detach().cpu(), counts=counts.cpu(), dx=dx, dp=dp, dw1=dw1, dw2=dw2)

    def check(index, actual, expected):
        for name, tensor in actual.items():
            reference = expected[name]
            label = (rank, index, modes[index], name)
            assert tensor.dtype == reference.dtype and tensor.shape == reference.shape, label
            assert torch.equal(tensor.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)), label

    def make():
        return SymmBuffer(
            group,
            e * size,
            m,
            topk,
            h,
            f,
            mma_type="bf16xbf16",
            need_grad=True,
            max_recv_tokens=m * size * e,
        )

    op = make()
    expect_error(make, "already owns")
    expected = [backward(i, forward(op, i)) for i in range(len(batches))]
    assert not torch.equal(expected[0]["dw1"], expected[1]["dw1"]), "same-plan microbatches must be distinguishable"
    pending = [forward(op, i) for i in range(len(batches))]
    with torch.npu.stream(torch.npu.Stream()):
        check(0, backward(0, forward(op)), expected[0])
    for i in reversed(range(len(pending))):
        check(i, backward(i, pending.pop()), expected[i])

    legacy = backward(0, forward(op, legacy_layout=True))
    for name in ("dw1", "dw2"):
        legacy[name] = legacy[name].transpose(1, 2)
    check(0, legacy, expected[0])
    for _ in range(args.stress_iterations):
        check(0, backward(0, forward(op)), expected[0])
    op.destroy()
    op.destroy()
    expect_error(lambda: forward(op), "destroyed")
    op = make()
    check(0, backward(0, forward(op)), expected[0])
    op.destroy()
    dist.destroy_process_group()
    print(f"[LIFECYCLE] rank={rank} PASS", flush=True)


if __name__ == "__main__":
    main()
