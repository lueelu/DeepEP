"""Real-NPU precision test for normal and cached notify_dispatch."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep.backends.clos import Buffer, Config, topk_idx_t
from deep_ep.backends.clos.config import _MAX_NUM_AIV


DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024


def _make_topk_idx(
    rank: int,
    world_size: int,
    num_tokens: int,
    experts_per_rank: int,
    num_topk: int,
    seed: int,
) -> torch.Tensor:
    num_experts = world_size * experts_per_rank
    generator = torch.Generator().manual_seed(seed + rank)
    rows = []
    for token in range(num_tokens):
        primary_rank = (token + 2 * rank) % world_size
        selected = [
            primary_rank * experts_per_rank + token % experts_per_rank,
            primary_rank * experts_per_rank + (token + 1) % experts_per_rank,
        ]
        for expert in torch.randperm(num_experts, generator=generator).tolist():
            if expert not in selected:
                selected.append(expert)
                if len(selected) == num_topk:
                    break
        rows.append(selected)
    return torch.tensor(rows, dtype=topk_idx_t)


def _layout_reference(
    topk_idx: torch.Tensor,
    world_size: int,
    experts_per_rank: int,
):
    num_experts = world_size * experts_per_rank
    rank_counts = torch.zeros(world_size, dtype=torch.int32)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32)
    membership = torch.zeros(topk_idx.size(0), world_size, dtype=torch.bool)
    for token, experts in enumerate(topk_idx.tolist()):
        visited_ranks = set()
        for expert in experts:
            expert_counts[expert] += 1
            visited_ranks.add(expert // experts_per_rank)
        for dst_rank in visited_ranks:
            rank_counts[dst_rank] += 1
            membership[token, dst_rank] = True
    return rank_counts, expert_counts, membership


def _notify_reference(rank: int, args):
    world_size = args.ep_size
    all_rank_counts = []
    all_expert_counts = []
    all_memberships = []
    local_membership = None
    for source_rank in range(world_size):
        topk_idx = _make_topk_idx(
            source_rank,
            world_size,
            args.num_tokens,
            args.experts_per_rank,
            args.num_topk,
            args.seed,
        )
        rank_counts, expert_counts, membership = _layout_reference(
            topk_idx,
            world_size,
            args.experts_per_rank,
        )
        all_rank_counts.append(rank_counts)
        all_expert_counts.append(expert_counts)
        all_memberships.append(membership)
        if source_rank == rank:
            local_membership = membership

    rank_prefix = torch.stack(all_rank_counts, dim=0).cumsum(dim=0, dtype=torch.int32)

    tokens_per_channel = (args.num_tokens + args.num_aiv - 1) // args.num_aiv
    channel_counts = torch.zeros(world_size, args.num_aiv, dtype=torch.int32)
    for channel in range(args.num_aiv):
        begin = min(channel * tokens_per_channel, args.num_tokens)
        end = min(begin + tokens_per_channel, args.num_tokens)
        channel_counts[:, channel] = local_membership[begin:end].sum(dim=0, dtype=torch.int32)
    channel_prefix = channel_counts.cumsum(dim=1, dtype=torch.int32)

    # handle[2]: exclusive channel starts from every source to this rank.
    # Compute directly from routes, independently of Notify's gather/shift.
    recv_channel_prefix = torch.zeros_like(channel_prefix)
    for source_rank, membership in enumerate(all_memberships):
        for channel in range(args.num_aiv):
            begin = min(channel * tokens_per_channel, args.num_tokens)
            recv_channel_prefix[source_rank, channel] = membership[:begin, rank].sum(dtype=torch.int32)

    expert_begin = rank * args.experts_per_rank
    expert_end = expert_begin + args.experts_per_rank
    local_expert_counts = torch.stack(all_expert_counts, dim=0).sum(dim=0, dtype=torch.int32)[expert_begin:expert_end]
    local_expert_counts = (
        (local_expert_counts + args.expert_alignment - 1) // args.expert_alignment * args.expert_alignment
    )
    return rank_prefix, channel_prefix, local_expert_counts, recv_channel_prefix


def _assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.cpu()
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(
            f"{name} metadata mismatch: actual dtype/shape="
            f"{actual.dtype}/{tuple(actual.shape)}, expected="
            f"{expected.dtype}/{tuple(expected.shape)}"
        )
    if not torch.equal(actual, expected):
        mismatch = torch.nonzero(actual != expected, as_tuple=False)
        raise AssertionError(f"{name} mismatch at {mismatch.size(0)} locations; first={mismatch[:16].tolist()}")


def run_rank(rank: int, args) -> None:
    world_size = args.ep_size
    num_experts = world_size * args.experts_per_rank
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.npu.set_device(args.first_device + rank)

    topk_idx_cpu = _make_topk_idx(
        rank,
        world_size,
        args.num_tokens,
        args.experts_per_rank,
        args.num_topk,
        args.seed,
    )
    (
        expected_rank_prefix,
        expected_channel_prefix,
        expected_expert_counts,
        expected_recv_prefix,
    ) = _notify_reference(rank, args)
    config = Config(args.num_aiv, 6, 256, 6, 128)

    buffer = None
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=args.dist_init_method,
            rank=rank,
            world_size=world_size,
        )
        buffer = Buffer(
            dist.group.WORLD,
            num_nvl_bytes=args.num_bytes,
            explicitly_destroy=True,
            shmem_ip_port=args.shmem_ip_port,
        )
        buffer.barrier()

        topk_idx = topk_idx_cpu.to(device="npu", dtype=topk_idx_t)
        x = torch.randn(
            args.num_tokens,
            args.hidden,
            device="npu",
            dtype=torch.float16,
        )
        (
            num_tokens_per_rank,
            _,
            num_tokens_per_expert,
            is_token_in_rank,
            layout_event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            num_experts,
            async_finish=True,
        )
        layout_event.current_stream_wait(release_handle=True)

        recv_x, _, _, counts, handle, event = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            expert_alignment=args.expert_alignment,
            config=config,
            async_finish=True,
        )
        event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()

        expected_recv_tokens = int(expected_rank_prefix[-1, rank])
        if recv_x.size(0) != expected_recv_tokens:
            raise AssertionError(f"recv token mismatch: actual={recv_x.size(0)}, expected={expected_recv_tokens}")
        _assert_equal("rank_prefix_matrix", handle[0], expected_rank_prefix)
        _assert_equal("channel_prefix_matrix", handle[1], expected_channel_prefix)
        _assert_equal("recv_channel_prefix_matrix", handle[2], expected_recv_prefix)
        actual_counts = torch.tensor(counts, dtype=torch.int32)
        _assert_equal(
            "num_recv_tokens_per_expert",
            actual_counts,
            expected_expert_counts,
        )

        cached_recv_x, _, _, cached_counts, cached_handle, cached_event = buffer.dispatch(
            x,
            handle=handle,
            config=config,
            async_finish=True,
        )
        cached_event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        if cached_recv_x.size(0) != expected_recv_tokens:
            raise AssertionError("cached dispatch receive shape changed")
        if cached_counts is not None or cached_handle is not None:
            raise AssertionError("cached dispatch must not return counts or a handle")
        _assert_equal("cached recv_channel_prefix_matrix", handle[2], expected_recv_prefix)

        buffer.barrier()
        print(
            f"rank {rank}/{world_size} notify precision passed: "
            f"channels={args.num_aiv}, recv_tokens={expected_recv_tokens}"
        )
    finally:
        if buffer is not None and buffer.is_available():
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--first-device", type=int, default=0)
    parser.add_argument("--num-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--num-tokens", type=int, default=257)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--experts-per-rank", type=int, default=31)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-aiv", type=int, default=32)
    parser.add_argument("--expert-alignment", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--dist-init-method",
        default="tcp://127.0.0.1:29500",
        help="EP group rendezvous; use a port distinct from --shmem-ip-port",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get(
            "ASCEND_DEEPEP_SHMEM_IP_PORT",
            "tcp://127.0.0.1:18899",
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_experts = args.ep_size * args.experts_per_rank
    if args.ep_size < 2:
        raise ValueError("--ep-size must be at least 2")
    if args.num_tokens <= 0 or args.hidden <= 0:
        raise ValueError("--num-tokens and --hidden must be positive")
    if args.num_topk < 2 or args.num_topk > num_experts:
        raise ValueError("--num-topk must be in [2, num_experts]")
    if args.num_aiv <= 0 or args.num_aiv > _MAX_NUM_AIV or args.num_aiv % 2:
        raise ValueError(f"--num-aiv must be an even integer in [2, {_MAX_NUM_AIV}]")
    if args.expert_alignment <= 0:
        raise ValueError("--expert-alignment must be positive")
    if args.num_bytes % DEVICE_LARGE_PAGE_SIZE:
        raise ValueError(f"--num-bytes must be aligned to {DEVICE_LARGE_PAGE_SIZE} bytes")
    mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()


# python test/test_notify_dispatch.py --ep-size 4 --num-aiv 32 \
#   --num-tokens 257 --experts-per-rank 31 --num-topk 8 \
#   --shmem-ip-port tcp://127.0.0.1:26666
