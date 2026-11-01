"""Real-NPU precision test for the DeepEP V1 dispatch layout interface."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep.backends.clos import Buffer, topk_idx_t


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
        dst_rank = (token + rank) % world_size
        first_local_expert = token % experts_per_rank
        selected = [
            dst_rank * experts_per_rank + first_local_expert,
            dst_rank * experts_per_rank + (first_local_expert + 1) % experts_per_rank,
        ]

        if num_topk > 2:
            for expert in torch.randperm(num_experts, generator=generator).tolist():
                if expert not in selected:
                    selected.append(expert)
                    if len(selected) == num_topk:
                        break
        if token % 17 == 16:
            selected = [-1] * num_topk
        elif token % 3 == 0:
            selected[-1] = -1
        rows.append(selected)

    return torch.tensor(rows, dtype=topk_idx_t)


def _reference_dispatch_layout(
    topk_idx: torch.Tensor,
    num_experts: int,
    world_size: int,
):
    experts_per_rank = num_experts // world_size
    num_tokens_per_rank = [0] * world_size
    num_tokens_per_expert = [0] * num_experts
    is_token_in_rank = [[False] * world_size for _ in range(topk_idx.size(0))]

    for token, experts in enumerate(topk_idx.tolist()):
        visited_ranks = set()
        for expert in experts:
            if expert == -1:
                continue
            num_tokens_per_expert[expert] += 1
            visited_ranks.add(expert // experts_per_rank)

        for dst_rank in visited_ranks:
            is_token_in_rank[token][dst_rank] = True
            num_tokens_per_rank[dst_rank] += 1

    return (
        torch.tensor(num_tokens_per_rank, dtype=torch.int32),
        torch.tensor(num_tokens_per_expert, dtype=torch.int32),
        torch.tensor(is_token_in_rank, dtype=torch.bool),
    )


def _assert_tensor_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.cpu()
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{name} dtype mismatch: actual={actual.dtype}, expected={expected.dtype}")
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    if torch.equal(actual, expected):
        return

    mismatch = torch.nonzero(actual != expected, as_tuple=False)
    first_locations = mismatch[:16].tolist()
    raise AssertionError(f"{name} mismatch at {mismatch.size(0)} locations; first locations={first_locations}")


def run_rank(rank: int, args) -> None:
    world_size = int(args.ep_size)
    device_id = int(args.first_device) + rank
    num_experts = world_size * args.experts_per_rank

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.npu.set_device(device_id)

    topk_idx_cpu = _make_topk_idx(
        rank,
        world_size,
        args.num_tokens,
        args.experts_per_rank,
        args.num_topk,
        args.seed,
    )
    expected_rank, expected_expert, expected_membership = _reference_dispatch_layout(
        topk_idx_cpu,
        num_experts,
        world_size,
    )
    topk_idx = topk_idx_cpu.to(device="npu", dtype=topk_idx_t)

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

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            num_experts,
            async_finish=True,
        )
        event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        buffer.barrier()

        if num_tokens_per_rdma_rank is not None:
            raise AssertionError("intranode layout must return None for RDMA rank counts")
        _assert_tensor_equal(
            "num_tokens_per_rank",
            num_tokens_per_rank,
            expected_rank,
        )
        _assert_tensor_equal(
            "num_tokens_per_expert",
            num_tokens_per_expert,
            expected_expert,
        )
        _assert_tensor_equal(
            "is_token_in_rank",
            is_token_in_rank,
            expected_membership,
        )

        print(
            f"rank {rank}/{world_size} dispatch layout precision passed: "
            f"tokens={args.num_tokens}, topk={args.num_topk}, "
            f"experts={num_experts}"
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
    parser.add_argument("--experts-per-rank", type=int, default=127)
    parser.add_argument("--num-topk", type=int, default=8)
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
            "tcp://127.0.0.1:8899",
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_experts = args.ep_size * args.experts_per_rank
    if args.ep_size < 2:
        raise ValueError("--ep-size must be greater than or equal to 2")
    if args.num_tokens <= 0:
        raise ValueError("--num-tokens must be greater than 0")
    if args.experts_per_rank < 2:
        raise ValueError("--experts-per-rank must be greater than or equal to 2")
    if args.num_topk < 2 or args.num_topk > num_experts:
        raise ValueError("--num-topk must be in [2, num_experts]")
    if args.num_bytes % DEVICE_LARGE_PAGE_SIZE != 0:
        raise ValueError(f"--num-bytes must be aligned to {DEVICE_LARGE_PAGE_SIZE} bytes")

    mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()


# python test/test_dispatch_layout.py --ep-size 4 --num-tokens 257 \
#   --experts-per-rank 127 --num-topk 8 \
#   --shmem-ip-port tcp://127.0.0.1:26665
