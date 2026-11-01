"""Real-NPU launch smoke test for the V2 elastic intranode pipeline."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep.backends.clos import ElasticBuffer, topk_idx_t


def run_rank(rank: int, args) -> None:
    device = args.first_device + rank
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.ep_size)
    os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"] = args.shmem_ip_port
    torch.npu.set_device(device)
    dist.init_process_group(
        backend="hccl",
        init_method=args.dist_init_method,
        rank=rank,
        world_size=args.ep_size,
    )

    buffer = None
    try:
        group = dist.group.WORLD
        num_bytes = ElasticBuffer.get_buffer_size_hint(
            group,
            args.num_tokens,
            args.hidden,
            args.num_topk,
        )
        buffer = ElasticBuffer(
            group,
            num_bytes=num_bytes,
            num_max_tokens_per_rank=args.num_tokens,
            hidden=args.hidden,
            num_topk=args.num_topk,
            deterministic=True,
            allow_multiple_reduction=True,
            explicitly_destroy=True,
        )

        x = torch.zeros(
            (args.num_tokens, args.hidden),
            dtype=torch.bfloat16,
            device="npu",
        )
        token_ids = torch.arange(args.num_tokens, dtype=torch.int64, device="npu").unsqueeze(1)
        topk_slots = torch.arange(args.num_topk, dtype=torch.int64, device="npu").unsqueeze(0)
        topk_idx = (token_ids * args.num_topk + topk_slots + rank).remainder(args.num_experts).to(topk_idx_t)
        topk_weights = torch.ones(
            (args.num_tokens, args.num_topk),
            dtype=torch.float32,
            device="npu",
        )

        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=args.num_experts,
            num_sms=args.num_aiv,
        )
        assert recv_x.dim() == 2
        assert recv_topk_idx.shape == recv_topk_weights.shape

        cached_recv_x, _, cached_recv_topk_weights, cached_handle, _ = buffer.dispatch(
            x,
            topk_weights=topk_weights,
            handle=handle,
            num_sms=args.num_aiv,
        )
        assert cached_handle is handle
        assert cached_recv_x.shape == recv_x.shape
        assert cached_recv_topk_weights.shape == recv_topk_weights.shape

        combined_x, combined_topk_weights, _ = buffer.combine(
            recv_x,
            handle,
            topk_weights=recv_topk_weights,
            num_sms=args.num_aiv,
        )
        assert combined_x.shape == x.shape
        assert combined_topk_weights.shape == topk_weights.shape

        dist.barrier(device_ids=[device])
        buffer.destroy()
        buffer = None
        if rank == 0:
            print(
                "V2 elastic kernel launch smoke test passed: "
                f"ep_size={args.ep_size}, tokens={args.num_tokens}, "
                f"hidden={args.hidden}, topk={args.num_topk}"
            )
    finally:
        # Do not enter a collective destroy from an asymmetric exception path.
        # BufferRuntime's destructor performs non-collective process teardown.
        buffer = None
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--first-device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--num-aiv", type=int, default=8)
    parser.add_argument("--dist-init-method", default="tcp://127.0.0.1:29500")
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT", "tcp://127.0.0.1:8899"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_experts % args.ep_size:
        raise ValueError("num_experts must be divisible by ep_size")
    mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()


# python test/test_v2_elastic_intranode.py --ep-size 4 --num-aiv 8
