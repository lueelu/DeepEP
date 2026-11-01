"""Real-NPU smoke test for Buffer runtime resource ownership."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep.backends.clos import Buffer, EventOverlap, topk_idx_t


def run_rank(rank: int, args) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.ep_size)
    torch.npu.set_device(args.first_device + rank)
    dist.init_process_group(
        backend="hccl",
        init_method=args.dist_init_method,
        rank=rank,
        world_size=args.ep_size,
    )

    buffer = None
    try:
        buffer = Buffer(
            dist.group.WORLD,
            num_bytes=args.num_bytes,
            transport=args.transport,
            low_latency_mode=args.low_latency_mode,
            explicitly_destroy=True,
            use_1g_huge_pages=args.use_1g_huge_pages,
            shmem_ip_port=args.shmem_ip_port,
        )
        assert buffer.is_available()
        assert buffer.comm_meta_ptr != 0
        if args.num_bytes:
            assert buffer.native.get_workspace_ptr() != 0
        else:
            assert buffer.native.get_workspace_ptr() == 0
        assert buffer.rank == rank
        assert buffer.num_ranks == args.ep_size
        assert buffer.group_size == args.ep_size
        assert buffer.num_bytes == args.num_bytes
        assert buffer.transport == args.transport
        assert buffer.low_latency_mode == args.low_latency_mode
        assert buffer.native.num_bytes() == args.num_bytes
        assert buffer.native.transport() == args.transport
        assert buffer.native.low_latency_mode() == args.low_latency_mode
        assert buffer.native.use_1g_huge_pages() == args.use_1g_huge_pages
        assert (
            buffer.native.data_op_engine_type()
            == {
                "auto": 0x01,
                "mte": 0x01,
                "sdma": 0x02,
                "roce": 0x04,
                "udma": 0x01 | 0x08,
            }[args.transport]
        )
        assert buffer.native.local_mem_size() >= args.num_bytes + 2 * 1024 * 1024
        assert buffer.native.local_mem_size() % (2 * 1024 * 1024) == 0
        assert buffer.get_comm_stream().device_index == args.first_device + rank
        expected_topk_idx_t = torch.int32 if int(os.environ.get("EP_NUM_TOPK_IDX_BITS", "64")) == 32 else torch.int64
        assert topk_idx_t == expected_topk_idx_t
        captured = buffer.capture()
        assert isinstance(captured, EventOverlap)
        captured.current_stream_wait(release_handle=True)
        assert captured.event is None
        buffer.barrier()
        buffer.destroy()
        assert not buffer.is_available()
        print(
            f"rank {rank}/{args.ep_size} Buffer resource initialization completed "
            f"(transport={args.transport}, low_latency={args.low_latency_mode}, "
            f"1g_huge_pages={args.use_1g_huge_pages})"
        )
    finally:
        if buffer is not None and buffer.is_available():
            buffer.destroy()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--first-device", type=int, default=0)
    parser.add_argument("--num-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument(
        "--transport",
        choices=("auto", "mte", "roce", "sdma", "udma"),
        default="auto",
    )
    parser.add_argument("--low-latency-mode", action="store_true")
    parser.add_argument("--use-1g-huge-pages", action="store_true")
    parser.add_argument(
        "--dist-init-method",
        default="tcp://127.0.0.1:29500",
        help="torch.distributed rendezvous used to construct deep_ep.Buffer(group, ...)",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT", "tcp://127.0.0.1:8899"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ep_size not in (2, 4, 8):
        raise ValueError("Buffer runtime smoke test supports EP 2/4/8")
    mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()


# python test/test_buffer_runtime.py --ep-size 4 --num-bytes 67108864 \
#   --dist-init-method tcp://127.0.0.1:29500 \
#   --shmem-ip-port tcp://127.0.0.1:26664
