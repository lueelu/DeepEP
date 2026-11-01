"""Real-NPU multi-rank smoke test for the V2 ElasticBuffer foundation."""

import argparse
import gc
import os
import weakref

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep.backends.clos import ElasticBuffer, EventHandle


DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024


def _trace(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def run_rank(rank: int, args: argparse.Namespace) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.ep_size)
    os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"] = args.shmem_ip_port
    os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = args.hccl_host_port_range
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = args.hccl_npu_port_range
    device_id = args.first_device + rank
    torch.npu.set_device(device_id)
    _trace(rank, f"initializing HCCL on NPU {device_id}")
    dist.init_process_group(
        backend="hccl",
        init_method=args.dist_init_method,
        rank=rank,
        world_size=args.ep_size,
    )
    _trace(rank, "HCCL initialized; constructing ElasticBuffer")

    buffer = None
    try:
        buffer = ElasticBuffer(
            dist.group.WORLD,
            num_max_tokens_per_rank=args.num_tokens,
            hidden=args.hidden,
            num_topk=args.num_topk,
            allow_multiple_reduction=True,
            explicitly_destroy=not args.implicit_destroy,
        )
        _trace(rank, "ElasticBuffer initialized")
        assert not buffer._destroyed
        assert buffer.rank_idx == rank
        assert buffer.num_ranks == args.ep_size
        assert buffer.num_bytes > 0
        assert buffer.num_bytes % DEVICE_LARGE_PAGE_SIZE == 0
        assert buffer.runtime.num_bytes() == buffer.num_bytes
        assert buffer.runtime.get_workspace_ptr() != 0
        assert buffer.get_physical_domain_size() == (1, args.ep_size)
        assert buffer.get_logical_domain_size() == (1, args.ep_size)
        assert buffer.get_comm_stream().device_index == device_id
        assert buffer.get_theoretical_num_qps(4) == 0
        _trace(rank, "runtime properties and local buffer size verified")

        captured = buffer.capture()
        assert isinstance(captured, EventHandle)
        captured.current_stream_wait()
        _trace(rank, "event capture verified; entering device barriers")

        for iteration in range(args.barrier_iters):
            if iteration == 0:
                _trace(rank, "launching first device barrier")
            buffer.barrier(
                use_comm_stream=iteration % 2 == 0,
                with_cpu_sync=iteration % 2 != 0,
                sequential=iteration % 3 != 0,
            )
            if iteration == 0:
                _trace(rank, "first device barrier completed")

        torch.npu.synchronize()
        allocated_bytes = buffer.num_bytes
        _trace(rank, "all device barriers completed; destroying buffer")
        if args.implicit_destroy:
            buffer_ref = weakref.ref(buffer)
            buffer = None
            gc.collect()
            assert buffer_ref() is None
        else:
            buffer.destroy()
            assert buffer._destroyed
        print(
            f"rank {rank}/{args.ep_size} ElasticBuffer runtime passed: "
            f"bytes={allocated_bytes}, barriers={args.barrier_iters}, "
            f"implicit_destroy={args.implicit_destroy}"
        )
    finally:
        if buffer is not None and not buffer._destroyed:
            if buffer.explicitly_destroy:
                buffer.destroy()
            else:
                buffer = None
                gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--first-device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--barrier-iters", type=int, default=20)
    parser.add_argument("--implicit-destroy", action="store_true")
    parser.add_argument(
        "--dist-init-method",
        default="tcp://127.0.0.1:29631",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT", "tcp://127.0.0.1:26831"),
    )
    parser.add_argument("--hccl-host-port-range", default="auto")
    parser.add_argument("--hccl-npu-port-range", default="auto")
    args = parser.parse_args()
    if args.ep_size not in (2, 4, 8):
        raise ValueError("ElasticBuffer runtime smoke test supports EP 2/4/8")
    if args.first_device < 0 or args.first_device + args.ep_size > torch.npu.device_count():
        raise ValueError("requested NPU device range is unavailable")
    if args.barrier_iters <= 0:
        raise ValueError("--barrier-iters must be positive")
    return args


def main() -> None:
    args = parse_args()
    mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()


# python test/test_elastic_runtime.py --ep-size 4
