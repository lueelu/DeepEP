# Copyright (c) 2026, Lu Lu
"""Standalone UBX RDMA Combine validation and clean Event performance.

Run with torchrun. HCCL is initialized and warmed before SHMEM; each timed
Combine follows the same MAX-all_reduce gate as the private benchmark.
"""

import argparse
from array import array
import csv
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import statistics
import time

from fixture import decode, make_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 48, 64, 72, 256])
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", default=None, help="Physical devices when no visibility mask is set")
    parser.add_argument("--endpoint", default=os.getenv("DEEPEP_SHMEM_ENDPOINT"))
    parser.add_argument("--output", default="results/ubx-combine")
    args = parser.parse_args()
    if (
        min(args.tokens) < 1
        or max(args.tokens) > min(args.capacity, 256)
        or len(set(args.tokens)) != len(args.tokens)
        or args.iters < 1
        or args.warmup < 0
        or not args.endpoint
    ):
        parser.error("Use positive cases/iterations and set --endpoint or DEEPEP_SHMEM_ENDPOINT")
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    if args.devices and os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Use --devices or ASCEND_RT_VISIBLE_DEVICES, not both")
    devices = list(map(int, args.devices.split(","))) if args.devices else list(range(local_world))
    if len(devices) != local_world or len(set(devices)) != len(devices):
        parser.error("Exactly one distinct device per local process is required")
    torch.npu.set_device(devices[local])
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))

    def stage(text):
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    # Fail collectively before any SHMEM bootstrap or device kernel launch.
    config = (vars(args), local_world)
    configs = [None] * world
    dist.all_gather_object(configs, config)
    if any(c != configs[0] for c in configs):
        raise RuntimeError("Ranks supplied different UBX benchmark configurations")
    output = Path(args.output)
    error = None
    try:
        output.mkdir(parents=True, exist_ok=True)
        sample_file = (output / f"rank-{rank}.csv").open("x", newline="")
        summary_file = (output / "summary.csv").open("x", newline="") if rank == 0 else None
    except OSError as exc:
        error = str(exc)
    errors = [None] * world
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"Choose a new writable --output directory: {errors}")
    stage("HCCL initialization and first all_reduce (before SHMEM)")
    hccl = dist.new_group(backend="hccl", timeout=timedelta(seconds=300))
    token = torch.ones(1, dtype=torch.float32, device="npu")

    def gate():
        work = dist.all_reduce(token, op=dist.ReduceOp.MAX, group=hccl, async_op=True)
        work.wait()
        return torch.npu.current_stream().npu_stream

    gate()
    torch.npu.synchronize()
    stream = torch.npu.current_stream().npu_stream

    from deep_ep.backends import ubx as native

    if native.ranks_per_host != local_world or world % local_world:
        raise RuntimeError("Rebuild UBX with DEEPEP_UBX_RANKS_PER_HOST matching --nproc-per-node")
    stage("UBX SHMEM MTE+ROCE initialization")
    session = native.Session(rank, world, args.endpoint, 180)
    cores = [None] * world
    dist.all_gather_object(cores, session.cores)
    if len(set(cores)) != 1:
        raise RuntimeError(f"Different active AIV counts: {cores}")
    samples = csv.writer(sample_file)
    samples.writerow(["bs", "iteration", "combine_event_us"])
    summary = csv.writer(summary_file) if summary_file else None
    if summary:
        summary.writerow(["bs", "samples", "rank_max_mean_us", "rank_max_min_us", "rank_max_p50_us", "rank_max_max_us"])
        print(
            f"UBX route-plan EP={world} ranks_per_host={local_world} AIV={session.cores} "
            f"send/recv={session.cores // 2}/{session.cores // 2} notify={session.cores - 8}..{session.cores - 1} "
            "QP=physical-core M4_input_lanes=6 data_l2=7 sync=hccl launch_gate=local-notify timer=ACL-Event",
            flush=True,
        )
    for bs in args.tokens:
        stage(f"BS={bs} CPU input and metadata")
        routes = array("i")
        routes.frombytes(native.routes(world, bs, args.topk, args.experts, args.seed))
        fixture = make_fixture(
            routes, world, rank, bs, args.hidden, args.topk, args.experts, local_world, args.capacity
        )
        dist.barrier()
        session.reset()
        dist.barrier()
        case = native.Combine(
            session,
            bs,
            args.hidden,
            args.topk,
            args.experts,
            args.capacity,
            True,
            *(fixture[key] for key in ("input", "destination", "counts", "weights", "types", "relay", "mask")),
        )

        def verify(case, sign=1):
            actual = array("H")
            actual.frombytes(case.output())
            for i, (a, e) in enumerate(zip(actual, fixture["expected"], strict=True)):
                av, ev = decode(a), sign * decode(e)
                if not abs(av - ev) <= 0.02 * (1 + abs(ev)):
                    raise AssertionError(f"rank={rank} BS={bs} element={i}: {av} != {ev}")

        stage(f"BS={bs} correctness and rotating-bank cache reuse")
        for sign in (1, -1, -1, 1, 1):
            data = array("H")
            data.frombytes(fixture["input"])
            if sign < 0:
                data = array("H", (v ^ 0x8000 for v in data))
            case.set_input(data.tobytes())
            case.run(stream, gate)
            verify(case, sign)
        stage(f"BS={bs} warmup={args.warmup}")
        for _ in range(args.warmup):
            case.run(stream, gate)
        measured = []
        for i in range(args.iters):
            if i % 10 == 0:
                stage(f"BS={bs} sampling={i}/{args.iters}")
            elapsed = case.run(stream, gate)
            samples.writerow([bs, i, elapsed])
            measured.append(elapsed)
        verify(case)
        sample_file.flush()
        gathered = [None] * world
        dist.all_gather_object(gathered, measured)
        maximum = [max(row[i] for row in gathered) for i in range(args.iters)]
        if summary:
            values = [statistics.mean(maximum), min(maximum), statistics.median(maximum), max(maximum)]
            summary.writerow([bs, args.iters, *values])
            summary_file.flush()
            print(
                f"BS={bs} mean/min/p50/max(us)="
                + "/".join(f"{v:.3f}" for v in values)
                + f" samples={args.iters} correctness=PASS route={fixture['route_sha256']}",
                flush=True,
            )
        del case
        gc.collect()
    dist.barrier()
    session.close()
    sample_file.close()
    if summary_file:
        summary_file.close()
        (output / "configuration.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    dist.destroy_process_group(hccl)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
