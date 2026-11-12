# Copyright (c) 2026, Lu Lu
"""V2 Combine-only benchmark with a real Dispatch handle; launch with torchrun."""

import argparse
import csv
from datetime import timedelta
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import struct

__test__ = False


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tokens", type=int, nargs="+", default=[32, 48, 64, 72, 256])
    result.add_argument("--hidden", type=int, default=3584)
    result.add_argument("--topk", type=int, default=8)
    result.add_argument("--experts", type=int, default=384)
    result.add_argument("--capacity", type=int, default=256)
    result.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iters", type=int, default=100)
    result.add_argument(
        "--timer",
        choices=["acl-event", "profiler"],
        default="acl-event",
        help="Default: clean native-launch ACL Events with profiler disabled",
    )
    result.add_argument(
        "--launch-gate",
        choices=["notify", "none"],
        default="notify",
        help="Hold HCCL until Combine is queued (default: notify); none is an ungated comparison",
    )
    result.add_argument(
        "--check-only", action="store_true", help="Run Dispatch/Combine E2E correctness without profiling"
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def rank_max_summary(gathered):
    if not gathered or not gathered[0] or any(len(row) != len(gathered[0]) for row in gathered):
        raise ValueError("All ranks must report the same nonempty iteration count")
    if any(not math.isfinite(value) or value <= 0 for row in gathered for value in row):
        raise ValueError("Invalid kernel duration")
    samples = [max(values) for values in zip(*gathered)]
    return {
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "p50_us": statistics.median(samples),
        "max_us": max(samples),
        "samples": len(samples),
    }


def rank_max_report(gathered):
    summary = rank_max_summary(gathered)
    iterations = []
    for iteration, values in enumerate(zip(*gathered), 1):
        rank = max(range(len(values)), key=values.__getitem__)
        iterations.append({"iteration": iteration, "rank": rank, "us": values[rank]})
    samples = [item["us"] for item in iterations]
    slowest = sorted(iterations, key=lambda item: item["us"], reverse=True)[:5]
    last50 = rank_max_summary([samples[-50:]])
    summary.update(
        max_iteration=slowest[0]["iteration"],
        max_rank=slowest[0]["rank"],
        first50_mean_us=statistics.mean(samples[:50]),
        last50_start_iteration=len(samples) - last50["samples"] + 1,
        **{f"last50_{key}": value for key, value in last50.items()},
    )
    blocks = [
        {
            "first_iteration": start + 1,
            "last_iteration": min(start + 10, len(samples)),
            "mean_us": statistics.mean(samples[start : start + 10]),
        }
        for start in range(0, len(samples), 10)
    ]
    return {"summary": summary, "iterations": iterations, "slowest5": slowest, "blocks10": blocks}


def route_digest(routes):
    return hashlib.sha256(
        b"".join(struct.pack("<i", expert) for rows in routes for row in rows for expert in row)
    ).hexdigest()


def event_samples(invoke, synchronize, timer, warmup, iterations):
    samples = []
    for iteration in range(warmup + iterations):
        output = invoke()
        synchronize()
        elapsed = timer.elapsed_us()
        if iteration >= warmup:
            samples.append(elapsed)
        del output
    return samples


def profile_combine(invoke, directory, iterations):
    import torch
    import torch_npu

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=torch_npu.profiler.ExportType.Text,
            data_simplification=False,
        ),
    ) as capture:
        for _ in range(iterations):
            output = invoke()
            torch.npu.synchronize()
            del output
        capture.step()


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < max(args.tokens) <= args.capacity <= 256 or min(args.tokens) < 1:
        raise ValueError("Require 0 < tokens <= capacity <= 256")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("Invalid warmup/iteration count")
    if os.environ.get("DEEPEP_LL_BACKEND", "rdma") != "rdma":
        raise ValueError("This benchmark tests the RDMA V2 Combine path")
    os.environ["DEEPEP_LL_BACKEND"] = "rdma"
    world, rank, local = (int(os.environ[key]) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if int(os.environ["LOCAL_WORLD_SIZE"]) != 16 or world not in (16, 32, 64, 128, 256):
        raise ValueError("Use the Dispatch topology: 16 participating ranks per host")
    if args.hidden < 4 or not 1 <= args.topk <= 16 or not world <= args.experts <= 1024 or args.experts % world:
        raise ValueError("Require H>=4, K=1..16, EP<=E<=1024 and E divisible by EP")
    if not os.environ.get("DEEPEP_SHMEM_ENDPOINT"):
        raise ValueError("Set DEEPEP_SHMEM_ENDPOINT to this job's bootstrap endpoint")

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    torch.npu.set_device(local)
    torch.set_num_threads(1)
    dist.init_process_group("hccl", timeout=timedelta(minutes=30))
    sync = torch.zeros(1, dtype=torch.int32, device="npu")
    dist.all_reduce(sync)
    torch.npu.synchronize()  # Materialize HCCL before the shared SHMEM Session.
    elastic_dir = Path(__file__).resolve().parents[1] / "elastic"
    spec = importlib.util.spec_from_file_location("ll_combine_reference", elastic_dir / "test_ep_low_latency.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    deep_ep = runpy.run_path(str(elastic_dir / "test_dispatch.py"))["_import_installed_deep_ep"]()
    options = dict(vars(args), output=str(args.output))
    peer_options = [None] * world
    dist.all_gather_object(peer_options, options)
    if any(value != options for value in peer_options):
        raise ValueError("All ranks must use the same Combine benchmark options")
    args.group_backend = "hccl"
    rank_dir = args.output / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    buffer = deep_ep.ElasticBuffer(
        group=dist.group.WORLD,
        num_bytes=8 << 30,
        num_max_tokens_per_rank=args.capacity,
        hidden=0,
        num_topk=0,
        explicitly_destroy=True,
        machine_type="mesh_clos",
    )
    if rank == 0:
        print(
            f"V2 UBX Combine EP={world} BS={args.tokens} warmup={args.warmup} iters={args.iters} "
            f"launch_gate={args.launch_gate} "
            f"sync=HCCL-each-iteration timer={args.timer} profiler={'on' if args.timer == 'profiler' else 'off'} "
            "stats=iteration-rank-MAX",
            flush=True,
        )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    with torch.no_grad():
        for bs in args.tokens:
            args.num_tokens = bs
            routes = reference.make_routes(world, bs, args.topk, args.experts, args.seed)
            route_hash = route_digest(routes)
            entries = reference.expert_entries(routes, args.experts, world, rank)
            x = reference.payload([rank] * bs, range(bs), args.hidden, dtype).npu()
            indices = torch.tensor(routes[rank], dtype=torch.int64, device="npu")
            weights = (torch.arange(1, args.topk + 1, dtype=torch.float32, device="npu") / 32).repeat(bs, 1)
            recv, _, _, handle, dispatched = buffer.dispatch(
                x,
                topk_idx=indices,
                topk_weights=weights,
                num_experts=args.experts,
                num_max_tokens_per_rank=args.capacity,
                do_expand=True,
                do_cpu_sync=False,
            )
            assert dispatched.event is None and handle.is_low_latency
            expert = reference.prepare_experts(recv, [len(rows) for rows in entries], rank, args.experts, world, dtype)
            expected = reference.expected_combine(routes, rank, args.hidden, dtype)
            torch.npu.synchronize()

            def invoke():
                # HCCL and Combine use the same comm stream. Both timers exclude
                # HCCL and Dispatch; the Event hook brackets only the native call.
                with torch.npu.stream(buffer.get_comm_stream()):
                    dist.all_reduce(sync, op=dist.ReduceOp.MAX, async_op=True).wait()
                    return buffer.combine(
                        expert,
                        handle,
                        allocate_on_comm_stream=True,
                        async_with_compute_stream=True,
                    )

            out, returned_weights, event = invoke()
            event.current_stream_wait(release_handle=True)

            def correctness():
                assert returned_weights is None
                torch.testing.assert_close(
                    out.cpu(),
                    expected,
                    rtol=0.003 if args.dtype == "fp16" else 0.01,
                    atol=0.004 if args.dtype == "fp16" else 0.03125,
                )

            reference.agree_check(correctness, args, rank)
            reference.check_async_combine(buffer, expert, handle, expected, out, args, rank)
            if args.check_only:
                (rank_dir / f"bs-{bs}-correctness.json").write_text(
                    json.dumps({"bs": bs, "correctness": "PASS"}) + "\n"
                )
                if rank == 0:
                    print(f"BS={bs} Dispatch->Combine E2E / async / bank-reuse correctness=PASS", flush=True)
                continue
            # Correctness calls materialize HCCL, tiling uploads and output
            # allocations before anything can block behind the launch gate.
            torch.npu.synchronize()
            gate = None
            if args.launch_gate == "notify":
                from launch_gate import LaunchGate

                gate = LaunchGate(buffer.get_comm_stream())

            def measured_invoke():
                return invoke() if gate is None else gate.run(invoke)

            samples = []
            try:
                directory = rank_dir / f"bs-{bs}"
                directory.mkdir(exist_ok=False)
                if rank == 0:
                    print(f"BS={bs} correctness=PASS warmup={args.warmup} sampling={args.iters}", flush=True)
                if args.timer == "acl-event":
                    from deep_ep._native import require_native
                    from launch_gate import KernelEventTimer

                    with KernelEventTimer(require_native("Combine benchmark"), buffer.get_comm_stream()) as timer:
                        samples = event_samples(measured_invoke, torch.npu.synchronize, timer, args.warmup, args.iters)
                else:
                    for _ in range(args.warmup):
                        warmed = measured_invoke()
                        torch.npu.synchronize()
                        del warmed
                    profile_combine(measured_invoke, directory, args.iters)

                    def parse():
                        samples.extend(reference.kernel_durations(directory, "ubx_combine_route_plan", args.iters))

                    reference.agree_check(parse, args, rank)
            finally:
                if gate is not None:
                    gate.close()
            sample_key = "combine_event_us" if args.timer == "acl-event" else "kernel_us"
            (directory / "samples.json").write_text(
                json.dumps(
                    {
                        sample_key: samples,
                        "timer": args.timer,
                        "launch_gate": args.launch_gate,
                        "route_sha256": route_hash,
                    }
                )
                + "\n"
            )
            gathered = [None] * world
            dist.all_gather_object(gathered, samples)
            if rank == 0:
                details = rank_max_report(gathered)
                summary = details["summary"]
                row = {
                    "bs": bs,
                    "timer": args.timer,
                    "launch_gate": args.launch_gate,
                    **summary,
                    "route_sha256": route_hash,
                }
                (args.output / f"bs-{bs}-rank-max.json").write_text(
                    json.dumps(
                        dict(
                            details,
                            bs=bs,
                            world_size=world,
                            warmup=args.warmup,
                            timer=args.timer,
                            launch_gate=args.launch_gate,
                            route_sha256=route_hash,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                report = args.output / "summary.csv"
                new = not report.exists()
                with report.open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if new:
                        writer.writeheader()
                    writer.writerow(row)
                print(
                    f"BS={bs} mean/min/p50/max(us)="
                    f"{summary['mean_us']:.3f}/{summary['min_us']:.3f}/"
                    f"{summary['p50_us']:.3f}/{summary['max_us']:.3f} samples={args.iters} route={route_hash}",
                    flush=True,
                )
                slowest = ",".join(
                    f"{item['iteration']}:r{item['rank']}:{item['us']:.3f}" for item in details["slowest5"]
                )
                blocks = ",".join(f"{item['mean_us']:.3f}" for item in details["blocks10"])
                print(
                    f"BS={bs} max_iter={summary['max_iteration']} max_rank={summary['max_rank']} "
                    f"first50_mean={summary['first50_mean_us']:.3f} "
                    f"last50[{summary['last50_start_iteration']}-{args.iters}] mean/min/p50/max(us)="
                    f"{summary['last50_mean_us']:.3f}/{summary['last50_min_us']:.3f}/"
                    f"{summary['last50_p50_us']:.3f}/{summary['last50_max_us']:.3f} "
                    f"slowest5(iter:rank:us)={slowest} block10_mean(us)={blocks}",
                    flush=True,
                )
    buffer.destroy()
    # Release the benchmark's closure cell and cyclic references while the
    # backend modules and process group are still alive.
    buffer = None
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
