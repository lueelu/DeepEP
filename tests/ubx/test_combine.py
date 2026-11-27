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
    result.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 32, 48, 64, 256])
    result.add_argument("--hidden", type=int, default=3584)
    result.add_argument("--topk", type=int, default=8)
    result.add_argument("--experts", type=int, default=384)
    result.add_argument("--capacity", type=int, default=256)
    result.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--iters", type=int, default=100)
    result.add_argument("--perf-tail", type=int, default=50, help="Number of final in-graph Combine calls to summarize")
    result.add_argument("--timer", choices=["profiler"], default="profiler", help=argparse.SUPPRESS)
    result.add_argument("--clock-lead-us", type=int, default=200000)
    result.add_argument("--clock-interval-us", type=int, default=1000)
    result.add_argument("--route", choices=["normal", "rank-hot"], default="normal",
                        help="rank-hot concentrates routes on the first K experts to exercise row tiling")
    result.add_argument(
        "--check-only", action="store_true", help="Run Dispatch/Combine E2E correctness without profiling"
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def rank_mean_report(gathered, tail):
    if tail <= 0 or not gathered or not gathered[0] or any(len(row) != len(gathered[0]) for row in gathered):
        raise ValueError("All ranks must report the same nonempty iteration count and a positive tail")
    if any(not math.isfinite(t) or t <= 0 for row in gathered for t in row):
        raise ValueError("Invalid kernel duration")
    iterations = [statistics.mean(column) for column in zip(*gathered)]
    selected = iterations[-tail:]
    return dict(mean_us=statistics.mean(selected), p50_us=statistics.median(selected),
                min_us=min(selected), max_us=max(selected), samples=len(selected),
                first_iteration=len(iterations) - len(selected) + 1, iterations_us=iterations)


def graph_kernel_durations(reference, directory, iterations):
    rows = reference.profiler_kernel_records(directory, ("clock_wait_table", "combine_put_jit"))
    if [row["entry"] for row in rows] != ["clock_wait_table", "combine_put_jit"] * iterations:
        raise ValueError("Expected one device wait and one PUT Combine per measured iteration")
    return [row["duration"] for row in rows[1::2]]


def check_clock_windows(peers):
    for index, records in enumerate(zip(*peers), start=1):
        if max(enter for enter, _ in records) >= min(release for _, release in records):
            raise ValueError(f"Ranks did not all finish before wait {index} released; increase the clock spacing")


def route_digest(routes):
    return hashlib.sha256(
        b"".join(struct.pack("<i", expert) for rows in routes for row in rows for expert in row)
    ).hexdigest()


def benchmark_graph(buffer, expert, handle, expected, args, reference, rank, directory):
    import torch
    import torch.distributed as dist
    import torch_npu

    stream = buffer.get_comm_stream()
    state = reference.clock_alignment_state(buffer)
    native, session = state["native"], state["runtime"].native

    def boundary():
        torch.npu.synchronize()
        dist.barrier()

    def invoke():
        return buffer.combine(expert, handle, allocate_on_comm_stream=True, async_with_compute_stream=True)

    # Compile and upload immutable tiling outside capture. Keep all captured
    # resources (including completion events and the borrowed output) alive.
    boundary()
    with torch.npu.stream(stream):
        eager, _, done = invoke()
        done.current_stream_wait(release_handle=True)
    boundary()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output, _, captured_event = invoke()
    # Change actual expert values across both inbox banks. Every rank finishes
    # the preceding replay before any rank reuses its symmetric workspace.
    for sign in (-1, 1, -1, 1):
        boundary()
        with torch.npu.stream(stream):
            expert.neg_()
            graph.replay()
        stream.synchronize()
        reference.agree_check(lambda: torch.testing.assert_close(output.cpu(), expected * sign,
                              rtol=0.01, atol=0.03125), args, rank)
    if args.check_only:
        return None
    for _ in range(args.warmup):
        boundary()
        with torch.npu.stream(stream):
            graph.replay()
    boundary()
    words, stride = reference.clock_wait_table_contract(native)
    cores = native.low_latency_core_config()["active_aiv"]
    table = torch.zeros(args.iters, dtype=torch.int64, device=expert.device)
    records = torch.full((args.iters, cores, stride), -1, dtype=torch.int64, device=expert.device)
    batch = args.case_id + 1

    def clock_wait(index):
        native.low_latency_clock_wait_table(session, table, args.iters, index, records, batch, True,
                                            stream.npu_stream)

    # Compile the wait kernel before graph capture. The deadline table has a
    # fixed address and is populated only after all ranks have captured.
    boundary()
    clock_wait(0)
    stream.synchronize()
    records.fill_(-1)
    boundary()
    measured = torch.npu.NPUGraph()
    captured = []
    with torch.npu.graph(measured, stream=stream):
        for i in range(args.iters):
            clock_wait(i)
            captured.append(invoke())
    boundary()
    model, _, _ = reference.calibrate_clock_model(state, args, "combine")
    record = torch.empty(1, dtype=torch.int64, device=expert.device)
    target = torch.empty_like(record)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=torch_npu.profiler.ExportType.Text, data_simplification=False),
    ):
        boundary()
        if rank == 0:
            native.low_latency_clock_sleep_at(session, 0, record, stream.npu_stream)
            stream.synchronize()
            target.fill_(reference.ll_clock_model.forward_reference(model, record.item()) + args.clock_lead_us * 1000)
        dist.broadcast(target, src=0)
        base = target.item()
        deadlines = [reference.ll_clock_model.local_deadline(model, base + i * args.clock_interval_us * 1000)
                     for i in range(args.iters)]
        with torch.npu.stream(stream):
            table.copy_(torch.tensor(deadlines, dtype=torch.int64))
            measured.replay()
        stream.synchronize()

    samples, windows = [], []

    def read_samples():
        rows = records.tolist()
        evidence = reference.validate_wait_records(rows, iterations=args.iters, cores=cores, words=words,
                       stride_words=stride, batch_tag=batch, table=deadlines, rank=rank, label="combine")
        for i, last_enter in enumerate(evidence["local_enter"]):
            first_release = min(slot[1] for slot in rows[i])
            # A wait starts only after this rank's preceding Combine finishes.
            # Compare in reference-clock units including the model uncertainty.
            upper = (reference.ll_clock_model.forward_reference(model, last_enter)
                     + reference.ll_clock_model.error_budget(model, last_enter)["total_ns"])
            lower = (reference.ll_clock_model.forward_reference(model, first_release)
                     - reference.ll_clock_model.error_budget(model, first_release)["total_ns"])
            windows.append((upper, lower))
        for index in {0, args.iters - 1}:
            torch.testing.assert_close(captured[index][0].cpu(), expected, rtol=0.01, atol=0.03125)
        samples.extend(graph_kernel_durations(reference, directory, args.iters))

    reference.agree_check(read_samples, args, rank)
    peers = reference.host_gather(args, windows, "combine.clock.windows")
    check_clock_windows(peers)
    return samples


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < max(args.tokens) <= args.capacity <= 256 or min(args.tokens) < 1:
        raise ValueError("Require 0 < tokens <= capacity <= 256")
    if args.warmup < 0 or min(args.iters, args.perf_tail, args.clock_lead_us, args.clock_interval_us) < 1:
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
    clock_options = reference.make_parser()
    for key in ("clock_align_samples", "clock_align_warmup_samples", "clock_align_timeout_secs",
                "clock_align_calibration_windows", "clock_align_calibration_gap_secs"):
        setattr(args, key, clock_options.get_default(key))
    rank_dir = args.output / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=False)
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
            f"timer={args.timer} gate=device-clock graph=ACLGraph "
            f"stats=per-iteration-rank-mean tail={args.perf_tail}",
            flush=True,
        )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    with torch.no_grad():
        for args.case_id, bs in enumerate(args.tokens):
            args.num_tokens = bs
            if args.route == "rank-hot":
                routes = [[list(range(args.topk)) for _ in range(bs)] for _ in range(world)]
            else:
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
            dispatched.current_stream_wait()
            assert handle.is_low_latency
            expert = reference.prepare_experts(recv, [len(rows) for rows in entries], rank, args.experts, world, dtype)
            expected = reference.expected_combine(routes, rank, args.hidden, dtype)
            torch.npu.synchronize()

            samples = benchmark_graph(buffer, expert, handle, expected, args, reference, rank,
                                      rank_dir / f"bs-{bs}-profile")
            if args.check_only:
                (rank_dir / f"bs-{bs}-correctness.json").write_text(json.dumps({"correctness": "PASS", "bs": bs}))
                continue
            gathered = reference.host_gather(args, samples, f"combine.bs{bs}.samples")
            if rank == 0:
                report = rank_mean_report(gathered, args.perf_tail)
                row = dict(bs=bs, timer=args.timer, route=args.route, world=world, route_sha256=route_hash,
                           **{key: value for key, value in report.items() if key != "iterations_us"})
                (args.output / f"bs-{bs}-rank-mean.json").write_text(
                    json.dumps(dict(row, iterations_us=report["iterations_us"], samples_by_rank=gathered), indent=2))
                summary = args.output / "summary.csv"
                new = not summary.exists()
                with summary.open("a", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=list(row))
                    if new:
                        writer.writeheader()
                    writer.writerow(row)
                print(f"BS={bs} mean={row['mean_us']:.3f} us p50={row['p50_us']:.3f} us "
                      f"iterations={row['first_iteration']}–{args.iters}", flush=True)
    buffer.destroy()
    # Release the benchmark's closure cell and cyclic references while the
    # backend modules and process group are still alive.
    buffer = None
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
