"""Matched fresh/cached dispatch: one Buffer, core/EP16 channel A/B and separate traces.

Run with torchrun. This is a dispatch diagnostic, not a MoE end-to-end benchmark.
"""

import argparse
import csv
from dataclasses import asdict
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import time

from bench_fused_notify import store_topology_collective
from stage_metrics import L2CacheFlush, prepare_v2_sample

TRACE_ENV = "DEEPEP_CLOS_DISPATCH_DEVICE_TIMING"
CHANNEL_ENV = "DEEPEP_CLOS_CACHED_CHANNEL_EP16"
TIMELINE = (
    "init_done",
    "begin_done",
    "metadata_done",
    "payload_queued",
    "published",
    "incoming_done",
    "quiet_done",
    "sync_done",
    "output_done",
    "handle_done",
    "work_done",
)


def decode_trace(raw, fields, rank, cores, cached, whole_channel, cycles_per_us):
    if len(raw) != cores or cycles_per_us <= 0:
        raise ValueError("trace core count/clock mismatch")
    result = []
    for core, values in enumerate(raw):
        if len(values) < len(fields):
            raise ValueError("truncated trace")
        row = dict(zip(fields, values))
        if [row[k] for k in ("version", "rank", "core", "cached", "whole_channel")] != [
            1,
            rank,
            core,
            int(cached),
            int(whole_channel),
        ]:
            raise ValueError("trace identity or effective whole-channel gate mismatch")
        stamps = [row[k] for k in TIMELINE]
        if stamps != sorted(stamps) or stamps[0] < 0 or not row["entry_cycles"]:
            raise ValueError("invalid trace timeline")
        previous = 0
        intervals = {}
        for name in TIMELINE:
            intervals[name + "_interval_us"] = (row[name] - previous) / cycles_per_us
            previous = row[name]
        intervals.update(
            {
                key + "_us": row[key] / cycles_per_us
                for key in ("src_cycles", "topk_cycles", "self_cycles", "list_cycles")
            }
        )
        result.append(dict(raw=row, intervals=intervals))
    return result


def capture_stages(directory, samples, cached):
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError("expected exactly one kernel_details.csv")
    with paths[0].open(encoding="utf-8-sig", newline="") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: float(row["Start Time(us)"]))
    names = ["cached_notify_combine_kernel"]  # out-of-measurement alignment
    if not cached:
        names.append("dispatch_layout_kernel")
    names += ["cached_notify_dispatch_kernel" if cached else "notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
    selected = [row for row in rows if row["Name"] in set(names)]
    if [row["Name"] for row in selected] != names * (samples + 4):
        raise RuntimeError("unexpected kernel sequence/sample count; retain raw capture for inspection")
    result = {
        name: [float(row["Duration(us)"]) for row in selected[i :: len(names)]][2:-2]
        for i, name in enumerate(names)
        if i
    }
    if any(not math.isfinite(t) or t <= 0 for values in result.values() for t in values):
        raise ValueError("invalid profiler duration")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--cores", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--cached-channel-modes", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trace-samples", type=int, default=3, help="0 for formal timing only/older baseline")
    parser.add_argument("--flush-l2", action="store_true")
    parser.add_argument("--huge-pages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-port", type=int, default=29881)
    parser.add_argument("--shmem-port", type=int, default=29882)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.tokens, args.hidden, args.topk, args.experts, args.samples, *args.cores) <= 0
        or min(args.warmup, args.trace_samples) < 0
    ):
        parser.error("invalid shape/core/sample count")
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep.backends.direct import buffer as bm, elastic as em
    from deep_ep.backends.direct.elastic import ElasticBuffer
    from deep_ep.backends.direct.dtypes import topk_idx_t
    from deep_ep.backends.direct._native import load_native_module
    from bench_fast_sweep import StoreGroup
    from bench_v2_bandwidth import loaded_identity
    from bandwidth_multinode import library_identity

    rank, world, device = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world not in (8, 16) or args.experts % world or args.topk > args.experts:
        parser.error("this experiment requires EP8/16 and experts divisible by EP")
    torch.npu.set_device(device)
    native = load_native_module()
    if args.trace_samples and getattr(native, "dispatch_device_timing_version", 0) != 1:
        raise RuntimeError("rebuild host and kernel libraries for device traces")
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        args.control_port,
        world,
        rank == 0,
        timeout=datetime.timedelta(seconds=600),
        wait_for_workers=True,
    )
    group = StoreGroup(store, rank, world)
    bm._bootstrap_all_gather = lambda g, c, n, value: group.gather(value)
    em._validate_group_consensus = lambda g, name, value: group.agree([name, value])
    group.agree(vars(args) | {"output": None})
    out = args.output / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    # Explicitly isolate these changes from earlier experimental metadata/notify options.
    for key in (
        TRACE_ENV,
        CHANNEL_ENV,
        "DEEPEP_CLOS_DISPATCH_SRC_IDX_256",
        "DEEPEP_CLOS_DISPATCH_TOPK_256",
        "DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY",
        "DEEPEP_CLOS_NOTIFY_HOST_PREP",
        "DEEPEP_CLOS_NOTIFY_DEFER_PLAN",
        "DEEPEP_CLOS_DISPATCH_HOST_TIMING",
    ):
        os.environ[key] = "0"
    with store_topology_collective(group):
        buffer = ElasticBuffer(
            group,
            num_max_tokens_per_rank=args.tokens,
            hidden=args.hidden,
            num_topk=args.topk,
            explicitly_destroy=True,
            transport="udma",
            udma_auto_policy=True,
            use_1g_huge_pages=args.huge_pages,
            shmem_ip_port=f"tcp://{os.environ['MASTER_ADDR']}:{args.shmem_port}",
        )
    flush = L2CacheFlush(torch, buffer, args.flush_l2)
    actual_huge_pages = bool(buffer.runtime.use_1g_huge_pages())
    if not all(group.gather(actual_huge_pages == args.huge_pages)):
        raise RuntimeError("native huge-page state differs from requested configuration")
    identity = loaded_identity()
    identities = group.gather(library_identity(identity))
    if any(item != identities[0] for item in identities):
        raise RuntimeError("ranks loaded different library bytes")
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                args=vars(args),
                rank=rank,
                world=world,
                buffer_initializations=1,
                identity=identity,
                actual_huge_pages=actual_huge_pages,
            ),
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )

    def done(result):
        if result[-1].event is not None:
            result[-1].current_stream_wait()
        torch.npu.synchronize()
        return result

    def snapshot(result):
        recv, idx, weight = result[:3]
        return [
            recv.cpu().clone(),
            None if idx is None else idx.cpu().clone(),
            None if weight is None else weight.cpu().masked_fill(idx.cpu() < 0, 0),
        ]

    try:
        gen = torch.Generator().manual_seed(20261122 + rank)
        x = (torch.randn((args.tokens, args.hidden), generator=gen) * 0.01).to(torch.bfloat16).npu()
        idx = torch.rand((args.tokens, args.experts), generator=gen).topk(args.topk, dim=1).indices
        owners = idx // (args.experts // world)
        local_rows = sum(int((owners == peer).any(1).sum()) for peer in range(world))
        payload = sum(group.gather(local_rows)) * args.hidden * 2 / world
        idx = idx.to(dtype=topk_idx_t, device="npu")
        weights = torch.rand((args.tokens, args.topk), generator=gen).npu()
        results = []
        for op in ("fresh", "fresh_weighted", "cached"):
            configs = [
                (core, flag)
                for core in args.cores
                for flag in args.cached_channel_modes
                if flag == 0 or (world == 16 and core == 8 and op == "cached")
            ]
            if not configs:
                continue
            reference = None
            for turn, (core, flag) in enumerate(configs + configs[::-1]):
                group.agree([op, turn, core, flag])
                torch.npu.synchronize()
                os.environ[CHANNEL_ENV] = str(flag)
                handle = None

                def invoke():
                    return done(
                        buffer.dispatch(
                            x,
                            handle=handle,
                            topk_idx=idx if handle is None else None,
                            topk_weights=weights if op == "fresh_weighted" else None,
                            num_experts=args.experts,
                            num_sms=core,
                            do_expand=False,
                        )
                    )

                initial = invoke()
                policy = asdict(initial[3]._udma_policy)
                if op == "cached":
                    handle = initial[3]
                checked = snapshot(invoke())
                if reference is None:
                    reference = checked
                else:
                    for actual, expected in zip(checked, reference):
                        if expected is None:
                            assert actual is None
                        else:
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for _ in range(args.warmup):
                    prepare_v2_sample(buffer, flush)
                    invoke()
                api = []
                for _ in range(args.samples):
                    prepare_v2_sample(buffer, flush)
                    start = time.perf_counter_ns()
                    invoke()
                    api.append((time.perf_counter_ns() - start) / 1000)
                capture = out / f"{op}_round{turn}_aiv{core}_channel{flag}"
                prof = torch_npu.profiler
                with prof.profile(
                    activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                    on_trace_ready=prof.tensorboard_trace_handler(str(capture)),
                    experimental_config=prof._ExperimentalConfig(
                        export_type=[prof.ExportType.Text],
                        profiler_level=prof.ProfilerLevel.Level0,
                        aic_metrics=prof.AiCMetrics.AiCoreNone,
                    ),
                ):
                    for _ in range(args.samples + 4):
                        prepare_v2_sample(buffer, flush)
                        invoke()
                stages = capture_stages(capture, args.samples, op == "cached")
                stages["api_wall_us"] = api
                (capture / "samples.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
                gathered = group.gather(stages)
                stats = {}
                for name in stages:
                    values = sorted(max(g[name][i] for g in gathered) for i in range(args.samples))
                    stats[name] = dict(
                        mean=statistics.mean(values),
                        p50=statistics.median(values),
                        p95=values[math.ceil(0.95 * len(values)) - 1],
                    )
                results.append(
                    dict(
                        op=op,
                        turn=turn,
                        cores=core,
                        channel=flag,
                        policy=policy,
                        stats_us=stats,
                        payload_bytes_per_rank=payload,
                        payload_gbps=payload / stats["v1_dispatch_udma_kernel"]["mean"] / 1000,
                    )
                )
                (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                if args.trace_samples:
                    traces = []
                    fields = list(native.dispatch_device_timing_fields)
                    os.environ[TRACE_ENV] = "1"
                    try:
                        for _ in range(args.trace_samples):
                            prepare_v2_sample(buffer, flush)
                            traced = invoke()
                            tensor = native.take_dispatch_device_timing()
                            if tensor is None:
                                raise RuntimeError("device trace missing")
                            # invoke has consumed the completion event; no asynchronous CPU read.
                            raw = tensor.cpu().tolist()
                            traces.append(
                                decode_trace(
                                    raw,
                                    fields,
                                    rank,
                                    core,
                                    op == "cached",
                                    op == "cached" and core == 8 and (world == 8 or flag == 1),
                                    native.dispatch_device_timing_cycles_per_us,
                                )
                            )
                            for actual, expected in zip(snapshot(traced), reference):
                                if expected is not None:
                                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    finally:
                        os.environ[TRACE_ENV] = "0"
                    (capture / "device_trace.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
            group.agree([op, "passed"])
    finally:
        os.environ[TRACE_ENV] = "0"
        buffer.destroy()
    group.agree("complete")
    (out / "exit.json").write_text('{"exit_code": 0, "variant_precision_passed": true}', encoding="utf-8")


if __name__ == "__main__":
    main()
