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
METADATA_ENV = "DEEPEP_CLOS_DISPATCH_CHANNEL_METADATA"
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


def cpu_inputs(torch, args, source, tokens=None, routing="random"):
    tokens = args.tokens if tokens is None else tokens
    gen = torch.Generator().manual_seed(20261122 + source)
    x = (torch.randn((tokens, args.hidden), generator=gen) * 0.01).to(torch.bfloat16)
    idx = torch.rand((tokens, args.experts), generator=gen).topk(args.topk, dim=1).indices
    weights = torch.rand((tokens, args.topk), generator=gen)
    if routing == "masked":
        idx[::7] = -1
        idx[:, -1] = -1
    elif routing == "invalid":
        idx.fill_(-1)
    elif routing == "skew":
        idx = torch.arange(args.topk).remainder(args.experts).expand(tokens, -1).clone()
    return x, idx, weights


def check_cpu_reference(torch, actual, args, rank, world, sizes=None, routing="random", cached=False, src_idx=None):
    """Derive receive rows from original routes, never from Notify/handle output."""
    recv, recv_idx, recv_weights = actual
    local_experts = args.experts // world
    first = 0
    for source in range(world):
        x, idx, weights = cpu_inputs(torch, args, source, None if sizes is None else sizes[source], routing)
        hits = (idx >= rank * local_experts) & (idx < (rank + 1) * local_experts)
        selected = hits.any(1)
        n = int(selected.sum())
        torch.testing.assert_close(recv[first : first + n], x[selected], rtol=0, atol=0)
        if src_idx is not None:
            torch.testing.assert_close(
                src_idx[first : first + n].to(torch.int64), torch.where(selected)[0], rtol=0, atol=0
            )
        expected_idx = (idx[selected] - rank * local_experts).masked_fill(~hits[selected], -1)
        if not cached:
            torch.testing.assert_close(recv_idx[first : first + n].to(torch.int64), expected_idx, rtol=0, atol=0)
        if recv_weights is not None:
            torch.testing.assert_close(
                recv_weights[first : first + n], weights[selected].masked_fill(~hits[selected], 0), rtol=0, atol=0
            )
        first += n
    if first != len(recv):
        raise AssertionError("receive row count differs from the independent routing reference")


def channel_comparison(results):
    """Summarize matched off/on turns; never add independently aggregated stages."""
    comparisons = []
    for key in sorted({(r["op"], r["cores"], r["channel"]) for r in results}):
        selected = [r for r in results if (r["op"], r["cores"], r["channel"]) == key]
        off = [r for r in selected if r["metadata_requested"] == 0]
        on = [r for r in selected if r["metadata_requested"] == 1]
        if not off or not on:
            continue
        if len(off) != len(on) or any(r["policy"] != off[0]["policy"] for r in selected):
            raise ValueError("unmatched metadata comparison")
        row = dict(op=key[0], cores=key[1], cached_channel=key[2], pairs=len(off))
        notify = "cached_notify_dispatch_kernel" if key[0] == "cached" else "notify_dispatch_kernel"
        for label, stage in (("notify", notify), ("comm", "v1_dispatch_udma_kernel"), ("api", "api_wall_us")):
            a = [r["stats_us"][stage]["mean"] for r in off]
            b = [r["stats_us"][stage]["mean"] for r in on]
            row.update(
                {
                    label + "_off_us": statistics.mean(a),
                    label + "_on_us": statistics.mean(b),
                    label + "_on_min_us": min(b),
                    label + "_on_max_us": max(b),
                    label + "_time_change_pct": (statistics.mean(b) / statistics.mean(a) - 1) * 100,
                    label + "_pair_delta_min_us": min(y - x for x, y in zip(a, b)),
                    label + "_pair_delta_max_us": max(y - x for x, y in zip(a, b)),
                }
            )
        payload = off[0]["payload_bytes_per_rank"]
        row["off_gbps"] = payload / row["comm_off_us"] / 1000
        row["on_gbps"] = payload / row["comm_on_us"] / 1000
        comparisons.append(row)
    return comparisons


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--cores", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--cached-channel-modes", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--metadata-modes", type=int, nargs="+", choices=[0, 1], default=[0])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trace-samples", type=int, default=3, help="0 for formal timing only/older baseline")
    parser.add_argument("--flush-l2", action="store_true")
    parser.add_argument("--huge-pages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-port", type=int, default=29881)
    parser.add_argument("--shmem-port", type=int, default=29882)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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
    if 1 in args.metadata_modes and getattr(native, "dispatch_channel_metadata_version", 0) != 1:
        raise RuntimeError("rebuild host and kernel libraries for channel metadata")
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
        METADATA_ENV,
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
                bandwidth="decimal GB/s, deduplicated BF16 payload including self, excluding metadata",
                aggregation="per-iteration slowest rank, then mean/P50/P95; API and profiler separate passes",
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
        if 1 in args.metadata_modes:
            precision_cases = []
            for routing in ("masked", "invalid", "skew", "empty_local"):
                sizes = [min(args.tokens, 65)] * world
                if routing == "empty_local":
                    sizes[0] = 0
                cpu_x, cpu_idx, cpu_w = cpu_inputs(torch, args, rank, sizes[rank], routing)
                dx, di, dw = cpu_x.npu(), cpu_idx.to(dtype=topk_idx_t, device="npu"), cpu_w.npu()
                for core in args.cores:
                    for weighted in (False, True):
                        for mode in args.metadata_modes:
                            group.agree(["precision", routing, core, weighted, mode])
                            os.environ[METADATA_ENV] = str(mode)
                            prepare_v2_sample(buffer, flush)
                            result = done(
                                buffer.dispatch(
                                    dx,
                                    topk_idx=di,
                                    topk_weights=dw if weighted else None,
                                    num_experts=args.experts,
                                    num_sms=core,
                                    do_expand=False,
                                )
                            )
                            if bool(native.last_dispatch_channel_metadata()) != bool(mode):
                                raise RuntimeError("precision launch used the wrong channel-metadata mode")
                            check_cpu_reference(
                                torch,
                                snapshot(result),
                                args,
                                rank,
                                world,
                                sizes,
                                routing,
                                src_idx=result[3]._dispatch_layout[3].cpu(),
                            )
                            precision_cases.append(dict(routing=routing, cores=core, weighted=weighted, mode=mode))
            group.agree("channel_metadata_precision_passed")
            (out / "precision.json").write_text(json.dumps(precision_cases, indent=2), encoding="utf-8")
        cpu_x, idx, cpu_w = cpu_inputs(torch, args, rank)
        x = cpu_x.npu()
        owners = idx // (args.experts // world)
        local_rows = sum(int((owners == peer).any(1).sum()) for peer in range(world))
        payload = sum(group.gather(local_rows)) * args.hidden * 2 / world
        idx = idx.to(dtype=topk_idx_t, device="npu")
        weights = cpu_w.npu()
        results = []
        for op in ("fresh", "fresh_weighted", "cached"):
            configs = [
                (core, flag, metadata)
                for core in args.cores
                for flag in args.cached_channel_modes
                for metadata in args.metadata_modes
                if flag == 0 or (world == 16 and core == 8 and op == "cached")
            ]
            if not configs:
                continue
            reference = None
            policies = {}
            for turn, (core, flag, metadata) in enumerate(configs + configs[::-1]):
                group.agree([op, turn, core, flag, metadata])
                torch.npu.synchronize()
                os.environ[CHANNEL_ENV] = str(flag)
                os.environ[METADATA_ENV] = str(metadata)
                handle = None

                def invoke():
                    result = done(
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
                    if getattr(native, "dispatch_channel_metadata_version", 0) == 1:
                        effective = bool(native.last_dispatch_channel_metadata())
                        if effective != bool(metadata and handle is None):
                            raise RuntimeError("native channel-metadata gate did not match this launch")
                    return result

                initial = invoke()
                policy = asdict(initial[3]._udma_policy)
                key = (core, flag)
                if key in policies and policies[key] != policy:
                    raise RuntimeError("metadata toggle changed the payload policy")
                policies[key] = policy
                if op == "cached":
                    handle = initial[3]
                checked_result = invoke()
                checked = snapshot(checked_result)
                if 1 in args.metadata_modes:
                    check_cpu_reference(
                        torch,
                        checked,
                        args,
                        rank,
                        world,
                        cached=op == "cached",
                        src_idx=checked_result[3]._dispatch_layout[3].cpu(),
                    )
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
                capture = out / f"{op}_round{turn}_aiv{core}_channel{flag}_metadata{metadata}"
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
                for actual, expected in zip(snapshot(invoke()), reference):
                    if expected is not None:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
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
                        metadata_requested=metadata,
                        metadata_effective=bool(metadata and op != "cached"),
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
        comparisons = channel_comparison(results)
        if comparisons:
            with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
                writer.writeheader()
                writer.writerows(comparisons)
    finally:
        os.environ[TRACE_ENV] = "0"
        os.environ[METADATA_ENV] = "0"
        buffer.destroy()
    group.agree("complete")
    (out / "exit.json").write_text('{"exit_code": 0, "variant_precision_passed": true}', encoding="utf-8")


if __name__ == "__main__":
    main()
