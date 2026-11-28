"""Fresh dispatch layout/notify measurement, one fixed-mode Buffer per rank, precision before timing."""

import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import time

HOST_PREP_ENV = "DEEPEP_CLOS_NOTIFY_HOST_PREP"
DEFER_PLAN_ENV = "DEEPEP_CLOS_NOTIFY_DEFER_PLAN"
HOST_TIMING_ENV = "DEEPEP_CLOS_DISPATCH_HOST_TIMING"
SIMT_LAYOUT_ENV = "DEEPEP_CLOS_V2_SIMT_LAYOUT"
MODES = {
    "baseline": (0, 0, 0),
    "fused": (1, 0, 0),
    "host": (1, 1, 0),
    "plan": (1, 0, 1),
    "both": (1, 1, 1),
    "simt": (0, 0, 0),
}


def mode_order(sweep, simt_layout=False, fused_layout_notify=False):
    names = (
        ["baseline", "simt"]
        if simt_layout
        else ([name for name in MODES if name != "simt"] if sweep else ["baseline", "fused"])
    )
    names = [name for name in names if bool(MODES[name][0]) == fused_layout_notify]
    return names + names[::-1]


def set_mode(name):
    for env, value in zip((HOST_PREP_ENV, DEFER_PLAN_ENV), MODES[name][1:]):
        os.environ[env] = str(value)
    os.environ[SIMT_LAYOUT_ENV] = str(int(name == "simt"))


def host_durations(rows, fields, samples, mode):
    """Same-call host intervals; never subtract profiler and unprofiled runs."""
    if len(rows) != samples:
        raise RuntimeError(f"expected {samples} Host rows, found {len(rows)}")
    intervals = {
        "api_to_notify_submit": ("api_entry", "notify_submitted"),
        "python_front": ("api_entry", "native_entry"),
        "native_prepare": ("native_entry", "notify_submit_begin"),
        "notify_launch": ("notify_submit_begin", "notify_submitted"),
        "post_launch_prepare": ("notify_submitted", "count_wait_begin"),
        "wait_total": ("count_wait_begin", "total_observed"),
        "wait_remaining_counts": ("total_observed", "counts_observed"),
        "notify_to_counts": ("notify_submitted", "counts_observed"),
        "count_to_outputs": ("counts_observed", "output_ready"),
        "comm_launch": ("comm_submit_begin", "comm_submitted"),
        "comm_submit_to_complete": ("comm_submitted", "complete"),
        "api_host_return": ("api_entry", "api_return"),
        "api_complete_diagnostic": ("api_entry", "complete"),
    }
    result = {key: [] for key in intervals}
    for raw in rows:
        if len(raw) != len(fields):
            raise RuntimeError("Host timing schema mismatch")
        row = dict(zip(fields, raw))
        if tuple(row[key] for key in ("fused", "host_prep", "deferred_plan")) != MODES[mode]:
            raise RuntimeError("native pipeline flags do not match requested mode")
        timestamps = [row[key] for key in fields if key not in ("fused", "host_prep", "deferred_plan", "mapped_counts")]
        if any(t <= 0 for t in timestamps) or timestamps != sorted(timestamps):
            raise RuntimeError("missing or out-of-order Host stage timestamps")
        for name, (start, end) in intervals.items():
            result[name].append((row[end] - row[start]) / 1000)
    return result


@contextmanager
def store_topology_collective(group):
    """Adapt constructor topology discovery to this benchmark's TCPStore group.

    Keep bind_group's real host identities and topology inference. Other EP
    groups retain their normal collective; restore it even if initialization fails.
    """
    from deep_ep import _machine

    original = _machine.gather_records

    def gather_records(ep_group, record):
        if ep_group is group:
            return group.gather(record)
        return original(ep_group, record)

    _machine.gather_records = gather_records
    try:
        yield
    finally:
        _machine.gather_records = original


def parse_capture(directory, fused, samples, expanded=False, guards=0, simt_layout=False):
    if fused and simt_layout:
        raise ValueError("SIMT layout must be separate from notify")
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one kernel_details.csv, found {len(paths)}")
    with paths[0].open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["Start Time(us)"]))
    from stage_metrics import normalize_kernel_rows

    rows = normalize_kernel_rows(rows)
    names = ["cached_notify_combine_kernel"]
    if not fused:
        names.append("dispatch_layout_simt_kernel" if simt_layout else "dispatch_layout_kernel")
    names += ["notify_dispatch_kernel", "direct_dispatch_udma_kernel"]
    if expanded:
        names.append("expand_kernel")
    known = set(names) | {"dispatch_layout_kernel", "dispatch_layout_simt_kernel", "expand_kernel"}
    selected = [r for r in rows if r["Name"] in known]
    if [r["Name"] for r in selected] != names * (samples + 2 * guards):
        raise RuntimeError("unexpected barrier/layout/notify/comm sequence or sample count")
    result = {
        name: [float(r["Duration(us)"]) for r in selected[i :: len(names)]][guards : guards + samples]
        for i, name in enumerate(names)
    }
    if any(not math.isfinite(t) or t <= 0 for values in result.values() for t in values):
        raise RuntimeError("invalid kernel duration")
    if simt_layout:
        result["dispatch_layout_kernel"] = result.pop("dispatch_layout_simt_kernel")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--expand", type=int, choices=(0, 1), default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--control-port", type=int, default=29871)
    p.add_argument("--shmem-port", type=int, default=29872)
    p.add_argument("--huge-pages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--flush-l2", action="store_true")
    p.add_argument("--precision-only", action="store_true")
    p.add_argument(
        "--fused-layout-notify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fix Buffer fusion at initialization; compare separate on/off runs",
    )
    p.add_argument(
        "--pipeline-sweep",
        action="store_true",
        help="fused/host/plan/both, then reverse order; requires --fused-layout-notify",
    )
    p.add_argument("--simt-layout-sweep", action="store_true", help="separate old/SIMT layout ABBA; fusion stays off")
    p.add_argument("--aiv", type=int, choices=range(2, 65, 2), default=8)
    p.add_argument("--host-timing", action="store_true", help="separate unprofiled Host diagnostic pass")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.pipeline_sweep and not args.fused_layout_notify:
        p.error("--pipeline-sweep requires --fused-layout-notify")
    if args.simt_layout_sweep and args.fused_layout_notify:
        p.error("--simt-layout-sweep requires --no-fused-layout-notify")
    if args.pipeline_sweep and args.simt_layout_sweep:
        p.error("choose either --pipeline-sweep or --simt-layout-sweep")
    if min(args.tokens, args.hidden, args.topk, args.experts, args.samples) <= 0 or args.warmup < 0:
        p.error("shape/sample counts must be positive; warmup must be nonnegative")
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep.backends.direct.elastic import ElasticBuffer
    from deep_ep.backends.direct import buffer as bm, elastic as em
    from deep_ep.backends.direct.dtypes import topk_idx_t
    from deep_ep.backends.direct._native import load_native_module
    from bench_fast_sweep import StoreGroup
    from stage_metrics import L2CacheFlush, prepare_v2_sample

    rank, world, device = (int(os.environ[n]) for n in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world < 2 or args.experts % world or args.topk > args.experts:
        p.error("EP must be >=2, divide experts, and topk must not exceed experts")
    torch.npu.set_device(device)
    native = load_native_module()
    if getattr(native, "fused_layout_notify_version", 0) != 2:
        raise RuntimeError("install this branch's native and device libraries before initializing Buffer")
    if getattr(native, "fused_notify_pipeline_version", 0) != 1:
        raise RuntimeError("rebuild native and device libraries for pipeline sweep/Host timing")
    if args.simt_layout_sweep and getattr(native, "v2_simt_layout_version", 0) != 1:
        raise RuntimeError("rebuild native and device libraries for V2 SIMT layout")
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
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                world=world,
                native_path=native.__file__,
                torch_version=torch.__version__,
                torch_npu_version=torch_npu.__version__,
                args=vars(args) | {"output": str(args.output)},
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = []
    old_env = {key: os.environ.get(key) for key in (HOST_PREP_ENV, DEFER_PLAN_ENV, HOST_TIMING_ENV, SIMT_LAYOUT_ENV)}
    os.environ[HOST_TIMING_ENV] = "0"
    with store_topology_collective(group):
        buffer = ElasticBuffer(
            group,
            combine_notify=True,
            fused_layout_notify=args.fused_layout_notify,
            num_max_tokens_per_rank=args.tokens,
            hidden=args.hidden,
            num_topk=args.topk,
            explicitly_destroy=True,
            transport="udma",
            udma_auto_policy=True,
            use_1g_huge_pages=args.huge_pages,
            shmem_ip_port=f"tcp://{os.environ['MASTER_ADDR']}:{args.shmem_port}",
        )
    if rank == 0:
        print(f"BUFFER_READY EP={world} AIV={args.aiv} initializations=1", flush=True)
    flush = L2CacheFlush(torch, buffer, args.flush_l2)

    def done(result):
        if result[-1].event is not None:
            result[-1].current_stream_wait()
        torch.npu.synchronize()
        return result

    def assert_equal(a, b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def snapshot(result, local_x, local_idx, all_idx):
        recv, recv_idx, recv_w, h, _ = done(result)
        layout = [t.cpu().clone() for t in h._dispatch_layout]
        memberships = []
        for idx in all_idx:
            valid = idx >= 0
            owners = idx.clamp_min(0) // (args.experts // world)
            memberships.append(torch.stack([(valid & (owners == peer)).any(1) for peer in range(world)], 1))
        assert_equal(layout[4], memberships[rank])
        counts = torch.stack([m.sum(0) for m in memberships]).to(torch.int32)
        assert_equal(layout[0], counts.cumsum(0).to(torch.int32))
        experts = sum((torch.bincount(idx[idx >= 0], minlength=args.experts) for idx in all_idx))
        e = args.experts // world
        assert_equal(h.num_unaligned_recv_tokens_per_expert.cpu(), experts[rank * e : (rank + 1) * e].to(torch.int32))
        length = (local_idx.shape[0] + args.aiv - 1) // args.aiv
        plan = h._dispatch_plan.cpu()
        stable = []
        for peer in range(world):
            total = 0
            for ch in range(args.aiv):
                begin, end = ch * length, min((ch + 1) * length, local_idx.shape[0])
                indices = torch.where(memberships[rank][begin:end, peer])[0] + begin
                total += len(indices)
                assert int(layout[1][peer, ch]) == total
                assert_equal(plan[peer, ch, : len(indices)], indices.to(torch.int32))
                if len(indices):
                    assert_equal(layout[5][indices, peer], torch.arange(len(indices), dtype=torch.int32))
                stable.append(indices)
        assert bool((layout[5][~memberships[rank]] == -1).all())
        outputs = [recv.cpu().clone()]
        if recv_idx is not None:
            outputs.append(recv_idx.cpu().clone())
        if recv_w is not None:
            outputs.append(
                recv_w.cpu().masked_fill(recv_idx.cpu() < 0, 0) if recv_idx is not None else recv_w.cpu().clone()
            )
        outputs += layout + stable
        # Exercise the handle in cached dispatch and both combine weight modes.
        cached = done(buffer.dispatch(local_x, handle=h, num_sms=args.aiv, do_expand=bool(args.expand)))
        assert_equal(cached[0], recv)
        for weighted in (False, True):
            w = recv_w if weighted else None
            combined = done(buffer.combine(recv, h, topk_weights=w, num_sms=args.aiv))
            outputs.append(combined[0].cpu().clone())
            if combined[1] is not None:
                outputs.append(combined[1].cpu().clone())
        return outputs

    try:
        gen = torch.Generator().manual_seed(20261122 + rank)
        x = (torch.randn((args.tokens, args.hidden), generator=gen) * 0.01).to(torch.bfloat16).npu()
        raw = torch.rand((args.tokens, args.experts), generator=gen).topk(args.topk, dim=1).indices
        weights = torch.rand((args.tokens, args.topk), generator=gen).npu()
        for routing in ("random", "masked", "hot", "empty", "empty_local"):
            if rank == 0:
                print(
                    f"PRECISION routing={routing} expand={args.expand} order={mode_order(args.pipeline_sweep, args.simt_layout_sweep, args.fused_layout_notify)}",
                    flush=True,
                )
            cpu_idx = raw.clone()
            if routing == "masked":
                cpu_idx[::3, -1] = -1
                cpu_idx[::7] = -1
            elif routing == "hot":
                cpu_idx[:] = torch.arange(args.topk)  # empty destination peers, repeated destination rank
            elif routing == "empty":
                cpu_idx.fill_(-1)
            elif routing == "empty_local" and rank == 0:
                cpu_idx = cpu_idx[:0]
            local_x, local_weights = x[: len(cpu_idx)], weights[: len(cpu_idx)]
            all_idx = [
                torch.tensor(v, dtype=torch.int64).reshape(-1, args.topk) for v in group.gather(cpu_idx.tolist())
            ]
            idx = cpu_idx.to(dtype=topk_idx_t, device="npu")
            for weighted in (False, True):
                baseline = None
                for mode in mode_order(args.pipeline_sweep, args.simt_layout_sweep, args.fused_layout_notify):
                    torch.npu.synchronize()
                    group.agree([routing, weighted, mode])
                    set_mode(mode)
                    result = buffer.dispatch(
                        local_x,
                        topk_idx=idx,
                        topk_weights=local_weights if weighted else None,
                        num_experts=args.experts,
                        num_sms=args.aiv,
                        do_expand=bool(args.expand),
                    )
                    actual = snapshot(result, local_x, cpu_idx, all_idx)
                    if baseline is None:
                        baseline = actual
                    else:
                        assert len(actual) == len(baseline)
                        for a, b in zip(actual, baseline):
                            assert_equal(a, b)
        group.agree("precision_passed")
        (out / "precision.json").write_text(json.dumps({"passed": True, "buffer_initializations": 1}), encoding="utf-8")
        if not args.precision_only:
            idx = raw.to(dtype=topk_idx_t, device="npu")
            owners = raw // (args.experts // world)
            local_rows = sum(int((owners == peer).any(1).sum()) for peer in range(world))
            mean_payload_bytes = sum(group.gather(local_rows)) * args.hidden * 2 / world
            for weighted in (False, True):
                reference_policy = None

                def invoke():
                    return done(
                        buffer.dispatch(
                            x,
                            topk_idx=idx,
                            topk_weights=weights if weighted else None,
                            num_experts=args.experts,
                            num_sms=args.aiv,
                            do_expand=bool(args.expand),
                        )
                    )

                for turn, mode in enumerate(
                    mode_order(args.pipeline_sweep, args.simt_layout_sweep, args.fused_layout_notify)
                ):
                    if rank == 0:
                        print(f"PERFORMANCE weights={weighted} round={turn} fused={mode}", flush=True)
                    torch.npu.synchronize()
                    group.agree(["performance", weighted, turn, mode])
                    set_mode(mode)
                    check = invoke()
                    policy = asdict(check[3]._udma_policy)
                    group.agree(policy)
                    if reference_policy is not None and policy != reference_policy:
                        raise RuntimeError("dispatch policy changed during the A/B comparison")
                    reference_policy = policy
                    for _ in range(args.warmup):
                        prepare_v2_sample(buffer, flush)
                        invoke()
                    api = []
                    for _ in range(args.samples):
                        prepare_v2_sample(buffer, flush)
                        start = time.perf_counter_ns()
                        invoke()
                        api.append((time.perf_counter_ns() - start) / 1000)
                    capture = out / f"weight{int(weighted)}_round{turn}_{mode}"
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
                    stages = parse_capture(
                        capture,
                        bool(MODES[mode][0]),
                        args.samples,
                        bool(args.expand),
                        guards=2,
                        simt_layout=mode == "simt",
                    )
                    stages["api_wall_us_unprofiled"] = api
                    stages["layout_plus_notify"] = [
                        a + b
                        for a, b in zip(
                            stages.get("dispatch_layout_kernel", [0] * args.samples), stages["notify_dispatch_kernel"]
                        )
                    ]
                    gathered = group.gather(stages)
                    summary = {
                        key: statistics.mean(max(g[key][i] for g in gathered) for i in range(args.samples))
                        for key in stages
                    }
                    rows.append(
                        dict(
                            weighted=weighted,
                            turn=turn,
                            mode=mode,
                            fused=MODES[mode][0],
                            host_prep=MODES[mode][1],
                            deferred_plan=MODES[mode][2],
                            means_us=summary,
                            policy=policy,
                            logical_payload_bytes_per_rank=mean_payload_bytes,
                            payload_gbps=mean_payload_bytes / summary["direct_dispatch_udma_kernel"] / 1000,
                        )
                    )
                    (capture / "samples.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
                    if args.host_timing:
                        for _ in range(args.warmup):
                            prepare_v2_sample(buffer, flush)
                            invoke()
                        native.take_dispatch_host_timings()
                        os.environ[HOST_TIMING_ENV] = "1"
                        try:
                            for _ in range(args.samples):
                                prepare_v2_sample(buffer, flush)
                                invoke()
                                native.complete_dispatch_host_timing()
                            host_rows = native.take_dispatch_host_timings()
                        finally:
                            os.environ[HOST_TIMING_ENV] = "0"
                        fields = list(native.dispatch_host_timing_fields)
                        durations = host_durations(host_rows, fields, args.samples, mode)
                        (capture / "host_timing.json").write_text(
                            json.dumps(
                                dict(
                                    fields=fields,
                                    rows_ns=host_rows,
                                    durations_us=durations,
                                    mapped_counts=[r[fields.index("mapped_counts")] for r in host_rows],
                                ),
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        host_all = group.gather(durations)
                        rows[-1]["host_diagnostic_means_us"] = {
                            key: statistics.mean(max(g[key][i] for g in host_all) for i in range(args.samples))
                            for key in durations
                        }
                        rows[-1]["mapped_counts_all_samples"] = all(
                            group.gather(all(r[fields.index("mapped_counts")] == 1 for r in host_rows))
                        )
        (out / "report.json").write_text(
            json.dumps(
                dict(
                    args=vars(args) | {"output": str(args.output)},
                    precision_passed=True,
                    buffer_initializations=1,
                    aiv=args.aiv,
                    results=rows,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [
                "weighted",
                "turn",
                "fused",
                "mode",
                "host_prep",
                "deferred_plan",
                "layout_us",
                "notify_us",
                "layout_notify_us",
                "comm_us",
                "expand_us",
                "api_wall_us",
                "payload_gbps",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                s = row["means_us"]
                writer.writerow(
                    dict(
                        weighted=row["weighted"],
                        turn=row["turn"],
                        fused=row["fused"],
                        mode=row["mode"],
                        host_prep=row["host_prep"],
                        deferred_plan=row["deferred_plan"],
                        layout_us=s.get("dispatch_layout_kernel", 0),
                        notify_us=s["notify_dispatch_kernel"],
                        layout_notify_us=s["layout_plus_notify"],
                        comm_us=s["direct_dispatch_udma_kernel"],
                        expand_us=s.get("expand_kernel", 0),
                        api_wall_us=s["api_wall_us_unprofiled"],
                        payload_gbps=row["payload_gbps"],
                    )
                )
        if args.host_timing and rows:
            with (out / "host_summary.csv").open("w", newline="", encoding="utf-8") as f:
                fields = ["weighted", "turn", "mode", "mapped_counts_all_samples"]
                fields += list(rows[0]["host_diagnostic_means_us"])
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row[key] for key in fields[:4]} | row["host_diagnostic_means_us"])
    finally:
        buffer.destroy()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    group.agree("complete")
    (out / "exit.json").write_text('{"exit_code": 0}', encoding="utf-8")


if __name__ == "__main__":
    main()
