"""Ordinary combine diagnostics using the dispatch benchmark's existing Buffer."""

import csv
from dataclasses import asdict
import json
import math
import os
import statistics
import time

from stage_metrics import prepare_v2_sample

TRACE_ENV = "DEEPEP_CLOS_COMBINE_DEVICE_TIMING"
COMPACT_ENV = "DEEPEP_CLOS_COMBINE_COMPACT_SELF"


def capture_combine_stages(directory, samples):
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError("expected one combine profiler export")
    with paths[0].open(encoding="utf-8-sig", newline="") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: float(row["Start Time(us)"]))
    names = ["cached_notify_combine_kernel", "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"]
    selected = [row for row in rows if row["Name"] in names]
    if [row["Name"] for row in selected] != names * (samples + 4):
        raise RuntimeError("unexpected combine kernel sequence; retain raw export")
    result = {name: [float(row["Duration(us)"]) for row in selected[i::3]][2:-2] for i, name in enumerate(names) if i}
    if any(not math.isfinite(value) or value <= 0 for values in result.values() for value in values):
        raise ValueError("invalid combine duration")
    return result


def rank_max_stats(group, values):
    gathered = group.gather(values)
    maxima = sorted(max(rank_values[i] for rank_values in gathered) for i in range(len(values)))
    return dict(
        mean=statistics.mean(maxima), p50=statistics.median(maxima), p95=maxima[math.ceil(0.95 * len(maxima)) - 1]
    )


def run_combine_diagnostics(
    buffer,
    native,
    torch,
    torch_npu,
    group,
    args,
    flush,
    cpu_x,
    owners,
    cpu_w,
    x,
    idx,
    weights,
    payload,
    out,
    decode_trace,
):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    multiplicity = sum((owners == peer).any(1).to(torch.float32) for peer in range(world))
    expected_x = (cpu_x.float() * multiplicity[:, None]).to(cpu_x.dtype)
    expected_w = cpu_w.masked_fill(owners < 0, 0)
    results = []
    configs = [
        (core, weighted, compact)
        for core in args.cores
        for weighted in (False, True)
        for compact in args.compact_self_modes
    ]
    os.environ[TRACE_ENV] = "0"

    def done(value):
        if value[-1].event is not None:
            value[-1].current_stream_wait()
        torch.npu.synchronize()
        return value

    def check(value, weighted):
        torch.testing.assert_close(value[0].cpu(), expected_x, rtol=0, atol=0)
        if weighted:
            torch.testing.assert_close(value[1].cpu(), expected_w, rtol=0, atol=0)
        elif value[1] is not None:
            raise AssertionError("unweighted combine returned weights")

    try:
        for turn, (core, weighted, compact) in enumerate(configs + configs[::-1]):
            group.agree(["combine_diagnostic", turn, core, weighted, compact])
            os.environ[COMPACT_ENV] = str(compact)
            # Prepare input outside all measured passes, without a second Buffer.
            received = done(
                buffer.dispatch(
                    x,
                    topk_idx=idx,
                    topk_weights=weights if weighted else None,
                    num_experts=args.experts,
                    num_sms=core,
                    do_expand=False,
                )
            )
            handle = received[3]
            plan = buffer._udma_auto_policy.for_launch(
                core, args.hidden * 2, "weighted" if weighted else "combine", bf16=True, expanded=False, v2=True
            )

            def invoke():
                return done(
                    buffer.combine(
                        received[0],
                        handle,
                        topk_weights=received[2] if weighted else None,
                        combine_num_sms=core,
                        reduce_num_sms=args.reduce_cores,
                    )
                )

            check(invoke(), weighted)
            for _ in range(args.warmup):
                prepare_v2_sample(buffer, flush)
                invoke()
            api = []
            for _ in range(args.samples):
                prepare_v2_sample(buffer, flush)
                start = time.perf_counter_ns()
                invoke()
                api.append((time.perf_counter_ns() - start) / 1000)
            capture = out / f"combine_round{turn}_aiv{core}_weighted{int(weighted)}_compact{compact}"
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
            stages = capture_combine_stages(capture, args.samples)
            stages["api_wall_us"] = api
            stats = {key: rank_max_stats(group, values) for key, values in stages.items()}
            (capture / "samples.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
            traces, diagnostic_api = [], []
            os.environ[TRACE_ENV] = "1"
            for _ in range(args.trace_warmup):
                prepare_v2_sample(buffer, flush)
                invoke()
            for _ in range(args.trace_samples):
                prepare_v2_sample(buffer, flush)
                start = time.perf_counter_ns()
                value = invoke()
                diagnostic_api.append((time.perf_counter_ns() - start) / 1000)
                tensor = native.take_combine_device_timing()
                if tensor is None:
                    raise RuntimeError("combine diagnostic tensor missing")
                traces.append(
                    decode_trace(
                        tensor.cpu().tolist(),
                        list(native.combine_device_timing_fields),
                        rank,
                        core,
                        False,
                        False,
                        native.dispatch_device_timing_cycles_per_us,
                        operation=1,
                    )
                )
                check(value, weighted)
            os.environ[TRACE_ENV] = "0"
            check(invoke(), weighted)
            diagnostic_stats = rank_max_stats(group, diagnostic_api)
            (capture / "device_trace.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
            (capture / "trace_overhead.json").write_text(
                json.dumps(
                    dict(
                        diagnostic_api_us=diagnostic_api,
                        diagnostic_api_stats=diagnostic_stats,
                        formal_api_stats=stats["api_wall_us"],
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            results.append(
                dict(
                    turn=turn,
                    cores=core,
                    reduce_cores=args.reduce_cores,
                    weighted=weighted,
                    compact_self=compact,
                    policy=asdict(plan),
                    stats_us=stats,
                    payload_bytes_per_rank=payload,
                    payload_gbps=payload / stats["v1_combine_udma_comm_kernel"]["mean"] / 1000,
                    note="Alignment notify is excluded; communication Begin is inside comm. Reduce is separate.",
                )
            )
            (out / "combine_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        os.environ[TRACE_ENV] = "0"
        os.environ[COMPACT_ENV] = "0"
    group.agree("combine_diagnostics_passed")
