# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Measurement protocol, rank aggregation and versioned result serialization."""

import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import statistics
import time

from ._config import Config, traffic

SCHEMA_VERSION = 1
TIMING = {
    "clock": "perf_counter_ns",
    "boundary": "host_call_to_device_output_ready",
    "includes": [
        "host_api",
        "allocation_in_api",
        "notify_for_fresh_dispatch",
        "completion_wait",
    ],
    "excludes": [
        "input_generation",
        "correctness",
        "pre_iteration_barrier",
        "sample_gather",
        "output_release",
    ],
    "rank_aggregation": "max_per_iteration_then_statistics",
    "percentile": "nearest_rank",
    "cache_policy": "reused_inputs_no_explicit_cache_flush",
    "roundtrip": "direct_dispatch_to_matching_handle_combine_no_expert_compute",
    "roundtrip_fp8": "received_fp8_to_bf16_conversion_included",
    "combine": "matching_handle_and_bf16_input_prepared_before_timing",
}

# Preserve the existing CSV columns and append versioned protocol details.
CSV_FIELDS = (
    "record_id,measured_at,revision,build_info_path,environment_path,device,topology_id,"
    "topology_path,nodes,world_size,api_family,operation,dtype,tokens_per_rank,hidden,"
    "topk,experts,variant,policy_path,num_aiv,seed,warmup,iterations,timing_boundary,"
    "rank_aggregation,latency_p50_us,latency_p95_us,effective_bytes,bandwidth_definition,"
    "bandwidth_gbps,baseline_revision,baseline_latency_us,speedup,correctness_result,"
    "reproduce_command,raw_results_path,notes,schema_version,status,backend,dispatch_dtype,"
    "combine_dtype,layout,dispatch_mode,alignment,with_weights,route,latency_mean_us,"
    "latency_min_us,latency_max_us,bandwidth_statistic,config_fingerprint"
).split(",")


def plan(config):
    config.validate()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "required_capabilities": sorted(config.required_capabilities()),
        "timing": dict(TIMING),
        "traffic": traffic(config),
        "correctness_result": "not_run",
    }


def aggregate_samples(samples, world_size, iterations):
    """samples is rank-major and iteration-aligned; reject partial/nonfinite data."""
    if type(world_size) is not int or type(iterations) is not int or min(world_size, iterations) < 1:
        raise ValueError("world_size and iterations must be positive integers")
    if not isinstance(samples, (tuple, list)) or len(samples) != world_size:
        raise ValueError("incomplete rank samples")
    for row in samples:
        if not isinstance(row, (tuple, list)) or len(row) != iterations:
            raise ValueError("incomplete iteration samples")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in row):
            raise ValueError("latency samples must be finite and positive")
    maximums = [max(row[i] for row in samples) for i in range(iterations)]
    ordered = sorted(maximums)
    return maximums, {
        "latency_mean_us": statistics.fmean(maximums),
        "latency_min_us": ordered[0],
        "latency_max_us": ordered[-1],
        "latency_p50_us": ordered[math.ceil(iterations * 0.50) - 1],
        "latency_p95_us": ordered[math.ceil(iterations * 0.95) - 1],
    }


def _verify_collectively(runtime):
    # Every rank participates even if its checker finds a local mismatch.
    # Transport/native failures still require the adapter's bounded timeout.
    error = None
    try:
        runtime.verify()
        runtime.synchronize()
    except Exception as caught:
        error = caught
    if runtime.all_ranks_ok(error is None) is not True:
        raise RuntimeError("correctness verification failed on at least one rank") from error
    if error is not None:
        raise RuntimeError("backend incorrectly accepted failed correctness") from error


def measure(config, runtime, *, clock=time.perf_counter_ns):
    """Run one prepared native session; teardown is owned by the caller.

    invoke() returns an output lease. wait() joins all communication/copy/reduce
    dependencies onto the observer stream; synchronize() waits for consumption.
    The lease remains live until after that boundary. No reference computations,
    rank aggregation, or release are inside the measured interval.
    """
    config.validate()
    if (
        type(runtime.world_size) is not int
        or runtime.world_size != config.world_size
        or type(runtime.rank) is not int
        or not 0 <= runtime.rank < config.world_size
    ):
        raise ValueError("backend rank/world size does not match the configuration")
    runtime.agree_config(config.fingerprint())
    runtime.prepare()
    _verify_collectively(runtime)

    samples = []
    for iteration in range(config.warmup + config.iterations):
        runtime.barrier()
        runtime.synchronize()
        timed = iteration >= config.warmup
        start = clock() if timed else None
        output = runtime.invoke()
        runtime.wait(output)
        runtime.synchronize()
        elapsed = (clock() - start) / 1000.0 if timed else None
        runtime.release(output)
        if timed:
            samples.append(elapsed)

    _verify_collectively(runtime)
    gathered = runtime.gather_samples(samples)
    maximums, stats = aggregate_samples(gathered, config.world_size, config.iterations)
    stats["bandwidth_gbps"] = traffic(config)["effective_bytes"] / (stats["latency_mean_us"] * 1000)
    stats["bandwidth_statistic"] = "effective_bytes_over_mean_rank_max_latency"
    return {
        "rank_samples_us": gathered,
        "rank_max_samples_us": maximums,
        "statistics": stats,
    }


def measured_result(config, measurements, *, revision, metadata):
    """Metadata is explicit and bounded; never scrape hostnames, env vars or paths."""
    import re

    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase Git SHA")
    allowed = {"backend", "device", "software", "topology", "variant"}
    if not isinstance(metadata, dict) or set(metadata) != allowed:
        raise ValueError("metadata must contain exactly backend/device/software/topology/variant")
    for value in metadata.values():
        if not isinstance(value, str) or not 1 <= len(value) <= 256 or any(ord(c) < 32 for c in value):
            raise ValueError("metadata values must be short, nonempty single-line strings")
        if value.lstrip().startswith(("=", "+", "-", "@")):
            raise ValueError("metadata must not start with a spreadsheet formula prefix")
    # Recompute rather than trust externally supplied summary numbers.
    maximums, stats = aggregate_samples(measurements["rank_samples_us"], config.world_size, config.iterations)
    report = plan(config)
    stats["bandwidth_gbps"] = report["traffic"]["effective_bytes"] / (stats["latency_mean_us"] * 1000)
    stats["bandwidth_statistic"] = "effective_bytes_over_mean_rank_max_latency"
    report.update(
        status="measured",
        correctness_result="passed_pre_and_post",
        measured_at=datetime.now(timezone.utc).isoformat(),
        revision=revision,
        metadata=dict(metadata),
        rank_samples_us=measurements["rank_samples_us"],
        rank_max_samples_us=maximums,
        statistics=stats,
    )
    return report


def summary_csv(report):
    if report.get("status") != "measured":
        raise ValueError("only verified measurements have a CSV summary")
    config = report["config"]
    row = {
        key: config[key]
        for key in (
            "world_size",
            "operation",
            "dtype",
            "hidden",
            "topk",
            "experts",
            "seed",
            "warmup",
            "iterations",
            "layout",
            "dispatch_mode",
            "alignment",
            "with_weights",
            "route",
        )
    }
    row.update(report["statistics"])
    row.update(
        schema_version=SCHEMA_VERSION,
        status="measured",
        record_id=report["config_fingerprint"][:16],
        config_fingerprint=report["config_fingerprint"],
        measured_at=report["measured_at"],
        revision=report["revision"],
        api_family=config["api"],
        backend=report["metadata"]["backend"],
        device=report["metadata"]["device"],
        variant=report["metadata"]["variant"],
        topology_id=report["metadata"]["topology"],
        tokens_per_rank=json.dumps(config["tokens_per_rank"], separators=(",", ":")),
        dispatch_dtype=config["dtype"],
        combine_dtype="bf16" if config["operation"] != "dispatch" else "",
        timing_boundary=TIMING["boundary"],
        rank_aggregation=TIMING["rank_aggregation"],
        effective_bytes=report["traffic"]["effective_bytes"],
        bandwidth_definition=report["traffic"]["definition"],
        correctness_result=report["correctness_result"],
        raw_results_path="result.json",
        environment_path="result.json",
        reproduce_command="torchrun <launch> -m benchmarks.bench_ep --config config.json --revision "
        + report["revision"]
        + " --output-dir <new-directory>",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue()


def write_result(directory, report):
    """Create a new run directory. Existing files/directories are never overwritten."""
    # Validate serialization and summary BEFORE reserving the destination.
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    config = Config.from_dict(report["config"])
    if report["status"] not in ("planned", "measured"):
        raise ValueError("invalid result status")
    csv_text = summary_csv(report) if report["status"] == "measured" else None
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    filename = "result.json" if csv_text is not None else "plan.json"
    (path / filename).write_text(encoded, encoding="utf-8")
    (path / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    if csv_text is not None:
        (path / "summary.csv").write_text(csv_text, encoding="utf-8")
