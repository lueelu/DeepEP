# Copyright (c) 2026, Lu Lu
# Modified by ryan_li 2026

"""Installed-wheel LL functional/performance ST, modeled on DeepEP test_ep.py.

Run explicitly; ordinary pytest never initializes NPU devices from this module.
Use --help for launch, configuration, and profiling options.
"""

import argparse
import csv
import hashlib
import math
import re
import tempfile
import json
from itertools import product
import os
from pathlib import Path
import random
import runpy
import statistics
import sys
from datetime import datetime, timedelta

__test__ = False
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_stage_timer_capability(requested, capability):
    if not requested:
        return
    if not capability["stage_timer"]:
        raise RuntimeError("All ranks must install a DEEPEP_LL_STAGE_TIMER=ON wheel")
    if capability["stage_timer_schema"] != 3:
        raise RuntimeError("LL byte diagnostics require timer schema 3; rebuild/install the matching wheel on every rank")


EXP_PREFIX_REPEATS = 100
SAMPLING_PROTOCOL = 'initial-reduce-exp100-batch-v3'
TCPSTORE_SAMPLING_PROTOCOL = 'initial-tcpstore-exp100-batch-v2'
CLOCK_ALIGNMENT_PROTOCOL = 'tilexr-sleep-at-per-iteration-v1'
CLOCK_CYCLES_PER_SECOND = 1_000_000_000
BALANCED_ROUTE_SEED = 20261014


def routing_metadata(world, b, e, seed, route):
    balanced = route == 'normal'
    return dict(generator='balanced-rank-distinct-roundrobin-v1' if balanced else 'hotspot-expert-sample-v1',
                seed_offset=seed, base_seed=BALANCED_ROUTE_SEED if balanced else None,
                rng_seed=BALANCED_ROUTE_SEED + seed + b * 1000 + 2 if balanced else seed,
                physical_stride=e // world, hot_multiplier_x2=2 if balanced else None)


def diagnostic_requested(args):
    return args.ll_stage_timer or getattr(args, 'll_reference_profile', False)


def validate_reference_profile_capability(requested, capability):
    if not requested:
        return
    if (capability.get('reference_profile') is not True or capability.get('stage_timer') is not False or
            type(capability.get('reference_profile_schema')) is not int or capability['reference_profile_schema'] != 4):
        raise RuntimeError('All ranks need the matching reference-profile schema 1 wheel with legacy stage tracing OFF')
    fingerprint = capability.get('installed_source_sha256')
    if not isinstance(fingerprint, str) or re.fullmatch(r'[0-9a-f]{64}', fingerprint) is None:
        raise RuntimeError('Reference profiling requires the installed source fingerprint on every rank')


MATRIX_FIELDS = ("num_tokens", "dtype", "num_experts", "hidden", "num_topk", "do_cpu_sync", "fp8_scale_dtype")


class MatrixValues(argparse.Action):
    """Keep scalar command compatibility while accepting space-separated values."""

    def __call__(self, parser, namespace, values, option_string=None):
        values = list(dict.fromkeys(values))
        setattr(namespace, self.dest, values[0] if len(values) == 1 else values)


def matrix_cases(args):
    axes = [value if isinstance(value, list) else [value] for value in (getattr(args, key) for key in MATRIX_FIELDS)]
    for values in product(*axes):
        case = argparse.Namespace(**vars(args))
        for key, value in zip(MATRIX_FIELDS, values):
            setattr(case, key, value)
        yield case


def case_config(args, m):
    return {
        "B": args.num_tokens,
        "M": m,
        "H": args.hidden,
        "num_topk": args.num_topk,
        "num_experts": args.num_experts,
        "dtype": args.dtype,
        "do_cpu_sync": bool(args.do_cpu_sync),
        "fp8_scale_dtype": args.fp8_scale_dtype,
    }


def compiled_group(native, world):
    value=native.low_latency_ranks_per_group
    group=value(world) if callable(value) else value
    if type(group) is not int or not 1 <= group <= 16 or world%group:
        raise RuntimeError('Invalid compiled LL logical group')
    return group


def comparison_metadata(args, world, m):
    runtime=getattr(args,'runtime_identity',None)
    if runtime is None: return None  # CPU fixtures / legacy custom callers.
    clock_aligned = bool(getattr(args, "clock_align", False))
    return dict(**runtime, comparison_schema=1, sampling_protocol=sampling_protocol(args),
                host_coordination=args.group_backend,
                exp_prefix_repeats=0 if clock_aligned else EXP_PREFIX_REPEATS,
                per_iteration_exp=0, per_iteration_all_reduce=False,
                per_iteration_host_collective=False,
                per_iteration_rank_alignment=CLOCK_ALIGNMENT_PROTOCOL if clock_aligned else "none",
                per_iteration_device_alignment=CLOCK_ALIGNMENT_PROTOCOL if clock_aligned else "none",
                clock_alignment=getattr(args, "clock_alignment_metadata", None),
                perf_enabled=not args.skip_perf_test, world_size=world,
                local_world_size=int(os.environ.get('LOCAL_WORLD_SIZE',args.num_processes)),
                case=case_config(args,m), route=args.route, seed=args.seed,
                routing=routing_metadata(world, args.num_tokens, args.num_experts, args.seed, args.route),
                route_sha256=getattr(args, 'sample_route_sha256', None),
                warmup=args.warmup, num_iters=args.num_iters, perf_tail=args.perf_tail,
                delay_rows=args.delay_rows, delay_cols=args.delay_cols, group_backend=args.group_backend,
                timer_mode=('reference' if getattr(args, 'll_reference_profile', False) else args.ll_timer_mode) if diagnostic_requested(args) else None,
                timer_iters=args.ll_timer_iters if diagnostic_requested(args) else 0,
                frequency_hz=args.ll_timer_frequency_hz, stage_capture=args.ll_stage_timer,
                reference_capture=bool(getattr(args, 'll_reference_profile', False)),
                expert_capture=args.ll_expert_timer, do_handle_copy=args.do_handle_copy,
                correctness_repeats=args.correctness_repeats, repeat_combine=args.repeat_combine,
                skip_check=args.skip_check, dispatch_only=args.dispatch_only,
                fresh_buffer_per_case=args.fresh_buffer_per_case,
                trace_ops=args.trace_ops, num_bytes=args.num_bytes,
                use_1g_huge_pages=(runtime['transport']!='rdma' if args.use_1g_huge_pages is None else bool(args.use_1g_huge_pages)),
                source_revision=os.environ.get('DEEPEP_LL_RUN_REVISION','unknown'),
                source_dirty=os.environ.get('DEEPEP_LL_RUN_DIRTY','unknown'))


def validate_native_cases(native, cases, world):
    """Host-only capability checks for every combination, before communication."""
    for case in cases:
        dtype = {"bf16": "bfloat16", "fp16": "float16", "fp8": "float8_e4m3fn"}[case.dtype]
        shape = (world, case.num_tokens, case.hidden, case.num_topk, case.num_experts)
        try:
            if native.low_latency_layout(*shape, dtype) is None:
                raise ValueError("dispatch layout rejected")
            native.low_latency_combine_tiling(
                *shape, "float16" if case.dtype == "fp16" else "bfloat16", validate_args(case, world)
            )
        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Unsupported LL combination {case_config(case, validate_args(case, world))}: {exc}"
            ) from exc


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Matrix axes accept space-separated values: --num-tokens 8 16 --dtype bf16 fp8 --do-cpu-sync 0 1",
    )
    parser.add_argument("--config", help="JSON settings and Cartesian matrix")
    parser.add_argument("--plan", action="store_true", help="Print the case matrix without importing Torch")
    parser.add_argument("--output-dir", default="benchmark-runs", help="Parent for a unique run directory and traces")
    parser.add_argument("--run-name", help="Optional fixed child directory name; existing directory is rejected")
    parser.add_argument('--backend',choices=('mte','rdma'),help='Optional override; default is determined by the installed kernel branch')
    parser.add_argument(
        "--perf-tail",
        type=int,
        default=50,
        help="Statistics use last N samples; default: min(50, num_iters); 0 uses all",
    )
    parser.add_argument("--num-processes", type=int, default=8, help="Local spawned processes, as in DeepEP")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--shmem-ip-port", default=None)
    parser.add_argument("--shmem-port", type=int, default=None)
    parser.add_argument(
        "--num-tokens", nargs="+", action=MatrixValues, type=int, default=16, help="Actual B, identical on all ranks"
    )
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=None, help="M; omitted means actual B")
    parser.add_argument("--hidden", nargs="+", action=MatrixValues, type=int, default=7168)
    parser.add_argument("--num-topk", nargs="+", action=MatrixValues, type=int, default=8)
    parser.add_argument("--num-experts", nargs="+", action=MatrixValues, type=int, default=128)
    parser.add_argument("--dtype", nargs="+", action=MatrixValues, choices=("bf16", "fp16", "fp8"), default="bf16")
    parser.add_argument(
        "--fp8-scale-dtype",
        nargs="+",
        action=MatrixValues,
        choices=("int32", "float32"),
        default="int32",
        help="FP8 scale formats; ignored for BF16/FP16",
    )
    parser.add_argument("--do-cpu-sync", nargs="+", action=MatrixValues, type=int, choices=(0, 1), default=1)
    parser.add_argument("--group-backend", choices=("tcpstore", "hccl", "gloo"), default="tcpstore")
    parser.add_argument(
        "--without-buffer-group",
        action="store_true",
        help="Keep the test control group, but construct Buffer with rank/world_size",
    )
    parser.add_argument("--route", choices=("normal", "hotspot"), default="normal")
    parser.add_argument("--do-handle-copy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--num-sms", type=int, default=0, help="LL only supports the default fixed schedule")
    parser.add_argument("--num-qps", type=int, default=0)
    parser.add_argument(
        "--num-bytes", type=int, default=8 << 30, help="LL requires exactly 8 GiB; no additional workspace is allocated"
    )
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trace-ops", action="store_true", help="Print per-rank Host call boundaries; affects performance"
    )
    parser.add_argument(
        "--fresh-buffer-per-case", action="store_true", help="Temporary diagnostic: recreate Buffer for each case"
    )
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-perf-test", action="store_true")
    parser.add_argument("--test-first-only", action="store_true")
    parser.add_argument("--delay-rows", type=int, default=118260)
    parser.add_argument("--delay-cols", type=int, default=30720)
    parser.add_argument(
        "--delay-exp-before", type=int, default=None, help="Compatibility: only 100 is accepted; fixed Exp prefix once per batch"
    )
    parser.add_argument(
        "--delay-exp-after", type=int, default=None, help="Compatibility: zero post-call Exp"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--clock-align", action="store_true",
                        help="Queue an absolute device-cycle wait before every measured operation")
    parser.add_argument("--clock-align-samples", type=int, default=64)
    parser.add_argument("--clock-align-warmup-samples", type=int, default=8)
    parser.add_argument("--clock-align-lead-us", type=int, default=200000)
    parser.add_argument("--clock-align-interval-us", type=int, default=1000)
    parser.add_argument("--clock-align-timeout-secs", type=int, default=10)
    parser.add_argument("--ll-stage-timer", action="store_true", help="Separate device-stage diagnostic calls; requires timer-enabled wheel")
    parser.add_argument("--ll-reference-profile", action="store_true", help="Separate dispatch-only reference cycle capture; requires reference ON and legacy timer OFF wheel")
    parser.add_argument('--ll-expert-timer', action='store_true', help='Opt-in dispatch per-expert receive work; implies stage capture, requires ABI 5')
    parser.add_argument("--ll-timer-iters", type=int, default=5)
    parser.add_argument("--ll-timer-mode", choices=("observe", "drain"), default="observe")
    parser.add_argument("--ll-timer-frequency-hz", type=float, default=1_000_000_000, help="GetSystemCycle frequency; default 1GHz (ticks/1000 us)")
    parser.add_argument("--dispatch-only", action="store_true")
    parser.add_argument("--repeat-combine", type=int, default=2)
    parser.add_argument("--correctness-repeats", type=int, default=3, help="Fresh routes on the same Buffer; outside profiling")
    parser.add_argument("--use-1g-huge-pages", type=int, choices=(0, 1), default=0, help="LL defaults to 0; explicit 1 is MTE-only")
    parser.add_argument("--output", help="New rank-0 JSON result file (will not overwrite)")
    return parser


DISPATCH_PROFILER_ENTRIES = ("dispatch_server_dedup_kernel",)


def kernel_durations(root, kernel, expected):
    """Read only the requested device kernel from torch_npu text exports."""
    rows = []
    entries = DISPATCH_PROFILER_ENTRIES if kernel == "dispatch_server_dedup_kernel" else (kernel,)
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, entries)) + r")\b")
    actual_entries = set()
    actual_task_names = set()
    for path in sorted(Path(root).rglob("kernel_details.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                normalized = {key.replace(" ", "").lower(): value for key, value in row.items() if key}
                name = normalized.get("name", normalized.get("opname", ""))
                matches = pattern.findall(name)
                if not matches:
                    continue
                actual_entries.update(matches)
                if kernel == "dispatch_server_dedup_kernel" and len(actual_entries) > 1:
                    raise ValueError(f"Mixed dispatch profiler entries in {root}: {sorted(actual_entries)}")
                actual_task_names.add(name.strip())
                if kernel == "dispatch_server_dedup_kernel" and len(actual_task_names) > 1:
                    raise ValueError(f"Mixed dispatch profiler task names in {root}: {sorted(actual_task_names)}")
                value = float(normalized.get("duration(us)", normalized.get("taskduration(us)", "nan")))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid kernel duration in {path}: {value}")
                start = normalized.get("starttime(us)")
                rows.append((float(start) if start else None, value))
    if len(rows) != expected:
        raise ValueError(f"{kernel}: expected {expected} profiler records, found {len(rows)} in {root}")
    if all(start is not None for start, _ in rows):
        rows.sort(key=lambda row: row[0])
    return [value for _, value in rows]


def sample_stats(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Kernel samples must be finite and positive")
    ordered = sorted(values)

    def percentile(p):
        position = (len(ordered) - 1) * p
        lo = int(position)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo)

    return {
        "samples": len(values),
        "mean_us": statistics.mean(values),
        "min_us": ordered[0],
        "median_us": statistics.median(values),
        "max_us": ordered[-1],
        "p50_us": percentile(0.50),
        "p90_us": percentile(0.90),
        "p95_us": percentile(0.95),
        "p99_us": percentile(0.99),
    }


def percentile(values, fraction):
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("Percentile requires nonempty values and a fraction in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo)


def summarize_kernel(samples_by_rank, kernel, tail):
    ranks = []
    selected_samples = []
    for rank, raw in enumerate(samples_by_rank):
        selected = raw[-tail:] if tail else raw
        selected_samples.extend(selected)
        ranks.append({"rank": rank, **sample_stats(selected), "kernel_us": raw})
    # Pool individual kernel durations after selecting each rank's stable tail.
    pooled = sample_stats(selected_samples)
    fastest = min(ranks, key=lambda row: row["mean_us"])
    slowest = max(ranks, key=lambda row: row["mean_us"])
    return {
        "measurement": "kernel_duration_us",
        "kernel": kernel,
        "perf_tail": tail,
        "ranks": ranks,
        "summary_statistics_scope": "pooled_rank_tail_kernel_samples",
        "kernel_samples": pooled["samples"],
        **{f"{name}_kernel_us": pooled[f"{name}_us"] for name in ("mean", "p99", "p90", "p50")},
        "min_rank_mean_kernel_us": fastest["mean_us"],
        "median_rank_mean_kernel_us": statistics.median(row["mean_us"] for row in ranks),
        "max_rank_mean_kernel_us": slowest["mean_us"],
        "fastest_rank": fastest["rank"],
        "slowest_rank": slowest["rank"],
    }


def case_result_summary(result):
    """Keep per-rank statistics in case results; raw samples have their own files."""
    summary = dict(result)
    for operation in ("dispatch", "combine"):
        if operation in result:
            summary[operation] = {
                **result[operation],
                "ranks": [
                    {key: value for key, value in row.items() if key != "kernel_us"}
                    for row in result[operation]["ranks"]
                ],
            }
    return summary


def write_reports(directory, report):
    directory = Path(directory)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    summaries, ranks = [], []
    for result in report["results"]:
        for operation in ("dispatch", "combine"):
            if operation not in result:
                continue
            perf = result[operation]
            identity=result.get('comparison') or {}
            base = {"case_id": result["case_id"], **result["config"], "route": result["route"], "operation": operation,
                    **{key:identity.get(key) for key in ('implementation','transport','logical_group','active_aiv','core_protocol','source_revision')}}
            summaries.append(
                {
                    **base,
                    **{
                        key: perf[key]
                        for key in (
                            "summary_statistics_scope",
                            "perf_tail",
                            "kernel_samples",
                            "mean_kernel_us",
                            "p99_kernel_us",
                            "p90_kernel_us",
                            "p50_kernel_us",
                            "min_rank_mean_kernel_us",
                            "median_rank_mean_kernel_us",
                            "max_rank_mean_kernel_us",
                            "slowest_rank",
                        )
                    },
                }
            )
            ranks.extend(
                {**base, **{key: value for key, value in rank.items() if key != "kernel_us"}} for rank in perf["ranks"]
            )
    for filename, rows in (("summary.csv", summaries), ("ranks.csv", ranks)):
        if rows:
            with (directory / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    config_path = probe.parse_known_args(argv)[0].config
    configured = []
    if config_path:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = PROJECT_ROOT / config_file
        config = json.loads(config_file.read_text())
        if not isinstance(config, dict):
            raise ValueError("ST config must be a JSON object")
        matrix = config.pop("matrix", {})
        if not isinstance(matrix, dict) or set(matrix) - set(MATRIX_FIELDS):
            raise ValueError("matrix supports only " + ", ".join(MATRIX_FIELDS))
        hosts = config.pop("hosts", None)
        if hosts is not None:
            if not isinstance(hosts, list) or not hosts or not all(isinstance(host, str) and host for host in hosts):
                raise ValueError("hosts must be a nonempty list of addresses")
            config.setdefault("num_nodes", len(hosts))
            config.setdefault("master_addr", hosts[0])
        if "loops" in config:
            if "num_iters" in config:
                raise ValueError("Use loops or num_iters, not both")
            config["num_iters"] = config.pop("loops")
        if set(config) & set(matrix):
            raise ValueError("Matrix axes must not also appear at the config root")
        config.update(matrix)
        actions = {action.dest: action for action in make_parser()._actions}
        for key, value in config.items():
            if key not in actions or key in {"help", "config", "plan", "node_rank"}:
                raise ValueError(f"Unknown or per-node config field: {key}")
            action = actions[key]
            option = "--" + key.replace("_", "-")
            if isinstance(action, argparse._StoreTrueAction):
                if type(value) is not bool:
                    raise ValueError(f"{key} must be boolean")
                if value:
                    configured.append(option)
            else:
                values = value if isinstance(value, list) else [value]
                if not values or (isinstance(value, list) and key not in MATRIX_FIELDS):
                    raise ValueError(f"Invalid values for {key}")
                configured.extend(
                    [
                        option,
                        *(
                            str(int(item)) if key == "do_cpu_sync" and isinstance(item, bool) else str(item)
                            for item in values
                        ),
                    ]
                )
    args = make_parser().parse_args(configured + argv)
    if args.ll_expert_timer: args.ll_stage_timer=True
    if args.ll_reference_profile and not any(value == '--ll-timer-iters' or value.startswith('--ll-timer-iters=') for value in configured + argv):
        args.ll_timer_iters = 1
    for name in ("output_dir", "output"):
        value = getattr(args, name)
        if value:
            path = Path(value)
            setattr(args, name, str(path if path.is_absolute() else PROJECT_ROOT / path))
    tail_explicit = any(option == "--perf-tail" or option.startswith("--perf-tail=") for option in configured + argv)
    if not tail_explicit:
        args.perf_tail = min(args.perf_tail, args.num_iters)
    resolve_delay_counts(args)
    if args.shmem_port is not None:
        if not 0 < args.shmem_port < 65536:
            raise ValueError("Invalid shmem_port")
        address = args.master_addr or os.environ.get("MASTER_ADDR")
        if not address:
            raise ValueError("shmem_port requires an explicit master address")
        if args.shmem_ip_port is not None:
            raise ValueError("Use shmem_port or shmem_ip_port, not both")
        args.shmem_ip_port = f"tcp://{address}:{args.shmem_port}"
    return args


def launch_config(args, environ):
    """Support DeepEP-style node identities and torchrun process identities."""
    if "LOCAL_RANK" in environ:
        if not all(key in environ for key in ("RANK", "WORLD_SIZE")):
            raise ValueError("torchrun requires RANK, WORLD_SIZE and LOCAL_RANK")
        world, rank, local = (int(environ[key]) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
        if not 0 <= rank < world or local < 0:
            raise ValueError("Invalid torchrun process identity")
        return world, rank, local
    nodes = int(environ.get("WORLD_SIZE", args.num_nodes))
    node = int(environ.get("RANK", args.node_rank))
    if nodes < 1 or args.num_processes < 1 or not 0 <= node < nodes:
        raise ValueError("Invalid node count, node rank or local process count")
    return nodes * args.num_processes, node * args.num_processes, None


def resolve_delay_counts(args):
    for name, default in (("delay_exp_before", EXP_PREFIX_REPEATS), ("delay_exp_after", 0)):
        if getattr(args, name) is None:
            setattr(args, name, default)


def validate_args(args, world):
    resolve_delay_counts(args)
    if any(isinstance(getattr(args, key), list) for key in MATRIX_FIELDS):
        capacities = []
        for case in matrix_cases(args):
            try:
                capacities.append(validate_args(case, world))
            except ValueError as exc:
                values = {key: getattr(case, key) for key in MATRIX_FIELDS}
                raise ValueError(f"Unsupported LL combination {values}: {exc}") from exc
        return max(capacities)
    if args.num_bytes != 8 << 30:
        raise ValueError("LL ST requires --num-bytes 8589934592 (exactly 8 GiB)")
    if world not in (16, 32, 64, 128, 256):
        raise ValueError("LL ST requires EP16/32/64/128/256; EP8 would exercise HT, not LL")
    m = args.num_tokens if args.num_max_tokens_per_rank is None else args.num_max_tokens_per_rank
    if not 0 < args.num_tokens <= m <= 256:
        raise ValueError("LL ST requires 0 < actual B <= M <= 256")
    if not 4 <= args.hidden <= 2147483647:
        raise ValueError("H must be >=4 for ST row identifiers and fit native int; native layout checks apply")
    if not 1 <= args.num_topk <= 16 or not world <= args.num_experts <= 1024 or args.num_experts % world:
        raise ValueError("Require K=1..16, EP<=E<=1024 and E divisible by EP")
    if args.num_sms != 0 or args.num_qps != 0:
        raise ValueError("LL ST requires --num-sms=0 and --num-qps=0; no HT fallback is permitted")
    if not 1 <= args.ll_timer_iters <= 1000:
        raise ValueError("ll-timer-iters must be 1..1000")
    if args.ll_timer_frequency_hz is not None and (not math.isfinite(args.ll_timer_frequency_hz) or args.ll_timer_frequency_hz <= 0):
        raise ValueError("Timer counter frequency must be positive and finite")
    if args.ll_expert_timer: args.ll_stage_timer=True
    if getattr(args, 'll_reference_profile', False):
        if args.ll_stage_timer or args.ll_expert_timer:
            raise ValueError('Reference profiling cannot run with legacy stage/expert tracing')
        if not args.dispatch_only:
            raise ValueError('Reference profiling requires --dispatch-only')
        if args.ll_timer_mode != 'observe':
            raise ValueError('Reference profiling has fixed scalar boundaries; drain is unsupported')
    if diagnostic_requested(args) and args.skip_check:
        raise ValueError("Stage diagnostics require correctness checks")
    if args.clock_align:
        if args.skip_perf_test or diagnostic_requested(args):
            raise ValueError("Clock alignment is only supported by plain OFF performance sampling")
        if not 8 <= args.clock_align_samples <= 256:
            raise ValueError("clock-align-samples must be in [8,256]")
        if not 0 <= args.clock_align_warmup_samples <= 32:
            raise ValueError("clock-align-warmup-samples must be in [0,32]")
        if not 1000 <= args.clock_align_lead_us <= 10_000_000:
            raise ValueError("clock-align-lead-us must be in [1000,10000000]")
        if not 100 <= args.clock_align_interval_us <= 1_000_000:
            raise ValueError("clock-align-interval-us must be in [100,1000000]")
        if not 1 <= args.clock_align_timeout_secs <= 60:
            raise ValueError("clock-align-timeout-secs must be in [1,60]")
    if args.warmup < 0 or args.num_iters < 1 or args.repeat_combine < 2:
        raise ValueError("Require warmup>=0, num-iters>=1 and repeat-combine>=2")
    if not 1 <= args.num_cpu_timeout_secs <= 300:
        raise ValueError("num-cpu-timeout-secs must be in [1,300]")
    if not (2 << 20) <= args.num_bytes <= (32 << 30) or args.num_bytes % (2 << 20):
        raise ValueError("num-bytes must be 2 MiB aligned and between 2 MiB and 32 GiB")
    if args.delay_rows <= 0 or args.delay_cols <= 0 or min(args.delay_exp_before, args.delay_exp_after) < 0:
        raise ValueError("Delay tensor dimensions must be positive and exp counts nonnegative")
    if (args.delay_exp_before, args.delay_exp_after) != (EXP_PREFIX_REPEATS, 0):
        raise ValueError("Reference profiler requires fixed batch delay-exp-before=100 and delay-exp-after=0; remove old per-iteration delay flags")
    if not 0 <= args.perf_tail <= args.num_iters:
        raise ValueError("Require 0 <= perf-tail <= num-iters")
    if args.output and Path(args.output).exists():
        raise ValueError("Output file already exists")
    return m


def make_random_expert_indices(  # 造数入口：返回 shape=[ep*tokens_per_rank, topk] 的全局专家号
    tokens_per_rank: int,  # 每个 rank 的 token 数（即 bs）
    ep: int = 64,  # 目标 rank 数；本函数按"目标 rank"来分配路由份额
    experts: int = 384,  # 全局专家总数；必须满足 experts == ep * physical_stride
    physical_stride: int = 6,  # 每个 rank 拥有的连续专家数（= experts // ep），用于拼全局专家号
    topk: int = 8,  # 每个 token 选几个不同的专家
    hot_multiplier_x2: int = 2,  # 热点倍率的 2 倍表示；2 表示最高 rank 恰为均值（即无热点）
    seed_offset: int = 0,
):
    seed = BALANCED_ROUTE_SEED + seed_offset  # 固定基种子，A/B 两次跑必须喂同一份输入
    total = ep * tokens_per_rank * topk  # 全局路由条数 = rank 数 × 每 rank token × topk
    mean = total // ep  # 每个目标 rank 的平均份额（整除）
    hot_count = mean * hot_multiplier_x2 // 2  # 热点 rank 的份额 = 均值 × (倍率/2)，倍率 2 时正好等于均值

    if hot_count == mean:  # 无热点（或倍率折算后等于均值）时走完全均匀分支
        counts = [mean] * ep  # 每个目标 rank 都拿 mean 条，总和精确等于 total
    else:  # 有热点：hot rank 先拿走 hot_count，其余按随机权重分摊
        rng = random.Random(seed + tokens_per_rank)  # 独立随机流；把 bs 混进种子，不同规模用不同数据但各自可复现
        weights = [rng.uniform(0.7, 1.3) for _ in range(ep - 1)]  # 除 hot rank 外每个 rank 的权重，0.7~1.3 倍
        raw = [(total - hot_count) * w / sum(weights) for w in weights]  # 剩余额度按权重归一化摊成浮点条数
        others = [math.floor(x) for x in raw]  # 先全部向下取整，保证不会超额
        order = sorted(  # 按"被截掉的小数部分"降序，决定谁先补 1 条
            range(ep - 1),  # 只对非 hot 的 rank 排序
            key=lambda i: (raw[i] - others[i], rng.random()),  # 主键是小数余数，同余数时用随机数打破平局
            reverse=True,  # 降序：余数大的排前面
        )
        for i in order[:total - hot_count - sum(others)]:  # 取整丢掉的条数（≥0，为 0 时切片为空）补给余数最大的若干个
            others[i] += 1  # 每条只补 1，保证 total 精确守恒
        counts = [hot_count] + others  # 第 0 个 rank 是热点，其余是摊分结果；sum(counts) == total

    remaining = counts.copy()  # 每个目标 rank 还剩多少条没分配，用来约束抽样
    occurrences = [0] * ep  # 每个目标 rank 已被选中次数，用于轮转其名下的局部专家号
    rng = random.Random(seed + tokens_per_rank * 1000 + hot_multiplier_x2)  # 第二阶段独立随机流；与上面的流刻意错开
    routes = []  # 输出：每个 token 一行，行内是 topk 个全局专家号

    for global_token in range(ep * tokens_per_rank):  # 逐个 token 生成路由（token 数 = ep × 每 rank token）
        tokens_left = ep * tokens_per_rank - global_token  # 含当前 token 在内还剩多少个 token 待生成
        chosen = [r for r, n in enumerate(remaining) if n == tokens_left]  # 剩余额度恰等于剩余 token 数的 rank 必须被选，否则永远填不满
        while len(chosen) < topk:  # 补足到 topk 个互不相同的目标 rank
            candidates = [  # 候选：还有额度、且本次 token 未选过的 rank
                r for r, n in enumerate(remaining)
                if n > 0 and r not in chosen  # 额度 >0 保证可分配，排除重复保证同 token 内专家不重复
            ]
            chosen.append(rng.choices(  # 按剩余额度加权抽一个，额度大的更容易被选（天然均衡收尾）
                candidates, weights=[remaining[r] for r in candidates]
            )[0])
        rng.shuffle(chosen)  # 打乱 topk 内顺序，避免每行都是"大额度 rank 在前"的固定排列

        routes.append([  # 生成本 token 的 topk 个全局专家号
            r * physical_stride + occurrences[r] % (experts // ep)  # 全局号 = rank 块基址 + 该 rank 名下轮转的局部专家号
            for r in chosen  # 对选中的每个目标 rank 逐个算号
        ])
        for r in chosen:  # 更新本 token 用掉的状态
            remaining[r] -= 1  # 扣掉一个额度
            occurrences[r] += 1  # 轮转计数 +1，使同一 rank 的局部专家号依次循环

    assert not any(remaining)  # 自检：所有额度用尽，说明每 rank 实际分配条数精确等于 counts
    return routes  # 返回 shape=[ep*tokens_per_rank, topk] 的嵌套列表


def make_routes(world, b, k, e, seed, route="normal"):
    """Default: user's exact rank quotas, distinct ranks, round-robin experts.

    seed is an offset from 20261014. Offset zero reproduces the supplied
    generator with hot_multiplier_x2=2 and physical_stride=E/EP.
    """
    if world < 1 or b < 1 or e < world or e % world or not 1 <= k <= world:
        raise ValueError('Routes require B>0, E divisible by EP, and 1<=K<=EP')
    if route not in ('normal', 'hotspot'):
        raise ValueError('Unknown routing mode')
    if route == 'normal':
        flat = make_random_expert_indices(b, ep=world, experts=e, physical_stride=e // world,
                                         topk=k, seed_offset=seed, hot_multiplier_x2=2)
        return [flat[source * b:(source + 1) * b] for source in range(world)]
    rng = random.Random(seed)
    result = []
    for _ in range(world):
        rows = []
        for _ in range(b):
            pool = min(e, max(k, 2 * (e // world)))
            chosen = rng.sample(range(pool), k)
            rows.append(chosen)
        result.append(rows)
    return result


def expert_entries(routes, experts, world, rank):
    local = experts // world
    result = [[] for _ in range(local)]
    for source, rows in enumerate(routes):
        for token, selected in enumerate(rows):
            for expert in selected:
                if rank * local <= expert < (rank + 1) * local:
                    result[expert - rank * local].append((source, token))
    return result


def payload(sources, tokens, hidden, dtype):
    import torch

    sources, tokens = torch.as_tensor(sources), torch.as_tensor(tokens)
    cols = torch.arange(hidden)
    values = ((sources[:, None] * 3 + tokens[:, None] * 5 + cols) % 17 - 8).to(dtype)
    # Four small exact integers identify every row, even with FP8 input.
    values[:, 0], values[:, 1] = sources // 16, sources % 16
    values[:, 2], values[:, 3] = tokens // 16, tokens % 16
    return values


def fp8_scales(sources, tokens, hidden, scale_dtype):
    import torch

    groups = (hidden + 127) // 128
    if scale_dtype == "int32":
        return torch.full((len(sources), groups), 0x7F7F7F7F, dtype=torch.int32)
    # Finite, positive non-unit values with distinct low mantissa bits per row/group.
    bits = (
        0x3F000001
        + torch.as_tensor(sources)[:, None] * 65536
        + torch.as_tensor(tokens)[:, None] * 128
        + torch.arange(groups)
    )
    return bits.to(torch.int32).contiguous().view(torch.float32)


def expected_combine(routes, rank, hidden, dtype, fp32_scales=False):
    import torch

    b, k = len(routes[rank]), len(routes[rank][0])
    x = payload([rank] * b, list(range(b)), hidden, dtype).float()
    if fp32_scales:
        x *= fp8_scales([rank] * b, list(range(b)), hidden, "float32").repeat_interleave(128, dim=1)[:, :hidden]
    expected = torch.zeros_like(x)
    for token in range(b):
        for slot in range(k):
            expert = routes[rank][token][slot]
            if expert >= 0:
                expert_value = (x[token] * ((expert % 4 + 1) / 4) + (expert % 3 - 1) / 8).to(dtype).float()
                expected[token] += expert_value * ((slot + 1) / 32)
    return expected.to(dtype)


def prepare_experts(recv, counts, rank, experts, world, expert_dtype):
    import torch

    values = recv[0] if isinstance(recv, tuple) else recv
    output = torch.zeros(values.shape, dtype=expert_dtype, device=values.device)
    start = 0
    for local, count in enumerate(counts):
        expert = rank * (experts // world) + local
        block = values[start : start + count].float()
        if isinstance(recv, tuple) and recv[1].dtype == torch.float32:
            block = block * recv[1][start : start + count].repeat_interleave(128, dim=1)[:, : values.shape[1]]
        output[start : start + count] = (block * ((expert % 4 + 1) / 4) + (expert % 3 - 1) / 8).to(expert_dtype)
        start += count
    return output


def verify_dispatch(recv, handle, entries, args, world, m):
    import torch

    counts = [len(rows) for rows in entries]
    r = sum(counts)
    exact = bool(args.do_cpu_sync) and not args.without_buffer_group
    c = world * m * min(args.num_topk, args.num_experts // world)
    values = recv[0] if isinstance(recv, tuple) else recv
    expected_rows = r if exact else c
    if tuple(values.shape) != (expected_rows, args.hidden) or handle.num_expanded_tokens != expected_rows:
        raise AssertionError("LL R/C shape or handle row count differs from the contract")
    if any(
        value is not None
        for value in (
            handle.expert_recv_counts,
            handle.num_unaligned_recv_tokens_per_expert,
            handle.psum_num_recv_tokens_per_expert,
        )
    ):
        raise AssertionError("LL handle count tensors must be None")
    if int(handle.send_counts[-1].item()) != r:
        raise AssertionError("LL send_counts does not end at exact R")
    if handle.num_recv_tokens != (None if exact else world * m):
        raise AssertionError("LL recv token metadata differs from R/C contract")
    if handle.num_recv_tokens_per_expert_list != (None if exact else []):
        raise AssertionError("Unexpected CPU expert-list field")
    if isinstance(recv, tuple):
        if recv[1].dtype != getattr(torch, args.fp8_scale_dtype) or recv[1].shape != (
            expected_rows,
            (args.hidden + 127) // 128,
        ):
            raise AssertionError("FP8 scale layout mismatch")
    start = 0
    for rows in entries:
        block = values[start : start + len(rows)].float().cpu()
        if rows:
            ids = (block[:, 0].to(torch.int64) * 16 + block[:, 1].to(torch.int64)) * args.num_tokens
            ids += block[:, 2].to(torch.int64) * 16 + block[:, 3].to(torch.int64)
            order = ids.argsort()
            sources, tokens = zip(*rows)
            reference = payload(sources, tokens, args.hidden, torch.float32)
            torch.testing.assert_close(block[order], reference, rtol=0, atol=0)
            if isinstance(recv, tuple):
                actual_scales = recv[1][start : start + len(rows)].cpu()[order].contiguous()
                reference_scales = fp8_scales(sources, tokens, args.hidden, args.fp8_scale_dtype)
                if not torch.equal(actual_scales.view(torch.int32), reference_scales.view(torch.int32)):
                    raise AssertionError("FP8 scale bits or row association were not preserved")
        start += len(rows)


def control_device(args, torch):
    return "npu" if args.group_backend == "hccl" else "cpu"


def sampling_protocol(args):
    if getattr(args, "clock_align", False):
        return CLOCK_ALIGNMENT_PROTOCOL
    return TCPSTORE_SAMPLING_PROTOCOL if args.group_backend == "tcpstore" else SAMPLING_PROTOCOL


def control_world(args):
    if args.group_backend == "tcpstore":
        return args.control_group.size()
    import torch.distributed as dist
    return dist.get_world_size()


def host_gather(args, value, stage):
    if args.group_backend == "tcpstore":
        return trace_call(args, stage, args.control_group.all_gather, value, stage)
    import torch.distributed as dist
    peers = [None] * dist.get_world_size()
    trace_call(args, stage, dist.all_gather_object, peers, value)
    return peers


def host_barrier(args, stage):
    if args is not None and args.group_backend == "tcpstore":
        return trace_call(args, stage, args.control_group.barrier, stage)
    import torch.distributed as dist
    return trace_call(args, stage, dist.barrier)


def initial_alignment(args, operation):
    """Align a batch before warmup, outside kernel timing."""
    if args.group_backend == "tcpstore":
        host_barrier(args, f"{operation}.initial_alignment")
        return None
    import torch
    import torch.distributed as dist
    alignment = torch.ones(1, dtype=torch.float32, device=control_device(args, torch))
    dist.all_reduce(alignment, op=dist.ReduceOp.SUM, async_op=False)
    return alignment


def trace_call(args, label, action, *values, **kwargs):
    """Log Host boundaries without introducing device synchronization."""
    enabled = getattr(args, "trace_ops", False)

    def report(state):
        import torch.distributed as dist

        print(
            "[LL ST TRACE] "
            + json.dumps(
                {
                    "time": datetime.now().isoformat(timespec="milliseconds"),
                    "rank": getattr(args, "trace_rank", None),
                    "case_id": getattr(args, "case_id", None),
                    "phase": getattr(args, "_case_phase", None),
                    "op": label,
                    "state": state,
                }
            ),
            flush=True,
        )

    if enabled:
        report("enter")
    try:
        result = action(*values, **kwargs)
    except Exception:
        if enabled:
            report("error")
        raise
    if enabled:
        report("return")
    return result


def agree_check(action, args, rank):
    """Make assertion failures visible to all peers before another device call."""
    error = None
    try:
        trace_call(args, "check.action", action)
    except Exception as exc:
        error = exc
        print(f"[rank {rank}] check failed: {exc}", flush=True)
    if args.group_backend == "tcpstore":
        errors = host_gather(args, None if error is None else f"{type(error).__name__}: {error}", "check.status")
        failures = [(peer, message) for peer, message in enumerate(errors) if message is not None]
        if failures:
            raise AssertionError(f"LL ST correctness failed: {failures}") from error
    else:
        import torch
        import torch.distributed as dist
        status = torch.tensor([int(error is not None)], dtype=torch.int32, device=control_device(args, torch))
        trace_call(args, "check.all_reduce", dist.all_reduce, status, op=dist.ReduceOp.MAX)
        if trace_call(args, "check.status.item", status.item):
            raise AssertionError("LL ST correctness failed on at least one rank") from error


def completion_boundary(args=None):
    """Test-side all-rank completion required by the existing LL call contract."""
    import torch

    trace_call(args, "boundary.synchronize.before", torch.npu.synchronize)
    host_barrier(args, "boundary.barrier")
    trace_call(args, "boundary.synchronize.after", torch.npu.synchronize)


def record_case_progress(args, rank, m, mode, phase, error=None):
    """Persist case identity outside timed loops so interrupted runs remain diagnosable."""
    previous = getattr(args, "_case_phase", None)
    args._case_phase = phase
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "rank": rank,
        "case_id": args.case_id,
        "route": mode,
        "config": case_config(args, m),
        "phase": phase,
    }
    if error is not None:
        entry.update(failed_phase=previous, error=f"{type(error).__name__}: {error}")
    directory = Path(args.run_directory) / f"case_{args.case_id:05d}" / mode
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"progress_rank_{rank:03d}.jsonl").open("a") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if rank == 0 or error is not None:
        print("[LL ST] " + json.dumps(entry, ensure_ascii=False), flush=True)


def enqueue_exp_prefix(delay):
    """Queue the fixed batch prefix, without a host wait or collective after it."""
    import torch
    for _ in range(EXP_PREFIX_REPEATS):
        torch.exp(delay, out=delay)


def estimate_clock_model(rows, rank):
    """Use TileXR's minimum-effective-RTT offset estimator for this short batch."""
    candidates = []
    for sequence, row in enumerate(rows):
        if len(row) != 8:
            continue
        t0, t1, t2, t3, reported_rtt, reported_offset, packed, _ = map(int, row)
        actual_sequence = packed & 0xffffffff
        status = (packed >> 32) & 0xffffffff
        if (status != 0 or actual_sequence != sequence or min(t0, t1, t2, t3) <= 0 or
                t3 < t0 or t2 < t1 or t3 - t0 < t2 - t1):
            continue
        rtt = (t3 - t0) - (t2 - t1)
        offset_sum = (t1 - t0) + (t2 - t3)
        offset = (abs(offset_sum) // 2) * (-1 if offset_sum < 0 else 1)
        if rtt != reported_rtt or offset != reported_offset:
            continue
        candidates.append((rtt, sequence, offset, t0 + (t3 - t0) // 2))
    minimum = max(4, math.ceil(len(rows) * 0.75))
    if len(candidates) < minimum:
        raise RuntimeError(
            f"Clock calibration rank {rank} has {len(candidates)}/{len(rows)} valid samples; require {minimum}"
        )
    best = min(candidates)
    return dict(
        rank=rank,
        state="ready",
        anchor_cycle=best[3],
        offset_cycles=0 if rank == 0 else best[2],
        rate_correction=0.0,
        min_round_trip_cycles=0 if rank == 0 else best[0],
        error_bound_cycles=0 if rank == 0 else (best[0] + 1) // 2,
        valid_samples=len(candidates),
        rejected_samples=len(rows) - len(candidates),
        sample_sequence=best[1],
    )


def clock_sleep_deadline(model, reference_cycle):
    """Invert the frozen affine model and round toward a non-early deadline."""
    rate = float(model["rate_correction"])
    displacement = math.ceil(
        (-float(model["offset_cycles"]) -
         rate * (int(reference_cycle) - int(model["anchor_cycle"]))) / (1.0 + rate)
    )
    deadline = int(reference_cycle) + displacement
    if not 0 <= deadline < 1 << 64:
        raise ValueError("Clock alignment deadline is outside uint64")
    return deadline


def prepare_clock_alignment(buffer, args, operation):
    import torch
    from deep_ep._native import require_native
    from deep_ep._runtime import runtime_of

    native = require_native("LL clock alignment")
    if getattr(native, "low_latency_clock_alignment_version", None) != 1:
        raise RuntimeError("Rebuild the wheel with LL clock alignment v1")
    runtime = runtime_of(buffer)
    if runtime.low_latency_comm_args is None:
        raise RuntimeError("Clock alignment requires an initialized LL communication argument tensor")
    generation = getattr(args, "_clock_alignment_generation", 0) + 1
    args._clock_alignment_generation = generation
    operation_id = 1 if operation == "dispatch" else 2
    magic = 0xC10C000000000000 | (int(args.case_id + 1) << 24) | (operation_id << 16) | generation
    samples = torch.empty(
        (args.clock_align_samples, 8), dtype=torch.int64, device=f"npu:{runtime.device}"
    )
    with torch.npu.stream(runtime.stream):
        native.low_latency_clock_sync(
            runtime.native,
            runtime.low_latency_comm_args,
            samples,
            args.clock_align_samples,
            args.clock_align_warmup_samples,
            magic,
            args.clock_align_timeout_secs * CLOCK_CYCLES_PER_SECOND,
            runtime.stream.npu_stream,
        )
    runtime.stream.synchronize()
    rows = samples.cpu().tolist()
    model = estimate_clock_model(rows, runtime.rank)
    local = {**model, "last_cycle": int(rows[-1][3])}
    models = host_gather(args, local, f"{operation}.clock.models")
    if [item["rank"] for item in models] != list(range(runtime.world_size)):
        raise RuntimeError("Clock models do not follow EP rank order")
    reference_target = int(models[0]["last_cycle"]) + args.clock_align_lead_us * 1000
    first_deadline = clock_sleep_deadline(model, reference_target)
    releases = torch.empty((args.num_iters,), dtype=torch.int64, device=f"npu:{runtime.device}")
    releases.record_stream(runtime.stream)
    metadata = dict(
        schema="deepep_clock_alignment.v1",
        source="TileXR PR 218 four-timestamp estimator and sleep-at semantics",
        cycles_per_second=CLOCK_CYCLES_PER_SECOND,
        samples=args.clock_align_samples,
        warmup_samples=args.clock_align_warmup_samples,
        lead_us=args.clock_align_lead_us,
        interval_us=args.clock_align_interval_us,
        reference_target=reference_target,
        first_local_deadline=first_deadline,
        max_model_error_bound_ns=max(item["error_bound_cycles"] for item in models),
        models=models,
    )
    return dict(runtime=runtime, native=native, model=model, releases=releases, metadata=metadata)


def enqueue_clock_alignment(state, args, iteration):
    import torch

    target = state["metadata"]["reference_target"] + iteration * args.clock_align_interval_us * 1000
    deadline = clock_sleep_deadline(state["model"], target)
    release = state["releases"][iteration:iteration + 1]
    with torch.npu.stream(state["runtime"].stream):
        state["native"].low_latency_clock_sleep_at(
            state["runtime"].native, deadline, release, state["runtime"].stream.npu_stream
        )


def finish_clock_alignment(state, args, operation, profile_dir):
    releases = [int(value) for value in state["releases"].cpu().tolist()]
    offset = int(state["model"]["offset_cycles"])
    corrected = [value + offset for value in releases]
    peers = host_gather(args, corrected, f"{operation}.clock.releases")
    targets = [
        state["metadata"]["reference_target"] + index * args.clock_align_interval_us * 1000
        for index in range(args.num_iters)
    ]
    skews = [max(row) - min(row) for row in zip(*peers)]
    lateness = [value - target for row, target in zip(zip(*peers), targets) for value in row]
    summary = {
        **state["metadata"],
        "release_skew_p50_ns": statistics.median(skews),
        "release_skew_p99_ns": percentile(skews, 0.99),
        "release_skew_max_ns": max(skews),
        "release_lateness_p50_ns": statistics.median(lateness),
        "release_lateness_p99_ns": percentile(lateness, 0.99),
        "release_lateness_max_ns": max(lateness),
    }
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    (Path(profile_dir) / "clock_alignment.json").write_text(
        json.dumps(dict(summary=summary, local_model=state["model"], local_release_cycles=releases,
                        local_corrected_cycles=corrected, reference_targets=targets), indent=2) + "\n"
    )
    if not hasattr(args, "clock_alignment_metadata"):
        args.clock_alignment_metadata = {}
    args.clock_alignment_metadata[operation] = summary
    return summary


def measure(buffer, invoke, args, operation, profile_dir, progress=None, after_profile=None):
    """Measure exported device kernel tasks; never substitute Host wall time."""
    import torch
    import torch.distributed as dist
    import torch_npu

    kernel = f"{operation}_server_dedup_kernel"
    if progress:
        progress(f"{operation}.warmup")
    completion_boundary(args)
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    clock_aligned = bool(getattr(args, "clock_align", False))
    delay = None if clock_aligned else torch.zeros(
        args.delay_rows, args.delay_cols, dtype=torch.float32, device="npu"
    )
    alignment = initial_alignment(args, operation)
    clock_state = None
    with torch_npu.profiler.profile(
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=torch_npu.profiler.ExportType.Text,
            l2_cache=False, data_simplification=False),
        profile_memory=True, with_stack=False, with_flops=False, with_modules=False,
    ) as profiler:
        for _ in range(args.warmup):
            output = invoke()
        torch.npu.synchronize()
        if clock_aligned:
            clock_state = prepare_clock_alignment(buffer, args, operation)
        else:
            # Submit the legacy padding while the profiler is still in its warmup step.
            # Do not synchronize after it: Host should enqueue the active batch.
            enqueue_exp_prefix(delay)
        profiler.step()
        for iteration in range(args.num_iters):
            if clock_state is not None:
                enqueue_clock_alignment(clock_state, args, iteration)
            output = invoke()
        host_barrier(args, f"{operation}.batch_submitted")
        torch.npu.synchronize()
        profiler.step()
    if after_profile is not None:
        # Append diagnostics only after profiler closes, before parsing/filesystem work.
        after_profile(delay, alignment)
    clock_summary = (
        finish_clock_alignment(clock_state, args, operation, profile_dir)
        if clock_state is not None else None
    )
    del output, delay, alignment, clock_state
    if progress:
        progress(f"{operation}.read_profile")
    samples = []

    def parse_profile():
        samples.extend(kernel_durations(profile_dir, kernel, args.num_iters))
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        (Path(profile_dir) / "samples.json").write_text(json.dumps({"kernel": kernel, "kernel_us": samples}) + "\n")

    rank = args.control_group.rank() if args.group_backend == "tcpstore" else dist.get_rank()
    agree_check(parse_profile, args, rank)
    gathered = host_gather(args, samples, f"{operation}.samples.all_gather")
    result = summarize_kernel(gathered, kernel, args.perf_tail)
    if clock_summary is not None:
        result["clock_alignment"] = clock_summary
    return result


def create_run_directory(parent, name=None):
    parent = Path(parent).absolute()
    if name is not None and (not name or name in ('.', '..') or '/' in name or '\\' in name):
        raise ValueError('--run-name must be a single nonempty directory name')
    parent.mkdir(parents=True, exist_ok=True)
    if name is None:
        return tempfile.mkdtemp(prefix=datetime.now().strftime('%m%d%H%M_'), dir=parent)
    selected = parent / name
    selected.mkdir(exist_ok=False)
    return str(selected)


def new_stage_trace(buffer, operation, args, rank):
    if getattr(args, 'll_reference_profile', False):
        from deep_ep.utils.ll_reference_profile import ReferenceTrace
        return ReferenceTrace(buffer, operation, frequency_hz=args.ll_timer_frequency_hz)
    from deep_ep.utils.ll_stage_timer import StageTrace
    if getattr(args, 'll_expert_timer', False) and operation == 'dispatch':
        import torch.distributed as dist
        return StageTrace(buffer, operation, args.ll_timer_mode, args.ll_timer_frequency_hz,
                          expert_config=dict(rank=rank,world=control_world(args),experts=args.num_experts))
    return StageTrace(buffer, operation, args.ll_timer_mode, args.ll_timer_frequency_hz)


def diagnostic_calls(buffer, invoke, check, args, operation, rank, m, mode,
                     *, prepared=None, prelude_buffers=None):
    """Sample a stabilized tail, retaining outputs/timers until all launches finish."""
    reference = bool(getattr(args, 'll_reference_profile', False))
    if reference:
        from deep_ep.utils.ll_reference_profile import capture
    else:
        from deep_ep.utils.ll_stage_timer import capture
    from deep_ep._native import require_native
    import torch
    import torch.distributed as dist
    world = control_world(args)
    logical_group = compiled_group(require_native("LL timer"),world)
    capture_mode = 'reference' if reference else args.ll_timer_mode
    capture_directory = 'll_reference_profile' if reference else 'll_stage_timer'
    directory = Path(args.run_directory) / f"case_{args.case_id:05d}" / mode / capture_directory / operation / f"rank_{rank:03d}"
    if prepared is None:
        prepared = []
        def allocate():
            for _ in range(args.ll_timer_iters):
                prepared.append(new_stage_trace(buffer, operation, args, rank))
        agree_check(allocate, args, rank)
    after_perf = prelude_buffers is not None
    if not after_perf:
        completion_boundary(args)
        delay = torch.zeros(args.delay_rows, args.delay_cols, dtype=torch.float32, device='npu')
        alignment = initial_alignment(args, operation)
        for _ in range(args.warmup):
            output = invoke()
        torch.npu.synchronize()
        enqueue_exp_prefix(delay)
        for _ in range(args.num_iters):
            output = invoke()
    else:
        delay, alignment = prelude_buffers
        # Profiler export can leave an idle gap. Prefix the diagnostic batch
        # once, sharing the original startup all_reduce, not adding another.
        enqueue_exp_prefix(delay)
    pending = []
    for iteration, trace in enumerate(prepared):
        with capture(buffer, operation, capture_mode, args.ll_timer_frequency_hz,
                     prepared=trace, defer_complete=True):
            output = invoke()
            output[-1].current_stream_wait()
        pending.append((iteration, trace, output))
    # No CPU copy, correctness collective or filesystem operation in the sample loop.
    completion_boundary(args)
    for iteration, trace, output in pending:
        agree_check(trace.complete, args, rank)
        agree_check(lambda: check(output), args, rank)
        metadata = dict(rank=rank, world_size=world, iteration=iteration,
                        comparison=comparison_metadata(args,world,m),
                        run_id=Path(args.run_directory).name, seed=args.seed, route=mode, case_id=args.case_id,
                        routing=routing_metadata(world, args.num_tokens, args.num_experts, args.seed, mode),
                        correctness="passed", config={**case_config(args, m),
                        "logical_group_size": logical_group},
                        protocol=f"{sampling_protocol(args)}; deferred validation/export; excluded from profiler",
                        sampling=dict(after_perf=after_perf, warmup=args.warmup, settle_iters=args.num_iters,
                                      diagnostic_iters=args.ll_timer_iters, delay_rows=args.delay_rows,
                                      delay_cols=args.delay_cols, group_backend=args.group_backend,
                                      sampling_protocol=sampling_protocol(args), exp_prefix_repeats=EXP_PREFIX_REPEATS,
                                      prefix_before='diagnostic_capture' if after_perf else 'settle_batch',
                                      initial_all_reduce=(False if args.group_backend == "tcpstore" else
                                                          'shared_with_perf' if after_perf else 'before_warmup'),
                                      initial_host_alignment='shared_with_perf' if after_perf else 'before_warmup',
                                      per_iteration_exp=0, per_iteration_all_reduce=False,
                                      per_iteration_host_collective=False, per_iteration_rank_alignment="none",
                                      expert_timer=bool(getattr(args,'ll_expert_timer',False)),
                                      reference_profile=reference))
        if reference:
            metadata.update(profile_seed=args.reference_profile_seed, route_sha256=args.reference_route_sha256)
        agree_check(lambda: trace.save(directory / f"sample_{iteration:04d}.json", metadata), args, rank)


def sample_operation(buffer, invoke, check, args, operation, rank, m, mode, progress):
    """Keep the diagnostic tail identical with and without external profiling."""
    prepared = None
    if diagnostic_requested(args):
        prepared = []
        def allocate():
            for _ in range(args.ll_timer_iters):
                prepared.append(new_stage_trace(buffer, operation, args, rank))
        agree_check(allocate, args, rank)
    def diagnostic(delay=None, alignment=None):
        diagnostic_calls(buffer, invoke, check, args, operation, rank, m, mode, prepared=prepared,
                         prelude_buffers=None if delay is None else (delay, alignment))
    if args.skip_perf_test:
        if diagnostic_requested(args):
            progress(f'{operation}.diagnostic')
            diagnostic()
        return None
    profile = Path(args.run_directory) / f'case_{args.case_id:05d}' / mode / operation / f'rank_{rank:03d}'
    return measure(buffer, invoke, args, operation, profile, progress,
                   after_profile=diagnostic if diagnostic_requested(args) else None)


def run_case(buffer, args, rank, world, m, mode):
    import torch

    def progress(phase):
        record_case_progress(args, rank, m, mode, phase)

    progress("prepare")
    routes = make_routes(world, args.num_tokens, args.num_topk, args.num_experts, args.seed, mode)
    entries = expert_entries(routes, args.num_experts, world, rank)
    counts = [len(rows) for rows in entries]
    expert_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    x = payload([rank] * args.num_tokens, range(args.num_tokens), args.hidden, expert_dtype)
    if args.dtype == "fp8":
        x = (
            x.to(torch.float8_e4m3fn).npu(),
            fp8_scales([rank] * args.num_tokens, range(args.num_tokens), args.hidden, args.fp8_scale_dtype).npu(),
        )
    else:
        x = x.npu()
    indices = torch.tensor(routes[rank], dtype=torch.int64, device="npu")
    weights = torch.arange(1, args.num_topk + 1, dtype=torch.float32, device="npu") / 32
    weights = weights.repeat(args.num_tokens, 1).contiguous()

    def dispatch():
        return buffer.dispatch(
            x,
            topk_idx=indices,
            topk_weights=weights,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=m,
            do_expand=True,
            do_cpu_sync=bool(args.do_cpu_sync),
            do_handle_copy=bool(args.do_handle_copy),
            num_compute_units=args.num_sms,
            num_qps=args.num_qps,
        )

    import hashlib

    mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if len(line.split()) >= 6}
    classes = [
        [p for p in mapped if "/deep_ep/_C." in p and p.endswith(".so")],
        [p for p in mapped if p.endswith("/libdeepep_kernels.so")],
        [p for p in mapped if p.endswith("/libshmem.so")],
        [p for p in mapped if p.endswith("/aclshmem_bootstrap_config_store.so")],
    ]
    if not all(len(paths) == 1 for paths in classes):
        raise AssertionError(f"Cannot audit native/SDK mappings: {classes}")
    libraries = {paths[0]: hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest() for paths in classes}
    live = {"pid": os.getpid(), "session": os.getsid(0), "rank": rank, "libraries": libraries}
    (Path(args.run_directory) / f"loaded_rank_{rank:03d}.json").write_text(json.dumps(live))
    progress("dispatch.check")
    completion_boundary(args)
    recv, recv_idx, recv_weights, handle, event = trace_call(args, "dispatch.invoke", dispatch)
    trace_call(args, "dispatch.event.wait", event.current_stream_wait)
    pending_checks = [(recv, recv_idx, recv_weights, handle, entries)]
    if not args.skip_check:
        if args.correctness_repeats < 1:
            raise ValueError("correctness-repeats must be positive")
        for repeat in range(1, args.correctness_repeats):
            # Exercise window reuse continuously; validate retained outputs after the batch.
            routes = make_routes(world, args.num_tokens, args.num_topk, args.num_experts, args.seed + repeat, mode)
            entries = expert_entries(routes, args.num_experts, world, rank)
            counts = [len(rows) for rows in entries]
            indices = torch.tensor(routes[rank], dtype=torch.int64, device="npu")
            recv, recv_idx, recv_weights, handle, event = dispatch()
            event.current_stream_wait()
            pending_checks.append((recv, recv_idx, recv_weights, handle, entries))
    # Restore the declared route after changed-seed correctness probes, for both
    # reference and full dispatch/combine performance sampling.
    if getattr(args, 'll_reference_profile', False) or (not args.skip_check and args.correctness_repeats > 1):
        # Correctness probes varied the seed above; capture the declared route.
        routes = make_routes(world, args.num_tokens, args.num_topk, args.num_experts, args.seed, mode)
        entries = expert_entries(routes, args.num_experts, world, rank)
        counts = [len(rows) for rows in entries]
        indices = torch.tensor(routes[rank], dtype=torch.int64, device='npu')
        recv, recv_idx, recv_weights, handle, event = dispatch()
        event.current_stream_wait()
        pending_checks.append((recv, recv_idx, recv_weights, handle, entries))
    trace_call(args, "dispatch.batch.synchronize", torch.npu.synchronize)

    def check_dispatch_batch():
        for payload, public_idx, public_weights, result_handle, expected_entries in pending_checks:
            # Never allow --skip-check to hide an HT fallback.
            if not result_handle.is_low_latency or public_idx is not None or public_weights is not None:
                raise AssertionError("ST did not execute the LL backend or returned unexpected public weights")
            if result_handle.source_weights is None or result_handle.destination_index is None:
                raise AssertionError("LL handle lost its native tables/weights")
            if not args.skip_check:
                verify_dispatch(payload, result_handle, expected_entries, args, world, m)

    agree_check(check_dispatch_batch, args, rank)
    del pending_checks
    args.reference_profile_seed = args.seed
    args.sample_route_sha256 = hashlib.sha256(
        json.dumps(routes, separators=(',', ':')).encode('utf-8')).hexdigest()
    args.reference_route_sha256 = args.sample_route_sha256
    if args.dispatch_only:
        result = {
            "case_id": args.case_id,
            "route": mode,
            "correctness": "skipped" if args.skip_check else "passed",
            "backend": "ll",
            "config": case_config(args, m),
            "tested_operations": ["dispatch"],
        }
        perf = sample_operation(buffer, dispatch,
            lambda output: verify_dispatch(output[0], output[3], entries, args, world, m),
            args, 'dispatch', rank, m, mode, progress)
        if perf is not None:
            result['dispatch'] = perf
        if rank == 0:
            print(json.dumps(result, ensure_ascii=False), flush=True)
        completion_boundary(args)
        progress("complete")
        return result
    combine = buffer.combine
    expert_output = prepare_experts(recv, counts, rank, args.num_experts, world, expert_dtype)
    expected = (
        expected_combine(
            routes,
            rank,
            args.hidden,
            expert_dtype,
            fp32_scales=args.dtype == "fp8" and args.fp8_scale_dtype == "float32",
        )
        if not args.skip_check
        else None
    )
    for repeat in range(args.repeat_combine):
        progress(f"combine.check.{repeat}")
        completion_boundary(args)
        out, returned_weights, completed = trace_call(
            args, "combine.invoke", combine, expert_output, handle, topk_weights=None
        )
        trace_call(args, "combine.event.wait", completed.current_stream_wait)
        trace_call(args, "device.synchronize", torch.npu.synchronize)

        def check_combine():
            if returned_weights is not None or tuple(out.shape) != (args.num_tokens, args.hidden):
                raise AssertionError("LL combine public outputs differ from the contract")
            if expected is not None:
                torch.testing.assert_close(
                    out.cpu(),
                    expected,
                    rtol=0.01 if args.dtype != "fp16" else 0.003,
                    atol=0.03125 if args.dtype != "fp16" else 0.004,
                )

        agree_check(check_combine, args, rank)
    def check_timed_combine(output):
        if output[1] is not None or tuple(output[0].shape) != (args.num_tokens, args.hidden):
            raise AssertionError("Instrumented combine output contract differs")
        torch.testing.assert_close(output[0].cpu(), expected,
                                   rtol=0.003 if args.dtype == "fp16" else 0.01,
                                   atol=0.004 if args.dtype == "fp16" else 0.03125)
    result = {
        "case_id": args.case_id,
        "route": mode,
        "correctness": "skipped" if args.skip_check else "passed",
        "backend": "ll",
        "config": case_config(args, m),
    }
    if not args.skip_perf_test or diagnostic_requested(args):
        if rank == 0:
            print(f"[LL ST] timing dispatch/combine: warmup={args.warmup}, iterations={args.num_iters}", flush=True)
        perf = sample_operation(buffer, dispatch,
            lambda output: verify_dispatch(output[0], output[3], entries, args, world, m),
            args, 'dispatch', rank, m, mode, progress)
        if perf is not None:
            result['dispatch'] = perf
        # Prepare fresh matching metadata and expert output outside the combine capture.
        progress("dispatch.refresh")
        completion_boundary(args)
        recv, _, _, handle, event = trace_call(args, "dispatch.invoke", dispatch)
        trace_call(args, "dispatch.event.wait", event.current_stream_wait)
        trace_call(args, "device.synchronize", torch.npu.synchronize)
        expert_output = prepare_experts(recv, counts, rank, args.num_experts, world, expert_dtype)
        perf = sample_operation(buffer,
            lambda: combine(expert_output, handle, topk_weights=None),
            check_timed_combine, args, 'combine', rank, m, mode, progress)
        if perf is not None:
            result['combine'] = perf
    if rank == 0:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    completion_boundary(args)
    progress("complete")
    return result


def worker(local_rank, args):
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    world, identity, external_local = launch_config(args, os.environ)
    rank = identity if external_local is not None else identity + local_rank
    args.trace_rank = rank
    m = validate_args(args, world)
    cases = list(matrix_cases(args))
    torch.npu.set_device(local_rank)
    torch.set_num_threads(1)
    helpers = runpy.run_path(str(Path(__file__).with_name("test_dispatch.py")))
    deep_ep = helpers["_import_installed_deep_ep"]()
    if rank == 0:
        print(f"[LL ST] Python: {sys.executable}; installed package: {deep_ep.__file__}", flush=True)
    from deep_ep._native import require_native

    native = require_native("LL ST")
    if not callable(getattr(native, "dispatch_low_latency", None)):
        raise RuntimeError("Installed wheel has no LL native entry; rebuild and install the migrated package")
    validate_native_cases(native, cases, world)
    control = None
    if args.group_backend == "tcpstore":
        from deep_ep._store import StoreGroup
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world))
        control = trace_call(args, "host_store.init", StoreGroup.from_env,
                             timeout=args.num_cpu_timeout_secs,
                             standalone=external_local is None, namespace="dispatch")
        args.control_group = control
        for case in cases:
            case.control_group = control
            case.trace_rank = rank
    else:
        trace_call(args, "process_group.init", dist.init_process_group,
            args.group_backend,
            init_method="env://",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=args.num_cpu_timeout_secs),
        )
    buffer = None
    failed = False
    try:
        # Agree before creating a SHMEM Session or launching an instrumented kernel.
        core_config = native.low_latency_core_config()
        implementation=getattr(native,'low_latency_implementation','unknown')
        backend=args.backend or ('rdma' if implementation=='ndr-hns1825' else 'mte')
        # One CLI choice drives Python and native initialization; stale shell state
        # does not change the selected branch's default in this experiment entry.
        os.environ['DEEPEP_LL_BACKEND']=backend
        if args.trace_ops:
            os.environ['DEEPEP_LL_INIT_TRACE'] = '1'
        capability = {"active_aiv": core_config["active_aiv"],
                      "transport": backend,
                      "rdma_compiled": bool(getattr(native, "low_latency_rdma_enabled", False)),
                      "roce_qps_per_peer": getattr(native, "low_latency_roce_qps_per_peer", None),
                      "logical_group": compiled_group(native,world),
                      "implementation": implementation,
                      "combine_host_protocol": getattr(deep_ep.ElasticBuffer, "ll_combine_host_protocol", 0),
                      "core_protocol": native.low_latency_core_protocol,
                      "clock_alignment_version": getattr(native, "low_latency_clock_alignment_version", None),
                      "stage_timer": bool(getattr(native, "low_latency_stage_timer_enabled", False)),
                      "stage_timer_schema": native.low_latency_stage_timer_layout()["schema"],
                      "expert_timer_requested": bool(args.ll_expert_timer)}
        if getattr(args, 'll_reference_profile', False):
            info = json.loads(Path(deep_ep.__file__).with_name('_build_info.json').read_text(encoding='utf-8'))
            if info.get('ll_reference_profile') is not True or info.get('ll_stage_timer') is not False:
                raise RuntimeError('Installed reference build metadata does not match the requested capture')
            capability.update(reference_profile=getattr(native, 'low_latency_reference_profile_enabled', None),
                              reference_profile_schema=native.low_latency_reference_profile_layout()['schema'],
                              installed_source_sha256=info.get('source_sha256'))
        capabilities = host_gather(args, capability, "capability.all_gather")
        if any(item != capabilities[0] for item in capabilities):
            raise RuntimeError("LL AIV/group/protocol/timer build differs across ranks")
        if capability['implementation'] not in ('ndr-hns1825','configurable-groups'):
            raise RuntimeError('Rebuild the wheel with the shared experiment implementation identity')
        requested_impl=os.environ.get('DEEPEP_LL_EXPECT_IMPLEMENTATION')
        if requested_impl and capability['implementation']!=requested_impl:
            raise RuntimeError('Installed wheel belongs to a different kernel branch')
        if capability['transport']=='rdma' and not capability['rdma_compiled']:
            raise RuntimeError('Requested RDMA is not compiled in this wheel; no MTE fallback')
        if args.clock_align and capability["clock_alignment_version"] != 1:
            raise RuntimeError("Clock-aligned sampling requires LL clock alignment v1 on every rank")
        validate_stage_timer_capability(args.ll_stage_timer, capability)
        validate_reference_profile_capability(getattr(args, 'll_reference_profile', False), capability)
        if args.ll_expert_timer and capability['core_protocol'] not in (5, 6, 7, 8, 9, 10):
            raise RuntimeError('Expert diagnostics require ABI 5 on every rank')
        args.ll_stage_timer_capabilities = capabilities
        run_directory = [None]

        def reserve_directory():
            if rank == 0:
                run_directory[0] = create_run_directory(args.output_dir, args.run_name)

        agree_check(reserve_directory, args, rank)
        directories = host_gather(args, run_directory[0], "run_directory.all_gather")
        args.run_directory = directories[0]
        modes = (args.route,)

        def save_plan():
            if rank == 0:
                manifest = {
                    "world_size": world,
                    "case_count": len(cases),
                    "routes": modes,
                    "cases": [
                        {"case_id": i, "config": case_config(case, validate_args(case, world))}
                        for i, case in enumerate(cases)
                    ],
                }
                (Path(args.run_directory) / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"[LL ST] run directory: {args.run_directory}", flush=True)

        agree_check(save_plan, args, rank)
        identity_args = (
            {"group": control} if control is not None else
            {"rank": rank, "world_size": world} if args.without_buffer_group else {"group": dist.group.WORLD}
        )
        if control is not None and args.without_buffer_group:
            raise ValueError("TCPStore NDR lifecycle requires passing StoreGroup to Buffer")

        def create_buffer(case):
            return trace_call(
                case,
                "buffer.create",
                deep_ep.ElasticBuffer,
                **identity_args,
                num_bytes=args.num_bytes,
                num_max_tokens_per_rank=m,
                hidden=0,
                num_topk=0,
                explicitly_destroy=True,
                num_cpu_timeout_secs=args.num_cpu_timeout_secs,
                use_1g_huge_pages=(capability["transport"] != "rdma" if args.use_1g_huge_pages is None else bool(args.use_1g_huge_pages)),
            )

        if not args.fresh_buffer_per_case:
            buffer = create_buffer(args)
        with torch.no_grad():
            results = []
            for case_id, case in enumerate(cases):
                case_m = validate_args(case, world)
                case.run_directory = args.run_directory
                case.case_id = case_id
                case.runtime_identity = capability
                if args.fresh_buffer_per_case:
                    buffer = create_buffer(case)
                for mode in modes:
                    try:
                        result = run_case(buffer, case, rank, world, case_m, mode)
                    except Exception as exc:
                        try:
                            record_case_progress(case, rank, case_m, mode, "failed", exc)
                        except Exception as log_error:
                            print(
                                f"[rank {rank}] case={case_id} route={mode}: cannot save failure: {log_error}",
                                flush=True,
                            )
                        raise
                    result["case_id"] = case_id
                    result['comparison'] = comparison_metadata(case,world,case_m)
                    results.append(result)

                    def save_case():
                        if rank == 0:
                            directory = Path(args.run_directory) / f"case_{case_id:05d}" / mode
                            directory.mkdir(parents=True, exist_ok=True)
                            (directory / "result.json").write_text(
                                json.dumps(case_result_summary(result), indent=2) + "\n"
                            )

                    agree_check(save_case, case, rank)
                if args.fresh_buffer_per_case:
                    trace_call(case, "buffer.destroy", buffer.destroy)
                    buffer = None
        if rank == 0:
            report = {
                "status": "completed",
                "world_size": world,
                "case_count": len(cases),
                "result_count": len(results),
                "config": {key: value for key, value in vars(args).items() if key != "control_group"},
                "package": deep_ep.__file__,
                "measurement": "torch_npu profiler kernel tasks; per-rank statistics and slowest rank mean",
                "alignment": (
                    f"{sampling_protocol(args)}; one initial Host batch alignment; "
                    + ("fresh four-timestamp calibration and one device sleep-at before every measured call"
                       if args.clock_align else "fixed Exp100 before active profiler.step")
                    + "; no per-iteration Host collective; Dispatch correctness validated after all repetitions"
                ),
                "host_coordination": args.group_backend,
                "results": results,
            }
            write_reports(args.run_directory, report)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, indent=2)
                    stream.write("\n")
            print(
                f"[LL ST] saved {len(cases)} combinations / {len(results)} results to {args.run_directory}", flush=True
            )
    except Exception as error:
        import traceback

        failed = True
        if control is not None:
            control.abort(error)
        traceback.print_exc()
        raise
    finally:
        # Runtime/device failures are not recovered or retried on HT.
        if not failed:
            try:
                if buffer is not None:
                    buffer.destroy()
                if control is not None:
                    control.close()
            except Exception as error:
                if control is not None:
                    control.abort(error)
                raise
        if control is None:
            dist.destroy_process_group()


def configure_endpoints(args, environ):
    address = args.master_addr or environ.get("MASTER_ADDR")
    port = args.master_port if args.master_port is not None else environ.get("MASTER_PORT")
    endpoint = args.shmem_ip_port or environ.get("DEEPEP_SHMEM_ENDPOINT")
    if not address or port is None or not endpoint:
        raise ValueError(
            "Explicit MASTER_ADDR, MASTER_PORT and SHMEM endpoint are required via arguments/config or environment"
        )
    try:
        port = int(port)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid master_port") from exc
    if not 0 < port < 65536:
        raise ValueError("Invalid master_port")
    environ.update(MASTER_ADDR=address, MASTER_PORT=str(port), DEEPEP_SHMEM_ENDPOINT=endpoint)


def main(argv=None):
    args = parse_args(argv)
    world, _, local = launch_config(args, os.environ)
    validate_args(args, world)
    if args.plan:
        cases = [
            {"case_id": i, **case_config(case, validate_args(case, world))} for i, case in enumerate(matrix_cases(args))
        ]
        clock_aligned = bool(args.clock_align)
        print(json.dumps({"world_size": world, "case_count": len(cases), "cases": cases,
                          "sampling": dict(protocol=sampling_protocol(args),
                                           exp_prefix_repeats=0 if clock_aligned else EXP_PREFIX_REPEATS,
                                           host_coordination=args.group_backend,
                                           per_iteration_exp=0, per_iteration_all_reduce=False,
                                           per_iteration_host_collective=False,
                                           per_iteration_rank_alignment=(
                                               CLOCK_ALIGNMENT_PROTOCOL if clock_aligned else "none"))}, indent=2))
        return
    configure_endpoints(args, os.environ)
    if local is not None:
        worker(local, args)
    else:
        import torch.multiprocessing as mp

        mp.spawn(worker, args=(args,), nprocs=args.num_processes, join=True)


if __name__ == "__main__":
    main()
