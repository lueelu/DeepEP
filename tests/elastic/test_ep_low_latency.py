# Copyright (c) 2026, Lu Lu
# Modified by ryan_li 2026

"""Installed-wheel LL functional/performance ST, modeled on DeepEP test_ep.py.

Run explicitly; ordinary pytest never initializes NPU devices from this module.
Use --help for launch, configuration, and profiling options.
"""

import argparse
import csv
import hashlib
import importlib.util
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
import time
from datetime import datetime, timedelta

__test__ = False


class _LazyClockModel:
    """Load the shared model on first use so the script starts without DeepEP.

    ``--plan`` and ``--help`` run from a bare checkout where the package is not
    importable yet. If the installed wheel predates the model module (an older
    AOT build), the repository copy is loaded instead, so the AOT and JIT runs
    still share exactly one implementation of the math.
    """

    _module = None

    def __getattr__(self, name):
        module = type(self)._module
        if module is None:
            try:
                from deep_ep.utils import ll_clock_model as module
            except ImportError:
                path = PROJECT_ROOT / "deep_ep" / "utils" / "ll_clock_model.py"
                spec = importlib.util.spec_from_file_location("deep_ep_ll_clock_model", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            type(self)._module = module
        return getattr(module, name)


ll_clock_model = _LazyClockModel()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


EXP_PREFIX_REPEATS = 100
SAMPLING_PROTOCOL = 'initial-reduce-exp100-batch-v3'
TCPSTORE_SAMPLING_PROTOCOL = 'initial-tcpstore-exp100-batch-v2'
CLOCK_ALIGNMENT_PROTOCOL = 'tilexr-sleep-at-per-iteration-v1'
CLOCK_ALIGNMENT_GRAPH_PROTOCOL = 'tilexr-graph-wait-table-v1'
CLOCK_CYCLES_PER_SECOND = 1_000_000_000
BALANCED_ROUTE_SEED = 20261014


def routing_metadata(world, b, e, seed, route):
    balanced = route == 'normal'
    return dict(generator='balanced-rank-distinct-roundrobin-v1' if balanced else 'hotspot-expert-sample-v1',
                seed_offset=seed, base_seed=BALANCED_ROUTE_SEED if balanced else None,
                rng_seed=BALANCED_ROUTE_SEED + seed + b * 1000 + 2 if balanced else seed,
                physical_stride=e // world, hot_multiplier_x2=2 if balanced else None)


def diagnostic_requested(args):
    return bool(getattr(args, 'll_reference_profile', False))


def validate_reference_profile_capability(requested, capability):
    if not requested:
        return
    if (capability.get('reference_profile') is not True or
            type(capability.get('reference_profile_schema')) is not int or capability['reference_profile_schema'] != 4):
        raise RuntimeError('All ranks need the matching reference-profile schema 4 wheel')
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
    graph_aligned = clock_aligned and getattr(args, "clock_align_mode", "stream") == "graph"
    alignment_protocol = CLOCK_ALIGNMENT_GRAPH_PROTOCOL if graph_aligned else CLOCK_ALIGNMENT_PROTOCOL
    return dict(**runtime, comparison_schema=1, sampling_protocol=sampling_protocol(args),
                host_coordination=args.group_backend,
                exp_prefix_repeats=(int(getattr(args, "clock_align_graph_exp_prefix", 0))
                                    if graph_aligned
                                    else 0 if clock_aligned else EXP_PREFIX_REPEATS),
                per_iteration_exp=0, per_iteration_all_reduce=False,
                per_iteration_host_collective=False,
                per_iteration_rank_alignment=alignment_protocol if clock_aligned else "none",
                per_iteration_device_alignment=alignment_protocol if clock_aligned else "none",
                clock_alignment=getattr(args, "clock_alignment_metadata", None),
                perf_enabled=not args.skip_perf_test, world_size=world,
                local_world_size=int(os.environ.get('LOCAL_WORLD_SIZE',args.num_processes)),
                case=case_config(args,m), route=args.route, seed=args.seed,
                routing=routing_metadata(world, args.num_tokens, args.num_experts, args.seed, args.route),
                route_sha256=getattr(args, 'sample_route_sha256', None),
                warmup=args.warmup, num_iters=args.num_iters, perf_tail=args.perf_tail,
                delay_rows=args.delay_rows, delay_cols=args.delay_cols, group_backend=args.group_backend,
                timer_mode='reference' if diagnostic_requested(args) else None,
                timer_iters=args.ll_timer_iters if diagnostic_requested(args) else 0,
                frequency_hz=args.ll_timer_frequency_hz,
                reference_capture=bool(getattr(args, 'll_reference_profile', False)),
                do_handle_copy=args.do_handle_copy,
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
        dtype = {"bf16": "bfloat16", "fp8": "float8_e4m3fn"}[case.dtype]
        shape = (world, case.num_tokens, case.hidden, case.num_topk, case.num_experts)
        try:
            if native.low_latency_layout(*shape, dtype) is None:
                raise ValueError("dispatch layout rejected")
            native.low_latency_combine_tiling(
                *shape, "bfloat16", validate_args(case, world)
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
    parser.add_argument("--dtype", nargs="+", action=MatrixValues, choices=("bf16", "fp8"), default="bf16")
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
    parser.add_argument("--clock-align-mode", choices=("stream", "graph"), default="stream",
                        help="stream: host-enqueue one sleep_at before each iteration; graph: capture one "
                             "wait+dispatch pair per iteration and refresh the deadline table per replay")
    parser.add_argument("--clock-align-replays", type=int, default=3,
                        help="Graph-mode replays with a refreshed deadline table; acceptance reads the last")
    parser.add_argument("--clock-align-calibration-windows", type=int, default=3,
                        help="Spaced four-timestamp windows fitted into one affine rate; 3+ holds the last out")
    parser.add_argument("--clock-align-calibration-gap-secs", type=float, default=5.0,
                        help="Host sleep between calibration windows; the slope needs a span that can resolve drift")
    parser.add_argument("--clock-align-model-max-age-secs", type=float, default=60.0,
                        help="Refit instead of re-anchoring once the fitted rate is older than this")
    parser.add_argument("--clock-align-max-pair-error-ns", type=float, default=None,
                        help="Fail before submitting the graph when the two-rank error budget exceeds this; "
                             "unset reports the budget without claiming a precision")
    parser.add_argument("--clock-align-graph-exp-prefix", type=int, default=0,
                        help="Graph mode: run the fixed Exp prefix after calibration and before the fresh "
                             "anchor; the benchmark protocol uses 100, the native acceptance run uses 0")
    parser.add_argument("--clock-align-samples", type=int, default=64)
    parser.add_argument("--clock-align-warmup-samples", type=int, default=8)
    parser.add_argument("--clock-align-lead-us", type=int, default=200000)
    parser.add_argument("--clock-align-interval-us", type=int, default=1000)
    parser.add_argument("--clock-align-timeout-secs", type=int, default=10)
    parser.add_argument("--ll-reference-profile", action="store_true", help="Separate dispatch-only reference cycle capture; requires a reference-profile wheel")
    parser.add_argument("--ll-timer-iters", type=int, default=5)
    parser.add_argument("--ll-timer-mode", choices=("observe", "drain"), default="observe")
    parser.add_argument("--ll-timer-frequency-hz", type=float, default=1_000_000_000, help="GetSystemCycle frequency; default 1GHz (ticks/1000 us)")
    parser.add_argument("--dispatch-only", action="store_true")
    parser.add_argument("--repeat-combine", type=int, default=2)
    parser.add_argument("--correctness-repeats", type=int, default=3, help="Fresh routes on the same Buffer; outside profiling")
    parser.add_argument("--use-1g-huge-pages", type=int, choices=(0, 1), default=0, help="LL defaults to 0; explicit 1 is MTE-only")
    parser.add_argument("--output", help="New rank-0 JSON result file (will not overwrite)")
    return parser


DISPATCH_LOGICAL_NAME = "dispatch_server_dedup_kernel"
# AOT and JIT expose different task names for the same logical Dispatch. Both
# are admitted here, once, so the ordinary and graph extraction paths agree.
DISPATCH_PROFILER_ENTRIES = ("dispatch_server_dedup_kernel", "dispatch_jit")
WAIT_PROFILER_ENTRY = "clock_wait_table"
PROFILER_LOGICAL_NAMES = {name: DISPATCH_LOGICAL_NAME for name in DISPATCH_PROFILER_ENTRIES}


def profiler_kernel_records(root, entries, logical=None):
    """Read matching device tasks from torch_npu text exports, ordered by start.

    Each row keeps the real task name, its source CSV, start and duration, and
    the logical name used for statistics. An alias never loses the original.
    """
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, entries)) + r")\b")
    aliases = PROFILER_LOGICAL_NAMES if logical is None else logical
    rows = []
    for path in sorted(Path(root).rglob("kernel_details.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                normalized = {key.replace(" ", "").lower(): value for key, value in row.items() if key}
                name = normalized.get("name", normalized.get("opname", ""))
                match = pattern.search(name)
                if not match:
                    continue
                value = float(normalized.get("duration(us)", normalized.get("taskduration(us)", "nan")))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid kernel duration in {path}: {value}")
                start = normalized.get("starttime(us)")
                entry = match.group(0)
                rows.append(dict(entry=entry, logical=aliases.get(entry, entry), name=name.strip(),
                                 duration=value, start=float(start) if start else None,
                                 source=str(path)))
    if rows and all(row["start"] is not None for row in rows):
        rows.sort(key=lambda row: row["start"])
    return rows


def kernel_durations(root, kernel, expected):
    """Read only the requested device kernel from torch_npu text exports."""
    dispatch = kernel == DISPATCH_LOGICAL_NAME
    entries = DISPATCH_PROFILER_ENTRIES if dispatch else (kernel,)
    rows = profiler_kernel_records(root, entries)
    if dispatch:
        rows = [row for row in rows if row["logical"] == DISPATCH_LOGICAL_NAME]
        actual_entries = {row["entry"] for row in rows}
        if len(actual_entries) > 1:
            raise ValueError(f"Mixed dispatch profiler entries in {root}: {sorted(actual_entries)}")
        actual_task_names = {row["name"] for row in rows}
        if len(actual_task_names) > 1:
            raise ValueError(f"Mixed dispatch profiler task names in {root}: {sorted(actual_task_names)}")
    if len(rows) != expected:
        raise ValueError(f"{kernel}: expected {expected} profiler records, found {len(rows)} in {root}")
    return [row["duration"] for row in rows]


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
    if getattr(args, 'll_reference_profile', False):
        if not args.dispatch_only:
            raise ValueError('Reference profiling requires --dispatch-only')
        if args.ll_timer_mode != 'observe':
            raise ValueError('Reference profiling has fixed scalar boundaries; drain is unsupported')
    if diagnostic_requested(args) and args.skip_check:
        raise ValueError("Stage diagnostics require correctness checks")
    if args.clock_align:
        if getattr(args, "clock_align_mode", "stream") == "graph":
            if not args.dispatch_only:
                raise ValueError("Graph clock alignment requires --dispatch-only")
            if args.do_cpu_sync != 0:
                raise ValueError("Graph clock alignment requires --do-cpu-sync 0; host count reads cannot be captured")
            if not 1 <= args.clock_align_replays <= 16:
                raise ValueError("clock-align-replays must be in [1,16]")
            if diagnostic_requested(args):
                raise ValueError("Graph clock alignment excludes reference profiling")
        elif args.skip_perf_test or diagnostic_requested(args):
            raise ValueError("Clock alignment is only supported by plain OFF performance sampling")
        if not 8 <= args.clock_align_samples <= 256:
            raise ValueError("clock-align-samples must be in [8,256]")
        if not 1 <= args.clock_align_calibration_windows <= 16:
            raise ValueError("clock-align-calibration-windows must be in [1,16]")
        if args.clock_align_calibration_windows < 3:
            raise ValueError(
                "A formal clock run needs at least 3 calibration windows: two train the rate and one "
                "is held out for the residual check")
        if not 0.0 <= args.clock_align_calibration_gap_secs <= 120.0:
            raise ValueError("clock-align-calibration-gap-secs must be in [0,120]")
        if not 0.0 < args.clock_align_model_max_age_secs <= 3600.0:
            raise ValueError("clock-align-model-max-age-secs must be in (0,3600]")
        if args.clock_align_max_pair_error_ns is not None:
            limit = args.clock_align_max_pair_error_ns
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError("clock-align-max-pair-error-ns must be positive and finite")
        if args.clock_align_graph_exp_prefix not in (0, EXP_PREFIX_REPEATS):
            raise ValueError(
                f"clock-align-graph-exp-prefix must be 0 or the fixed {EXP_PREFIX_REPEATS} prefix")
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
        if getattr(args, "clock_align_mode", "stream") == "graph":
            return CLOCK_ALIGNMENT_GRAPH_PROTOCOL
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


def clock_alignment_state(buffer, native=None):
    """Resolve the clock entries once; calibration and replay share them."""
    from deep_ep._native import require_native
    from deep_ep._runtime import runtime_of

    native = native or require_native("LL clock alignment")
    if getattr(native, "low_latency_clock_alignment_version", None) != 1:
        raise RuntimeError("Rebuild the wheel with LL clock alignment v1")
    runtime = runtime_of(buffer)
    if runtime.low_latency_comm_args is None:
        raise RuntimeError("Clock alignment requires an initialized LL communication argument tensor")
    return dict(runtime=runtime, native=native)


def clock_window(state, args, operation, window, keep_rows=False):
    """Run one four-timestamp exchange and reduce it to a calibration anchor."""
    import torch

    runtime = state["runtime"]
    generation = getattr(args, "_clock_alignment_generation", 0) + 1
    args._clock_alignment_generation = generation
    operation_id = 1 if operation == "dispatch" else 2
    magic = 0xC10C000000000000 | (int(args.case_id + 1) << 24) | (operation_id << 16) | generation
    samples = torch.empty(
        (args.clock_align_samples, 8), dtype=torch.int64, device=f"npu:{runtime.device}"
    )
    with torch.npu.stream(runtime.stream):
        state["native"].low_latency_clock_sync(
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
    anchor = ll_clock_model.window_from_rows(rows, runtime.rank, window)
    return (anchor, rows) if keep_rows else anchor


def validate_clock_models(models, world, phase="calibration", rank=None, replay_id=None):
    """Admission for every gathered model, with the failing phase in the error."""
    where = f"phase={phase} rank={rank}" + ("" if replay_id is None else f" replay={replay_id}")
    try:
        ll_clock_model.validate_models(models, world)
    except ValueError as exc:
        raise RuntimeError(f"Clock model admission failed ({where}): {exc}") from exc
    return models


def calibrate_clock_model(state, args, operation, replay_id=None):
    """Fit one affine model from windows spaced far enough apart to see drift.

    One window resolves the offset only; the slope needs a span whose anchor
    errors are small against the rate being claimed, so the windows are
    separated by a Host sleep and each uses a fresh protocol generation. The
    gathered models are admitted immediately: an unresolved, clamped or
    unvalidated rank must stop the run here, not at the first deadline. The raw
    rows are returned so a report can keep the untouched evidence.
    """
    count = int(args.clock_align_calibration_windows)
    gap = float(args.clock_align_calibration_gap_secs)
    windows, raw = [], []
    for index in range(count):
        anchor, rows = clock_window(state, args, operation, index, keep_rows=True)
        windows.append(anchor)
        raw.append(rows)
        if index + 1 < count and gap > 0.0:
            time.sleep(gap)
    rank = state["runtime"].rank
    world = state["runtime"].world_size
    model = ll_clock_model.fit_model(windows, rank)
    models = host_gather(args, model, f"{operation}.clock.models")
    validate_clock_models(models, world, "calibration", rank, replay_id)
    return model, models, raw


def reference_base(models, lead_us):
    """Common reference-domain instant that is still ahead of every rank."""
    latest = max(ll_clock_model.reference_now(item) for item in models)
    return latest + int(lead_us) * 1000


def deadline_budgets(model, table, phase, rank, replay_id=None):
    """Per-iteration budget at the deadline actually handed to the device."""
    budgets = []
    for index, deadline in enumerate(table):
        try:
            budgets.append(ll_clock_model.error_budget(model, deadline))
        except ValueError as exc:
            suffix = "" if replay_id is None else f" replay={replay_id}"
            raise RuntimeError(
                f"Clock error budget unavailable (phase={phase} rank={rank} node={index}{suffix}): "
                f"{exc}") from exc
    return budgets


def peer_payloads(payloads, world, iterations, phase, rank, replay_id=None):
    """Validate one gathered peer payload list: coverage, order and lengths."""
    where = f"phase={phase} rank={rank}" + ("" if replay_id is None else f" replay={replay_id}")
    if not isinstance(payloads, list) or len(payloads) != world:
        raise RuntimeError(f"Clock peer payload needs {world} ranks ({where})")
    if [item.get("rank") if isinstance(item, dict) else None for item in payloads] != list(range(world)):
        raise RuntimeError(f"Clock peer payload does not follow EP rank order ({where})")
    for item in payloads:
        for field in ("release", "error_ns"):
            values = item.get(field)
            if values is None:
                continue
            if not isinstance(values, list) or len(values) != iterations:
                raise RuntimeError(f"Clock peer {field} length differs on rank {item.get('rank')} ({where})")
    return payloads


def per_iteration_pair_bounds(errors_by_rank, world):
    """Worst-rank and worst-two-rank budgets per iteration.

    The pair bound takes the two largest budgets of the same iteration, so it
    never adds extremes collected from different iterations.
    """
    iterations = len(errors_by_rank[0]) if errors_by_rank else 0
    global_max, pair = [], []
    for index in range(iterations):
        column = [int(row[index]) for row in errors_by_rank]
        global_max.append(max(column))
        if world <= 1:
            pair.append(0)
        else:
            top = sorted(column, reverse=True)[:2]
            pair.append(top[0] + top[1])
    return global_max, pair


def check_submit_budget(payloads, world, iterations, limit, phase, rank, replay_id=None):
    """Fail before a graph is submitted when the two-rank budget exceeds the limit."""
    payloads = peer_payloads(payloads, world, iterations, phase, rank, replay_id)
    errors = [item["error_ns"] for item in payloads]
    for row in errors:
        for value in row:
            if not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    f"Clock peer budget {value!r} is invalid (phase={phase} rank={rank})")
    limits = {item.get("limit") for item in payloads}
    if len(limits) != 1:
        raise RuntimeError(
            f"--clock-align-max-pair-error-ns differs across ranks: {sorted(limits, key=str)}")
    global_max, pair = per_iteration_pair_bounds(errors, world)
    peak = max(pair) if pair else 0
    if limit is not None and peak > limit:
        suffix = "" if replay_id is None else f" replay={replay_id}"
        raise RuntimeError(
            f"Clock pair error bound {peak} ns exceeds --clock-align-max-pair-error-ns={limit} "
            f"(phase={phase} rank={rank}{suffix})")
    return dict(
        global_max_error_ns=max(global_max) if global_max else 0,
        pair_error_bounds_ns=pair,
        pair_error_bound_ns_max=peak,
        max_pair_error_ns_limit=None if limit is None else int(limit),
        accuracy_status="reported_only" if limit is None else "threshold_met",
    )


def release_rows(payloads, world, iterations, phase, rank, replay_id=None):
    """Per-rank corrected releases, validated like every other peer payload."""
    payloads = peer_payloads(payloads, world, iterations, phase, rank, replay_id)
    return [item["release"] for item in payloads]


def prepare_clock_alignment(buffer, args, operation, allocate_releases=True):
    """Calibrate each rank and choose the common target of iteration zero.

    The budget is evaluated at the deadline actually handed to the device, not
    at the anchor, so a long horizon is not reported as an anchor-only error.
    """
    import torch

    state = clock_alignment_state(buffer)
    model, models, raw = calibrate_clock_model(state, args, operation)
    runtime = state["runtime"]
    reference_target = reference_base(models, args.clock_align_lead_us)
    first_deadline = ll_clock_model.local_deadline(model, reference_target)
    releases = None
    if allocate_releases:
        releases = torch.empty((args.num_iters,), dtype=torch.int64, device=f"npu:{runtime.device}")
        releases.record_stream(runtime.stream)
    targets = [reference_target + index * args.clock_align_interval_us * 1000
               for index in range(args.num_iters)]
    deadlines = [ll_clock_model.local_deadline(model, target) for target in targets]
    budgets = deadline_budgets(model, deadlines, operation, runtime.rank)
    limit = args.clock_align_max_pair_error_ns
    payloads = host_gather(
        args,
        dict(rank=runtime.rank, error_ns=[item["total_ns"] for item in budgets], limit=limit),
        f"{operation}.clock.budget",
    )
    budget_summary = check_submit_budget(payloads, runtime.world_size, len(targets), limit,
                                         operation, runtime.rank)
    metadata = dict(
        schema="deepep_clock_alignment.v2",
        model_schema=ll_clock_model.MODEL_SCHEMA,
        source="four-timestamp exchange with an affine rate fit and a held-out residual",
        cycles_per_second=CLOCK_CYCLES_PER_SECOND,
        samples=args.clock_align_samples,
        warmup_samples=args.clock_align_warmup_samples,
        calibration_windows=int(args.clock_align_calibration_windows),
        calibration_gap_secs=float(args.clock_align_calibration_gap_secs),
        lead_us=args.clock_align_lead_us,
        interval_us=args.clock_align_interval_us,
        reference_target=reference_target,
        first_local_deadline=first_deadline,
        first_deadline_error=ll_clock_model.error_budget(model, first_deadline),
        deadline_budget_ns_max=max(item["total_ns"] for item in budgets),
        rate_ppm=model["rate_ppm"],
        rate_resolved=bool(model["rate_resolved"]),
        residual_cycles=model["residual_cycles"],
        residual_validated=bool(model["residual_validated"]),
        max_model_error_bound_ns=max(
            ll_clock_model.error_budget(item, item["l0"])["total_ns"] for item in models
        ),
        models=models,
        calibration_rows=raw,
        **budget_summary,
    )
    return dict(runtime=runtime, native=state["native"], model=model, releases=releases, metadata=metadata)


def enqueue_clock_alignment(state, args, iteration):
    import torch

    target = state["metadata"]["reference_target"] + iteration * args.clock_align_interval_us * 1000
    deadline = ll_clock_model.local_deadline(state["model"], target)
    release = state["releases"][iteration:iteration + 1]
    with torch.npu.stream(state["runtime"].stream):
        state["native"].low_latency_clock_sleep_at(
            state["runtime"].native, deadline, release, state["runtime"].stream.npu_stream
        )


def finish_clock_alignment(state, args, operation, profile_dir):
    """Report model-domain releases and the global peer budget.

    The local budget is only one rank's error; the reported worst-rank and
    two-rank budgets come from every peer's own deadline budget.
    """
    model = state["model"]
    runtime = state["runtime"]
    releases = [int(value) for value in state["releases"].cpu().tolist()]
    corrected = [ll_clock_model.forward_reference(model, value) for value in releases]
    targets = [
        state["metadata"]["reference_target"] + index * args.clock_align_interval_us * 1000
        for index in range(args.num_iters)
    ]
    deadlines = [ll_clock_model.local_deadline(model, target) for target in targets]
    bounds = deadline_budgets(model, deadlines, operation, runtime.rank)
    limit = args.clock_align_max_pair_error_ns
    peers = host_gather(
        args,
        dict(rank=runtime.rank, release=corrected,
             error_ns=[item["total_ns"] for item in bounds], limit=limit),
        f"{operation}.clock.releases",
    )
    budget_summary = check_submit_budget(peers, runtime.world_size, len(targets), limit,
                                         operation, runtime.rank)
    rows = release_rows(peers, runtime.world_size, len(targets), operation, runtime.rank)
    skews = [max(row) - min(row) for row in zip(*rows)]
    lateness = [value - target for row, target in zip(zip(*rows), targets) for value in row]
    summary = {
        **state["metadata"],
        "release_skew_p50_ns": statistics.median(skews),
        "release_skew_p99_ns": percentile(skews, 0.99),
        "release_skew_max_ns": max(skews),
        "release_lateness_p50_ns": statistics.median(lateness),
        "release_lateness_p99_ns": percentile(lateness, 0.99),
        "release_lateness_max_ns": max(lateness),
        "local_error_bound_ns_max": max(item["total_ns"] for item in bounds),
        "release_pair_error_bounds_ns": budget_summary["pair_error_bounds_ns"],
        "release_pair_error_bound_ns_max": budget_summary["pair_error_bound_ns_max"],
        "global_max_error_ns": budget_summary["global_max_error_ns"],
        "accuracy_status": budget_summary["accuracy_status"],
        "alignment_scope": "model domain; records are required for physical evidence",
    }
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    (Path(profile_dir) / "clock_alignment.json").write_text(
        json.dumps(dict(summary=summary, local_model=model, local_release_cycles=releases,
                        local_corrected_cycles=corrected, reference_targets=targets,
                        local_error_bounds=bounds), indent=2) + "\n"
    )
    if not hasattr(args, "clock_alignment_metadata"):
        args.clock_alignment_metadata = {}
    args.clock_alignment_metadata[operation] = summary
    return summary


def clock_wait_table_contract(native):
    """Read the installed wheel's in-graph wait ABI before touching the device.

    The 128-byte record stride is an ABI property: a wheel that still writes a
    32-byte slot would silently overlap neighbouring cores, so a mismatch is
    rejected instead of being reinterpreted.
    """
    fields = [str(name) for name in getattr(native, "low_latency_clock_wait_record_fields", [])]
    words = getattr(native, "low_latency_clock_wait_record_words", None)
    stride_words = getattr(native, "low_latency_clock_wait_record_stride_words", None)
    stride_bytes = getattr(native, "low_latency_clock_wait_record_stride_bytes", None)
    alignment = getattr(native, "low_latency_clock_wait_record_alignment_bytes", None)
    schema = getattr(native, "low_latency_clock_wait_schema", None)
    arg_bytes = getattr(native, "low_latency_clock_wait_arg_bytes", None)
    cycles = getattr(native, "low_latency_clock_cycles_per_second", None)
    if (words != 4 or fields != ["enter", "release", "batch", "core"] or
            schema != 2 or stride_words != 16 or stride_bytes != 128 or alignment != 128 or
            arg_bytes != 40 or
            cycles != CLOCK_CYCLES_PER_SECOND or not hasattr(native, "low_latency_clock_wait_table")):
        raise RuntimeError(
            "Rebuild the wheel with the in-graph clock wait-table ABI "
            "(schema 2, four fields, 128-byte record slots, 40-byte arguments)"
        )
    return int(words), int(stride_words)


def validate_wait_records(rows, *, iterations, cores, words, stride_words, batch_tag, table,
                          rank, label):
    """Every participating core must own a complete record in every call.

    ``table=None`` checks only record completeness, which is what the capture
    pass can prove: no real deadline exists yet. A replay passes its deadline
    table so an enter after the deadline or a release before it also fails.
    """
    local_enter, local_release, core_skew = [], [], []
    late_enters = early = 0
    for index in range(iterations):
        node = rows[index]
        if len(node) != cores:
            raise RuntimeError(
                f"Rank {rank} {label} node {index} exposes {len(node)} wait slots, expected {cores}")
        node_enters, node_releases = [], []
        for core in range(cores):
            slot = node[core]
            if len(slot) != stride_words:
                raise RuntimeError(
                    f"Rank {rank} {label} node {index} core {core} slot width {len(slot)} "
                    f"!= {stride_words}")
            enter, release, tag, owner = (int(value) for value in slot[:words])
            if tag != batch_tag or owner != core:
                raise RuntimeError(
                    f"Rank {rank} {label} node {index} core {core} slot is not this call's record")
            if enter <= 0 or release < enter:
                raise RuntimeError(
                    f"Rank {rank} {label} node {index} core {core} has no wait record "
                    f"({list(slot[:words])})")
            if table is not None:
                if enter >= table[index]:
                    late_enters += 1
                if release < table[index]:
                    early += 1
            node_enters.append(enter)
            node_releases.append(release)
        local_enter.append(max(node_enters))
        local_release.append(max(node_releases))
        core_skew.append(max(node_releases) - min(node_releases))
    if late_enters:
        raise RuntimeError(
            f"Rank {rank} {label} entered after its deadline on {late_enters} cores; "
            "raise --clock-align-lead-us, widen --clock-align-interval-us, or shorten the "
            "calibration gap")
    if early:
        raise RuntimeError(f"Rank {rank} {label} released early on {early} cores")
    return dict(local_enter=local_enter, local_release=local_release, core_skew=core_skew)


def summarize_replay(model, evidence, targets, table, budgets, budget_summary, args, operation,
                     rank, world, replay_id, base):
    """Convert one replay's records into reference-domain skew and peer budgets.

    Budgets were gathered before the graph was submitted, so this only needs the
    corrected releases; the model domain skew is reported separately from the
    propagated error bounds.
    """
    releases, enters, core_skew = evidence["local_release"], evidence["local_enter"], evidence["core_skew"]
    corrected = [ll_clock_model.forward_reference(model, value) for value in releases]
    stage = f"{operation}.graph.clock.releases.{replay_id}"
    peers = host_gather(args, dict(rank=rank, release=corrected), stage)
    rows = release_rows(peers, world, len(table), f"{operation}.graph", rank, replay_id)
    skews = [max(row) - min(row) for row in zip(*rows)]
    lateness = [value - target for row, target in zip(zip(*rows), targets) for value in row]
    margins = [deadline - enter for deadline, enter in zip(table, enters)]
    overshoot = [release - deadline for release, deadline in zip(releases, table)]
    return dict(
        replay_id=int(replay_id),
        base_reference_cycle=int(base),
        deadline_margin_p50_ns=statistics.median(margins),
        deadline_margin_min_ns=min(margins),
        release_overshoot_p50_ns=statistics.median(overshoot),
        release_overshoot_max_ns=max(overshoot),
        core_skew_p50_ns=statistics.median(core_skew),
        core_skew_max_ns=max(core_skew),
        release_skew_p50_ns=statistics.median(skews),
        release_skew_p99_ns=percentile(skews, 0.99),
        release_skew_max_ns=max(skews),
        release_lateness_p50_ns=statistics.median(lateness),
        release_lateness_p99_ns=percentile(lateness, 0.99),
        release_lateness_max_ns=max(lateness),
        local_error_bound_ns_max=max(item["total_ns"] for item in budgets),
        pair_error_bounds_ns=budget_summary["pair_error_bounds_ns"],
        pair_error_bound_ns_max=budget_summary["pair_error_bound_ns_max"],
        global_max_error_ns=budget_summary["global_max_error_ns"],
        accuracy_status=budget_summary["accuracy_status"],
        local_enter_cycles=enters,
        local_release_cycles=releases,
        corrected_release_cycles=corrected,
        deadline_table=table,
        reference_targets=targets,
    )


def replay_with_profile(graph, stream, directory, iterations, operation):
    """Replay one graph under the profiler and return the Dispatch durations.

    With ``wait=0, warmup=0, active=1`` the first step is the recording one, so
    the replay must happen before the first ``profiler.step()``: stepping first
    would stop and export an empty window.
    """
    import torch
    import torch_npu

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with torch_npu.profiler.profile(
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=torch_npu.profiler.ExportType.Text,
            l2_cache=False, data_simplification=False),
        profile_memory=True, with_stack=False, with_flops=False, with_modules=False,
    ) as profiler:
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.synchronize()
        profiler.step()
    return graph_replay_perf(directory, iterations, operation)


def graph_replay_perf(root, iterations, operation):
    """Extract Dispatch durations and prove the wait -> Dispatch order.

    Both AOT and JIT task names map to the same logical Dispatch, so the same
    extraction works for either build; nothing outside the two admitted names
    is merged into the samples.
    """
    rows = profiler_kernel_records(root, (WAIT_PROFILER_ENTRY, *DISPATCH_PROFILER_ENTRIES))
    waits = [row for row in rows if row["logical"] == WAIT_PROFILER_ENTRY]
    dispatches = [row for row in rows if row["logical"] == DISPATCH_LOGICAL_NAME]
    if len(waits) != iterations or len(dispatches) != iterations:
        raise ValueError(
            f"Graph replay profiler saw {len(waits)} waits and {len(dispatches)} Dispatch tasks; "
            f"expected {iterations} each in {root}")
    if [row["logical"] for row in rows] != [WAIT_PROFILER_ENTRY, DISPATCH_LOGICAL_NAME] * iterations:
        raise ValueError(f"Graph replay did not alternate wait -> Dispatch in {root}")
    return [row["duration"] for row in dispatches]


def expected_send_counts(entries, world):
    """Cumulative (local expert, source rank) status prefix of one rank.

    The public count output follows the existing count ABI: statuses are laid
    out expert-major with one entry per source rank and each entry is the
    running total. This is not the per-expert receive count, and it is not the
    padded capacity region.
    """
    values, running = [], 0
    for rows in entries:
        per_source = [0] * world
        for source, _ in rows:
            per_source[source] += 1
        for source in range(world):
            running += per_source[source]
            values.append(running)
    return values


def fill_count_sentinels(captured):
    """Overwrite every node's count output so a stale value cannot pass."""
    for index, output in enumerate(captured):
        handle = output[3] if len(output) > 3 else None
        counts = getattr(handle, "send_counts", None)
        if counts is None:
            raise AssertionError(f"Captured node {index} lost its count output")
        counts.fill_(-1)


def verify_graph_outputs(captured, expected_counts, check, rank, replay_id):
    """Every node must rewrite its own counts, payload and scales.

    Counts are compared against values derived from the route and the count
    ABI, never against another node's output; payload and scales reuse the
    caller's existing verify_dispatch rules.
    """
    for index, output in enumerate(captured):
        handle = output[3] if len(output) > 3 else None
        counts = getattr(handle, "send_counts", None)
        if counts is None:
            raise AssertionError(f"Rank {rank} replay {replay_id} node {index} lost its count output")
        actual = [int(value) for value in counts.cpu().tolist()]
        if len(actual) != len(expected_counts):
            raise AssertionError(
                f"Rank {rank} replay {replay_id} node {index} count width {len(actual)} "
                f"!= {len(expected_counts)}")
        if actual != expected_counts:
            mismatch = next(item for item, (a, b) in enumerate(zip(actual, expected_counts)) if a != b)
            raise AssertionError(
                f"Rank {rank} replay {replay_id} node {index} count output differs at {mismatch}: "
                f"got {actual[mismatch]}, expected {expected_counts[mismatch]}")
        if check is None:
            continue
        try:
            check(output)
        except AssertionError as exc:
            raise AssertionError(f"Rank {rank} replay {replay_id} node {index} payload: {exc}") from exc
    return len(captured)


def assert_records_untouched(rows, rank, label):
    """Capture only records operations; a written record there is a bug."""
    for node_index, node in enumerate(rows):
        for core_index, slot in enumerate(node):
            if any(int(value) != -1 for value in slot):
                raise RuntimeError(
                    f"Rank {rank} {label} node {node_index} core {core_index} already has a record "
                    "before the probe replay")


def probe_capture_records(clear, replay, rows, *, iterations, cores, words, stride_words,
                          batch_tag, rank):
    """Validate the record path with one explicit replay, never with the capture.

    Capture enqueues the operations but does not execute them, so the record
    buffer still holds its sentinel. One replay against the all-zero (already
    expired) deadline table executes every documented wait and must produce a
    complete record set before calibration starts.
    """
    clear()
    replay()
    return validate_wait_records(rows(), iterations=iterations, cores=cores, words=words,
                                 stride_words=stride_words, batch_tag=batch_tag, table=None,
                                 rank=rank, label="probe_replay")


def graph_clock_alignment(buffer, invoke, args, operation, rank, world, mode, check=None,
                          expected_counts=None):
    """Capture one wait+operation pair per iteration, then replay with a refreshed table.

    Preparation order is part of the protocol: resources, first compile, core
    pinning, warmup and capture all finish before calibration, so the model is
    still fresh when the measured replay runs. Capture only enqueues operations,
    so records are validated by one explicit probe replay first. Afterwards every
    replay re-anchors, rewrites the table at the captured address, refills the
    sentinels and is validated on its own evidence.
    """
    import torch
    import torch_npu  # noqa: F401

    if operation != "dispatch":
        raise ValueError("Graph clock alignment covers dispatch only")
    if bool(args.do_cpu_sync):
        raise ValueError("Graph clock alignment cannot capture host count reads")
    if not hasattr(torch.npu, "NPUGraph"):
        raise RuntimeError("Installed torch_npu does not expose NPUGraph capture")

    state = clock_alignment_state(buffer)
    native, runtime = state["native"], state["runtime"]
    words, stride_words = clock_wait_table_contract(native)
    cores = int(native.low_latency_core_config()["active_aiv"])
    iterations = int(args.num_iters)
    if cores <= 0 or iterations < 1:
        raise ValueError("Graph clock alignment requires active AIV cores and positive iterations")
    if expected_counts is None and check is not None:
        raise ValueError("Graph clock alignment needs the expected count output with a payload check")
    device = f"npu:{runtime.device}"
    stream = runtime.stream
    replays = int(args.clock_align_replays)
    limit_ns = args.clock_align_max_pair_error_ns
    directory = (Path(args.run_directory) / f"case_{args.case_id:05d}" / mode / "clock_graph" /
                 operation / f"rank_{rank:03d}")
    directory.mkdir(parents=True, exist_ok=True)
    # The captured scalar tag cannot change on replay; replay identity stays on
    # the Host side, next to the per-replay sentinel refill.
    batch_tag = 1

    deadlines = torch.zeros((iterations,), dtype=torch.int64, device=device)
    records = torch.full((iterations, cores, stride_words), -1, dtype=torch.int64, device=device)

    completion_boundary(args)
    for _ in range(max(1, args.warmup)):
        invoke()
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        # First compile, core pinning and a wait launch before capture starts.
        for index in range(min(max(1, args.warmup), iterations)):
            native.low_latency_clock_wait_table(
                runtime.native, deadlines, iterations, index, records, batch_tag, False,
                stream.npu_stream)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    captured = []
    with torch.npu.graph(graph, stream=stream):
        for index in range(iterations):
            native.low_latency_clock_wait_table(
                runtime.native, deadlines, iterations, index, records, batch_tag, True,
                stream.npu_stream)
            captured.append(invoke())
    torch.npu.synchronize()

    def replay_graph():
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.synchronize()

    # Capture does not execute anything, so the record buffer must still hold
    # its sentinel. One explicit replay against the all-zero (already expired)
    # deadline table proves the record path before calibration. A failure keeps
    # the exact record snapshot instead of only its exception text.
    try:
        assert_records_untouched(records.cpu().tolist(), rank, "capture")
        probe = probe_capture_records(
            lambda: records.fill_(-1), replay_graph, lambda: records.cpu().tolist(),
            iterations=iterations, cores=cores, words=words, stride_words=stride_words,
            batch_tag=batch_tag, rank=rank)
    except Exception:
        (directory / "probe_failure.json").write_text(
            json.dumps(dict(cores=cores, stride_words=stride_words,
                            records=records.cpu().tolist()), indent=2) + "\n")
        raise

    model, models, raw = calibrate_clock_model(state, args, operation)
    calibration_rows = [dict(phase="initial", windows=raw)]
    interval = int(args.clock_align_interval_us) * 1000
    exp_prefix = int(getattr(args, "clock_align_graph_exp_prefix", 0))
    if exp_prefix:
        # The formal protocol runs the fixed Exp prefix after calibration and
        # before the fresh anchor: it belongs to the sampling conditions and
        # never enters the captured graph.
        with torch.npu.stream(stream):
            delay = torch.zeros((args.delay_rows, args.delay_cols), dtype=torch.float32,
                                device=device)
            for _ in range(exp_prefix):
                torch.exp(delay, out=delay)
        torch.npu.synchronize()
    reports, perf = [], None
    reanchor_checks, ages = [], []
    for replay_id in range(replays):
        # Re-anchor for every replay; refit only when the rate is older than the
        # admitted age or the new anchor disagrees with the old prediction. The
        # recovery is bounded to one refit per replay, never a retry loop.
        reanchor = None
        if ll_clock_model.model_age_seconds(model) > float(args.clock_align_model_max_age_secs):
            model, models, raw = calibrate_clock_model(state, args, operation, replay_id)
            calibration_rows.append(dict(phase=f"replay {replay_id}", windows=raw))
        else:
            window = clock_window(state, args, operation,
                                  int(args.clock_align_calibration_windows) + replay_id)
            residual = ll_clock_model.reanchor_residual(model, window)
            allowed = ll_clock_model.reanchor_limit(model, window)
            reanchor = dict(residual_cycles=residual, limit_cycles=allowed,
                            accepted=abs(residual) <= allowed)
            if abs(residual) > allowed:
                model, models, raw = calibrate_clock_model(state, args, operation, replay_id)
                calibration_rows.append(dict(phase=f"replay {replay_id}", windows=raw))
            else:
                model = ll_clock_model.reanchor(model, window)
                models = host_gather(args, model, f"{operation}.clock.reanchor.{replay_id}")
                validate_clock_models(models, world, "reanchor", rank, replay_id)
        reanchor_checks.append(reanchor)
        ages.append(round(ll_clock_model.model_age_seconds(model), 3))

        base = reference_base(models, args.clock_align_lead_us)
        targets = [base + index * interval for index in range(iterations)]
        table = [ll_clock_model.local_deadline(model, target) for target in targets]
        budgets = deadline_budgets(model, table, f"{operation}.graph", rank, replay_id)
        budget_payloads = host_gather(
            args,
            dict(rank=rank, error_ns=[item["total_ns"] for item in budgets], limit=limit_ns),
            f"{operation}.graph.budget.{replay_id}",
        )
        # The two-rank budget is checked before the graph is submitted, so a
        # model that cannot meet the configured precision never runs.
        budget_summary = check_submit_budget(budget_payloads, world, iterations, limit_ns,
                                             f"{operation}.graph", rank, replay_id)

        with torch.npu.stream(stream):
            records.fill_(-1)   # a missing write must never look like valid evidence
            deadlines.copy_(torch.tensor(table, dtype=torch.int64, device=device))
            if captured:
                fill_count_sentinels(captured)
        # Same-stream ordering makes the refreshed table, sentinels and counts
        # visible to the replay without any extra flush of the storages.
        runtime.stream.synchronize()
        profiled = not args.skip_perf_test and replay_id + 1 == replays
        samples = None
        if profiled:
            samples = replay_with_profile(graph, stream, directory / f"replay_{replay_id:02d}",
                                          iterations, operation)
        else:
            replay_graph()
        try:
            evidence = validate_wait_records(
                records.cpu().tolist(), iterations=iterations, cores=cores, words=words,
                stride_words=stride_words, batch_tag=batch_tag, table=table, rank=rank,
                label=f"replay {replay_id}")
            checked = (verify_graph_outputs(captured, expected_counts, check, rank, replay_id)
                       if captured else 0)
        except Exception:
            (directory / f"replay_{replay_id:02d}_failure.json").write_text(
                json.dumps(dict(cores=cores, stride_words=stride_words, table=table,
                                records=records.cpu().tolist()), indent=2) + "\n")
            raise
        report = summarize_replay(model, evidence, targets, table, budgets, budget_summary, args,
                                  operation, rank, world, replay_id, base)
        report.update(reanchor=reanchor, count_outputs_checked=checked)
        reports.append(report)
        (directory / f"replay_{replay_id:02d}.json").write_text(json.dumps(dict(
            summary={key: value for key, value in report.items()
                     if key not in ("local_enter_cycles", "local_release_cycles",
                                    "corrected_release_cycles")},
            model=model,
            records=records.cpu().tolist(),
        ), indent=2) + "\n")
        if samples is not None:
            gathered = host_gather(args, samples, f"{operation}.graph.perf.{replay_id}")
            perf = summarize_kernel(gathered, DISPATCH_LOGICAL_NAME, args.perf_tail)
            perf.update(profiled_replay=replay_id, waits_verified=iterations,
                        record_stride_bytes=stride_words * 8)

    stripped = [{
        key: value for key, value in item.items()
        if key not in ("local_enter_cycles", "local_release_cycles",
                       "corrected_release_cycles", "deadline_table", "reference_targets")
    } for item in reports]
    summary = dict(
        schema="deepep_clock_alignment.graph.v2",
        protocol=CLOCK_ALIGNMENT_GRAPH_PROTOCOL,
        route=mode,
        iterations=iterations,
        replays=replays,
        cores=cores,
        batch_id=batch_tag,
        record_words=words,
        record_stride_words=stride_words,
        record_stride_bytes=stride_words * 8,
        record_fields=["enter", "release", "batch", "core"],
        probe=dict(records_complete=True, cores=cores,
                   valid_samples=probe and len(probe["local_release"]) or 0),
        samples=args.clock_align_samples,
        calibration_windows=int(args.clock_align_calibration_windows),
        calibration_gap_secs=float(args.clock_align_calibration_gap_secs),
        lead_us=int(args.clock_align_lead_us),
        interval_us=int(args.clock_align_interval_us),
        exp_prefix_repeats=exp_prefix,
        model=model,
        models=models,
        replay=stripped[-1],
        replay_summaries=stripped,
        reanchor_checks=reanchor_checks,
        calibration_age_secs=ages,
        calibration_rows=calibration_rows,
        acceptance=dict(records_complete=True, all_cores_waited=True, no_early_release=True,
                        replays_verified=len(reports),
                        count_outputs_checked=min(item["count_outputs_checked"] for item in reports)
                        if reports else 0),
        performance=perf,
        alignment_scope="model domain; propagated bounds reported separately",
    )
    (directory / "clock_alignment.json").write_text(
        json.dumps(dict(summary=summary, final_replay=reports[-1]), indent=2) + "\n"
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
    from deep_ep.utils.ll_reference_profile import ReferenceTrace
    return ReferenceTrace(buffer, operation, frequency_hz=args.ll_timer_frequency_hz)


def diagnostic_calls(buffer, invoke, check, args, operation, rank, m, mode,
                     *, prepared=None, prelude_buffers=None):
    """Sample a stabilized tail, retaining outputs/timers until all launches finish."""
    reference = bool(getattr(args, 'll_reference_profile', False))
    from deep_ep.utils.ll_reference_profile import capture
    from deep_ep._native import require_native
    import torch
    import torch.distributed as dist
    world = control_world(args)
    logical_group = compiled_group(require_native("LL timer"),world)
    capture_mode = 'reference'
    capture_directory = 'll_reference_profile'
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
    expert_dtype = torch.bfloat16
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
        if args.clock_align and getattr(args, 'clock_align_mode', 'stream') == 'graph':
            progress("dispatch.graph")
            clock_summary = graph_clock_alignment(
                buffer, dispatch, args, 'dispatch', rank, world, mode,
                check=None if args.skip_check else
                (lambda output: verify_dispatch(output[0], output[3], entries, args, world, m)),
                expected_counts=expected_send_counts(entries, world))
            result['clock_alignment'] = clock_summary
            if clock_summary.get('performance') is not None:
                result['dispatch'] = clock_summary['performance']
        else:
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
                    rtol=0.01,
                    atol=0.03125,
                )

        agree_check(check_combine, args, rank)
    def check_timed_combine(output):
        if output[1] is not None or tuple(output[0].shape) != (args.num_tokens, args.hidden):
            raise AssertionError("Instrumented combine output contract differs")
        torch.testing.assert_close(output[0].cpu(), expected,
                                   rtol=0.01,
                                   atol=0.03125)
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
    if args.clock_align and getattr(args, "clock_align_mode", "stream") == "graph":
        clock_wait_table_contract(native)
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
                      "clock_alignment_version": getattr(native, "low_latency_clock_alignment_version", None)}
        if getattr(args, 'll_reference_profile', False):
            info = json.loads(Path(deep_ep.__file__).with_name('_build_info.json').read_text(encoding='utf-8'))
            if info.get('ll_reference_profile') is not True:
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
        validate_reference_profile_capability(getattr(args, 'll_reference_profile', False), capability)
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
            if args.clock_align and getattr(args, 'clock_align_mode', 'stream') == 'graph':
                clock_note = ("one captured wait+Dispatch pair per iteration; each replay re-anchors the affine "
                              "clock model, refreshes the deadline table at the captured address and is verified "
                              "against its own per-core records; the profiled replay yields Dispatch durations")
            elif args.clock_align:
                clock_note = "fresh four-timestamp calibration and one device sleep-at before every measured call"
            else:
                clock_note = "fixed Exp100 before active profiler.step"
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
                    + clock_note
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
        graph_aligned = clock_aligned and getattr(args, "clock_align_mode", "stream") == "graph"
        print(json.dumps({"world_size": world, "case_count": len(cases), "cases": cases,
                          "sampling": dict(protocol=sampling_protocol(args),
                                           exp_prefix_repeats=(
                                               int(args.clock_align_graph_exp_prefix) if graph_aligned
                                               else 0 if clock_aligned else EXP_PREFIX_REPEATS),
                                           host_coordination=args.group_backend,
                                           per_iteration_exp=0, per_iteration_all_reduce=False,
                                           per_iteration_host_collective=False,
                                           per_iteration_rank_alignment=(
                                               (CLOCK_ALIGNMENT_GRAPH_PROTOCOL if graph_aligned
                                                else CLOCK_ALIGNMENT_PROTOCOL)
                                               if clock_aligned else "none"))}, indent=2))
        return
    configure_endpoints(args, os.environ)
    if local is not None:
        worker(local, args)
    else:
        import torch.multiprocessing as mp

        mp.spawn(worker, args=(args,), nprocs=args.num_processes, join=True)


if __name__ == "__main__":
    main()
