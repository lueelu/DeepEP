# Copyright (c) 2026, Lu Lu
# Modified by ryan_li 2026

"""Installed-wheel LL functional/performance ST, modeled on DeepEP test_ep.py.

Run explicitly; ordinary pytest never initializes NPU devices from this module.
Use --help for launch, configuration, and profiling options.
"""

import argparse
import csv
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
    parser.add_argument("--perf-tail", type=int, default=0, help="Statistics use last N samples; 0 uses all")
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
    parser.add_argument("--group-backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument(
        "--without-buffer-group",
        action="store_true",
        help="Keep the test control group, but construct Buffer with rank/world_size",
    )
    parser.add_argument("--route", choices=("normal",), default="normal")
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
    parser.add_argument("--delay-exp-before", type=int, default=None, help="Default: max(num_iters // 2, 20)")
    parser.add_argument("--delay-exp-after", type=int, default=None, help="Default: max(num_iters // 2, 20)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--repeat-combine", type=int, default=2)
    parser.add_argument("--output", help="New rank-0 JSON result file (will not overwrite)")
    return parser


def kernel_durations(root, kernel, expected):
    """Read only the requested device kernel from torch_npu text exports."""
    rows = []
    for path in sorted(Path(root).rglob("kernel_details.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                normalized = {key.replace(" ", "").lower(): value for key, value in row.items() if key}
                name = normalized.get("name", normalized.get("opname", ""))
                if not re.search(r"\b" + re.escape(kernel) + r"\b", name):
                    continue
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
        "p95_us": percentile(0.95),
        "p99_us": percentile(0.99),
    }


def summarize_kernel(samples_by_rank, kernel, tail):
    ranks = []
    for rank, raw in enumerate(samples_by_rank):
        selected = raw[-tail:] if tail else raw
        ranks.append({"rank": rank, **sample_stats(selected), "kernel_us": raw})
    fastest = min(ranks, key=lambda row: row["mean_us"])
    slowest = max(ranks, key=lambda row: row["mean_us"])
    return {
        "measurement": "kernel_duration_us",
        "kernel": kernel,
        "perf_tail": tail,
        "ranks": ranks,
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
            base = {"case_id": result["case_id"], **result["config"], "route": result["route"], "operation": operation}
            summaries.append(
                {
                    **base,
                    **{
                        key: perf[key]
                        for key in (
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
        config = json.loads(Path(config_path).read_text())
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
    default = max(args.num_iters // 2, 20)
    for name in ("delay_exp_before", "delay_exp_after"):
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
    if world not in (16, 32, 64, 128):
        raise ValueError("LL ST requires EP16/32/64/128; EP8 would exercise HT, not LL")
    m = args.num_tokens if args.num_max_tokens_per_rank is None else args.num_max_tokens_per_rank
    if not 0 < args.num_tokens <= m <= 256:
        raise ValueError("LL ST requires 0 < actual B <= M <= 256")
    if not 4 <= args.hidden <= 2147483647:
        raise ValueError("H must be >=4 for ST row identifiers and fit native int; native layout checks apply")
    if not 1 <= args.num_topk <= 16 or not world <= args.num_experts <= 1024 or args.num_experts % world:
        raise ValueError("Require K=1..16, EP<=E<=1024 and E divisible by EP")
    if args.num_sms != 0 or args.num_qps != 0:
        raise ValueError("LL ST requires --num-sms=0 and --num-qps=0; no HT fallback is permitted")
    if args.warmup < 0 or args.num_iters < 1 or args.repeat_combine < 2:
        raise ValueError("Require warmup>=0, num-iters>=1 and repeat-combine>=2")
    if not 1 <= args.num_cpu_timeout_secs <= 300:
        raise ValueError("num-cpu-timeout-secs must be in [1,300]")
    if not (2 << 20) <= args.num_bytes <= (32 << 30) or args.num_bytes % (2 << 20):
        raise ValueError("num-bytes must be 2 MiB aligned and between 2 MiB and 32 GiB")
    if args.delay_rows <= 0 or args.delay_cols <= 0 or min(args.delay_exp_before, args.delay_exp_after) < 0:
        raise ValueError("Delay tensor dimensions must be positive and exp counts nonnegative")
    if not 0 <= args.perf_tail <= args.num_iters:
        raise ValueError("Require 0 <= perf-tail <= num-iters")
    if args.output and Path(args.output).exists():
        raise ValueError("Output file already exists")
    return m


def make_routes(world, b, k, e, seed):
    """Independent CPU routes: no repeated experts within any token."""
    rng = random.Random(seed)
    result = []
    for _ in range(world):
        rows = []
        for _ in range(b):
            chosen = rng.sample(range(e), k)
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
                    "rank": dist.get_rank(),
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
    import torch
    import torch.distributed as dist

    error = None
    try:
        trace_call(args, "check.action", action)
    except Exception as exc:
        error = exc
        print(f"[rank {rank}] check failed: {exc}", flush=True)
    status = torch.tensor([int(error is not None)], dtype=torch.int32, device=control_device(args, torch))
    trace_call(args, "check.all_reduce", dist.all_reduce, status, op=dist.ReduceOp.MAX)
    if trace_call(args, "check.status.item", status.item):
        raise AssertionError("LL ST correctness failed on at least one rank") from error


def completion_boundary(args=None):
    """Test-side all-rank completion required by the existing LL call contract."""
    import torch
    import torch.distributed as dist

    trace_call(args, "boundary.synchronize.before", torch.npu.synchronize)
    trace_call(args, "boundary.barrier", dist.barrier)
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


def measure(invoke, args, operation, profile_dir, progress=None):
    """Measure exported device kernel tasks; never substitute Host wall time."""
    import torch
    import torch.distributed as dist
    import torch_npu

    kernel = f"{operation}_server_dedup_kernel"
    if progress:
        progress(f"{operation}.warmup")
    for _ in range(args.warmup):
        completion_boundary(args)
        output = trace_call(args, f"{operation}.invoke", invoke)
        trace_call(args, f"{operation}.event.wait", output[-1].current_stream_wait)
        trace_call(args, "device.synchronize", torch.npu.synchronize)
        del output
    completion_boundary(args)
    if progress:
        progress(f"{operation}.profile")
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=torch_npu.profiler.ExportType.Text,
            l2_cache=False,
            data_simplification=False,
        ),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
    ):
        # One capture, no profiler.step(), matching profile_operator's launch pattern.
        alignment = torch.randn(128, dtype=torch.float32, device=control_device(args, torch))
        request = trace_call(
            args, f"{operation}.alignment.initial", dist.all_reduce, alignment, op=dist.ReduceOp.SUM, async_op=True
        )
        trace_call(args, f"{operation}.alignment.wait", request.wait)
        delay = torch.zeros(args.delay_rows, args.delay_cols, dtype=torch.float32, device="npu")
        for _ in range(args.delay_exp_before):
            torch.exp(delay, out=delay)
        output = None
        for iteration in range(args.num_iters):
            with torch.profiler.record_function(f"deepep_ll_{operation}"):
                if operation == "combine" and args.group_backend == "gloo":
                    # CPU collectives cannot order completion of prior NPU combine calls.
                    completion_boundary(args)
                trace_call(
                    args,
                    f"{operation}.alignment.iteration.{iteration}",
                    dist.all_reduce,
                    alignment,
                    op=dist.ReduceOp.SUM,
                    async_op=False,
                )
                output = trace_call(args, f"{operation}.invoke", invoke)
            if iteration == 0:
                for _ in range(args.delay_exp_after):
                    torch.exp(delay, out=delay)
        # Finish device work once before exporting; no per-iteration NPU sync/event wait.
        trace_call(args, "device.synchronize", torch.npu.synchronize)
        del output, delay, alignment
    if progress:
        progress(f"{operation}.read_profile")
    samples = []

    def parse_profile():
        samples.extend(kernel_durations(profile_dir, kernel, args.num_iters))
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        (Path(profile_dir) / "samples.json").write_text(json.dumps({"kernel": kernel, "kernel_us": samples}) + "\n")

    agree_check(parse_profile, args, dist.get_rank())
    gathered = [None] * dist.get_world_size()
    trace_call(args, f"{operation}.samples.all_gather", dist.all_gather_object, gathered, samples)
    return summarize_kernel(gathered, kernel, args.perf_tail)


def run_case(buffer, args, rank, world, m, mode):
    import torch

    def progress(phase):
        record_case_progress(args, rank, m, mode, phase)

    progress("prepare")
    routes = make_routes(world, args.num_tokens, args.num_topk, args.num_experts, args.seed)
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

    progress("dispatch.check")
    completion_boundary(args)
    recv, recv_idx, recv_weights, handle, event = trace_call(args, "dispatch.invoke", dispatch)
    trace_call(args, "dispatch.event.wait", event.current_stream_wait)
    trace_call(args, "device.synchronize", torch.npu.synchronize)

    def check_backend():
        if not handle.is_low_latency or recv_idx is not None or recv_weights is not None:
            raise AssertionError("ST did not execute the LL backend or returned unexpected public weights")
        if handle.source_weights is None or handle.destination_index is None:
            raise AssertionError("LL handle lost its native tables/weights")

    agree_check(check_backend, args, rank)  # Never allow --skip-check to hide an HT fallback.
    if not args.skip_check:
        agree_check(lambda: verify_dispatch(recv, handle, entries, args, world, m), args, rank)
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
            args, "combine.invoke", buffer.combine, expert_output, handle, topk_weights=None
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
    result = {
        "case_id": args.case_id,
        "route": mode,
        "correctness": "skipped" if args.skip_check else "passed",
        "backend": "ll",
        "config": case_config(args, m),
    }
    if not args.skip_perf_test:
        if rank == 0:
            print(f"[LL ST] timing dispatch/combine: warmup={args.warmup}, iterations={args.num_iters}", flush=True)
        profile = Path(args.run_directory) / f"case_{args.case_id:05d}" / mode
        rank_directory = f"rank_{rank:03d}"
        result["dispatch"] = measure(dispatch, args, "dispatch", profile / "dispatch" / rank_directory, progress)
        # Prepare fresh matching metadata and expert output outside the combine capture.
        progress("dispatch.refresh")
        completion_boundary(args)
        recv, _, _, handle, event = trace_call(args, "dispatch.invoke", dispatch)
        trace_call(args, "dispatch.event.wait", event.current_stream_wait)
        trace_call(args, "device.synchronize", torch.npu.synchronize)
        expert_output = prepare_experts(recv, counts, rank, args.num_experts, world, expert_dtype)
        result["combine"] = measure(
            lambda: buffer.combine(expert_output, handle, topk_weights=None),
            args,
            "combine",
            profile / "combine" / rank_directory,
            progress,
        )
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
    dist.init_process_group(
        args.group_backend,
        init_method="env://",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=args.num_cpu_timeout_secs),
    )
    buffer = None
    try:
        run_directory = [None]

        def reserve_directory():
            if rank == 0:
                parent = Path(args.output_dir).absolute()
                parent.mkdir(parents=True, exist_ok=True)
                run_directory[0] = tempfile.mkdtemp(prefix=datetime.now().strftime("%m%d%H%M_"), dir=parent)

        agree_check(reserve_directory, args, rank)
        dist.broadcast_object_list(run_directory, src=0)
        args.run_directory = run_directory[0]
        modes = ("normal",)

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
            {"rank": rank, "world_size": world} if args.without_buffer_group else {"group": dist.group.WORLD}
        )

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
            )

        if not args.fresh_buffer_per_case:
            buffer = create_buffer(args)
        with torch.no_grad():
            results = []
            for case_id, case in enumerate(cases):
                case_m = validate_args(case, world)
                case.run_directory = args.run_directory
                case.case_id = case_id
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
                "config": vars(args),
                "package": deep_ep.__file__,
                "measurement": "torch_npu profiler kernel tasks; per-rank statistics and slowest rank mean",
                "alignment": "separate dispatch/combine captures: all_reduce + exp padding; Gloo combine uses completion boundaries",
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
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Runtime/device failures are not recovered or retried on HT.
        if buffer is not None:
            buffer.destroy()
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
        print(json.dumps({"world_size": world, "case_count": len(cases), "cases": cases}, indent=2))
        return
    configure_endpoints(args, os.environ)
    if local is not None:
        worker(local, args)
    else:
        import torch.multiprocessing as mp

        mp.spawn(worker, args=(args,), nprocs=args.num_processes, join=True)


if __name__ == "__main__":
    main()
