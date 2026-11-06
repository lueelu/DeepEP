#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""MegaMoE BF16: one entry for full forward/backward precision and timing."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import importlib
from importlib.machinery import EXTENSION_SUFFIXES, PathFinder
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import struct
import subprocess
import sys
import sysconfig
import time
import traceback
from urllib.parse import urlsplit
from uuid import uuid4


DATA_PROFILES = (
    "random",
    "normal",
    "zero_x",
    "zero_w1",
    "zero_w2",
    "zero_dy",
    "near_zero",
    "cancellation",
    "outliers",
    "equal_weights",
    "one_hot_weights",
    "clip_boundary",
)


def parse_devices(value):
    try:
        devices = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated physical device IDs") from error
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("Device IDs must be distinct and nonnegative")
    return devices


def resolved_save_mode(mode, limit):
    if mode != "auto":
        return mode
    limit = struct.unpack("f", struct.pack("f", limit))[0]
    return "mxfp8" if limit <= 0 or limit == float.fromhex("0x1.fffffep127") else "bf16"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", type=parse_devices, nargs="?", default=None)
    parser.add_argument(
        "--atk", action="store_true", help="Explicit ATK request; all precision runs use ATK by default"
    )
    parser.add_argument(
        "--atk-ratios",
        type=float,
        nargs=3,
        metavar=("MAX", "MEAN", "RMSE"),
        default=None,
        help="Explicit ATK ratio limits; default 5 / 1.5 / 1.5, number_count_radio stays 2",
    )
    parser.add_argument("--suite", choices=("priority", "model", "dtype", "gather", "all"), default=None)
    parser.add_argument(
        "--skip-mxfp8",
        action="store_true",
        help="Exclude suite cases resolving to MXFP8, including auto without clipping",
    )
    parser.add_argument(
        "--isolate-cases", action="store_true", help="Start fresh workers/runtime per case instead of reusing them"
    )
    parser.add_argument(
        "--diagnose-mxfp8",
        action="store_true",
        help="Opt-in saved-state diagnosis: fused controls and Gate/Up reference replay; requires --skip performance",
    )
    parser.add_argument("--suite-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", action="append", default=None, help="Select an ID within --suite; repeatable")
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List the selected suite without importing Torch/ATK or running devices",
    )
    parser.add_argument(
        "--skip",
        choices=("precision", "performance"),
        default=None,
        help="Skip precision or performance; by default run precision, then performance",
    )
    parser.add_argument(
        "--seq", type=int, default=129, help="Tokens per rank entering this operator, not model sequence length"
    )
    parser.add_argument("--hidden", type=int, default=256)
    width = parser.add_mutually_exclusive_group()
    width.add_argument("--ffn", type=int, default=512, help="SwiGLU output width F; GMM1 output is 2*F")
    width.add_argument("--gmm-n", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--topk-weight-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--topk-weight-values",
        choices=("bf16-rounded", "full-precision"),
        default=None,
        help="Default: BF16-rounded for BF16, full precision for FP32; rounded FP32 equals BF16 p.float()",
    )
    parser.add_argument("--expert-per-rank", type=int, default=2)
    parser.add_argument(
        "--routed-scaling-factor",
        type=float,
        default=1.0,
        help="Scale normalized route weights before dtype conversion",
    )
    parser.add_argument(
        "--routing-mode",
        choices=("balanced", "random", "imbalanced", "hot_expert", "hot_rank", "local_only", "remote_only", "ragged"),
        default="balanced",
    )
    parser.add_argument(
        "--data-profile", choices=DATA_PROFILES, default="random", help="Reproducible input-value scenario"
    )
    parser.add_argument("--save-gmm1-output", choices=("none", "bf16", "mxfp8", "auto"), default="bf16")
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--chunk-rows", type=int, default=256, help="CPU FP64 expert-row chunk size")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument(
        "--log",
        choices=("quiet", "rank0", "all"),
        default="quiet",
        help="quiet: results only; rank0: + phase log and perf stages for rank 0; all: every rank",
    )
    parser.add_argument(
        "--baseline",
        choices=("on", "off"),
        default="on",
        help="Time the small-operator baseline next to the MegaMoE BF16 operator; off skips it",
    )
    parser.add_argument("--local-mem-size", type=int, default=None, help="SHMEM bytes; default auto, aligned to 2 MiB")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--shmem-ip-port", default="tcp://127.0.0.1:29862")
    parser.add_argument("--output", type=Path, default=None, help="New report directory")
    args = parser.parse_args(argv)
    # Keep earlier standalone commands usable while publishing only F in the
    # new manifest/report schema. N denotes GMM1's full Gate+Up width, 2*F.
    if args.gmm_n is not None:
        if args.gmm_n <= 0 or args.gmm_n % 512:
            parser.error("gmm-n must be positive and divisible by 512")
        args.ffn = args.gmm_n // 2
    del args.gmm_n
    if args.topk_weight_values is None:
        args.topk_weight_values = "bf16-rounded" if args.topk_weight_dtype == "bf16" else "full-precision"
    if args.topk_weight_dtype == "bf16" and args.topk_weight_values != "bf16-rounded":
        parser.error("full-precision route weights require --topk-weight-dtype fp32")
    if (args.list_cases or args.case_id) and not args.suite:
        parser.error("--list-cases and --case-id require --suite")
    if args.skip_mxfp8 and (not args.suite or args.diagnose_mxfp8):
        parser.error("--skip-mxfp8 requires --suite and cannot be combined with --diagnose-mxfp8")
    if (args.isolate_cases or args.suite_worker) and not args.suite:
        parser.error("--isolate-cases requires --suite")
    if args.suite_worker and (args.isolate_cases or args.output is None):
        parser.error("Internal suite workers require --output and persistent execution")
    if args.atk and args.skip == "precision":
        parser.error("--atk requires precision; use --skip performance for accuracy only")
    args.atk = args.skip != "precision"
    if args.diagnose_mxfp8 and args.skip != "performance":
        parser.error("--diagnose-mxfp8 requires --skip performance")
    if args.atk_ratios is not None and (
        not args.atk or not all(math.isfinite(value) and value > 0 for value in args.atk_ratios)
    ):
        parser.error("--atk-ratios requires precision and three finite positive limits")
    if args.suite:
        if "RANK" in os.environ:
            parser.error("Run --suite once with python, rather than under torchrun; single cases support torchrun")
        owned = {
            "--seq",
            "--hidden",
            "--ffn",
            "--gmm-n",
            "--topk",
            "--topk-weight-dtype",
            "--topk-weight-values",
            "--expert-per-rank",
            "--routed-scaling-factor",
            "--routing-mode",
            "--data-profile",
            "--save-gmm1-output",
            "--swiglu-limit",
            "--seed",
        }
        if any(token.partition("=")[0] in owned for token in (sys.argv[1:] if argv is None else argv)):
            parser.error("--suite defines shape, routing, data profile, save mode, clipping and seed; omit case flags")
    inherited = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    args.devices = args.devices or (
        parse_devices(inherited) if inherited else list(range(int(os.environ.get("LOCAL_WORLD_SIZE", "2"))))
    )
    world = int(os.environ.get("WORLD_SIZE", str(len(args.devices))))
    if not 2 <= world <= 1024 or not 1 <= args.seq <= 16384 or not 1 <= args.expert_per_rank <= 32:
        parser.error("Requires EP in 2..1024, seq in 1..16384 and local experts in 1..32")
    if args.hidden <= 0 or args.hidden % 256 or args.ffn <= 0 or args.ffn % 256:
        parser.error("Requires hidden % 256 == 0 and ffn % 256 == 0, both positive")
    if world * args.expert_per_rank > 2048 or not 1 <= args.topk <= world * args.expert_per_rank:
        parser.error("Requires total experts <= 2048 and 1 <= topk <= total experts")
    if args.seed < 0 or args.chunk_rows <= 0 or args.warmup < 0 or args.loops <= 0 or not 1 <= args.timeout <= 3600:
        parser.error("Invalid seed, chunk size, iteration count or timeout")
    if not math.isfinite(args.swiglu_limit) or abs(args.swiglu_limit) > float.fromhex("0x1.fffffep127"):
        parser.error("swiglu-limit must be finite and representable as float32")
    args.swiglu_limit = struct.unpack("f", struct.pack("f", args.swiglu_limit))[0]
    if not math.isfinite(args.routed_scaling_factor) or not 0 < args.routed_scaling_factor <= float.fromhex(
        "0x1.fep127"
    ):
        parser.error("routed-scaling-factor must be positive, finite and within the BF16 finite range")
    args.routed_scaling_factor = struct.unpack("f", struct.pack("f", args.routed_scaling_factor))[0]
    if args.routed_scaling_factor == 0:
        parser.error("routed-scaling-factor must remain positive in FP32")
    if args.diagnose_mxfp8 and not args.suite:
        if resolved_save_mode(args.save_gmm1_output, args.swiglu_limit) != "mxfp8":
            parser.error("--diagnose-mxfp8 requires a case resolving to mxfp8")
    if args.routing_mode == "imbalanced" and args.topk == world * args.expert_per_rank:
        parser.error("imbalanced routing requires topk < total experts; selecting every expert cannot skew counts")
    if args.routing_mode == "hot_expert" and args.topk != 1:
        parser.error("hot_expert requires topk=1 to keep expert IDs distinct per token")
    if args.routing_mode in ("hot_rank", "local_only", "remote_only") and args.topk > args.expert_per_rank:
        parser.error(f"{args.routing_mode} requires topk <= expert-per-rank")
    if args.routing_mode == "ragged" and (args.topk != 1 or args.expert_per_rank < 6 or args.seq < 386):
        parser.error("ragged requires topk=1, expert-per-rank>=6 and seq>=386")
    if args.data_profile == "clip_boundary":
        limit_bits = struct.unpack("I", struct.pack("f", args.swiglu_limit))[0]
        if not 0 < args.swiglu_limit < float.fromhex("0x1.fep127") or limit_bits & 0xFFFF:
            parser.error("clip_boundary requires a positive, exactly BF16-representable limit below BF16 max")
    endpoint = urlsplit(args.shmem_ip_port)
    if (
        endpoint.scheme != "tcp"
        or not endpoint.hostname
        or endpoint.path
        or not endpoint.port
        or not 1024 <= endpoint.port <= 65533
    ):
        parser.error("shmem-ip-port must be tcp://<IPv4>:<port>, with port in 1024..65533")
    import ipaddress

    address = ipaddress.IPv4Address(endpoint.hostname)
    if world > len(args.devices) and address.is_loopback:
        parser.error("Multi-node jobs require a rank-0 address reachable from every node")
    if args.local_mem_size is not None and (args.local_mem_size < 2 << 20 or args.local_mem_size % (2 << 20)):
        parser.error("local-mem-size must be positive and aligned to 2 MiB")
    if "LOCAL_WORLD_SIZE" in os.environ and int(os.environ["LOCAL_WORLD_SIZE"]) != len(args.devices):
        parser.error("The device list must match torchrun's local process count")
    return args


def suite_cases(args, *, include_skipped=False):
    manifest = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    defaults = manifest["defaults"]
    seen, selected = set(), []
    for item in manifest["cases"]:
        identity = item["id"]
        if not re.fullmatch(r"[a-z0-9_]+", identity) or identity in seen:
            raise ValueError(f"Invalid/duplicate case ID: {identity}")
        seen.add(identity)
        if not item.get("covers", "").strip():
            raise ValueError(f"Missing coverage rationale in {identity}")
        if not item["suites"] or set(item["suites"]) - {"priority", "model", "dtype", "gather"}:
            raise ValueError(f"Invalid suite membership in {identity}")
        unknown = set(item) - set(defaults) - {"id", "suites", "covers"}
        if unknown:
            raise ValueError(f"Unknown case fields in {identity}: {sorted(unknown)}")
        if args.suite == "all" or args.suite in item["suites"]:
            selected.append({**defaults, **{k: v for k, v in item.items() if k != "suites"}})
    if args.case_id:
        unknown = set(args.case_id) - {case["id"] for case in selected}
        if unknown:
            raise ValueError(f"Case IDs absent from suite {args.suite}: {sorted(unknown)}")
        selected = [case for case in selected if case["id"] in args.case_id]
    if not selected:
        raise ValueError(f"No cases in suite {args.suite}")
    if args.skip_mxfp8 and not include_skipped:
        selected = [
            case for case in selected if resolved_save_mode(case["save_gmm1_output"], case["swiglu_limit"]) != "mxfp8"
        ]
        if not selected:
            raise ValueError(
                "No cases remain after --skip-mxfp8; choose a suite/case containing BF16 or recompute cases"
            )
    return selected


def launch_options(args):
    argv = []
    for key in ("chunk_rows", "warmup", "loops", "timeout", "skip", "local_mem_size", "baseline", "log"):
        value = getattr(args, key)
        if value is not None:
            argv += ["--" + key.replace("_", "-"), str(value)]
    if args.atk:
        argv += ["--atk"]
    if args.diagnose_mxfp8:
        argv += ["--diagnose-mxfp8"]
    if args.atk_ratios is not None:
        argv += ["--atk-ratios", *map(str, args.atk_ratios)]
    return argv


def case_argv(args, case, index, output):
    endpoint = urlsplit(args.shmem_ip_port)
    argv = [
        ",".join(map(str, args.devices)),
        "--shmem-ip-port",
        f"tcp://{endpoint.hostname}:{endpoint.port + 4 * index}",
    ]
    for key, value in case.items():
        if key not in ("id", "covers"):
            argv += ["--" + key.replace("_", "-"), str(value)]
    argv += launch_options(args)
    return argv + ["--output", str(output)]


def stream_job(command, log_path, on_line=None):
    with (
        log_path.open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=os.name == "posix",
        ) as process,
    ):
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                visible = on_line(line) if on_line is not None else True
                if visible:
                    print(line, end="", flush=True)
        except BaseException:
            # Only this Popen-created job group is terminated, never other jobs.
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            raise
    return process.returncode


def read_case_result(args, report_file, result_code=None):
    result = json.loads(report_file.read_text(encoding="utf-8"))
    ranks = result["ranks"]
    if {rank["rank"] for rank in ranks} != set(range(len(args.devices))) or len(ranks) != len(args.devices):
        raise RuntimeError(f"Incomplete rank coverage in {report_file}")
    if result_code is None:
        result_code = 0 if result["passed"] else 2
    if result["passed"] != all(rank["passed"] for rank in ranks) or result["passed"] != (result_code == 0):
        raise RuntimeError(f"Inconsistent report/result code in {report_file}")
    if args.atk:
        from precision_metrics import OUTPUTS

        if not result.get("atk_requested") or not result["precision_run"]:
            raise RuntimeError(f"ATK precision did not run in {report_file}")
        for rank in ranks:
            check = rank.get("precision", {}).get("atk", {})
            outputs = check.get("outputs", {})
            if (
                check.get("provider") != "private"
                or check.get("standard") != "cv_fused_double_benchmark"
                or set(outputs) != set(OUTPUTS)
                or check.get("passed") != all(value.get("passed", False) for value in outputs.values())
                or (rank["passed"] and not check.get("passed"))
            ):
                raise RuntimeError(f"Incomplete/inconsistent ATK results in {report_file}, rank={rank['rank']}")
    return {
        "status": "PASS" if result["passed"] else "FAIL",
        "report": str(report_file),
        "precision_run": result["precision_run"],
        "performance_run": result["performance_run"],
        "result_code": result_code,
    }


def case_progress(index, count, record, log_path):
    case = record["config"]
    print(
        f"[CASE] {index + 1}/{count} {record['id']} "
        f"M={case['seq']} H={case['hidden']} F={case['ffn']} TopK={case['topk']} "
        f"E/rank={case['expert_per_rank']} save={case['save_gmm1_output']} data={case['data_profile']} "
        f"clip={case['swiglu_limit']} p_scale={case['routed_scaling_factor']} "
        f"p={case['topk_weight_dtype']}/{case['topk_weight_values']} "
        f"log={log_path}",
        flush=True,
    )


def run_persistent_suite(args, cases, output, summary, save):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        ",".join(map(str, args.devices)),
        "--suite",
        args.suite,
        "--suite-worker",
        "--output",
        str(output),
        "--shmem-ip-port",
        args.shmem_ip_port,
        *launch_options(args),
    ]
    if args.skip_mxfp8:
        command += ["--skip-mxfp8"]
    for case_id in args.case_id or ():
        command += ["--case-id", case_id]
    active, case_log, completed, case_started = None, None, 0, None

    def line_received(line):
        nonlocal active, case_log, completed, case_started
        begin = re.fullmatch(r"\[CASE_BEGIN\] ([a-z0-9_]+)", line.strip())
        end = re.fullmatch(r"\[CASE_END\] ([a-z0-9_]+)", line.strip())
        if begin:
            if active is not None or completed >= len(cases) or begin[1] != cases[completed]["id"]:
                raise RuntimeError("Unexpected case-begin marker from persistent workers")
            active = summary["cases"][completed]
            active["status"] = "RUNNING"
            case_started = time.perf_counter()
            case_log = (output / (active["id"] + ".log")).open("w", encoding="utf-8")
            case_progress(completed, len(cases), active, Path(case_log.name))
            save()
        if case_log is not None:
            case_log.write(line)
            case_log.flush()
        if end:
            if active is None or end[1] != active["id"]:
                raise RuntimeError("Unexpected case-end marker from persistent workers")
            case_log.close()
            case_log = None
            active["elapsed_seconds"] = time.perf_counter() - case_started
            active.update(read_case_result(args, output / active["id"] / "summary.json"))
            save()
            print(
                f"[CASE] {active['id']} {active['status']} elapsed={active['elapsed_seconds']:.2f}s "
                f"suite_elapsed={summary['elapsed_seconds']:.2f}s",
                flush=True,
            )
            completed += 1
            active = None
        return begin is None and end is None

    try:
        exit_code = stream_job(command, output / "suite.log", line_received)
        summary["worker_exit_code"] = exit_code
        if exit_code != 0 or completed != len(cases):
            raise RuntimeError(f"Persistent workers exited with code {exit_code}; completed {completed}/{len(cases)}")
        runtime_file = output / "suite_runtime.json"
        runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
        ranks = runtime["ranks"]
        if len(ranks) != len(args.devices) or {rank["rank"] for rank in ranks} != set(range(len(args.devices))):
            raise RuntimeError("Incomplete persistent runtime cleanup report")
        for rank in ranks:
            for name in ("operator_init", "operator_close"):
                elapsed = rank["phase_seconds"].get(name)
                if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                    raise RuntimeError(f"Missing/invalid {name} timing for rank {rank['rank']}")
        summary["runtime_report"] = str(runtime_file)
        summary["shmem_bytes"] = runtime["shmem_bytes"]
        return False
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["status"] = "ERROR"
        if active is not None:
            active.update(status="ERROR", error=summary["error"], elapsed_seconds=time.perf_counter() - case_started)
        elif completed < len(cases):
            summary["cases"][completed].update(status="ERROR", error=summary["error"])
        save()
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        print(f"[SUITE ERROR] {summary['error']}", flush=True)
        return True
    finally:
        if case_log is not None:
            case_log.close()


def suite_performance_summary(cases, args):
    """One row per case: the run's at-a-glance table (rank-0 medians).

    Columns match the reviewed layout: shape, three latencies, speedup,
    MFU and roofline floor/overhead for the backward and complete scopes.
    """
    rows = []
    for record in cases:
        if record.get("status") != "PASS" or not record.get("performance_run") or not record.get("report"):
            continue
        try:
            result = json.loads(Path(record["report"]).read_text(encoding="utf-8"))
            perf = result["ranks"][0]["performance"]
        except (OSError, KeyError, IndexError, ValueError):
            continue
        config = record["config"]
        speeds = perf.get("baseline", {}).get("speedup") or {}
        mfu = perf.get("mfu", {})
        roofline = perf.get("roofline", {})
        speed_cell = (
            f"{speeds['forward']:.2f}/{speeds['backward']:.2f}/{speeds['complete']:.2f}" if speeds else "--/--/--"
        )
        mfu_cell = "/".join(f"{mfu[scope]['mfu']:.0f}" for scope in ("forward", "backward", "complete") if scope in mfu)
        floor_cell = "/".join(
            f"{roofline[scope]['floor_ms']:.2f}" for scope in ("forward", "backward", "complete") if scope in roofline
        )
        overhead = roofline.get("complete", {}).get("overhead_ms")
        rows.append(
            f"[SUMMARY] {record['id']:<14}"
            f"{config['seq']:>5}{config['hidden']:>6}{config['ffn']:>6}{config['topk']:>4}"
            f"{config['expert_per_rank'] * len(args.devices):>5}"
            f"{perf['forward_ms']['median']:>9.2f}"
            f"{perf['backward_ms']['median']:>9.2f}"
            f"{perf['complete_ms']['median']:>9.2f}  "
            f"{speed_cell:<15} "
            f"{mfu_cell:<12}"
            f"{floor_cell:<20}" + (f"{overhead:>+12.2f}" if overhead is not None else f"{'--':>12}")
        )
    if not rows:
        return []
    header = (
        f"[SUMMARY] {len(rows)} shapes @ EP{len(args.devices)} save={args.save_gmm1_output} "
        f"warmup={args.warmup} loops={args.loops} (rank 0 median)"
    )
    columns = (
        f"[SUMMARY] {'case':<14}{'M':>5}{'H':>6}{'F':>6}{'tk':>4}{'E':>5}"
        f"{'fwd_ms':>9}{'bwd_ms':>9}{'fb_ms':>9}  {'speedup f/b/fb':<15} "
        f"{'MFU% f/b/fb':<12}{'floor f/b/fb':<20}{'fb_overhead':>12}"
    )
    return [header, columns, *rows]


def run_suite(args):
    requested_cases = suite_cases(args, include_skipped=True)
    cases = suite_cases(args)
    selected_ids = {case["id"] for case in cases}
    skipped = [
        {
            "id": case["id"],
            "status": "SKIPPED",
            "reason": "--skip-mxfp8",
            "requested_save_mode": case["save_gmm1_output"],
            "resolved_save_mode": resolved_save_mode(case["save_gmm1_output"], case["swiglu_limit"]),
        }
        for case in requested_cases
        if case["id"] not in selected_ids
    ]
    output = args.output or Path(__file__).resolve().parent / "output" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.suite + "_" + uuid4().hex[:8]
    )
    commands = [
        case_argv(args, case, index if args.isolate_cases else 0, output / case["id"])
        for index, case in enumerate(cases)
    ]
    for command in commands:
        parse_args(command)  # Validate every case before creating directories or starting any job.
    if args.skip_mxfp8 and not args.suite_worker:
        print(
            f"[SELECTION] requested={len(requested_cases)} selected={len(cases)} skipped_mxfp8={len(skipped)} "
            f"ids={','.join(case['id'] for case in skipped) or 'none'}",
            flush=True,
        )
    if args.list_cases:
        print("id\tEP\tM/rank\tH\tF\tTopK\tE/rank\trouting\tdata\tsave\tclip\tp_scale\tp_dtype\tp_values\tseed\tcovers")
        for case in cases:
            print(
                "\t".join(
                    map(
                        str,
                        (
                            case["id"],
                            len(args.devices),
                            case["seq"],
                            case["hidden"],
                            case["ffn"],
                            case["topk"],
                            case["expert_per_rank"],
                            case["routing_mode"],
                            case["data_profile"],
                            case["save_gmm1_output"],
                            case["swiglu_limit"],
                            case["routed_scaling_factor"],
                            case["topk_weight_dtype"],
                            case["topk_weight_values"],
                            case["seed"],
                            case["covers"],
                        ),
                    )
                )
            )
        print(f"{len(cases)} cases; list only, no imports of Torch/ATK and no device execution")
        return 0
    if args.suite_worker:
        installed_root()
        import torch.multiprocessing as mp

        mp.spawn(run_suite_rank, args=(args, cases, output), nprocs=len(args.devices), join=True)
        return 0
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.devices))
    suite_started = time.perf_counter()
    mode = "isolated" if args.isolate_cases else "persistent"
    print(f"[SUITE] mode={mode} EP={len(args.devices)} cases={len(cases)} output={output}", flush=True)
    installed_root()
    if args.atk:
        from atk_compare import require_atk

        require_atk()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "suite": args.suite,
        "execution_mode": mode,
        "status": "RUNNING",
        "devices": args.devices,
        "world_size": len(args.devices),
        "atk_requested": args.atk,
        "skip": args.skip,
        "skip_mxfp8": args.skip_mxfp8,
        "requested_case_count": len(requested_cases),
        "selected_case_count": len(cases),
        "skipped_cases": skipped,
        "passed": False,
        "cases": [{"id": case["id"], "config": case, "status": "NOT_RUN"} for case in cases],
    }
    summary_file = output / "suite_summary.json"

    def save():
        summary["elapsed_seconds"] = time.perf_counter() - suite_started
        summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save()
    if not args.isolate_cases:
        runtime_error = run_persistent_suite(args, cases, output, summary, save)
    else:
        runtime_error = False
        for index, (record, command) in enumerate(zip(summary["cases"], commands, strict=True)):
            case_started = time.perf_counter()
            case_progress(index, len(cases), record, output / (record["id"] + ".log"))
            try:
                exit_code = stream_job(
                    [sys.executable, "-u", str(Path(__file__).resolve()), *command], output / (record["id"] + ".log")
                )
                record["elapsed_seconds"] = time.perf_counter() - case_started
                record["exit_code"] = exit_code
                report_file = output / record["id"] / "summary.json"
                if exit_code in (0, 2) and report_file.is_file():
                    record.update(read_case_result(args, report_file, exit_code))
                else:
                    record["status"] = "ERROR"
                    runtime_error = True
            except BaseException as error:
                summary["status"] = "ERROR"
                record.update(
                    status="ERROR",
                    error=f"{type(error).__name__}: {error}",
                    elapsed_seconds=time.perf_counter() - case_started,
                )
                save()
                raise
            save()
            print(
                f"[CASE] {record['id']} {record['status']} elapsed={record['elapsed_seconds']:.2f}s "
                f"suite_elapsed={summary['elapsed_seconds']:.2f}s log={output / (record['id'] + '.log')}",
                flush=True,
            )
            if runtime_error:
                break  # Device/communication/setup errors stop the suite; remaining cases stay NOT_RUN.
    summary["passed"] = not runtime_error and all(record["status"] == "PASS" for record in summary["cases"])
    summary["status"] = "ERROR" if runtime_error else "PASS" if summary["passed"] else "FAIL"
    save()
    print(
        f"[SUITE] {summary['status']} elapsed={summary['elapsed_seconds']:.2f}s "
        f"selected={len(cases)} skipped={len(skipped)} report={summary_file}",
        flush=True,
    )
    for line in suite_performance_summary(summary["cases"], args):
        print(line, flush=True)
    return 1 if runtime_error else 0 if summary["passed"] else 2


_IDENTITY = None


def runtime_identity():
    """Best-effort identity of the installed binary, cached for the process.

    The sha256 of each loaded extension is authoritative; the git revision is
    only a label and stays null when the wheel does not record one.
    """
    global _IDENTITY
    if _IDENTITY is not None:
        return _IDENTITY
    import hashlib
    import importlib.util
    from importlib.metadata import PackageNotFoundError, version

    def digest(path):
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return None

    identity = {"revision": None, "wheel": {}, "libraries": {}}
    try:
        identity["wheel"] = {"name": "ascend-deepep", "version": version("ascend-deepep")}
    except PackageNotFoundError:
        pass
    for label, name in (
        ("deep_ep._C", "deep_ep._C"),
        ("megamoe_train._C", "deep_ep_experimental.megamoe_train._C"),
    ):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.origin:
            continue
        origin = Path(spec.origin)
        identity["libraries"][label] = {"path": str(origin), "sha256": digest(origin)}
        if label == "megamoe_train._C":
            kernel = origin.with_name("libdeepep_megamoe_train_bf16.so")
            if kernel.is_file():
                identity["libraries"]["megamoe_train_kernel"] = {
                    "path": str(kernel),
                    "sha256": digest(kernel),
                }
    # A full 40-hex revision only; never guess or truncate.
    for candidate in (os.environ.get("DEEPEP_BUILD_REVISION"),):
        value = (candidate or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", value):
            identity["revision"] = value
    if identity["revision"] is None:
        try:
            info = json.loads((Path(sys.modules["deep_ep"].__file__).parent / "_build_info.json").read_text())
        except (KeyError, OSError, ValueError):
            info = {}
        for key in ("revision", "source_revision", "git_revision", "commit"):
            value = str(info.get(key, "")).strip().lower()
            if re.fullmatch(r"[0-9a-f]{40}", value):
                identity["revision"] = value
                break
    _IDENTITY = identity
    return identity


def installed_root():
    locations = list(dict.fromkeys(sysconfig.get_path(name) for name in ("purelib", "platlib")))
    spec = PathFinder.find_spec("deep_ep", locations)
    hint = "Build with DEEPEP_BUILD_MEGAMOE_TRAIN=ON and install its wheel in this Python environment."
    if spec is None or spec.origin is None:
        raise RuntimeError(f"No installed deep_ep package for {sys.executable}. {hint}")
    root = Path(spec.origin).resolve().parent.parent
    for name in ("deep_ep", "deep_ep_experimental", "deep_ep_experimental.megamoe_train"):
        directory = root.joinpath(*name.split("."))
        if not (directory / "__init__.py").is_file():
            raise RuntimeError(f"Missing installed package {directory}. {hint}")
        if name != "deep_ep_experimental" and not any(
            (directory / ("_C" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES
        ):
            raise RuntimeError(f"Missing installed native extension {name}._C. {hint}")
    for name, module in tuple(sys.modules.items()):
        top = name.partition(".")[0]
        if top in ("deep_ep", "deep_ep_experimental"):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root / top):
                raise RuntimeError(
                    f"{name} was already loaded from {origin}; expected installed wheel at {root}. Restart the process."
                )
    if not sys.path or Path(sys.path[0]).resolve() != root:
        sys.path.insert(0, str(root))
    return root


def make_inputs(args, rank, world, global_tokens=True):
    """Generate shared CPU snapshots; preserve the selected route-weight dtype."""
    import torch

    def rng(source, stream):
        seed = (args.seed + source * 1000003 + stream * 1000000007) % (1 << 63)
        return torch.Generator(device="cpu").manual_seed(seed)

    def sample(generator, shape, bound):
        value = torch.empty(shape, dtype=torch.float32, device="cpu")
        if args.data_profile == "normal":
            return value.normal_(mean=0.0, std=bound / 3, generator=generator).to(torch.bfloat16)
        return value.uniform_(-bound, bound, generator=generator).to(torch.bfloat16)

    xs, probabilities, upstreams, all_ids = [], [], [], []
    total = world * args.expert_per_rank
    for source in range(world):
        if args.routing_mode == "balanced":
            ids = (
                ((torch.arange(args.seq * args.topk, device="cpu") + source * args.seq * args.topk) % total)
                .reshape(args.seq, args.topk)
                .to(torch.int32)
            )
        elif args.routing_mode in ("hot_expert", "hot_rank", "local_only", "remote_only"):
            ids = (torch.arange(args.seq)[:, None] + torch.arange(args.topk)[None, :]) % args.expert_per_rank
            if args.routing_mode == "hot_expert":
                ids.zero_()
            elif args.routing_mode == "local_only":
                ids += source * args.expert_per_rank
            elif args.routing_mode == "remote_only":
                ids += ((source + 1) % world) * args.expert_per_rank
            ids = ids.to(torch.int32)
        elif args.routing_mode == "ragged":
            # Send to the next rank with exact per-expert row counts, including
            # an empty expert, one row, and both sides of the 128-row tile.
            rows = torch.tensor([0, 1, 127, 128, 129, args.seq - 385] + [0] * (args.expert_per_rank - 6))
            ids = torch.repeat_interleave(torch.arange(args.expert_per_rank), rows).reshape(-1, 1)
            ids = (ids + ((source + 1) % world) * args.expert_per_rank).to(torch.int32)
        else:
            ids = torch.empty((args.seq, args.topk), dtype=torch.int32, device="cpu")
            generator = rng(source, 1)
            skew = torch.zeros(total, dtype=torch.float32, device="cpu")
            if args.routing_mode == "imbalanced":
                skew = -2.0 * torch.arange(total, dtype=torch.float32, device="cpu") / max(total - 1, 1)
            for begin in range(0, args.seq, 256):
                stop = min(begin + 256, args.seq)
                uniform_scores = torch.rand(
                    (stop - begin, total), dtype=torch.float32, device="cpu", generator=generator
                ).clamp_min(torch.finfo(torch.float32).tiny)
                scores = -torch.log(-torch.log(uniform_scores)) + skew
                ids[begin:stop] = scores.topk(args.topk, dim=1).indices.to(torch.int32)
        all_ids.append(ids)
        if global_tokens or source == rank:
            xs.append(sample(rng(source, 0), (args.seq, args.hidden), 0.1))
            logits = torch.empty((args.seq, args.topk), dtype=torch.float32, device="cpu")
            logits.uniform_(-1, 1, generator=rng(source, 2))
            probabilities.append(torch.softmax(logits, dim=1) * args.routed_scaling_factor)
            upstreams.append(sample(rng(source, 3), (args.seq, args.hidden), 0.1))
    ids = torch.cat(all_ids)
    counts = torch.bincount(ids.reshape(-1).long(), minlength=total)
    w1 = sample(rng(rank, 4), (args.expert_per_rank, 2 * args.ffn, args.hidden), 1.0)
    w2 = sample(rng(rank, 5), (args.expert_per_rank, args.hidden, args.ffn), 1.0)
    inputs = {
        "x": torch.cat(xs),
        "ids": ids if global_tokens else all_ids[rank],
        "p": torch.cat(probabilities),
        "w1": w1,
        "w2": w2,
        "dy": torch.cat(upstreams),
        "all_counts": counts,
        "source_to_destination_rows": torch.stack(
            [torch.bincount(routes.long().reshape(-1) // args.expert_per_rank, minlength=world) for routes in all_ids]
        ),
    }
    apply_data_profile(inputs, args, token_start=0 if global_tokens else rank * args.seq)
    # Convert once, after profiles. Rounded FP32 and BF16 runs have identical
    # values and RNG streams; full FP32 retains the router's additional bits.
    if args.topk_weight_values == "bf16-rounded":
        inputs["p"] = inputs["p"].to(torch.bfloat16)
    if args.topk_weight_dtype == "fp32":
        inputs["p"] = inputs["p"].float()
    return inputs


def apply_data_profile(inputs, args, token_start=0):
    """Alter only the shared input snapshot, never a reference result."""
    import torch

    profile = args.data_profile
    if profile.startswith("zero_"):
        inputs[profile.removeprefix("zero_")].zero_()
    elif profile == "near_zero":
        inputs["x"].mul_(2**-10)
        inputs["dy"].mul_(2**-10)
    elif profile == "cancellation":
        # Equal adjacent Gate/Up channels and opposite W2 columns cancel
        # output contributions while retaining a nontrivial weight gradient.
        inputs["w1"][:, 1::2, :] = inputs["w1"][:, ::2, :].clone()
        inputs["w2"][:, :, 1::2] = -inputs["w2"][:, :, ::2]
    elif profile == "outliers":
        inputs["x"][:, 0] = 8.0
        inputs["x"][:, 1] = -8.0
    elif profile == "equal_weights":
        inputs["p"].fill_(args.routed_scaling_factor / args.topk)
    elif profile == "one_hot_weights":
        slots = (torch.arange(len(inputs["p"])) + token_start).remainder(args.topk).reshape(-1, 1)
        inputs["p"].zero_().scatter_(1, slots, args.routed_scaling_factor)
    elif profile == "clip_boundary":
        limit = torch.tensor(args.swiglu_limit, dtype=torch.bfloat16)
        below = torch.nextafter(limit, torch.zeros_like(limit))
        above = torch.nextafter(limit, torch.full_like(limit, math.inf))
        values = torch.stack((below, limit, above, -below, -limit, -above))
        # x=[1,0,...] makes GMM1 exactly select the first weight column.
        # Cartesian pairs cover Gate's upper boundary and both Up boundaries.
        columns = torch.arange(args.ffn)
        gate = values[columns.remainder(6)]
        up = values[columns.div(6, rounding_mode="floor").remainder(6)]
        inputs["x"].zero_()
        inputs["x"][:, 0] = 1.0
        inputs["w1"].zero_()
        inputs["w1"][:, :, 0] = torch.cat((gate, up))


def combine_baseline(partial, args, rank, world, device):
    """Move route values through Gloo, then apply the selected NPU combine."""
    import torch
    import torch.distributed as dist

    from npu_baseline import unpermute_routes

    m, h = args.seq, args.hidden
    for destination in range(world):
        # Every row/slot has one owning expert. FP32 carries BF16 y/dx and
        # either BF16 or FP32 dp exactly; no route contributions are combined.
        buffer = torch.zeros((m, args.topk, 2 * h + 1), dtype=torch.float32, device="cpu")
        for route in partial["routes"]:
            mask = route["rows"] // m == destination
            rows, slots = route["rows"][mask] % m, route["slots"][mask]
            buffer[rows, slots, :h] = route["y"][mask].float()
            buffer[rows, slots, h : 2 * h] = route["dx"][mask].float()
            buffer[rows, slots, -1] = route["dp"][mask].float()
        dist.reduce(buffer, dst=destination)
        if rank == destination:
            ids = partial["ids"][rank * m : (rank + 1) * m]
            result = unpermute_routes(
                buffer, ids, device, reference=partial["reference"], topk_weight_dtype=partial["topk_weight_dtype"]
            )
    result.update(dw1=partial["dw1"], dw2=partial["dw2"])
    return result


def local_truth(inputs, args, rank):
    import torch.distributed as dist

    from cpu_reference import run_reference

    result = run_reference(inputs, rank * args.expert_per_rank, args.chunk_rows, args.swiglu_limit)
    sl = slice(rank * args.seq, (rank + 1) * args.seq)
    for name in ("y", "dx", "dtopk_weights"):
        dist.all_reduce(result[name])
        result[name] = result[name][sl].clone()
    return result


# ---------------------------------------------------------------------------
# Performance protocol and the small-operator baseline.
#
# Ported from ascend_deepep_hkq@f575fc1 test/bigop_fused: test_speed.py owns the
# three measurement scopes and the per-iteration synchronization, timing.py owns
# the statistics, golden.py owns the dropless small-operator chain that is timed
# here as the baseline. The source tree removed its kernel Timer
# instrumentation, so nothing here builds, enables or reads a device Timer: every
# sample is one synchronized NPU Event window.
#
# The source times its `BigOp.apply(x, moe_layer)` chain, which runs a router
# before the dispatch. MegaMoE BF16 takes routing ids/weights directly and this
# package has no router, so both paths here start at permute1: the router cost
# that is common to both sides in the source stays outside this entry.
# ---------------------------------------------------------------------------

PEAK_TFLOPS = 432.0  # source comment: Ascend950 BF16 cube peak
ROOFLINE_TFLOPS = round(PEAK_TFLOPS * 0.95, 1)  # 410.4: the source's derated roofline
FORWARD_RFK = 6.0  # GMM1 (4) + GMM2 (2) per routed row
SAVED_MODE_RFK = {"none": 16.0, "bf16": 12.0, "mxfp8": 12.0, "auto": 12.0}
BASELINE_BUDGET_BYTES = 16 << 30  # packed-route working set the baseline may allocate


def event_stats(samples):
    """Sample list plus the source's mean/median/min/max summary."""
    return {
        "samples": list(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "min": min(samples),
        "max": max(samples),
    }


def bench_scope(body, prepare, args):
    """warmup+loops Event windows, synchronized before and after every window.

    The source synchronizes each iteration and rebuilds a forward graph for
    every backward sample, so queued SHMEM calls never share one workspace.
    """
    import torch

    samples = []
    for iteration in range(args.warmup + args.loops):
        torch.npu.synchronize()
        state = prepare() if prepare is not None else None
        if prepare is not None:
            torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        body(state)
        end.record()
        torch.npu.synchronize()
        if iteration >= args.warmup:
            samples.append(float(start.elapsed_time(end)))
    return samples


def exchanged_rows(group, value, receive_splits, send_splits):
    """Blocking all_to_all_single with Megatron's split orientation.

    The source keeps an asynchronous handle only to overlap a shared expert this
    entry does not have; the blocking form moves the same bytes.
    """
    import torch.distributed as dist

    if dist.get_world_size(group) == 1:
        return value
    value = value.contiguous()
    output = value.new_empty([sum(receive_splits), *value.shape[1:]])
    dist.all_to_all_single(output, value, output_split_sizes=receive_splits, input_split_sizes=send_splits, group=group)
    return output


def reorder_chunks(value, sizes, order, local_experts=None):
    """Megatron sort_chunks_by_idxs; zero-row chunks keep empty experts valid.

    The source skips permute2/unpermute1 entirely with one local expert, where
    the reorder is an identity copy; keep that so neither side pays for it.
    """
    import torch

    if local_experts == 1:
        return value
    chunks = torch.split(value, sizes, dim=0)
    return torch.cat([chunks[index] for index in order], dim=0)


def clip_gate_up(gate, up, limit):
    """Source clip semantics: Gate one-sided, Up two-sided, disabled for limit<=0."""
    import torch

    if not 0 < limit < torch.finfo(torch.bfloat16).max:
        return gate, up
    return gate.clamp_max(limit), up.clamp(-limit, limit)


class BaselinePlan:
    """Row order, exchange splits, expert counts and weight views of one case."""

    __slots__ = (
        "group",
        "tokens",
        "topk",
        "hidden",
        "local_experts",
        "swiglu_limit",
        "order",
        "route_row",
        "counts",
        "input_splits",
        "output_splits",
        "received_sizes",
        "return_sizes",
        "sort_index",
        "restore_index",
        "weight1",
        "weight2",
        "recompute",
    )


def baseline_plan(ids, w1, w2, group, args, rank, world, mode):
    """Derive the baseline chain metadata from the shared routing ids.

    Returns ``(plan, None)`` or ``(None, reason)``, so a shape the baseline
    cannot afford prints why it is absent instead of failing the case. Every
    value here is built once, outside all Event windows.
    """
    import torch
    import torch.distributed as dist

    local_experts = args.expert_per_rank
    tokens, topk = (int(size) for size in ids.shape)
    routes = tokens * topk
    working_set = routes * (args.hidden + args.ffn) * 2 * 6
    if working_set > BASELINE_BUDGET_BYTES:
        return None, (
            f"packed routes need ~{working_set / 2**30:.1f} GiB, above the "
            f"{BASELINE_BUDGET_BYTES >> 30} GiB baseline budget"
        )
    flat = ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    experts = world * local_experts
    gathered = [None] * world
    dist.all_gather_object(gathered, torch.bincount(flat, minlength=experts).tolist())
    # Rows rank `source` produces for this rank, split by local expert.
    sent_to_us = torch.tensor(gathered, dtype=torch.int64)[:, rank * local_experts : (rank + 1) * local_experts]
    order = torch.argsort(flat, stable=True)
    plan = BaselinePlan()
    plan.group = group
    plan.tokens, plan.topk, plan.hidden = tokens, topk, args.hidden
    plan.local_experts, plan.swiglu_limit, plan.recompute = local_experts, args.swiglu_limit, mode == "none"
    plan.order = order.to(ids.device)
    plan.route_row = (order // topk).to(ids.device)
    plan.counts = sent_to_us.sum(dim=0).to(ids.device)
    plan.input_splits = torch.bincount(flat // local_experts, minlength=world).tolist()
    plan.output_splits = sent_to_us.sum(dim=1).tolist()
    plan.received_sizes = sent_to_us.reshape(-1).tolist()
    plan.return_sizes = sent_to_us.T.reshape(-1).tolist()
    plan.sort_index = torch.arange(experts).reshape(world, local_experts).T.reshape(-1).tolist()
    plan.restore_index = torch.arange(experts).reshape(local_experts, world).T.reshape(-1).tolist()
    # Megatron stores [H, 2F*E] and [F*E, H]; the operator takes per-expert
    # tensors, so read both once here rather than copying inside a window.
    plan.weight1 = w1.detach().transpose(1, 2).contiguous()
    plan.weight2 = w2.detach().transpose(1, 2).contiguous()
    if sum(plan.received_sizes) != sum(plan.output_splits) or int(plan.counts.sum()) != sum(plan.received_sizes):
        return None, "inconsistent dispatch splits"
    return plan, None


_BASELINE_CHAIN = None
_HCCL = {}


def device_group():
    """One reusable HCCL subgroup for the baseline; the default group is Gloo.

    MegaMoE BF16 communicates inside its own runtime, so the baseline
    chain needs a Torch-level device collective. Creation is cached: a suite
    must not accumulate one communicator per case.
    """
    import torch.distributed as dist

    if "group" not in _HCCL:
        try:
            _HCCL["group"] = dist.new_group(ranks=list(range(dist.get_world_size())), backend="hccl")
            _HCCL["reason"] = None
        except Exception as error:  # capability probe; the reason is reported, not hidden
            _HCCL["group"] = None
            _HCCL["reason"] = f"no HCCL subgroup for all_to_all ({type(error).__name__}: {error})"
    return _HCCL["group"], _HCCL["reason"]


def baseline_chain():
    """Build the baseline autograd Function once; keeps module import Torch-free."""
    global _BASELINE_CHAIN
    if _BASELINE_CHAIN is not None:
        return _BASELINE_CHAIN

    import torch

    class BaselineChain(torch.autograd.Function):
        """The source's dropless small-operator chain with its hand-written backward.

        permute1 -> dispatch -> permute2 -> GMM1 -> SwiGLU*probs -> GMM2 ->
        unpermute1 -> combine -> unpermute2. ``recompute`` mirrors the source's
        coupled regime: MegaMoE BF16 saves Gate/Up for bf16/mxfp8/auto and
        recomputes GMM1 for none, so the baseline performs the same GEMM work
        (12 versus 16 RFK).
        """

        @staticmethod
        def forward(ctx, x, topk_weights, w1, w2, plan):
            from npu_baseline import _grouped_matmul

            counts = plan.counts
            global_x = reorder_chunks(
                exchanged_rows(plan.group, x.index_select(0, plan.route_row), plan.output_splits, plan.input_splits),
                plan.received_sizes,
                plan.sort_index,
                plan.local_experts,
            )
            global_p = reorder_chunks(
                exchanged_rows(
                    plan.group,
                    topk_weights.reshape(-1).index_select(0, plan.order),
                    plan.output_splits,
                    plan.input_splits,
                ),
                plan.received_sizes,
                plan.sort_index,
                plan.local_experts,
            )
            gate_up = _grouped_matmul(global_x, plan.weight1, counts)
            gate, up = clip_gate_up(*torch.chunk(gate_up, 2, dim=-1), plan.swiglu_limit)
            activation = torch.nn.functional.silu(gate) * up
            if global_p.dtype == torch.float32:
                # FP32 p must not promote the following BF16 GMM operands.
                intermediate = (activation.float() * global_p.unsqueeze(-1)).to(activation.dtype)
            else:
                intermediate = activation * global_p.unsqueeze(-1)
            packed_y = _grouped_matmul(intermediate, plan.weight2, counts)
            combined = exchanged_rows(
                plan.group,
                reorder_chunks(packed_y, plan.return_sizes, plan.restore_index, plan.local_experts),
                plan.input_splits,
                plan.output_splits,
            )
            output = torch.zeros((plan.tokens, plan.hidden), dtype=combined.dtype, device=combined.device)
            output.scatter_add_(0, plan.route_row.unsqueeze(1).expand(-1, plan.hidden), combined)
            ctx.plan = plan
            saved = [global_x, global_p] if plan.recompute else [global_x, global_p, gate_up]
            ctx.save_for_backward(*saved)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            from npu_baseline import _grouped_matmul, _grouped_wgrad

            plan = ctx.plan
            counts = plan.counts
            global_x, global_p = ctx.saved_tensors[0], ctx.saved_tensors[1]
            weight1_transposed = plan.weight1.transpose(-1, -2).contiguous()
            weight2_transposed = plan.weight2.transpose(-1, -2).contiguous()
            grad_packed_y = reorder_chunks(
                exchanged_rows(
                    plan.group,
                    grad_output.index_select(0, plan.route_row),
                    plan.output_splits,
                    plan.input_splits,
                ),
                plan.received_sizes,
                plan.sort_index,
                plan.local_experts,
            )
            gate_up = _grouped_matmul(global_x, plan.weight1, counts) if plan.recompute else ctx.saved_tensors[2]
            gate, up = clip_gate_up(*torch.chunk(gate_up, 2, dim=-1), plan.swiglu_limit)
            activation = torch.nn.functional.silu(gate) * up
            if global_p.dtype == torch.float32:
                intermediate = (activation.float() * global_p.unsqueeze(-1)).to(activation.dtype)
            else:
                intermediate = activation * global_p.unsqueeze(-1)
            grad_intermediate = _grouped_matmul(grad_packed_y, weight2_transposed, counts)
            grad_w2 = _grouped_wgrad(intermediate, grad_packed_y, counts)
            grad_w2 = grad_w2.reshape(plan.weight2.shape).transpose(1, 2).contiguous()
            grad_activation = grad_intermediate * global_p.unsqueeze(-1)
            if global_p.dtype == torch.float32:
                grad_p = (grad_intermediate.float() * activation.float()).sum(dim=-1)
            else:
                grad_p = (grad_intermediate * activation).sum(dim=-1)
            # The source differentiates the same PyTorch graph it ran in forward
            # instead of substituting a different activation backward.
            with torch.enable_grad():
                replay = gate_up.detach().requires_grad_(True)
                replay_gate, replay_up = clip_gate_up(*torch.chunk(replay, 2, dim=-1), plan.swiglu_limit)
                (grad_gate_up,) = torch.autograd.grad(
                    torch.nn.functional.silu(replay_gate) * replay_up, replay, grad_activation
                )
            grad_x = _grouped_matmul(grad_gate_up, weight1_transposed, counts)
            grad_w1 = _grouped_wgrad(global_x, grad_gate_up, counts)
            grad_w1 = grad_w1.reshape(plan.weight1.shape).transpose(1, 2).contiguous()
            grad_x = reorder_chunks(grad_x, plan.return_sizes, plan.restore_index, plan.local_experts)
            grad_p = reorder_chunks(grad_p, plan.return_sizes, plan.restore_index, plan.local_experts)
            grad_x = exchanged_rows(plan.group, grad_x, plan.input_splits, plan.output_splits)
            grad_p = exchanged_rows(plan.group, grad_p, plan.input_splits, plan.output_splits)
            grad_input = torch.zeros((plan.tokens, plan.hidden), dtype=grad_x.dtype, device=grad_x.device)
            grad_input.scatter_add_(0, plan.route_row.unsqueeze(1).expand(-1, plan.hidden), grad_x)
            grad_weights = torch.zeros(plan.tokens * plan.topk, dtype=grad_p.dtype, device=grad_p.device)
            grad_weights.index_copy_(0, plan.order, grad_p)
            return grad_input, grad_weights.reshape(plan.tokens, plan.topk), grad_w1, grad_w2, None

    _BASELINE_CHAIN = BaselineChain
    return _BASELINE_CHAIN


def measure(op, tensors, capacity, args, mode):
    """Three Event scopes for the MegaMoE BF16 operator, optional small-operator baseline.

    Ported protocol: identical warmup/loop counts for both sides, one window per
    sample, a rebuilt forward graph for every backward sample, and the coupled
    backward regime. The reference is informational; it never decides the case.
    """
    import torch
    import torch.distributed as dist

    x, ids, p, w1, w2, dy = tensors
    leaves = (x, p, w1, w2)
    rank = dist.get_rank()

    def stage(name):
        """Progress markers: a silent hang in this phase is otherwise invisible."""
        if rank == 0 and _LOG_LEVEL != "quiet":
            print(f"[PERF] stage {name}", flush=True)

    def reset():
        for leaf in leaves:
            leaf.grad = None

    def megamoe_forward():
        reset()
        y, counts = op(
            x, ids, p, w1, w2, capacity, save_gmm1_output=args.save_gmm1_output, swiglu_limit=args.swiglu_limit
        )
        return y, counts

    def megamoe_complete():
        y, counts = megamoe_forward()
        y.backward(dy)
        return y, counts

    stage(f"scopes start warmup={args.warmup} loops={args.loops}")
    megamoe = {}
    for name, body, prepare in (
        ("forward", lambda state: megamoe_forward(), None),
        ("backward", lambda state: state[0].backward(dy), megamoe_forward),
        ("complete", lambda state: megamoe_complete(), None),
    ):
        megamoe[f"{name}_ms"] = event_stats(bench_scope(body, prepare, args))
        stage(f"{name} scope done")
    y, counts = megamoe_complete()
    if not all(bool(torch.isfinite(value).all().item()) for value in (y, x.grad, p.grad, w1.grad, w2.grad)):
        raise RuntimeError("Non-finite output or gradient after timing")
    stage("megamoe_bf16 valid")
    report = {
        "operator": "megamoe_bf16",
        "warmup": args.warmup,
        "loops": args.loops,
        "scope": (
            "Synchronized NPU Event latency of forward, autograd backward with the graph built outside the "
            "window, and forward+backward; reference, reporting and loss construction excluded; no device Timer"
        ),
        "baseline": {"status": "OFF", "reason": "--baseline off"},
        **megamoe,
    }
    route_flops = args.seq * args.topk * args.ffn * args.hidden  # FLOPs per 1 RFK
    backward_rfk = SAVED_MODE_RFK.get(mode, 12.0)
    scoped_rfk = {
        "forward": FORWARD_RFK,
        "backward": backward_rfk,
        "complete": FORWARD_RFK + backward_rfk,
    }

    def utilization(scopes_ms):
        """GMM-only achieved TFLOPS and its share of the BF16 cube peak.

        Non-GEMM work (all-to-all, permute, activation) is inside the window, so
        this is the operator's GMM utilization: achieved / peak, not a claim
        about the end-to-end model.
        """
        result = {}
        for name, rfk_value in scoped_rfk.items():
            latency = scopes_ms[f"{name}_ms"]["median"]
            flops = rfk_value * route_flops
            throughput = flops / (latency * 1e-3) / 1e12 if latency > 0 else float("nan")
            result[name] = {
                "rfk": rfk_value,
                "flops": flops,
                "tflops": throughput,
                "mfu": 100.0 * throughput / PEAK_TFLOPS,
            }
        return result

    megamoe_utilization = utilization(megamoe)
    if args.baseline == "on":
        stage("baseline plan start")
        group, reason = device_group()
        plan = None
        if group is not None:
            plan, reason = baseline_plan(ids, w1, w2, group, args, rank, dist.get_world_size(), mode)
        if plan is None:
            report["baseline"] = {"status": "SKIPPED", "reason": reason}
            stage(f"baseline skipped: {reason}")
        else:
            stage("baseline plan ready")
            chain = baseline_chain()

            def reference_forward():
                reset()
                return chain.apply(x, p, w1, w2, plan)

            def reference_complete():
                out = reference_forward()
                out.backward(dy)
                return out

            reference = {}
            for name, body, prepare in (
                ("forward", lambda state: reference_forward(), None),
                ("backward", lambda state: state.backward(dy), reference_forward),
                ("complete", lambda state: reference_complete(), None),
            ):
                reference[f"{name}_ms"] = event_stats(bench_scope(body, prepare, args))
                stage(f"baseline {name} scope done")
            out = reference_complete()
            if all(bool(torch.isfinite(value).all().item()) for value in (out, x.grad, p.grad, w1.grad, w2.grad)):
                speedup = {
                    scope: reference[f"{scope}_ms"]["median"] / max(megamoe[f"{scope}_ms"]["median"], 1e-9)
                    for scope in ("forward", "backward", "complete")
                }
                reference_utilization = utilization(reference)
                report["baseline"] = {
                    "status": "RUN",
                    "recompute_gmm1": plan.recompute,
                    "speedup": speedup,
                    "mfu": {name: value["mfu"] for name, value in reference_utilization.items()},
                    **reference,
                }
            else:
                report["baseline"] = {
                    "status": "SKIPPED",
                    "reason": "baseline produced non-finite values; its samples are not comparable",
                }
    report["mfu"] = {
        "peak_tflops": PEAK_TFLOPS,
        "peak_note": (
            "432 TFLOPS is the BF16 cube peak inherited from the source harness comment; confirm it against the "
            "official Ascend950 dense-BF16 specification before quoting an absolute utilization"
        ),
        "scope": (
            "GMM-only FLOPs (RFK model) divided by the windowed latency, as a share of the BF16 cube peak; "
            "all-to-all, permute and activation time stays inside the window, so this is not a model-level MFU"
        ),
        **megamoe_utilization,
    }
    # The source printed a backward-only roofline. Keep its derated denominator,
    # but report every scope so the three numbers stay comparable.
    roofline = {}
    for name, value in megamoe_utilization.items():
        measured_ms = megamoe[f"{name}_ms"]["median"]
        floor_ms = value["flops"] / (ROOFLINE_TFLOPS * 1e12) * 1e3
        roofline[name] = {
            "rfk": value["rfk"],
            "flops": value["flops"],
            "measured_ms": measured_ms,
            "floor_ms": floor_ms,
            "overhead_ms": measured_ms - floor_ms,
            "tflops": value["tflops"],
        }
    report["roofline"] = {"mode": mode, "roofline_tflops": ROOFLINE_TFLOPS, **roofline}
    return report, counts


_LOG_LEVEL = "all"


def log_visible(rank):
    """Phase logs follow --log: all ranks, rank 0 only, or silent."""
    return _LOG_LEVEL == "all" or (_LOG_LEVEL == "rank0" and rank == 0)


def phase_timer(rank, phase_seconds):
    @contextmanager
    def phase(name):
        started = time.perf_counter()
        visible = log_visible(rank)
        if visible:
            print(f"[STAGE] rank={rank} {name}", flush=True)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            phase_seconds[name] = elapsed
            if visible:
                print(f"[TIME] rank={rank} {name}={elapsed:.3f}s", flush=True)

    return phase


def local_mem_bytes(args, world):
    needed = (72 << 10) + world * math.ceil(world * args.expert_per_rank / 128) * 128 * 4
    weight_bytes = 4 if args.topk_weight_dtype == "fp32" else 2
    needed += 4 * args.seq * args.topk * args.hidden
    needed += math.ceil(weight_bytes * args.seq * args.topk / 512) * 512 + (2 << 20)
    return args.local_mem_size or math.ceil(max(2 << 30, needed) / (2 << 20)) * (2 << 20)


def prepare_worker(local_rank, args, rank, world, phase_seconds):
    phase = phase_timer(rank, phase_seconds)
    with phase("worker_imports"):
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401

    if log_visible(rank):
        print(f"[STAGE] rank={rank} logical_device={local_rank} physical_device={args.devices[local_rank]}", flush=True)
    with phase("device_setup"):
        torch.npu.set_device(local_rank)
        installed_root()
        from deep_ep._native import require_native

        require_native("MegaMoE BF16 precision validation")
        importlib.import_module("deep_ep_experimental.megamoe_train._C")
        from deep_ep_experimental.megamoe_train import MegaMoeBF16Train

    with phase("process_group_setup"):
        endpoint = urlsplit(args.shmem_ip_port)
        store = dist.TCPStore(
            endpoint.hostname, endpoint.port + 2, world, rank == 0, timeout=timedelta(seconds=args.timeout)
        )
        dist.init_process_group(
            "gloo", store=store, rank=rank, world_size=world, timeout=timedelta(seconds=args.timeout)
        )
    return MegaMoeBF16Train


def execute_case(local_rank, args, rank, world, op, local_mem, phase_seconds, execution_mode):
    import torch
    import torch.distributed as dist

    from npu_baseline import BIGOP_REFERENCE_VERSION, run_baseline
    from precision_metrics import evaluate

    phase = phase_timer(rank, phase_seconds)
    config = {key: value for key, value in vars(args).items() if key not in ("devices", "output")}
    configurations = [None] * world
    dist.all_gather_object(configurations, config)
    if any(value != configurations[0] for value in configurations):
        raise ValueError("All ranks must use identical test configuration")
    directory = [
        str(
            args.output
            or (
                Path(__file__).resolve().parent
                / "output"
                / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
            )
        )
        if rank == 0
        else None
    ]
    dist.broadcast_object_list(directory, src=0)
    output = Path(directory[0])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    precision = args.skip != "precision"
    performance = args.skip != "performance"
    with phase("prepare_inputs"):
        inputs = make_inputs(args, rank, world, global_tokens=precision)
    route_counts = inputs["all_counts"].reshape(world, args.expert_per_rank)
    routing = {
        "expert_rows_per_rank": route_counts.tolist(),
        "received_rows_per_rank": route_counts.sum(dim=1).tolist(),
        "source_to_destination_rows": inputs["source_to_destination_rows"].tolist(),
        "empty_experts": int((route_counts == 0).sum().item()),
        "zero_receive_ranks": int((route_counts.sum(dim=1) == 0).sum().item()),
    }
    if log_visible(rank):
        print(
            f"[INPUT] rank={rank} profile={args.data_profile} routing={args.routing_mode} "
            f"received_rows={routing['received_rows_per_rank']} empty_experts={routing['empty_experts']}",
            flush=True,
        )
    expected = inputs["all_counts"][rank * args.expert_per_rank : (rank + 1) * args.expert_per_rank]
    capacity = (
        math.ceil(int(inputs["all_counts"].reshape(world, args.expert_per_rank).sum(dim=1).max().item()) / 128) * 128
    )
    raw_baseline = mixed_baseline = truth = None
    trace = {} if args.diagnose_mxfp8 else None
    mode = resolved_save_mode(args.save_gmm1_output, args.swiglu_limit)
    if precision:
        with phase("cpu_fp64_reference"):
            truth = local_truth(inputs, args, rank)
        with phase("bigop_reference"):
            raw_baseline = combine_baseline(
                run_baseline(
                    inputs,
                    f"npu:{local_rank}",
                    rank * args.expert_per_rank,
                    "bf16",
                    args.swiglu_limit,
                    reference="bigop",
                    trace=trace if mode == "bf16" else None,
                ),
                args,
                rank,
                world,
                f"npu:{local_rank}",
            )
        if mode == "bf16":
            # Both checks use the same uncompressed small-op reference.
            mixed_baseline = raw_baseline
        else:
            with phase("megamoe_contract_reference"):
                mixed_baseline = combine_baseline(
                    run_baseline(
                        inputs, f"npu:{local_rank}", rank * args.expert_per_rank, mode, args.swiglu_limit, trace=trace
                    ),
                    args,
                    rank,
                    world,
                    f"npu:{local_rank}",
                )
    sl = slice(rank * args.seq, (rank + 1) * args.seq) if precision else slice(None)

    def tensor(value, dtype=torch.bfloat16, grad=False):
        return value.detach().to(device=f"npu:{local_rank}", dtype=dtype).contiguous().requires_grad_(grad)

    with phase("inputs_to_npu"):
        x, ids, p, w1, w2, dy = (
            tensor(inputs["x"][sl], grad=True),
            tensor(inputs["ids"][sl], torch.int32),
            tensor(inputs["p"][sl], inputs["p"].dtype, grad=True),
            tensor(inputs["w1"], grad=True),
            tensor(inputs["w2"], grad=True),
            tensor(inputs["dy"][sl]),
        )
    tensors = (x, ids, p, w1, w2, dy)
    if not args.diagnose_mxfp8:
        del inputs
    if log_visible(rank):
        print(
            f"[STAGE] rank={rank} fused forward/backward; max_recv_tokens={capacity} shmem_bytes={local_mem}",
            flush=True,
        )
    # On runtime/device exceptions restart the job; do not attempt a collective
    # close while peers may still be executing a different operation.
    with phase("fused_forward_backward"):
        y, counts = op(
            x, ids, p, w1, w2, capacity, save_gmm1_output=args.save_gmm1_output, swiglu_limit=args.swiglu_limit
        )
        y.backward(dy)
        torch.npu.synchronize()
    values = dict(y=y, dx=x.grad, dtopk_weights=p.grad, dw1=w1.grad, dw2=w2.grad, counts=counts)
    if any(value is None for value in values.values()):
        raise RuntimeError("Fused forward/backward did not produce every required result")
    if counts.dtype != torch.int32 or tuple(counts.shape) != (args.expert_per_rank,):
        raise RuntimeError("Invalid expert-count dtype or shape")
    report = {
        "rank": rank,
        "config": config,
        "world_size": world,
        "execution_mode": execution_mode,
        "identity": runtime_identity(),
        "resolved_save_mode": mode,
        "max_recv_tokens": capacity,
        "shmem_bytes": local_mem,
        "routing": routing,
        "phase_seconds": phase_seconds,
        "precision_run": precision,
        "performance_run": False,
        "references": {
            "mixed_tolerance": "megamoe_contract",
            "atk": "bigop_fp32_boundaries_vs_cpu_fp64",
            "bigop_version": BIGOP_REFERENCE_VERSION,
        },
    }
    if precision:
        with phase("precision_metrics"):
            actual = {name: value.detach().cpu().clone() for name, value in values.items()}
            report["precision"] = evaluate(actual, mixed_baseline, expected)
        if args.atk:
            with phase("atk_compare"):
                from atk_compare import DEFAULT_RATIOS, compare_with_atk

                report["precision"]["atk"] = compare_with_atk(
                    actual,
                    raw_baseline,
                    truth,
                    output / f"atk_rank_{rank}",
                    args.atk_ratios or DEFAULT_RATIOS,
                )
            report["precision"]["passed"] &= report["precision"]["atk"]["passed"]
        report["passed"] = report["precision"]["passed"]
        checks = report["precision"]["outputs"]
        mixed_ok = all(value["mixed_tolerance"]["passed"] for value in checks.values())
        print(
            f"[PRECISION] rank={rank} mixed_tolerance={'PASS' if mixed_ok else 'FAIL'} reference=megamoe_contract "
            f"counts={'PASS' if report['precision']['counts_passed'] else 'FAIL'}",
            flush=True,
        )
        if args.atk:
            atk_report = report["precision"]["atk"]
            print(
                f"[ATK] rank={rank} {'PASS' if atk_report['passed'] else 'FAIL'} standard={atk_report['standard']} provider={atk_report['provider']}",
                flush=True,
            )
            for name, check in atk_report["outputs"].items():
                if not check["passed"]:
                    print(f"[ATK FAIL] rank={rank} {name}: {check['result']}", flush=True)
        for name, checks in report["precision"].get("outputs", {}).items():
            if not checks["mixed_tolerance"]["passed"]:
                print(f"[FAIL] rank={rank} {name}: {checks}", flush=True)
        if args.diagnose_mxfp8:
            from mxfp8_diagnostics import diagnose_mxfp8

            def fused_control(save_mode):
                for leaf in (x, p, w1, w2):
                    leaf.grad = None
                control_y, control_counts = op(
                    x, ids, p, w1, w2, capacity, save_gmm1_output=save_mode, swiglu_limit=args.swiglu_limit
                )
                control_y.backward(dy)
                torch.npu.synchronize()
                return {
                    name: value.detach().cpu().clone()
                    for name, value in dict(
                        y=control_y, dx=x.grad, dtopk_weights=p.grad, dw1=w1.grad, dw2=w2.grad, counts=control_counts
                    ).items()
                }

            with phase("mxfp8_diagnosis"):
                report["mxfp8_diagnosis"] = diagnose_mxfp8(
                    inputs,
                    trace,
                    actual,
                    mixed_baseline,
                    raw_baseline,
                    truth,
                    args,
                    rank,
                    f"npu:{local_rank}",
                    output / f"mxfp8_rank_{rank}",
                    combine=lambda partial: combine_baseline(partial, args, rank, world, f"npu:{local_rank}"),
                    fused=fused_control,
                )
            del inputs, trace
        del actual
    else:
        report["passed"] = all(
            bool(torch.isfinite(value).all().item()) for name, value in values.items() if name != "counts"
        )
        report["passed"] &= torch.equal(counts.cpu().to(torch.int64), expected)
        report["precision"] = {
            "status": "NOT_RUN",
            "reason": "--skip precision",
            "scope": "finite/count smoke checks only",
        }
    del values, y, counts, raw_baseline, mixed_baseline, truth
    failed = torch.tensor(int(not report["passed"]), dtype=torch.int32)
    dist.all_reduce(failed)
    if not int(failed.item()) and performance:
        report["performance"], counts = measure(op, tensors, capacity, args, mode)
        report["performance_run"] = True
        counts_ok = torch.equal(counts.cpu().to(torch.int64), expected)
        report["performance"]["passed"] = counts_ok
        report["passed"] &= counts_ok
    else:
        report["performance"] = {
            "status": "NOT_RUN",
            "reason": "--skip performance" if not performance else "precision or smoke checks failed",
        }
    return report, output


def performance_summary(reports):
    """One terminal block: a line per scope with both sides, speedup and MFU.

    Workers only collect data; rank 0 presents it here, after the JSON exists.
    """
    first = reports[0]
    perf = first["performance"]
    baseline = perf.get("baseline", {})
    speeds = baseline.get("speedup") or {}
    base_mfu = baseline.get("mfu") or {}
    roofline = perf.get("roofline", {})
    scopes = ("forward", "backward", "complete")
    rfk = "/".join(f"{roofline[scope]['rfk']:.0f}" for scope in scopes if scope in roofline)
    lines = [
        f"[PERF] rank=0 mode={roofline.get('mode', '?')} warmup={perf['warmup']} loops={perf['loops']} "
        f"({rfk} RFK, GMM peak {PEAK_TFLOPS:.0f} TF)"
    ]
    for scope in scopes:
        row = f"[PERF] {scope:<9} megamoe_bf16={perf[f'{scope}_ms']['median']:8.3f}ms"
        if scope in speeds:
            row += f" baseline={baseline[f'{scope}_ms']['median']:8.3f}ms speedup={speeds[scope]:5.2f}x"
        else:
            row += " baseline=        --"
        mfu = perf.get("mfu", {}).get(scope, {}).get("mfu")
        if mfu is not None:
            row += f" MFU={mfu:5.1f}%" + (f"/{base_mfu[scope]:.1f}%" if scope in base_mfu else "")
        lines.append(row)
    if roofline:
        floor = "/".join(f"{roofline[scope]['floor_ms']:.2f}" for scope in scopes)
        overhead = "/".join(f"{roofline[scope]['overhead_ms']:+.2f}" for scope in scopes)
        lines.append(
            f"[PERF] roofline floor={floor}ms overhead={overhead}ms @ {roofline['roofline_tflops']} TF (fwd/bwd/fb)"
        )
    if baseline.get("status") != "RUN":
        lines.append(f"[PERF] baseline {str(baseline.get('status', '?')).lower()}: {baseline.get('reason', '')}")
    for report in reports[1:]:
        other = report["performance"]
        other_speed = other.get("baseline", {}).get("speedup") or {}
        row = (
            f"[PERF] rank={report['rank']} "
            f"forward={other['forward_ms']['median']:.3f}ms "
            f"backward={other['backward_ms']['median']:.3f}ms "
            f"complete={other['complete_ms']['median']:.3f}ms"
        )
        if other_speed:
            row += f" speedup={other_speed['forward']:.2f}/{other_speed['backward']:.2f}/{other_speed['complete']:.2f}x"
        lines.append(row)
    return lines


def write_case_reports(args, rank, world, report, output):
    import torch.distributed as dist

    from precision_metrics import write_report
    from npu_baseline import BIGOP_REFERENCE_VERSION

    precision, performance = report["precision_run"], report["performance_run"]
    write_report(output / f"rank_{rank}.json", report)
    reports = [None] * world
    dist.all_gather_object(reports, report)
    passed = all(value["passed"] for value in reports)
    if rank == 0:
        write_report(
            output / "summary.json",
            {
                "passed": passed,
                "skip": args.skip,
                "execution_mode": report["execution_mode"],
                "atk_requested": args.atk,
                "precision_run": precision,
                "precision_passed": all(value["precision"]["passed"] for value in reports) if precision else None,
                "performance_run": all(value["performance_run"] for value in reports),
                "ranks": reports,
                "precision_reference_version": BIGOP_REFERENCE_VERSION,
            },
        )
        print(f"[RESULT] {'PASS' if passed else 'FAIL'} skip={args.skip or 'none'} output={output}", flush=True)
        if passed and performance:
            for line in performance_summary(reports):
                print(line, flush=True)
    return passed


def run_rank(local_rank, args, rank=None, world=None):
    rank = local_rank if rank is None else rank
    world = len(args.devices) if world is None else world
    phase_seconds = {}
    phase = phase_timer(rank, phase_seconds)
    operator_type = prepare_worker(local_rank, args, rank, world, phase_seconds)
    import torch.distributed as dist

    try:
        local_mem = local_mem_bytes(args, world)
        os.environ["DEEPEP_SHMEM_ENDPOINT"] = args.shmem_ip_port
        with phase("operator_init"):
            op = operator_type(dist.group.WORLD, local_mem, timeout=args.timeout)
        report, output = execute_case(local_rank, args, rank, world, op, local_mem, phase_seconds, "isolated")
        with phase("operator_close"):
            op.close()
        passed = write_case_reports(args, rank, world, report, output)
    finally:
        dist.destroy_process_group()
    if not passed:
        raise SystemExit(2)


def run_suite_rank(local_rank, args, cases, output):
    rank, world = local_rank, len(args.devices)
    lifetime_seconds = {}
    phase = phase_timer(rank, lifetime_seconds)
    configs = [parse_args(case_argv(args, case, 0, output / case["id"])) for case in cases]
    local_mem = max(local_mem_bytes(config, world) for config in configs)
    operator_type = prepare_worker(local_rank, args, rank, world, lifetime_seconds)
    import torch.distributed as dist

    os.environ["DEEPEP_SHMEM_ENDPOINT"] = args.shmem_ip_port
    with phase("operator_init"):
        op = operator_type(dist.group.WORLD, local_mem, timeout=args.timeout)
    for case, config in zip(cases, configs, strict=True):
        # Barriers keep begin/end markers outside all ranks' case output.
        dist.barrier()
        if rank == 0:
            print(f"[CASE_BEGIN] {case['id']}", flush=True)
        dist.barrier()
        report, case_output = execute_case(local_rank, config, rank, world, op, local_mem, {}, "persistent")
        write_case_reports(config, rank, world, report, case_output)
        # execute_case has returned: its graphs, inputs and scratch tensors are
        # no longer live. Only the shared operator/runtime spans cases.
        dist.barrier()
        if rank == 0:
            print(f"[CASE_END] {case['id']}", flush=True)
        dist.barrier()

    # Runtime/device exceptions leave this path immediately; do not perform
    # collective close while peers may still be in another operation.
    with phase("operator_close"):
        op.close()
    lifetime = [None] * world
    dist.all_gather_object(lifetime, {"rank": rank, "phase_seconds": lifetime_seconds})
    if rank == 0:
        (output / "suite_runtime.json").write_text(
            json.dumps({"shmem_bytes": local_mem, "ranks": lifetime}, indent=2) + "\n", encoding="utf-8"
        )
    dist.barrier()
    dist.destroy_process_group()


def main(argv=None):
    global _LOG_LEVEL
    args = parse_args(argv)
    _LOG_LEVEL = args.log
    if args.suite:
        return run_suite(args)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.devices))
    installed_root()
    if args.atk:
        from atk_compare import require_atk

        require_atk()
    if "RANK" in os.environ:
        run_rank(int(os.environ["LOCAL_RANK"]), args, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
        return 0
    import torch.multiprocessing as mp

    try:
        mp.spawn(run_rank, args=(args,), nprocs=len(args.devices), join=True)
    except mp.ProcessExitedException as error:
        return 2 if error.exit_code == 2 else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
