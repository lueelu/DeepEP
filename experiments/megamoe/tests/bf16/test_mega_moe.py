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
import struct
import subprocess
import sys
import sysconfig
import time
import traceback
from urllib.parse import urlsplit
from uuid import uuid4

from accuracy import check_accuracy, combine_baseline, prepare_references
from perf_compare import group_performance, fused_forward, measure, performance_summary, summarize_event_rounds


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", type=parse_devices, nargs="?", default=None)
    parser.add_argument("--bitwise-bigop", action="store_true", help="Require bitwise equality to uncompressed BigOp")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare-forward", action="store_true", help="Compare three BF16 forward paths")
    comparison.add_argument(
        "--compare-backward",
        action="store_true",
        help="Compare training forward and full backward for MegaMoE, BigOp and DeepEP MC (save=bf16)",
    )
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
        help="Exclude suite cases that save Gate/Up as MXFP8",
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
    parser.add_argument("--api", choices=("functional",), default="functional", help="Unified in-place BF16 API")
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
    parser.add_argument("--save-gmm1-output", choices=("none", "bf16", "mxfp8"), default="bf16")
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--chunk-rows", type=int, default=256, help="CPU FP64 expert-row chunk size")
    parser.add_argument(
        "--timer", action="store_true", help="One separate forward/backward Timer capture after measurement"
    )
    parser.add_argument("--timer-ranks", default=None, help="Selected EP ranks, e.g. 0-7 or 0,3; default first eight")
    parser.add_argument("--inference", action="store_true", help="Run the postweighted BF16 inference forward")
    parser.add_argument(
        "--bitwise",
        action="store_true",
        help="Require exact output and gradient bits against the NPU small-op reference for the selected save mode",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--tensor-hashes",
        action="store_true",
        help="Record full input and result SHA-256 outside timing for comparisons across builds",
    )
    parser.add_argument(
        "--save-topk-grad",
        action="store_true",
        help="Save each rank's complete dTopK tensor on CPU for offline bitwise comparison",
    )
    parser.add_argument(
        "--profile-forward-kernel",
        action="store_true",
        help="Profile training forward in a separate pass and report device kernel P50/P95",
    )
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
    parser.add_argument(
        "--local-mem-size",
        type=int,
        default=None,
        help="Legacy size hint; the report uses the symmetric buffer capacity allocated by the operator",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--shmem-ip-port", default="tcp://127.0.0.1:29862")
    parser.add_argument("--output", type=Path, default=None, help="New report directory")
    args = parser.parse_args(argv)
    if args.bitwise_bigop and args.skip == "precision":
        parser.error("--bitwise-bigop requires a precision run")
    compare = args.compare_forward or args.compare_backward
    if args.compare_backward and args.save_gmm1_output != "bf16":
        parser.error("--compare-backward requires --save-gmm1-output bf16")
    if compare and (
        args.suite
        or args.skip
        or args.bitwise_bigop
        or args.atk
        or args.timer
        or args.baseline == "off"
        or args.atk_ratios
        or args.diagnose_mxfp8
        or "RANK" in os.environ
    ):
        parser.error("The three-way comparison is a standalone benchmark; run once with python")
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
    if args.inference and (args.bitwise_bigop or args.compare_forward or args.compare_backward):
        parser.error("--inference uses --bitwise for its postweighted forward reference")
    if args.inference and (args.diagnose_mxfp8 or args.timer):
        parser.error("--inference supports accuracy and performance; the training backward Timer requires training")
    if args.bitwise and args.skip == "precision":
        parser.error("--bitwise requires precision")
    if args.profile_forward_kernel and (args.inference or args.skip == "performance"):
        parser.error("--profile-forward-kernel requires training performance")
    if args.save_topk_grad and args.inference:
        parser.error("--save-topk-grad requires training")
    if args.atk and args.skip == "precision":
        parser.error("--atk requires precision; use --skip performance for accuracy only")
    args.atk = args.skip != "precision" and not compare
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
    if compare and world > 8:
        parser.error("DeepEP MC supports at most 8 ranks")
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
        if args.save_gmm1_output != "mxfp8":
            parser.error("--diagnose-mxfp8 requires MXFP8 saving")
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
        selected = [case for case in selected if case["save_gmm1_output"] != "mxfp8"]
        if not selected:
            raise ValueError(
                "No cases remain after --skip-mxfp8; choose a suite/case containing BF16 or recompute cases"
            )
    return selected


def launch_options(args):
    argv = []
    for key in ("chunk_rows", "warmup", "loops", "timeout", "skip", "local_mem_size", "baseline", "log", "api"):
        value = getattr(args, key)
        if value is not None:
            argv += ["--" + key.replace("_", "-"), str(value)]
    if args.inference:
        argv.append("--inference")
    if args.bitwise:
        argv.append("--bitwise")
    if args.tensor_hashes:
        argv.append("--tensor-hashes")
    if args.save_topk_grad:
        argv.append("--save-topk-grad")
    if args.profile_forward_kernel:
        argv.append("--profile-forward-kernel")
    if args.atk:
        argv += ["--atk"]
    if args.bitwise_bigop:
        argv += ["--bitwise-bigop"]
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
                or set(outputs) != ({"y"} if args.inference else set(OUTPUTS))
                or check.get("passed") != all(value.get("passed", False) for value in outputs.values())
                or (rank["passed"] and not check.get("passed"))
            ):
                raise RuntimeError(f"Incomplete/inconsistent ATK results in {report_file}, rank={rank['rank']}")
    if args.bitwise:
        from precision_metrics import OUTPUTS

        expected = {"y", "counts"} if args.inference else {*OUTPUTS, "counts"}
        for rank in ranks:
            exact = rank.get("precision", {}).get("bitwise", {})
            if set(exact) != expected or (rank["passed"] and not all(value["passed"] for value in exact.values())):
                raise RuntimeError(f"Incomplete/inconsistent bitwise results in {report_file}, rank={rank['rank']}")
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
            "resolved_save_mode": case["save_gmm1_output"],
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
        ("megamoe._C", "deep_ep_experimental.megamoe._C"),
    ):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.origin:
            continue
        origin = Path(spec.origin)
        identity["libraries"][label] = {"path": str(origin), "sha256": digest(origin)}
        if label == "megamoe._C":
            kernel = origin.with_name("libmegamoe_kernel.so")
            if kernel.is_file():
                identity["libraries"]["megamoe_bf16_kernel"] = {
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
    hint = "Build with DEEPEP_BUILD_MEGAMOE=ON and install its wheel in this Python environment."
    if spec is None or spec.origin is None:
        raise RuntimeError(f"No installed deep_ep package for {sys.executable}. {hint}")
    root = Path(spec.origin).resolve().parent.parent
    for name in ("deep_ep", "deep_ep_experimental", "deep_ep_experimental.megamoe"):
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


def make_buffer(operator_type, group, args, world):
    return operator_type(
        group,
        args.expert_per_rank * world,
        args.seq,
        args.topk,
        args.hidden,
        args.ffn,
        mma_type="bf16xbf16",
        need_grad=not args.inference,
        max_recv_tokens=None if args.inference else args.seq * args.topk * world,
        timeout=args.timeout,
    )


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
        importlib.import_module("deep_ep_experimental.megamoe._C")
        from deep_ep_experimental.megamoe import SymmBuffer

    with phase("process_group_setup"):
        endpoint = urlsplit(args.shmem_ip_port)
        store = dist.TCPStore(
            endpoint.hostname, endpoint.port + 2, world, rank == 0, timeout=timedelta(seconds=args.timeout)
        )
        dist.init_process_group(
            "gloo", store=store, rank=rank, world_size=world, timeout=timedelta(seconds=args.timeout)
        )
    return SymmBuffer


def tensor_hash(value):
    import hashlib
    import torch

    data = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def execute_case(local_rank, args, rank, world, op, phase_seconds, execution_mode):
    import torch
    import torch.distributed as dist

    from npu_baseline import BIGOP_REFERENCE_VERSION

    phase = phase_timer(rank, phase_seconds)
    local_mem = op.symmetric_buffer_bytes
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
    if rank != 0:
        output.mkdir(parents=True, exist_ok=True)
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
    capacity = op.max_recv_tokens if args.inference else math.ceil(max(routing["received_rows_per_rank"]) / 128) * 128
    raw_baseline = mixed_baseline = truth = None
    trace = {} if args.diagnose_mxfp8 else None
    mode = "none" if args.inference else args.save_gmm1_output
    if precision:
        truth, raw_baseline, mixed_baseline = prepare_references(
            inputs, args, rank, world, f"npu:{local_rank}", mode, trace, phase
        )
    sl = slice(rank * args.seq, (rank + 1) * args.seq) if precision else slice(None)

    def tensor(value, dtype=torch.bfloat16, grad=False):
        return value.detach().to(device=f"npu:{local_rank}", dtype=dtype).contiguous().requires_grad_(grad)

    with phase("inputs_to_npu"):
        x, ids, p, w1, w2, dy = (
            tensor(inputs["x"][sl], grad=not args.inference),
            tensor(inputs["ids"][sl], torch.int32),
            tensor(inputs["p"][sl], inputs["p"].dtype, grad=not args.inference),
            tensor(inputs["w1"], grad=not args.inference),
            tensor(inputs["w2"], grad=not args.inference),
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
        y, counts = fused_forward(
            op,
            x,
            ids,
            p,
            w1,
            w2,
            capacity,
            save_gmm1_output=args.save_gmm1_output,
            swiglu_limit=args.swiglu_limit,
            api=args.api,
        )
        if not args.inference:
            y.backward(dy)
        torch.npu.synchronize()
    values = dict(y=y, counts=counts)
    if not args.inference:
        values.update(dx=x.grad, dtopk_weights=p.grad, dw1=w1.grad, dw2=w2.grad)
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
    if args.tensor_hashes:
        report["input_sha256"] = {key: tensor_hash(t) for key, t in zip(("x", "ids", "p", "w1", "w2", "dy"), tensors)}
        report["output_sha256"] = {key: tensor_hash(t) for key, t in values.items()}
    if args.save_topk_grad:
        torch.save(values["dtopk_weights"].detach().cpu().contiguous(), output / f"rank_{rank}_dtopk_weights.pt")
    if precision:
        actual, report["precision"] = check_accuracy(
            values, mixed_baseline, raw_baseline, truth, expected, args, phase, output, rank
        )
        report["passed"] = report["precision"]["passed"]
        if args.bitwise_bigop:
            bitwise = report["precision"]["bitwise_bigop"]
            failures = {k: v for k, v in bitwise["outputs"].items() if not v["passed"]}
            print(f"[BITWISE] rank={rank} {'FAIL ' + str(failures) if failures else 'PASS'}", flush=True)
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
                control_y, control_counts = fused_forward(
                    op,
                    x,
                    ids,
                    p,
                    w1,
                    w2,
                    capacity,
                    save_gmm1_output=save_mode,
                    swiglu_limit=args.swiglu_limit,
                    api=args.api,
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
        if args.inference:
            from inference_performance import measure_inference

            report["performance"], counts = measure_inference(op, tensors, args, output / f"rank_{rank}_profiler")
        else:
            report["performance"], counts = measure(op, tensors, capacity, args, mode, output)
        report["performance_run"] = True
        counts_ok = torch.equal(counts.cpu().to(torch.int64), expected)
        report["performance"]["passed"] = counts_ok
        report["passed"] &= counts_ok
    else:
        report["performance"] = {
            "status": "NOT_RUN",
            "reason": "--skip performance" if not performance else "precision or smoke checks failed",
        }
    if not int(failed.item()) and getattr(args, "timer", False):
        report["timer"] = collect_timer(op, tensors, capacity, args, rank, world, output)
    return report, output


def collect_timer(op, tensors, capacity, args, rank, world, output):
    """Keep diagnostic sampling and CPU export outside all Event windows."""
    import hashlib
    import torch
    import torch.distributed as dist
    from deep_ep_experimental.megamoe.timer.bf16.__main__ import export, parse_ranks

    ranks = parse_ranks(args.timer_ranks) if args.timer_ranks else list(range(min(world, 8)))
    if any(value >= world for value in ranks):
        raise ValueError("Timer rank exceeds this EP group")
    x, ids, p, w1, w2, dy = tensors

    def snapshot():
        for leaf in (x, p, w1, w2):
            leaf.grad = None
        y, counts = fused_forward(
            op,
            x,
            ids,
            p,
            w1,
            w2,
            capacity,
            save_gmm1_output=args.save_gmm1_output,
            swiglu_limit=args.swiglu_limit,
            api=args.api,
        )
        y.backward(dy)
        torch.npu.synchronize()
        return {
            name: hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in dict(y=y, counts=counts, dx=x.grad, dp=p.grad, dw1=w1.grad, dw2=w2.grad).items()
        }

    dist.barrier()
    plain = snapshot()
    dist.barrier()
    from deep_ep_experimental.megamoe.timer.bf16.capture import TimerCapture

    with TimerCapture(op, enabled=rank in ranks) as capture:
        timed = snapshot()
    checks = {name: plain[name] == timed[name] for name in plain}
    capture.export(output / "timer", parse=False)
    dist.barrier()
    if rank == 0:
        export(output / "timer", ranks)
    dist.barrier()
    if not all(checks.values()):
        raise AssertionError(f"Timer on/off result mismatch: {checks}")
    return dict(passed=True, bitwise_equal=checks, ranks=ranks, output=str(output / "timer"))


def write_case_reports(args, rank, world, report, output):
    import torch
    import torch.distributed as dist

    from precision_metrics import write_report
    from npu_baseline import BIGOP_REFERENCE_VERSION

    precision, performance = report["precision_run"], report["performance_run"]
    write_report(output / f"rank_{rank}.json", report)
    passed_flag = torch.tensor(int(report["passed"]), dtype=torch.int32, device="cpu")
    dist.all_reduce(passed_flag, op=dist.ReduceOp.MIN)
    passed = bool(passed_flag.item())
    reports = [None] * world if rank == 0 else None
    dist.gather_object(report, reports, dst=0)
    if rank == 0:
        if [value["rank"] for value in reports] != list(range(world)):
            raise ValueError("Incomplete or out-of-order EP rank reports")
        event_ms = (
            summarize_event_rounds(reports)
            if performance and not args.inference and all(r["performance_run"] for r in reports)
            else {}
        )
        write_report(
            output / "summary.json",
            {
                "passed": passed,
                "skip": args.skip,
                "execution_mode": report["execution_mode"],
                "atk_requested": args.atk,
                "precision_run": precision,
                "precision_passed": all(value["precision"]["passed"] for value in reports) if precision else None,
                **(
                    {"bitwise_bigop_passed": all(r["precision"]["bitwise_bigop"]["passed"] for r in reports)}
                    if args.bitwise_bigop
                    else {}
                ),
                "performance_run": all(value["performance_run"] for value in reports),
                "event_tail_half_rank_max_mean_ms": event_ms,
                "ep_performance": group_performance(reports) if all(r["performance_run"] for r in reports) else {},
                "ranks": reports,
                "precision_reference_version": BIGOP_REFERENCE_VERSION,
            },
        )
        print(f"[RESULT] {'PASS' if passed else 'FAIL'} skip={args.skip or 'none'} output={output}", flush=True)
        if passed and performance:
            for name, timings in event_ms.items():
                print(
                    f"[EVENT] {name} tail-half mean of slowest EP rank: "
                    + " ".join(f"{scope}={ms:.3f}ms" for scope, ms in timings.items()),
                    flush=True,
                )
            for key, stats in group_performance(reports).items():
                if key.endswith("_ms") and isinstance(stats, dict) and "p95" in stats:
                    print(f"[EP] {key}: P50={stats['median']:.3f} P95={stats['p95']:.3f} ms", flush=True)
            for line in [] if args.inference else performance_summary(reports):
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
        os.environ["DEEPEP_SHMEM_ENDPOINT"] = args.shmem_ip_port
        with phase("operator_init"):
            op = make_buffer(operator_type, dist.group.WORLD, args, world)
        report, output = execute_case(local_rank, args, rank, world, op, phase_seconds, "isolated")
        with phase("operator_close"):
            op.destroy()
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
    operator_type = prepare_worker(local_rank, args, rank, world, lifetime_seconds)
    import torch.distributed as dist

    os.environ["DEEPEP_SHMEM_ENDPOINT"] = args.shmem_ip_port
    peak_shmem_bytes = 0
    for case, config in zip(cases, configs, strict=True):
        with phase("operator_init"):
            op = make_buffer(operator_type, dist.group.WORLD, config, world)
            peak_shmem_bytes = max(peak_shmem_bytes, op.symmetric_buffer_bytes)
        # Barriers keep begin/end markers outside all ranks' case output.
        dist.barrier()
        if rank == 0:
            print(f"[CASE_BEGIN] {case['id']}", flush=True)
        dist.barrier()
        report, case_output = execute_case(local_rank, config, rank, world, op, {}, "per-configuration")
        write_case_reports(config, rank, world, report, case_output)
        # Different shapes use different fixed buffer configurations.
        with phase("operator_close"):
            op.destroy()
        dist.barrier()
        if rank == 0:
            print(f"[CASE_END] {case['id']}", flush=True)
        dist.barrier()

    lifetime = [None] * world
    dist.all_gather_object(lifetime, {"rank": rank, "phase_seconds": lifetime_seconds})
    if rank == 0:
        (output / "suite_runtime.json").write_text(
            json.dumps({"shmem_bytes": peak_shmem_bytes, "ranks": lifetime}, indent=2) + "\n", encoding="utf-8"
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
    if args.compare_forward or args.compare_backward:
        from forward_compare import compare_forward

        return compare_forward(args)
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
