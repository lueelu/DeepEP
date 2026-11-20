# Copyright (c) 2026, Lu Lu

"""DSV41 full MegaMoE: EP16/32 x (original BS 1..24 multiplied by six)."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile

BATCHES = tuple(range(6, 145, 6))


def batches(value):
    selected = [int(part) for part in value.split(",")]
    if not selected or len(set(selected)) != len(selected) or any(bs not in BATCHES for bs in selected):
        raise argparse.ArgumentTypeError("BS must be distinct values from 6,12,...,144 (already multiplied by six)")
    return sorted(selected)


def specification(args, ep):
    return dict(
        suite="dsv41_full_megamoe_bs_x6_v1",
        ep_size=ep,
        ranks_per_node=16,
        original_bs=[bs // 6 for bs in args.bs_values],
        bs_multiplier=6,
        input=args.input,
        routing=args.routing,
        perf_method=args.perf_method,
        config=dict(
            schema_version=1,
            model=dict(
                hidden=5120,
                intermediate_hidden=2304,
                num_experts=384 + ep,
                num_topk=6,
                num_shared_experts=args.shared_experts,
            ),
            bs_values=args.bs_values,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        ),
    )


def command(args, directory):
    return [
        sys.executable,
        "-I",
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(args.ep_size // 16),
        "--nproc-per-node",
        "16",
        "--node-rank",
        str(args.node_rank),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "--rdzv-conf",
        "timeout=600",
        "--max-restarts",
        "0",
        "--module",
        "deep_ep_experimental.megamoe.benchmark",
        "--config",
        str(directory / f"config_node_{args.node_rank}.json"),
        "--mode",
        "perf",
        "--input",
        args.input,
        "--routing",
        args.routing,
        "--perf-method",
        args.perf_method,
        "--output-dir",
        str(directory / "measurements"),
    ]


def preflight(args, env):
    # Isolated Python resolves the installed wheel, just like the worker processes.
    code = """import json, sys
from importlib.resources import files
import torch, torch_npu
from deep_ep_experimental.megamoe.api import _extension
ext = _extension()
info = json.loads(files('deep_ep').joinpath('_build_info.json').read_text())
if torch.npu.device_count() != 16:
    raise RuntimeError('Exactly 16 visible NPUs per node are required')
if ext.timer_enabled:
    raise RuntimeError('Rebuild with DEEPEP_MEGAMOE_TIMER=OFF for this performance suite')
if int(sys.argv[1]) == 32:
    if info['shmem']['rdma_backend'] not in ('HNS_1825', 'XSCALE'):
        raise RuntimeError('EP32 requires a RoCE-enabled wheel and matching SHMEM library')
    if getattr(ext, 'hierarchical_max_tokens', 32) < int(sys.argv[2]):
        raise RuntimeError('Installed wheel has the old RoCE BS limit; rebuild this checkout')
print('[CHECK] Installed build:', json.dumps(info), flush=True)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(args.ep_size), str(max(args.bs_values))],
        env=env,
        cwd=tempfile.gettempdir(),
        check=True,
    )


def write_json(path, value, mode="w"):
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def collect(directory):
    manifests = sorted(directory.glob("launcher_node_*.json"))
    if not manifests:
        raise ValueError(f"No DSV41 launcher manifest in {directory}")
    spec = json.loads(manifests[0].read_text(encoding="utf-8"))["specification"]
    if spec.get("suite") != "dsv41_full_megamoe_bs_x6_v1":
        raise ValueError(f"Unsupported suite in {directory}")
    for path in manifests[1:]:
        if json.loads(path.read_text(encoding="utf-8"))["specification"] != spec:
            raise ValueError(f"Inconsistent node configurations in {directory}")
    ep, config = spec["ep_size"], spec["config"]
    model = config["model"]
    # Each rank's report contains the collective samples of every EP rank.
    # Node 1's local disk therefore suffices too; never combine local-only means.
    rank_dir = directory / "measurements/rank_0"
    if not rank_dir.exists():
        rank_dir = directory / "measurements/rank_16"
    benchmark_manifest = rank_dir / "manifest.json"
    if benchmark_manifest.exists():
        observed = json.loads(benchmark_manifest.read_text(encoding="utf-8"))
        if (
            observed["ep_size"] != ep
            or observed["mode"] != "perf"
            or any(observed[key] != spec[key] for key in ("input", "routing", "perf_method"))
            or any(observed["config"][key] != value for key, value in config.items())
        ):
            raise ValueError(f"Benchmark configuration differs from launcher manifest: {benchmark_manifest}")
    rows = []
    for bs in config["bs_values"]:
        row = dict(
            ep=ep,
            original_bs=bs // 6,
            bs=bs,
            global_tokens=ep * bs,
            hidden=model["hidden"],
            intermediate_hidden=model["intermediate_hidden"],
            topk=model["num_topk"],
            physical_experts=model["num_experts"],
            local_experts=model["num_experts"] // ep,
            shared_experts=model["num_shared_experts"],
            input=spec["input"],
            weights="mxfp4",
            routing=spec["routing"],
            perf_method=spec["perf_method"],
            max_rank_mean_us=None,
            min_rank_mean_us=None,
            mean_rank_mean_us=None,
            per_round_min_mean_us=None,
            per_round_max_mean_us=None,
            recorded_rounds=config["iters"],
            selected_rounds=None,
            status="missing",
            error="",
            source=str(rank_dir / f"bs_{bs}/profiling.json"),
        )
        path = Path(row["source"])
        if path.exists():
            try:
                if not benchmark_manifest.exists():
                    raise ValueError("Missing benchmark manifest")
                report = json.loads(path.read_text(encoding="utf-8"))
                samples = report["per_rank_rounds_us"]  # Already restricted to the tail half.
                rounds = (config["iters"] + 1) // 2
                if (
                    report["metric"] != "megamoe_kernel_duration_us"
                    or report["timer_pass"]
                    or report["scope"] != "all_ep_ranks"
                    or report["iteration_selection"] != "tail_half_ceil"
                    or report["recorded_rounds"] != config["iters"]
                    or report["rounds"] != rounds
                    or report["perf_method"] != spec["perf_method"]
                    or len(samples) != ep
                    or any(len(values) != rounds for values in samples)
                    or any(not math.isfinite(v) or v <= 0 for values in samples for v in values)
                ):
                    raise ValueError("Incomplete ranks/rounds or incompatible measurement")
                means = [statistics.mean(values) for values in samples]
                row.update(
                    max_rank_mean_us=max(means),
                    min_rank_mean_us=min(means),
                    mean_rank_mean_us=statistics.mean(means),
                    per_round_min_mean_us=statistics.mean(min(values) for values in zip(*samples)),
                    per_round_max_mean_us=statistics.mean(max(values) for values in zip(*samples)),
                    selected_rounds=rounds,
                    status="ok",
                )
            except (KeyError, TypeError, ValueError) as error:
                row.update(status="invalid", error=str(error))
        rows.append(row)
    return spec, rows


def summarize(directories, output):
    rows, seen = [], set()
    for directory in directories:
        _, cases = collect(directory.resolve())
        for case in cases:
            key = case["ep"], case["bs"]
            if key in seen:
                raise ValueError(f"Duplicate EP/BS {key}; supply one run per EP")
            seen.add(key)
            rows.append(case)
    rows.sort(key=lambda row: (row["ep"], row["bs"]))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", rows)
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        latency = f"{row['max_rank_mean_us']:.3f} us" if row["status"] == "ok" else row["status"]
        print(f"[RESULT] EP={row['ep']} original_BS={row['original_bs']} BS={row['bs']} max-rank-mean={latency}")
    print(f"[SUMMARY] {output / 'summary.csv'} ({sum(r['status'] == 'ok' for r in rows)}/{len(rows)} complete)")
    return all(row["status"] == "ok" for row in rows)


def run(args):
    if sys.platform != "linux":
        raise ValueError("NPU execution requires Linux; plan and summarize work without devices")
    if args.node_rank >= args.ep_size // 16:
        raise ValueError("EP16 uses node 0; EP32 uses nodes 0 and 1")
    if args.ep_size == 32 and args.master_addr in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("EP32 requires --master-addr reachable from both nodes")
    directory = args.output_dir.expanduser().resolve()
    spec = specification(args, args.ep_size)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "ASCEND_LAUNCH_BLOCKING"):
        env.pop(key, None)
    env.update(
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONSAFEPATH="1",
        DEEPEP_SHMEM_ENDPOINT=f"tcp://{args.master_addr}:{args.shmem_port}",
    )
    preflight(args, env)
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("launcher_node_*.json"):
        if json.loads(path.read_text(encoding="utf-8"))["specification"] != spec:
            raise ValueError(f"Different case matrix already exists in {directory}; choose a fresh output directory")
    launch = command(args, directory)
    # Exclusive files allow both nodes to share storage without overwriting a run.
    write_json(directory / f"launcher_node_{args.node_rank}.json", dict(specification=spec, command=launch), "x")
    write_json(directory / f"config_node_{args.node_rank}.json", spec["config"], "x")
    print(f"[SUITE] EP{args.ep_size} S={args.shared_experts} BS={args.bs_values}; full MegaMoE, {args.input} x MXFP4")
    print("[EXEC] " + shlex.join(launch), flush=True)
    status = dict(state="running", returncode=None)
    status_path = directory / f"status_node_{args.node_rank}.json"
    write_json(status_path, status)
    try:
        with (directory / f"node_{args.node_rank}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                launch,
                env=env,
                cwd=tempfile.gettempdir(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
        status.update(state="completed" if code == 0 else "failed", returncode=code)
    except BaseException as error:
        status.update(state="interrupted", error=str(error))
        raise
    finally:
        write_json(status_path, status)
    complete = summarize([directory], directory / f"summary_node_{args.node_rank}")
    return code if code else (0 if complete else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--ep-size", type=int, choices=(16, 32), required=name == "run")
        sub.add_argument("--bs-values", type=batches, default=list(BATCHES), help="Actual tokens; default 6,12,...,144")
        sub.add_argument("--shared-experts", type=int, choices=(0, 1), default=1)
        sub.add_argument("--input", choices=("fp8", "bf16"), default="fp8")
        sub.add_argument("--routing", choices=("balanced", "random"), default="balanced")
        sub.add_argument("--perf-method", choices=("legacy", "shmem"), default="legacy")
        sub.add_argument("--warmup", type=int, default=20)
        sub.add_argument("--iters", type=int, default=100)
        sub.add_argument("--seed", type=int, default=20261029)
        if name == "run":
            sub.add_argument("--node-rank", type=int, choices=(0, 1), default=0)
            sub.add_argument("--master-addr", default="127.0.0.1")
            sub.add_argument("--master-port", type=int, default=29654)
            sub.add_argument("--shmem-port", type=int, default=26733)
            sub.add_argument("--output-dir", type=Path, required=True, help="New run directory; one per EP")
    summary = commands.add_parser("summarize")
    summary.add_argument("runs", type=Path, nargs="+")
    summary.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "summarize":
            return 0 if summarize(args.runs, args.output_dir) else 1
        if args.warmup < 0 or args.iters <= 0 or args.seed < 0:
            raise ValueError("warmup/seed must be nonnegative and iters positive")
        if args.action == "plan":
            plans = [specification(args, ep) for ep in ((args.ep_size,) if args.ep_size else (16, 32))]
            print(json.dumps(dict(case_count=sum(len(p["config"]["bs_values"]) for p in plans), plans=plans), indent=2))
            return 0
        if (
            any(not 1024 <= port <= 65535 for port in (args.master_port, args.shmem_port))
            or args.master_port == args.shmem_port
        ):
            raise ValueError("Master/SHMEM ports must be distinct and in [1024,65535]")
        return run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"[ERROR] {error}\n")


if __name__ == "__main__":
    sys.exit(main())
