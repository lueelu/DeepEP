# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""python -m experiments.ep_gmm_fused.tests.run {accuracy,bench,profile,report}."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .framework.cases import generate, load_cases, save_cases
from .performance.metrics import summarize


def execute(command, log, timeout, env=None):
    """Timeout terminates the entire torchrun/profiler process group on Linux."""
    with Path(log).open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=(os.name == "posix")
        )
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            else:
                process.kill()
                process.wait()
            raise
    if status:
        raise RuntimeError(f"worker exit status {status}; see {log}")


def report_run(root, baseline=None):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    results = []
    for selection in plan["selections"]:
        directory = root / selection["case_id"]
        ranks = [json.loads(p.read_text()) for p in sorted(directory.glob("rank-*.json"))]
        expected = set(range(selection["world_size"]))
        seen = {r["global_rank"] for r in ranks}
        passed = seen == expected and all(r.get("passed", False) for r in ranks)
        row = {
            "case_id": selection["case_id"],
            "passed": passed,
            "expected_ranks": sorted(expected),
            "observed_ranks": sorted(seen),
            "rank_reports": ranks,
            "category": selection.get("category", "legacy"),
            "status": "passed" if passed else ("not_run" if not ranks else "failed_or_incomplete"),
        }
        if plan["mode"] == "profile":
            captures = [json.loads(p.read_text()) for p in directory.glob("capture-*.json")]
            row["captures"] = captures
            row["passed"] &= len(captures) == len(expected) and all(c.get("passed") for c in captures)
        elif plan["mode"] == "bench":
            captures = [json.loads(p.read_text()) for p in sorted(directory.glob("capture-*.json"))]
            row["captures"] = captures
            row["passed"] &= (
                len(captures) == len(expected)
                and {c.get("global_rank") for c in captures} == expected
                and all(c.get("passed", False) for c in captures)
            )
            if row["passed"]:
                metrics = {c["metric"] for c in captures}
                if len(metrics) != 1:
                    raise ValueError("msprof metrics differ across ranks")
                row["msprof"] = summarize(
                    [c["samples_us"] for c in captures], sample_semantics="max_rank_" + metrics.pop()
                )
        row["status"] = "passed" if row["passed"] else ("not_run" if not ranks else "failed_or_incomplete")
        results.append(row)
    result = {
        "schema_version": 1,
        "mode": plan["mode"],
        "passed": bool(results) and all(r["passed"] for r in results),
        "results": results,
        "plan_sha256": hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
    }
    if baseline:
        prior = json.loads(Path(baseline).read_text())
        by_id = {r["case_id"]: r for r in prior["results"]}
        for row in results:
            base = by_id.get(row["case_id"])
            if not base or not base["passed"] or "msprof" not in base or "msprof" not in row:
                raise ValueError(
                    "baseline requires matching passing msprof cases; NPU Event results are not comparable"
                )
            if base["msprof"]["sample_semantics"] != row["msprof"]["sample_semantics"]:
                raise ValueError("baseline timing semantics differ")
            if len(base["rank_reports"]) != len(row["rank_reports"]):
                raise ValueError("baseline topology differs")
            for current, previous in zip(row["rank_reports"], base["rank_reports"]):
                for key in ("torch", "torch_npu", "device", "visible_devices", "stack_versions"):
                    if current["environment"][key] != previous["environment"][key]:
                        raise ValueError(f"baseline environment differs: {key}")
                if current["timing"] != previous["timing"]:
                    raise ValueError("baseline timing configuration differs")
            if [c["profiler"]["version"] for c in row["captures"]] != [
                c["profiler"]["version"] for c in base["captures"]
            ]:
                raise ValueError("baseline msprof version differs")
            row["speedup_p50"] = base["msprof"]["p50_us"] / row["msprof"]["p50_us"]
        result["baseline"] = str(Path(baseline).resolve())
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# EP_GMM_FUSED validation",
        "",
        f"Mode: {plan['mode']}; passed: {result['passed']}",
        "",
    ]
    if plan["mode"] == "bench":
        lines += ["| Case | Execution passed | msprof p50 of max-rank kernel duration (us) |", "|---|---|---|"]
        lines += [f"| {r['case_id']} | {r['passed']} | {r.get('msprof', {}).get('p50_us', '')} |" for r in results]
        lines += ["", "Accuracy: not run. Composition samples sum the two target kernel durations."]
    else:
        lines += ["| Case | Result |", "|---|---|"]
        lines += [f"| {r['case_id']} | {r['passed']} |" for r in results]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("accuracy", "bench", "profile", "report"))
    parser.add_argument("--suite", choices=("smoke", "model", "generalization", "all", "nightly"), default="smoke")
    parser.add_argument("--cases", help="saved schema_version=1 case manifest (exact replay)")
    parser.add_argument("--case-id")
    parser.add_argument("--ep-size", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--operator", choices=("alltoallv_gmm", "gmm_alltoallv", "composition"))
    parser.add_argument("--category", action="append", help="include category; repeat to select several")
    parser.add_argument("--exclude-category", action="append", default=[])
    parser.add_argument("--model", help="model ID substring")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--golden-backend", choices=("npu-fp32",), default="npu-fp32")
    parser.add_argument("--golden-block-rows", type=int, default=1024)
    parser.add_argument("--reference-cache", help="existing local scratch directory (temporary files cleaned on exit)")
    parser.add_argument("--blas-threads", type=int, default=1, help="CPU BLAS threads per worker")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", default="benchmark-runs/ep_gmm_fused")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap", choices=("default", "group"), default="group")
    parser.add_argument("--baseline")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--metric", choices=("PipeTimeline", "InstrTimeline", "op_summary"), default="PipeTimeline")
    parser.add_argument("--pipes")
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args(argv)
    if args.mode == "bench":
        if args.pipes:
            parser.error("bench uses msprof op_summary; pipe selection is only supported by profile")
        args.metric = "op_summary"
    if (
        min(args.shard_count, args.golden_block_rows, args.blas_threads) < 1
        or not 0 <= args.shard_index < args.shard_count
    ):
        parser.error("invalid shard, block size or BLAS thread count")
    if args.reference_cache and not Path(args.reference_cache).is_dir():
        parser.error("reference-cache must be an existing local directory")
    if args.mode == "report":
        return 0 if report_run(args.output, args.baseline)["passed"] else 1
    if min(args.iterations, args.batches, args.nnodes, args.timeout) < 1 or args.warmup < 0:
        parser.error("iterations/batches/nnodes/timeout must be positive; warmup nonnegative")
    if args.nnodes > 1 and (not args.run_id or not args.nproc_per_node):
        parser.error("multi-node runs require a shared --run-id, shared output filesystem and --nproc-per-node")
    if not 0 <= args.node_rank < args.nnodes:
        parser.error("node-rank outside nnodes")
    cases = load_cases(args.cases) if args.cases else generate(args.suite, args.seed)
    from .framework.catalog import category, counts

    cases = [
        c
        for c in cases
        if (args.case_id is None or c.case_id == args.case_id)
        and (args.ep_size is None or c.ep_size == args.ep_size)
        and (args.operator is None or c.operator == args.operator)
        and (not args.category or category(c) in args.category)
        and category(c) not in args.exclude_category
        and (not args.model or args.model in (c.scenario or {}).get("model", ""))
    ]
    cases = cases[args.shard_index :: args.shard_count]
    if not cases:
        parser.error("case selection is empty")
    if args.mode != "accuracy" and any(c.expected_error for c in cases):
        parser.error("select positive cases for benchmark/profiling")
    if args.mode != "accuracy" and any(
        category(c) == "state_reuse" or (c.scenario or {}).get("contract") == "reject_zero_input" for c in cases
    ):
        parser.error("sequences and rejection contracts are accuracy-only")
    if args.mode == "profile" and any(c.operator == "composition" for c in cases):
        parser.error("profile one fused operator at a time with --operator")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        parser.error("run-id must be a simple directory name")
    root = (Path(args.output) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=(args.nnodes > 1))
    # Exclusive per-node claim permits concurrent nodes, rejects stale/reused runs.
    with (root / f"node-{args.node_rank}.claim").open("x") as claim:
        claim.write(str(os.getpid()))
    profiler_config = None
    if args.mode in ("bench", "profile"):
        from .profiling.timeline import discover

        profiler_config = root / f"profiler-node-{args.node_rank}.json"
        profiler_config.write_text(
            json.dumps(
                {
                    "profiler": None if args.dry_run else discover(args.metric).record(),
                    "metric": args.metric,
                    "pipes": args.pipes,
                },
                indent=2,
            )
        )
    selections, commands = [], []
    for case in cases:
        nproc = args.nproc_per_node or (case.scenario or {}).get("required_world_size", case.ep_size)
        world = nproc * args.nnodes
        required = (case.scenario or {}).get("required_world_size")
        if required and (world != required or (not args.dry_run and args.nproc_per_node is None)):
            parser.error("domain_isolation requires explicit 64-rank topology; exclude the category for smaller runs")
        if nproc < 1 or world % case.ep_size:
            parser.error("nproc-per-node * nnodes must be divisible by each case's EP size")
        directory = root / case.case_id
        directory.mkdir(exist_ok=(args.nnodes > 1))
        case_file = directory / f"case-node-{args.node_rank}.json"
        case_file.write_text(json.dumps(asdict(case), indent=2) + "\n")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={args.nnodes}",
            f"--node_rank={args.node_rank}",
            f"--nproc_per_node={nproc}",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}",
            "--max_restarts=0",
            "-m",
            "experiments.ep_gmm_fused.tests.framework.worker",
            "--case",
            str(case_file),
            "--output",
            str(directory),
            "--mode",
            "capture" if args.mode == "profile" else args.mode,
            "--bootstrap",
            args.bootstrap,
            "--timeout",
            str(args.timeout),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--batches",
            str(args.batches),
        ]
        command += ["--golden-backend", args.golden_backend, "--golden-block-rows", str(args.golden_block_rows)]
        if args.reference_cache:
            command += ["--reference-cache", str(Path(args.reference_cache).resolve())]
        if profiler_config:
            command += ["--profiler-config", str(profiler_config)]
        selections.append(
            {"case_id": case.case_id, "world_size": world, "category": category(case), "scenario": case.scenario}
        )
        commands.append(command)
    plan = {
        "schema_version": 1,
        "mode": args.mode,
        "selections": selections,
        "commands": commands,
        "category_counts": counts(cases),
        "arguments": vars(args),
    }
    if args.node_rank == 0:
        (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        save_cases(root / "cases.json", cases)
    print(f"{len(cases)} cases; artifacts: {root}", flush=True)
    if args.dry_run:
        return 0
    if os.name != "posix":
        parser.error("device execution requires Linux with Ascend950; CPU tests and dry-run work here")
    try:
        worker_env = dict(os.environ)
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
            worker_env[name] = str(args.blas_threads)
        for selection, command in zip(selections, commands):
            print(f"Running {selection['case_id']}", flush=True)
            execute(
                command, root / selection["case_id"] / f"worker-node-{args.node_rank}.log", args.timeout, worker_env
            )
    finally:
        if args.node_rank == 0:
            result = report_run(root, args.baseline)
    return 0 if args.node_rank != 0 or result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
