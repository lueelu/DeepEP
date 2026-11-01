"""Sweep V1/V2 UDMA schedules with existing precision and profiler benchmarks.

One command per node. Every candidate uses fresh rank processes and SHMEM;
there are no runtime private-attribute mutations or retries of failed jobs.
--dry-run writes the complete plan without importing torch or contacting nodes.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

from route_workloads import DEFAULTS as ROUTE_DEFAULTS, options as route_options, validate as validate_routes

import bench_v1_bandwidth as v1
import bench_v2_bandwidth as v2
from bandwidth_multinode import content_identity, library_identity
from stage_metrics import file_identity, write_csv, write_json


ORDERS = ("rank", "group", "group_staggered", "group_round_robin")
PARAMETERS = (
    "api",
    "udma_schedule",
    "peers_per_worker",
    "wqes_per_visit",
    "dispatch_peer_order",
    "combine_peer_order",
    "ranks_per_group",
    "peers_per_group_per_worker",
    "workers_per_peer",
    "qps_per_worker",
    "aiv",
    "dtype",
    "expand",
) + tuple(ROUTE_DEFAULTS)
METRICS = ("caller_us", "device_span_us", "device_sum_us", "comm_us", "combine_stage_us", "fused_combine_us")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--apis", choices=("v1", "v2"), nargs="+", default=["v1", "v2"])
    parser.add_argument("--schedules", choices=("peer", "token"), nargs="+", default=["peer", "token"])
    parser.add_argument("--peers-per-worker", type=int, choices=(1, 2, 4, 8), nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--wqes-per-visit", type=int, choices=(4, 8, 16, 32), nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--dispatch-peer-orders", choices=ORDERS, nargs="+", default=list(ORDERS[:3]))
    parser.add_argument("--combine-peer-orders", choices=ORDERS, nargs="+", default=list(ORDERS[:3]))
    parser.add_argument("--workers-per-peer", type=int, choices=(1, 2), nargs="+", default=[1])
    parser.add_argument("--qps-per-worker", type=int, choices=(1, 2), nargs="+", default=[1])
    parser.add_argument("--group-peer-counts", type=int, choices=(1, 2, 4), nargs="+", default=[1, 2, 4])
    parser.add_argument("--paired-peer-orders", action="store_true", help="only matching dispatch/combine orders")
    parser.add_argument(
        "--group-sizes", type=int, nargs="+", help="default: min(8, EP); invalid divisors are recorded as skipped"
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="independent measurement rounds; 3 recommended for finalists"
    )
    parser.add_argument("--route-modes", nargs="+", choices=("uniform", "rank_hotspot", "local_group"))
    parser.add_argument("--route-skews", nargs="+", type=float, default=[2.0, 4.0])
    parser.add_argument("--dry-run", action="store_true", help="write plan only; no NPU/runtime/network use")
    sweep, remaining = parser.parse_known_args(argv)
    if sweep.help:
        parser.print_help()
        print("\nShared shape, topology, profiling and output options:")
        v2.arguments(["--help"])
    if sweep.repeats <= 0:
        parser.error("repeats must be positive")
    for name in (
        "apis",
        "schedules",
        "peers_per_worker",
        "wqes_per_visit",
        "dispatch_peer_orders",
        "combine_peer_orders",
        "group_sizes",
        "group_peer_counts",
        "workers_per_peer",
        "qps_per_worker",
    ):
        values = getattr(sweep, name)
        if values is not None and (len(values) != len(set(values)) or not values):
            parser.error(f"duplicate/empty {name}")
    forbidden = (
        "--worker",
        "--udma-schedule",
        "--dispatch-peer-order",
        "--combine-peer-order",
        "--ranks-per-group",
        "--peers-per-group-per-worker",
        "--sweep-plan-sha256",
    )
    if any(arg.split("=")[0] in forbidden for arg in remaining):
        parser.error("use the sweep plural options; worker/internal options are not accepted")
    # Validate shared options with a legal scalar peer count even for EP64.
    base = v2.arguments(remaining + ["--peers-per-worker", "8"])
    return sweep, base


def make_plan(sweep, base):
    candidates, skipped = [], []
    sizes = sweep.group_sizes or [min(8, base.ep_size)]
    for api, dtype, expand, aiv, schedule in itertools.product(
        sweep.apis, base.dtypes, base.expand, base.aiv, sweep.schedules
    ):
        if schedule == "token":
            # Token ownership does not use peers_per_worker/group order. Do not
            # benchmark identical kernels sixteen times under different labels.
            topologies = [("rank", "rank", None)]
            peer_counts = [sweep.peers_per_worker[0]]
        else:
            topologies = [
                (dispatch, combine, size)
                for dispatch, combine in itertools.product(sweep.dispatch_peer_orders, sweep.combine_peer_orders)
                if not sweep.paired_peer_orders or dispatch == combine
                for size in ([None] if dispatch == combine == "rank" else sizes)
            ]
            peer_counts = sweep.peers_per_worker
        expanded = []
        for dispatch, combine, size in topologies:
            rr = "group_round_robin" in (dispatch, combine)
            # Both RR operations ignore the legacy total-peer setting.
            legacy = peer_counts[:1] if dispatch == combine == "group_round_robin" else peer_counts
            for peers, group_peers in itertools.product(legacy, sweep.group_peer_counts if rr else [None]):
                expanded.append((dispatch, combine, size, peers, group_peers))
        for (dispatch, combine, size, peers, group_peers), visit, lanes, qps in itertools.product(
            expanded,
            sweep.wqes_per_visit,
            [1] if schedule == "token" else sweep.workers_per_peer,
            [1] if schedule == "token" else sweep.qps_per_worker,
        ):
            config = dict(
                api=api,
                dtype=dtype,
                expand=expand,
                aiv=aiv,
                udma_schedule=schedule,
                dispatch_peer_order=dispatch,
                combine_peer_order=combine,
                ranks_per_group=size,
                peers_per_worker=peers,
                wqes_per_visit=visit,
                peers_per_group_per_worker=group_peers,
                workers_per_peer=lanes,
                qps_per_worker=qps,
            )
            reason = "two QPs per worker require workers_per_peer=1" if qps == 2 and lanes != 1 else None
            if size is not None and (size <= 0 or base.ep_size % size):
                reason = "ranks_per_group must be a positive divisor of EP"
            if schedule == "peer" and reason is None:
                for order in (dispatch, combine):
                    workers = (
                        math.ceil(size / group_peers)
                        if order == "group_round_robin"
                        else math.ceil((base.ep_size - 1) / peers)
                    )
                    if workers * lanes > aiv or workers * lanes * qps > 32:
                        reason = "required sending workers exceed AIV/QP limit"
                    if order == "group_round_robin" and math.ceil(size / workers) * (base.ep_size // size) > 64:
                        reason = "group_round_robin exceeds 64 peers per worker"
            if reason:
                skipped.append(dict(config=config, reason=reason))
            else:
                candidates.append(config)
    # Workload grouping stays fixed while scheduling topology is scanned.
    workload_configs = []
    for mode in sweep.route_modes or [base.route_mode]:
        ratios = (
            sweep.route_skews
            if sweep.route_modes and mode == "rank_hotspot"
            else [base.route_skew if mode == "rank_hotspot" else 1.0]
        )
        for ratio in ratios:
            workload = {**route_options(base), "route_mode": mode, "route_skew": ratio}
            probe = copy.copy(base)
            for key, value in workload.items():
                setattr(probe, key, value)
            validate_routes(probe)
            if workload not in workload_configs:
                workload_configs.append(workload)
    candidates = [{**config, **workload} for config in candidates for workload in workload_configs]
    jobs = []
    for repeat in range(sweep.repeats):
        order = list(enumerate(candidates))
        # Identical on every node; vary order between repetitions to reduce drift.
        random.Random(base.seed + repeat).shuffle(order)
        for candidate, config in order:
            jobs.append(
                dict(name=f"c{candidate:05d}_r{repeat + 1:02d}", candidate=candidate, repeat=repeat + 1, config=config)
            )
    contract = v2.suite_contract(base)
    for key in (
        "peers_per_worker",
        "wqes_per_visit",
        "udma_schedule",
        "dispatch_peer_order",
        "combine_peer_order",
        "ranks_per_group",
        "peers_per_group_per_worker",
        "workers_per_peer",
        "qps_per_worker",
        "sweep_plan_sha256",
    ):
        contract.pop(key, None)
    plan = dict(
        format_version=1,
        contract=contract,
        jobs=jobs,
        skipped=skipped,
        repeats=sweep.repeats,
        candidates=len(candidates),
        notes=[
            "Every job initializes fresh rank processes and SHMEM.",
            "Token schedule ignores peer-count/group-order sweeps.",
            "V1 fused combine is not separately timed communication.",
            "Default dispatch uses GM output; no zero-copy performance is inferred.",
            "Rank=all bandwidth is group aggregate, not per-card bandwidth.",
            "Best results are observed within the selected grid, not a global optimum.",
            "Random routing is fixed across scheduler/dtype/expand variants; load groups are independent.",
            "Rankings compare only matching routing workloads.",
        ],
    )
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def job_arguments(base, job, root, digest):
    args = copy.deepcopy(base)
    for name, value in job["config"].items():
        if name not in ("dtype", "expand", "aiv"):
            setattr(args, name, value)
    args.dtypes = [job["config"]["dtype"]]
    args.expand = [job["config"]["expand"]]
    args.aiv = [job["config"]["aiv"]]
    args.output = root / job["name"]
    args.sweep_plan_sha256 = digest
    return args


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def atomic_csv(path, rows):
    if rows:
        temporary = path.with_suffix(path.suffix + ".tmp")
        write_csv(temporary, rows)
        os.replace(temporary, path)


def rankings(rows, repeats):
    """Compare only matching API/dtype/expand/operator, using group max-rank time.

    A candidate is eligible only after every requested repetition was reviewed.
    Null timers remain unavailable; neither fused reduce nor missing results
    become zero or participate in a pure-communication ranking.
    """
    grouped = {}
    for row in rows:
        if row["rank"] == "all":
            workload_key = json.dumps(
                {k: row.get(k, "legacy_recorded" if k == "route_mode" else v) for k, v in ROUTE_DEFAULTS.items()},
                sort_keys=True,
            )
            key = (row["api"], row["dtype"], row["expand"], row["op"], row["candidate"], workload_key)
            grouped.setdefault(key, []).append(row)
    tables = {}
    for (api, dtype, expand, op, candidate, workload_key), runs in grouped.items():
        if len(runs) != repeats or {r["repeat"] for r in runs} != set(range(1, repeats + 1)):
            continue
        for metric in METRICS:
            values = [r.get(metric) for r in runs]
            if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in values):
                continue
            group = (api, dtype, expand, op, metric, workload_key)
            entry = {k: runs[0][k] for k in PARAMETERS if k not in ROUTE_DEFAULTS}
            entry.update(json.loads(workload_key))
            entry.update(
                op=op,
                candidate=candidate,
                metric=metric,
                median_us=statistics.median(values),
                min_us=min(values),
                max_us=max(values),
                repeats=repeats,
            )
            # Carry already verified bandwidth fields without inventing a timer.
            for name in runs[0]:
                if name.endswith("GBps"):
                    samples = [r.get(name) for r in runs]
                    entry["median_" + name] = (
                        statistics.median(samples)
                        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in samples)
                        else None
                    )
            tables.setdefault(group, []).append(entry)
    result = []
    for key in sorted(tables):
        for position, entry in enumerate(sorted(tables[key], key=lambda r: (r["median_us"], r["candidate"])), 1):
            result.append(dict(position=position, **entry))
    return result


def publish(root, plan, state, rows):
    table = rankings(rows, plan["repeats"])
    atomic_csv(root / "summary.csv", rows)
    atomic_csv(root / "ranking.csv", table)
    atomic_csv(root / "best.csv", [r for r in table if r["position"] == 1])
    atomic_json(
        root / "report.json",
        dict(
            status=state["status"],
            complete=state["status"] == "complete",
            planned=len(plan["jobs"]),
            completed=len(state["completed"]),
            failed=state.get("failed"),
            not_completed=len(plan["jobs"]) - len(state["completed"]),
            plan_sha256=plan["sha256"],
            notes=plan["notes"],
            summary=rows,
            ranking=table,
        ),
    )
    atomic_json(root / "state.json", state)


def review_job(root, plan, job, *, reparse=False):
    child = root / job["name"]
    if plan["contract"]["nnodes"] > 1:
        child /= "node0"
    child_plan = json.loads((child / "plan.json").read_text(encoding="utf-8"))
    args = child_plan["arguments"]
    if args.get("sweep_plan_sha256") != plan["sha256"] or child_plan["api"] != job["config"]["api"]:
        raise ValueError("child report belongs to another sweep/API")
    for key, value in job["config"].items():
        actual = args[{"dtype": "dtypes", "aiv": "aiv", "expand": "expand"}.get(key, key)]
        if actual != ([value] if key in ("dtype", "aiv", "expand") else value):
            raise ValueError(f"child sweep configuration mismatch: {key}")
    for key in (
        "ep_size",
        "nnodes",
        "num_tokens",
        "hidden",
        "num_topk",
        "experts_per_rank",
        "warmup",
        "iterations",
        "seed",
        "huge_pages",
    ):
        if args[key] != plan["contract"][key]:
            raise ValueError(f"child workload mismatch: {key}")
    if args.get("flush_l2_cache", False) != plan["contract"].get("flush_l2_cache", False):
        raise ValueError("child L2 eviction mode mismatch")
    if args.get("combine_bias", "two") != plan["contract"].get("combine_bias", "two"):
        raise ValueError("child combine bias mode mismatch")
    if reparse:
        v2.report(child)
    report = json.loads((child / "report.json").read_text(encoding="utf-8"))
    if not report["passed"]:
        raise ValueError("child benchmark did not pass")
    first_case = child_plan["cases"][0]["name"]
    if "source" in plan and json.dumps(child_plan.get("source_identity"), sort_keys=True) != json.dumps(
        plan["source"], sort_keys=True
    ):
        raise ValueError("source checkout changed during sweep")
    record = json.loads((child / first_case / "rank0/result.json").read_text(encoding="utf-8"))
    # Keep the global sweep index bounded at EP64. Every rank's full result
    # remains in the child report/CSV; rankings always use the all-rank row.
    rows = [
        {**job["config"], "candidate": job["candidate"], "repeat": job["repeat"], "job": job["name"], **r}
        for r in report["summary"]
        if r["rank"] in (0, "all")
    ]
    return rows, library_identity(record["identity"]), file_identity(child / "report.json")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(sweep, base):
    plan = make_plan(sweep, base)
    if not plan["jobs"]:
        raise ValueError("no valid configurations in requested grid")
    plan["source"] = content_identity(v2.git_identity(), v2.REPO)
    plan.pop("sha256")
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    root = base.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / f"sweep_plan_node{base.node_rank}.json", plan)
    print(
        f"SWEEP_PLAN candidates={plan['candidates']} jobs={len(plan['jobs'])} "
        f"skipped={len(plan['skipped'])} repeats={plan['repeats']} sha256={plan['sha256']}",
        flush=True,
    )
    if sweep.dry_run:
        print(f"DRY_RUN: {root}; no NPU work started", flush=True)
        return
    if os.name != "posix":
        raise RuntimeError("NPU execution requires Linux; --dry-run/--report-only need no NPU")
    # A second sweep in this checkout must not interleave cases from another plan.
    import fcntl
    import tempfile

    lock_key = hashlib.sha256(str(v2.REPO).encode()).hexdigest()[:16]
    with open(Path(tempfile.gettempdir()) / f"deepep_sweep_{lock_key}.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status="running", started=time.time(), completed=[], current=None)
        rows, libraries = [], None

        def save():
            if base.node_rank == 0:
                publish(root, plan, state, rows)
            else:
                atomic_json(root / f"state_node{base.node_rank}.json", state)

        save()
        try:
            for job in plan["jobs"]:
                state["current"] = job["name"]
                save()
                current_source = content_identity(v2.git_identity(), v2.REPO)
                # JSON normalizes tuples when reading plans copied from Linux.
                if json.dumps(current_source, sort_keys=True) != json.dumps(plan["source"], sort_keys=True):
                    raise ValueError("source checkout changed; no next job started")
                print(
                    f"SWEEP_RUN {len(state['completed']) + 1}/{len(plan['jobs'])} "
                    f"{job['name']} {json.dumps(job['config'])}",
                    flush=True,
                )
                args = job_arguments(base, job, root, plan["sha256"])
                module = v1 if args.api == "v1" else v2
                with (root / f"{job['name']}_node{base.node_rank}.log").open("x", encoding="utf-8") as log:
                    with (
                        contextlib.redirect_stdout(Tee(sys.stdout, log)),
                        contextlib.redirect_stderr(Tee(sys.stderr, log)),
                    ):
                        v2.launch(args, entrypoint=module.__file__)
                item = dict(job=job["name"])
                if base.node_rank == 0:
                    data, identity, artifact = review_job(root, plan, job)
                    if libraries is not None and libraries != identity:
                        raise ValueError("loaded library identity changed between sweep jobs")
                    libraries = identity
                    rows.extend(data)
                    item["report"] = artifact
                state["completed"].append(item)
                save()
            state.update(status="complete", current=None, ended=time.time())
            save()
            print(f"SWEEP_COMPLETE {root}", flush=True)
        except BaseException:
            state.update(status="failed", failed=state["current"], ended=time.time(), error=traceback.format_exc())
            save()
            raise


def report_only(root):
    root = root.resolve()
    plan = json.loads((root / "sweep_plan_node0.json").read_text(encoding="utf-8"))
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    jobs = {job["name"]: job for job in plan["jobs"]}
    rows, libraries = [], None
    seen = set()
    for completed in state["completed"]:
        name = completed["job"]
        if name in seen:
            raise ValueError("duplicate completed job")
        seen.add(name)
        data, identity, _ = review_job(root, plan, jobs[name], reparse=True)
        if libraries is not None and identity != libraries:
            raise ValueError("loaded library identity differs between sweep jobs")
        libraries = identity
        rows.extend(data)
    if state["status"] == "complete" and len(seen) != len(jobs):
        raise ValueError("complete sweep has missing jobs")
    publish(root, plan, state, rows)
    print(f"SWEEP_REPORT {len(seen)}/{len(jobs)} status={state['status']} {root / 'report.json'}")


if __name__ == "__main__":
    sweep_options, common_options = arguments()
    if common_options.report_only:
        report_only(common_options.report_only)
    else:
        run(sweep_options, common_options)
