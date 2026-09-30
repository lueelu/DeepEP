# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""Lifecycle scenarios run inside one already-initialized UDMA domain."""

from dataclasses import replace
import json
import time

from ..reference.golden import route_matrix
from ..reference.routing import route_statistics


def domain_groups(world, ep, layout="contiguous"):
    if world % ep:
        raise ValueError("world must be divisible by EP")
    if layout == "interleaved":
        domains = world // ep
        return [[domain + slot * domains for slot in range(ep)] for domain in range(domains)]
    return [list(range(start, start + ep)) for start in range(0, world, ep)]


def sequence_cases(case):
    s = case.scenario
    result = []
    for repetition in range(s.get("repetitions", 2)):
        for index in range(3):
            options = dict(category="sequence_step", parent=case.case_id, step=len(result))
            result.append(
                replace(
                    case,
                    m=129 if s["sequence"] == "size" and index == 1 else case.m,
                    route="hot_rank" if s["sequence"] == "route" and index == 1 else "balanced",
                    seed=case.seed + repetition * 1009 + index * 101,
                    scenario=options,
                )
            )
    return result


def run_sequence(case, rank, package, torch, npu, device, args, output, global_rank):
    from .worker import prepare, check

    steps = sequence_cases(case)
    tables = [route_matrix(step) for step in steps]
    max_rows = max(
        step.m
        if step.operator == "gmm_alltoallv"
        else int(table[:, rank * step.local_experts : (rank + 1) * step.local_experts].sum())
        for step, table in zip(steps, tables)
    )
    pool = {"out": torch.empty((max_rows, case.n), device=device, dtype=torch.bfloat16)}
    prepared, reports, snapshots = [], [], []
    log = output / f"progress-rank-{global_rank}.jsonl"

    def progress(index, status):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(step=index, status=status, time_ns=time.time_ns())) + "\n")

    try:
        # Prepare before enqueue so input copies don't synchronize between DUT calls.
        for step in steps:
            prepared.append(
                prepare(step, rank, package, torch, device, cache_dir=args.reference_cache, output_pool=pool)
            )
        for index, (step, (call, table)) in enumerate(zip(steps, prepared)):
            progress(index, "submitted")
            if case.scenario["submission"] == "queued":
                result = call()
                snapshots.append(result.clone())  # Same stream; preserve shared output before reuse.
            else:
                reports.append(
                    check(
                        step,
                        rank,
                        call,
                        table,
                        torch,
                        npu,
                        backend=args.golden_backend,
                        block_rows=args.golden_block_rows,
                    )
                )
                progress(index, "checked")
        if snapshots:
            npu.synchronize()
            for index, (step, (call, table), result) in enumerate(zip(steps, prepared, snapshots)):
                progress(index, "completed")
                reports.append(
                    check(
                        step,
                        rank,
                        call,
                        table,
                        torch,
                        npu,
                        result=result,
                        backend=args.golden_backend,
                        block_rows=args.golden_block_rows,
                    )
                )
                progress(index, "checked")
        return dict(
            passed=all(r["passed"] for r in reports),
            submission=case.scenario["submission"],
            steps=[
                dict(case=step.record(), routing=route_statistics(step, table), accuracy=report)
                for step, table, report in zip(steps, tables, reports)
            ],
        )
    finally:
        for call, _ in prepared:
            call.close()
