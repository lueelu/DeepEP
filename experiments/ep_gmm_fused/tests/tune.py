"""Torchrun entry point for the eight matched EP64/128 cases. Timings use NPU Events."""

import argparse
from dataclasses import replace
from datetime import timedelta
import importlib
import json
import os
from pathlib import Path
from statistics import median

from .framework.cases import CaseSpec, save_cases


def matched_cases(world):
    for op in ("alltoallv_gmm", "gmm_alltoallv"):
        for width in (3072, 6144):
            k, n = (3584, width) if op == "alltoallv_gmm" else (width, 3584)
            for experts in (7, 14):
                yield CaseSpec(operator=op, ep_size=world, m=65536, n=n, k=k, local_experts=experts,
                               seed=20261127, preallocate=True)


def measure(call, profile, iterations, repeats, align, torch, npu, dist):
    call.set_tiling(profile)
    dist.barrier()
    events = []
    for _ in range(repeats):
        for _ in range(align):
            call()
        start, end = npu.Event(enable_timing=True), npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        events.append((start, end))
    events[-1][1].synchronize()
    us = torch.tensor([start.elapsed_time(end) * 1000 / iterations for start, end in events], dtype=torch.float64)
    dist.all_reduce(us, op=dist.ReduceOp.MAX)
    samples = us.tolist()
    if not all(0 < value < float("inf") for value in samples):
        raise RuntimeError(f"invalid NPU Event samples {samples}")
    return {"tiling": profile, "p50_us": median(samples), "max_rank_us_per_repeat": samples}


def run_case(case, args, package, tiles, config, torch, npu, dist):
    from .framework.worker import prepare

    local, rank = int(os.environ["LOCAL_RANK"]), dist.get_rank()
    shape = dict(m=case.m, n=case.n, k=case.k)
    op = tiles.ALL_TO_ALLV if case.operator == "alltoallv_gmm" else tiles.GROUPED_MATMUL
    spec = dict(operator=op, shape=shape, expert_num=case.local_experts * case.ep_size, world_size=case.ep_size,
                output_all_to_allv=False, workspace_bytes=config.workspace_data_bytes)
    cores = npu.get_device_properties(local).cube_core_num
    universe = tiles.candidates_for_case(spec, strategy="exhaustive", max_candidates=None, aic_cores=cores)
    candidates = universe if len(universe) <= args.candidates else tiles.candidates_for_case(
        spec, max_candidates=args.candidates, seed=case.seed, aic_cores=cores)
    if not candidates or candidates[0] != tiles.default_tiling(spec):
        raise RuntimeError(f"invalid default tiling for {spec}")
    call, _ = prepare(case, rank, package, torch, f"npu:{local}", cache_dir=args.reference_cache)
    try:
        if rank == 0:
            print(f"{case.operator} K={case.k} N={case.n} G={case.local_experts}, "
                  f"screening {len(candidates)}/{len(universe)} legal tilings", flush=True)
        for _ in range(3):
            call()
        npu.synchronize()
        # Measure the actual native fallback. Replay finalists with explicit
        # profiles; recording the effective default also makes the manifest portable.
        screened = [measure(call, p, 2, 1, 1, torch, npu, dist) for p in [None, *candidates[1:]]]
        ranked = sorted(screened[1:], key=lambda r: r["p50_us"])
        finalists = [None, *(r["tiling"] for r in ranked[:args.top])]
        if rank == 0:
            print(f"  independently retiming {len(finalists)} finalists × {args.repeats} repeats", flush=True)
        final = [measure(call, p, args.iterations, args.repeats, 3, torch, npu, dist) for p in finalists]
    finally:
        call.close()
    # Preserve the fallback unless improvement exceeds 5% and both spreads.
    anchor = final[0]
    best = min(final, key=lambda row: row["p50_us"])
    def spread(row):
        v = row["max_rank_us_per_repeat"]
        return (max(v) - min(v)) / row["p50_us"]

    threshold = max(.05, spread(anchor), spread(best))
    selected = best if 1 - best["p50_us"] / anchor["p50_us"] > threshold else anchor
    for row in final:
        row["event_global_tflops"] = 2 * case.ep_size * case.m * case.k * case.n / (row["p50_us"] * 1e6)
        row["event_mfu_percent"] = 100 * row["event_global_tflops"] / (case.ep_size * args.peak_tflops)
    profile = selected["tiling"] or candidates[0]
    result = dict(case=case.record(), default_tiling=candidates[0], workspace_bytes=config.workspace_data_bytes,
                  ranks_per_node=config.ranks_per_node, default_mfu_percent=anchor["event_mfu_percent"],
                  selected_mfu_percent=selected["event_mfu_percent"], selected_p50_us=selected["p50_us"],
                  selected_global_tflops=selected["event_global_tflops"],
                  gain_pp=selected["event_mfu_percent"] - anchor["event_mfu_percent"],
                  selection_threshold_percent=threshold * 100, legal_grid_count=len(universe),
                  screened_count=len(screened), finalists_count=len(finalists), screen_iterations=2, screen_repeats=1,
                  peak_tflops_per_rank=args.peak_tflops, final_iterations=args.iterations, final_repeats=args.repeats,
                  selected_tiling=profile, screen=screened, final=final, accuracy_checked=False)
    if rank == 0:
        print(f"  default MFU {anchor['event_mfu_percent']:.3f}%  selected {result['selected_mfu_percent']:.3f}%  "
              f"({result['gain_pp']:+.3f} pp), {result['selected_p50_us']/1000:.3f} ms, "
              f"{result['selected_global_tflops']:.1f} global TFLOPS", flush=True)
    return result, replace(case, tiling=profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh shared output directory")
    parser.add_argument("--peak-tflops", type=float, required=True, help="per-card dense BF16 peak, e.g. 432")
    parser.add_argument("--candidates", type=int, default=512, help="quick-screen budget, including the fallback")
    parser.add_argument("--top", type=int, default=16, help="non-default finalists, in addition to fallback")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference-cache", help="existing local directory for temporary input/weight files")
    args = parser.parse_args()
    if min(args.top, args.iterations, args.repeats) < 1 or args.candidates < 2 or not 0 < args.peak_tflops < float("inf"):
        parser.error("expected at least two candidates, positive sampling counts and a finite positive peak")

    from .framework.installed import load_installed_ep_gmm_fused
    import torch
    import torch.distributed as dist
    import torch_npu

    world, local = int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world not in (64, 128):
        parser.error("launch on 64 or 128 ranks with torchrun")
    npu = torch_npu.npu
    npu.set_device(local)
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    try:
        package = load_installed_ep_gmm_fused()
        tiles = importlib.import_module(package.__name__ + "._tiling")
        if dist.get_rank() == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            with (args.output / "tune.claim").open("x") as claim:
                claim.write(str(os.getpid()))
        measured, selected = [], []
        with package.udma_group(group=dist.group.WORLD):
            config = package.get_udma_config()
            if not config.ranks_per_node or config.workspace_data_bytes != tiles.LARGE_EP_WORKSPACE_BYTES:
                raise RuntimeError("the matched multi-node tune requires an active striped UDMA domain")
            if os.environ.get("DEEPEP_LARGE_EP_PIPELINE_CHUNKS") == "0":
                raise RuntimeError("DEEPEP_LARGE_EP_PIPELINE_CHUNKS=0 disables the reverse pipeline for this tune")
            for case in matched_cases(world):
                row, picked = run_case(case, args, package, tiles, config, torch, npu, dist)
                measured.append(row)
                selected.append(picked)
                if dist.get_rank() == 0:
                    (args.output / f"{case.case_id}.json").write_text(json.dumps(row, indent=2) + "\n")
        if dist.get_rank() == 0:
            save_cases(args.output / "selected-cases.json", selected)
            (args.output / "tuning-summary.json").write_text(json.dumps(dict(
                ep_size=world, metric="median_of_max_rank_npu_event_us_per_call",
                accuracy_checked=False, cases=measured), indent=2) + "\n")
            lines = [f"EP{world} NPU Event, {args.peak_tflops:g} peak TFLOPS/card, accuracy not checked.",
                     "Operator           K     N   G    grid    final  default MFU   tuned MFU   delta pp   global TFLOPS   p50 ms"]
            for row in measured:
                c = row["case"]
                coverage = f"{row['screened_count']}/{row['legal_grid_count']}"
                lines.append(f"{c['operator']:<15} {c['k']:4}  {c['n']:4}  {c['local_experts']:2}  "
                             f"{coverage:>6} {row['finalists_count']:7} {row['default_mfu_percent']:11.3f}% "
                             f"{row['selected_mfu_percent']:10.3f}% {row['gain_pp']:+9.3f} "
                             f"{row['selected_global_tflops']:15.1f} {row['selected_p50_us']/1000:8.3f}")
            text = "\n".join(lines)
            (args.output / "comparison.txt").write_text(text + "\n")
            print(text, flush=True)
            print(f"8 cases complete. {args.output}/tuning-summary.json", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
