# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Decode selected ranks only. Raw buffers are never modified."""

import argparse
import json
from pathlib import Path

from .parse_timer_csv import decode
from .breakdown import rank_breakdown, write_breakdown
from .trace_parser import (
    apply_wait_mode,
    first_start_cycle,
    raw_events_for_rank,
    synchronized_rank_bases,
    trace_events,
    write_summary,
)


def parse_ranks(value):
    result = []
    for part in value.split(","):
        bounds = part.split("-")
        if len(bounds) == 1:
            result.append(int(part))
        elif len(bounds) == 2:
            start, end = map(int, bounds)
            if start > end:
                raise ValueError("Rank range must be ascending")
            result.extend(range(start, end + 1))
        else:
            raise ValueError("Use ranks such as 0-7 or 0,3,7")
    if not result or min(result) < 0 or max(result) >= 128 or len(set(result)) != len(result):
        raise ValueError("Ranks must be unique integers in [0,127]")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="One case's timer directory")
    parser.add_argument("--ranks", type=parse_ranks, default=parse_ranks("0-7"))
    parser.add_argument("--include-wait", action="store_true")
    parser.add_argument("--merge", action="store_true", help="Additionally merge the selected ranks only")
    parser.add_argument(
        "--time-origin",
        choices=("rank", "sync"),
        default="rank",
        help="sync is an approximate barrier-anchor alignment, NOT absolute time",
    )
    args = parser.parse_args()
    events = {}
    for rank in args.ranks:
        stem = args.directory / f"rank_{rank}_npu_time"
        raw = stem.with_suffix(".bin")
        if not raw.exists():
            raw = stem.with_suffix(".csv")
        parsed = args.directory / f"{stem.name}_parsed.csv"
        decode(raw, parsed)
        events[rank] = raw_events_for_rank(parsed, rank)
        if not events[rank]:
            raise ValueError(f"rank={rank}: empty trace; check timer-enabled build and collection")
    bases = {rank: first_start_cycle(raw) for rank, raw in events.items()}
    write_breakdown(
        [rank_breakdown(raw, rank) for rank, raw in events.items()], args.directory, args.directory.parent.name
    )
    if args.time_origin == "sync":
        bases, _ = synchronized_rank_bases(events)
        print("[TIMER] Approximate sync-anchor alignment; not a global latency measurement.")
    merged = []
    for rank, raw in events.items():
        trace = trace_events(raw, "keep", bases[rank], args.include_wait, args.time_origin)
        output = args.directory / f"rank_{rank}_npu_time.json"
        output.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
        write_summary(apply_wait_mode(raw, args.include_wait), args.directory / f"rank_{rank}_summary.csv")
        if args.merge:
            merged.extend(trace)
        print(f"[TIMER] rank={rank} events={len(trace)} {output}")
    if args.merge:
        (args.directory / "all_rank_trace.json").write_text(json.dumps(merged), encoding="utf-8")


if __name__ == "__main__":
    main()
