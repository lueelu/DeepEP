#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Create Chrome/Perfetto/MindStudio JSON and a compact stage summary."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import re
from statistics import median

from .protocol import CYCLE_TO_US

SYNC_ANCHOR_EVENT = "AIV_INPUT_CROSS_RANK_SYNC"


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def event_names(fields):
    field_set = set(fields)
    result = []
    for field in fields:
        if not field.endswith("_start_cycle"):
            continue
        name = field[: -len("_start_cycle")]
        if name != "KERNEL_TIMING" and f"{name}_end_cycle" in field_set:
            result.append(name)
    return result


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(item for item in intervals if item[1] > item[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def compressed_cycle(cycle, removed):
    delta = 0
    for start, end in removed:
        if cycle >= end:
            delta += end - start
        elif cycle > start:
            delta += cycle - start
            break
        else:
            break
    return cycle - delta


def apply_input_sync_mode(events, mode):
    if mode == "keep":
        return events
    removed = merge_intervals(
        (event["start_cycle"], event["end_cycle"]) for event in events if "INPUT_CROSS_RANK_SYNC" in event["name"]
    )
    result = []
    for event in events:
        if "INPUT_CROSS_RANK_SYNC" in event["name"]:
            continue
        copied = dict(event)
        if mode == "compress":
            copied["start_cycle"] = compressed_cycle(event["start_cycle"], removed)
            copied["end_cycle"] = compressed_cycle(event["end_cycle"], removed)
            if copied["end_cycle"] <= copied["start_cycle"]:
                continue
        result.append(copied)
    return result


def apply_wait_mode(events, include_wait):
    if include_wait:
        return events
    return [event for event in events if "WAIT" not in event["name"]]


def raw_events_for_rank(path: Path, rank: int):
    fields, rows = read_rows(path)
    names = event_names(fields)
    events = []
    for row in rows:
        for name in names:
            start = int(row[f"{name}_start_cycle"])
            end = int(row[f"{name}_end_cycle"])
            if start <= 0 or end <= start:
                continue
            wave = {}
            if f"{name}_wave_id" in row:
                wave_id = int(row[f"{name}_wave_id"])
                if wave_id < -1:
                    raise ValueError(f"{path}: {name} has invalid wave ID {wave_id}")
                wave = {"wave_id": wave_id}
            events.append(
                {
                    **wave,
                    "name": name,
                    "core_type": row["core_type"],
                    "group_id": int(row["group_id"]),
                    "sub_id": int(row["sub_id"]),
                    "rank": rank,
                    "start_cycle": start,
                    "end_cycle": end,
                    "task": json.loads(row.get(f"{name}_task") or "{}"),
                }
            )
    # 在过滤 WAIT 之前编号；tile/token 批次的 WAIT 与计算共享身份，空 WAIT 槽不挤占编号。
    groups = defaultdict(set)
    keys = {}
    for index, event in enumerate(events):
        if "wave_id" not in event:
            continue
        family = event_family(event["name"])
        task = event.get("task", {})
        if task.get("granularity") == "tile" or (task and family in ("AIV_COMBINE", "AIV_COMBINE_WAIT")):
            family = family.removesuffix("_WAIT")
        key = (event["group_id"], event["core_type"], event["sub_id"], family, event["wave_id"])
        suffix = re.search(r"_(\d+)$", event["name"])
        slot = int(suffix[1]) if suffix else 0
        groups[key].add(slot)
        keys[index] = (key, slot)
    indices = {key: {slot: i for i, slot in enumerate(sorted(slots))} for key, slots in groups.items()}
    for index, (key, slot) in keys.items():
        events[index]["wave_index"] = indices[key][slot]
    return events


def first_start_cycle(events):
    return min((event["start_cycle"] for event in events), default=0)


def synchronization_anchor(events):
    ends = [event["end_cycle"] for event in events if event["name"] == SYNC_ANCHOR_EVENT]
    if not ends:
        return None
    return int(median(ends))


def synchronized_rank_bases(raw_by_rank):
    anchors = {rank: synchronization_anchor(events) for rank, events in raw_by_rank.items()}
    missing = [rank for rank, anchor in anchors.items() if anchor is None]
    if missing:
        raise ValueError(
            f"missing {SYNC_ANCHOR_EVENT} for rank(s) {missing}; "
            "use --time-origin rank if this pipeline has no input sync"
        )
    earliest_relative = min(
        event["start_cycle"] - anchors[rank] for rank, events in raw_by_rank.items() for event in events
    )
    bases = {rank: anchor + earliest_relative for rank, anchor in anchors.items()}
    return bases, anchors


def trace_events(
    raw_events,
    input_sync_mode,
    base_cycle,
    include_wait=False,
    time_origin="global",
):
    selected = apply_wait_mode(raw_events, include_wait)
    selected = apply_input_sync_mode(selected, input_sync_mode)
    result = []
    for event in selected:
        task = event.get("task", {})
        granularity = task.get("granularity")
        display_name = event["name"]
        wave_metadata = {}
        if "wave_id" in event:
            wave_id = event["wave_id"]
            wave_label = str(wave_id) if wave_id >= 0 else "global"
            family = event_family(event["name"])
            display_name = f"wave_{wave_label}_{family}_{event['wave_index']}"
            wave_metadata = {
                "wave_id": wave_id if wave_id >= 0 else None,
                "wave_scope": "shared" if "_SHARED_" in family else ("routed" if wave_id >= 0 else "global"),
                "wave_index": event["wave_index"],
                "event_family": family,
                "raw_event_name": event["name"],
                "display_index_semantics": "per_core_per_family_per_wave_task_or_interval",
            }
        result.append(
            {
                "ph": "X",
                "cat": "dependency_wait"
                if "WAIT" in event["name"]
                else ("aic_op" if event["core_type"] == "AIC" else "aiv_op"),
                "pid": f"rank{event['rank']}",
                "tid": (f"group_{event['group_id']}_{event['core_type']}_{event['sub_id']}"),
                "name": display_name,
                "ts": round((event["start_cycle"] - base_cycle) / CYCLE_TO_US, 6),
                "dur": round(
                    (event["end_cycle"] - event["start_cycle"]) / CYCLE_TO_US,
                    6,
                ),
                "args": {
                    "rank": event["rank"],
                    "group_id": event["group_id"],
                    "core_type": event["core_type"],
                    "sub_id": event["sub_id"],
                    "input_sync_mode": input_sync_mode,
                    "include_wait": include_wait,
                    "time_origin": time_origin,
                    "dynamic_index_semantics": (
                        f"per_core_{granularity}" if granularity else "per_core_interval_not_expert_or_tile"
                    ),
                    **task,
                    **wave_metadata,
                },
            }
        )
    return result


def event_family(name: str) -> str:
    name = re.sub(r"^wave_(?:\d+|global)_", "", name)
    return re.sub(r"_\d+$", "", name)


def write_summary(events, path: Path) -> None:
    grouped = defaultdict(list)
    for event in events:
        grouped[(event["rank"], event_family(event["name"]))].append(event)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "rank",
                "event_family",
                "event_count",
                "first_start_cycle",
                "last_end_cycle",
                "envelope_us",
                "sum_active_us",
                "max_event_us",
            ]
        )
        for (rank, family), items in sorted(grouped.items()):
            first = min(item["start_cycle"] for item in items)
            last = max(item["end_cycle"] for item in items)
            durations = [(item["end_cycle"] - item["start_cycle"]) / CYCLE_TO_US for item in items]
            writer.writerow(
                [
                    rank,
                    family,
                    len(items),
                    first,
                    last,
                    f"{(last - first) / CYCLE_TO_US:.3f}",
                    f"{sum(durations):.3f}",
                    f"{max(durations):.3f}",
                ]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timer-dir", type=Path, default=Path("timer"))
    parser.add_argument("--output-dir", type=Path, default=Path("timer_json"))
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, default=7)
    parser.add_argument("--output", default="all_rank_trace.json")
    parser.add_argument(
        "--input-sync-mode",
        choices=("keep", "hide", "compress"),
        default="keep",
    )
    parser.add_argument(
        "--include-wait",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show dependency WAIT events in JSON traces and summary (hidden by default)",
    )
    parser.add_argument(
        "--time-origin",
        choices=("sync", "global", "rank"),
        default="sync",
        help=(
            "sync removes per-NPU cycle offsets using the early input cross-rank "
            "sync (default); global exposes raw device-cycle offsets; rank "
            "independently zeros each rank"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_by_rank = {}
    for rank in range(args.rank_start, args.rank_end + 1):
        path = args.timer_dir / f"rank_{rank}_npu_time_parsed.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        raw_by_rank[rank] = raw_events_for_rank(path, rank)

    global_base = first_start_cycle([event for events in raw_by_rank.values() for event in events])
    rank_bases = {rank: first_start_cycle(events) for rank, events in raw_by_rank.items()}
    sync_bases = None
    sync_anchors = None
    if args.time_origin == "sync":
        sync_bases, sync_anchors = synchronized_rank_bases(raw_by_rank)
    valid_rank_bases = [cycle for cycle in rank_bases.values() if cycle > 0]
    if valid_rank_bases:
        spread_us = (max(valid_rank_bases) - min(valid_rank_bases)) / CYCLE_TO_US
        print(f"[TIMER] time origin={args.time_origin}; rank first-event spread={spread_us:.3f} us")
    if sync_anchors:
        anchor_spread_us = (max(sync_anchors.values()) - min(sync_anchors.values())) / CYCLE_TO_US
        print(f"[TIMER] aligned on median {SYNC_ANCHOR_EVENT} end; raw anchor spread={anchor_spread_us:.3f} us")

    merged_raw = []
    merged_trace = []
    for rank, raw in raw_by_rank.items():
        selected_raw = apply_wait_mode(raw, args.include_wait)
        if args.time_origin == "sync":
            base_cycle = sync_bases[rank]
        elif args.time_origin == "global":
            base_cycle = global_base
        else:
            base_cycle = rank_bases[rank]
        trace = trace_events(
            raw,
            args.input_sync_mode,
            base_cycle,
            args.include_wait,
            args.time_origin,
        )
        merged_raw.extend(selected_raw)
        merged_trace.extend(trace)
        output = args.output_dir / f"rank_{rank}_npu_time.json"
        output.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[TIMER] rank {rank}: {len(trace)} events -> {output}")
    merged_output = args.output_dir / args.output
    merged_output.write_text(
        json.dumps(merged_trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_output = args.output_dir / "timer_summary.csv"
    write_summary(merged_raw, summary_output)
    print(f"[TIMER] merged {len(merged_trace)} events -> {merged_output}")
    print(f"[TIMER] summary -> {summary_output}")


if __name__ == "__main__":
    main()
