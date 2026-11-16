# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""Decode actual training coordinates and emit rank-local Chrome/Perfetto traces."""

from array import array
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

from deep_ep._timer import core_info, merge_intervals, validate_event_lane, write_summary

from . import protocol as p


def decode(path, rank):
    path = Path(path)
    raw = array("q")
    if path.stat().st_size < p.HEADER_WORDS * 8 or path.stat().st_size % 8:
        raise ValueError(f"{path}: truncated training Timer")
    with path.open("rb") as stream:
        raw.fromfile(stream, path.stat().st_size // 8)
    if sys.byteorder != "little":
        raw.byteswap()
    header = list(raw[: p.HEADER_WORDS])
    if header[:2] != [p.MAGIC, p.VERSION]:
        raise ValueError(f"{path}: unsupported training Timer protocol")
    if header[4] not in (0, 1) or header[5:8] != [len(p.FIXED_NAMES), len(p.DYNAMIC_NAMES), p.METADATA_WORDS]:
        raise ValueError(f"{path}: inconsistent training Timer event schema")
    layout = p.layout(header[2], header[3])
    if len(raw) != layout["elements"]:
        raise ValueError(f"{path}: training Timer size differs from header")
    phase = "forward" if header[4] == 0 else "backward"
    events = []
    for core in range(layout["cores"]):
        group, engine, sub = core_info(core)
        slots = [(name, i, None, None) for i, name in enumerate(p.FIXED_NAMES)]
        for family, name in enumerate(p.DYNAMIC_NAMES):
            count = raw[layout["counts"] + core * layout["count_stride"] + family]
            if not 0 <= count <= layout["capacity"]:
                raise ValueError(f"rank={rank} core={core} {name}: Timer overflow ({count})")
            slots.extend((name, len(p.FIXED_NAMES) + family * layout["capacity"] + i, family, i) for i in range(count))
        for name, slot, family, index in slots:
            offset = p.HEADER_WORDS + core * layout["stride"] + slot * 2
            start, end = raw[offset : offset + 2]
            if start == end == 0:
                continue
            if not 0 < start <= end:
                raise ValueError(f"rank={rank} core={core} {name}: invalid timestamp pair")
            validate_event_lane(name, core, engine, start, end)
            task = {}
            if family is not None:
                offset = (
                    layout["metadata"]
                    + ((core * len(p.DYNAMIC_NAMES) + family) * layout["capacity"] + index) * p.METADATA_WORDS
                )
                first, second, tile, depth, columns, context, parent = raw[offset : offset + p.METADATA_WORDS]
                mask = (1 << 32) - 1
                if first or second:
                    task = dict(
                        expert_id=(first >> 32) - 1,
                        row_start=first & mask,
                        row_count=second & mask,
                        n_tile=(second >> 32) - 1,
                        tile_m=tile >> 32,
                        tile_n=tile & mask,
                        k_start=depth >> 32,
                        k_count=depth & mask,
                        n_start=columns >> 32,
                        n_count=columns & mask,
                        peer_rank=(context >> 32) - 1,
                        task_flags=context & mask,
                        parent_family=(parent >> 32) - 1,
                        parent_slot=(parent & mask) - 1,
                    )
                    if task["expert_id"] < 0 or task["row_count"] <= 0:
                        raise ValueError(f"rank={rank} core={core}: invalid task coordinates")
                    task["m_tile"] = task["row_start"] // task["tile_m"] if task["tile_m"] else None
                    task["coordinate_system"] = (
                        "weight_gradient" if name in ("AIC_BW_DW1", "AIC_BW_DW2") else "expert_rows"
                    )
                if name in ("AIC_SYNC_WAIT", "AIV_SYNC_WAIT"):
                    task = {}
            events.append(
                dict(
                    name=name,
                    rank=rank,
                    phase=phase,
                    core_id=core,
                    group_id=group,
                    core_type=engine,
                    sub_id=sub,
                    start_cycle=start,
                    end_cycle=end,
                    slot=index,
                    task=task,
                )
            )
    if not any(e["name"] == "KERNEL_TIMING" for e in events):
        raise ValueError(f"{path}: no recorded kernel interval")
    return events, dict(
        protocol_version=header[1],
        phase=phase,
        lanes=header[2],
        capacity=header[3],
        bytes=len(raw) * 8,
        max_recv_tokens=header[8],
        M=header[9],
        H=header[10],
        F=header[11],
        experts_per_rank=header[12],
        topk=header[13],
        topk_weight_dtype="fp32" if header[14] else "bf16",
        path_flags=header[15],
    )


def chrome_events(events, origin, *, include_wait=True):
    # A task envelope can contain more than one real dependency wait. Cut
    # those intervals from its displayed work without moving other events.
    waits = defaultdict(list)
    for event in events:
        if event["name"].endswith("WAIT"):
            waits[event["core_id"]].append((event["start_cycle"], event["end_cycle"]))
    result = []
    for event in events:
        name = event["name"]
        is_wait = name.endswith("WAIT")
        if is_wait and not include_wait:
            continue
        intervals = [(event["start_cycle"], event["end_cycle"])]
        if event["slot"] is not None and not is_wait:
            for left, right in merge_intervals(waits[event["core_id"]]):
                intervals = [(a, min(b, left)) for a, b in intervals if a < left] + [
                    (max(a, right), b) for a, b in intervals if b > right
                ]
        for start, end in intervals:
            if end <= start:
                continue
            task = event["task"]
            label = name + (f" e{task['expert_id']} m{task['row_start']} n{task['n_start']}" if task else "")
            result.append(
                dict(
                    name=label,
                    cat="wait" if is_wait else "stage",
                    ph="X",
                    pid=event["rank"],
                    tid=f"group{event['group_id']}/{event['core_type']}{event['sub_id']}",
                    ts=(start - origin) / p.CYCLE_TO_US,
                    dur=(end - start) / p.CYCLE_TO_US,
                    args=dict(
                        event_family=name,
                        phase=event["phase"],
                        rank=event["rank"],
                        core_id=event["core_id"],
                        task_id=event["slot"],
                        start_cycle=start,
                        end_cycle=end,
                        **task,
                    ),
                )
            )
    return result


def expert_events(events, origin, *, include_wait=False):
    """One expert-stage interval per physical lane, using the existing trace naming."""
    wait_ends = {}
    for event in events:
        task = event["task"]
        parent, slot = task.get("parent_family", -1), task.get("parent_slot", -1)
        if event["name"].endswith("WAIT") and parent >= 0 and slot >= 0:
            key = (event["phase"], event["core_id"], p.DYNAMIC_NAMES[parent], slot)
            wait_ends[key] = max(wait_ends.get(key, 0), event["end_cycle"])

    grouped = defaultdict(list)
    for event in events:
        name = event["name"]
        if name == "KERNEL_TIMING" or name.endswith("WAIT"):
            continue
        if name in ("AIV_SAVE_GATE_UP", "AIV_BW_DECODE_GATE_UP"):
            continue
        start = event["start_cycle"]
        if event["slot"] is not None:
            # Recorded tile dependencies precede the matrix/activation/copy
            # operation. Preparation before those waits stays in the detail view.
            start = max(start, wait_ends.get((event["phase"], event["core_id"], name, event["slot"]), start))
        if start >= event["end_cycle"]:
            continue
        expert = event["task"].get("expert_id", -1)
        grouped[(event["phase"], event["core_id"], name, expert)].append(dict(event, start_cycle=start))

    result = []
    for (_, _, name, expert), values in grouped.items():
        event = values[0]
        start = min(value["start_cycle"] for value in values)
        end = max(value["end_cycle"] for value in values)
        args = dict(
            rank=event["rank"],
            phase=event["phase"],
            group_id=event["group_id"],
            core_type=event["core_type"],
            core_id=event["core_id"],
            sub_id=event["sub_id"],
            event_family=name,
            start_cycle=start,
            end_cycle=end,
            is_wait=False,
        )
        if expert >= 0:
            args.update(expert_id=expert, granularity="expert", recorded_tasks=len(values))
        result.append(
            dict(
                ph="X",
                cat="aic_op" if event["core_type"] == "AIC" else "aiv_op",
                pid=f"rank{event['rank']}",
                tid=f"group_{event['group_id']}_{event['core_type']}_{event['sub_id']}",
                name=f"{name}_{expert}" if expert >= 0 else name,
                ts=(start - origin) / p.CYCLE_TO_US,
                dur=(end - start) / p.CYCLE_TO_US,
                args=args,
            )
        )
    if include_wait:
        first_slots = {
            (phase, core, name, expert): min(value["slot"] for value in values if value["slot"] is not None)
            for (phase, core, name, expert), values in grouped.items()
            if any(value["slot"] is not None for value in values)
        }
        for event in chrome_events(events, origin):
            if event["cat"] != "wait":
                continue
            core = event["args"]["core_id"]
            group, engine, sub = core_info(core)
            family = event["args"]["event_family"]
            if family.endswith("_UB_WAIT"):
                continue
            expert = event["args"].get("expert_id", -1)
            parent = event["args"].get("parent_family", -1)
            if parent >= 0:
                key = (event["args"]["phase"], core, p.DYNAMIC_NAMES[parent], expert)
                if event["args"].get("parent_slot") != first_slots.get(key):
                    continue
            event.update(
                pid=f"rank{event['args']['rank']}",
                tid=f"group_{group}_{engine}_{sub}",
                name=f"{family}_{expert}" if expert >= 0 else family,
                cat="dependency_wait",
            )
            event["args"]["is_wait"] = True
            event["args"]["wait_scope"] = "expert_entry" if expert >= 0 else "shared_stage"
            result.append(event)
    return sorted(result, key=lambda event: (event["args"]["core_id"], event["ts"], event["name"]))


def export_rank(directory, rank, *, include_wait=False):
    directory = Path(directory)
    phases = {}
    for phase in ("forward", "backward"):
        path = directory / f"rank_{rank}_{phase}_time.bin"
        if path.exists():
            events, info = decode(path, rank)
            if info["phase"] != phase:
                raise ValueError(f"{path}: phase does not match filename")
            phases[phase] = events
            (directory / f"rank_{rank}_{phase}_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            with (directory / f"rank_{rank}_{phase}_parsed.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["rank", "core", "phase", "event", "start_cycle", "end_cycle", "task"])
                writer.writerows(
                    (rank, e["core_id"], phase, e["name"], e["start_cycle"], e["end_cycle"], json.dumps(e["task"]))
                    for e in events
                )
    if not phases:
        raise ValueError(f"rank={rank}: no raw training Timer files")
    all_events = [e for events in phases.values() for e in events]
    origin = min(e["start_cycle"] for e in all_events if e["name"] == "KERNEL_TIMING")
    detail = chrome_events(all_events, origin)
    combined = expert_events(all_events, origin, include_wait=include_wait)
    for phase in phases:
        selected = [event for event in combined if event["args"]["phase"] == phase]
        selected_detail = [event for event in detail if event["args"]["phase"] == phase]
        summary_events = [
            dict(
                rank=rank,
                name=e["args"]["event_family"],
                start_cycle=e["args"]["start_cycle"],
                end_cycle=e["args"]["end_cycle"],
            )
            for e in selected_detail
        ]
        write_summary(summary_events, directory / f"rank_{rank}_{phase}_summary.csv")
        lanes = defaultdict(lambda: defaultdict(list))
        for event in selected_detail:
            lanes[event["args"]["event_family"]][event["args"]["core_id"]].append(
                (event["args"]["start_cycle"], event["args"]["end_cycle"])
            )
        breakdown = {
            name: dict(
                max_lane_active_us=max(
                    sum(b - a for a, b in merge_intervals(intervals)) / p.CYCLE_TO_US for intervals in by_core.values()
                )
            )
            for name, by_core in lanes.items()
        }
        (directory / f"rank_{rank}_{phase}_breakdown.json").write_text(
            json.dumps(breakdown, indent=2), encoding="utf-8"
        )
        (directory / f"rank_{rank}_{phase}_trace.json").write_text(
            json.dumps(selected, ensure_ascii=False), encoding="utf-8"
        )
        (directory / f"rank_{rank}_{phase}_detail_trace.json").write_text(
            json.dumps(selected_detail, ensure_ascii=False), encoding="utf-8"
        )
    (directory / f"rank_{rank}_trace.json").write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    return combined
