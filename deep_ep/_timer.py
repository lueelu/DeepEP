# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026
# Modified by xixing 2026

"""Pure-Python helpers shared by MegaMoE device timer exporters."""

import csv
import re
from collections import defaultdict
from pathlib import Path

CYCLE_TO_US = 1000.0


def core_info(core_id: int) -> tuple[int, str, int]:
    group_id, lane = divmod(core_id, 3)
    return (
        (group_id, "AIC", 0)
        if lane == 0
        else (
            group_id,
            "AIV",
            lane - 1,
        )
    )


def validate_event_lane(name: str, core_id: int, core_type: str, start: int, end: int) -> None:
    if start <= 0 or end <= start:
        return
    expected = "AIC" if name.startswith("AIC_") else ("AIV" if name.startswith("AIV_") else None)
    if expected is not None and core_type != expected:
        raise ValueError(
            f"{name}: nonzero event found in logical core {core_id} "
            f"({core_type}); expected {expected}. Rebuild with the corrected "
            "MIX_AIC_1_2 timer core mapping and recollect the raw trace."
        )


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(item for item in intervals if item[1] > item[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


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
