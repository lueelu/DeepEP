#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Decode raw per-rank MegaMoe timer buffers into readable CSV files."""

import argparse
from array import array
import csv
import json
from pathlib import Path
import sys

from .protocol import (
    COUNTERS_PER_CORE,
    CYCLE_TO_US,
    DYNAMIC_ITER_PER_CORE,
    DYNAMIC_NAMES,
    FIXED_NAMES,
    MAX_DYNAMIC_ITER,
    N_CORE_COUNT,
    N_TIMING_COUNTER,
    TOTAL_BUFFER_SIZE,
    TASK_METADATA_OFFSET,
    WAVE_ID_OFFSET,
    WAVE_ID_PER_CORE,
    task_metadata,
)


def load_raw(path: Path):
    # 原始二进制采用 little-endian int64，避免数千万 Python int 的内存开销。
    values = array("q")
    if path.suffix == ".bin":
        if path.stat().st_size != TOTAL_BUFFER_SIZE * 8:
            raise ValueError(f"{path}: invalid binary timer size; expected {TOTAL_BUFFER_SIZE * 8} bytes")
        with path.open("rb") as stream:
            values.fromfile(stream, TOTAL_BUFFER_SIZE)
        if sys.byteorder != "little":
            values.byteswap()
    else:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                values.extend(int(item.strip()) for item in row if item.strip())
    if len(values) != TOTAL_BUFFER_SIZE:
        raise ValueError(
            f"{path}: got {len(values)} int64 values, expected "
            f"{TOTAL_BUFFER_SIZE}; rebuild kernels/extensions and keep the "
            "C++ and Python timer schemas synchronized"
        )
    return values


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


def event_names(dynamic_counts: list[int]) -> list[tuple[str, int]]:
    result = [(name, index) for index, name in enumerate(FIXED_NAMES)]
    for type_id, prefix in enumerate(DYNAMIC_NAMES):
        count = max(dynamic_counts[core * DYNAMIC_ITER_PER_CORE + type_id] for core in range(N_CORE_COUNT))
        if count < 0 or count > MAX_DYNAMIC_ITER:
            raise ValueError(
                f"{prefix}: device reported {count} tasks, but the trace "
                f"capacity is {MAX_DYNAMIC_ITER}; increase MAX_DYNAMIC_ITER "
                "in both schemas and rebuild"
            )
        result.extend(
            (
                f"{prefix}_{task_index}",
                len(FIXED_NAMES) + type_id * MAX_DYNAMIC_ITER + task_index,
            )
            for task_index in range(count)
        )
    return result


def decode(raw_path: Path, output_path: Path) -> None:
    raw = load_raw(raw_path)
    counters = memoryview(raw)[:N_TIMING_COUNTER]
    dynamic_counts = raw[N_TIMING_COUNTER:TASK_METADATA_OFFSET]
    for core in range(N_CORE_COUNT):
        for type_id, name in enumerate(DYNAMIC_NAMES):
            count = dynamic_counts[core * DYNAMIC_ITER_PER_CORE + type_id]
            if not 0 <= count <= MAX_DYNAMIC_ITER:
                raise ValueError(f"{raw_path}: core={core} {name} count={count}; rebuild/increase timer capacity")
    names = event_names(dynamic_counts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["core_id", "group_id", "core_type", "sub_id"]
        for name, _ in names:
            header.extend(
                (
                    f"{name}_start_cycle",
                    f"{name}_end_cycle",
                    f"{name}_duration(us)",
                    f"{name}_task",
                    f"{name}_wave_id",
                )
            )
        header.extend(f"{name}_task_count" for name in DYNAMIC_NAMES)
        writer.writerow(header)
        for core_id in range(N_CORE_COUNT):
            group_id, core_type, sub_id = core_info(core_id)
            row: list[object] = [core_id, group_id, core_type, sub_id]
            base = core_id * COUNTERS_PER_CORE
            for name, logical_index in names:
                start = counters[base + logical_index * 2]
                end = counters[base + logical_index * 2 + 1]
                validate_event_lane(name, core_id, core_type, start, end)
                duration = (end - start) / CYCLE_TO_US if end > start else 0.0
                task = task_metadata(raw, core_id, logical_index) if end > start > 0 else {}
                wave = raw[WAVE_ID_OFFSET + core_id * WAVE_ID_PER_CORE + logical_index]
                if end > start > 0 and wave < -1:
                    raise ValueError(f"{raw_path}: core={core_id} {name} invalid wave ID {wave}")
                row.extend(
                    (start, end, f"{duration:.3f}", json.dumps(task, separators=(",", ":")) if task else "", wave)
                )
            dynamic_base = core_id * DYNAMIC_ITER_PER_CORE
            row.extend(dynamic_counts[dynamic_base + type_id] for type_id in range(len(DYNAMIC_NAMES)))
            writer.writerow(row)
    print(f"[TIMER] parsed {raw_path} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("timer"))
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, default=7)
    args = parser.parse_args()
    for rank in range(args.rank_start, args.rank_end + 1):
        raw_path = args.dir / f"rank_{rank}_npu_time.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        decode(raw_path, args.dir / f"rank_{rank}_npu_time_parsed.csv")


if __name__ == "__main__":
    main()
