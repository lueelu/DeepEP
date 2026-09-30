# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

from pathlib import Path
import re
import struct

import pytest

from deep_ep_experimental.megamoe.timer import protocol as p
from deep_ep_experimental.megamoe.timer.parse_timer_csv import decode, load_raw
from deep_ep_experimental.megamoe.timer.trace_parser import (
    first_start_cycle,
    raw_events_for_rank,
    synchronized_rank_bases,
    trace_events,
)

ROOT = Path(__file__).resolve().parents[3]


def test_device_schema_and_extended_size():
    header = (ROOT / "experiments/megamoe/csrc/kernels/mega_moe/common/ascend_timer_v2.hpp").read_text(encoding="utf-8")

    def names(enum, sentinel):
        body = re.search(r"enum\s+" + enum + r"\s*\{(.*?)\};", header, re.S)[1]
        return [item.split("=")[0].strip() for item in body.split(",") if item.strip() != sentinel]

    assert names("FixedTiming", "FIXED_TIMING_COUNT") == p.FIXED_NAMES
    assert names("DynamicTimingType", "DYNAMIC_TYPE_COUNT_ENUM") == p.DYNAMIC_NAMES
    for name in ("MAX_DYNAMIC_ITER", "N_CORE_COUNT"):
        assert int(re.search(rf"{name}\s*=\s*(\d+)", header)[1]) == getattr(p, name)
    assert p.TOTAL_BUFFER_SIZE == p.WAVE_ID_OFFSET + p.WAVE_ID_PER_CORE * p.N_CORE_COUNT
    assert p.TASK_METADATA_SIZE == p.N_CORE_COUNT * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER * 2


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "rank_0_npu_time.bin"
    with path.open("wb") as stream:
        stream.truncate(p.TOTAL_BUFFER_SIZE * 8)

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        for core, name, start, end in (
            (0, "AIC_GMM1", 102000, 107000),
            (0, "AIC_GMM1_WAIT", 100000, 102000),
            (1, "AIV_GMM1_PROLOGUE", 101000, 105000),
            (2, "AIV_DISPATCH", 100000, 103000),
        ):
            family = p.DYNAMIC_NAMES.index(name)
            logical = len(p.FIXED_NAMES) + family * p.MAX_DYNAMIC_ITER
            put(core * p.COUNTERS_PER_CORE + logical * 2, start, end)
            put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family, 1)
            metadata = (
                p.TASK_METADATA_OFFSET
                + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + family * p.MAX_DYNAMIC_ITER) * 2
            )
            put(metadata, (3 << 32) | 256, (2 << 32) | 96)
            put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical, 4)
    return path


def test_binary_to_csv_to_trace_wait_metadata(raw_file):
    parsed = raw_file.with_suffix(".parsed.csv")
    decode(raw_file, parsed)
    events = raw_events_for_rank(parsed, 0)
    assert len(events) == 4
    assert {event["group_id"] for event in events} == {0}
    trace = trace_events(events, "keep", first_start_cycle(events), include_wait=False, time_origin="rank")
    assert len(trace) == 3
    compute = next(event for event in trace if event["args"]["event_family"] == "AIC_GMM1")
    assert compute["dur"] == 5 and compute["ts"] == 2
    assert compute["args"]["expert_id"] == 2
    assert compute["args"]["n_tile"] == 1
    assert compute["args"]["wave_id"] == 4
    assert len(trace_events(events, "keep", first_start_cycle(events), include_wait=True)) == 4


def test_truncated_and_overflow_rejected(raw_file):
    with raw_file.open("r+b") as stream:
        stream.seek(p.N_TIMING_COUNTER * 8)
        stream.write(struct.pack("<q", p.MAX_DYNAMIC_ITER + 1))
    with pytest.raises(ValueError, match="count"):
        decode(raw_file, raw_file.with_suffix(".csv"))
    with raw_file.open("r+b") as stream:
        stream.truncate(8)
    with pytest.raises(ValueError, match="size"):
        load_raw(raw_file)


def test_sync_alignment_is_explicit_and_requires_anchor():
    with pytest.raises(ValueError, match="missing"):
        synchronized_rank_bases({0: [{"name": "AIC_GMM1", "start_cycle": 100, "end_cycle": 200}]})
