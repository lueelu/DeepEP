# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

from pathlib import Path
import re
import struct
import csv
import json

import pytest

from deep_ep_experimental.megamoe.timer import protocol as p
from deep_ep_experimental.megamoe.timer.parse_timer_csv import decode, load_raw
from deep_ep_experimental.megamoe.timer.trace_parser import (
    first_start_cycle,
    raw_events_for_rank,
    synchronized_rank_bases,
    trace_events,
    write_startup_breakdown,
)

ROOT = Path(__file__).resolve().parents[3]


def test_device_schema_and_extended_size():
    header = (ROOT / "experiments/megamoe/csrc_ep32/kernels/mega_moe/common/ascend_timer_v2.hpp").read_text(
        encoding="utf-8"
    )

    def names(enum, sentinel):
        body = re.search(r"enum\s+" + enum + r"\s*\{(.*?)\};", header, re.S)[1]
        return [item.split("=")[0].strip() for item in body.split(",") if item.strip() != sentinel]

    assert names("FixedTiming", "FIXED_TIMING_COUNT") == p.FIXED_NAMES
    assert names("DynamicTimingType", "DYNAMIC_TYPE_COUNT_ENUM") == p.DYNAMIC_NAMES
    for name in ("MAX_DYNAMIC_ITER", "N_CORE_COUNT", "COUNT_SNAPSHOT_MAX_WORDS"):
        assert int(re.search(rf"{name}\s*=\s*(\d+)", header)[1]) == getattr(p, name)
    assert names("CountDiagnostic", "COUNT_DIAGNOSTIC_COUNT") == p.COUNT_DIAGNOSTIC_NAMES
    assert p.COUNT_SNAPSHOT_OFFSET % 4 == 0
    assert p.TOTAL_BUFFER_SIZE == p.COUNT_SNAPSHOT_OFFSET + p.COUNT_SNAPSHOT_SLOTS_PER_CORE * p.N_CORE_COUNT
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


def test_startup_details_and_combine_decisions_round_trip(raw_file, tmp_path):
    core = 2
    with raw_file.open("r+b") as stream:

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        fixed = p.FIXED_NAMES.index("AIV_COUNT_METADATA_WAIT")
        put(core * p.COUNTERS_PER_CORE + fixed * 2, 100000, 120000)
        stats = dict(
            COUNT_PROFILE_VALID=1,
            COUNT_SNAPSHOT_ATTEMPTS=3,
            COUNT_SNAPSHOT_FAILURES=2,
            COUNT_DMA_CYCLES=6000,
            COUNT_CHECK_CYCLES=9000,
            COUNT_BACKOFF_CYCLES=400,
            COUNT_SUCCESS_DMA_CYCLES=2000,
            COUNT_SUCCESS_CHECK_CYCLES=3000,
            COUNT_LAST_STALE_WORDS=2,
            COUNT_LAST_STALE_SOURCE_MASK=(1 << 7) | (1 << 23),
            COUNT_LAST_STALE_SOURCE_PLUS_ONE=8,
            COUNT_LAST_STALE_EXPERT_PLUS_ONE=6,
            COUNT_LAST_OBSERVED_EPOCH=128,
            COUNT_EXPECTED_EPOCH=129,
            COUNT_DIAGNOSTIC_CYCLES=700,
            COUNT_SNAPSHOT_BYTES=2048,
            COUNT_SNAPSHOT_SOURCES=32,
            COUNT_SNAPSHOT_EXPERTS=12,
            COUNT_CAPTURED_SNAPSHOT_WORDS=512,
        )
        snapshot = [129 << 24 if i % 16 < 12 else 0 for i in range(512)]
        snapshot[7 * 16 + 5] = 128 << 24
        snapshot[23 * 16 + 11] = 130 << 24
        stream.seek((p.COUNT_SNAPSHOT_OFFSET + core * p.COUNT_SNAPSHOT_SLOTS_PER_CORE) * 8)
        stream.write(struct.pack("<512I", *snapshot))
        put(
            p.COUNT_DIAGNOSTIC_OFFSET + core * len(p.COUNT_DIAGNOSTIC_NAMES),
            *(stats.get(name, 0) for name in p.COUNT_DIAGNOSTIC_NAMES),
        )
        for family, records in (
            (
                "AIV_RELAY_STEP",
                [
                    ((0 << 32) | 31, (168 << 32) | 12),
                    ((5 << 32) | 31, (168 << 32) | 12),
                    ((7 << 32) | 2, 512),
                    ((15 << 32) | 19, (8 << 32) | 3),
                    ((15 << 32) | 3, (0xF0000 << 32) | 0x80000008),
                ],
            ),
            ("AIV_COMBINE_RUN", [((7 << 32) | 1, 0x302), ((7 << 32) | 16, 0x401)]),
            ("AIV_COMBINE_REDUCE", [((4 << 32) | 31, (0x85 << 32) | 3584)]),
            (
                "AIV_COMBINE_ASYNC",
                [
                    ((4 << 32) | 17, 1),
                    ((7 << 32) | 3, (1 << 32) | 1),
                    ((7 << 32) | 3, (1 << 32) | 2),
                    ((7 << 32) | 3, (1 << 32) | 3),
                ],
            ),
            ("AIV_COMBINE_READY", [((4 << 32) | 31, (0x85 << 32) | 0x81), ((4 << 32) | 31, (0x85 << 32) | 0x85)]),
            ("AIV_COMBINE_SYNC", [((4 << 32) | 1, 3), ((4 << 32) | 4, 0)]),
        ):
            type_id = p.DYNAMIC_NAMES.index(family)
            put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + type_id, len(records))
            for index, words in enumerate(records):
                slot = type_id * p.MAX_DYNAMIC_ITER + index
                logical = len(p.FIXED_NAMES) + slot
                put(core * p.COUNTERS_PER_CORE + logical * 2, 110000 + index * 1000, 110500 + index * 1000)
                put(p.TASK_METADATA_OFFSET + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + slot) * 2, *words)
    parsed = raw_file.with_suffix(".parsed.csv")
    decode(raw_file, parsed)
    raw = raw_events_for_rank(parsed, 0)
    visible = trace_events(raw, "keep", first_start_cycle(raw), include_wait=True)
    count = next(e for e in visible if e["args"].get("event_family") == "AIV_COUNT_METADATA_WAIT")
    assert count["args"]["snapshot_retries"] == 2
    assert count["args"]["last_failed_sources"] == [7, 23]
    assert count["args"]["failed_snapshot_available"] is True
    assert count["args"]["last_failed_local_expert_id"] == 5
    assert count["args"]["snapshot_dma_us"] == 6 and count["args"]["diagnostic_us"] == 0.7
    relay = [e for e in visible if e["args"].get("granularity") == "relay_step"]
    assert [e["args"]["relay_step"] for e in relay] == [
        "packet_wait",
        "count_publish",
        "clos_copy",
        "dispatch_flag_check",
        "dispatch_flag_check",
    ]
    assert relay[2]["args"]["source_host"] == 1 and relay[2]["args"]["bytes"] == 512
    assert relay[3]["args"]["consumer_rank"] == 18 and relay[3]["args"]["rank"] == 0
    assert relay[4]["args"]["source_ranks"] == [16, 17, 18, 19] and relay[4]["args"]["row_count"] == 8
    assert relay[0]["args"]["source_rank"] == 30 and relay[0]["args"]["first_expert"] == 168
    runs = [e["args"] for e in visible if e["args"].get("granularity") == "combine_run"]
    assert runs[0]["decision"] == "defer" and runs[0]["stop_reason"] == "producer_not_ready"
    assert runs[1]["run_rows"] == 16 and runs[1]["already_deferred"]
    reduced = next(e["args"] for e in visible if e["args"].get("granularity") == "combine_reduce")
    assert (reduced["qp_index"], reduced["token_index"], reduced["expert_rows"], reduced["hidden"]) == (4, 31, 3, 3584)
    async_row = next(e["args"] for e in visible if e["args"].get("granularity") == "combine_async")
    assert (async_row["expert_id"], async_row["row_start"], async_row["scope"]) == (3, 17, "row_lifetime")
    assert next(e["tid"] for e in visible if e["args"].get("granularity") == "combine_async").endswith("/COMBINE_ASYNC")
    phases = [e for e in visible if e["args"].get("granularity") == "combine_async" and "buffer_slot" in e["args"]]
    assert [e["args"]["scope"] for e in phases] == ["row_lifetime", "issue", "reuse_wait"]
    assert phases[0]["args"]["observation"] == "software_retirement"
    assert phases[0]["tid"].endswith("/COMBINE_ASYNC/slot0")
    assert phases[1]["tid"].endswith("/COMBINE_ISSUE")
    assert phases[2]["tid"].endswith("/COMBINE_REUSE_WAIT")
    readiness = [e["args"] for e in visible if e["args"].get("granularity") == "combine_ready"]
    assert [e["token_ready"] for e in readiness] == [False, True]
    sync = [e["args"] for e in visible if e["args"].get("granularity") == "combine_sync"]
    assert [(e["sync_phase"], e["pending_rows"]) for e in sync] == [
        ("before_export_sync", 3),
        ("after_host_barrier", 0),
    ]
    hidden = trace_events(raw, "keep", first_start_cycle(raw), include_wait=False)
    assert not any(e["args"].get("relay_step") == "packet_wait" for e in hidden)
    assert any(e["args"].get("relay_step") == "count_publish" for e in hidden)
    report = tmp_path / "startup.csv"
    write_startup_breakdown({0: raw}, report)
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    count_row = next(row for row in rows if row["event_family"] == "AIV_COUNT_METADATA_WAIT")
    assert json.loads(count_row["last_failed_sources"]) == [7, 23]
    assert count_row["snapshot_retries"] == "2"


def test_sync_alignment_is_explicit_and_requires_anchor():
    with pytest.raises(ValueError, match="missing"):
        synchronized_rank_bases({0: [{"name": "AIC_GMM1", "start_cycle": 100, "end_cycle": 200}]})


def test_unavailable_failed_snapshot_does_not_report_an_empty_source_set():
    result = p.failed_count_snapshot([], 2, {"COUNT_CAPTURED_SNAPSHOT_WORDS": 0})
    assert result["failed_snapshot_available"] is False
    assert result["last_failed_sources"] is None
    assert result["last_failed_source_mask"] is None


def test_rdma_binary_metadata_and_wait_visibility(raw_file):
    core = 1
    with raw_file.open("r+b") as stream:

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        for phase in range(3):
            for name, commit, size in (
                ("AIV_RDMA_POST", 0, 3584),
                ("AIV_RDMA_QUIET_WAIT", 0, 0),
                ("AIV_RDMA_RECV_WAIT", 1, 4),
            ):
                family = p.DYNAMIC_NAMES.index(name)
                slot = family * p.MAX_DYNAMIC_ITER + phase
                logical = len(p.FIXED_NAMES) + slot
                put(core * p.COUNTERS_PER_CORE + logical * 2, 101000 + phase * 1000, 101500 + phase * 1000)
                put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family, 3)
                put(
                    p.TASK_METADATA_OFFSET + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + slot) * 2,
                    (17 << 32) | (phase + 1),
                    (commit << 32) | size,
                )
                put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical, p.NO_WAVE)
    parsed = raw_file.with_suffix(".parsed.csv")
    decode(raw_file, parsed)
    raw = raw_events_for_rank(parsed, 0)
    base = first_start_cycle(raw)
    hidden = trace_events(raw, "keep", base)
    shown = trace_events(raw, "keep", base, include_wait=True)
    assert len(hidden) == 6 and len(shown) == 13
    rdma = [e for e in shown if e["args"].get("granularity") == "rdma"]
    assert len(rdma) == 9
    assert {e["args"]["phase"] for e in rdma} == set(p.RDMA_PHASES)
    for event in rdma:
        args = event["args"]
        assert event["tid"] == "group_0_AIV_0/RDMA"
        assert args["peer_rank"] == 16 and args["wave_id"] is None
        assert "expert_id" not in args and "n_tile" not in args
        if args["event_family"] == "AIV_RDMA_POST":
            assert args["kind"] == "data" and args["bytes"] == 3584
            assert event["cat"] == "rdma_post"
        else:
            assert event["cat"] == "rdma_wait"
    for event in hidden:
        counterpart = next(e for e in shown if e["name"] == event["name"] and e["tid"] == event["tid"])
        assert (event["ts"], event["dur"]) == (counterpart["ts"], counterpart["dur"])


@pytest.mark.parametrize("first,second", [(0, 0), (33 << 32 | 1, 4), (1 << 32 | 4, 4), (1 << 32 | 1, 2 << 32)])
def test_rdma_invalid_metadata_rejected(first, second):
    class Metadata:
        def __getitem__(self, key):
            return first, second

    logical = len(p.FIXED_NAMES) + p.DYNAMIC_NAMES.index("AIV_RDMA_POST") * p.MAX_DYNAMIC_ITER
    with pytest.raises(ValueError, match="invalid RDMA metadata"):
        p.task_metadata(Metadata(), 1, logical)


def test_startup_roundtrip_keeps_waits_in_diagnostic_csv(raw_file):
    phases = [
        (2, "AIV_GATEWAY_PAYLOAD_BROADCAST", 100000, 100400),
        (3, "AIC_MOE_STARTUP", 100000, 110000),
        (3, "AIC_COUNT_READY_WAIT", 101000, 109000),
        (5, "AIV_MOE_STARTUP", 100000, 110000),
        (5, "AIV_COUNT_TABLE_PREPARE", 100500, 109500),
        (5, "AIV_COUNT_METADATA_WAIT", 101000, 104000),
        (5, "AIV_GATEWAY_PAYLOAD_CHECK", 104000, 107000),
        (5, "AIV_COUNT_PREFIX_BUILD", 107000, 108000),
        (5, "AIV_COUNT_TABLE_PUBLISH", 108000, 109000),
    ]
    with raw_file.open("r+b") as stream:

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        for core, name, start, end in phases:
            slot = p.FIXED_NAMES.index(name)
            put(core * p.COUNTERS_PER_CORE + slot * 2, start, end)
            put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + slot, p.NO_WAVE)
        core = 5
        family = p.DYNAMIC_NAMES.index("AIV_GATEWAY_PAYLOAD_WAIT")
        slot = len(p.FIXED_NAMES) + family * p.MAX_DYNAMIC_ITER
        put(core * p.COUNTERS_PER_CORE + slot * 2, 104500, 106500)
        put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + slot, p.NO_WAVE)
        put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family, 1)
        put(
            p.TASK_METADATA_OFFSET + (core * len(p.DYNAMIC_NAMES) + family) * p.MAX_DYNAMIC_ITER * 2,
            (18 << 32) | 2,
            130,
        )
    parsed = raw_file.with_suffix(".parsed.csv")
    decode(raw_file, parsed)
    raw = raw_events_for_rank(parsed, 0)
    base = first_start_cycle(raw)
    shown = trace_events(raw, "keep", base, include_wait=True)
    startup = [e for e in shown if e["tid"].endswith("/STARTUP")]
    assert len(startup) == 10
    gateway = next(e for e in startup if e["args"].get("granularity") == "startup_dependency")
    assert gateway["name"] == "AIV_GATEWAY_PAYLOAD_WAIT_source17_gateway1_seq130"
    assert gateway["args"]["source_rank"] == 17 and gateway["args"]["gateway_rank"] == 1
    assert gateway["args"]["wave_id"] is None and "expert_id" not in gateway["args"]
    assert gateway["ts"] == 4.5 and gateway["dur"] == 2
    broadcast = next(e for e in startup if e["name"] == "AIV_GATEWAY_PAYLOAD_BROADCAST")
    assert broadcast["ts"] == 0 and broadcast["dur"] == 0.4
    hidden = trace_events(raw, "keep", base)
    assert not any("WAIT" in e["name"] for e in hidden)
    for event in hidden:
        counterpart = next(e for e in shown if e["name"] == event["name"] and e["tid"] == event["tid"])
        assert (event["ts"], event["dur"]) == (counterpart["ts"], counterpart["dur"])
    report = raw_file.with_name("startup_breakdown.csv")
    write_startup_breakdown({0: raw}, report)
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert sum("WAIT" in row["event_family"] for row in rows) == 3
    assert sum(row["scope"] == "envelope" for row in rows) == 4
    dependency = next(row for row in rows if row["event_family"] == "AIV_GATEWAY_PAYLOAD_WAIT")
    assert (dependency["source_rank"], dependency["gateway_rank"], dependency["sequence"]) == ("17", "1", "130")
    assert float(dependency["duration_us"]) == 2


@pytest.mark.parametrize("first,second", [(0, 0), (18 << 32 | 3, 130), (18 << 32 | 2, 0), (18 << 32 | 2, 1 << 32)])
def test_gateway_invalid_metadata_rejected(first, second):
    class Metadata:
        def __getitem__(self, key):
            return first, second

    logical = len(p.FIXED_NAMES) + p.DYNAMIC_NAMES.index("AIV_GATEWAY_PAYLOAD_WAIT") * p.MAX_DYNAMIC_ITER
    with pytest.raises(ValueError, match="invalid gateway wait metadata"):
        p.task_metadata(Metadata(), 2, logical)


@pytest.mark.parametrize("scan_cycles", [0, 2501, 2**53 + 17, 2**63 + 17, 2**64 - 1])
def test_combine_progress_metadata_preserves_unsigned_counters(scan_cycles):
    first = 128 | (120 << 16) | (4 << 32) | (1024 << 48)

    def signed(value):
        return value if value < 2**63 else value - 2**64

    class Metadata:
        def __getitem__(self, key):
            return signed(first), signed(scan_cycles)

    logical = len(p.FIXED_NAMES) + p.DYNAMIC_NAMES.index("AIV_COMBINE_PROGRESS") * p.MAX_DYNAMIC_ITER
    task = p.task_metadata(Metadata(), 1, logical)
    assert task == dict(
        granularity="combine_progress",
        pending_before=128,
        pending_after=120,
        posts=4,
        scanned_rows=1024,
        scan_control_cycles=scan_cycles,
        scan_control_us=scan_cycles / p.CYCLE_TO_US,
    )
    assert type(task["scan_control_cycles"]) is int


@pytest.mark.parametrize("family", ["AIV_COMBINE_PROGRESS", "AIV_COMBINE_DRAIN"])
def test_empty_combine_call_is_metadata_not_an_unidentified_task(family):
    class Metadata:
        def __getitem__(self, key):
            return 0, 0

    logical = len(p.FIXED_NAMES) + p.DYNAMIC_NAMES.index(family) * p.MAX_DYNAMIC_ITER
    task = p.task_metadata(Metadata(), 1, logical)
    assert task["pending_before"] == task["pending_after"] == 0
    assert task["granularity"] == ("combine_progress" if family == "AIV_COMBINE_PROGRESS" else "combine_drain")
    if family == "AIV_COMBINE_PROGRESS":
        assert task["posts"] == task["scanned_rows"] == task["scan_control_cycles"] == 0
    else:
        assert task == dict(granularity="combine_drain", pending_before=0, pending_after=0)


@pytest.mark.parametrize(
    "family,first,second",
    [
        ("AIV_COMBINE_PROGRESS", 1025, 0),
        ("AIV_COMBINE_PROGRESS", 1 | 2 << 16, 0),
        ("AIV_COMBINE_PROGRESS", 8 | 9 << 32 | 8 << 48, 0),
        ("AIV_COMBINE_PROGRESS", 8 | 1 << 32 | 7 << 48, 0),
        ("AIV_COMBINE_PROGRESS", 8 | 1 << 32 | 1025 << 48, 0),
        ("AIV_COMBINE_PROGRESS", 8 | 8 << 48, 0),
        ("AIV_COMBINE_PROGRESS", 8 | 8 << 16 | 1 << 32 | 8 << 48, 0),
        ("AIV_COMBINE_DRAIN", 1025, 0),
        ("AIV_COMBINE_DRAIN", 1 | 2 << 16, 0),
        ("AIV_COMBINE_DRAIN", 8 | 1 << 32, 0),
        ("AIV_COMBINE_DRAIN", 8 | 1 << 48, 0),
        ("AIV_COMBINE_DRAIN", 8, 1),
    ],
)
def test_combine_invalid_metadata_rejected(family, first, second):
    class Metadata:
        def __getitem__(self, key):
            return first, second

    logical = len(p.FIXED_NAMES) + p.DYNAMIC_NAMES.index(family) * p.MAX_DYNAMIC_ITER
    with pytest.raises(ValueError, match="invalid combine metadata"):
        p.task_metadata(Metadata(), 1, logical)


def test_combine_progress_binary_csv_text_roundtrip(raw_file, tmp_path):
    from deep_ep_experimental.megamoe.timer import text_transfer as transfer

    core = 1
    with raw_file.open("r+b") as stream:

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        for family, start, end in (
            ("AIV_COMBINE_INIT", 110100, 110900),
            ("AIV_HOST_COMBINE_BARRIER_WAIT", 131000, 140000),
        ):
            logical = p.FIXED_NAMES.index(family)
            put(core * p.COUNTERS_PER_CORE + logical * 2, start, end)
            put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical, p.NO_WAVE)

        for name, index, start, end, first, second, wave in (
            ("AIV_COMBINE_PROGRESS", 0, 111000, 120000, 128 | 120 << 16 | 4 << 32 | 32 << 48, 2501, 4),
            ("AIV_COMBINE_PROGRESS", 1, 145000, 153000, 120 | 6 << 32 | 256 << 48, 6203, p.NO_WAVE),
            ("AIV_COMBINE_DRAIN", 0, 142000, 160000, 120, 0, p.NO_WAVE),
            ("AIV_RDMA_POST", 0, 112000, 114000, 17 << 32 | 3, 14336, 4),
        ):
            family = p.DYNAMIC_NAMES.index(name)
            slot = family * p.MAX_DYNAMIC_ITER + index
            logical = len(p.FIXED_NAMES) + slot
            put(core * p.COUNTERS_PER_CORE + logical * 2, start, end)
            put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family, index + 1)
            put(p.TASK_METADATA_OFFSET + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + slot) * 2, first, second)
            put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical, wave)

    parsed = raw_file.with_name("rank_0_npu_time_parsed.csv")
    decode(raw_file, parsed)
    raw = raw_events_for_rank(parsed, 0)
    base = first_start_cycle(raw)
    shown = trace_events(raw, "keep", base, include_wait=True, time_origin="rank")
    combine = [event for event in shown if event["tid"].endswith("/COMBINE_PROGRESS")]
    assert len(combine) == 5
    assert all(event["tid"] == "group_0_AIV_0/COMBINE_PROGRESS" for event in combine)
    assert all("expert_id" not in event["args"] for event in combine)
    progress = [event for event in combine if event["args"].get("granularity") == "combine_progress"]
    assert [(event["dur"], event["args"]["scan_control_cycles"]) for event in progress] == [(9, 2501), (8, 6203)]
    assert [event["args"]["wave_id"] for event in progress] == [4, None]
    drain = next(event for event in combine if event["args"].get("granularity") == "combine_drain")
    assert drain["dur"] == 18 and drain["args"]["pending_before"] == 120 and drain["args"]["pending_after"] == 0
    for key in ("posts", "scanned_rows", "scan_control_cycles", "scan_control_us"):
        assert key not in drain["args"]
    child = next(event for event in shown if event["args"].get("granularity") == "rdma")
    assert child["tid"].endswith("/RDMA") and child["dur"] == 2
    hidden = trace_events(raw, "keep", base, time_origin="rank")
    assert not any("WAIT" in event["args"]["event_family"] for event in hidden)
    assert sum(event["tid"].endswith("/COMBINE_PROGRESS") for event in hidden) == 4
    for event in hidden:
        counterpart = next(e for e in shown if e["name"] == event["name"] and e["tid"] == event["tid"])
        assert (event["ts"], event["dur"]) == (counterpart["ts"], counterpart["dur"])

    output, index = transfer.export_text(raw_file.parent, tmp_path / "export", max_bytes=1024)
    parts = [transfer.show_part(output, number) for number in range(1, len(index["parts"]) + 1)]
    restored = tmp_path / "restored"
    transfer.restore_text(parts, restored)
    assert (restored / "timer" / parsed.name).read_bytes() == parsed.read_bytes()
    assert json.loads((restored / "timer/rank_0_npu_time.json").read_text(encoding="utf-8")) == shown
    with (restored / "timer/combine_progress_breakdown.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert next(row for row in rows if row["phase"] == "host_barrier_wait")["elapsed_us"] == "9.000000"
    measured = [row for row in rows if row["phase"] == "progress"]
    assert [(row["scan_control_cycles"], row["scan_control_us"]) for row in measured] == [
        ("2501", "2.501"),
        ("6203", "6.203"),
    ]
    assert [(row["posts"], row["scanned_rows"]) for row in measured] == [("4", "32"), ("6", "256")]
    drain_row = next(row for row in rows if row["phase"] == "drain")
    assert drain_row["scope"] == "envelope" and drain_row["elapsed_us"] == "18.000000"
    assert (drain_row["pending_before"], drain_row["pending_after"]) == ("120", "0")
    for key in ("posts", "scanned_rows", "scan_control_cycles", "scan_control_us"):
        assert drain_row[key] == ""
    with (restored / "timer/startup_breakdown.csv").open(newline="", encoding="utf-8") as stream:
        assert not list(csv.DictReader(stream))


def test_legacy_parsed_csv_renders_without_new_events(tmp_path):
    from deep_ep_experimental.megamoe.timer.__main__ import main

    # Previous parsed schemas do not need new raw-buffer offsets or event columns.
    parsed = tmp_path / "rank_0_npu_time_parsed.csv"
    fields = [
        "core_id",
        "group_id",
        "core_type",
        "sub_id",
        "AIC_GMM1_0_start_cycle",
        "AIC_GMM1_0_end_cycle",
        "AIC_GMM1_task_count",
    ]
    with parsed.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerow([0, 0, "AIC", 0, 2**55 + 1000, 2**55 + 6000, 1])
    main([str(tmp_path), "--ranks", "0", "--include-wait"])
    trace = json.loads((tmp_path / "rank_0_npu_time.json").read_text(encoding="utf-8"))
    assert len(trace) == 1 and trace[0]["name"] == "AIC_GMM1_0"
    assert trace[0]["ts"] == 0 and trace[0]["dur"] == 5
    with (tmp_path / "combine_progress_breakdown.csv").open(newline="", encoding="utf-8") as stream:
        assert not list(csv.DictReader(stream))


def test_previous_raw_schema_size_is_rejected(tmp_path):
    fixed, dynamic = len(p.FIXED_NAMES) - 2, len(p.DYNAMIC_NAMES) - 2
    items = fixed + dynamic * p.MAX_DYNAMIC_ITER
    counters = (items * 2 + 15) // 16 * 16 * p.N_CORE_COUNT
    iterations = (dynamic + 15) // 16 * 16 * p.N_CORE_COUNT
    metadata = dynamic * p.MAX_DYNAMIC_ITER * 2 * p.N_CORE_COUNT
    waves = (items + 15) // 16 * 16 * p.N_CORE_COUNT
    previous_size = counters + iterations + metadata + waves
    assert previous_size < p.TOTAL_BUFFER_SIZE
    previous = tmp_path / "old_timer.bin"
    with previous.open("wb") as stream:
        stream.truncate(previous_size * 8)
    with pytest.raises(ValueError, match="invalid binary timer size"):
        load_raw(previous)


def test_multi_qp_payload_trace_ranges_and_collective_wait(raw_file):
    core = 2  # g0 AIV1
    with raw_file.open("r+b") as stream:

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        records = [
            ("AIV_RDMA_POST", 17 << 32 | 16 << 8 | 2, 112 << 48 | 8 << 40 | 59392),
            ("AIV_PAYLOAD_ALL_LANES_WAIT", 0x81 << 32 | 4, 257),
        ]
        for name, first, second in records:
            family = p.DYNAMIC_NAMES.index(name)
            slot = family * p.MAX_DYNAMIC_ITER
            logical = len(p.FIXED_NAMES) + slot
            put(core * p.COUNTERS_PER_CORE + logical * 2, 101000, 102000)
            put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family, 1)
            put(p.TASK_METADATA_OFFSET + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + slot) * 2, first, second)
            put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical, p.NO_WAVE)
    parsed = raw_file.with_suffix(".parsed.csv")
    decode(raw_file, parsed)
    raw = raw_events_for_rank(parsed, 0)
    events = trace_events(raw, "keep", first_start_cycle(raw), include_wait=True)
    post = next(e for e in events if e["args"].get("granularity") == "rdma")
    assert post["name"].endswith("_qp8")
    assert {k: post["args"][k] for k in ("qp", "lane", "row_start", "row_end", "row_count", "bytes")} == dict(
        qp=8, lane=7, row_start=112, row_end=128, row_count=16, bytes=59392
    )
    wait = next(e for e in events if e["name"] == "AIV_PAYLOAD_ALL_LANES_WAIT")
    assert wait["args"]["initial_missing_lanes"] == [0, 7]
    assert wait["args"]["poll_rounds"] == 4
    assert wait["args"]["sequence"] == 257
    assert wait["tid"].endswith("/STARTUP")
    hidden = trace_events(raw, "keep", first_start_cycle(raw), include_wait=False)
    assert not any(e["name"] == "AIV_PAYLOAD_ALL_LANES_WAIT" for e in hidden)

    report = raw_file.with_suffix(".startup.csv")
    write_startup_breakdown({0: raw}, report)
    with report.open(newline="", encoding="utf-8") as stream:
        row = next(row for row in csv.DictReader(stream) if row["event_family"] == "AIV_PAYLOAD_ALL_LANES_WAIT")
    assert json.loads(row["initial_missing_lanes"]) == [0, 7]
    assert row["poll_rounds"] == "4"


@pytest.mark.parametrize("qp", [9, 16, 17])
@pytest.mark.parametrize(
    "family,lo,second",
    [
        ("AIV_COMBINE_READY", 0, (1 << 32) | 1),
        ("AIV_COMBINE_SYNC", 1, 0),
        ("AIV_COMBINE_REDUCE", 0, (1 << 32) | 5120),
    ],
)
def test_combine_trace_qp_boundaries(qp, family, lo, second):
    from deep_ep_experimental.megamoe.timer import protocol as p

    class Raw:
        def __getitem__(self, _):
            return [(qp << 32) | lo, second]

    slot = len(p.FIXED_NAMES) + p.DYNAMIC_NAMES.index(family) * p.MAX_DYNAMIC_ITER
    if qp > 16:
        with pytest.raises(ValueError):
            p.task_metadata(Raw(), 0, slot)
    else:
        decoded = p.task_metadata(Raw(), 0, slot)
        assert (decoded["qp_index"], decoded["lane_count"]) == (qp, 16)


@pytest.mark.parametrize("family", ["AIV_RDMA_POST", "AIV_RDMA_RECV_WAIT", "AIV_RDMA_QUIET_WAIT"])
@pytest.mark.parametrize("phase,qp", [(1, 0), (1, 8), (1, 9), (2, 9), (2, 16), (2, 17)])
def test_rdma_qp_is_phase_specific(family, phase, qp):
    class Raw:
        def __getitem__(self, _):
            return [(17 << 32) | (phase + 1), (qp << 40) | 4096]

    slot = len(p.FIXED_NAMES) + p.DYNAMIC_NAMES.index(family) * p.MAX_DYNAMIC_ITER
    if qp > (16 if phase == 2 else 8):
        with pytest.raises(ValueError, match="RDMA"):
            p.task_metadata(Raw(), 0, slot)
    else:
        task = p.task_metadata(Raw(), 0, slot)
        assert task.get("qp", 0) == qp
        if qp:
            assert task["lane_count"] == (16 if phase == 2 else 8)
