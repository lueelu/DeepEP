# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""CPU protocol/trace checks; these tests never load the native extension."""

from array import array
import importlib
import json
from pathlib import Path
import re

import pytest


@pytest.fixture
def timer(monkeypatch):
    python = Path(__file__).resolve().parents[2] / "python"
    monkeypatch.syspath_prepend(str(python))
    package = importlib.import_module("deep_ep_experimental")
    monkeypatch.setattr(package, "__path__", [*package.__path__, str(python / "deep_ep_experimental")])
    return importlib.import_module("deep_ep_experimental.megamoe.timer.bf16.trace_parser")


def raw_buffer(timer, phase=0, *, with_waves=False, fixed_count=None):
    p = timer.p
    if fixed_count is None:
        fixed_count = len(p.FIXED_NAMES)
    layout = p.layout(3, 2, with_waves=with_waves, fixed_count=fixed_count)
    raw = array("q", [0]) * layout["elements"]
    if with_waves:
        raw[layout["waves"] :] = array("q", [-1]) * (layout["elements"] - layout["waves"])
    raw[:16] = array(
        "q",
        [
            p.MAGIC,
            (p.BACKWARD_WAVE_VERSION if phase else p.FORWARD_WAVE_VERSION) if with_waves else p.VERSION,
            3,
            2,
            phase,
            fixed_count,
            len(p.DYNAMIC_NAMES),
            7,
            258,
            129,
            7168,
            3072,
            2,
            4,
            1,
            1,
        ],
    )
    raw[16:18] = array("q", [100000, 120000])
    return raw, layout


def put_event(timer, raw, layout, name, index, start, end, *, m=384, tile_m=192, wave=None, core=0):
    p = timer.p
    family = p.DYNAMIC_NAMES.index(name)
    logical = raw[5] + family * layout["capacity"] + index
    if wave is not None:
        raw[layout["waves"] + core * layout["wave_stride"] + logical] = wave
    start_offset = p.HEADER_WORDS + core * layout["stride"] + logical * 2
    raw[start_offset : start_offset + 2] = array("q", [start, end])
    raw[layout["counts"] + core * layout["count_stride"] + family] = index + 1
    offset = layout["metadata"] + ((core * len(p.DYNAMIC_NAMES) + family) * layout["capacity"] + index) * 7
    raw[offset : offset + 7] = array(
        "q", [(3 << 32) | m, (2 << 32) | 66, (tile_m << 32) | 256, 258, (256 << 32) | 128, 0, 0]
    )


def test_schema_matches_device_header(timer):
    header = (Path(__file__).resolve().parents[2] / "csrc/kernels/mega_moe/timer/ascend_timer.hpp").read_text(
        encoding="utf-8"
    )
    for enum, sentinel, expected in (
        ("FixedTiming", "FIXED_TIMING_COUNT", timer.p.FIXED_NAMES),
        ("DynamicTimingType", "DYNAMIC_TYPE_COUNT", timer.p.DYNAMIC_NAMES),
    ):
        body = re.search(r"enum " + enum + r"\s*\{(.*?)\};", header, re.S)[1]
        names = [x.split("=")[0].strip() for x in body.split(",") if x.strip() and sentinel not in x]
        assert names == expected


def test_real_tile_coordinates_and_multiple_wait_intervals(timer, tmp_path):
    raw, layout = raw_buffer(timer)
    put_event(timer, raw, layout, "AIC_GMM1", 0, 100000, 110000)
    put_event(timer, raw, layout, "AIC_GMM1_WAIT", 0, 101000, 102000)
    put_event(timer, raw, layout, "AIC_GMM1_WAIT", 1, 104000, 106000)
    path = tmp_path / "rank_0_forward_time.bin"
    path.write_bytes(raw.tobytes())
    events, info = timer.decode(path, 0)
    assert info["lanes"] == 3 and info["topk_weight_dtype"] == "fp32"
    work = next(e for e in events if e["name"] == "AIC_GMM1")
    assert work["task"]["m_tile"] == 2
    assert work["task"]["row_count"] == 66
    trace = timer.chrome_events(events, 100000)
    assert sum(e["dur"] for e in trace if e["args"]["event_family"] == "AIC_GMM1") == 7
    assert sum(e["dur"] for e in trace if e["cat"] == "wait") == 3
    assert not any(e["cat"] == "wait" for e in timer.chrome_events(events, 100000, include_wait=False))


def test_decode_capture_before_dispatch_weight_timer(timer, tmp_path):
    raw, layout = raw_buffer(timer, fixed_count=15)
    put_event(timer, raw, layout, "AIC_GMM2", 0, 101000, 112000)
    path = tmp_path / "rank_0_forward_time.bin"
    path.write_bytes(raw.tobytes())
    events, _ = timer.decode(path, 0)
    assert [(e["name"], e["start_cycle"]) for e in events] == [("KERNEL_TIMING", 100000), ("AIC_GMM2", 101000)]


def test_weight_gradient_coordinates_keep_reduction_dimension(timer, tmp_path):
    raw, layout = raw_buffer(timer, phase=1)
    put_event(timer, raw, layout, "AIC_BW_DW2", 0, 101000, 111000, m=128, tile_m=128)
    path = tmp_path / "rank_0_backward_time.bin"
    path.write_bytes(raw.tobytes())
    events, _ = timer.decode(path, 0)
    dw = next(e["task"] for e in events if e["name"] == "AIC_BW_DW2")
    assert dw["row_start"] == 128 and dw["k_count"] == 258 and dw["n_start"] == 256


def test_backward_wave_wait_and_borrowed_dw2_remain_separate(timer, tmp_path):
    raw, layout = raw_buffer(timer, phase=1, with_waves=True)
    put_event(timer, raw, layout, "AIC_BW_DW2", 0, 101000, 103000, wave=7)
    put_event(timer, raw, layout, "AIC_BW_DHP_WAIT", 0, 103000, 105000, wave=7)
    put_event(timer, raw, layout, "AIC_BW_DHP", 0, 105000, 107000, wave=7)
    put_event(timer, raw, layout, "AIC_BW_DW2", 1, 111000, 113000)
    (tmp_path / "rank_0_backward_time.bin").write_bytes(raw.tobytes())
    events = timer.export_rank(tmp_path, 0, include_wait=True)
    assert [(e["name"], e["ts"], e["dur"]) for e in events] == [
        ("wave_7_AIC_BW_DW2_EARLY_E2", 1, 2),
        ("wave_7_AIC_BW_DHP_WAIT_E2", 3, 2),
        ("wave_7_AIC_BW_DHP_E2", 5, 2),
        ("AIC_BW_DW2_E2", 11, 2),
    ]
    assert timer.export_rank(tmp_path, 0) == [e for e in events if not e["args"]["is_wait"]]


def test_backward_expert_overview_keeps_tiles_in_detail_and_splits_at_wait(timer, tmp_path):
    raw, layout = raw_buffer(timer, phase=1)
    put_event(timer, raw, layout, "AIC_BW_DW1", 0, 101000, 103000)
    put_event(timer, raw, layout, "AIC_BW_DW1", 1, 103100, 106000)
    put_event(timer, raw, layout, "AIC_BW_DHP", 0, 108000, 110000)
    put_event(timer, raw, layout, "AIC_BW_DHP_WAIT", 0, 110000, 112000)
    put_event(timer, raw, layout, "AIC_BW_DHP", 1, 112000, 115000)
    (tmp_path / "rank_0_backward_time.bin").write_bytes(raw.tobytes())
    visible = timer.export_rank(tmp_path, 0, include_wait=True)
    work = timer.export_rank(tmp_path, 0)
    assert work == [e for e in visible if not e["args"]["is_wait"]]
    assert [(e["name"], e["ts"], e["dur"]) for e in work] == [
        ("AIC_BW_DW1_E2", 1, 5),
        ("AIC_BW_DHP_E2", 8, 2),
        ("AIC_BW_DHP_E2", 12, 3),
    ]
    assert work[0]["args"]["recorded_fragments"] == 2
    assert work[0]["args"]["active_us"] == pytest.approx(4.9)
    detail = json.loads((tmp_path / "rank_0_backward_detail_trace.json").read_text())["traceEvents"]
    assert [e["args"]["task_id"] for e in detail if e["args"]["event_family"] == "AIC_BW_DW1"] == [0, 1]


def test_backward_dp_overview_collapses_sends_only_on_their_own_lane(timer, tmp_path):
    raw, layout = raw_buffer(timer, phase=1)
    fixed = timer.p.FIXED_NAMES.index("AIV_BW_DP_REDUCE")
    offset = timer.p.HEADER_WORDS + layout["stride"] + fixed * 2
    raw[offset : offset + 2] = array("q", [101000, 119000])
    put_event(timer, raw, layout, "AIV_BW_DP_SEND", 0, 102000, 106000, core=1)
    put_event(timer, raw, layout, "AIV_BW_DP_SEND", 1, 109000, 113000, core=1)
    put_event(timer, raw, layout, "AIV_BW_DP_SEND", 0, 107000, 108000, core=2)
    (tmp_path / "rank_0_backward_time.bin").write_bytes(raw.tobytes())
    for waits in (False, True):
        overview = timer.export_rank(tmp_path, 0, include_wait=waits)
        assert [(e["tid"], e["name"], e["ts"], e["dur"]) for e in overview] == [
            ("group_0_AIV_0", "AIV_BW_DP_REDUCE_SEND", 1, 18),
            ("group_0_AIV_1", "AIV_BW_DP_SEND_E2", 7, 1),
        ]
        detail = json.loads((tmp_path / "rank_0_backward_detail_trace.json").read_text())["traceEvents"]
        sends = [e for e in detail if e["args"]["event_family"] == "AIV_BW_DP_SEND"]
        assert [(e["args"]["core_id"], e["ts"], e["dur"]) for e in sends] == [(1, 2, 4), (1, 9, 4), (2, 7, 1)]


def test_overflow_and_truncated_protocol_fail(timer, tmp_path):
    raw, layout = raw_buffer(timer)
    raw[layout["counts"]] = layout["capacity"] + 1
    path = tmp_path / "invalid.bin"
    path.write_bytes(raw.tobytes())
    with pytest.raises(ValueError, match="overflow"):
        timer.decode(path, 0)
    path.write_bytes(raw.tobytes()[:-8])
    with pytest.raises(ValueError, match="size"):
        timer.decode(path, 0)


def test_forward_backward_export_preserves_same_rank_gap(timer, tmp_path):
    for phase, shift in ((0, 0), (1, 100000)):
        raw, layout = raw_buffer(timer, phase)
        raw[16:18] = array("q", [100000 + shift, 120000 + shift])
        name = "forward" if phase == 0 else "backward"
        put_event(timer, raw, layout, "AIC_GMM1" if phase == 0 else "AIC_BW_DX", 0, 101000 + shift, 111000 + shift)
        (tmp_path / f"rank_0_{name}_time.bin").write_bytes(raw.tobytes())
    combined = timer.export_rank(tmp_path, 0)
    assert not any(e["args"]["event_family"] == "KERNEL_TIMING" for e in combined)
    assert [e["ts"] for e in combined] == [1, 101]
    assert (tmp_path / "rank_0_forward_detail_trace.json").exists()


def test_export_keeps_submicrosecond_slices_on_one_lane(timer, tmp_path):
    export = importlib.import_module(timer.__package__ + ".__main__").export
    for phase, name in enumerate(("AIC_GMM1", "AIC_BW_DHP")):
        phase_name = "backward" if phase else "forward"
        raw, layout = raw_buffer(timer, phase=phase, with_waves=True)
        base = 100000 + phase * 20000
        raw[16:18] = array("q", [base, base + 20000])
        put_event(timer, raw, layout, name, 0, base + 1000, base + 3140, wave=0)
        # A short adjacent task in the next wave can round into the previous one at microsecond precision.
        put_event(timer, raw, layout, name, 1, base + 3160, base + 3420, wave=1)
        (tmp_path / f"rank_0_{phase_name}_time.bin").write_bytes(raw.tobytes())
    export(tmp_path, (0,))
    for path in tmp_path.glob("*_trace.json"):
        doc = json.loads(path.read_text())
        assert doc["displayTimeUnit"] == "ns"
        slices = [e for e in doc["traceEvents"] if e["args"]["task_id"] == 1]
        assert slices
        for e in slices:
            assert (e["ts"], e["dur"]) == (23.16 if e["args"]["phase"] == "backward" else 3.16, 0.26)


def test_expert_view_uses_existing_names_and_excludes_pre_dependency_preparation(timer, tmp_path):
    raw, layout = raw_buffer(timer)
    put_event(timer, raw, layout, "AIC_GMM1", 0, 100000, 110000)
    put_event(timer, raw, layout, "AIC_GMM1", 1, 111000, 119000)
    put_event(timer, raw, layout, "AIC_GMM1_WAIT", 0, 101000, 105000)
    put_event(timer, raw, layout, "AIC_GMM1_WAIT", 1, 112000, 114000)
    family = timer.p.DYNAMIC_NAMES.index("AIC_GMM1_WAIT")
    parent = timer.p.DYNAMIC_NAMES.index("AIC_GMM1")
    raw[layout["metadata"] + family * layout["capacity"] * 7 + 6] = ((parent + 1) << 32) | 1
    raw[layout["metadata"] + (family * layout["capacity"] + 1) * 7 + 6] = ((parent + 1) << 32) | 2
    path = tmp_path / "rank_0_forward_time.bin"
    path.write_bytes(raw.tobytes())
    events, _ = timer.decode(path, 0)
    actual = timer.expert_events(events, 100000)
    assert len(actual) == 1
    assert actual[0]["name"] == "AIC_GMM1_2"
    assert actual[0]["tid"] == "group_0_AIC_0" and actual[0]["pid"] == "rank0"
    assert actual[0]["ts"] == 5 and actual[0]["dur"] == 14
    assert actual[0]["args"]["recorded_tasks"] == 2
    with_wait = timer.expert_events(events, 100000, include_wait=True)
    waits = [e for e in with_wait if e["args"]["is_wait"]]
    assert len(waits) == 1 and waits[0]["name"] == "AIC_GMM1_WAIT_2"
    assert waits[0]["ts"] == 1 and waits[0]["dur"] == 4


def test_forward_wave_export_keeps_gaps_and_wait_toggle_across_ranks(timer, tmp_path):
    export = importlib.import_module(timer.__package__ + ".__main__").export
    for rank in (0, 1):
        raw, layout = raw_buffer(timer, with_waves=True)
        put_event(timer, raw, layout, "AIC_GMM1", 0, 101000, 105000, wave=1)
        put_event(timer, raw, layout, "AIC_GMM1_WAIT", 0, 102000, 103000, wave=1)
        put_event(timer, raw, layout, "AIC_GMM2", 0, 106000, 110000, wave=0)
        put_event(timer, raw, layout, "AIC_GMM1", 1, 111000, 116000, wave=1)
        (tmp_path / f"rank_{rank}_forward_time.bin").write_bytes(raw.tobytes())
    plain, waits = tmp_path / "plain", tmp_path / "waits"
    export(tmp_path, (0, 1), output_dir=plain)
    export(tmp_path, (0, 1), include_wait=True, output_dir=waits)
    for file in ("rank_0_forward_trace.json", "rank_0_forward_detail_trace.json", "all_rank_forward_trace.json"):
        hidden = json.loads((plain / file).read_text())["traceEvents"]
        visible = json.loads((waits / file).read_text())["traceEvents"]
        assert hidden == [e for e in visible if e["cat"] not in ("wait", "dependency_wait")]
        assert len(visible) > len(hidden)
    merged = json.loads((plain / "all_rank_forward_trace.json").read_text())["traceEvents"]
    assert {e["pid"] for e in merged} == {"rank0", "rank1"}
    assert [(e["name"], e["ts"], e["dur"]) for e in merged if e["pid"] == "rank0"] == [
        ("wave_1_AIC_GMM1_0", 1, 1),
        ("wave_1_AIC_GMM1_0", 3, 2),
        ("wave_0_AIC_GMM2_0", 6, 4),
        ("wave_1_AIC_GMM1_1", 11, 5),
    ]


def test_forward_wave_protocol_rejects_missing_wave(timer, tmp_path):
    raw, layout = raw_buffer(timer, with_waves=True)
    put_event(timer, raw, layout, "AIC_GMM2", 0, 105000, 109000)
    path = tmp_path / "rank_0_forward_time.bin"
    path.write_bytes(raw.tobytes())
    with pytest.raises(ValueError, match="invalid Wave ID"):
        timer.decode(path, 0)
