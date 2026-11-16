# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""CPU protocol/trace checks; these tests never load the native extension."""

from array import array
import importlib
from pathlib import Path
import re

import pytest


@pytest.fixture
def timer(monkeypatch):
    python = Path(__file__).resolve().parents[1] / "python"
    monkeypatch.syspath_prepend(str(python))
    package = importlib.import_module("deep_ep_experimental")
    monkeypatch.setattr(package, "__path__", [*package.__path__, str(python / "deep_ep_experimental")])
    return importlib.import_module("deep_ep_experimental.megamoe_train.timer.trace_parser")


def raw_buffer(timer, phase=0):
    p = timer.p
    layout = p.layout(3, 2)
    raw = array("q", [0]) * layout["elements"]
    raw[:16] = array(
        "q",
        [
            p.MAGIC,
            p.VERSION,
            3,
            2,
            phase,
            len(p.FIXED_NAMES),
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


def put_event(timer, raw, layout, name, index, start, end, *, m=384, tile_m=192):
    p = timer.p
    family = p.DYNAMIC_NAMES.index(name)
    logical = len(p.FIXED_NAMES) + family * layout["capacity"] + index
    raw[16 + logical * 2 : 18 + logical * 2] = array("q", [start, end])
    raw[layout["counts"] + family] = index + 1
    offset = layout["metadata"] + (family * layout["capacity"] + index) * 7
    raw[offset : offset + 7] = array(
        "q", [(3 << 32) | m, (2 << 32) | 66, (tile_m << 32) | 256, 258, (256 << 32) | 128, 0, 0]
    )


def test_schema_matches_device_header(timer):
    header = (Path(__file__).resolve().parents[1] / "csrc/kernels/bf16/timer/ascend_timer.hpp").read_text(
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


def test_weight_gradient_coordinates_keep_reduction_dimension(timer, tmp_path):
    raw, layout = raw_buffer(timer, phase=1)
    put_event(timer, raw, layout, "AIC_BW_DW2", 0, 101000, 111000, m=128, tile_m=128)
    path = tmp_path / "rank_0_backward_time.bin"
    path.write_bytes(raw.tobytes())
    events, _ = timer.decode(path, 0)
    dw = next(e["task"] for e in events if e["name"] == "AIC_BW_DW2")
    assert dw["row_start"] == 128 and dw["k_count"] == 258 and dw["n_start"] == 256


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
