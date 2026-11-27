# Copyright (c) 2026, Lu Lu
# Modified by candy_cloud_fly 2026

"""CPU checks for LL Host contracts; these do not qualify device kernels."""

from types import SimpleNamespace

import pytest

from deep_ep.backends.mesh_clos import elastic


def test_handle_schema(monkeypatch):
    monkeypatch.setattr(elastic, "require_native", lambda _: None)
    args = [True, 32, 1, 64, 0, None, None, 0, None, None, None, None, None, None, None, None]
    ht, ll = elastic.EPHandle(*args), elastic.EPHandle(*args, is_low_latency=True)
    assert vars(ht).keys() == vars(ll).keys()
    assert not ht.is_low_latency and ll.is_low_latency
    assert ll.dst is None and ll.forward_list is None
    assert ht.source_weights is None and ht.send_counts is None


def make_buffer(group=None):
    buffer = elastic.ElasticBuffer.__new__(elastic.ElasticBuffer)
    buffer._runtime = SimpleNamespace(group=group, rank=0, world_size=2, closed=True)
    return buffer


def test_no_group_does_not_read_inputs():
    assert make_buffer()._notify_low_latency(object(), 32) is None


def make_ll_buffer(world=16):
    buffer = elastic.ElasticBuffer.__new__(elastic.ElasticBuffer)
    buffer._runtime = SimpleNamespace(group=None, rank=0, world_size=world, closed=True)
    buffer.hidden = 0
    buffer.num_topk = 0
    return buffer


def test_smooth_int8_plan_accepts_replicated_fp32_table(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = {}

    class Native:
        @staticmethod
        def low_latency_layout(*args):
            calls["args"] = args
            return {"capacity_rows": 64, "smooth_int8": bool(args[-1])}

    monkeypatch.setattr(elastic, "require_native", lambda _: Native)
    b, h, k, e = 8, 512, 8, 32
    hidden = torch.randn(b, h, dtype=torch.bfloat16)
    topk = torch.zeros(b, k, dtype=torch.int64)
    weights = torch.rand(b, k, dtype=torch.float32)
    smooth = torch.rand(e, h, dtype=torch.float32)
    plan = make_ll_buffer()._low_latency_plan(hidden, topk, weights, e, b, smooth)
    assert plan["smooth_int8"] is True
    assert calls["args"][-1] is True


def test_smooth_int8_plan_rejects_unsupported_tables(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        elastic, "require_native", lambda _: SimpleNamespace(low_latency_layout=lambda *args: {"capacity_rows": 1})
    )
    b, h, k, e = 8, 512, 8, 32
    hidden = torch.randn(b, h, dtype=torch.bfloat16)
    topk = torch.zeros(b, k, dtype=torch.int64)
    weights = torch.rand(b, k, dtype=torch.float32)
    buffer = make_ll_buffer()
    assert buffer._low_latency_plan(hidden, topk, weights, e, b, torch.rand(e, h, dtype=torch.float16)) is None
    assert buffer._low_latency_plan(hidden, topk, weights, e, b, torch.rand(e, h // 2, dtype=torch.float32)) is None
    noncontiguous = torch.rand(h, e, dtype=torch.float32).t()
    assert buffer._low_latency_plan(hidden, topk, weights, e, b, noncontiguous) is None
    # Prequantized FP8 input cannot be smoothed a second time.
    fp8 = torch.zeros(b, h, dtype=torch.float8_e4m3fn)
    scales = torch.ones(b, (h + 127) // 128, dtype=torch.float32)
    assert buffer._low_latency_plan((fp8, scales), topk, weights, e, b, torch.rand(e, h, dtype=torch.float32)) is None
