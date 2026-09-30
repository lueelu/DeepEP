# Copyright (c) 2026, Lu Lu
# Modified by candy_cloud_fly 2026

"""CPU checks for LL Host contracts; these do not qualify device kernels."""

from types import SimpleNamespace

from deep_ep.buffers import elastic


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
