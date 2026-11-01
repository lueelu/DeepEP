"""CPU policy and mocked API contracts, not NPU precision/performance tests."""

from unittest.mock import Mock, patch
from contextlib import nullcontext

import pytest
import torch

from deep_ep.backends.clos import Config, topk_idx_t
from deep_ep.backends.clos.config import _MAX_NUM_AIV
from deep_ep.backends.clos.udma_policy import resolve_udma_policy
from deep_ep.backends.clos.buffer import _prepare_shmem_bootstrap
from test_public_api import _make_buffer, _Group, _FakeNpu
import test_v2_elastic_api as v2_api


def policy(ep=32, **kwargs):
    args = dict(
        enabled=True,
        world_size=ep,
        transport="udma",
        schedule="peer",
        dispatch_order="rank",
        combine_order="rank",
        ranks_per_group=None,
        peers_per_group_per_worker=None,
    )
    args.update(kwargs)
    return resolve_udma_policy(**args)


@pytest.mark.parametrize(
    "ep,p,w,visit",
    [(4, 1, 2, 8), (8, 1, 2, 8), (16, 1, 1, 8), (32, 4, 1, 32), (64, 4, 1, 32), (128, 4, 1, 32), (256, 8, 1, 32)],
)
def test_ep_defaults(ep, p, w, visit):
    r = policy(ep)
    assert (r.peers_per_worker, r.workers_per_peer, r.qps_per_worker, r.wqes_per_visit) == (p, w, 1, visit)
    assert not hasattr(r, "num_aiv")


def test_even_core_counts_validate_without_adjusting_schedule():
    for ep in (2, 4, 8, 16, 32, 64, 128, 256):
        r = policy(ep)
        signature = r.signature()
        for cores in range(2, _MAX_NUM_AIV + 1, 2):
            if cores < r.send_workers:
                with pytest.raises(ValueError, match="requires"):
                    r.validate_num_aiv(cores)
            else:
                assert r.config(cores).num_aiv == cores
            assert r.signature() == signature
        assert r.config(0).num_aiv == 0
    with pytest.raises(ValueError, match="32-QP"):
        policy(512)
    # Unlike the rejected implementation, do not change P4 to P8 to fit 20 cores.
    with pytest.raises(ValueError, match="requires 32"):
        policy(128).validate_num_aiv(20)


def test_explicit_overrides_are_ignored_and_core_only_config(caplog):
    from deep_ep.backends.clos.udma_policy import _warn_ignored

    _warn_ignored.cache_clear()
    r = policy(8, workers=1, qps=2, peers=4, wqes=16)
    assert (r.workers_per_peer, r.qps_per_worker, r.peers_per_worker, r.wqes_per_visit) == (2, 1, 1, 8)
    assert "udma_auto_policy=False" in caplog.text
    c = r.resolve_config(Config(20, 4, 128), 32)
    assert (c.num_aiv, c.udma_peers_per_worker, c.udma_wqes_per_visit) == (20, 1, 8)
    assert (c.num_max_nvl_chunked_send_tokens, c.num_max_nvl_chunked_recv_tokens) == (4, 128)
    c = r.resolve_config(Config(20, udma_peers_per_worker=4, udma_wqes_per_visit=16), 32)
    assert (c.num_aiv, c.udma_peers_per_worker, c.udma_wqes_per_visit) == (20, 1, 8)
    assert policy().resolve_config(Config(20, udma_wqes_per_visit=8), 32).udma_wqes_per_visit == 32
    assert (Config().udma_peers_per_worker, Config().udma_wqes_per_visit) == (4, 8)
    for overrides in (
        dict(workers=2, qps=2),
        dict(schedule="token"),
        dict(dispatch_order="group"),
        dict(peers=16),
        dict(workers=True),
    ):
        assert policy(**overrides).signature() == policy().signature()
    assert policy(transport="mte") is None
    with pytest.raises(ValueError, match="manual UDMA"):
        policy(enabled=False)


def test_warnings_only_once_on_rank_zero(caplog):
    from deep_ep.backends.clos.udma_policy import _warn_ignored

    _warn_ignored.cache_clear()
    policy(8, workers=2, peers=1, wqes=8, qps=1)
    policy(8, workers=1, rank=1)
    assert not caplog.records
    for _ in range(3):
        policy(8, workers=1)
    assert len(caplog.records) == 1
    assert "effective udma_workers_per_peer=2" in caplog.text
    _warn_ignored.cache_clear()
    caplog.clear()
    r = policy(32)
    for _ in range(3):
        r.resolve_config(Config(20, udma_wqes_per_visit=8), 32, rank=1)
    assert not caplog.records
    for _ in range(3):
        r.resolve_config(Config(20, udma_wqes_per_visit=8), 32)
    assert len(caplog.records) == 1


@pytest.mark.parametrize("missing", ["schedule", "dispatch_order", "combine_order", "workers", "qps", "peers", "wqes"])
def test_manual_missing_required_fields(missing):
    options = dict(
        enabled=False, schedule="peer", dispatch_order="rank", combine_order="rank", workers=1, qps=1, peers=4, wqes=8
    )
    options[missing] = None
    with pytest.raises(ValueError, match="manual UDMA"):
        policy(**options)


def test_manual_relevant_fields_only():
    assert policy(enabled=False, schedule="token", wqes=16) is None
    assert (
        policy(
            enabled=False,
            schedule="peer",
            dispatch_order="group_round_robin",
            combine_order="group_round_robin",
            workers=1,
            qps=1,
            ranks_per_group=16,
            peers_per_group_per_worker=2,
            wqes=8,
        )
        is None
    )
    with pytest.raises(ValueError, match="udma_peers_per_group_per_worker"):
        policy(
            enabled=False,
            schedule="peer",
            dispatch_order="group_round_robin",
            combine_order="rank",
            workers=1,
            qps=1,
            ranks_per_group=16,
            peers=4,
            wqes=8,
        )


def test_v1_original_core_api_and_cached_policy(monkeypatch):
    from deep_ep.backends.clos import Buffer

    monkeypatch.setattr("deep_ep.backends.clos.event._get_npu_api", lambda: _FakeNpu())
    monkeypatch.setattr("deep_ep.backends.clos.buffer._use_comm_stream", lambda _: nullcontext())
    monkeypatch.setattr(Buffer, "num_sms", 20)
    b = _make_buffer(group=_Group(size=8), transport="udma")
    x = torch.ones((4, 8), dtype=torch.bfloat16)
    idx = torch.zeros((4, 1), dtype=topk_idx_t)
    counts, _, experts, member, _ = b.get_dispatch_layout(idx, 16)
    kwargs = dict(num_tokens_per_rank=counts, num_tokens_per_expert=experts, is_token_in_rank=member)
    for cores in (20, 32):
        rx, _, _, _, h, _ = b.dispatch(x, config=Config(cores), **kwargs)
        assert b.native.udma_schedule == (1, 8)
        b.dispatch(x, handle=h, config=Config(cores))
        b.combine(rx, h, config=Config(cores))
        assert b.native.udma_schedule == (1, 8)
    rx, _, _, _, h, _ = b.dispatch(x, **kwargs)
    assert b.native.dispatch_args[-4] == 20  # Native core count, before chunk sizes/zero-copy.
    with pytest.raises(ValueError, match="requires 14"):
        b.dispatch(x, config=Config(12), **kwargs)
    b.destroy()


@pytest.fixture
def elastic_api():
    fixture = v2_api.ElasticApiTest()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.tearDown()


def test_v2_can_use_different_counts_for_new_handles(elastic_api):
    b = elastic_api.make(transport="udma")
    assert b.udma_workers_per_peer == 2
    for cores in (20, 32):
        rx, _, weights, h, _ = elastic_api.fresh(b, num_sms=cores)
        assert h.num_sms == cores and b.runtime.udma_schedule == (1, 8)
        b.dispatch(torch.ones((13, 128), dtype=torch.bfloat16), handle=h)
        b.combine(rx, h, topk_weights=weights)
        assert b.runtime.udma_schedule == (1, 8)
        with pytest.raises(ValueError, match="must match"):
            b.combine(rx, h, num_sms=32 if cores == 20 else 20)
    with pytest.raises(ValueError, match="requires 14"):
        elastic_api.fresh(b, num_sms=12)
    # num_sms=0 retains the API's original theoretical selection, not a new fixed default.
    with patch.object(b, "get_theoretical_num_sms", return_value=20) as choose:
        _, _, _, h, _ = elastic_api.fresh(b, num_sms=0)
        assert h.num_sms == 20
        choose.assert_called_once()
    b.destroy()


def test_v2_default_ignores_overrides_and_manual_parameters(elastic_api):
    b = elastic_api.make(
        transport="udma",
        udma_workers_per_peer=1,
        udma_peers_per_worker=2,
        udma_wqes_per_visit=16,
        udma_schedule="token",
        udma_dispatch_peer_order="group",
        udma_ranks_per_group=4,
    )
    elastic_api.fresh(b, num_sms=20)
    assert b.runtime.udma_schedule == (1, 8) and b.udma_workers_per_peer == 2
    assert b.udma_schedule == "peer" and b.udma_dispatch_peer_order == "rank"
    assert b.udma_ranks_per_group is None
    b.destroy()
    b = elastic_api.manual_make(
        transport="udma", udma_peers_per_worker=2, udma_wqes_per_visit=16, udma_qps_per_worker=2
    )
    assert b.udma_auto_policy is None
    elastic_api.fresh(b, num_sms=20)
    assert b.runtime.udma_schedule == (2, 16) and b.udma_qps_per_worker == 2
    b.destroy()
    with pytest.raises(ValueError, match="manual UDMA"):
        elastic_api.make(transport="udma", udma_auto_policy=False)


def test_v1_manual_config_required_and_auto_conflicts(monkeypatch):
    from test_public_api import _make_manual_buffer

    monkeypatch.setattr("deep_ep.backends.clos.event._get_npu_api", lambda: _FakeNpu())
    monkeypatch.setattr("deep_ep.backends.clos.buffer._use_comm_stream", lambda _: nullcontext())
    x = torch.ones((4, 8), dtype=torch.bfloat16)
    b = _make_manual_buffer(transport="udma")
    for c in (None, Config(20), Config(20, udma_peers_per_worker=1)):
        with pytest.raises(ValueError, match="explicit Config"):
            b.dispatch(x, config=c)
        with pytest.raises(ValueError, match="explicit Config"):
            b.combine(x, None, config=c)
    idx = torch.zeros((4, 1), dtype=topk_idx_t)
    counts, _, experts, member, _ = b.get_dispatch_layout(idx, 4)
    c = Config(20, udma_peers_per_worker=1, udma_wqes_per_visit=16)
    rx, _, _, _, h, _ = b.dispatch(
        x, config=c, num_tokens_per_rank=counts, num_tokens_per_expert=experts, is_token_in_rank=member
    )
    b.dispatch(x, handle=h, config=c)
    b.combine(rx, h, config=c)
    assert b.native.udma_schedule == (1, 16)
    b.destroy()
    b = _make_buffer(transport="udma", udma_schedule="token", udma_qps_per_worker=2)
    assert b.udma_schedule == "peer" and b.udma_qps_per_worker == 1
    b.destroy()


@pytest.mark.parametrize("transport", ["mte", "auto", "roce", "sdma"])
def test_non_udma_defaults_unchanged(transport, elastic_api):
    b = _make_buffer(transport=transport)
    assert b.udma_auto_policy is None and b.udma_workers_per_peer == 1
    b.destroy()
    b = elastic_api.make(transport=transport)
    assert b.udma_auto_policy is None and b.udma_workers_per_peer == 1
    b.destroy()


def test_bootstrap_rejects_auto_policy_disagreement():
    native = Mock()
    native.get_shmem_bootstrap.return_value = (0, [])

    def gather(group, comm, size, value):
        return [value, (value[0][:-1], 0, [])]

    with (
        patch("deep_ep.backends.clos.buffer.load_native_module", return_value=native),
        patch("deep_ep.backends.clos.buffer._bootstrap_all_gather", side_effect=gather),
    ):
        with pytest.raises(ValueError, match="all EP ranks"):
            _prepare_shmem_bootstrap(
                object(), None, 0, 2, 64, "udma", False, None, udma_auto_options=policy(2).signature()
            )
    native.get_shmem_unique_id.assert_not_called()


def test_effective_report_uses_runtime_values(elastic_api):
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
    from bench_v2_bandwidth import effective_udma_config

    b = elastic_api.make(transport="udma", udma_peers_per_worker=8, udma_wqes_per_visit=32)
    effective = effective_udma_config(b)
    assert effective["udma_auto_policy"] is True
    assert effective["udma_peers_per_worker"] == 1
    assert effective["udma_wqes_per_visit"] == 8
    assert effective["udma_workers_per_peer"] == 2
    b.destroy()
    b = elastic_api.manual_make(transport="udma", udma_peers_per_worker=2, udma_wqes_per_visit=16)
    effective = effective_udma_config(b)
    assert effective["udma_auto_policy"] is False
    assert effective["udma_peers_per_worker"] == 2 and effective["udma_wqes_per_visit"] == 16
    b.destroy()


def test_manual_token_and_group_reach_native(elastic_api):
    b = elastic_api.make(transport="udma", udma_auto_policy=False, udma_schedule="token", udma_wqes_per_visit=16)
    assert b.runtime.v1_token_schedule
    elastic_api.fresh(b, num_sms=20)
    assert b.runtime.udma_schedule == (4, 16)
    b.destroy()
    b = elastic_api.make(
        transport="udma",
        udma_auto_policy=False,
        udma_schedule="peer",
        udma_dispatch_peer_order="group_round_robin",
        udma_combine_peer_order="group_round_robin",
        udma_ranks_per_group=4,
        udma_peers_per_group_per_worker=2,
        udma_workers_per_peer=1,
        udma_qps_per_worker=2,
        udma_wqes_per_visit=16,
    )
    assert b.runtime.udma_peer_options == (3, 3, 4, 2, 1, 2)
    elastic_api.fresh(b, num_sms=20)
    b.destroy()
