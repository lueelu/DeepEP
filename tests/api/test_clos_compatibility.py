"""Host routing regressions; dtype-only doubles never execute device operations."""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
DTYPES_ONLY = """
import sys
from types import ModuleType, SimpleNamespace
torch = ModuleType('torch')
torch.int32, torch.int64 = 'int32', 'int64'
sys.modules['torch'] = torch
"""


def run_host(code, *, dtypes=True):
    result = subprocess.run(
        [sys.executable, "-c", (DTYPES_ONLY if dtypes else "") + textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("first", ["ascend_deepep", "ascend_deepep.elastic", "deep_ep"])
def test_legacy_first_import_is_lazy_and_uses_public_entries(first):
    run_host(
        f"""
        import importlib, sys
        importlib.import_module({first!r})
        import ascend_deepep as old, deep_ep as new
        from ascend_deepep.buffer import Buffer
        from ascend_deepep.elastic import ElasticBuffer, EPHandle
        from ascend_deepep.config import Config
        from ascend_deepep.event import EventHandle, EventOverlap
        assert Buffer is old.Buffer is new.Buffer
        assert ElasticBuffer is old.ElasticBuffer is new.ElasticBuffer
        assert EPHandle is old.EPHandle is new.EPHandle
        assert Config is old.Config is new.Config
        assert EventHandle is old.EventHandle is new.EventHandle
        assert EventOverlap is old.EventOverlap is new.EventOverlap
        assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
        assert 'deep_ep._C' not in sys.modules and 'deep_ep._clos_C' not in sys.modules
    """,
        dtypes=False,
    )


def test_legacy_extra_exports_reuse_backend_objects():
    run_host("""
        import ascend_deepep as old, deep_ep as new
        from deep_ep.backends import clos
        from ascend_deepep.expand import ExpandHandle
        from ascend_deepep.resources import Handle, Slot, SlotState
        from ascend_deepep.dtypes import topk_idx_t
        from ascend_deepep.utils import align_up
        for name in ('ExpandHandle', 'Handle', 'Slot', 'SlotState', 'topk_idx_t'):
            assert getattr(old, name) is getattr(new, name) is getattr(clos, name)
        assert align_up is clos.utils.align_up
        assert align_up(33, 32) == 64
        assert isinstance(object.__new__(clos.EPHandle), old.EPHandle)
    """)


@pytest.mark.parametrize("ep", [2, 4, 8, 16, 32, 64, 128])
def test_public_config_and_helpers_work_with_auto_policy(ep):
    run_host(f"""
        import ascend_deepep as old, deep_ep as new
        from deep_ep.backends.clos.config import Config as InternalConfig
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        policy = resolve_udma_policy(True, {ep}, 'udma', None, None, None, None, None)
        for config in (None, new.Config(), old.Config(), InternalConfig(),
                       new.Buffer.get_dispatch_config({ep}), new.Buffer.get_combine_config({ep})):
            resolved = policy.resolve_config(config, 32)
            assert (resolved.num_aiv or 32) == 32
            assert resolved.udma_peers_per_worker == policy.peers_per_worker
        assert new.Config().num_aiv == 32
        assert not new.Config()._udma_peers_explicit
        assert not new.Config()._udma_wqes_explicit
        for keyword in ('num_aiv', 'num_sms', 'num_compute_units'):
            resolved = policy.resolve_config(new.Config(**{{keyword: 64}}), 32)
            assert resolved.num_aiv == 64
        config = new.Config(32, 8, 512, 16, 1024, udma_peers_per_worker=2, udma_wqes_per_visit=16)
        resolved = policy.resolve_config(config, 32)
        assert (resolved.num_max_nvl_chunked_send_tokens, resolved.num_max_nvl_chunked_recv_tokens,
                resolved.num_max_rdma_chunked_send_tokens, resolved.num_max_rdma_chunked_recv_tokens) == (8, 512, 16, 1024)
        assert config.udma_peers_per_worker == 2 and config.udma_wqes_per_visit == 16
        assert config.get_nvl_buffer_size_hint(12288, {ep}, 129) == config._impl.get_nvl_buffer_size_hint(12288, {ep}, 129)
        for bad in (object(), SimpleNamespace(_impl=config._impl)):
            try:
                policy.resolve_config(bad, 32)
            except TypeError:
                pass
            else:
                raise AssertionError('unrelated configuration accepted')
    """)


@pytest.mark.parametrize("operation", ["dispatch", "combine"])
def test_v1_public_config_reaches_native_argument_conversion(operation):
    run_host(f"""
        import deep_ep
        from deep_ep.backends.clos import buffer as m
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        from deep_ep._clos_adapter import Buffer
        class Converted(Exception): pass
        def convert(config):
            assert config.num_aiv == 32
            assert config.udma_peers_per_worker == 1
            raise Converted
        m._config_to_native_args = convert
        m._get_allocation_stream_id = lambda _: 0
        m._is_capturing = lambda: False
        obj = SimpleNamespace(_ensure_alive=lambda: None, num_max_tokens_per_rank=None,
            _udma_auto_policy=resolve_udma_policy(True, 8, 'udma', None, None, None, None, None), rank=0)
        kwargs = {{'config': deep_ep.Config()}}
        if {operation!r} == 'combine': kwargs['handle'] = ()
        try:
            getattr(Buffer, {operation!r})(obj, object(), **kwargs)
        except Converted:
            pass
        else:
            raise AssertionError('public Config did not reach native argument conversion')
    """)


@pytest.mark.parametrize("operation", ["fresh", "cached", "combine"])
@pytest.mark.parametrize("requested", [None, 0, 8, 32, 64])
def test_v2_core_defaults_and_handle_inheritance(operation, requested):
    run_host(f"""
        from deep_ep.backends.clos import elastic as m
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        from deep_ep._clos_adapter import ElasticBuffer
        from contextlib import contextmanager
        class ReadyForDevice(Exception): pass
        counts = []
        policy = resolve_udma_policy(True, 8, 'udma', None, None, None, None, None)
        def resolve(count):
            counts.append(count)
            return policy.for_num_aiv(count)
        @contextmanager
        def stop_before_device(*args):
            raise ReadyForDevice
            yield
        for name in ('_validate_dispatch_scope', '_validate_dispatch_tensors', '_validate_combine_tensors'):
            setattr(m, name, lambda *a: None)
        m._get_allocation_stream_id = lambda _: 0
        m._try_record_event = lambda _: None
        m._use_stream_after_event = stop_before_device
        x = SimpleNamespace(size=lambda dim: (129, 6144)[dim])
        topk = SimpleNamespace(size=lambda dim: (129, 8)[dim])
        handle = SimpleNamespace(num_sms=64, num_experts=256, num_max_tokens_per_rank=129,
            expert_alignment=1, do_expand=False, topk_idx=topk,
            _dispatch_layout=(None,) * 6)
        def no_estimate(*args): raise AssertionError('default still uses theoretical estimator')
        obj = SimpleNamespace(_ensure_alive=lambda: None, _validate_handle=lambda _: None,
            use_fp8_dispatch=False, num_max_tokens_per_rank=129, num_ranks=8, hidden=6144, num_topk=8,
            get_buffer_size_hint=lambda *a: 0, group=None, allow_hybrid_mode=True,
            num_bytes=1, num_device_aiv=64, get_theoretical_num_sms=no_estimate,
            _udma_auto_policy=SimpleNamespace(for_num_aiv=resolve), transport='udma',
            _communication_stream=None, _unpack_bias=lambda _: (None, None),
            _udma_config=SimpleNamespace(udma_peers_per_worker=1, udma_wqes_per_visit=8))
        requested = {requested!r}
        mismatch = {operation!r} != 'fresh' and requested in (8, 32)
        kwargs = {{}} if requested is None else {{'num_sms': requested}}
        try:
            if {operation!r} == 'combine':
                ElasticBuffer.combine(obj, x, handle, **kwargs)
            elif {operation!r} == 'cached':
                ElasticBuffer.dispatch(obj, x, handle=handle, **kwargs)
            else:
                ElasticBuffer.dispatch(obj, x, topk_idx=topk, num_experts=256, **kwargs)
        except ReadyForDevice:
            assert not mismatch
            assert counts == [({requested} or 32) if {operation!r} == 'fresh' else 64]
        except ValueError as exc:
            assert mismatch and 'must match' in str(exc), str(exc)
        else:
            raise AssertionError('did not stop at native submission boundary')
    """)


@pytest.mark.parametrize(
    "ep,cores,peers,workers",
    [
        (4, 2, 4, 2),
        (8, 2, 8, 2),
        (8, 8, 2, 2),
        (16, 2, 8, 1),
        (16, 8, 2, 1),
        (32, 4, 8, 1),
        (32, 8, 4, 1),
        (64, 8, 8, 1),
        (128, 20, 8, 1),
    ],
)
def test_low_core_policy_preserves_core_count_and_runtime_qp_lanes(ep, cores, peers, workers):
    run_host(f"""
        import deep_ep
        from dataclasses import FrozenInstanceError
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        base = resolve_udma_policy(True, {ep}, 'udma', None, None, None, None, None)
        before = base.signature()
        selected = base.for_num_aiv({cores})
        assert (selected.peers_per_worker, selected.workers_per_peer) == ({peers}, {workers})
        assert selected.workers_per_peer == base.workers_per_peer
        assert selected.qps_per_worker == base.qps_per_worker == 1
        assert selected.wqes_per_visit == base.wqes_per_visit
        assert selected.send_workers <= {cores}
        for config in (deep_ep.Config({cores}), base.config({cores})):
            effective = base.resolve_config(config, 32)
            assert effective.num_aiv == {cores}
            assert effective.udma_peers_per_worker == {peers}
        assert base.signature() == before  # A later handle cannot change earlier handles' schedule.
        assert base.for_num_aiv({cores}) == selected
        assert base.for_num_aiv(32) == base
        assert base.for_num_aiv(64) == base
        try: selected.peers_per_worker = 1
        except FrozenInstanceError: pass
        else: raise AssertionError('policy is mutable')
    """)


def test_auto_policy_budget_boundaries_and_manual_opt_out():
    run_host("""
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        for ep in range(2, 257):
            base = resolve_udma_policy(True, ep, 'udma', None, None, None, None, None)
            for cores in range(2, 65, 2):
                feasible = ((ep - 1 + 7) // 8) * base.workers_per_peer <= min(cores, 32)
                try: chosen = base.for_num_aiv(cores)
                except ValueError:
                    assert not feasible, (ep, cores)
                else:
                    assert feasible and chosen.send_workers <= min(cores, 32), (ep, cores)
                    assert chosen.peers_per_worker in (1, 2, 4, 8)
                    assert chosen.config(cores).num_aiv == cores
        base = resolve_udma_policy(True, 32, 'udma', None, None, None, None, None)
        assert base.for_num_aiv(0) == base
        for bad in (False, True, 8.0, -2, 1, 3, 66, 2):
            try: base.for_num_aiv(bad)
            except (TypeError, ValueError): pass
            else: raise AssertionError(('invalid/unrepresentable budget accepted', bad))
        assert base.for_num_aiv(0) == base
        assert resolve_udma_policy(False, 16, 'udma', 'peer', 'rank', 'rank', None, None,
                                   workers=1, qps=1, peers=1, wqes=8) is None
    """)


def test_bandwidth_reports_the_operation_policy_not_the_buffer_default():
    run_host("""
        sys.path.insert(0, 'benchmarks/clos')
        from bench_v2_bandwidth import effective_udma_config
        from deep_ep.backends.clos.config import Config
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        base = resolve_udma_policy(True, 8, 'udma', None, None, None, None, None)
        buffer = SimpleNamespace(udma_auto_policy=base, udma_schedule='peer',
            udma_dispatch_peer_order='rank', udma_combine_peer_order='rank',
            udma_ranks_per_group=None, udma_peers_per_group_per_worker=None,
            udma_workers_per_peer=2, udma_qps_per_worker=1)
        for config in (effective_udma_config(buffer, num_aiv=8), effective_udma_config(buffer, Config(8))):
            assert config['udma_peers_per_worker'] == 2
            assert config['udma_workers_per_peer'] == 2
            assert config['udma_wqes_per_visit'] == 8
        assert effective_udma_config(buffer, num_aiv=32)['udma_peers_per_worker'] == 1
        assert base.peers_per_worker == 1
    """)
