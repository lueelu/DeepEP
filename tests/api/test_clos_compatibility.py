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


def test_topology_entries_share_direct_implementation_without_native_loading():
    run_host("""
        import deep_ep
        from deep_ep.backends.direct import api, buffer, elastic
        assert isinstance(object.__new__(api.Buffer), deep_ep.Buffer)
        assert isinstance(object.__new__(api.ElasticBuffer), deep_ep.ElasticBuffer)
        from deep_ep.backends import global_clos, mesh_pcie
        assert global_clos.Buffer is api.Buffer
        assert global_clos.ElasticBuffer is api.ElasticBuffer
        assert issubclass(mesh_pcie.Buffer, api.Buffer)
        assert issubclass(mesh_pcie.ElasticBuffer, api.ElasticBuffer)
        for entry in (global_clos, mesh_pcie):
            assert entry.Buffer.destroy is buffer.Buffer.destroy
            assert entry.ElasticBuffer.destroy is elastic.ElasticBuffer.destroy
            assert isinstance(object.__new__(entry.Buffer), deep_ep.Buffer)
            assert isinstance(object.__new__(entry.ElasticBuffer), deep_ep.ElasticBuffer)
        assert 'deep_ep._C' not in sys.modules
        assert 'deep_ep._clos_C' not in sys.modules
    """)


@pytest.mark.parametrize(
    "first",
    ["deep_ep", "ascend_deepep"]
    + [
        f"ascend_deepep.{name}"
        for name in ("buffer", "elastic", "config", "event", "expand", "resources", "dtypes", "utils")
    ],
)
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
        from deep_ep.backends import direct as clos
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


def test_consolidated_submodules_preserve_imports_and_exports():
    run_host("""
        import importlib, importlib.util, sys
        import ascend_deepep as old
        aliases = {
            'buffer': ('deep_ep.buffer', ('Buffer',)),
            'elastic': ('deep_ep.buffers.elastic', ('ElasticBuffer', 'EPHandle')),
            'config': ('deep_ep', ('Config',)),
            'event': ('deep_ep.utils.event', ('EventHandle', 'EventOverlap')),
            'expand': ('deep_ep.backends.direct.expand', ('ExpandHandle',)),
            'resources': ('deep_ep.backends.direct.resources', ('Handle', 'Slot', 'SlotState')),
            'dtypes': ('deep_ep.backends.direct.dtypes', ('topk_idx_t',)),
            'utils': ('deep_ep.backends.direct.utils', ('align_up',)),
        }
        for name, (target, exports) in aliases.items():
            fullname = 'ascend_deepep.' + name
            module = importlib.import_module(fullname)
            assert getattr(old, name) is module is sys.modules[fullname]
            assert module.__name__ == fullname
            assert importlib.util.find_spec(fullname).name == fullname
            assert module.__all__ == list(exports)
            assert set(exports) <= set(dir(module))
            scope = {}
            exec('from ' + fullname + ' import *', scope)
            for symbol in exports:
                assert scope[symbol] is getattr(importlib.import_module(target), symbol)
            try:
                module.not_a_legacy_export
            except AttributeError:
                pass
            else:
                raise AssertionError('unknown compatibility export accepted')
        saved = old.buffer
        importlib.reload(old)
        assert old.buffer is saved
        scope = {'__package__': 'ascend_deepep'}
        exec('from .buffer import Buffer', scope)
        assert scope['Buffer'] is old.Buffer
        assert 'deep_ep._C' not in sys.modules and 'deep_ep._clos_C' not in sys.modules
    """)


@pytest.mark.parametrize("ep", [2, 4, 8, 16, 32, 64, 128])
def test_public_config_and_helpers_work_with_auto_policy(ep):
    run_host(f"""
        import ascend_deepep as old, deep_ep as new
        from deep_ep.backends.direct.config import Config as InternalConfig
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
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
        from deep_ep.backends.direct import buffer as m
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        from deep_ep.backends.direct.api import Buffer
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
        from deep_ep.backends.direct import elastic as m
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        from deep_ep.backends.direct.api import ElasticBuffer
        from contextlib import contextmanager
        class ReadyForDevice(Exception): pass
        counts = []
        policy = resolve_udma_policy(True, 8, 'udma', None, None, None, None, None)
        def resolve(count, *args, **kwargs):
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
        x = SimpleNamespace(size=lambda dim: (129, 6144)[dim], element_size=lambda: 2, dtype="torch.bfloat16")
        topk = SimpleNamespace(size=lambda dim: (129, 8)[dim])
        handle = SimpleNamespace(num_sms=64, num_experts=256, num_max_tokens_per_rank=129,
            expert_alignment=1, do_expand=False, topk_idx=topk,
            _dispatch_layout=(None,) * 6)
        def no_estimate(*args): raise AssertionError('default still uses theoretical estimator')
        obj = SimpleNamespace(_ensure_alive=lambda: None, _validate_handle=lambda _: None,
            use_fp8_dispatch=False, num_max_tokens_per_rank=129, num_ranks=8, hidden=6144, num_topk=8,
            get_buffer_size_hint=lambda *a: 0, group=None, allow_hybrid_mode=True,
            num_bytes=1, num_device_aiv=64, get_theoretical_num_sms=no_estimate,
            _udma_auto_policy=SimpleNamespace(for_launch=resolve), transport='udma',
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
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
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
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        for ep in range(2, 257):
            base = resolve_udma_policy(True, ep, 'udma', None, None, None, None, None)
            for cores in range(2, 65, 2):
                feasible = ((ep - 1 + 7) // 8) * base.workers_per_peer <= min(cores, 32)
                try: chosen = base.for_num_aiv(cores)
                except ValueError:
                    assert not feasible, (ep, cores)
                else:
                    assert chosen.valid_group_capacity() and chosen.send_workers <= min(cores, 32), (ep, cores)
                    assert chosen.peers_per_worker in (1, 2, 4, 8)
                    assert chosen.config(cores).num_aiv == cores
        base = resolve_udma_policy(True, 32, 'udma', None, None, None, None, None)
        assert base.for_num_aiv(0) == base
        for bad in (False, True, 8.0, -2, 1, 3, 66):
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
        from deep_ep.backends.direct.config import Config
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
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


def test_udma_combine_core_overrides_reach_native_and_preserve_handle():
    run_host("""
        from contextlib import nullcontext
        from deep_ep.backends.direct import elastic as m
        calls, policies = [], []
        epilogue = SimpleNamespace(run=lambda: ("out", None))
        def begin(*args):
            calls.append(args)
            return epilogue
        b = object.__new__(m.ElasticBuffer)
        b._destroyed = False
        b._native = SimpleNamespace(_intranode_combine_begin=begin)
        b.num_device_aiv = 64
        b.transport = "udma"
        b._udma_auto_policy = SimpleNamespace(for_launch=lambda n, *a, **kw: (
            policies.append(n) or SimpleNamespace(peers_per_worker=4, wqes_per_visit=8, native_plan=lambda: [])))
        b._aggregate_pipeline_chunk_tokens = 0
        b._communication_stream = None
        b._validate_handle = lambda h: None
        b._unpack_bias = lambda bias: (None, None)
        b._finish_comm_event = lambda *a: "event"
        m._validate_combine_tensors = lambda *a: None
        m._get_allocation_stream_id = lambda *a: None
        m._try_record_event = lambda *a: None
        m._use_stream_after_event = lambda *a: nullcontext()
        m._record_recv_outputs_on_current_stream = lambda *a: None
        layout = tuple(object() for _ in range(6))
        h = SimpleNamespace(num_sms=48, do_expand=False, _dispatch_layout=layout,
                            num_max_tokens_per_rank=128)
        for opts, expected in [({}, (48,48)),
                ({'combine_num_sms':8}, (8,48)),
                ({'reduce_num_sms':8}, (48,8)),
                ({'num_sms':48,'combine_num_sms':20,'reduce_num_sms':8}, (20,8)),
                ({'combine_num_sms':64,'reduce_num_sms':16}, (64,16))]:
            assert b.combine(SimpleNamespace(size=lambda dim: (128,2816)[dim], element_size=lambda: 2, dtype="torch.bfloat16"), h, **opts) == ("out", None, "event")
            assert (calls[-1][8], calls[-1][-2]) == expected
            assert policies[-1] == expected[0]
            assert calls[-1][6] is layout[2]
            assert h.num_sms == 48 and h._dispatch_layout is layout
        for field in ('combine_num_sms', 'reduce_num_sms'):
            for value in (-2, 3, 66, True, 8.0):
                before = len(calls)
                try:
                    b.combine(SimpleNamespace(size=lambda dim: (128,2816)[dim], element_size=lambda: 2, dtype="torch.bfloat16"), h, **{field:value})
                except (ValueError, TypeError):
                    pass
                else:
                    raise AssertionError((field,value))
                assert len(calls) == before
        # Expanded input aggregation retains the original core count.
        aggregates = []
        b._native.aggregate = lambda *args: (aggregates.append(args) or (args[0], None))
        b._use_single_source_combine = False
        h.do_expand = True
        h._expand_layout = (None, None, object(), None)
        b.combine(SimpleNamespace(size=lambda dim: (128,2816)[dim], element_size=lambda: 2, dtype="torch.bfloat16"), h, combine_num_sms=8, reduce_num_sms=16)
        assert aggregates[-1][4] == 48
        assert (calls[-1][8], calls[-1][-2]) == (8,16)
        before = len(calls)
        b._aggregate_pipeline_chunk_tokens = 32
        try:
            b.combine(SimpleNamespace(size=lambda dim: (128,2816)[dim], element_size=lambda: 2, dtype="torch.bfloat16"), h, combine_num_sms=8)
        except ValueError as e:
            assert 'pipeline' in str(e)
        else:
            raise AssertionError('pipeline accepted changed channel count')
        assert len(calls) == before
        h.do_expand = False
        b._aggregate_pipeline_chunk_tokens = 0
        b.num_device_aiv = 56
        for field in ('combine_num_sms', 'reduce_num_sms'):
            try:
                b.combine(SimpleNamespace(size=lambda dim: (128,2816)[dim], element_size=lambda: 2, dtype="torch.bfloat16"), h, **{field:64})
            except ValueError:
                pass
            else:
                raise AssertionError('device core limit ignored')
        b.transport = "mte"
        for opts in ({'combine_num_sms':8}, {'reduce_num_sms':8}):
            try:
                b.combine(SimpleNamespace(size=lambda dim: (128,2816)[dim], element_size=lambda: 2, dtype="torch.bfloat16"), h, **opts)
            except ValueError as e:
                assert 'UDMA' in str(e)
            else:
                raise AssertionError('MTE accepted independent cores')
        b._destroyed = True
    """)


def test_launch_profiles_are_immutable_cached_and_topology_scoped():
    run_host("""
        from dataclasses import replace, FrozenInstanceError
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy, _PROFILES
        for (ep, cores, bucket, op), (groups, order, p, w, q, visit, copy) in _PROFILES.items():
            for group in groups:
                base = replace(resolve_udma_policy(True, ep, 'udma', None, None, None, None, None),
                               ranks_per_group=group, topology_known=True)
                before = base.signature()
                args = (cores, 2048 if bucket == 'small' else 5632, op)
                plan = base.for_launch(*args, bf16=True)
                assert plan is base.for_launch(*args, bf16=True)
                assert plan.native_plan()[:7] == [1, order, group, p, w, q, visit]
                assert plan.self_copy == copy
                assert plan.send_workers <= cores and plan.send_workers * q <= 32
                assert plan.valid_group_capacity()
                assert base.signature() == before
                try: plan.peers_per_worker = 8
                except FrozenInstanceError: pass
                else: raise AssertionError('mutable launch policy')
                for kwargs in ({'expanded': True}, {'v2': False}, {'bf16': False}):
                    options = {'bf16': True, **kwargs}
                    fallback = base.for_launch(*args, **options)
                    assert fallback.peer_order == 0 and fallback.self_copy == fallback.weight_mode == 0
        base = resolve_udma_policy(True, 32, 'udma', None, None, None, None, None)
        for known, group in ((True,16), (False,8)):
            policy = replace(base, ranks_per_group=group, topology_known=known)
            assert policy.for_launch(32,5632,'combine',bf16=True).peer_order == 0
        # Weighted low-core cases must not opportunistically take the SIMT path.
        low = replace(base, world_size=8, ranks_per_group=8, topology_known=True)
        assert low.for_launch(8,5632,'weighted',bf16=True).weight_mode == 1
    """)


def test_auto_topology_uses_ep_rank_order_and_collects_only_at_construction():
    run_host("""
        from deep_ep import _machine
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy, _contiguous_group_size
        assert _contiguous_group_size(['a']*16 + ['b']*16) == 16
        assert _contiguous_group_size(['a','b','a','b']) == 0
        assert _contiguous_group_size(['a']*8 + ['b']*16) == 0
        assert _contiguous_group_size([None]*8) == 0
        calls = []
        group = object()
        def gather(g, record):
            assert g is group
            calls.append(record)
            return ['a']*16 + ['b']*16
        _machine.gather_records = gather
        policy = resolve_udma_policy(True,32,'udma',None,None,None,None,None).bind_group(group)
        assert policy.ranks_per_group == 16 and policy.topology_known
        for cores in (8,20,32,64):
            for op in ('dispatch','combine','weighted'):
                policy.for_launch(cores,5632,op,bf16=True)
        assert len(calls) == 1
    """)


def test_large_ep_low_core_fallback_and_invalid_launch_inputs():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        for ep in (4,8,16,32,64,128):
            for group in (8,16):
                if group > ep: continue
                base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                               ranks_per_group=group, topology_known=True)
                for cores in (8,20,32,64):
                    for size in (1024,2048,4096,5632,12288,14336):
                        for op in ('dispatch','combine','weighted'):
                            p = base.for_launch(cores,size,op,bf16=True)
                            assert p.send_workers <= cores and p.send_workers*p.qps_per_worker <= 32
                            assert p.valid_group_capacity()
                            assert p.ranks_per_group == group
                if ep == 128:
                    assert base.for_launch(8,5632,'combine',bf16=True).peer_order == 3
                for cores in (True,False,8.0,3,-2,66):
                    try: base.for_launch(cores,5632,'dispatch',bf16=True)
                    except (TypeError,ValueError): pass
                    else: raise AssertionError(cores)
    """)


def test_benchmark_reports_separate_launch_plans():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        sys.path.insert(0, 'benchmarks/clos')
        from bench_v2_bandwidth import effective_udma_config
        base = replace(resolve_udma_policy(True,8,'udma',None,None,None,None,None),
                       ranks_per_group=8,topology_known=True)
        b = SimpleNamespace(udma_auto_policy=base, udma_schedule='peer',
            udma_dispatch_peer_order='rank',udma_combine_peer_order='rank',udma_ranks_per_group=8,
            udma_peers_per_group_per_worker=None,udma_workers_per_peer=2,udma_qps_per_worker=1)
        result = effective_udma_config(b,num_aiv=8,row_bytes=5632,bf16=True)
        plans = result['operation_plans']
        assert plans['dispatch']['qps_per_worker'] == 2
        assert plans['weighted']['qps_per_worker'] == 1
        assert plans['weighted']['weight_mode'] == 1
        assert plans['combine']['self_copy'] == 2
    """)


def test_reserved_native_masks_cover_all_supported_launch_owners(tmp_path):
    import shutil

    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler required for native schedule coverage")
    source = tmp_path / "schedule.cpp"
    source.write_text(
        r"""
        #include "common/udma_schedule.h"
        #include <cassert>
        using namespace ascend_deepep::udma;
        int main() {
            for (unsigned ep : {4U,8U,16U,32U,64U,128U})
              for (unsigned g : {4U,8U,16U}) {
                if (g > ep) continue;
                for (unsigned rank=0; rank<ep; ++rank)
                  for (unsigned order=0; order<=3; ++order)
                    for (unsigned p : {1U,2U,4U,8U}) {
                      if (order==3 && p==8) continue;
                      unsigned owners=Workers(ep,p,order,g);
                      for (unsigned lanes : {1U,2U}) {
                        if (owners*lanes>32) continue;
                        for (unsigned owner=0; owner<owners; ++owner) {
                          unsigned count=WorkerCount(rank,ep,owner,p,order,g);
                          for (unsigned i=0; i<count; ++i) {
                            unsigned peer=WorkerPeer(rank,ep,owner,p,i,order,g);
                            assert(peer<ep && peer!=rank);
                            assert(Owner(rank,peer,ep,p,order,g)==owner);
                            auto mask=TuningPeerMask(rank,peer,ep,g);
                            assert(mask==TuningPeerMask(peer,rank,ep,g));
                            for (unsigned lane=0; lane<lanes; ++lane)
                              assert(mask & (1ULL << (owner*lanes+lane)));
                          }
                        }
                      }
                    }
              }
        }
    """.replace("#include <cassert>", "#include <cassert>\n#include <initializer_list>")
    )
    binary = tmp_path / "schedule"
    built = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(ROOT / "csrc/direct"), str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    subprocess.run([str(binary)], check=True, timeout=60)
