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


def test_multi_buffer_bootstrap_uses_fresh_uid_and_capacity_independent_identity():
    run_host("""
        from deep_ep.backends.direct import buffer
        keys, uids = [], []
        def state(key):
            keys.append(key)
            return len(keys) - 1, []
        def unique():
            uids.append([len(uids) + 1])
            return uids[-1]
        buffer.load_native_module = lambda: SimpleNamespace(
            get_shmem_bootstrap=state, get_shmem_unique_id=unique)
        buffer._bootstrap_all_gather = lambda group, comm, size, value: [value] * size
        group = object()
        first = buffer._prepare_shmem_bootstrap(group, None, 0, 2, 64 << 20, 'udma', False, None, 17)
        second = buffer._prepare_shmem_bootstrap(group, None, 0, 2, 128 << 20, 'udma', False, None, 129)
        assert first[0] != second[0] and first[1] == second[1]
        assert keys[0] == keys[1] and len(uids) == 2
    """)


def test_extended_visit_config_is_explicit_and_preserved():
    run_host("""
        from deep_ep.backends.direct.config import Config
        for visit in (4,8,16,32,64,128):
            config = Config(8, udma_peers_per_worker=8, udma_wqes_per_visit=visit)
            assert config.udma_wqes_per_visit == visit and config._udma_wqes_explicit
        assert Config().udma_wqes_per_visit == 8
        for bad in (0,33,96,129,256,(1<<32)+64):
            try:
                Config(8, udma_wqes_per_visit=bad)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid visit accepted')
    """)


def test_qp_reservation_participates_in_rank_consensus_and_buffer_identity():
    run_host("""
        import os
        from deep_ep.backends.direct import buffer
        buffer.load_native_module = lambda: SimpleNamespace(get_shmem_bootstrap=lambda key: (0, []))
        buffer._bootstrap_all_gather = lambda group, comm, size, value: [value] * size
        group = object()
        def prepare():
            return buffer._prepare_shmem_bootstrap(group, None, 0, 2, 64<<20, 'udma', False, 'tcp://host:123')
        os.environ.pop('DEEPEP_CLOS_QP_PLANS', None)
        original = prepare()[1]
        os.environ['DEEPEP_CLOS_QP_PLANS'] = 'rank:8:1:1'
        reserved = prepare()[1]
        assert original != reserved and prepare()[1] == reserved
        def mismatched(group, comm, size, value):
            options, state, uid = value
            return [value, (options[:-1], state, uid)]
        buffer._bootstrap_all_gather = mismatched
        try:
            prepare()
        except ValueError:
            pass
        else:
            raise AssertionError('QP reservation mismatch was accepted')
    """)


@pytest.mark.parametrize("peer_state", [-1, 7])
def test_multi_buffer_bootstrap_rejects_incompatible_or_divergent_ids_before_uid(peer_state):
    run_host(f"""
        from deep_ep.backends.direct import buffer
        def forbidden():
            raise AssertionError('must reject before opening a UID listener')
        buffer.load_native_module = lambda: SimpleNamespace(
            get_shmem_bootstrap=lambda key: (1, []), get_shmem_unique_id=forbidden)
        buffer._bootstrap_all_gather = lambda group, comm, size, value: [value, (value[0], {peer_state}, [])]
        try:
            buffer._prepare_shmem_bootstrap(object(), None, 0, 2, 64 << 20, 'udma', False, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError('rank disagreement accepted')
    """)


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
                    assert chosen.peers_per_worker in (1, 2, 4, 8, 16)
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
        b._use_early_single_source_combine = False
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
            assert (calls[-1][8], calls[-1][-3]) == expected
            assert calls[-1][-1] is layout[1]
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
        assert (calls[-1][8], calls[-1][-3]) == (8,16)
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


def test_serial_expanded_combine_uses_unexpanded_plan_after_aggregate():
    run_host("""
        from contextlib import nullcontext
        from dataclasses import replace
        from deep_ep.backends.direct import elastic as m
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        m._validate_combine_tensors = lambda *a: None
        m._get_allocation_stream_id = lambda *a: None
        m._try_record_event = lambda *a: None
        m._use_stream_after_event = lambda *a: nullcontext()
        m._record_recv_outputs_on_current_stream = lambda *a: None
        b = object.__new__(m.ElasticBuffer)
        b._destroyed = False
        b.num_device_aiv = 64
        b.transport = 'udma'
        b._udma_schedule = 'peer'
        b._udma_qps_per_worker = 1
        b._communication_stream = None
        b._validate_handle = lambda h: None
        b._unpack_bias = lambda bias: (None,None)
        b._finish_comm_event = lambda *a: None
        x = SimpleNamespace(size=lambda dim: (128,2816)[dim],element_size=lambda:2,dtype='torch.bfloat16')
        h = SimpleNamespace(num_sms=32,do_expand=True,num_max_tokens_per_rank=128,
            _dispatch_layout=tuple(object() for _ in range(6)),
            _expand_layout=(None,None,object(),None),_single_source_slots=object())
        base = replace(resolve_udma_policy(True,4,'udma',None,None,None,None,None),
                       ranks_per_group=4,topology_known=True)
        b._udma_auto_policy = base
        for early, chunk in ((False,0),(True,0),(False,32)):
            b._use_early_single_source_combine = early
            b._aggregate_pipeline_chunk_tokens = chunk
            for single in (False,True):
                if chunk and not single: continue
                b._use_single_source_combine = single
                for weights in (None,object()):
                    events, calls = [], []
                    def aggregate(*args):
                        events.append('aggregate')
                        return x,weights
                    def begin(*args):
                        events.append('begin')
                        calls.append(args)
                        return SimpleNamespace(run=lambda: (x,weights))
                    b._native = SimpleNamespace(aggregate=aggregate,_intranode_combine_begin=begin)
                    b.combine(x,h,topk_weights=weights)
                    overlap = early or bool(chunk)
                    expected = base.for_launch(32,5632,'weighted' if weights is not None else 'combine',
                                               bf16=True,expanded=overlap)
                    assert calls[-1][-2] == expected.native_plan()
                    assert calls[-1][-1] is h._dispatch_layout[1]
                    assert events == (['begin'] if single and overlap else ['aggregate','begin'])
                    assert calls[-1][15] is (x if single else None)
                    assert calls[-1][16] is (h._single_source_slots if single else None)
                    assert calls[-1][17] is (h._expand_layout[2] if single and overlap else None)
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
                expected_order = 0
                rank_p = p * (ep//group if order == 3 else 1)
                assert plan.native_plan()[:7] == [1, expected_order, group, rank_p, w, q, visit]
                assert plan.self_copy == (copy or 2)
                assert plan.send_workers <= cores and plan.send_workers * q <= 32
                assert plan.valid_group_capacity()
                assert base.signature() == before
                try: plan.peers_per_worker = 8
                except FrozenInstanceError: pass
                else: raise AssertionError('mutable launch policy')
                for kwargs in ({'expanded': True}, {'v2': False}, {'bf16': False}):
                    options = {'bf16': True, **kwargs}
                    fallback = base.for_launch(*args, **options)
                    assert fallback.peer_order == 0 and fallback.self_copy == 2 and fallback.weight_mode == 0
        base = resolve_udma_policy(True, 32, 'udma', None, None, None, None, None)
        for known, group in ((True,16), (False,8)):
            policy = replace(base, ranks_per_group=group, topology_known=known)
            assert policy.for_launch(32,5632,'combine',bf16=True).peer_order == 0
    """)


def test_weighted_combine_uses_row_bytes_and_spare_aivs():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy

        for ep, group, cores in ((4,2,8), (8,8,20), (16,8,20), (32,8,32), (64,8,48), (128,16,64)):
            base = resolve_udma_policy(True,ep,'udma',None,None,None,None,None)
            policy = replace(base, ranks_per_group=group, topology_known=True)
            for row_bytes, mode in ((5632,2), (7168,2), (7170,1), (8192,1), (12288,1)):
                plan = policy.for_launch(cores,row_bytes,'weighted',bf16=True)
                assert plan.send_workers < cores, (ep, cores, plan)
                expected = 0 if ep == 128 else 2 if ep == 64 or (ep in (16,32) and row_bytes <= 8192) else mode
                if ep in (8,16):
                    expected = policy.for_launch(8 if cores < 32 else 32,row_bytes,'weighted',bf16=True).weight_mode
                assert plan.weight_mode == expected, (ep, cores, row_bytes, plan)
        # Rank ordering frees the RR self-owner slot without changing UDMA.
        base = resolve_udma_policy(True,8,'udma',None,None,None,None,None)
        policy = replace(base, ranks_per_group=8, topology_known=True)
        plan = policy.for_launch(8,7168,'weighted',bf16=True)
        assert plan.send_workers == 7 and plan.weight_mode == 1

        base = resolve_udma_policy(True,64,'udma',None,None,None,None,None)
        for group, known, cores in ((16,True,32), (8,True,32), (16,False,32), (16,True,48)):
            policy = replace(base, ranks_per_group=group, topology_known=known)
            for row_bytes in (8192, 9216, 12288):
                plan = policy.for_launch(cores,row_bytes,'weighted',bf16=True)
                assert plan.weight_mode == 2
    """)


def test_ep16_weight_row_limit_preserves_payload_plan_across_core_budgets():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy

        base = resolve_udma_policy(True,16,'udma',None,None,None,None,None)
        for group in (1,2,4,8,16):
            for known in (False,True):
                policy = replace(base,ranks_per_group=group,topology_known=known)
                for cores in range(8,65,2):
                    reference = policy.for_launch(cores,12288,'weighted',bf16=True)
                    assert reference.weight_mode == 1
                    for size in (5632,7168,7170,8190,8192,8194,12288):
                        plan = policy.for_launch(cores,size,'weighted',bf16=True)
                        tier = 8 if cores < 32 else 32
                        expected = 2 if size <= 8192 and plan.send_workers < tier else 1
                        assert plan.weight_mode == expected, (group,known,cores,size,plan)
                        assert replace(plan,weight_mode=1) == reference
        # The updated visit applies throughout the existing 32-core tier.
        policy = replace(base,ranks_per_group=8,topology_known=True)
        assert policy.for_launch(48,8192,'weighted',bf16=True).native_plan() == [1,0,8,2,1,1,8,2,2]
        assert policy.for_launch(48,12288,'weighted',bf16=True).native_plan() == [1,0,8,2,1,1,8,2,1]
    """)


def test_fitted_weight_payload_plans_and_reports_follow_row_preference():
    run_host("""
        from dataclasses import asdict, replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        sys.path.insert(0, 'benchmarks/clos')
        from bench_v2_bandwidth import effective_udma_config

        for ep, group, mode in ((32,8,1), (64,16,2)):
            base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                           ranks_per_group=group, topology_known=True)
            before = base.signature()
            for cores in ((0,32,34,48,62) if ep == 32 else (0,32,34,48,62,64)):
                for row_bytes in (8192,8193,8194,9216,10240,12286,12287,12288):
                    plan = base.for_launch(cores,row_bytes,'weighted',bf16=True)
                    expected_mode = 2 if row_bytes <= 8192 else mode
                    expected_order, expected_p = (0,4)
                    assert plan.native_plan() == [1,expected_order,group,expected_p,1,1,8,2,expected_mode]
                    assert plan.send_workers == group
                    assert plan.send_workers < 32 and plan.valid_group_capacity()
                    assert plan is base.for_launch(cores,row_bytes,'weighted',bf16=True)
                    buffer = SimpleNamespace(udma_auto_policy=base, udma_schedule='peer',
                        udma_dispatch_peer_order='rank',udma_combine_peer_order='rank',
                        udma_ranks_per_group=group,udma_peers_per_group_per_worker=None,
                        udma_workers_per_peer=1,udma_qps_per_worker=1)
                    report = effective_udma_config(buffer,num_aiv=cores,row_bytes=row_bytes,bf16=True)
                    assert report['operation_plans']['weighted'] == asdict(plan)
            # Unweighted and dispatch retain their existing plans at both widths.
            for op in ('dispatch','combine'):
                assert base.for_launch(32,8192,op,bf16=True) == base.for_launch(32,5632,op,bf16=True)
                assert base.for_launch(32,12288,op,bf16=True) == base.for_launch(32,5632,op,bf16=True)
            # Width/topology scopes remain unchanged; EP32 combine is not inherited.
            for size in (7168,7170,8190,8191,12289,12290):
                p = base.for_launch(32,size,'weighted',bf16=True)
                assert p.peer_order == 0
            for cores in ((8,20,64) if ep == 32 else (8,20)):
                p = base.for_launch(cores,8192,'weighted',bf16=True)
                assert p.peer_order == 0
                weight_cores = (8 if cores < 32 else 32) if ep == 64 else cores
                assert p.weight_mode == (2 if p.send_workers < weight_cores else 1)
            for other in (replace(base,topology_known=False), replace(base,ranks_per_group=24-group)):
                p = other.for_launch(32,8192,'weighted',bf16=True)
                assert p.peer_order == 0 and p.weight_mode == 2
            for opts in ({'expanded':True}, {'v2':False}, {'bf16':False}):
                p = base.for_launch(32,8192,'weighted',**{'bf16':True,**opts})
                assert p.native_plan()[1:] == [0,group,4,1,1,32,2,0]
            assert base.signature() == before
    """)


def test_weight_preference_does_not_return_to_simt_as_rows_grow():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy, _simt_weight_row_fits

        for ep, limit in ((4,7168),(16,8192),(32,8192),(48,7168),(64,12288),(96,12288)):
            assert _simt_weight_row_fits(ep,limit)
            assert not _simt_weight_row_fits(ep,limit+1)
        for ep in (128,192,256):
            assert _simt_weight_row_fits(ep,1<<20)
        sizes = sorted(set(range(256,32769,256)) | {2047,2049,4095,4097,7167,7169,8191,8193,12287,12289})
        for ep in (4,8,16,32,48,64,96,128,192,256):
            for group in (1,2,4,8,16):
                if ep % group: continue
                for known in (False,True):
                    base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                                   ranks_per_group=group,topology_known=known)
                    for cores in range(8,65,2):
                        try: base.for_num_aiv(cores)
                        except ValueError: continue
                        udma_seen = False
                        for size in sizes:
                            plan = base.for_launch(cores,size,'weighted',bf16=True)
                            # Legacy mode 0 uses SIMT when a helper exists,
                            # under the normal SIMT-enabled peer kernel build.
                            preferred = _simt_weight_row_fits(ep,size)
                            weight_cores = (8 if cores < 32 else 32) if ep in (8,16,64) else min(cores,32)
                            simt = plan.weight_mode != 1 and plan.send_workers < weight_cores
                            # The low-core rank override preserves the weight
                            # mode selected before removing the RR self owner.
                            rank_keeps_udma = weight_cores == 8 and known and (
                                (ep == 8 and group == 8 and 2048 <= size <= 12288) or
                                (ep == 4 and group == 4 and 4096 < size <= 12288))
                            assert simt == (preferred and plan.send_workers < weight_cores and not rank_keeps_udma)
                            # Capacity fallback is separate from the row rule:
                            # no helper must remain legal even in the SIMT band.
                            assert not (udma_seen and preferred), (ep,size)
                            udma_seen |= not preferred
    """)


def test_core_tiers_preserve_actual_budgets_and_capacity():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy, _PROFILES

        for (ep, tier, bucket, op), (groups, order, p, w, q, visit, copy) in _PROFILES.items():
            for group in groups:
                base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                               ranks_per_group=group, topology_known=True)
                for cores in (range(8,16,2) if tier == 8 else range(32,64,2)):
                    plan = base.for_launch(cores,2048 if bucket == 'small' else 5632,op,bf16=True)
                    rank_p = p * (ep//group if order == 3 else 1)
                    assert plan.native_plan()[:7] == [1,0,group,rank_p,w,q,visit]
                    assert base.config(cores).num_aiv == cores
                    selected_senders = ((group+p-1)//p if order == 3 else (ep-2)//p+1)*w
                    if op == 'combine':
                        assert plan.self_copy == (1 if cores >= 2*selected_senders else 2)
                    if op == 'weighted':
                        weight_cores = tier if ep in (8,16) else cores
                        assert plan.weight_mode == (2 if weight_cores > selected_senders else 1)
        # Scan legal budgets and both sides of row-width boundaries, including
        # unknown topology and widths outside the measured profiles.
        for ep in (4,8,16,32,64,128,256):
            for group in (1,min(ep,8),min(ep,16)):
                for known in (False,True):
                    base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                                   ranks_per_group=group, topology_known=known)
                    for cores in range(2,65,2):
                        for size in (2048,4096,5632,7168,8190,8192,10240,12288,12290,14336):
                            for op in ('dispatch','combine','weighted'):
                                try:
                                    plan = base.for_launch(cores,size,op,bf16=True)
                                except ValueError:
                                    # A budget rejected by the base allocator is
                                    # not a new tier-selection failure.
                                    try: base.for_num_aiv(cores)
                                    except ValueError: continue
                                    raise
                                assert plan.send_workers <= cores
                                assert plan.send_workers*plan.qps_per_worker <= 32
                                assert plan.valid_group_capacity()
    """)


def test_low_core_rank_defaults_match_measured_plans():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy

        # Rank candidates keep sender capacity valid across the whole row interval.
        expected = {
            (4,'dispatch'): (1,1,2,32,3,0),
            (4,'combine'): (1,2,1,32,6,0),
            (4,'weighted'): (1,2,1,32,6,1),
            (8,'dispatch'): (2,2,1,32,8,0),
            (8,'combine'): (1,1,1,32,7,0),
            (8,'weighted'): (1,1,1,32,7,1),
            (16,'dispatch'): (2,1,1,16,8,0),
            (16,'combine'): (2,1,1,8,8,0),
            (16,'weighted'): (2,1,1,8,8,1),
            (32,'dispatch'): (4,1,1,32,8,0),
            (32,'combine'): (4,1,1,32,8,0),
            (32,'weighted'): (4,1,1,32,8,1),
        }
        for (ep,op),(p,w,q,visit,senders,mode) in expected.items():
            group = min(ep,16)
            base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                           ranks_per_group=group,topology_known=True)
            original = base.native_plan()
            for hidden in (2816,3072,4096,5120,6144):
                plan = base.for_launch(8,hidden*2,op,bf16=True)
                assert plan.native_plan() == [1,0,group,p,w,q,visit,2,mode]
                assert plan.send_workers == senders
                assert plan.valid_group_capacity()
            assert base.native_plan() == original
    """)


def test_32_core_rank_defaults_preserve_payload_and_weight_plans():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy

        # EP8/16 rank candidates and EP32 visit32 retain the actual core budget.
        # EP4 preserves its pre-existing rank configuration.
        # P/W/Q, visit, sender count, self-copy, H2816/H6144 weight mode.
        expected = {
            (4,'dispatch'): (1,2,1,32,6,2,0,0),
            (4,'combine'): (1,2,1,32,6,1,0,0),
            (4,'weighted'): (1,2,1,32,6,2,2,1),
            (8,'dispatch'): (1,1,1,16,7,2,0,0),
            (8,'combine'): (1,1,1,32,7,1,0,0),
            (8,'weighted'): (1,1,1,32,7,2,2,1),
            (16,'dispatch'): (4,2,1,16,8,2,0,0),
            (16,'combine'): (2,1,1,8,8,1,0,0),
            (16,'weighted'): (2,1,1,8,8,2,2,1),
            (32,'dispatch'): (4,1,1,32,8,2,0,0),
            (32,'combine'): (4,1,1,32,8,1,0,0),
            (32,'weighted'): (4,1,1,32,8,2,2,1),
        }
        for (ep,op),(p,w,q,visit,senders,copy,small_mode,large_mode) in expected.items():
            group = min(ep,16)
            base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                           ranks_per_group=group,topology_known=True)
            for cores in (0,32,34,48,62):
                for hidden,mode in ((2816,small_mode),(6144,large_mode)):
                    plan = base.for_launch(cores,hidden*2,op,bf16=True)
                    assert plan.native_plan() == [1,0,group,p,w,q,visit,copy,mode]
                    assert plan.send_workers == senders
                    assert plan.send_workers*q <= 32
        # Rank is also selected outside fitted topologies/row widths, without
        # manufacturing physical groups or changing the user's actual budget.
        for ep in (4,8,16,32):
            for group in (1,2,4,ep):
                for known in (False,True):
                    base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                                   ranks_per_group=group,topology_known=known)
                    for cores in (0,32,34,48,62):
                        for size in (2046,2048,4096,4098,5632,8192,12288,12290,14336):
                            for op in ('dispatch','combine','weighted'):
                                plan = base.for_launch(cores,size,op,bf16=True)
                                assert plan.peer_order == 0
                                assert plan.ranks_per_group == group
                                assert plan.send_workers <= (cores or 32)
    """)


def test_reported_300_operator_plans_allow_rank_conversion_and_copy_first_removal():
    # Frozen native plans from the November 16 EP8-128 reports, not generated
    # from the selector. Keep all five operators for each of the 60 shapes.
    run_host("""
        import csv
        import json
        from dataclasses import replace
        from pathlib import Path
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy

        with Path('tests/api/fixtures/udma_plans_20261116.csv').open() as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == 300
        cases = {}
        operations = {'dispatch_with_weights', 'dispatch_without_weights',
                      'cached_dispatch', 'combine_weighted', 'combine_unweighted'}
        for row in rows:
            key = tuple(int(row[k]) for k in ('EP','tokens','hidden','topk','experts','aiv','G'))
            ep, tokens, hidden, topk, experts, cores, group = key
            cases.setdefault(key, []).append(row['operator'])
            op = {'combine_weighted':'weighted', 'combine_unweighted':'combine'}.get(row['operator'], 'dispatch')
            base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                           ranks_per_group=group, topology_known=True)
            plan = base.for_launch(cores,hidden*2,op,bf16=True)
            historical = json.loads(row['native_plan'])
            # Preserve the historical fixture; normalize rank order,
            # its RR ownership width, copy-first, and the EP32 combine visit.
            # Never regenerate the recorded measurements.
            historical[7] = historical[7] or 2
            senders = int(row['send_aivs'])
            if ep == 64 or (ep in (4,8,16,32) and cores == 32):
                if historical[1] == 3:
                    historical[3] *= ep//group
                historical[1] = 0
                senders = ((ep-2)//historical[3]+1)*historical[4]
            if ep == 32 and op == 'combine':
                historical[6] = 32
            if ep == 64 and cores == 32 and op == 'dispatch':
                # Intentional sender-budget change; retain the raw fixture.
                historical[3:6] = [8,1,1]
                senders = 8
            if cores > 32:
                # These budgets now deliberately inherit the separately checked
                # 32-core anchor; the old 64-core fixture remains unmodified.
                inherited = base.for_launch(32,hidden*2,op,bf16=True)
                historical = inherited.native_plan()
                senders = inherited.send_workers
            assert plan.native_plan() == historical, (key, op, plan)
            assert plan.send_workers == senders, (key, op, plan)
        assert len(cases) == 60
        assert all(len(ops) == 5 and set(ops) == operations for ops in cases.values())
    """)


def test_metadata_starvation_adjustment_preserves_low_core_and_unweighted_plans():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy

        for ep, group, p, workers in ((8,2,2,2), (16,4,2,1)):
            base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                           ranks_per_group=group,topology_known=True)
            before = base.signature()
            for size in (2048,5632,12288,14336):
                for op in ('dispatch','weighted'):
                    plan = base.for_launch(16,size,op,bf16=True)
                    assert plan.native_plan()[:7] == [1,0,group,p,workers,1,32]
                    assert plan.send_workers == 8
                    assert plan.valid_group_capacity()
                    assert plan is base.for_launch(16,size,op,bf16=True)
                    if op == 'weighted':
                        assert plan.self_copy == 2
                        assert plan.weight_mode == 1
                    # A different API/dtype/expansion still uses the old baseline.
                    for options in ({'expanded':True}, {'v2':False}, {'bf16':False}):
                        old = base.for_launch(16,size,op,**{'bf16':True,**options})
                        assert old.send_workers == (14 if ep == 8 else 15)
                unweighted = base.for_launch(16,size,'combine',bf16=True)
                assert unweighted.send_workers == 8
                assert unweighted.self_copy == 1
            assert base.signature() == before
            assert base.config(16).num_aiv == 16

        # Actual four-node topologies: keep all AIV8 launch plans unchanged.
        for ep, group, order, p, w in ((4,1,0,1,2), (8,2,0,2,2), (16,4,0,2,1),
                                      (32,8,0,4,1), (64,16,0,8,1), (128,16,0,16,1)):
            base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                           ranks_per_group=group,topology_known=True)
            for op in ('dispatch','combine','weighted'):
                for size in (2048,5632,12288,14336):
                    plan = base.for_launch(8,size,op,bf16=True)
                    mode = (2 if ep == 4 and size <= 7168 else 1) if op == 'weighted' else 0
                    assert plan.native_plan() == [1,order,group,p,w,1,32,2,mode]

        # EP64 dispatch retains eight senders throughout the supported core range.
        base = replace(resolve_udma_policy(True,64,'udma',None,None,None,None,None),
                       ranks_per_group=16,topology_known=True)
        assert base.for_launch(16,5632,'dispatch',bf16=True).send_workers == 8
        assert base.for_launch(18,5632,'dispatch',bf16=True).send_workers == 8
        assert base.for_launch(20,5632,'dispatch',bf16=True).send_workers == 8
    """)


def test_all_large_core_launches_inherit_complete_32_core_plans():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        for ep in (4,8,16,32,64,128,256):
            for group in sorted({1,min(ep,8),min(ep,16)}):
                for known in (False,True):
                    base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                                   ranks_per_group=group,topology_known=known)
                    for op in ('dispatch','combine','weighted'):
                        for size in (32,2047,2048,4096,4097,5632,8191,8192,12288,12289,14336,32768,65536):
                            for opts in ({'bf16':True},{'bf16':False},{'bf16':True,'v2':False},
                                         {'bf16':True,'expanded':True}):
                                anchor = base.for_launch(32,size,op,**opts)
                                for cores in range(34,65,2):
                                    plan = base.for_launch(cores,size,op,**opts)
                                    assert plan.native_plan() == anchor.native_plan(), (ep,group,known,op,size,cores,opts)
                                    assert plan.send_workers == anchor.send_workers
                                    assert plan.send_workers <= cores and plan.send_workers*plan.qps_per_worker <= 32
                                    assert base.config(cores).num_aiv == cores
        # Preserve the measured visit intervals; only eliminate core-budget fallbacks.
        for ep,group in ((32,8),(32,16),(64,16)):
            base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                           ranks_per_group=group,topology_known=True)
            for hidden in (2816,4096,6144,7168,16384):
                expected = (4 if hidden<4096 else 8 if hidden<=6144 else 32) if (ep,group)==(32,8) else (
                    8 if ep==64 and 4096<=hidden<=6144 else 32)
                for cores in (32,48,64):
                    assert base.for_launch(cores,hidden*2,'weighted',bf16=True).wqes_per_visit == expected
    """)


def test_sender_tiers_inherit_complete_plans_without_changing_launch_budget():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        for ep in (4,8,16,32,64):
            for group in sorted({min(ep,8),min(ep,16)}):
                for known in (False,True):
                    base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                                   ranks_per_group=group,topology_known=known)
                    for op in (('dispatch','combine','weighted') if ep in (8,16,64) else ('dispatch',)):
                        for size in (1024,2048,4096,4097,5632,8192,12288,12289,14336):
                            for cores in range(8,65,2):
                                tier = 8 if cores < 32 else 32
                                anchor = base.for_launch(tier,size,op,bf16=True)
                                plan = base.for_launch(cores,size,op,bf16=True)
                                expected = anchor.native_plan()
                                if op == 'combine' and cores < 32:
                                    expected[7] = 1 if cores >= 2*anchor.send_workers else 2
                                assert plan.native_plan() == expected, (ep,group,known,op,size,cores)
                                assert plan.send_workers == anchor.send_workers
                                assert plan.send_workers <= cores
                                assert plan.send_workers*plan.qps_per_worker <= 32
                                assert plan.valid_group_capacity()
                                assert base.config(cores).num_aiv == cores
    """)


def test_ep4_wide_dispatch_retains_helpers_and_reserved_qps():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        base = replace(resolve_udma_policy(True,4,'udma',None,None,None,None,None),
                       ranks_per_group=4,topology_known=True,extend_large_rows=False)
        assert 'rank:1:1:2' in base.reservation_spec().split(';')
        for size in (12288,12289,14336,16384,24576,32768,65536):
            for cores in range(8,32,2):
                plan = base.for_launch(cores,size,'dispatch',bf16=True)
                assert plan.native_plan() == [1,0,4,1,1,2,32,2,0]
                assert plan.send_workers == 3 and cores-plan.send_workers >= 5
                assert plan.send_workers*plan.qps_per_worker == 6
                assert base.config(cores).num_aiv == cores
            for cores in (0,*range(32,65,2)):
                plan = base.for_launch(cores,size,'dispatch',bf16=True)
                assert plan.native_plan() == [1,0,4,1,2,1,32,2,0]
            # Changing dispatch must not change combine ownership or fallback
            # eligibility for other dtypes, API versions, expansion or topology.
            for cores in (8,16,30,32,64):
                for op in ('combine','weighted'):
                    plan = base.for_launch(cores,size,op,bf16=True)
                    assert (plan.peers_per_worker,plan.workers_per_peer,plan.qps_per_worker) == (1,2,1)
                for opts in ({'bf16':False},{'bf16':True,'v2':False},{'bf16':True,'expanded':True}):
                    plan = base.for_launch(cores,size,'dispatch',**opts)
                    assert (plan.peers_per_worker,plan.workers_per_peer,plan.qps_per_worker) == (1,2,1)
                for group, known in ((1,True),(2,True),(4,False)):
                    plan = replace(base,ranks_per_group=group,topology_known=known).for_launch(
                        cores,size,'dispatch',bf16=True)
                    assert (plan.peers_per_worker,plan.workers_per_peer,plan.qps_per_worker) == (1,2,1)
    """)


def test_ep32_unweighted_combine_preserves_eight_core_sender_plan():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        for group in (8,16):
            for known in (False,True):
                base = replace(resolve_udma_policy(True,32,'udma',None,None,None,None,None),
                               ranks_per_group=group,topology_known=known)
                for cores in (0,*range(8,65,2)):
                    for size in (1024,2048,4096,5632,8192,12288,14336):
                        plan = base.for_launch(cores,size,'combine',bf16=True)
                        reference = base.for_launch(8,size,'combine',bf16=True)
                        assert plan.native_plan()[:7] == reference.native_plan()[:7]
                        assert plan.native_plan()[:7] == [1,0,group,4,1,1,32]
                        assert plan.send_workers == 8
                        assert base.config(cores).num_aiv == cores
                        assert plan.self_copy == (1 if (cores or 32) >= 16 else 2)
    """)


def test_ep8_ep16_wide_rows_inherit_fixed_sender_tiers_by_default():
    run_host("""
        import os
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        os.environ.pop('DEEPEP_CLOS_EXTEND_LARGE_ROW_POLICY',None)
        expected = {
            (8,8,'dispatch'):(2,2,1,32), (8,32,'dispatch'):(1,1,1,16),
            (8,8,'combine'):(1,1,1,32), (8,32,'combine'):(1,1,1,32),
            (16,8,'dispatch'):(2,1,1,16), (16,32,'dispatch'):(4,2,1,16),
            (16,8,'combine'):(2,1,1,8), (16,32,'combine'):(2,1,1,8),
        }
        for ep in (8,16):
            base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                           ranks_per_group=ep,topology_known=True)
            assert not base.extend_large_rows
            reserved = set(base.reservation_spec().split(';'))
            for cores in range(8,65,2):
                tier = 8 if cores < 32 else 32
                for op in ('dispatch','combine','weighted'):
                    p,w,q,visit = expected[(ep,tier,'combine' if op=='weighted' else op)]
                    for size in (12288,12289,14336,16384,24576,32766,32768,65536):
                        plan = base.for_launch(cores,size,op,bf16=True)
                        assert (plan.peer_order,plan.peers_per_worker,plan.workers_per_peer,
                                plan.qps_per_worker,plan.wqes_per_visit) == (0,p,w,q,visit)
                        assert plan.native_plan() == base.for_launch(cores,12288,op,bf16=True).native_plan()
                        assert f'rank:{p}:{w}:{q}' in reserved
                        assert plan.send_workers <= cores and plan.send_workers*q <= 32
                        assert base.config(cores).num_aiv == cores
                        if op == 'weighted': assert plan.weight_mode == 1
                        for opts in ({'bf16':False},{'bf16':True,'expanded':True},{'bf16':True,'v2':False}):
                            fallback = base.for_launch(cores,size,op,**opts)
                            original = replace(base.for_num_aiv(cores),wqes_per_visit=32,self_copy=0,weight_mode=0)
                            assert fallback.native_plan() == original.native_plan()
    """)


def test_large_row_extension_defaults_and_compatibility_option_are_reserved():
    run_host("""
        import os
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        def resolve(ep):
            return replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                           ranks_per_group=min(ep,16), topology_known=True)
        os.environ.pop('DEEPEP_CLOS_EXTEND_LARGE_ROW_POLICY',None)
        old = resolve(8)
        assert not old.extend_large_rows
        old_plan = old.for_launch(32,14336,'combine',bf16=True).native_plan()
        assert old_plan[3:6] == [1,1,1]
        os.environ['DEEPEP_CLOS_EXTEND_LARGE_ROW_POLICY'] = '1'
        enabled = resolve(8)
        assert enabled.extend_large_rows and old.signature() != enabled.signature()
        # Configuration is immutable per Buffer; changing the environment cannot
        # leave cached launch plans inconsistent with its original QP reservation.
        assert old.for_launch(32,14336,'combine',bf16=True).native_plan() == old_plan
        names = ('rank','group','group_staggered','group_round_robin')
        for ep in (4,8,16,32,64,128):
            base = resolve(ep)
            disabled = replace(base,extend_large_rows=False)
            reserved = set(base.reservation_spec().split(';'))
            for cores in range(8,65,2):
                for op in ('dispatch','combine','weighted'):
                    for size in (2048,4096,5632,8192,12288,12289,14336,32768):
                        plan = base.for_launch(cores,size,op,bf16=True)
                        reference = disabled.for_launch(cores,12288 if ep in (4,8,16) and size>12288 else size,
                                                       op,bf16=True)
                        assert plan.native_plan() == reference.native_plan(), (ep,cores,op,size)
                        key = f'{names[plan.peer_order]}:{plan.peers_per_worker}:{plan.workers_per_peer}:{plan.qps_per_worker}'
                        assert key in reserved and plan.send_workers <= cores
                        assert plan.send_workers*plan.qps_per_worker <= 32
                    for opts in ({'bf16':False},{'bf16':True,'v2':False},{'bf16':True,'expanded':True}):
                        assert base.for_launch(cores,14336,op,**opts).native_plan() == disabled.for_launch(
                            cores,14336,op,**opts).native_plan()
        os.environ['DEEPEP_CLOS_EXTEND_LARGE_ROW_POLICY'] = '0'
        assert not resolve(8).extend_large_rows
        assert enabled.for_launch(32,14336,'combine',bf16=True).native_plan()[3:6] == [1,1,1]
        for bad in ('true','2',''):
            os.environ['DEEPEP_CLOS_EXTEND_LARGE_ROW_POLICY'] = bad
            try: resolve(8)
            except ValueError: pass
            else: raise AssertionError('invalid large-row option accepted')
    """)


def test_ep32_dispatch_profile_survives_extra_cores():
    run_host("""
        from dataclasses import replace, asdict
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        sys.path.insert(0, 'benchmarks/clos')
        from bench_v2_bandwidth import effective_udma_config
        base = replace(resolve_udma_policy(True,32,'udma',None,None,None,None,None),
                       ranks_per_group=8,topology_known=True)
        buffer = SimpleNamespace(udma_auto_policy=base, udma_schedule='peer',
            udma_dispatch_peer_order='rank',udma_combine_peer_order='rank',
            udma_ranks_per_group=8,udma_peers_per_group_per_worker=None,
            udma_workers_per_peer=1,udma_qps_per_worker=1)
        for cores in (32,34,48,64):
            assert base.config(cores).num_aiv == cores
            for size in (5632,8192,12288):
                plan = base.for_launch(cores,size,'dispatch',bf16=True)
                assert plan.native_plan() == [1,0,8,4,1,1,32,2,0]
                report = effective_udma_config(buffer,num_aiv=cores,row_bytes=size,bf16=True)
                assert report['operation_plans']['dispatch'] == asdict(plan)
            for size in (1024,2048,4096,14336):
                assert base.for_launch(cores,size,'dispatch',bf16=True).peer_order == 0
            for other in (replace(base,topology_known=False),replace(base,ranks_per_group=16)):
                assert other.for_launch(cores,5632,'dispatch',bf16=True).peer_order == 0
            for options in ({'expanded':True}, {'v2':False}, {'bf16':False}):
                assert base.for_launch(cores,5632,'dispatch',**{'bf16':True,**options}).peer_order == 0
                for op in ('combine','weighted'):
                    assert base.for_launch(cores,12288,op,bf16=True).peer_order == 0
    """)


def test_combine_copy_uses_sender_and_spare_aivs_instead_of_fixed_core_count():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        base = replace(resolve_udma_policy(True,4,'udma',None,None,None,None,None),
                       ranks_per_group=4, topology_known=True)
        for cores in (20,32,48,64):
            plan = base.for_launch(cores,5632,'combine',bf16=True)
            assert plan.send_workers == 6 and plan.self_copy == 1
        assert base.for_launch(10,5632,'combine',bf16=True).self_copy == 2
        assert base.for_launch(14,5632,'combine',bf16=True).self_copy == 2
        assert base.for_launch(16,5632,'combine',bf16=True).self_copy == 1
        for ep in (4,8,16,32,64,128):
            policy = replace(base, world_size=ep, ranks_per_group=min(ep,16))
            for cores in (8,20,32,48,64):
                for row_bytes in (2048,5632,12288):
                    plan = policy.for_launch(cores,row_bytes,'combine',bf16=True)
                    expected = 1 if min(cores,32)-plan.send_workers >= plan.send_workers else 2
                    if ep == 128 and cores >= 16:
                        expected = 2  # Fixed rank/P8/visit16, but never copy before sending.
                    assert plan.self_copy == expected
        # Two QPs still belong to one payload sender AIV.
        q2 = replace(base, workers_per_peer=1, qps_per_worker=2, topology_known=False)
        plan = q2.for_launch(12,5632,'combine',bf16=True)
        assert plan.send_workers == 3 and plan.self_copy == 1
        # EP128 keeps the complete fixed candidate with larger core budgets.
        large = replace(base, world_size=128, ranks_per_group=16)
        plan = large.for_launch(48,5632,'combine',bf16=True)
        assert plan.send_workers == 16 and plan.self_copy == 2
    """)


def test_ep128_defaults_preserve_fixed_candidate_across_shapes_and_budgets():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        base = replace(resolve_udma_policy(True,128,'udma',None,None,None,None,None),
                       ranks_per_group=16, topology_known=True)
        before = base.signature()
        native = [1,0,16,8,1,1,16,2,0]
        assert (base.peers_per_worker, base.wqes_per_visit) == (8,16)
        for op in ('dispatch','combine','weighted'):
            for cores in (0,16,20,30,32,34,48,64):
                for size in (2048,5600,5632,5664,7168,8192,12288,14336):
                    plan = base.for_launch(cores,size,op,bf16=True)
                    assert plan.native_plan() == native
                    assert plan.send_workers == 16 and plan.valid_group_capacity()
                    assert plan is base.for_launch(cores,size,op,bf16=True)
            # Unknown locality does not alter rank order; the low-core path
            # still chooses a legal fallback instead of exceeding its budget.
            assert replace(base,topology_known=False).for_launch(32,5632,op,bf16=True).native_plan() == native
            small = base.for_launch(8,5632,op,bf16=True)
            assert small.send_workers <= 8 and small.valid_group_capacity()
            for other in (replace(base,world_size=64), replace(base,world_size=256)):
                assert other.for_launch(32,5632,op,bf16=True).native_plan() != native
            for options in ({'bf16':False}, {'expanded':True}, {'v2':False}):
                assert base.for_launch(32,5632,op,**{'bf16':True,**options}).native_plan() == native
        assert base.signature() == before
    """)


def test_ep128_low_core_rank_p16_preserves_larger_core_plans():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.config import Config
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        assert Config(8,udma_peers_per_worker=16).udma_peers_per_worker == 16
        for group in (1,2,4,8,16,32,64,128):
            for known in (False,True):
                base = replace(resolve_udma_policy(True,128,'udma',None,None,None,None,None),
                               ranks_per_group=group,topology_known=known)
                assert base.config(8).udma_peers_per_worker == 16
                for op in ('dispatch','combine','weighted'):
                    for size in (2048,5632,8192,12288,14336):
                        for opts in ({'bf16':True},{'bf16':False},{'bf16':True,'expanded':True},{'bf16':True,'v2':False}):
                            for cores in (8,10,12,14):
                                plan = base.for_launch(cores,size,op,**opts)
                                mode = (2 if cores > 8 else 1) if op == 'weighted' and opts == {'bf16':True} else 0
                                assert plan.native_plan() == [1,0,group,16,1,1,32,2,mode]
                                assert plan.send_workers == 8
                                assert plan.valid_group_capacity()
        base = replace(base,ranks_per_group=16,topology_known=True)
        for cores in (10,12,14):
            assert base.for_launch(cores,5632,'dispatch',bf16=True).native_plan() == [1,0,16,16,1,1,32,2,0]
        for cores in (0,*range(16,65,2)):
            for op in ('dispatch','combine','weighted'):
                assert base.for_launch(cores,5632,op,bf16=True).native_plan() == [1,0,16,8,1,1,16,2,0]
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


def test_ep64_dispatch_32_cores_uses_eight_senders_without_changing_other_paths():
    run_host("""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        base = resolve_udma_policy(True,64,'udma',None,None,None,None,None)
        for known in (False,True):
            for group in (8,16,32,64):
                policy = replace(base, ranks_per_group=group, topology_known=known)
                for size in (2048,5632,8192,12288,14336,16384):
                    for cores in (0,32):
                        plan = policy.for_launch(cores,size,'dispatch',bf16=True)
                        assert plan.native_plan()[1:7] == [0,group,8,1,1,32]
                        assert plan.send_workers == 8 and (cores or 32)-plan.send_workers == 24
                    for cores in (8,16,30,48,64):
                        plan = policy.for_launch(cores,size,'dispatch',bf16=True)
                        assert plan.peers_per_worker == 8
                    for kwargs in ({'bf16':False}, {'bf16':True,'expanded':True},
                                   {'bf16':True,'v2':False}):
                        assert policy.for_launch(32,size,'dispatch',**kwargs).peers_per_worker == 4
                    assert policy.for_launch(32,size,'combine',bf16=True).send_workers == 16
                    assert policy.for_launch(32,size,'weighted',bf16=True).send_workers == 16
    """)


@pytest.mark.parametrize("ep", [4, 8, 16, 32, 64, 128, 256])
def test_auto_qp_reservation_covers_dynamic_launches_and_legacy_paths(ep):
    run_host(f"""
        from dataclasses import replace
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        base = resolve_udma_policy(True,{ep},'udma',None,None,None,None,None)
        names = ('rank','group','group_staggered','group_round_robin')
        sizes = (32,2047,2048,2049,4095,4096,4097,5632,7168,8191,8192,8193,12287,12288,12289,14336,32768)
        for group in (1,2,4,8,16,32,64,128,256):
            if group > {ep}: continue
            for known in (False,True):
                policy = replace(base,ranks_per_group=group,topology_known=known)
                reserved = set(policy.reservation_spec().split(';'))
                assert policy.signature()[-1] == policy.reservation_spec()
                for cores in range(0,65,2):
                    try: policy.for_num_aiv(cores)
                    except ValueError: continue
                    for size in sizes:
                        for op in ('dispatch','combine','weighted'):
                            for kwargs in ({{'bf16':True}}, {{'bf16':False}},
                                           {{'bf16':True,'expanded':True}}, {{'bf16':True,'v2':False}}):
                                p = policy.for_launch(cores,size,op,**kwargs)
                                key = f'{{names[p.peer_order]}}:{{p.peers_per_worker}}:{{p.workers_per_peer}}:{{p.qps_per_worker}}'
                                assert key in reserved, (group,known,cores,size,op,kwargs,key,reserved)
        if {ep} in (32,64,128):
            policy = replace(base,ranks_per_group=16,topology_known=True)
            expected = {{32:'rank:4:1:1;rank:8:1:1',
                        64:'rank:4:1:1;rank:8:1:1;group_round_robin:4:1:1',
                        128:'rank:8:1:1;rank:16:1:1;group_round_robin:4:1:1'}}
            assert policy.reservation_spec() == expected[{ep}]
    """)


def test_auto_qp_masks_preserve_control_symmetry_and_reduce_tuning_union(tmp_path):
    import json
    import shutil

    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for automatic QP mask coverage")
    generated = subprocess.run(
        [
            sys.executable,
            "-c",
            DTYPES_ONLY
            + textwrap.dedent("""
            import json
            from dataclasses import replace
            from deep_ep.backends.direct.udma_policy import resolve_udma_policy
            cases = []
            for ep in (4,8,16,32,64,128):
                group = min(ep,16)
                policy = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                                 ranks_per_group=group,topology_known=True)
                cases.append((ep,group,policy.reservation_spec()))
            print(json.dumps(cases))
        """),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    source = ['#include "runtime/udma_qp_reservation.h"', "using namespace ascend_deepep::udma;"]
    names = ("rank", "group", "group_staggered", "group_round_robin")
    for ep, group, spec in json.loads(generated.stdout):
        expressions = []
        for entry in spec.split(";"):
            order, p, w, q = entry.split(":")
            expressions.append(f"QpReservation::Mask(r,p,{ep},{group},{names.index(order)},{p},{int(w) * int(q)})")
        source.append(f"constexpr uint64_t mask{ep}(unsigned r,unsigned p) {{return " + "|".join(expressions) + ";}")
        source.append(f"""
constexpr bool check{ep}() {{
    unsigned before=0,after=0;
    for(unsigned p=0;p<{ep};++p) {{
        auto m=mask{ep}(0,p), old=TuningPeerMask(0,p,{ep},{group});
        if(m!=mask{ep}(p,0) || (p==0 ? m!=0 : !(m&1)) || (m&~old)) return false;
        for(unsigned b=0;b<32;++b) {{before+=(old>>b)&1; after+=(m>>b)&1;}}
    }}
    return after < before;
}}
static_assert(check{ep}(), "automatic mask must retain control and symmetry while reducing reservations");
""")
    path = tmp_path / "auto_qp_masks.cpp"
    path.write_text("\n".join(source))
    budget = (
        "-fconstexpr-ops-limit=100000000" if Path(compiler).name.startswith("g++") else "-fconstexpr-steps=100000000"
    )
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            budget,
            "-I" + str(ROOT / "csrc/direct"),
            "-c",
            str(path),
            "-o",
            str(tmp_path / "auto_qp_masks.o"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
                    assert base.for_launch(8,5632,'combine',bf16=True).native_plan()[:7] == [1,0,group,16,1,1,32]
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
        assert plans['dispatch']['qps_per_worker'] == 1
        assert plans['dispatch']['workers_per_peer'] == 2
        assert plans['weighted']['qps_per_worker'] == 1
        assert plans['weighted']['weight_mode'] == 1
        assert plans['combine']['self_copy'] == 2
        for early, chunk in ((False,0),(True,0),(False,32)):
            b._use_early_single_source_combine = early
            b._aggregate_pipeline_chunk_tokens = chunk
            report = effective_udma_config(b,num_aiv=8,row_bytes=5632,bf16=True,expanded=True)
            from dataclasses import asdict
            for op in ('dispatch','combine','weighted'):
                expected = base.for_launch(8,5632,op,bf16=True,
                    expanded=op == 'dispatch' or early or bool(chunk))
                assert report['operation_plans'][op] == asdict(expected)
    """)


@pytest.mark.parametrize("expand", [0, 1])
def test_benchmark_worker_reports_the_expand_mode_passed_to_dispatch(expand):
    run_host(
        f"expand = {expand}\n"
        + textwrap.dedent("""
        import os
        from dataclasses import asdict, replace
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import Mock
        from deep_ep.backends import global_clos
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        sys.path.insert(0, 'benchmarks/clos')
        import bench_v2_bandwidth as bench

        class DispatchReached(Exception):
            pass

        policy = replace(resolve_udma_policy(True,32,'udma',None,None,None,None,None),
                         ranks_per_group=8, topology_known=True)
        buffer = Mock(udma_auto_policy=policy)
        buffer._use_early_single_source_combine = False
        buffer._aggregate_pipeline_chunk_tokens = 0
        for name in ('schedule', 'dispatch_peer_order', 'combine_peer_order', 'ranks_per_group',
                     'peers_per_group_per_worker', 'workers_per_peer', 'qps_per_worker'):
            setattr(buffer, 'udma_' + name, None)
        buffer.dispatch.side_effect = DispatchReached
        global_clos.ElasticBuffer = Mock(return_value=buffer)
        dist = Mock(group=SimpleNamespace(WORLD=object()))
        dist.all_gather_object.side_effect = lambda items, value: items.__setitem__(slice(None), [value]*len(items))
        sys.modules.update({'torch.distributed': dist, 'torch_npu': Mock(), 'common': Mock(), 'profiling': Mock()})
        torch.distributed, torch.npu, torch.set_num_threads = dist, Mock(), Mock()
        bench.route_counts = bench.route_statistics = Mock(return_value={})
        bench.loaded_identity = bench.library_identity = Mock(return_value={})
        bench.byte_ledger = Mock(return_value={'official_logical_bytes_per_row': 0})
        tensor = Mock(**{'numel.return_value': 0, 'element_size.return_value': 2})
        tensor.to.return_value = tensor
        tensor.tolist.return_value = []
        golden = sys.modules['common']
        golden._make_topk_idx.return_value = tensor
        golden._make_weights.return_value = tensor
        golden._to_npu.return_value = tensor
        os.environ.update(RANK='0', WORLD_SIZE='32', LOCAL_RANK='0', LOCAL_WORLD_SIZE='8')
        with TemporaryDirectory() as output:
            args = bench.arguments(['--ep-size','32','--nnodes','4','--node-rank','0',
                                    '--master-addr','192.0.2.1','--run-id','expand-test','--hidden','2816','--aiv','32',
                                    '--dtypes','bf16','--expand',str(expand),'--output',str(Path(output))])
            try:
                bench.worker(args)
            except DispatchReached:
                actual = buffer.dispatch.call_args.kwargs['do_expand']
                assert actual is bool(expand)
                assert args.effective_udma_config['operation_plans'] == {
                    op: asdict(policy.for_launch(32,5632,op,bf16=True,expanded=actual and op == 'dispatch'))
                    for op in ('dispatch','combine','weighted')
                }
            else:
                raise AssertionError('worker never called dispatch')
    """)
    )


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
              for (unsigned g : {1U,2U,4U,8U,16U}) {
                if (g > ep) continue;
                for (unsigned rank=0; rank<ep; ++rank)
                  for (unsigned order=0; order<=3; ++order)
                    for (unsigned p : {1U,2U,4U,8U,16U}) {
                      if (p==16 && order!=0) continue;
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
    try:
        subprocess.run([str(binary)], check=True, timeout=60)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked execution (4551)")
        raise


@pytest.mark.parametrize("legacy_guard", [False, True])
def test_python_plans_pass_native_launch_validation(tmp_path, legacy_guard):
    """Evaluate the native guards against real Python plans, without CANN.

    Fixed arrays replace vector input and rejected plans return a sentinel so
    every check executes at compile time, including on hosts that block test EXEs.
    The negative control reinstates the old dispatch guard and must fail.
    """
    import json
    import re
    import shutil

    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for native launch validation")
    generated = subprocess.run(
        [
            sys.executable,
            "-c",
            DTYPES_ONLY
            + textwrap.dedent("""
                import csv, json
                from dataclasses import replace
                from deep_ep.backends.direct.udma_policy import resolve_udma_policy
                cases = []
                with open('tests/api/fixtures/udma_plans_20261116.csv') as f:
                    for row in csv.DictReader(f):
                        ep, group, cores, hidden = (int(row[k]) for k in ('EP','G','aiv','hidden'))
                        op = {'combine_weighted':'weighted','combine_unweighted':'combine'}.get(row['operator'],'dispatch')
                        base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                                       ranks_per_group=group,topology_known=True)
                        cases.append([ep,group,cores,op!='dispatch',base.for_launch(cores,hidden*2,op,bf16=True).native_plan()])
                assert len(cases) == 300
                for ep in (4,8,16,32,64,128,256):
                    group = min(ep,8)
                    base = replace(resolve_udma_policy(True,ep,'udma',None,None,None,None,None),
                                   ranks_per_group=group,topology_known=True)
                    for cores in (8,16,32,48,64):
                        for op in ('dispatch','combine','weighted'):
                            plan = base.for_launch(cores,5632,op,bf16=True)
                            cases.append([ep,group,cores,op!='dispatch',plan.native_plan()])
                print(json.dumps(cases))
            """),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert generated.returncode == 0, generated.stderr
    cases = json.loads(generated.stdout)
    native = (ROOT / "csrc/direct/runtime/shmem_instance.cpp").read_text()
    # The shared extractor handles free functions; drop only the member qualifier.
    body = cpp_functions(native.replace("int64_t cores) const", "int64_t cores)"), "ResolveLaunch")
    body = body.replace(
        "ShmemInstance::LaunchPlan ShmemInstance::ResolveLaunch(const std::vector<int64_t>& values",
        "template<size_t N> constexpr LaunchPlan ResolveLaunch(const std::array<int64_t,N>& values",
    )
    body = re.sub(r'throw std::invalid_argument\("[^"\n]*"\);', "return {UINT32_MAX,0,0,0,0,0,0,0};", body)
    if legacy_guard:
        fixed = "(!combine && ((copy != 0 && copy != 2) || weight))"
        assert fixed in body
        body = body.replace(fixed, "(!combine && (copy || weight))")
    header = (ROOT / "csrc/direct/runtime/shmem_instance.h").read_text()
    begin = header.index("struct LaunchPlan {")
    plan_type = header[begin : header.index("};", begin) + 2]
    source = r"""
#include <array>
#include <cstddef>
#include "common/udma_schedule.h"
struct ShmemInstance {
    int64_t world_size_;
    uint32_t ranks_per_group_;
    bool auto_resources_=true;
    uint32_t dispatch_peer_order_=0, combine_peer_order_=0, peers_per_group_per_worker_=1;
    uint32_t workers_per_peer_=1, qps_per_worker_=1;
    // Reservation construction is tested separately. Exercise both outcomes
    // here while keeping the extracted native launch guards constexpr.
    struct Reservation {
        bool allowed=true;
        constexpr bool Allows(uint32_t,uint32_t,uint32_t,uint32_t) const { return allowed; }
    } qp_reservation_;
    // PLAN_TYPE
    // BODY
};
constexpr bool matches(unsigned ep,unsigned group,int cores,bool combine,std::array<int64_t,9> v) {
    ShmemInstance instance{ep,group};
    const auto r=instance.ResolveLaunch(v,combine,1,32,cores);
    return r.order==v[1] && r.group==v[2] && r.peers==v[3] && r.workers==v[4] &&
           r.qps==v[5] && r.visit==v[6] && r.self_copy==v[7] && r.weight_mode==v[8];
}
constexpr bool reject(int index,int64_t value,int cores=8) {
    ShmemInstance instance{8,8};
    std::array<int64_t,9> v{1,0,8,1,1,1,32,2,0};
    v[index]=value;
    return instance.ResolveLaunch(v,false,1,32,cores).order==UINT32_MAX;
}
static_assert(matches(8,8,8,false,{1,0,8,1,1,1,32,0,0})); // Legacy marker stays legal.
static_assert(reject(0,2) && reject(1,4) && reject(2,4) && reject(3,3));
static_assert(reject(4,3) && reject(5,3) && reject(6,3) && reject(6,256));
static_assert(reject(6,96) && reject(6,129) && reject(6,(int64_t{1}<<32)+64));
static_assert(matches(64,16,8,false,{1,0,16,8,1,1,64,2,0}));
static_assert(matches(64,16,8,true,{1,0,16,8,1,1,128,2,1}));
static_assert(matches(128,16,8,false,{1,0,16,16,1,1,32,2,0}));
static_assert(matches(128,16,8,true,{1,0,16,16,1,1,32,2,1}));
static_assert(reject(7,-1) && reject(7,1) && reject(7,3));
static_assert(reject(8,-1) && reject(8,1) && reject(8,2) && reject(8,3));
static_assert(reject(7,2,-2) && reject(7,2,7) && reject(7,2,6));
static_assert(reject(4,2)); // Too many sending AIVs.
constexpr bool resources() {
    ShmemInstance instance{8,8};
    const std::array<int64_t,9> good{1,0,8,1,1,1,32,2,0};
    if(instance.ResolveLaunch(std::array<int64_t,8>{},false,1,32,8).order!=UINT32_MAX)return false;
    instance.auto_resources_=false;
    if(instance.ResolveLaunch(good,false,1,32,8).order!=UINT32_MAX)return false;
    const auto legacy=instance.ResolveLaunch(std::array<int64_t,0>{},false,2,16,8);
    if(legacy.self_copy!=-1 || legacy.peers!=2 || legacy.visit!=16)return false;
    instance={64,16};
    if(instance.ResolveLaunch(std::array<int64_t,9>{1,0,16,1,1,1,32,2,0},false,1,32,64).order!=UINT32_MAX)
        return false; // Physical QP capacity still applies.
    if(instance.ResolveLaunch(std::array<int64_t,9>{1,3,16,8,1,1,32,2,0},false,1,32,64).order!=UINT32_MAX)
        return false; // RR peer capacity still applies.
    return true;
}
static_assert(resources());
constexpr bool unreserved() {
    ShmemInstance instance{8,8};
    instance.qp_reservation_.allowed=false;
    const std::array<int64_t,9> plan{1,0,8,1,1,1,32,2,0};
    return instance.ResolveLaunch(plan,false,1,32,8).order==UINT32_MAX &&
           instance.ResolveLaunch(plan,true,1,32,8).order==UINT32_MAX &&
           instance.ResolveLaunch(std::array<int64_t,0>{},false,2,16,8).order==UINT32_MAX &&
           instance.ResolveLaunch(std::array<int64_t,0>{},true,2,16,8).order==UINT32_MAX;
}
static_assert(unreserved());
// CASES
"""
    checks = "\n".join(
        f"static_assert(matches({ep},{group},{cores},{str(combine).lower()},{{" + ",".join(map(str, plan)) + "}));"
        for ep, group, cores, combine, plan in cases
    )
    path = tmp_path / "native_launch.cpp"
    path.write_text(source.replace("// PLAN_TYPE", plan_type).replace("// BODY", body).replace("// CASES", checks))
    built = subprocess.run(
        [compiler, "-std=c++17", "-I", str(ROOT / "csrc/direct"), "-c", str(path), "-o", str(tmp_path / "launch.o")],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if legacy_guard:
        assert built.returncode != 0 and "static assertion" in built.stderr, built.stderr
    else:
        assert built.returncode == 0, built.stderr
