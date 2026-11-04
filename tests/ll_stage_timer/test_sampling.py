"""CPU event-order tests for stabilized diagnostics and deferred record collection."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('after_perf', [False, True])
@pytest.mark.parametrize('operation', ['dispatch', 'combine'])
@pytest.mark.parametrize('backend', ['hccl', 'gloo'])
def test_tail_has_prelude_and_defers_all_checks(monkeypatch, tmp_path, after_perf, operation, backend):
    from tests.elastic import test_ep_low_latency as st
    from deep_ep.utils import ll_stage_timer as timer
    from deep_ep import _native
    events, saved, invoked = [], [], []
    class Tensor:
        def fill_(self, value): events.append('fill')
    dist = SimpleNamespace(get_world_size=lambda: 16, ReduceOp=SimpleNamespace(SUM=0),
                           all_reduce=lambda *a, **k: events.append('reduce'))
    torch = SimpleNamespace(float32=0, zeros=lambda *a, **k: Tensor(), ones=lambda *a, **k: Tensor(),
                            distributed=dist,
                            npu=SimpleNamespace(synchronize=lambda: events.append('sync'),
                                                _sleep=lambda cycles: events.append('sleep')))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.distributed', dist)
    monkeypatch.setattr(_native, 'require_native', lambda _: SimpleNamespace(low_latency_ranks_per_group=lambda world: 4))
    monkeypatch.setattr(st, 'control_device', lambda *a: 'npu')
    monkeypatch.setattr(st, 'completion_boundary', lambda *a: events.append('boundary'))
    monkeypatch.setattr(st, 'agree_check', lambda fn, *a: fn())
    monkeypatch.setattr(st, 'case_config', lambda *a: {'B': 32})
    class Trace:
        def __init__(self, *a): events.append('allocate')
        def complete(self): events.append('copy')
        def save(self, path, meta): events.append('save'); saved.append(meta)
    monkeypatch.setattr(timer, 'StageTrace', Trace)
    @contextmanager
    def capture(*a, prepared, defer_complete, **kw):
        assert defer_complete
        events.append('capture')
        yield prepared
    monkeypatch.setattr(timer, 'capture', capture)
    def invoke():
        events.append('invoke')
        result=(len(invoked), SimpleNamespace(current_stream_wait=lambda: events.append('wait')))
        invoked.append(result)
        return result
    def check(output):
        assert output in invoked[-2:]
        events.append('check')
    args=SimpleNamespace(run_directory=str(tmp_path), case_id=0, ll_timer_iters=2,
                         ll_timer_mode='observe', ll_timer_frequency_hz=1e9, seed=0, num_tokens=8, num_experts=384,
                         warmup=2, num_iters=3, device_busy_cycles=20_000_000, group_backend=backend)
    st.diagnostic_calls(object(), invoke, check, args, operation, 0, 256, 'normal',
                        after_perf=after_perf)
    prefix=['allocate']*2
    if not after_perf:
        prefix += ['boundary', 'reduce'] + ['invoke']*2 + ['sync']
        prefix += ['sleep'] + ['invoke']*3
    else:
        prefix += ['sleep']
    assert events == prefix + ['capture', 'invoke', 'wait']*2 + ['boundary'] + ['copy', 'check', 'save']*2
    assert [item['iteration'] for item in saved]==[0, 1]
    assert all(item['sampling']['after_perf']==after_perf for item in saved)
    assert events.count('reduce') == (0 if after_perf else 1)
    assert all(item['sampling']['sampling_protocol'] == 'initial-reduce-sleep-batch-v4' for item in saved)
    assert all(item['sampling']['device_busy'] == 'npu_sleep' and item['sampling']['device_busy_cycles'] == 20_000_000
               and not item['sampling']['per_iteration_all_reduce'] for item in saved)


def test_deferred_context_preserves_records_until_complete(monkeypatch):
    spec=importlib.util.spec_from_file_location('deferred_timer', ROOT/'deep_ep/utils/ll_stage_timer.py')
    timer=importlib.util.module_from_spec(spec); spec.loader.exec_module(timer)
    copied=[]
    buffer=object()
    trace=SimpleNamespace(buffer=buffer, operation='dispatch', mode='observe', frequency_hz=1e9,
                          launches=0, tensor=SimpleNamespace(device='npu:0'), complete=lambda: copied.append(True))
    with timer.capture(buffer, 'dispatch', prepared=trace, defer_complete=True):
        timer.attach(buffer, {'input': SimpleNamespace(device='npu:0')}, 'dispatch')
    assert copied==[] and not timer._state.active
    trace.complete()
    assert copied==[True]
    with pytest.raises(ValueError, match='unused'):
        with timer.capture(buffer, 'dispatch', prepared=trace, defer_complete=True): pass


@pytest.mark.parametrize('skip_perf', [False, True])
@pytest.mark.parametrize('stage_timer', [False, True])
def test_sample_operation_selects_one_schedule(monkeypatch, tmp_path, skip_perf, stage_timer):
    from tests.elastic import test_ep_low_latency as st
    from deep_ep.utils import ll_stage_timer as timer
    events=[]
    monkeypatch.setattr(timer, 'StageTrace', lambda *a: events.append('allocate') or object())
    monkeypatch.setattr(st, 'agree_check', lambda fn, *a: fn())
    def diagnostic(*a, prepared, after_perf):
        assert len(prepared)==2
        assert after_perf==(not skip_perf)
        events.append('diagnostic')
    def measure(*a, after_profile):
        events.append('perf100')
        if after_profile is not None:
            after_profile()
        return {'samples': 100}
    monkeypatch.setattr(st, 'diagnostic_calls', diagnostic)
    monkeypatch.setattr(st, 'measure', measure)
    args=SimpleNamespace(ll_stage_timer=stage_timer, ll_timer_iters=2, ll_timer_mode='drain',
                         ll_timer_frequency_hz=1e9, skip_perf_test=skip_perf,
                         run_directory=str(tmp_path), case_id=0)
    result=st.sample_operation(object(), None, None, args, 'dispatch', 0, 256, 'normal', lambda *a: None)
    assert events==(['allocate', 'allocate'] if stage_timer else []) + ([] if skip_perf else ['perf100']) + (['diagnostic'] if stage_timer else [])
    assert (result is None)==skip_perf
