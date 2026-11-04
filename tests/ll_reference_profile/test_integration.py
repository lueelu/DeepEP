"""CPU contracts for the independent reference build, capture and ST schedule."""

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_ep import _native
from deep_ep.utils import ll_reference_profile as reference
from deep_ep.utils import ll_stage_timer as legacy
from tests.elastic import test_ep_low_latency as st


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load('reference_one_click', ROOT / 'tests/ll_rdma/one_click.py')


class CaptureTests(unittest.TestCase):
    def setUp(self):
        reference._state.active = {}
        legacy._state.active = {}

    def test_allocates_actual_cores_aligned_and_rejects_wrong_native_layout(self):
        created = []
        class Tensor:
            device = 'npu:3'
            def __init__(self, address=520): self.address = address
            def data_ptr(self): return self.address
            def narrow(self, axis, offset, length):
                created.append(('narrow', axis, offset, length))
                return Tensor(self.address + offset * 8)
            def view(self, *shape):
                created.append(('view', shape))
                return self
        torch = SimpleNamespace(int64='int64', zeros=lambda words, **kw: created.append((words, kw)) or Tensor())
        layout = dict(schema=4, slots=80, fields=8, words=list(reference.WORDS))
        native = SimpleNamespace(low_latency_reference_profile_enabled=True, low_latency_stage_timer_enabled=False,
                                 low_latency_reference_profile_layout=lambda: layout,
                                 low_latency_core_config=lambda: dict(active_aiv=56, device=3, runtime_device_queried=True))
        with patch.dict(sys.modules, {'torch': torch}), patch.object(_native, 'require_native', return_value=native):
            trace = reference.ReferenceTrace(object())
            self.assertEqual(trace.tensor.data_ptr() % 512, 0)
            self.assertEqual(created[-1], ('view', (56, 80, 8)))
            self.assertEqual(created[0][0], 56 * 640 + 63)
            layout['schema'] = True
            with self.assertRaisesRegex(RuntimeError, 'layout mismatch'):
                reference.ReferenceTrace(object())
            layout['schema'] = 4
            native.low_latency_reference_profile_enabled = 1
            with self.assertRaisesRegex(RuntimeError, 'Rebuild'):
                reference.ReferenceTrace(object())
            native.low_latency_reference_profile_enabled = True
            native.low_latency_stage_timer_enabled = True
            with self.assertRaisesRegex(RuntimeError, 'Rebuild'):
                reference.ReferenceTrace(object())

    def prepared(self, buffer):
        trace = reference.ReferenceTrace.__new__(reference.ReferenceTrace)
        trace.buffer, trace.operation, trace.mode = buffer, 'dispatch', 'reference'
        trace.frequency_hz, trace.launches = 1e9, 0
        trace.tensor = SimpleNamespace(device='npu:0')
        trace.raw, trace.report = None, None
        return trace

    def test_capture_attaches_only_reference_and_defers_copy(self):
        buffer = object()
        trace = self.prepared(buffer)
        tensors = {'input': SimpleNamespace(device='npu:0')}
        with reference.capture(buffer, prepared=trace, defer_complete=True):
            reference.attach(buffer, tensors, 'dispatch')
            self.assertIs(tensors['ll_reference_profile'], trace.tensor)
            self.assertNotIn('ll_stage_timer', tensors)
            self.assertNotIn('ll_stage_timer_mode', tensors)
            with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                reference.attach(buffer, tensors, 'dispatch')
            with self.assertRaisesRegex(RuntimeError, 'Nested'):
                with legacy.capture(buffer, 'dispatch'): pass
        self.assertIsNone(trace.raw)
        self.assertEqual(reference._state.active, {})
        with self.assertRaisesRegex(ValueError, 'unused'):
            with reference.capture(buffer, prepared=trace, defer_complete=True): pass

    def test_wrong_operation_or_other_diagnostic_never_attaches(self):
        for operation, extra in (('combine', {}), ('dispatch', {'ll_stage_timer': object()}),
                                 ('dispatch', {'ll_expert_timer': object()})):
            with self.subTest(operation=operation, extra=list(extra)):
                buffer = object()
                trace = self.prepared(buffer)
                reference._state.active[id(buffer)] = trace
                tensors = dict(input=SimpleNamespace(device='npu:0'), **extra)
                with self.assertRaises(RuntimeError): reference.attach(buffer, tensors, operation)
                self.assertNotIn('ll_reference_profile', tensors)
                self.assertEqual(trace.launches, 0)
                reference._state.active.clear()

    def test_save_preserves_format_raw_and_metadata_without_overwrite(self):
        trace = self.prepared(object())
        trace.raw = [[[2**63 + 17]]]
        trace.report = dict(schema=1, records=[])
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as directory:
            path = Path(directory) / 'sample_0000.json'
            trace.save(path, dict(rank=17, profile_seed=0))
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['format'], 'll-reference-cycles-v1')
            self.assertEqual(saved['raw'], trace.raw)
            self.assertEqual(saved['metadata']['rank'], 17)
            with self.assertRaises(FileExistsError): trace.save(path)


class LauncherTests(unittest.TestCase):
    def test_reference_default_and_explicit_capture_count(self):
        args = st.parse_args(['--ll-reference-profile', '--dispatch-only'])
        self.assertEqual(args.ll_timer_iters, 1)
        self.assertEqual(st.validate_args(args, 32), args.num_tokens)
        explicit = st.parse_args(['--ll-reference-profile', '--dispatch-only', '--ll-timer-iters', '3'])
        self.assertEqual(explicit.ll_timer_iters, 3)
        self.assertEqual(st.parse_args(['--ll-stage-timer']).ll_timer_iters, 5)

    def test_mixed_or_combine_reference_requests_rejected_before_launch(self):
        for options in ([], ['--ll-stage-timer', '--dispatch-only'], ['--ll-expert-timer', '--dispatch-only'],
                        ['--ll-timer-mode', 'drain', '--dispatch-only'], ['--skip-check', '--dispatch-only']):
            with self.subTest(options=options), self.assertRaises(ValueError):
                st.validate_args(st.parse_args(['--ll-reference-profile', *options]), 32)

    def test_modes_clear_stale_build_flags_and_reference_is_dispatch_only(self):
        with patch.dict('os.environ', {'DEEPEP_LL_STAGE_TIMER': 'ON', 'DEEPEP_LL_REFERENCE_PROFILE': 'ON'}):
            for mode in ('reference', 'perf'):
                args = launch.parse_args(['1', '192.0.2.1', 'cpu', '--mode', mode])
                env = launch.child_environment(args)
                self.assertEqual(env['DEEPEP_LL_STAGE_TIMER'], 'ON' if mode == 'timer' else 'OFF')
                self.assertEqual(env['DEEPEP_LL_REFERENCE_PROFILE'], 'ON' if mode == 'reference' else 'OFF')
                arguments = launch.test_arguments(args)
                options = st.parse_args(arguments[:2] + arguments[4:])
                self.assertEqual(st.validate_args(options, 32), 256)
                self.assertEqual(options.dispatch_only, mode == 'reference')
                self.assertEqual(options.ll_reference_profile, mode == 'reference')
                self.assertEqual((options.warmup, options.num_iters, options.perf_tail), (10, 100, 50))
                if mode == 'reference': self.assertEqual(options.ll_timer_iters, 1)
                command = launch.command_plan(args, '/project with spaces/python')[0]
                self.assertEqual(command, ['git', 'submodule', 'update', '--init', '--recursive', '--depth', '1', '--', 'third-party/shmem'])

    def test_reference_identity_is_typed_and_needs_source_fingerprint(self):
        valid = dict(reference_profile=True, stage_timer=False, reference_profile_schema=4,
                     installed_source_sha256='a' * 64)
        st.validate_reference_profile_capability(True, valid)
        for key, value in (('reference_profile', 1), ('stage_timer', 0), ('reference_profile_schema', True),
                           ('installed_source_sha256', None), ('installed_source_sha256', 'old')):
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                st.validate_reference_profile_capability(True, dict(valid, **{key: value}))

    def test_reference_cache_isolated_but_legacy_timer_still_reuses_cache(self):
        build = load('reference_build_support', ROOT / 'build_support.py')
        plain = dict(source_sha256='source', ll_stage_timer=False, ll_reference_profile=False)
        self.assertEqual(build.build_cache_key(plain), build.build_cache_key(dict(plain, ll_stage_timer=True)))
        self.assertNotEqual(build.build_cache_key(plain), build.build_cache_key(dict(plain, ll_reference_profile=True)))


class SamplingTests(unittest.TestCase):
    def test_reference_tail_keeps_prelude_and_exports_only_after_all_captures(self):
        events, metadata = [], []
        buffer = object()
        class Tensor:
            device = 'npu:0'
            def fill_(self, value): events.append('fill')
        class Trace:
            def __init__(self, owner, operation, frequency_hz):
                self.buffer, self.operation, self.mode = owner, operation, 'reference'
                self.frequency_hz, self.launches = frequency_hz, 0
                self.tensor = Tensor()
                events.append('allocate')
            def complete(self): events.append('copy')
            def save(self, path, meta):
                self_test.assertIn('ll_reference_profile', path.parts)
                metadata.append(meta)
                events.append('save')
        self_test = self
        dist = SimpleNamespace(get_world_size=lambda: 32, ReduceOp=SimpleNamespace(SUM=0),
                               all_reduce=lambda *a, **k: events.append('reduce'))
        torch = SimpleNamespace(float32=0, zeros=lambda *a, **k: Tensor(), ones=lambda *a, **k: Tensor(),
                                exp=lambda *a, **k: events.append('exp'), distributed=dist,
                                npu=SimpleNamespace(synchronize=lambda: events.append('sync')))
        def invoke():
            reference.attach(buffer, {'input': Tensor()}, 'dispatch')
            events.append('invoke')
            return (SimpleNamespace(current_stream_wait=lambda: events.append('wait')),)
        args = SimpleNamespace(run_directory=str(ROOT / '.tmp/reference-cpu'), case_id=0,
                               ll_reference_profile=True, ll_timer_iters=2, ll_timer_mode='observe',
                               ll_timer_frequency_hz=1e9, seed=0, num_tokens=8, num_experts=384, warmup=2, num_iters=3,
                               delay_rows=1, delay_cols=1, group_backend='hccl',
                               reference_profile_seed=0, reference_route_sha256='a' * 64)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist}))
            stack.enter_context(patch.object(_native, 'require_native', return_value=SimpleNamespace(low_latency_ranks_per_group=lambda _: 16)))
            stack.enter_context(patch.object(reference, 'ReferenceTrace', Trace))
            stack.enter_context(patch.object(st, 'control_device', return_value='npu'))
            stack.enter_context(patch.object(st, 'completion_boundary', side_effect=lambda _: events.append('boundary')))
            stack.enter_context(patch.object(st, 'agree_check', side_effect=lambda fn, *a: fn()))
            stack.enter_context(patch.object(st, 'case_config', return_value=dict(B=8)))
            st.diagnostic_calls(buffer, invoke, lambda _: events.append('check'), args, 'dispatch', 17, 256, 'normal')
        expected = ['allocate'] * 2 + ['boundary', 'reduce'] + ['invoke'] * 2 + ['sync']
        expected += ['exp'] * 200 + ['invoke'] * 3
        expected += ['invoke', 'wait'] * 2
        expected += ['boundary'] + ['copy', 'check', 'save'] * 2
        self.assertEqual(events, expected)
        self.assertEqual([item['iteration'] for item in metadata], [0, 1])
        self.assertTrue(all(item['rank'] == 17 and item['world_size'] == 32 and item['profile_seed'] == 0 for item in metadata))
        self.assertTrue(all(item['route_sha256'] == 'a' * 64 for item in metadata))


if __name__ == '__main__':
    (ROOT / '.tmp').mkdir(exist_ok=True)
    unittest.main()
