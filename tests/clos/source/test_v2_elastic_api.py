"""CPU contracts for the V2 API sharing V1 native kernels (not NPU precision)."""

import os
import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock, patch
from types import SimpleNamespace

import torch
import deep_ep
from deep_ep.backends.clos import ElasticBuffer, EPHandle, topk_idx_t
from test_public_api import _FakeBufferRuntime, _Group
from deep_ep.backends.clos.config import _MAX_NUM_AIV


class _Stream:
    def synchronize(self):
        pass


class _Event(deep_ep.EventHandle):
    def __init__(self, stream=None):
        self.stream = stream
        self._event = object()
        self._device = 0
        self._stream = stream
        self.wait_count = 0

    def current_stream_wait(self):
        self.wait_count += 1


class _ElasticRuntime(_FakeBufferRuntime):
    def __init__(self, *args):
        super().__init__(*args)
        self.use_1g_huge_pages = args[6]
        self.barriers = []
        self.destroyed = False

    def _intranode_combine_begin(self, *args):
        self.phase_log = getattr(self, "phase_log", [])
        self.single_combine_args = args[15:]
        if len(args) > 17 and args[17] is not None:
            self.phase_log.append("pipeline" if args[18] else "early_submit")
            x, values = self.aggregate(args[0], None, args[1], args[17], args[8], args[11], args[16], args[5])
            self.phase_log.append("aggregate")
            args = (x, values, *args[2:])
        self.phase_log.append("communication")
        result = self.intranode_combine(*args[:15])
        consumed = False

        def run():
            nonlocal consumed
            if consumed:
                raise RuntimeError("combine epilogue has already been consumed")
            consumed = True
            self.phase_log.append("reduce")
            return result

        self.last_epilogue = SimpleNamespace(run=run)
        return self.last_epilogue

    def num_device_aiv(self):
        return 64

    def elastic_barrier(self, num_aiv):
        self.barriers.append(num_aiv)

    def destroy(self):
        self.destroyed = True
        super().destroy()

    def intranode_dispatch_with_plan(self, *args):
        result = super().intranode_dispatch_with_plan(*args)
        # Native rebuilds the ephemeral plan when cached dispatch carries weights.
        if args[8] is not None and args[2] is not None:
            result = (*result[:-1], result[-1].clone() if result[-1] is not None else None)
        self.returned_plan = result[-1]
        return result

    def intranode_dispatch_with_handle(self, *args):
        self.public_export_calls = getattr(self, "public_export_calls", 0) + 1
        self.defer_public_expert_prefix = args[-1]
        args = args[:-1]
        result = self.intranode_dispatch_with_plan(*args)
        world, experts = self.args[1], args[6].numel() // self.args[1]
        rows, k = result[0].size(0), args[2].size(1)
        public = [
            torch.zeros(world, dtype=torch.int32),
            torch.zeros(experts, dtype=torch.int32),
            torch.zeros(experts, dtype=torch.int32),
            torch.full((rows, k + 2), -1, dtype=torch.int32),
            torch.full(args[2].shape, -1, dtype=torch.int32),
        ]
        self.public_outputs = public
        return result, public

    def expand(self, *args):
        result = super().expand(*args[:12])
        if len(args) > 12 and args[12] is not None:
            offsets, counts, mapping = result[3:]
            args[12].copy_(offsets[:-1] + counts)
            args[13][:, 2:].copy_(mapping)
        if len(args) > 14 and args[14] is not None:
            mapping = result[5]
            valid = mapping >= 0
            args[14].copy_(torch.where(valid.sum(1) == 1, mapping.amax(1), -1))
        return result

    def aggregate(self, *args):
        self.single_aggregate_args = args[6:]
        return super().aggregate(*args[:6])


class ElasticApiTest(unittest.TestCase):
    def setUp(self):
        self.stream = _Stream()
        self.compute_stream = _Stream()
        self.device_sync = Mock()
        self.patches = (
            patch("deep_ep.backends.clos.elastic._prepare_shmem_bootstrap", return_value=([], "")),
            patch(
                "deep_ep.backends.clos.elastic.get_native_class",
                return_value=_ElasticRuntime,
            ),
            patch(
                "deep_ep.backends.clos.elastic._create_communication_stream",
                return_value=self.stream,
            ),
            patch(
                "deep_ep.backends.clos.elastic._try_record_event",
                side_effect=lambda stream: _Event(stream),
            ),
            patch(
                "deep_ep.backends.clos.elastic._current_npu_stream",
                return_value=self.compute_stream,
            ),
            patch(
                "deep_ep.backends.clos.elastic._use_comm_stream",
                side_effect=lambda stream: nullcontext(),
            ),
            patch("deep_ep.backends.clos.elastic.EventHandle", _Event),
            patch(
                "deep_ep.backends.clos.elastic._synchronize_npu_device",
                self.device_sync,
            ),
        )
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def make(self, **kwargs):
        return ElasticBuffer(
            _Group(size=8), num_max_tokens_per_rank=17, hidden=128, num_topk=4, explicitly_destroy=True, **kwargs
        )

    def manual_make(self, **kwargs):
        # Existing scheduler tests explicitly exercise manual tuning.
        options = dict(
            udma_auto_policy=False,
            udma_schedule="peer",
            udma_dispatch_peer_order="rank",
            udma_combine_peer_order="rank",
            udma_workers_per_peer=1,
            udma_qps_per_worker=1,
            udma_peers_per_worker=4,
            udma_wqes_per_visit=8,
        )
        options.update(kwargs)
        return self.make(**options)

    def fresh(self, buffer, **kwargs):
        x = torch.ones((13, 128), dtype=torch.bfloat16)
        idx = torch.zeros((13, 4), dtype=topk_idx_t)
        weights = torch.ones((13, 4), dtype=torch.float32)
        return buffer.dispatch(
            x, topk_idx=idx, topk_weights=weights, num_experts=32, num_sms=kwargs.pop("num_sms", _MAX_NUM_AIV), **kwargs
        )

    def test_exports_and_shared_backend(self):
        from deep_ep._clos_adapter import ElasticBuffer as ClosElasticBuffer

        self.assertTrue(issubclass(ClosElasticBuffer, ElasticBuffer))
        self.assertIsInstance(object.__new__(EPHandle), deep_ep.EPHandle)
        for transport in ("udma", "mte"):
            with self.subTest(transport=transport):
                b = self.manual_make(transport=transport, udma_peers_per_worker=1)
                self.assertTrue(b.runtime.v1_shared_jetty)
                self.assertFalse(b.runtime.v1_token_schedule)
                x, idx, weights, h, _ = self.fresh(b)
                self.assertEqual(len(h._dispatch_layout), 6)
                self.assertEqual(b.runtime.udma_schedule, (1, 8))
                self.assertEqual(b.runtime.dispatch_args[15], False)
                self.assertEqual(h.num_max_tokens_per_rank, 17)
                self.assertEqual(h.num_sms, _MAX_NUM_AIV)
                self.assertEqual(h._dispatch_plan is None, transport == "mte")
                b.combine(x, h, topk_weights=weights)
                self.assertEqual(b.runtime.udma_schedule, (1, 8))
                self.assertEqual(b.runtime.phase_log, ["communication", "reduce"])
                for removed in (True, False):
                    with self.subTest(split_reduce=removed), self.assertRaises(TypeError):
                        b.combine(x, h, split_reduce=removed)
                b.destroy()

    def test_peer_orders_forwarded_and_cached_handle_reused(self):
        orders = ("rank", "group", "group_staggered")
        for dispatch in orders:
            for combine in orders:
                b = self.manual_make(
                    transport="udma",
                    udma_dispatch_peer_order=dispatch,
                    udma_combine_peer_order=combine,
                    udma_ranks_per_group=4,
                )
                self.assertEqual(b.runtime.udma_peer_options, (orders.index(dispatch), orders.index(combine), 4))
                for name in ("udma_dispatch_peer_order", "udma_combine_peer_order", "udma_ranks_per_group"):
                    with self.assertRaises(AttributeError):
                        setattr(b, name, getattr(b, name))
                x, _, weights, handle, _ = self.fresh(b)
                cached = b.dispatch(torch.ones((13, 128), dtype=torch.bfloat16), handle=handle)
                self.assertIs(cached[3], handle)
                b.combine(x, handle, topk_weights=weights)
                b.destroy()
        for options in (
            {"udma_dispatch_peer_order": "group"},
            {"udma_combine_peer_order": "bad"},
            {"udma_ranks_per_group": 3},
            {"udma_ranks_per_group": True},
            {"transport": "mte", "udma_ranks_per_group": 4},
            {"udma_schedule": "token", "udma_ranks_per_group": 4},
        ):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                self.manual_make(**({"transport": "udma"} | options))

    def test_group_round_robin_cached_interfaces(self):
        b = self.manual_make(
            transport="udma",
            udma_dispatch_peer_order="group_round_robin",
            udma_combine_peer_order="group_round_robin",
            udma_ranks_per_group=2,
            udma_peers_per_group_per_worker=2,
            udma_workers_per_peer=2,
        )
        self.assertEqual(b.runtime.udma_peer_options, (3, 3, 2, 2, 2))
        for expanded in (False, True):
            x, _, weights, h, _ = self.fresh(b, do_expand=expanded)
            b.combine(x, h, topk_weights=weights)
            cached = b.dispatch(torch.ones((13, 128), dtype=torch.bfloat16), handle=h, do_expand=expanded)
            b.combine(cached[0], cached[3])
        b.destroy()

    def test_two_qps_per_worker_cached_interfaces(self):
        b = self.manual_make(transport="udma", udma_qps_per_worker=2)
        self.assertEqual(b.runtime.udma_peer_options, (0, 0, 0, 0, 1, 2))
        self.assertEqual(b.udma_qps_per_worker, 2)
        for expanded in (False, True):
            x, _, weights, h, _ = self.fresh(b, do_expand=expanded)
            b.combine(x, h, topk_weights=weights)
            cached = b.dispatch(torch.ones((13, 128), dtype=torch.bfloat16), handle=h, do_expand=expanded)
            b.combine(cached[0], cached[3])
        b.destroy()
        with self.assertRaises(ValueError):
            self.manual_make(transport="udma", udma_workers_per_peer=2, udma_qps_per_worker=2)

    def test_single_source_cache_is_private_and_reused(self):
        b = self.manual_make(transport="udma", udma_peers_per_worker=1)
        self.assertFalse(b._use_early_single_source_combine)
        b._use_early_single_source_combine = True
        x, _, weights, h, _ = self.fresh(b, do_expand=True)
        table = h._single_source_slots
        mapping = h._expand_layout[2]
        expected = torch.where((mapping >= 0).sum(1) == 1, mapping.amax(1), -1)
        self.assertTrue(torch.equal(table, expected))
        b.combine(x, h, topk_weights=weights)
        self.assertIs(b.runtime.single_combine_args[0], x)
        self.assertIs(b.runtime.single_combine_args[1], table)
        self.assertIsNone(b.runtime.aggregate_args[1])  # weights are a separate result
        self.assertIs(b.runtime.single_aggregate_args[1], h._dispatch_layout[0])
        cached = b.dispatch(torch.ones((13, 128), dtype=torch.bfloat16), handle=h, do_expand=True)
        self.assertIs(cached[3]._single_source_slots, table)
        replacement = cached[0].clone()
        b.combine(replacement, h)
        self.assertIs(b.runtime.single_combine_args[0], replacement)
        b._use_early_single_source_combine = False
        b.runtime.phase_log = []
        b.combine(replacement, h)
        self.assertEqual(b.runtime.phase_log, ["communication", "reduce"])
        self.assertIsNone(b.runtime.single_combine_args[2])
        b._use_single_source_combine = False
        b.combine(replacement, h)
        self.assertEqual(b.runtime.single_combine_args, (None, None, None, 0))
        self.assertEqual(b.runtime.single_aggregate_args, (None, None))
        b.destroy()

    def test_private_aggregate_pipeline_cached_and_guards(self):
        b = self.manual_make(transport="udma", udma_peers_per_worker=1)
        self.assertEqual(b._aggregate_pipeline_chunk_tokens, 0)
        x, _, weights, h, _ = self.fresh(b, do_expand=True, num_sms=32)
        for chunk in (32, 64):
            b._aggregate_pipeline_chunk_tokens = chunk
            b.runtime.phase_log = []
            b.combine(x, h, topk_weights=weights)
            self.assertEqual(b.runtime.phase_log, ["pipeline", "aggregate", "communication", "reduce"])
            self.assertIs(b.runtime.single_combine_args[2], h._expand_layout[2])
            self.assertEqual(b.runtime.single_combine_args[3], chunk)
            self.assertIsNone(b.runtime.aggregate_args[1])
            cached = b.dispatch(torch.ones((13, 128), dtype=torch.bfloat16), handle=h, do_expand=True)
            b.combine(cached[0], cached[3])
            self.assertIs(b.runtime.single_combine_args[2], h._expand_layout[2])
        for chunk in (-1, 1, 128):
            b._aggregate_pipeline_chunk_tokens = chunk
            with self.assertRaisesRegex(ValueError, "pipeline chunk"):
                b.combine(x, h)
        b._aggregate_pipeline_chunk_tokens = 32
        for attr, value in (
            ("transport", "mte"),
            ("_use_single_source_combine", False),
            ("_use_early_single_source_combine", True),
        ):
            old = getattr(b, attr)
            setattr(b, attr, value)
            with self.assertRaisesRegex(ValueError, "pipeline requires"):
                b.combine(x, h)
            setattr(b, attr, old)
        b.destroy()

    def test_private_self_send_pipeline_requires_independent_qps(self):
        b = self.manual_make(transport="udma", udma_schedule="token")
        x, _, weights, h, _ = self.fresh(b, do_expand=True, num_sms=32)
        for chunk in (32, 64):
            b._aggregate_pipeline_chunk_tokens = chunk
            b.combine(x, h, topk_weights=weights, num_sms=32)
            self.assertEqual(b.runtime.single_combine_args[3], chunk)
            with (
                patch("deep_ep.backends.clos.elastic._MAX_NUM_AIV", 64),
                patch("deep_ep.backends.clos.config._MAX_NUM_AIV", 64),
            ):
                x64, _, _, h64, _ = self.fresh(b, do_expand=True, num_sms=64)
                with self.assertRaisesRegex(ValueError, "self-send aggregate pipeline"):
                    b.combine(x64, h64, num_sms=64)
        b.num_ranks = 16
        with self.assertRaisesRegex(ValueError, "self-send aggregate pipeline"):
            b.combine(x, h, num_sms=32)
        b.destroy()

    def test_combine_epilogue_dependency_and_completion_order(self):
        # The gate is not an input readiness event: communication must precede it.
        for expanded in (False, True):
            for allocation in (False, True):
                for asynchronous in (False, True):
                    with self.subTest(expanded=expanded, allocation=allocation, asynchronous=asynchronous):
                        b = self.manual_make()
                        x, _, w, h, _ = self.fresh(b, do_expand=expanded)
                        log = b.runtime.phase_log = []
                        gate = _Event()
                        gate.current_stream_wait = lambda: log.append("epilogue_wait")
                        previous = _Event()
                        previous.current_stream_wait = lambda: log.append("input_wait")
                        completion = _Event()
                        completion.current_stream_wait = lambda: log.append("caller_wait")

                        def record(_):
                            if not log:
                                return previous
                            log.append("completion_record")
                            return completion

                        # Use a recorded prologue event without requiring explicit
                        # previous_event when allocation remains on caller stream.
                        with patch("deep_ep.backends.clos.elastic._try_record_event", side_effect=record) as rec:
                            result = b.combine(
                                x,
                                h,
                                topk_weights=w,
                                previous_event_before_epilogue=gate,
                                allocate_on_comm_stream=allocation,
                                async_with_compute_stream=asynchronous,
                            )
                        self.assertEqual(
                            log,
                            ["input_wait", "communication", "epilogue_wait", "reduce", "completion_record"]
                            + ([] if asynchronous else ["caller_wait"]),
                        )
                        self.assertEqual(rec.call_count, 2)
                        if asynchronous:
                            self.assertIs(result[-1].event, completion)
                            self.assertTrue(any(v is b.runtime.last_epilogue for v in result[-1]._retained_objects))
                            result[-1].current_stream_wait(release_handle=True)
                            self.assertEqual(log[-1], "caller_wait")
                        with self.assertRaises(RuntimeError):
                            b.runtime.last_epilogue.run()
                        b.destroy()

    def test_event_constraints_match_legacy_elastic(self):
        b = self.manual_make()
        x, _, _, h, _ = self.fresh(b)
        with self.assertRaises(ValueError):
            b.combine(x, h, previous_event=_Event(), allocate_on_comm_stream=False)
        for bad in (True, object()):
            with self.assertRaises(TypeError):
                b.combine(x, h, previous_event_before_epilogue=bad)
        # Epilogue-only event permits caller-stream allocation in both APIs.
        self.fresh(b, previous_event_before_epilogue=_Event(), allocate_on_comm_stream=False)
        b.combine(x, h, previous_event_before_epilogue=_Event(), allocate_on_comm_stream=False)
        b.destroy()

    def test_early_combine_keeps_input_and_epilogue_events_on_their_sides(self):
        b = self.manual_make(transport="udma")
        b._use_early_single_source_combine = True
        x, _, w, h, _ = self.fresh(b, do_expand=True)
        log = b.runtime.phase_log = []
        dependency, gate, completion = _Event(), _Event(), _Event()
        dependency.current_stream_wait = lambda: log.append("input_wait")
        gate.current_stream_wait = lambda: log.append("epilogue_wait")
        completion.current_stream_wait = lambda: log.append("caller_wait")

        def record(_):
            if not log:
                return dependency
            log.append("completion_record")
            return completion

        with patch("deep_ep.backends.clos.elastic._try_record_event", side_effect=record):
            result = b.combine(
                x, h, topk_weights=w, previous_event_before_epilogue=gate, async_with_compute_stream=True
            )
        self.assertEqual(
            log,
            [
                "input_wait",
                "early_submit",
                "aggregate",
                "communication",
                "epilogue_wait",
                "reduce",
                "completion_record",
            ],
        )
        self.assertIs(b.runtime.single_combine_args[2], h._expand_layout[2])
        self.assertIs(result[-1].event, completion)
        self.assertTrue(any(v is b.runtime.last_epilogue for v in result[-1]._retained_objects))
        result[-1].current_stream_wait(release_handle=True)
        self.assertEqual(log[-1], "caller_wait")
        b.destroy()

    def test_cached_plan_and_new_weights(self):
        b = self.manual_make(transport="udma")
        x, _, w, h, _ = self.fresh(b)
        original = h._dispatch_plan
        _, idx, weights, cached, _ = b.dispatch(x, handle=h)
        self.assertIs(cached, h)
        self.assertIsNone(idx)
        self.assertIsNone(weights)
        self.assertIs(b.runtime.cached_dispatch_plan, original)
        self.assertIs(b.runtime.returned_plan, original)
        _, idx, weights, _, _ = b.dispatch(x, handle=h, topk_weights=w * 2)
        self.assertIsNone(idx)
        self.assertIsNotNone(weights)
        self.assertIs(b.runtime.dispatch_args[2], h.topk_idx)
        self.assertIsNot(b.runtime.returned_plan, original)
        self.assertIs(h._dispatch_plan, original)
        b.destroy()

    def test_public_handle_ownership_and_cached_reuse(self):
        fields = (
            "psum_num_recv_tokens_per_scaleup_rank",
            "psum_num_recv_tokens_per_expert",
            "num_unaligned_recv_tokens_per_expert",
            "recv_src_metadata",
            "dst_buffer_slot_idx",
        )
        for expanded in (False, True):
            b = self.manual_make(transport="udma")
            _, _, _, h, _ = self.fresh(b, do_expand=expanded, expert_alignment=8 if expanded else 1)
            self.assertEqual(b.runtime.defer_public_expert_prefix, expanded)
            tensors = [getattr(h, n) for n in fields]
            for actual, native in zip(tensors, b.runtime.public_outputs):
                self.assertIs(actual, native)
                self.assertEqual(actual.dtype, torch.int32)
                self.assertTrue(actual.is_contiguous())
            self.assertEqual(tensors[3].shape, (h.num_recv_tokens, 6))
            self.assertEqual(tensors[4].shape, (13, 4))
            snapshots = [t.clone() for t in tensors]
            b.dispatch(torch.ones((13, 128), dtype=torch.bfloat16), handle=h, do_expand=expanded, num_sms=_MAX_NUM_AIV)
            self.assertEqual(b.runtime.public_export_calls, 1)
            if expanded:
                self.assertIsNone(b.runtime.expand_args[2])
            for n, original, saved in zip(fields, tensors, snapshots):
                self.assertIs(getattr(h, n), original)
                torch.testing.assert_close(original, saved, rtol=0, atol=0)
            self.assertIsNone(h.token_metadata_at_forward)
            self.assertIsNone(h.channel_linked_list)
            b.destroy()

    def test_public_handle_reference_rejects_wrong_fields(self):
        from test_v2_elastic_v1_kernel import check_public_handle, PUBLIC_HANDLE_FIELDS

        routes = [torch.tensor([[0, 1, 2], [-1, 3, -1]]), torch.tensor([[3, 0, -1]])]
        h = SimpleNamespace(
            num_experts=4, num_max_tokens_per_rank=4, expert_alignment=4, num_recv_tokens=2, do_expand=True
        )
        values = [
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([2, 5], dtype=torch.int32),
            torch.tensor([2, 1], dtype=torch.int32),
            torch.tensor([[0, 0, 0, 4, -1], [4, 4, -1, 1, -1]], dtype=torch.int32),
            torch.tensor([[0, -1, 0], [-1, 1, -1]], dtype=torch.int32),
        ]
        for name, value in zip(PUBLIC_HANDLE_FIELDS, values):
            setattr(h, name, value)
        check_public_handle(h, routes, 0)
        for name in PUBLIC_HANDLE_FIELDS:
            original = getattr(h, name)
            bad = original.clone()
            bad.view(-1)[0] += 1
            setattr(h, name, bad)
            with self.assertRaises(AssertionError):
                check_public_handle(h, routes, 0)
            setattr(h, name, original)

    def test_fixed_capacity_and_rejection(self):
        b = self.manual_make(transport="udma")
        x, _, _, h, _ = self.fresh(b, do_cpu_sync=False)
        self.assertEqual(x.shape[0], 17 * 8)
        self.assertEqual(h.num_recv_tokens, 17 * 8)
        with self.assertRaisesRegex(NotImplementedError, "requires do_cpu_sync"):
            self.fresh(b, do_expand=True, do_cpu_sync=False)
        with self.assertRaisesRegex(ValueError, "match EPHandle"):
            b.dispatch(torch.ones((13, 128), dtype=torch.bfloat16), handle=h, num_sms=32)
        b.destroy()

    def test_async_retains_rebuilt_plan(self):
        b = self.manual_make(transport="udma")
        x, _, w, h, _ = self.fresh(b)
        with patch("deep_ep.backends.clos.elastic._try_record_event", side_effect=lambda _: _Event()):
            _, _, _, _, event = b.dispatch(x, handle=h, topk_weights=w, async_with_compute_stream=True)
        self.assertTrue(any(t is b.runtime.returned_plan for t in event._retained_objects))
        self.assertIsNot(b.runtime.returned_plan, h._dispatch_plan)
        b.destroy()

    def test_constructor_tuning_validation(self):
        for options in (
            {"udma_peers_per_worker": 3},
            {"udma_wqes_per_visit": 7},
            {"udma_schedule": "wrong"},
            {"transport": "mte", "udma_schedule": "token"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.manual_make(**options)
        b = self.manual_make(transport="udma", udma_schedule="token")
        self.assertTrue(b.runtime.v1_token_schedule)
        b.destroy()

    def test_transport_selection(self):
        with patch.dict(os.environ, {"ASCEND_DEEPEP_TRANSPORT": "udma"}):
            b = self.manual_make()
            self.assertEqual(b.transport, "udma")
            b.destroy()
            b = self.manual_make(transport="mte")
            self.assertEqual(b.transport, "mte")
            b.destroy()

    def test_foreign_handle_rejected(self):
        b, other = self.manual_make(), self.manual_make()
        x, _, _, h, _ = self.fresh(b)
        with self.assertRaisesRegex(ValueError, "different ElasticBuffer"):
            other.combine(x, h)
        b.destroy()
        other.destroy()

    def test_size_hint_alignment_and_overflow(self):
        for fp8 in (False, True):
            n = ElasticBuffer.get_buffer_size_hint(_Group(size=8), 8192, 6144, 8, fp8)
            self.assertEqual(n % (2 * 1024 * 1024), 0)
            self.assertGreaterEqual(n, 8192 * 8 * 6144 * 2)  # BF16 combine even with FP8 dispatch
        with self.assertRaises(OverflowError):
            ElasticBuffer.get_buffer_size_hint(_Group(size=8), 1 << 62, 6144, 8)

    def test_constructor_resource_estimate(self):
        buffer = ElasticBuffer(
            _Group(),
            num_max_tokens_per_rank=8,
            hidden=16,
            num_topk=2,
            explicitly_destroy=True,
            use_1g_huge_pages=True,
        )
        self.assertTrue(buffer.use_1g_huge_pages)
        self.assertTrue(buffer.runtime.use_1g_huge_pages)
        self.assertEqual(buffer.get_theoretical_num_sms(8, 2), 6)
        self.assertEqual(buffer.get_theoretical_num_qps(6), 0)
        with self.assertRaises(ValueError):
            buffer.get_theoretical_num_sms(8, 9)
        buffer.destroy()

    def test_constructor_rejects_non_bool_huge_page_option(self):
        with self.assertRaisesRegex(TypeError, "use_1g_huge_pages must be a bool"):
            ElasticBuffer(
                _Group(),
                num_max_tokens_per_rank=8,
                hidden=16,
                num_topk=2,
                use_1g_huge_pages=1,
            )

    def test_constructor_rejects_huge_page_mismatch_before_native_init(self):
        def gather_mismatched(values, value, group):
            del group
            values[:] = [value] * 4
            if isinstance(value, bool):
                values[-1] = not value

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=gather_mismatched,
            ),
            self.assertRaisesRegex(
                ValueError,
                "all ranks must use the same use_1g_huge_pages",
            ),
        ):
            ElasticBuffer(
                _Group(),
                num_max_tokens_per_rank=8,
                hidden=16,
                num_topk=2,
                use_1g_huge_pages=True,
            )

    def test_constructor_rejects_rank_capacity_mismatch_before_native_init(self):
        def gather_mismatched(values, value, group):
            del value, group
            values[:] = [8, 9, 8, 8]

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=gather_mismatched,
            ),
            self.assertRaisesRegex(
                ValueError,
                "all ranks must use the same num_max_tokens_per_rank",
            ),
        ):
            ElasticBuffer(
                _Group(),
                num_max_tokens_per_rank=8,
                hidden=16,
                num_topk=2,
            )

    def test_implicit_destructor_uses_collective_destroy_path(self):
        buffer = ElasticBuffer(
            _Group(),
            num_max_tokens_per_rank=8,
            hidden=16,
            num_topk=2,
            explicitly_destroy=False,
        )
        runtime = buffer.runtime
        with self.assertRaises(RuntimeError):
            buffer.destroy()

        buffer.__del__()
        self.assertTrue(buffer._destroyed)
        self.assertTrue(runtime.destroyed)
        self.assertEqual(runtime.barriers, [1, 1])
        self.assertEqual(self.device_sync.call_count, 4)

    def test_barrier_orders_compute_and_communication_streams(self):
        active_stream = {"value": None}
        log = []

        @contextmanager
        def use_stream(stream):
            previous = active_stream["value"]
            active_stream["value"] = stream
            try:
                yield
            finally:
                active_stream["value"] = previous

        class TrackingEvent:
            def __init__(self, stream=None):
                log.append(("record", stream))

            def current_stream_wait(self):
                log.append(("wait", active_stream["value"]))

        with (
            patch(
                "deep_ep.backends.clos.elastic._use_comm_stream",
                side_effect=use_stream,
            ),
            patch("deep_ep.backends.clos.elastic.EventHandle", TrackingEvent),
            patch(
                "deep_ep.backends.clos.elastic._synchronize_npu_device",
                side_effect=lambda: log.append(("device_sync", None)),
            ),
        ):
            buffer = ElasticBuffer(
                _Group(),
                num_max_tokens_per_rank=8,
                hidden=16,
                num_topk=2,
                explicitly_destroy=True,
            )
            log.clear()
            original_barrier = buffer.runtime.elastic_barrier

            def tracked_barrier(num_aiv):
                original_barrier(num_aiv)
                log.append(("barrier", active_stream["value"]))

            buffer.runtime.elastic_barrier = tracked_barrier
            buffer.barrier(with_cpu_sync=True, sequential=False)

        self.assertEqual(
            log,
            [
                ("record", self.compute_stream),
                ("wait", self.stream),
                ("device_sync", None),
                ("barrier", self.stream),
                ("record", self.stream),
                ("device_sync", None),
                ("wait", self.compute_stream),
            ],
        )
        self.assertEqual(buffer.runtime.barriers, [1, 1])
        buffer.destroy()


if __name__ == "__main__":
    unittest.main()
