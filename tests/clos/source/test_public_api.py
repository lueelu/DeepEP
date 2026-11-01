import gc
import inspect
import os
import unittest
import warnings
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
from weakref import ref

import torch

import deep_ep
import deep_ep.buffer as deep_ep_buffer_module
import deep_ep.backends.clos.buffer as buffer_module
from deep_ep.buffers.legacy import Buffer as DeepEPBuffer
from deep_ep.utils import envs
from deep_ep.utils.event import EventHandle as DeepEPEventHandle
from deep_ep.utils.math import (
    align,
    calc_diff,
    count_bytes,
    create_grouped_scores,
    inplace_unique,
    per_token_cast_back,
    per_token_cast_to_fp8,
    safe_div,
)

from deep_ep.backends.clos import (
    Buffer,
    Config,
    EventHandle,
    EventOverlap,
    ExpandHandle,
    Handle,
    Slot,
    SlotState,
    topk_idx_t,
)
from deep_ep.backends.clos._native import _resolve_native_class
from deep_ep.backends.clos.config import _MAX_NUM_AIV
from deep_ep.backends.clos.utils import align_up


class _FakeBufferRuntime:
    def __init__(
        self,
        rank,
        num_ranks,
        num_bytes,
        transport,
        low_latency_mode,
        ip_port,
        use_1g_huge_pages,
        unique_id,
        bootstrap_key,
        v1_shared_jetty,
        v1_token_schedule=False,
    ):
        self.args = (
            rank,
            num_ranks,
            num_bytes,
            transport,
            low_latency_mode,
            ip_port,
            use_1g_huge_pages,
        )
        self.initialized = True
        self.barrier_count = 0
        self.unique_id = unique_id
        self.bootstrap_key = bootstrap_key
        self.v1_shared_jetty = v1_shared_jetty
        self.v1_token_schedule = v1_token_schedule

    def is_initialized(self):
        return self.initialized

    def barrier(self):
        self.barrier_count += 1

    def destroy(self):
        self.initialized = False

    def get_comm_meta_ptr(self):
        return 0xA30000

    def get_workspace_ptr(self):
        return 0xB40000

    def transport(self):
        return self.args[3]

    def low_latency_mode(self):
        return self.args[4]

    def get_dispatch_layout(self, topk_idx, num_experts, allocation_stream_id=None):
        self.allocation_stream_id = allocation_stream_id
        self.layout_args = (topk_idx, num_experts)
        num_tokens = topk_idx.size(0)
        num_ranks = self.args[1]
        return (
            torch.zeros((num_ranks,), dtype=torch.int32),
            None,
            torch.zeros((num_experts,), dtype=torch.int32),
            torch.zeros((num_tokens, num_ranks), dtype=torch.bool),
        )

    def intranode_dispatch_with_plan(self, *args):
        cached_plan = args[20] if len(args) > 20 else None
        self.cached_dispatch_plan = cached_plan
        result = self.intranode_dispatch(*args[:20])
        plan = None
        if self.transport() == "udma":
            x, channels = args[0], args[12] or 32
            stride = ((x.size(0) + channels - 1) // channels + 7) // 8 * 8
            plan = (
                cached_plan
                if cached_plan is not None
                else torch.empty((self.args[1], channels, stride), dtype=torch.int32)
            )
        return (*result, plan)

    def intranode_dispatch(
        self,
        x,
        x_scales,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        cached_num_recv_tokens,
        cached_rank_prefix_matrix,
        cached_channel_prefix_matrix,
        expert_alignment,
        num_worst_tokens,
        num_aiv,
        num_max_nvl_chunked_send_tokens,
        num_max_nvl_chunked_recv_tokens,
        zero_copy,
        allocation_stream_id=None,
        num_max_tokens_per_rank=None,
        udma_peers_per_worker=4,
        udma_wqes_per_visit=8,
    ):
        self.udma_schedule = (udma_peers_per_worker, udma_wqes_per_visit)
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.allocation_stream_id = allocation_stream_id
        self.dispatch_args = (
            x,
            x_scales,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            cached_num_recv_tokens,
            cached_rank_prefix_matrix,
            cached_channel_prefix_matrix,
            expert_alignment,
            num_worst_tokens,
            num_aiv,
            num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens,
            zero_copy,
        )
        num_ranks = self.args[1]
        local_num_tokens = x.size(0)
        num_tokens = x.size(0) if cached_num_recv_tokens == 0 else cached_num_recv_tokens
        if num_worst_tokens > 0:
            num_tokens = num_worst_tokens
        hidden = x.size(1)
        num_topk = topk_idx.size(1) if topk_idx is not None else 0
        num_channels = num_aiv or 32
        recv_x = torch.zeros((num_tokens, hidden), dtype=x.dtype)
        recv_x_scales = (
            torch.zeros((num_tokens, x_scales.size(1)), dtype=x_scales.dtype) if x_scales is not None else None
        )
        recv_topk_idx = torch.full((num_tokens, num_topk), -1, dtype=topk_idx.dtype) if topk_idx is not None else None
        recv_topk_weights = (
            torch.zeros((num_tokens, num_topk), dtype=topk_weights.dtype) if topk_weights is not None else None
        )
        rank_prefix_matrix = (
            cached_rank_prefix_matrix
            if cached_rank_prefix_matrix is not None
            else torch.zeros((num_ranks, num_ranks), dtype=torch.int32)
        )
        channel_prefix_matrix = (
            cached_channel_prefix_matrix
            if cached_channel_prefix_matrix is not None
            else torch.zeros((num_ranks, num_channels), dtype=torch.int32)
        )
        return (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            [0] * (num_tokens_per_expert.numel() // num_ranks)
            if num_tokens_per_expert is not None and num_worst_tokens == 0
            else [],
            rank_prefix_matrix,
            channel_prefix_matrix,
            torch.zeros((num_ranks, num_channels), dtype=torch.int32),
            torch.zeros((num_tokens,), dtype=torch.int32),
            torch.zeros((local_num_tokens, num_ranks), dtype=torch.int32),
        )

    def intranode_combine(
        self,
        x,
        topk_weights,
        bias_0,
        bias_1,
        src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        send_head,
        num_aiv,
        num_max_nvl_chunked_send_tokens,
        num_max_nvl_chunked_recv_tokens,
        allocation_stream_id=None,
        num_max_tokens_per_rank=None,
        udma_peers_per_worker=4,
        udma_wqes_per_visit=8,
    ):
        self.udma_schedule = (udma_peers_per_worker, udma_wqes_per_visit)
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.allocation_stream_id = allocation_stream_id
        self.combine_args = (
            x,
            topk_weights,
            bias_0,
            bias_1,
            src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            send_head,
            num_aiv,
            num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens,
        )
        num_tokens = send_head.size(0)
        recv_x = torch.zeros((num_tokens, x.size(1)), dtype=x.dtype)
        recv_topk_weights = (
            torch.zeros((num_tokens, topk_weights.size(1)), dtype=topk_weights.dtype)
            if topk_weights is not None
            else None
        )
        return recv_x, recv_topk_weights

    def expand(
        self,
        x,
        x_scales,
        topk_idx,
        topk_weights,
        num_tokens_per_expert,
        expert_alignment,
        cached_num_expanded_tokens,
        cached_expert_offsets,
        cached_actual_expert_counts,
        cached_recv_to_expanded_idx,
        num_aiv,
        allocation_stream_id=None,
    ):
        self.allocation_stream_id = allocation_stream_id
        self.expand_args = (
            x,
            x_scales,
            topk_idx,
            topk_weights,
            num_tokens_per_expert,
            expert_alignment,
            cached_num_expanded_tokens,
            cached_expert_offsets,
            cached_actual_expert_counts,
            cached_recv_to_expanded_idx,
            num_aiv,
        )
        cached = cached_recv_to_expanded_idx is not None
        if cached:
            expert_offsets = cached_expert_offsets
            actual_counts = cached_actual_expert_counts
            mapping = cached_recv_to_expanded_idx
            num_expanded_tokens = cached_num_expanded_tokens
        else:
            counts = torch.tensor(num_tokens_per_expert, dtype=torch.int32)
            expert_offsets = torch.cat(
                (
                    torch.zeros((1,), dtype=torch.int32),
                    counts.cumsum(0, dtype=torch.int32),
                )
            )
            actual_counts = torch.zeros_like(counts)
            mapping = torch.full(topk_idx.shape, -1, dtype=torch.int32)
            tails = expert_offsets[:-1].clone()
            for token in range(topk_idx.size(0)):
                for slot in range(topk_idx.size(1)):
                    local_expert = int(topk_idx[token, slot])
                    if 0 <= local_expert < len(num_tokens_per_expert):
                        expanded_slot = int(tails[local_expert])
                        mapping[token, slot] = expanded_slot
                        tails[local_expert] += 1
                        actual_counts[local_expert] += 1
            num_expanded_tokens = sum(num_tokens_per_expert)

        expanded_x = torch.zeros((num_expanded_tokens, x.size(1)), dtype=x.dtype)
        expanded_scales = (
            torch.zeros(
                (num_expanded_tokens, x_scales.size(1)),
                dtype=x_scales.dtype,
            )
            if x_scales is not None
            else None
        )
        expanded_weights = (
            torch.zeros((num_expanded_tokens,), dtype=topk_weights.dtype) if topk_weights is not None else None
        )
        for token in range(mapping.size(0)):
            for slot in range(mapping.size(1)):
                expanded_slot = int(mapping[token, slot])
                if expanded_slot < 0:
                    continue
                expanded_x[expanded_slot] = x[token]
                if expanded_scales is not None:
                    expanded_scales[expanded_slot] = x_scales[token]
                if expanded_weights is not None:
                    expanded_weights[expanded_slot] = topk_weights[token, slot]
        return (
            expanded_x,
            expanded_scales,
            expanded_weights,
            expert_offsets,
            actual_counts,
            mapping,
        )

    def aggregate(
        self,
        x,
        topk_weights,
        expanded_values,
        recv_to_expanded_idx,
        num_aiv,
        allocation_stream_id=None,
    ):
        self.allocation_stream_id = allocation_stream_id
        self.aggregate_args = (
            x,
            topk_weights,
            expanded_values,
            recv_to_expanded_idx,
            num_aiv,
        )
        output = torch.zeros((recv_to_expanded_idx.size(0), x.size(1)), dtype=torch.float32)
        values = torch.zeros(recv_to_expanded_idx.shape, dtype=torch.float32) if expanded_values is not None else None
        for token in range(recv_to_expanded_idx.size(0)):
            for slot in range(recv_to_expanded_idx.size(1)):
                expanded_slot = int(recv_to_expanded_idx[token, slot])
                if expanded_slot < 0:
                    continue
                scale = float(topk_weights[expanded_slot]) if topk_weights is not None else 1.0
                output[token] += x[expanded_slot].float() * scale
                if values is not None:
                    values[token, slot] = expanded_values[expanded_slot]
        return output.to(x.dtype), values


class _FakeCommunicationStream:
    def __init__(self):
        self.synchronize_count = 0

    def synchronize(self):
        self.synchronize_count += 1


class _Group:
    def __init__(self, rank=0, size=2):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _Comm:
    def __init__(self, rank=0, size=2):
        self._rank = rank
        self._size = size

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size


def _make_buffer(**kwargs):
    parameters = {
        "group": _Group(),
        "num_nvl_bytes": 64 * 1024 * 1024,
        "explicitly_destroy": True,
    }
    parameters.update(kwargs)
    with (
        patch("deep_ep.backends.clos.buffer.get_native_class", return_value=_FakeBufferRuntime),
        patch(
            "deep_ep.backends.clos.buffer._create_communication_stream",
            return_value=_FakeCommunicationStream(),
        ),
        patch("deep_ep.backends.clos.buffer._prepare_shmem_bootstrap", return_value=([], "test")),
    ):
        return Buffer(**parameters)


class ConfigTest(unittest.TestCase):
    def test_shared_udma_tuning_validation(self):
        for peers in (1, 2, 4, 8):
            for visit in (4, 8, 16, 32):
                cfg = Config(32, udma_peers_per_worker=peers, udma_wqes_per_visit=visit)
                self.assertEqual((cfg.udma_peers_per_worker, cfg.udma_wqes_per_visit), (peers, visit))
        for field, invalid in (("udma_peers_per_worker", (0, 3, 64)), ("udma_wqes_per_visit", (0, 1, 3, 64))):
            for value in invalid:
                with self.assertRaises(ValueError):
                    Config(**{field: value})
            for value in (True, 4.0, "4"):
                with self.assertRaises(TypeError):
                    Config(**{field: value})

    def tearDown(self):
        Buffer.set_num_sms(32)

    def test_ascend_num_aiv_field(self):
        config = Config(num_aiv=_MAX_NUM_AIV)
        self.assertEqual(config.num_aiv, _MAX_NUM_AIV)
        self.assertEqual(config.num_sms, _MAX_NUM_AIV)
        self.assertFalse(config.uses_platform_defaults)

    def test_zero_sizes_select_platform_defaults(self):
        self.assertEqual(Config().num_sms, 0)
        self.assertTrue(Config().uses_platform_defaults)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            Config(num_aiv=-1)

    def test_set_num_sms_updates_recommended_configs(self):
        Buffer.set_num_sms(32)
        for num_ranks in (2, 3, 4, 8, 16, 1024):
            self.assertEqual(Buffer.get_dispatch_config(num_ranks), Config(32))
            self.assertEqual(Buffer.get_combine_config(num_ranks), Config(32))
        Buffer.set_num_sms(new_num_sms=18)
        self.assertEqual(Buffer.get_dispatch_config(2).num_sms, 18)
        Buffer.set_num_sms(_MAX_NUM_AIV)
        self.assertEqual(Buffer.get_dispatch_config(2).num_sms, _MAX_NUM_AIV)
        Buffer.set_num_sms(0)
        self.assertTrue(Buffer.get_dispatch_config(2).uses_platform_defaults)
        self.assertFalse(Buffer.is_sm90_compiled())

    def test_intranode_hints_do_not_switch_to_rdma_above_eight_ranks(self):
        config = Config(20, 6, 256, 6, 128)
        self.assertGreater(config.get_nvl_buffer_size_hint(4096, 1024), 0)
        self.assertEqual(config.get_rdma_buffer_size_hint(4096, 1024), 0)

    def test_invalid_num_sms_and_world_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            Buffer.set_num_sms(19)
        with self.assertRaises(ValueError):
            Buffer.set_num_sms(-2)
        with self.assertRaises(TypeError):
            Buffer.set_num_sms(True)
        with self.assertRaises(ValueError):
            Buffer.set_num_sms(_MAX_NUM_AIV + 2)
        with self.assertRaises(ValueError):
            Buffer.get_dispatch_config(1)
        with self.assertRaises(ValueError):
            Buffer.get_dispatch_config(1025)
        with self.assertRaises(TypeError):
            Buffer.get_combine_config(True)

    def test_nvl_size_hint_uses_default_or_explicit_token_capacity(self):
        config = Config(
            num_aiv=20,
            num_max_nvl_chunked_send_tokens=8,
            num_max_nvl_chunked_recv_tokens=128,
        )
        nvl_hint = config.get_nvl_buffer_size_hint(4096, 4)
        self.assertGreater(nvl_hint, 0)
        self.assertEqual(nvl_hint % (2 * 1024 * 1024), 0)
        self.assertEqual(
            nvl_hint,
            config.get_nvl_buffer_size_hint(4096, 4, num_tokens=8192),
        )
        self.assertGreater(
            nvl_hint,
            config.get_nvl_buffer_size_hint(4096, 4, num_tokens=1024),
        )
        self.assertGreater(
            config.get_nvl_buffer_size_hint(4096, 4, num_tokens=16384),
            nvl_hint,
        )
        self.assertEqual(config.get_rdma_buffer_size_hint(4096, 4), 0)
        self.assertGreater(config.get_nvl_buffer_size_hint(8192, 4), nvl_hint)
        self.assertGreater(config.get_nvl_buffer_size_hint(4096, 8), nvl_hint)

        legacy = Config(
            num_sms=20,
            num_max_shmem_chunked_send_tokens=8,
            num_max_shmem_chunked_recv_tokens=128,
        )
        self.assertEqual(legacy.num_aiv, 20)
        self.assertEqual(legacy.num_max_nvl_chunked_send_tokens, 8)
        self.assertEqual(legacy.num_max_nvl_chunked_recv_tokens, 128)
        self.assertEqual(legacy.num_max_shmem_chunked_send_tokens, 8)
        self.assertEqual(legacy.num_max_shmem_chunked_recv_tokens, 128)
        self.assertFalse(hasattr(legacy, "get_shmem_buffer_size_hint"))

        positional = Config(20, 8, 128, 6, 256)
        self.assertEqual(positional, config)

        with self.assertRaises(ValueError):
            Config(
                num_max_nvl_chunked_send_tokens=8,
                num_max_shmem_chunked_send_tokens=9,
            )

    def test_size_hint_and_reserved_rdma_values_are_validated(self):
        for invalid_hidden_bytes in (0, -1):
            with self.assertRaises(ValueError):
                Config().get_nvl_buffer_size_hint(invalid_hidden_bytes, 2)
            with self.assertRaises(ValueError):
                Config().get_rdma_buffer_size_hint(invalid_hidden_bytes, 2)
        with self.assertRaises(TypeError):
            Config().get_rdma_buffer_size_hint(True, 2)
        with self.assertRaises(ValueError):
            Config().get_rdma_buffer_size_hint(4096, 1025)
        for invalid_num_tokens in (0, -1):
            with self.assertRaises(ValueError):
                Config().get_nvl_buffer_size_hint(4096, 2, num_tokens=invalid_num_tokens)
        for invalid_num_tokens in (True, 1.5, "8192"):
            with self.assertRaises(TypeError):
                Config().get_nvl_buffer_size_hint(4096, 2, num_tokens=invalid_num_tokens)
        with self.assertRaises(ValueError):
            Config(num_aiv=19)
        with self.assertRaises(ValueError):
            Config(num_aiv=_MAX_NUM_AIV + 2)
        with self.assertRaises(ValueError):
            Config(num_max_nvl_chunked_send_tokens=256)
        with self.assertRaises(TypeError):
            Config(num_max_rdma_chunked_send_tokens=True)
        with self.assertRaises(ValueError):
            Config(
                num_max_rdma_chunked_send_tokens=65,
                num_max_rdma_chunked_recv_tokens=128,
            )
        with self.assertRaises(ValueError):
            Config(num_aiv=20, num_sms=24)

    def test_size_hint_covers_shared_udma_workspace(self):
        # Native layout byte counts for BF16 H6144, K8 (int64 indices),
        # six experts/rank, ordinary dispatch, without FP8 scales.
        # These include notify, payload and synchronization regions before
        # SHMEM allocation rounding; do not derive them from the Python hint.
        cases = (
            (2, 32, 129, 3198976),
            (8, 32, 129, 12792320),
            (16, 32, 129, 25585152),
            (32, 32, 1, 430592),
            (34, 32, 1, 458752),
            (48, 32, 1, 648704),
            (64, 16, 1, 856576),
            (64, 18, 1, 859648),
            (64, 32, 1, 868864),
            (64, 48, 1, 881152),
            (64, 64, 1, 893440),
            (64, 32, 129, 102351360),
            (128, 32, 1, 1769472),
            (128, 32, 129, 204734464),
        )
        for ranks, aiv, tokens, required in cases:
            if aiv > _MAX_NUM_AIV:
                continue
            with self.subTest(ranks=ranks, aiv=aiv, tokens=tokens):
                hint = Config(aiv).get_nvl_buffer_size_hint(12288, ranks, tokens)
                self.assertGreaterEqual(hint, required)
                self.assertEqual(hint % (2 * 1024**2), 0)
        self.assertEqual(
            Config().get_nvl_buffer_size_hint(12288, 64, 1),
            Config(32).get_nvl_buffer_size_hint(12288, 64, 1),
        )


class DtypeTest(unittest.TestCase):
    def test_topk_idx_t_matches_current_ascend_kernel(self):
        expected = torch.int32 if int(os.environ.get("EP_NUM_TOPK_IDX_BITS", "64")) == 32 else torch.int64
        self.assertIs(topk_idx_t, expected)

    def test_deep_ep_package_exports_v1_api(self):
        from deep_ep._clos_adapter import Buffer as ClosBuffer

        self.assertTrue(issubclass(ClosBuffer, Buffer))
        self.assertEqual(deep_ep.Config(num_compute_units=32).num_aiv, Config(32).num_aiv)
        self.assertIs(deep_ep.topk_idx_t, topk_idx_t)
        self.assertIs(DeepEPBuffer, deep_ep.Buffer)
        self.assertTrue(callable(DeepEPEventHandle.current_stream_wait))

    def test_blackwell_compatibility_helper_is_false_on_npu(self):
        self.assertFalse(buffer_module._is_blackwell())
        self.assertFalse(Buffer._is_blackwell())
        self.assertIs(deep_ep.buffer, deep_ep_buffer_module)
        self.assertFalse(deep_ep.buffer._is_blackwell())


class _FakeEvent:
    def __init__(self):
        self.recorded_stream = None
        self.waited_streams = []

    def record(self, stream):
        self.recorded_stream = stream

    def wait(self, stream):
        self.waited_streams.append(stream)


class _FakeNpu:
    Event = _FakeEvent

    def __init__(self):
        self.stream = object()

    def current_stream(self):
        return self.stream


class EventTest(unittest.TestCase):
    def setUp(self):
        cpu_npu = patch.object(torch, "npu", SimpleNamespace(), create=True)
        cpu_npu.start()
        self.addCleanup(cpu_npu.stop)
        self.npu = _FakeNpu()
        self.npu_patch = patch("deep_ep.backends.clos.event._get_npu_api", return_value=self.npu)
        self.npu_patch.start()

    def tearDown(self):
        self.npu_patch.stop()

    def test_event_handle_records_and_current_stream_waits(self):
        handle = EventHandle()
        self.assertIs(handle.event.recorded_stream, self.npu.stream)

        waiting_stream = object()
        self.npu.stream = waiting_stream
        handle.current_stream_wait()
        self.assertEqual(handle.event.waited_streams, [waiting_stream])

    def test_event_handle_can_record_an_explicit_stream(self):
        communication_stream = object()
        handle = EventHandle(communication_stream)
        self.assertIs(handle.event.recorded_stream, communication_stream)

    def test_try_record_event_propagates_all_event_errors(self):
        for error_type in (ImportError, AttributeError, TypeError, RuntimeError):
            error = error_type("event record failed")
            with self.subTest(error=error_type), patch.object(buffer_module, "EventHandle", side_effect=error):
                with self.assertRaises(error_type) as raised:
                    buffer_module._try_record_event(None)
                self.assertIs(raised.exception, error)

    def test_dependency_event_failure_prevents_native_submission(self):
        for error_type in (ImportError, AttributeError, TypeError, RuntimeError):
            buffer = _make_buffer()
            try:
                with (
                    self.subTest(error=error_type),
                    patch.object(buffer_module, "EventHandle", side_effect=error_type("dependency failed")),
                    patch.object(buffer._native, "get_dispatch_layout") as launch,
                ):
                    with self.assertRaisesRegex(error_type, "dependency failed"):
                        buffer.get_dispatch_layout(torch.zeros((1, 1), dtype=topk_idx_t), 2)
                    launch.assert_not_called()
                    self.assertIsNone(buffer._failed_event_tensors)
            finally:
                buffer.destroy()

    def test_dependency_wait_failure_prevents_native_submission(self):
        buffer = _make_buffer()
        try:
            with (
                patch.object(_FakeEvent, "wait", side_effect=RuntimeError("wait failed")),
                patch.object(buffer._native, "get_dispatch_layout") as launch,
            ):
                with self.assertRaisesRegex(RuntimeError, "wait failed"):
                    buffer.get_dispatch_layout(torch.zeros((1, 1), dtype=topk_idx_t), 2)
                launch.assert_not_called()
        finally:
            buffer.destroy()

    def test_completion_event_failure_never_returns_success(self):
        for error_type in (ImportError, AttributeError, TypeError, RuntimeError):
            for async_finish in (False, True):
                buffer = _make_buffer()
                try:
                    with (
                        self.subTest(error=error_type, async_finish=async_finish),
                        patch.object(self.npu, "Event", side_effect=[_FakeEvent(), error_type("completion failed")]),
                        patch.object(
                            buffer._native, "get_dispatch_layout", wraps=buffer._native.get_dispatch_layout
                        ) as launch,
                    ):
                        with self.assertRaisesRegex(error_type, "completion failed"):
                            buffer.get_dispatch_layout(
                                torch.zeros((1, 1), dtype=topk_idx_t), 2, async_finish=async_finish
                            )
                        launch.assert_called_once()
                        self.assertIsNotNone(buffer._failed_event_tensors)
                        with self.assertRaisesRegex(RuntimeError, "failed completion event"):
                            buffer.get_dispatch_layout(torch.zeros((1, 1), dtype=topk_idx_t), 2)
                        launch.assert_called_once()
                finally:
                    buffer.destroy()

    def test_failed_completion_retains_outputs_until_successful_destroy(self):
        buffer = _make_buffer()
        native = buffer.native
        payload = torch.ones(1)
        payload_ref = ref(payload)
        try:
            with patch.object(_FakeEvent, "wait", side_effect=TypeError("completion wait failed")):
                with self.assertRaisesRegex(TypeError, "completion wait failed"):
                    buffer._finish_comm_event(False, (payload,))
            del payload
            gc.collect()
            self.assertIsNotNone(payload_ref())
            with patch.object(buffer._communication_stream, "synchronize", side_effect=RuntimeError("drain failed")):
                with self.assertRaisesRegex(RuntimeError, "drain failed"):
                    buffer.destroy()
            self.assertTrue(native.initialized)
            self.assertIsNotNone(payload_ref())
            buffer.destroy()
            gc.collect()
            self.assertFalse(native.initialized)
            self.assertIsNone(payload_ref())
        finally:
            buffer.destroy()

    def test_record_failure_is_propagated_before_and_after_submission(self):
        for fail_on_completion in (False, True):
            buffer = _make_buffer()
            error = TypeError("event record failed")
            records = [None, error] if fail_on_completion else [error]
            try:
                with (
                    self.subTest(completion=fail_on_completion),
                    patch.object(_FakeEvent, "record", side_effect=records),
                    patch.object(
                        buffer._native,
                        "get_dispatch_layout",
                        wraps=buffer._native.get_dispatch_layout,
                    ) as launch,
                ):
                    with self.assertRaises(TypeError) as raised:
                        buffer.get_dispatch_layout(torch.zeros((1, 1), dtype=topk_idx_t), 2)
                    self.assertIs(raised.exception, error)
                    self.assertEqual(launch.call_count, int(fail_on_completion))
                    self.assertEqual(buffer._failed_event_tensors is not None, fail_on_completion)
            finally:
                buffer.destroy()

    def test_successful_completion_keeps_sync_and_async_semantics(self):
        for async_finish in (False, True):
            buffer = _make_buffer()
            payload = torch.ones(1)
            try:
                with self.subTest(async_finish=async_finish), patch.object(_FakeEvent, "wait") as wait:
                    overlap = buffer._finish_comm_event(async_finish, (payload,))
                    self.assertIsNone(buffer._failed_event_tensors)
                    self.assertEqual(wait.call_count, int(not async_finish))
                    if async_finish:
                        self.assertIsInstance(overlap.event, EventHandle)
                        self.assertIs(overlap.extra_tensors[0], payload)
                        self.assertIs(
                            overlap.event.event.recorded_stream,
                            buffer._communication_stream,
                        )
                    else:
                        self.assertIsNone(overlap.event)
                        self.assertIsNone(overlap.extra_tensors)
            finally:
                buffer.destroy()

    def test_event_failure_finalizer_retains_native_stream_and_outputs(self):
        orphaned = []
        with patch.object(buffer_module, "_ORPHANED_NATIVE_RESOURCES", orphaned):
            buffer = _make_buffer()
            native = buffer.native
            stream = buffer._communication_stream
            payload = torch.ones(1)
            payload_ref = ref(payload)
            with patch.object(buffer_module, "EventHandle", side_effect=RuntimeError("record failed")):
                with self.assertRaisesRegex(RuntimeError, "record failed"):
                    buffer._finish_comm_event(True, (payload,))
            buffer_ref = ref(buffer)
            del payload
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                del buffer
                gc.collect()
            self.assertIsNone(buffer_ref())
            self.assertEqual(len(orphaned), 1)
            self.assertIs(orphaned[0][0], native)
            self.assertIs(orphaned[0][1], stream)
            self.assertIsNotNone(payload_ref())
            self.assertTrue(native.initialized)
            self.assertEqual(stream.synchronize_count, 0)
            self.assertTrue(any(item.category is ResourceWarning for item in caught))
            orphaned.clear()  # CPU-only fake resources; no device work exists.

    def test_comm_stream_waits_for_default_or_explicit_dependency(self):
        default_dependency = SimpleNamespace(current_stream_wait=lambda: None)
        with (
            patch.object(
                buffer_module,
                "_try_record_event",
                return_value=default_dependency,
            ) as record_event,
            patch.object(
                default_dependency,
                "current_stream_wait",
            ) as default_wait,
        ):
            with buffer_module._use_comm_stream_after_dependency(object(), None):
                pass
        record_event.assert_called_once_with(None)
        default_wait.assert_called_once_with()

        explicit_dependency = SimpleNamespace(current_stream_wait=lambda: None)
        previous_event = SimpleNamespace(event=explicit_dependency)
        with (
            patch.object(
                buffer_module,
                "_try_record_event",
            ) as record_event,
            patch.object(
                explicit_dependency,
                "current_stream_wait",
            ) as explicit_wait,
        ):
            with buffer_module._use_comm_stream_after_dependency(object(), previous_event):
                pass
        record_event.assert_not_called()
        explicit_wait.assert_called_once_with()

    def test_operation_type_error_is_not_swallowed_by_stream_context(self):
        with patch.object(torch, "npu", SimpleNamespace(stream=lambda _: nullcontext())):
            with self.assertRaisesRegex(TypeError, "operation failed"):
                with buffer_module._use_comm_stream(object()):
                    raise TypeError("operation failed")

    def test_event_overlap_wait_hook_and_release(self):
        overlap = EventOverlap(EventHandle(), extra_tensors=(object(),))
        calls = []
        overlap.register_hook_after_wait(hook_after_wait=lambda: calls.append("waited"))
        overlap.current_stream_wait(release_handle=True)
        self.assertEqual(calls, ["waited"])
        self.assertIsNone(overlap.event)
        self.assertIsNone(overlap.extra_tensors)
        self.assertIsNone(overlap.hook_after_wait)
        with self.assertRaises(RuntimeError):
            overlap.current_stream_wait()

    def test_event_overlap_context_manager_waits(self):
        overlap = EventOverlap(EventHandle())
        event = overlap.event.event
        with overlap(release_handle=True) as entered:
            self.assertIs(entered, overlap)
            self.assertEqual(event.waited_streams, [])
        self.assertEqual(event.waited_streams, [self.npu.stream])
        self.assertIsNone(overlap.event)

    def test_event_overlap_validates_event_hook_and_flags(self):
        raw_event = object()
        self.assertIs(EventOverlap(raw_event).event, raw_event)
        overlap = EventOverlap(EventHandle())
        with self.assertRaises(TypeError):
            overlap.register_hook_after_wait(None)
        overlap.register_hook_after_wait(lambda: None)
        with self.assertRaises(RuntimeError):
            overlap.register_hook_after_wait(lambda: None)
        with self.assertRaises(TypeError):
            overlap.current_stream_wait(release_handle=1)

    def test_buffer_capture_returns_event_overlap(self):
        overlap = Buffer.capture()
        self.assertIsInstance(overlap, EventOverlap)
        self.assertIsInstance(overlap.event, EventHandle)
        self.assertIs(overlap.event.event.recorded_stream, self.npu.stream)


class UtilsTest(unittest.TestCase):
    def test_align_up(self):
        self.assertEqual(align_up(0, 512), 0)
        self.assertEqual(align_up(513, 512), 1024)
        with self.assertRaises(ValueError):
            align_up(-1, 512)

    def test_fp8_round_trip_supports_unaligned_hidden(self):
        torch.manual_seed(7)
        x = torch.randn((3, 130), dtype=torch.bfloat16)
        x_fp8, scales = per_token_cast_to_fp8(x)
        restored = per_token_cast_back(x_fp8, scales)
        self.assertEqual(x_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(x_fp8.shape), (3, 130))
        self.assertEqual(tuple(scales.shape), (3, 2))
        self.assertEqual(restored.dtype, torch.bfloat16)
        self.assertLess(calc_diff(restored, x), 1e-3)

    def test_deep_ep_math_compatibility_helpers(self):
        self.assertEqual(align(129, 128), 256)
        self.assertEqual(safe_div(0, 0), 0)
        with self.assertRaises(ZeroDivisionError):
            safe_div(1, 0)

        values = torch.tensor([[1, 1, 3, -1], [2, 0, 2, 0]])
        inplace_unique(values, 4)
        torch.testing.assert_close(
            values,
            torch.tensor([[3, 1, -1, -1], [2, 0, -1, -1]]),
        )

        scores = torch.arange(8, dtype=torch.float32).view(2, 4)
        group_idx = torch.tensor([[0], [1]])
        grouped = create_grouped_scores(scores, group_idx, 2)
        torch.testing.assert_close(
            grouped,
            torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 6.0, 7.0]]),
        )
        self.assertEqual(count_bytes(scores, (group_idx, None)), 48)


class EnvsCompatibilityTest(unittest.TestCase):
    def test_init_dist_matches_deep_ep_api_and_uses_hccl(self):
        self.assertEqual(
            list(inspect.signature(envs.init_dist).parameters),
            ["local_rank", "num_local_ranks", "seed"],
        )

        fake_group = object()
        signature = SimpleNamespace(parameters={})
        with (
            patch.dict(
                envs.os.environ,
                {
                    "MASTER_ADDR": "10.0.0.1",
                    "MASTER_PORT": "9000",
                    "WORLD_SIZE": "2",
                    "RANK": "1",
                },
                clear=False,
            ),
            patch.object(envs.dist, "is_initialized", return_value=False),
            patch.object(envs.dist, "init_process_group") as init_group,
            patch.object(envs.dist, "new_group", return_value=fake_group),
            patch.object(envs.dist, "get_rank", return_value=5),
            patch.object(envs.inspect, "signature", return_value=signature),
            patch.object(torch.npu, "set_device") as set_device,
            patch.object(torch, "set_default_dtype") as set_dtype,
            patch.object(torch, "set_default_device") as set_default_device,
        ):
            rank, world_size, group = envs.init_dist(1, 4, seed=11)

        self.assertEqual((rank, world_size, group), (5, 8, fake_group))
        init_group.assert_called_once_with(
            backend="hccl",
            init_method="tcp://10.0.0.1:9000",
            world_size=8,
            rank=5,
        )
        set_device.assert_called_once_with(1)
        set_dtype.assert_called_once_with(torch.bfloat16)
        set_default_device.assert_called_once_with("npu")
        self.assertEqual(envs.get_local_seed(), 16)
        self.assertEqual(envs.get_global_seed(), 11)


class BufferConstructionTest(unittest.TestCase):
    def test_bootstrap_mode_is_selected_only_by_constructor_argument(self):
        # Exercise the real constructor -> bootstrap -> native argument path.
        # Only the distributed exchange and native runtime are CPU substitutes.
        uid = [1, 2, 3]
        explicit = "tcp://127.0.0.1:30001"
        for inherited in (None, "tcp://127.0.0.1:30000", "", "invalid"):
            environment = {} if inherited is None else {"ASCEND_DEEPEP_SHMEM_IP_PORT": inherited}
            for kwargs in ({}, {"shmem_ip_port": None}, {"shmem_ip_port": explicit}):
                for transport in ("mte", "udma"):
                    group = _Group()
                    native = Mock()
                    native.get_shmem_bootstrap.return_value = (0, [])
                    native.get_shmem_unique_id.return_value = uid

                    def gather(values, value, group):
                        values[:] = [value] * 2

                    with (
                        self.subTest(env=inherited, kwargs=kwargs, transport=transport),
                        patch.dict(os.environ, environment, clear=True),
                        patch("deep_ep.backends.clos.buffer.load_native_module", return_value=native),
                        patch("deep_ep.backends.clos.buffer.get_native_class", return_value=_FakeBufferRuntime),
                        patch(
                            "deep_ep.backends.clos.buffer._create_communication_stream",
                            return_value=_FakeCommunicationStream(),
                        ),
                        patch("torch.distributed.all_gather_object", side_effect=gather) as exchange,
                    ):
                        buffer = Buffer(
                            group,
                            num_nvl_bytes=64 * 1024**2,
                            explicitly_destroy=True,
                            transport=transport,
                            **kwargs,
                        )
                        try:
                            endpoint = kwargs.get("shmem_ip_port")
                            self.assertEqual(buffer.native.args[5], endpoint or "")
                            self.assertEqual(buffer.native.unique_id, [] if endpoint else uid)
                            self.assertEqual(native.get_shmem_unique_id.call_count, int(endpoint is None))
                            self.assertEqual(exchange.call_count, 1 if endpoint else 2)
                            self.assertTrue(all(call.kwargs["group"] is group for call in exchange.call_args_list))
                            self.assertEqual(os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT"), inherited)
                        finally:
                            buffer.destroy()

    def test_invalid_explicit_endpoint_does_not_fall_back_to_environment(self):
        for endpoint in ("", 123, False):
            with (
                self.subTest(endpoint=endpoint),
                patch.dict(os.environ, {"ASCEND_DEEPEP_SHMEM_IP_PORT": "tcp://127.0.0.1:30000"}),
                patch("deep_ep.backends.clos.buffer._prepare_shmem_bootstrap") as prepare,
                self.assertRaises(ValueError),
            ):
                Buffer(_Group(), num_nvl_bytes=64 * 1024**2, shmem_ip_port=endpoint)
            prepare.assert_not_called()

    def test_buffer_constructor_and_runtime_resources(self):
        buffer = _make_buffer()
        try:
            self.assertTrue(buffer.is_available())
            self.assertEqual(buffer.comm_meta_ptr, 0xA30000)
            self.assertEqual(buffer.native.args[:3], (0, 2, 64 * 1024 * 1024))
            self.assertIs(buffer.runtime, buffer.native)
            self.assertEqual(buffer.num_bytes, 64 * 1024 * 1024)
            self.assertEqual(buffer.num_nvl_bytes, 64 * 1024 * 1024)
            self.assertEqual(buffer.num_rdma_bytes, 0)
            self.assertEqual(buffer.transport, "auto")
            self.assertFalse(buffer.low_latency_mode)
            self.assertEqual(buffer.group_size, 2)
            self.assertIs(buffer.get_comm_stream(), buffer.communication_stream)
            buffer.barrier()
            self.assertEqual(buffer.native.barrier_count, 1)
            communication_stream = buffer.get_comm_stream()
        finally:
            buffer.destroy()
        self.assertFalse(buffer.is_available())
        self.assertEqual(communication_stream.synchronize_count, 1)
        with self.assertRaises(RuntimeError):
            buffer.get_comm_stream()

    def test_process_group_constructor(self):
        parameters = {
            "group": _Group(rank=1),
            "num_nvl_bytes": 64 * 1024 * 1024,
            "explicitly_destroy": True,
        }
        with (
            patch("deep_ep.backends.clos.buffer.get_native_class", return_value=_FakeBufferRuntime),
            patch(
                "deep_ep.backends.clos.buffer._create_communication_stream",
                return_value=_FakeCommunicationStream(),
            ),
            patch("deep_ep.backends.clos.buffer._prepare_shmem_bootstrap", return_value=([], "test")),
        ):
            buffer = Buffer(**parameters)
        try:
            self.assertEqual(buffer.rank, 1)
            self.assertEqual(buffer.num_ranks, 2)
            self.assertEqual(buffer.group_size, 2)
            self.assertIs(buffer.group, parameters["group"])
        finally:
            buffer.destroy()

    def test_buffer_constraints_are_rejected_before_native_init(self):
        with self.assertRaises(ValueError):
            _make_buffer(group=_Group(size=1))
        with self.assertRaises(ValueError):
            _make_buffer(group=_Group(size=1025))
        with self.assertRaises(ValueError):
            _make_buffer(num_nvl_bytes=-1)
        with self.assertRaises(ValueError):
            _make_buffer(num_rdma_bytes=-1)
        with self.assertRaises(ValueError):
            _make_buffer(num_bytes=-1)
        with self.assertRaises(TypeError):
            _make_buffer(num_bytes=True)
        with self.assertRaises(TypeError):
            _make_buffer(explicitly_destroy=1)
        with self.assertRaises(TypeError):
            _make_buffer(transport=1)
        with self.assertRaises(ValueError):
            _make_buffer(transport="unsupported")
        with self.assertRaises(TypeError):
            _make_buffer(low_latency_mode=1)
        with self.assertRaises(TypeError):
            _make_buffer(use_1g_huge_pages=1)
        with self.assertRaises(TypeError):
            _make_buffer(num_qps_per_rank=True)
        with self.assertRaises(ValueError):
            _make_buffer(num_qps_per_rank=0)
        for name in (
            "allow_nvlink_for_low_latency_mode",
            "allow_mnnvl",
            "enable_shrink",
        ):
            with self.subTest(name=name), self.assertRaises(TypeError):
                _make_buffer(**{name: 1})

    def test_transport_and_low_latency_mode_reach_native_runtime(self):
        buffer = _make_buffer(transport="UDMA", low_latency_mode=True, use_1g_huge_pages=True)
        try:
            self.assertEqual(buffer.transport, "udma")
            self.assertTrue(buffer.low_latency_mode)
            self.assertEqual(buffer.native.args[3:5], ("udma", True))
            self.assertTrue(buffer.use_1g_huge_pages)
            self.assertTrue(buffer.native.args[6])
        finally:
            buffer.destroy()

    def test_constructor_signature_preserves_v1_and_optional_capacity(self):
        self.assertEqual(
            tuple(inspect.signature(Buffer).parameters),
            (
                "group",
                "num_nvl_bytes",
                "num_rdma_bytes",
                "low_latency_mode",
                "num_qps_per_rank",
                "allow_nvlink_for_low_latency_mode",
                "allow_mnnvl",
                "explicitly_destroy",
                "enable_shrink",
                "comm",
                "num_bytes",
                "transport",
                "udma_schedule",
                "use_1g_huge_pages",
                "shmem_ip_port",
                "num_max_tokens_per_rank",
            ),
        )
        with self.assertRaises(TypeError):
            _make_buffer(hidden=128)

    def test_num_bytes_keyword_remains_supported_as_alias(self):
        buffer = _make_buffer(num_nvl_bytes=0, num_bytes=32 * 1024 * 1024)
        try:
            self.assertEqual(buffer.num_bytes, 32 * 1024 * 1024)
            self.assertEqual(buffer.num_nvl_bytes, 32 * 1024 * 1024)
            self.assertEqual(buffer.native.args[:3], (0, 2, 32 * 1024 * 1024))
        finally:
            buffer.destroy()

        with self.assertRaises(ValueError):
            _make_buffer(num_nvl_bytes=64, num_bytes=32)

    def test_token_schedule_is_explicit_and_immutable(self):
        for schedule in ("peer", "token"):
            buffer = _make_buffer(transport="udma", udma_schedule=schedule)
            try:
                self.assertEqual(buffer.udma_schedule, schedule)
                self.assertEqual(buffer.native.v1_token_schedule, schedule == "token")
                with self.assertRaises(AttributeError):
                    buffer.udma_schedule = "peer"
            finally:
                buffer.destroy()
        for options in ({"udma_schedule": "token"}, {"udma_schedule": "unknown"}):
            with self.assertRaises(ValueError):
                _make_buffer(**options)

    def test_mpi_comm_constructor_and_source_validation(self):
        comm = _Comm(rank=1, size=4)
        buffer = _make_buffer(group=None, comm=comm)
        try:
            self.assertEqual((buffer.rank, buffer.num_ranks), (1, 4))
            self.assertIs(buffer.group, comm)
        finally:
            buffer.destroy()

        with self.assertRaises(ValueError):
            _make_buffer(comm=_Comm())
        with self.assertRaises(ValueError):
            _make_buffer(group=None, comm=None)
        with self.assertRaises(TypeError):
            _make_buffer(group=None, comm=object())

    def test_upstream_buffer_options_are_retained(self):
        buffer = _make_buffer(
            num_rdma_bytes=4096,
            num_qps_per_rank=8,
            allow_nvlink_for_low_latency_mode=False,
            allow_mnnvl=True,
            enable_shrink=True,
        )
        try:
            self.assertEqual(buffer.num_rdma_bytes, 4096)
            self.assertEqual(buffer.num_qps_per_rank, 8)
            self.assertFalse(buffer.allow_nvlink_for_low_latency_mode)
            self.assertTrue(buffer.allow_mnnvl)
            self.assertTrue(buffer.enable_shrink)
        finally:
            buffer.destroy()

    def test_finalizer_does_not_run_collective_destroy(self):
        buffer = _make_buffer(explicitly_destroy=False)
        native = buffer.native
        buffer_ref = ref(buffer)
        del buffer
        gc.collect()
        self.assertIsNone(buffer_ref())
        self.assertTrue(native.initialized)

    def test_finalizer_retains_active_slot_resources(self):
        orphaned = []
        with patch("deep_ep.backends.clos.buffer._ORPHANED_NATIVE_RESOURCES", orphaned):
            buffer = _make_buffer(explicitly_destroy=False)
            native = buffer.native
            slot = buffer.acquire_slot()
            buffer_ref = ref(buffer)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                del buffer
                gc.collect()

            self.assertIsNone(buffer_ref())
            self.assertEqual(orphaned, [(native, slot)])
            self.assertTrue(native.initialized)
            self.assertTrue(any(item.category is ResourceWarning for item in caught))

            slot.release()
            orphaned.clear()


class NativeLoadingTest(unittest.TestCase):
    def test_missing_native_class_runtime_error_becomes_import_error(self):
        class MissingClassNamespace:
            def __getattr__(self, name):
                raise RuntimeError(f"class {name} is not registered")

        fake_classes = SimpleNamespace(AscendDeepEP=MissingClassNamespace())
        fake_torch = SimpleNamespace(classes=fake_classes)
        with self.assertRaisesRegex(ImportError, "MissingClass"):
            _resolve_native_class(fake_torch, "MissingClass")


class SlotAndHandleTest(unittest.TestCase):
    def test_shared_udma_config_reaches_all_native_calls(self):
        x = torch.zeros((3, 64), dtype=torch.bfloat16)
        idx = torch.zeros((3, 2), dtype=topk_idx_t)
        counts, _, experts, membership, _ = self.buffer.get_dispatch_layout(idx, 4)
        self.assertTrue(self.buffer.native.v1_shared_jetty)
        for peers, visit in ((1, 8), (2, 4), (8, 32)):
            config = Config(32, udma_peers_per_worker=peers, udma_wqes_per_visit=visit)
            recv, _, _, _, handle, _ = self.buffer.dispatch(
                x,
                topk_idx=idx,
                num_tokens_per_rank=counts,
                num_tokens_per_expert=experts,
                is_token_in_rank=membership,
                config=config,
            )
            self.assertEqual(self.buffer.native.udma_schedule, (peers, visit))
            recv, _, _, _, _, _ = self.buffer.dispatch(x, handle=handle, config=config)
            self.assertEqual(self.buffer.native.udma_schedule, (peers, visit))
            self.buffer.combine(recv, handle, config=config)
            self.assertEqual(self.buffer.native.udma_schedule, (peers, visit))

    def test_v1_handle_retains_owner_and_rejects_other_buffer(self):
        other = _make_buffer()
        try:
            x = torch.zeros((3, 64), dtype=torch.bfloat16)
            idx = torch.zeros((3, 2), dtype=topk_idx_t)
            counts, _, experts, member, _ = self.buffer.get_dispatch_layout(idx, 4)
            recv, _, _, _, handle, _ = self.buffer.dispatch(
                x, topk_idx=idx, num_tokens_per_rank=counts, num_tokens_per_expert=experts, is_token_in_rank=member
            )
            self.assertIsInstance(handle, tuple)
            self.assertEqual(len(handle), 6)
            self.assertIs(handle._buffer_owner, self.buffer)
            with self.assertRaisesRegex(ValueError, "different Buffer"):
                other.dispatch(x, handle=handle)
            with self.assertRaisesRegex(ValueError, "different Buffer"):
                other.combine(recv, handle)
        finally:
            other.destroy()

    def test_dispatch_private_plan_cached_and_legacy_handle(self):
        buffer = _make_buffer(transport="udma")
        try:
            x = torch.zeros((17, 64), dtype=torch.bfloat16)
            idx = torch.zeros((17, 2), dtype=topk_idx_t)
            nr, _, ne, member, _ = buffer.get_dispatch_layout(idx, 4)
            _, _, _, _, handle, _ = buffer.dispatch(
                x, topk_idx=idx, num_tokens_per_rank=nr, num_tokens_per_expert=ne, is_token_in_rank=member
            )
            self.assertEqual(len(handle), 6)
            self.assertIsNotNone(handle._dispatch_plan)
            plan = handle._dispatch_plan
            # Other dispatches must not replace the handle's routing plan.
            _, _, _, _, second, _ = buffer.dispatch(
                x, topk_idx=idx, num_tokens_per_rank=nr, num_tokens_per_expert=ne, is_token_in_rank=member
            )
            self.assertIsNot(second._dispatch_plan, plan)
            with (
                patch.object(buffer_module, "_record_recv_outputs_on_current_stream") as record,
                patch.object(buffer, "_finish_comm_event", wraps=buffer._finish_comm_event) as finish,
            ):
                buffer.dispatch(x, handle=handle, async_finish=True)
                self.assertTrue(any(plan is t for call in record.call_args_list for t in call.args))
                self.assertTrue(any(plan is t for t in finish.call_args.args[1]))
            self.assertIs(buffer.native.cached_dispatch_plan, plan)
            buffer.dispatch(x, handle=tuple(handle))
            self.assertIsNone(buffer.native.cached_dispatch_plan)
            self.assertIs(handle._dispatch_plan, plan)
        finally:
            buffer.destroy()

    def test_unsafe_fixed_capacity_rejected_before_native_launch(self):
        buffer = _make_buffer(num_max_tokens_per_rank=9)
        try:
            with patch.object(buffer.native, "intranode_dispatch_with_plan") as launch:
                for tokens in (0, 3, 9):
                    for capacity in (1, 17, 19):
                        with self.assertRaisesRegex(ValueError, "group_size.*capacity"):
                            buffer.dispatch(torch.zeros((tokens, 64)), num_worst_tokens=capacity)
                launch.assert_not_called()
        finally:
            buffer.destroy()

    def test_fixed_capacity_survives_cached_dispatch_and_combine(self):
        buffer = _make_buffer(num_max_tokens_per_rank=9)
        try:
            for tokens in (3, 0):
                x = torch.zeros((tokens, 64), dtype=torch.bfloat16)
                idx = torch.zeros((tokens, 2), dtype=torch.int64)
                per_rank, _, per_expert, membership, _ = buffer.get_dispatch_layout(idx, 4)
                recv, _, _, _, handle, _ = buffer.dispatch(
                    x,
                    topk_idx=idx,
                    num_tokens_per_rank=per_rank,
                    num_tokens_per_expert=per_expert,
                    is_token_in_rank=membership,
                    num_worst_tokens=18,
                )
                buffer.dispatch(x, handle=handle)
                self.assertEqual(buffer.native.num_max_tokens_per_rank, 9)
                output, _, _ = buffer.combine(recv, handle)
                self.assertEqual(tuple(output.shape), (tokens, 64))
                self.assertEqual(buffer.native.num_max_tokens_per_rank, 9)
            with self.assertRaises(AttributeError):
                buffer.num_max_tokens_per_rank = 10
            with self.assertRaisesRegex(ValueError, "exceeds"):
                buffer.dispatch(torch.zeros((10, 64)))
        finally:
            buffer.destroy()

    def test_graph_dynamic_dispatch_rejected_before_native_launch(self):
        with patch.object(buffer_module, "_is_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "num_worst_tokens"):
                self.buffer.dispatch(torch.zeros((3, 64)))
            with self.assertRaisesRegex(RuntimeError, "outside"):
                self.buffer.destroy()

    def setUp(self):
        cpu_npu = patch.object(torch, "npu", SimpleNamespace(), create=True)
        cpu_npu.start()
        self.addCleanup(cpu_npu.stop)
        self.event_patch = patch.object(
            buffer_module,
            "_try_record_event",
            side_effect=lambda stream: SimpleNamespace(current_stream_wait=lambda: None),
        )
        self.event_patch.start()
        self.addCleanup(self.event_patch.stop)
        self.buffer = _make_buffer()

    def tearDown(self):
        if self.buffer.slot.state is SlotState.ACQUIRED:
            self.buffer.slot.release()
        if self.buffer.slot.state is SlotState.DISPATCH_READY:
            self.buffer.slot.transition(SlotState.RELEASABLE)
            self.buffer.slot.release()
        self.buffer.destroy()

    def test_state_machine_and_dispatch_handle(self):
        slot = self.buffer.acquire_slot()
        generation = slot.generation
        tensor_reference = object()
        slot.hold(tensor_reference)
        slot.transition(SlotState.DISPATCH_IN_FLIGHT)
        slot.transition(SlotState.DISPATCH_READY)
        put_offset = object()
        handle = self.buffer.create_handle(slot, put_offset)
        self.assertIsInstance(handle, Handle)
        self.assertIs(handle.put_offset, put_offset)
        self.assertEqual(handle.generation, generation)
        self.buffer.validate_handle(handle)
        slot.transition(SlotState.COMBINE_IN_FLIGHT)
        slot.transition(SlotState.RELEASABLE)
        slot.release()
        with self.assertRaises(RuntimeError):
            handle.validate(self.buffer)
        with self.assertRaises(RuntimeError):
            _ = handle.put_offset

    def test_old_handle_cannot_access_reused_named_slots(self):
        slot = self.buffer.acquire_slot()
        slot.transition(SlotState.DISPATCH_READY)
        handle = self.buffer.create_handle(slot, object())
        slot.transition(SlotState.RELEASABLE)
        slot.release()
        reacquired = self.buffer.acquire_slot()
        self.assertEqual(reacquired.generation, handle.generation + 1)
        with self.assertRaises(RuntimeError):
            handle.validate(self.buffer)
        reacquired.release()

    def test_cross_buffer_handle_and_busy_destroy_are_rejected(self):
        other = _make_buffer()
        try:
            slot = self.buffer.acquire_slot()
            with self.assertRaises(RuntimeError):
                self.buffer.destroy()
            slot.transition(SlotState.DISPATCH_READY)
            handle = self.buffer.create_handle(slot, object())
            with self.assertRaises(ValueError):
                other.validate_handle(handle)
            slot.transition(SlotState.RELEASABLE)
            slot.release()
        finally:
            other.destroy()

    def test_v1_buffer_interfaces_call_native_runtime(self):
        x = torch.ones((4, 8), dtype=torch.float16)
        topk_idx = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=topk_idx_t)
        topk_weights = torch.ones((4, 2), dtype=torch.float32)

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            layout_event,
        ) = self.buffer.get_dispatch_layout(
            topk_idx,
            4,
            previous_event=SimpleNamespace(event=None),
        )
        self.assertEqual(num_tokens_per_rank.shape, (2,))
        self.assertIsNone(num_tokens_per_rdma_rank)
        self.assertEqual(num_tokens_per_expert.shape, (4,))
        self.assertEqual(is_token_in_rank.shape, (4, 2))
        self.assertIsInstance(layout_event, EventOverlap)

        recv_x, recv_topk_idx, recv_topk_weights, counts, handle, event = self.buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            config=Config(20, 6, 256, 6, 128),
        )
        self.assertEqual(recv_x.shape, x.shape)
        self.assertEqual(recv_topk_idx.shape, topk_idx.shape)
        self.assertEqual(recv_topk_weights.shape, topk_weights.shape)
        self.assertEqual(counts, [0, 0])
        self.assertEqual(len(handle), 6)
        self.assertEqual(handle[5].shape, (x.size(0), 2))
        self.assertIsInstance(event, EventOverlap)
        self.assertEqual(self.buffer.native.dispatch_args[-4:], (20, 6, 256, False))

        (
            no_topk_recv_x,
            no_topk_idx,
            no_topk_weights,
            no_topk_counts,
            no_topk_handle,
            _,
        ) = self.buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            config=Config(20, 6, 256, 6, 128),
        )
        self.assertEqual(no_topk_recv_x.shape, x.shape)
        self.assertIsNone(no_topk_idx)
        self.assertIsNone(no_topk_weights)
        self.assertEqual(no_topk_counts, [0, 0])
        self.assertEqual(len(no_topk_handle), 6)

        combined_x, combined_topk_weights, combine_event = self.buffer.combine(
            recv_x,
            handle,
            topk_weights=recv_topk_weights,
            config=Config(20, 4, 256, 6, 128),
        )
        self.assertEqual(combined_x.shape, x.shape)
        self.assertEqual(combined_topk_weights.shape, topk_weights.shape)
        self.assertIsInstance(combine_event, EventOverlap)
        self.assertEqual(self.buffer.native.combine_args[-3:], (20, 4, 256))
        for removed in (True, False):
            with self.subTest(split_reduce=removed), self.assertRaises(TypeError):
                self.buffer.combine(recv_x, handle, split_reduce=removed)

        cached_recv_x, cached_topk_idx, cached_topk_weights, cached_counts, cached_handle, _ = self.buffer.dispatch(
            x,
            handle=handle,
            config=Config(20, 6, 256, 6, 128),
        )
        self.assertEqual(cached_recv_x.shape, x.shape)
        self.assertIsNone(cached_topk_idx)
        self.assertIsNone(cached_topk_weights)
        self.assertIsNone(cached_counts)
        self.assertIsNone(cached_handle)
        self.assertFalse(self.buffer.native.dispatch_args[-1])

        worst_recv_x, _, _, _, worst_handle, _ = self.buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            num_worst_tokens=8,
            config=Config(20, 6, 256, 6, 128),
        )
        self.assertEqual(worst_recv_x.shape, (8, x.size(1)))
        self.assertEqual(worst_handle[5].shape, (x.size(0), 2))

    def test_dispatch_zero_copy_is_explicit_and_applies_to_cached_calls(self):
        x = torch.zeros((4, 8), dtype=torch.bfloat16)
        topk_idx = torch.zeros((4, 1), dtype=topk_idx_t)
        counts, _, experts, membership, _ = self.buffer.get_dispatch_layout(topk_idx, 4)
        kwargs = dict(num_tokens_per_rank=counts, num_tokens_per_expert=experts, is_token_in_rank=membership)
        _, _, _, _, handle, _ = self.buffer.dispatch(x, zero_copy=True, **kwargs)
        self.assertTrue(self.buffer.native.dispatch_args[-1])
        self.buffer.dispatch(x, handle=handle, zero_copy=True)
        self.assertTrue(self.buffer.native.dispatch_args[-1])
        self.buffer.dispatch(x, handle=handle)
        self.assertFalse(self.buffer.native.dispatch_args[-1])
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "zero_copy must be a bool"):
                    self.buffer.dispatch(x, zero_copy=value, **kwargs)
                with self.assertRaisesRegex(TypeError, "zero_copy must be a bool"):
                    self.buffer.dispatch(x, handle=handle, zero_copy=value)

    def test_owned_recv_outputs_record_the_consumer_stream(self):
        consumer_stream = object()
        tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
        recorded = []
        tensor.record_stream = recorded.append
        fake_npu = SimpleNamespace(current_stream=lambda device: consumer_stream)
        with patch.object(torch, "npu", fake_npu, create=True):
            buffer_module._record_recv_outputs_on_current_stream(tensor, None)
        self.assertEqual(recorded, [consumer_stream])

    def test_all_v1_operations_forward_allocation_stream_and_record_outputs(self):
        x = torch.arange(32, dtype=torch.float16).reshape(4, 8)
        idx = torch.tensor([[0, 1]] * 4, dtype=topk_idx_t)
        weights = torch.ones((4, 2), dtype=torch.float32)
        caller = SimpleNamespace(stream_id=1234)
        fake_npu = SimpleNamespace(current_stream=lambda: caller)
        for on_comm in (False, True):
            with (
                self.subTest(allocate_on_comm_stream=on_comm),
                patch.object(torch, "npu", fake_npu),
                patch.object(buffer_module, "_record_recv_outputs_on_current_stream") as record,
            ):
                kw = dict(allocate_on_comm_stream=on_comm, async_finish=True)
                expected = None if on_comm else caller.stream_id
                ranks, _, experts, mask, _ = self.buffer.get_dispatch_layout(idx, 4, **kw)
                self.assertEqual(self.buffer._native.allocation_stream_id, expected)
                record.assert_called_once_with(ranks, None, experts, mask)
                record.reset_mock()
                rx, ri, rw, _, handle, _ = self.buffer.dispatch(
                    x,
                    topk_idx=idx,
                    topk_weights=weights,
                    num_tokens_per_rank=ranks,
                    num_tokens_per_expert=experts,
                    is_token_in_rank=mask,
                    **kw,
                )
                self.assertEqual(self.buffer._native.allocation_stream_id, expected)
                self.assertEqual(record.call_count, 2)
                self.assertEqual(tuple(id(t) for t in record.call_args.args), tuple(id(t) for t in handle))
                record.reset_mock()
                self.buffer.dispatch(x, handle=handle, **kw)
                self.assertEqual(self.buffer._native.allocation_stream_id, expected)
                self.assertEqual(record.call_count, 1)
                record.reset_mock()
                self.buffer.combine(rx, handle, topk_weights=rw, bias=(x, x), **kw)
                self.assertEqual(self.buffer._native.allocation_stream_id, expected)
                self.assertEqual(record.call_count, 1)
                record.reset_mock()
                ex, ew, eh, _ = self.buffer.expand(x, idx, topk_weights=weights, num_tokens_per_expert=[4, 4], **kw)
                self.assertEqual(self.buffer._native.allocation_stream_id, expected)
                self.assertEqual(len(record.call_args.args), 6)
                self.buffer.expand(x, handle=eh, **kw)
                self.assertEqual(self.buffer._native.allocation_stream_id, expected)
                record.reset_mock()
                self.buffer.aggregate(ex, eh, topk_weights=ew, **kw)
                self.assertEqual(self.buffer._native.allocation_stream_id, expected)
                self.assertEqual(record.call_count, 1)

    def test_zero_copy_registers_handle_but_not_shmem_payload(self):
        x = torch.zeros((4, 8), dtype=torch.float16)
        idx = torch.zeros((4, 2), dtype=topk_idx_t)
        ranks, _, experts, mask, _ = self.buffer.get_dispatch_layout(idx, 4)
        with patch.object(buffer_module, "_record_recv_outputs_on_current_stream") as record:
            *_, handle, _ = self.buffer.dispatch(
                x,
                topk_idx=idx,
                num_tokens_per_rank=ranks,
                num_tokens_per_expert=experts,
                is_token_in_rank=mask,
                zero_copy=True,
            )
            record.assert_called_once()
            self.assertEqual(tuple(id(t) for t in record.call_args.args), tuple(id(t) for t in handle))
            record.reset_mock()
            self.buffer.dispatch(x, handle=handle, zero_copy=True)
            record.assert_not_called()

    def test_expand_and_aggregate_interfaces(self):
        x = torch.arange(32, dtype=torch.float16).reshape(4, 8)
        topk_idx = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=topk_idx_t)
        topk_weights = torch.ones((4, 2), dtype=torch.float32)

        expanded_x, expanded_weights, handle, event = self.buffer.expand(
            x,
            topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_expert=[2, 2],
            expert_alignment=1,
            config=Config(20),
        )
        self.assertEqual(expanded_x.shape, (4, 8))
        self.assertEqual(expanded_weights.shape, (4,))
        self.assertIsInstance(handle, ExpandHandle)
        self.assertEqual(handle.num_expanded_tokens, 4)
        self.assertEqual(handle.expert_offsets.tolist(), [0, 2, 4])
        self.assertEqual(handle.actual_expert_counts.tolist(), [2, 2])
        self.assertIsInstance(event, EventOverlap)

        aggregated_x, aggregated_values, aggregate_event = self.buffer.aggregate(
            expanded_x,
            handle,
            topk_weights=expanded_weights,
            expanded_values=torch.arange(4, dtype=torch.float32),
            config=Config(20),
        )
        torch.testing.assert_close(aggregated_x[:2], x[:2] * 2)
        torch.testing.assert_close(aggregated_x[2:], torch.zeros_like(x[2:]))
        self.assertEqual(aggregated_values.shape, topk_idx.shape)
        self.assertIsInstance(aggregate_event, EventOverlap)

        cached_x, cached_weights, cached_handle, _ = self.buffer.expand(
            x * 2,
            handle=handle,
            config=Config(20),
        )
        torch.testing.assert_close(cached_x, expanded_x * 2)
        self.assertIsNone(cached_weights)
        self.assertIs(cached_handle, handle)


class StandaloneSlotTest(unittest.TestCase):
    def test_unbound_slot_has_no_named_native_resources(self):
        slot = Slot(3)
        with self.assertRaises(RuntimeError):
            _ = slot.comm_meta_ptr


if __name__ == "__main__":
    unittest.main()


# python -m unittest test.test_public_api
