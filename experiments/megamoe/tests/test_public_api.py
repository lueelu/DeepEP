# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

from unittest.mock import Mock

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api
from api_test_support import Tensor, quantized_backend as quantized_backend, weights
from deep_ep_experimental.megamoe.validation import collective_check


def test_output_and_stats_aliases_rejected_before_launch(backend):
    b = backend.buffer
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    with pytest.raises(ValueError, match="alias"):
        api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=b.topk_idx[0])
    with pytest.raises(ValueError, match="overlap"):
        api.fp8_fp4_mega_moe(y, *weights(), b, x=y)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


@pytest.mark.parametrize("bad", [-1, 4, 0])
def test_collective_routing_preflight_rejects_bad_ids(backend, bad):
    backend.buffer.topk_idx.data[0, 1] = bad
    with pytest.raises(ValueError, match="rank=0.*token=0"):
        backend.buffer.validate_inputs(17)


def test_collective_remote_error_and_bs_mismatch(backend):
    def mismatched(output, value, **kwargs):
        output[:] = [value, (None, 96)]

    backend.dist.all_gather_object = mismatched
    with pytest.raises(ValueError, match="ranks disagree.*rank=1: 96"):
        collective_check(backend.buffer.group, "prepare", None, 17)

    def failed(output, value, **kwargs):
        output[:] = [value, ("out of memory", 17)]

    backend.dist.all_gather_object = failed
    with pytest.raises(ValueError, match="rank=1: out of memory"):
        collective_check(backend.buffer.group, "prepare", None, 17)


def test_timer_disabled_and_destroyed_buffer_fail_explicitly(backend):
    backend.ext.timer_enabled = False
    with pytest.raises(RuntimeError, match="DEEPEP_MEGAMOE_TIMER"):
        backend.buffer.allocate_timer()
    backend.buffer.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), *weights(), backend.buffer)


def test_fp8_weights_remain_nd_without_fp4_format_cast(backend):
    pair = (Tensor(np.zeros((1, 128, 128)), "float8_e4m3fn"), Tensor(np.full((1, 128, 2, 2), 127), "float8_e8m0fnu"))
    for weight, scale in api.transform_weights_for_mega_moe(pair, pair):
        assert weight is pair[0]
        np.testing.assert_array_equal(scale.data, pair[1].data)
    assert not any(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events)


def test_store_group_public_configuration_and_prepare_without_process_group(backend, monkeypatch):
    from deep_ep._store import StoreGroup

    stages = []
    group = StoreGroup(None, 0, 2, prefix="api-test")

    def gather(value, stage):
        stages.append(stage)
        return [value, value]

    group.all_gather = gather

    def forbidden(*args, **kwargs):
        pytest.fail("Store API accessed a ProcessGroup")

    monkeypatch.setattr(backend.dist, "get_world_size", forbidden)
    monkeypatch.setattr(backend.dist, "all_gather_object", forbidden)
    buffer = api.SymmBuffer(group, 4, 17, 2, 1024, 256)
    buffer.topk_idx.data[:] = [0, 1]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    buffer.validate_inputs(17)
    buffer.prepare(17)
    buffer.destroy()
    assert stages == [
        "check:configuration",
        "check:input allocation",
        "check:input preflight",
        "check:prepare",
        "check:prepare BS=17",
    ]


def test_store_group_ep256_with_eight_bf16_experts_per_rank(backend, monkeypatch):
    group = api.StoreGroup(None, 255, 256, prefix="ep256", bf16=True)
    monkeypatch.setattr(group, "all_gather", lambda value, stage: [value] * group.size())
    backend.ext.bf16_enabled = True
    backend.ext.bf16_buffer_size = Mock(return_value=2 << 20)
    options = dict(mma_type="bf16xbf16", need_grad=True, max_recv_tokens=128)
    with pytest.raises(ValueError, match="num_experts"):
        api.SymmBuffer(group, 256 * 9, 17, 2, 1024, 256, **options)
    for mma_type in ("fp8xfp4", "fp8xfp8"):
        with pytest.raises(ValueError, match="world_size"):
            api.SymmBuffer(group, 256 * 8, 17, 2, 1024, 256, mma_type=mma_type)
    buf = api.SymmBuffer(group, 256 * 8, 17, 2, 1024, 256, **options)
    try:
        backend.ext.bf16_buffer_size.assert_called_once_with(17, 1024, 256, 2, 256, 8, 128, True)
    finally:
        buf.destroy()


def test_non_finite_route_weights_and_bad_capacity_fail_before_launch(backend):
    backend.buffer.topk_weights.data[2, 0] = np.nan
    with pytest.raises(ValueError, match="token=2.*finite"):
        backend.buffer.validate_inputs(17)
    with pytest.raises(ValueError, match="prepare.*BS"):
        backend.buffer.prepare(18)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


def test_plan_allocation_failure_is_collective_and_retryable(backend):
    plan_type = backend.ext.Plan

    def fail(*args):
        raise RuntimeError("workspace allocation failed")

    backend.ext.Plan = fail
    with pytest.raises(ValueError, match="rank=0: workspace allocation failed"):
        backend.buffer.prepare(17)
    assert backend.buffer._plan is None and backend.buffer._plan_bs is None
    backend.ext.Plan = plan_type
    backend.buffer.prepare(17)
    assert backend.buffer._plan_bs == 17


@pytest.mark.parametrize("need_grad,capacity", [(False, None), (False, 80), (True, 80), (True, None)])
def test_bf16_receive_capacity(backend, need_grad, capacity):
    args = (backend.buffer.group, 8, 17, 3, 1024, 256)
    backend.ext.bf16_enabled = True
    backend.ext._inference_max_recv_tokens = Mock(return_value=102)
    backend.ext.bf16_buffer_size = Mock(return_value=2 << 20)
    options = {} if capacity is None else {"max_recv_tokens": capacity}
    if need_grad and capacity is None:
        with pytest.raises(ValueError, match="max_recv_tokens"):
            api.SymmBuffer(*args, mma_type="bf16xbf16", need_grad=True, **options)
        backend.ext._inference_max_recv_tokens.assert_not_called()
        backend.ext.bf16_buffer_size.assert_not_called()
        return

    buf = api.SymmBuffer(*args, mma_type="bf16xbf16", need_grad=need_grad, **options)
    try:
        if capacity is None:
            backend.ext._inference_max_recv_tokens.assert_called_once_with(17, 2, 3, 4)
            assert buf.max_recv_tokens == 102
        else:
            backend.ext._inference_max_recv_tokens.assert_not_called()
            assert buf.max_recv_tokens == capacity
        backend.ext.bf16_buffer_size.assert_called_once_with(17, 1024, 256, 3, 2, 4, buf.max_recv_tokens, need_grad)
    finally:
        buf.destroy()
