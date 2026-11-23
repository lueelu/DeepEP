# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

import sys

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api
from api_test_support import DTYPES, Tensor, quantized_backend as quantized_backend, weights


@pytest.mark.parametrize("bf16", [False, True])
def test_public_workflow_output_stats_timer_and_preparation_reuse(backend, bf16):
    b = backend.buffer
    assert b.x.shape == (17, 1024) and b.x_sf.shape == (17, 32)
    original = b.x
    assert b.allocate_inputs() is b and b.x is original
    x = Tensor(np.ones((17, 1024)), "bfloat16") if bf16 else None
    b.validate_inputs(17, x=x)
    b.prepare(17)
    b.prepare(17)
    y, stats = Tensor(np.zeros((17, 1024)), "bfloat16"), Tensor([7, 11], "int32")
    assert api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=stats, x=x) is None
    np.testing.assert_array_equal(y.data, 3)
    np.testing.assert_array_equal(stats.data, [9, 14])
    timer = b.allocate_timer()
    assert not timer.data.any()
    api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=stats, x=x, timer=timer)
    np.testing.assert_array_equal(stats.data, [11, 17])
    launches = [event[1] for event in backend.events if isinstance(event, tuple) and event[0] == "launch"]
    assert launches[0][8] is None and launches[1][8] is timer
    assert (launches[0][1] is None) == bf16
    assert launches[0][-1] is y and launches[1][-1] is y
    assert (
        sum(event == ("prepare", 17) for event in backend.events if isinstance(event, tuple) and event[0] == "prepare")
        == 1
    )
    assert stats.streams == [b._runtime.stream, b._runtime.stream]
    assert b.workspace_bytes == 4096 and b.symmetric_buffer_bytes == 2 << 20
    b.prepare(1)
    assert b._plan.bs == 1 and b._plan_bs == 1


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_fp4_and_packed_e8m0_are_reinterpreted_without_numeric_cast(backend, dtype):
    raw = np.arange(512 * 64, dtype=np.uint8).reshape(1, 512, 64)
    # Distinct row-dependent bytes expose both weight and scale column mismatches.
    row_ids = np.arange(512, dtype=np.uint16)
    raw[0, :, 0] = row_ids % 256
    raw[0, :, 1] = row_ids // 256
    sf = np.empty((1, 512, 4), dtype=np.uint8)
    sf[0, :, 0] = row_ids % 256
    sf[0, :, 1] = row_ids // 256
    sf[0, :, 2:] = np.array([129, 140], dtype=np.uint8)
    pair = Tensor(raw.view(DTYPES[dtype]), dtype), Tensor(sf.view(np.int32), "int32")
    order = np.array([*range(128), *range(256, 384), *range(128, 256), *range(384, 512)])
    prepared = api.transform_weights_for_mega_moe(pair, pair)
    for idx, (weight, scale) in enumerate(prepared):
        rows = order if idx == 0 else np.arange(512)
        np.testing.assert_array_equal(weight.data, raw[:, rows])
        np.testing.assert_array_equal(scale.data, sf[:, rows].reshape(1, 512, 2, 2))
        assert weight.dtype == "uint8" and scale.dtype == "float8_e8m0fnu"
        assert weight.npu_format == 50
    assert sum(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events) == 3
    planar_weight, planar_scale = prepared[0].select(False)
    np.testing.assert_array_equal(planar_weight.data, raw)
    np.testing.assert_array_equal(planar_scale.data, sf.reshape(1, 512, 2, 2))


def test_shared_fp4_preparation_retains_planar_weights_and_scales(backend):
    raw = np.arange(512 * 64, dtype=np.uint8).reshape(1, 512, 64)
    sf = np.arange(512 * 4, dtype=np.uint8).reshape(1, 512, 2, 2)
    pair = Tensor(raw, "uint8"), Tensor(sf, "float8_e8m0fnu")
    for weight, scale in api.transform_weights_for_mega_moe(pair, pair, shared=True):
        np.testing.assert_array_equal(weight.data, raw)
        np.testing.assert_array_equal(scale.data, sf)


@pytest.mark.parametrize("bf16", [False, True])
def test_routed_layout_switches_with_actual_bs_without_hot_path_cast(backend, bf16):
    b = backend.buffer
    b.num_max_tokens_per_rank = 4096
    b.x = Tensor(np.zeros((4096, 1024)), "float8_e4m3fn")
    b.x_sf = Tensor(np.zeros((4096, 32)), "float8_e8m0fnu")
    b.topk_idx = Tensor(np.tile([0, 1], (4096, 1)), "int32")
    b.topk_weights = Tensor(np.ones((4096, 2)), "float32")
    raw = weights()
    for weight, _ in raw:
        weight.npu_format = 2
    raw[0][0].data[:, :, 0] = np.arange(512, dtype=np.uint16) % 251
    w1, w2 = api.transform_weights_for_mega_moe(*raw)
    # Preparing weights must not share writable storage with the caller's raw scales.
    assert w1.select(False)[1].untyped_storage().data_ptr() != raw[0][1].untyped_storage().data_ptr()
    casts = sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events)
    for bs in (96, 256, 257, 4096, 96):
        output = Tensor(np.zeros((bs, 1024)), "bfloat16")
        x = Tensor(np.ones((bs, 1024)), "bfloat16") if bf16 else None
        api.fp8_fp4_mega_moe(output, w1, w2, b, x=x)
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        selected = w1.select(bs <= 256)
        assert launch[4] is selected[0] and launch[6] is selected[1]
        assert b._plan.bs == bs  # actual BS, not max capacity or per-expert M
    assert sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events) == casts
    with pytest.raises(ValueError, match="keep its returned l1 pair intact"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((257, 1024)), "bfloat16"), tuple(w1), w2, b)


@pytest.mark.parametrize("layout", [True, False])
@pytest.mark.parametrize("index", [0, 1])
def test_prepared_layout_mutation_is_not_silently_cached(backend, layout, index):
    raw = weights()
    for weight, _ in raw:
        weight.npu_format = 2
    w1, _ = api.transform_weights_for_mega_moe(*raw)
    tensor = w1.select(layout)[index]
    tensor.add_(Tensor(np.ones(tensor.shape), tensor.dtype))
    # Even requesting the other layout must fail rather than using an out-of-date copy.
    with pytest.raises(ValueError, match="modified in-place"):
        w1.select(not layout)


def test_fp4_transform_rejects_double_preparation(backend):
    pair = Tensor(np.zeros((1, 256, 64)), "uint8"), Tensor(np.ones((1, 256, 2, 2)), "float8_e8m0fnu")
    pair[0].npu_format = 50
    with pytest.raises(ValueError, match="already converted NZ"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_routed_transform_rejects_incomplete_gate_up_pair(backend):
    pair = Tensor(np.zeros((1, 128, 64)), "uint8"), Tensor(np.ones((1, 128, 2, 2)), "float8_e8m0fnu")
    with pytest.raises(ValueError, match="divisible by 128"):
        api.transform_weights_for_mega_moe(pair, pair)


@pytest.mark.parametrize("weight_format", [29, 50])
def test_prepared_fp4_nz_variants_reach_launch(backend, weight_format):
    w1, w2 = weights()
    w1[0].npu_format = w2[0].npu_format = weight_format
    api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), w1, w2, backend.buffer)
    assert any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


@pytest.mark.parametrize("name", ["l1_weights", "l2_weights", "shared_l1_weights", "shared_l2_weights"])
@pytest.mark.parametrize("weight_format", [2, 4, 51, 52, 53, 54])
def test_invalid_fp4_layout_rejected_with_tensor_context_before_launch(backend, name, weight_format):
    b = backend.buffer
    b.num_shared_experts = 2
    pairs = dict(zip(("l1_weights", "l2_weights", "shared_l1_weights", "shared_l2_weights"), (*weights(), *weights())))
    pairs[name][0].npu_format = weight_format
    with pytest.raises(ValueError, match=rf"{name}:.*got format={weight_format}, dtype=uint8, shape="):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), sym_buffer=b, **pairs)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


def test_format_cast_returning_nd_fails_at_preparation(backend, monkeypatch):
    monkeypatch.setattr(sys.modules["torch_npu"], "npu_format_cast", lambda t, *a, **k: t)
    pair = (Tensor(np.zeros((1, 256, 64)), "uint8"), Tensor(np.full((1, 256, 2, 2), 127), "float8_e8m0fnu"))
    with pytest.raises(ValueError, match="l1_weights:.*got format=2"):
        api.transform_weights_for_mega_moe(pair, pair)
