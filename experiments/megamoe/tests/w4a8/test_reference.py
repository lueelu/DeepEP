# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

import numpy as np
import pytest

from deep_ep_experimental.megamoe.reference import (
    FP4,
    FP8_POSITIVE,
    bf16,
    dequantize_input,
    error_metrics,
    expert_forward,
    fp8_decode,
    fp8_encode,
    local_reference,
    quantize_input,
    unpack_weight,
)


def test_all_fp8_finite_codes_roundtrip():
    codes = np.concatenate((np.arange(127), np.arange(128, 255))).astype(np.uint8)
    np.testing.assert_array_equal(fp8_encode(fp8_decode(codes)), codes)


def test_fp8_nearest_even_and_saturation():
    mid = (FP8_POSITIVE[:-1] + FP8_POSITIVE[1:]) / 2
    expected = np.arange(126)
    expected += expected % 2
    np.testing.assert_array_equal(fp8_encode(mid), expected)
    np.testing.assert_array_equal(fp8_encode([1000, -1000]), [126, 254])
    with pytest.raises(ValueError):
        fp8_decode(np.array([127], np.uint8))


def test_quant_block_scale_and_zero():
    x = np.full((2, 64), 0.5, dtype=np.float32)
    x[:, 32:] = 4.0
    codes, scale = quantize_input(x)
    np.testing.assert_array_equal(scale, [[118, 121], [118, 121]])
    np.testing.assert_array_equal(dequantize_input(codes, scale), x)
    codes, scale = quantize_input(np.zeros((1, 32), np.float32))
    assert not codes.any()
    assert not scale.any()


def test_fp4_nibble_and_scale_order():
    packed = np.tile(np.arange(16, dtype=np.uint8) | (np.arange(16, dtype=np.uint8)[::-1] << 4), 2).reshape(1, 1, 32)
    scales = np.array([[[[127, 128]]]], dtype=np.uint8)
    result = unpack_weight((packed, scales))[0, 0]
    np.testing.assert_array_equal(result[:32:2], FP4)
    np.testing.assert_array_equal(result[1:32:2], FP4[::-1])
    np.testing.assert_array_equal(result[32:], result[:32] * 2)


def test_local_moe_decomposition_and_empty_expert():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 32)).astype(np.float32) * 0.1
    w1 = rng.normal(size=(4, 64, 32)).astype(np.float32) * 0.1
    w2 = rng.normal(size=(4, 32, 32)).astype(np.float32) * 0.1
    ids = np.array([[0, 2], [2, 0], [0, 0]])
    weights = np.array([[0.2, 0.8], [0.7, 0.3], [0.5, 0.5]], np.float32)
    full = local_reference(x, ids, weights, w1, w2, 0)
    split = local_reference(x, ids, weights, w1[:2], w2[:2], 0)
    split += local_reference(x, ids, weights, w1[2:], w2[2:], 2)
    np.testing.assert_allclose(full, split, atol=1e-8)
    expected = expert_forward(x[2:3], w1[0], w2[0])
    np.testing.assert_allclose(full[2:3], expected)
    assert error_metrics(full, full)["relative_rmse"] == 0
    assert error_metrics(np.full_like(full, np.nan), full)["finite"] is False


def test_bf16_ties_to_even():
    values = np.array([0x3F808000, 0x3F818000], np.uint32).view(np.float32)
    np.testing.assert_array_equal(bf16(values).view(np.uint32), [0x3F800000, 0x3F820000])


def test_route_weight_does_not_commute_with_fp8_quantization():
    # 隔离量化与乘权重的顺序；不是对 NVIDIA kernel 的逐位模拟。
    rng = np.random.default_rng(31)
    activated = rng.normal(size=(8, 64)).astype(np.float32)
    w2 = rng.normal(size=(32, 64)).astype(np.float32)
    weight = np.array([0.1, -0.37, 0.8, 0.003, 0.19, 0, 0.47, 1], np.float32)[:, None]
    current = bf16(dequantize_input(*quantize_input(activated)) @ w2.T) * weight
    before_quant = bf16(dequantize_input(*quantize_input(activated * weight)) @ w2.T)
    assert error_metrics(current, before_quant)["relative_rmse"] > 1e-3
    np.testing.assert_array_equal(current[5], 0)


@pytest.mark.parametrize("routing", ["balanced", "random", "skewed"])
def test_benchmark_routing_stress_patterns(routing):
    from deep_ep_experimental.megamoe.benchmark import inputs, load_config
    from deep_ep_experimental.megamoe.validation import check_routing

    config = load_config("smoke")
    first = inputs(config, 0, 96, routing, world=2)
    repeat = inputs(config, 0, 96, routing, world=2)
    for a, b in zip(first, repeat):
        np.testing.assert_array_equal(a, b)
    _, indices, weights = first
    check_routing(indices.tolist(), weights.tolist(), 8, 2)
    if routing == "skewed":
        assert set(indices.ravel()) == {0, 1}
    if routing == "random":
        assert (weights < 0).any()


@pytest.mark.parametrize("world", [1, 8, 32, 64, 128])
@pytest.mark.parametrize("bs", [1, 17, 96, 4096])
def test_balanced_routes_match_legacy_rank_first_order(world, bs):
    from deep_ep_experimental.megamoe.benchmark import inputs

    # Small H keeps this a routing test; EP/E/top-k/BS match the real experiment.
    experts, topk = 384, 6
    local = experts // world
    config = dict(seed=13, model=dict(hidden=32, num_experts=experts, num_topk=topk))
    all_ids = []
    for rank in range(world):
        _, ids, _ = inputs(config, rank, bs, world=world)
        assert ids.dtype == np.int32 and ids.shape == (bs, topk)
        assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
        # Every source spreads routes evenly across destination ranks.
        peer_counts = np.bincount((ids // local).ravel(), minlength=world)
        assert peer_counts.max() - peer_counts.min() <= 1
        if world == 64 and bs == 96:
            np.testing.assert_array_equal(peer_counts, np.full(world, 9))
            # Expert 0 has work from every source: AIV0/AIV1 pairs get 2/1 rows.
            assert np.count_nonzero(ids == 0) == (2 if rank % 2 == 0 else 1)
        all_ids.append(ids)
    actual = np.concatenate(all_ids)
    # Enumerate the old global route stream, then split it into per-source tokens.
    # This also detects accidentally restarting the local-expert cursor at each source.
    cycle = [rank * local + expert for expert in range(local) for rank in range(world)]
    expected = np.resize(np.array(cycle, dtype=np.int32), actual.size).reshape(actual.shape)
    np.testing.assert_array_equal(actual, expected)
    counts = np.bincount(actual.ravel(), minlength=experts)
    assert counts.max() - counts.min() <= 1
    if actual.size % experts == 0:
        np.testing.assert_array_equal(counts, np.full(experts, actual.size // experts))
    if world == 64 and bs in (96, 4096):
        np.testing.assert_array_equal(counts, np.full(experts, bs))


@pytest.mark.parametrize("routing", ["balanced", "random", "skewed"])
def test_routing_update_preserves_input_and_route_weight_rng(routing):
    from deep_ep_experimental.megamoe.benchmark import inputs

    rank, bs, topk, experts = 3, 17, 6, 384
    config = dict(seed=13, model=dict(hidden=32, num_experts=experts, num_topk=topk))
    rng = np.random.default_rng(config["seed"] + rank * 1009 + bs)
    expected_x = bf16(rng.normal(0, 0.2, (bs, 32)).astype(np.float32))
    if routing == "random":
        expected_ids = np.array([rng.choice(experts, topk, replace=False) for _ in range(bs)])
    elif routing == "skewed":
        expected_ids = np.tile(np.arange(topk), (bs, 1))
    weights = rng.random((bs, topk), dtype=np.float32)
    if routing == "random":
        weights = 2 * weights - 1
    weights /= np.abs(weights).sum(axis=1, keepdims=True)
    x, ids, actual_weights = inputs(config, rank, bs, routing, world=64)
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(actual_weights, weights)
    if routing != "balanced":
        np.testing.assert_array_equal(ids, expected_ids)


@pytest.mark.parametrize("world,rank", [(0, 0), (7, 0), (64, -1), (64, 64)])
def test_balanced_routes_require_valid_ep_and_rank(world, rank):
    from deep_ep_experimental.megamoe.benchmark import inputs

    config = dict(seed=13, model=dict(hidden=32, num_experts=384, num_topk=6))
    with pytest.raises(ValueError):
        inputs(config, rank, 96, world=world)
