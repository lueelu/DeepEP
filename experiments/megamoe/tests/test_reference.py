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


def test_mixed_shared_fp8_uses_the_same_logical_values_and_scales():
    from deep_ep_experimental.megamoe.reference import FP4, fp4_to_fp8, fp8_decode

    packed = np.arange(256, dtype=np.uint8)
    values = fp8_decode(fp4_to_fp8(packed))
    np.testing.assert_array_equal(values[::2], FP4[packed & 15])
    np.testing.assert_array_equal(values[1::2], FP4[packed >> 4])


@pytest.mark.parametrize("limit", [None, 0, 2, 5, 10])
def test_activation_clamp_applies_asymmetrically_before_swiglu(limit):
    from deep_ep_experimental.megamoe.reference import bf16, dequantize_input, expert_forward, quantize_input

    x = np.eye(32, dtype=np.float32)
    gate = np.resize(np.array([-12, -4, 4, 12], np.float32), (32, 32))
    up = np.resize(np.array([-15, 9, -6, 20], np.float32), (32, 32))
    w1 = np.concatenate((gate, up), axis=1).T
    w2 = np.eye(32, dtype=np.float32)
    g = gate if limit is None else np.minimum(gate, limit)
    u = up if limit is None else np.clip(up, -limit, limit)
    golden = bf16(dequantize_input(*quantize_input(bf16(u * (g / (1 + np.exp(-g)))))))
    np.testing.assert_array_equal(expert_forward(x, w1, w2, limit), golden)


@pytest.mark.parametrize("bf16", [False, True])
def test_eplb_replay_driver_rebinds_slots_and_alternates_views(monkeypatch, bf16):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from deep_ep_experimental.megamoe import api, benchmark, validation, weights

    torch = pytest.importorskip("torch")
    recording = []
    replays = []

    class Graph:
        def replay(self):
            replays.append(self.view)
            self.run()

    @contextmanager
    def capture(graph):
        recording.append(graph)
        yield
        recording.pop()

    class Owner:
        def __init__(self, slots, prepared=False):
            self.count = len(slots)
            self.tables = {f: torch.empty((4, self.count), dtype=torch.int64) for f in (False, True)}
            self.l1 = self.l2 = SimpleNamespace(owner=self)
            values = [int(pair[0]) for pair in slots] if prepared else slots
            self.bind(values)

        def bind(self, values):
            def projection():
                return tuple([torch.tensor(v) for v in values] for _ in range(2))

            self.planar, self.fused, self.second = (projection() for _ in range(3))
            for f, table in self.tables.items():
                first = self.fused if f else self.planar
                table.copy_(
                    torch.tensor(
                        [
                            [t.data_ptr() for t in group]
                            for group in (first[0], self.second[0], first[1], self.second[1])
                        ]
                    )
                )

        def select(self, fused):
            return None, None, self.tables[fused]

        def expert_views(self, *, layout):
            return self.fused if layout == "fused" else self.planar, self.second

        def update_experts(self, indices, l1, l2, *, replace, input_layout="nd", planar_l1=None):
            assert list(indices) == list(range(self.count)) and replace
            if input_layout == "prepared":
                assert planar_l1 is not None
                self.bind([int(t) for t in l1[0]])
            else:
                assert l1 == l2 == [20, 30, 10]
                self.bind(l1)

        def refresh_gmm1(self, slots, *, source_layout):
            assert slots == list(range(self.count)) and source_layout == "planar"
            for destinations, sources in zip(self.fused, self.planar):
                for dst, src in zip(destinations, sources):
                    dst.copy_(src)

        def release_retired(self):
            pass

    owner = Owner([10, 20, 30])

    def launch(y, l1, l2, view, *, x, cumulative_local_expert_recv_stats, **kwargs):
        def run():
            ids = view.topk_idx.long()
            values = torch.stack(l1.owner.fused[0])[ids]
            y.copy_((x if bf16 else view.x) + (values * view.topk_weights).sum(-1, keepdim=True))
            cumulative_local_expert_recv_stats.add_(torch.bincount(ids.flatten(), minlength=3))

        if recording:
            recording[-1].run = run
            recording[-1].view = view._view_id
        else:
            run()

    def check(group, stage, error):
        assert error is None, stage

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(graph=capture, NPUGraph=Graph, synchronize=lambda: None), raising=False
    )
    monkeypatch.setattr(api, "fp8_fp4_mega_moe", launch)
    monkeypatch.setattr(weights, "prepare_weights_for_mega_moe", lambda l1, l2, **kw: Owner(l1, prepared=True))
    monkeypatch.setattr(validation, "collective_check", check)

    def view(n, index):
        return SimpleNamespace(
            _view_id=index,
            x=torch.ones(n, 4),
            x_sf=torch.ones(n, 1),
            topk_idx=torch.zeros(n, 2, dtype=torch.int32),
            topk_weights=torch.ones(n, 2),
        )

    target, draft = view(8, 0), view(4, 1)
    act = target.x.clone() if bf16 else None
    y, stats = torch.zeros_like(target.x), torch.zeros(3, dtype=torch.int32)
    graph = Graph()
    with capture(graph):
        launch(y, owner.l1, owner.l2, target, x=act, cumulative_local_expert_recv_stats=stats)
    config = dict(seed=42, model=dict(num_experts=3, num_topk=2, hidden=4))
    benchmark._check_eplb_replays(
        None, config, 0, 1, target, draft, (owner.l1, owner.l2), act, {}, (graph, y, stats), [[10, 20, 30]] * 2
    )
    assert replays == [0, 1] * 4 and [int(t) for t in owner.fused[0]] == [10, 20, 30]
    for obj, n, pattern in ((target, 8, "balanced"), (draft, 4, "skewed")):
        ids = benchmark.inputs(config, 0, n, pattern, world=1)[1]
        np.testing.assert_array_equal(obj.topk_idx.numpy(), (ids - 1) % 3)
