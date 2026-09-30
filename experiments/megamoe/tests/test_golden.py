# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""CPU Golden equivalence, cache lifecycle and small-report-only accuracy."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from deep_ep_experimental.megamoe import golden
from deep_ep_experimental.megamoe.benchmark import accuracy, inputs
from deep_ep_experimental.megamoe.reference import (
    bf16,
    dequantize_input,
    expert_forward,
    local_reference,
    make_weights,
    quantize_input,
    unpack_weight,
)


@pytest.fixture
def config():
    return dict(
        schema_version=1,
        model=dict(hidden=64, intermediate_hidden=64, num_experts=4, num_topk=2, num_shared_experts=1),
        bs_values=[7],
        seed=13,
        warmup=0,
        iters=1,
    )


def naive(config, world, source, bs, routing, check_tokens):
    """Old per-source/per-owner oracle, with no cached/batched Golden implementation."""
    model = config["model"]
    local = model["num_experts"] // world
    counts = sum(
        np.bincount(inputs(config, r, bs, routing, world=world)[1].ravel(), minlength=model["num_experts"])
        for r in range(world)
    )
    x, ids, weights = inputs(config, source, bs, routing, world=world)
    selected = golden.token_indices(bs, check_tokens)
    x = dequantize_input(*quantize_input(x[selected]))
    result = np.zeros_like(x)
    for owner in range(world):
        raw = make_weights(local, model["hidden"], model["intermediate_hidden"], config["seed"] + owner)
        result += local_reference(x, ids[selected], weights[selected], *map(unpack_weight, raw), owner * local)
    if model["num_shared_experts"]:
        raw = make_weights(
            model["num_shared_experts"], model["hidden"], model["intermediate_hidden"], config["seed"] + 100000
        )
        for w1, w2 in zip(*map(unpack_weight, raw)):
            result += expert_forward(x, w1, w2)
    return bf16(result), counts[source * local : (source + 1) * local], selected


@pytest.mark.parametrize("routing", ["balanced", "random", "skewed"])
@pytest.mark.parametrize("check_tokens", [0, 3])
@pytest.mark.parametrize("shared", [0, 1])
def test_parallel_chunked_matches_old_oracle(config, routing, check_tokens, shared):
    config["model"]["num_shared_experts"] = shared
    spec = golden.contract(config, 2, 7, "fp8", routing, check_tokens)
    serial = golden.generate(config, spec, [0, 1], workers=1, chunk_tokens=256)
    parallel = golden.generate(config, spec, [0, 1], workers=3, chunk_tokens=2)
    for rank in (0, 1):
        expected = naive(config, 2, rank, 7, routing, check_tokens)
        for a, b, c in zip(serial[rank], parallel[rank], expected):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(a, c, atol=1e-8)
    if routing == "skewed":
        assert not parallel[1][1].any()  # 无本地专家仍需校验本 rank 收到的输出。


def test_cache_reuse_invalidation_and_corruption(config, tmp_path, monkeypatch):
    spec = golden.prepare(config, 2, 7, [0, 1], tmp_path)
    original = golden.load(tmp_path, spec, 0)
    monkeypatch.setattr(golden, "generate", lambda *a, **k: pytest.fail("cached Golden recomputed"))
    golden.prepare(config, 2, 7, [0, 1], tmp_path, workers=2)
    changed = copy.deepcopy(config)
    changed.update(warmup=20, iters=30)
    assert golden.contract(changed, 2, 7, "bf16", "balanced", 0) == spec
    changed["seed"] += 1
    with pytest.raises(FileNotFoundError, match="Golden missing"):
        golden.load(tmp_path, golden.contract(changed, 2, 7, "bf16", "balanced", 0), 0)
    path = golden.cache_path(tmp_path, spec, 0)
    with np.load(path, allow_pickle=False) as saved:
        metadata = saved["metadata"]
    tampered = original[0].copy()
    tampered[0, 0] += 1
    np.savez(path, metadata=metadata, output=tampered, counts=original[1], selected=original[2])
    with pytest.raises(ValueError, match="checksum"):
        golden.prepare(config, 2, 7, [0, 1], tmp_path)


def test_partial_cache_only_computes_missing_ranks(config, tmp_path, monkeypatch):
    spec = golden.prepare(config, 2, 7, [0], tmp_path)
    original = golden.generate
    seen = []

    def generate(config, spec, ranks, **kwargs):
        seen.append(ranks)
        return original(config, spec, ranks, **kwargs)

    monkeypatch.setattr(golden, "generate", generate)
    golden.prepare(config, 2, 7, [0, 1], tmp_path)
    assert seen == [[1]]
    np.testing.assert_array_equal(golden.load(tmp_path, spec, 1)[0], naive(config, 2, 1, 7, "balanced", 0)[0])


def test_routing_source_change_invalidates_cached_golden(config, tmp_path, monkeypatch):
    spec = golden.prepare(config, 2, 7, [0], tmp_path)
    original = golden.inspect.getsource
    # The real previous/current input generators have different source hashes.
    # Simulate that change to ensure the same model/seed/"balanced" name cannot
    # accidentally reuse Golden computed for a different route order.
    monkeypatch.setattr(
        golden.inspect,
        "getsource",
        lambda obj: original(obj) + ("\n# different input routing\n" if obj is inputs else ""),
    )
    changed = golden.contract(config, 2, 7, "bf16", "balanced", 0)
    assert changed["reference_sha256"] != spec["reference_sha256"]
    assert golden.cache_path(tmp_path, changed, 0) != golden.cache_path(tmp_path, spec, 0)
    with pytest.raises(FileNotFoundError, match="Golden missing"):
        golden.load(tmp_path, changed, 0)


def test_legacy_mixed_error_boundaries():
    expected = np.ones(100, dtype=np.float32)
    actual = expected.copy()
    actual[:5] = 1.05
    assert golden.compare(actual, expected)["passed"]  # 恰好 5% 元素超过 1%。
    actual[5] = 1.05
    assert not golden.compare(actual, expected)["passed"]
    actual[:] = 1
    actual[0] = 2
    assert not golden.compare(actual, expected)["passed"]  # 少数大误差也不能被均值隐藏。
    assert golden.compare(np.array([0.009]), np.zeros(1))["passed"]
    for bad in (np.nan, np.inf, -np.inf):
        assert not golden.compare(np.array([bad]), np.zeros(1))["passed"]
    assert not golden.compare(np.zeros(2), np.zeros(1))["passed"]


def test_accuracy_uses_only_compact_control_report(config, tmp_path):
    spec = golden.prepare(config, 2, 7, [0], tmp_path, input_kind="fp8")
    output, counts, _ = golden.load(tmp_path, spec, 0)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Group:
        def all_gather(self, report, stage):
            assert len(json.dumps(report)) < 2048
            return [report]

        def reduce_sum(self, *args):
            pytest.fail("large tensor Store reduction")

    def check(values, recv, cumulative):
        return accuracy(
            config,
            0,
            2,
            7,
            Tensor(values),
            Tensor(recv),
            0,
            cumulative_stats=Tensor(cumulative),
            group=Group(),
            golden_dir=tmp_path,
            input_kind="fp8",
        )[0]

    assert check(output, counts, 7 + 2 * counts)["passed"]
    assert not check(output, counts + 1, 7 + 2 * counts)["passed"]
    assert not check(output, counts, 2 * counts)["passed"]
    assert not check(output + 10, counts, 7 + 2 * counts)["passed"]


def test_incomplete_atomic_save_is_not_published(config, tmp_path, monkeypatch):
    spec = golden.contract(config, 2, 7, "bf16", "balanced", 0)
    arrays = naive(config, 2, 0, 7, "balanced", 0)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(golden.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        golden.save(tmp_path, spec, 0, *arrays)
    assert not golden.cache_path(tmp_path, spec, 0).exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_cpu_only_cli_cache_is_readable_by_benchmark(config, tmp_path):
    case = tmp_path / "case.json"
    case.write_text(json.dumps(config), encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    # 新解释器验证 -m 入口及线程环境，不依赖 pytest 已加载的 NumPy/Torch。
    program = (
        "import runpy, sys; "
        f"sys.path[:0] = {[str(root), str(root / 'experiments/megamoe/python')]!r}; "
        "runpy.run_module('deep_ep_experimental.megamoe.golden', run_name='__main__'); "
        "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"
    )
    command = [
        sys.executable,
        "-c",
        program,
        "--config",
        str(case),
        "--ep-size",
        "2",
        "--ranks",
        "0-1",
        "--input",
        "fp8",
        "--golden-dir",
        str(tmp_path / "cache"),
        "--workers",
        "2",
        "--chunk-tokens",
        "2",
    ]
    for marker in ("saved", "reuse"):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, env=dict(os.environ, OMP_NUM_THREADS="1")
        )
        assert result.returncode == 0, result.stderr
        assert marker in result.stdout
    spec = golden.contract(config, 2, 7, "fp8", "balanced", 0)
    np.testing.assert_array_equal(golden.load(tmp_path / "cache", spec, 0)[0], naive(config, 2, 0, 7, "balanced", 0)[0])
