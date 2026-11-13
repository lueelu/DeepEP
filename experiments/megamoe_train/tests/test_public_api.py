# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""CPU tests for the public training bridge, with a stub native executor."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "python"))
    return importlib.import_module("deep_ep_experimental.megamoe_train.api")


@pytest.fixture
def operator(api):
    calls = []

    def forward(x, ids, weights, w1, w2, capacity, limit, mode, out=None):
        calls.append((weights.dtype, capacity, limit, mode, out is not None))
        if out is None:
            out = x.clone()
        else:
            out.copy_(x)
        return SimpleNamespace(
            y=out,
            counts=torch.full((w1.shape[0],), 7, dtype=torch.int32),
            saved_state=torch.zeros(1, dtype=torch.uint8),
            dispatch_output=torch.zeros(1, dtype=torch.uint8),
        )

    def backward(dy, weights, w1, w2, *args):
        return SimpleNamespace(
            dx=dy,
            dtopk_weights=torch.full_like(weights, 2),
            dw1=torch.full_like(w1, 3),
            dw2=torch.full_like(w2, 4),
        )

    op = api.MegaMoeBF16Train.__new__(api.MegaMoeBF16Train)
    torch.nn.Module.__init__(op)
    op._check_ready = lambda: None
    op._runtime = SimpleNamespace(validate_topk_weight_type=lambda *args: None)
    op._executor = SimpleNamespace(
        forward=forward,
        forward_out=lambda out, *args: forward(*args, out=out),
        backward=backward,
    )
    return op, calls


def inputs(dtype):
    x = torch.ones((3, 8), dtype=torch.bfloat16, requires_grad=True)
    ids = torch.zeros((3, 2), dtype=torch.int32)
    weights = torch.full((3, 2), 0.5, dtype=dtype, requires_grad=True)
    w1 = torch.ones((2, 8, 8), dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.ones((2, 8, 4), dtype=torch.bfloat16, requires_grad=True)
    return x, ids, weights, w1, w2


def test_public_exports_include_training_entry_and_legacy_module(api):
    package = importlib.import_module("deep_ep_experimental.megamoe_train")
    assert package.bf16_mega_moe_train is api.bf16_mega_moe_train
    assert package.SymmBuffer is package.MegaMoeBF16Train is api.MegaMoeBF16Train
    assert set(package.__all__) == {"bf16_mega_moe_train", "SymmBuffer", "MegaMoeBF16Train"}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["bf16", "none", "mxfp8", "auto"])
def test_output_first_api_retains_gradients_options_and_cumulative_counts(api, operator, dtype, mode):
    op, calls = operator
    x, ids, weights, w1, w2 = inputs(dtype)
    y = torch.empty_like(x, requires_grad=False)
    pointer = y.data_ptr()
    stats = torch.full((2,), 5, dtype=torch.int32)
    assert api.SymmBuffer is api.MegaMoeBF16Train
    assert (
        api.bf16_mega_moe_train(
            y,
            w1,
            w2,
            op,
            cumulative_local_expert_recv_stats=stats,
            x=x,
            topk_idx=ids,
            topk_weights=weights,
            max_recv_tokens=32,
            save_gmm1_output=mode,
            swiglu_limit=0.25,
        )
        is None
    )
    assert y.data_ptr() == pointer and torch.equal(y, x)
    assert torch.equal(stats, torch.full_like(stats, 12))
    y.float().sum().backward()
    for tensor, expected in ((x, 1), (weights, 2), (w1, 3), (w2, 4)):
        assert tensor.grad.dtype == tensor.dtype
        assert torch.equal(tensor.grad, torch.full_like(tensor, expected))
    assert calls == [(dtype, 32, 0.25, mode, True)]


def test_legacy_module_call_still_returns_y_and_counts(api, operator):
    op, calls = operator
    x, ids, weights, w1, w2 = inputs(torch.bfloat16)
    y, counts = op(x, ids, weights, w1, w2, 32)
    assert not counts.requires_grad and counts.tolist() == [7, 7]
    y.float().sum().backward()
    assert all(value.grad is not None for value in (x, weights, w1, w2))
    assert calls == [(torch.bfloat16, 32, 0.0, "bf16", False)]


def test_queued_outputs_keep_independent_graphs_and_topk_dtypes(api, operator):
    op, _ = operator
    pending = []
    for dtype in (torch.bfloat16, torch.float32):
        x, ids, weights, w1, w2 = inputs(dtype)
        y = torch.empty_like(x, requires_grad=False)
        api.bf16_mega_moe_train(
            y,
            w1,
            w2,
            op,
            x=x,
            topk_idx=ids,
            topk_weights=weights,
            max_recv_tokens=32,
        )
        pending.append((y, x, weights))
    for y, x, weights in reversed(pending):
        y.sum().backward()
        assert weights.grad.dtype == weights.dtype
        assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("out_kind", ["alias", "wrong_dtype", "wrong_shape", "noncontiguous", "requires_grad"])
def test_invalid_output_is_rejected_before_native_launch(api, operator, out_kind):
    op, calls = operator
    x, ids, weights, w1, w2 = inputs(torch.float32)
    out = {
        "alias": x.detach(),
        "wrong_dtype": torch.empty_like(x, dtype=torch.float32),
        "wrong_shape": torch.empty((2, 8), dtype=x.dtype),
        "noncontiguous": torch.empty((8, 3), dtype=x.dtype).transpose(0, 1),
        "requires_grad": torch.empty_like(x, requires_grad=True),
    }[out_kind]
    with pytest.raises(ValueError):
        api.bf16_mega_moe_train(out, w1, w2, op, x=x, topk_idx=ids, topk_weights=weights, max_recv_tokens=32)
    assert not calls


@pytest.mark.parametrize(
    "options, error",
    [
        ({"shared_l1_weights": torch.ones(1)}, NotImplementedError),
        ({"shared_l2_weights": torch.ones(1)}, NotImplementedError),
        ({"shared_l1_weights": torch.ones(1), "shared_l2_weights": torch.ones(1)}, NotImplementedError),
        ({"activation": "relu"}, NotImplementedError),
        ({"fast_math": False}, NotImplementedError),
        ({"swiglu_limit": 0.5, "activation_clamp": 1.0}, ValueError),
        ({"swiglu_limit": float("nan")}, ValueError),
        ({"save_gmm1_output": "fp32"}, ValueError),
        ({"max_recv_tokens": 0}, ValueError),
        ({"cumulative_local_expert_recv_stats": torch.zeros(2)}, ValueError),
    ],
)
def test_invalid_options_are_rejected_before_native_launch(api, operator, options, error):
    op, calls = operator
    x, ids, weights, w1, w2 = inputs(torch.float32)
    kwargs = {"max_recv_tokens": 32, **options}
    with pytest.raises(error):
        api.bf16_mega_moe_train(
            torch.empty_like(x, requires_grad=False),
            w1,
            w2,
            op,
            x=x,
            topk_idx=ids,
            topk_weights=weights,
            **kwargs,
        )
    assert not calls


def test_deepgemm_activation_clamp_maps_to_training_limit(api, operator):
    op, calls = operator
    x, ids, weights, w1, w2 = inputs(torch.float32)
    api.bf16_mega_moe_train(
        torch.empty_like(x, requires_grad=False),
        w1,
        w2,
        op,
        activation_clamp=0.125,
        x=x,
        topk_idx=ids,
        topk_weights=weights,
        max_recv_tokens=32,
    )
    assert calls[0][2] == 0.125
