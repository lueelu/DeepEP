# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""Unified BF16 bridge: real autograd with a CPU native Plan double."""

import importlib
import inspect
import threading
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
api = importlib.import_module("deep_ep_experimental.megamoe.api")
bf16 = importlib.import_module("deep_ep_experimental.megamoe._bf16")


@pytest.fixture
def buffer(monkeypatch):
    calls, plans = [], []
    stream = SimpleNamespace(wait_stream=lambda other: None)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *a: None)
    monkeypatch.setattr(api, "npu_api", lambda: SimpleNamespace(current_stream=lambda device: stream))

    class Plan:
        def __init__(self, address, capacity, rank, world_size, like, weight_before_gmm2, ranks_per_node):
            self.before = weight_before_gmm2
            self.ranks_per_node = ranks_per_node
            plans.append(self)

        def prepare(self, *args):
            self.spec = args

        def forward(self, out, x, ids, p, w1, w2, capacity, limit, mode, capture_timer, save_dispatched_x=True):
            calls.append((p.dtype, mode, self.before, limit))
            out.copy_(x)
            return SimpleNamespace(
                y=out,
                counts=torch.full((2,), 7, dtype=torch.int32),
                saved_state=torch.zeros(1, dtype=torch.uint8),
                dispatch_output=torch.zeros(1, dtype=torch.uint8) if save_dispatched_x else None,
                timer=torch.zeros(1, dtype=torch.int64) if capture_timer else torch.Tensor(),
            )

        def backward(self, dy, p, w1, w2, counts, saved, dispatch, capacity, limit, mode, capture_timer):
            assert mode == self.spec[-2]
            return SimpleNamespace(
                dx=dy,
                dtopk_weights=torch.full_like(p, 2),
                dw1=torch.full_like(w1, 3),
                dw2=torch.full_like(w2, 4),
                timer=torch.ones(1, dtype=torch.int64) if capture_timer else torch.Tensor(),
            )

    monkeypatch.setattr(api, "_extension", lambda: SimpleNamespace(BF16Plan=Plan))
    monkeypatch.setattr(bf16, "require_native", lambda _: SimpleNamespace(megamoe_workspace_address=lambda _: 4096))
    b = api.SymmBuffer.__new__(api.SymmBuffer)
    b._device = torch.device("cpu")
    b._lock = threading.Lock()
    b.group = SimpleNamespace(size=lambda: 2)
    b.num_experts = 4
    b.num_max_tokens_per_rank = 3
    b.num_topk = 2
    b.hidden = 8
    b.intermediate_hidden = 4
    b.max_recv_tokens = 32
    b.save_dispatched_x = True
    b.dispatch_rebuilder = None
    b.need_grad = True
    b.ranks_per_node = 0
    b.gmm1_save_type = "bf16"
    b.mma_type = "bf16xbf16"
    b._plan = b._plan_key = b._plan_bs = None
    b._runtime = object.__new__(api._MegaRuntime)
    b._runtime.closed = True
    b._runtime.ensure_alive = lambda: None
    b._runtime.native = SimpleNamespace(capacity=2 << 20)
    b._runtime.rank, b._runtime.world_size = 0, 2
    b._runtime.device, b._runtime.stream = 0, stream
    return b, calls, plans


def inputs(b, dtype=torch.float32, grad=True):
    x = torch.ones(3, 8, dtype=torch.bfloat16, requires_grad=grad)
    p = torch.full((3, 2), 0.5, dtype=dtype, requires_grad=grad)
    b.x = x
    b.topk_idx = torch.zeros(3, 2, dtype=torch.int32)
    b.topk_weights = p
    w1 = torch.ones(2, 8, 8, dtype=torch.bfloat16, requires_grad=grad)
    w2 = torch.ones(2, 8, 4, dtype=torch.bfloat16, requires_grad=grad)
    return x, p, w1, w2


def call(b, w1, w2, **kwargs):
    y = torch.empty(3, 8, dtype=torch.bfloat16)
    assert api.bf16_mega_moe(y, w1, w2, b, **kwargs) is None
    return y


def test_public_signature_and_export():
    assert inspect.signature(api.SymmBuffer).parameters["save_dispatched_x"].default is True
    assert list(inspect.signature(api.bf16_mega_moe).parameters) == [
        "y",
        "l1_weights",
        "l2_weights",
        "sym_buffer",
        "shared_l1_weights",
        "shared_l2_weights",
        "cumulative_local_expert_recv_stats",
        "activation",
        "activation_clamp",
        "fast_math",
    ]
    package = importlib.import_module("deep_ep_experimental.megamoe")
    assert package.bf16_mega_moe is api.bf16_mega_moe


def test_bf16_waits_only_when_caller_changes_stream(buffer, monkeypatch):
    b, _, _ = buffer
    waits = []
    first = b._runtime.stream
    first.wait_stream = lambda previous: waits.append((first, previous))
    second = SimpleNamespace(wait_stream=lambda previous: waits.append((second, previous)))
    current = first
    monkeypatch.setattr(api, "npu_api", lambda: SimpleNamespace(current_stream=lambda device: current))
    _, _, w1, w2 = inputs(b)
    y = call(b, w1, w2)
    current = second
    call(b, w1, w2)
    call(b, w1, w2)
    current = first
    y.sum().backward()
    assert waits == [(second, first), (first, second)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["bf16", "none", "mxfp8"])
def test_original_tensor_gradients_inplace_output_counts_and_save_type(buffer, dtype, mode):
    b, calls, plans = buffer
    b.gmm1_save_type = mode
    x, p, w1, w2 = inputs(b, dtype)
    stats = torch.zeros(2, dtype=torch.int32)
    y = call(b, w1, w2, cumulative_local_expert_recv_stats=stats, activation_clamp=0.25)
    y.sum().backward()
    assert calls == [(dtype, mode, True, 0.25)] and stats.tolist() == [7, 7]
    assert torch.equal(x.grad, torch.ones_like(x)) and p.grad.dtype == dtype
    assert torch.equal(p.grad, torch.full_like(p, 2))
    assert torch.equal(w1.grad, torch.full_like(w1, 3)) and torch.equal(w2.grad, torch.full_like(w2, 4))


def test_queued_graphs_snapshot_plan_dtype_and_save_type(buffer):
    b, calls, plans = buffer
    pending = []
    for dtype, mode in [(torch.bfloat16, "bf16"), (torch.float32, "none"), (torch.float32, "mxfp8")]:
        b.gmm1_save_type = mode
        x, p, w1, w2 = inputs(b, dtype)
        pending.append((call(b, w1, w2), x, p, w1, w2))
    assert len(plans) == 3
    for y, x, p, w1, w2 in reversed(pending):
        y.sum().backward()
        assert p.grad.dtype == p.dtype and torch.equal(x.grad, torch.ones_like(x))


def test_external_dispatch_is_bound_per_microbatch(buffer):
    b, _, _ = buffer
    b.save_dispatched_x = False
    _, _, w1, w2 = inputs(b)
    with pytest.raises(ValueError, match="dispatch_rebuilder"):
        call(b, w1, w2)

    seen = []

    def graph(index):
        def rebuild():
            seen.append(index)
            return torch.full((1,), index, dtype=torch.uint8)

        b.dispatch_rebuilder = rebuild
        return call(b, w1, w2)

    first, second = graph(1), graph(2)
    b.dispatch_rebuilder = None
    second.sum().backward()
    first.sum().backward()
    assert seen == [2, 1]


def test_destroy_releases_prepared_bf16_plan_and_inputs(buffer):
    b, _, _ = buffer
    _, _, w1, w2 = inputs(b)
    call(b, w1, w2)
    assert b._plan is not None
    calls = []
    b._runtime.close = lambda: calls.append("close")
    b.destroy()
    assert calls == ["close"]
    assert b._plan is b._plan_key is b._plan_bs is None
    assert b.x is b.topk_idx is b.topk_weights is None


def test_forward_and_prepare_validate_routing_inputs_once(buffer, monkeypatch):
    b, calls, _ = buffer
    _, _, w1, w2 = inputs(b)
    checked = []
    validate = bf16.validate_buffer_inputs

    def track(owner, bs):
        checked.append(bs)
        validate(owner, bs)

    monkeypatch.setattr(bf16, "validate_buffer_inputs", track)

    def check(group, stage, error, *args):
        if error:
            raise ValueError(error)

    monkeypatch.setattr(api, "collective_check", check)
    b.prepare(3)
    call(b, w1, w2)
    assert checked == [3, 3]
    b.topk_weights = b.topk_weights.to(torch.float64)
    with pytest.raises(ValueError, match="topk_weights"):
        call(b, w1, w2)
    with pytest.raises(ValueError, match="topk_weights"):
        b.prepare(3)
    assert len(calls) == 1


@pytest.mark.parametrize("need_grad", [False, True])
def test_bf16_plan_uses_buffer_peer_topology(buffer, need_grad):
    b, _, plans = buffer
    b.need_grad = need_grad
    _, _, w1, w2 = inputs(b, grad=need_grad)
    call(b, w1, w2)
    b.ranks_per_node = 1
    call(b, w1, w2)
    call(b, w1, w2)
    assert [plan.ranks_per_node for plan in plans] == [0, 1]


def test_prepared_plan_is_reused_and_no_grad_keeps_preweight_semantics(buffer):
    b, calls, plans = buffer
    x, p, w1, w2 = inputs(b)
    b.prepare = lambda bs: b._prepare(bs)
    b.prepare(3)
    y = call(b, w1, w2)
    y.sum().backward()
    assert len(plans) == 1
    with torch.no_grad():
        y = call(b, w1, w2)
    assert not y.requires_grad and calls[-1][1:3] == ("none", True)


def test_inference_is_postweighted_and_rejects_requested_gradient(buffer):
    b, calls, plans = buffer
    b.need_grad = False
    x, p, w1, w2 = inputs(b, grad=False)
    b.gmm1_save_type = "mxfp8"
    b._prepare(3)
    assert plans[0].spec[-2] == "none"
    y = call(b, w1, w2)
    assert len(plans) == 1
    assert calls[-1][1:3] == ("none", False) and not y.requires_grad
    x.requires_grad_()
    with pytest.raises(ValueError, match="need_grad=True"):
        call(b, w1, w2)


@pytest.mark.parametrize("kind", ["alias", "shape", "dtype", "strided", "grad"])
def test_invalid_output_rejected_before_launch(buffer, kind):
    b, calls, _ = buffer
    x, p, w1, w2 = inputs(b)
    y = {
        "alias": x.detach(),
        "shape": torch.empty(2, 7, dtype=x.dtype),
        "dtype": torch.empty(3, 8),
        "strided": torch.empty(8, 3, dtype=x.dtype).T,
        "grad": torch.empty(3, 8, dtype=x.dtype, requires_grad=True),
    }[kind]
    with pytest.raises(ValueError):
        api.bf16_mega_moe(y, w1, w2, b)
    assert not calls


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"shared_l1_weights": torch.ones(1)}, NotImplementedError),
        ({"shared_l2_weights": torch.ones(1)}, NotImplementedError),
        ({"activation": "relu"}, NotImplementedError),
        ({"fast_math": False}, NotImplementedError),
        ({"activation_clamp": float("nan")}, ValueError),
        ({"activation_clamp": -1}, ValueError),
    ],
)
def test_invalid_options_rejected(buffer, kwargs, error):
    b, calls, _ = buffer
    x, p, w1, w2 = inputs(b)
    with pytest.raises(error):
        call(b, w1, w2, **kwargs)
    assert not calls


def test_capture_state_belongs_to_its_graph(buffer, monkeypatch):
    b, calls, _ = buffer
    x, p, w1, w2 = inputs(b)
    package = importlib.import_module("deep_ep_experimental.megamoe")
    monkeypatch.setattr(package, "_C", SimpleNamespace(timer_enabled=True), raising=False)
    from deep_ep_experimental.megamoe.timer.bf16.capture import TimerCapture

    with TimerCapture(b) as capture:
        y = call(b, w1, w2)
    assert b._timer_capture is None
    y.sum().backward()
    assert capture.forward is not None and capture.backward is not None


def test_active_batch_within_capacity_backpropagates_to_original_rows(buffer):
    b, calls, _ = buffer
    x, p, w1, w2 = inputs(b)
    y = torch.empty(2, 8, dtype=torch.bfloat16)
    api.bf16_mega_moe(y, w1, w2, b)
    y.sum().backward()
    assert torch.equal(x.grad[:2], torch.ones_like(x[:2]))
    assert torch.count_nonzero(x.grad[2:]) == 0
    assert torch.equal(p.grad[:2], torch.full_like(p[:2], 2))
    assert torch.count_nonzero(p.grad[2:]) == 0


@pytest.mark.parametrize("chosen", range(4))
def test_only_requested_leaf_receives_gradient(buffer, chosen):
    b, _, _ = buffer
    values = inputs(b, grad=False)
    values[chosen].requires_grad_()
    call(b, values[2], values[3]).sum().backward()
    for i, value in enumerate(values):
        assert (value.grad is not None) == (i == chosen)


def test_fp32_router_gradient_stays_connected_to_original_logits(buffer):
    b, _, _ = buffer
    _, _, w1, w2 = inputs(b, grad=False)
    logits = (torch.arange(12, dtype=torch.float32).reshape(3, 4) / 4).requires_grad_()
    p, ids = logits.softmax(dim=-1).topk(2, dim=-1)
    p.retain_grad()
    b.topk_idx = ids.to(torch.int32)
    b.topk_weights = p
    call(b, w1, w2).sum().backward()

    reference = logits.detach().clone().requires_grad_()
    (2 * reference.softmax(dim=-1).topk(2, dim=-1).values.sum()).backward()
    assert torch.equal(p.grad, torch.full_like(p, 2))
    assert torch.count_nonzero(logits.grad) > 0
    assert torch.equal(logits.grad, reference.grad)
