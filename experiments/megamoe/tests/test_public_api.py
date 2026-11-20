# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Execute public wrappers using CPU storage doubles; this does not simulate the NPU kernel."""

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api
from deep_ep_experimental.megamoe.validation import collective_check


class Device(str):
    @property
    def type(self):
        return self.split(":")[0]


DTYPES = dict(
    uint8=np.uint8,
    int8=np.int8,
    int32=np.int32,
    int64=np.int64,
    float32=np.float32,
    bfloat16=np.float32,
    float8_e4m3fn=np.uint8,
    float8_e8m0fnu=np.uint8,
)


class Tensor:
    def __init__(self, data, dtype, device="npu:0"):
        self.data, self.dtype, self.device = np.asarray(data, dtype=DTYPES[dtype]), dtype, Device(device)
        self.requires_grad = False
        self.streams = []
        self.npu_format = 2
        self._version = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, index):
        result = Tensor(self.data[index], self.dtype, self.device)
        result.npu_format = self.npu_format
        return result

    def view(self, dtype):
        return Tensor(self.data.view(DTYPES[dtype]), dtype, self.device)

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape), self.dtype, self.device)

    def transpose(self, first, second):
        return Tensor(self.data.swapaxes(first, second), self.dtype, self.device)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data), self.dtype, self.device)

    def clone(self):
        return Tensor(self.data.copy(), self.dtype, self.device)

    def is_contiguous(self):
        return self.data.flags.c_contiguous

    def untyped_storage(self):
        base = self.data
        while isinstance(base.base, np.ndarray):
            base = base.base
        return SimpleNamespace(data_ptr=lambda: base.ctypes.data, nbytes=lambda: base.nbytes)

    def data_ptr(self):
        return self.data.ctypes.data

    def storage_offset(self):
        return (self.data_ptr() - self.untyped_storage().data_ptr()) // self.data.itemsize

    def numel(self):
        return self.data.size

    def element_size(self):
        return self.data.itemsize

    def stride(self):
        return tuple(n // self.data.itemsize for n in self.data.strides)

    def unsqueeze(self, axis):
        result = Tensor(np.expand_dims(self.data, axis), self.dtype, self.device)
        result.npu_format = self.npu_format
        return result

    def is_inference(self):
        return False

    def copy_(self, other, non_blocking=False):
        np.copyto(self.data, other.data)
        self._version += 1
        return self

    def record_stream(self, stream):
        self.streams.append(stream)

    def cpu(self):
        return self

    def tolist(self):
        return self.data.tolist()

    def add_(self, other):
        self.data += other.data
        self._version += 1
        return self


@pytest.fixture
def backend(monkeypatch):
    events = []

    class Group:
        def __init__(self, size=2):
            self._size = size

        def size(self):
            return self._size

    def gather(output, value, **kwargs):
        output[:] = [value] * len(output)

    dist = SimpleNamespace(ProcessGroup=Group, get_world_size=lambda group: group.size(), all_gather_object=gather)
    torch = SimpleNamespace(
        Tensor=Tensor,
        **{name: name for name in DTYPES},
        device=Device,
        empty=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=DTYPES[dtype]), dtype, device),
        zeros=lambda shape, dtype, device: Tensor(np.zeros(shape), dtype, device),
        empty_like=lambda t, dtype=None, device=None: Tensor(
            np.empty(t.shape, dtype=DTYPES[dtype or t.dtype]), dtype or t.dtype, device or t.device
        ),
        tensor=lambda data, dtype, device, pin_memory=False: Tensor(data, dtype, device),
        distributed=dist,
        inference_mode=lambda _: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    def format_cast(t, acl_format, *, customize_dtype, input_dtype):
        # 检查实际传入的协议，不能把所有 format_cast 参数都静默忽略。
        assert t.dtype == "uint8" and acl_format in (2, 29)
        if acl_format == 2:
            assert t.npu_format == 50
        assert customize_dtype == "float8_e4m3fn" and input_dtype == "float4_e2m1fn_x2"
        events.append(("format_cast", t))
        t = Tensor(t.data.copy(), t.dtype, t.device)
        t.npu_format = 2 if acl_format == 2 else 50  # FP4 打包成 uint8 后，输出码与请求的29不同。
        return t

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda t: t.npu_format, npu_format_cast=format_cast, float4_e2m1fn_x2="float4_e2m1fn_x2"
        ),
    )
    stream = SimpleNamespace(wait_stream=lambda _: events.append("wait"), synchronize=lambda: events.append("sync"))
    npu = SimpleNamespace(
        Event=lambda: SimpleNamespace(record=lambda stream: None, query=lambda: True),
        is_current_stream_capturing=lambda: False,
        device=lambda _: contextlib.nullcontext(),
        current_device=lambda: 0,
        current_stream=lambda _: stream,
        stream=lambda _: contextlib.nullcontext(),
    )
    monkeypatch.setattr(api, "npu_api", lambda: npu)
    monkeypatch.setattr(
        api,
        "require_native",
        lambda _: SimpleNamespace(
            megamoe_workspace_address=lambda _: 4096,
            megamoe_combine_reduce_version=4,
            megamoe_combine_qps=16,
            megamoe_dispatch_qps=8,
            megamoe_dispatch_workers=14,
        ),
    )

    class Runtime:
        def __init__(self, group, capacity, timeout, *, ranks_per_node=0):
            self.device, self.stream, self.native = 0, stream, SimpleNamespace(capacity=capacity)
            self.ranks_per_node = ranks_per_node
            self.closed = False

        def ensure_alive(self):
            if self.closed:
                raise RuntimeError("Buffer runtime has been destroyed")

        def close(self):
            events.append("close")
            self.closed = True

    monkeypatch.setattr(api, "_MegaRuntime", Runtime)

    class Plan:
        def __init__(self, example, *args):
            self.bs, self.workspace_bytes = example.shape[0], 4096
            self.options = args
            events.append(("prepare", self.bs))

        def forward(self, *args):
            events.append(("launch", args))
            output = args[-1]
            output.data.fill(3)
            return output, Tensor([2, 3], "int32")

    ext = SimpleNamespace(
        buffer_size=lambda *args: 2 << 20,
        Plan=Plan,
        ffts_config=lambda: 123,
        timer_enabled=True,
        timer_numel=64,
        small_batch_max_tokens=256,
        hierarchical_max_tokens=256,
        combine_reduce_version=4,
        combine_qps=16,
        dispatch_qps=8,
        dispatch_workers=14,
    )
    monkeypatch.setattr(api, "_extension", lambda: ext)
    from deep_ep_experimental.megamoe import _platform_api as platform_api

    ep32_ext = ext
    monkeypatch.setattr(platform_api, "npu_api", api.npu_api)
    monkeypatch.setattr(platform_api, "require_native", api.require_native)
    monkeypatch.setattr(platform_api, "_MegaRuntime", lambda group, capacity, timeout, **kw: Runtime(group, capacity, timeout))
    class Plan:
        def __init__(
            self,
            example,
            intermediate,
            topk,
            experts,
            world,
            max_tokens,
            is_a8w4,
            shared_experts,
            ranks_per_node,
            ffts,
            address,
            capacity,
            backend="mte",
            topk_weights_type=0,
        ):
            self.bs, self.workspace_bytes = example.shape[0], 4096
            self.backend = backend
            self.topk_weights_type = topk_weights_type
            self.local_experts = experts
            events.append(("prepare", self.bs))
            events.append(("plan_backend", backend))

        def forward(
            self,
            x,
            scales,
            ids,
            route_weights,
            w1,
            w2,
            s1,
            s2,
            timer,
            shared_w1,
            shared_w2,
            shared_s1,
            shared_s2,
            out,
            generation=0,
            activation_clamp=None,
        ):
            args = (
                x,
                scales,
                ids,
                route_weights,
                w1,
                w2,
                s1,
                s2,
                timer,
                shared_w1,
                shared_w2,
                shared_s1,
                shared_s2,
                out,
                generation,
            )
            events.append(("launch", args))
            events.append(("activation_clamp", activation_clamp))
            output = out if out is not None else Tensor(np.zeros(x.shape), "bfloat16")
            output.data.fill(3)
            if self.backend == "urma" and timer is not None:
                timer.data[-1] = generation
            return output, Tensor(np.arange(self.local_experts) + 2, "int32")

    def buffer_size(*args, backend="mte", topk_weights_type=0):
        events.append(("buffer_size", args, backend))
        return 2 << 20

    ext = SimpleNamespace(
        buffer_size=buffer_size,
        Plan=Plan,
        ffts_config=lambda: 123,
        timer_enabled=True,
        timer_numel=64,
        allocate_timer=lambda example, backend="mte": Tensor(np.zeros(65536 if backend == "urma" else 64), "int64"),
        small_batch_max_tokens=256,
    )
    monkeypatch.setattr(platform_api, "_extension", lambda: platform_ext)
    platform_ext = ext
    ext = ep32_ext
    buffer = api.SymmBuffer(Group(), 4, 17, 2, 1024, 256)
    buffer.topk_idx.data[:] = [0, 1]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    yield SimpleNamespace(buffer=buffer, events=events, ext=ext, platform_ext=platform_ext, dist=dist, group_type=Group)
    buffer.destroy()


def weights():
    prepared = (
        (Tensor(np.zeros((2, 512, 512)), "uint8"), Tensor(np.full((2, 512, 16, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 1024, 128)), "uint8"), Tensor(np.full((2, 1024, 4, 2), 127), "float8_e8m0fnu")),
    )
    for weight, _ in prepared:
        weight.npu_format = 50
    return prepared


def fp8_weights():
    return (
        (Tensor(np.zeros((2, 512, 1024)), "float8_e4m3fn"), Tensor(np.full((2, 512, 16, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 1024, 256)), "float8_e4m3fn"), Tensor(np.full((2, 1024, 4, 2), 127), "float8_e8m0fnu")),
    )


@pytest.mark.parametrize("roce", [False, True])
def test_benchmark_helpers_use_live_owner_and_select_shmem_transport(backend, roce):
    log = []
    backend.ext.benchmark_sleep = lambda cycles: log.append(("sleep", cycles))
    backend.ext.benchmark_barrier = lambda selected: log.append(("barrier", selected))
    backend.buffer._hierarchical_roce = roce
    backend.buffer.benchmark_sleep(20_000_000)
    backend.buffer.benchmark_barrier()
    assert log == [("sleep", 20_000_000), ("barrier", roce)]
    assert not backend.events  # No host completion/teardown or MegaMoE call.
    backend.buffer._runtime.closed = True
    with pytest.raises(RuntimeError, match="destroyed"):
        backend.buffer.benchmark_barrier()
    assert len(log) == 2


def test_shmem_benchmark_detects_old_wheel_before_capture(backend):
    with pytest.raises(RuntimeError, match="rebuilding"):
        backend.buffer.benchmark_sleep(0)


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
    assert b._plan.bs == b._plan_bs == b.num_max_tokens_per_rank


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
        assert b._plan.bs == b.num_max_tokens_per_rank
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


def test_fp8_weights_remain_nd_without_fp4_format_cast(backend):
    pair = (Tensor(np.zeros((1, 128, 128)), "float8_e4m3fn"), Tensor(np.full((1, 128, 2, 2), 127), "float8_e8m0fnu"))
    for weight, scale in api.transform_weights_for_mega_moe(pair, pair):
        assert weight is pair[0]
        np.testing.assert_array_equal(scale.data, pair[1].data)
    assert not any(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events)


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
        "check:MegaMoE product selection",
        "check:buffer implementation",
        "check:configuration",
        "check:input allocation",
        "check:input preflight",
        "check:prepare",
        "check:prepare BS=17",
    ]


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


def hierarchical_buffer(backend, *, size=32, mma_type="fp8xfp4", capacity=257, **options):
    from deep_ep._store import StoreGroup

    group = StoreGroup(None, 0, size, prefix="hierarchical-api-test")
    group.all_gather = lambda value, stage: [value] * size
    buffer = api.SymmBuffer(
        group,
        size * 2,
        capacity,
        2,
        1024,
        256,
        ranks_per_node=16,
        mma_type=mma_type,
        **options,
    )
    buffer.topk_idx.data[:] = [0, size]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    return buffer


@pytest.mark.parametrize("mma_type", ["fp8xfp4"])
def test_hierarchical_sequence_survives_plan_recreation_and_old_epoch_wrap(backend, mma_type):
    b = hierarchical_buffer(backend, mma_type=mma_type)
    assert b._runtime.ranks_per_node == 16 and b._hierarchical_roce
    pairs = weights() if mma_type == "fp8xfp4" else fp8_weights()
    try:
        for sequence in range(1, 260):
            bs = (1, 31, 32, 33, 138, 144, 255, 256)[sequence % 8]
            api.mega_moe(
                (b.x[:bs], b.x_sf[:bs]),
                b.topk_idx[:bs],
                b.topk_weights[:bs],
                *pairs,
                b,
                out=Tensor(np.zeros((bs, 1024)), "bfloat16"),
            )
            launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
            assert launch[-2] == sequence and b._sequence == sequence
            assert b._plan_bs == bs
        with pytest.raises(ValueError, match=r"BS in \[1,256\]"):
            b.prepare(257)
        assert b._sequence == 259
    finally:
        b.destroy()


def test_hierarchical_old_wheel_retains_bs32_limit(backend):
    del backend.ext.hierarchical_max_tokens
    b = hierarchical_buffer(backend)
    try:
        b.prepare(32)
        with pytest.raises(ValueError, match=r"BS in \[1,32\]"):
            b.prepare(33)
        assert b._plan_bs == 32
    finally:
        b.destroy()


def test_hierarchical_launch_failure_and_sequence_exhaustion(backend, monkeypatch):
    b = hierarchical_buffer(backend)
    try:
        b.prepare(1)
        plan = b._plan
        launch = plan.forward

        def fail(*args):
            raise RuntimeError("launch rejected")

        monkeypatch.setattr(plan, "forward", fail)
        with pytest.raises(RuntimeError, match="launch rejected"):
            api.fp8_fp4_mega_moe(Tensor(np.zeros((1, 1024)), "bfloat16"), *weights(), b)
        assert b._sequence == 0
        monkeypatch.setattr(plan, "forward", launch)
        b._sequence = 0xFFFFFFFE
        api.fp8_fp4_mega_moe(Tensor(np.zeros((1, 1024)), "bfloat16"), *weights(), b)
        assert b._sequence == 0xFFFFFFFF
        count = len(backend.events)
        with pytest.raises(RuntimeError, match="sequence exhausted"):
            api.fp8_fp4_mega_moe(Tensor(np.zeros((1, 1024)), "bfloat16"), *weights(), b)
        assert not any(isinstance(e, tuple) and e[0] == "launch" for e in backend.events[count:])
    finally:
        b.destroy()


@pytest.mark.parametrize("size,mma_type", [(64, "fp8xfp4")])
def test_hierarchical_unsupported_configuration_fails_collectively(backend, size, mma_type):
    with pytest.raises(ValueError, match="MTE\\+RoCE MegaMoE requires EP32"):
        hierarchical_buffer(backend, size=size, mma_type=mma_type)


def test_sixteen_rank_single_host_keeps_mte(backend):
    b = hierarchical_buffer(backend, size=16)
    try:
        assert not b._hierarchical_roce
        b.prepare(32)
        api.fp8_fp4_mega_moe(Tensor(np.zeros((32, 1024)), "bfloat16"), *weights(), b)
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        assert launch[-2] == 0 and b._sequence == 0
    finally:
        b.destroy()


@pytest.mark.parametrize("protocol", list(range(5, 18)))
def test_extension_rejects_native_without_expanded_hierarchical_support(monkeypatch, protocol):
    api._extension.cache_clear()
    core = SimpleNamespace(create_megamoe_runtime=lambda: None)
    extension = SimpleNamespace(api_protocol=protocol)
    monkeypatch.setattr(api, "require_native", lambda _: core)
    monkeypatch.setattr(api.importlib, "import_module", lambda _: extension)
    try:
        if protocol != 17:
            with pytest.raises(RuntimeError, match="versions differ"):
                api._extension()
        else:
            assert api._extension() is extension
    finally:
        api._extension.cache_clear()


def test_hierarchical_plan_has_no_selectable_algorithm_arguments(backend):
    sizes = []
    backend.ext.buffer_size = lambda *args: sizes.append(args) or (2 << 20)
    b = hierarchical_buffer(backend)
    try:
        for bs in (1, 32, 256):
            b.prepare(bs)
            assert len(b._plan.options) == 14
            assert b._plan.options[-5:] == (4096, 2 << 20, (2, 257, 2, 1024), 0, 1)
        assert sizes == [(32, 2, 257, 2, 1024, 16, 1)]
        assert b._runtime.ranks_per_node == 16
    finally:
        b.destroy()


@pytest.mark.parametrize(
    "name",
    [
        "combine_reduce_qps",
        "dispatch_workers",
        "reduce_lane_init",
        "combine_sparse_scan",
        "reduce_double_buffer",
        "combine_single_row",
        "combine_ready_trace",
        "dispatch_skip_empty",
        "dispatch_lookahead",
        "dispatch_single_row",
        "combine_cross_expert",
    ],
)
def test_retired_algorithm_switches_are_not_public_api(backend, monkeypatch, name):
    monkeypatch.setattr(api, "_MegaRuntime", lambda *a, **kw: pytest.fail("Retired option started SHMEM"))
    with pytest.raises(TypeError, match="unexpected keyword"):
        hierarchical_buffer(backend, **{name: True})


@pytest.mark.parametrize("case", ["fp8", "old-core"])
def test_qp8_rejects_incompatible_mode_or_old_runtime_before_shmem(backend, monkeypatch, case):
    monkeypatch.setattr(api, "_MegaRuntime", lambda *a, **kw: pytest.fail("Invalid configuration started SHMEM"))
    if case == "old-core":
        monkeypatch.setattr(api, "require_native", lambda _: SimpleNamespace(megamoe_combine_reduce_version=1))
    with pytest.raises(ValueError, match="FP8xFP4|Rebuild"):
        hierarchical_buffer(backend, mma_type="fp8xfp8" if case == "fp8" else "fp8xfp4")


def test_legacy_positional_fast_math_with_keyword_clamp(backend):
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    prefix = (y, *weights(), backend.buffer, None, None, None, (1, 1, 32), "swiglu")
    api.fp8_fp4_mega_moe(*prefix, True, activation_clamp=0.25)
    launched = next(item[1] for item in backend.events if isinstance(item, tuple) and item[0] == "launch")
    assert launched[-3] == 0.25
    with pytest.raises(NotImplementedError, match="default math"):
        api.fp8_fp4_mega_moe(*prefix, False)


@pytest.mark.parametrize("clamp", [None, 0, 2, 5, 10, 1 / 3])
def test_public_clamp_is_a_captured_fp32_scalar(backend, clamp):
    b = backend.buffer
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    api.fp8_fp4_mega_moe(y, *weights(), b, activation_clamp=clamp)
    args = next(item[1] for item in backend.events if isinstance(item, tuple) and item[0] == "launch")
    assert args[-3] == (np.finfo(np.float32).max if clamp is None else float(np.float32(clamp)))


@pytest.mark.parametrize("bad", [True, -1, float("inf"), float("nan"), 2**129, Tensor([1], "float32")])
def test_clamp_rejects_invalid_or_device_values(bad):
    with pytest.raises((ValueError, TypeError)):
        api._activation_clamp(bad)


def test_views_cache_separate_domains_freeze_and_close_independently(backend):
    root = hierarchical_buffer(backend, capacity=64, num_max_configs=3)
    child = api.SymmBuffer(root.group, 64, 32, 2, 1024, 256, ranks_per_node=16, base=root)
    root.prepare(32)
    child.prepare(32)
    child.prepare(16)
    plans = [root._plans[32], child._plans[32], child._plans[16]]
    assert [p.options[-2] for p in plans] == [0, 1, 2]
    api.fp8_fp4_mega_moe(Tensor(np.zeros((32, 1024)), "bfloat16"), *weights(), root)
    assert root._domains.frozen and child._domains is root._domains
    root.destroy()
    assert not child._runtime.closed
    child.prepare(32)
    assert child._plan is plans[1]
    with pytest.raises(ValueError, match="frozen"):
        child.prepare(31)
    assert child._domains.failed
    child.destroy()
    assert child._runtime.closed


def test_list_replacement_preserves_both_captured_tables_and_unchanged_experts(backend):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    def raw_groups():
        return [[(w[i].clone(), s[i].clone()) for i in range(2)] for w, s in weights()]

    l1, l2 = raw_groups()
    owner = prepare_weights_for_mega_moe(l1, l2)
    tables = [owner.select(f)[2] for f in (True, False)]
    old = [t.data.copy() for t in tables]
    l1, l2 = raw_groups()
    owner.update_experts([1], l1[1:], l2[1:], replace=True)
    for fused, table, previous in zip((True, False), tables, old):
        assert table is owner.select(fused)[2]
        assert np.array_equal(table.data[:, 0], previous[:, 0])
        assert np.all(table.data[:, 1] != previous[:, 1])
    api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), owner.l1, owner.l2, backend.buffer)
    assert owner in backend.buffer._owners
    assert owner.check_versions()
    owner.release_retired()


def test_list_failed_table_copy_requires_full_recovery(backend, monkeypatch):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    groups = [[(w[i].clone(), s[i].clone()) for i in range(2)] for w, s in weights()]
    owner = prepare_weights_for_mega_moe(*groups)
    table = owner.select(False)[2]
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("table copy failed")

        patch.setattr(table, "copy_", fail)
        with pytest.raises(RuntimeError, match="copy failed"):
            owner.update_experts([1], *(pairs[1:] for pairs in groups), replace=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        owner.select(True)
    with pytest.raises(RuntimeError, match="retry all"):
        owner.update_experts([0], *(pairs[:1] for pairs in groups))
    owner.update_experts([1], *(pairs[1:] for pairs in groups))
    assert owner.select(False)[2] is table
    assert table.data[0, 1] == owner.expert_views(1)["planar_l1"][0].data_ptr()


def test_external_eplb_refresh_updates_both_layouts_in_place(backend):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    owner = prepare_weights_for_mega_moe(*weights(), input_layout="planar_nz")
    views = owner.expert_views(1)
    planar, fused = views["planar_l1"], views["l1"]
    addresses = [t.data_ptr() for pair in (planar, fused) for t in pair]
    for target, inverse in (("planar", False), ("fused", True)):
        src, dst = (fused, planar) if inverse else (planar, fused)
        for t in src:
            for block, values in enumerate(np.split(t.data, 4)):
                values.fill(8 + 2 * block + inverse)
        owner.refresh_gmm1(target, [1])
        for source, result in zip(src, dst):
            expected = np.concatenate([np.split(source.data, 4)[i] for i in (0, 2, 1, 3)])
            np.testing.assert_array_equal(result.data, expected)
        assert addresses == [t.data_ptr() for pair in (planar, fused) for t in pair]
        assert owner.check_versions()


def test_prepared_routed_layouts_are_independent(backend):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    l1, l2 = weights()
    for planar in (l1, (l1[0], l1[1].clone()), (l1[0][0:2], l1[1])):
        with pytest.raises(ValueError, match="overlap"):
            prepare_weights_for_mega_moe(l1, l2, input_layout="prepared", planar_l1=planar)


@pytest.mark.parametrize("interrupt", [False, True])
def test_owner_swaps_aliasing_sources_before_first_write(backend, monkeypatch, interrupt):
    l1, l2 = [[(w[i].clone(), s[i].clone()) for i in range(2)] for w, s in weights()]
    l1[0][0].data.fill(1)
    l1[1][0].data.fill(2)
    owner = api.transform_weights_for_mega_moe(l1, l2)[0].owner
    planar, second = owner.expert_views(layout="planar")
    fused, _ = owner.expert_views(layout="fused")
    before = [w.data_ptr() for w in fused[0]]

    def flip(pair):
        return tuple(list(reversed(group)) for group in pair)

    if interrupt:
        with monkeypatch.context() as patch:

            def fail_copy(_):
                raise RuntimeError("interrupted swap")

            patch.setattr(fused[0][1], "copy_", fail_copy)
            with pytest.raises(RuntimeError, match="interrupted swap"):
                owner.update(flip(fused), flip(second), planar_l1=flip(planar), input_layout="prepared")
        with pytest.raises(RuntimeError, match="incomplete"):
            owner.select(True)
    owner.update(flip(fused), flip(second), planar_l1=flip(planar), input_layout="prepared")
    assert [w.data_ptr() for w in fused[0]] == before
    assert [w.data.flat[0] for w in fused[0]] == [2, 1]
    assert [w.data.flat[0] for w in planar[0]] == [2, 1]
    owner.refresh_gmm1([1], source_layout="planar")
    fused[0][1].data = fused[0][1].data.copy()
    with pytest.raises(RuntimeError, match="rebound"):
        owner.check_versions()


def test_capture_guards_weight_mutations(backend, monkeypatch):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    owner = prepare_weights_for_mega_moe(*weights(), input_layout="planar_nz")
    monkeypatch.setattr(api.npu_api(), "is_current_stream_capturing", lambda: True)
    for action in (
        owner.commit_prepared_update,
        owner.release_retired,
        owner.refresh_gmm1,
        lambda: owner.update(*weights()),
        lambda: prepare_weights_for_mega_moe(*weights()),
    ):
        with pytest.raises(RuntimeError, match="capture"):
            action()


@pytest.mark.parametrize("overlap", [False, True])
def test_prepared_scales_can_share_disjoint_intervals(backend, overlap):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    fused, second = weights()
    template = fused[1]
    count = template.numel()
    backing = np.zeros(2 * count, dtype=template.data.dtype)

    def scale(start):
        return Tensor(backing[start : start + count].reshape(template.shape), template.dtype)

    planar = (fused[0].clone(), scale(count - int(overlap)))
    planar[0].npu_format = fused[0].npu_format
    fused = (fused[0], scale(0))
    if overlap:
        with pytest.raises(ValueError, match="overlap"):
            prepare_weights_for_mega_moe(fused, second, input_layout="prepared", planar_l1=planar)
    else:
        assert prepare_weights_for_mega_moe(fused, second, input_layout="prepared", planar_l1=planar).check_versions()


@pytest.mark.parametrize("layout", ["nd", "prepared"])
@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("containers", [None, (list, list), (list, tuple), (tuple, list), (tuple, tuple)])
def test_parallel_expert_sequences_and_linear_format_30(backend, layout, size, containers):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    projections = []
    for w, s in fp8_weights():
        pairs = [(w[0].clone(), s[0].clone()) for _ in range(size)]
        for i, (weight, scale) in enumerate(pairs):
            weight.npu_format = (30, 0, 2)[i]
            scale.npu_format = (2, 30, 0)[i]
            weight.data.fill(10 + i)
            scale.data.fill(30 + i)
        projections.append(
            pairs if containers is None else tuple(cls(seq) for cls, seq in zip(containers, zip(*pairs)))
        )
    if layout == "nd":
        owner = api.transform_weights_for_mega_moe(*projections, shared=True)[0].owner
    else:
        owner = prepare_weights_for_mega_moe(*projections, input_layout=layout, shared=True)
    for w, s in owner.expert_views(layout="fused"):
        assert [int(t.data.flat[0]) for t in w] == list(range(10, 10 + size))
        assert [int(t.data.flat[0]) for t in s] == list(range(30, 30 + size))


@pytest.mark.parametrize("selector", ["indices", "experts", "positional"])
@pytest.mark.parametrize("replace", [False, True])
def test_expert_selectors_share_update_semantics(backend, selector, replace):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    groups = [[(w[i].clone(), s[i].clone()) for i in range(2)] for w, s in weights()]
    owner = prepare_weights_for_mega_moe(*groups)
    tables = [owner.select(f)[2] for f in (False, True)]
    previous = [t.data.copy() for t in tables]
    new1, new2 = [[(group[0][0].clone(), group[0][1].clone())] for group in groups]
    new1[0][0].data.fill(81)
    if selector == "positional":
        owner.update_experts([1], new1, new2, replace=replace)
    else:
        owner.update_experts(**{selector: [1]}, l1=new1, l2=new2, replace=replace)
    for f, t, old in zip((False, True), tables, previous):
        assert t is owner.select(f)[2]
        np.testing.assert_array_equal(t.data[:, 0], old[:, 0])
        assert np.all(t.data[:, 1] != old[:, 1]) if replace else np.array_equal(t.data[:, 1], old[:, 1])
    for layout in ("planar", "fused"):
        assert owner.expert_views(layout=layout)[0][0][1].data.flat[0] == 81
    for kwargs in (
        dict(indices=[1], experts=[1], l1=new1, l2=new2),
        dict(l1=new1, l2=new2),
        dict(indices=[1], l1=new1),
        dict(indices=[0, 0], l1=groups[0], l2=groups[1]),
    ):
        with pytest.raises((ValueError, TypeError)):
            owner.update_experts(**kwargs)
    assert owner.check_versions()


def test_expert_sequence_and_format_errors(backend):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe as prepare

    l1, l2 = fp8_weights()
    w, s = l1
    for malformed in (((), ()), ((w[0],), (s[0], s[1])), ((w[0],), s), [(w[0],)], [(w[0], None)]):
        with pytest.raises((TypeError, ValueError)):
            prepare(malformed, [(l2[0][0], l2[1][0])], input_layout="prepared")
    for value in (w, s):
        value.npu_format = 50
        with pytest.raises(ValueError):
            prepare(l1, l2, input_layout="prepared")
        value.npu_format = 2
    w.npu_format = 30
    w.data = w.data[..., ::-1]
    for layout in ("prepared", "nd"):
        with pytest.raises(ValueError):
            prepare(l1, l2, input_layout=layout)
    l1, l2 = weights()
    l1[0].npu_format = 30
    with pytest.raises(ValueError, match="NZ"):
        prepare(l1, l2, shared=True, input_layout="prepared")


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_list_singleton_axes_normalize_before_schema_and_update(backend, packed, shared):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    groups = []
    for proj, (w, s) in enumerate(fp8_weights() if shared else weights()):
        entries = []
        for i in range(3):
            weight, scale = w[0].clone(), s[0].clone()
            weight.data.fill(11 + i)
            scale.data.fill(121 + i)
            if packed:
                scale = Tensor(scale.data.reshape(scale.shape[0], -1).view(np.int32), "int32")
            if (i + proj) % 2:
                weight = weight.unsqueeze(0)
            if (i + proj) % 3:
                scale = scale.unsqueeze(0)
            entries.append((weight, scale))
        groups.append(entries)
    owner = prepare_weights_for_mega_moe(*groups, shared=shared)
    slots = [owner.select(fused)[2] for fused in (False, True)]
    for projection in owner.expert_views(layout="fused"):
        for i, (w, s) in enumerate(zip(*projection)):
            assert w.ndim == 2 and s.ndim == 3
            assert np.all(w.data == 11 + i) and np.all(s.data == 121 + i)
    owner.update(*groups)
    owner.update_experts(indices=[2], l1=groups[0][1:2], l2=groups[1][1:2])
    assert np.all(owner.expert_views(2)["l1"][0].data == 12)
    for invalid in (groups[0][0][0].unsqueeze(0).unsqueeze(0), weights()[0][0]):
        with pytest.raises(ValueError):
            owner.update_experts([0], [(invalid, groups[0][0][1])], groups[1][:1], replace=True)
    assert all(table is owner.select(fused)[2] for fused, table in zip((False, True), slots))
    assert owner.check_versions()


@pytest.mark.parametrize("packed", [False, True])
def test_import_independent_2d_nz_and_reject_partial_first_slice(backend, packed):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    def nz2(t):
        result = t[0].clone()
        result.npu_format = 50
        return result

    l1, l2 = weights()

    def entries(pair):
        w, s = pair

        def scale():
            t = s[0].clone()
            return Tensor(t.data.reshape(t.shape[0], -1).view(np.int32), "int32") if packed else t

        return [(nz2(w), scale().unsqueeze(0)), (nz2(w).unsqueeze(0), scale())]

    first, second, planar = entries(l1), entries(l2), entries(l1)
    owner = prepare_weights_for_mega_moe(first, second, input_layout="prepared", planar_l1=planar)
    table = owner.select(True)[2]
    before = table.data.copy()
    owner.refresh_gmm1([0], source_layout="planar")
    assert [e[1].ndim for e in backend.events if isinstance(e, tuple) and e[0] == "format_cast"][-2:] == [2, 2]
    owner.update_experts(indices=[0], l1=first[1:], l2=second[1:], planar_l1=planar[1:], input_layout="prepared")
    np.testing.assert_array_equal(table.data, before)
    for partial in (l1[0][0], l1[0][0:1]):
        with pytest.raises(ValueError, match="complete allocation"):
            owner.update_experts(
                indices=[0],
                l1=[(partial, first[0][1])],
                l2=second[:1],
                input_layout="prepared",
                planar_l1=planar[:1],
                replace=True,
            )
    scale = first[1][1]
    for malformed in (
        Tensor(np.stack([scale.data, scale.data]), scale.dtype),
        Tensor(scale.data[:, :-1].copy(), scale.dtype),
    ):
        with pytest.raises(ValueError):
            owner.update_experts(
                [0], [(first[0][0], malformed)], second[:1], input_layout="prepared", planar_l1=planar[:1], replace=True
            )
    np.testing.assert_array_equal(table.data, before)
    assert owner.check_versions()


@pytest.mark.parametrize("shared,packed", [(False, False), (False, True), (True, False)])
def test_raw_list_offset_casts_use_owning_2d_storage(backend, shared, packed):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    groups = []
    for weight, scale in weights():
        weight.npu_format = 2
        weight.data[:] = np.arange(weight.numel(), dtype=np.uint32).reshape(weight.shape) // weight.shape[-1]
        weight.data[1] += 17
        scale.data[1] += 3
        weight = weight.view("int8")
        scale = scale.view("uint8")
        if packed:
            scale = scale.reshape((*weight.shape[:2], -1)).view("int32")
        entries = [(weight[0], scale[0]), (weight[1:2], scale[1:2])]
        assert entries[1][0].is_contiguous() and entries[1][0].storage_offset() > 0
        groups.append(entries)
    clones = [[(w.clone(), s.clone()) for w, s in group] for group in groups]
    baseline = prepare_weights_for_mega_moe(*clones, shared=shared)
    # The owner may bind ND scale storage directly. Keep the update source independent.
    first, second = api.transform_weights_for_mega_moe(
        *[[(w, s.clone()) for w, s in group] for group in groups], shared=shared
    )
    owner = first.owner
    assert second.owner is owner

    def compare(i, expected):
        for name, pair in owner.expert_views(i).items():
            for actual, ref in zip(pair, baseline.expert_views(expected)[name]):
                np.testing.assert_array_equal(actual.data, ref.data)
            assert pair[0].ndim == 2 and pair[0].storage_offset() == 0

    compare(0, 0)
    compare(1, 1)
    ptrs = [t.data_ptr() for pair in owner.expert_views(0).values() for t in pair]
    owner.update(*(group[::-1] for group in groups))
    compare(0, 1)
    compare(1, 0)
    assert ptrs == [t.data_ptr() for pair in owner.expert_views(0).values() for t in pair]
    owner.update_experts([1], *(group[1:] for group in groups), replace=True)
    compare(1, 1)
    casts = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "format_cast"]
    assert casts and all(t.ndim == 2 and t.storage_offset() == 0 for t in casts)


def test_capture_guard_uses_bound_device_and_restores_caller(backend, monkeypatch):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe
    from deep_ep_experimental.megamoe.weights import _outside_capture

    current = Device("npu:1")
    capturing = {Device("npu:0")}

    @contextlib.contextmanager
    def device(target):
        nonlocal current
        previous, current = current, target
        try:
            yield
        finally:
            current = previous

    npu = api.npu_api()
    monkeypatch.setattr(npu, "device", device)
    monkeypatch.setattr(npu, "is_current_stream_capturing", lambda: current in capturing)
    groups = [[(w[0].clone(), s[0].clone())] for w, s in weights()]
    with pytest.raises(RuntimeError, match="capture"):
        prepare_weights_for_mega_moe(*groups)
    assert current == "npu:1" and not backend.events
    capturing.clear()
    owner = prepare_weights_for_mega_moe(*groups)
    table = owner.select(True)[2].data.copy()
    capturing.add(Device("npu:0"))
    backend.events.clear()
    for action in (
        lambda: owner.update(*groups),
        lambda: owner.update_experts([0], *groups, replace=True),
        owner.refresh_gmm1,
        owner.commit_prepared_update,
    ):
        with pytest.raises(RuntimeError, match="capture"):
            action()
        assert current == "npu:1" and not backend.events
        assert owner.check_versions()
        np.testing.assert_array_equal(table, owner.select(True)[2].data)
    capturing.clear()
    capturing.add(current)
    _outside_capture(Device("npu:0"))
    assert current == "npu:1"


def test_prepared_list_exports_physical_nz_owners_after_index_and_replace(backend):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    def projection(template):
        w, s = template
        owners, pairs = [], []
        for i in range(3):
            physical = (w[:1] if i == 1 else w[0]).clone()
            physical.npu_format = 50
            logical = physical
            if i == 2:
                logical = physical.unsqueeze(0)
                logical._base = physical
            owners.append(physical)
            pairs.append((logical, s[0].clone()))
        return owners, pairs

    l1, l2 = weights()
    fused, second, planar = [projection(t) for t in (l1, l2, l1)]
    owner = prepare_weights_for_mega_moe(fused[1], second[1], input_layout="prepared", planar_l1=planar[1])
    for i in range(3):
        assert all(w.ndim == 2 for w, _ in owner.expert_views(i).values())

    def check():
        for layout, first in (("planar", planar), ("fused", fused)):
            for (ws, ss), (originals, pairs) in zip(owner.expert_views(layout=layout), (first, second)):
                for w, scale, original, pair in zip(ws, ss, originals, pairs):
                    assert w is original and w.npu_format == 50
                    assert w.data_ptr() == w.untyped_storage().data_ptr() == original.data_ptr()
                    assert scale.shape == pair[1].shape and scale.data_ptr() == pair[1].data_ptr()

    check()
    updates = [projection(t) for t in (l1, l2, l1)]
    owner.update_experts(
        [1], [updates[0][1][0]], [updates[1][1][0]], planar_l1=[updates[2][1][0]], input_layout="prepared", replace=True
    )
    for original, new in zip((fused, second, planar), updates):
        for group, replacements in zip(original, new):
            group[1] = replacements[0]
    check()
    owner.release_retired()


def mock_board(monkeypatch, board):
    from deep_ep import _machine

    monkeypatch.setattr(_machine, "physical_device", lambda: 17)
    monkeypatch.setattr(_machine, "board_id", lambda physical: board)


@pytest.mark.parametrize("ep", [4, 8, 16])
@pytest.mark.parametrize("capacity,selected", [(255, "mte"), (256, "mte"), (257, "urma")])
def test_ubx_selects_backend_by_buffer_capacity(backend, monkeypatch, ep, capacity, selected):
    mock_board(monkeypatch, 0x44)
    b = api.SymmBuffer(backend.group_type(ep), ep * 2, capacity, 2, 1024, 256, ranks_per_node=ep)
    try:
        b.prepare(1)
        assert b._backend_name == selected
        if selected == "mte":
            assert isinstance(b._plan, backend.ext.Plan) and not b._hierarchical_roce
        else:
            assert isinstance(b._plan, backend.platform_ext.Plan) and b._plan.backend == "urma"
    finally:
        b.destroy()


@pytest.mark.parametrize("board", [3, 7, None, 0x6C])
def test_other_products_keep_mte_for_large_capacity(backend, monkeypatch, board):
    mock_board(monkeypatch, board)
    b = api.SymmBuffer(backend.group_type(16), 32, 257, 2, 1024, 256)
    try:
        assert b._backend_name == "mte" and not b._hierarchical_roce
    finally:
        b.destroy()


def test_ubx_ep32_decode_and_shared_view(backend, monkeypatch):
    mock_board(monkeypatch, 0x44)
    b = hierarchical_buffer(backend, capacity=256)
    from deep_ep import _machine

    monkeypatch.setattr(_machine, "board_id", lambda _: pytest.fail("view queried hardware again"))
    child = api.SymmBuffer(b.group, 64, 16, 2, 1024, 256, ranks_per_node=16, base=b)
    try:
        b.prepare(256)
        child.prepare(1)
        assert child._hierarchical_roce and b._hierarchical_roce and child._runtime is b._runtime
    finally:
        child.destroy()
        b.destroy()


@pytest.mark.parametrize("capacity,local", [(257, 16), (256, 8), (256, 0)])
def test_ubx_ep32_rejects_unsupported_layout_before_shmem(backend, monkeypatch, capacity, local):
    mock_board(monkeypatch, 0x44)
    monkeypatch.setattr(api, "_MegaRuntime", lambda *a, **kw: pytest.fail("unexpected SHMEM initialization"))
    with pytest.raises(ValueError, match="UBX EP32 requires"):
        api.SymmBuffer(backend.group_type(32), 64, capacity, 2, 1024, 256, ranks_per_node=local)


def urma_buffer(backend, monkeypatch, *, shared=0):
    mock_board(monkeypatch, 0x44)
    b = api.SymmBuffer(backend.group_type(16), 32, 2048, 2, 256, 256, num_shared_experts=shared)
    b.topk_idx.data[:] = [0, 1]
    b.topk_weights.data[:] = [0.4, 0.6]
    raw = (
        (Tensor(np.zeros((2, 512, 128)), "uint8"), Tensor(np.full((2, 512, 4, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 256, 128)), "uint8"), Tensor(np.full((2, 256, 4, 2), 127), "float8_e8m0fnu")),
    )
    return b, api.transform_weights_for_mega_moe(*raw)


def test_urma_uses_unified_plan_and_generation_survives_bs_changes(backend, monkeypatch):
    b, (w1, w2) = urma_buffer(backend, monkeypatch)

    def forbidden_readback(*args, **kwargs):
        pytest.fail("Normal forward must not read routing tensors on Host")

    monkeypatch.setattr(Tensor, "cpu", forbidden_readback)
    stats = Tensor([7, 11], "int32")
    generations = []
    plans = []
    casts_before = sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events)
    try:
        assert not hasattr(b, "_urma")
        for bs in (17, 1024, 1025, 2048, 17):
            y = Tensor(np.zeros((bs, 256)), "bfloat16")
            assert api.fp8_fp4_mega_moe(y, w1, w2, b, cumulative_local_expert_recv_stats=stats) is None
            launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
            assert launch[-2] is y
            assert launch[4] is w1.select(True)[0]  # URMA uses fused layout even above the MTE threshold.
            assert isinstance(b._plan, backend.platform_ext.Plan)
            assert b._plan.backend == "urma"
            generations.append(launch[-1])
            plans.append(b._plan)
            np.testing.assert_array_equal(y.data, 3)
        assert generations == [1, 2, 3, 4, 5]
        assert len({id(plan) for plan in plans}) == 1  # generation advances on one reused max-capacity plan.
        np.testing.assert_array_equal(stats.data, [17, 26])
        assert stats.streams == [b._runtime.stream] * 5
        assert sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events) == casts_before
    finally:
        b.destroy()


def test_urma_shared_shapes_reach_same_forward_without_extra_host_interface(backend, monkeypatch):
    import inspect

    b, weights = urma_buffer(backend, monkeypatch, shared=1)
    # Independent shared I=512; routed I=256. Native Plan is responsible for
    # inferring and preparing shared resources from these existing operands.
    sw1 = Tensor(np.zeros((1, 1024, 256)), "float8_e4m3fn")
    ss1 = Tensor(np.full((1, 1024, 4, 2), 127), "float8_e8m0fnu")
    sw2 = Tensor(np.zeros((1, 256, 512)), "float8_e4m3fn")
    ss2 = Tensor(np.full((1, 256, 8, 2), 127), "float8_e8m0fnu")
    y = Tensor(np.zeros((17, 256)), "bfloat16")
    try:
        assert "shared_intermediate_hidden" not in inspect.signature(api.SymmBuffer).parameters
        api.fp8_fp4_mega_moe(y, *weights, b, shared_l1_weights=(sw1, ss1), shared_l2_weights=(sw2, ss2))
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        assert launch[9:13] == (sw1, sw2, ss1, ss2)
        assert b._plan.backend == "urma"
        assert b._plan.workspace_bytes == b.workspace_bytes
    finally:
        b.destroy()


def test_urma_timer_uses_unified_forward_and_plan_diagnostics(backend, monkeypatch):
    b, weights = urma_buffer(backend, monkeypatch)
    try:
        timer = b.allocate_timer()
        assert timer.data.size > backend.platform_ext.timer_numel
        y = Tensor(np.zeros((17, 256)), "bfloat16")
        api.fp8_fp4_mega_moe(y, *weights, b, timer=timer)
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        assert launch[8] is timer
        assert timer.data[-1] == launch[-1]
        assert not hasattr(b._plan, "diagnostic_timer")
        assert not hasattr(b._plan, "cv_rows")
        assert not hasattr(b, "_urma")
    finally:
        b.destroy()


def test_urma_plan_allocation_failure_poisoned_without_caching_shape(backend, monkeypatch):
    b, weights = urma_buffer(backend, monkeypatch)
    original_plan = backend.platform_ext.Plan
    attempts = []

    def fail(*args, **kwargs):
        attempts.append((args, kwargs))
        raise RuntimeError("URMA workspace allocation failed")

    backend.platform_ext.Plan = fail
    y = Tensor(np.zeros((17, 256)), "bfloat16")
    try:
        with pytest.raises(RuntimeError, match="URMA workspace allocation failed"):
            api.fp8_fp4_mega_moe(y, *weights, b)
        assert b._failure == "URMA workspace allocation failed"
        assert b._plan is None and b._plan_bs is None
        assert len(attempts) == 1
        backend.platform_ext.Plan = original_plan
        with pytest.raises(RuntimeError, match="buffer failed.*restart"):
            api.fp8_fp4_mega_moe(y, *weights, b)
        assert b._plan is None and b._plan_bs is None
        assert len(attempts) == 1
        assert not any(isinstance(e, tuple) and e[0] == "launch" for e in backend.events)
    finally:
        backend.platform_ext.Plan = original_plan
        b.destroy()


def test_activation_clamp_reaches_native_forward(backend):
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    api.fp8_fp4_mega_moe(y, *weights(), backend.buffer, activation_clamp=0.25)
    assert [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1][-3] == 0.25
    with pytest.raises(ValueError, match="activation_clamp"):
        api.fp8_fp4_mega_moe(y, *weights(), backend.buffer, activation_clamp=-1)


@pytest.mark.parametrize("mode", [0, 1])
def test_route_weight_mode_reaches_plan(backend, mode):
    b = api.SymmBuffer(backend.group_type(), 4, 17, 2, 1024, 256, topk_weights_type=mode)
    try:
        b.prepare(17)
        assert getattr(b._plan, "topk_weights_type", 0) == mode
    finally:
        b.destroy()


@pytest.mark.parametrize("mode", [-1, 2, True, 1.0])
def test_invalid_route_weight_mode(backend, mode):
    with pytest.raises((ValueError, TypeError), match="topk_weights_type"):
        api.SymmBuffer(backend.group_type(), 4, 17, 2, 1024, 256, topk_weights_type=mode)
