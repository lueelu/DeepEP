# Copyright (c) 2026, Lu Lu
"""Public V2 stream/ownership contracts without importing Torch or an NPU."""

from contextlib import contextmanager
import sys
import threading
from types import SimpleNamespace

import pytest

from deep_ep import ElasticBuffer, EPHandle
from deep_ep.backends.mesh_clos import elastic as mesh
from deep_ep.backends.ubx import elastic as ubx
from deep_ep.utils import event as events


@pytest.fixture
def setup(monkeypatch):
    log = []
    device = SimpleNamespace(type="npu", index=0)
    caller = SimpleNamespace(device=0, npu_stream=1)
    comm = SimpleNamespace(device=0, npu_stream=2)
    active = [caller]

    class Tensor:
        requires_grad = False

        def __init__(self, shape, dtype="torch.bfloat16", device=device):
            self.shape, self.dtype, self.device = shape, dtype, device

        def is_contiguous(self):
            return True

        def record_stream(self, stream):
            log.append(("use", self, stream))

        def add_(self, other):
            log.append(("add", self, other))
            return self

    class Event:
        def record(self, stream):
            log.append(("record", self, stream))

        def wait(self, stream):
            log.append(("wait", self, stream))

    @contextmanager
    def stream(value):
        previous, active[0] = active[0], value
        try:
            yield
        finally:
            active[0] = previous

    npu = SimpleNamespace(
        stream=stream,
        current_device=lambda: 0,
        current_stream=lambda _: active[0],
        Event=Event,
    )
    torch = SimpleNamespace(Tensor=Tensor, empty=Tensor, bfloat16="torch.bfloat16", float16="torch.float16")
    monkeypatch.setitem(sys.modules, "torch", torch)
    for module in (mesh, ubx, events):
        monkeypatch.setattr(module, "npu_api", lambda: npu, raising=False)

    def tiling(*args, **kwargs):
        log.append(("tiling", args, kwargs))
        return bytes(16)

    def launch(session, tensors, *args):
        assert active[0] is comm and args[-1] == comm.npu_stream
        log.append(("launch", tensors))

    native = SimpleNamespace(
        low_latency_core_config=lambda: {"active_aiv": 56},
        low_latency_combine_tiling=tiling,
        combine_low_latency=launch,
    )
    for module in (mesh, ubx):
        monkeypatch.setattr(module, "require_native", lambda _: native)
    runtime = SimpleNamespace(
        device=0,
        world_size=32,
        stream=comm,
        native=object(),
        ll_backend="rdma",
        closed=True,
        notify_lock=threading.Lock(),
        notify_failed=False,
        notify_inflight=None,
        ensure_alive=lambda: None,
        low_latency_comm_args=Tensor((529944,), "torch.uint8"),
    )
    buffer = ElasticBuffer.__new__(ElasticBuffer)
    buffer._runtime, buffer._handle_token = runtime, object()
    buffer._upload_low_latency = lambda data, dev: Tensor((len(data),), "torch.uint8", dev)
    handle = EPHandle.__new__(EPHandle)
    handle.__dict__.update(
        _buffer_token=buffer._handle_token,
        is_low_latency=True,
        metadata_only=False,
        do_expand=True,
        expert_alignment=1,
        num_tokens=32,
        hidden=3584,
        num_topk=8,
        num_experts=384,
        num_max_tokens_per_rank=256,
        num_expanded_tokens=12,
    )
    for name in ("destination_index", "send_counts", "source_weights", "token_type", "relay_read_index", "source_mask"):
        setattr(handle, name, Tensor((1,)))
    return SimpleNamespace(
        buffer=buffer,
        handle=handle,
        runtime=runtime,
        native=native,
        log=log,
        x=Tensor((12, 3584)),
        Tensor=Tensor,
        comm=comm,
        caller=caller,
    )


def test_public_rdma_call_uses_dispatch_tensors_and_default_stream_dependency(setup):
    s = setup
    output, weights, event = s.buffer.combine(s.x, s.handle, num_sms=56, num_qps=56)
    assert output.shape == (32, 3584) and weights is None and event.event is None
    launched = next(entry[1] for entry in s.log if entry[0] == "launch")
    assert launched["input"] is s.x and launched["output"] is output
    assert launched["comm_args"] is s.runtime.low_latency_comm_args
    assert launched["relay_read_index"] is s.handle.relay_read_index
    assert next(entry[2] for entry in s.log if entry[0] == "tiling") == {"rdma": True}
    order = [entry[0] for entry in s.log if entry[0] != "use"]
    assert order == ["record", "wait", "tiling", "launch", "record", "wait"]
    assert s.log[-1][2] is s.caller  # Consumer storage is registered after the wait.
    assert s.runtime.notify_inflight is None


def test_async_previous_and_epilogue_events_bias_and_repeated_handle(setup):
    s = setup
    ready, epilogue = events.EventHandle(), events.EventHandle()
    bias = s.Tensor((32, 3584))
    s.log.clear()
    outputs = []
    for _ in range(3):
        x = s.Tensor((12, 3584))
        output, _, event = s.buffer.combine(
            x,
            s.handle,
            bias=(bias, bias),
            previous_event=ready,
            previous_event_before_epilogue=epilogue,
            allocate_on_comm_stream=True,
            async_with_compute_stream=True,
        )
        assert event.event is not None and event.event._stream is s.comm
        assert any(value is x for value in event._retained_objects)
        assert any(value is s.handle for value in event._retained_objects)
        assert s.log[-1][2] is s.comm  # Async return did not wait on the caller.
        event.current_stream_wait(release_handle=True)
        assert event.event is None and event._retained_objects is None
        outputs.append(output)
    assert len({id(output) for output in outputs}) == 3
    assert len([entry for entry in s.log if entry[0] == "tiling"]) == 1
    assert len([entry for entry in s.log if entry[0] == "add"]) == 6
    for dependency in (ready, epilogue):
        assert sum(entry == ("wait", dependency._event, s.comm) for entry in s.log) == 3


@pytest.mark.parametrize(
    "option",
    [
        {"topk_weights": object()},
        {"num_sms": 32},
        {"num_qps": 2},
        {"bias": ()},
        {"async_with_compute_stream": 1},
    ],
)
def test_reject_incompatible_options_before_launch(setup, option):
    s = setup
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        s.buffer.combine(s.x, s.handle, **option)
    assert not any(entry[0] == "launch" for entry in s.log)
    assert not s.runtime.notify_failed


def test_reject_wrong_owner_device_and_unexpanded_handle(setup):
    s = setup
    s.handle._buffer_token = object()
    with pytest.raises(ValueError, match="this buffer"):
        s.buffer.combine(s.x, s.handle)
    s.handle._buffer_token = s.buffer._handle_token
    s.x.device = SimpleNamespace(type="npu", index=1)
    with pytest.raises(ValueError, match="buffer NPU"):
        s.buffer.combine(s.x, s.handle)
    s.x.device.index = 0
    s.handle.do_expand = False
    with pytest.raises(NotImplementedError, match="expanded"):
        s.buffer.combine(s.x, s.handle)


def test_launch_failure_retains_storage_and_prevents_reuse(setup):
    s = setup

    def fail(*_):
        raise RuntimeError("launch failure")

    s.native.combine_low_latency = fail
    with pytest.raises(RuntimeError, match="launch failure"):
        s.buffer.combine(s.x, s.handle)
    assert s.runtime.notify_failed and any(value is s.x for value in s.runtime.notify_inflight)
    with pytest.raises(RuntimeError, match="restart"):
        s.buffer.combine(s.x, s.handle)


def test_pod_path_does_not_enter_rdma_adapter(setup, monkeypatch):
    s = setup
    s.runtime.ll_backend = "mte"
    monkeypatch.setattr(ubx, "combine", lambda *_: pytest.fail("POD entered the RDMA adapter"))
    output, _, event = s.buffer.combine(s.x, s.handle)
    assert output.shape == (32, 3584) and event.event is None
    assert next(entry[2] for entry in s.log if entry[0] == "tiling") == {}
