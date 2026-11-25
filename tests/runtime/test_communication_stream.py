"""CPU checks for Direct V1/V2 process/device communication-stream ownership."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier
from types import SimpleNamespace
import sys

import pytest

from deep_ep.backends.direct import buffer, elastic


@pytest.fixture
def streams(monkeypatch):
    state = SimpleNamespace(device=0, pid=100, created=[], fail=False)

    def create(*, device):
        if state.fail:
            raise RuntimeError("stream creation failed")
        stream = SimpleNamespace(device=device, npu_stream=len(state.created) + 1)
        state.created.append(stream)
        return stream

    monkeypatch.setattr(buffer, "_COMMUNICATION_STREAMS", {})
    monkeypatch.setattr(buffer.os, "getpid", lambda: state.pid)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(current_device=lambda: state.device, Stream=create))
    )
    return state


def test_v1_v2_share_stream_but_devices_and_processes_do_not(streams):
    first = buffer.get_global_comm_stream()
    assert elastic.get_global_comm_stream() is first
    streams.device = 1
    second = elastic.get_global_comm_stream()
    assert second is not first and second.device == 1
    streams.device = 0
    assert buffer.get_global_comm_stream() is first
    streams.pid += 1
    assert buffer.get_global_comm_stream() is not first
    assert len(streams.created) == 3


def test_simultaneous_first_access_creates_one_stream(streams):
    ready = Barrier(8)

    def access(index):
        ready.wait(timeout=10)
        module = buffer if index % 2 else elastic
        return module.get_global_comm_stream()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(access, range(8)))
    assert all(stream is results[0] for stream in results)
    assert len(streams.created) == 1


def test_creation_failure_does_not_publish_a_stream(streams):
    streams.fail = True
    with pytest.raises(RuntimeError, match="stream creation failed"):
        buffer.get_global_comm_stream()
    assert buffer._COMMUNICATION_STREAMS == {}
    streams.fail = False
    assert buffer.get_global_comm_stream() is elastic.get_global_comm_stream()
    assert len(streams.created) == 1


def test_output_registration_preserves_devices_order_and_duplicates(monkeypatch):
    import gc
    import weakref

    @dataclass(frozen=True)
    class Device:
        type: str
        index: int

    queries, registrations = [], []
    current = {0: object(), 1: object()}

    def current_stream(device):
        queries.append(device.index)
        return current[device.index]

    class Tensor:
        def __init__(self, name, device):
            self.name, self.device = name, device

        def record_stream(self, stream):
            registrations.append((self.name, stream))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(current_stream=current_stream)))
    first = Tensor("first", Device("npu", 0))
    second = Tensor("second", Device("npu", 1))
    cpu = Tensor("cpu", Device("cpu", 0))
    refs = [weakref.ref(t) for t in (first, second)]
    cache = {}
    # Separate dispatch batches retain their original registration order.
    buffer._record_recv_outputs_on_current_stream(None, cpu, first, streams=cache)
    buffer._record_recv_outputs_on_current_stream(second, first, streams=cache)
    assert queries == [0, 1]
    assert registrations == [("first", current[0]), ("second", current[1]), ("first", current[0])]
    # A later API call must observe changed caller streams, even on the same device.
    current[0] = object()
    buffer._record_recv_outputs_on_current_stream(first, first)
    assert queries == [0, 1, 0]
    assert registrations[-2:] == [("first", current[0])] * 2
    del first, second
    gc.collect()
    assert all(ref() is None for ref in refs)  # The stream cache must not extend tensor ownership.


def test_output_registration_does_not_hide_failures(monkeypatch):
    from collections import namedtuple

    device = namedtuple("Device", "type index")("npu", 0)
    calls = []

    def fail(stream):
        calls.append(stream)
        raise RuntimeError("registration failed")

    stream = object()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device: stream))
    )
    tensor = SimpleNamespace(device=device, record_stream=fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        buffer._record_recv_outputs_on_current_stream(tensor, tensor)
    assert calls == [stream]
