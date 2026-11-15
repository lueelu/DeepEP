"""CPU checks for Direct V1/V2 process/device communication-stream ownership."""

from concurrent.futures import ThreadPoolExecutor
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
