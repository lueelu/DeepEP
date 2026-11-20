# Copyright (c) 2026, Lu Lu

"""RoCE teardown ordering and failure containment without NPU hardware."""

import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from deep_ep import _runtime
from deep_ep._store import StoreGroup


@pytest.fixture
def roce_runtime(monkeypatch):
    log = []
    runtime = object.__new__(_runtime.Runtime)
    runtime.closed = False
    runtime.use_roce = True
    runtime.device = 3
    runtime.world_size = 2
    runtime.notify_lock = threading.RLock()
    runtime.group = StoreGroup(None, 0, 2, prefix="roce-lifecycle")
    runtime.group.all_gather = lambda packet, stage: log.append(("gather", stage)) or [packet, packet]
    runtime.group.abort = lambda error: log.append(("abort", str(error)))
    runtime.native = SimpleNamespace(
        close=lambda **kwargs: log.append(("close", kwargs)),
        barrier=lambda **kwargs: log.append(("barrier", kwargs)),
    )
    npu = SimpleNamespace(
        current_device=lambda: 3,
        synchronize=lambda device: log.append(("device sync", device)),
    )
    monkeypatch.setattr(_runtime, "npu_api", lambda: npu)
    monkeypatch.setattr(_runtime, "_OWNER", runtime)
    return runtime, npu, log


def test_roce_barrier_then_close_waits_all_streams_and_peers(roce_runtime):
    runtime, _, log = roce_runtime
    runtime.barrier()
    assert _runtime._OWNER is runtime and not runtime.closed
    runtime.close()
    runtime.close()
    assert log == [
        ("device sync", 3),
        ("gather", "RoCE runtime barrier"),
        ("barrier", {"host_coordinated": True}),
        ("device sync", 3),
        ("gather", "RoCE runtime close"),
        ("close", {"host_coordinated": True}),
    ]
    assert runtime.closed and _runtime._OWNER is None


@pytest.mark.parametrize("operation", ["barrier", "close"])
@pytest.mark.parametrize("fault", ["local", "peer", "store", "order", "native"])
def test_roce_failure_retains_owner_and_prevents_reuse(roce_runtime, operation, fault):
    runtime, npu, log = roce_runtime

    def fail(*args, **kwargs):
        raise RuntimeError("injected fault")

    def gather(packet, stage):
        log.append(("gather", stage))
        if fault == "local":
            assert "injected fault" in packet[1]
        if fault == "store":
            fail()
        return [packet, ["wrong operation" if fault == "order" else stage, "peer fault" if fault == "peer" else None]]

    runtime.group.all_gather = gather
    if fault == "local":
        npu.synchronize = fail
    if fault == "native":
        setattr(runtime.native, operation, fail)
    with pytest.raises(RuntimeError):
        getattr(runtime, operation)()
    assert _runtime._OWNER is runtime and not runtime.closed and runtime.roce_failed
    assert any(entry[0] == "gather" for entry in log)
    assert log[-1][0] == "abort"
    assert not any(entry[0] in ("barrier", "close") for entry in log)
    with pytest.raises(RuntimeError, match="restart the distributed job"):
        runtime.ensure_alive()


def test_roce_process_group_exchanges_status_after_device_completion(roce_runtime, monkeypatch):
    runtime, _, log = roce_runtime
    runtime.group = object()
    dist = ModuleType("torch.distributed")

    def gather(peers, packet, group):
        assert log == [("device sync", 3)] and group is runtime.group
        peers[:] = [packet, packet]
        log.append(("process group", packet))

    dist.all_gather_object = gather
    torch = ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    runtime.close()
    assert log[-1] == ("close", {"host_coordinated": True})
    assert runtime.closed and _runtime._OWNER is None
