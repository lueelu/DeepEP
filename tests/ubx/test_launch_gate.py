# Copyright (c) 2026, Lu Lu
"""CPU queue model for the benchmark gate, including delayed Host submission."""

import ctypes
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("ubx_launch_gate", Path(__file__).with_name("launch_gate.py"))
gate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate_module)


class Function:
    def __init__(self, operation):
        self.operation = operation

    def __call__(self, *args):
        return self.operation(*args)


class QueueModel:
    # Values above 32 bits also catch pointer truncation in the ctypes adapter.
    target = 0x100000001
    control = 0x100000002
    notify = 0x100000003

    def __init__(self, failure=""):
        self.failure = failure
        self.held = False
        self.host = []
        self.device = []
        self.executed = []
        self.calls = []
        self.tick = 0
        self.expected = ["HCCL", "Combine", "completion-event"]
        for name in (
            "CreateNotify",
            "CreateStream",
            "WaitAndResetNotify",
            "RecordNotify",
            "SynchronizeStreamWithTimeout",
            "DestroyStream",
            "DestroyNotify",
        ):
            setattr(self, "aclrt" + name, Function(lambda *args, name=name: self.acl(name, *args)))

    @property
    def npu_stream(self):
        self.device.extend(self.host)
        self.host.clear()
        self.progress()
        return self.target

    def progress(self):
        if self.held:
            return
        for name, duration in self.device:
            self.executed.append((name, self.tick))
            self.tick += duration
        self.device.clear()

    def submit(self, name, duration):
        self.host.append((name, duration))
        self.tick += 100  # Arbitrarily slow Host between enqueue operations.
        self.progress()
        assert self.held and not self.executed

    def acl(self, name, *args):
        self.calls.append(name)
        if self.failure == name:
            return 42
        if name in ("CreateNotify", "CreateStream"):
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = (
                self.notify if name == "CreateNotify" else self.control
            )
        elif name == "WaitAndResetNotify":
            assert args[0].value == self.notify and args[1].value == self.target
            assert args[2] == 180000 and not self.held
            self.held = True
        elif name == "RecordNotify":
            assert args[0].value == self.notify and args[1].value == self.control
            assert self.held and not self.host
            assert [name for name, _ in self.device] == self.expected
            self.held = False
            self.progress()
        elif name == "SynchronizeStreamWithTimeout":
            assert args[0].value in (self.target, self.control) and args[1] == 180000
            assert not self.held and not self.device and not self.host
        elif name in ("DestroyStream", "DestroyNotify"):
            assert not self.held and not self.device and not self.host
            assert args[0].value == (self.control if name == "DestroyStream" else self.notify)
        return 0


def exercise(model):
    gate = gate_module.LaunchGate(model)
    output = object()

    def invoke():
        model.submit("HCCL", 200)
        if model.failure == "callback":
            raise ValueError("injected submission failure")
        model.submit("Combine", 5)
        model.submit("completion-event", 0)
        return output

    for _ in range(3):
        model.executed.clear()
        assert gate.run(invoke) is output
        assert [name for name, _ in model.executed] == ["HCCL", "Combine", "completion-event"]
        assert model.executed[2][1] - model.executed[1][1] == 5
    gate.close()
    gate.close()
    return gate


def test_delayed_host_queue_drained_before_release_and_repeated_reuse(monkeypatch):
    model = QueueModel()
    monkeypatch.setattr(gate_module.ctypes, "CDLL", lambda name: model)
    gate = exercise(model)
    assert model.aclrtCreateNotify.argtypes == [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64]
    assert model.aclrtWaitAndResetNotify.argtypes == [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    assert model.aclrtSynchronizeStreamWithTimeout.argtypes == [ctypes.c_void_p, ctypes.c_int32]
    assert model.calls.count("DestroyNotify") == 1
    assert model.calls.count("RecordNotify") == 3
    with pytest.raises(RuntimeError, match="closed"):
        gate.run(lambda: None)


@pytest.mark.parametrize("operation", ["CreateNotify", "CreateStream"])
def test_creation_failure_reports_acl_status_and_releases_created_notify(monkeypatch, operation):
    model = QueueModel(operation)
    monkeypatch.setattr(gate_module.ctypes, "CDLL", lambda name: model)
    with pytest.raises(RuntimeError, match="ACL error 42"):
        gate_module.LaunchGate(model)
    assert ("DestroyNotify" in model.calls) == (operation == "CreateStream")


class EventQueueModel(QueueModel):
    def __init__(self):
        super().__init__()
        self.expected = ["HCCL", "producer-event", "start", "Combine", "end", "completion-event"]
        self.events = []
        self.destroyed = []
        for name in ("CreateEventWithFlag", "RecordEvent", "EventElapsedTime", "DestroyEvent"):
            setattr(self, "aclrt" + name, Function(lambda *args, name=name: self.event(name, *args)))

    def event(self, name, *args):
        if name == "CreateEventWithFlag":
            assert args[1] == 8
            self.events.append(0x200000001 + len(self.events))
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = self.events[-1]
        elif name == "RecordEvent":
            assert self.held and not self.host and args[1].value == self.target
            self.device.append(("start" if args[0].value == self.events[0] else "end", 0))
        elif name == "EventElapsedTime":
            assert not self.held and not self.device
            assert (args[1].value, args[2].value) == tuple(self.events)
            times = dict(self.executed)
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_float))[0] = (times["end"] - times["start"]) / 1000
        elif name == "DestroyEvent":
            assert not self.held and not self.device
            self.destroyed.append(args[0].value)
        return 0


def test_native_event_boundary_excludes_hccl_framework_events_and_host_delay(monkeypatch):
    model = EventQueueModel()
    monkeypatch.setattr(gate_module.ctypes, "CDLL", lambda name: model)

    def native_launch(stream):
        assert stream == model.target and model.held
        model.device.append(("Combine", 5))

    native = SimpleNamespace(combine_low_latency=native_launch)
    gate = gate_module.LaunchGate(model)

    def public_call():
        model.submit("HCCL", 200)
        model.submit("producer-event", 3)
        native.combine_low_latency(model.npu_stream)
        model.submit("completion-event", 7)
        return "output"

    with gate_module.KernelEventTimer(native, model) as timer:
        for _ in range(3):
            model.executed.clear()
            assert gate.run(public_call) == "output"
            assert timer.elapsed_us() == pytest.approx(5)
        with pytest.raises(RuntimeError, match="did not reach"):
            timer.elapsed_us()
    gate.close()
    assert native.combine_low_latency is native_launch
    assert set(model.destroyed) == set(model.events)
    assert model.aclrtEventElapsedTime.argtypes == [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]


@pytest.mark.parametrize(
    "operation", ["WaitAndResetNotify", "callback", "RecordNotify", "SynchronizeStreamWithTimeout", "DestroyNotify"]
)
def test_submission_or_device_failure_exits_worker_without_collective_cleanup(operation):
    # Test the real fail-fast exit, not an exception that could enter Torch teardown.
    result = subprocess.run([sys.executable, __file__, operation], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    assert "UBX launch gate failed:" in result.stderr
    assert (
        "injected submission failure" in result.stderr if operation == "callback" else "ACL error 42" in result.stderr
    )


if __name__ == "__main__":
    model = QueueModel(sys.argv[1])
    gate_module.ctypes.CDLL = lambda name: model
    exercise(model)
