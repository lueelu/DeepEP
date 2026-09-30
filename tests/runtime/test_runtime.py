# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Control-path fault injection; these tests do not establish NPU qualification."""

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import deep_ep
from deep_ep import _runtime
from deep_ep.utils import event as events


class FakeEvent:
    def __init__(self, log):
        self.log = log

    def record(self, stream):
        self.log.append(("record", stream))

    def wait(self, stream):
        self.log.append(("wait", stream))


@pytest.fixture
def event_backend(monkeypatch):
    log = []
    npu = SimpleNamespace(
        current_device=lambda: 0,
        current_stream=lambda device: "consumer",
        Event=lambda: FakeEvent(log),
    )
    monkeypatch.setattr(events, "require_native", lambda operation: None)
    monkeypatch.setattr(events, "npu_api", lambda: npu)
    return log, npu


def test_record_wait_hook_and_release(event_backend):
    log, _ = event_backend
    wrapper = events.EventOverlap(events.EventHandle())
    wrapper.register_hook_after_wait(lambda: log.append("hook"))
    with wrapper(release_handle=True):
        log.append("body")
    assert log == [("record", "consumer"), "body", ("wait", "consumer"), "hook"]
    assert wrapper.event is None
    with pytest.raises(RuntimeError, match="released"):
        wrapper.current_stream_wait()


def test_event_failure_does_not_drop_resources_or_hook(event_backend):
    log, _ = event_backend
    wrapper = events.EventOverlap(events.EventHandle())

    def hook():
        log.append("hook")

    wrapper.register_hook_after_wait(hook)

    def fail(stream):
        raise RuntimeError("wait fault")

    wrapper.event._event.wait = fail
    with pytest.raises(RuntimeError, match="wait fault"):
        wrapper.current_stream_wait(release_handle=True)
    assert wrapper.event is not None and wrapper._hook is hook


def test_context_wait_runs_on_body_exception(event_backend):
    log, _ = event_backend
    wrapper = events.EventOverlap(events.EventHandle())
    with pytest.raises(ValueError, match="body"):
        with wrapper(True):
            raise ValueError("body")
    assert ("wait", "consumer") in log and wrapper.event is None


def test_empty_event_wrong_device_and_hooks(event_backend):
    _, npu = event_backend
    with pytest.raises(ValueError, match="recorded"):
        events.EventOverlap(None)
    handle = events.EventHandle()
    npu.current_device = lambda: 1
    with pytest.raises(RuntimeError, match="another"):
        handle.current_stream_wait()
    wrapper = events.EventOverlap(handle)
    with pytest.raises(TypeError):
        wrapper.register_hook_after_wait(42)
    wrapper.register_hook_after_wait(lambda: None)
    with pytest.raises(RuntimeError, match="already"):
        wrapper.register_hook_after_wait(lambda: None)


@pytest.mark.parametrize("value", [None, -1, 0, True, 1.5, 2**63])
def test_capacity_rejection(value):
    with pytest.raises(ValueError):
        _runtime.integer("capacity", value, 2 << 20, 1 << 30)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "tcp://localhost:9000",
        "tcp://127.0.0.1:80",
        "tcp://127.0.0.1:65536",
        "tcp://127.0.0.1:1/",
        "udp://127.0.0.1:9000",
    ],
)
def test_invalid_endpoint(monkeypatch, value):
    monkeypatch.setenv("DEEPEP_SHMEM_ENDPOINT", value)
    with pytest.raises(ValueError):
        _runtime.endpoint()


def test_host_boot_id_is_read_from_proc(monkeypatch):
    def read(path):
        assert path.as_posix() == "/proc/sys/kernel/random/boot_id"
        return " 11111111-2222-4333-8444-555555555555\n"

    monkeypatch.setattr(_runtime.Path, "read_text", read)
    assert _runtime._host_boot_id() == "11111111222243338444555555555555"


@pytest.mark.parametrize("value", ["", "cpu1", "not-a-uuid", "\ufffd"])
def test_invalid_host_boot_id_returns_collective_failure_marker(monkeypatch, value):
    monkeypatch.setattr(_runtime.Path, "read_text", lambda _: value)
    assert _runtime._host_boot_id() == ""


def test_unreadable_host_boot_id_returns_collective_failure_marker(monkeypatch):
    def unreadable(_):
        raise PermissionError("proc is not readable")

    monkeypatch.setattr(_runtime.Path, "read_text", unreadable)
    assert _runtime._host_boot_id() == ""


@pytest.mark.parametrize(
    "cls,options",
    [
        (deep_ep.Buffer, {"low_latency_mode": True}),
        (deep_ep.Buffer, {"num_scaleout_bytes": 1024, "explicitly_destroy": True}),
        (deep_ep.Buffer, {"allow_multinode_memory_transport": True, "explicitly_destroy": True}),
        (deep_ep.Buffer, {"allow_memory_transport_for_low_latency_mode": False, "explicitly_destroy": True}),
        (deep_ep.Buffer, {"num_qps_per_rank": 32, "explicitly_destroy": True}),
        (deep_ep.Buffer, {"comm": object()}),
        (deep_ep.Buffer, {}),
        (deep_ep.ElasticBuffer, {"use_fp8_dispatch": True}),
        (deep_ep.ElasticBuffer, {"num_cpu_bytes": 1024}),
        (deep_ep.ElasticBuffer, {"num_device_timeout_secs": 1}),
        (deep_ep.ElasticBuffer, {"num_allocated_qps": 1, "num_bytes": 4 << 20, "explicitly_destroy": True}),
        (deep_ep.ElasticBuffer, {"explicitly_destroy": True}),
        (deep_ep.ElasticBuffer, {}),
    ],
)
def test_unsupported_constructor_before_runtime(monkeypatch, cls, options):
    module = importlib.import_module(cls.__module__)
    monkeypatch.setattr(module, "require_native", lambda operation: None)

    def forbidden(*args):
        pytest.fail("Acquired runtime for unsupported option")

    monkeypatch.setattr(module, "Runtime", forbidden)
    with pytest.raises(NotImplementedError):
        cls(None, **options)


def test_close_failure_retains_owner(monkeypatch):
    runtime = object.__new__(_runtime.Runtime)
    runtime.closed = False
    runtime.ensure_alive = lambda: None
    runtime.stream = SimpleNamespace(synchronize=lambda: None)

    def fail():
        raise RuntimeError("close fault")

    runtime.native = SimpleNamespace(close=fail)
    monkeypatch.setattr(_runtime, "_OWNER", runtime)
    with pytest.raises(RuntimeError, match="close fault"):
        runtime.close()
    assert _runtime._OWNER is runtime and not runtime.closed
    runtime.native.close = lambda: None
    runtime.close()
    runtime.close()
    assert _runtime._OWNER is None and runtime.closed


def test_scaleup_capacity_reaches_runtime_without_translation(monkeypatch):
    module = importlib.import_module(deep_ep.Buffer.__module__)
    monkeypatch.setattr(module, "require_native", lambda operation: None)
    calls = []
    resource = SimpleNamespace(closed=True)

    def create(group, capacity):
        calls.append((group, capacity))
        return resource

    monkeypatch.setattr(module, "Runtime", create)
    group = object()
    buffer = deep_ep.Buffer(group, num_scaleup_bytes=4 << 20, explicitly_destroy=True)
    assert calls == [(group, 4 << 20)]
    assert buffer._runtime is resource


def test_import_does_not_load_framework_or_extension():
    root = Path(deep_ep.__file__).resolve().parent.parent
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'torch_npu'} or name == 'deep_ep._C':
            raise AssertionError(name)
sys.meta_path.insert(0, Block())
import deep_ep
assert len(deep_ep.__all__) == 6
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mismatch", [None, "capacity", "timeout", "endpoint", "busy", "engine"])
def test_bootstrap_without_host_device_checks(monkeypatch, mismatch):
    pytest.importorskip("torch")
    import torch.distributed as dist

    monkeypatch.setattr(_runtime, "endpoint", lambda: "tcp://xxx.xxx.xxx.xxx:xxxxx")
    monkeypatch.setattr(_runtime, "_OWNER", None)
    monkeypatch.setattr(_runtime, "rank_and_world", lambda *args: (0, 2))
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    created = []
    native = SimpleNamespace(
        runtime_busy=lambda: False,
        Runtime=lambda *args: created.append(args) or SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(_runtime, "require_native", lambda operation: native)
    monkeypatch.setattr(
        _runtime,
        "npu_api",
        lambda: SimpleNamespace(
            current_device=lambda: 0, Stream=lambda **kwargs: SimpleNamespace(synchronize=lambda: None)
        ),
    )

    def gather(peers, local, group):
        # Only collective resource settings remain; no host name or device index.
        assert local.numel() == 5
        assert local.tolist()[:2] == [2 << 20, 120]
        assert local.tolist()[3:] == [0, 0]
        peers[0].copy_(local)
        peers[1].copy_(local)
        if mismatch:
            peers[1][{"capacity": 0, "timeout": 1, "endpoint": 2, "busy": 3, "engine": 4}[mismatch]] += 1

    monkeypatch.setattr(dist, "all_gather", gather)
    if mismatch:
        message = {
            "capacity": "workspace capacity",
            "timeout": "timeout",
            "endpoint": "bootstrap endpoint",
            "busy": "already owns",
            "engine": "runtime engines",
        }[mismatch]
        with pytest.raises((ValueError, RuntimeError), match=message):
            _runtime.Runtime(object(), 2 << 20)
        assert not created
    else:
        runtime = _runtime.Runtime(object(), 2 << 20)
        assert len(created) == 1
        runtime.close()
