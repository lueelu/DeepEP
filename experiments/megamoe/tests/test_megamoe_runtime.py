# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""MegaMoE bootstrap checks with mocked native and collective backends."""

from types import SimpleNamespace

import pytest

from deep_ep import _runtime
from deep_ep._store import StoreGroup
from deep_ep_experimental.megamoe.api import _MegaRuntime


@pytest.mark.parametrize("backend", ["store", "gloo"])
@pytest.mark.parametrize("case", ["same-host", "different-host", "missing-host", "mapping", "device", "engine"])
def test_megamoe_bootstrap(monkeypatch, backend, case):
    monkeypatch.setattr(_runtime, "endpoint", lambda: "tcp://127.0.0.1:19091")
    monkeypatch.setattr(_runtime, "_OWNER", None)
    monkeypatch.setattr(_runtime, "_host_boot_id", lambda: "test-host")
    created = []

    def create(*args, backend="pod"):
        created.append((*args, backend))
        return SimpleNamespace(close=lambda: None)

    def core_runtime(*args):
        pytest.fail("MegaMoE must use its own native runtime factory")

    native = SimpleNamespace(runtime_busy=lambda: False, Runtime=core_runtime, create_megamoe_runtime=create)
    monkeypatch.setattr(_runtime, "require_native", lambda operation: native)
    monkeypatch.setattr(
        _runtime,
        "npu_api",
        lambda: SimpleNamespace(
            current_device=lambda: 0, Stream=lambda **kwargs: SimpleNamespace(synchronize=lambda: None)
        ),
    )

    def exchange(local):
        peer = list(local)
        peer[6] = 1
        if case == "different-host":
            peer[4] += 1
            peer[6] = 0
        elif case == "missing-host":
            peer[4] = 0
        elif case == "mapping":
            peer[5] += 1
        elif case == "device":
            peer[6] = 0
        elif case == "engine":
            peer[7] = 1
        return [local, peer]

    if backend == "store":
        group = StoreGroup(None, 0, 2, prefix="test")
        monkeypatch.setattr(group, "all_gather", lambda local, stage: exchange(local))
    else:
        torch = pytest.importorskip("torch")
        import torch.distributed as dist

        group = object()
        monkeypatch.setattr(_runtime, "rank_and_world", lambda *args: (0, 2))
        monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")

        def gather(peers, local, group):
            for peer, values in zip(peers, exchange(local.tolist())):
                peer.copy_(torch.tensor(values, dtype=local.dtype))

        monkeypatch.setattr(dist, "all_gather", gather)

    errors = {
        "missing-host": "host identity is required",
        "mapping": "common visible-device mapping",
        "device": "distinct local NPU",
        "engine": "runtime engines",
    }
    if case in errors:
        with pytest.raises((ValueError, RuntimeError), match=errors[case]):
            _MegaRuntime(group, 2 << 20)
        assert not created
    else:
        runtime = _MegaRuntime(group, 2 << 20)
        try:
            assert len(created) == 1
            assert created[0][:3] == (0, 2, 2 << 20)
            assert created[0][-1] == "pod"
        finally:
            runtime.close()


@pytest.fixture
def unified_runtime_environment(monkeypatch):
    """CPU doubles for resource setup, without Torch or tensor readbacks."""
    monkeypatch.setattr(_runtime, "endpoint", lambda: "tcp://127.0.0.1:19091")
    monkeypatch.setattr(_runtime, "_OWNER", None)
    monkeypatch.setattr(_runtime, "_host_boot_id", lambda: "local-host")
    calls, events, aborted = [], [], []
    state = SimpleNamespace(local_error="", peer_error="", different_hosts=False, mismatch=False)

    class NativeSession:
        @property
        def peer_access_error(self):
            events.append("peer_access_error")
            return state.local_error

        def close(self):
            events.append("close")

        def barrier(self):
            pytest.fail("No device barrier before peer-access agreement")

    def create(rank, size, capacity, address, timeout, backend="pod"):
        calls.append((rank, size, capacity, address, timeout, backend))
        events.append("create")
        return NativeSession()

    def ordinary(*args, **kwargs):
        pytest.fail("MegaMoE must use the unified MegaMoE factory")

    monkeypatch.setattr(
        _runtime,
        "require_native",
        lambda operation: SimpleNamespace(runtime_busy=lambda: False, Runtime=ordinary, create_megamoe_runtime=create),
    )
    monkeypatch.setattr(
        _runtime,
        "npu_api",
        lambda: SimpleNamespace(
            current_device=lambda: 0,
            Stream=lambda **kwargs: SimpleNamespace(synchronize=lambda: events.append("stream-sync")),
        ),
    )
    group = StoreGroup(None, 0, 4, prefix="unified-runtime-test")

    def exchange(value, stage):
        events.append(stage)
        if stage == "runtime configuration":
            peers = []
            for rank in range(4):
                row = list(value)
                row[6] = rank
                if state.different_hosts and rank >= 2:
                    row[4] += 1
                    row[6] -= 2
                peers.append(row)
            if state.mismatch:
                peers[1][-1] ^= 32
            return peers
        peers = [value] * 4
        if state.peer_error:
            peers[1] = state.peer_error
        return peers

    monkeypatch.setattr(group, "all_gather", exchange)
    monkeypatch.setattr(group, "abort", aborted.append)
    return SimpleNamespace(group=group, state=state, calls=calls, events=events, aborted=aborted)


@pytest.mark.parametrize("backend", ["pod", "urma"])
def test_unified_factory_routes_backend_and_preserves_lifetime(unified_runtime_environment, backend):
    env = unified_runtime_environment
    runtime = _MegaRuntime(env.group, 2 << 20, backend=backend)
    assert env.calls == [(0, 4, 2 << 20, "tcp://127.0.0.1:19091", 120, backend)]
    assert ("peer_access_error" in env.events) == (backend == "urma")
    assert _runtime._OWNER is runtime
    runtime.close()
    assert env.events[-2:] == ["stream-sync", "close"]
    assert _runtime._OWNER is None
    runtime.close()
    assert env.events.count("close") == 1


@pytest.mark.parametrize("backend", ["unsupported", "auto", "", None, True])
def test_unified_factory_rejects_invalid_backend_before_initialization(unified_runtime_environment, backend):
    env = unified_runtime_environment
    with pytest.raises((TypeError, ValueError), match="backend"):
        _MegaRuntime(env.group, 2 << 20, backend=backend)
    assert env.calls == []
    assert _runtime._OWNER is None


def test_urma_multi_host_requires_peer_access_not_opt_in(unified_runtime_environment):
    env = unified_runtime_environment
    env.state.different_hosts = True
    runtime = _MegaRuntime(env.group, 2 << 20, backend="urma")
    assert len(env.calls) == 1 and "peer_access_error" in env.events
    assert env.aborted == []
    runtime.close()


@pytest.mark.parametrize("failure_origin", ["local", "peer"])
def test_peer_access_failure_is_shared_and_retains_owner(unified_runtime_environment, failure_origin):
    env = unified_runtime_environment
    if failure_origin == "local":
        env.state.local_error = "peer mapping unavailable"
    else:
        env.state.peer_error = "peer mapping unavailable"
    with pytest.raises(RuntimeError, match="peer mapping unavailable"):
        _MegaRuntime(env.group, 2 << 20, backend="urma")
    assert len(env.calls) == 1
    assert len(env.aborted) == 1
    assert _runtime._OWNER is not None
    assert "close" not in env.events and "stream-sync" not in env.events


def test_backend_mismatch_fails_before_shmem(unified_runtime_environment):
    env = unified_runtime_environment
    env.state.mismatch = True
    with pytest.raises(ValueError, match="Ranks disagree"):
        _MegaRuntime(env.group, 2 << 20, backend="urma")
    assert env.calls == []
    assert _runtime._OWNER is None


def test_backend_specific_capacity_limits(unified_runtime_environment):
    env = unified_runtime_environment
    with pytest.raises(ValueError, match="capacity"):
        _MegaRuntime(env.group, 2 << 30, backend="pod")
    assert env.calls == []
    runtime = _MegaRuntime(env.group, 2 << 30, backend="urma")
    assert env.calls[0][2] == 2 << 30
    runtime.close()
