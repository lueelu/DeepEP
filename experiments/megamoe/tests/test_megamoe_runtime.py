# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""MegaMoE bootstrap checks with mocked native and collective backends."""

from types import SimpleNamespace

import pytest

from deep_ep import _runtime
from deep_ep._store import StoreGroup
from deep_ep_experimental.megamoe.api import _MegaRuntime


@pytest.mark.parametrize("backend", ["store", "gloo"])
@pytest.mark.parametrize(
    "case",
    ["same-host", "different-host", "missing-host", "mapping", "device", "engine", "bf16", "bf16-debug-mismatch"],
)
def test_megamoe_bootstrap(monkeypatch, backend, case):
    monkeypatch.setattr(_runtime, "endpoint", lambda: "tcp://127.0.0.1:19091")
    monkeypatch.setattr(_runtime, "_OWNERS", set())
    monkeypatch.setattr(_runtime, "_host_boot_id", lambda: "test-host")
    created = []

    def create(*args, bf16=False):
        created.append((*args, bf16))
        return SimpleNamespace(close=lambda: None)

    def core_runtime(*args):
        pytest.fail("MegaMoE must use its own native runtime factory")

    native = SimpleNamespace(
        runtime_busy=lambda: False, next_instance_id=lambda: 0, Runtime=core_runtime, create_megamoe_runtime=create
    )
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
        elif case == "bf16-debug-mismatch":
            peer[-1] ^= 16  # Peer enabled debug_collectives.
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
        "bf16-debug-mismatch": "debug_collectives",
    }
    bf16 = case in ("bf16", "bf16-debug-mismatch")
    if case in errors:
        with pytest.raises((ValueError, RuntimeError), match=errors[case]):
            _MegaRuntime(group, 2 << 20, bf16=bf16)
        assert not created
    else:
        runtime = _MegaRuntime(group, 2 << 20, bf16=bf16)
        try:
            assert len(created) == 1
            assert created[0][:3] == (0, 2, 2 << 20)
            assert created[0][-1] is bf16
            if bf16:
                with pytest.raises(RuntimeError, match="already owns"):
                    _MegaRuntime(group, 2 << 20, bf16=True)
                assert len(created) == 1
        finally:
            runtime.close()
