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

    def create(*args, backend="mte"):
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
            peer[-1] = 1
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
            assert created[0][-1] == "mte"
        finally:
            runtime.close()


@pytest.mark.parametrize(
    "world,ranks_per_node,case,error",
    [
        (32, 16, "normal", None),
        (32, 16, "qp-mismatch", "QP configuration"),
        (16, 16, "no-rdma", None),
        (32, 8, "no-rdma", None),
        (32, 16, "no-rdma", "requires HNS_1825"),
        (32, 16, "xscale", "requires HNS_1825"),
        (32, 16, "sdk-mismatch", "requires HNS_1825"),
        (32, 16, "peer-no-rdma", "requires HNS_1825"),
        (32, 16, "backend-mismatch", "disagree on SHMEM RDMA backend"),
        (32, 16, "interleaved", "contiguous groups of 16"),
        (32, 16, "one-host", "contiguous groups of 16"),
        (32, 16, "config-mismatch", "disagree on ranks_per_node"),
        (64, 16, "normal", "supports EP16 or two-host EP32"),
    ],
)
def test_megamoe_roce_collective_bootstrap(monkeypatch, world, ranks_per_node, case, error):
    monkeypatch.setattr(_runtime, "endpoint", lambda: "tcp://127.0.0.1:19091")
    monkeypatch.setattr(_runtime, "_OWNER", None)
    monkeypatch.setattr(_runtime, "_host_boot_id", lambda: "test-host")
    created = []
    native = SimpleNamespace(
        Runtime=lambda *args: pytest.fail("Unexpected core runtime"),
        runtime_busy=lambda: False,
        create_megamoe_runtime=lambda *args, backend: created.append((*args, backend)) or SimpleNamespace(close=lambda **kwargs: None),
        shmem_rdma_backend="NONE" if case == "no-rdma" else ("XSCALE" if case == "xscale" else "HNS_1825"),
        megamoe_combine_qps=16,
        shmem_sdk_matches=case != "sdk-mismatch",
    )
    monkeypatch.setattr(_runtime, "require_native", lambda operation: native)
    monkeypatch.setattr(
        _runtime,
        "npu_api",
        lambda: SimpleNamespace(
            current_device=lambda: 0,
            synchronize=lambda device: None,
            Stream=lambda **kwargs: SimpleNamespace(synchronize=lambda: None),
        ),
    )

    def exchange(local, stage):
        if stage == "RoCE runtime close":
            return [local] * world
        peers = []
        for rank in range(world):
            peer = list(local)
            peer[4] += rank // ranks_per_node
            peer[6] = rank % ranks_per_node
            if case == "interleaved":
                peer[4], peer[6] = local[4] + rank % 2, rank // 2
            elif case == "one-host":
                peer[4], peer[6] = local[4], rank
            if rank == world - 1:
                if case == "peer-no-rdma":
                    peer[8] = 0
                elif case == "backend-mismatch":
                    peer[8] = 1
                elif case == "config-mismatch":
                    peer[7] = 8
                elif case == "qp-mismatch":
                    peer[-1] += 16 * 4
            peers.append(peer)
        return peers

    group = StoreGroup(None, 0, world, prefix="test-roce")
    monkeypatch.setattr(group, "all_gather", exchange)
    if error is not None:
        with pytest.raises((ValueError, RuntimeError), match=error):
            _MegaRuntime(group, 2 << 20, ranks_per_node=ranks_per_node)
        assert not created
        assert _runtime._OWNER is None
    else:
        runtime = _MegaRuntime(group, 2 << 20, ranks_per_node=ranks_per_node)
        try:
            assert runtime.use_roce == (ranks_per_node == 16 and world > 16)
            assert len(created) == 1
            assert created[0][5:] == (("ep32" if runtime.use_roce else "mte"),)
        finally:
            runtime.close()
