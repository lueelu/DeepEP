# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Concurrent host protocol tests; no NPU/Gloo/HCCL is used."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep._store import StoreGroup


class MemoryStore:
    def __init__(self):
        self.values = {}
        self.changed = threading.Condition()

    def set(self, key, value):
        with self.changed:
            self.values[key] = value.encode() if isinstance(value, str) else value
            self.changed.notify_all()

    def get(self, key):
        with self.changed:
            if not self.changed.wait_for(lambda: key in self.values, timeout=3):
                raise TimeoutError(key)
            return self.values[key]

    def check(self, keys):
        with self.changed:
            return all(key in self.values for key in keys)

    def wait(self, keys, timeout):
        with self.changed:
            if not self.changed.wait_for(lambda: self.check(keys), timeout=timeout.total_seconds()):
                raise TimeoutError(f"missing {keys}")

    def compare_set(self, key, old, new):
        with self.changed:
            if self.values.get(key, b"") == old:
                self.set(key, new)
            return self.values[key]

    def add(self, key, value):
        with self.changed:
            result = int(self.values.get(key, b"0")) + value
            self.set(key, str(result))
            return result

    def delete_key(self, key):
        with self.changed:
            self.values.pop(key, None)


def together(groups, operation):
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [pool.submit(operation, group) for group in groups]
        return [future.result(timeout=10) for future in futures]


def groups(size, store=None, prefix="test"):
    store = store or MemoryStore()
    return [StoreGroup(store, rank, size, prefix=prefix, timeout=2) for rank in range(size)]


@pytest.mark.parametrize("size", [2, 8, 64, 128])
def test_gather_barrier_cleanup_and_close(size):
    peers = groups(size)
    store = peers[0]._store

    def work(group):
        for step in range(3):
            values = group.all_gather({"rank": group.rank(), "round": step}, f"round {step}")
            assert values == [{"rank": rank, "round": step} for rank in range(size)]
            group.barrier(f"round {step} done")
        group.close()
        group.close()

    together(peers, work)
    assert not any("collective/" in key for key in store.values)
    with pytest.raises(RuntimeError, match="closed"):
        peers[0].barrier()


def test_mismatched_stage_reports_rank_and_sequence_on_all_peers():
    def work(group):
        with pytest.raises(RuntimeError, match="collective order mismatch") as exc:
            group.barrier("timer" if group.rank() == 0 else "profiling")
        assert "collective/0" in str(exc.value)
        assert "profiling" in str(exc.value) and "timer" in str(exc.value)

    together(groups(4), work)


def test_abort_wakes_blocked_peers_and_preserves_first_error():
    peers = groups(3)

    def work(group):
        if group.rank() == 2:
            group.abort("original allocation failed")
            group.abort("secondary failure")
        else:
            with pytest.raises(RuntimeError, match="original allocation failed"):
                group.barrier("weights ready")

    together(peers, work)
    assert b"secondary" not in peers[0]._store.get("test/abort")


def test_missing_peer_times_out_with_stage_and_rank():
    group = StoreGroup(MemoryStore(), 0, 2, prefix="missing", timeout=0.02)
    with pytest.raises(RuntimeError, match="rank=0 stage=collective/0/barrier:weights"):
        group.barrier("weights")


@pytest.mark.parametrize("dst", [0, 2])
def test_chunked_accuracy_sum_and_bounded_key_lifetime(dst):
    peers = groups(4)
    arrays = [np.full((160, 1024), rank + 0.25, np.float32) for rank in range(4)]
    together(peers, lambda group: group.reduce_sum(arrays[group.rank()], dst, "reference"))
    np.testing.assert_array_equal(arrays[dst], 7.0)
    for rank in range(4):
        if rank != dst:
            np.testing.assert_array_equal(arrays[rank], rank + 0.25)
    assert not any("reduce/" in key or "collective/" in key for key in peers[0]._store.values)


def test_reduce_shape_mismatch_is_collective_error():
    def work(group):
        with pytest.raises(RuntimeError, match="reduce metadata mismatch"):
            group.reduce_sum(np.zeros((group.rank() + 1,), np.float32), 0, "reference")

    together(groups(2), work)


def test_session_namespaces_do_not_cross_talk():
    store = MemoryStore()
    first, second = groups(2, store, "first"), groups(2, store, "second")

    def work(group):
        assert group.all_gather(group._prefix, "config") == [group._prefix] * 2

    together(first + second, work)


def test_from_env_connects_to_agent_store_without_process_group(monkeypatch):
    store, calls = MemoryStore(), []
    for key, value in dict(
        RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29654", TORCHELASTIC_RESTART_COUNT="0"
    ).items():
        monkeypatch.setenv(key, value)

    def connect(*args, **kwargs):
        calls.append((args, kwargs))
        return store

    # 无 init_process_group/Gloo/HCCL 方法；调用任何一个都会失败。
    monkeypatch.setitem(sys.modules, "torch.distributed", SimpleNamespace(TCPStore=connect))
    group = StoreGroup.from_env()
    args, kwargs = calls[0]
    assert args[:4] == ("127.0.0.1", 29654, None, False)
    assert kwargs == {"use_libuv": False}
    assert group.rank() == 0 and group.size() == 1
    group.close()


def test_real_tcpstore_local_protocol():
    torch = pytest.importorskip("torch", reason="Real TCPStore test needs CPU PyTorch")
    from datetime import timedelta

    server = torch.distributed.TCPStore(
        "127.0.0.1", 0, None, True, timedelta(seconds=5), wait_for_workers=False, use_libuv=False
    )
    clients = [
        torch.distributed.TCPStore("127.0.0.1", server.port, None, False, timedelta(seconds=5), use_libuv=False)
        for _ in range(4)
    ]
    peers = [StoreGroup(client, rank, 4, prefix="actual", timeout=5) for rank, client in enumerate(clients)]

    def work(group):
        assert group.all_gather(group.rank(), "actual socket") == [0, 1, 2, 3]
        group.close()

    together(peers, work)
    assert not torch.distributed.is_initialized()


def test_torchrun_workers_reuse_agent_store_without_process_group(tmp_path):
    pytest.importorskip("torch", reason="Actual torchrun test needs CPU PyTorch")
    worker = tmp_path / "store_worker.py"
    root = str(Path(__file__).resolve().parents[3])
    worker.write_text(
        f"""
import sys
sys.path.insert(0, {root!r})
import torch.distributed as dist
from deep_ep._store import StoreGroup
def forbidden(*args, **kwargs):
    raise AssertionError('ProcessGroup must not be created')
dist.init_process_group = forbidden
group = StoreGroup.from_env(timeout=15)
assert group.all_gather(group.rank(), 'worker ranks') == [0, 1]
assert not dist.is_initialized()
group.close()
print('STORE_WORKER_OK', flush=True)
""",
        encoding="utf-8",
    )
    import os
    import socket

    env = dict(os.environ, USE_LIBUV="0")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [sys.executable, "-m", "torch.distributed.run"]
    if sys.platform == "win32":
        # CPU Windows wheel 无 libuv；仅给测试 agent 显式选择原生 TCPStore 后端。
        # worker 与生产环境使用同一 StoreGroup.from_env，仍是真实跨进程 socket。
        command = [
            sys.executable,
            "-c",
            "from functools import partial; "
            "import torch.distributed.elastic.rendezvous.static_tcp_rendezvous as r; "
            "r.TCPStore = partial(r.TCPStore, use_libuv=False); "
            "from torch.distributed.run import main; main()",
        ]
    result = subprocess.run(
        command + ["--master-addr=127.0.0.1", f"--master-port={port}", "--nnodes=1", "--nproc-per-node=2", str(worker)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("STORE_WORKER_OK") == 2
