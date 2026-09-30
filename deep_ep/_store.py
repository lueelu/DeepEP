# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""TCPStore host coordination for MegaMoE; no ProcessGroup or device transport."""

from contextlib import contextmanager
from datetime import timedelta
import json
import os
import uuid

_CONTROL_BYTES = 1024 * 1024
_CHUNK_BYTES = 256 * 1024


class StoreGroup:
    """Single-threaded coordinator. All ranks must call operations in the same order.

    Use from_env() once per torchrun worker, or supply an isolated Store namespace.
    The caller owns the Store server; close this group after collective SHMEM teardown.
    """

    def __init__(self, store, rank, size, *, prefix, timeout=300):
        if not 0 <= rank < size <= 128 or not prefix or timeout <= 0:
            raise ValueError("StoreGroup requires rank in [0,size), size<=128, prefix and positive timeout")
        self._store, self._rank, self._size = store, rank, size
        self._prefix, self._timeout = prefix.rstrip("/") + "/", timedelta(seconds=timeout)
        self._sequence = 0

    @classmethod
    def from_env(cls, timeout=300):
        """Connect to torchrun's existing TCPStore, without initializing a ProcessGroup."""
        from torch.distributed import TCPStore

        rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        # torchrun 的 agent 已监听 MASTER_PORT；worker 只连接，不争抢端口。
        store = TCPStore(
            os.environ["MASTER_ADDR"],
            int(os.environ["MASTER_PORT"]),
            None,
            False,
            timedelta(seconds=timeout),
            use_libuv=False,
        )
        key = "megamoe/session/" + os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
        if rank == 0:
            store.set(key, uuid.uuid4().hex)
        session = store.get(key).decode("ascii")
        group = cls(store, rank, size, prefix=f"megamoe/{session}", timeout=timeout)
        group.barrier("connect")
        print(f"[HOST_STORE] rank={rank}/{size} ready (no ProcessGroup)", flush=True)
        return group

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def _set(self, key, value):
        self._store.set(self._prefix + key, value)

    def _delete(self, key):
        self._store.delete_key(self._prefix + key)

    def abort(self, error):
        if self._store is None:
            return
        try:
            message = f"rank={self._rank}: {error}".encode("utf-8", errors="replace")[:_CONTROL_BYTES]
            self._store.compare_set(self._prefix + "abort", b"", message)
            # 登记等待 key 后再检查 abort，避免异常与阻塞等待之间的竞态。
            for peer in range(self._size):
                key = self._prefix + f"waiting/{peer}"
                if self._store.check([key]):
                    self._set(self._store.get(key).decode("utf-8"), b"")
        except Exception:
            pass  # Store 失联时保留原始异常；远端仍受超时和启动器约束。

    def _get(self, key):
        self._set(f"waiting/{self._rank}", key)

        def check_abort():
            if self._store.check([self._prefix + "abort"]):
                error = self._store.get(self._prefix + "abort").decode("utf-8", errors="replace")
                raise RuntimeError(f"waiting key={key}: peer aborted: {error}")

        check_abort()
        self._store.wait([self._prefix + key], self._timeout)
        check_abort()
        return self._store.get(self._prefix + key)

    @contextmanager
    def _operation(self, stage):
        if self._store is None:
            raise RuntimeError("StoreGroup is closed")
        try:
            yield
        except Exception as error:
            message = f"TCPStore rank={self._rank} stage={stage}: {error}"
            self.abort(message)
            raise RuntimeError(message) from error

    def all_gather(self, value, stage):
        """Gather JSON-compatible control data, with explicit operation order checking."""
        key = f"collective/{self._sequence}"
        self._sequence += 1
        with self._operation(f"{key}/{stage}"):
            packet = json.dumps(dict(stage=stage, value=value), allow_nan=False).encode("utf-8")
            if len(packet) > _CONTROL_BYTES:
                raise ValueError(f"control message exceeds {_CONTROL_BYTES} bytes")
            self._set(f"{key}/rank/{self._rank}", packet)
            if self._rank == 0:
                entries = [json.loads(self._get(f"{key}/rank/{peer}")) for peer in range(self._size)]
                if any(entry["stage"] != stage for entry in entries):
                    raise ValueError(f"collective order mismatch: {[(i, e['stage']) for i, e in enumerate(entries)]}")
                self._set(f"{key}/result", json.dumps([e["value"] for e in entries], allow_nan=False))
            result = json.loads(self._get(f"{key}/result"))
            # 最后一个读取者回收，单调序号不复用旧 key。
            if self._store.add(self._prefix + f"{key}/readers", 1) == self._size:
                for peer in range(self._size):
                    self._delete(f"{key}/rank/{peer}")
                self._delete(f"{key}/result")
                self._delete(f"{key}/readers")
            return result

    def barrier(self, stage="barrier"):
        self.all_gather(None, f"barrier:{stage}")

    def reduce_sum(self, array, dst, stage):
        """Sum CPU float32 reference output to dst in bounded chunks, outside timing."""
        import numpy as np

        with self._operation(stage):
            if array.dtype != np.float32 or not array.flags.c_contiguous or not 0 <= dst < self._size:
                raise ValueError("reduce_sum requires contiguous CPU float32 and a valid destination")
            metadata = [dst, list(array.shape), array.dtype.str]
            key = f"reduce/{self._sequence}"
            peers = self.all_gather(metadata, f"reduce:{stage}")
            if any(peer != metadata for peer in peers):
                raise ValueError(f"reduce metadata mismatch: {peers}")
            flat = array.reshape(-1)
            chunk = _CHUNK_BYTES // array.itemsize
            for start in range(0, flat.size, chunk):
                part = flat[start : start + chunk]
                self._set(f"{key}/{start}/{self._rank}", part.tobytes())
                if self._rank == dst:
                    total = np.zeros_like(part)
                    for peer in range(self._size):
                        payload = self._get(f"{key}/{start}/{peer}")
                        if len(payload) != part.nbytes:
                            raise ValueError(f"reduce payload size mismatch at rank={peer}")
                        total += np.frombuffer(payload, dtype=np.float32)
                        self._delete(f"{key}/{start}/{peer}")
                    part[:] = total
                self.barrier(f"{key}/{start}/read")

    def close(self):
        if self._store is None:
            return
        with self._operation("close"):
            self.barrier("close")
            if self._rank == 0:
                # rank0/agent 最后退出，确保客户端已读取末次屏障结果。
                for peer in range(1, self._size):
                    self._get(f"closed/{peer}")
            else:
                self._set(f"closed/{self._rank}", b"1")
            self._store = None
