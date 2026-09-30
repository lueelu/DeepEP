# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Shared V1/V2 resource owner; no EP algorithms are enabled by this module."""

import hashlib
import ipaddress
import os
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path
from uuid import UUID

from ._native import npu_api, require_native
from ._store import StoreGroup

_OWNER = None
_LOCK = threading.Lock()


def integer(name, value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def endpoint():
    value = os.environ.get("DEEPEP_SHMEM_ENDPOINT", "")
    if not value.startswith("tcp://") or len(value) >= 64:
        raise ValueError("Set DEEPEP_SHMEM_ENDPOINT=tcp://<IPv4>:<unused-port>")
    host, separator, port = value[6:].partition(":")
    if not separator or not port.isdecimal() or not 1024 <= int(port) <= 65535:
        raise ValueError("DEEPEP_SHMEM_ENDPOINT requires an unprivileged TCP port")
    ipaddress.IPv4Address(host)
    return value


def _host_boot_id():
    # hostname 可能在不同机器上重复；Linux boot_id 在本机各进程间一致。
    try:
        return UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()).hex
    except (OSError, ValueError):
        # 先参加配置交换，让所有 rank 一起失败，不能退回不可靠的 hostname。
        return ""


def rank_and_world(group, rank=None, world_size=None):
    """Resolve an explicit SHMEM identity or an existing framework group."""
    if group is None:
        from .buffers._elastic_layout import WORLD_SIZES

        integer("world_size", world_size, 2, 256)
        if world_size not in WORLD_SIZES:
            raise ValueError("Runtime supports EP2/4/8/16/32/64/128/256")
        integer("rank", rank, 0, world_size - 1)
        return rank, world_size
    if rank is not None or world_size is not None:
        raise ValueError("Pass either group or rank/world_size, not both")
    import torch.distributed as dist

    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be an initialized torch.distributed ProcessGroup")
    return dist.get_rank(group), dist.get_world_size(group)


class Runtime:
    """One explicit collective lifetime per process; a live owner is retained."""

    _megamoe = False

    def __init__(self, group, capacity, timeout=120, *, udma=False, rank=None, world_size=None):
        global _OWNER
        native = require_native("Runtime.__init__")
        integer("capacity", capacity, 2 << 20, (1 if self._megamoe else 32) << 30)
        if capacity % (2 << 20):
            raise ValueError("capacity must be aligned to 2 MiB")
        integer("timeout", timeout, 1, 300)
        address = endpoint()
        store_group = self._megamoe and isinstance(group, StoreGroup)
        if self._megamoe and (group is None or udma):
            raise ValueError("MegaMoE requires a ProcessGroup or StoreGroup and the MTE runtime")
        if store_group:
            if rank is not None or world_size is not None:
                raise ValueError("Pass either group or rank/world_size, not both")
            rank, size = group.rank(), group.size()
        else:
            rank, size = rank_and_world(group, rank, world_size)
        factory = native.Runtime
        if self._megamoe:
            integer("world_size", size, 2, 128)
            factory = getattr(native, "create_megamoe_runtime", None)
            if factory is None:
                raise RuntimeError("Rebuild with DEEPEP_BUILD_MEGAMOE=ON")
        else:
            from .buffers._elastic_layout import WORLD_SIZES

            if size not in WORLD_SIZES:
                raise ValueError("Runtime supports EP2/4/8/16/32/64/128/256")
        if group is not None and not store_group:
            import torch
            import torch.distributed as dist

            backend = str(dist.get_backend(group))
            if backend not in {"gloo", "hccl"}:
                raise NotImplementedError("Runtime bootstrap supports Gloo or HCCL process groups")
        npu = npu_api()
        self.device = npu.current_device()
        self.group = group
        self.low_latency_comm_args = None
        self.closed = False
        with _LOCK:
            busy = _OWNER is not None or native.runtime_busy()

            if group is None:
                if busy:
                    raise RuntimeError("This process already owns a SHMEM runtime")
            else:

                def digest(value):
                    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:7], "little")

                local_values = [capacity, timeout, digest(address), int(busy)]
                if self._megamoe:
                    host_id = _host_boot_id()
                    local_values.extend(
                        [
                            digest(host_id) if host_id else 0,
                            digest(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")),
                            self.device,
                        ]
                    )
                local_values.append(int(udma))
                if store_group:
                    values = group.all_gather(local_values, "runtime configuration")
                else:
                    local = torch.tensor(
                        local_values, dtype=torch.int64, device=f"npu:{self.device}" if backend == "hccl" else "cpu"
                    )
                    peers = [torch.empty_like(local) for _ in range(size)]
                    dist.all_gather(peers, local, group=group)
                    values = [value.cpu().tolist() for value in peers]
                if any(value[3] for value in values):
                    raise RuntimeError("A rank already owns a SHMEM runtime; destroy it collectively first")
                if any(value[:3] != values[0][:3] for value in values):
                    raise ValueError("Ranks disagree on workspace capacity, timeout or bootstrap endpoint")
                if any(value[-1] != values[0][-1] for value in values):
                    raise ValueError("Ranks disagree on MTE/URMA runtime engines")
                if self._megamoe:
                    invalid_hosts = [i for i, value in enumerate(values) if value[4] == 0]
                    if invalid_hosts:
                        raise RuntimeError(
                            f"Cannot read a valid Linux /proc/sys/kernel/random/boot_id on group ranks {invalid_hosts}; "
                            "host identity is required before SHMEM initialization"
                        )
                    hosts = {value[4] for value in values}
                    for host in hosts:
                        local = [value for value in values if value[4] == host]
                        if len({value[5] for value in local}) != 1:
                            raise ValueError("Ranks on one host must use a common visible-device mapping")
                    devices = {}
                    for peer_rank, value in enumerate(values):
                        key = (value[4], value[6])
                        if key in devices:
                            raise ValueError(
                                f"Each rank must use a distinct local NPU: group ranks {devices[key]} and "
                                f"{peer_rank} use local device {value[6]} on the same Linux host"
                            )
                        devices[key] = peer_rank
            self.rank, self.world_size = rank, size
            self.capacity = capacity
            self.notify_lock = threading.RLock()
            self.notify_failed = False
            self.notify_inflight = None
            self.stream = npu.Stream(device=self.device)
            self.native = (
                native.Runtime(rank, size, capacity, address, timeout, udma)
                if udma
                else factory(rank, size, capacity, address, timeout)
            )
            _OWNER = self
            if capacity == 8 << 30 and 16 <= size <= 128:
                # LL peers must finish initialization before any payload writes.
                self.native.barrier()

    def ensure_alive(self):
        if self.closed:
            raise RuntimeError("Buffer runtime has been destroyed")
        if npu_api().current_device() != self.device:
            raise RuntimeError("Use Buffer on its creating NPU device")

    def close(self):
        global _OWNER
        with getattr(self, "notify_lock", nullcontext()), _LOCK:
            if self.closed:
                return
            self.ensure_alive()
            self.stream.synchronize()
            self.native.close()
            self.notify_inflight = None
            self.low_latency_comm_args = None
            self.closed = True
            _OWNER = None

    def barrier(self):
        self.ensure_alive()
        npu_api().current_stream(self.device).synchronize()
        self.stream.synchronize()
        self.native.barrier()


def runtime_of(buffer):
    runtime = getattr(buffer, "_runtime", None)
    if runtime is None:
        raise RuntimeError("Buffer runtime was not initialized")
    return runtime


def warn_unclosed(buffer):
    runtime = getattr(buffer, "_runtime", None)
    if runtime is not None and not runtime.closed:
        warnings.warn(
            "Buffer.destroy() must be called collectively before process-group teardown; "
            "resources are retained to avoid a collective in garbage collection",
            ResourceWarning,
            stacklevel=2,
        )
