# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by zhu-mingzhe71 2026
"""EP_GMM_FUSED adapter for the public Session's UDMA communication domain."""

import ipaddress
import os
import socket
import hashlib
import importlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator, Optional

import torch

from deep_ep._native import require_native
from ._tiling import LARGE_EP_WORKSPACE_BYTES, WORKSPACE_BYTES


@dataclass(frozen=True)
class UdmaConfig:
    rank: int
    world_size: int
    init_method: str
    heap_size: int
    device: str
    mode: str = "default"
    members: tuple = ()
    ranks_per_node: int = 0
    workspace_data_bytes: int = WORKSPACE_BYTES
    pipeline_chunks: int = -1


_config: Optional[UdmaConfig] = None
_anchor: Optional[torch.Tensor] = None
_group_ref = None
_lock = threading.RLock()
_leases = 0
_context_owned = False
_session = None
_closing = False
_CONTROL_BYTES = 4096


def _env_int(name: str, default: Optional[int] = None) -> int:
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise RuntimeError(f"{name} is required")
        return default
    return int(value)


def _resolve_master_addr(host: str) -> str:
    """Return the numeric IPv4 address required by the SHMEM bootstrap."""
    try:
        return str(ipaddress.IPv4Address(host))
    except ipaddress.AddressValueError:
        try:
            return socket.gethostbyname(host)
        except OSError as error:
            raise RuntimeError(
                f"Unable to resolve UDMA bootstrap host {host!r} to an IPv4 address. "
                "Set UDMA_MASTER_ADDR to a reachable numeric IPv4 address."
            ) from error


def _node_width(rank, world_size, members):
    """Infer a contiguous torchrun world, or use the explicit group-local layout."""
    if world_size not in (64, 128):
        return 0
    override = os.environ.get("DEEPEP_UDMA_RANKS_PER_NODE")
    if override is not None:
        try:
            width = int(override)
        except ValueError:
            return -1
        return width if width == 0 or (0 < width < world_size and world_size % width == 0) else -1
    if members and members != tuple(range(world_size)):
        return 0
    try:
        width = int(os.environ["LOCAL_WORLD_SIZE"])
        if (0 < width < world_size and world_size % width == 0 and int(os.environ["WORLD_SIZE"]) == world_size
                and int(os.environ["RANK"]) == rank and int(os.environ["LOCAL_RANK"]) == rank % width
                and int(os.environ["GROUP_RANK"]) == rank // width):
            return width
    except (KeyError, ValueError):
        pass
    return 0


def _control_device(group, device):
    backend = str(torch.distributed.get_backend(group)).lower()
    if backend == "hccl":
        return device
    if backend == "gloo":
        return "cpu"
    raise ValueError(f"group bootstrap supports HCCL or Gloo, got {backend}")


def _agree(group, device, config, valid, uid_size):
    """Fixed-size exchange also detects partial/reconfigured initialization."""
    common = (config.world_size, config.heap_size, config.members, config.mode, uid_size,
              config.ranks_per_node, config.workspace_data_bytes, config.pipeline_chunks)
    digest = hashlib.sha256(json.dumps(common).encode()).digest()
    payload = torch.tensor([int(valid), int(_config is not None), *digest], dtype=torch.uint8, device=device)
    peers = [torch.empty_like(payload) for _ in range(config.world_size)]
    torch.distributed.all_gather(peers, payload, group=group)
    values = [peer.cpu().tolist() for peer in peers]
    if any(not value[0] for value in values) or any(value[1:] != values[0][1:] for value in values):
        raise RuntimeError("EP group initialization disagrees across ranks (domain/device/heap/state/UID ABI)")


def init_udma_group(
    *, group=None, rank=None, world_size=None, local_rank=None, init_method=None, heap_size=1 << 30
) -> UdmaConfig:
    """Initialize one process-local UDMA domain, collectively across its EP ranks.

    ``group=None`` preserves the PR's DEFAULT bootstrap and environment fallback.
    ``group=ep_group`` borrows a Gloo/HCCL ProcessGroup only to exchange an official
    SHMEM unique ID. It uses the currently bound NPU unless local_rank is supplied.
    A process can own one domain. Calls on all members must have the same order.
    Initialize, invoke operators and destroy in order on the same host thread.
    Existing external SHMEM instances cannot be implicitly adopted.
    """
    global _anchor, _config, _group_ref, _session, _closing
    with _lock:
        import torch_npu

        if _closing:
            raise RuntimeError("UDMA setup or cleanup did not complete; destroy the group collectively before reuse")
        if group is not None:
            if any(value is not None for value in (rank, world_size, init_method)):
                raise ValueError("group cannot be combined with rank, world_size or init_method")
            dist = torch.distributed
            if not dist.is_initialized():
                raise RuntimeError("initialize torch.distributed before using group")
            rank, world_size = dist.get_rank(group), dist.get_world_size(group)
            if rank < 0:
                raise ValueError("this process is not a member of group")
            members = tuple(dist.get_process_group_ranks(group))
            local_rank = torch_npu.npu.current_device() if local_rank is None else local_rank
            requested = UdmaConfig(rank, world_size, "", heap_size, f"npu:{local_rank}", "unique_id", members)
        else:
            rank = _env_int("RANK", 0) if rank is None else rank
            world_size = _env_int("WORLD_SIZE", 1) if world_size is None else world_size
            local_rank = _env_int("LOCAL_RANK", rank) if local_rank is None else local_rank
            if init_method is None:
                host = _resolve_master_addr(
                    os.environ.get("UDMA_MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
                )
                port = int(os.environ.get("UDMA_MASTER_PORT", int(os.environ.get("MASTER_PORT", "29500")) + 1))
                if not 1 <= port <= 65535:
                    raise ValueError("UDMA bootstrap port must be in [1, 65535]")
                init_method = f"tcp://{host}:{port}"
            requested = UdmaConfig(rank, world_size, init_method, heap_size, f"npu:{local_rank}")
        width = _node_width(rank, world_size, requested.members)
        chunks = os.environ.get("DEEPEP_LARGE_EP_PIPELINE_CHUNKS", "") if width > 0 else ""
        chunk_count = {"": -1, "0": 0, "1": 1, "2": 2, "4": 4}.get(chunks, -2)
        data_bytes = LARGE_EP_WORKSPACE_BYTES if width > 0 else WORKSPACE_BYTES
        requested = replace(requested, ranks_per_node=width, workspace_data_bytes=data_bytes, pipeline_chunks=chunk_count)
        workspace_bytes = requested.workspace_data_bytes + _CONTROL_BYTES
        valid = (
            all(isinstance(v, int) and not isinstance(v, bool) for v in (rank, world_size, heap_size, local_rank))
            and 0 <= rank < world_size <= 128
            and local_rank >= 0
            and width >= 0 and chunk_count >= -1 and workspace_bytes < heap_size < 1 << 63
            and (_config is None or (requested == _config and group is _group_ref))
        )
        if group is None and not valid:
            raise ValueError("invalid or incompatible UDMA rank/world/device/heap configuration")
        torch_npu.npu.set_device(local_rank)
        core = require_native("EP_GMM_FUSED")
        if not hasattr(core, "create_udma_runtime"):
            raise RuntimeError("Install a wheel built with DEEPEP_BUILD_EP_GMM_FUSED=ON")
        importlib.import_module(f"{__package__}._C")
        anchor = _anchor if _anchor is not None else torch.empty(0, dtype=torch.uint8, device=requested.device)
        native = torch.ops.deep_ep_ep_gmm_fused
        if group is not None:
            device = _control_device(group, requested.device)
            uid_size = core.udma_unique_id_size()
            valid = valid and (_config is not None or not core.runtime_busy())
            _agree(group, device, requested, valid, uid_size)
        if _config is not None:
            return _config
        uid_bytes = []
        if group is not None:
            # The source argument is a GLOBAL rank, even when EP does not include 0.
            root = members[0]
            uid = torch.empty(uid_size, dtype=torch.uint8, device=device)
            status = torch.ones(1, dtype=torch.uint8, device=device)
            root_error = None
            if rank == 0:
                try:
                    uid.copy_(torch.tensor(core.udma_unique_id(), dtype=torch.uint8, device=device))
                except Exception as error:
                    root_error = error
                    status.zero_()
            dist.broadcast(status, src=root, group=group)
            if not status.item():
                raise RuntimeError("EP root failed to create SHMEM unique ID") from root_error
            dist.broadcast(uid, src=root, group=group)
            uid_bytes = uid.cpu().tolist()
            torch_npu.npu.synchronize()
        session = core.create_udma_runtime(
            rank, world_size, heap_size, workspace_bytes, requested.init_method, uid_bytes, width > 0
        )
        # Retain the owner even if attachment fails. Cleanup is an explicit
        # collective; never start it implicitly on one failed rank.
        _session = session
        _anchor, _config, _group_ref = anchor, requested, group
        _closing = True
        native._attach_udma(anchor, rank, world_size, core.udma_workspace_address(session), session.capacity,
                            width, chunk_count)
        _closing = False
        return requested


def destroy_udma_group() -> None:
    """Synchronize and free owned resources; never destroy the borrowed ProcessGroup."""
    global _anchor, _config, _group_ref, _session, _closing
    with _lock:
        if _leases:
            raise RuntimeError("cannot destroy UDMA while udma_group contexts are active")
        if _session is not None:
            import torch_npu

            _closing = True
            with torch_npu.npu.device(_config.device):
                # Stop borrowing only after all kernels using the workspace finish.
                torch.ops.deep_ep_ep_gmm_fused._detach_udma(_anchor)
                _session.close()
            _anchor = _config = _group_ref = _session = None
            _closing = False


def get_udma_config() -> UdmaConfig:
    if _closing:
        raise RuntimeError("UDMA setup or cleanup did not complete; the workspace is unavailable")
    if _config is None:
        raise RuntimeError("UDMA is not initialized; call init_udma_group() first")
    return _config


@contextmanager
def udma_group(**kwargs) -> Iterator[UdmaConfig]:
    global _leases, _context_owned
    with _lock:
        was_initialized = _config is not None
        config = init_udma_group(**kwargs)
        if _leases == 0:
            _context_owned = not was_initialized
        _leases += 1
    try:
        yield config
    finally:
        with _lock:
            _leases -= 1
            if _leases == 0 and _context_owned:
                _context_owned = False
                destroy_udma_group()
