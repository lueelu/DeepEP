# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DeepEP-compatible distributed environment helpers for Ascend NPU."""

from __future__ import annotations

import inspect
import os
import random
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


_local_rank: Optional[int] = None
_local_seed: Optional[int] = None
_global_seed: Optional[int] = None


def init_seed(global_seed: int) -> None:
    """Initialize Python and PyTorch seeds using DeepEP's rank-local rule."""

    global _local_seed, _global_seed
    _local_seed = global_seed + dist.get_rank()
    _global_seed = global_seed
    torch.manual_seed(_local_seed)
    random.seed(_local_seed)


def get_local_seed() -> Optional[int]:
    return _local_seed


def get_global_seed() -> Optional[int]:
    return _global_seed


def dist_print(value: str = "", once_in_node: bool = False) -> None:
    """Print collectively using the same behavior as DeepEP's helper."""

    if _local_rank is None:
        raise RuntimeError("init_dist must be called before dist_print")
    if not once_in_node or _local_rank == 0:
        print(value, flush=True)
    dist.barrier()


def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0) -> Tuple[int, int, dist.ProcessGroup]:
    """Initialize DeepEP's spawned-process topology with the HCCL backend.

    As in upstream DeepEP, ``WORLD_SIZE`` and ``RANK`` describe node count and
    node rank when the caller spawns ``num_local_ranks`` processes per node.
    The returned process group contains every resulting global rank.
    """

    if not isinstance(local_rank, int) or isinstance(local_rank, bool):
        raise TypeError("local_rank must be an integer")
    if not isinstance(num_local_ranks, int) or isinstance(num_local_ranks, bool):
        raise TypeError("num_local_ranks must be an integer")
    if num_local_ranks <= 0 or local_rank < 0 or local_rank >= num_local_ranks:
        raise ValueError("local_rank must be in [0, num_local_ranks)")
    if dist.is_initialized():
        raise RuntimeError("torch.distributed is already initialized")

    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", "1"))
    node_rank = int(os.getenv("RANK", "0"))
    world_size = num_nodes * num_local_ranks
    rank = node_rank * num_local_ranks + local_rank
    if num_nodes <= 0 or node_rank < 0 or node_rank >= num_nodes:
        raise ValueError("RANK must be in [0, WORLD_SIZE) for node-spawn mode")

    global _local_rank
    _local_rank = local_rank
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.npu.set_device(local_rank)
    parameters = {
        "backend": "hccl",
        "init_method": f"tcp://{master_addr}:{master_port}",
        "world_size": world_size,
        "rank": rank,
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        parameters["device_id"] = torch.device(f"npu:{local_rank}")
    dist.init_process_group(**parameters)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("npu")
    init_seed(seed)
    group = dist.new_group(list(range(world_size)))
    return rank, world_size, group


def get_physical_domain_size(
    group: dist.ProcessGroup,
) -> Tuple[int, int]:
    """Return the current intranode CANN SHMEM topology."""

    return 1, dist.get_world_size(group)


def get_logical_domain_size(group: dist.ProcessGroup, allow_hybrid_mode: bool = True) -> Tuple[int, int]:
    """Return one scale-out domain and all ranks in the scale-up domain."""

    del allow_hybrid_mode
    return 1, dist.get_world_size(group)


def check_nvlink_connections(group: dist.ProcessGroup) -> None:
    """Compatibility no-op; CANN SHMEM validates the selected transport."""

    del group


def check_torch_deterministic() -> None:
    if torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory:
        raise AssertionError("deterministic algorithms and fill_uninitialized_memory cannot both be enabled")


@lru_cache()
def get_nvlink_gbs(factor: float = 0.9) -> float:
    """NVLink is not present on Ascend; retain the API with a zero value."""

    del factor
    return 0.0


@lru_cache()
def check_fast_rdma_atomic_support(nic_name: str = "") -> bool:
    del nic_name
    return False


@lru_cache()
def get_rdma_gbs(nic_name: str = "") -> float:
    """Inter-node RDMA benchmarking is not implemented by this backend."""

    del nic_name
    return 0.0


__all__ = [
    "check_fast_rdma_atomic_support",
    "check_nvlink_connections",
    "check_torch_deterministic",
    "dist_print",
    "get_global_seed",
    "get_local_seed",
    "get_logical_domain_size",
    "get_nvlink_gbs",
    "get_physical_domain_size",
    "get_rdma_gbs",
    "init_dist",
    "init_seed",
]
