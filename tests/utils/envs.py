# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Lazy, explicitly requested HCCL setup. Import/collection touches no device."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import os


@dataclass(frozen=True)
class Launch:
    rank: int
    local_rank: int
    world_size: int


def read_launch(environ):
    try:
        rank, local_rank, world, local_world = (
            int(environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        )
        port = int(environ["MASTER_PORT"])
        address = environ["MASTER_ADDR"]
    except (KeyError, ValueError) as error:
        raise RuntimeError("Device tests require a torchrun launch") from error
    if (
        world < 1
        or not 0 <= rank < world
        or not 1 <= local_world <= world
        or not 0 <= local_rank < local_world
        or not address
        or not 1 <= port <= 65535
    ):
        raise RuntimeError("Invalid torchrun environment")
    return Launch(rank, local_rank, world)


@contextmanager
def owned_process_group(torch, launch):
    """Only destroy the group created here; never reuse or tear down a caller's group."""
    dist = torch.distributed
    if dist.is_initialized():
        raise RuntimeError("Device tests require a fresh process; a process group already exists")
    if not torch.npu.is_available() or launch.local_rank >= torch.npu.device_count():
        raise RuntimeError("Requested NPU is unavailable")
    torch.npu.set_device(launch.local_rank)
    try:
        dist.init_process_group(
            backend="hccl",
            init_method="env://",
            rank=launch.rank,
            world_size=launch.world_size,
            timeout=timedelta(seconds=90),
        )
        yield dist.group.WORLD
    finally:
        # Includes partial initialization; no teardown barrier that could hang
        # when a peer has already failed. Transport operations have a timeout.
        if dist.is_initialized():
            dist.destroy_process_group()


@contextmanager
def npu_session():
    launch = read_launch(os.environ)  # Reject missing launch before framework import.
    import torch
    import torch_npu  # noqa: F401 -- registers the NPU backend

    with owned_process_group(torch, launch) as group:
        yield torch, group, launch
