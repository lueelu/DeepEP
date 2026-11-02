# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by zhu-mingzhe71 2026
"""Route-table helpers matching CatCCOS ``moe_route_table.h`` exactly."""

from typing import List, Tuple


def _tokens_for(
    m: int,
    ep_size: int,
    local_expert_num: int,
    src_rank: int,
    dst_ep: int,
    local_expert: int,
) -> int:
    tokens_per_ep, ep_remainder = divmod(m, ep_size)
    rotated_ep = (dst_ep + ep_size - src_rank % ep_size) % ep_size
    pair_tokens = tokens_per_ep + int(rotated_ep < ep_remainder)
    tokens_per_expert, expert_remainder = divmod(pair_tokens, local_expert_num)
    rotation = (src_rank + dst_ep) % local_expert_num
    rotated_expert = (local_expert + local_expert_num - rotation) % local_expert_num
    return tokens_per_expert + int(rotated_expert < expert_remainder)


def build_global_route_matrix(m: int, ep_size: int, expert_num: int) -> List[List[int]]:
    """Build ``[source rank][global expert]`` token counts for balanced TP=1 routing."""
    if m <= 0 or ep_size <= 0 or expert_num <= 0:
        raise ValueError("m, ep_size, and expert_num must be positive")
    if ep_size > 8:
        raise ValueError("ep_size must not exceed 8")
    if expert_num % ep_size:
        raise ValueError("expert_num must be divisible by ep_size")

    local_expert_num = expert_num // ep_size
    if local_expert_num > 64:
        raise ValueError("local expert count must not exceed 64")
    matrix = [
        [
            _tokens_for(m, ep_size, local_expert_num, src_rank, dst_ep, local_expert)
            for dst_ep in range(ep_size)
            for local_expert in range(local_expert_num)
        ]
        for src_rank in range(ep_size)
    ]
    if any(sum(row) != m for row in matrix):
        raise RuntimeError("route construction did not preserve source token counts")
    for dst_ep in range(ep_size):
        received = sum(
            matrix[src][dst_ep * local_expert_num + local_expert]
            for src in range(ep_size)
            for local_expert in range(local_expert_num)
        )
        if received != m:
            raise RuntimeError("route construction did not preserve destination token counts")
    return matrix


def build_rank_route_tables(m: int, rank: int, ep_size: int, expert_num: int) -> Tuple[List[int], List[int]]:
    """Return the two flattened CatCCOS route tables for one rank."""
    if rank < 0 or rank >= ep_size:
        raise ValueError("rank must be in [0, ep_size)")
    matrix = build_global_route_matrix(m, ep_size, expert_num)
    local_expert_num = expert_num // ep_size
    local = list(matrix[rank])
    global_for_local = [
        matrix[src][rank * local_expert_num + local_expert]
        for src in range(ep_size)
        for local_expert in range(local_expert_num)
    ]
    return local, global_for_local


def build_rank_route_tensors(m: int, rank: int, ep_size: int, expert_num: int, device):
    """Build flat contiguous int64 NPU route tensors accepted by the operator."""
    local, global_for_local, _ = build_rank_route(m, rank, ep_size, expert_num, device)
    return local, global_for_local


def build_rank_route(m: int, rank: int, ep_size: int, expert_num: int, device):
    """Build NPU route tensors and the host-known destination row count.

    Returning ``received_rows`` alongside the device tensors lets callers size
    data-dependent outputs without synchronizing a device-side reduction back
    to the host.
    """
    import torch

    local, global_for_local = build_rank_route_tables(m, rank, ep_size, expert_num)
    return (
        torch.tensor(local, dtype=torch.int64, device=device),
        torch.tensor(global_for_local, dtype=torch.int64, device=device),
        sum(global_for_local),
    )
