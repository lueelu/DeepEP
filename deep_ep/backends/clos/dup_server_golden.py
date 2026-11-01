"""CPU golden metadata builders for the duplicate-server prototype.

This module does not launch Notify or dispatch kernels. It builds source-side
communication inputs and destination-side dispatch metadata for validation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Union

import torch


_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)
_INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class DupServerDispatchMeta:
    """Precomputed no-weight dispatch metadata for one rank.

    ``recv_route_position`` and ``recv_src_rank`` are ordered by
    ``local_expert`` first and ``src_rank`` second. Segment
    ``local_expert * world_size + src_rank`` is delimited by two adjacent
    entries in ``recv_segment_offset``. ``recv_expert_offset`` gives the same
    layout at expert granularity. ``recv_chunk_token_end[src_rank, chunk]``
    stores the original-token upper bound of the matching compact Dispatch chunk.
    """

    send_token_count: torch.Tensor
    send_token_ids: torch.Tensor
    send_gateway_lane: torch.Tensor
    recv_route_position: torch.Tensor
    recv_gateway_lane: torch.Tensor
    recv_src_rank: torch.Tensor
    recv_segment_offset: torch.Tensor
    recv_chunk_token_end: torch.Tensor
    recv_expert_offset: torch.Tensor
    num_recv_tokens: int
    num_recv_tokens_per_expert: List[int]
    rank: int
    world_size: int
    ranks_per_server: int
    num_servers: int
    num_tokens: int
    num_topk: int
    num_experts: int
    num_local_experts: int
    num_max_tokens_per_rank: int
    num_tokens_per_chunk: int
    gateway_policy: str

    def to(
        self,
        device: Union[str, torch.device],
        non_blocking: bool = False,
    ) -> "DupServerDispatchMeta":
        """Move tensor fields to ``device`` while preserving host metadata."""

        return replace(
            self,
            send_token_count=self.send_token_count.to(device=device, non_blocking=non_blocking),
            send_token_ids=self.send_token_ids.to(device=device, non_blocking=non_blocking),
            send_gateway_lane=self.send_gateway_lane.to(device=device, non_blocking=non_blocking),
            recv_route_position=self.recv_route_position.to(device=device, non_blocking=non_blocking),
            recv_gateway_lane=self.recv_gateway_lane.to(device=device, non_blocking=non_blocking),
            recv_src_rank=self.recv_src_rank.to(device=device, non_blocking=non_blocking),
            recv_segment_offset=self.recv_segment_offset.to(device=device, non_blocking=non_blocking),
            recv_chunk_token_end=self.recv_chunk_token_end.to(device=device, non_blocking=non_blocking),
            recv_expert_offset=self.recv_expert_offset.to(device=device, non_blocking=non_blocking),
            num_recv_tokens_per_expert=list(self.num_recv_tokens_per_expert),
        )


@dataclass(frozen=True)
class DupServerSourceNotifyMeta:
    """Communication-ready source metadata for one rank."""

    send_token_count: torch.Tensor
    send_token_ids: torch.Tensor
    send_gateway_lane: torch.Tensor
    send_route_count: torch.Tensor
    send_expert_prefix: torch.Tensor
    packed_route_position: torch.Tensor
    packed_gateway_lane: torch.Tensor
    source_server_mask: torch.Tensor
    packed_topk_weights: Optional[torch.Tensor]


def build_dup_server_golden_notify(
    all_topk_idx: torch.Tensor,
    rank: int,
    num_experts: int,
    *,
    num_max_tokens_per_rank: Optional[int] = None,
    ranks_per_server: int = 8,
    num_tokens_per_chunk: int = 128,
    gateway_policy: str = "source_lane",
) -> DupServerDispatchMeta:
    """Build no-weight dispatch metadata for one rank on the CPU.

    Args:
        all_topk_idx: Global routing snapshot with shape ``[R, N, K]``. Values
            must be ``-1`` or a global expert ID in ``[0, num_experts)``.
        rank: Rank whose source-send and destination-receive metadata is built.
        num_experts: Number of global experts. Experts are assigned to ranks in
            contiguous rank-major order.
        num_max_tokens_per_rank: Capacity of each fixed server token list. It
            defaults to ``N`` and must be at least ``N``.
        ranks_per_server: Number of strictly consecutive ranks in one server.
        num_tokens_per_chunk: Number of compact destination-server tokens in one
            Dispatch chunk.
        gateway_policy: Only ``source_lane`` is supported. Each destination
            server uses the same lane as the source rank.

    Returns:
        CPU metadata. Call ``meta.to("npu")`` before passing it to a future
        dispatch kernel. Golden construction and the copy must remain outside
        the measured dispatch interval.
    """

    _validate_inputs(
        all_topk_idx,
        rank,
        num_experts,
        num_max_tokens_per_rank,
        ranks_per_server,
    )

    routes = all_topk_idx.detach().contiguous()
    world_size, num_tokens, num_topk = map(int, routes.shape)
    capacity = num_tokens if num_max_tokens_per_rank is None else int(num_max_tokens_per_rank)
    num_servers = world_size // ranks_per_server
    num_local_experts = num_experts // world_size
    experts_per_server = num_local_experts * ranks_per_server
    _validate_gateway_policy(gateway_policy)
    if not isinstance(num_tokens_per_chunk, int) or num_tokens_per_chunk <= 0:
        raise ValueError("num_tokens_per_chunk must be a positive integer")

    send_token_count, send_token_ids, send_gateway_lane = _build_source_send_metadata(
        routes[rank],
        num_servers=num_servers,
        experts_per_server=experts_per_server,
        num_local_experts=num_local_experts,
        source_lane=rank % ranks_per_server,
        source_server=rank // ranks_per_server,
        gateway_policy=gateway_policy,
        capacity=capacity,
    )
    (
        recv_route_position,
        recv_gateway_lane,
        recv_src_rank,
        recv_segment_offset,
        recv_expert_offset,
        num_recv_tokens_per_expert,
    ) = _build_destination_route_metadata(
        routes,
        rank=rank,
        num_local_experts=num_local_experts,
        ranks_per_server=ranks_per_server,
        gateway_policy=gateway_policy,
    )
    recv_chunk_token_end = _build_recv_chunk_token_end(
        routes,
        destination_server=rank // ranks_per_server,
        experts_per_server=experts_per_server,
        capacity=capacity,
        num_tokens_per_chunk=num_tokens_per_chunk,
    )

    return DupServerDispatchMeta(
        send_token_count=send_token_count,
        send_token_ids=send_token_ids,
        send_gateway_lane=send_gateway_lane,
        recv_route_position=recv_route_position,
        recv_gateway_lane=recv_gateway_lane,
        recv_src_rank=recv_src_rank,
        recv_segment_offset=recv_segment_offset,
        recv_chunk_token_end=recv_chunk_token_end,
        recv_expert_offset=recv_expert_offset,
        num_recv_tokens=int(recv_route_position.numel()),
        num_recv_tokens_per_expert=num_recv_tokens_per_expert,
        rank=rank,
        world_size=world_size,
        ranks_per_server=ranks_per_server,
        num_servers=num_servers,
        num_tokens=num_tokens,
        num_topk=num_topk,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        num_max_tokens_per_rank=capacity,
        num_tokens_per_chunk=num_tokens_per_chunk,
        gateway_policy=gateway_policy,
    )


def build_dup_server_golden_source_notify(
    topk_idx: torch.Tensor,
    num_experts: int,
    *,
    topk_weights: Optional[torch.Tensor] = None,
    world_size: int,
    source_rank: int = 0,
    num_max_tokens_per_rank: Optional[int] = None,
    ranks_per_server: int = 8,
    gateway_policy: str = "source_lane",
) -> DupServerSourceNotifyMeta:
    """Build the source-side outputs produced before Notify communication."""

    if not isinstance(topk_idx, torch.Tensor):
        raise TypeError("topk_idx must be a torch.Tensor")
    if topk_idx.device.type != "cpu" or topk_idx.dim() != 2:
        raise ValueError("topk_idx must be a CPU tensor with shape [N, K]")
    if topk_idx.dtype not in _INTEGER_DTYPES:
        raise TypeError("topk_idx must use an integer dtype")
    if ranks_per_server <= 0:
        raise ValueError("ranks_per_server must be positive")
    if world_size <= 0 or world_size % ranks_per_server:
        raise ValueError("world_size must be divisible by ranks_per_server")
    if not isinstance(source_rank, int) or not 0 <= source_rank < world_size:
        raise ValueError("source_rank must be in [0, world_size)")
    if world_size // ranks_per_server > 64:
        raise ValueError("source_server_mask supports at most 64 servers")
    if num_experts <= 0 or num_experts % world_size:
        raise ValueError("num_experts must be positive and divisible by world_size")
    _validate_gateway_policy(gateway_policy)

    num_tokens, num_topk = map(int, topk_idx.shape)
    capacity = num_tokens if num_max_tokens_per_rank is None else num_max_tokens_per_rank
    if num_tokens <= 0 or num_topk <= 0:
        raise ValueError("N and K must be positive")
    if capacity < num_tokens:
        raise ValueError("num_max_tokens_per_rank must be no smaller than N")
    if capacity * num_topk > _INT32_MAX:
        raise ValueError("num_max_tokens_per_rank*K must fit int32")
    invalid = topk_idx.lt(-1) | topk_idx.ge(num_experts)
    if bool(invalid.any()):
        first = torch.nonzero(invalid, as_tuple=False)[0].tolist()
        raise ValueError(f"topk_idx contains an invalid expert ID at {tuple(first)}")

    if topk_weights is not None:
        if not isinstance(topk_weights, torch.Tensor):
            raise TypeError("topk_weights must be a torch.Tensor or None")
        if topk_weights.device.type != "cpu" or topk_weights.dtype != torch.float32:
            raise TypeError("topk_weights must be a CPU float32 tensor")
        if tuple(topk_weights.shape) != tuple(topk_idx.shape):
            raise ValueError("topk_weights must have the same shape as topk_idx")

    num_servers = world_size // ranks_per_server
    experts_per_server = num_experts // num_servers
    num_local_experts = num_experts // world_size
    route_capacity = capacity * num_topk
    send_token_count, send_token_ids, send_gateway_lane = _build_source_send_metadata(
        topk_idx,
        num_servers=num_servers,
        experts_per_server=experts_per_server,
        num_local_experts=num_local_experts,
        source_lane=source_rank % ranks_per_server,
        source_server=source_rank // ranks_per_server,
        gateway_policy=gateway_policy,
        capacity=capacity,
    )

    flat_topk = topk_idx.reshape(-1).to(torch.int64)
    valid_positions = torch.nonzero(flat_topk.ge(0), as_tuple=False).flatten()
    valid_experts = flat_topk[valid_positions]
    order = torch.argsort(valid_experts, stable=True)
    sorted_positions = valid_positions[order]
    expert_count = torch.bincount(valid_experts, minlength=num_experts).to(torch.int32)
    server_expert_count = expert_count.reshape(num_servers, experts_per_server)
    send_expert_prefix = torch.cat(
        (
            torch.zeros((num_servers, 1), dtype=torch.int32),
            torch.cumsum(server_expert_count, dim=1, dtype=torch.int32),
        ),
        dim=1,
    ).contiguous()
    send_route_count = send_expert_prefix[:, -1].contiguous()
    global_expert_prefix = torch.cat(
        (
            torch.zeros((1,), dtype=torch.int64),
            torch.cumsum(expert_count, dim=0, dtype=torch.int64),
        )
    )

    packed_route_position = torch.full((route_capacity,), -1, dtype=torch.int32)
    packed_gateway_lane = torch.full((route_capacity,), 0xFF, dtype=torch.uint8)
    packed_topk_weights = (
        torch.zeros((num_servers, route_capacity), dtype=torch.float32) if topk_weights is not None else None
    )
    flat_weights = topk_weights.reshape(-1) if topk_weights is not None else None
    for dst_server in range(num_servers):
        first_expert = dst_server * experts_per_server
        route_begin = int(global_expert_prefix[first_expert].item())
        route_end = int(global_expert_prefix[first_expert + experts_per_server].item())
        count = route_end - route_begin
        server_positions = sorted_positions[route_begin:route_end]
        packed_route_position[route_begin:route_end] = server_positions.to(torch.int32)
        for packed_index, route_position in enumerate(server_positions.tolist(), start=route_begin):
            token_id = route_position // num_topk
            packed_gateway_lane[packed_index] = _gateway_lane_for_token_server(
                topk_idx[token_id],
                destination_server=dst_server,
                experts_per_server=experts_per_server,
                num_local_experts=num_local_experts,
                source_lane=source_rank % ranks_per_server,
                source_server=source_rank // ranks_per_server,
                gateway_policy=gateway_policy,
            )
        if packed_topk_weights is not None and flat_weights is not None:
            packed_topk_weights[dst_server, :count] = flat_weights[server_positions]

    source_server_mask = torch.zeros((num_tokens,), dtype=torch.int64)
    for dst_server in range(num_servers):
        count = int(send_token_count[dst_server].item())
        token_ids = send_token_ids[dst_server, :count].to(torch.int64)
        server_bit = torch.tensor(1, dtype=torch.int64) << dst_server
        source_server_mask[token_ids] |= server_bit

    return DupServerSourceNotifyMeta(
        send_token_count=send_token_count,
        send_token_ids=send_token_ids,
        send_gateway_lane=send_gateway_lane,
        send_route_count=send_route_count,
        send_expert_prefix=send_expert_prefix,
        packed_route_position=packed_route_position,
        packed_gateway_lane=packed_gateway_lane,
        source_server_mask=source_server_mask,
        packed_topk_weights=packed_topk_weights,
    )


def _build_source_send_metadata(
    local_topk_idx: torch.Tensor,
    *,
    num_servers: int,
    experts_per_server: int,
    num_local_experts: int,
    source_lane: int,
    source_server: int,
    gateway_policy: str,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_topk = map(int, local_topk_idx.shape)
    token_ids = torch.arange(num_tokens, dtype=torch.int64).view(-1, 1)
    token_ids = token_ids.expand(num_tokens, num_topk)
    valid = local_topk_idx.ge(0)

    membership = torch.zeros((num_tokens, num_servers), dtype=torch.bool)
    if bool(valid.any()):
        dst_servers = torch.div(
            local_topk_idx[valid].to(torch.int64),
            experts_per_server,
            rounding_mode="floor",
        )
        membership[token_ids[valid], dst_servers] = True

    send_token_count = membership.sum(dim=0, dtype=torch.int32)
    send_token_ids = torch.full((num_servers, capacity), -1, dtype=torch.int32)
    send_gateway_lane = torch.full((num_servers, capacity), 0xFF, dtype=torch.uint8)
    for dst_server in range(num_servers):
        ids = torch.nonzero(membership[:, dst_server], as_tuple=False).flatten()
        if ids.numel():
            send_token_ids[dst_server, : ids.numel()] = ids.to(torch.int32)
            for compact_index, token_id in enumerate(ids.tolist()):
                send_gateway_lane[dst_server, compact_index] = _gateway_lane_for_token_server(
                    local_topk_idx[token_id],
                    destination_server=dst_server,
                    experts_per_server=experts_per_server,
                    num_local_experts=num_local_experts,
                    source_lane=source_lane,
                    source_server=source_server,
                    gateway_policy=gateway_policy,
                )

    return send_token_count.contiguous(), send_token_ids, send_gateway_lane


def _build_destination_route_metadata(
    all_topk_idx: torch.Tensor,
    *,
    rank: int,
    num_local_experts: int,
    ranks_per_server: int,
    gateway_policy: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    world_size = int(all_topk_idx.size(0))
    num_topk = int(all_topk_idx.size(2))
    num_experts = num_local_experts * world_size
    num_servers = world_size // ranks_per_server
    experts_per_server = num_experts // num_servers
    destination_server = rank // ranks_per_server
    first_expert = rank * num_local_experts
    route_segments: List[torch.Tensor] = []
    gateway_lane_segments: List[torch.Tensor] = []
    source_segments: List[torch.Tensor] = []
    segment_offsets = [0]
    expert_offsets = [0]
    expert_counts: List[int] = []

    flattened_by_source = [all_topk_idx[src_rank].reshape(-1) for src_rank in range(world_size)]
    total_routes = 0
    for local_expert in range(num_local_experts):
        global_expert = first_expert + local_expert
        expert_route_count = 0
        for src_rank in range(world_size):
            route_positions = (
                torch.nonzero(
                    flattened_by_source[src_rank].eq(global_expert),
                    as_tuple=False,
                )
                .flatten()
                .to(torch.int32)
            )
            route_segments.append(route_positions)
            gateway_lane_segments.append(
                torch.tensor(
                    [
                        _gateway_lane_for_token_server(
                            all_topk_idx[src_rank, int(position) // num_topk],
                            destination_server=destination_server,
                            experts_per_server=experts_per_server,
                            num_local_experts=num_local_experts,
                            source_lane=src_rank % ranks_per_server,
                            source_server=src_rank // ranks_per_server,
                            gateway_policy=gateway_policy,
                        )
                        for position in route_positions.tolist()
                    ],
                    dtype=torch.uint8,
                )
            )
            source_segments.append(torch.full((route_positions.numel(),), src_rank, dtype=torch.int32))
            total_routes += int(route_positions.numel())
            expert_route_count += int(route_positions.numel())
            segment_offsets.append(total_routes)
        expert_counts.append(expert_route_count)
        expert_offsets.append(total_routes)

    nonempty_segments = [segment for segment in route_segments if segment.numel()]
    if nonempty_segments:
        route_position = torch.cat(nonempty_segments).contiguous()
        source_rank = torch.cat([segment for segment in source_segments if segment.numel()]).contiguous()
        gateway_lane = torch.cat([segment for segment in gateway_lane_segments if segment.numel()]).contiguous()
    else:
        route_position = torch.empty((0,), dtype=torch.int32)
        source_rank = torch.empty((0,), dtype=torch.int32)
        gateway_lane = torch.empty((0,), dtype=torch.uint8)

    return (
        route_position,
        gateway_lane,
        source_rank,
        torch.tensor(segment_offsets, dtype=torch.int32),
        torch.tensor(expert_offsets, dtype=torch.int32),
        expert_counts,
    )


def _validate_gateway_policy(gateway_policy: str) -> None:
    if gateway_policy != "source_lane":
        raise ValueError("gateway_policy='source_lane' is the only supported policy")


def _gateway_lane_for_token_server(
    token_routes: torch.Tensor,
    *,
    destination_server: int,
    experts_per_server: int,
    num_local_experts: int,
    source_lane: int,
    source_server: int,
    gateway_policy: str,
) -> int:
    _validate_gateway_policy(gateway_policy)
    return source_lane


def _build_recv_chunk_token_end(
    all_topk_idx: torch.Tensor,
    *,
    destination_server: int,
    experts_per_server: int,
    capacity: int,
    num_tokens_per_chunk: int,
) -> torch.Tensor:
    """Build exact compact Dispatch chunk boundaries for one destination server."""

    world_size, num_tokens, _ = map(int, all_topk_idx.shape)
    chunk_capacity = (capacity + num_tokens_per_chunk - 1) // num_tokens_per_chunk
    num_token_chunks = (num_tokens + num_tokens_per_chunk - 1) // num_tokens_per_chunk
    chunk_token_end = torch.full((world_size, chunk_capacity), num_tokens, dtype=torch.int32)
    expert_begin = destination_server * experts_per_server
    membership = (all_topk_idx.ge(expert_begin) & all_topk_idx.lt(expert_begin + experts_per_server)).any(dim=2)

    for source_rank in range(world_size):
        token_ids = torch.nonzero(membership[source_rank], as_tuple=False).flatten()
        compact_chunks = (token_ids.numel() + num_tokens_per_chunk - 1) // num_tokens_per_chunk
        boundary_chunks = min(compact_chunks, num_token_chunks - 1)
        for chunk_id in range(boundary_chunks):
            compact_end = min(
                (chunk_id + 1) * num_tokens_per_chunk,
                token_ids.numel(),
            )
            chunk_token_end[source_rank, chunk_id] = token_ids[compact_end - 1] + 1

    return chunk_token_end


def _validate_inputs(
    all_topk_idx: torch.Tensor,
    rank: int,
    num_experts: int,
    num_max_tokens_per_rank: Optional[int],
    ranks_per_server: int,
) -> None:
    if not isinstance(all_topk_idx, torch.Tensor):
        raise TypeError("all_topk_idx must be a torch.Tensor")
    if all_topk_idx.device.type != "cpu":
        raise ValueError("golden notify requires all_topk_idx on the CPU")
    if all_topk_idx.dim() != 3:
        raise ValueError("all_topk_idx must have shape [R, N, K]")
    if all_topk_idx.dtype not in _INTEGER_DTYPES:
        raise TypeError("all_topk_idx must use an integer dtype")

    world_size, num_tokens, num_topk = map(int, all_topk_idx.shape)
    if world_size < 2 or num_tokens <= 0 or num_topk <= 0:
        raise ValueError("R, N, and K must all be positive and R must be >= 2")
    if not isinstance(rank, int) or rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, R)")
    if not isinstance(ranks_per_server, int) or ranks_per_server <= 0:
        raise ValueError("ranks_per_server must be a positive integer")
    if world_size % ranks_per_server:
        raise ValueError("world_size must be divisible by ranks_per_server")
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    if num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size")
    if num_tokens * num_topk > _INT32_MAX:
        raise ValueError("N*K must fit int32 route_position")
    if world_size * num_tokens * num_topk > _INT32_MAX:
        raise ValueError("R*N*K must fit int32 segment offsets")

    if num_max_tokens_per_rank is not None:
        if not isinstance(num_max_tokens_per_rank, int) or num_max_tokens_per_rank < num_tokens:
            raise ValueError("num_max_tokens_per_rank must be an integer no smaller than N")

    invalid = all_topk_idx.lt(-1) | all_topk_idx.ge(num_experts)
    if bool(invalid.any()):
        first = torch.nonzero(invalid, as_tuple=False)[0].tolist()
        value = int(all_topk_idx[tuple(first)].item())
        raise ValueError(f"all_topk_idx contains an invalid expert ID at {tuple(first)}: {value}")


__all__ = [
    "DupServerDispatchMeta",
    "DupServerSourceNotifyMeta",
    "build_dup_server_golden_notify",
    "build_dup_server_golden_source_notify",
]
