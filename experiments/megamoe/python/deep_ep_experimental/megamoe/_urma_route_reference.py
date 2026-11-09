"""Routing contract shared by the URMA runner and its CPU oracle.

Every top-k entry is a separate routing instance. Expansion sorts by global
expert, then by the original flattened [token, top-k] position. There is no
balanced-routing shortcut and no token deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceRoutes:
    order: object
    reduce_index: object
    expert_counts: object
    expert_offsets: object


@dataclass(frozen=True)
class HostRoutes:
    order: object
    reduce_index: object
    expert_counts: object
    expert_offsets: object
    peer_counts: object
    peer_offsets: object
    num_experts: int
    world: int
    tokens: int
    topk: int


def plan_routes(topk_idx, num_experts, world):
    """Independent NumPy oracle; never used to prepare NPU benchmark routes."""
    import numpy as np

    ids = np.asarray(topk_idx)
    if ids.ndim != 2 or ids.dtype.kind not in "iu" or not ids.size:
        raise ValueError("routes must be a nonempty integer [tokens, top-k] array")
    if type(world) is not int or world <= 0 or num_experts <= 0 or num_experts % world:
        raise ValueError("positive world must divide num_experts")
    if np.any(ids < 0) or np.any(ids >= num_experts):
        raise ValueError("expert id is outside [0, num_experts)")
    flat = ids.reshape(-1).astype(np.int64)
    order = np.argsort(flat, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    counts = np.bincount(flat, minlength=num_experts).astype(np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), counts.cumsum()))
    peer_counts = counts.reshape(world, num_experts // world).sum(axis=1)
    peer_offsets = np.concatenate((np.zeros(1, dtype=np.int64), peer_counts.cumsum()))
    return HostRoutes(
        order, inverse.reshape(ids.shape), counts, offsets, peer_counts, peer_offsets, num_experts, world, *ids.shape
    )


@dataclass(frozen=True)
class Fragment:
    source_rank: int
    destination_rank: int
    cv_id: int
    expert: int
    local_expert: int
    peer_row: int
    cv_row: int
    expert_row: int
    source_expanded_row: int
    rows: int


def cv_fragments(routes, destination_rank, cv_rows):
    """Intersect each peer's fixed-row communication window with expert runs.

    peer_row is the stable row in that source->destination stream and
    source_expanded_row is the target of the return PUT. These intersections
    describe communication, not the expert-major GMM task boundaries.
    """
    if type(cv_rows) is not int or cv_rows <= 0 or not routes:
        raise ValueError("positive cv_rows and at least one rank are required")
    first = routes[0]
    if len(routes) != first.world or not 0 <= destination_rank < first.world:
        raise ValueError("route table/rank does not match world")
    if any((r.world, r.num_experts) != (first.world, first.num_experts) for r in routes):
        raise ValueError("inconsistent expert topology")
    local_experts = first.num_experts // first.world
    fragments = []
    for source, route in enumerate(routes):
        base = int(route.peer_offsets[destination_rank])
        peer_rows = int(route.peer_counts[destination_rank])
        for cv in range((peer_rows + cv_rows - 1) // cv_rows):
            window_begin, window_end = cv * cv_rows, min((cv + 1) * cv_rows, peer_rows)
            for local in range(local_experts):
                expert = destination_rank * local_experts + local
                expert_begin = int(route.expert_offsets[expert]) - base
                expert_end = int(route.expert_offsets[expert + 1]) - base
                begin, end = max(window_begin, expert_begin), min(window_end, expert_end)
                if begin < end:
                    fragments.append(
                        Fragment(
                            source,
                            destination_rank,
                            cv,
                            expert,
                            local,
                            begin,
                            begin - window_begin,
                            begin - expert_begin,
                            base + begin,
                            end - begin,
                        )
                    )
    return fragments


@dataclass(frozen=True)
class RowSegment:
    source_rank: int
    cv_id: int
    receive_row: int
    return_row: int
    tile_row: int
    rows: int


@dataclass(frozen=True)
class ExpertMGroup:
    expert: int
    local_expert: int
    row_begin: int
    expert_row: int
    rows: int
    segments: tuple[RowSegment, ...]


def expert_m_groups(routes, destination_rank, cv_rows):
    """CPU oracle for expert-major M256 gather/scatter descriptors.

    Build each expert's ordered row identities first, then partition those
    identities into M256 groups. This intentionally differs from the native
    interval/prefix construction, so coverage tests exercise an independent
    algorithm. Communication CV size never changes the compute partition.
    """
    if type(cv_rows) is not int or cv_rows <= 0 or not routes:
        raise ValueError("positive cv_rows and at least one rank are required")
    first = routes[0]
    if len(routes) != first.world or not 0 <= destination_rank < first.world:
        raise ValueError("route table/rank does not match world")
    if any((r.world, r.num_experts) != (first.world, first.num_experts) for r in routes):
        raise ValueError("inconsistent expert topology")
    local_experts = first.num_experts // first.world
    receive_bases, receive_rows = [], 0
    for route in routes:
        receive_bases.append(receive_rows)
        receive_rows += int(route.peer_counts[destination_rank])
    groups, compute_row = [], 0
    for local in range(local_experts):
        expert = destination_rank * local_experts + local
        identities = []
        for source, route in enumerate(routes):
            begin, end = map(int, route.expert_offsets[expert : expert + 2])
            peer_base = int(route.peer_offsets[destination_rank])
            for expanded_row in range(begin, end):
                peer_row = expanded_row - peer_base
                identities.append((source, peer_row // cv_rows, receive_bases[source] + peer_row, expanded_row))
        for offset in range(0, len(identities), 256):
            rows = identities[offset : offset + 256]
            segments = []
            start = 0
            while start < len(rows):
                source, cv, receive_row, return_row = rows[start]
                end = start + 1
                while end < len(rows) and rows[end] == (
                    source,
                    cv,
                    receive_row + end - start,
                    return_row + end - start,
                ):
                    end += 1
                segments.append(RowSegment(source, cv, receive_row, return_row, start, end - start))
                start = end
            groups.append(ExpertMGroup(expert, local, compute_row, offset, len(rows), tuple(segments)))
            compute_row += len(rows)
    assert compute_row == receive_rows
    return groups


def remote_peer_for_worker(rank, worker, world=16):
    """One remote PE per communication worker; the self PE uses local copies."""
    if not 0 <= rank < world or not 0 <= worker < world - 1:
        raise ValueError("invalid rank or remote worker")
    return worker + int(worker >= rank)


def simulate_roundtrip(inputs, topk_idx, topk_weights, num_experts, cv_rows, expert_fn):
    """CPU descriptor/return/reduction oracle, not a hardware protocol emulator.

    expert_fn(expert_id, rows) may implement the complete expert transformation.
    Empty destinations and expert runs are intentionally supported.
    """
    import numpy as np

    world = len(inputs)
    if not world or len(topk_idx) != world or len(topk_weights) != world:
        raise ValueError("one input, route and weight tensor per rank is required")
    routes = [plan_routes(ids, num_experts, world) for ids in topk_idx]
    expanded = []
    returned = []
    written = []
    for x, weights, route in zip(inputs, topk_weights, routes):
        x, weights = np.asarray(x), np.asarray(weights)
        if x.ndim != 2 or x.shape[0] != route.tokens or weights.shape != route.reduce_index.shape:
            raise ValueError("input or top-k weight shape does not match routes")
        expanded.append(x[route.order // route.topk])
        returned.append(np.empty((route.order.size, x.shape[1]), dtype=x.dtype))
        written.append(np.zeros(route.order.size, dtype=np.int32))
    for destination in range(world):
        for group in expert_m_groups(routes, destination, cv_rows):
            gathered = np.concatenate(
                [
                    expanded[segment.source_rank][segment.return_row : segment.return_row + segment.rows]
                    for segment in group.segments
                ],
                axis=0,
            )
            value = np.asarray(expert_fn(group.expert, gathered))
            if value.shape != gathered.shape:
                raise ValueError("expert transform must preserve the output hidden shape")
            for segment in group.segments:
                source = segment.source_rank
                section = slice(segment.return_row, segment.return_row + segment.rows)
                returned[source][section] = value[segment.tile_row : segment.tile_row + segment.rows]
                written[source][section] += 1
    if any(not np.all(count == 1) for count in written):
        raise AssertionError("return PUT coverage must be exactly once per routing instance")
    return [
        (values[route.reduce_index].astype(np.float32) * np.asarray(weights)[..., None]).sum(axis=1)
        for values, route, weights in zip(returned, routes, topk_weights)
    ]


class CompletionLedger:
    """Small CPU specification of delayed PUT completion and final reduction.

    It checks only lifecycle invariants. It cannot prove SHMEM memory ordering,
    device progress, or queue correctness; those require actual NPU validation.
    """

    def __init__(self, world=16):
        if world < 2:
            raise ValueError("world must be at least two")
        self.world = world
        self.dispatch_done = set()
        self.submitted = set()
        self.completed = set()
        self.barrier_ranks = set()

    def finish_dispatch(self, rank):
        if not 0 <= rank < self.world:
            raise ValueError("invalid rank")
        self.dispatch_done.add(rank)

    def submit_return(self, rank, peer, cv):
        if rank not in self.dispatch_done:
            raise RuntimeError("Combine cannot precede this rank's Dispatch completion")
        if not 0 <= peer < self.world or cv < 0:
            raise ValueError("invalid peer/CV")
        key = (rank, peer, cv)
        if key in self.submitted:
            raise RuntimeError("duplicate return PUT")
        if rank in self.barrier_ranks:
            raise RuntimeError("return PUT after final barrier arrival")
        self.submitted.add(key)

    def quiet(self, rank):
        self.completed.update(key for key in self.submitted if key[0] == rank)

    def arrive_final_barrier(self, rank):
        if rank not in self.dispatch_done or any(key not in self.completed for key in self.submitted if key[0] == rank):
            raise RuntimeError("all local PUTs must complete before final barrier")
        self.barrier_ranks.add(rank)

    def can_reduce(self):
        return len(self.barrier_ranks) == self.world
