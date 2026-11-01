"""Standalone harness for duplicate-server notify and dispatch development.

The CPU golden path is the reference model.  Full NPU dispatch consumes the
EPHandle produced by the Notify kernel; the scale-out-only mode can still use
golden send metadata to isolate payload transfer performance.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import html
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import torch
import torch.multiprocessing as mp

from deep_ep.backends.clos.dup_server_golden import (
    DupServerDispatchMeta,
    build_dup_server_golden_notify,
    build_dup_server_golden_source_notify,
)


PROFILE_OPERATORS = ("notify-prepare", "notify", "dispatch", "full-dispatch")
PROFILE_SUMMARY_TIMEOUT_SECONDS = 30.0
PROFILE_SUMMARY_STABLE_SECONDS = 0.5
DUP_SERVER_RANKS_PER_SERVER = 8
DUP_SERVER_SERVERS_PER_FRAME = 8
DUP_SERVER_MAX_SERVERS = 64
DUP_DISPATCH_FIELD_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
DUP_DISPATCH_STAGE_COLORS = {
    "producer_transfer": "#0072B2",
    "local_partition_copy": "#CC79A7",
    "receive_init": "#6B7280",
    "receive_wait": "#D55E00",
    "route_issue": "#009E73",
    "receiver_overhead": "#D1D5DB",
    "receiver_untraced": "#A8A29E",
    "final_submit": "#56B4E9",
    "cqe_wait": "#332288",
    "local_mte_drain": "#E69F00",
    # Legacy schema-2 stages retained so older raw logs remain renderable.
    "route_scan": "#F0E442",
    "remote_get_submit": "#009E73",
    "local_mte_copy": "#882255",
    "route_issue_other": "#999933",
    "receiver_bookkeeping": "#44AA99",
    "trace_record": "#AA4499",
}


class _ProfileComplete(Exception):
    pass


def _log_rank_stage(rank: int, message: str) -> None:
    print(f"[dup_server][rank={rank}] {message}", flush=True)


def _route_mode(args: argparse.Namespace) -> str:
    route = getattr(args, "route", None)
    if route is not None:
        return route
    return "gateway_aligned_remote" if getattr(args, "gateway_aligned_remote", False) else "balanced"


def _align_remote_routes_to_source_lane(
    routes: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    """Align one remote route and balance the rest across receiver lanes."""

    experts_per_rank = args.num_experts // args.ep_size
    experts_per_server = experts_per_rank * args.ranks_per_server
    num_servers = args.ep_size // args.ranks_per_server
    if args.ranks_per_server <= 1:
        return

    source_rank = torch.arange(args.ep_size, dtype=torch.int64).view(-1, 1)
    token_id = torch.arange(routes.shape[1], dtype=torch.int64).view(1, -1)
    source_server = torch.div(source_rank, args.ranks_per_server, rounding_mode="floor")
    source_lane = source_rank.remainder(args.ranks_per_server)
    valid = routes.ge(0)
    route_server = torch.where(
        valid,
        torch.div(routes, experts_per_server, rounding_mode="floor"),
        -1,
    )
    route_local_expert = torch.where(
        valid,
        routes.remainder(experts_per_rank),
        0,
    )

    for destination_server in range(num_servers):
        in_server = route_server.eq(destination_server)
        touches_server = in_server.any(dim=2)
        remote_routes = source_server.ne(destination_server) & touches_server
        if not bool(remote_routes.any()):
            continue

        route_ordinal = in_server.to(torch.int64).cumsum(dim=2) - 1
        # Keep the anchor on source_lane, then walk a hash-selected permutation
        # of every nonzero lane offset to balance (destination rank, gateway lane).
        balance_base = (source_rank * 131 + token_id * 17 + destination_server * 29 + args.seed).remainder(
            args.ranks_per_server - 1
        )
        nonzero_lane_offset = 1 + (balance_base.unsqueeze(2) + route_ordinal - 1).remainder(args.ranks_per_server - 1)
        balanced_lane = torch.where(
            route_ordinal.eq(0),
            source_lane.unsqueeze(2),
            (source_lane.unsqueeze(2) + nonzero_lane_offset).remainder(args.ranks_per_server),
        )
        balanced_route = (
            destination_server * args.ranks_per_server + balanced_lane
        ) * experts_per_rank + route_local_expert
        balance_mask = remote_routes.unsqueeze(2) & in_server
        routes[balance_mask] = balanced_route[balance_mask]


def make_global_topk_idx(args: argparse.Namespace) -> torch.Tensor:
    """Create a reproducible global routing snapshot in the selected mode."""

    route_mode = _route_mode(args)
    if route_mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        routes = torch.randint(
            args.num_experts,
            (args.ep_size, args.num_tokens, args.num_topk),
            dtype=torch.int64,
            generator=generator,
        )
        for topk_slot in range(1, args.num_topk):
            duplicate = routes[..., topk_slot : topk_slot + 1].eq(routes[..., :topk_slot]).any(dim=-1)
            while bool(duplicate.any()):
                routes[..., topk_slot][duplicate] = torch.randint(
                    args.num_experts,
                    (int(duplicate.sum().item()),),
                    dtype=torch.int64,
                    generator=generator,
                )
                duplicate = routes[..., topk_slot : topk_slot + 1].eq(routes[..., :topk_slot]).any(dim=-1)
    elif route_mode in ("balanced", "gateway_aligned_remote"):
        src_rank = torch.arange(args.ep_size, dtype=torch.int64).view(-1, 1, 1)
        token_id = torch.arange(args.num_tokens, dtype=torch.int64).view(1, -1, 1)
        topk_slot = torch.arange(args.num_topk, dtype=torch.int64).view(1, 1, -1)
        routes = (src_rank * 97 + token_id * 17 + topk_slot * 19 + args.seed).remainder(args.num_experts)
    else:
        raise ValueError(f"unsupported route mode: {route_mode}")

    if args.drop_route_every:
        linear = torch.arange(routes.numel()).reshape(routes.shape)
        routes[linear.remainder(args.drop_route_every).eq(0)] = -1

    if route_mode == "gateway_aligned_remote":
        _align_remote_routes_to_source_lane(routes, args)

    return routes.to(torch.int32).contiguous()


def test_gateway_aligned_remote_routes_preserve_server_membership() -> None:
    base_args = {
        "ep_size": 16,
        "num_tokens": 31,
        "num_topk": 8,
        "num_experts": 64,
        "ranks_per_server": 4,
        "seed": 1234,
        "drop_route_every": 7,
    }
    baseline = make_global_topk_idx(argparse.Namespace(**base_args, route="balanced"))
    legacy = make_global_topk_idx(argparse.Namespace(**base_args))
    aligned = make_global_topk_idx(argparse.Namespace(**base_args, route="gateway_aligned_remote"))
    assert torch.equal(legacy, baseline)

    experts_per_rank = base_args["num_experts"] // base_args["ep_size"]
    experts_per_server = experts_per_rank * base_args["ranks_per_server"]
    baseline_server = torch.where(
        baseline.ge(0),
        torch.div(baseline, experts_per_server, rounding_mode="floor"),
        -1,
    )
    aligned_server = torch.where(
        aligned.ge(0),
        torch.div(aligned, experts_per_server, rounding_mode="floor"),
        -1,
    )
    assert torch.equal(aligned_server, baseline_server)
    assert torch.equal(
        aligned.remainder(experts_per_rank),
        baseline.remainder(experts_per_rank),
    )
    changed = aligned.ne(baseline)
    changed_source = torch.nonzero(changed, as_tuple=True)[0]
    assert changed_source.numel() != 0
    changed_server = baseline_server[changed]
    changed_source_server = torch.div(changed_source, base_args["ranks_per_server"], rounding_mode="floor")
    assert bool(changed_server.ne(changed_source_server).all())

    source_rank = torch.arange(base_args["ep_size"], dtype=torch.int64).view(-1, 1)
    source_server = torch.div(source_rank, base_args["ranks_per_server"], rounding_mode="floor")
    source_lane = source_rank.remainder(base_args["ranks_per_server"])
    num_servers = base_args["ep_size"] // base_args["ranks_per_server"]
    aligned_rank = torch.div(aligned, experts_per_rank, rounding_mode="floor")
    for destination_server in range(num_servers):
        touches_server = aligned_server.eq(destination_server).any(dim=2)
        touches_aligned_rank = aligned_rank.eq(
            destination_server * base_args["ranks_per_server"] + source_lane.view(-1, 1, 1)
        ).any(dim=2)
        required = source_server.ne(destination_server) & touches_server
        assert bool(touches_aligned_rank[required].all())


def test_gateway_aligned_remote_routes_balance_receiver_pairs() -> None:
    args = argparse.Namespace(
        ep_size=16,
        num_tokens=4096,
        num_topk=8,
        num_experts=64,
        ranks_per_server=4,
        seed=1234,
        drop_route_every=0,
        route="gateway_aligned_remote",
    )
    routes = make_global_topk_idx(args).to(torch.int64)
    experts_per_rank = args.num_experts // args.ep_size
    source_lane = torch.arange(args.ep_size, dtype=torch.int64).view(-1, 1, 1).remainder(args.ranks_per_server)
    destination_rank = torch.div(routes, experts_per_rank, rounding_mode="floor")
    destination_lane = destination_rank.remainder(args.ranks_per_server)
    rank_counts = torch.bincount(
        destination_rank[routes.ge(0)],
        minlength=args.ep_size,
    )
    assert int(rank_counts.max() - rank_counts.min()) <= 32

    remote_get = routes.ge(0) & destination_lane.ne(source_lane)
    receiver_pair = destination_rank * args.ranks_per_server + source_lane
    pair_counts = torch.bincount(
        receiver_pair[remote_get],
        minlength=args.ep_size * args.ranks_per_server,
    ).reshape(args.ep_size, args.ranks_per_server)
    active_pairs = pair_counts[
        torch.arange(args.ep_size).remainder(args.ranks_per_server).view(-1, 1)
        != torch.arange(args.ranks_per_server).view(1, -1)
    ]

    assert int(active_pairs.max() - active_pairs.min()) <= 32


def test_random_routes_are_seeded_and_valid() -> None:
    base_args = {
        "ep_size": 16,
        "num_tokens": 31,
        "num_topk": 8,
        "num_experts": 64,
        "ranks_per_server": 4,
        "drop_route_every": 0,
        "route": "random",
    }
    first = make_global_topk_idx(argparse.Namespace(**base_args, seed=1234))
    repeated = make_global_topk_idx(argparse.Namespace(**base_args, seed=1234))
    changed_seed = make_global_topk_idx(argparse.Namespace(**base_args, seed=1235))

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed_seed)
    assert first.dtype == torch.int32
    assert first.is_contiguous()
    assert bool(first.ge(0).all())
    assert bool(first.lt(base_args["num_experts"]).all())
    sorted_routes = first.sort(dim=-1).values
    assert bool(sorted_routes[..., 1:].ne(sorted_routes[..., :-1]).all())


def build_golden_metadata(
    all_topk_idx: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Dict[int, DupServerDispatchMeta], List[float]]:
    """Build metadata for one selected rank or for the complete EP group."""

    ranks: Iterable[int]
    if args.rank is None:
        ranks = range(args.ep_size)
    else:
        ranks = (args.rank,)

    metadata = {}
    rank_times = []
    for rank in ranks:
        begin = time.perf_counter()
        meta = build_dup_server_golden_notify(
            all_topk_idx,
            rank=rank,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=args.num_max_tokens_per_rank,
            ranks_per_server=args.ranks_per_server,
            num_tokens_per_chunk=args.num_tokens_per_chunk,
            gateway_policy=args.gateway_policy,
        )
        elapsed = time.perf_counter() - begin
        metadata[rank] = meta
        rank_times.append(elapsed)
        if args.verbose:
            print(f"  rank {rank:>3}: {elapsed:.3f}s, recv_routes={meta.num_recv_tokens}")

    return metadata, rank_times


def validate_golden_metadata(
    all_topk_idx: torch.Tensor,
    metadata: Dict[int, DupServerDispatchMeta],
) -> None:
    """Check source deduplication and every destination route segment."""

    for rank, meta in metadata.items():
        _validate_source_metadata(all_topk_idx[rank], meta)
        _validate_destination_metadata(all_topk_idx, meta)


def validate_source_notify_precision(
    actual: Any,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor | None,
    source_rank: int,
    args: argparse.Namespace,
) -> None:
    expected = build_dup_server_golden_source_notify(
        topk_idx,
        args.num_experts,
        topk_weights=topk_weights,
        world_size=args.ep_size,
        source_rank=source_rank,
        num_max_tokens_per_rank=args.num_max_tokens_per_rank,
        ranks_per_server=args.ranks_per_server,
        gateway_policy=args.gateway_policy,
    )
    for name in (
        "send_token_count",
        "send_token_ids",
        "send_gateway_lane",
        "send_route_count",
        "send_expert_prefix",
        "packed_route_position",
        "packed_gateway_lane",
        "source_server_mask",
    ):
        actual_tensor = getattr(actual, name).cpu()
        expected_tensor = getattr(expected, name)
        if not torch.equal(actual_tensor, expected_tensor):
            raise AssertionError(f"duplicate-server Notify {name} mismatch")
    if topk_weights is not None:
        if not torch.equal(
            actual.packed_topk_weights.cpu(),
            expected.packed_topk_weights,
        ):
            raise AssertionError("duplicate-server Notify packed_topk_weights mismatch")


def validate_full_notify_precision(
    actual: Any,
    all_topk_idx: torch.Tensor,
    rank: int,
    args: argparse.Namespace,
) -> None:
    """Validate gateway metadata and destination expert segments."""

    if int(actual.notify_status.cpu().item()) != 0:
        raise AssertionError("duplicate-server Notify receive capacity overflow")
    expected = build_dup_server_golden_notify(
        all_topk_idx,
        rank=rank,
        num_experts=args.num_experts,
        num_max_tokens_per_rank=args.num_max_tokens_per_rank,
        ranks_per_server=args.ranks_per_server,
        num_tokens_per_chunk=args.num_tokens_per_chunk,
        gateway_policy=args.gateway_policy,
    )
    actual_segment_offset = actual.recv_segment_offset.cpu()
    if not torch.equal(actual_segment_offset, expected.recv_segment_offset):
        raise AssertionError("duplicate-server Notify recv_segment_offset mismatch")
    num_recv_routes = int(actual_segment_offset[-1].item())
    actual_recv_routes = actual.recv_route_position[:num_recv_routes].cpu()
    if not torch.equal(actual_recv_routes, expected.recv_route_position):
        raise AssertionError("duplicate-server Notify recv_route_position mismatch")
    actual_recv_gateway_lane = actual.recv_gateway_lane[:num_recv_routes].cpu()
    if not torch.equal(actual_recv_gateway_lane, expected.recv_gateway_lane):
        raise AssertionError("duplicate-server Notify recv_gateway_lane mismatch")

    if not torch.equal(actual.recv_chunk_token_end.cpu(), expected.recv_chunk_token_end):
        raise AssertionError("duplicate-server Notify recv_chunk_token_end mismatch")

    num_servers = args.ep_size // args.ranks_per_server
    num_local_experts = args.num_experts // args.ep_size
    experts_per_server = num_local_experts * args.ranks_per_server
    destination_server = rank // args.ranks_per_server
    gateway_lane = rank % args.ranks_per_server
    expert_begin = destination_server * experts_per_server
    gateway_route_count = actual.gateway_route_count.cpu()
    gateway_route_position = actual.gateway_route_position.cpu()
    gateway_slot_mask = actual.gateway_slot_mask.cpu()
    recv_weight_segment_begin = actual.recv_weight_segment_begin.cpu()
    expert_count_rows = []
    for source_rank in range(args.ep_size):
        flat_routes = all_topk_idx[source_rank].reshape(-1).to(torch.int64)
        valid_routes = flat_routes[(flat_routes >= 0) & (flat_routes < args.num_experts)]
        expert_count_rows.append(torch.bincount(valid_routes, minlength=args.num_experts))
    expert_prefix = torch.cat(
        (
            torch.zeros((args.ep_size, 1), dtype=torch.int64),
            torch.stack(expert_count_rows).cumsum(dim=1),
        ),
        dim=1,
    )

    for source_server in range(num_servers):
        source_rank = source_server * args.ranks_per_server + gateway_lane
        source_expected = build_dup_server_golden_source_notify(
            all_topk_idx[source_rank],
            args.num_experts,
            world_size=args.ep_size,
            source_rank=source_rank,
            num_max_tokens_per_rank=args.num_max_tokens_per_rank,
            ranks_per_server=args.ranks_per_server,
            gateway_policy=args.gateway_policy,
        )
        route_base = int(source_expected.send_route_count[:destination_server].sum().item())
        expected_count = int(source_expected.send_route_count[destination_server].item())
        expected_gateway_routes = source_expected.packed_route_position[route_base : route_base + expected_count]
        route_count = int(gateway_route_count[source_server].item())
        if route_count != expected_gateway_routes.numel():
            raise AssertionError("duplicate-server Notify gateway_route_count mismatch")
        if not torch.equal(
            gateway_route_position[source_server, :route_count],
            expected_gateway_routes,
        ):
            raise AssertionError("duplicate-server Notify gateway_route_position mismatch")

        expected_mask = torch.zeros(args.num_tokens, dtype=torch.int32)
        source_routes = all_topk_idx[source_rank]
        for topk_slot in range(args.num_topk):
            slot_expert = source_routes[:, topk_slot]
            valid = slot_expert.ge(expert_begin) & slot_expert.lt(expert_begin + experts_per_server)
            expected_mask[valid] += 1 << topk_slot
        if not torch.equal(gateway_slot_mask[source_server, : args.num_tokens], expected_mask):
            raise AssertionError("duplicate-server Notify gateway_slot_mask mismatch")

    for local_expert in range(num_local_experts):
        global_expert = rank * num_local_experts + local_expert
        for source_rank in range(args.ep_size):
            expected_begin = int(
                (expert_prefix[source_rank, global_expert] - expert_prefix[source_rank, expert_begin]).item()
            )
            segment = local_expert * args.ep_size + source_rank
            if int(recv_weight_segment_begin[segment].item()) != expected_begin:
                raise AssertionError("duplicate-server Notify recv_weight_segment_begin mismatch")


def _validate_source_metadata(
    local_topk_idx: torch.Tensor,
    meta: DupServerDispatchMeta,
) -> None:
    experts_per_server = meta.num_local_experts * meta.ranks_per_server
    for dst_server in range(meta.num_servers):
        expert_begin = dst_server * experts_per_server
        expert_end = expert_begin + experts_per_server
        membership = (local_topk_idx.ge(expert_begin) & local_topk_idx.lt(expert_end)).any(dim=1)
        expected_ids = torch.nonzero(membership, as_tuple=False).flatten().to(torch.int32)
        count = int(meta.send_token_count[dst_server].item())
        actual_ids = meta.send_token_ids[dst_server, :count]
        if not torch.equal(actual_ids, expected_ids):
            raise AssertionError(f"rank {meta.rank} server {dst_server}: send_token_ids mismatch")
        if not bool(meta.send_token_ids[dst_server, count:].eq(-1).all()):
            raise AssertionError(f"rank {meta.rank} server {dst_server}: unused send_token_ids entries must be -1")
        expected_lanes = []
        for token_id in expected_ids.tolist():
            if meta.gateway_policy == "source_lane" or dst_server == meta.rank // meta.ranks_per_server:
                lane = meta.rank % meta.ranks_per_server
            else:
                lane = next(
                    (int(expert) - expert_begin) // meta.num_local_experts
                    for expert in local_topk_idx[token_id].tolist()
                    if expert_begin <= int(expert) < expert_end
                )
            expected_lanes.append(lane)
        if meta.send_gateway_lane[dst_server, :count].tolist() != expected_lanes:
            raise AssertionError(f"rank {meta.rank} server {dst_server}: send_gateway_lane mismatch")
        if not bool(meta.send_gateway_lane[dst_server, count:].eq(0xFF).all()):
            raise AssertionError(f"rank {meta.rank} server {dst_server}: unused gateway lanes must be 255")


def _validate_destination_metadata(
    all_topk_idx: torch.Tensor,
    meta: DupServerDispatchMeta,
) -> None:
    expected_offset_size = meta.num_local_experts * meta.world_size + 1
    if meta.recv_segment_offset.numel() != expected_offset_size:
        raise AssertionError(f"rank {meta.rank}: recv_segment_offset has invalid shape")
    if not bool((meta.recv_segment_offset[1:] >= meta.recv_segment_offset[:-1]).all()):
        raise AssertionError(f"rank {meta.rank}: recv_segment_offset is not monotonic")
    if int(meta.recv_segment_offset[-1].item()) != meta.num_recv_tokens:
        raise AssertionError(f"rank {meta.rank}: final segment offset does not match num_recv_tokens")
    if tuple(meta.recv_src_rank.shape) != (meta.num_recv_tokens,):
        raise AssertionError(f"rank {meta.rank}: recv_src_rank has invalid shape")
    if tuple(meta.recv_expert_offset.shape) != (meta.num_local_experts + 1,):
        raise AssertionError(f"rank {meta.rank}: recv_expert_offset has invalid shape")

    first_expert = meta.rank * meta.num_local_experts
    for local_expert in range(meta.num_local_experts):
        global_expert = first_expert + local_expert
        segment_begin = local_expert * meta.world_size
        segment_end = (local_expert + 1) * meta.world_size
        if int(meta.recv_expert_offset[local_expert].item()) != int(
            meta.recv_segment_offset[segment_begin].item()
        ) or int(meta.recv_expert_offset[local_expert + 1].item()) != int(meta.recv_segment_offset[segment_end].item()):
            raise AssertionError(f"rank {meta.rank}, expert {global_expert}: expert offset mismatch")
        expert_count = 0
        for src_rank in range(meta.world_size):
            segment = local_expert * meta.world_size + src_rank
            begin = int(meta.recv_segment_offset[segment].item())
            end = int(meta.recv_segment_offset[segment + 1].item())
            actual = meta.recv_route_position[begin:end]
            actual_gateway_lane = meta.recv_gateway_lane[begin:end]
            actual_source = meta.recv_src_rank[begin:end]
            expected = (
                torch.nonzero(
                    all_topk_idx[src_rank].reshape(-1).eq(global_expert),
                    as_tuple=False,
                )
                .flatten()
                .to(torch.int32)
            )
            if not torch.equal(actual, expected):
                raise AssertionError(
                    f"rank {meta.rank}, expert {global_expert}, source {src_rank}: recv route mismatch"
                )
            if not bool(actual_source.eq(src_rank).all()):
                raise AssertionError(
                    f"rank {meta.rank}, expert {global_expert}, source {src_rank}: recv source rank mismatch"
                )
            expected_gateway_lanes = []
            destination_server = meta.rank // meta.ranks_per_server
            expert_begin = destination_server * meta.num_local_experts * meta.ranks_per_server
            expert_end = expert_begin + meta.num_local_experts * meta.ranks_per_server
            for route_position in expected.tolist():
                if meta.gateway_policy == "source_lane" or src_rank // meta.ranks_per_server == destination_server:
                    lane = src_rank % meta.ranks_per_server
                else:
                    token_id = route_position // meta.num_topk
                    lane = next(
                        (int(expert) - expert_begin) // meta.num_local_experts
                        for expert in all_topk_idx[src_rank, token_id].tolist()
                        if expert_begin <= int(expert) < expert_end
                    )
                expected_gateway_lanes.append(lane)
            if actual_gateway_lane.tolist() != expected_gateway_lanes:
                raise AssertionError(
                    f"rank {meta.rank}, expert {global_expert}, source {src_rank}: recv gateway lane mismatch"
                )
            expert_count += int(expected.numel())

        if expert_count != meta.num_recv_tokens_per_expert[local_expert]:
            raise AssertionError(f"rank {meta.rank}, expert {global_expert}: expert route count mismatch")


def make_dispatch_x(rank: int, args: argparse.Namespace) -> torch.Tensor:
    """Create BF16 rows whose source rank and token ID are easy to recover."""

    row_values = (
        (torch.arange(args.num_tokens, device="npu", dtype=torch.int32) * 7 + rank * 31)
        .remainder(251)
        .to(torch.bfloat16)
    )
    return row_values.view(-1, 1).expand(-1, args.hidden).contiguous()


def validate_dispatch_precision(
    all_topk_idx: torch.Tensor,
    gateway_x: torch.Tensor,
    rank: int,
    args: argparse.Namespace,
) -> None:
    """Validate every remotely populated payload row on this rank."""

    destination_server = rank // args.ranks_per_server
    local_rank = rank % args.ranks_per_server
    experts_per_server = args.num_experts * args.ranks_per_server // args.ep_size
    expert_begin = destination_server * experts_per_server
    expert_end = expert_begin + experts_per_server

    for source_rank in range(args.ep_size):
        source_server = source_rank // args.ranks_per_server
        if source_server == destination_server:
            continue
        if args.gateway_policy == "source_lane" and source_rank % args.ranks_per_server != local_rank:
            continue

        membership = (all_topk_idx[source_rank].ge(expert_begin) & all_topk_idx[source_rank].lt(expert_end)).any(dim=1)

        token_ids = torch.nonzero(membership, as_tuple=False).flatten().to(torch.int64)
        if token_ids.numel() == 0:
            continue

        owner = source_server
        token_ids_npu = token_ids.to(device="npu")
        actual = gateway_x[owner].index_select(0, token_ids_npu)
        expected_rows = (
            (token_ids_npu.to(torch.int32) * 7 + source_rank * 31).remainder(251).to(torch.bfloat16).view(-1, 1)
        )
        mismatch_mask = actual != expected_rows
        mismatch_count = int(torch.count_nonzero(mismatch_mask).item())
        if mismatch_count:
            mismatch = torch.nonzero(mismatch_mask, as_tuple=False)[0].cpu().tolist()
            token_id = int(token_ids[mismatch[0]].item())
            hidden_id = int(mismatch[1])
            raise AssertionError(
                f"rank {rank}: source_rank={source_rank}, token={token_id}, "
                f"hidden={hidden_id}, actual={actual[mismatch[0], hidden_id].item()}, "
                f"expected={expected_rows[mismatch[0], 0].item()}, mismatches={mismatch_count}"
            )


def validate_full_dispatch_precision(
    meta: DupServerDispatchMeta,
    recv_x: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    """Validate expert-major rows produced by the complete hierarchical dispatch."""

    if tuple(recv_x.shape) != (meta.num_recv_tokens, args.hidden):
        raise AssertionError(
            f"rank {meta.rank}: recv_x has shape {tuple(recv_x.shape)}, expected {(meta.num_recv_tokens, args.hidden)}"
        )
    if meta.num_recv_tokens == 0:
        return

    source_rank = meta.recv_src_rank.to(device="npu", dtype=torch.int32)
    route_position = meta.recv_route_position.to(device="npu", dtype=torch.int32)
    token_id = torch.div(route_position, args.num_topk, rounding_mode="floor")
    expected = (token_id * 7 + source_rank * 31).remainder(251).to(torch.bfloat16).view(-1, 1)
    actual = recv_x.clone()
    mismatch_mask = actual != expected
    mismatch_count = int(torch.count_nonzero(mismatch_mask).item())
    if mismatch_count:
        mismatch = torch.nonzero(mismatch_mask, as_tuple=False)[0].cpu().tolist()
        output_index, hidden_id = map(int, mismatch)
        raise AssertionError(
            f"rank {meta.rank}: output={output_index}, hidden={hidden_id}, "
            f"source_rank={int(meta.recv_src_rank[output_index].item())}, "
            f"route_position={int(meta.recv_route_position[output_index].item())}, "
            f"actual={actual[output_index, hidden_id].item()}, "
            f"expected={expected[output_index, 0].item()}, mismatches={mismatch_count}"
        )


def _consume_profile_result(result: Any) -> None:
    event = result[-1] if isinstance(result, (tuple, list)) and result else result
    if event is not None and getattr(event, "event", None) is not None:
        event.current_stream_wait(release_handle=True)


def _profile_csv_snapshot(profile_dir: str) -> Dict[str, tuple[int, int]]:
    snapshot: Dict[str, tuple[int, int]] = {}
    root = Path(profile_dir)
    if not root.exists():
        return snapshot
    for path in root.rglob("op_summary_*.csv"):
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[str(path.resolve())] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _wait_for_profile_csvs(
    profile_dir: str,
    before: Dict[str, tuple[int, int]],
) -> List[Path]:
    deadline = time.monotonic() + PROFILE_SUMMARY_TIMEOUT_SECONDS
    stable_since: float | None = None
    previous: Dict[str, tuple[int, int]] = {}
    current: Dict[str, tuple[int, int]] = {}
    while True:
        snapshot = _profile_csv_snapshot(profile_dir)
        current = {path: signature for path, signature in snapshot.items() if before.get(path) != signature}
        now = time.monotonic()
        if current and current == previous:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= PROFILE_SUMMARY_STABLE_SECONDS:
                break
        else:
            previous = current
            stable_since = now if current else None
        if now >= deadline:
            break
        time.sleep(0.1)

    return [Path(path) for path, _ in sorted(current.items(), key=lambda item: (item[1][0], item[0]))]


def _read_profile_durations(path: Path) -> Dict[str, List[float]]:
    durations: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or ())
        required = {"Op Name", "Task Duration(us)"}
        if not required.issubset(fieldnames):
            raise ValueError(f"missing columns {sorted(required - fieldnames)}")
        for row in reader:
            op_name = (row.get("Op Name") or "").strip()
            duration_text = (row.get("Task Duration(us)") or "").strip()
            if not op_name or not duration_text:
                continue
            try:
                duration_us = float(duration_text)
            except ValueError:
                continue
            if math.isfinite(duration_us) and duration_us >= 0.0:
                durations.setdefault(op_name, []).append(duration_us)
    return durations


def _profile_percentile(ordered: List[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _filter_profile_outliers(values: List[float]) -> List[float]:
    if len(values) < 4:
        return list(values)
    ordered = sorted(values)
    q1 = _profile_percentile(ordered, 0.25)
    q3 = _profile_percentile(ordered, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = [value for value in values if lower <= value <= upper]
    return filtered or list(values)


def _format_profile_stats(values: List[float]) -> str:
    filtered = _filter_profile_outliers(values)
    return (
        f"samples={len(values)}, min={min(values):.3f} us, "
        f"max={max(values):.3f} us, avg={statistics.fmean(filtered):.3f} us "
        f"(IQR kept={len(filtered)}/{len(values)}, "
        f"raw_avg={statistics.fmean(values):.3f} us)"
    )


def _print_profile_summary(
    profile_dir: str,
    before: Dict[str, tuple[int, int]],
) -> None:
    paths = _wait_for_profile_csvs(profile_dir, before)
    if not paths:
        print(
            f"[profile-summary] no new op_summary_*.csv found under "
            f"{profile_dir} within {PROFILE_SUMMARY_TIMEOUT_SECONDS:.0f}s",
            flush=True,
        )
        return

    root = Path(profile_dir).resolve()
    combined: Dict[str, List[float]] = {}
    print(
        f"\n[profile-summary] rank0_visible_files={len(paths)}, directory={root}",
        flush=True,
    )
    for path in paths:
        try:
            durations = _read_profile_durations(path)
        except (OSError, ValueError) as exc:
            print(f"  file={path}: skipped ({exc})", flush=True)
            continue
        try:
            display_path = path.resolve().relative_to(root)
        except ValueError:
            display_path = path.resolve()
        print(f"  file={display_path}", flush=True)
        if not durations:
            print("    no valid operator durations", flush=True)
            continue
        for op_name in sorted(durations):
            values = durations[op_name]
            combined.setdefault(op_name, []).extend(values)
            print(
                f"    op={op_name}: {_format_profile_stats(values)}",
                flush=True,
            )

    if combined:
        print("  combined:", flush=True)
        for op_name in sorted(combined):
            print(
                f"    op={op_name}: {_format_profile_stats(combined[op_name])}",
                flush=True,
            )


def _dup_dispatch_trace_fields(line: str) -> Dict[str, str]:
    return dict(DUP_DISPATCH_FIELD_PATTERN.findall(line))


def _trace_int(fields: Dict[str, str], name: str, default: int = 0) -> int:
    value = fields.get(name)
    return default if value is None else int(value, 0)


def _add_timeline_segment(
    core_info: Dict[str, Any],
    stage: str,
    start_cycle: int,
    end_cycle: int,
    detail: str,
) -> None:
    if end_cycle <= start_cycle:
        return
    core_info["segments"].append(
        {
            "stage": stage,
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "detail": detail,
        }
    )


def _parse_dup_dispatch_timeline(trace_text: str) -> tuple[int, Dict[int, Dict[str, Any]]]:
    cycles_per_us = 1000
    cores: Dict[int, Dict[str, Any]] = {}
    receiver_events: Dict[tuple[int, int], Dict[str, int]] = {}

    def core_info(core: int) -> Dict[str, Any]:
        return cores.setdefault(
            core,
            {
                "role": "unknown",
                "gateway_lane": None,
                "use_remote_get": 0,
                "receive_start_cycle": None,
                "receive_end_cycle": None,
                "drain_begin_cycle": None,
                "segments": [],
            },
        )

    for line in trace_text.splitlines():
        if "[ascend_deepep][dup_dispatch_" not in line:
            continue
        fields = _dup_dispatch_trace_fields(line)
        if "cycles_per_us" in fields:
            cycles_per_us = _trace_int(fields, "cycles_per_us", cycles_per_us)
        if "core" not in fields:
            continue
        core = _trace_int(fields, "core")
        info = core_info(core)

        if "[dup_dispatch_timer]" in line:
            role = fields.get("role")
            if role:
                info["role"] = role
            if "gateway_lane" in fields:
                info["gateway_lane"] = _trace_int(fields, "gateway_lane")
            if "use_remote_get" in fields:
                info["use_remote_get"] = _trace_int(fields, "use_remote_get")
            stage = fields.get("stage")
            if stage in ("cross_server_transfer", "local_partition_copy"):
                display_stage = "producer_transfer" if role == "producer" else stage
                _add_timeline_segment(
                    info,
                    display_stage,
                    _trace_int(fields, "start_cycle"),
                    _trace_int(fields, "end_cycle"),
                    stage,
                )
            elif stage == "receiver_window":
                info["receive_start_cycle"] = _trace_int(fields, "receive_start_cycle")
                info["receive_end_cycle"] = _trace_int(fields, "receive_end_cycle")
                _add_timeline_segment(
                    info,
                    "receive_init",
                    _trace_int(fields, "entry_cycle"),
                    _trace_int(fields, "receive_start_cycle"),
                    f"gateway_lane={info['gateway_lane']}",
                )
            elif stage == "final_drain_timeline":
                drain_begin = _trace_int(fields, "drain_begin_cycle")
                info["drain_begin_cycle"] = drain_begin
                submit_done = _trace_int(fields, "final_submit_done_cycle")
                receive_end = _trace_int(fields, "receive_end_cycle")
                _add_timeline_segment(info, "final_submit", drain_begin, submit_done, "final ordered GET")
                wait_stage = "cqe_wait" if info["use_remote_get"] else "local_mte_drain"
                _add_timeline_segment(info, wait_stage, submit_done, receive_end, "final receiver drain")
            continue

        if "[dup_dispatch_trace]" not in line or "sequence" not in fields:
            continue
        sequence = _trace_int(fields, "sequence")
        event = receiver_events.setdefault((core, sequence), {})
        for name, value in fields.items():
            if name in {"stage", "role"}:
                continue
            try:
                event[name] = int(value, 0)
            except ValueError:
                continue
        if "gateway_lane" in fields:
            info["gateway_lane"] = _trace_int(fields, "gateway_lane")
            info["role"] = "receiver"

    receiver_cursor: Dict[int, int] = {}
    receiver_event_count: Dict[int, int] = {}
    for (core, sequence), event in sorted(receiver_events.items()):
        required = {"wait_begin_cycle", "ready_cycle", "issue_done_cycle"}
        if not required.issubset(event):
            continue
        info = core_info(core)
        chunk_begin = event.get("chunk_begin", 0)
        chunk_end = event.get("chunk_end", chunk_begin + 1)
        if "source_rank" in event:
            source_detail = f"source_rank={event['source_rank']}"
        else:
            source_detail = f"source_server={event.get('source_server', -1)}"
        quiet_end = info.get("receive_end_cycle")
        quiet_detail = ""
        if quiet_end is not None and quiet_end >= event["issue_done_cycle"]:
            quiet_detail = (
                f", submit_to_quiet_upper_bound={(quiet_end - event['issue_done_cycle']) / cycles_per_us:.3f} us"
            )
        detail = f"seq={sequence}, {source_detail}, chunks=[{chunk_begin},{chunk_end}){quiet_detail}"
        wait_begin = event["wait_begin_cycle"]
        ready = event["ready_cycle"]
        issue_done = event["issue_done_cycle"]
        previous_end = receiver_cursor.get(core, info.get("receive_start_cycle"))
        if previous_end is not None:
            _add_timeline_segment(
                info, "receiver_overhead", previous_end, wait_begin, "bookkeeping and trace recording"
            )
        _add_timeline_segment(info, "receive_wait", wait_begin, ready, detail)

        if any(name in event for name in ("route_scan_cycles", "remote_get_submit_cycles", "local_mte_copy_cycles")):
            issue_cursor = ready
            issue_stages = (
                ("route_scan", event.get("route_scan_cycles", 0)),
                ("remote_get_submit", event.get("remote_get_submit_cycles", 0)),
                ("local_mte_copy", event.get("local_mte_copy_cycles", 0)),
                ("route_issue_other", event.get("route_issue_other_cycles", 0)),
            )
            for stage, duration in issue_stages:
                stage_end = min(issue_cursor + duration, issue_done)
                _add_timeline_segment(info, stage, issue_cursor, stage_end, detail)
                issue_cursor = stage_end
            _add_timeline_segment(info, "route_issue_other", issue_cursor, issue_done, detail)
        else:
            _add_timeline_segment(info, "route_issue", ready, issue_done, detail)

        bookkeeping_done = event.get("bookkeeping_done_cycle", issue_done)
        trace_done = event.get("trace_done_cycle", bookkeeping_done)
        _add_timeline_segment(info, "receiver_bookkeeping", issue_done, bookkeeping_done, detail)
        _add_timeline_segment(info, "trace_record", bookkeeping_done, trace_done, detail)
        receiver_cursor[core] = trace_done
        receiver_event_count[core] = receiver_event_count.get(core, 0) + 1

    for core, info in cores.items():
        receive_start = info.get("receive_start_cycle")
        drain_begin = info.get("drain_begin_cycle")
        if receive_start is None or drain_begin is None:
            continue
        if receiver_event_count.get(core, 0) == 0:
            _add_timeline_segment(
                info, "receiver_untraced", receive_start, drain_begin, "no complete per-batch trace records"
            )
            continue
        _add_timeline_segment(
            info,
            "receiver_overhead",
            receiver_cursor[core],
            drain_begin,
            "final bookkeeping and trace recording",
        )

    return cycles_per_us, cores


def _write_dup_dispatch_timeline(trace_text: str, output_path: Path) -> tuple[Path, Path]:
    cycles_per_us, cores = _parse_dup_dispatch_timeline(trace_text)
    cores = {core: info for core, info in cores.items() if info["segments"]}
    if not cores:
        raise ValueError("no dup_server dispatch timer records were found")

    all_segments = [segment for info in cores.values() for segment in info["segments"]]
    first_cycle = min(segment["start_cycle"] for segment in all_segments)
    last_cycle = max(segment["end_cycle"] for segment in all_segments)
    span_cycles = max(1, last_cycle - first_cycle)
    left_margin = 210
    plot_width = 1260
    right_margin = 520
    top_margin = 115
    row_height = 34
    bottom_margin = 55
    width = left_margin + plot_width + right_margin
    height = top_margin + len(cores) * row_height + bottom_margin

    svg: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#111827}.label{font-size:12px}"
        ".small{font-size:10px;fill:#475569}.title{font-size:18px;font-weight:700}</style>",
        '<text x="20" y="28" class="title">dup_server dispatch timeline (rank 0)</text>',
        '<text x="20" y="48" class="small">Absolute cross-core cycles; each compact receiver event shows '
        "flag wait followed by route scan and transfer submission.</text>",
    ]

    legend_x = 20
    legend_y = 72
    present_stages = {segment["stage"] for segment in all_segments}
    for stage, color in DUP_DISPATCH_STAGE_COLORS.items():
        if stage not in present_stages:
            continue
        label_width = 24 + len(stage) * 7
        if legend_x + label_width > width - 20:
            legend_x = 20
            legend_y += 20
        svg.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 17}" y="{legend_y}" class="small">{html.escape(stage)}</text>')
        legend_x += label_width

    for tick in range(11):
        x = left_margin + plot_width * tick / 10
        elapsed_us = span_cycles * tick / 10 / cycles_per_us
        svg.append(
            f'<line x1="{x:.2f}" y1="{top_margin - 10}" x2="{x:.2f}" y2="{height - bottom_margin}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        svg.append(f'<text x="{x:.2f}" y="{height - 25}" text-anchor="middle" class="small">{elapsed_us:.1f} us</text>')

    csv_rows: List[List[Any]] = []
    for row, core in enumerate(sorted(cores)):
        info = cores[core]
        y = top_margin + row * row_height
        gateway = "" if info["gateway_lane"] is None else f"/gw{info['gateway_lane']}"
        svg.append(
            f'<text x="{left_margin - 10}" y="{y + 17}" text-anchor="end" class="label">'
            f"core {core} {html.escape(info['role'] + gateway)}</text>"
        )
        svg.append(
            f'<line x1="{left_margin}" y1="{y + 22}" x2="{left_margin + plot_width}" y2="{y + 22}" '
            'stroke="#f1f5f9" stroke-width="1"/>'
        )
        totals: Dict[str, int] = {}
        for segment in sorted(info["segments"], key=lambda item: item["start_cycle"]):
            stage = segment["stage"]
            start_cycle = segment["start_cycle"]
            end_cycle = segment["end_cycle"]
            duration_cycles = end_cycle - start_cycle
            totals[stage] = totals.get(stage, 0) + duration_cycles
            x = left_margin + (start_cycle - first_cycle) * plot_width / span_cycles
            segment_width = max(0.75, duration_cycles * plot_width / span_cycles)
            duration_us = duration_cycles / cycles_per_us
            title = html.escape(f"core={core}, stage={stage}, duration={duration_us:.3f} us, {segment['detail']}")
            color = DUP_DISPATCH_STAGE_COLORS.get(stage, "#64748b")
            svg.append(
                f'<rect x="{x:.2f}" y="{y + 4}" width="{segment_width:.2f}" height="18" '
                f'fill="{color}" stroke="#ffffff" stroke-width="0.35">'
                f"<title>{title}</title></rect>"
            )
            csv_rows.append(
                [
                    core,
                    info["role"],
                    "" if info["gateway_lane"] is None else info["gateway_lane"],
                    stage,
                    (start_cycle - first_cycle) / cycles_per_us,
                    (end_cycle - first_cycle) / cycles_per_us,
                    duration_us,
                    segment["detail"],
                ]
            )
        core_begin = min(segment["start_cycle"] for segment in info["segments"])
        core_end = max(segment["end_cycle"] for segment in info["segments"])
        total_us = (core_end - core_begin) / cycles_per_us
        summary_names = (
            "receive_wait",
            "route_issue",
            "receiver_overhead",
            "route_scan",
            "remote_get_submit",
            "local_mte_copy",
            "cqe_wait",
        )
        summary = ", ".join(
            f"{name}={totals[name] / cycles_per_us:.1f}" for name in summary_names if totals.get(name, 0) != 0
        )
        summary_x = left_margin + plot_width + 10
        svg.append(f'<text x="{summary_x}" y="{y + 12}" class="small">total={total_us:.1f} us</text>')
        svg.append(f'<text x="{summary_x}" y="{y + 26}" class="small">{html.escape(summary)} us</text>')

    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("core", "role", "gateway_lane", "stage", "start_us", "end_us", "duration_us", "detail"))
        writer.writerows(csv_rows)
    return output_path, csv_path


def _flush_native_streams() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError):
        pass


def _capture_rank0_timer_run(
    args: argparse.Namespace,
    fn: Callable[[], Any],
) -> Any:
    output_path = Path(args.timeline_output or Path(args.profile_dir) / "dup_dispatch_timeline_rank0.svg")
    log_path = output_path.with_suffix(".log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _flush_native_streams()
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            os.dup2(log_file.fileno(), stdout_fd)
            os.dup2(log_file.fileno(), stderr_fd)
            result = fn()
            torch.npu.synchronize()
            _flush_native_streams()
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)

    trace_text = log_path.read_text(encoding="utf-8", errors="replace")
    if trace_text:
        print(trace_text, end="" if trace_text.endswith("\n") else "\n", flush=True)
    try:
        svg_path, csv_path = _write_dup_dispatch_timeline(trace_text, output_path)
        print(
            f"[dup_server][timeline] svg={svg_path.resolve()}, csv={csv_path.resolve()}, log={log_path.resolve()}",
            flush=True,
        )
    except Exception as exc:
        print(f"[dup_server][timeline] generation failed: {exc}; raw_log={log_path.resolve()}", flush=True)
    return result


def test_dup_dispatch_timeline_generation(tmp_path: Path) -> None:
    trace_text = "\n".join(
        (
            "[AIV Block 0/12] [ascend_deepep][dup_dispatch_timer] use_urma_get=1 cycles_per_us=1000 trace_schema=3",
            "[AIV Block 0/12] [ascend_deepep][dup_dispatch_timer] role=producer core=0 "
            "stage=cross_server_transfer start_cycle=1000 end_cycle=5000 cycles=4000",
            "[AIV Block 4/12] [ascend_deepep][dup_dispatch_timer] role=receiver core=4 gateway_lane=0 "
            "stage=receiver_window use_remote_get=1 entry_cycle=1000 receive_start_cycle=1100 "
            "receive_end_cycle=7000",
            "[AIV Block 4/12] [ascend_deepep][dup_dispatch_trace] core=4 gateway_lane=0 sequence=0 "
            "stage=compact source_rank=9 chunk_begin=0 chunk_end=2 merged_chunks=2 "
            "wait_begin_cycle=1100 ready_cycle=2000 issue_done_cycle=6000",
            "[AIV Block 4/12] [ascend_deepep][dup_dispatch_timer] role=receiver core=4 gateway_lane=0 "
            "stage=final_drain_timeline drain_begin_cycle=6120 final_submit_done_cycle=6200 "
            "receive_end_cycle=7000 final_submit_cycles=80 final_wait_cycles=800",
        )
    )
    output_path = tmp_path / "timeline.svg"
    svg_path, csv_path = _write_dup_dispatch_timeline(trace_text, output_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "dup_server dispatch timeline" in svg_text
    assert "core 4 receiver/gw0" in svg_text
    assert "route_issue" in svg_text
    assert "receiver_overhead" in svg_text
    assert "chunks=[0,2)" in csv_text
    assert "source_rank=9" in csv_text
    assert "submit_to_quiet_upper_bound=1.000 us" in csv_text
    assert ",cqe_wait," in csv_text


def _profile_operator(
    rank: int,
    args: argparse.Namespace,
    operator: str,
    fn: Callable[[], Any],
    *,
    stop_after: bool = True,
) -> None:
    import torch.distributed as dist
    import torch_npu

    os.makedirs(args.profile_dir, exist_ok=True)
    _log_rank_stage(
        rank,
        f"profile {operator}: warmup begin ({args.profile_warmup} iterations)",
    )
    for _ in range(args.profile_warmup):
        _consume_profile_result(fn())
    _log_rank_stage(rank, f"profile {operator}: waiting for warmup kernels")
    torch_npu.npu.synchronize()
    _log_rank_stage(rank, f"profile {operator}: warmup complete, capture begin")

    device_id = torch.npu.current_device()
    dist.barrier(device_ids=[device_id])
    profile_files_before: Dict[str, tuple[int, int]] = {}
    if rank == 0:
        try:
            profile_files_before = _profile_csv_snapshot(args.profile_dir)
        except OSError as exc:
            print(
                f"[profile-summary] failed to snapshot {args.profile_dir}: {exc}",
                flush=True,
            )
    dist.barrier(device_ids=[device_id])

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=True,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
        host_sys=[],
        sys_io=False,
        sys_interconnection=False,
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=1,
            active=1,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )

    prof.start()
    for _ in range(2):
        for _ in range(args.profile_iters):
            _consume_profile_result(fn())
        torch_npu.npu.synchronize()
        prof.step()
    prof.stop()
    _log_rank_stage(rank, f"profile {operator}: waiting for captured kernels")
    torch_npu.npu.synchronize()
    _log_rank_stage(rank, f"profile {operator}: capture complete")

    dist.barrier(device_ids=[device_id])

    if rank == 0:
        print(
            f"profile operator {operator} written to {args.profile_dir}: iters_per_step={args.profile_iters}",
            flush=True,
        )
        try:
            _print_profile_summary(args.profile_dir, profile_files_before)
        except Exception as exc:
            print(f"[profile-summary] parsing failed: {exc}", flush=True)
    dist.barrier(device_ids=[device_id])
    if stop_after:
        raise _ProfileComplete


def validate_launch_args(args: argparse.Namespace) -> None:
    if not args.master_addr:
        raise ValueError("--master-addr must not be empty")
    if args.master_port <= 0 or args.master_port > 65535:
        raise ValueError("--master-port must be in [1, 65535]")
    if args.nproc_per_node is None:
        if args.node_rank != 0:
            raise ValueError("--node-rank requires --nproc-per-node")
        return
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be greater than 0")
    if args.ep_size % args.nproc_per_node:
        raise ValueError("--ep-size must be divisible by --nproc-per-node")
    num_nodes = args.ep_size // args.nproc_per_node
    if args.node_rank < 0 or args.node_rank >= num_nodes:
        raise ValueError(f"--node-rank must be in [0, {num_nodes})")


def global_rank_for_local(local_rank: int, args: argparse.Namespace) -> int:
    if args.nproc_per_node is None:
        return local_rank
    return args.node_rank * args.nproc_per_node + local_rank


def local_process_count(args: argparse.Namespace) -> int:
    return args.ep_size if args.nproc_per_node is None else args.nproc_per_node


def effective_dist_init_method(args: argparse.Namespace) -> str:
    if args.dist_init_method:
        return args.dist_init_method
    return f"tcp://{args.master_addr}:{args.master_port}"


def effective_shmem_ip_port(args: argparse.Namespace) -> str:
    if args.shmem_ip_port:
        return args.shmem_ip_port
    shmem_port = args.master_port + 1
    if shmem_port > 65535:
        raise ValueError("--shmem-ip-port is required when --master-port is 65535")
    return f"tcp://{args.master_addr}:{shmem_port}"


def run_dispatch(
    local_rank: int,
    args: argparse.Namespace,
    global_rank: int | None = None,
) -> None:
    """Run one local duplicate-server dispatch worker."""

    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from deep_ep.backends.clos import ElasticBuffer, topk_idx_t

    rank = global_rank if global_rank is not None else global_rank_for_local(local_rank, args)
    device_id = args.first_device + local_rank
    mode = "notify-prepare" if args.run_notify_prepare else "full-dispatch" if args.run_dispatch else "scale-out"
    _log_rank_stage(
        rank,
        f"worker start: mode={mode}, local_rank={local_rank}, device={device_id}",
    )

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(device_id)
    os.environ["WORLD_SIZE"] = str(args.ep_size)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["ASCEND_DEEPEP_TRANSPORT"] = "udma"
    os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"] = args.shmem_ip_port
    _log_rank_stage(rank, "setting NPU device")
    torch.npu.set_device(device_id)
    _log_rank_stage(rank, "HCCL process-group initialization begin")
    dist.init_process_group(
        backend="hccl",
        init_method=effective_dist_init_method(args),
        rank=rank,
        world_size=args.ep_size,
    )
    _log_rank_stage(rank, "HCCL process-group initialized; initial barrier begin")
    dist.barrier(device_ids=[device_id])
    _log_rank_stage(rank, "initial barrier complete")

    buffer = None
    try:
        _log_rank_stage(rank, f"building {args.route} global routing input")
        all_topk_idx = make_global_topk_idx(args)
        num_bytes = ElasticBuffer.get_dup_server_scaleout_buffer_size_hint(
            dist.group.WORLD,
            args.num_max_tokens_per_rank,
            args.hidden,
            args.ranks_per_server,
            args.num_tokens_per_chunk,
            args.num_topk,
            args.num_experts,
            args.gateway_policy,
            args.remote_get_servers,
        )
        _log_rank_stage(
            rank,
            f"ElasticBuffer initialization begin: workspace={num_bytes} bytes, 1g_huge_pages={args.use_1g_huge_pages}",
        )
        buffer = ElasticBuffer(
            dist.group.WORLD,
            num_bytes=num_bytes,
            num_max_tokens_per_rank=args.num_max_tokens_per_rank,
            hidden=args.hidden,
            num_topk=args.num_topk,
            allow_multiple_reduction=True,
            explicitly_destroy=True,
            use_1g_huge_pages=args.use_1g_huge_pages,
        )
        _log_rank_stage(rank, "ElasticBuffer initialization complete")

        if args.run_notify_prepare:
            _log_rank_stage(rank, "preparing Notify input tensors")
            local_topk_idx = all_topk_idx[rank].to(dtype=topk_idx_t)
            local_topk_weights = None
            if args.with_topk_weights:
                local_topk_weights = torch.arange(
                    args.num_tokens * args.num_topk,
                    dtype=torch.float32,
                ).reshape(args.num_tokens, args.num_topk)
                local_topk_weights.div_(max(1, local_topk_weights.numel()))
            npu_topk_idx = local_topk_idx.to(device="npu")
            npu_topk_weights = local_topk_weights.to(device="npu") if local_topk_weights is not None else None
            _log_rank_stage(rank, "Notify input tensors ready")

            def notify_prepare(*, enable_timer: bool = True):
                return buffer.notify_dup_server(
                    npu_topk_idx,
                    args.num_experts,
                    topk_weights=npu_topk_weights,
                    ranks_per_server=args.ranks_per_server,
                    num_tokens_per_chunk=args.num_tokens_per_chunk,
                    num_sms=args.num_sms,
                    enable_timer=enable_timer,
                    async_with_compute_stream=True,
                    gateway_policy=args.gateway_policy,
                )

            if args.profile == "notify-prepare":
                _profile_operator(
                    rank,
                    args,
                    "notify-prepare",
                    lambda: notify_prepare(enable_timer=False),
                    stop_after=False,
                )
            _log_rank_stage(rank, "notify-prepare kernel launch begin")
            notify_meta, notify_event = notify_prepare()
            _log_rank_stage(rank, "notify-prepare kernel submitted; waiting for NPU")
            if notify_event.event is not None:
                notify_event.current_stream_wait(release_handle=True)
            torch.npu.synchronize()
            _log_rank_stage(rank, "notify-prepare kernel complete")
            if not args.skip_check:
                _log_rank_stage(rank, "Notify precision validation begin")
                validate_source_notify_precision(
                    notify_meta,
                    local_topk_idx,
                    local_topk_weights,
                    rank,
                    args,
                )
                validate_full_notify_precision(notify_meta, all_topk_idx, rank, args)
                _log_rank_stage(rank, "Notify precision validation complete")
            if args.verbose or rank == 0:
                print(
                    f"rank={rank}, mode=notify-prepare, aiv={notify_meta.num_sms}, "
                    f"weights={'on' if local_topk_weights is not None else 'off'}, "
                    f"precision={'skipped' if args.skip_check else 'passed'}",
                    flush=True,
                )
            _log_rank_stage(rank, "ElasticBuffer destruction begin")
            buffer.destroy()
            buffer = None
            _log_rank_stage(rank, "ElasticBuffer destruction complete")
            return

        _log_rank_stage(rank, "building CPU golden Dispatch reference")
        meta = build_dup_server_golden_notify(
            all_topk_idx,
            rank=rank,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=args.num_max_tokens_per_rank,
            ranks_per_server=args.ranks_per_server,
            num_tokens_per_chunk=args.num_tokens_per_chunk,
            gateway_policy=args.gateway_policy,
        )
        x = make_dispatch_x(rank, args)
        dispatch_handle = None
        if args.run_dispatch:
            _log_rank_stage(rank, "building Dispatch EPHandle with Notify kernel")
            local_topk_idx = all_topk_idx[rank].to(device="npu", dtype=topk_idx_t)
            dispatch_handle, prepare_overlap = buffer.notify_dup_server(
                local_topk_idx,
                args.num_experts,
                ranks_per_server=args.ranks_per_server,
                num_tokens_per_chunk=args.num_tokens_per_chunk,
                num_sms=args.num_sms,
                async_with_compute_stream=True,
                gateway_policy=args.gateway_policy,
            )
            if prepare_overlap.event is not None:
                prepare_overlap.current_stream_wait(release_handle=True)
            torch.npu.synchronize()
            if not args.skip_check:
                validate_source_notify_precision(
                    dispatch_handle,
                    all_topk_idx[rank],
                    None,
                    rank,
                    args,
                )
                validate_full_notify_precision(
                    dispatch_handle,
                    all_topk_idx,
                    rank,
                    args,
                )
            dispatch_handle.max_inflight_same_frame_servers = args.max_inflight_same_frame_servers
            dispatch_handle.use_urma_get = args.use_urma_get
            _log_rank_stage(rank, "Dispatch EPHandle ready")
            send_token_count = dispatch_handle.send_token_count
            send_token_ids = dispatch_handle.send_token_ids
            send_gateway_lane = dispatch_handle.send_gateway_lane
        else:
            _log_rank_stage(rank, "moving scale-out golden metadata to NPU")
            send_token_count = meta.send_token_count.to(device="npu")
            send_token_ids = meta.send_token_ids.to(device="npu")
            send_gateway_lane = meta.send_gateway_lane.to(device="npu")
            _log_rank_stage(rank, "scale-out metadata submitted to NPU")

        def notify_scaleout():
            return buffer.notify_dup_server_scaleout(
                ranks_per_server=args.ranks_per_server,
                max_inflight_same_frame_servers=(args.max_inflight_same_frame_servers),
                num_tokens_per_chunk=args.num_tokens_per_chunk,
                num_sms=args.num_sms,
                async_with_compute_stream=True,
                gateway_policy=args.gateway_policy,
            )

        def dispatch_scaleout(*, enable_timer: bool = True):
            notify_event = notify_scaleout().event
            return buffer.dispatch_dup_server_scaleout(
                x,
                send_token_count,
                send_token_ids,
                ranks_per_server=args.ranks_per_server,
                max_inflight_same_frame_servers=(args.max_inflight_same_frame_servers),
                num_tokens_per_chunk=args.num_tokens_per_chunk,
                num_sms=args.num_sms,
                previous_event=notify_event,
                async_with_compute_stream=True,
                enable_timer=enable_timer,
                send_gateway_lane=send_gateway_lane,
                gateway_policy=args.gateway_policy,
            )

        def dispatch_full(*, enable_timer: bool = True):
            nonlocal dispatch_handle
            assert dispatch_handle is not None
            if enable_timer or args.remote_get_servers:
                result = buffer.dispatch_dup_server(
                    x,
                    ranks_per_server=args.ranks_per_server,
                    max_inflight_same_frame_servers=args.max_inflight_same_frame_servers,
                    num_tokens_per_chunk=args.num_tokens_per_chunk,
                    num_sms=args.num_sms,
                    use_urma_get=args.use_urma_get,
                    async_with_compute_stream=True,
                    enable_timer=enable_timer,
                    gateway_policy=args.gateway_policy,
                    remote_get_servers=args.remote_get_servers,
                    remote_get_rx_tx_gate=args.remote_get_rx_tx_gate,
                    handle=dispatch_handle,
                    do_cpu_sync=False,
                    do_expand=True,
                )
            else:
                result = buffer.dispatch(
                    x,
                    num_sms=args.num_sms,
                    async_with_compute_stream=True,
                    handle=dispatch_handle,
                    do_cpu_sync=False,
                    do_expand=True,
                )
            dispatch_handle = result[3]
            return result

        if args.profile == "notify":
            _profile_operator(rank, args, "notify", notify_scaleout)
        if args.profile == "dispatch":
            _profile_operator(
                rank,
                args,
                "dispatch",
                lambda: dispatch_scaleout(enable_timer=False),
                stop_after=False,
            )
        if args.profile == "full-dispatch":
            _profile_operator(
                rank,
                args,
                "full-dispatch",
                lambda: dispatch_full(enable_timer=False),
                stop_after=False,
            )
        _log_rank_stage(rank, f"{mode} timer run begin")
        if args.run_dispatch:
            if rank == 0:
                timer_result = _capture_rank0_timer_run(args, dispatch_full)
            else:
                timer_result = dispatch_full()
                torch.npu.synchronize()
            dispatch_output = timer_result[0]
        else:
            dispatch_output, _ = dispatch_scaleout()
            torch.npu.synchronize()
        dist.barrier(device_ids=[device_id])
        _log_rank_stage(rank, f"{mode} timer run complete")

        _log_rank_stage(rank, f"{mode} warmup begin ({args.warmup} iterations)")
        for _ in range(args.warmup):
            if args.run_dispatch:
                dispatch_output = dispatch_full(enable_timer=False)[0]
            else:
                notify_event = notify_scaleout().event
                dispatch_output, _ = buffer.dispatch_dup_server_scaleout(
                    x,
                    send_token_count,
                    send_token_ids,
                    send_gateway_lane=send_gateway_lane,
                    ranks_per_server=args.ranks_per_server,
                    max_inflight_same_frame_servers=args.max_inflight_same_frame_servers,
                    num_tokens_per_chunk=args.num_tokens_per_chunk,
                    num_sms=args.num_sms,
                    previous_event=notify_event,
                    enable_timer=False,
                    gateway_policy=args.gateway_policy,
                )
        _log_rank_stage(rank, f"{mode} warmup submitted; waiting for NPU")
        torch.npu.synchronize()
        _log_rank_stage(rank, f"{mode} warmup complete")

        comm_stream = buffer.get_comm_stream()
        start_events = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
        end_events = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
        _log_rank_stage(rank, f"{mode} timed run begin ({args.iterations} iterations)")
        for index in range(args.iterations):
            start_events[index].record(comm_stream)
            if args.run_dispatch:
                dispatch_output = dispatch_full(enable_timer=False)[0]
            else:
                notify_event = notify_scaleout().event
                dispatch_output, _ = buffer.dispatch_dup_server_scaleout(
                    x,
                    send_token_count,
                    send_token_ids,
                    send_gateway_lane=send_gateway_lane,
                    ranks_per_server=args.ranks_per_server,
                    max_inflight_same_frame_servers=args.max_inflight_same_frame_servers,
                    num_tokens_per_chunk=args.num_tokens_per_chunk,
                    num_sms=args.num_sms,
                    previous_event=notify_event,
                    enable_timer=False,
                    gateway_policy=args.gateway_policy,
                )
            end_events[index].record(comm_stream)
        _log_rank_stage(rank, f"{mode} timed run submitted; waiting for NPU")
        torch.npu.synchronize()
        _log_rank_stage(rank, f"{mode} timed run complete")
        dispatch_times_ms = [float(start.elapsed_time(end)) for start, end in zip(start_events, end_events)]

        if not args.skip_check:
            _log_rank_stage(rank, "Dispatch precision validation begin")
            if args.run_dispatch:
                validate_full_dispatch_precision(meta, dispatch_output, args)
            else:
                validate_dispatch_precision(all_topk_idx, dispatch_output, rank, args)
            torch.npu.synchronize()
            _log_rank_stage(rank, "Dispatch precision validation complete")
        if args.verbose or rank == 0:
            receive_mode = "urma_get" if args.use_urma_get else "datacopy"
            if args.run_dispatch:
                recommended_num_sms = buffer.get_dup_server_num_sms(
                    args.ranks_per_server,
                    args.max_inflight_same_frame_servers,
                    args.remote_get_servers,
                )
                resolved_num_sms = min(
                    args.num_sms or recommended_num_sms,
                    recommended_num_sms,
                )
            else:
                resolved_num_sms = buffer.get_dup_server_scaleout_num_sms(
                    args.ranks_per_server,
                    args.max_inflight_same_frame_servers,
                )
            print(
                f"rank={rank}, mode={'full' if args.run_dispatch else 'scaleout'}, "
                f"aiv={resolved_num_sms}, "
                f"receive={receive_mode if args.run_dispatch else 'n/a'}, "
                f"route={args.route}, "
                f"gateway_policy={args.gateway_policy}, "
                f"remote_get_servers={args.remote_get_servers}, "
                f"remote_get_rx_tx_gate={args.remote_get_rx_tx_gate}, "
                f"tokens_per_chunk={args.num_tokens_per_chunk}, "
                f"workspace={num_bytes}, "
                f"avg_dispatch={statistics.mean(dispatch_times_ms):.3f} ms, "
                f"precision={'skipped' if args.skip_check else 'passed'}",
                flush=True,
            )
        _log_rank_stage(rank, "ElasticBuffer destruction begin")
        buffer.destroy()
        buffer = None
        _log_rank_stage(rank, "ElasticBuffer destruction complete")
    except _ProfileComplete:
        pass
    finally:
        if buffer is not None and not buffer._destroyed:
            _log_rank_stage(rank, "ElasticBuffer cleanup begin")
            buffer.destroy()
            _log_rank_stage(rank, "ElasticBuffer cleanup complete")
        if dist.is_initialized():
            _log_rank_stage(rank, "process-group destruction begin")
            dist.destroy_process_group()
            _log_rank_stage(rank, "process-group destruction complete")


def _percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def print_summary(
    route_time: float,
    golden_time: float,
    check_time: float | None,
    rank_times: List[float],
    metadata: Dict[int, DupServerDispatchMeta],
    args: argparse.Namespace,
) -> None:
    total_recv_routes = sum(meta.num_recv_tokens for meta in metadata.values())
    print("\nDuplicate-server golden notify")
    print(f"  shape: R={args.ep_size}, N={args.num_tokens}, K={args.num_topk}, E={args.num_experts}")
    print(f"  servers: {args.ep_size // args.ranks_per_server}, ranks/server={args.ranks_per_server}")
    print(f"  route mode: {args.route}")
    print(f"  generated ranks: {len(metadata)}, total recv routes={total_recv_routes}")
    print(f"  routing generation: {route_time:.3f}s")
    print(f"  golden generation:  {golden_time:.3f}s")
    print(
        "  per-rank golden:    "
        f"min={min(rank_times):.3f}s, "
        f"median={statistics.median(rank_times):.3f}s, "
        f"p95={_percentile(rank_times, 0.95):.3f}s, "
        f"max={max(rank_times):.3f}s"
    )
    if check_time is not None:
        print(f"  validation:         {check_time:.3f}s (passed)")
    else:
        print("  validation:         skipped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build duplicate-server golden notify metadata")
    parser.add_argument("--ep-size", type=int, default=64)
    parser.add_argument("--first-device", type=int, default=0)
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="zero-based node index; requires --nproc-per-node",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="enable multi-node mode and spawn this many local NPU processes",
    )
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help="HCCL rendezvous host shared by every node",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "29500")),
        help="HCCL rendezvous port shared by every node",
    )
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--ranks-per-server", type=int, default=8)
    parser.add_argument("--num-max-tokens-per-rank", type=int)
    parser.add_argument(
        "--rank",
        type=int,
        help="only generate this rank; default generates the complete group",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--route",
        choices=("balanced", "random", "gateway_aligned_remote"),
        default="balanced",
        help="synthetic top-k routing rule",
    )
    parser.add_argument(
        "--gateway-policy",
        choices=("source_lane",),
        default="source_lane",
        help=(
            "select the remote-server payload gateway: preserve the source "
            "lane or use the token's first matching top-k rank; local-server "
            "traffic always stays on the source lane"
        ),
    )
    parser.add_argument(
        "--drop-route-every",
        type=int,
        default=0,
        help="set every Nth synthetic route to -1; 0 keeps every route",
    )
    parser.add_argument(
        "--gateway-aligned-remote",
        dest="legacy_gateway_aligned_remote",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--num-cpu-threads", type=int, default=1)
    parser.add_argument("--run-notify-prepare", action="store_true")
    parser.add_argument("--run-scaleout", action="store_true")
    parser.add_argument("--run-dispatch", action="store_true")
    parser.add_argument("--with-topk-weights", action="store_true")
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument(
        "--use-1g-huge-pages",
        action="store_true",
        help="back the symmetric communication workspace with 1 GiB HBM pages",
    )
    receive_copy_group = parser.add_mutually_exclusive_group()
    receive_copy_group.add_argument(
        "--use-urma-get",
        dest="use_urma_get",
        action="store_true",
        help="use URMA GET for cross-card scale-up payload reads (default)",
    )
    receive_copy_group.add_argument(
        "--use-datacopy",
        dest="use_urma_get",
        action="store_false",
        help="use DataCopyPad for cross-card scale-up payload reads",
    )
    parser.set_defaults(use_urma_get=True)
    parser.add_argument("--max-inflight-same-frame-servers", type=int, default=3)
    parser.add_argument(
        "--num-tokens-per-chunk",
        type=int,
        default=128,
        help="publish one completion update per compact valid-token chunk",
    )
    parser.add_argument(
        "--remote-get-servers",
        type=int,
        default=0,
        help=(
            "source_lane direct-GET server count: EP8=0, EP32=0..1, "
            "EP64=0..3, EP128=0..7; maximum selects the half-server path"
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--remote-get-rx-tx-gate",
        action="store_true",
        help="wait for server-wide incoming/outgoing PUT completion before Direct GET (all ranks must agree)",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=PROFILE_OPERATORS,
        default=None,
        metavar="OP",
        help="profile Notify preparation, signal reset, scale-out, or full dispatch",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=5,
        help="unrecorded warmup calls for the selected operator",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=10,
        help="selected operator calls in each profiler step",
    )
    parser.add_argument(
        "--profile-dir",
        default="./profile/dup_server",
        help=("torch_npu profiler output directory; rank 0 summarizes new op_summary CSV files automatically"),
    )
    parser.add_argument(
        "--timeline-output",
        default=None,
        help=(
            "rank-0 dispatch timer SVG path; full dispatch defaults to "
            "<profile-dir>/dup_dispatch_timeline_rank0.svg and also writes CSV and raw log files"
        ),
    )
    parser.add_argument(
        "--dist-init-method",
        default=None,
        help="legacy HCCL rendezvous override",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT"),
        help=("CANN SHMEM endpoint; defaults to tcp://MASTER_ADDR:(MASTER_PORT+1)"),
    )
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.legacy_gateway_aligned_remote:
        if args.route != "balanced":
            parser.error("--gateway-aligned-remote cannot be combined with a non-default --route")
        args.route = "gateway_aligned_remote"
    del args.legacy_gateway_aligned_remote
    return args


def required_dup_server_num_sms(args: argparse.Namespace) -> int:
    """Return the minimum AIV count for the selected NPU path."""

    if args.run_notify_prepare:
        return args.ranks_per_server + 1

    num_servers = args.ep_size // args.ranks_per_server
    max_wave_peers = 0
    for frame_begin in range(0, num_servers, DUP_SERVER_SERVERS_PER_FRAME):
        frame_servers = min(
            DUP_SERVER_SERVERS_PER_FRAME,
            num_servers - frame_begin,
        )
        wave_peers = (
            num_servers
            - frame_servers
            + min(
                args.max_inflight_same_frame_servers,
                frame_servers - 1,
            )
        )
        max_wave_peers = max(max_wave_peers, wave_peers)
    if args.run_scaleout:
        return max(1, max_wave_peers)
    if args.remote_get_servers:
        return 16
    return max_wave_peers + 1 + args.ranks_per_server


def validate_args(args: argparse.Namespace) -> None:
    if args.ep_size < 2:
        raise ValueError("ep_size must be at least 2")
    if args.ranks_per_server <= 0:
        raise ValueError("ranks_per_server must be positive")
    if args.ep_size % args.ranks_per_server:
        raise ValueError("ep_size must be divisible by ranks_per_server")
    if args.num_experts % args.ep_size:
        raise ValueError("num_experts must be divisible by ep_size")
    run_modes = sum((args.run_notify_prepare, args.run_scaleout, args.run_dispatch))
    if run_modes > 1:
        raise ValueError("--run-notify-prepare, --run-scaleout and --run-dispatch are mutually exclusive")
    run_npu = run_modes == 1
    if run_npu and (
        args.ranks_per_server != DUP_SERVER_RANKS_PER_SERVER
        or args.ep_size < args.ranks_per_server
        or args.ep_size % args.ranks_per_server
        or args.ep_size // args.ranks_per_server > DUP_SERVER_MAX_SERVERS
    ):
        raise ValueError(
            "URMA scale-out requires 8 ranks/server and ep_size to be a "
            "multiple of 8 with at most 64 servers (512 ranks)"
        )
    if args.num_topk <= 0 or args.num_topk > args.num_experts:
        raise ValueError("num_topk must be in (0, num_experts]")
    if args.run_notify_prepare and args.num_topk > 16:
        raise ValueError("--run-notify-prepare requires num_topk <= 16")
    if args.num_tokens <= 0 or args.hidden <= 0:
        raise ValueError("num_tokens and hidden must be positive")
    if run_npu and args.hidden * 2 % 32:
        raise ValueError("URMA scale-out requires hidden * sizeof(bfloat16) to be 32-byte aligned")
    if args.rank is not None and not 0 <= args.rank < args.ep_size:
        raise ValueError("rank must be in [0, ep_size)")
    if args.drop_route_every < 0:
        raise ValueError("drop_route_every must be non-negative")
    if args.num_cpu_threads <= 0:
        raise ValueError("num_cpu_threads must be positive")
    if args.num_sms < 0:
        raise ValueError("num_sms must be non-negative")
    if not 1 <= args.max_inflight_same_frame_servers <= 7:
        raise ValueError("max_inflight_same_frame_servers must be in [1, 7]")
    if run_npu and args.num_sms:
        required_num_sms = required_dup_server_num_sms(args)
        if args.num_sms < required_num_sms:
            raise ValueError(
                f"selected duplicate-server path requires at least {required_num_sms} "
                f"AIVs, got --num-sms={args.num_sms}"
            )
    if args.num_tokens_per_chunk <= 0:
        raise ValueError("num_tokens_per_chunk must be positive")
    if args.remote_get_servers < 0:
        raise ValueError("remote_get_servers must be non-negative")
    if args.remote_get_rx_tx_gate and args.remote_get_servers == 0:
        raise ValueError("--remote-get-rx-tx-gate requires --remote-get-servers > 0")
    if args.remote_get_servers:
        if not args.run_dispatch:
            raise ValueError("remote_get_servers requires --run-dispatch")
        if args.ep_size not in (32, 64, 128):
            raise ValueError("remote_get_servers requires --ep-size 32, 64, or 128; --ep-size 8 has no remote server")
        if args.gateway_policy != "source_lane":
            raise ValueError("remote_get_servers requires --gateway-policy source_lane")
        max_remote_get_servers = args.ep_size // args.ranks_per_server // 2 - 1
        if args.remote_get_servers > max_remote_get_servers:
            raise ValueError(
                f"remote_get_servers must be in [0, {max_remote_get_servers}] for --ep-size {args.ep_size}"
            )
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    if args.profile is not None and not run_npu:
        raise ValueError("--profile requires a duplicate-server NPU run mode")
    if args.profile == "notify-prepare" and not args.run_notify_prepare:
        raise ValueError("--profile notify-prepare requires --run-notify-prepare")
    if args.run_notify_prepare and args.profile not in (None, "notify-prepare"):
        raise ValueError("--run-notify-prepare only supports --profile notify-prepare")
    if args.profile == "full-dispatch" and not args.run_dispatch:
        raise ValueError("--profile full-dispatch requires --run-dispatch")
    if args.with_topk_weights and not args.run_notify_prepare:
        raise ValueError("--with-topk-weights requires --run-notify-prepare")
    if args.profile_warmup < 0 or args.profile_iters <= 0:
        raise ValueError("profile_warmup must be non-negative and profile_iters positive")
    if not args.profile_dir:
        raise ValueError("profile_dir must not be empty")
    if args.timeline_output is not None and not args.run_dispatch:
        raise ValueError("--timeline-output requires --run-dispatch")
    if args.timeline_output == "":
        raise ValueError("timeline_output must not be empty")
    if args.timeline_output is not None and Path(args.timeline_output).suffix.lower() != ".svg":
        raise ValueError("timeline_output must use the .svg suffix")
    if args.num_max_tokens_per_rank is None:
        args.num_max_tokens_per_rank = args.num_tokens
    elif args.num_max_tokens_per_rank < args.num_tokens:
        raise ValueError("num_max_tokens_per_rank must be no smaller than num_tokens")


def main() -> None:
    args = parse_args()
    validate_args(args)
    validate_launch_args(args)
    args.shmem_ip_port = effective_shmem_ip_port(args)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_num_interop_threads(1)

    if args.run_notify_prepare or args.run_scaleout or args.run_dispatch:
        external_torchrun = (
            args.nproc_per_node is None
            and all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
            and int(os.environ["WORLD_SIZE"]) == args.ep_size
        )
        if external_torchrun:
            run_dispatch(
                int(os.environ["LOCAL_RANK"]),
                args,
                global_rank=int(os.environ["RANK"]),
            )
            return

        local_count = local_process_count(args)
        first_global_rank = global_rank_for_local(0, args)
        last_global_rank = global_rank_for_local(local_count - 1, args)
        print(
            "Launching duplicate-server "
            f"{'notify-prepare' if args.run_notify_prepare else 'dispatch' if args.run_dispatch else 'scale-out'} "
            "with "
            f"ep_size={args.ep_size}, node_rank={args.node_rank}, "
            f"local_processes={local_count}, "
            f"global_ranks={first_global_rank}-{last_global_rank}, "
            f"first_device={args.first_device}, "
            f"master={args.master_addr}:{args.master_port}, "
            f"shmem_ip_port={args.shmem_ip_port}",
            flush=True,
        )
        if local_count == 1:
            run_dispatch(0, args)
        else:
            mp.spawn(
                run_dispatch,
                args=(args,),
                nprocs=local_count,
                join=True,
            )
        return

    begin = time.perf_counter()
    all_topk_idx = make_global_topk_idx(args)
    route_time = time.perf_counter() - begin

    begin = time.perf_counter()
    metadata, rank_times = build_golden_metadata(all_topk_idx, args)
    golden_time = time.perf_counter() - begin

    check_time = None
    if not args.skip_check:
        begin = time.perf_counter()
        validate_golden_metadata(all_topk_idx, metadata)
        check_time = time.perf_counter() - begin

    print_summary(
        route_time,
        golden_time,
        check_time,
        rank_times,
        metadata,
        args,
    )


if __name__ == "__main__":
    main()


# Golden only:
# python test/test_dup_server.py
# python test/test_dup_server.py --rank 0 --verbose
# python test/test_dup_server.py --skip-check
#
# Source Notify preparation and precision check:
# python test/test_dup_server.py --run-notify-prepare --ep-size 64 \
#   --nproc-per-node 8 --node-rank 0 --master-addr 10.0.0.1 \
#   --master-port 29500
#
# URMA scale-out (one torchrun worker per rank):
# torchrun ... test/test_dup_server.py --run-scaleout --ep-size 64
# torchrun ... test/test_dup_server.py --run-scaleout --ep-size 64 \
#   --profile dispatch --profile-warmup 5 --profile-iters 10 \
#   --profile-dir ./profile/dup_server
# Multi-node spawn mode (run once per node with node-rank 0..7):
# python test/test_dup_server.py --run-scaleout --ep-size 64 \
#   --nproc-per-node 8 --node-rank 0 --master-addr 10.0.0.1 \
#   --master-port 29500
# Complete hierarchical dispatch:
# python test/test_dup_server.py --run-dispatch --ep-size 64 \
#   --nproc-per-node 8 --node-rank 0 --master-addr 10.0.0.1 \
#   --master-port 29500 --timeline-output ./profile/dup_server/dispatch.svg
# Adjustable remote GET (EP32/64/128 maxima are 1/3/7):
# python test/test_dup_server.py --run-dispatch --ep-size 64 --remote-get-servers 3 \
#   --num-tokens 8192 --hidden 7168 --num-topk 16 --num-experts 384 \
#   --num-tokens-per-chunk 128 --warmup 50 --iterations 50 \
#   --nproc-per-node 8 --node-rank 0 --master-addr 10.0.0.1 --master-port 29500
