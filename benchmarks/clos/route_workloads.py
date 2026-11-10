"""Reproducible synthetic router workloads, independent of kernel scheduling.

No torch/device dependency. Biased routing takes top-k of independent uniform
expert scores, boosting one pool. Uniform order statistics avoid materializing
T x E scores: only the largest K scores of each pool can reach the final top-k.
These are synthetic loads, not a model-router trace.
"""

import functools
import hashlib
import heapq
import json
import math
import random

DEFAULTS = dict(
    route_mode="uniform",
    route_skew=2.0,
    route_group_size=None,
    route_hotspot=0,
    route_local_fraction=0.8,
    route_mask_rate=0.0,
)


def add_arguments(parser):
    parser.add_argument(
        "--route-mode",
        choices=("uniform", "rank_hotspot", "local_group"),
        default="uniform",
        help="random expert scores; no forced expert/rank hits",
    )
    parser.add_argument(
        "--route-skew", type=float, default=2.0, help="target deduplicated receive rows: hot rank / mean non-hot rank"
    )
    parser.add_argument(
        "--route-group-size", type=int, help="fixed contiguous load group size, independent of scheduling group size"
    )
    parser.add_argument("--route-hotspot", type=int, default=0, help="hot rank index (rank_hotspot only)")
    parser.add_argument(
        "--route-local-fraction",
        type=float,
        default=0.8,
        help="target fraction of deduplicated rows within source load group",
    )
    parser.add_argument(
        "--route-mask-rate",
        type=float,
        default=0.0,
        help="independent probability of replacing a selected expert by -1",
    )


def options(args):
    return {k: getattr(args, k, v) for k, v in DEFAULTS.items()}


def validate(args):
    o = options(args)
    world = args.ep_size
    g = o["route_group_size"]
    if o["route_mode"] not in ("uniform", "rank_hotspot", "local_group"):
        raise ValueError("unknown route-mode")
    if not math.isfinite(o["route_skew"]) or o["route_skew"] < 1:
        raise ValueError("route-skew must be finite and >= 1")
    if not math.isfinite(o["route_mask_rate"]) or not 0 <= o["route_mask_rate"] <= 1:
        raise ValueError("route-mask-rate must be in [0, 1]")
    if not math.isfinite(o["route_local_fraction"]) or not 0 < o["route_local_fraction"] < 1:
        raise ValueError("route-local-fraction must be in (0, 1)")
    if g is not None and (g <= 0 or world % g):
        raise ValueError("route-group-size must divide EP size")
    if o["route_mode"] == "local_group" and (g is None or g == world):
        raise ValueError("group workloads require route-group-size and at least two load groups")
    if not 0 <= o["route_hotspot"] < world:
        raise ValueError("route-hotspot is outside the rank range")


def _pool_top(rng, count, take, factor, offset):
    ids = rng.sample(range(offset, offset + count), min(take, count))
    value = factor
    result = []
    for i, expert in enumerate(ids):
        value *= rng.random() ** (1.0 / (count - i))
        result.append((value, expert))
    return result


def _sample(rng, experts, k, hot_experts, factor):
    candidates = _pool_top(rng, hot_experts, k, factor, 0) + _pool_top(rng, experts - hot_experts, k, 1.0, hot_experts)
    selected = [expert for _, expert in heapq.nlargest(k, candidates)]
    # Do not couple top-k slot position to rank or score ordering.
    rng.shuffle(selected)
    return selected


def _observed(rows, per_rank, hot_ranks, world, fraction):
    hot = cold = 0
    for row in rows:
        peers = {e // per_rank for e in row}
        hot += sum(p < hot_ranks for p in peers)
        cold += sum(p >= hot_ranks for p in peers)
    if fraction:
        return hot / (hot + cold) if hot + cold else 0.0
    cold_groups = (world - hot_ranks) / hot_ranks
    return hot * cold_groups / cold if cold else float("inf")


@functools.lru_cache(maxsize=32)
def _calibrate(world, per_rank, k, hot_ranks, target, fraction, seed):
    """Fixed pilot/common RNG; no source rank, dtype, expand or scheduler inputs."""

    def measure(factor):
        rng = random.Random(seed ^ 0x5A17C0DE)
        rows = (_sample(rng, world * per_rank, k, hot_ranks * per_rank, factor) for _ in range(1024))
        return _observed(rows, per_rank, hot_ranks, world, fraction)

    low, high = 1e-6, 1e6
    lo_value, hi_value = measure(low), measure(high)
    if target < lo_value - 0.02 or target > hi_value + 0.02:
        raise ValueError(f"route target {target} unreachable for EP/E/K; pilot range {lo_value}..{hi_value}")
    best = (float("inf"), 1.0)
    for _ in range(22):
        factor = math.sqrt(low * high)
        actual = measure(factor)
        best = min(best, (abs(actual - target), factor))
        if actual < target:
            low = factor
        else:
            high = factor
    return best[1]


def generate(rank, tokens, world, per_rank, k, seed=20260823, **kwargs):
    o = {**DEFAULTS, **kwargs}
    from types import SimpleNamespace

    validate(SimpleNamespace(ep_size=world, **o))
    if not 0 <= rank < world or tokens < 0 or not 1 <= k <= world * per_rank:
        raise ValueError("invalid rank/tokens/experts/top-k")
    rng = random.Random(seed + rank)
    mask_rng = random.Random((seed + rank) ^ 0x341AD5)
    experts = world * per_rank
    mode = o["route_mode"]
    factor, hot_ranks, first = 1.0, 0, 0
    if mode != "uniform":
        hot_ranks = 1 if mode == "rank_hotspot" else o["route_group_size"]
        first = (rank // hot_ranks if mode == "local_group" else o["route_hotspot"]) * hot_ranks
        fraction = mode == "local_group"
        target = o["route_local_fraction"] if fraction else o["route_skew"]
        factor = _calibrate(world, per_rank, k, hot_ranks, target, fraction, seed)
    rows = []
    for _ in range(tokens):
        if mode == "uniform":
            selected = rng.sample(range(experts), k)
        else:
            selected = _sample(rng, experts, k, hot_ranks * per_rank, factor)
            # Rotate a canonical hot pool to the requested physical ranks.
            selected = [(e + first * per_rank) % experts for e in selected]
        rows.append([e if mask_rng.random() >= o["route_mask_rate"] else -1 for e in selected])
    return rows


def statistics(routes, per_rank, group_size=None):
    world = len(routes)
    g = group_size or 1
    if g <= 0 or world % g:
        raise ValueError("invalid load group size")
    matrix = [[0] * world for _ in range(world)]
    group_matrix = [[0] * (world // g) for _ in range(world // g)]
    fanout, contributions = {}, {}
    experts = [0] * (world * per_rank)
    for src, rows in enumerate(routes):
        for row in rows:
            counts = {}
            seen = set()
            for e in row:
                if e == -1:
                    continue
                if not 0 <= e < len(experts) or e in seen:
                    raise ValueError("invalid or duplicate expert in route")
                seen.add(e)
                experts[e] += 1
                peer = e // per_rank
                counts[peer] = counts.get(peer, 0) + 1
            fanout[str(len(counts))] = fanout.get(str(len(counts)), 0) + 1
            for peer, count in counts.items():
                matrix[src][peer] += 1
                group_matrix[src // g][peer // g] += 1
                contributions[str(count)] = contributions.get(str(count), 0) + 1
    receive = [sum(row[p] for row in matrix) for p in range(world)]
    send = [sum(row) for row in matrix]
    group_receive = [sum(row[p] for row in group_matrix) for p in range(world // g)]

    def imbalance(values):
        return max(values) * len(values) / sum(values) if sum(values) else None

    def ratios(values):
        return [v * (len(values) - 1) / (sum(values) - v) if sum(values) > v else None for v in values]

    total = sum(send)
    return dict(
        route_sha256=hashlib.sha256(json.dumps(routes, separators=(",", ":")).encode()).hexdigest(),
        rows=matrix,
        group_rows=group_matrix,
        load_group_size=g,
        send_rows=send,
        recv_rows=receive,
        recv_max_over_mean=imbalance(receive),
        rank_over_other_mean=ratios(receive),
        group_over_other_mean=ratios(group_receive),
        fanout_histogram=fanout,
        contribution_histogram=contributions,
        expert_hits=experts,
        valid_expert_hits=sum(experts),
        single_contribution_fraction=contributions.get("1", 0) / total if total else None,
        local_group_row_fraction=sum(group_matrix[i][i] for i in range(world // g)) / total if total else None,
    )
