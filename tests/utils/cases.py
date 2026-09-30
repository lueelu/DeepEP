# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Deterministic global inputs, with deliberately unequal token counts."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Case:
    name: str
    experts: int
    hidden: int
    topk: int
    x: tuple
    routes: tuple
    weights: Optional[tuple]
    dtype: str = "bfloat16"
    expert_alignment: int = 1
    distribution: str = "tagged"

    @property
    def world_size(self):
        return len(self.x)

    def validate(self):
        if self.world_size < 1 or self.experts < 1 or self.experts % self.world_size:
            raise ValueError("Experts must divide evenly across nonempty ranks")
        if self.hidden < 1 or self.topk < 1 or self.topk > self.experts:
            raise ValueError("hidden and topk must be positive and topk must not exceed experts")
        if self.expert_alignment < 1:
            raise ValueError("expert_alignment must be positive")
        if self.distribution not in ("tagged", "uniform", "normal", "small", "outlier"):
            raise ValueError("unknown input distribution")
        if self.dtype not in ("bfloat16", "float8_e4m3fn"):
            raise ValueError("dtype must be bfloat16 or float8_e4m3fn")
        if len(self.routes) != self.world_size:
            raise ValueError("Routing rank count mismatch")
        if self.weights is not None and len(self.weights) != self.world_size:
            raise ValueError("Weight rank count mismatch")
        for rank, (tokens, routes) in enumerate(zip(self.x, self.routes)):
            if len(tokens) != len(routes):
                raise ValueError("Token/routing count mismatch")
            if self.weights is not None and len(self.weights[rank]) != len(tokens):
                raise ValueError("Token/weight count mismatch")
            for token, (x, indices) in enumerate(zip(tokens, routes)):
                if len(x) != self.hidden or len(indices) != self.topk:
                    raise ValueError("Token/routing shape mismatch")
                if any(type(e) is not int or e < -1 or e >= self.experts for e in indices):
                    raise ValueError("Expert index must be -1 or in [0, num_experts)")
                if self.weights is not None and len(self.weights[rank][token]) != self.topk:
                    raise ValueError("Weight shape mismatch")


# Every input token has a unique BF16-exact tag in column zero. Two ranks and
# two experts per rank keep the reference hand-checkable. Repeated slots count
# separately; -1 slots contribute nothing, including an entirely masked token.
ROUTES = {
    "normal": (((0, 2), (1, 3), (2, 1)), ((3, 0), (2, 1))),
    "same_rank": (((0, 1), (2, 3), (0, 1)), ((2, 3),)),
    "repeated_slot": (((0, 0), (2, 2), (1, 3)), ((3, 3),)),
    "masked": (((-1, 2), (0, -1), (-1, -1)), ((3, -1),)),
    "empty_expert": (((0, 2), (2, 0), (0, 2)), ((2, 0),)),
    "skew": (((0, 1), (0, 1), (0, 1)), ((0, 1),)),
    "zero_token_rank": ((), ((0, 2), (1, 3))),
    "all_masked": (((-1, -1), (-1, -1)), ((-1, -1),)),
}


def make_case(name, *, hidden=4, with_weights=True, dtype="bfloat16", expert_alignment=1):
    routes = ROUTES[name]
    x = tuple(
        tuple(tuple(1 + rank * 8 + token + col % 3 for col in range(hidden)) for token in range(len(rank_routes)))
        for rank, rank_routes in enumerate(routes)
    )
    weights = (
        tuple(
            tuple(
                tuple((1 + rank * 8 + token * 2 + slot) / 16 for slot in range(2)) for token in range(len(rank_routes))
            )
            for rank, rank_routes in enumerate(routes)
        )
        if with_weights
        else None
    )
    case = Case(name, 4, hidden, 2, x, routes, weights, dtype, expert_alignment)
    case.validate()
    return case


def make_matrix_case(
    name,
    *,
    world_size,
    experts,
    hidden,
    topk,
    tokens_per_rank,
    dtype="bfloat16",
    with_weights=True,
    expert_alignment=1,
    distribution="tagged",
):
    """Create a deterministic, non-random qualification case of arbitrary size.

    Values are intentionally non-integral.  Route slots cover local and remote
    experts, include deterministic masks, and preserve repeated destinations.
    The formula is stable across Python versions so checked-in golden digests can
    detect an accidental oracle change.
    """
    if len(tokens_per_rank) != world_size:
        raise ValueError("tokens_per_rank must contain one entry per rank")
    x, routes, weights = [], [], []

    def unit(seed):
        return ((seed * 1103515245 + 12345) & 0x7FFFFFFF) / 0x80000000

    def value_for(rank, token, col, outliers):
        linear = token * hidden + col
        seed = (rank + 1) * 1000003 + (token + 1) * 1009 + col
        if distribution == "tagged":
            return (((rank + 1) * 131 + (token + 3) * 17 + (col + 5) * 29) % 2041 - 1020) / 64.0
        if distribution == "uniform":
            return unit(seed) * 10.0 - 5.0
        if distribution == "normal":
            standard = sum(unit(seed + offset * 7919) for offset in range(12)) - 6.0
            mean = ((rank * 17 + token * 3) % 201) - 100
            deviation = 1 + ((rank * 11 + token * 5) % 25)
            return mean + standard * deviation
        if distribution == "small":
            threshold = 2**-8 if dtype == "bfloat16" else 2**-4
            return (unit(seed) * 2.0 - 1.0) * threshold
        base = unit(seed) * 0.01 - 0.005
        return base * 1000.0 if linear in outliers else base

    for rank, count in enumerate(tokens_per_rank):
        rank_x, rank_routes, rank_weights = [], [], []
        total_values = count * hidden
        outlier_count = max(1, total_values // 1000) if total_values else 0
        outliers = (
            {min(total_values - 1, index * total_values // outlier_count) for index in range(outlier_count)}
            if outlier_count
            else set()
        )
        for token in range(count):
            rank_x.append(tuple(value_for(rank, token, col, outliers) for col in range(hidden)))
            token_routes = []
            for slot in range(topk):
                expert = (rank * 11 + token * 7 + slot * 13) % experts
                if (rank * 5 + token * 3 + slot + 1) % 97 == 0:
                    expert = -1
                token_routes.append(expert)
            rank_routes.append(tuple(token_routes))
            if with_weights:
                rank_weights.append(
                    tuple(((rank + 1) * 19 + (token + 1) * 7 + slot * 3) % 127 / 128.0 for slot in range(topk))
                )
        x.append(tuple(rank_x))
        routes.append(tuple(rank_routes))
        if with_weights:
            weights.append(tuple(rank_weights))
    case = Case(
        name,
        experts,
        hidden,
        topk,
        tuple(x),
        tuple(routes),
        tuple(weights) if with_weights else None,
        dtype,
        expert_alignment,
        distribution,
    )
    case.validate()
    return case
