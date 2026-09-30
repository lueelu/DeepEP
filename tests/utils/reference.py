# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""All-rank CPU oracle; no Torch, collectives, backend handles or device metadata.

Semantics follow the pinned DeepSeek reference (see third-party/README.md).
This is a small mathematical model, not a device backend or a reduction-order
simulator. Alignment is one; accumulation uses Python arithmetic.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Row:
    source: int
    token: int
    slots: tuple
    indices: Optional[tuple]  # Local expert IDs; None in expanded layout.
    x: tuple
    weights: Optional[tuple]  # One element in expanded layout.
    expert: Optional[int] = None  # Global expert in expanded layout.


@dataclass(frozen=True)
class Dispatch:
    expanded: bool
    ranks: tuple
    counts: tuple  # Per rank, per local expert, counting slots (not unique tokens).


@dataclass(frozen=True)
class Payload:
    x: tuple
    weights: Optional[tuple]


def dispatch(case, *, expanded=False):
    case.validate()
    local_experts = case.experts // case.world_size
    ranks, counts = [], []
    for destination in range(case.world_size):
        rows = []
        expert_counts = [0] * local_experts
        for source, routes in enumerate(case.routes):
            for token, indices in enumerate(routes):
                slots = tuple(s for s, e in enumerate(indices) if e >= 0 and e // local_experts == destination)
                for slot in slots:
                    expert_counts[indices[slot] % local_experts] += 1
                if expanded:
                    for slot in slots:
                        weights = None if case.weights is None else (case.weights[source][token][slot],)
                        rows.append(
                            Row(
                                source,
                                token,
                                (slot,),
                                None,
                                case.x[source][token],
                                weights,
                                indices[slot],
                            )
                        )
                elif slots:
                    local = tuple(e % local_experts if s in slots else -1 for s, e in enumerate(indices))
                    weights = None if case.weights is None else case.weights[source][token]
                    rows.append(Row(source, token, slots, local, case.x[source][token], weights))
        if expanded:
            rows.sort(key=lambda row: (row.expert, row.source, row.token, row.slots))
        ranks.append(tuple(rows))
        counts.append(tuple(expert_counts))
    return Dispatch(expanded, tuple(ranks), tuple(counts))


def expert_payload(case, layout, *, use_received_x=False):
    """Simulated expert outputs plus an independent FP32-like value channel.

    combine-only inputs come from global source data. Roundtrip inputs come
    from received x, so dispatch errors cannot be hidden by regenerating x.
    No gate multiplication is performed. Values are intentionally different
    from dispatch weights to exercise the return channel (e.g. dProb).
    """
    ranks = []
    for rows in layout.ranks:
        payloads = []
        for row in rows:
            source_x = row.x if use_received_x else case.x[row.source][row.token]
            factors = [case.routes[row.source][row.token][slot] + 1 for slot in row.slots]
            x = tuple(value * sum(factors) for value in source_x)
            if case.weights is None:
                weights = None
            else:
                slots = row.slots if layout.expanded else range(case.topk)
                # Poison nonlocal values: combine must not add them to a slot.
                weights = tuple(
                    0.5 - case.weights[row.source][row.token][s] if s in row.slots else 777.0 for s in slots
                )
            payloads.append(Payload(x, weights))
        ranks.append(tuple(payloads))
    return tuple(ranks)


def combine(case, layout, payloads):
    """Reduce one payload per received row; weights never scale payload.x."""
    case.validate()
    if len(payloads) != case.world_size or len(layout.ranks) != case.world_size:
        raise ValueError("Combine rank count mismatch")
    output = [[[0.0] * case.hidden for _ in tokens] for tokens in case.x]
    weights = None if case.weights is None else [[[0.0] * case.topk for _ in tokens] for tokens in case.x]
    for rows, rank_payloads in zip(layout.ranks, payloads):
        if len(rows) != len(rank_payloads):
            raise ValueError("Combine row count mismatch")
        for row, payload in zip(rows, rank_payloads):
            if len(payload.x) != case.hidden:
                raise ValueError("Combine hidden dimension mismatch")
            width = 1 if layout.expanded else case.topk
            if (weights is None) != (payload.weights is None) or (
                payload.weights is not None and len(payload.weights) != width
            ):
                raise ValueError("Combine weight shape mismatch")
            for col, value in enumerate(payload.x):
                output[row.source][row.token][col] += value
            if weights is not None:
                for index, slot in enumerate(row.slots):
                    weights[row.source][row.token][slot] += payload.weights[index if layout.expanded else slot]

    def freeze(data):
        return tuple(tuple(tuple(row) for row in rank) for rank in data)

    return freeze(output), None if weights is None else freeze(weights)


def expected_expert_combine(case):
    """Direct source-side answer, deliberately independent of dispatch/Row."""
    case.validate()
    output, weights = [], []
    for source, routes in enumerate(case.routes):
        rank_x, rank_weights = [], []
        for token, indices in enumerate(routes):
            factor = sum(expert + 1 for expert in indices if expert >= 0)
            rank_x.append(tuple(value * factor for value in case.x[source][token]))
            if case.weights is not None:
                rank_weights.append(
                    tuple(
                        0.5 - case.weights[source][token][slot] if expert >= 0 else 0.0
                        for slot, expert in enumerate(indices)
                    )
                )
        output.append(tuple(rank_x))
        weights.append(tuple(rank_weights))
    return tuple(output), None if case.weights is None else tuple(weights)
