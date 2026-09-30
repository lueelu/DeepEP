# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Synchronous, all-rank V1 functional stub. No devices or native fallback.

This test adapter models rank-deduplicated BF16 dispatch/combine, not the
public Buffer ABI, SHMEM transport, asynchronous events, or kernel timing.
It deliberately does not import the routing oracle in tests.utils.reference.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Optional

from tests.utils.precision import _float32 as fp32, quantize_bfloat16 as bf16


def finite_float32(value):
    if type(value) not in (int, float):
        raise ValueError("payloads must contain finite numeric values")
    try:
        result = fp32(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("payload is outside finite FP32 range") from error
    if not isfinite(result):
        raise ValueError("payloads must contain finite numeric values")
    return result


def finite_bfloat16(value):
    result = bf16(finite_float32(value))
    if not isfinite(result):
        raise ValueError("payload is outside finite BF16 range")
    return result


@dataclass(frozen=True)
class Received:
    x: tuple
    topk_idx: tuple
    topk_weights: Optional[tuple]
    counts: tuple


@dataclass(frozen=True, eq=False)
class _Handle:
    origins: tuple
    valid_slots: tuple
    token_counts: tuple
    hidden: int
    topk: int


@dataclass(frozen=True)
class Dispatched:
    ranks: tuple
    handle: _Handle


class FunctionalStub:
    """One synchronous pending operation, with immutable received snapshots.

    A successful fresh dispatch invalidates the previous handle. A successful
    combine consumes its handle. These are test-adapter constraints, not a
    claim about native Buffer handle reuse or asynchronous ownership.
    """

    def __init__(self):
        self._active = None
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("functional stub is destroyed")

    @staticmethod
    def _validate(case):
        if any(type(value) is not int for value in (case.hidden, case.experts, case.topk, case.expert_alignment)):
            raise ValueError("shape and alignment fields must be integers")
        if case.dtype != "bfloat16" or case.expert_alignment != 1:
            raise ValueError("functional stub supports BF16 and alignment=1 only")
        if case.world_size != 2 or case.topk not in (1, 2):
            raise ValueError("functional stub supports two simulated ranks and top-k 1/2 only")
        if not 1 <= case.hidden <= 512 or not 1 <= case.experts <= 64:
            raise ValueError("functional stub requires hidden <= 512 and experts <= 64")
        if any(len(rank) > 64 for rank in case.x):
            raise ValueError("functional stub allows at most 64 tokens per rank")
        case.validate()

    def dispatch(self, case):
        self._check_open()
        self._validate(case)
        local_experts = case.experts // case.world_size
        data, indices, weights, origins, valid_slots = ([[], []] for _ in range(5))
        counts = [[0] * local_experts for _ in range(2)]
        # Source-first scatter, deliberately separate from destination-first
        # oracle construction. Route metadata lives only in this handle.
        for source, tokens in enumerate(case.x):
            for token, values in enumerate(tokens):
                row = tuple(finite_bfloat16(value) for value in values)
                routes = tuple(case.routes[source][token])
                gates = (
                    None
                    if case.weights is None
                    else tuple(finite_float32(value) for value in case.weights[source][token])
                )
                destinations = sorted({expert // local_experts for expert in routes if expert >= 0})
                for destination in destinations:
                    local = tuple(
                        expert % local_experts if expert >= 0 and expert // local_experts == destination else -1
                        for expert in routes
                    )
                    slots = tuple(slot for slot, expert in enumerate(local) if expert >= 0)
                    data[destination].append(row)
                    indices[destination].append(local)
                    if gates is not None:
                        weights[destination].append(gates)
                    origins[destination].append((source, token))
                    valid_slots[destination].append(slots)
                    for slot in slots:
                        counts[destination][local[slot]] += 1
        handle = _Handle(
            tuple(map(tuple, origins)),
            tuple(map(tuple, valid_slots)),
            tuple(map(len, case.x)),
            case.hidden,
            case.topk,
        )
        received = tuple(
            Received(
                tuple(data[r]), tuple(indices[r]), None if case.weights is None else tuple(weights[r]), tuple(counts[r])
            )
            for r in range(2)
        )
        self._active = handle
        return Dispatched(received, handle)

    def combine(self, x, handle, *, topk_weights=None):
        self._check_open()
        if handle is None or handle is not self._active:
            raise ValueError("foreign, stale or consumed functional handle")
        if len(x) != 2 or (topk_weights is not None and len(topk_weights) != 2):
            raise ValueError("combine rank count mismatch")
        payloads, values = [], []
        # Validate the entire operation before reducing or consuming the handle.
        for rank in range(2):
            if len(x[rank]) != len(handle.origins[rank]):
                raise ValueError("combine row count mismatch")
            if topk_weights is not None and len(topk_weights[rank]) != len(x[rank]):
                raise ValueError("combine value row count mismatch")
            rank_x, rank_values = [], []
            for index, row in enumerate(x[rank]):
                if len(row) != handle.hidden:
                    raise ValueError("combine hidden dimension mismatch")
                converted = tuple(finite_bfloat16(value) for value in row)
                if converted != tuple(row):
                    raise ValueError("combine inputs must already be BF16-representable")
                rank_x.append(converted)
                if topk_weights is not None:
                    if len(topk_weights[rank][index]) != handle.topk:
                        raise ValueError("combine value width mismatch")
                    rank_values.append(tuple(finite_float32(value) for value in topk_weights[rank][index]))
            payloads.append(rank_x)
            values.append(rank_values)
        output = [[[0.0] * handle.hidden for _ in range(n)] for n in handle.token_counts]
        returned = (
            None if topk_weights is None else [[[0.0] * handle.topk for _ in range(n)] for n in handle.token_counts]
        )
        for rank, rows in enumerate(handle.origins):
            for index, (source, token) in enumerate(rows):
                for column, value in enumerate(payloads[rank][index]):
                    output[source][token][column] = finite_float32(output[source][token][column] + value)
                if returned is not None:
                    for slot in handle.valid_slots[rank][index]:
                        returned[source][token][slot] = finite_float32(
                            returned[source][token][slot] + values[rank][index][slot]
                        )
        result = tuple(tuple(tuple(finite_bfloat16(v) for v in row) for row in rank) for rank in output)
        returned = None if returned is None else tuple(tuple(map(tuple, rank)) for rank in returned)
        self._active = None
        return result, returned

    def destroy(self):
        self._active = None
        self._closed = True


def create_backend(name):
    if name != "functional-stub":
        raise ValueError("this test runner requires backend='functional-stub'; no native fallback exists")
    return FunctionalStub()
