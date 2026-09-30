# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""No imports from the DUT: route order, BF16 rounding and CPU FP32 test oracle."""

import numpy as np


def bf16(value):
    a = np.asarray(value, dtype=np.float32)
    bits = a.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def route_matrix(case):
    """rows = source rank, columns = destination rank then local expert."""
    from .routing import special_route

    special = special_route(case)
    if special is not None:
        return special
    p, e = case.ep_size, case.local_experts
    rng = np.random.default_rng(case.seed)
    table = np.zeros((p, p * e), dtype=np.int64)
    for source in range(p):
        weights = np.ones(p * e, dtype=np.float64)
        if case.route == "random":
            weights = rng.lognormal(0, 2, p * e)
        elif case.route in ("hot_expert", "hot_rank"):
            weights[: 1 if case.route == "hot_expert" else e] = 100
        elif case.route in ("local", "remote", "ring"):
            weights[:] = 0
            target = source if case.route == "local" else (source + 1) % p
            if case.route == "remote" and p > 1:
                weights[:] = 1
                weights[source * e : (source + 1) * e] = 0
            else:
                weights[target * e : (target + 1) * e] = 1
        elif case.route == "empty_experts":
            weights[:] = 0
            weights[::e] = 1
        raw = weights / weights.sum() * case.m
        counts = np.floor(raw).astype(np.int64)
        for index in np.argsort(-(raw - counts), kind="stable")[: case.m - counts.sum()]:
            counts[index] += 1
        # Keep every destination's input nonzero (a PR host constraint for inverse).
        # Hot traffic remains strongly skewed, with zero expert/source segments.
        table[source] = counts
    for target in range(p):
        if table[:, target * e : (target + 1) * e].sum() == 0:
            donor = int(table.reshape(p, p, e).sum(axis=(0, 2)).argmax())
            source, local = np.unravel_index(table[:, donor * e : (donor + 1) * e].argmax(), (p, e))
            table[source, donor * e + local] -= 1
            table[source, target * e] += 1
    return table


def data(shape, kind, seed, weight=False):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.25, shape).astype(np.float32)
    if kind == "zero":
        values.fill(0)
    elif kind == "uniform":
        values = rng.uniform(-1, 1, shape).astype(np.float32)
    elif kind == "small":
        values *= 1e-5
    elif kind == "sparse":
        values *= rng.random(shape) < 0.05
    elif kind == "cancellation":
        values = np.broadcast_to(np.where(np.arange(shape[-1]) % 2, -1.0, 1.0), shape).copy().astype(np.float32)
    elif kind == "outliers":
        values.flat[::127] *= 64
    elif kind == "labels":
        if weight:
            values.fill(0)
            for index in range(min(shape[-2:])):
                values[..., index, index] = 1
        else:
            # Exact BF16 digits encode rank and row; multi-column IDs avoid the
            # 127-row collisions of a scalar modulo label.
            values = ((np.arange(np.prod(shape)).reshape(shape) + seed) % 127 - 63).astype(np.float32)
            if shape[-1] >= 4:
                rows = np.arange(shape[0])
                values[:, 0] = seed % 127 - 63
                values[:, 1] = rows % 127 - 63
                values[:, 2] = rows // 127 % 127 - 63
                values[:, 3] = rows // (127 * 127) % 127 - 63
    return bf16(values)


def inputs(case, rank, table, *, second=False):
    e = case.local_experts
    rows = case.m if case.operator != "gmm_alltoallv" else int(table[:, rank * e : (rank + 1) * e].sum())
    k, n = (case.n, case.k) if second else (case.k, case.n)
    x = data((rows, k), case.x_data, case.seed + rank * 101)
    w = data((e, k, n), case.w_data, case.seed + rank * 101 + (9001 if second else 5003), True)
    return x, w


def gemm(x, w, block=256):
    result = np.empty((x.shape[0], w.shape[1]), dtype=np.float32)
    weight = w.astype(np.float32)
    for row in range(0, x.shape[0], block):
        result[row : row + block] = x[row : row + block].astype(np.float32) @ weight
    return result


def dispatched(case, rank, table):
    chunks = []
    for expert in range(case.local_experts):
        col = rank * case.local_experts + expert
        for source in range(case.ep_size):
            x = data((case.m, case.k), case.x_data, case.seed + source * 101)
            offset = int(table[source, :col].sum())
            chunks.append(x[offset : offset + int(table[source, col])].copy())
    return np.concatenate(chunks, axis=0)


def grouped(x, w, counts):
    pieces, offset = [], 0
    for expert, count in enumerate(counts):
        count = int(count)
        pieces.append(gemm(x[offset : offset + count], w[expert]))
        offset += count
    assert offset == len(x)
    return np.concatenate(pieces, axis=0)


def reference(case, rank, table):
    e = case.local_experts
    if case.operator == "alltoallv_gmm":
        x = dispatched(case, rank, table)
        _, w = inputs(case, rank, table)
        return grouped(x, w, table[:, rank * e : (rank + 1) * e].sum(axis=0)), x
    chunks = []
    for owner in range(case.ep_size):
        x, w = inputs(case, owner, table)
        counts = table[:, owner * e : (owner + 1) * e].sum(axis=0)
        if case.operator == "composition":
            x = bf16(grouped(dispatched(case, owner, table), w, counts))
            _, w = inputs(case, owner, table, second=True)
        y = grouped(x, w, counts)
        offset = 0
        for expert in range(e):
            col = owner * e + expert
            start = offset + int(table[:rank, col].sum())
            chunks.append(y[start : start + int(table[rank, col])].copy())
            offset += int(counts[expert])
    return np.concatenate(chunks, axis=0), None


def compare(actual, expected):
    from .precision import Comparison

    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return {"passed": False, "reason": "shape mismatch", "actual": list(a.shape), "expected": list(b.shape)}
    check = Comparison()
    check.update(a.reshape(-1, 1), b.reshape(-1, 1), 0)
    report = check.report()
    for error in report["top_errors"]:
        error["index"] = [int(i) for i in np.unravel_index(error["index"][0], a.shape)]
    return report
