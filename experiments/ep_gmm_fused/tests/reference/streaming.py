# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""Bounded-RAM full-output reference; each source is generated once per check.

Disk-backed BF16-rounded FP32 inputs preserve the legacy RNG stream.
The reference uses independent FP32 matmul and retains FP32 final results.
"""

from contextlib import AbstractContextManager
from pathlib import Path
import tempfile
import time

import numpy as np

from .golden import bf16
from .precision import Comparison
from .routing import token_order


def fill_data(output, kind, seed, weight=False, block_elements=262144):
    """Bitwise equivalent to golden.data without full-size temporary arrays."""
    shape, flat = output.shape, output.reshape(-1)
    rng = np.random.default_rng(seed)
    for start in range(0, flat.size, block_elements):
        stop = min(start + block_elements, flat.size)
        if kind in ("zero", "labels", "cancellation"):
            values = np.zeros(stop - start, dtype=np.float32)
        else:
            values = rng.normal(0, 0.25, stop - start).astype(np.float32)
        flat[start:stop] = values
    for start in range(0, flat.size, block_elements):
        stop = min(start + block_elements, flat.size)
        values = np.array(flat[start:stop])
        indices = np.arange(start, stop)
        if kind == "uniform":
            values = rng.uniform(-1, 1, stop - start).astype(np.float32)
        elif kind == "small":
            values *= 1e-5
        elif kind == "sparse":
            values *= rng.random(stop - start) < 0.05
        elif kind == "outliers":
            values[indices % 127 == 0] *= 64
        elif kind == "cancellation":
            values = np.where(indices % shape[-1] % 2, -1.0, 1.0).astype(np.float32)
        elif kind == "labels":
            if weight:
                values = ((indices // shape[-1]) % shape[-2] == indices % shape[-1]).astype(np.float32)
            else:
                values = ((indices + seed) % 127 - 63).astype(np.float32)
                if shape[-1] >= 4:
                    rows, col = indices // shape[-1], indices % shape[-1]
                    for digit, encoded in enumerate(
                        (
                            np.full_like(rows, seed % 127 - 63),
                            rows % 127 - 63,
                            rows // 127 % 127 - 63,
                            rows // (127 * 127) % 127 - 63,
                        )
                    ):
                        values[col == digit] = encoded[col == digit]
        flat[start:stop] = bf16(values)


class ReferenceStore(AbstractContextManager):
    def __init__(self, case, table, directory=None):
        self.case, self.table = case, table
        self.directory = tempfile.TemporaryDirectory(prefix="ep-gmm-fused-reference-", dir=directory)
        self.arrays = {}
        self.generated_bytes = 0
        self.generation_seconds = 0.0

    def array(self, key, shape, kind, seed, weight=False):
        if key not in self.arrays:
            begin = time.perf_counter()
            path = Path(self.directory.name) / (key + ".npy")
            value = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
            fill_data(value, kind, seed, weight)
            self.arrays[key] = value
            self.generated_bytes += value.nbytes
            self.generation_seconds += time.perf_counter() - begin
        return self.arrays[key]

    def x(self, rank):
        c, e = self.case, self.case.local_experts
        rows = c.m if c.operator != "gmm_alltoallv" else int(self.table[:, rank * e : (rank + 1) * e].sum())
        key = f"x-{rank}"
        if c.operator != "gmm_alltoallv" and (c.scenario or {}).get("top_k"):
            if key not in self.arrays:
                q = c.scenario["top_k"]
                tokens = self.array(f"tokens-{rank}", (c.m // q, c.k), c.x_data, c.seed + rank * 101)
                path = Path(self.directory.name) / (key + ".npy")
                output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, c.k))
                order = token_order(c, rank)
                for begin in range(0, rows, 1024):
                    output[begin : begin + 1024] = tokens[order[begin : begin + 1024]]
                self.arrays[key] = output
                self.generated_bytes += output.nbytes
                self.release(f"tokens-{rank}")
            return self.arrays[key]
        return self.array(key, (rows, c.k), c.x_data, c.seed + rank * 101)

    def w(self, rank, second=False):
        c = self.case
        k, n = (c.n, c.k) if second else (c.k, c.n)
        return self.array(
            f"w-{rank}-{second}",
            (c.local_experts, k, n),
            c.w_data,
            c.seed + rank * 101 + (9001 if second else 5003),
            True,
        )

    def release(self, key):
        array = self.arrays.pop(key, None)
        if array is not None:
            filename = Path(array.filename)
            array._mmap.close()
            filename.unlink()

    def __exit__(self, *args):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()
        self.directory.cleanup()


def segments(case, rank, table, store, block_rows=1024):
    """Yield output offset, X block, expert W, public dispatch block.

    Inverse computes only the destination rank's rows, eliminating EP-fold GEMM.
    No full dispatched tensor or full reference output is materialized.
    """
    e, out = case.local_experts, 0
    if case.operator == "alltoallv_gmm":
        weights = store.w(rank)
        starts = table.cumsum(axis=1) - table
        counts = table[:, rank * e : (rank + 1) * e].sum(axis=0)
        expert_starts = counts.cumsum() - counts
        # Source-major traversal lets us release each source after one pass;
        # explicit output offsets retain expert-major destination semantics.
        for source in range(case.ep_size):
            x = store.x(source)
            for expert in range(e):
                col = rank * e + expert
                out = int(expert_starts[expert] + table[:source, col].sum())
                count, start = int(table[source, col]), int(starts[source, col])
                if not count:
                    continue
                for pos in range(0, count, block_rows):
                    block = x[start + pos : start + min(pos + block_rows, count)]
                    yield out, block, weights[expert], block
                    out += len(block)
            store.release(f"x-{source}")
    elif case.operator == "gmm_alltoallv":
        for owner in range(case.ep_size):
            weights, x, offset = store.w(owner), store.x(owner), 0
            for expert in range(e):
                col = owner * e + expert
                start = offset + int(table[:rank, col].sum())
                count = int(table[rank, col])
                for pos in range(0, count, block_rows):
                    block = x[start + pos : start + min(pos + block_rows, count)]
                    yield out, block, weights[expert], None
                    out += len(block)
                offset += int(table[:, col].sum())
            store.release(f"x-{owner}")
            store.release(f"w-{owner}-False")
    elif case.operator == "composition":
        # Only rows originally sent by this rank contribute to its final output.
        # Both expert matmuls run through the supplied FP32 callback.
        x = store.x(rank)
        for owner in range(case.ep_size):
            first, second = store.w(owner), store.w(owner, second=True)
            for expert in range(e):
                count = int(table[rank, owner * e + expert])
                for pos in range(0, count, block_rows):
                    length = min(block_rows, count - pos)
                    yield out, x[out : out + length], (first[expert], second[expert]), None
                    out += length
            store.release(f"w-{owner}-False")
            store.release(f"w-{owner}-True")
        store.release(f"x-{rank}")
    else:
        raise ValueError("unknown operator")


def check_streaming(case, rank, table, store, actual_block, exported_block=None, block_rows=1024, device_matmul=None):
    if device_matmul is None:
        raise ValueError("an independent FP32 matmul callback is required")
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    check = Comparison()
    exact, labels, rows = True, True, 0
    started = time.perf_counter()
    for offset, x, w, dispatch in segments(case, rank, table, store, block_rows):
        actual = actual_block(offset, len(x))
        if case.operator == "composition":
            middle = device_matmul(x, w[0])
            if middle.dtype != np.float32:
                raise TypeError("reference matmul must return FP32")
            # This BF16 boundary exists between the two public DUT calls.
            y32 = device_matmul(bf16(middle), w[1])
        else:
            y32 = device_matmul(x, w)
        if y32.dtype != np.float32:
            raise TypeError("reference matmul must return FP32")
        check.update(actual, y32, offset)  # Keep final golden unrounded.
        if exported_block is not None:
            exact &= bool(np.array_equal(exported_block(offset, len(x)), dispatch))
        if case.x_data == case.w_data == "labels":
            labels &= bool(np.array_equal(actual, y32))
        rows += len(x)
    expected_rows = (
        int(table[:, rank * case.local_experts : (rank + 1) * case.local_experts].sum())
        if case.operator == "alltoallv_gmm"
        else case.m
    )
    assert rows == expected_rows, "incomplete reference coverage"
    report = dict(
        fp32=check.report(),
        export_exact=exact,
        labels_exact=labels,
        golden_dtype="float32",
        golden_rounding="none",
        checked_rows=rows,
        reference_seconds=time.perf_counter() - started,
        generated_bytes=store.generated_bytes,
        generation_seconds=store.generation_seconds,
    )
    report["passed"] = report["fp32"]["passed"] and exact and labels
    return report
