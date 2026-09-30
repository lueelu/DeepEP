# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""Versioned, replayable cases shared by correctness, benchmark and profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
from pathlib import Path
import random

ROUTES = ("balanced", "random", "hot_expert", "hot_rank", "local", "remote", "ring", "empty_experts")
DATA = ("normal", "uniform", "zero", "small", "sparse", "cancellation", "outliers", "labels")


@dataclass(frozen=True)
class CaseSpec:
    operator: str = "alltoallv_gmm"
    ep_size: int = 2
    local_experts: int = 2
    m: int = 129
    k: int = 256
    n: int = 256
    route: str = "balanced"
    x_data: str = "normal"
    w_data: str = "normal"
    seed: int = 20260920
    route_device: str = "npu"
    preallocate: bool = False
    export: bool = False
    explicit_received: bool = True
    tiling: dict | None = None
    expected_error: str | None = None
    scenario: dict | None = None

    def __post_init__(self):
        if self.expected_error not in (None, "experts", "zero_input", "dtype", "route_dtype", "route_sum", "capacity"):
            raise ValueError("unknown expected error contract")
        if self.operator not in {"alltoallv_gmm", "gmm_alltoallv", "composition"}:
            raise ValueError("unknown operator")
        if self.ep_size not in (1, 2, 4, 8) or self.local_experts < 1:
            raise ValueError("invalid EP size or local expert count")
        if min(self.m, self.k, self.n) < 1:
            raise ValueError("m/k/n must be positive; zero input is tested as an API rejection")
        limit = 64 if self.operator == "alltoallv_gmm" else 32
        if self.local_experts > limit and self.expected_error != "experts":
            raise ValueError(f"operator supports at most {limit} local experts")
        if self.route not in ROUTES or self.x_data not in DATA or self.w_data not in DATA:
            raise ValueError("unknown route/data distribution")
        if self.route_device not in ("host", "cpu", "npu"):
            raise ValueError("route_device must be host/cpu/npu")
        if self.export and self.operator != "alltoallv_gmm":
            raise ValueError("export is only defined for alltoallv_gmm")

    @property
    def case_id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def record(self):
        return {"case_id": self.case_id, **asdict(self)}


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported case schema")
    cases = [CaseSpec(**{k: v for k, v in item.items() if k != "case_id"}) for item in data["cases"]]
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("case selection is empty or contains duplicates")
    return cases


def save_cases(path, cases):
    Path(path).write_text(
        json.dumps({"schema_version": 1, "cases": [c.record() for c in cases]}, indent=2) + "\n", encoding="utf-8"
    )


def pairwise(factors):
    """Greedily cover every value pair; deterministic tie breaking and no dependency."""
    names = list(factors)
    candidates = list(itertools.product(*(factors[n] for n in names)))

    def pairs(row):
        return {(i, row[i], j, row[j]) for i in range(len(row)) for j in range(i + 1, len(row))}

    remaining = set().union(*(pairs(row) for row in candidates))
    while remaining:
        best = max(candidates, key=lambda row: len(pairs(row) & remaining))
        remaining -= pairs(best)
        yield dict(zip(names, best))


def generate(suite="smoke", seed=20260920, random_count=24):
    if suite in ("model", "generalization", "all"):
        from .catalog import generate_catalog

        return generate_catalog(suite, seed)
    if suite not in ("smoke", "nightly"):
        raise ValueError(f"unknown suite: {suite}")
    base = CaseSpec(seed=seed)
    cases = [
        replace(base, operator=op, ep_size=ep)
        for ep in (1, 2)
        for op in ("alltoallv_gmm", "gmm_alltoallv", "composition")
    ]
    if suite == "smoke":
        # Upstream PR 11602 regressions belong to smoke.
        cases += [
            replace(base, ep_size=8, local_experts=32, m=131072, k=2048, n=n, export=export, preallocate=True)
            for n in (512, 1024)
            for export in (False, True)
        ] + [
            replace(
                base, operator="gmm_alltoallv", ep_size=8, local_experts=32, m=131072, k=k, n=2048, preallocate=True
            )
            for k in (512, 1024)
        ]
    if suite != "smoke":
        for op in ("alltoallv_gmm", "gmm_alltoallv"):
            for ep in (1, 2, 4, 8):
                c = replace(base, operator=op, ep_size=ep)
                cases += [
                    replace(c, m=m)
                    for m in (
                        1,
                        15,
                        16,
                        17,
                        127,
                        128,
                        129,
                        255,
                        256,
                        257,
                        1279,
                        1280,
                        1281,
                        1535,
                        1536,
                        1537,
                        4095,
                        4096,
                        4097,
                        16385,
                    )
                ]
                cases += [
                    replace(c, local_experts=e, m=max(129, ep * e))
                    for e in (
                        (1, 2, 3, 8, 16, 31, 32, 33, 63, 64) if op == "alltoallv_gmm" else (1, 2, 3, 8, 16, 31, 32)
                    )
                ]
                cases += [
                    replace(c, k=k, n=n) for k, n in ((32, 32), (255, 257), (257, 255), (7168, 2048), (2048, 7168))
                ]
                cases += [replace(c, route=r, m=1024) for r in ROUTES]
        factors = {
            "route": ROUTES,
            "x_data": DATA,
            "w_data": DATA,
            "route_device": ("host", "cpu", "npu"),
            "preallocate": (False, True),
        }
        rows = list(pairwise(factors))
        cases += [replace(base, operator=op, **row) for op in ("alltoallv_gmm", "gmm_alltoallv") for row in rows]
        cases += [
            replace(base, export=True, preallocate=p, route_device=r, explicit_received=e)
            for p, r, e in itertools.product((False, True), ("host", "cpu", "npu"), (False, True))
        ]
        cases += [
            replace(base, operator=op, x_data="labels", w_data="labels", route=r, export=op == "alltoallv_gmm")
            for op in ("alltoallv_gmm", "gmm_alltoallv")
            for r in ROUTES
        ]
        cases += [replace(base, operator="composition", ep_size=ep) for ep in (4, 8)]
        rng = random.Random(seed)
        cases += [
            replace(
                base,
                seed=seed + i + 1,
                ep_size=rng.choice((1, 2, 4, 8)),
                operator=rng.choice(("alltoallv_gmm", "gmm_alltoallv", "composition")),
                local_experts=rng.choice((1, 3, 8, 31)),
                m=rng.randint(128, 4097),
                k=rng.choice((32, 128, 256, 1024)),
                n=rng.choice((32, 256, 512)),
                route=rng.choice(ROUTES),
                x_data=rng.choice(DATA),
                w_data=rng.choice(DATA),
            )
            for i in range(random_count)
        ]
        for op, limit in (("alltoallv_gmm", 64), ("gmm_alltoallv", 32)):
            cases.append(replace(base, operator=op, local_experts=limit + 1, expected_error="experts"))
            cases += [
                replace(base, operator=op, expected_error=error)
                for error in ("zero_input", "dtype", "route_dtype", "route_sum")
            ]
    return list({case.case_id: case for case in cases}.values())
