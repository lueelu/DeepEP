# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""Deterministic, executable training and directed coverage catalog."""

from collections import Counter
from dataclasses import replace
from functools import lru_cache
import importlib.util
import json
from pathlib import Path

from .cases import CaseSpec, generate

OPS = ("alltoallv_gmm", "gmm_alltoallv")
TRAIN_M = (4096, 8192, 16384, 32768, 65536, 131072, 262144)


@lru_cache(None)
def tiling_module():
    path = Path(__file__).resolve().parents[2] / "python/deep_ep_experimental/ep_gmm_fused/_tiling.py"
    spec = importlib.util.spec_from_file_location("ep_gmm_fused_test_tiling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiling_case(c):
    t = tiling_module()
    return dict(
        operator=t.ALL_TO_ALLV if c.operator == OPS[0] else t.GROUPED_MATMUL,
        shape=dict(m=c.m, k=c.k, n=c.n),
        world_size=c.ep_size,
        expert_num=c.ep_size * c.local_experts,
        output_all_to_allv=c.export,
    )


def validate_profile(c, cores=32):
    t, spec = tiling_module(), tiling_case(c)
    return t.validate_tiling(
        spec["operator"],
        c.tiling,
        m=c.m,
        k=c.k,
        n=c.n,
        world_size=c.ep_size,
        aic_cores=cores,
        output_all_to_allv=c.export,
    )


def profiles():
    data = json.loads((Path(__file__).resolve().parents[1] / "MODEL_CASES.json").read_text(encoding="utf-8"))
    for model in data["models"]:
        for deployment in model["deployments"]:
            if deployment["local_experts"] <= 32:
                yield model, deployment


def model_case(model, deployment, projection, target, route, seed, category):
    q = model["top_k"]
    fields = model["source"]["field_values"]
    return CaseSpec(
        operator=projection["operator"],
        ep_size=deployment["ep_size"],
        local_experts=deployment["local_experts"],
        m=((target + q - 1) // q) * q,
        k=projection["k"],
        n=projection["n"],
        route=route,
        seed=seed,
        scenario=dict(
            category=category,
            model=model["model_id"],
            projection=projection["role"],
            top_k=q,
            target_m=target,
            routing_seed=seed,
            n_group=fields.get("n_group", 1),
            topk_group=fields.get("topk_group", 1),
            config_revision=model["source"]["revision"],
        ),
    )


def model_cases(seed):
    return [
        model_case(model, dep, proj, m, route, seed, "model")
        for model, dep in profiles()
        for proj in dep["projections"]
        for m in TRAIN_M
        for route in ("balanced", "random")
    ]


def directed_cases(seed):
    result = []

    def add(c, category, **options):
        c = replace(c, scenario=dict(category=category, **options))
        result.append(c)
        return c

    for op in OPS:
        for ep in (2, 4, 8):
            base = CaseSpec(operator=op, ep_size=ep, local_experts=4, seed=seed)
            if op == OPS[1] and ep == 2:
                wide = replace(base, local_experts=1, m=128, k=256, n=2176, route="local")
                add(wide, "ub_regression", copy_rows="default")
                profile = tiling_module().default_tiling(tiling_case(wide))
                for rows in (29, 30):
                    add(
                        replace(
                            wide,
                            tiling=dict(profile, local_copy_tile_rows=rows),
                            expected_error="capacity" if rows == 30 else None,
                        ),
                        "ub_regression",
                        copy_rows=rows,
                    )
            for boundary in (16, 32, 64, 128, 256, 1280, 1536):
                for delta in (-1, 0, 1):
                    segment = boundary + delta
                    add(replace(base, m=segment * ep * 4), "segment_boundary", segment_rows=segment, boundary=boundary)
            for k in (255, 256, 257):
                for n in (127, 128, 129):
                    add(replace(base, k=k, n=n, m=1025), "kn_tail")
            for m in (131072, 262144, 524288, 1048576):
                for delta in (-1, 1):
                    add(replace(base, m=m + delta, k=32, n=32), "large_m_boundary", boundary=m)
            for pattern in ("one_empty", "one_receiver"):
                add(
                    replace(base, m=1024),
                    "zero_receive",
                    pattern=pattern,
                    contract="reject_zero_input" if op == OPS[1] else "positive",
                )
            # Capacity neighbors are derived from the real host constraints.
            t = tiling_module()
            spec = tiling_case(base)
            profile = t.default_tiling(spec)
            for resource in ("workspace", "ub"):
                if resource == "ub":
                    field = "local_copy_tile_rows"
                    cap = (t.UB_BYTES - t.UDMA_SCRATCH_BYTES) // (4 * 256)
                elif op == OPS[0]:
                    field = "stage_rows"
                    cap = t.WORKSPACE_BYTES // (2 * ep * base.k * 2) // profile["m0"]
                else:
                    field = "comm_interval"
                    cap = t.WORKSPACE_BYTES // (4 * 32 * profile["m0"] * base.n)
                for delta in (-1, 0, 1):
                    value = (cap + delta) * (profile["m0"] if field == "stage_rows" else 1)
                    candidate = dict(profile, **{field: value})
                    add(
                        replace(base, tiling=candidate, expected_error="capacity" if delta == 1 else None),
                        "capacity",
                        resource=resource,
                        neighbor=delta,
                        aic_cores=32,
                    )
            for family in ("size", "route"):
                for submission in ("checked", "queued"):
                    add(
                        replace(base, m=4096, preallocate=True),
                        "state_reuse",
                        sequence=family,
                        submission=submission,
                        repetitions=2,
                    )
            for delay in ("first", "rotating"):
                add(replace(base, m=1024), "rank_stagger", delay_pattern=delay, delay_seconds=0.1)
    # Same inputs/weights and independent reference for every legal candidate.
    for ep in (2, 4, 8):
        for op, export in ((OPS[0], False), (OPS[0], True), (OPS[1], False)):
            base = CaseSpec(operator=op, ep_size=ep, local_experts=4, m=1024, export=export, seed=seed)
            candidates = tiling_module().candidates_for_case(
                tiling_case(base), max_candidates=16, seed=seed, aic_cores=32
            )
            if len(candidates) != 16:
                raise ValueError("tiling coverage budget unavailable")
            for candidate in candidates:
                add(
                    replace(base, tiling=candidate),
                    "tiling",
                    aic_cores=32,
                    comparison="independent_numeric",
                    accumulation_order="not_proven_identical",
                )
    representative = {
        "Qwen/Qwen3-30B-A3B",
        "openai/gpt-oss-120b",
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
    }
    for model, dep in profiles():
        if model["model_id"] not in representative:
            continue
        for proj in dep["projections"]:
            if proj["role"] == "single_gate_or_up":
                continue
            for category, m in (("large_load", 524288), ("stress", 1048576)):
                for route in ("balanced", "random"):
                    result.append(model_case(model, dep, proj, m, route, seed, category))
            for m in (131072, 262144):
                for ratio in (2,) if dep["ep_size"] == 4 else (2, 4):
                    c = model_case(model, dep, proj, m, "hot_rank", seed, "imbalance")
                    result.append(replace(c, scenario=dict(c.scenario, receive_ratio=ratio)))
    for op in OPS:
        for layout in ("contiguous", "interleaved"):
            for route in ("balanced", "random"):
                add(
                    CaseSpec(operator=op, ep_size=4, m=1024, route=route, seed=seed),
                    "domain_isolation",
                    required_world_size=64,
                    layout=layout,
                )
    return result


def generate_catalog(suite, seed=20260920):
    if suite == "model":
        cases = model_cases(seed)
    elif suite == "generalization":
        smoke_ids = {c.case_id for c in generate("smoke", seed)}
        retained = [c for c in generate("nightly", seed) if c.local_experts <= 32 and c.case_id not in smoke_ids]
        cases = retained + directed_cases(seed)
    elif suite == "all":
        cases = generate("smoke", seed) + generate_catalog("model", seed) + generate_catalog("generalization", seed)
    else:
        raise ValueError(suite)
    if len(cases) != len({c.case_id for c in cases}):
        raise ValueError("duplicate generated case IDs")
    return cases


def category(c):
    return (c.scenario or {}).get("category", "legacy")


def counts(cases):
    return dict(sorted(Counter(category(c) for c in cases).items()))
