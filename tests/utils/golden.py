# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Versioned golden manifest construction for the EP reference matrix."""

from dataclasses import replace
from hashlib import sha256
import json

from .matrix import DEVICE_MATRIX, MATRIX, iter_matrix_cases
from .precision import compare_dual, emulate_upstream_case, quantize_bfloat16
from .reference import dispatch, expected_expert_combine


SCHEMA = "deepep-reference-golden/v2"
UPSTREAM_COMMIT = "a56d6156febcd9976e55adc85b5155bfac9f28f8"


def stable_digest(value):
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("ascii")).hexdigest()


def _upstream_benchmark(case):
    quantized_x = emulate_upstream_case(case)
    quantized_case = replace(case, x=quantized_x)
    values, weights = expected_expert_combine(quantized_case)
    # Pinned upstream tests restore FP8 dispatch to BF16 and use BF16 combine
    # tensors.  Record the observable output dtype, not an ideal FP64 sum.
    values = tuple(tuple(tuple(quantize_bfloat16(value) for value in row) for row in rank) for rank in values)
    return values, weights


def build_manifest():
    records = []
    for spec, case in zip(MATRIX, iter_matrix_cases()):
        ordinary, expanded = dispatch(case), dispatch(case, expanded=True)
        golden_x, golden_weights = expected_expert_combine(case)
        benchmark_x, benchmark_weights = _upstream_benchmark(case)
        report = compare_dual(benchmark_x, benchmark_x, golden_x, dtype=case.dtype, level="L2")
        records.append(
            {
                "name": case.name,
                "config": json.loads(json.dumps(spec)),
                "valid_route_slots": sum(expert >= 0 for rank in case.routes for token in rank for expert in token),
                "ordinary_rows": [len(rows) for rows in ordinary.ranks],
                "expanded_rows": [len(rows) for rows in expanded.ranks],
                "expert_counts_sha256": stable_digest(ordinary.counts),
                "cpu_fp64_combine_sha256": stable_digest((golden_x, golden_weights)),
                "upstream_dtype_combine_sha256": stable_digest((benchmark_x, benchmark_weights)),
                "upstream_dtype_error": report.benchmark.__dict__,
            }
        )
    return {
        "schema": SCHEMA,
        "upstream_commit": UPSTREAM_COMMIT,
        "precision_standard": "Ascend operator precision standard 2.1 / L2",
        "device_qualification_cases": json.loads(json.dumps(DEVICE_MATRIX)),
        "notes": [
            "CPU FP64 mathematical oracle is the golden truth.",
            "Second benchmark emulates the pinned upstream BF16 or per-128 E4M3FN dispatch path and BF16 observable combine output.",
            "Device qualification must replace or supplement emulation with captured upstream-device results using the same inputs.",
            "Device qualification cases require one torchrun execution for each EP world size and are not materialized during CPU manifest checks.",
        ],
        "cases": records,
    }
