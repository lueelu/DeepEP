# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by zhu-mingzhe71 2026
"""Explicit reuse of a tiling measured on the deployment cluster."""

import json
from pathlib import Path

from ._tiling import validate_tiling


def load_tiling_config(path, case_id):
    """Load one successful case from ``best_tilings.json`` for ``tiling=``.

    Load once outside the hot loop and pass the same mapping on every rank. The
    case id is explicit because routing, export mode and deployment topology can
    change the best configuration even when tensor dimensions are identical.
    A reusable recommendation must have passed both performance collection and
    accuracy validation on its recorded workload.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("tiling results require schema_version=1")
    cases = payload.get("cases")
    if not isinstance(cases, list) or any(not isinstance(item, dict) for item in cases):
        raise ValueError("tiling results must contain a cases list")
    matches = [item for item in cases if item.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError("expected exactly one tiling result for case_id={!r}".format(case_id))
    item = matches[0]
    if item.get("passed") is not True or item.get("valid") is not True or not item.get("tiling"):
        raise ValueError("case {!r} has no successfully measured best tiling".format(case_id))
    if item.get("accuracy_checked") is not True:
        raise ValueError("best tiling must be accuracy checked before reuse")
    if not isinstance(item.get("shape"), dict) or any(
        isinstance(item["shape"].get(axis), bool)
        or not isinstance(item["shape"].get(axis), int)
        or item["shape"][axis] <= 0
        for axis in ("m", "n", "k")
    ):
        raise ValueError("tiling result is missing the measured shape")
    if item.get("world_size") is None or not isinstance(item.get("output_all_to_allv"), bool):
        raise ValueError("tiling result is missing world_size or export mode")
    return validate_tiling(
        item.get("operator"),
        item["tiling"],
        output_all_to_allv=item.get("output_all_to_allv", False),
        m=item["shape"]["m"],
        n=item["shape"]["n"],
        k=item["shape"]["k"],
        world_size=item.get("world_size"),
    )
