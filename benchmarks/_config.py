# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Strict, framework-free workload configuration and logical byte accounting."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path

MAX_CONFIG_BYTES = 65536


@dataclass(frozen=True)
class Config:
    api: str = "elastic"
    operation: str = "dispatch"
    dtype: str = "bf16"  # Dispatch input; combine always receives BF16.
    layout: str = "expanded"
    dispatch_mode: str = "fresh"
    world_size: int = 2
    tokens_per_rank: tuple = (16, 15)
    hidden: int = 256
    topk: int = 2
    experts: int = 8
    alignment: int = 1
    with_weights: bool = True
    route: str = "balanced"
    seed: int = 0
    warmup: int = 10
    iterations: int = 50

    def validate(self):
        choices = {
            "api": ("legacy", "elastic"),
            "operation": ("dispatch", "combine", "roundtrip"),
            "dtype": ("bf16", "fp8"),
            "layout": ("ordinary", "expanded"),
            "dispatch_mode": ("fresh", "cached"),
            "route": ("balanced", "skew", "masked", "duplicate"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        limits = {
            "world_size": (1, 1024),
            "hidden": (1, 65536),
            "topk": (1, 256),
            "experts": (1, 65536),
            "seed": (0, 2**31 - 1),
            "warmup": (0, 10000),
            "iterations": (1, 10000),
            "alignment": (1, 128),
        }
        for name, (minimum, maximum) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if type(self.with_weights) is not bool:
            raise ValueError("with_weights must be boolean")
        if not isinstance(self.tokens_per_rank, (list, tuple)):
            raise ValueError("tokens_per_rank must be an array")
        if len(self.tokens_per_rank) != self.world_size or any(
            type(t) is not int or not 0 <= t <= 1048576 for t in self.tokens_per_rank
        ):
            raise ValueError("tokens_per_rank must contain one nonnegative integer per rank")
        if not 1 <= sum(self.tokens_per_rank) <= 1048576:
            raise ValueError("total tokens must be in [1, 1048576]")
        if sum(self.tokens_per_rank) * self.topk > 8388608:
            raise ValueError("planning supports at most 8388608 route slots")
        if self.world_size * self.iterations > 1000000:
            raise ValueError("at most 1000000 rank samples can be retained")
        if self.experts % self.world_size or self.topk > self.experts:
            raise ValueError("experts must divide across ranks and topk must not exceed experts")
        if self.alignment not in (1, 128):
            raise ValueError("alignment must be 1 or 128")
        if self.api == "legacy" and (self.layout != "ordinary" or self.dtype != "bf16" or self.alignment != 1):
            raise ValueError("legacy benchmark currently covers ordinary BF16 with alignment=1")
        if self.dtype == "fp8" and self.hidden % 128:
            raise ValueError("FP8 per-128-column scaling requires hidden divisible by 128")
        if self.api == "elastic" and self.operation != "dispatch" and self.hidden % 256:
            raise ValueError("pinned upstream V2 combine requires hidden divisible by 256")
        if self.operation == "combine" and self.dispatch_mode == "cached":
            raise ValueError("dispatch_mode=cached applies to dispatch or roundtrip")
        return self

    def to_dict(self):
        result = asdict(self)
        result["tokens_per_rank"] = list(self.tokens_per_rank)
        return result

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("configuration must be a JSON object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("unknown configuration fields: " + ", ".join(sorted(unknown)))
        data = dict(data)
        if isinstance(data.get("tokens_per_rank"), list):
            data["tokens_per_rank"] = tuple(data["tokens_per_rank"])
        return cls(**data).validate()

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def required_capabilities(self):
        required = {self.api, "dispatch", self.layout, self.dtype}
        if self.operation != "dispatch":
            required.add("combine")
        if self.operation == "roundtrip":
            required.add("roundtrip")
        if self.dispatch_mode == "cached":
            required.add("cached")
        if self.alignment != 1:
            required.add("alignment128")
        if len(set(self.tokens_per_rank)) > 1:
            required.add("unequal_tokens")
        if 0 in self.tokens_per_rank:
            required.add("zero_tokens")
        if self.world_size == 1:
            required.add("single_rank")
        if self.with_weights:
            required.add("weights")
        required.add("route_" + self.route)
        return required


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_config(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("configuration exceeds 64 KiB")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("invalid configuration JSON") from error


def route_for_token(config, rank, token):
    """Stable integer formula, shared with native adapters; duplicates stay slots."""
    indices = [(config.seed + rank * 17 + token * 7 + slot) % config.experts for slot in range(config.topk)]
    if config.route == "skew":
        indices = [e % (config.experts // config.world_size) for e in indices]
    elif config.route == "duplicate" and config.topk > 1:
        indices[1] = indices[0]
    elif config.route == "masked":
        indices = [-1 if (config.seed + rank + token + slot) % 5 == 0 else e for slot, e in enumerate(indices)]
    return tuple(indices)


def hidden_value(config, rank, token, column):
    """Exactly BF16-representable input, before optional per-128 FP8 encoding."""
    return ((config.seed + rank * 13 + token * 7 + column) % 127 - 63) / 16.0


def weight_value(config, rank, token, slot):
    return ((config.seed + rank * 19 + token * 7 + slot * 3) % 127) / 128.0


def traffic(config):
    """Useful payload bytes, aggregated over all ranks; NOT physical traffic.

    ordinary sends one row to each distinct destination rank; expanded counts
    every valid route slot. Includes local routes; excludes all metadata,
    scales, padding and route values. FP8 dispatch and BF16 combine differ.
    """
    config.validate()
    rows = [0] * config.world_size
    local_experts = config.experts // config.world_size
    for rank, tokens in enumerate(config.tokens_per_rank):
        for token in range(tokens):
            destinations = [e // local_experts for e in route_for_token(config, rank, token) if e >= 0]
            rows[rank] += len(set(destinations)) if config.layout == "ordinary" else len(destinations)
    dispatch = [n * config.hidden * (1 if config.dtype == "fp8" else 2) for n in rows]
    combine = [n * config.hidden * 2 for n in rows]
    measured = (
        dispatch
        if config.operation == "dispatch"
        else (combine if config.operation == "combine" else [a + b for a, b in zip(dispatch, combine)])
    )
    return {
        "definition": "aggregate_useful_hidden_payload_bytes_v1",
        "includes": ["local_routes", "valid_route_multiplicity"],
        "excludes": [
            "padding",
            "fp8_scales",
            "indices",
            "weights",
            "protocol",
            "link_retransmission",
        ],
        "rows_per_source_rank": rows,
        "dispatch_bytes_per_source_rank": dispatch,
        "combine_bytes_per_source_rank": combine,
        "effective_bytes": sum(measured),
    }
