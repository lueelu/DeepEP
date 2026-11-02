# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared helpers for real-NPU operator benchmarks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist


L2_CACHE_FLUSH_BYTES = 256_000_000


def make_topk_idx(
    rank: int,
    num_ranks: int,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    route: str,
    seed: int = 0,
) -> torch.Tensor:
    """Create repeatable top-k routes for benchmark inputs."""

    if route == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed + rank)
        scores = torch.randn(
            (num_tokens, num_experts),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).abs()
        return torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False).indices.to(torch.int32)

    experts_per_rank = num_experts // num_ranks
    token_ids = torch.arange(num_tokens, dtype=torch.int64, device="cpu").view(-1, 1)
    topk_slots = torch.arange(num_topk, dtype=torch.int64, device="cpu").view(1, -1)
    if route == "balanced":
        target_ranks = (token_ids + topk_slots + rank) % num_ranks
    elif route == "local":
        target_ranks = torch.full((num_tokens, num_topk), rank, dtype=torch.int64, device="cpu")
    elif route == "remote":
        target_ranks = torch.full(
            (num_tokens, num_topk),
            (rank + 1) % num_ranks,
            dtype=torch.int64,
            device="cpu",
        )
    else:
        raise ValueError(f"unsupported route: {route}")
    local_experts = (token_ids + topk_slots) % experts_per_rank
    return (target_ranks * experts_per_rank + local_experts).to(torch.int32)


def destination_count(num_ranks: int, num_topk: int, route: str) -> int:
    """Return the number of distinct destination ranks selected per token."""

    if route == "balanced":
        return min(num_ranks, num_topk)
    return 1


def dispatch_layout_reference(topk_idx: torch.Tensor, num_ranks: int, num_experts: int):
    """Calculate layout outputs without using the measured kernels."""

    routes = topk_idx.to(device="cpu", dtype=torch.int64)
    experts_per_rank = num_experts // num_ranks
    rank_idx = routes // experts_per_rank
    num_tokens_per_rank = torch.stack([(rank_idx == rank).any(dim=1).sum() for rank in range(num_ranks)]).to(
        torch.int32
    )
    num_tokens_per_expert = torch.bincount(routes.flatten(), minlength=num_experts).to(torch.int32)
    is_token_in_rank = torch.stack([(rank_idx == rank).any(dim=1) for rank in range(num_ranks)], dim=1)
    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def percentile(values: List[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""

    if not values:
        raise ValueError("values must not be empty")
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_times(
    name: str,
    times_ms: List[float],
    *,
    num_ranks: int,
    num_tokens_per_rank: int,
    logical_bytes_per_rank: int = 0,
) -> Dict[str, Any]:
    """Build stable latency, throughput and per-rank bandwidth metrics."""

    if not times_ms:
        raise ValueError("times_ms must not be empty")
    mean_ms = sum(times_ms) / len(times_ms)
    mean_seconds = mean_ms / 1e3
    result: Dict[str, Any] = {
        "op": name,
        "iterations": len(times_ms),
        "mean_us": mean_ms * 1e3,
        "min_us": min(times_ms) * 1e3,
        "p50_us": percentile(times_ms, 0.50) * 1e3,
        "p95_us": percentile(times_ms, 0.95) * 1e3,
        "max_us": max(times_ms) * 1e3,
        "aggregate_tokens_per_second": (num_tokens_per_rank * num_ranks / mean_seconds),
    }
    if logical_bytes_per_rank > 0:
        result["per_rank_logical_payload_gbps"] = logical_bytes_per_rank / mean_seconds / 1e9
    return result


def _release_events(value: Any) -> None:
    if value is None:
        return
    if hasattr(value, "event") and hasattr(value, "current_stream_wait"):
        if value.event is not None:
            value.current_stream_wait(release_handle=True)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _release_events(item)


def flush_l2_cache(cache: Optional[torch.Tensor], stream: Any) -> None:
    """Evict device L2 contents by overwriting a 256 MB scratch tensor."""

    if cache is None:
        return
    with torch.npu.stream(stream):
        cache.zero_()


def benchmark_operator(
    name: str,
    fn: Callable[[], Any],
    *,
    buffer: Any,
    timing_group: Any = None,
    num_warmups: int,
    num_iterations: int,
    num_tokens_per_rank: int,
    logical_bytes_per_rank: int = 0,
    flush_l2: bool = True,
    max_rank_latency: bool = False,
) -> Dict[str, Any]:
    """Measure work enqueued on the Buffer communication stream.

    Each rank records timing-enabled NPU events on the same communication
    stream used by the operator. By default each rank keeps its local samples,
    matching DeepEP's benchmark behavior; ``max_rank_latency`` enables the
    previous elementwise MAX reduction across ranks.
    """

    if num_warmups < 0:
        raise ValueError("num_warmups must be greater than or equal to zero")
    if num_iterations < 2:
        raise ValueError("num_iterations must be at least two")

    comm_stream = buffer.get_comm_stream()
    l2_flush_cache = None
    if flush_l2:
        l2_flush_cache = torch.empty(
            L2_CACHE_FLUSH_BYTES // 4,
            dtype=torch.int32,
            device=f"npu:{torch.npu.current_device()}",
        )
    torch.npu.synchronize()
    for _ in range(num_warmups):
        _release_events(fn())

    start_events = [torch.npu.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(num_iterations)]
    for index in range(num_iterations):
        flush_l2_cache(l2_flush_cache, comm_stream)
        start_events[index].record(comm_stream)
        outputs = fn()
        end_events[index].record(comm_stream)
        _release_events(outputs)
    torch.npu.synchronize()

    # DeepEP intentionally discards the first measured sample. It is often
    # affected by lazy runtime work that remains after the warmup lifecycle.
    local_times_ms = [float(start.elapsed_time(end)) for start, end in zip(start_events, end_events)][1:]

    reported_times_ms = local_times_ms
    if max_rank_latency:
        reduction_group = timing_group if timing_group is not None else buffer.group
        reduction_backend = dist.get_backend(reduction_group)
        max_times = torch.tensor(local_times_ms, dtype=torch.float32)
        if reduction_backend == "hccl":
            max_times = max_times.to(device=f"npu:{torch.npu.current_device()}")
        dist.all_reduce(
            max_times,
            op=dist.ReduceOp.MAX,
            group=reduction_group,
        )
        reported_times_ms = [float(value) for value in max_times.cpu().tolist()]
    return summarize_times(
        name,
        reported_times_ms,
        num_ranks=buffer.group_size,
        num_tokens_per_rank=num_tokens_per_rank,
        logical_bytes_per_rank=logical_bytes_per_rank,
    )


def write_json(path: Optional[str], document: Dict[str, Any]) -> None:
    """Write one benchmark report when an output path is requested."""

    if path is None:
        return
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
