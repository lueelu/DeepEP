# Copyright (c) 2026, Lu Lu

"""Installed-wheel Combine precision test; payload Dispatch is explicitly opt-in.

NPU: python tests/elastic/test_combine.py --ep-size 2 --nproc-per-node 2 --output /tmp/combine-check
"""

import copy
from contextlib import nullcontext
import json
import math
from pathlib import Path
import re
import runpy
import pytest
import importlib.util
import shutil
import subprocess
import sys
import argparse
from types import ModuleType, SimpleNamespace
import hashlib
import time
import os


_notify = runpy.run_path(str(Path(__file__).with_name("test_notify.py")))
_combine_metrics = runpy.run_path(str(Path(__file__).with_name("combine_metrics.py")))
_combine_chunks = runpy.run_path(str(Path(__file__).with_name("combine_chunks.py")))
_combine_peers = runpy.run_path(str(Path(__file__).with_name("combine_peers.py")))
_combine_cases = runpy.run_path(str(Path(__file__).with_name("combine_cases.py")))
_ROUTES = ("normal", "random", "duplicates", "invalid", "empty", "local_duplicates", "local_experts")


def make_combine_routes(world, tokens, args, mode):
    if mode in ("hotspot", "rotating_hotspot"):
        local, lanes = args.experts // world, min(world, 8)
        return [
            [
                [
                    (
                        ((source + token // 7) % ((world + 7) // 8) * 8 if mode == "rotating_hotspot" else 0)
                        + (slot + (token // 3 if mode == "rotating_hotspot" else 0)) % lanes
                    )
                    * local
                    + slot // lanes
                    for slot in range(args.topk)
                ]
                for token in range(tokens)
            ]
            for source in range(world)
        ]
    routes = _notify["make_routes"](
        world,
        tokens,
        args.topk,
        args.experts,
        0,
        mode,
        seed=args.seed,
        without_replacement=getattr(args, "without_replacement", False),
    )
    if getattr(args, "local_expert_first", False):
        local_experts = args.experts // world
        for source, rows in enumerate(routes):
            first, end = source * local_experts, (source + 1) * local_experts
            for row in rows:
                # Stable partition: other servers retain their first-hit gateway.
                row.sort(key=lambda expert: not first <= expert < end)
    return routes


@pytest.mark.parametrize("world", [8, 16, 32, 64, 128, 256, 512])
def test_reuse_capacity_selection_keeps_full_layout(world):
    from deep_ep.buffers._combine_reuse import select_window
    from deep_ep.buffers._elastic_layout import combine_layout

    key = (world, 16384, 6, world * 16384, world * 16384 * 5, 256, 7168)
    full = combine_layout(*key)
    assert select_window(key, full["workspace_bytes"]) == (0, full)
    assert select_window(key, full["workspace_bytes"] + (1 << 30)) == (0, full)
    for window in (1, 7, 63, 256):
        bounded = combine_layout(world, 16384, 6, window, window * 6, 256, 7168)
        selected, actual = select_window(key, bounded["workspace_bytes"])
        assert actual["workspace_bytes"] <= bounded["workspace_bytes"]
        assert selected >= window and selected <= 256
        assert select_window(key, full["workspace_bytes"], window) == (window, bounded)
    with pytest.raises(ValueError, match="fixed buffers"):
        select_window(key, 1)


def test_reuse_profile_does_not_require_the_normal_prepare_kernel(tmp_path):
    (tmp_path / "kernel_details.csv").write_text(
        "Name,Duration(us)\ncombine_reuse_kernel,123\ncombine_reuse_kernel,127\n",
        encoding="utf-8",
    )
    result = _notify["read_kernel_summary"](tmp_path, "combine", 2, combine_reuse=True)
    assert result["durations_us"] == [123, 127] and "prepare" not in result
    with pytest.raises(ValueError):
        _notify["read_kernel_summary"](tmp_path, "combine", 2)


@pytest.mark.parametrize("world", [8, 16, 32])
@pytest.mark.parametrize("mode", ["random", "hotspot", "rotating_hotspot", "duplicates", "invalid", "empty"])
def test_reuse_periods_preserve_every_contribution(world, mode):
    torch = pytest.importorskip("torch")
    from deep_ep.buffers._combine_reuse import build_plan

    tokens, topk = 17, 6
    args = SimpleNamespace(topk=topk, experts=384, seed=2026, without_replacement=True)
    routes = make_combine_routes(world, tokens, args, mode)
    plans = _notify["build_notify_golden"](routes, args.experts, 256)["plans"]
    handles = []
    for rank, plan in enumerate(plans):
        forward = torch.full((world, max(1, max(plan["forward_counts"])), 6), -1, dtype=torch.int32)
        backward = torch.full((min(world, 8), max(1, max(plan["backward_counts"])), 3), -1, dtype=torch.int64)
        for source, rows in enumerate(plan["forward_list"]):
            if rows:
                forward[source, : len(rows)] = torch.tensor(rows, dtype=torch.int32)
        for lane, rows in enumerate(plan["backward_list"]):
            if rows:
                backward[lane, : len(rows)] = torch.tensor(rows, dtype=torch.int64)
        entries = _notify["expert_rows"](routes, args.experts, rank)
        handles.append(
            SimpleNamespace(
                forward_list=forward,
                forward_counts=torch.tensor(plan["forward_counts"]),
                backward_list=backward,
                backward_counts=torch.tensor(plan["backward_counts"]),
                topk_idx=torch.tensor(routes[rank]),
                server_mask=torch.tensor(plan["server_mask"]),
                weight_return_meta=torch.tensor(
                    [(s * tokens + t) * topk + k for s, t, k in entries], dtype=torch.int32
                ),
            )
        )
    for window in (1, 3, 8, 17):
        sent = {}
        for rank, handle in enumerate(handles):
            packed, offsets = build_plan(handle, window)
            original = handle.backward_list.clone()
            assert build_plan(handle, window)[0] is packed
            for lane in range(min(world, 8)):
                rows = packed[offsets[lane] : offsets[lane + 1]].tolist()
                assert [r[2] >> 32 for r in rows] == sorted(r[2] >> 32 for r in rows)
                for _, expert, address in rows:
                    key = (rank // 8 * 8 + lane, address >> 32, address & 0xFFFFFFFF)
                    assert key not in sent
                    sent[key] = int(handle.weight_return_meta[expert])
            assert torch.equal(original, handle.backward_list)
        received = set()
        for rank, handle in enumerate(handles):
            partial = set()
            for source, count in enumerate(handle.forward_counts.tolist()):
                rows = handle.forward_list[source, :count].tolist()
                task = 0
                while task < count:
                    end, token, _, _, _, _ = rows[task]
                    epoch = token // window * world + source + 1
                    key = (epoch, token % window)
                    assert key not in partial
                    partial.add(key)
                    for _, _, _, peer, expert, gather in rows[task:end]:
                        if gather >= 0:
                            key = (rank, epoch, gather % (window * topk))
                            assert key not in received
                            assert sent[key] == int(handles[peer].weight_return_meta[expert])
                            received.add(key)
                    task = end
        assert received == set(sent)


@pytest.mark.parametrize("world,experts", [(8, 384), (16, 384), (32, 512), (128, 384), (256, 512)])
def test_without_replacement_golden_and_local_priority(world, experts):
    dispatch = runpy.run_path(str(Path(__file__).with_name("test_dispatch.py")))
    args = SimpleNamespace(topk=6, experts=experts, seed=2026, without_replacement=True, local_expert_first=False)
    routes = make_combine_routes(world, 3, args, "random")
    assert routes == dispatch["make_routes"](world, 3, 6, experts, 0, "random", seed=2026, without_replacement=True)
    args.local_expert_first = True
    ordered = make_combine_routes(world, 3, args, "random")
    local = experts // world
    for source, rows in enumerate(ordered):
        for token, row in enumerate(rows):
            original = routes[source][token]
            assert row == [e for e in original if e // local == source] + [e for e in original if e // local != source]
    golden = _notify["build_notify_golden"](ordered, experts, 256)
    assert sum(golden["rank_rows"]) == world * 3 * 6
    for rank in (0, world - 1):
        reference = build_reference(ordered, experts, rank)
        assert reference["rank_rows"] == golden["rank_rows"]
        assert reference["gather_rows"] == golden["gather_rows"]
        assert reference["weight_return_meta"] == [
            (source * 3 + token) * 6 + slot for source, token, slot in _notify["expert_rows"](ordered, experts, rank)
        ]


def profile_combine(args, rank, scenario, buffer, expert, invoke, validate, info, log):
    """Clean profiler capture, then one separately validated instrumented call."""
    import torch

    api_events = []
    if getattr(args, "profile_api_time", False):
        call = invoke

        def invoke():
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record(buffer._runtime.stream)
            result = call()
            end.record(buffer._runtime.stream)
            api_events.append((start, end))
            return result

    if getattr(args, "profile_flush_l2", False):
        invoke = _notify["with_l2_flush"](args, invoke, buffer._runtime.stream)

    chunk_tokens = getattr(args, "chunk_tokens", 256)
    old = getattr(buffer, "_combine_diagnostics", None)
    try:
        buffer._combine_diagnostics = None
        clean = _notify["profile_operator"](args, rank, scenario, invoke, validate, info, log)
        if api_events:
            assert len(api_events) == args.profile_warmup + args.profile_iters
            durations = [a.elapsed_time(b) * 1000 for a, b in api_events[args.profile_warmup :]]
            timing = dict(
                rank=rank,
                tokens=args.tokens,
                chunk_tokens=args.chunk_tokens,
                input_storage=getattr(args, "input_storage", "gm"),
                flush_l2=getattr(args, "profile_flush_l2", False),
                durations_us=durations,
                kernel_durations_us=clean["durations_us"],
                precision_checked=True,
                scope="communication stream: entry barrier, optional input copy, prepare, kernel, exit barrier, output copy",
                excludes="L2 flush and Notify",
            )
            (args.output / f"rank-{rank}.api-timings.json").write_text(json.dumps(timing) + "\n", encoding="utf-8")
        if getattr(buffer, "_combine_last_reuse_tokens", 0):
            if clean["kernel"] != "combine_reuse_kernel":
                raise RuntimeError("Reuse profiling selected the wrong kernel")
            return dict(clean_profile=clean, reuse_tokens=buffer._combine_last_reuse_tokens, precision_checked=True)
        kernel = "combine_ep8_kernel" if getattr(args, "ep8_rank_pipeline", False) else "combine_kernel"
        profile_kernel = kernel.replace("_kernel", "_profile_kernel")
        if clean["kernel"] != kernel:
            raise RuntimeError("Combine performance must use the uninstrumented kernel")
        # The summary is fixed-size; chunk GM records scale with cores/groups/chunks.
        # Allocation, readback and export are outside clean profiling.
        profile = torch.zeros((64, _combine_metrics["WORDS"]), dtype=torch.int64, device=expert.device)
        sample = 1
        chunk_profile = torch.zeros(
            _combine_chunks["shape"](args.ep_size, args.tokens, chunk_tokens), dtype=torch.int64, device=expert.device
        )
        peer_profile = torch.zeros(
            _combine_peers["shape"](args.ep_size, args.tokens, chunk_tokens), dtype=torch.int64, device=expert.device
        )
        buffer._combine_diagnostics = dict(profile=profile, sample=sample, chunks=chunk_profile, peers=peer_profile)
        log(
            f"diagnostic.begin kernel={profile_kernel} buffer_bytes="
            f"{64 * _combine_metrics['WORDS'] * 8 + (chunk_profile.numel() + peer_profile.numel()) * 8}"
        )
        result = invoke()
        torch.npu.synchronize()
        log("diagnostic.synchronize.done")
        validate(result)
        cores = _combine_metrics["decode_profile"](
            profile.cpu().tolist(),
            rank=rank,
            world=args.ep_size,
            tokens=args.tokens,
            topk=args.topk,
            weights=not args.without_weights,
            hidden=getattr(args, "hidden", 7168),
            sample=sample,
            chunk_tokens=chunk_tokens,
            ep8_rank_pipeline=getattr(args, "ep8_rank_pipeline", False),
        )
        events = _combine_chunks["decode"](
            chunk_profile.cpu().tolist(),
            rank=rank,
            world=args.ep_size,
            tokens=args.tokens,
            sample=sample,
            chunk_tokens=chunk_tokens,
            ep8_rank_pipeline=getattr(args, "ep8_rank_pipeline", False),
        )
        directory = args.output / "diagnostics" / scenario
        directory.mkdir(parents=True, exist_ok=True)
        peer_events = _combine_peers["decode"](
            peer_profile.cpu().tolist(),
            rank=rank,
            world=args.ep_size,
            tokens=args.tokens,
            sample=sample,
            chunk_tokens=chunk_tokens,
        )
        peer_artifacts = _combine_peers["write_trace"](
            directory / f"rank-{rank}.peers.json",
            events=peer_events,
            rank=rank,
            world=args.ep_size,
            tokens=args.tokens,
            sample=sample,
            chunk_tokens=chunk_tokens,
            origin=min(c["raw_words"][_combine_metrics["FIELDS"].index("Begin")] for c in cores),
            kernel=profile_kernel,
        )
        chunk_path = directory / f"rank-{rank}.chunks.json"
        chunk_trace = dict(
            local_expert_first=getattr(args, "local_expert_first", False),
            deduplication="rank" if getattr(args, "no_server_dedup", False) else "server",
            schema_version=_combine_chunks["VERSION"],
            rank=rank,
            world_size=args.ep_size,
            tokens=args.tokens,
            topk=args.topk,
            hidden=getattr(args, "hidden", 7168),
            sample=sample,
            chunk_tokens=chunk_tokens,
            with_weights=not args.without_weights,
            kernel=profile_kernel,
            precision_checked=True,
            source_sha256=info["source_sha256"],
            ticks_per_us=1000,
            note=_combine_chunks["NOTE"],
            events=events,
        )
        temporary = chunk_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(chunk_trace, separators=(",", ":")) + "\n", encoding="utf-8")
        temporary.replace(chunk_path)
        _combine_chunks["write_timeline"](chunk_path, chunk_trace)
        diagnostic = dict(
            local_expert_first=getattr(args, "local_expert_first", False),
            deduplication="rank" if getattr(args, "no_server_dedup", False) else "server",
            traffic=getattr(args, "combine_traffic", {}),
            peer_profile=peer_artifacts,
            chunk_profile=dict(
                schema_version=_combine_chunks["VERSION"],
                events=len(events),
                file=str(chunk_path),
                svg=str(chunk_path.with_suffix(".svg")),
                csv=str(chunk_path.with_suffix(".csv")),
                perfetto=str(chunk_path.with_suffix(".perfetto.json")),
            ),
            schema_version=_combine_metrics["VERSION"],
            ticks_per_us=_combine_metrics["TICKS_PER_US"],
            rank=rank,
            world_size=args.ep_size,
            tokens=args.tokens,
            topk=args.topk,
            experts=args.experts,
            hidden=getattr(args, "hidden", 7168),
            scenario=scenario,
            seed=args.seed,
            with_weights=not args.without_weights,
            sample=sample,
            chunk_tokens=chunk_tokens,
            kernel=profile_kernel,
            precision_checked=True,
            source_sha256=info["source_sha256"],
            cores=cores,
            note=_combine_metrics["NOTE"] + " " + _combine_chunks["NOTE"],
            clean_profile=dict(kernel=clean["kernel"], samples=clean["samples"], avg_us=clean["avg_us"]),
        )
        # Also save immediately: later repeated-call checks or teardown can fail.
        directory = args.output / "diagnostics" / scenario
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank-{rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        log(f"diagnostic.done precision_checked=1 output={path} perfetto={diagnostic['chunk_profile']['perfetto']}")
        return diagnostic
    finally:
        buffer._combine_diagnostics = old


def build_reference(routes, experts, rank, chunk_tokens=256, *, server_dedup=True):
    source = Path(__file__).with_name("combine_reference.py")
    spec = importlib.util.spec_from_file_location("_deepep_combine_reference", source)
    planner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = planner
    spec.loader.exec_module(planner)
    # Replace only its synthetic route generator, not metadata construction.
    # The reference represents every invalid expert with -1.
    planner.route_experts = lambda source, token, case: [
        expert if 0 <= expert < experts else -1 for expert in routes[source][token]
    ]
    world, tokens, topk = len(routes), len(routes[0]), len(routes[0][0])
    case = planner.CombineCase(world, tokens, topk, experts, chunk_tokens=chunk_tokens, server_dedup=server_dedup)
    plan = planner.build_plan(case, rank)
    result = {
        "forward_list": [list(plan.forward_rows(source)) for source in range(world)],
        "backward_list": [list(plan.backward_rows(lane)) for lane in range(plan.lanes)],
        "forward_counts": list(plan.forward_count),
        "backward_counts": list(plan.backward_count),
        "chunk_ranges": list(plan.chunk_ranges),
        "chunk_masks": list(plan.chunk_masks),
        "server_mask": list(plan.server_mask),
        "rank_rows": list(plan.rank_rows),
        "gather_rows": list(plan.gather_rows),
        "weight_return_meta": list(plan.route_ids),
    }
    if not server_dedup:
        servers, local_experts = max((world + 7) // 8, topk), experts // world
        for source, rows in enumerate(result["forward_list"]):
            for index, row in enumerate(rows):
                token = row[1]
                peers = list(dict.fromkeys(e // local_experts for e in routes[source][token] if 0 <= e < experts))
                flat_row = token * servers + peers.index(rank)
                rows[index] = (row[0], -flat_row - 1, *row[2:])
    return result


def install_rank_return_plan(handle, reference):
    """Test-only replacement after real Notify validation, before the first Combine."""
    import torch

    if any(reference["backward_counts"]) or any(reference["gather_rows"]) or any(reference["chunk_masks"]):
        raise ValueError("Rank return plans must not contain remote gather tasks")
    for name in ("forward_counts", "backward_counts", "server_mask", "chunk_ranges", "chunk_masks", "gather_rows"):
        target = getattr(handle, name)
        target.copy_(torch.tensor(reference[name], dtype=target.dtype, device=target.device).reshape(target.shape))
    for peer, rows in enumerate(reference["forward_list"]):
        if rows:
            target = handle.forward_list[peer, : len(rows)]
            target.copy_(torch.tensor(rows, dtype=target.dtype, device=target.device))
    # Keep the original allocation/strides so both variants reserve the same SHMEM.
    handle._combine_layout_cache = None
    handle._rank_return_plan = True
    handle._routing_versions = {
        name: (getattr(handle, name).data_ptr(), getattr(handle, name)._version) for name in handle._routing_versions
    }


def combine_traffic(reference, rank, hidden):
    counts = dict(self_return_rows=0, same_server_return_rows=0, cross_server_return_rows=0)
    for source, tasks in enumerate(reference["forward_list"]):
        key = (
            "self_return_rows"
            if source == rank
            else "same_server_return_rows"
            if source // 8 == rank // 8
            else "cross_server_return_rows"
        )
        cursor = 0
        while cursor < len(tasks):
            counts[key] += 1
            end = tasks[cursor][0]
            if not cursor < end <= len(tasks):
                raise ValueError("Invalid forward run boundary")
            cursor = end
    counts["gather_rows"] = sum(reference["backward_counts"])
    counts["source_partial_rows"] = sum((mask & ((1 << 64) - 1)).bit_count() for mask in reference["server_mask"])
    counts["cross_server_payload_bytes"] = counts["cross_server_return_rows"] * hidden * 2
    counts["gather_payload_bytes"] = counts["gather_rows"] * hidden * 2
    return counts


def assert_metadata_equal(actual, expected, label):
    """Report the first differing coordinate, including packed chunk markers."""
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{label}: length actual={len(actual)}, expected={len(expected)}")
        for index, (value, reference) in enumerate(zip(actual, expected)):
            assert_metadata_equal(value, reference, f"{label}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def check_handle(handle, reference, rank):
    if handle.metadata_only or handle.metadata_version != 7:
        raise AssertionError(f"rank {rank}: expected payload dispatch metadata v7")
    # Check counts first, then copy/compare only valid rows of padded lists.
    for name in (
        "forward_counts",
        "backward_counts",
        "server_mask",
        "rank_rows",
        "gather_rows",
        "chunk_ranges",
        "chunk_masks",
    ):
        actual = getattr(handle, name).detach().cpu().reshape(-1).tolist()
        assert_metadata_equal(actual, reference[name], f"rank {rank}: {name}")
    for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
        for lane, count in enumerate(reference[counts]):
            actual = getattr(handle, name)[lane, :count].detach().cpu().tolist()
            assert_metadata_equal(actual, reference[name][lane], f"rank {rank}: {name}[{lane}]")
    assert_metadata_equal(
        handle.weight_return_meta.detach().cpu().tolist(),
        reference["weight_return_meta"],
        f"rank {rank}: weight_return_meta",
    )


def payload(source, tokens, generation, hidden=7168):
    import torch

    return ((torch.arange(tokens)[:, None] + torch.arange(hidden)[None, :] + source + generation) % 7 - 3).to(
        torch.bfloat16
    )


def expected_output(routes, experts, rank, generation, hidden=7168):
    import torch

    base = payload(rank, len(routes[rank]), generation, hidden).float()
    selected = torch.tensor(routes[rank], dtype=torch.int64)
    valid = (selected >= 0) & (selected < experts)
    slots = torch.arange(selected.shape[1])[None, :]
    count = valid.sum(dim=1, keepdim=True)
    bias = ((selected % 3 - 1 + slots % 3 - 1) * valid).sum(dim=1, keepdim=True)
    # Bounded integer fixtures make this factored sum exact in FP32/BF16.
    out = base * count + bias
    return out.to(torch.bfloat16)


def weight_payload(ids, generation):
    import torch

    bits = 0x3E800000 + ((ids + generation * 104729) % (1 << 25))
    return bits.to(torch.int32).view(torch.float32)


def expected_weights(routes, experts, rank, generation):
    import torch

    selected = torch.tensor(routes[rank], dtype=torch.int64)
    tokens, topk = selected.shape
    ids = torch.arange(tokens * topk).reshape(tokens, topk) + rank * tokens * topk
    return weight_payload(ids, generation).masked_fill((selected < 0) | (selected >= experts), 0)


def check_dispatch_inputs(
    enabled,
    handle,
    reference,
    rank,
    entries,
    generation,
    fp8,
    scale_value,
    recv,
    recv_weights,
    input_weights,
    weight_ids,
    log,
    context,
    hidden=7168,
):
    """Optional payload Dispatch oracle; Notify is checked independently."""
    if not enabled:
        return
    import torch

    log(f"{context} dispatch_golden.begin (CPU)")
    sources = torch.tensor([s + t for s, t, _ in entries], dtype=torch.int32)[:, None]
    expected_recv = ((sources + torch.arange(hidden)[None, :] + generation) % 7 - 3).to(torch.bfloat16)
    log(f"{context} dispatch_golden.done")
    if fp8:
        _notify["assert_bytes"](
            recv[0],
            (expected_recv.float() / scale_value).to(torch.float8_e4m3fn),
            f"{context} FP8 hidden",
            progress=log,
        )
        _notify["assert_bytes"](
            recv[1],
            torch.full((len(entries), hidden // 128), scale_value, dtype=torch.float32),
            f"{context} FP8 scales",
            progress=log,
        )
    else:
        _notify["assert_bytes"](recv, expected_recv, f"{context} dispatch input to experts", progress=log)
    if input_weights is None:
        assert recv_weights is None
    else:
        _notify["assert_bytes"](
            recv_weights,
            weight_payload(weight_ids, generation),
            f"{context} dispatch weights",
            progress=log,
        )


def golden_combine_handle(buffer, args, indices, reference, routes):
    """Build a test-only Combine handle without launching Notify or Dispatch."""
    import torch
    from deep_ep._native import require_native
    from deep_ep.buffers.elastic import EPHandle, _PERSISTENT

    rank, world = buffer._runtime.rank, buffer._runtime.world_size
    specification = require_native("Combine golden layout").notify_layout(
        rank, world, args.tokens, args.topk, args.experts, args.chunk_tokens, combine_only=True
    )
    tensors = {
        name: torch.zeros(shape, dtype=getattr(torch, dtype), device=indices.device)
        for name, (shape, dtype) in specification["tensors"].items()
        if name in _PERSISTENT
    }
    for name in _PERSISTENT:
        if name in ("forward_list", "backward_list"):
            for peer, rows in enumerate(reference[name]):
                if rows:
                    tensors[name][peer, : len(rows)].copy_(
                        torch.tensor(rows, dtype=tensors[name].dtype, device=indices.device)
                    )
        elif name == "weight_return_meta":
            tensors[name] = torch.tensor(reference[name], dtype=torch.int32, device=indices.device)
        elif name in reference:
            tensors[name].copy_(
                torch.tensor(reference[name], dtype=tensors[name].dtype, device=indices.device).reshape(
                    tensors[name].shape
                )
            )
    selected = torch.tensor(routes, dtype=torch.int64).flatten()
    totals = torch.bincount(selected[(selected >= 0) & (selected < args.experts)], minlength=args.experts)
    tensors["expert_totals"].copy_(totals)
    local = args.experts // world
    count_list = totals[rank * local : (rank + 1) * local].tolist()
    counts = tensors["expert_totals"][rank * local : (rank + 1) * local].clone()
    handle = EPHandle(
        True,
        args.experts,
        1,
        args.tokens,
        32,
        indices,
        None,
        reference["rank_rows"][rank],
        count_list,
        None,
        counts.cumsum(0, dtype=torch.int32),
        counts,
        None,
        None,
        None,
        None,
    )
    for name, tensor in tensors.items():
        setattr(handle, name, tensor)
    handle.address_stride = max(1, max(reference["rank_rows"]))
    handle._gather_stride = max(1, max(reference["gather_rows"]))
    handle.rank, handle.world_size = rank, world
    handle.num_tokens, handle.num_topk = args.tokens, args.topk
    handle.hidden, handle.chunk_tokens = args.hidden, args.chunk_tokens
    handle._buffer_token, handle.metadata_only = buffer._handle_token, False
    handle._routing_versions = {
        name: (getattr(handle, name).data_ptr(), getattr(handle, name)._version) for name in ("topk_idx", *_PERSISTENT)
    }
    return handle


def prepare_routing(buffer, args, x, indices, input_weights, handle, reference=None, routes=None):
    """Prepare routing outside profiling; payload Dispatch is opt-in."""
    if args.check_dispatch:
        if hasattr(args, "no_server_dedup"):
            buffer._notify_rank_dedup = args.no_server_dedup
        recv, _, recv_weights, handle, event = buffer.dispatch(
            x,
            topk_weights=input_weights,
            topk_idx=indices,
            num_experts=args.experts if handle is None else None,
            num_max_tokens_per_rank=args.tokens,
            handle=handle,
            do_expand=True,
            async_with_compute_stream=True,
        )
        return recv, recv_weights, handle, event
    if handle is not None:
        return None, None, handle, buffer.capture()
    if getattr(args, "golden_notify", False):
        handle = golden_combine_handle(buffer, args, indices, reference, routes)
        return None, None, handle, buffer.capture()
    _, _, _, handle, event = buffer._notify(
        x=x,
        topk_idx=indices,
        topk_weights=None,
        stats=None,
        num_experts=args.experts,
        capacity=args.tokens,
        alignment=1,
        cores=getattr(args, "notify_num_cores", 64),
        qps=0,
        previous_event=None,
        previous_epilogue=None,
        async_with_compute=True,
        allocate_on_comm=False,
        handle=None,
        copy_handle=True,
        cpu_sync=True,
        expand=True,
        zero_padding=False,
        aligned_sf=False,
        chunk_tokens=getattr(args, "chunk_tokens", 256),
        rank_dedup=getattr(args, "no_server_dedup", False),
    )
    from deep_ep.buffers.elastic import _PERSISTENT

    # Test-only fixture: expanded expert inputs are constructed independently below,
    # so admit this Notify handle to Combine without executing a payload dispatch.
    # Production handle validation is unchanged; freeze every routing tensor used here.
    handle._routing_versions = {
        name: (getattr(handle, name).data_ptr(), getattr(handle, name)._version)
        for name in ("topk_idx", *_PERSISTENT, *(("_peer_plan",) if hasattr(handle, "_peer_plan") else ()))
    }
    handle.metadata_only = False
    return None, None, handle, event


def direct_expert_input(entries, generation, device, hidden=7168):
    """Expanded expert input in expert/source/token/slot order, without transport."""
    import torch

    sources = torch.tensor(
        [source + token + generation for source, token, _ in entries], dtype=torch.int32, device=device
    )
    columns = torch.arange(hidden, dtype=torch.int32, device=device)
    return ((sources[:, None] + columns[None, :]) % 7 - 3).float()


def hotspot_expert_output(entries, bias, generation, device, hidden, pool=None):
    import torch

    with torch.npu.memory.use_mem_pool(pool) if pool is not None else nullcontext():
        output = torch.empty((len(entries), hidden), dtype=torch.bfloat16, device=device)
    for start in range(0, len(entries), 4096):
        end = min(start + 4096, len(entries))
        values = direct_expert_input(entries[start:end], generation, device, hidden)
        output[start:end].copy_((values + bias[start:end]).to(torch.bfloat16))
    return output


def test_hotspot_input_tiling_preserves_values():
    torch = pytest.importorskip("torch")
    entries = [(i % 8, i // 8, i % 6) for i in range(4103)]
    bias = torch.tensor([i % 5 - 2 for i in range(len(entries))], dtype=torch.float32)[:, None]
    expected = (direct_expert_input(entries, 2, "cpu", 128) + bias).to(torch.bfloat16)
    assert torch.equal(hotspot_expert_output(entries, bias, 2, "cpu", 128), expected)


def worker(local_rank, args):
    if args.dense_channels:
        os.environ["DEEPEP_SHMEM_DENSE_QPS"] = "1"
    import torch
    import torch_npu  # noqa: F401

    deep_ep = _notify["_import_installed_deep_ep"]()
    from deep_ep.buffers._elastic_layout import elastic_size_hint

    info = json.loads(Path(deep_ep.__file__).with_name("_build_info.json").read_text())
    if "elastic-combine-expanded-v3" not in info.get("capabilities", []):
        raise RuntimeError("Installed wheel does not contain expanded Combine; rebuild/install first")
    if "elastic-notify-metadata-v7" not in info.get("capabilities", []):
        raise RuntimeError("Rebuild/install the wheel with Notify metadata v7")
    if getattr(args, "no_server_dedup", False) and "elastic-combine-rank-return-v1" not in info.get("capabilities", []):
        raise RuntimeError("Rebuild/install all ranks with rank return support before --no-server-dedup")
    if args.topk > (args.ep_size + 7) // 8 and "elastic-combine-return-slots-v2" not in info.get("capabilities", []):
        raise RuntimeError("Rebuild/install all ranks with expanded Combine return slots")
    if args.profile_op and not {
        "elastic-combine-profile-v4",
        "elastic-combine-peers-v1",
        "elastic-combine-chunk-tuning-v1",
    }.issubset(info.get("capabilities", [])):
        raise RuntimeError("Rebuild/install all ranks for Combine profile v3 and peers v1 before profiling")
    if "runtime-combine-sparse-qp-v1" not in info.get("capabilities", []) or (
        args.profile_op and "elastic-combine-chunks-v2" not in info.get("capabilities", [])
    ):
        raise RuntimeError("Rebuild/install all ranks for Combine sparse QPs and chunk diagnostics")
    torch.npu.set_device(local_rank)
    # Reuse the producer stream so allocator caches do not grow per iteration.
    args._producer_stream = torch.npu.Stream()
    rank, world = _notify["_worker_identity"](local_rank, args)
    log = _notify["_progress_logger"]("combine", rank)
    log(
        f"worker.begin route={args.route} seed={args.seed} bootstrap={args.bootstrap} "
        f"chunk_tokens={args.chunk_tokens} local_expert_first={getattr(args, 'local_expert_first', False)}"
    )
    cases = _combine_cases["from_args"](args)
    modes = _ROUTES if args.route == "all" else (args.route,)
    capacity = 0
    input_storages = getattr(args, "input_storage_list", None) or [getattr(args, "input_storage", "gm")]
    huge_allocator, huge_pool = None, None
    if "gm-huge" in input_storages:
        from deep_ep import _C

        if getattr(_C, "gm_huge_allocator_version", lambda: 0)() != 1:
            raise RuntimeError("Rebuild/install all ranks with the huge GM allocator")
        huge_allocator = torch.npu.memory.NPUPluggableAllocator(
            _C.__file__, "deepep_gm_huge_alloc", "deepep_gm_huge_free"
        )
        huge_pool = torch.npu.MemPool(huge_allocator.allocator())
    staging_capacity = 0
    first_references = {}
    seen_tokens = set()
    # Row counts are independent of chunk partitioning. The size hint uses chunk256,
    # which reserves at least as much control space as chunk512/1024.
    for case in cases:
        tokens = case["tokens"]
        if tokens in seen_tokens:
            continue
        seen_tokens.add(tokens)
        for mode in modes:
            log(f"tokens={tokens} route={mode} capacity_golden.begin (CPU)")
            routes = make_combine_routes(world, tokens, args, mode)
            plan = build_reference(routes, args.experts, rank, case["chunk_tokens"])
            if "shmem" in input_storages:
                staging_capacity = max(staging_capacity, (max(plan["rank_rows"]) * args.hidden * 2 + 511) // 512 * 512)
            capacity = max(
                capacity,
                elastic_size_hint(
                    world,
                    tokens,
                    args.topk,
                    args.experts,
                    max(1, max(plan["rank_rows"])),
                    max(1, max(plan["gather_rows"])),
                    fp8=args.dispatch_dtype == "fp8",
                    hidden=args.hidden,
                ),
            )
            if case is cases[0]:
                first_references[mode] = plan
            log(f"tokens={tokens} route={mode} capacity_golden.done session_bytes={capacity}")
        del plan, routes
    if staging_capacity:
        capacity = (capacity + 2 * staging_capacity + (2 << 20) - 1) // (2 << 20) * (2 << 20)
    max_tokens = max(case["tokens"] for case in cases)
    log(f"buffer.init.begin bytes={capacity} max_tokens={max_tokens} cases={len(cases)}")
    started = time.monotonic()
    buffer = deep_ep.ElasticBuffer(
        rank=rank,
        world_size=world,
        num_bytes=capacity,
        num_max_tokens_per_rank=max_tokens,
        hidden=args.hidden,
        num_topk=args.topk,
        use_fp8_dispatch=args.dispatch_dtype == "fp8",
        explicitly_destroy=True,
        bootstrap=args.bootstrap,
        combine_only=not args.check_dispatch and not args.dense_channels,
        use_1g_huge_pages=getattr(args, "use_1g_huge_pages", True),
    )
    init_seconds = time.monotonic() - started
    workspace_allocation_bytes = buffer._runtime.native.workspace_allocation_bytes
    log(f"buffer.init.done seconds={init_seconds:.6f} use_1g_huge_pages={getattr(args, 'use_1g_huge_pages', True)}")
    retained, retained_weights = None, None
    try:
        for index, case in enumerate(cases):
            current = _combine_cases["case_args"](args, case, len(cases) > 1)
            # No SHMEM finalize/reinit and no change to QPs between cases.
            torch.npu.synchronize()
            retained, retained_weights = None, None
            log(f"case.begin {index + 1}/{len(cases)} id={case['case_id']}")
            for storage in input_storages:
                selected = copy.copy(current)
                selected.input_storage = storage
                if storage == "gm-huge":
                    selected._gm_huge_pool = huge_pool
                if len(input_storages) > 1:
                    selected.output = current.output / f"input-{storage}"
                buffer._combine_stage_input = storage == "shmem"
                log(f"input_storage={storage} case={case['case_id']}")
                retained, retained_weights = run_case(
                    buffer, selected, rank, world, info, log, first_references if index == 0 else {}
                )
                torch.npu.synchronize()
            torch.npu.synchronize()
            log(f"case.done {index + 1}/{len(cases)} id={case['case_id']}")
    except BaseException:
        _notify["_log_failure"](log)
        raise
    finally:
        log("buffer.destroy.begin (may wait for device work/peers)")
        buffer.destroy()
        log("buffer.destroy.done")
    if retained is not None:
        _notify["assert_bytes"](*retained, "output after destroy")
    if retained_weights is not None:
        _notify["assert_bytes"](*retained_weights, "weights after destroy")
    if len(cases) > 1:
        path = args.output / f"rank-{rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                dict(
                    rank=rank,
                    world_size=world,
                    passed=True,
                    cases=cases,
                    initialization_count=1,
                    without_replacement=getattr(args, "without_replacement", False),
                    local_expert_first=getattr(args, "local_expert_first", False),
                    buffer_init_seconds=init_seconds,
                    session_bytes=capacity,
                    use_1g_huge_pages=buffer._runtime.native.use_1g_huge_pages,
                    workspace_allocation_bytes=workspace_allocation_bytes,
                    max_tokens=max_tokens,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def run_case(buffer, args, rank, world, info, log, references):
    import torch

    if buffer is not None:
        buffer._combine_reuse_tokens = getattr(args, "reuse_window_tokens", 0)
    if getattr(args, "combine_capacity_mib", 0):
        buffer._combine_capacity_limit = args.combine_capacity_mib << 20
    args.output.mkdir(parents=True, exist_ok=True)
    modes = _ROUTES if args.route == "all" else (args.route,)
    checks, retained, retained_weights = 0, None, None
    diagnostics, traffic, input_allocations = [], {}, []
    for mode in modes:
        log(f"route={mode} routes_and_expert_rows.begin (CPU)")
        routes = make_combine_routes(world, args.tokens, args, mode)
        entries = _notify["expert_rows"](routes, args.experts, rank)
        log(f"route={mode} routes_and_expert_rows.done rows={len(entries)}")
        if mode in references:
            reference = references.pop(mode)
            log(f"route={mode} notify_reference.reused")
        else:
            log(f"route={mode} notify_reference.begin (CPU)")
            reference = build_reference(routes, args.experts, rank, args.chunk_tokens)
            log(f"route={mode} notify_reference.done")
        weight_ids = torch.tensor([(s * args.tokens + t) * args.topk + k for s, t, k in entries], dtype=torch.int64)
        handle = None
        for generation in range(args.repeat):
            context = f"route={mode} iter={generation + 1}/{args.repeat}"
            log(f"{context} inputs.begin")
            indices = torch.tensor(routes[rank], dtype=torch.int64, device="npu") if handle is None else None
            x = (
                payload(rank, args.tokens, generation, args.hidden).npu()
                if args.check_dispatch
                else torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16, device="npu")
            )
            fp8 = args.dispatch_dtype == "fp8"
            scale_value = 0.5 if generation % 2 else 1.0
            dispatch_input = (
                (
                    (x.float() / scale_value).to(torch.float8_e4m3fn),
                    torch.full((args.tokens, args.hidden // 128), scale_value, dtype=torch.float32, device="npu"),
                )
                if fp8
                else x
            )
            input_ids = torch.arange(args.tokens * args.topk) + rank * args.tokens * args.topk
            input_weights = (
                None
                if args.without_weights or not args.check_dispatch
                else weight_payload(input_ids, generation).reshape(args.tokens, args.topk).npu()
            )
            preparation = (
                "dispatch"
                if args.check_dispatch
                else "golden_notify"
                if getattr(args, "golden_notify", False)
                else "notify_only"
            )
            log(f"{context} {preparation}.enter mode={'fresh' if handle is None else 'cached'}")
            recv, recv_weights, handle, event = prepare_routing(
                buffer, args, dispatch_input, indices, input_weights, handle, reference, routes
            )
            log(f"{context} {preparation}.return")
            event.current_stream_wait()
            # Validate the installed metadata once per plan, outside profiling.
            if generation == 0:
                log(f"{context} notify_metadata.check.begin")
                if getattr(args, "no_server_dedup", False):
                    reference = build_reference(routes, args.experts, rank, args.chunk_tokens, server_dedup=False)
                    if getattr(args, "golden_notify", False):
                        install_rank_return_plan(handle, reference)
                        event = buffer.capture()
                        log(f"{context} rank_return_golden.installed")
                    elif not handle._rank_return_plan:
                        raise AssertionError("Real Notify did not produce a rank-return handle")
                check_handle(handle, reference, rank)
                log(f"{context} notify_metadata.check.passed")
                traffic[mode] = combine_traffic(reference, rank, args.hidden)
                args.combine_traffic = traffic[mode]
                log(f"{context} traffic={json.dumps(traffic[mode], sort_keys=True)}")
            check_dispatch_inputs(
                args.check_dispatch,
                handle,
                reference,
                rank,
                entries,
                generation,
                fp8,
                scale_value,
                recv,
                recv_weights,
                input_weights,
                weight_ids,
                log,
                context,
                args.hidden,
            )
            # Producer stream is deliberately distinct; Combine must honor its event.
            log(f"{context} expert_compute.enqueue.begin")
            producer = args._producer_stream
            with torch.npu.stream(producer):
                event.current_stream_wait()
                bias = torch.tensor(
                    [routes[s][t][k] % 3 - 1 + k % 3 - 1 for s, t, k in entries], dtype=torch.float32, device="npu"
                )[:, None]
                pool = getattr(args, "_gm_huge_pool", None)
                if not args.check_dispatch and mode in ("hotspot", "rotating_hotspot"):
                    expert = hotspot_expert_output(entries, bias, generation, "npu", args.hidden, pool)
                else:
                    if args.check_dispatch:
                        expert_input = recv[0].float() * recv[1].repeat_interleave(128, dim=1) if fp8 else recv.float()
                    else:
                        expert_input = direct_expert_input(entries, generation, "npu", args.hidden)
                    expert_float = expert_input + bias
                    with torch.npu.memory.use_mem_pool(pool) if pool is not None else nullcontext():
                        expert = expert_float.to(torch.bfloat16)
                    del expert_float, expert_input
                if getattr(args, "input_storage", "gm") == "gm-huge" and expert.numel():
                    from deep_ep import _C

                    allocation = _C.gm_huge_allocation_info(expert.data_ptr(), expert.numel() * expert.element_size())
                    input_allocations.append(allocation)
                    log(f"{context} input_storage=gm-huge allocation={allocation}")
                weights = None if args.without_weights else weight_payload(weight_ids, generation).npu()
                before = buffer.capture()
            log(f"{context} expert_compute.enqueued; combine.enter")
            output, values, completed = buffer.combine(
                expert, handle, topk_weights=weights, previous_event=before, async_with_compute_stream=True
            )
            log(f"{context} combine.return (async; device completion not yet confirmed)")
            completed.current_stream_wait()
            args.combine_reuse = bool(getattr(buffer, "_combine_last_reuse_tokens", 0))
            log(f"{context} combine_golden.begin (CPU)")
            golden = expected_output(routes, args.experts, rank, generation, args.hidden)
            log(f"{context} combine_golden.done")
            _notify["assert_bytes"](output, golden, f"{context} initial combine", progress=log)
            if getattr(args, "compare_full", False) and args.combine_reuse:
                force = buffer._combine_reuse_tokens
                limit = getattr(buffer, "_combine_capacity_limit", buffer._runtime.capacity // 2)
                try:
                    buffer._combine_reuse_tokens = 0
                    buffer._combine_capacity_limit = buffer._runtime.capacity // 2
                    full, full_weights, done = buffer.combine(expert, handle, topk_weights=weights)
                    done.current_stream_wait()
                    assert buffer._combine_last_reuse_tokens == 0
                    _notify["assert_bytes"](output, full.cpu(), f"{context} reuse versus full", progress=log)
                    if weights is not None:
                        _notify["assert_bytes"](
                            values, full_weights.cpu(), f"{context} reuse weights versus full", progress=log
                        )
                finally:
                    buffer._combine_reuse_tokens = force
                    buffer._combine_capacity_limit = limit
                    buffer._combine_last_reuse_tokens = force or handle._combine_reuse_plan[0]
            weight_golden = None
            if weights is None:
                assert values is None
            else:
                weight_golden = expected_weights(routes, args.experts, rank, generation)
                _notify["assert_bytes"](values, weight_golden, f"{context} weights", progress=log)
            if args.profile_op and generation == 0:

                def invoke():
                    result = buffer.combine(expert, handle, topk_weights=weights, async_with_compute_stream=True)
                    result[-1].current_stream_wait()
                    return result

                def validate(result):
                    if weight_golden is None:
                        assert result[1] is None
                    else:
                        _notify["assert_bytes"](result[1], weight_golden, "profile combine weights")
                    _notify["assert_bytes"](result[0], golden, "profile combine")

                diagnostics.append(
                    profile_combine(
                        args,
                        rank,
                        f"{mode}-{args.dispatch_dtype}-weights{int(weights is not None)}"
                        + ("-no-server-dedup" if getattr(args, "no_server_dedup", False) else "")
                        + ("-local-expert-first" if getattr(args, "local_expert_first", False) else ""),
                        buffer,
                        expert,
                        invoke,
                        validate,
                        info,
                        log,
                    )
                )
            # The same handle and workspace must support consecutive combines.
            log(f"{context} repeated_combine.enter")
            again, no_weights, completed = buffer.combine(expert, handle)
            log(f"{context} repeated_combine.return (device completion not yet confirmed)")
            completed.current_stream_wait()
            assert no_weights is None
            _notify["assert_bytes"](again, golden, f"{context} repeated combine", progress=log)
            if weights is not None:
                changed = weight_payload(weight_ids, generation + 1).npu()
                again, changed_out, completed = buffer.combine(expert, handle, topk_weights=changed)
                completed.current_stream_wait()
                _notify["assert_bytes"](again, golden, f"{context} changed weights hidden")
                _notify["assert_bytes"](
                    changed_out,
                    expected_weights(routes, args.experts, rank, generation + 1),
                    f"{context} changed weights",
                )
                _notify["assert_bytes"](values, weight_golden, f"{context} retained weights")
            log(f"{context} retained_output_and_negative_checks.begin")
            if retained is not None:
                _notify["assert_bytes"](*retained, "retained output")
            retained = (output, golden)
            if retained_weights is not None:
                _notify["assert_bytes"](*retained_weights, "retained weights from previous iteration")
            if values is not None:
                retained_weights = (values, weight_golden)
            corrupt = golden.clone()
            corrupt[0, 0] += 1
            try:
                _notify["assert_bytes"](output, corrupt, "negative control")
            except AssertionError:
                pass
            else:
                raise AssertionError("Comparator accepted corrupted golden")
            checks += 1
            log(f"{context} iteration.passed")
        print(f"[combine][rank={rank}] route={mode} precision passed", flush=True)
    # Complete all streams before the next case can drop its routing/input tensors.
    torch.npu.synchronize()
    report_path = args.output / f"rank-{rank}.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            dict(
                rank=rank,
                world_size=world,
                checks=checks,
                passed=True,
                bootstrap=args.bootstrap,
                tokens=args.tokens,
                case_id=args.case_id,
                hidden=args.hidden,
                session_bytes=buffer._runtime.capacity,
                use_1g_huge_pages=buffer._runtime.native.use_1g_huge_pages,
                workspace_allocation_bytes=buffer._runtime.native.workspace_allocation_bytes,
                chunk_tokens=args.chunk_tokens,
                input_storage=getattr(args, "input_storage", "gm"),
                input_allocations=input_allocations,
                reuse_tokens=getattr(buffer, "_combine_last_reuse_tokens", 0),
                reuse_compared_with_full=getattr(args, "compare_full", False),
                topk=args.topk,
                experts=args.experts,
                route=args.route,
                dispatch_dtype=args.dispatch_dtype,
                dispatch_checked=args.check_dispatch,
                dispatch_executed=args.check_dispatch,
                notify_checked=not getattr(args, "golden_notify", False),
                notify_executed=not getattr(args, "golden_notify", False),
                local_expert_first=getattr(args, "local_expert_first", False),
                deduplication="rank" if getattr(args, "no_server_dedup", False) else "server",
                combine_metadata_source=("golden" if getattr(args, "golden_notify", False) else "notify"),
                traffic=traffic,
                combine_diagnostics=diagnostics,
                test_files_sha256={
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in (
                        "test_combine.py",
                        "test_notify.py",
                        "combine_reference.py",
                        "combine_metrics.py",
                        "combine_chunks.py",
                        "combine_peers.py",
                        "combine_cases.py",
                    )
                },
                seed=args.seed,
                without_replacement=getattr(args, "without_replacement", False),
                with_weights=not args.without_weights,
                combine_only=not args.check_dispatch and not args.dense_channels,
                source_sha256=info["source_sha256"],
            ),
            indent=2,
        )
        + "\n"
    )
    temporary.replace(report_path)
    return retained, retained_weights


@pytest.mark.parametrize("world", [8, 32])
def test_local_expert_first_preserves_routes_and_local_preferred_traffic(monkeypatch, world):
    experts, tokens, topk = world * 2, 3, 6
    original = []
    for source in range(world):
        neighbor = source // 8 * 8 + (source + 1) % 8
        remote = (source + 8) % world if world > 8 else neighbor
        original.append(
            [
                [neighbor * 2, remote * 2, source * 2, -1, source * 2 + 1, neighbor * 2],
                [neighbor * 2, neighbor * 2 + 1, -1, experts, -1, -1],
                [-1] * topk,
            ]
        )
    monkeypatch.setitem(_notify, "make_routes", lambda *a, **k: copy.deepcopy(original))
    args = SimpleNamespace(topk=topk, experts=experts, seed=1)
    assert make_combine_routes(world, tokens, args, "random") == original
    args.local_expert_first = True
    ordered = make_combine_routes(world, tokens, args, "random")

    def gateways(row):
        result = {}
        for expert in row:
            if 0 <= expert < experts:
                result.setdefault(expert // 16, expert // 2)
        return result

    changed = 0
    for source, (before, after) in enumerate(zip(original, ordered, strict=True)):
        assert after[1:] == before[1:]
        for old, new in zip(before, after, strict=True):
            assert sorted(old) == sorted(new)
            assert [e for e in new if not source * 2 <= e < source * 2 + 2] == [
                e for e in old if not source * 2 <= e < source * 2 + 2
            ]
            old_gateways, new_gateways = gateways(old), gateways(new)
            for server, gateway in old_gateways.items():
                if server != source // 8:
                    assert new_gateways[server] == gateway
            if any(source * 2 <= e < source * 2 + 2 for e in old):
                assert new_gateways[source // 8] == source
                changed += old_gateways[source // 8] != source
        assert after[0][:2] == [source * 2, source * 2 + 1]

    totals = [{}, {}]
    for rank in range(world):
        plans = [build_reference(routes, experts, rank, 512) for routes in (original, ordered)]
        assert plans[0]["rank_rows"] == plans[1]["rank_rows"]
        assert plans[0]["server_mask"] == plans[1]["server_mask"]
        for total, plan in zip(totals, plans, strict=True):
            for name, count in combine_traffic(plan, rank, 7168).items():
                total[name] = total.get(name, 0) + count
    assert changed == world
    assert totals[0] == totals[1]


def test_local_expert_first_cli_checks_before_launch(monkeypatch, tmp_path):
    prefix = ["test_combine.py", "--ep-size", "32", "--experts", "512", "--output", str(tmp_path)]
    calls = []
    monkeypatch.setitem(_notify, "_launch", lambda args, worker: calls.append(args))
    monkeypatch.setattr(sys, "argv", prefix + ["--local-expert-first"])
    main()
    assert len(calls) == 1 and calls[0].local_expert_first and not calls[0].no_server_dedup
    monkeypatch.setattr(sys, "argv", prefix + ["--local-expert-first", "--no-server-dedup", "--golden-notify"])
    main()
    assert len(calls) == 2 and calls[1].local_expert_first and calls[1].no_server_dedup and calls[1].golden_notify
    for extra in ([], ["--no-server-dedup"]):
        monkeypatch.setattr(sys, "argv", prefix + ["--local-expert-first", "--with-weights", *extra])
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
    assert len(calls) == 2


@pytest.mark.parametrize("world", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("chunk_tokens", [256, 512])
@pytest.mark.parametrize("local_expert_first", [False, True])
def test_rank_return_plan_has_unique_slots_and_complete_reduction(monkeypatch, world, chunk_tokens, local_expert_first):
    tokens, topk, experts = 7, 6, world * 2
    return_slots = max((world + 7) // 8, topk)
    routes = _notify["make_routes"](world, tokens, topk, experts, 0, "random", seed=1)
    # Force duplicate experts, distinct ranks in one server, self-return and invalid routes.
    for source in range(world):
        routes[source][0] = [0, 0, 2, 4, experts - 2, -1]
        routes[source][1] = [source * 2] * topk
        routes[source][2] = [-1] * topk
    original = copy.deepcopy(routes)
    monkeypatch.setitem(_notify, "make_routes", lambda *a, **k: copy.deepcopy(original))
    args = SimpleNamespace(topk=topk, experts=experts, seed=1, local_expert_first=local_expert_first)
    routes = make_combine_routes(world, tokens, args, "random")
    writes, expected_writes = {}, set()
    for source in range(world):
        for token, selected in enumerate(routes[source]):
            peers = list(dict.fromkeys(e // 2 for e in selected if 0 <= e < experts))
            expected_writes.update((source, token * return_slots + slot) for slot in range(len(peers)))
    for rank in range(world):
        plan = build_reference(routes, experts, rank, chunk_tokens, server_dedup=False)
        assert not any(plan["backward_counts"] + plan["gather_rows"] + plan["chunk_masks"])
        for source, tasks in enumerate(plan["forward_list"]):
            cursor = 0
            while cursor < len(tasks):
                end, encoded, primary, *_ = tasks[cursor]
                flat = -encoded - 1
                assert encoded < 0 and 0 <= flat < tokens * return_slots
                if local_expert_first and source == rank:
                    assert flat % return_slots == 0
                assert (source, flat) not in writes
                rows = [primary]
                for task in tasks[cursor:end]:
                    assert task[1] == encoded
                    if task[3] >= 0:
                        assert task[3] == rank and task[5] == -1
                        rows.append(task[4])
                ids = [plan["weight_return_meta"][row] for row in rows]
                writes[source, flat] = sum(identity % 17 - 8 for identity in ids)
                cursor = end
        for token, mask in enumerate(plan["server_mask"]):
            peers = {e // 2 for e in routes[rank][token] if 0 <= e < experts}
            assert mask == (1 << len(peers)) - 1
        baseline = build_reference(routes, experts, rank, chunk_tokens)
        assert plan["rank_rows"] == baseline["rank_rows"]
        actual = combine_traffic(plan, rank, 7168)
        assert actual["gather_payload_bytes"] == 0
        assert actual["cross_server_return_rows"] >= combine_traffic(baseline, rank, 7168)["cross_server_return_rows"]
    assert set(writes) == expected_writes
    for source in range(world):
        for token, selected in enumerate(routes[source]):
            actual = sum(writes.get((source, token * return_slots + slot), 0) for slot in range(topk))
            expected = sum(
                ((source * tokens + token) * topk + slot) % 17 - 8
                for slot, expert in enumerate(selected)
                if 0 <= expert < experts
            )
            assert actual == expected


@pytest.mark.parametrize("world", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("topk", [1, 6, 16])
def test_combine_return_slots_cover_topk(world, topk):
    combine_layout = runpy.run_path(str(Path(__file__).parents[2] / "deep_ep/buffers/_elastic_layout.py"))[
        "combine_layout"
    ]

    tokens, hidden = 16384, 7168
    layout = combine_layout(world, tokens, topk, 1, 1, 512, hidden)
    expected = tokens * max((world + 7) // 8, topk) * hidden * 2
    assert layout["output"] - layout["returned_partial"] == expected


@pytest.mark.parametrize(
    "extra, expected",
    [
        ([], True),
        (["--rank-dedup"], True),
        (["--golden-notify"], True),
        (["--with-dispatch"], True),
        (["--profile-op", "combine"], True),
        (["--ep-size", "16"], False),
        (["--server-dedup"], False),
        (["--with-weights"], True),
    ],
)
def test_ep8_rank_pipeline_cli_selects_kernel_automatically(monkeypatch, tmp_path, extra, expected):
    calls = []
    monkeypatch.setitem(_notify, "_launch", lambda args, worker: calls.append(args))
    prefix = ["test_combine.py", "--ep-size", "8", "--experts", "384", "--output", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", prefix + extra)
    main()
    assert len(calls) == 1 and calls[0].ep8_rank_pipeline == expected


def test_ep8_rank_pipeline_profile_selects_the_requested_kernel(tmp_path):
    csv = tmp_path / "kernel_details.csv"
    csv.write_text(
        "Name,Duration(us)\ncombine_prepare_kernel,3\ncombine_prepare_kernel,4\n"
        "combine_kernel,1200\ncombine_kernel,1300\n"
        "combine_ep8_kernel,400\ncombine_ep8_kernel,500\n"
        "combine_ep8_profile_kernel,600\n",
        encoding="utf-8",
    )
    normal = _notify["read_kernel_summary"](tmp_path, "combine", 2)
    optimized = _notify["read_kernel_summary"](tmp_path, "combine", 2, ep8_rank_pipeline=True)
    assert normal["avg_us"] == 1250 and optimized["avg_us"] == 450
    assert optimized["kernel"] == "combine_ep8_kernel"
    assert optimized["prepare"]["avg_us"] == 3.5


def test_rank_return_cli_checks_before_launch(monkeypatch, tmp_path):
    prefix = ["test_combine.py", "--ep-size", "128", "--experts", "384", "--output", str(tmp_path)]
    calls = []
    monkeypatch.setitem(_notify, "_launch", lambda args, worker: calls.append(args))
    monkeypatch.setattr(sys, "argv", prefix + ["--no-server-dedup"])
    main()
    assert len(calls) == 1 and calls[0].no_server_dedup and not calls[0].check_dispatch
    monkeypatch.setattr(sys, "argv", prefix + ["--no-server-dedup", "--ep-size", "32"])
    main()
    assert len(calls) == 2 and calls[1].ep_size == 32
    for extra in (["--with-dispatch"],):
        monkeypatch.setattr(sys, "argv", prefix + ["--no-server-dedup", *extra])
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
    assert len(calls) == 2


@pytest.mark.parametrize("world", [32, 64])
@pytest.mark.parametrize("chunk_tokens", [256, 512])
def test_rank_return_chunk_boundaries_preserve_original_tokens(world, chunk_tokens):
    routes = _notify["make_routes"](world, 517, 6, 128, 0, "random", seed=1)
    return_slots = max((world + 7) // 8, 6)
    plan = build_reference(routes, 128, 0, chunk_tokens, server_dedup=False)
    physical_chunks = math.ceil(517 / chunk_tokens) + 3
    ends = [chunk_tokens * (i + 1) // 4 for i in range(4)]
    ends += [min((i + 2) * chunk_tokens, 517) for i in range(physical_chunks - 4)]
    for source, tasks in enumerate(plan["forward_list"]):
        previous = 0
        for chunk, token_end in enumerate(ends):
            offset = (source * physical_chunks + chunk) * 2
            begin, end = plan["chunk_ranges"][offset : offset + 2]
            assert begin == previous
            assert all((-task[1] - 1) // return_slots < token_end for task in tasks[begin:end])
            if end < len(tasks):
                assert (-tasks[end][1] - 1) // return_slots >= token_end
            previous = end
        assert previous == len(tasks)


def test_rank_return_metadata_replacement_refreshes_versions():
    torch = pytest.importorskip("torch")

    routes = _notify["make_routes"](64, 3, 6, 128, 0, "random", seed=1)
    reference = build_reference(routes, 128, 0, server_dedup=False)
    names = ("forward_counts", "backward_counts", "server_mask", "chunk_ranges", "chunk_masks", "gather_rows")
    handle = SimpleNamespace(
        **{name: torch.zeros(len(reference[name]), dtype=torch.int64) for name in names},
        forward_list=torch.zeros((64, 3 * 6, 6), dtype=torch.int32),
        _gather_stride=123,
        _combine_layout_cache=("old", {}),
    )
    handle._routing_versions = {
        name: (getattr(handle, name).data_ptr(), getattr(handle, name)._version) for name in (*names, "forward_list")
    }
    install_rank_return_plan(handle, reference)
    assert handle._gather_stride == 123 and handle._combine_layout_cache is None
    for name, version in handle._routing_versions.items():
        tensor = getattr(handle, name)
        assert version == (tensor.data_ptr(), tensor._version)
    for source, rows in enumerate(reference["forward_list"]):
        assert handle.forward_list[source, : len(rows)].tolist() == [list(row) for row in rows]


def test_local_routes_require_reduction_without_remote_gather():
    for mode in ("local_duplicates", "local_experts"):
        routes = _notify["make_routes"](16, 3, 4, 32, 0, mode)
        for source, tokens in enumerate(routes):
            for token, row in enumerate(tokens):
                assert {expert // 2 for expert in row} == {(source + token) % 16}
                assert len(set(row)) == (1 if mode == "local_duplicates" else 2)
        golden = _notify["build_notify_golden"](routes, 32, 256)
        assert sum(golden["rank_rows"]) == 16 * 3 * 4
        assert not any(golden["gather_rows"])
        for rank, plan in enumerate(golden["plans"]):
            assert not any(plan["chunk_masks"])
            for records in plan["forward_list"]:
                for _, _, primary, peer, row, gather in records:
                    assert peer == rank and row != primary and gather == -1


def test_encoded_weight_metadata():
    for world in (2, 16):
        tokens, topk, experts = 7, 6, world * 2
        for mode in _ROUTES:
            routes = _notify["make_routes"](world, tokens, topk, experts, 0, mode)
            all_ids = []
            for rank in range(world):
                plan = build_reference(routes, experts, rank)
                ids = plan["weight_return_meta"]
                decoded = [(value // (tokens * topk), value // topk % tokens, value % topk) for value in ids]
                assert decoded == _notify["expert_rows"](routes, experts, rank)
                all_ids.extend(ids)
            expected = [
                (source * tokens + token) * topk + slot
                for source in range(world)
                for token in range(tokens)
                for slot, expert in enumerate(routes[source][token])
                if 0 <= expert < experts
            ]
            assert sorted(all_ids) == expected
            assert len(all_ids) == len(set(all_ids))


def test_sq_pressure_routes():
    routes = _notify["make_routes"](128, 8, 1, 128, 0, "sq_pressure")
    active = [rank for rank, rows in enumerate(routes) if rows[0] == [0]]
    assert active == [8, 20, 48, 52, 80, 84, 112, 116]
    assert all(row == ([0] if rank in active else [-1]) for rank, rows in enumerate(routes) for row in rows)
    reference = build_reference(routes, 128, 0)
    assert reference["rank_rows"] == [64] + [0] * 127


def test_large_shape_and_capacity_bounds(monkeypatch):
    import importlib
    import pytest
    from deep_ep.buffers._elastic_layout import combine_layout, dispatch_layout, elastic_size_hint

    elastic = importlib.import_module("deep_ep.buffers.elastic")
    monkeypatch.setattr(elastic, "require_native", lambda *_: None)
    for world in (256, 512):
        for tokens in (4608, 4609, 8192, 16383, 16384):
            for topk in (6, 16):
                rows = 4 * tokens * topk
                for fp8 in (False, True):
                    d = dispatch_layout(world, tokens, topk, 1024, rows, fp8, weights=True)
                    c = combine_layout(world, tokens, topk, rows, rows)
                    size = elastic_size_hint(world, tokens, topk, 1024, rows, rows, fp8)
                    assert max(d["workspace_bytes"], c["workspace_bytes"]) <= size // 2
                    assert size % (2 << 20) == 0
                    assert (
                        elastic.ElasticBuffer.get_buffer_size_hint(
                            None, tokens, 7168, topk, use_fp8_dispatch=fp8, world_size=world
                        )
                        == size
                    )
                    worst = dispatch_layout(world, tokens, topk, 1024, world * tokens * topk, fp8)
                    assert world * worst["address_stride"] < 2**63
    # Preserve the dense returned_partial layout, with full-width byte offsets.
    c = combine_layout(512, 16384, 16, 1, 1)
    assert c["output"] - c["returned_partial"] == 14 << 30
    assert elastic_size_hint(512, 16384, 16, 1024, 1048576, 1048576) > 32 << 30
    for world, tokens in ((1024, 16384), (512, 16385)):
        with pytest.raises(ValueError):
            dispatch_layout(world, tokens, 6, 1024, 1)
        with pytest.raises(ValueError):
            combine_layout(world, tokens, 6, 1, 1)


def test_hidden_layouts_match_native(tmp_path):
    import shutil
    import subprocess
    from deep_ep.buffers._elastic_layout import combine_layout, dispatch_layout, elastic_size_hint, validate_hidden

    for width in (0, 15, 17, 7169, 7296, True):
        with pytest.raises(ValueError, match="hidden"):
            validate_hidden(width)
    with pytest.raises(ValueError, match="128"):
        validate_hidden(144, True)
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    source = tmp_path / "hidden_layout.cpp"
    source.write_text(
        r"""

#include <iostream>
#include "kernels/dispatch/tiling.hpp"
#include "kernels/combine/tiling.hpp"
int main()
{
    for (unsigned world : {2U, 8U, 128U, 256U, 512U})
        for (unsigned h : {16U, 48U, 128U, 144U, 1024U, 4096U, 6144U, 7040U, 7152U, 7168U})
            for (bool fp8 : {false, true}) {
                if (fp8 && h % 128) continue;
                auto d = ascend_deepep::MakeDispatchTiling(0, world, 33, 6, 1024, 128, fp8, true, h);
                auto c = ascend_deepep::MakeCombineTiling(0, world, 33, 6, 128, 256, 256, h);
                std::cout << world << ' ' << h << ' ' << fp8 << ' ' << d.input << ' ' << d.output << ' ' << d.weights
                          << ' ' << d.scales << ' ' << d.inbox << ' ' << d.inbox_stride << ' ' << d.workspace_bytes
                          << ' ' << c.gather_input_offset_bytes << ' ' << c.server_partial_offset_bytes << ' '
                          << c.returned_partial_offset_bytes << ' ' << c.output_offset_bytes << ' '
                          << c.weight_recv_offset_bytes << ' ' << ascend_deepep::CombineWorkspaceBytes(c) << '\n';
            }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("hidden_layout.exe" if sys.platform == "win32" else "hidden_layout")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I" + str(Path(__file__).resolve().parents[2] / "csrc"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=60,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
    for line in result.stdout.splitlines():
        world, width, fp8, *actual = map(int, line.split())
        d = dispatch_layout(world, 33, 6, 1024, 128, bool(fp8), True, width)
        c = combine_layout(world, 33, 6, 128, 256, 256, width)
        expected = [d[k] for k in ("input", "output", "weights", "scales", "inbox", "inbox_stride", "workspace_bytes")]
        expected += [
            c[k]
            for k in (
                "gather_input",
                "server_partial",
                "returned_partial",
                "output",
                "weight_recv",
                "workspace_bytes",
            )
        ]
        assert actual == expected, (world, width, fp8)
        assert d["output"] - d["input"] == (33 * width * (1 if fp8 else 2) + 511) // 512 * 512
        assert c["gather_input"] == c["control_slot_num"] * 512
        assert c["server_partial"] - c["gather_input"] == 256 * width * 2
        size = elastic_size_hint(world, 33, 6, 1024, 128, 256, bool(fp8), width)
        assert max(d["workspace_bytes"], c["workspace_bytes"]) <= size // 2


def test_ep512_reference_high_ranks():
    # Independent planners must agree across the half-world boundary and rank511.
    for mode in ("normal", "random", "duplicates", "invalid", "empty", "local_experts"):
        routes = _notify["make_routes"](512, 3, 6, 1024, 0, mode)
        golden = _notify["build_notify_golden"](routes, 1024, 256)
        for rank in (0, 255, 256, 511):
            plan = build_reference(routes, 1024, rank)
            expected = golden["plans"][rank]
            for name in (
                "forward_list",
                "backward_list",
                "forward_counts",
                "backward_counts",
                "server_mask",
                "rank_rows",
                "gather_rows",
                "chunk_masks",
            ):
                assert plan[name] == expected[name], (mode, rank, name)
            assert plan["chunk_ranges"] == [
                value for source in expected["chunk_ranges"] for pair in source for value in pair
            ]
            ids = plan["weight_return_meta"]
            assert [(i // 18, i // 6 % 3, i % 6) for i in ids] == _notify["expert_rows"](routes, 1024, rank)


def test_dispatch_check_is_opt_in(monkeypatch, tmp_path):
    import pytest

    calls = []
    monkeypatch.setitem(_notify, "_launch", lambda args, worker: calls.append(args))
    for flag in (None, "--with-dispatch", "--check-dispatch"):
        argv = ["test_combine.py", "--output", str(tmp_path), "--repeat", "1"]
        if flag:
            argv.append(flag)
        monkeypatch.setattr(sys, "argv", argv)
        main()
        assert calls[-1].check_dispatch is bool(flag)
    monkeypatch.setattr(sys, "argv", ["test_combine.py", "--output", str(tmp_path), "--dispatch-dtype", "fp8"])
    with pytest.raises(SystemExit):
        main()
    # Even invalid prerequisite objects must not trigger the payload oracle by default.
    check_dispatch_inputs(False, *([None] * 13))


@pytest.mark.parametrize("notify_cores", [32, 64])
@pytest.mark.parametrize("rank_dedup", [False, True])
def test_routing_without_dispatch_and_explicit_opt_in(notify_cores, rank_dedup):
    from types import SimpleNamespace
    from deep_ep.buffers.elastic import _PERSISTENT

    tensor = SimpleNamespace(data_ptr=lambda: 123, _version=0)
    handle = SimpleNamespace(metadata_only=True, **{name: tensor for name in ("topk_idx", *_PERSISTENT, "_peer_plan")})
    ready = object()
    calls = []

    class Buffer:
        def _notify(self, **kwargs):
            calls.append("notify")
            assert kwargs["x"] == "hidden" and kwargs["topk_idx"] == "routes"
            assert kwargs["expand"] and kwargs["topk_weights"] is None
            assert kwargs["cores"] == notify_cores
            assert kwargs["rank_dedup"] == rank_dedup
            return None, None, None, handle, ready

        def capture(self):
            return ready

        def dispatch(self, *args, **kwargs):
            calls.append("dispatch")
            assert options.check_dispatch, "Default Combine test must never execute payload Dispatch"
            assert kwargs["num_max_tokens_per_rank"] == options.tokens
            return "received", None, "weights", handle, ready

    buffer = Buffer()
    options = SimpleNamespace(
        check_dispatch=False, experts=16, tokens=3, notify_num_cores=notify_cores, no_server_dedup=rank_dedup
    )
    result = prepare_routing(buffer, options, "hidden", "routes", None, None)
    assert result == (None, None, handle, ready)
    assert calls == ["notify"] and not handle.metadata_only
    assert handle._routing_versions == {name: (123, 0) for name in ("topk_idx", *_PERSISTENT, "_peer_plan")}
    assert prepare_routing(buffer, options, "hidden", None, None, handle) == result
    assert calls == ["notify"]  # Cached generations reuse metadata without Dispatch or another Notify.
    # A Combine-only Session omits the Dispatch peer plan entirely.
    del handle._peer_plan
    assert prepare_routing(buffer, options, "hidden", "routes", None, None) == result
    assert "_peer_plan" not in handle._routing_versions
    calls.clear()
    options.check_dispatch = True
    assert prepare_routing(buffer, options, "hidden", "routes", "weights", None) == (
        "received",
        "weights",
        handle,
        ready,
    )
    assert calls == ["dispatch"]


@pytest.mark.parametrize("world", (8, 32, 64))
@pytest.mark.parametrize("mode", ("random", "duplicates", "empty"))
def test_golden_routing_skips_device_notify(monkeypatch, world, mode):
    torch = pytest.importorskip("torch")
    import deep_ep._native as native_module
    import deep_ep.buffers.elastic as elastic

    args = SimpleNamespace(
        tokens=9, topk=6, experts=world * 2, chunk_tokens=256, hidden=7168, check_dispatch=False, golden_notify=True
    )
    routes = _notify["make_routes"](world, args.tokens, args.topk, args.experts, 0, mode, seed=2026)
    reference = build_reference(routes, args.experts, world - 1, args.chunk_tokens)
    chunks = len(reference["chunk_ranges"]) // (2 * world)
    specs = {
        "forward_list": ((world, args.tokens * args.topk, 6), "int32"),
        "backward_list": ((min(world, 8), world * args.tokens * args.topk, 3), "int64"),
        "server_mask": ((args.tokens,), "int64"),
        "dst": ((args.tokens, args.topk), "int64"),
        "expert_totals": ((args.experts,), "int32"),
        "weight_return_meta": ((max(1, reference["rank_rows"][world - 1]),), "int32"),
        "chunk_ranges": ((world, chunks, 2), "int32"),
        "chunk_masks": ((world // min(world, 64), chunks), "int32"),
    }
    for name in ("forward_counts", "backward_counts", "rank_rows", "gather_rows"):
        specs[name] = ((len(reference[name]),), "int32")
    native = SimpleNamespace(notify_layout=lambda *a, **kw: {"tensors": specs})
    monkeypatch.setattr(native_module, "require_native", lambda *a: native)
    monkeypatch.setattr(elastic, "require_native", lambda *a: native)
    ready = object()
    buffer = SimpleNamespace(
        _runtime=SimpleNamespace(rank=world - 1, world_size=world), _handle_token=object(), capture=lambda: ready
    )
    indices = torch.tensor(routes[world - 1], dtype=torch.int64)
    _, _, handle, event = prepare_routing(buffer, args, None, indices, None, None, reference, routes)
    check_handle(handle, reference, world - 1)
    assert event is ready and isinstance(handle, elastic.EPHandle)
    oracle = _notify["build_notify_golden"](routes, args.experts, args.chunk_tokens)
    assert handle.rank_rows.tolist() == oracle["rank_rows"]
    assert handle.gather_rows.tolist() == oracle["gather_rows"]
    assert handle.num_expanded_tokens == sum(handle.num_recv_tokens_per_expert_list)
    assert prepare_routing(buffer, args, None, None, None, handle) == (None, None, handle, ready)


def test_notify_checked_without_payload_dispatch(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pytest

    args = SimpleNamespace(
        output=tmp_path,
        route="random",
        tokens=1,
        hidden=7168,
        topk=1,
        experts=2,
        seed=1,
        chunk_tokens=256,
        repeat=1,
        check_dispatch=False,
        dispatch_dtype="bf16",
        without_weights=True,
    )
    reference, handle = object(), object()
    monkeypatch.setitem(_notify, "make_routes", lambda *a, **k: [[[0]], [[1]]])
    monkeypatch.setitem(_notify, "expert_rows", lambda *a: [])
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            int64="int64",
            bfloat16="bf16",
            tensor=lambda *a, **k: None,
            empty=lambda *a, **k: None,
            arange=lambda *a: 0,
        ),
    )
    event = SimpleNamespace(current_stream_wait=lambda: None)
    monkeypatch.setattr(sys.modules[__name__], "prepare_routing", lambda *a: (None, None, handle, event))

    def reject(actual, expected, rank):
        assert actual is handle and expected is reference and rank == 0
        raise AssertionError("Notify metadata corruption detected before Combine")

    monkeypatch.setattr(sys.modules[__name__], "check_handle", reject)
    with pytest.raises(AssertionError, match="Notify metadata corruption"):
        run_case(None, args, 0, 2, None, lambda message: None, {"random": reference})


def test_direct_expert_inputs_match_combine_golden():
    import pytest

    torch = pytest.importorskip("torch")
    for world in (2, 16):
        experts, tokens, topk = world * 2, 3, 6
        for mode in _ROUTES:
            routes = _notify["make_routes"](world, tokens, topk, experts, 0, mode)
            for generation in (0, 1):
                recovered = torch.zeros((world, tokens, 7168), dtype=torch.float32)
                for rank in range(world):
                    entries = _notify["expert_rows"](routes, experts, rank)
                    values = direct_expert_input(entries, generation, "cpu")
                    assert tuple(values.shape) == (len(entries), 7168)
                    for row, (source, token, slot) in enumerate(entries):
                        assert torch.equal(values[row], payload(source, tokens, generation)[token].float())
                        bias = routes[source][token][slot] % 3 - 1 + slot % 3 - 1
                        recovered[source, token] += (values[row] + bias).to(torch.bfloat16).float()
                for source in range(world):
                    assert torch.equal(
                        recovered[source].to(torch.bfloat16), expected_output(routes, experts, source, generation)
                    )


def test_ep256_combine_cli(monkeypatch, tmp_path):
    import pytest

    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_combine.py",
            "--ep-size",
            "256",
            "--nproc-per-node",
            "8",
            "--node-rank",
            "31",
            "--tokens",
            "4096",
            "--topk",
            "6",
            "--experts",
            "512",
            "--route",
            "random",
            "--seed",
            "2026",
            "--without-weights",
            "--profile-op",
            "combine",
            "--output",
            str(tmp_path / "ep256"),
        ],
    )
    monkeypatch.setitem(_notify, "_launch", lambda args, worker: calls.append(args))
    main()
    (args,) = calls
    assert (args.ep_size, args.node_rank, args.tokens, args.topk, args.experts) == (256, 31, 4096, 6, 512)
    assert not args.check_dispatch
    assert args.without_weights and args.profile_op == "combine"
    assert args.route == "random" and args.seed == 2026
    # Non-integer seeds fail before any device launch.
    monkeypatch.setattr(sys, "argv", ["test_combine.py", "--seed", "bad", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and len(calls) == 1


def test_rank_return_row_codec(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    source = tmp_path / "return_row.cpp"
    source.write_text(
        r"""

#include "kernels/combine/tiling.hpp"
#include <cassert>
#include <cstdint>
#include <initializer_list>
int main()
{
    using ascend_deepep::CombineReturnRow;
    for (uint32_t world : {2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U, 512U}) {
        for (uint32_t topk : {1U, 6U, 16U}) {
            uint32_t servers = ascend_deepep::CombineReturnSlots(world, topk);
            assert(servers >= (world + 7U) / 8U && servers >= topk);
            auto t = ascend_deepep::MakeCombineTiling(0, world, 16384, topk, 1, 1, 512);
            assert(t.output_offset_bytes - t.returned_partial_offset_bytes == uint64_t(16384) * servers * 7168 * 2);
            for (uint32_t token : {0U, 63U, 255U, 511U, 4095U, 16383U}) {
                for (uint32_t rank = 0; rank < world; ++rank) {
                    assert(CombineReturnRow(token, servers, rank) == uint64_t(token) * servers + rank / 8);
                    for (uint32_t slot = 0; slot < servers; ++slot) {
                        uint32_t row = token * servers + slot;
                        assert(CombineReturnRow(-int32_t(row) - 1, servers, rank) == row);
                    }
                }
            }
        }
    }
    assert(CombineReturnRow(INT32_MIN, 64, 511) == uint64_t(INT32_MAX));
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("return_row.exe" if sys.platform == "win32" else "return_row")
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(Path(__file__).parents[2] / "csrc"), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True)


def test_combine_reduction_predicates(tmp_path):
    """Execute the kernel's actual predicates on Host; does not test NPU synchronization."""
    import importlib.util
    import shutil
    import subprocess
    import sys

    import pytest

    root = Path(__file__).resolve().parents[2]
    source = root / "csrc/kernels/combine/combine.cpp"
    if not source.is_file():
        pytest.skip("Kernel source test runs before packaging")
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler:
        command = [compiler]
    elif importlib.util.find_spec("ziglang"):
        command = [sys.executable, "-m", "ziglang", "c++"]
    else:
        pytest.skip("Host predicate test requires a C++17 compiler")
    code = source.read_text(encoding="utf-8")

    def section(begin, end):
        return code[code.index(begin) : code.index(end, code.index(begin))]

    helpers = section("__aicore__ inline uint32_t CombineGroup(", "__aicore__ inline uint32_t CombinePeer(")
    helpers += section("__aicore__ inline uint32_t CrossPeerOwner(", "__aicore__ inline uint32_t SelfCopyDoneSlot(")
    helpers += r"""

void cross_peer_plan_test()
{
    // The old full-rank scan is the oracle, including ascending peer order,
    // sparse QP availability and the EP64 alternating-owner fallback.
    for (uint32_t world : {8U, 16U, 32U, 64U, 128U, 256U, 512U})
        for (uint32_t tokens : {1U, 257U})
            for (uint32_t rank = 0; rank < world; ++rank) {
                auto t = ascend_deepep::MakeCombineTiling(rank, world, tokens, 1, 1, 1, 256);
                for (uint32_t owners : {8U, 16U}) {
                    t.num_cross_put_cores = owners;
                    for (uint32_t owner = 0; owner < owners; ++owner) {
                        std::vector<uint32_t> expected;
                        for (uint32_t peer = 0; peer < world; ++peer)
                            if (peer / 8 != rank / 8 && CrossPeerAssigned(t, peer, owner)) expected.push_back(peer);
                        uint32_t storage[130];
                        std::fill(storage, storage + 130, UINT32_MAX);
                        const auto count = BuildCrossPeerList(t, owner, storage + 1);
                        assert(count <= 128 && storage[0] == UINT32_MAX && storage[count + 1] == UINT32_MAX);
                        assert(std::vector<uint32_t>(storage + 1, storage + 1 + count) == expected);
                        if (owners == 16)
                            for (auto peer : expected) {
                                const auto pair = 3U << (owner * 2U);
                                assert((first_hit_schedule::CombineQpMask(rank, peer, world) & pair) == pair);
                                assert((first_hit_schedule::CombineQpMask(peer, rank, world) & pair) == pair);
                            }
                    }
                }
            }
}
namespace cross_ranges_test {
using bfloat16_t = uint16_t;
"""
    helpers += section("constexpr uint32_t kReduceTileElements =", "constexpr uint32_t kReduceOutputSlots =")
    helpers += r"""

enum class HardEvent { S_MTE2, MTE2_S };
constexpr uint32_t EVENT_ID2 = 2;
uint32_t event_state = 0, copies = 0;
template <HardEvent event>
void SetFlag(uint32_t id)
{
    assert(id == 2);
    if constexpr (event == HardEvent::S_MTE2) {
        assert(event_state == 0 || event_state == 5);
        event_state = 1;
    } else {
        assert(event_state == 3);
        event_state = 4;
    }
}
template <HardEvent event>
void WaitFlag(uint32_t id)
{
    assert(id == 2);
    if constexpr (event == HardEvent::S_MTE2) {
        assert(event_state == 1);
        event_state = 2;
    } else {
        assert(event_state == 4);
        event_state = 5;
    }
}
template <class T>
struct LocalTensor {
    T* data;
};
template <class T>
struct GlobalTensor {
    T* data;
    GlobalTensor operator[](uint64_t i) const
    {
        return {data + i};
    }
};
struct DataCopyExtParams {
    uint16_t blocks;
    uint32_t bytes, src_stride, dst_stride, reserved;
    DataCopyExtParams(uint16_t n, uint32_t b, uint32_t s, uint32_t d, uint32_t r)
        : blocks(n), bytes(b), src_stride(s), dst_stride(d), reserved(r)
    {
    }
};
template <class T>
struct DataCopyPadExtParams {
    bool enabled;
    uint8_t left, right;
    T value;
};
const int32_t* gm_end;
void DataCopyPad(LocalTensor<int32_t> dst, GlobalTensor<int32_t> src, DataCopyExtParams p,
                 DataCopyPadExtParams<int32_t> pad)
{
    assert(event_state == 2 && p.blocks <= 512 && p.bytes == 8 && p.src_stride % 4 == 0);
    assert(p.dst_stride == 0 && p.reserved == 0 && !pad.enabled);
    for (uint32_t peer = 0; peer < p.blocks; ++peer) {
        const auto* begin = src.data + peer * (p.bytes + p.src_stride) / 4;
        assert(begin + 2 <= gm_end);
        std::copy(begin, begin + 2, dst.data + peer * 8);  // 8-byte blocks have a 32-byte UB stride.
    }
    ++copies;
    event_state = 3;
}
"""
    helpers += section(
        "__aicore__ inline void LoadCrossChunkRanges(",
        "template <bool EnableTrace, bool EnableProfile>\n__aicore__ inline void CrossSend(",
    )
    helpers += r"""

void run()
{
    for (uint32_t world : {128U, 256U, 512U})
        for (uint32_t tokens : {1U, 257U, 4096U, 16384U})
            for (uint32_t chunk_tokens : {256U, 512U, 1024U}) {
                auto t = ascend_deepep::MakeCombineTiling(0, world, tokens, 6, 1, 1, chunk_tokens);
                std::vector<int32_t> gm(world * t.chunks_per_group * 2);
                for (uint32_t i = 0; i < gm.size(); ++i) gm[i] = int32_t(i + 17);
                gm_end = gm.data() + gm.size();
                // Guard the exact existing 16 KiB BF16 bank, including the EP512 endpoint.
                std::vector<int32_t> ub(kReduceTileElements * 2 / 4 + 2, -123);
                GlobalTensor<int32_t> source{gm.data()};
                event_state = 0;
                copies = 0;
                for (uint32_t chunk = 0; chunk < t.chunks_per_group; ++chunk) {
                    LoadCrossChunkRanges({ub.data() + 1}, source, t, chunk);
                    assert(event_state == 5 && ub.front() == -123 && ub.back() == -123);
                    for (uint32_t peer = 0; peer < world; ++peer)
                        for (uint32_t field = 0; field < 2; ++field)
                            assert(ub[1 + peer * 8 + field] == gm[(peer * t.chunks_per_group + chunk) * 2 + field]);
                }
                assert(copies == t.chunks_per_group);
            }
}
}  // namespace cross_ranges_test
"""
    helpers += section(
        "template <typename TaskReader>\n__aicore__ inline bool TokenNeedsGatewayReduce(",
        "template <bool EnableTrace, bool EnableProfile>\n__aicore__ inline void ReduceGateway(",
    )
    helpers += section("constexpr uint32_t kWeightThreads =", "__aicore__ inline __gm__ uint64_t* WeightFlag(")
    # Compile the production dispatch schedule too: Notify and Combine must
    # agree with it for every high-rank source/destination and credit edge.
    dispatch_code = (root / "csrc/kernels/dispatch/dispatch.cpp").read_text(encoding="utf-8")
    helpers += "\nnamespace dispatch_contract {\n"
    helpers += dispatch_code[
        dispatch_code.index("constexpr uint32_t kMaxDispatchGroupWidth") : dispatch_code.index(
            "// 128P uses the frozen"
        )
    ]
    helpers += dispatch_code[
        dispatch_code.index("__aicore__ inline uint32_t GetDispatchGroupWidth(") : dispatch_code.index(
            "__aicore__ inline void DispatchSplitRange("
        )
    ]
    helpers += "\n}\n"

    def function(source, name):
        begin = source.index("__simt_vf__", source.rfind("\n", 0, source.index("inline void " + name)) + 1)
        return source[begin : source.index("\n}\n", begin) + 3]

    notify = (root / "csrc/kernels/notify/notify.cpp").read_text(encoding="utf-8")
    scale = (root / "csrc/kernels/dispatch/scale_pipeline.hpp").read_text(encoding="utf-8")
    helpers += "\nnamespace destination_test {\nconstexpr uint32_t kNotifySimtThreads = 128, kSimtThreads = 128;\n"
    helpers += "constexpr uint32_t kTileTokens = ascend_deepep::kNotifyLocalTopkTileTokens, kMetaFields = 6;\n"
    helpers += notify[notify.index("constexpr uint32_t kWireRankMask") : notify.index("template <uint32_t CoreCount")]
    helpers += "template <uint32_t CoreCount, bool RankDedup = false>\n" + function(notify, "ParseLocalTopkTokensVf")
    helpers += "template <bool CachedHist, uint32_t CoreCount, bool RankDedup>\n" + function(notify, "ParseLocalTopkVf")
    helpers += "template <uint32_t CoreCount>\n" + function(notify, "NotifyResolveDstVf")
    helpers += function(scale, "Decode")
    helpers += "\n}\n" + function(code, "LoadGatewayPageVf")
    helpers += r"""

template <uint32_t Cores>
void local_gateway_parse_test()
{
    using namespace destination_test;
    for (uint32_t world : {8U, 16U, 32U, 64U, 128U, 256U, 512U}) {
        constexpr uint32_t tokens = 35, topk = 6;
        const uint32_t experts = world * 2, stride = (experts + world + 15) / 16 * 16;
        for (uint32_t rank : {0U, world / 2, world - 1}) {
            std::vector<int64_t> indices(tokens * topk), masks(tokens), fallback_masks(tokens), cached_masks(tokens);
            std::vector<int32_t> meta(tokens * topk * 6), fallback(meta.size()), cached(meta.size());
            std::vector<int32_t> hist(3 * stride), cache(16 * topk * 2), hist_ub(stride);
            for (uint32_t t = 0; t < tokens; ++t) {
                const uint32_t other = rank / 8 * 8 + (rank + 1) % 8, remote = (rank + 8) % world;
                int64_t row[topk] = {
                    int64_t(other * 2), int64_t(remote * 2), int64_t(rank * 2 + 1), int64_t(rank * 2), -1,
                    int64_t(other * 2)};
                if (t % 3 == 0) row[2] = row[3] = -1;
                if (t % 3 == 1) row[4] = experts;
                for (uint32_t k = 0; k < topk; ++k) indices[t * topk + k] = t == tokens - 1 ? -1 : row[k];
            }
            const auto original = indices;
            for (uint32_t core = 0; core < Cores; ++core) {
                std::fill(hist_ub.begin(), hist_ub.end(), 0);
                for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x) {
                    ParseLocalTopkTokensVf<Cores>(indices.data(), masks.data(), meta.data(), rank, world, tokens, topk,
                                                  experts, 256, core, cache.data());
                    ParseLocalTopkVf<false, Cores, false>(indices.data(), fallback_masks.data(), fallback.data(),
                                                          hist.data(), rank, world, tokens, topk, experts, 256, core);
                    ParseLocalTopkVf<true, Cores, false>(indices.data(), cached_masks.data(), cached.data(),
                                                         hist.data(), rank, world, tokens, topk, experts, 256, core,
                                                         hist_ub.data());
                }
            }
            assert(indices == original && masks == fallback_masks && masks == cached_masks);
            assert(fallback == cached);
            for (uint32_t t = 0; t < tokens; ++t)
                for (uint32_t k = 0; k < topk; ++k) {
                    const int64_t value = indices[t * topk + k];
                    const int32_t expert = value >= 0 && value < experts ? int32_t(value) : -1;
                    int32_t primary = -1;
                    if (expert >= 0) {
                        for (uint32_t j = 0; j < topk; ++j) {
                            const auto e = indices[t * topk + j];
                            if (e < 0 || e >= experts || e / 16 != expert / 16) continue;
                            if (primary < 0) primary = j;
                            if (e / 2 == rank) {
                                primary = j;
                                break;
                            }
                        }
                    }
                    const auto* row = meta.data() + (t * topk + k) * 6;
                    assert(row[0] == expert && row[2] == primary);
                    assert(row[1] == (primary < 0 ? -1 : indices[t * topk + primary] / 2));
                    assert(row[5] == (primary < 0 ? 0 : primary == int32_t(k) ? 7 : 9));
                    for (uint32_t field : {0U, 1U, 2U, 3U, 5U})
                        assert(row[field] == fallback[(t * topk + k) * 6 + field]);
                }
        }
    }
}
void large_shape_test()
{
    local_gateway_parse_test<32>();
    local_gateway_parse_test<64>();
    for (uint32_t world : {8U, 16U, 512U}) {
        using namespace destination_test;
        constexpr uint32_t tokens = 35, topk = 16;
        const uint32_t experts = world * 2, stride = (experts + world + 15) / 16 * 16;
        std::vector<int64_t> indices(tokens * topk), masks(tokens, -1), fallback_masks(tokens, -1);
        std::vector<int32_t> meta(tokens * topk * 6, -97), fallback(meta.size(), -97), cache(3 * 16 * topk * 2, -97);
        std::vector<int32_t> hist(3 * stride, -97);
        for (bool empty : {false, true}) {
            for (uint32_t t = 0; t < tokens; ++t)
                for (uint32_t k = 0; k < topk; ++k)
                    indices[t * topk + k] = empty || (t + k) % 5 == 0 ? -1 : (t * 11 + k % 4 * 2) % experts;
            for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
                ParseLocalTopkTokensVf<1, true>(indices.data(), masks.data(), meta.data(), 0, world, tokens, topk,
                                                experts, 256, 0, cache.data());
            for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
                ParseLocalTopkVf<false, 1, true>(indices.data(), fallback_masks.data(), fallback.data(), hist.data(), 0,
                                                 world, tokens, topk, experts, 256, 0);
            assert(masks == fallback_masks);
            for (uint32_t t = 0; t < tokens; ++t) {
                std::vector<int32_t> peers, primaries;
                for (uint32_t k = 0; k < topk; ++k) {
                    int32_t expert = indices[t * topk + k], primary = -1, slot = -1;
                    if (expert >= 0) {
                        auto hit = std::find(peers.begin(), peers.end(), expert / 2);
                        slot = hit - peers.begin();
                        if (hit == peers.end()) {
                            peers.push_back(expert / 2);
                            primaries.push_back(k);
                        }
                        primary = primaries[slot];
                    }
                    auto* row = meta.data() + (t * topk + k) * 6;
                    assert(row[0] == expert && row[1] == (expert < 0 ? -1 : expert / 2));
                    assert(row[2] == primary && row[3] == slot);
                    assert(row[5] == (expert < 0 ? 0 : primary == int32_t(k) ? 7 : 9));
                    for (uint32_t field : {0U, 1U, 2U, 3U, 5U})
                        assert(row[field] == fallback[(t * topk + k) * 6 + field]);
                }
                assert(masks[t] == int64_t((1U << peers.size()) - 1));
            }
        }
    }
    {
        using namespace destination_test;
        int64_t topk[6]{0, 1023, 1016, -1, 512, 1023}, mask = 0;
        int32_t meta[36]{}, cache[192]{};
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            ParseLocalTopkTokensVf<1>(topk, &mask, meta, 0, 512, 1, 6, 1024, 256, 0, cache);
        assert(uint64_t(mask) == ((1ULL << 63) | (1ULL << 32) | 1ULL));
        assert(meta[6 + 1] == 511 && meta[12 + 1] == 511 && meta[12 + 2] == 1 && meta[18] == -1);
        for (uint32_t rank = 0; rank < 512; ++rank)
            for (uint32_t k = 0; k < 16; ++k) {
                const uint32_t word = rank | (k << kWirePrimaryTopkShift) | ((15 - k) << kWireSecondaryTopkShift);
                assert((word & kWireRankMask) == rank);
                assert(((word >> kWirePrimaryTopkShift) & kWireTopkMask) == k);
                assert(((word >> kWireSecondaryTopkShift) & kWireTopkMask) == 15 - k);
            }
    }
    constexpr uint32_t n = 6;
    int32_t meta[n * 6]{}, bases[512]{}, stride = 1 << 27;
    int64_t encoded[n]{};
    for (uint32_t i = 0; i < n; ++i) {
        meta[i * 6] = i == 5 ? -1 : (i == 4 ? 256 : 511);
        meta[i * 6 + 4] = 123 + i;
        meta[i * 6 + 5] = i % 2 ? 0 : 2;
    }
    for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
        destination_test::NotifyResolveDstVf<1>(meta, bases, &stride, encoded, 512, 512, n, 0);
    assert(encoded[0] == int64_t(511) * stride + 123 && encoded[5] == INT64_MIN);
    assert(encoded[1] == ~(int64_t(511) * stride + 124));
    int32_t ranks[n]{}, rows[n]{}, indices[n]{};
    for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
        destination_test::Decode(encoded, ranks, rows, indices, n, 0, stride, 27);
    for (uint32_t i = 0; i < n; ++i) {
        assert(ranks[i] == (i == 5 ? -1 : (i == 4 ? 256 : 511)));
        assert(rows[i] == (i == 5 ? -1 : 123 + int32_t(i)) && indices[i] == int32_t(i));
    }
    for (uint32_t world : {128U, 256U, 512U}) {
        const uint32_t chunks = 67, groups = world / 64, page = ascend_deepep::GatewayControlChunks(world, chunks);
        const uint32_t mask_offset = (world * page * 2 + 7) / 8 * 8;
        assert((mask_offset + (groups * page + 7) / 8 * 8) * 4 <= 44 * 1024);
        std::vector<int32_t> ranges(world * chunks * 2), masks(groups * chunks),
            out(mask_offset + groups * page + 8, -123);
        for (uint32_t i = 0; i < ranges.size(); ++i) ranges[i] = int32_t(i + 17);
        for (uint32_t i = 0; i < masks.size(); ++i) masks[i] = int32_t(i % 256);
        for (uint32_t begin = 0; begin < chunks; begin += page) {
            for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
                LoadGatewayPageVf(out.data(), ranges.data(), masks.data(), world, groups, chunks, page, begin,
                                  mask_offset);
            for (uint32_t r = 0; r < world; ++r)
                for (uint32_t c = 0; c < page; ++c)
                    for (uint32_t k = 0; k < 2; ++k)
                        assert(out[(r * page + c) * 2 + k] ==
                               (begin + c < chunks ? ranges[(r * chunks + begin + c) * 2 + k] : 0));
            for (uint32_t g = 0; g < groups; ++g)
                for (uint32_t c = 0; c < page; ++c)
                    assert(out[mask_offset + g * page + c] == (begin + c < chunks ? masks[g * chunks + begin + c] : 0));
            for (uint32_t i = mask_offset + groups * page; i < out.size(); ++i) assert(out[i] == -123);
        }
    }
    for (uint32_t rank = 0; rank < 512; ++rank)
        for (uint32_t owner = 0; owner < 8; ++owner) {
            bool seen[64]{};
            for (uint32_t i = 0; i < 64; ++i) {
                auto peer = ascend_deepep::scale_clos::Peer(rank, 512, owner, i);
                assert(peer < 512 && peer % 8 == owner && !seen[peer / 8]);
                seen[peer / 8] = true;
            }
        }
    const auto scale = ascend_deepep::scale_pipeline::Make(512, 224);
    const auto weight = ascend_deepep::aux_pipeline::Make(512, 0);
    assert(scale.peers == 16 && scale.batch == 8 && scale.bytes <= 132096);
    assert(weight.peers == 16 && weight.batch && weight.bytes <= 132096);
}
"""
    transport = (root / "csrc/kernels/dispatch/peer_transport.hpp").read_text(encoding="utf-8")
    credit_parts = []
    for declaration in (
        "constexpr uint64_t kClosPollCyclesPerUs =",
        "constexpr uint64_t kClosWaitTimeoutCycles =",
    ):
        begin = transport.index(declaration)
        credit_parts.append(transport[begin : transport.index(";", begin) + 1])
    for signature in (
        "__aicore__ inline void PauseClosAckPoll(",
        "__aicore__ inline bool EnsureClosSqCredit(",
    ):
        begin = transport.index(signature)
        credit_parts.append(transport[begin : transport.index("\n}\n", begin) + 3])
    credit_helper = "\n".join(credit_parts)
    helpers += (
        r"""

namespace dispatch_credit_test {
namespace AscendC {
uint64_t cycles = 0;
uint64_t GetSystemCycle()
{
    return cycles++;
}
template <unsigned n>
void Nop()
{
    cycles += n;
}
}
struct aclshmemi_udma_queue_state_t {
    uint32_t sq_head = 0, sq_tail = 0, cqe_cnt = 0, cq_tail = 0;
} queue;
struct DeepepSecondaryState {
    uint64_t clos_relay_mailbox = 0, magic = 17;
    uint32_t rank = 0, world = 512, clos_table_owner = 0;
    uint8_t* sync = nullptr;
};
constexpr uint32_t MAX_RETRY_TIMES = 100, PIPE_MTE3 = 0, EVENT_ID0 = 0, kUdmaSyncSoCqeConfig = 0,
                   kPeerCompletionValueSlot = 0;
uint64_t mailbox[1024]{}, sink = 0;
uint32_t ack_reads = 0, checkpoints = 0, phase_now = 0;
bool ack_acquired = false;
bool aclshmemi_udma_sq_has_credit(aclshmemi_udma_queue_state_t* q, uint32_t n)
{
    return uint32_t(q->sq_head - q->sq_tail) + n + 10 < 32768;
}
uint64_t* ClosRelayAssistOwner(uint64_t, uint32_t)
{
    return mailbox;
}
bool ClosRelayAssistGeneration(uint64_t*, uint64_t)
{
    ++ack_reads;
    ack_acquired = ack_reads >= 3;
    return ack_acquired;
}
template <class... T>
void aclshmemi_kernel_abort(const char*, T...)
{
    assert(false);
}
void trap()
{
    assert(false);
}
uint32_t GetGroupRankIndex(uint32_t p, uint32_t)
{
    return p;
}
uint64_t* GetPeerCompletionFlag(uint32_t, uint32_t, uint32_t, uint8_t*)
{
    return &sink;
}
uint32_t DeepepTransportQp(DeepepSecondaryState&, uint32_t, uint32_t q)
{
    return q;
}
template <class T, uint32_t, uint32_t>
void aclshmemx_udma_qp_put_nbi(T*, T*, T*, uint32_t, uint32_t, uint32_t, uint32_t)
{
    assert(!phase_now || ack_acquired);
    assert(aclshmemi_udma_sq_has_credit(&queue, 1));
    ++queue.sq_head;
    ++queue.cqe_cnt;
    ++checkpoints;
}
void aclshmemx_udma_qp_quiet(uint32_t, uint32_t)
{
    assert(!phase_now || ack_acquired);
    queue.sq_tail = queue.sq_head;
    queue.cq_tail = queue.cqe_cnt;
}
"""
        + credit_helper
        + r"""

void run()
{
    DeepepSecondaryState s;
    uint8_t scratch[512]{};
    assert(EnsureClosSqCredit(s, 511, 0, 0, &queue, 64, scratch));
    assert(!ack_reads && !checkpoints);
    // Two phases, each much larger than SQ; include uint32 counter wrap.
    queue.sq_head = queue.sq_tail = UINT32_MAX - 100;
    for (phase_now = 0; phase_now < 2; ++phase_now) {
        for (uint32_t batch = 0; batch < 4096; ++batch) {
            assert(EnsureClosSqCredit(s, 511, 0, phase_now, &queue, 64, scratch));
            queue.sq_head += 64;
            assert(uint32_t(queue.sq_head - queue.sq_tail) < 32768);
        }
        ++queue.sq_head;
        ++queue.cqe_cnt;  // The actual SO1/SO2, not a checkpoint.
    }
    assert(checkpoints > 2 && ack_reads >= 3 && ack_acquired && AscendC::cycles >= 2000);
}
}
"""
    )
    credit_test = (
        r"""

namespace credit_test {
constexpr uint32_t kWqeBytes = 64, kWqeBatchScratchBytes = 128 * kWqeBytes;
constexpr uint32_t ACLSHMEM_UDMA_AGGREGATE_CREDIT_GUARD = 10, EVENT_ID0 = 0;
namespace shm {
constexpr uint32_t UDMA_SQ_BASKBLK_CNT = 32768;
}
struct aclshmemi_udma_queue_state_t {
    uint32_t sq_head = 0, sq_tail = 0, cqe_cnt = 0, cq_tail = 0;
};
struct Context {
    uint32_t depth;
    uintptr_t state_addr, buf_addr;
} contexts[2];
aclshmemi_udma_queue_state_t queues[2];
uint32_t completed_head[2]{}, polls[2]{}, copies = 0, checkpoints[2]{};
uint8_t rings[2][32768 * kWqeBytes]{}, scratch[2 * kWqeBatchScratchBytes]{};
bool stuck = false, expect_failure = false;
bool failure_observed = false;
struct Address {
    uint8_t logicPos;
    uint64_t bufferAddr;
};
template <class T>
struct LocalTensor {
    Address address_{};
};
enum class TPosition { VECOUT };
int* aclshmemi_udma_qp_info_fetch()
{
    return nullptr;
}
Context* aclshmemi_udma_get_qp_ctx(int*, uint32_t, uint32_t qp)
{
    assert(qp == 16 || qp == 17);
    return &contexts[qp - 16];
}
void aclshmemx_udma_qp_quiet(uint32_t, uint32_t qp)
{
    auto lane = qp - 16;
    auto& q = queues[lane];
    ++polls[lane];
    if (!stuck) {
        q.sq_tail = completed_head[lane];
        q.cq_tail = q.cqe_cnt;
    }
}
template <class... Args>
void aclshmemi_kernel_abort(const char*, Args...)
{
    if (expect_failure) {
        failure_observed = true;
        return;
    }
    std::abort();
}
void aclshmemi_udma_copy_wqe_from_ub(uint8_t* dst, LocalTensor<uint8_t> src, uint32_t bytes, uint32_t)
{
    ++copies;
    std::memcpy(dst, reinterpret_cast<void*>(src.address_.bufferAddr), bytes);
}
void aclshmemi_udma_post_send_update_info(uint32_t head, Context*, aclshmemi_udma_queue_state_t* q)
{
    assert(uint32_t(head - q->sq_tail) < shm::UDMA_SQ_BASKBLK_CNT);
    q->sq_head = head;
}
constexpr uint32_t PIPE_MTE3 = 0, kCompletionSoConfig = 0;
template <class T, uint32_t, uint32_t>
void aclshmemx_udma_qp_put_nbi(T*, T*, T*, uint32_t n, uint32_t, uint32_t qp, uint32_t)
{
    assert(n == 1);
    auto lane = qp - 16;
    auto& q = queues[lane];
    assert(uint32_t(q.sq_head - q.sq_tail) + 1 + 10 < 32768);
    ++q.sq_head;
    ++q.cqe_cnt;
    ++checkpoints[lane];
    completed_head[lane] = q.sq_head;
}
struct Batch {
    static constexpr bool Enabled = false;
    struct ProfileStub {
        uint64_t Clock()
        {
            return 0;
        }
        void Add(uint32_t, uint64_t) {}
        void Elapsed(uint32_t, uint64_t) {}
        void ChunkSend(uint32_t, uint32_t, uint32_t) {}
    } profile_stub;
    ProfileStub* profile = &profile_stub;
    uint32_t direct_rows[2]{}, reduced_rows[2]{};
    struct {
        uint32_t pending_count = 0;
    } state[2];
    uint32_t peer[2]{8, 8}, so_reserve = 16;
    uint64_t *credit_source = nullptr, *credit_sink[2]{};
"""
        + section("    __aicore__ inline void FlushLane(", "    __aicore__ inline void Flush(")
        + r"""

}
;
void reset(uint32_t start = 0)
{
    stuck = false;
    copies = 0;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        queues[lane] = {start, start, 0, 0};
        completed_head[lane] = start;
        polls[lane] = 0;
        checkpoints[lane] = 0;
        contexts[lane] = {32768, reinterpret_cast<uintptr_t>(&queues[lane]), reinterpret_cast<uintptr_t>(rings[lane])};
    }
}
void expect_abort(Batch& b)
{
    const auto before = copies;
    expect_failure = true;
    failure_observed = false;
    b.FlushLane(0, 8, scratch);
    assert(failure_observed);
    expect_failure = false;
    assert(copies == before);
}
void simulate(uint32_t tokens, bool empty = false, uint32_t world = 128)
{
    auto t = ascend_deepep::MakeCombineTiling(0, world, tokens, 1, 8 * tokens, 1, 256);
    std::vector<uint32_t> peers;
    for (uint32_t p = 8; p < world; ++p)
        if (CrossPeerAssigned(t, p, 8)) peers.push_back(p);
    if (world == 128)
        assert((peers == std::vector<uint32_t>{8, 20, 48, 52, 80, 84, 112, 116}));
    else
        assert(peers.size() == world / 16);
    Batch b;
    b.so_reserve = 0;
    for (auto p : peers)
        for (uint32_t c = 0; c < t.chunks_per_group; ++c)
            if (CrossSoRequired(t, 0, p, c)) ++b.so_reserve;
    assert(b.so_reserve == peers.size() * std::min(3U, (tokens + 255U) / 256U));
    uint32_t payload[2]{};
    for (uint32_t c = 0; c < t.chunks_per_group; ++c) {
        auto begin = c ? std::min<uint64_t>(tokens, ChunkTokenEnd(t, c - 1)) : 0;
        auto end = std::min<uint64_t>(tokens, ChunkTokenEnd(t, c));
        for (uint32_t g = 0; g < world / t.group_size; ++g) {
            std::vector<uint32_t> group;
            for (auto p : peers)
                if (CombineGroup(0, world, p, t.group_size) == g) group.push_back(p);
            uint32_t left[2][8]{};
            assert(group.size() <= 8);
            if (!empty)
                for (uint32_t i = 0; i < group.size(); ++i)
                    for (uint32_t n = 0; n < end - begin; ++n) ++left[n & 1U][i];
            bool remaining = true;
            while (remaining) {
                remaining = false;
                for (uint32_t lane = 0; lane < 2; ++lane)
                    for (uint32_t i = 0; i < group.size(); ++i) {
                        auto n = std::min(left[lane][i], t.cross_tokens_per_turn);
                        left[lane][i] -= n;
                        for (uint32_t j = 0; j < n; ++j) {
                            ++payload[lane];
                            b.peer[lane] = group[i];
                            if (++b.state[lane].pending_count == 32) b.FlushLane(lane, 8, scratch);
                        }
                        remaining |= left[lane][i] != 0;
                    }
            }
            for (uint32_t lane = 0; lane < 2; ++lane) b.FlushLane(lane, 8, scratch);
            for (auto p : group)
                if (CrossSoRequired(t, 0, p, c))
                    for (uint32_t lane = 0; lane < 2; ++lane) {
                        auto& q = queues[lane];
                        assert(uint32_t(q.sq_head - q.sq_tail) + 1 + 10 < 32768);
                        ++q.sq_head;
                        ++q.cqe_cnt;
                        completed_head[lane] = q.sq_head;
                    }
        }
    }
    if (tokens == 6000 && !empty) assert(payload[0] == 24000 && payload[1] == 24000);
    if ((world == 128 && tokens <= 4096) || empty)
        assert(polls[0] == 0 && polls[1] == 0);
    else if (uint64_t(tokens / 2) * peers.size() >= 32768)
        assert(polls[0] > 0 && polls[1] > 0);
    // Final quiet must still drain both lanes before the next invocation.
    for (uint32_t lane = 0; lane < 2; ++lane) {
        aclshmemx_udma_qp_quiet(8, 16 + lane);
        assert(queues[lane].sq_head == queues[lane].sq_tail);
        polls[lane] = 0;
    }
}
void run()
{
    // Bound each no-SO interval for every supported token count at chunk=256.
    for (uint32_t world : {128U, 256U})
        for (uint32_t tokens = 1; tokens <= (world == 256 ? 4608U : 10240U); ++tokens) {
            auto t = ascend_deepep::MakeCombineTiling(0, world, tokens, 1, 1, 1, 256);
            uint32_t previous = 0, lane0 = 0;
            for (uint32_t c = 0; c < t.chunks_per_group; ++c) {
                auto end = std::min<uint64_t>(tokens, ChunkTokenEnd(t, c));
                auto count = end - previous;
                previous = end;
                lane0 += (count + 1) / 2;
                if (CrossSoRequired(t, 0, 8, c)) {
                    assert(lane0 <= 2560);
                    const uint32_t peers = world == 256 ? 16U : 8U;
                    assert(uint64_t(lane0) * peers + 32 + peers * 3 + 10 < 32768);
                    lane0 = 0;
                }
            }
            assert(lane0 == 0);
        }
    for (uint32_t world : {16U, 32U, 64U, 128U, 256U})
        for (uint32_t rank = 0; rank < world; ++rank) {
            auto t = ascend_deepep::MakeCombineTiling(rank, world, world == 256 ? 4608 : 10240, 1, 1, 1, 256);
            for (uint32_t owner = 0; owner < 16; ++owner) {
                uint32_t peers = 0, so = 0;
                for (uint32_t p = 0; p < world; ++p)
                    if (p / 8 != rank / 8 && CrossPeerAssigned(t, p, owner)) {
                        ++peers;
                        for (uint32_t c = 0; c < t.chunks_per_group; ++c)
                            if (CrossChunkOwner(t, rank, p, c) == owner && CrossSoRequired(t, rank, p, c)) ++so;
                    }
                // Each owner reserves at most three window notifications per peer.
                assert(peers <= (world == 256 ? 16U : 8U) && so <= (world == 256 ? 48U : 24U));
            }
        }
    reset();
    Batch b;
    b.FlushLane(0, 8, scratch);
    assert(copies == 0 && polls[0] == 0);
    queues[0].sq_head = 32712;
    b.state[0].pending_count = 32;
    // No old CQE: a private checkpoint creates reclaimable credit without
    // publishing any data-ready flag or overwriting the staged UB batch.
    std::memset(scratch, 0x5a, 32 * kWqeBytes);
    b.FlushLane(0, 8, scratch);
    assert(checkpoints[0] == 1 && polls[0] == 1);
    assert(queues[0].sq_tail == 32713 && queues[0].sq_head == 32745);
    for (uint32_t i = 0; i < 32 * kWqeBytes; ++i) assert(rings[0][(32713 * kWqeBytes + i) % sizeof(rings[0])] == 0x5a);
    reset();
    queues[0].sq_head = 32712;
    b.state[0].pending_count = 32;
    stuck = true;
    expect_abort(b);
    assert(polls[0] == 1);  // Recheck even after a checkpoint.
    reset();
    contexts[0].depth = 1024;
    b.state[0].pending_count = 1;
    expect_abort(b);
    assert(polls[0] == 0);
    reset(32760);
    b.state[0].pending_count = 32;
    std::memset(scratch, 0x5a, 32 * kWqeBytes);
    b.FlushLane(0, 8, scratch);
    assert(copies == 2 && polls[0] == 0);
    for (uint32_t i = 0; i < 32 * kWqeBytes; ++i) assert(rings[0][(32760 * kWqeBytes + i) % sizeof(rings[0])] == 0x5a);
    reset();
    simulate(256);
    simulate(257);
    simulate(512);
    simulate(513);
    simulate(4096);
    simulate(6000, true);
    reset();
    simulate(6000);
    simulate(6000);  // Second invocation crosses the 16-bit CQE index range.
    reset(UINT32_MAX - 100);
    simulate(10240);
    simulate(10240);
    reset();
    simulate(4096, false, 256);
    simulate(4096, false, 256);
    reset(UINT32_MAX - 100);
    simulate(4608, false, 256);
    for (uint32_t world : {256U, 512U}) {
        reset(UINT32_MAX - 100);
        simulate(16384, false, world);
        simulate(16384, false, world);
        assert(checkpoints[0] > 0 && checkpoints[1] > 0);
        reset();
        simulate(16384, true, world);
        assert(checkpoints[0] == 0 && checkpoints[1] == 0);
    }
}
}  // namespace credit_test
"""
    )
    harness = tmp_path / "combine_predicates.cpp"
    harness.write_text(
        r"""

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#define __aicore__
#define __simt_vf__
#define __gm__
#define __ubuf__
#define __launch_bounds__(n)
#include "kernels/common.hpp"
#include "kernels/combine/tiling.hpp"
#include "kernels/combine/profile.hpp"
#include "kernels/notify/tiling.hpp"
#include "kernels/dispatch/tiling.hpp"
#include "kernels/dispatch/peer_order.hpp"
using CombineTiling = ascend_deepep::PutCombineTiling;
namespace combine_control = ascend_deepep::combine_control;
namespace diagnostics = ascend_deepep::combine_diagnostics;
struct {
    uint32_t x;
} threadIdx;
void asc_threadfence() {}
template <class T>
T asc_ldcg(T* p)
{
    return *p;
}
template <class T>
void asc_stcg(T* p, T v)
{
    *p = v;
}
namespace simt {
std::vector<std::vector<float>> receive;
std::vector<std::vector<uint64_t>> done;
std::vector<std::vector<uint32_t>> writes;
void aclshmem_float_p(float* address, float value, int32_t peer)
{
    const auto index = address - receive[0].data();
    assert(++writes.at(peer).at(index) == 1);
    receive.at(peer).at(index) = value;
}
void aclshmem_uint64_p(uint64_t* address, uint64_t value, int32_t peer)
{
    done.at(peer).at(address - done[0].data()) = value;
}
}  // namespace simt
template <class T>
struct GlobalTensor {
    std::vector<T> values;
    T GetValue(uint64_t index)
    {
        return values.at(index);
    }
};
"""
        + helpers
        + credit_test
        + r"""

int main()
{
    credit_test::run();
    large_shape_test();
    cross_peer_plan_test();
    cross_ranges_test::run();
    dispatch_credit_test::run();
    static_assert(ascend_deepep::kMaxDispatchRanks >= 512);
    for (uint32_t world : {2U, 8U, 16U, 32U, 64U, 128U, 256U, 512U}) {
        const uint32_t width = first_hit_schedule::Width(world);
        for (uint32_t rank = 0; rank < world; ++rank) {
            std::vector<uint32_t> seen(world, 0);
            const auto index = dispatch_contract::GetGroupRankIndex(rank, world);
            assert(index == first_hit_schedule::Index(rank, world));
            for (uint32_t group = 0; group < world / width; ++group)
                for (uint32_t lane = 0; lane < width; ++lane) {
                    auto peer = first_hit_schedule::Peer(rank, world, group, lane);
                    assert(peer < world && ++seen[peer] == 1);
                    assert(first_hit_schedule::Group(rank, world, peer) == group);
                    assert(dispatch_contract::GetGroupRankIndex(peer, world) == lane);
                }
            for (auto visits : seen) assert(visits == 1);
        }
    }
    for (uint32_t world : {2U, 16U, 64U, 128U, 256U}) {
        constexpr uint32_t values = 7 * 6;
        simt::receive.assign(world, std::vector<float>(values, 0));
        simt::writes.assign(world, std::vector<uint32_t>(values, 0));
        simt::done.assign(world, std::vector<uint64_t>(world * 64, 0));
        std::vector<int32_t> ids;
        std::vector<float> weights;
        for (uint32_t id = world * values; id-- > 0;) {
            if (id % 5 == 0) continue;
            ids.push_back(id);
            weights.push_back(id * 0.125f + 0.25f);
        }
        const uint32_t reducers = world <= 8 ? 56 : 40;
        for (uint32_t reducer = 0; reducer < reducers; ++reducer) {
            for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x) {
                ScatterWeights(weights.data(), ids.data(), simt::receive[0].data(), ids.size() * reducer / reducers,
                               ids.size() * (reducer + 1) / reducers, values);
                ScatterWeights(nullptr, nullptr, simt::receive[0].data(), 0, 0, values);
            }
        }
        for (threadIdx.x = 0; threadIdx.x < 64; ++threadIdx.x) {
            NotifyWeightRanks(simt::done[0].data() + (world - 1) * 64, world);
        }
        for (uint32_t peer = 0; peer < world; ++peer) {
            assert(simt::done[peer][(world - 1) * 64] == 1);
            for (uint32_t slot = 0; slot < values; ++slot) {
                uint32_t id = peer * values + slot;
                assert(simt::receive[peer][slot] == (id % 5 == 0 ? 0 : id * 0.125f + 0.25f));
            }
        }
    }
    GlobalTensor<int32_t> tasks{std::vector<int32_t>(18, -1)};
    tasks.values[0] = 1;
    tasks.values[6] = tasks.values[12] = 3;
    assert(!TokenNeedsGatewayReduce(tasks, 0, 0));
    tasks.values[9] = 0;  // local duplicate still needs reduction, even with zero gather mask
    assert(TokenNeedsGatewayReduce(tasks, 0, 1));
    assert(AlignTaskToTokenBoundary(tasks, 0, 0, 0) == 0);
    assert(AlignTaskToTokenBoundary(tasks, 0, 0, 1) == 1);
    assert(AlignTaskToTokenBoundary(tasks, 0, 0, 2) == 3);
    for (uint32_t world : {2U, 8U, 16U, 32U, 64U, 128U, 256U}) {
        for (uint32_t tokens : {1U, 3U, 63U, 256U, 257U, 1025U, 4096U, 4608U, 10240U}) {
            if (world == 256 && tokens > 4608) continue;  // Resident controls exceed 44 KiB.
            auto t = ascend_deepep::MakeCombineTiling(0, world, tokens, 2, 1, 1);
            auto n = ascend_deepep::MakeNotifyTiling(0, world, tokens, 2, world * 4, 256, 7168);
            auto d = ascend_deepep::MakeDispatchTiling(0, world, tokens, 2, world * 4, 1, false);
            std::cout << world << ' ' << tokens << ' ' << n.workspace_bytes << ' ' << n.backward_group_offsets << ' '
                      << n.source_chunk_masks << ' ' << n.expert_totals << ' ' << n.local_prefix_segments << ' '
                      << t.control_slot_num << ' ' << t.gather_input_offset_bytes << ' '
                      << t.server_partial_offset_bytes << ' ' << t.returned_partial_offset_bytes << ' '
                      << t.output_offset_bytes << ' ' << t.weight_recv_offset_bytes << ' '
                      << t.weight_local_done_offset_bytes << ' ' << t.weight_done_offset_bytes << ' '
                      << ascend_deepep::CombineWorkspaceBytes(t) << ' ' << d.sync << ' ' << d.book << ' ' << d.input
                      << ' ' << d.output << ' ' << d.weights << ' ' << d.scales << ' ' << d.inbox << ' '
                      << d.inbox_stride << ' ' << d.address_stride << ' ' << d.workspace_bytes << '\n';
            assert(t.num_so == 3U);
            uint32_t windows = 0;
            for (uint32_t c = 0; c < t.chunks_per_group; c = SoWindowEnd(t, c) + 1U) ++windows;
            assert(windows == std::min(3U, (tokens + 255U) / 256U));
            if (tokens == 4096) {
                // Preserve the first half, then split the second half: 8 + 4 + 4 logical chunks.
                assert(SoWindowEnd(t, 0) == 10U);
                assert(SoWindowEnd(t, 11) == 14U);
                assert(SoWindowEnd(t, 15) == 18U);
            }
            assert(sizeof(t) == 160);
            assert(t.weight_topk == 2 && !t.with_weights && t.weight_rows == 0);
            assert(t.weight_recv_offset_bytes % 512 == 0);
            assert(t.weight_local_done_offset_bytes >= t.weight_recv_offset_bytes + uint64_t(tokens) * 2 * 4);
            assert(t.weight_done_offset_bytes == t.weight_local_done_offset_bytes + 64 * 512);
            uint32_t end = 0;
            for (uint32_t c = 0; c < t.chunks_per_group; ++c) {
                auto next = std::min<uint64_t>(tokens, ChunkTokenEnd(t, c));
                assert(next > end && next <= tokens);
                end = next;
                assert(SoWindowEnd(t, c) >= c && SoWindowEnd(t, c) < t.chunks_per_group);
                assert(CrossCompletionSlot(t, world - 1, c, 1) < t.control_slot_num - 64);
            }
            assert(end == tokens);
            if (world <= 8) continue;
            for (uint32_t rank = 0; rank < world; ++rank) {
                t.rank_id = rank;
                for (uint32_t peer = 0; peer < world; ++peer) {
                    if (rank / 8 == peer / 8) continue;
                    // Every owner that sends in a window must issue its own final SO.
                    for (uint32_t begin = 0; begin < t.chunks_per_group;) {
                        uint32_t last[16];
                        std::fill(last, last + 16, UINT32_MAX);
                        const uint32_t finish = SoWindowEnd(t, begin);
                        for (uint32_t c = begin; c <= finish; ++c) {
                            const auto owner = CrossChunkOwner(t, rank, peer, c);
                            assert(owner < 16 && CrossPeerAssigned(t, peer, owner));
                            last[owner] = c;
                        }
                        for (uint32_t c = begin; c <= finish; ++c) {
                            assert(CrossSoRequired(t, rank, peer, c) == (last[CrossChunkOwner(t, rank, peer, c)] == c));
                        }
                        begin = finish + 1;
                    }
                }
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    use_wasi = sys.platform == "win32" and importlib.util.find_spec("wasmtime") and not compiler
    executable = tmp_path / (
        "combine_predicates.wasm"
        if use_wasi
        else "combine_predicates.exe"
        if sys.platform == "win32"
        else "combine_predicates"
    )
    result = subprocess.run(
        [
            *command,
            *(["-target", "wasm32-wasi", "-fno-exceptions"] if use_wasi else []),
            "-std=c++17",
            "-O2",
            "-UNDEBUG",
            "-I",
            str(root / "csrc"),
            "-I",
            str(Path(__file__).parent),
            "-I",
            str(tmp_path),
            str(harness),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if use_wasi:
        import wasmtime

        engine = wasmtime.Engine()
        module = wasmtime.Module.from_file(engine, str(executable))
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        wasi = wasmtime.WasiConfig()
        output = tmp_path / "stdout"
        wasi.stdout_file = str(output)
        with wasmtime.Store(engine) as store:
            store.set_wasi(wasi)
            instance = linker.instantiate(store, module)
            try:
                instance.exports(store)["_start"](store)
            except wasmtime.ExitTrap as error:
                assert error.code == 0
        stdout = output.read_text(encoding="utf-8")
    else:
        stdout = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30).stdout
    from deep_ep.buffers._elastic_layout import combine_layout, dispatch_layout, workspace_layout

    for line in stdout.splitlines():
        world, tokens, *actual = map(int, line.split())
        notify = workspace_layout(world, tokens, 2, world * 4)
        combine = combine_layout(world, tokens, 2, 1, 1)
        expected = [
            notify[name]
            for name in (
                "workspace_bytes",
                "backward_group_offsets",
                "source_chunk_masks",
                "expert_totals",
                "local_prefix_segments",
            )
        ]
        expected += [
            combine[name]
            for name in (
                "control_slot_num",
                "gather_input",
                "server_partial",
                "returned_partial",
                "output",
                "weight_recv",
                "weight_local_done",
                "weight_done",
                "workspace_bytes",
            )
        ]
        dispatch = dispatch_layout(world, tokens, 2, world * 4, 1)
        expected += [
            dispatch[name]
            for name in (
                "sync",
                "book",
                "input",
                "output",
                "weights",
                "scales",
                "inbox",
                "inbox_stride",
                "address_stride",
                "workspace_bytes",
            )
        ]
        assert actual == expected, (world, tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _notify["_add_launch_args"](parser)
    _notify["add_notify_args"](parser)
    parser.add_argument("--bootstrap", choices=("default", "mpi"), default="default")
    parser.add_argument(
        "--use-1g-huge-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1 GiB SHMEM workspace pages (default); --no-use-1g-huge-pages selects the standard allocator",
    )
    _combine_cases["add_arguments"](parser, 257)
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument("--input-storage", choices=("gm", "gm-huge", "shmem"), default="gm")
    storage.add_argument("--input-storage-list", choices=("gm", "gm-huge", "shmem"), nargs="+")
    parser.add_argument(
        "--profile-api-time", action="store_true", help="Also time the communication stream including staging copies"
    )
    parser.add_argument("--cases-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--reuse-window-tokens", type=int, default=0, help="Force a bounded server Combine window")
    parser.add_argument(
        "--combine-capacity-mib", type=int, default=0, help="Limit Combine workspace to test automatic reuse"
    )
    parser.add_argument(
        "--compare-full", action="store_true", help="Also compare reuse output with the full-buffer kernel"
    )
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--golden-notify", action="store_true", help="Construct Combine metadata on CPU without running Notify"
    )
    parser.add_argument(
        "--local-expert-first",
        action="store_true",
        help="Place source-rank experts first in top-k (without weights)",
    )
    _notify["add_dedup_args"](parser, "no_server_dedup")
    parser.add_argument(
        "--dispatch-dtype",
        choices=("bf16", "fp8"),
        default="bf16",
        help="FP8 always transports scales; expert/Combine payloads are BF16",
    )
    parser.add_argument(
        "--with-dispatch",
        "--check-dispatch",
        dest="check_dispatch",
        action="store_true",
        help="Execute and validate payload Dispatch (default: Notify metadata + directly constructed expert inputs)",
    )
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--with-weights", dest="without_weights", action="store_false", help="Enable weight return")
    weights.add_argument(
        "--without-weights", dest="without_weights", action="store_true", help="Disable weight return (default)"
    )
    parser.set_defaults(without_weights=True)
    parser.add_argument(
        "--dense-channels", action="store_true", help="Build all QPs per peer for initialization A/B comparison"
    )
    parser.add_argument(
        "--route", choices=("all", *_ROUTES, "sq_pressure", "hotspot", "rotating_hotspot"), default="all"
    )
    _notify["add_random_route_args"](parser)
    parser.add_argument("--output", type=Path, required=True)
    _notify["add_profile_args"](parser, ("combine",))
    args = parser.parse_args()
    _notify["resolve_dedup_args"](args, "no_server_dedup")
    args.ep8_rank_pipeline = args.ep_size == 8 and args.no_server_dedup
    if args.reuse_window_tokens < 0 or args.combine_capacity_mib < 0:
        parser.error("Reuse window and workspace limit must be nonnegative")
    if (args.reuse_window_tokens or args.combine_capacity_mib) and args.no_server_dedup:
        parser.error("Reuse requires server deduplication")
    if args.profile_api_time and args.profile_op != "combine":
        parser.error("--profile-api-time requires --profile-op combine")
    _notify["validate_random_route_args"](parser, args)
    _notify["validate_profile_args"](parser, args)
    if args.golden_notify and args.check_dispatch:
        parser.error("--golden-notify cannot use --with-dispatch")
    if args.no_server_dedup and args.check_dispatch and args.ep_size not in (8, 16):
        parser.error("--no-server-dedup with --with-dispatch supports EP8/16")
    if args.local_expert_first and not args.without_weights:
        parser.error("--local-expert-first requires --without-weights")
    if not 0 < args.hidden <= 7168 or args.hidden % (128 if args.dispatch_dtype == "fp8" else 16):
        parser.error("hidden must be <=7168 and divisible by 16 (BF16) or 128 (FP8)")
    try:
        cases = _combine_cases["from_args"](args)
    except ValueError as exc:
        parser.error(str(exc))
    args.tokens, args.chunk_tokens = cases[0]["tokens"], cases[0]["chunk_tokens"]
    if args.check_dispatch and any(case["chunk_tokens"] != 256 for case in cases):
        parser.error("--chunk-tokens 512/1024 requires the default Notify-only Combine test")
    if args.check_dispatch and any(case.get("notify_num_cores", args.notify_num_cores) != 64 for case in cases):
        parser.error("--notify-num-cores 32 requires the default Notify-only Combine test")
    if args.dispatch_dtype == "fp8" and not args.check_dispatch:
        parser.error("--dispatch-dtype fp8 requires --with-dispatch")
    if args.ep_size > 512 or not 1 <= args.tokens <= 16384 or not 1 <= args.topk <= 16 or args.repeat < 1:
        parser.error("EP<=512, tokens1..16384, topk1..16, repeat>=1 required")
    if args.experts < args.ep_size or args.experts > 1024 or args.experts % args.ep_size:
        parser.error("experts must be divisible by EP and <=1024")
    if args.route == "sq_pressure" and (
        args.ep_size != 128
        or args.topk != 1
        or args.experts != 128
        or any(case["tokens"] < 6000 for case in cases)
        or args.repeat < 2
    ):
        parser.error("sq_pressure requires EP128, topk=1, experts=128, tokens>=6000 and repeat>=2")
    args.output.mkdir(parents=True, exist_ok=True)
    _notify["_launch"](args, worker)


_summary_metrics = runpy.run_path(str(Path(__file__).with_name("combine_metrics.py")))

_summary_FIELDS = _summary_metrics["FIELDS"]


def _summary_records(rank=0, world=128, tokens=4096):
    result = []
    cross = 16 if world > 8 else 0
    for core in range(64):
        role = 0 if core < min(world, 8) else 1 if 8 <= core < 8 + cross else 2 if core >= 8 + cross else 3
        v = dict.fromkeys(_summary_FIELDS, 0)
        v.update(
            Magic=_summary_metrics["MAGIC"],
            Version=_summary_metrics["VERSION"],
            Rank=rank,
            Core=core,
            World=world,
            Tokens=tokens,
            Role=role,
            WithWeights=1,
            Sample=1,
            Topk=6,
            Groups=world // min(world, 64),
            Chunks=(tokens + 255) // 256 + min(4, tokens) - 1,
            ChunkTokens=256,
            CrossCores=cross,
            FinalTokens=len(range(core, tokens, 64)),
        )
        for i, name in enumerate(
            (
                "Begin",
                "InitDone",
                "WeightSendBegin",
                "WeightSendEnd",
                "ProducerBegin",
                "GatherEnd",
                "LocalReturnBegin",
                "LocalReturnEnd",
                "ProducerEnd",
                "SourceBegin",
                "SourceEnd",
                "QuietBegin",
                "QuietEnd",
                "ReturnsBegin",
                "ReturnsEnd",
                "WeightWaitBegin",
                "WeightWaitEnd",
                "End",
            )
        ):
            v[name] = (i + 1) * 1000
        if role == 1:
            v.update(
                DirectRows=3,
                ReducedRows=2,
                PayloadWqes=5,
                Lane0PayloadRows=4,
                Lane1PayloadRows=1,
                CreditWaitTicks=250,
                Lane0CreditWaitTicks=200,
                Lane1CreditWaitTicks=50,
                SqPressureCount=3,
                Lane0SqPressureCount=2,
                Lane1SqPressureCount=1,
                Lane0QuietTicks=500,
                Lane1QuietTicks=100,
                FirstDirectIssue=6000,
                FirstReducedIssue=7000,
                CrossReduceWaitTicks=1000,
                CrossSubmitTicks=1000,
                CrossBarrierTicks=1000,
                CrossPeerSetupTicks=200,
                CrossSoReserveTicks=50,
            )
        v["SourceWaitTicks"] = 250
        result.append([v[name] for name in _summary_FIELDS] + [0] * (_summary_metrics["WORDS"] - len(_summary_FIELDS)))
    return result


def _summary_decode(rows, rank=0, world=128, tokens=4096):
    return _summary_metrics["decode_profile"](
        rows, rank=rank, world=world, tokens=tokens, topk=6, weights=True, sample=1
    )


@pytest.mark.parametrize("world,tokens", [(2, 1), (8, 257), (128, 4096), (512, 16384)])
def test_summary_roles_clock_conversion_and_large_shapes(world, tokens):
    decoded = _summary_decode(_summary_records(world=world, tokens=tokens), world=world, tokens=tokens)
    assert sum((c["counters"]["FinalTokens"] for c in decoded)) == tokens
    assert decoded[0]["phases_us"]["source_wait"] == 0.25
    assert decoded[0]["phases_us"]["source_after_wait"] == 0.75
    assert decoded[-1]["role"] == "gateway"
    if world > 8:
        assert [decoded[i]["role"] for i in (7, 8, 23, 24)] == ["local", "cross", "cross", "gateway"]
        assert decoded[8]["phases_us"]["cross_peer_setup"] == 0.2
        assert decoded[8]["phases_us"]["cross_so_reserve"] == 0.05
        assert decoded[8]["phases_us"]["cross_first_payload"] == 1.0
        assert decoded[8]["phases_us"]["lane0_credit_wait"] == 0.2
        assert decoded[8]["phases_us"]["lane1_quiet"] == 0.1
        assert decoded[8]["counters"]["Lane0PayloadBytes"] == 4 * 14336


@pytest.mark.parametrize(
    "field", ["Lane0PayloadRows", "Lane1CreditWaitTicks", "Lane0SqPressureCount", "Lane1QuietTicks"]
)
def test_summary_reject_inconsistent_lane_statistics(field):
    rows = _summary_records()
    rows[8][_summary_FIELDS.index(field)] += 1000
    with pytest.raises(ValueError):
        _summary_decode(rows)


def test_summary_legacy_summary_has_no_invented_lane_statistics():
    rows = _summary_records()
    for index, row in enumerate(rows):
        rows[index] = row[:63] + [0]
        rows[index][1] = 3
    decoded = _summary_decode(rows)
    assert "Lane0PayloadBytes" not in decoded[8]["counters"]
    assert "lane0_quiet" not in decoded[8]["phases_us"]


@pytest.mark.parametrize(
    "fault", ["missing", "rank", "version", "sample", "role", "order", "negative", "wait", "wqes", "tokens"]
)
def test_summary_reject_corrupt_or_stale_records(fault):
    rows = _summary_records()
    if fault == "missing":
        rows.pop()
    else:
        core, field, value = {
            "rank": (0, "Rank", 1),
            "version": (0, "Version", 0),
            "sample": (0, "Sample", 2),
            "role": (8, "Role", 0),
            "order": (0, "End", 1),
            "negative": (0, "Begin", -1),
            "wait": (0, "SourceWaitTicks", 100000),
            "wqes": (8, "PayloadWqes", 6),
            "tokens": (0, "FinalTokens", 0),
        }[fault]
        rows[core][_summary_FIELDS.index(field)] = value
    with pytest.raises(ValueError):
        _summary_decode(rows)


def test_summary_cpp_python_abi_matches():
    path = Path(__file__).parents[2] / "csrc/kernels/combine/profile.hpp"
    enum = path.read_text().split("enum Field : uint32_t {", 1)[1].split("};", 1)[0]
    names = [re.split("\\s*=", item.strip())[0] for item in enum.split(",")]
    assert names == _summary_FIELDS + ["FieldCount"]


def test_summary_cluster_summary_revalidates_and_identifies_slowest():
    reports = []
    for rank in range(2):
        cores = _summary_decode(_summary_records(rank, 2, 4096), rank, 2, 4096)
        d = dict(
            schema_version=_summary_metrics["VERSION"],
            ticks_per_us=1000,
            rank=rank,
            world_size=2,
            tokens=4096,
            topk=6,
            hidden=7168,
            experts=512,
            scenario="random-bf16-weights1",
            seed=2026,
            sample=1,
            with_weights=True,
            source_sha256="source",
            precision_checked=True,
            kernel="combine_profile_kernel",
            cores=cores,
        )
        reports.append(
            dict(
                rank=rank,
                world_size=2,
                passed=True,
                combine_diagnostics=[d],
                **{k: d[k] for k in ("tokens", "topk", "experts", "seed", "with_weights", "source_sha256")},
            )
        )
    reports[1]["combine_diagnostics"][0]["cores"][63]["raw_words"][_summary_FIELDS.index("End")] += 1000
    result = _summary_metrics["summarize"](reports)
    assert result["phase_maxima_us"]["gateway/kernel_envelope"] == dict(us=18.0, rank=1, core=63)
    with pytest.raises(ValueError):
        _summary_metrics["summarize"](reports[:1])
    bad = copy.deepcopy(reports)
    bad[1]["combine_diagnostics"][0]["precision_checked"] = False
    with pytest.raises(ValueError):
        _summary_metrics["summarize"](bad)


@pytest.mark.parametrize("fault", [None, "capture", "trace", "precision"])
@pytest.mark.parametrize("flush_l2", [False, True])
def test_summary_profile_capture_excludes_diagnostics_and_restores_hook(tmp_path, monkeypatch, fault, flush_l2):
    from contextlib import nullcontext

    test = runpy.run_path(str(Path(__file__).with_name("test_combine.py")))
    old = {"previous": "hook"}
    buffer = SimpleNamespace(_combine_diagnostics=old, _runtime=SimpleNamespace(stream=object()))
    calls, clears = [], []
    active = False
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: _summary_records()))
    chunk_data = object()
    chunk_tensor = SimpleNamespace(numel=lambda: 64 * 76 * 32, cpu=lambda: SimpleNamespace(tolist=lambda: chunk_data))

    def zeros(shape, **kwargs):
        assert not active
        return tensor if shape == (64, _summary_metrics["WORDS"]) else chunk_tensor

    fake = SimpleNamespace(
        zeros=zeros,
        int64="int64",
        int32="int32",
        empty=lambda *a, **kw: SimpleNamespace(zero_=lambda: clears.append(len(calls))),
        npu=SimpleNamespace(synchronize=lambda: None, stream=lambda _: nullcontext()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    def decode_chunks(value, **kwargs):
        assert not active and value is chunk_data
        assert buffer._combine_diagnostics["chunks"] is chunk_tensor
        return [dict(chunk=0, kind="final_reduce")]

    def export_chunks(path, value):
        assert not active and value["precision_checked"]
        path.with_suffix(".svg").write_text("<svg/>")
        path.with_suffix(".csv").write_text("chunk\n0\n")
        path.with_suffix(".perfetto.json").write_text('{"traceEvents":[]}')

    test["_combine_peers"]["decode"] = lambda *a, **kw: []
    test["_combine_peers"]["write_trace"] = lambda *a, **kw: dict(events=0)
    test["_combine_chunks"]["decode"] = decode_chunks
    test["_combine_chunks"]["write_timeline"] = export_chunks

    def invoke():
        traced = buffer._combine_diagnostics is not None
        calls.append(traced)
        assert not (traced and active)
        if traced and fault == "trace":
            raise RuntimeError("trace failed")
        return traced

    def validate(result):
        if result and fault == "precision":
            raise RuntimeError("precision failed")

    def capture(*args):
        nonlocal active
        active = True
        try:
            assert buffer._combine_diagnostics is None
            if fault == "capture":
                raise RuntimeError("capture failed")
            args[3]()
            validate(args[3]())
        finally:
            active = False
        return dict(kernel="combine_kernel", samples=2, avg_us=3.0)

    test["_notify"]["profile_operator"] = capture
    args = SimpleNamespace(
        output=tmp_path,
        ep_size=128,
        tokens=4096,
        topk=6,
        experts=512,
        seed=2026,
        without_weights=False,
        profile_flush_l2=flush_l2,
    )

    def run():
        return test["profile_combine"](
            args,
            0,
            "random-bf16-weights1",
            buffer,
            SimpleNamespace(device="npu:0"),
            invoke,
            validate,
            dict(source_sha256="source"),
            lambda _: None,
        )

    if fault:
        with pytest.raises(RuntimeError):
            run()
    else:
        result = run()
        assert calls == [False, False, True]
        assert result["clean_profile"]["kernel"] == "combine_kernel"
        assert result["kernel"] == "combine_profile_kernel"
        assert result["chunk_profile"]["events"] == 1
        assert Path(result["chunk_profile"]["file"]).is_file()
        assert Path(result["chunk_profile"]["svg"]).is_file()
        assert Path(result["chunk_profile"]["perfetto"]).is_file()
        saved = json.loads(next(tmp_path.rglob("rank-0.json")).read_text())
        assert saved == result and saved["precision_checked"]
    assert buffer._combine_diagnostics is old
    assert clears == (list(range(len(calls))) if flush_l2 else [])


_session_ROOT = Path(__file__).parents[2]

_session_CASES = runpy.run_path(str(_session_ROOT / "tests/elastic/combine_cases.py"))


def test_session_order_capacity_validation_and_output_paths(tmp_path):
    parser = argparse.ArgumentParser()
    _session_CASES["add_arguments"](parser, 16384)
    args = parser.parse_args(["--tokens-list", "4096", "16384", "4096", "--chunk-tokens-list", "256", "512"])
    args.output = tmp_path
    cases = _session_CASES["from_args"](args)
    assert [(c["tokens"], c["chunk_tokens"]) for c in cases] == [
        (4096, 256),
        (4096, 512),
        (16384, 256),
        (16384, 512),
        (4096, 256),
        (4096, 512),
    ]
    assert len({_session_CASES["case_args"](args, c, True).output for c in cases}) == 6
    assert _session_CASES["case_args"](args, cases[0], False).output == tmp_path
    assert _session_CASES["from_args"](parser.parse_args([])) == _session_CASES["make_cases"]([16384], [256])
    for t, c in ((0, 256), (32768, 256), (4096, 128), (True, 256), (4096, 256.0)):
        with pytest.raises(ValueError):
            _session_CASES["make_cases"]([t], [c])
    for broken in ([], list(reversed(cases)), [dict(cases[0], case_id="../../escape")]):
        with pytest.raises(ValueError):
            _session_CASES["validate_cases"](broken)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    assert _session_CASES["from_args"](SimpleNamespace(cases_file=path)) == cases
    core_cases = _session_CASES["make_cases"]([4096, 16384], [256, 512], [32, 64, 32])
    assert len(core_cases) == 12 and len({c["case_id"] for c in core_cases}) == 12
    assert _session_CASES["validate_cases"](core_cases) == core_cases
    assert _session_CASES["case_args"](args, core_cases[0], True).notify_num_cores == 32
    for cores in ([], [0], [63], [True], [32.0]):
        with pytest.raises(ValueError):
            _session_CASES["make_cases"]([4096], [256], cores)


@pytest.mark.parametrize("fail_second", [False, True])
@pytest.mark.parametrize("storages", [["gm"], ["gm", "shmem"]])
def test_session_one_buffer_for_grow_shrink_and_fail_fast(tmp_path, monkeypatch, fail_second, storages):
    spec = importlib.util.spec_from_file_location("combine_session", _session_ROOT / "tests/elastic/test_combine.py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    events = []
    producers = []

    def make_producer():
        stream = object()
        producers.append(stream)
        return stream

    npu = SimpleNamespace(
        set_device=lambda rank: events.append(("device", rank)),
        synchronize=lambda: events.append(("sync",)),
        Stream=make_producer,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=npu))
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    layout = ModuleType("deep_ep.buffers._elastic_layout")
    layout.elastic_size_hint = lambda world, tokens, *a, **kw: tokens * 100
    monkeypatch.setitem(sys.modules, "deep_ep.buffers._elastic_layout", layout)
    (tmp_path / "_build_info.json").write_text(
        json.dumps(
            dict(
                capabilities=[
                    "elastic-combine-expanded-v3",
                    "elastic-notify-metadata-v7",
                    "runtime-combine-sparse-qp-v1",
                ]
            )
        )
    )
    instances = []

    class Buffer:
        def __init__(self, **kwargs):
            self.options = kwargs
            self._runtime = SimpleNamespace(
                native=SimpleNamespace(
                    use_1g_huge_pages=kwargs["use_1g_huge_pages"], workspace_allocation_bytes=1 << 30
                )
            )
            self.destroyed = 0
            instances.append(self)

        def destroy(self):
            self.destroyed += 1
            events.append(("destroy",))

    monkeypatch.setitem(
        test._notify,
        "_import_installed_deep_ep",
        lambda: SimpleNamespace(__file__=str(tmp_path / "__init__.py"), ElasticBuffer=Buffer),
    )
    monkeypatch.setitem(test._notify, "_worker_identity", lambda *_: (0, 128))
    monkeypatch.setitem(test._notify, "_progress_logger", lambda *_: lambda text: None)
    monkeypatch.setitem(test._notify, "_log_failure", lambda _: events.append(("failed",)))
    route_tokens = []

    def routes(world, tokens, *a, **kw):
        route_tokens.append(tokens)
        return tokens

    monkeypatch.setitem(test._notify, "make_routes", routes)
    monkeypatch.setattr(test, "build_reference", lambda *a: dict(rank_rows=[1], gather_rows=[2]))
    cases = _session_CASES["make_cases"]([4096, 16384, 4096], [256, 512], [32, 64])
    args = SimpleNamespace(
        ep_size=128,
        tokens=4096,
        tokens_list=[4096, 16384, 4096],
        hidden=4096,
        chunk_tokens=256,
        chunk_tokens_list=[256, 512],
        notify_num_cores_list=[32, 64],
        topk=6,
        experts=512,
        seed=2026,
        route="random",
        bootstrap="mpi",
        profile_op=None,
        dispatch_dtype="bf16",
        check_dispatch=False,
        dense_channels=False,
        output=tmp_path,
        input_storage_list=storages,
    )
    visited = []

    def run_case(buffer, current, rank, world, info, log, references):
        assert events[-1] == ("sync",)
        assert buffer is instances[0]
        assert len(producers) == 1 and current._producer_stream is producers[0]
        assert bool(references) == (not visited)
        references.clear()
        assert buffer._combine_stage_input == (current.input_storage == "shmem")
        visited.append((current.tokens, current.chunk_tokens, current.notify_num_cores, current.input_storage))
        events.append(("case", current.case_id))
        if fail_second and len(visited) == 2:
            raise RuntimeError("device failure")
        current.output.mkdir(parents=True)
        (current.output / "rank-0.json").write_text('{"passed":true}')
        return (None, None)

    monkeypatch.setattr(test, "run_case", run_case)
    expected = [(c["tokens"], c["chunk_tokens"], c["notify_num_cores"], storage) for c in cases for storage in storages]
    if fail_second:
        with pytest.raises(RuntimeError, match="device failure"):
            test.worker(0, args)
        assert visited == expected[:2]
        assert not (tmp_path / "rank-0.json").exists()
        first_output = tmp_path / "cases" / cases[0]["case_id"]
        if len(storages) > 1:
            first_output /= "input-gm"
        assert (first_output / "rank-0.json").is_file()
    else:
        test.worker(0, args)
        assert visited == expected
        summary = json.loads((tmp_path / "rank-0.json").read_text())
        assert summary["initialization_count"] == 1 and summary["cases"] == cases
        assert summary["buffer_init_seconds"] >= 0
    assert route_tokens == [4096, 16384]
    assert len(instances) == 1 and instances[0].destroyed == 1
    assert instances[0].options["num_max_tokens_per_rank"] == 16384
    assert instances[0].options["num_bytes"] == (2 << 20 if "shmem" in storages else 1638400)
    assert instances[0].options["use_1g_huge_pages"] is True
    assert events[-1] == ("destroy",)


_chunk_ROOT = Path(__file__).resolve().parents[2]

_chunk_chunk = runpy.run_path(str(_chunk_ROOT / "tests/elastic/combine_chunks.py"))


def _chunk_records(ep8_rank_pipeline=False):
    world, tokens, sample = (8 if ep8_rank_pipeline else 16, 257, 7)
    _, count, words = _chunk_chunk["shape"](world, tokens)
    data = [[[0] * words for _ in range(count)] for _ in range(64)]
    chunks = count // 3
    for core in range(64):
        for k in range(chunks):
            kinds = (
                [5] + ([4] if core < 32 else [2])
                if ep8_rank_pipeline
                else [5] + ([4] if core < 8 else [3] if core < 24 else [2])
            )
            for kind in kinds:
                row = dict(
                    Magic=_chunk_chunk["MAGIC"],
                    Version=_chunk_chunk["VERSION"],
                    Sample=sample,
                    KindId=kind,
                    Core=core,
                    Group=0,
                    Chunk=k,
                    Peer=(core // 4 if ep8_rank_pipeline else core) if kind == 4 else 2**32 - 1,
                    Begin=100 + k * 1000,
                    WorkBegin=110 + k * 1000,
                    WorkEnd=120 + k * 1000,
                    End=130 + k * 1000,
                    WindowEnd=k,
                )
                index = (2 if kind == 5 else 1 if kind == 4 else 0) * chunks + k
                data[core][index] = [row.get(f, 0) for f in _chunk_chunk["FIELDS"]]
    return (data, dict(rank=0, world=world, tokens=tokens, sample=sample))


@pytest.mark.parametrize("weights", [False, True])
def test_ep8_rank_pipeline_diagnostics_validate_new_core_ownership(weights):
    rows = _summary_records(world=8)
    for core, row in enumerate(rows):
        row[_summary_FIELDS.index("WithWeights")] = int(weights)
        row[_summary_FIELDS.index("Role")] = 0 if core < 32 else 2
        if core < 32:
            row[_summary_FIELDS.index("GatherEnd")] = row[_summary_FIELDS.index("ProducerBegin")]
    args = dict(rank=0, world=8, tokens=4096, topk=6, weights=weights, sample=1)
    decoded = _summary_metrics["decode_profile"](rows, **args, ep8_rank_pipeline=True)
    assert [c["role"] for c in decoded] == ["local"] * 32 + ["gateway"] * 32
    assert all(c["phases_us"]["gather"] == 0 for c in decoded[:32])
    with pytest.raises(ValueError, match="Missing/stale"):
        _summary_metrics["decode_profile"](rows, **args)

    records, args = _chunk_records(ep8_rank_pipeline=True)
    events = _chunk_chunk["decode"](records, **args, ep8_rank_pipeline=True)
    local = [e for e in events if e["kind"] == "local_return"]
    assert {e["core"] for e in local} == set(range(32))
    assert all(e["peer"] == e["core"] // 4 for e in local)
    trace = _chunk_report(events)
    trace.update(world_size=8, kernel="combine_ep8_profile_kernel")
    assert _chunk_chunk["perfetto_trace"](trace, "ep8.json")["traceEvents"]
    with pytest.raises(ValueError):
        _chunk_chunk["decode"](records, **args)
    records[4][5][_chunk_chunk["FIELDS"].index("Peer")] = 4
    with pytest.raises(ValueError, match="local peer/core"):
        _chunk_chunk["decode"](records, **args, ep8_rank_pipeline=True)


def _chunk_report(events, **extra):
    return dict(
        schema_version=1,
        rank=0,
        world_size=16,
        tokens=257,
        ticks_per_us=1000,
        kernel="combine_profile_kernel",
        precision_checked=True,
        source_sha256="source",
        note=_chunk_chunk["NOTE"],
        events=events,
        **extra,
    )


def test_chunk_chunk_exports_and_rejects_corruption(tmp_path):
    data, args = _chunk_records()
    events = _chunk_chunk["decode"](data, **args)
    trace = _chunk_report(events)
    output = tmp_path / "rank-0.chunks.json"
    _chunk_chunk["write_timeline"](output, trace)
    assert "gateway_reduce" in output.with_suffix(".svg").read_text()
    assert "WorkEnd_us" in output.with_suffix(".csv").read_text()
    perfetto = json.loads(output.with_suffix(".perfetto.json").read_text())
    envelopes = [e for e in perfetto["traceEvents"] if e["name"].startswith("chunk ")]
    assert len(envelopes) == len(events)
    assert {e["args"]["core"] for e in envelopes} == set(range(64))
    assert not list(tmp_path.glob("*.tmp"))
    for field, value in (("Sample", 8), ("Core", 30), ("WorkEnd", 99), ("Magic", 0)):
        bad = copy.deepcopy(data)
        bad[24][0][_chunk_chunk["FIELDS"].index(field)] = value
        with pytest.raises(ValueError):
            _chunk_chunk["decode"](bad, **args)
    bad = copy.deepcopy(data)
    bad[24][0] = [0] * _chunk_chunk["WORDS"]
    with pytest.raises(ValueError, match="Missing gateway"):
        _chunk_chunk["decode"](bad, **args)


def _chunk_perfetto_fixture():
    base = 2**55

    def event(core, kind, index, times, **counters):
        return dict(
            rank=0,
            core=core,
            kind=kind,
            group=0,
            chunk=index,
            token_begin=index * 64,
            token_end=(index + 1) * 64,
            peer=None,
            window_end=1,
            counters=dict(Tokens=0, Rows=0, DirectRows=0, ReducedRows=0, PayloadWqes=0, Sos=0) | counters,
            times_ticks={key: base + value for key, value in times.items()},
        )

    return _chunk_report(
        [
            event(
                0,
                "final_reduce",
                0,
                dict(WaitBegin=100, Ready=200, Begin=220, WorkBegin=230, WorkEnd=250, End=260),
                Tokens=1,
            ),
            event(0, "final_reduce", 1, dict(Ready=200, Begin=300, WorkBegin=310, WorkEnd=330, End=340), Tokens=1),
            event(
                8,
                "cross_send",
                0,
                dict(
                    Begin=150,
                    WorkBegin=155,
                    WorkEnd=280,
                    End=290,
                    FirstSend0=180,
                    LastSend0=180,
                    FirstDirect=180,
                    LastDirect=180,
                ),
                DirectRows=2,
                PayloadWqes=2,
            ),
            event(24, "gateway_reduce", 0, dict(Begin=105, WorkBegin=110, WorkEnd=120, End=130)),
            event(25, "gateway_reduce", 0, dict(Begin=105, WorkBegin=110, WorkEnd=120, End=130), Tokens=1),
        ]
    )


def test_chunk_perfetto_preserves_so_window_waits_submission_instants_and_empty_reductions():
    trace = _chunk_chunk["perfetto_trace"](_chunk_perfetto_fixture(), "rank-0.chunks.json")
    assert trace["otherData"]["time_origin_tick"] == 2**55 + 100
    events = [e for e in trace["traceEvents"] if e["ph"] != "M"]
    waits = [e for e in events if e["name"].startswith("SO_window_wait")]
    assert len(waits) == 1 and (waits[0]["ts"], waits[0]["dur"]) == (0, 0.1)
    final = [e for e in events if e["name"].startswith("chunk ") and e["args"]["core"] == 0]
    assert [(e["ts"], e["dur"]) for e in final] == [(0, 0.16), (0.2, 0.04)]
    sends = [e for e in events if e["name"].startswith(("First", "Last"))]
    assert len(sends) == 4 and all((e["ph"] == "I" and e["ts"] == 0.08 and ("dur" not in e) for e in sends))
    done = [e for e in events if e["name"].startswith("gateway_reduce_done")]
    assert len(done) == 1 and done[0]["args"]["core"] == 25 and (done[0]["ts"] == 0.02)
    assert any((e["name"].startswith("no_reduce") and e["args"]["core"] == 24 for e in events))


def test_chunk_perfetto_empty_cross_owner_has_no_work_slice(tmp_path):
    trace = _chunk_perfetto_fixture()
    cross = trace["events"][2]
    cross["counters"] = dict.fromkeys(cross["counters"], 0)
    cross["times_ticks"] = {key: cross["times_ticks"][key] for key in ("Begin", "WorkEnd", "End")}
    cross["times_ticks"].update(WaitBegin=2**55 + 151, Ready=2**55 + 154)
    path = tmp_path / "empty-owner.chunks.json"
    _chunk_chunk["write_timeline"](path, trace)
    exported = json.loads(path.with_suffix(".perfetto.json").read_text())
    events = [e for e in exported["traceEvents"] if e.get("args", {}).get("core") == 8]
    assert any(e["name"].startswith("chunk ") for e in events)
    assert any(e["name"].startswith("wait ") for e in events)
    assert not any(e["name"].startswith("cross_send |") for e in events)
    assert "WorkBegin" not in cross["times_ticks"]
    cross["counters"]["PayloadWqes"] = 1
    with pytest.raises(ValueError, match="Missing chunk work timestamp"):
        _chunk_chunk["perfetto_trace"](trace, path.name)


def test_chunk_lane_payload_and_barrier_roundtrip(tmp_path):
    data, args = _chunk_records()
    row = data[8][0]
    for field, value in dict(
        DirectRows=4,
        PayloadWqes=4,
        Lane0Rows=3,
        Lane1Rows=1,
        FirstDirect=115,
        LastDirect=116,
        FirstSend0=115,
        LastSend0=115,
        FirstSend1=116,
        LastSend1=116,
    ).items():
        row[_chunk_chunk["FIELDS"].index(field)] = value
    events = _chunk_chunk["decode"](data, **args)
    event = next((e for e in events if e["core"] == 8 and e["kind"] == "cross_send" and (e["chunk"] == 0)))
    assert event["counters"]["Lane0Rows"] == 3 and event["counters"]["Lane1Rows"] == 1
    row[_chunk_chunk["FIELDS"].index("Lane0Rows")] = 2
    with pytest.raises(ValueError, match="lane payload"):
        _chunk_chunk["decode"](data, **args)
    trace = _chunk_perfetto_fixture()
    trace["schema_version"] = _chunk_chunk["VERSION"]
    trace["world_size"] = 256
    cross = trace["events"][2]
    cross["times_ticks"].update(BarrierBegin=2**55 + 282, BarrierEnd=2**55 + 288)
    path = tmp_path / "rank-0.chunks.json"
    _chunk_chunk["write_timeline"](path, trace)
    exported = json.loads(path.with_suffix(".perfetto.json").read_text())
    (barrier,) = [e for e in exported["traceEvents"] if e["name"].startswith("cross_group_barrier")]
    assert barrier["dur"] == 0.006
    assert "#eb9634" in path.with_suffix(".svg").read_text()
    assert "BarrierBegin_us" in path.with_suffix(".csv").read_text()
    cross["times_ticks"]["BarrierBegin"] = 2**55 + 279
    with pytest.raises(ValueError, match="barrier"):
        _chunk_chunk["perfetto_trace"](trace, "invalid.json")


@pytest.mark.parametrize("fault", ["precision", "duplicate", "overlap", "send"])
def test_chunk_perfetto_rejects_unvalidated_or_misleading_timelines(fault):
    trace = _chunk_perfetto_fixture()
    if fault == "precision":
        trace["precision_checked"] = False
    elif fault == "duplicate":
        trace["events"].append(copy.deepcopy(trace["events"][0]))
    elif fault == "overlap":
        trace["events"][1]["times_ticks"].update(Begin=2**55 + 240, WorkBegin=2**55 + 245)
    else:
        trace["events"][2]["times_ticks"]["FirstSend0"] = 2**55 + 149
    with pytest.raises(ValueError):
        _chunk_chunk["perfetto_trace"](trace, "rank-0.chunks.json")


@pytest.mark.parametrize("tokens", [1, 3, 4, 255, 256, 257, 4096, 16384])
@pytest.mark.parametrize("chunk_tokens", [256, 512, 1024])
def test_chunk_chunk_boundaries_and_storage(tokens, chunk_tokens):
    for world in (128, 256, 512):
        cores, count, words = _chunk_chunk["shape"](world, tokens, chunk_tokens)
        chunks = count // (world // 64 + 2)
        ranges = [_chunk_chunk["token_range"](c, tokens, chunk_tokens) for c in range(chunks)]
        assert ranges[0][0] == 0 and ranges[-1][1] == tokens
        assert all((a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
        assert cores * count * words * 8 <= 12 * 1024 * 1024


def test_chunk_notify_chunk_layout_matches_host_tiling(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    layout = runpy.run_path(str(_chunk_ROOT / "deep_ep/buffers/_elastic_layout.py"))
    source = tmp_path / "chunk_layout.cpp"
    source.write_text(
        r"""

#include "kernels/notify/tiling.hpp"
#include "kernels/combine/tiling.hpp"
#include <iostream>

int main()
{
    for (unsigned p : {128U, 256U, 512U})
        for (unsigned n : {4096U, 16384U})
            for (unsigned c : {256U, 512U, 1024U}) {
                auto t = ascend_deepep::MakeNotifyTiling(0, p, n, 6, 512, c, 7168);
                auto b = ascend_deepep::MakeCombineTiling(0, p, n, 6, 4 * n * 6, 4 * n * 6, c);
                std::cout << p << ' ' << n << ' ' << c << ' ' << t.chunks << ' ' << t.workspace_bytes << ' '
                          << t.forward_tile_offsets << ' ' << t.source_chunk_masks << ' '
                          << ascend_deepep::CombineWorkspaceBytes(b) << '\n';
            }
}
"""
    )
    binary = tmp_path / "chunk_layout"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(_chunk_ROOT / "csrc"), str(source), "-o", str(binary)], check=True
    )
    for line in subprocess.check_output([str(binary)], text=True).splitlines():
        world, tokens, size, count, nbytes, forward, masks, cbytes = map(int, line.split())
        notify = layout["workspace_layout"](world, tokens, 6, 512, size)
        combine = layout["combine_layout"](world, tokens, 6, 4 * tokens * 6, 4 * tokens * 6, size)
        assert count == (tokens + size - 1) // size + 3
        assert (nbytes, forward, masks) == tuple(
            (notify[n] for n in ("workspace_bytes", "forward_tile_offsets", "source_chunk_masks"))
        )
        assert cbytes == combine["workspace_bytes"]
        hint = layout["elastic_size_hint"](world, tokens, 6, 512, 4 * tokens * 6, 4 * tokens * 6)
        assert max(nbytes, cbytes) <= hint // 2


def test_chunk_qp_mask_matches_scheduler_for_ep512(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler:
        command = [compiler]
    elif importlib.util.find_spec("ziglang"):
        command = [sys.executable, "-m", "ziglang", "c++"]
    else:
        pytest.skip("Requires a host C++ compiler")
    source = tmp_path / "mask.cpp"
    source.write_text(
        r"""

#include "kernels/common.hpp"
#include "kernels/combine/tiling.hpp"
#include "runtime/udma_peer_masks.hpp"
#include "kernels/dispatch/peer_protocol.hpp"
#include <cassert>
#include <algorithm>
#include <initializer_list>

unsigned legacy_owner(unsigned sender, unsigned peer, unsigned world, unsigned owners, unsigned chunk)
{
    using namespace first_hit_schedule;
    unsigned order = Group(sender, world, peer) * Width(world) + Index(peer, world), index = order;
    for (unsigned local = sender / 8 * 8; local < sender / 8 * 8 + 8; ++local)
        if (Group(sender, world, local) * Width(world) + Index(local, world) < order) --index;
    index = index / 8 * 8 + (index % 8 + 8 - sender % 8) % 8;
    if (world == 64 && owners == 16 && index >= 48) return index % 8 + chunk % 2 * 8;
    return (index + (owners == 16 ? 8 : 0)) % owners;
}

unsigned window_end(unsigned chunk, unsigned logical_chunks, unsigned splits, unsigned num_so)
{
    unsigned logical = chunk < splits ? 0 : chunk - splits + 1, count = std::min(num_so, logical_chunks);
    if (count == 3 && logical_chunks >= 4) {
        unsigned a = (logical_chunks + 1) / 2, b = a + (logical_chunks - a + 1) / 2;
        return (logical < a ? a : logical < b ? b : logical_chunks) - 1 + splits - 1;
    }
    unsigned window = logical * count / logical_chunks;
    return ((window + 1) * logical_chunks + count - 1) / count - 1 + splits - 1;
}

int main()
{
    using namespace first_hit_schedule;
    for (unsigned world : {16U, 32U, 64U, 128U, 256U, 512U}) {
        unsigned min_count = ~0U, max_count = 0;
        for (unsigned rank = 0; rank < world; ++rank) {
            unsigned count = 0, used = 0;
            for (unsigned peer = 0; peer < world; ++peer) {
                unsigned mask = CombineQpMask(rank, peer, world);
                const uint64_t joint = deepep::udma_admission::PeerMask(rank, peer, world);
                assert(joint == deepep::udma_admission::PeerMask(peer, rank, world));
                assert((joint & mask) == mask && (joint >> 32U) == 0U);
                assert(mask == CombineQpMask(peer, rank, world));
                if (peer == rank) {
                    assert(mask == 0 && joint == 0);
                    continue;
                }
                assert(mask & 1U);
                if (peer / 8 == rank / 8)
                    assert(mask == 1 && joint == 1);
                else
                    for (unsigned parity = 0; parity < 2; ++parity) {
                        for (unsigned destination : {peer, rank}) {
                            const unsigned owner = ascend_deepep::DefaultClosPeerTableOwner(destination);
                            const uint64_t pair = uint64_t{3} << (2U * owner);
                            assert((joint & pair) == pair);
                        }
                        unsigned owner = CombineOwner(rank, peer, world, 16, parity);
                        assert(owner < 16 && (mask & (3U << (owner * 2))) == (3U << (owner * 2)));
                        for (unsigned owners : {1U, 2U, 4U, 8U, 16U})
                            assert(CombineOwner(rank, peer, world, owners, parity) ==
                                   legacy_owner(rank, peer, world, owners, parity));
                    }
                used |= mask;
                count += __builtin_popcount(mask);
            }
            if (world >= 32) assert(used == ~0U);
            min_count = std::min(min_count, count);
            max_count = std::max(max_count, count);
        }
        if (world == 128) assert(min_count == 558 && max_count == 566);
        if (world == 256) assert(min_count == 1153 && max_count == 1168);
        if (world == 512) assert(min_count == 2337 && max_count == 2368);
    }

    for (unsigned logical = 1; logical <= 64; ++logical)
        for (unsigned splits = 1; splits <= 4; ++splits)
            for (unsigned num_so : {1U, 2U, 3U, 4U, 7U, 64U, 128U}) {
                unsigned observed = 0;
                for (unsigned chunk = 0; chunk < logical + splits - 1; ++chunk)
                    observed += chunk == window_end(chunk, logical, splits, num_so);
                ascend_deepep::PutCombineTiling t{};
                t.chunks_per_group = logical + splits - 1;
                t.first_chunk_splits = splits;
                t.num_so = num_so;
                for (unsigned shared : {0U, 1U})
                    for (unsigned peers : {0U, 1U, 7U, 8U, 15U, 16U, 31U, 32U}) {
                        t.shared_jetty = shared;
                        unsigned expected = shared ? observed * peers : (peers ? observed : 0);
                        assert(ascend_deepep::CombineFixedOwnerSoReserve(t, peers) == expected);
                    }
            }
}
"""
    )
    binary = tmp_path / "mask"
    subprocess.run(
        [*command, "-std=c++17", "-O2", "-I" + str(_chunk_ROOT / "csrc"), str(source), "-o", str(binary)], check=True
    )
    subprocess.run([str(binary)], check=True)


def test_chunk_lane_phase_survives_sparse_chunks_and_group_reindexing():
    peers = (9, 17, 65, 73)
    ordinals = {peer: [0, 0] for peer in peers}
    sent = {peer: [[], []] for peer in peers}
    totals = dict.fromkeys(peers, 0)
    for chunk_id in range(67):
        for group in ((9, 17), (65, 73)):
            for peer in group:
                n = (0, 3, 1, 5, 2)[(chunk_id + peer) % 5]
                tokens = list(range(totals[peer], totals[peer] + n))
                for lane in (0, 1):
                    cursor = 0
                    while cursor < n:
                        issued = 0
                        while cursor < n and issued < 2:
                            selected = int(ordinals[peer][lane] % 4 == 3)
                            ordinals[peer][lane] += 1
                            token = tokens[cursor]
                            cursor += 1
                            if selected == lane:
                                sent[peer][lane].append(token)
                                issued += 1
                totals[peer] += n
                assert ordinals[peer][0] == ordinals[peer][1] == totals[peer]
    for peer in peers:
        assert set(sent[peer][0]).isdisjoint(sent[peer][1])
        assert sorted(sent[peer][0] + sent[peer][1]) == list(range(totals[peer]))
        assert len(sent[peer][1]) == totals[peer] // 4
    source = (_chunk_ROOT / "csrc/kernels/combine/combine.cpp").read_text(encoding="utf-8")
    cross = source.split("__aicore__ inline void CrossSend(", 1)[1].split("constexpr uint32_t kWeightThreads", 1)[0]
    assert "token_ordinal[lane][peer_indices[index]]++" in cross
    setup, chunk_loops = cross.split("for (uint32_t chunk =", 1)
    assert setup.count("token_ordinal[0][index] = token_ordinal[1][index] = 0U") == 1
    assert "for (uint32_t index = 0; index < all_peer_count; ++index)" in setup
    assert "token_ordinal[0][index] = token_ordinal[1][index] = 0" not in chunk_loops


_peer_ROOT = Path(__file__).parents[2]

_peer_peer = runpy.run_path(str(_peer_ROOT / "tests/elastic/combine_peers.py"))


def _peer_records(world=128, tokens=4096, rank=0, chunk_tokens=256):
    _, _, chunks, _, _ = _peer_peer["shape"](world, tokens, chunk_tokens)
    rows = [[[[[0] * 16 for _ in range(2)] for _ in range(chunks)] for _ in range(world)] for _ in range(2)]
    origin = 2**55
    for (direction, gateway, chunk, lane), (core, qp, window) in _peer_peer["expected_records"](
        rank, world, tokens, chunk_tokens
    ).items():
        begin = origin + window * 10000 + 100 + lane
        row = [
            _peer_peer["MAGIC"],
            1,
            1,
            direction,
            rank,
            core,
            gateway,
            chunk,
            lane,
            qp,
            begin,
            begin + 80,
            0,
            0,
            0,
            0,
        ]
        if not direction:
            row[12:] = [origin + window * 10000 + 1000, origin + window * 10000 + 1200, lane + 1, int(lane == 0)]
        rows[direction][gateway][chunk][lane] = row
    return rows


@pytest.mark.parametrize(
    "world,tokens", [(2, 1), (8, 257), (16, 4096), (64, 4096), (128, 4096), (256, 16384), (512, 16384)]
)
@pytest.mark.parametrize("chunk_tokens", [256, 512, 1024])
def test_peer_complete_coverage_and_bounded_wait_bank(world, tokens, chunk_tokens):
    events = _peer_peer["decode"](
        _peer_records(world, tokens, chunk_tokens=chunk_tokens),
        rank=0,
        world=world,
        tokens=tokens,
        sample=1,
        chunk_tokens=chunk_tokens,
    )
    expected = _peer_peer["expected_records"](0, world, tokens, chunk_tokens)
    assert len(events) == len(expected)
    for window in _peer_peer["window_ends"](tokens, chunk_tokens):
        for core in range(64):
            assert sum((d == 0 and c == core and (w == window) for (d, *_), (c, q, w) in expected.items())) <= 16
    if world == 64:
        assert any((e["chunk"] != e["window_end"] for e in events))
    else:
        assert all((e["chunk"] == e["window_end"] for e in events))
    assert math.prod(_peer_peer["shape"](world, tokens, chunk_tokens)) * 8 <= 17563648


@pytest.mark.parametrize(
    "field,value",
    [("Magic", 0), ("Version", 2), ("Rank", 1), ("Core", 63), ("Qp", 45), ("Observed", 0), ("Polls", 0), ("Flags", 0)],
)
def test_peer_reject_wrong_identity_or_censoring(field, value):
    rows = _peer_records()
    first = _peer_peer["window_ends"](4096)[0]
    rows[0][8][first][0][_peer_peer["FIELDS"].index(field)] = value
    with pytest.raises(ValueError):
        _peer_peer["decode"](rows, rank=0, world=128, tokens=4096, sample=1)


def test_peer_missing_completion_and_unexpected_slot():
    rows = _peer_records()
    first = _peer_peer["window_ends"](4096)[0]
    missing = copy.deepcopy(rows)
    missing[0][8][first][0] = [0] * 16
    with pytest.raises(ValueError, match="Missing peer"):
        _peer_peer["decode"](missing, rank=0, world=128, tokens=4096, sample=1)
    rows[0][8][0][0] = rows[0][8][first][0]
    with pytest.raises(ValueError, match="Unexpected peer"):
        _peer_peer["decode"](rows, rank=0, world=128, tokens=4096, sample=1)


def test_peer_perfetto_precise_local_origin_censored_waits_and_unique_barriers(tmp_path):
    events = _peer_peer["decode"](_peer_records(), rank=0, world=128, tokens=4096, sample=1)
    path = tmp_path / "rank-0.peers.json"
    artifact = _peer_peer["write_trace"](path, events=events, rank=0, world=128, tokens=4096, origin=2**55, sample=1)
    trace = json.loads(Path(artifact["perfetto"]).read_text())["traceEvents"]
    slices = [e for e in trace if e["ph"] == "X"]
    observations = [e for e in slices if e["cat"] == "combine.peer"]
    assert len(observations) == len(events)
    assert all((e["dur"] == 0.08 for e in observations))
    assert sum((e["cat"] == "combine.barrier" for e in slices)) == 64 * 3
    latest = artifact["summary"]["latest_pending"]
    assert latest and all((e["polls"] > 1 for e in latest))
    assert any(("arrival earlier" in e["name"] for e in observations))


def test_peer_cpp_python_record_abi():
    header = (_peer_ROOT / "csrc/kernels/combine/peer_profile.hpp").read_text()
    enum = header.split("enum Field : uint32_t {", 1)[1].split("};", 1)[0]
    assert [re.split("\\s*=", item.strip())[0] for item in enum.split(",")] == _peer_peer["FIELDS"]


@pytest.mark.parametrize("rank", range(8))
def test_ep8_window_source_pool_requires_every_owner(rank):
    records, args = _chunk_records(ep8_rank_pipeline=True)
    args["rank"] = rank
    chunks = len(records[0]) // 3
    ends = _combine_peers["window_ends"](args["tokens"])
    fields = {name: i for i, name in enumerate(_chunk_chunk["FIELDS"])}
    result = [[[0] * _chunk_chunk["WORDS"] for _ in records[0]] for _ in range(64)]
    for core in range(64):
        for chunk in range(chunks):
            end = next(e for e in ends if e >= chunk)
            begin = 32 if end != ends[-1] else 11
            kinds = ([4] if core < 11 else []) + ([2] if core >= 32 else []) + ([5] if core >= begin else [])
            for kind in kinds:
                index = (2 if kind == 5 else 1 if kind == 4 else 0) * chunks + chunk
                row = records[0 if kind in (4, 5) else 32][index].copy()
                row[fields["Core"]], row[fields["WindowEnd"]] = core, end
                row[fields["Flags"]] = (
                    (0x45600000 | 64 << 8 | (64 - begin))
                    if kind == 5
                    else _chunk_chunk["EP8_SINGLE_QP_FLAG"]
                    if kind == 4
                    else 0
                )
                if kind == 4:
                    row[fields["Peer"]] = (core if core < rank else core + 1) if core < 7 else rank
                result[core][index] = row
    _chunk_chunk["decode"](result, **args, ep8_rank_pipeline=True)
    result[62][2 * chunks] = [0] * _chunk_chunk["WORDS"]
    with pytest.raises(ValueError, match="Missing final-reduce"):
        _chunk_chunk["decode"](result, **args, ep8_rank_pipeline=True)


def test_ep8_summary_decoder_validates_compact_core_roles():
    rows = _summary_records(world=8)
    fields = {name: i for i, name in enumerate(_summary_FIELDS)}
    for core, row in enumerate(rows):
        row[fields["WithWeights"]] = 0
        row[fields["Role"]] = 0 if core < 11 else 2 if core >= 32 else 3
    decoded = _summary_metrics["decode_profile"](
        rows, rank=0, world=8, tokens=4096, topk=6, weights=False, sample=1, ep8_rank_pipeline=True
    )
    assert [d["role"] for d in decoded] == ["local"] * 11 + ["idle"] * 21 + ["gateway"] * 32
    rows[12][fields["Role"]] = 0
    with pytest.raises(ValueError, match="Missing/stale"):
        _summary_metrics["decode_profile"](
            rows, rank=0, world=8, tokens=4096, topk=6, weights=False, sample=1, ep8_rank_pipeline=True
        )


if __name__ == "__main__":
    main()
