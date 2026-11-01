"""Real-NPU precision test for the V2 API backed by intranode kernels.

Run one mode at a time on a single node, for example::

    python test/test_v2_elastic_v1_kernel.py --mode bf16 --ep-size 4
    python test/test_v2_elastic_v1_kernel.py --mode fp8 --ep-size 4
    python test/test_v2_elastic_v1_kernel.py --mode expand --ep-size 4
    python test/test_v2_elastic_v1_kernel.py --mode bf16 --ep-size 4 --uneven-tokens
    python test/test_v2_elastic_v1_kernel.py --mode expand --ep-size 8 --num-aiv 32 --local-stage-checks
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
from route_workloads import (
    add_arguments as add_route_arguments,
    options as route_options,
    generate as generate_routes,
    validate as validate_routes,
)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep.backends.clos import ElasticBuffer, topk_idx_t


PUBLIC_HANDLE_FIELDS = (
    "psum_num_recv_tokens_per_scaleup_rank",
    "psum_num_recv_tokens_per_expert",
    "num_unaligned_recv_tokens_per_expert",
    "recv_src_metadata",
    "dst_buffer_slot_idx",
)


def check_public_handle(handle, routes_by_rank, rank):
    """Independent CPU routing reference; never derive expected fields from a kernel handle."""
    world = len(routes_by_rank)
    k = routes_by_rank[rank].shape[1]
    experts = handle.num_experts // world
    capacity, alignment = handle.num_max_tokens_per_rank, handle.expert_alignment
    counts = [0] * experts
    received, rank_prefix = [], []
    for source, routes in enumerate(routes_by_rank):
        for token, route in enumerate(routes.tolist()):
            local = [
                (j, e % experts) for j, e in enumerate(route) if 0 <= e < handle.num_experts and e // experts == rank
            ]
            if local:
                received.append((source, token, local))
                for _, expert in local:
                    counts[expert] += 1
        rank_prefix.append(len(received))
    starts, aligned_ends, total = [], [], 0
    for count in counts:
        starts.append(total)
        total += (count + alignment - 1) // alignment * alignment
        aligned_ends.append(total)
    expert_prefix = [start + n for start, n in zip(starts, counts)] if handle.do_expand else aligned_ends
    metadata = torch.full((handle.num_recv_tokens, k + 2), -1, dtype=torch.int32)
    tails = starts.copy()
    for row, (source, token, local) in enumerate(received):
        metadata[row, 0] = source * capacity + token
        metadata[row, 1] = source * k + local[0][0]
        if handle.do_expand:
            for lane, expert in local:
                metadata[row, lane + 2] = tails[expert]
                tails[expert] += 1
    slots = torch.full(routes_by_rank[rank].shape, -1, dtype=torch.int32)
    peer_tails = [0] * world
    for token, route in enumerate(routes_by_rank[rank].tolist()):
        seen = set()
        for lane, expert in enumerate(route):
            if not 0 <= expert < handle.num_experts:
                continue
            peer = expert // experts
            if peer not in seen:
                slots[token, lane] = rank * capacity + peer_tails[peer]
                peer_tails[peer] += 1
                seen.add(peer)
    expected = [
        torch.tensor(rank_prefix, dtype=torch.int32),
        torch.tensor(expert_prefix, dtype=torch.int32),
        torch.tensor(counts, dtype=torch.int32),
        metadata,
        slots,
    ]
    snapshots = []
    for name, reference in zip(PUBLIC_HANDLE_FIELDS, expected):
        actual = getattr(handle, name)
        assert actual.dtype == torch.int32 and actual.is_contiguous(), name
        value = actual.cpu()
        torch.testing.assert_close(value, reference, rtol=0, atol=0, msg=name)
        snapshots.append((name, actual.data_ptr(), value.clone()))
    single_slots = getattr(handle, "_single_source_slots", None)
    if single_slots is not None:
        mapping = metadata[:, 2:]
        single_reference = torch.where((mapping >= 0).sum(1) == 1, mapping.amax(1), -1)
        torch.testing.assert_close(single_slots.cpu(), single_reference, rtol=0, atol=0)
        snapshots.append(("_single_source_slots", single_slots.data_ptr(), single_reference.clone()))
    return snapshots


def check_public_handle_unchanged(handle, snapshots):
    for name, pointer, saved in snapshots:
        actual = getattr(handle, name)
        assert actual.data_ptr() == pointer, name
        torch.testing.assert_close(actual.cpu(), saved, rtol=0, atol=0, msg=name)


def check_public_handle_edges(buffer, rank, args):
    # Small independent cases include no receiver, an empty sender, fixed-capacity
    # padding, odd K, duplicate destinations and uneven expert alignment.
    saved = []
    k = args.num_topk
    experts = args.num_experts // args.ep_size
    for empty in (False, True):
        routes = []
        for source in range(args.ep_size):
            rows = []
            for t in range(0 if source == 0 else 9 + source):
                peer = (source + t) % args.ep_size
                rows.append([-1] * k if empty else [peer * experts, peer * experts + 1] + [-1] * (k - 2))
            routes.append(torch.tensor(rows, dtype=topk_idx_t).reshape(-1, k))
        for expanded, cpu_sync in ((False, True), (False, False), (True, True)):
            x = torch.ones((routes[rank].size(0), args.hidden), dtype=torch.bfloat16, device="npu")
            result, _, _, h, _ = buffer.dispatch(
                x,
                topk_idx=routes[rank].to("npu"),
                num_experts=args.num_experts,
                num_sms=args.num_aiv,
                expert_alignment=8 if expanded else 1,
                do_expand=expanded,
                do_cpu_sync=cpu_sync,
            )
            snapshots = check_public_handle(h, routes, rank)
            _, _, _, cached, _ = buffer.dispatch(x, handle=h, num_sms=args.num_aiv, do_expand=expanded)
            assert cached is h
            check_public_handle_unchanged(h, snapshots)
            saved.append((h, snapshots, result))
    for h, snapshots, _ in saved:
        check_public_handle_unchanged(h, snapshots)
    print(f"[rank {rank}] public handle edge cases passed: 6 fresh + 6 cached", flush=True)


def check_single_source_combine(buffer, rank, args):
    """Differential full-combine checks, including all 16-bit payload patterns.

    Public metadata is also checked against CPU routes. The fused/local public
    aggregate result is not used to manufacture expected table entries.
    """
    device = f"npu:{args.first_device + rank}"
    checks = 0
    experts = args.num_experts // args.ep_size
    # ElasticBuffer's public payload contract is BF16. FP16 remains covered by
    # the separate local/V1 tests; do not widen the public API for this check.
    for dtype in (torch.bfloat16,):
        for alignment, pattern in ((1, "single"), (8, "mixed"), (16, "empty")):
            routes = []
            for source in range(args.ep_size):
                rows = []
                for token in range(17 + source % 3):
                    peer = (source + token) % args.ep_size
                    row = [-1] * args.num_topk
                    if pattern != "empty":
                        row[0] = peer * experts
                    if pattern == "mixed" and token % 2:
                        # Duplicate expert routes count as two contributions.
                        row[1] = row[0]
                        row[2] = ((peer + 1) % args.ep_size) * experts + 1
                    rows.append(row)
                routes.append(torch.tensor(rows, dtype=topk_idx_t))
            source = torch.ones((len(routes[rank]), args.hidden), dtype=dtype, device=device)
            recv, _, _, h, _ = buffer.dispatch(
                source,
                topk_idx=routes[rank].to(device),
                num_experts=args.num_experts,
                num_sms=args.num_aiv,
                do_expand=True,
                expert_alignment=alignment,
            )
            snapshots = check_public_handle(h, routes, rank)
            table = h._single_source_slots
            assert table is not None
            # Fresh inputs each time; a cached table must never cache a GM pointer.
            for repeat in (0, 1):
                bits = (torch.arange(recv.numel(), dtype=torch.int32) + rank * 997 + repeat * 31).to(torch.int16)
                expert_x = bits.view(dtype).reshape(recv.shape).to(device)
                values = torch.arange(recv.size(0), dtype=torch.float32, device=device) / 128
                bias = (torch.full_like(source, 1), torch.full_like(source, -1)) if repeat else None
                outputs = []
                modes = [(False, False, 0), (True, False, 0), (True, True, 0)]
                if args.aggregate_pipeline_chunk:
                    modes.append((True, False, args.aggregate_pipeline_chunk))
                for enabled, early, chunk in modes:
                    buffer._use_single_source_combine = enabled
                    buffer._use_early_single_source_combine = early
                    buffer._aggregate_pipeline_chunk_tokens = chunk
                    out, weights, event = buffer.combine(
                        expert_x,
                        h,
                        topk_weights=values if repeat else None,
                        bias=bias,
                        num_sms=args.num_aiv,
                        async_with_compute_stream=True,
                    )
                    event.current_stream_wait(release_handle=True)
                    outputs.append((out.cpu().view(torch.int16), weights.cpu() if weights is not None else None))
                assert torch.equal(outputs[0][0], outputs[1][0]), (
                    "single-source bitwise mismatch",
                    dtype,
                    pattern,
                    repeat,
                    torch.nonzero(outputs[0][0] != outputs[1][0])[:8].tolist(),
                )
                assert torch.equal(outputs[0][0], outputs[2][0]), (
                    "early single-source bitwise mismatch",
                    pattern,
                    repeat,
                )
                if repeat:
                    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
                    torch.testing.assert_close(outputs[0][1], outputs[2][1], rtol=0, atol=0)
                if args.aggregate_pipeline_chunk:
                    assert torch.equal(outputs[0][0], outputs[3][0]), ("pipeline bitwise mismatch", pattern, repeat)
                    if repeat:
                        torch.testing.assert_close(outputs[0][1], outputs[3][1], rtol=0, atol=0)
                checks += 1
                _, _, _, cached, _ = buffer.dispatch(source, handle=h, do_expand=True, num_sms=args.num_aiv)
                assert cached is h and cached._single_source_slots is table
                check_public_handle_unchanged(h, snapshots)
    print(f"[rank {rank}] single-source combine checks passed: {checks}, BF16, fresh/cached", flush=True)
    buffer._use_single_source_combine = True
    buffer._use_early_single_source_combine = False
    buffer._aggregate_pipeline_chunk_tokens = args.aggregate_pipeline_chunk


def _source_x(source_rank, num_tokens, hidden, mode, delta=0):
    token = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    column = torch.arange(hidden, dtype=torch.float32).unsqueeze(0)
    values = 2 * (source_rank + 1) + token.remainder(4) + column.remainder(2) * 0.5 + delta
    for digit in range(min(hidden, 4)):
        values[:, digit] = (token[:, 0] // (16**digit)).remainder(16)
    dtype = torch.float8_e4m3fn if mode == "fp8" else torch.bfloat16
    payload = values.to(dtype)
    if mode != "fp8":
        return payload
    scales = 1.0 + source_rank * 0.125 + torch.arange(num_tokens).float().unsqueeze(1) * 0.015625
    return payload, scales


def _topk(source_rank, num_tokens, world_size, num_experts, num_topk, mode, args):
    # Mode changes output representation only, never the input routing.
    key = (source_rank, num_tokens, world_size, num_experts, num_topk, args.seed, tuple(route_options(args).items()))
    cache = vars(args).setdefault("_route_cache", {})
    if key not in cache:
        cache[key] = torch.tensor(
            generate_routes(
                source_rank,
                num_tokens,
                world_size,
                num_experts // world_size,
                num_topk,
                args.seed,
                **route_options(args),
            ),
            dtype=topk_idx_t,
        )
    return cache[key]


def _routes(source_rank, token, world_size, num_experts, num_topk, mode, args):
    return _topk(source_rank, _rank_num_tokens(source_rank, args), world_size, num_experts, num_topk, mode, args)[
        token
    ].tolist()


def _route_weights(rank, args):
    n = _rank_num_tokens(rank, args)
    weights = _weights(n, args.num_topk)
    routes = _topk(rank, n, args.ep_size, args.num_experts, args.num_topk, args.mode, args)
    return weights.masked_fill(routes < 0, 0)


def _weights(num_tokens, num_topk):
    return torch.tensor(
        [[(route + 1) / 4 + token / 1024 for route in range(num_topk)] for token in range(num_tokens)],
        dtype=torch.float32,
    )


def _payload(value):
    return value[0] if isinstance(value, tuple) else value


def _rank_num_tokens(rank, args):
    # Uneven mode varies local T while args.num_tokens remains the shared capacity.
    return max(1, args.num_tokens - rank) if args.uneven_tokens else args.num_tokens


def _check_received(
    rank,
    args,
    payload,
    scales,
    recv_topk_idx,
    recv_weights,
    handle,
    delta,
    weight_scale=1,
):
    experts_per_rank = args.num_experts // args.ep_size
    rank_prefix, _, _, src_idx, _, _ = handle._dispatch_layout
    rank_prefix = rank_prefix.cpu()
    src_idx = src_idx.cpu()
    mapping = handle._expand_layout[2].cpu() if args.mode == "expand" else None
    previous_end = 0
    # Rank prefixes delimit each source's rows, even when source lengths differ.
    for source_rank in range(args.ep_size):
        end = int(rank_prefix[source_rank, rank])
        source = _source_x(source_rank, _rank_num_tokens(source_rank, args), args.hidden, args.mode, delta)
        source_payload = _payload(source)
        source_scales = source[1] if args.mode == "fp8" else None
        for row in range(previous_end, end):
            token = int(src_idx[row])
            routes = _routes(source_rank, token, args.ep_size, args.num_experts, args.num_topk, args.mode, args)
            for route, expert in enumerate(routes):
                is_local = expert // experts_per_rank == rank
                expected_weight = weight_scale * ((route + 1) / 4 + token / 1024)
                if mapping is not None:
                    slot = int(mapping[row, route])
                    assert (slot >= 0) == is_local
                    if is_local:
                        assert torch.equal(payload[slot], source_payload[token])
                        if recv_weights is not None:
                            assert recv_weights[slot] == expected_weight
                else:
                    if recv_topk_idx is not None:
                        assert int(recv_topk_idx[row, route]) == (expert % experts_per_rank if is_local else -1)
                    if recv_weights is not None:
                        assert recv_weights[row, route] == (expected_weight if is_local else 0.0)
            if mapping is None:
                assert torch.equal(payload[row], source_payload[token])
            if scales is not None:
                if mapping is None:
                    assert torch.equal(scales[row], source_scales[token])
                else:
                    for route, expert in enumerate(routes):
                        if expert // experts_per_rank == rank:
                            assert torch.equal(scales[int(mapping[row, route])], source_scales[token])
        previous_end = end
    assert previous_end == handle.num_recv_tokens


def _expected_combine(rank, args, delta=0):
    experts_per_rank = args.num_experts // args.ep_size
    num_tokens = _rank_num_tokens(rank, args)
    source = _payload(_source_x(rank, num_tokens, args.hidden, args.mode, delta)).float()
    result = torch.full((num_tokens, args.hidden), 3.0, dtype=torch.float32)
    for token in range(num_tokens):
        destinations = [
            expert // experts_per_rank
            for expert in _routes(rank, token, args.ep_size, args.num_experts, args.num_topk, args.mode, args)
            if expert >= 0
        ]
        if args.mode != "expand":
            # Non-expanded dispatch sends one copy per destination rank.
            destinations = sorted(set(destinations))
        for dst in destinations:
            result[token] += source[token] + dst + 1
    return result.to(torch.bfloat16)


def check_workspace_reuse(buffer, rank, args, handle, bias):
    """Keep outputs alive across consecutive combines and cached dispatches.

    Intentionally skew rank arrival and alternate weight presence (and hence
    payload extent). No host readback/barrier between the two combine calls.
    Expected tensors come only from CPU source/routing, never device outputs.
    """
    pending = []
    n = _rank_num_tokens(rank, args)
    for iteration in range(args.reuse_iterations):
        delta = iteration % 4
        source = _source_x(rank, n, args.hidden, args.mode, delta)
        x = tuple(t.to("npu") for t in source) if isinstance(source, tuple) else source.to("npu")
        scale = 1 + iteration % 2
        received, _, weights, cached, _ = buffer.dispatch(
            x,
            handle=handle,
            topk_weights=(_route_weights(rank, args) * scale).to("npu"),
            num_sms=args.num_aiv,
            do_expand=args.mode == "expand",
        )
        assert cached is handle
        expert_x = _payload(received).to(torch.bfloat16) + rank + 1
        expected = _expected_combine(rank, args, delta)
        for weighted in (False, True):
            if rank == (iteration + int(weighted)) % args.ep_size:
                time.sleep(0.02)
            output, out_weights, _ = buffer.combine(
                expert_x, handle, topk_weights=weights if weighted else None, bias=bias, num_sms=args.num_aiv
            )
            # Queue an actual local consumer, and retain both original and copy.
            # Next combine may not overwrite the previous reduction's inputs.
            pending.append(
                (
                    output,
                    output.clone(),
                    out_weights,
                    expected,
                    _route_weights(rank, args) * scale if weighted else None,
                )
            )
    for output, consumed, weights, expected, expected_weights in pending:
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(consumed.cpu(), expected, rtol=0, atol=0)
        if expected_weights is None:
            assert weights is None
        else:
            torch.testing.assert_close(weights.cpu(), expected_weights, rtol=0, atol=0)
    if args.reuse_iterations:
        print(
            f"[rank {rank}] workspace reuse passed: {args.reuse_iterations} cached dispatches, "
            f"{2 * args.reuse_iterations} combines, delayed ranks, exact CPU golden",
            flush=True,
        )


def run_rank(rank, args):
    device = args.first_device + rank
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(args.ep_size),
    )
    torch.npu.set_device(device)
    dist.init_process_group(
        backend="hccl",
        init_method=args.dist_init_method,
        rank=rank,
        world_size=args.ep_size,
    )
    buffer = None
    try:
        group = dist.group.WORLD
        num_topk = args.num_topk
        local_tokens = _rank_num_tokens(rank, args)

        # All ranks reserve the same capacity; local token counts may differ.
        buffer = ElasticBuffer(
            group,
            num_bytes=ElasticBuffer.get_buffer_size_hint(
                group,
                args.num_tokens,
                args.hidden,
                num_topk,
                use_fp8_dispatch=args.mode == "fp8",
            ),
            num_max_tokens_per_rank=args.num_tokens,
            hidden=args.hidden,
            num_topk=num_topk,
            use_fp8_dispatch=args.mode == "fp8",
            explicitly_destroy=True,
            shmem_ip_port=args.shmem_ip_port,
            transport=args.transport,
            udma_schedule=args.udma_schedule,
            udma_auto_policy=False,
            udma_dispatch_peer_order="rank",
            udma_combine_peer_order="rank",
            udma_workers_per_peer=args.udma_workers_per_peer,
            udma_qps_per_worker=args.udma_qps_per_worker,
            udma_peers_per_worker=args.udma_peers_per_worker,
            udma_wqes_per_visit=args.udma_wqes_per_visit,
        )
        buffer._aggregate_pipeline_chunk_tokens = args.aggregate_pipeline_chunk
        print(f"[rank {rank}] aggregate pipeline chunk={args.aggregate_pipeline_chunk}", flush=True)

        # Fresh dispatch: check payload, routes, weights, and FP8 scales.
        source = _source_x(rank, local_tokens, args.hidden, args.mode)
        x = tuple(t.to("npu") for t in source) if isinstance(source, tuple) else source.to("npu")
        topk_idx = _topk(rank, local_tokens, args.ep_size, args.num_experts, num_topk, args.mode, args).to("npu")
        weights = _route_weights(rank, args).to("npu")
        recv_x, recv_idx, recv_weights, handle, event = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=weights,
            num_experts=args.num_experts,
            num_sms=args.num_aiv,
            do_expand=args.mode == "expand",
            expert_alignment=2 if args.mode == "expand" else 1,
        )
        assert event.event is None
        assert handle.do_expand == (args.mode == "expand")
        assert (recv_idx is None) == (args.mode == "expand")
        routes_by_rank = [
            _topk(p, _rank_num_tokens(p, args), args.ep_size, args.num_experts, num_topk, args.mode, args)
            for p in range(args.ep_size)
        ]
        public_snapshot = check_public_handle(handle, routes_by_rank, rank)

        # Snapshot for immediate checks, but retain the original NPU outputs.
        recv_payload = _payload(recv_x).cpu().clone()
        recv_scales = recv_x[1].cpu().clone() if isinstance(recv_x, tuple) else None
        recv_idx_cpu = recv_idx.cpu().clone() if recv_idx is not None else None
        recv_weights_cpu = recv_weights.cpu().clone()
        _check_received(
            rank,
            args,
            recv_payload,
            recv_scales,
            recv_idx_cpu,
            recv_weights_cpu,
            handle,
            delta=0,
        )

        # Combine: compare against an independent source-rank reference.
        # Own the expert result: dispatch returns a separately allocated
        # tensor with zero_copy=False.
        expert_x = (recv_payload.float().to(torch.bfloat16) + rank + 1).to("npu")
        bias = (
            torch.ones((local_tokens, args.hidden), dtype=torch.bfloat16, device="npu"),
            torch.full((local_tokens, args.hidden), 2, dtype=torch.bfloat16, device="npu"),
        )
        combined_x, combined_weights, combine_event = buffer.combine(
            expert_x,
            handle,
            topk_weights=recv_weights_cpu.to("npu"),
            bias=bias,
            num_sms=args.num_aiv,
        )
        assert combine_event.event is None
        torch.testing.assert_close(combined_x.cpu(), _expected_combine(rank, args), rtol=0, atol=0)
        torch.testing.assert_close(combined_weights.cpu(), _route_weights(rank, args), rtol=0, atol=0)

        # Cached dispatch/combine: reuse routing metadata with new payload and weights.
        cached_source = _source_x(rank, local_tokens, args.hidden, args.mode, delta=1)
        cached_x = (
            tuple(t.to("npu") for t in cached_source) if isinstance(cached_source, tuple) else cached_source.to("npu")
        )
        updated_weights = weights * 2
        cached_recv_x, cached_idx, cached_weights, cached_handle, cached_event = buffer.dispatch(
            cached_x,
            handle=handle,
            topk_weights=updated_weights,
            num_sms=args.num_aiv,
            do_expand=args.mode == "expand",
        )
        assert cached_event.event is None
        assert cached_handle is handle
        assert cached_idx is None and cached_weights is not None
        cached_payload = _payload(cached_recv_x).cpu().clone()
        cached_scales = cached_recv_x[1].cpu().clone() if isinstance(cached_recv_x, tuple) else None
        cached_weights_cpu = cached_weights.cpu().clone()
        _check_received(
            rank,
            args,
            cached_payload,
            cached_scales,
            None,
            cached_weights_cpu,
            handle,
            delta=1,
            weight_scale=2,
        )

        cached_expert_x = (cached_payload.float().to(torch.bfloat16) + rank + 1).to("npu")
        cached_combined_x, cached_combined_weights, _ = buffer.combine(
            cached_expert_x,
            handle,
            topk_weights=cached_weights_cpu.to("npu"),
            bias=bias,
            num_sms=args.num_aiv,
        )
        torch.testing.assert_close(cached_combined_x.cpu(), _expected_combine(rank, args, delta=1), rtol=0, atol=0)
        torch.testing.assert_close(
            cached_combined_weights.cpu(),
            _route_weights(rank, args) * 2,
            rtol=0,
            atol=0,
        )

        # A cached dispatch may also omit route weights entirely.
        payload_only_x, payload_only_idx, payload_only_weights, payload_only_handle, _ = buffer.dispatch(
            cached_x,
            handle=handle,
            num_sms=args.num_aiv,
            do_expand=args.mode == "expand",
        )
        assert payload_only_handle is handle
        assert payload_only_idx is None and payload_only_weights is None
        _check_received(
            rank,
            args,
            _payload(payload_only_x).cpu().clone(),
            payload_only_x[1].cpu().clone() if isinstance(payload_only_x, tuple) else None,
            None,
            None,
            handle,
            delta=1,
        )

        # Re-read the original NPU tensors after later communication; checking
        # only the CPU snapshots above would not detect overwritten outputs.
        _check_received(
            rank,
            args,
            _payload(recv_x).cpu(),
            recv_x[1].cpu() if isinstance(recv_x, tuple) else None,
            recv_idx.cpu() if recv_idx is not None else None,
            recv_weights.cpu(),
            handle,
            delta=0,
        )
        torch.testing.assert_close(combined_x.cpu(), _expected_combine(rank, args), rtol=0, atol=0)
        torch.testing.assert_close(combined_weights.cpu(), _route_weights(rank, args), rtol=0, atol=0)

        check_workspace_reuse(buffer, rank, args, handle, bias)

        if args.local_stage_checks:
            from test_local_expand_aggregate import check_local_stages

            local_result = check_local_stages(buffer.runtime, f"npu:{device}", args.num_aiv)
            print(
                f"[rank {rank}] local expand/aggregate passed: {json.dumps(local_result, sort_keys=True)}", flush=True
            )

        if args.public_handle_edges:
            check_public_handle_edges(buffer, rank, args)
        if args.single_source_checks:
            check_single_source_combine(buffer, rank, args)
        check_public_handle_unchanged(handle, public_snapshot)

        dist.barrier(device_ids=[device])
        buffer.destroy()
        buffer = None
        check_public_handle_unchanged(handle, public_snapshot)
        if rank == 0:
            print(f"V2 API / intranode kernel precision passed: mode={args.mode}, ep_size={args.ep_size}")
    finally:
        # Collective destroy is unsafe after an asymmetric rank failure.
        buffer = None
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    add_route_arguments(parser)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--transport", choices=("mte", "udma"), default="udma")
    parser.add_argument("--udma-schedule", choices=("peer", "token"), default="peer")
    parser.add_argument("--udma-peers-per-worker", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--udma-wqes-per-visit", type=int, choices=(4, 8, 16, 32), default=8)
    parser.add_argument("--udma-workers-per-peer", type=int, choices=(1, 2), default=1)
    parser.add_argument("--udma-qps-per-worker", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--reuse-iterations",
        type=int,
        default=0,
        help="stress consecutive weighted/unweighted combines with staggered rank arrival",
    )
    parser.add_argument("--mode", choices=("bf16", "fp8", "expand"), required=True)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--first-device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--uneven-tokens", action="store_true")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--num-aiv", type=int, default=8)
    parser.add_argument("--public-handle-edges", action="store_true")
    parser.add_argument("--single-source-checks", action="store_true")
    parser.add_argument("--aggregate-pipeline-chunk", type=int, choices=(0, 32, 64), default=0)
    parser.add_argument(
        "--local-stage-checks",
        action="store_true",
        help="also check local BF16/FP16 expand, cached expand and aggregate boundaries",
    )
    parser.add_argument("--dist-init-method", default="tcp://127.0.0.1:29500")
    parser.add_argument("--shmem-ip-port", default=None)
    args = parser.parse_args()
    try:
        validate_routes(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.aggregate_pipeline_chunk and (
        args.transport != "udma"
        or args.udma_schedule not in ("peer", "token")
        or args.num_aiv not in (32, 64)
        or args.mode != "expand"
        or (args.udma_schedule == "token" and (args.num_aiv != 32 or args.ep_size > 8))
    ):
        parser.error("aggregate pipeline requires expanded peer UDMA and 32/64 AIV")
    if args.reuse_iterations < 0:
        parser.error("reuse-iterations must be nonnegative")
    if args.ep_size < 2 or args.num_tokens < 1 or args.hidden < 1:
        parser.error("ep-size must be >= 2; num-tokens and hidden must be positive")
    if args.single_source_checks and (
        args.transport != "udma" or args.num_tokens < 19 or args.num_topk < 3 or args.num_experts < args.ep_size * 2
    ):
        parser.error("single-source checks require UDMA, capacity >=19, K>=3 and >=2 experts/rank")
    if args.uneven_tokens and args.num_tokens < 2:
        parser.error("uneven-tokens requires num-tokens >= 2")
    if args.num_experts < args.ep_size or args.num_experts % args.ep_size:
        parser.error("num-experts must be positive and divisible by ep-size")
    if args.public_handle_edges and (
        args.num_tokens < args.ep_size + 8 or args.num_topk < 2 or args.num_experts < args.ep_size * 2
    ):
        parser.error("public-handle-edges needs capacity >= ep-size+8, topk >= 2 and >= 2 experts/rank")
    if not 1 <= args.num_topk <= min(128, args.num_experts):
        parser.error("num-topk must be between 1 and min(128, num-experts)")
    mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()
