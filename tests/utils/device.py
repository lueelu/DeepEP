# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Public-API adapter for generalized BF16/FP8 EP qualification.

No backend emulation or exception-to-skip fallback. Native calls and completion
waits must really work when the user opts into these tests.
"""

from contextlib import contextmanager
from dataclasses import replace

from .checks import assert_close, assert_dispatch
from .precision import (
    compare_dual,
    emulate_upstream_case,
    encode_upstream_fp8_row,
    quantize_bfloat16,
)
from .reference import Dispatch, dispatch, expected_expert_combine, expert_payload


def tensor(torch, values, shape, dtype, device):
    return torch.tensor(values, dtype=dtype, device=device).reshape(shape)


def _host_list(value):
    return value.cpu().tolist() if hasattr(value, "cpu") else list(value)


def _wait_if_async(event):
    """Insert a stream dependency when the returned wrapper holds an event.

    Otherwise this helper performs no wait; that is not proof of communication
    completion. It neither constructs nor requires an empty event wrapper.
    Future synchronous algorithm return types must follow the public contract
    defined with their implementation, not be inferred from this helper.
    """
    if getattr(event, "event", None) is not None:
        event.current_stream_wait()


def _valid_positions(case, expanded, handle, expected_rows):
    if not expanded or case.expert_alignment == 1:
        return tuple(range(expected_rows))
    psum = _host_list(handle.psum_num_recv_tokens_per_expert)
    positions, previous = [], 0
    for end in psum:
        start = (previous + case.expert_alignment - 1) // case.expert_alignment
        start *= case.expert_alignment
        if end < start:
            raise AssertionError("Expanded expert prefix sum precedes its aligned start")
        positions.extend(range(start, end))
        previous = end
    assert len(positions) == expected_rows, "Expanded valid row count differs"
    return tuple(positions)


def _semantic_counts(handle, fallback, alignment):
    if alignment == 1:
        return _host_list(fallback)
    exact = getattr(handle, "num_unaligned_recv_tokens_per_expert", None)
    if exact is None:
        raise AssertionError("Aligned dispatch must expose unaligned expert counts")
    return _host_list(exact)


def cast_to_fp8(torch, x):
    """Pinned upstream per-token/per-128 E4M3FN representation."""
    n, hidden = x.shape
    aligned = (hidden + 127) // 128 * 128
    padded = torch.nn.functional.pad(x, (0, aligned - hidden), value=0)
    grouped = padded.reshape(n, -1, 128)
    amax = grouped.abs().float().amax(dim=2).clamp(min=1e-4)
    values = (grouped * (448.0 / amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return (
        values.reshape(n, aligned)[:, :hidden].contiguous(),
        (amax / 448.0).contiguous(),
    )


def cast_from_fp8(torch, x):
    values, scales = x
    n, hidden = values.shape
    aligned = (hidden + 127) // 128 * 128
    padded = torch.nn.functional.pad(values, (0, aligned - hidden), value=0)
    restored = padded.float().reshape(n, -1, 128) * scales.float().reshape(n, -1, 1)
    return restored.reshape(n, aligned).to(torch.bfloat16)[:, :hidden].contiguous()


@contextmanager
def buffer_for(session, case, api):
    import deep_ep

    _, group, _ = session
    if api == "legacy":
        configs = [
            deep_ep.Buffer.get_dispatch_config(case.world_size),
            deep_ep.Buffer.get_combine_config(case.world_size),
        ]
        scaleup = max(c.get_scaleup_buffer_size_hint(case.hidden * 2, case.world_size) for c in configs)
        scaleout = max(c.get_scaleout_buffer_size_hint(case.hidden * 2, case.world_size) for c in configs)
        buffer = deep_ep.Buffer(group, num_scaleup_bytes=scaleup, num_scaleout_bytes=scaleout, explicitly_destroy=True)
    else:
        buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=max(map(len, case.x)),
            hidden=case.hidden,
            num_topk=case.topk,
            use_fp8_dispatch=case.dtype == "float8_e4m3fn",
            explicitly_destroy=True,
        )
    try:
        yield buffer
    finally:
        buffer.destroy()


def launch_dispatch(session, buffer, case, api, expanded, *, handle=None):
    import deep_ep

    torch, _, launch = session
    rank = launch.rank
    device = f"npu:{launch.local_rank}"
    n = len(case.x[rank])
    x = tensor(torch, case.x[rank], (n, case.hidden), torch.bfloat16, device)
    if case.dtype == "float8_e4m3fn":
        if api != "elastic":
            raise AssertionError("FP8 qualification is currently scoped to the upstream V2 path")
        x = cast_to_fp8(torch, x)
    index_dtype = getattr(deep_ep, "topk_idx_t", torch.int64)
    idx = tensor(torch, case.routes[rank], (n, case.topk), index_dtype, device)
    weights = None if case.weights is None else tensor(torch, case.weights[rank], (n, case.topk), torch.float32, device)
    if api == "legacy":
        if handle is None:
            per_rank, per_scaleout, per_expert, mask, event = buffer.get_dispatch_layout(idx, case.experts)
            _wait_if_async(event)
            args = dict(
                topk_idx=idx,
                topk_weights=weights,
                num_tokens_per_rank=per_rank,
                num_tokens_per_scaleout_rank=per_scaleout,
                num_tokens_per_expert=per_expert,
                is_token_in_rank=mask,
            )
        else:
            args = dict(handle=handle)
        rx, ri, rw, counts, new_handle, event = buffer.dispatch(x, **args)
        _wait_if_async(event)
        assert rx.dtype == torch.bfloat16
        if ri is not None:
            assert ri.dtype == torch.int64
        if rw is not None:
            assert rw.dtype == torch.float32
        return rx, ri, rw, counts, new_handle
    if handle is None:
        args = dict(
            topk_idx=idx,
            num_experts=case.experts,
            expert_alignment=case.expert_alignment,
            num_max_tokens_per_rank=max(map(len, case.x)),
            do_expand=expanded,
        )
    else:
        args = dict(handle=handle)  # Omitted do_expand must inherit the cached layout.
    rx, ri, rw, new_handle, event = buffer.dispatch(x, topk_weights=weights, **args)
    _wait_if_async(event)
    if case.dtype == "float8_e4m3fn":
        assert isinstance(rx, tuple) and rx[0].dtype == torch.float8_e4m3fn
        assert rx[1].dtype == torch.float32
    else:
        assert rx.dtype == torch.bfloat16
    if ri is not None:
        assert ri.dtype == index_dtype
    if rw is not None:
        assert rw.dtype == torch.float32
    counts = _semantic_counts(
        new_handle,
        new_handle.num_recv_tokens_per_expert_list,
        case.expert_alignment,
    )
    return rx, ri, rw, counts, new_handle


def snapshot(case, rank, expanded, received, *, elastic):
    rx, ri, rw, counts, handle = received
    observed_case = replace(case, x=emulate_upstream_case(case))
    oracle = dispatch(observed_case, expanded=expanded)
    n = len(oracle.ranks[rank])
    if elastic:
        field = "num_expanded_tokens" if expanded else "num_recv_tokens"
        observed_rows = getattr(handle, field)
        if expanded:
            assert observed_rows >= n, f"{field} is smaller than valid rows"
        else:
            assert observed_rows == n, f"{field} differs"
    else:
        assert rx.shape[0] == n, "V1 received row count differs"
    rx_bf16 = cast_from_fp8(__import__("torch"), rx) if case.dtype == "float8_e4m3fn" else rx
    positions = _valid_positions(case, expanded, handle, n)
    assert rx_bf16.shape[0] > max(positions, default=-1)
    all_x = rx_bf16.cpu().tolist()
    x = [all_x[index] for index in positions]
    all_indices = None if ri is None else ri.cpu().tolist()
    indices = None if all_indices is None else [all_indices[index] for index in positions]
    all_weights = None if rw is None else rw.cpu().tolist()
    weights = None if all_weights is None else [all_weights[index] for index in positions]
    assert_dispatch(observed_case, rank, expanded, x, indices, weights, counts)
    if case.dtype == "float8_e4m3fn":
        all_raw_values = rx[0].float().cpu().tolist()
        all_raw_scales = rx[1].float().cpu().tolist()
        raw_values = [all_raw_values[index] for index in positions]
        raw_scales = [all_raw_scales[index] for index in positions]
        candidates = list(oracle.ranks[rank])
        for i, values in enumerate(x):
            match = next(row for row in candidates if row.x == tuple(values))
            candidates.remove(match)
            expected_values, expected_scales = encode_upstream_fp8_row(case.x[match.source][match.token])
            assert_close(raw_values[i], expected_values, label="FP8 payload")
            assert_close(raw_scales[i], expected_scales, label="FP8 scales")
    return x, indices, weights, tuple(counts)


def observed_rows(case, rank, expanded, raw):
    """Recover test tags from actual outputs, not private native routing tables.

    Expanded duplicate slots are distinguished by their unique dispatch weight
    tags. Without weights their identical expert outputs are interchangeable.
    """
    x, indices, weights, counts = raw
    candidates = list(dispatch(case, expanded=expanded).ranks[rank])
    experts = [
        rank * (case.experts // case.world_size) + expert for expert, count in enumerate(counts) for _ in range(count)
    ]
    rows = []
    for i, values in enumerate(x):
        matches = [
            row
            for row in candidates
            if row.x == tuple(values)
            and (not expanded or row.expert == experts[i])
            and (not expanded or weights is None or row.weights == (weights[i],))
        ]
        assert matches, "Received token cannot be matched to independent input tags"
        row = matches[0]
        candidates.remove(row)
        rows.append(replace(row, x=tuple(values)))
    assert not candidates
    ranks = tuple(tuple(rows) if r == rank else () for r in range(case.world_size))
    return Dispatch(expanded, ranks, ())


def run_operation(session, buffer, case, api, expanded, operation):
    torch, _, launch = session
    rank = launch.rank
    received = launch_dispatch(session, buffer, case, api, expanded)
    raw = snapshot(case, rank, expanded, received, elastic=api == "elastic")
    if operation == "dispatch":
        return
    observed_case = replace(case, x=emulate_upstream_case(case))
    layout = observed_rows(observed_case, rank, expanded, raw)
    payloads = expert_payload(observed_case, layout, use_received_x=operation == "roundtrip")[rank]
    n = len(payloads)
    device = f"npu:{launch.local_rank}"
    payload_x = [p.x for p in payloads]
    payload_weights = None if case.weights is None else [p.weights[0] if expanded else p.weights for p in payloads]
    combine_rows = n
    if expanded and case.expert_alignment > 1:
        positions = _valid_positions(case, True, received[-1], n)
        combine_rows = received[-1].num_expanded_tokens
        padded_x = [[0.0] * case.hidden for _ in range(combine_rows)]
        padded_weights = None if payload_weights is None else [0.0] * combine_rows
        for position, values in zip(positions, payload_x):
            padded_x[position] = values
        if payload_weights is not None:
            for position, value in zip(positions, payload_weights):
                padded_weights[position] = value
        payload_x, payload_weights = padded_x, padded_weights
    x = tensor(torch, payload_x, (combine_rows, case.hidden), torch.bfloat16, device)
    shape = (combine_rows,) if expanded else (combine_rows, case.topk)
    weights = (
        None
        if case.weights is None
        else tensor(
            torch,
            payload_weights,
            shape,
            torch.float32,
            device,
        )
    )
    out, values, event = buffer.combine(x, received[-1], topk_weights=weights)
    _wait_if_async(event)
    benchmark_x, _ = expected_expert_combine(observed_case)
    benchmark_x = tuple(
        tuple(tuple(quantize_bfloat16(value) for value in row) for row in rank_rows) for rank_rows in benchmark_x
    )
    golden_x, golden_values = expected_expert_combine(case)
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == (len(case.x[rank]), case.hidden)
    report = compare_dual(
        out.cpu().tolist(),
        benchmark_x[rank],
        golden_x[rank],
        dtype=case.dtype,
        level="L2",
    )
    assert report.passed, report.to_dict()
    if golden_values is None:
        assert values is None
    else:
        assert values.dtype == torch.float32
        assert tuple(values.shape) == (len(case.x[rank]), case.topk)
        assert_close(values.cpu().tolist(), golden_values[rank], label="combined values")


def run_cached(session, buffer, case, api, expanded):
    """Reuse routing with new x; new-weight and multi-handle lifetimes are later tests."""
    rank = session[2].rank
    first = launch_dispatch(session, buffer, case, api, expanded)
    snapshot(case, rank, expanded, first, elastic=api == "elastic")
    changed = replace(
        case,
        x=tuple(tuple(tuple(v + 1 for v in x) for x in tokens) for tokens in case.x),
    )
    cached = launch_dispatch(session, buffer, changed, api, expanded, handle=first[-1])
    if api == "elastic":
        snapshot(changed, rank, expanded, cached, elastic=True)
    else:
        # V1 cached dispatch omits index/weight/count channels by contract.
        from collections import Counter

        expected = dispatch(changed).ranks[rank]
        actual = cached[0].cpu().tolist()
        assert Counter(map(tuple, actual)) == Counter(row.x for row in expected)
