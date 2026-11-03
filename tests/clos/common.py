"""Shared independent CPU references and precision checks for Clos validation."""

import sys
from pathlib import Path
from typing import Any, Tuple, Union

import torch
from deep_ep.backends.direct import Buffer, topk_idx_t

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/clos"))
from route_workloads import options as route_options, generate as generate_routes

DispatchX = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _make_topk_idx(rank: int, args) -> torch.Tensor:
    opts = route_options(args)
    key = (
        rank,
        args.ep_size,
        args.experts_per_rank,
        args.num_tokens,
        args.num_topk,
        args.seed,
        tuple(opts.items()),
        topk_idx_t,
    )
    cache = vars(args).setdefault("_topk_reference_cache", {})
    if key not in cache:
        rows = generate_routes(
            rank, args.num_tokens, args.ep_size, args.experts_per_rank, args.num_topk, args.seed, **opts
        )
        cache[key] = torch.tensor(rows, dtype=topk_idx_t).reshape(args.num_tokens, args.num_topk)
    return cache[key]


def _compute_dtype(args) -> torch.dtype:
    return torch.bfloat16 if args.dtype in ("bf16", "fp8") else torch.float16


def _make_x(rank: int, args, delta: float = 0.0) -> torch.Tensor:
    values = (
        torch.arange(args.num_tokens * args.hidden, dtype=torch.float32)
        .remainder_(2048)
        .reshape(args.num_tokens, args.hidden)
    )
    return (values * 0.01 + rank * 16.0 + delta).to(_compute_dtype(args))


def _cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden = x.size(1)
    aligned_hidden = (hidden + 127) // 128 * 128
    x_padded = torch.nn.functional.pad(x, (0, aligned_hidden - hidden))
    x_view = x_padded.view(x.size(0), -1, 128)
    amax = x_view.abs().float().amax(dim=2).clamp(1e-4)
    quantized = (x_view * (448.0 / amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    scales = (amax / 448.0).contiguous().T.contiguous().T
    return quantized.view(x.size(0), aligned_hidden)[:, :hidden].contiguous(), scales


def _make_dispatch_x(
    rank: int,
    args,
    delta: float = 0.0,
    use_int_scales: bool = False,
) -> DispatchX:
    x = _make_x(rank, args, delta)
    if args.dtype != "fp8":
        return x
    x_fp8, scales = _cast_to_fp8(x)
    if use_int_scales:
        scale_values = torch.arange(scales.numel(), dtype=torch.int32).reshape(scales.shape)
        scale_values += rank * scales.numel() + int(delta * 1000)
        scales = scale_values.T.contiguous().T
    return x_fp8, scales


def _to_npu(value: DispatchX) -> DispatchX:
    if isinstance(value, tuple):
        x = value[0].to(device="npu")
        scales = value[1].to(device="npu")
        if not value[1].is_contiguous():
            scales = scales.T.contiguous().T
        return x, scales
    return value.to(device="npu")


def _x_tensor(value: DispatchX) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _dequantize_x(value: DispatchX, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, tuple):
        return value
    x, scales = value
    if x.numel() == 0:
        return x.to(dtype)
    hidden = x.size(1)
    aligned_hidden = (hidden + 127) // 128 * 128
    x_padded = torch.nn.functional.pad(x.float(), (0, aligned_hidden - hidden))
    restored = x_padded.view(x.size(0), -1, 128) * scales.float().unsqueeze(2)
    return restored.view(x.size(0), aligned_hidden)[:, :hidden].to(dtype)


def _make_weights(rank: int, args) -> torch.Tensor:
    values = torch.arange(args.num_tokens * args.num_topk, dtype=torch.float32).reshape(args.num_tokens, args.num_topk)
    return values * 0.001 + rank + 0.25


def _layout_reference(topk_idx: torch.Tensor, args):
    num_experts = args.ep_size * args.experts_per_rank
    token, slot = (topk_idx != -1).nonzero(as_tuple=True)
    experts = topk_idx[token, slot].long()
    expert_counts = torch.bincount(experts, minlength=num_experts).to(torch.int32)
    membership = torch.zeros(args.num_tokens, args.ep_size, dtype=torch.bool)
    membership[token, experts // args.experts_per_rank] = True
    rank_counts = membership.sum(dim=0, dtype=torch.int32)
    return rank_counts, expert_counts, membership


def _channel_metadata(membership: torch.Tensor, args):
    counts = torch.zeros(args.ep_size, args.num_aiv, dtype=torch.int32)
    send_head = torch.full((args.num_tokens, args.ep_size), -1, dtype=torch.int32)
    tokens_per_channel = (args.num_tokens + args.num_aiv - 1) // args.num_aiv
    for channel in range(args.num_aiv):
        begin = min(channel * tokens_per_channel, args.num_tokens)
        end = min(begin + tokens_per_channel, args.num_tokens)
        routed = membership[begin:end]
        send_head[begin:end] = torch.where(routed, routed.cumsum(dim=0, dtype=torch.int32) - 1, -1)
        counts[:, channel] = membership[begin:end].sum(dim=0, dtype=torch.int32)
    inclusive = counts.cumsum(dim=1, dtype=torch.int32)
    exclusive = torch.zeros_like(inclusive)
    exclusive[:, 1:] = inclusive[:, :-1]
    return inclusive, exclusive, send_head


def _make_x_rows(rank: int, args, tokens: torch.Tensor, delta: float):
    # Convert absolute element indices BEFORE remainder: the original FP32
    # arange rounds indices above 2**24. Integer modulo would change the data.
    indices = tokens.long().unsqueeze(1) * args.hidden
    indices = indices + torch.arange(args.hidden, dtype=torch.int64)
    values = indices.to(torch.float32).remainder_(2048)
    return (values * 0.01 + rank * 16.0 + delta).to(_compute_dtype(args))


def _dispatch_reference_plan(rank: int, args):
    key = (
        rank,
        args.ep_size,
        args.experts_per_rank,
        args.num_tokens,
        args.num_topk,
        args.num_aiv,
        args.seed,
        args.expert_alignment,
        topk_idx_t,
    )
    cached = getattr(args, "_dispatch_reference_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    rank_counts, expert_counts, recv_channel_prefix = [], [], []
    source_tokens, recv_topk, recv_weights = [], [], []
    expert_begin = rank * args.experts_per_rank
    expert_end = expert_begin + args.experts_per_rank
    tokens_per_channel = (args.num_tokens + args.num_aiv - 1) // args.num_aiv
    for source_rank in range(args.ep_size):
        topk = _make_topk_idx(source_rank, args)
        counts, per_expert, membership = _layout_reference(topk, args)
        rank_counts.append(counts)
        expert_counts.append(per_expert)
        # Only the source rank's send_head is checked locally. Other sources
        # need just this destination's channel counts, not full T x EP scans.
        if source_rank == rank:
            local_prefix, _, local_send_head = _channel_metadata(membership, args)
        channel_counts = torch.zeros(args.num_aiv, dtype=torch.int32)
        for channel in range(args.num_aiv):
            begin = channel * tokens_per_channel
            channel_counts[channel] = membership[begin : begin + tokens_per_channel, rank].sum(dtype=torch.int32)
        exclusive = torch.zeros_like(channel_counts)
        exclusive[1:] = channel_counts.cumsum(0, dtype=torch.int32)[:-1]
        recv_channel_prefix.append(exclusive)
        # Channels partition contiguous token ranges; nonzero preserves the
        # original source-rank/channel/token ordering and rank deduplication.
        tokens = membership[:, rank].nonzero().flatten()
        source_tokens.append(tokens)
        selected = topk.index_select(0, tokens)
        local = (selected >= expert_begin) & (selected < expert_end)
        recv_topk.append(torch.where(local, selected - expert_begin, -1))
        weights = _make_weights(source_rank, args).index_select(0, tokens)
        recv_weights.append(torch.where(local, weights, 0.0))

    local_counts = torch.stack(expert_counts).sum(0, dtype=torch.int32)[expert_begin:expert_end]
    local_counts = (local_counts + args.expert_alignment - 1) // args.expert_alignment * args.expert_alignment
    plan = {
        "source_tokens": source_tokens,
        "recv_src_idx": torch.cat(source_tokens).to(torch.int32),
        "recv_topk_idx": torch.cat(recv_topk),
        "recv_topk_weights": torch.cat(recv_weights),
        "rank_prefix": torch.stack(rank_counts).cumsum(0, dtype=torch.int32),
        "local_channel_prefix": local_prefix,
        "recv_channel_prefix": torch.stack(recv_channel_prefix),
        "send_head": local_send_head,
        "expert_counts": local_counts,
    }
    args._dispatch_reference_cache = key, plan
    return plan


def _dispatch_reference(
    rank: int,
    args,
    delta: float = 0.0,
    use_int_scales: bool = False,
):
    plan = _dispatch_reference_plan(rank, args)
    num_recv = plan["recv_src_idx"].numel()
    recv_x = torch.empty(
        (num_recv, args.hidden), dtype=(torch.float8_e4m3fn if args.dtype == "fp8" else _compute_dtype(args))
    )
    recv_scales = None
    if args.dtype == "fp8":
        num_scales = (args.hidden + 127) // 128
        recv_scales = torch.empty((num_recv, num_scales), dtype=(torch.int32 if use_int_scales else torch.float32))
    offset = 0
    for source_rank, tokens in enumerate(plan["source_tokens"]):
        for begin in range(0, tokens.numel(), 128):
            selected = tokens[begin : begin + 128]
            payload = _make_x_rows(source_rank, args, selected, delta)
            end = offset + selected.numel()
            if recv_scales is not None:
                payload, scales = _cast_to_fp8(payload)
                if use_int_scales:
                    scales = (
                        selected[:, None] * num_scales
                        + torch.arange(num_scales)
                        + source_rank * args.num_tokens * num_scales
                        + int(delta * 1000)
                    ).to(torch.int32)
                recv_scales[offset:end] = scales
            recv_x[offset:end] = payload
            offset = end
    result = {key: value for key, value in plan.items() if key != "source_tokens"}
    result["recv_x"] = (recv_x, recv_scales) if recv_scales is not None else recv_x
    return result


def _expand_reference(
    recv_x: DispatchX,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    aligned_expert_counts: torch.Tensor,
):
    expert_offsets = torch.zeros(aligned_expert_counts.numel() + 1, dtype=torch.int32)
    expert_offsets[1:] = aligned_expert_counts.cumsum(dim=0, dtype=torch.int32)
    actual_expert_counts = torch.zeros_like(aligned_expert_counts)
    recv_to_expanded_idx = torch.full(recv_topk_idx.shape, -1, dtype=torch.int32)
    tails = expert_offsets[:-1].clone()

    num_expanded_tokens = int(expert_offsets[-1])
    recv_x_tensor = _x_tensor(recv_x)
    expanded_x = torch.zeros((num_expanded_tokens, recv_x_tensor.size(1)), dtype=recv_x_tensor.dtype)
    expanded_x_scales = None
    if isinstance(recv_x, tuple):
        expanded_x_scales = torch.zeros((num_expanded_tokens, recv_x[1].size(1)), dtype=recv_x[1].dtype)
    expanded_topk_weights = torch.zeros(num_expanded_tokens, dtype=recv_topk_weights.dtype)
    for token in range(recv_topk_idx.size(0)):
        for route in range(recv_topk_idx.size(1)):
            local_expert = int(recv_topk_idx[token, route])
            if local_expert < 0:
                continue
            expanded_slot = int(tails[local_expert])
            tails[local_expert] += 1
            actual_expert_counts[local_expert] += 1
            recv_to_expanded_idx[token, route] = expanded_slot
            expanded_x[expanded_slot] = recv_x_tensor[token]
            if expanded_x_scales is not None:
                expanded_x_scales[expanded_slot] = recv_x[1][token]
            expanded_topk_weights[expanded_slot] = recv_topk_weights[token, route]

    return {
        "expanded_x": ((expanded_x, expanded_x_scales) if expanded_x_scales is not None else expanded_x),
        "expanded_topk_weights": expanded_topk_weights,
        "expert_offsets": expert_offsets,
        "actual_expert_counts": actual_expert_counts,
        "recv_to_expanded_idx": recv_to_expanded_idx,
    }


def _combine_reference(
    rank: int,
    args,
    bias_0=None,
    bias_1=None,
    delta: float = 0.0,
):
    source_x = _dequantize_x(_make_dispatch_x(rank, args, delta=delta), _compute_dtype(args))
    _, _, membership = _layout_reference(_make_topk_idx(rank, args), args)
    accumulator = torch.zeros_like(source_x, dtype=torch.float32)
    if bias_0 is not None:
        accumulator += bias_0.float()
    if bias_1 is not None:
        accumulator += bias_1.float()
    for dst_rank in range(args.ep_size):
        routed = membership[:, dst_rank]
        accumulator[routed] += source_x[routed].float()
    combined_weights = _make_weights(rank, args)
    combined_weights[_make_topk_idx(rank, args) == -1] = 0
    return accumulator.to(source_x.dtype), combined_weights


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if isinstance(actual, tuple) or isinstance(expected, tuple):
        if not isinstance(actual, tuple) or not isinstance(expected, tuple):
            raise AssertionError(f"{name} FP8 tuple mismatch")
        _assert_equal(f"{name}.x", actual[0], expected[0])
        _assert_equal(f"{name}.x_scales", actual[1], expected[1])
        return
    actual = actual.cpu()
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(
            f"{name} metadata mismatch: actual={actual.dtype}/"
            f"{tuple(actual.shape)}, expected={expected.dtype}/"
            f"{tuple(expected.shape)}"
        )
    if not torch.equal(actual, expected):
        mismatch = torch.nonzero(actual != expected, as_tuple=False)
        first_index = tuple(mismatch[0].tolist())
        raise AssertionError(
            f"{name} mismatch at {mismatch.size(0)} locations; "
            f"first={mismatch[:16].tolist()}; "
            f"first_actual={actual[first_index].item()}, "
            f"first_expected={expected[first_index].item()}"
        )


def _assert_recv_storage(name: str, tensor: DispatchX, buffer: Buffer, zero_copy: bool) -> None:
    if isinstance(tensor, tuple):
        _assert_recv_storage(f"{name}.x", tensor[0], buffer, zero_copy)
        _assert_recv_storage(f"{name}.x_scales", tensor[1], buffer, zero_copy)
        return
    if not tensor.is_contiguous():
        raise AssertionError(f"{name} receive output must be contiguous")
    if tensor.numel() == 0:
        return
    workspace_begin = int(buffer.native.get_workspace_ptr())
    workspace_end = workspace_begin + buffer.num_nvl_bytes
    tensor_begin = tensor.data_ptr()
    tensor_end = tensor_begin + tensor.numel() * tensor.element_size()
    in_workspace = workspace_begin <= tensor_begin and tensor_end <= workspace_end
    overlaps_workspace = tensor_begin < workspace_end and workspace_begin < tensor_end
    if (zero_copy and not in_workspace) or (not zero_copy and overlaps_workspace):
        raise AssertionError(
            f"{name} storage mismatch (zero_copy={zero_copy}): "
            f"tensor=[{tensor_begin}, {tensor_end}), "
            f"workspace=[{workspace_begin}, {workspace_end})"
        )
