"""Real-NPU precision test for the V1 intranode communication path."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
from route_workloads import (
    add_arguments as add_route_arguments,
    options as route_options,
    generate as generate_routes,
    validate as validate_routes,
)
from typing import Any, Callable, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep.backends.clos import Buffer, Config, topk_idx_t
from deep_ep.backends.clos.config import _MAX_NUM_AIV
from deep_ep.backends.clos import ExpandHandle, EventOverlap


DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024
PROFILE_OPERATORS = (
    "layout",
    "dispatch",
    "expand",
    "cached-expand",
    "aggregate",
    "combine",
    "dispatch-no-topk",
    "combine-no-topk",
    "cached-dispatch",
    "cached-combine",
)


class _ProfileComplete(Exception):
    pass


def _map_test_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, ExpandHandle):
        return ExpandHandle(
            fn(value.expert_offsets),
            fn(value.actual_expert_counts),
            fn(value.recv_to_expanded_idx),
            value.num_tokens_per_expert_list,
            value.num_recv_tokens,
            value.num_expanded_tokens,
            value.num_topk,
            value.expert_alignment,
        )
    if isinstance(value, tuple):
        return tuple(_map_test_tensors(v, fn) for v in value)
    if isinstance(value, list):
        return [_map_test_tensors(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map_test_tensors(v, fn) for k, v in value.items()}
    return value


class _V1StreamTestBuffer:
    """Exercise early release; numerical expectations remain the CPU golden."""

    operations = {"get_dispatch_layout", "dispatch", "combine", "expand", "aggregate"}

    def __init__(self, buffer, args):
        self.buffer, self.args = buffer, args
        self.producer = torch.npu.Stream() if args.stream_lifetime else None
        self.delay_storage = {}
        self.calls = 0

    def __getattr__(self, name):
        method = getattr(self.buffer, name)
        if name not in self.operations:
            return method

        def call(*args, **kwargs):
            kwargs.setdefault("allocate_on_comm_stream", self.args.allocate_on_comm_stream)
            if not self.args.stream_lifetime:
                return method(*args, **kwargs)
            return self._stress(name, method, args, kwargs)

        return call

    def _delay(self, stream):
        # Queued arithmetic gives allocation pressure an opportunity to overlap
        # the custom kernel. No host sleep or device synchronize before release.
        with torch.npu.stream(stream):
            if stream.stream_id not in self.delay_storage:
                a = torch.ones((1024, 1024), device="npu", dtype=torch.float16)
                self.delay_storage[stream.stream_id] = (a, a.clone(), torch.empty_like(a))
            a, b, out = self.delay_storage[stream.stream_id]
            for _ in range(32):
                torch.mm(a, b, out=out)

    @staticmethod
    def _pressure(specs, stream):
        junk = []
        with torch.npu.stream(stream):
            for shape, dtype in set(specs):
                for _ in range(2):
                    # Same byte-size allocator bins, with native float32 fill.
                    # uint8 Fill falls back to AICPU on some CANN releases.
                    elements = 1
                    for dimension in shape:
                        elements *= dimension
                    nbytes = elements * torch.empty((), dtype=dtype).element_size()
                    tensor = torch.empty(((nbytes + 3) // 4,), dtype=torch.float32, device="npu")
                    tensor.fill_(123.0)
                    junk.append(tensor)
        return junk

    def _assert_allocation(self, data, input_ptrs, caller):
        expected = (self.buffer.get_comm_stream() if self.args.allocate_on_comm_stream else caller).npu_stream
        snapshot = torch.npu.memory_snapshot()

        def check(t):
            if not t.numel() or t.data_ptr() in input_ptrs:
                return t
            if self.args.zero_copy:
                lo = self.buffer._native.get_workspace_ptr()
                if lo <= t.data_ptr() < lo + self.args.num_bytes:
                    return t
            for segment in snapshot:
                if segment["address"] <= t.data_ptr() < segment["address"] + segment["total_size"]:
                    if segment["stream"] != expected:
                        raise AssertionError(f"allocation stream mismatch: {segment['stream']} != {expected}")
                    return t
            raise AssertionError("owned output missing from NPU allocator snapshot")

        _map_test_tensors(data, check)

    def _stress(self, name, method, args, kwargs):
        caller = torch.npu.current_stream()
        self.calls += 1
        explicit = self.calls % 2 == 1
        producer = self.producer if explicit else caller
        specs, input_ptrs = [], set()

        def clone_input(t):
            clone = t.clone()
            specs.append((tuple(clone.shape), clone.dtype))
            input_ptrs.add(clone.data_ptr())
            return clone

        with torch.npu.stream(producer):
            producer.wait_stream(caller)
            previous = kwargs.get("previous_event")
            if previous is not None:
                previous.current_stream_wait()
            copies = _map_test_tensors((args, kwargs), clone_input)
            copies[1]["previous_event"] = Buffer.capture() if explicit else None
        self._delay(self.buffer.get_comm_stream())
        result = method(*copies[0], **copies[1])
        # Retain only the completion event, not EventOverlap.extra_tensors.
        event = EventOverlap(result[-1].event)
        data = result[:-1]
        del result, copies
        junk_inputs = self._pressure(specs, producer)
        event.current_stream_wait()
        self._assert_allocation(data, input_ptrs, caller)
        if not self.args.zero_copy:
            self._delay(caller)
            output_specs = []

            def consume(t):
                output_specs.append((tuple(t.shape), t.dtype))
                return t.clone()

            consumed = _map_test_tensors(data, consume)
            if name == "expand" and kwargs.get("handle") is not None:
                # Public cached expand returns the original handle object.
                consumed = (*consumed[:-1], kwargs["handle"])
            del data, event
            allocation_stream = self.buffer.get_comm_stream() if self.args.allocate_on_comm_stream else caller
            junk_outputs = self._pressure(output_specs, allocation_stream)
            # Reclaim only after the pressure writes and queued consumption;
            # golden checks outside this wrapper inspect the consumed copies.
            caller.synchronize()
            allocation_stream.synchronize()
            data = consumed
            del junk_outputs
        elif name == "expand" and kwargs.get("handle") is not None:
            data = (*data[:-1], kwargs["handle"])
        producer.synchronize()
        del junk_inputs
        return (*data, Buffer.capture())


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


DispatchX = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


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


def _cpu_index_select_x(value: DispatchX, index: torch.Tensor) -> DispatchX:
    if isinstance(value, tuple):
        return (
            value[0].cpu().index_select(0, index),
            value[1].cpu().index_select(0, index),
        )
    return value.cpu().index_select(0, index)


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


def _aggregate_reference(
    expanded_x: torch.Tensor,
    recv_to_expanded_idx: torch.Tensor,
    expanded_topk_weights: torch.Tensor,
    expanded_values: torch.Tensor,
):
    num_recv_tokens, num_topk = recv_to_expanded_idx.shape
    accumulator = torch.zeros((num_recv_tokens, expanded_x.size(1)), dtype=torch.float32)
    aggregated_values = torch.zeros((num_recv_tokens, num_topk), dtype=expanded_values.dtype)
    for token in range(num_recv_tokens):
        for route in range(num_topk):
            expanded_slot = int(recv_to_expanded_idx[token, route])
            if expanded_slot < 0:
                continue
            accumulator[token] += expanded_x[expanded_slot].float() * expanded_topk_weights[expanded_slot]
            aggregated_values[token, route] = expanded_values[expanded_slot]
    return accumulator.to(expanded_x.dtype), aggregated_values


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


def _consume_profile_result(result: Any) -> None:
    if not isinstance(result, (tuple, list)) or not result:
        return
    event = result[-1]
    if event is not None and getattr(event, "event", None) is not None:
        event.current_stream_wait(release_handle=True)


def _profile_operator(
    rank: int,
    args,
    operator: str,
    fn: Callable[[], Any],
    *,
    sample_callback=None,
    guard_iters: int = 0,
    synchronize_each_iteration: bool = False,
    before_iteration=None,
) -> None:
    if guard_iters < 0 or (guard_iters and sample_callback is None):
        raise ValueError("profile guards require a sample callback and a nonnegative count")
    os.makedirs(args.profile_dir, exist_ok=True)

    for _ in range(args.profile_warmup):
        if before_iteration is not None:
            before_iteration()
        _consume_profile_result(fn())
        if synchronize_each_iteration:
            torch_npu.npu.synchronize()
    torch_npu.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=True,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
        host_sys=[],
        sys_io=False,
        sys_interconnection=False,
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=1,
            active=1,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )

    prof.start()
    for step in range(2):
        guards = guard_iters if step == 1 and sample_callback is not None else 0
        for sample in range(-guards, args.profile_iters + guards):
            # Keep rank alignment outside sample markers, including warmup and guards.
            if before_iteration is not None:
                before_iteration()
            if step == 1 and sample_callback is not None:
                sample_callback(sample, lambda: _consume_profile_result(fn()))
            else:
                _consume_profile_result(fn())
            # Stream waits only order device work. Drain each invocation on the
            # host as well so released large outputs can be reused instead of
            # accumulating while Python queues subsequent iterations. Keep this
            # outside the callback's end marker and all measured kernel times.
            if synchronize_each_iteration:
                torch_npu.npu.synchronize()
        torch_npu.npu.synchronize()
        prof.step()
    prof.stop()
    torch_npu.npu.synchronize()

    if rank == 0:
        detail = (
            f"measured_iters={args.profile_iters}; guard_iters_per_side={guard_iters}; "
            f"captured_iters={args.profile_iters + 2 * guard_iters}"
            if sample_callback is not None
            else f"iters_per_step={args.profile_iters}"
        )
        print(
            f"profile operator {operator} written to {args.profile_dir}: "
            f"{detail}; host_sync_each_iteration={synchronize_each_iteration}"
        )
    raise _ProfileComplete


def validate_launch_args(args) -> None:
    if not args.master_addr:
        raise ValueError("--master-addr must not be empty")
    if args.master_port <= 0 or args.master_port > 65535:
        raise ValueError("--master-port must be in [1, 65535]")
    if args.nproc_per_node is None:
        if args.node_rank != 0:
            raise ValueError("--node-rank requires --nproc-per-node")
        return
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be greater than 0")
    if args.ep_size % args.nproc_per_node:
        raise ValueError("--ep-size must be divisible by --nproc-per-node")
    num_nodes = args.ep_size // args.nproc_per_node
    if args.node_rank < 0 or args.node_rank >= num_nodes:
        raise ValueError(f"--node-rank must be in [0, {num_nodes})")


def global_rank_for_local(local_rank: int, args) -> int:
    if args.nproc_per_node is None:
        return local_rank
    return args.node_rank * args.nproc_per_node + local_rank


def local_process_count(args) -> int:
    return args.ep_size if args.nproc_per_node is None else args.nproc_per_node


def effective_dist_init_method(args) -> str:
    if args.dist_init_method:
        return args.dist_init_method
    return f"tcp://{args.master_addr}:{args.master_port}"


def effective_shmem_ip_port(args) -> str:
    if args.shmem_ip_port:
        return args.shmem_ip_port
    shmem_port = args.master_port + 1
    if shmem_port > 65535:
        raise ValueError("--shmem-ip-port is required when --master-port is 65535")
    return f"tcp://{args.master_addr}:{shmem_port}"


def _profile_bf16_transport(rank: int, args, buffer, config) -> None:
    """Prepare only the inputs needed by the requested communication profile.

    This is not a precision run. Run without --profile for full independent
    golden checks, including expand/aggregate and all cached metadata.
    """
    topk_idx = _make_topk_idx(rank, args).to(device="npu")
    x = _to_npu(_make_dispatch_x(rank, args))
    counts, _, expert_counts, membership, event = buffer.get_dispatch_layout(
        topk_idx, args.ep_size * args.experts_per_rank, async_finish=True
    )
    event.current_stream_wait(release_handle=True)
    dispatch_kwargs = dict(
        num_tokens_per_rank=counts,
        num_tokens_per_expert=expert_counts,
        is_token_in_rank=membership,
        expert_alignment=args.expert_alignment,
        config=config,
        async_finish=True,
        zero_copy=args.zero_copy,
    )
    if args.profile in ("dispatch", "combine"):
        dispatch_kwargs.update(
            topk_idx=topk_idx,
            topk_weights=_make_weights(rank, args).to(device="npu"),
        )
    if args.profile == "dispatch":
        _profile_operator(rank, args, args.profile, lambda: buffer.dispatch(x, **dispatch_kwargs))
    recv_x, _, recv_weights, _, handle, event = buffer.dispatch(x, **dispatch_kwargs)
    event.current_stream_wait(release_handle=True)
    torch_npu.npu.synchronize()
    combine_kwargs = dict(config=config, async_finish=True)
    if args.profile == "combine":
        bias_0 = torch.full_like(x, 0.5)
        bias_1 = (
            torch.arange(args.num_tokens * args.hidden, dtype=torch.float32)
            .remainder_(2048)
            .reshape(args.num_tokens, args.hidden)
            * 0.001
        ).to(dtype=_compute_dtype(args), device="npu")
        combine_kwargs.update(topk_weights=recv_weights.clone(), bias=(bias_0, bias_1))
    else:
        cached_x = _to_npu(_make_dispatch_x(rank, args, delta=1.0))
        cached_kwargs = dict(handle=handle, config=config, async_finish=True, zero_copy=args.zero_copy)
        if args.profile == "cached-dispatch":
            _profile_operator(rank, args, args.profile, lambda: buffer.dispatch(cached_x, **cached_kwargs))
        recv_x, _, _, _, _, event = buffer.dispatch(cached_x, **cached_kwargs)
        event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
    # Combine overwrites dispatch SHMEM payload; retain an independent input.
    combine_x = recv_x.clone()
    _profile_operator(rank, args, args.profile, lambda: buffer.combine(combine_x, handle, **combine_kwargs))


def run_rank(local_rank: int, args) -> None:
    rank = global_rank_for_local(local_rank, args)
    os.environ["RANK"] = str(rank)
    device_id = args.first_device + local_rank
    os.environ["LOCAL_RANK"] = str(device_id)
    os.environ["WORLD_SIZE"] = str(args.ep_size)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.npu.set_device(device_id)
    dist.init_process_group(
        backend="hccl",
        init_method=effective_dist_init_method(args),
        rank=rank,
        world_size=args.ep_size,
    )
    group = dist.new_group(ranks=list(range(args.ep_size)))
    dist.barrier(group=group, device_ids=[device_id])

    if args.num_aiv is None:
        dispatch_config = Buffer.get_dispatch_config(args.ep_size)
        combine_config = Buffer.get_combine_config(args.ep_size)
        if dispatch_config.num_aiv != combine_config.num_aiv:
            raise AssertionError("default dispatch/combine configs must use the same num_aiv")
        args.num_aiv = dispatch_config.num_aiv
        config = Config(
            args.num_aiv, udma_peers_per_worker=args.udma_peers_per_worker, udma_wqes_per_visit=args.udma_wqes_per_visit
        )
        config_source = "default-aiv"
    else:
        config = Config(
            args.num_aiv,
            6,
            256,
            6,
            128,
            udma_peers_per_worker=args.udma_peers_per_worker,
            udma_wqes_per_visit=args.udma_wqes_per_visit,
        )
        config_source = "explicit"

    profile_transport = (
        args.dtype == "bf16"
        and not args.inject_error
        and args.profile in ("dispatch", "cached-dispatch", "combine", "cached-combine")
    )
    reference = None if profile_transport else _dispatch_reference(rank, args)
    if args.inject_error:
        reference_x = _x_tensor(reference["recv_x"])
        if reference_x.numel() == 0:
            raise AssertionError("--inject-error requires at least one received token")
        first_value = float(reference_x.view(-1)[0])
        reference_x.view(-1)[0] = first_value + 1.0
    buffer = None
    try:
        buffer = Buffer(
            group,
            num_nvl_bytes=args.num_bytes,
            transport=args.transport,
            udma_schedule=args.udma_schedule,
            udma_auto_policy=False,
            udma_dispatch_peer_order="rank",
            udma_combine_peer_order="rank",
            udma_workers_per_peer=1,
            udma_qps_per_worker=1,
            explicitly_destroy=True,
            shmem_ip_port=args.shmem_ip_port,
        )
        buffer = _V1StreamTestBuffer(buffer, args)
        buffer.barrier()

        if profile_transport:
            _profile_bf16_transport(rank, args, buffer, config)

        topk_idx_cpu = _make_topk_idx(rank, args)
        topk_idx = topk_idx_cpu.to(device="npu")
        topk_weights = _make_weights(rank, args).to(device="npu")
        x = _to_npu(_make_dispatch_x(rank, args))
        if args.dtype == "fp8" and x[1].size(1) > 1:
            expected_scale_stride = (1, x[1].size(0))
            if x[1].is_contiguous() or x[1].stride() != expected_scale_stride:
                raise AssertionError(
                    f"FP8 input scales must use DeepEP V1's column-major layout, got stride={x[1].stride()}"
                )
        if args.profile == "layout":
            _profile_operator(
                rank,
                args,
                "layout",
                lambda: buffer.get_dispatch_layout(
                    topk_idx,
                    args.ep_size * args.experts_per_rank,
                    async_finish=True,
                ),
            )
        (
            num_tokens_per_rank,
            _,
            num_tokens_per_expert,
            is_token_in_rank,
            layout_event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            args.ep_size * args.experts_per_rank,
            async_finish=True,
        )
        layout_event.current_stream_wait(release_handle=True)

        dispatch_kwargs = {
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "num_tokens_per_rank": num_tokens_per_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "expert_alignment": args.expert_alignment,
            "config": config,
            "async_finish": True,
            "zero_copy": args.zero_copy,
        }
        if args.profile == "dispatch":
            _profile_operator(
                rank,
                args,
                "dispatch",
                lambda: buffer.dispatch(x, **dispatch_kwargs),
            )
        recv_x, recv_topk_idx, recv_topk_weights, counts, handle, event = buffer.dispatch(x, **dispatch_kwargs)
        event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()

        _assert_recv_storage("recv_x", recv_x, buffer, args.zero_copy)
        _assert_recv_storage("recv_topk_idx", recv_topk_idx, buffer, args.zero_copy)
        _assert_recv_storage("recv_topk_weights", recv_topk_weights, buffer, args.zero_copy)
        _assert_equal("recv_x", recv_x, reference["recv_x"])
        _assert_equal("recv_src_idx", handle[3], reference["recv_src_idx"])
        _assert_equal("recv_topk_idx", recv_topk_idx, reference["recv_topk_idx"])
        _assert_equal(
            "recv_topk_weights",
            recv_topk_weights,
            reference["recv_topk_weights"],
        )
        _assert_equal("rank_prefix", handle[0], reference["rank_prefix"])
        _assert_equal(
            "local_channel_prefix",
            handle[1],
            reference["local_channel_prefix"],
        )
        _assert_equal(
            "recv_channel_prefix",
            handle[2],
            reference["recv_channel_prefix"],
        )
        _assert_equal("send_head", handle[5], reference["send_head"])
        _assert_equal(
            "expert_counts",
            torch.tensor(counts, dtype=torch.int32),
            reference["expert_counts"],
        )

        expand_reference = _expand_reference(
            reference["recv_x"],
            reference["recv_topk_idx"],
            reference["recv_topk_weights"],
            reference["expert_counts"],
        )
        expand_kwargs = {
            "topk_weights": recv_topk_weights,
            "num_tokens_per_expert": counts,
            "expert_alignment": args.expert_alignment,
            "config": config,
            "async_finish": True,
        }
        if args.profile == "expand":
            _profile_operator(
                rank,
                args,
                "expand",
                lambda: buffer.expand(
                    recv_x,
                    recv_topk_idx,
                    **expand_kwargs,
                ),
            )
        (
            expanded_x,
            expanded_topk_weights,
            expand_handle,
            expand_event,
        ) = buffer.expand(
            recv_x,
            recv_topk_idx,
            **expand_kwargs,
        )
        expand_event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        if expanded_topk_weights is None:
            raise AssertionError("expand did not return top-k weights")
        expanded_x_tensor = _x_tensor(expanded_x)
        reference_expanded_x_tensor = _x_tensor(expand_reference["expanded_x"])
        if (
            expanded_x_tensor.shape != reference_expanded_x_tensor.shape
            or expanded_x_tensor.dtype != reference_expanded_x_tensor.dtype
        ):
            raise AssertionError("expanded_x shape or dtype mismatch")
        if (
            expanded_topk_weights.shape != expand_reference["expanded_topk_weights"].shape
            or expanded_topk_weights.dtype != expand_reference["expanded_topk_weights"].dtype
        ):
            raise AssertionError("expanded_topk_weights shape or dtype mismatch")
        if expand_handle.num_recv_tokens != _x_tensor(recv_x).size(0):
            raise AssertionError("expand handle recv token count mismatch")
        if expand_handle.num_expanded_tokens != sum(counts):
            raise AssertionError("expand handle expanded token count mismatch")
        if expand_handle.num_topk != args.num_topk:
            raise AssertionError("expand handle top-k count mismatch")
        if expand_handle.expert_alignment != args.expert_alignment:
            raise AssertionError("expand handle expert alignment mismatch")
        if expand_handle.num_tokens_per_expert_list != counts:
            raise AssertionError("expand handle expert count list mismatch")
        _assert_equal(
            "expand_expert_offsets",
            expand_handle.expert_offsets,
            expand_reference["expert_offsets"],
        )
        _assert_equal(
            "expand_actual_expert_counts",
            expand_handle.actual_expert_counts,
            expand_reference["actual_expert_counts"],
        )
        _assert_equal(
            "expand_recv_to_expanded_idx",
            expand_handle.recv_to_expanded_idx,
            expand_reference["recv_to_expanded_idx"],
        )

        # Padding rows reserve aligned GMM capacity but are intentionally
        # undefined. Compare every mapped row and skip only those padding rows.
        valid_expanded_slots = expand_reference["recv_to_expanded_idx"].reshape(-1)
        valid_expanded_slots = valid_expanded_slots[valid_expanded_slots >= 0].to(torch.long)
        _assert_equal(
            "expanded_x",
            _cpu_index_select_x(expanded_x, valid_expanded_slots),
            _cpu_index_select_x(expand_reference["expanded_x"], valid_expanded_slots),
        )
        _assert_equal(
            "expanded_topk_weights",
            expanded_topk_weights.cpu().index_select(0, valid_expanded_slots),
            expand_reference["expanded_topk_weights"].index_select(0, valid_expanded_slots),
        )

        if isinstance(recv_x, tuple):
            cached_expand_input: DispatchX = (
                recv_x[0].clone(),
                recv_x[1].clone().mul_(2.0),
            )
            cached_expand_reference_input: DispatchX = (
                reference["recv_x"][0],
                reference["recv_x"][1] * 2.0,
            )
        else:
            cached_expand_input = recv_x.clone().add_(1.0)
            cached_expand_reference_input = reference["recv_x"] + 1.0
        cached_expand_reference = _expand_reference(
            cached_expand_reference_input,
            reference["recv_topk_idx"],
            reference["recv_topk_weights"],
            reference["expert_counts"],
        )
        cached_expand_kwargs = {
            "handle": expand_handle,
            "config": config,
            "async_finish": True,
        }
        if args.profile == "cached-expand":
            _profile_operator(
                rank,
                args,
                "cached-expand",
                lambda: buffer.expand(
                    cached_expand_input,
                    **cached_expand_kwargs,
                ),
            )
        (
            cached_expanded_x,
            cached_expanded_topk_weights,
            cached_expand_handle,
            cached_expand_event,
        ) = buffer.expand(cached_expand_input, **cached_expand_kwargs)
        cached_expand_event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        if cached_expanded_topk_weights is not None:
            raise AssertionError("cached expand unexpectedly returned weights")
        if cached_expand_handle is not expand_handle:
            raise AssertionError("cached expand did not reuse its handle")
        _assert_equal(
            "cached_expanded_x",
            _cpu_index_select_x(cached_expanded_x, valid_expanded_slots),
            _cpu_index_select_x(cached_expand_reference["expanded_x"], valid_expanded_slots),
        )

        expanded_values_reference = torch.arange(
            expand_handle.num_expanded_tokens,
            dtype=torch.float32,
        )
        expanded_values = expanded_values_reference.to(device="npu")
        aggregate_input_x = _dequantize_x(expanded_x, _compute_dtype(args))
        aggregate_kwargs = {
            "topk_weights": expanded_topk_weights,
            "config": config,
            "async_finish": True,
        }
        if args.profile == "aggregate":
            _profile_operator(
                rank,
                args,
                "aggregate",
                lambda: buffer.aggregate(
                    aggregate_input_x,
                    expand_handle,
                    **aggregate_kwargs,
                ),
            )
        aggregated_x, aggregated_values, aggregate_event = buffer.aggregate(
            aggregate_input_x,
            expand_handle,
            expanded_values=expanded_values,
            **aggregate_kwargs,
        )
        aggregate_event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        if aggregated_values is None:
            raise AssertionError("aggregate did not return expanded values")
        aggregate_x_reference, aggregate_values_reference = _aggregate_reference(
            _dequantize_x(expand_reference["expanded_x"], _compute_dtype(args)),
            expand_reference["recv_to_expanded_idx"],
            expand_reference["expanded_topk_weights"],
            expanded_values_reference,
        )
        _assert_equal("aggregated_x", aggregated_x, aggregate_x_reference)
        _assert_equal(
            "aggregated_values",
            aggregated_values,
            aggregate_values_reference,
        )

        bias_0 = torch.full_like(_make_x(rank, args).to(device="npu"), 0.5)
        bias_1 = (
            torch.arange(
                args.num_tokens * args.hidden,
                dtype=torch.float32,
            )
            .remainder_(2048)
            .reshape(args.num_tokens, args.hidden)
            * 0.001
        ).to(dtype=_compute_dtype(args), device="npu")
        # Owned receive outputs can feed combine directly. Borrowed SHMEM
        # inputs need a snapshot before combine reuses the same workspace.
        combine_x = _dequantize_x(recv_x, _compute_dtype(args))
        combine_topk_weights = recv_topk_weights
        if args.zero_copy:
            combine_x = combine_x.clone()
            combine_topk_weights = combine_topk_weights.clone()
        combine_kwargs = {
            "topk_weights": combine_topk_weights,
            "bias": (bias_0, bias_1),
            "config": config,
            "async_finish": True,
        }
        if args.profile == "combine":
            _profile_operator(
                rank,
                args,
                "combine",
                lambda: buffer.combine(
                    combine_x,
                    handle,
                    **combine_kwargs,
                ),
            )
        combined_x, combined_topk_weights, combine_event = buffer.combine(
            combine_x,
            handle,
            **combine_kwargs,
        )
        combine_event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        combined_x_reference, combined_weights_reference = _combine_reference(
            rank,
            args,
            bias_0.cpu(),
            bias_1.cpu(),
        )
        _assert_equal("combined_x", combined_x, combined_x_reference)
        if not args.zero_copy:
            _assert_equal("recv_x retained after combine", recv_x, reference["recv_x"])
            _assert_equal("recv_idx retained after combine", recv_topk_idx, reference["recv_topk_idx"])
            _assert_equal("recv_weights retained after combine", recv_topk_weights, reference["recv_topk_weights"])
        _assert_equal(
            "combined_topk_weights",
            combined_topk_weights,
            combined_weights_reference,
        )

        no_topk_dispatch_kwargs = {
            "num_tokens_per_rank": num_tokens_per_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "expert_alignment": args.expert_alignment,
            "config": config,
            "async_finish": True,
            "zero_copy": args.zero_copy,
        }
        if args.profile == "dispatch-no-topk":
            _profile_operator(
                rank,
                args,
                "dispatch-no-topk",
                lambda: buffer.dispatch(x, **no_topk_dispatch_kwargs),
            )
        (
            no_topk_recv_x,
            no_topk_idx,
            no_topk_weights,
            no_topk_counts,
            no_topk_handle,
            no_topk_event,
        ) = buffer.dispatch(x, **no_topk_dispatch_kwargs)
        no_topk_event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        _assert_recv_storage("no_topk_recv_x", no_topk_recv_x, buffer, args.zero_copy)
        _assert_equal("no_topk_recv_x", no_topk_recv_x, reference["recv_x"])
        if no_topk_idx is not None or no_topk_weights is not None:
            raise AssertionError("dispatch without top-k returned top-k metadata")
        _assert_equal(
            "no_topk_rank_prefix",
            no_topk_handle[0],
            reference["rank_prefix"],
        )
        _assert_equal(
            "no_topk_local_channel_prefix",
            no_topk_handle[1],
            reference["local_channel_prefix"],
        )
        _assert_equal(
            "no_topk_recv_channel_prefix",
            no_topk_handle[2],
            reference["recv_channel_prefix"],
        )
        _assert_equal(
            "no_topk_recv_src_idx",
            no_topk_handle[3],
            reference["recv_src_idx"],
        )
        _assert_equal(
            "no_topk_send_head",
            no_topk_handle[5],
            reference["send_head"],
        )
        _assert_equal(
            "no_topk_expert_counts",
            torch.tensor(no_topk_counts, dtype=torch.int32),
            reference["expert_counts"],
        )

        no_topk_combine_kwargs = {
            "config": config,
            "async_finish": True,
        }
        no_topk_combine_x = _dequantize_x(no_topk_recv_x, _compute_dtype(args))
        if args.zero_copy:
            no_topk_combine_x = no_topk_combine_x.clone()
        if args.profile == "combine-no-topk":
            _profile_operator(
                rank,
                args,
                "combine-no-topk",
                lambda: buffer.combine(
                    no_topk_combine_x,
                    no_topk_handle,
                    **no_topk_combine_kwargs,
                ),
            )
        no_topk_combined_x, no_topk_combined_weights, no_topk_combine_event = buffer.combine(
            no_topk_combine_x,
            no_topk_handle,
            **no_topk_combine_kwargs,
        )
        no_topk_combine_event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        no_topk_combined_reference, _ = _combine_reference(rank, args)
        _assert_equal(
            "no_topk_combined_x",
            no_topk_combined_x,
            no_topk_combined_reference,
        )
        if no_topk_combined_weights is not None:
            raise AssertionError("combine without top-k returned top-k weights")

        cached_reference = _dispatch_reference(rank, args, delta=1.0)
        cached_x = _to_npu(_make_dispatch_x(rank, args, delta=1.0))
        cached_dispatch_kwargs = {
            "handle": no_topk_handle,
            "config": config,
            "async_finish": True,
            "zero_copy": args.zero_copy,
        }
        if args.profile == "cached-dispatch":
            _profile_operator(
                rank,
                args,
                "cached-dispatch",
                lambda: buffer.dispatch(cached_x, **cached_dispatch_kwargs),
            )
        (
            cached_recv_x,
            cached_topk_idx,
            cached_topk_weights,
            cached_counts,
            cached_handle,
            cached_event,
        ) = buffer.dispatch(cached_x, **cached_dispatch_kwargs)
        cached_event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        _assert_recv_storage("cached_recv_x", cached_recv_x, buffer, args.zero_copy)
        _assert_equal("cached_recv_x", cached_recv_x, cached_reference["recv_x"])
        if any(
            value is not None
            for value in (
                cached_topk_idx,
                cached_topk_weights,
                cached_counts,
                cached_handle,
            )
        ):
            raise AssertionError("cached dispatch returned uncached metadata")

        cached_combine_kwargs = {
            "config": config,
            "async_finish": True,
        }
        cached_combine_x = _dequantize_x(cached_recv_x, _compute_dtype(args))
        if args.zero_copy:
            cached_combine_x = cached_combine_x.clone()
        if args.profile == "cached-combine":
            _profile_operator(
                rank,
                args,
                "cached-combine",
                lambda: buffer.combine(
                    cached_combine_x,
                    no_topk_handle,
                    **cached_combine_kwargs,
                ),
            )
        cached_combined_x, cached_combined_weights, cached_combine_event = buffer.combine(
            cached_combine_x,
            no_topk_handle,
            **cached_combine_kwargs,
        )
        cached_combine_event.current_stream_wait(release_handle=True)
        torch_npu.npu.synchronize()
        cached_combined_reference, _ = _combine_reference(
            rank,
            args,
            delta=1.0,
        )
        _assert_equal(
            "cached_combined_x",
            cached_combined_x,
            cached_combined_reference,
        )
        if cached_combined_weights is not None:
            raise AssertionError("cached combine without top-k returned top-k weights")

        if args.dtype == "fp8":
            row_reference = _dispatch_reference(rank, args, delta=2.0)
            row_x_cpu = _make_dispatch_x(rank, args, delta=2.0)
            row_x_cpu = row_x_cpu[0], row_x_cpu[1].contiguous()
            row_x = _to_npu(row_x_cpu)
            if not row_x[1].is_contiguous():
                raise AssertionError("row-major FP8 scale input is not contiguous")
            row_recv_x, _, _, _, _, row_event = buffer.dispatch(
                row_x,
                **dispatch_kwargs,
            )
            row_event.current_stream_wait(release_handle=True)
            torch_npu.npu.synchronize()
            _assert_equal(
                "row_major_fp8_recv_x",
                row_recv_x,
                row_reference["recv_x"],
            )

            int_reference = _dispatch_reference(
                rank,
                args,
                delta=3.0,
                use_int_scales=True,
            )
            int_x_cpu = _make_dispatch_x(
                rank,
                args,
                delta=3.0,
                use_int_scales=True,
            )
            int_x = _to_npu(int_x_cpu)
            int_recv_x, _, _, _, _, int_event = buffer.dispatch(
                int_x,
                **dispatch_kwargs,
            )
            int_event.current_stream_wait(release_handle=True)
            torch_npu.npu.synchronize()
            _assert_equal(
                "column_major_int32_scale_recv_x",
                int_recv_x,
                int_reference["recv_x"],
            )

        # All-invalid routes cover empty receive intervals and every padding
        # byte, without allocating a worst-case output for the large test shape.
        empty_topk = torch.full_like(topk_idx[:1], -1)
        empty_counts, _, empty_experts, empty_membership, empty_event = buffer.get_dispatch_layout(
            empty_topk, args.ep_size * args.experts_per_rank, async_finish=True
        )
        empty_event.current_stream_wait(release_handle=True)
        empty_x = tuple(t[:1].contiguous() for t in x) if isinstance(x, tuple) else x[:1]
        for capacity in (0, args.ep_size):
            padded_x, padded_idx, padded_weights, _, padded_handle, padded_event = buffer.dispatch(
                empty_x,
                topk_idx=empty_topk,
                topk_weights=topk_weights[:1],
                num_tokens_per_rank=empty_counts,
                num_tokens_per_expert=empty_experts,
                is_token_in_rank=empty_membership,
                num_worst_tokens=capacity,
                config=config,
                async_finish=True,
                zero_copy=args.zero_copy,
            )
            padded_event.current_stream_wait(release_handle=True)
            torch_npu.npu.synchronize()
            padded_parts = padded_x if isinstance(padded_x, tuple) else (padded_x,)
            for part in padded_parts:
                _assert_equal("padding payload", part, torch.zeros((capacity, part.size(1)), dtype=part.dtype))
            _assert_equal("padding indices", padded_idx, torch.full((capacity, args.num_topk), -1, dtype=topk_idx_t))
            _assert_equal(
                "padding weights", padded_weights, torch.zeros((capacity, args.num_topk), dtype=torch.float32)
            )
            _assert_recv_storage("padding payload", padded_x, buffer, args.zero_copy)
            _assert_recv_storage("padding indices", padded_idx, buffer, args.zero_copy)
            _assert_recv_storage("padding weights", padded_weights, buffer, args.zero_copy)
            cached_padding, _, _, _, _, padded_event = buffer.dispatch(
                empty_x,
                handle=padded_handle,
                config=config,
                async_finish=True,
                zero_copy=args.zero_copy,
            )
            padded_event.current_stream_wait(release_handle=True)
            torch_npu.npu.synchronize()
            cached_parts = cached_padding if isinstance(cached_padding, tuple) else (cached_padding,)
            for part in cached_parts:
                _assert_equal("cached padding payload", part, torch.zeros((capacity, part.size(1)), dtype=part.dtype))
            _assert_recv_storage("cached padding payload", cached_padding, buffer, args.zero_copy)

        if not args.zero_copy:
            _assert_equal("recv_x retained after reuse", recv_x, reference["recv_x"])
            _assert_equal("recv_idx retained after reuse", recv_topk_idx, reference["recv_topk_idx"])
            _assert_equal("recv_weights retained after reuse", recv_topk_weights, reference["recv_topk_weights"])
            _assert_equal("cached_recv_x retained after combine", cached_recv_x, cached_reference["recv_x"])
        buffer.barrier()
        if not args.zero_copy:
            buffer.destroy()
            _assert_equal("recv_x retained after destroy", recv_x, reference["recv_x"])
            _assert_equal("recv_idx retained after destroy", recv_topk_idx, reference["recv_topk_idx"])
            _assert_equal("recv_weights retained after destroy", recv_topk_weights, reference["recv_topk_weights"])
            _assert_equal("cached_recv_x retained after destroy", cached_recv_x, cached_reference["recv_x"])
        print(
            f"rank {rank}/{args.ep_size} V1 intranode precision passed: "
            f"recv_tokens={_x_tensor(recv_x).size(0)}, "
            f"dtype={args.dtype}, channels={args.num_aiv}, "
            f"expanded_tokens={expand_handle.num_expanded_tokens}, "
            f"config={config_source}, transport={args.transport}, zero_copy={args.zero_copy}, "
            f"allocate_on_comm_stream={args.allocate_on_comm_stream}, "
            f"stream_lifetime={args.stream_lifetime}"
        )
    except _ProfileComplete:
        buffer.barrier()
    finally:
        if buffer is not None and buffer.is_available():
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    add_route_arguments(parser)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--allocate-on-comm-stream", action="store_true")
    parser.add_argument(
        "--stream-lifetime",
        action="store_true",
        help="release native inputs/outputs early under cross-stream allocator pressure",
    )
    parser.add_argument("--zero-copy", action="store_true", help="return dispatch payloads as borrowed SHMEM views")
    parser.add_argument("--first-device", type=int, default=0)
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="zero-based node index; requires --nproc-per-node",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="enable multi-node mode and spawn this many local NPU processes",
    )
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help="HCCL rendezvous host shared by every node",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "29500")),
        help="HCCL rendezvous port shared by every node",
    )
    parser.add_argument(
        "--dist-init-method",
        default=None,
        help="legacy HCCL tcp://HOST:PORT override",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT"),
        help=("CANN SHMEM endpoint; defaults to tcp://MASTER_ADDR:(MASTER_PORT+1)"),
    )
    parser.add_argument(
        "--num-bytes",
        type=int,
        default=None,
        help=("symmetric workspace bytes; omit to calculate from the V1 size hint using --num-tokens"),
    )
    parser.add_argument(
        "--transport",
        choices=("auto", "mte", "roce", "sdma", "udma"),
        default="auto",
    )
    parser.add_argument("--num-tokens", type=int, default=67)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp8"),
        default="fp16",
        help=(
            "dispatch payload dtype; FP8 uses E4M3 data with per-128-value "
            "float32 scales and BF16 for aggregate/combine"
        ),
    )
    parser.add_argument("--experts-per-rank", type=int, default=7)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument("--udma-schedule", choices=("peer", "token"), default="peer")
    parser.add_argument("--udma-peers-per-worker", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--udma-wqes-per-visit", type=int, choices=(4, 8, 16, 32), default=8)
    parser.add_argument(
        "--num-aiv",
        type=int,
        default=None,
        help="explicit AIV count; omit to use Buffer recommended configs",
    )
    parser.add_argument("--expert-alignment", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--profile",
        choices=PROFILE_OPERATORS,
        default=None,
        metavar="OP",
        help="profile one API operator while running the precision flow",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=5,
        help="unrecorded warmup calls for the selected operator",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=10,
        help="selected operator calls in each profiler step",
    )
    parser.add_argument(
        "--profile-dir",
        default="./profile",
        help="torch_npu profiler output directory",
    )
    parser.add_argument(
        "--inject-error",
        action="store_true",
        help="corrupt one recv_x reference value to verify the guard",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    validate_routes(args)
    if args.stream_lifetime and args.profile is not None:
        raise ValueError("--stream-lifetime cannot be combined with profiling")
    if args.ep_size < 2:
        raise ValueError("--ep-size must be at least 2")
    if args.num_tokens <= 0 or args.hidden <= 0:
        raise ValueError("--num-tokens and --hidden must be positive")
    if args.experts_per_rank <= 0:
        raise ValueError("--experts-per-rank must be positive")
    num_experts = args.ep_size * args.experts_per_rank
    if args.num_topk < 1 or args.num_topk > num_experts:
        raise ValueError("--num-topk must be in [1, num_experts]")
    if args.num_aiv is not None and (args.num_aiv <= 0 or args.num_aiv > _MAX_NUM_AIV or args.num_aiv % 2):
        raise ValueError(f"--num-aiv must be an even integer in [2, {_MAX_NUM_AIV}]")
    if args.expert_alignment <= 0:
        raise ValueError("--expert-alignment must be positive")
    if args.profile_warmup < 0:
        raise ValueError("--profile-warmup must be non-negative")
    if args.profile_iters <= 0:
        raise ValueError("--profile-iters must be positive")
    if args.profile is not None and not args.profile_dir:
        raise ValueError("--profile-dir must not be empty")
    if args.dist_init_method:
        prefix = "tcp://"
        if not args.dist_init_method.startswith(prefix):
            raise ValueError("--dist-init-method must use tcp://HOST:PORT")
        host_port = args.dist_init_method[len(prefix) :]
        host, separator, port = host_port.rpartition(":")
        if not separator or not host or not port.isdigit():
            raise ValueError("--dist-init-method must use tcp://HOST:PORT")
        args.master_addr = host
        args.master_port = int(port)
    validate_launch_args(args)
    args.shmem_ip_port = effective_shmem_ip_port(args)
    if args.num_bytes is None:
        size_config = (
            Buffer.get_dispatch_config(args.ep_size)
            if args.num_aiv is None
            else Config(
                args.num_aiv,
                6,
                256,
                6,
                128,
                udma_peers_per_worker=args.udma_peers_per_worker,
                udma_wqes_per_visit=args.udma_wqes_per_visit,
            )
        )
        # FP8 dispatch uses one byte per hidden element, but this end-to-end
        # test dequantizes to BF16 before combine. Size the shared workspace
        # for the larger phase because dispatch and combine reuse one Buffer.
        hidden_bytes = args.hidden * 2
        args.num_bytes = size_config.get_nvl_buffer_size_hint(
            hidden_bytes,
            args.ep_size,
            num_tokens=args.num_tokens,
        )
    if args.num_bytes % DEVICE_LARGE_PAGE_SIZE:
        raise ValueError(f"--num-bytes must be aligned to {DEVICE_LARGE_PAGE_SIZE}")
    local_count = local_process_count(args)
    first_global_rank = global_rank_for_local(0, args)
    last_global_rank = global_rank_for_local(local_count - 1, args)
    print(
        f"Launching V1 intranode test with ep_size={args.ep_size}, "
        f"node_rank={args.node_rank}, local_processes={local_count}, "
        f"global_ranks={first_global_rank}-{last_global_rank}, "
        f"first_device={args.first_device}, "
        f"master={args.master_addr}:{args.master_port}, "
        f"shmem_ip_port={args.shmem_ip_port}",
        flush=True,
    )
    if local_count == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=local_count, join=True)


if __name__ == "__main__":
    main()


# python test/test_v1_intranode.py --ep-size 4 \
#   --shmem-ip-port tcp://127.0.0.1:26667
# python test/test_v1_intranode.py --ep-size 4 --num-topk 1
# python test/test_v1_intranode.py --ep-size 4 --dtype fp8
# python test/test_v1_intranode.py --ep-size 4 --num-aiv 48
# python test/test_v1_intranode.py --ep-size 4 --transport udma
# python test/test_v1_intranode.py --ep-size 4 --inject-error
# python test/test_v1_intranode.py --ep-size 4 --profile expand
# python test/test_v1_intranode.py --ep-size 4 --profile cached-expand
# python test/test_v1_intranode.py --ep-size 4 --profile aggregate
# Multi-node example (run once per node with node-rank 0..7):
# python test/test_v1_intranode.py --ep-size 64 --nproc-per-node 8 \
#   --node-rank 0 --master-addr 10.0.0.1 --master-port 29500 \
#   --transport udma
