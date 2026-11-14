# Copyright (c) 2026, Lu Lu
"""Tensor/handle adapter for the isolated UBX route-plan kernel."""

from ..._native import npu_api, require_native
from ..._runtime import runtime_of
from ...utils.event import EventHandle, EventOverlap


def combine(
    buffer,
    x,
    handle,
    topk_weights,
    bias,
    num_compute_units,
    num_qps,
    previous_event,
    previous_event_before_epilogue,
    async_with_compute_stream,
    allocate_on_comm_stream,
):
    import torch

    runtime = runtime_of(buffer)
    runtime.ensure_alive()
    if handle._buffer_token is not buffer._handle_token or handle.metadata_only:
        raise ValueError("UBX Combine requires this buffer's successful LL Dispatch handle")
    if not handle.do_expand or handle.expert_alignment != 1:
        raise NotImplementedError("UBX Combine requires expanded, unpadded LL rows")
    if topk_weights is not None:
        raise NotImplementedError("LL Combine applies Dispatch weights from the handle; pass topk_weights=None")
    if type(async_with_compute_stream) is not bool or type(allocate_on_comm_stream) is not bool:
        raise TypeError("Stream options must be bool")
    if previous_event is not None and not allocate_on_comm_stream:
        raise ValueError("previous_event requires allocate_on_comm_stream=True")
    for dependency in (previous_event, previous_event_before_epilogue):
        if dependency is not None and (not isinstance(dependency, EventHandle) or dependency._device != runtime.device):
            raise ValueError("Combine events must belong to the buffer NPU")
    b, k, h, e = handle.num_tokens, handle.num_topk, handle.hidden, handle.num_experts
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype not in (torch.bfloat16, torch.float16)
        or tuple(x.shape) != (handle.num_expanded_tokens, h)
        or not x.is_contiguous()
        or x.requires_grad
        or x.device.type != "npu"
        or x.device.index != runtime.device
    ):
        raise ValueError("UBX Combine requires contiguous BF16/FP16 expert output on the buffer NPU")
    biases = () if bias is None else bias if isinstance(bias, tuple) else (bias,)
    if len(biases) > 2 or (isinstance(bias, tuple) and len(biases) != 2):
        raise ValueError("bias must be a Tensor or a pair of Tensors")
    for value in biases:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (b, h)
            or value.dtype != x.dtype
            or value.device != x.device
            or not value.is_contiguous()
            or value.requires_grad
        ):
            raise ValueError("Combine bias must match the output shape, dtype and device")

    native, npu = require_native("ElasticBuffer.combine"), npu_api()
    cores = native.low_latency_core_config()["active_aiv"]
    if type(num_compute_units) is not int or num_compute_units not in (0, cores):
        raise ValueError("UBX Combine uses the active AIV count pinned by Dispatch")
    if type(num_qps) is not int or num_qps not in (0, cores):
        raise ValueError("UBX Combine uses one QP per active physical AIV")
    dtype = str(x.dtype).removeprefix("torch.")
    key = (runtime.world_size, b, h, k, e, dtype, handle.num_max_tokens_per_rank, cores)
    with runtime.notify_lock:
        if runtime.notify_failed:
            raise RuntimeError("A previous communication launch failed; restart the distributed job")
        producer = previous_event if previous_event is not None else EventHandle()
        output = None if allocate_on_comm_stream else torch.empty((b, h), dtype=x.dtype, device=x.device)
        try:
            with npu.stream(runtime.stream):
                producer.current_stream_wait()
                # This kernel fuses reduction and output. An epilogue dependency
                # therefore precedes the whole kernel, rather than a second launch.
                if previous_event_before_epilogue is not None:
                    previous_event_before_epilogue.current_stream_wait()
                cached = getattr(handle, "_ubx_combine_tiling", None)
                if cached is None or cached[0] != key:
                    tiling = native.low_latency_combine_tiling(*key[:-1], rdma=True)
                    cached = (key, buffer._upload_low_latency(tiling, x.device))
                    handle._ubx_combine_tiling = cached
                if output is None:
                    output = torch.empty((b, h), dtype=x.dtype, device=x.device)
                tensors = {
                    "input": x,
                    "output": output,
                    "destination_index": handle.destination_index,
                    "send_counts": handle.send_counts,
                    "source_weights": handle.source_weights,
                    "token_type": handle.token_type,
                    "relay_read_index": handle.relay_read_index,
                    "source_mask": handle.source_mask,
                    "comm_args": runtime.low_latency_comm_args,
                    "tiling": cached[1],
                }
                retained = (*tensors.values(), *biases, handle)
                runtime.notify_inflight = retained
                for value in (*tensors.values(), *biases):
                    if not isinstance(value, torch.Tensor):
                        raise ValueError("LL handle is missing a required tensor")
                    value.record_stream(runtime.stream)
                capacity = handle.num_max_tokens_per_rank
                native.combine_low_latency(
                    runtime.native,
                    tensors,
                    b,
                    h,
                    k,
                    e,
                    dtype,
                    runtime.stream.npu_stream,
                    capacity=capacity,
                )
                for value in biases:
                    output.add_(value)
                event = EventOverlap._from_resources(EventHandle(), retained)
            if not async_with_compute_stream:
                event.current_stream_wait(release_handle=True)
            runtime.notify_inflight = None
            return output, None, event
        except Exception:
            runtime.notify_failed = True
            raise
