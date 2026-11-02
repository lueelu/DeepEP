# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark the public Buffer layout, dispatch and combine operators."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep._clos_adapter import Buffer
from deep_ep import Config, topk_idx_t
from bench_utils import (
    L2_CACHE_FLUSH_BYTES,
    benchmark_operator,
    dispatch_layout_reference,
    make_topk_idx,
    write_json,
)


DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024
SUPPORTED_OPS = (
    "all",
    "layout",
    "dispatch",
    "cached-dispatch",
    "expand",
    "cached-expand",
    "aggregate",
    "combine",
    "full-dispatch",
    "full-combine",
    "e2e",
)


def debug_progress(rank: int, phase: str) -> None:
    """Emit opt-in per-rank progress while diagnosing collective stalls."""

    if int(os.environ.get("ASCEND_DEEPEP_BENCH_DEBUG", "0")):
        print(f"[rank {rank}] {phase}", flush=True)


def wait_event(event: Any) -> None:
    if event is not None and getattr(event, "event", None) is not None:
        event.current_stream_wait(release_handle=True)


def materialize_combine_inputs(
    buffer: Buffer,
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    topk_weights: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Move SHMEM-backed dispatch outputs into combine-owned storage."""

    with torch.npu.stream(buffer.get_comm_stream()):
        materialized_x = per_token_cast_back(x, dtype).clone()
        materialized_topk_weights = topk_weights.clone() if topk_weights is not None else None
    return materialized_x, materialized_topk_weights


def per_token_cast_to_fp8(
    x: torch.Tensor,
    *,
    column_major_scales: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize groups of 128 values using DeepEP's E4M3 convention."""

    hidden = x.size(1)
    aligned_hidden = (hidden + 127) // 128 * 128
    x_padded = torch.nn.functional.pad(x, (0, aligned_hidden - hidden))
    x_view = x_padded.view(x.size(0), -1, 128)
    amax = x_view.abs().float().amax(dim=2).clamp(1e-4)
    quantized = (x_view * (448.0 / amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    scales = (amax / 448.0).contiguous()
    if column_major_scales:
        scales = scales.T.contiguous().T
    return quantized.view(x.size(0), aligned_hidden)[:, :hidden].contiguous(), scales


def per_token_cast_back(
    value: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a dispatch FP8 tuple for correctness and combine checks."""

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


def config_dict(config: Config, label: Union[str, int]) -> Dict[str, Any]:
    return {
        "label": label,
        "num_sms": config.num_sms,
        "nvl_send_tokens": config.num_max_nvl_chunked_send_tokens,
        "nvl_recv_tokens": config.num_max_nvl_chunked_recv_tokens,
    }


def config_with_nvl_send(base: Config, send_tokens: int) -> Config:
    return Config(
        base.num_aiv,
        send_tokens,
        base.num_max_nvl_chunked_recv_tokens,
        base.num_max_rdma_chunked_send_tokens,
        base.num_max_rdma_chunked_recv_tokens,
    )


def dispatch_config_candidates(base: Config):
    return [(chunk, config_with_nvl_send(base, chunk)) for chunk in range(4, 33, 2)] + [("default", base)]


def combine_config_candidates(base: Config):
    return [(chunk, config_with_nvl_send(base, chunk)) for chunk in range(1, 17)] + [("default", base)]


def tune_operator(
    name: str,
    candidates: Sequence[Tuple[Union[str, int], Config]],
    fn_factory,
    *,
    dtype_name: str,
    logical_bytes_per_rank: int,
    buffer: Buffer,
    timing_group: Any,
    rank: int,
    args: argparse.Namespace,
):
    """Scan all candidates and select rank 0's lowest mean latency."""

    candidate_results = []
    for label, config in candidates:
        result = benchmark_operator(
            name,
            fn_factory(config),
            buffer=buffer,
            timing_group=timing_group,
            num_warmups=args.warmup,
            num_iterations=args.iterations,
            num_tokens_per_rank=args.num_tokens,
            logical_bytes_per_rank=logical_bytes_per_rank,
            flush_l2=args.flush_l2_cache,
            max_rank_latency=args.max_rank_latency,
        )
        result["dtype"] = dtype_name
        result["tuned"] = True
        result["tuning_config"] = config_dict(config, label)
        candidate_results.append(result)

    # Keep DeepEP's rank-0 selection rule and propagate it through the global
    # HCCL group. The reference reductions run before Buffer construction, so
    # this communicator has already completed its lazy device initialization.
    buffer.barrier()
    best_index = (
        min(
            range(len(candidate_results)),
            key=lambda index: candidate_results[index]["mean_us"],
        )
        if rank == 0
        else -1
    )
    selected_index = torch.tensor(
        [best_index],
        dtype=torch.int32,
        device=f"npu:{torch.npu.current_device()}",
    )
    gathered_indices = [torch.empty_like(selected_index) for _ in range(buffer.group_size)]
    dist.all_gather(gathered_indices, selected_index, group=timing_group)
    torch.npu.synchronize()
    best_index = int(gathered_indices[0].item())
    if best_index < 0 or best_index >= len(candidate_results):
        raise RuntimeError(f"rank 0 selected invalid tuning index {best_index}")
    buffer.barrier()
    return (
        candidates[best_index][1],
        candidate_results[best_index],
        candidate_results,
    )


def print_deepep_report(report: Dict[str, Any]) -> None:
    """Print rank-0 results in DeepEP's familiar benchmark style."""

    config = report["config"]
    print(
        f"[config] num_tokens={config['num_tokens']}, hidden={config['hidden']}, num_topk={config['num_topk']}",
        flush=True,
    )
    for result in report["results"]:
        op = result["op"]
        mean_us = result["mean_us"]
        bandwidth = result.get(
            "algorithm_bandwidth_gbps",
            result.get("per_rank_logical_payload_gbps"),
        )
        dtype_name = str(result.get("dtype", config["dtype"])).upper()
        if op == "layout":
            print(
                f"[layout] Kernel performance: {mean_us / 1000:.3f} ms",
                flush=True,
            )
        elif op == "cached-dispatch":
            prefix = "[tuning] Best" if result.get("tuned") else "[benchmark]"
            print(
                f"{prefix} dispatch ({dtype_name}): {bandwidth:.2f} GB/s (NVL), t: {mean_us:.2f} us",
                flush=True,
            )
        elif op == "combine":
            prefix = "[tuning] Best" if result.get("tuned") else "[benchmark]"
            print(
                f"{prefix} combine: {bandwidth:.2f} GB/s (NVL), t: {mean_us:.2f} us",
                flush=True,
            )
        elif op == "dispatch":
            print(
                f"[benchmark] First dispatch ({dtype_name}): {bandwidth:.2f} GB/s (NVL), t: {mean_us:.2f} us",
                flush=True,
            )
        elif op in ("expand", "cached-expand", "aggregate"):
            print(
                f"[benchmark] {op}: {mean_us:.2f} us",
                flush=True,
            )
        elif op in ("full-dispatch", "full-combine"):
            print(
                f"[benchmark] {op}: {bandwidth:.2f} GB/s, t: {mean_us:.2f} us",
                flush=True,
            )
        elif op == "dispatch-combine":
            print(
                f"[benchmark] Dispatch + combine: {bandwidth:.2f} GB/s, t: {mean_us:.2f} us",
                flush=True,
            )


def get_layout(buffer: Buffer, topk_idx: torch.Tensor, num_experts: int):
    outputs = buffer.get_dispatch_layout(
        topk_idx,
        num_experts,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    return outputs


def run_correctness_matrix(
    *,
    buffer: Buffer,
    x: torch.Tensor,
    x_pure_rand: torch.Tensor,
    x_fp8: Tuple[torch.Tensor, torch.Tensor] | None,
    x_pure_rand_fp8: Tuple[torch.Tensor, torch.Tensor] | None,
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_weights_pure_rand: torch.Tensor,
    num_recv_tokens: int,
    expected_local_expert_counts: List[int],
    dispatch_config: Config,
    combine_config: Config,
    rank: int,
    num_ranks: int,
    args: argparse.Namespace,
) -> None:
    """Exercise DeepEP's sync/async, event, top-k, cached and FP8 matrix."""

    experts_per_rank = args.num_experts // num_ranks
    factor = is_token_in_rank.sum(dim=1, keepdim=True).to(x.dtype)
    if args.dtype == "fp8":
        if x_fp8 is None or x_pure_rand_fp8 is None:
            raise RuntimeError("FP8 correctness inputs were not initialized")
        inputs: List[Tuple[str, Any, torch.Tensor]] = [
            ("FP8-row-major", x_pure_rand_fp8, topk_weights_pure_rand),
            ("FP8-column-major", x_fp8, topk_weights),
        ]
    else:
        inputs = [
            (f"{args.dtype.upper()}-random", x_pure_rand, topk_weights_pure_rand),
            (args.dtype.upper(), x, topk_weights),
        ]
    if args.dtype != "fp8" and x_fp8 is not None:
        inputs.append(("FP8", x_fp8, topk_weights))

    num_cases = 0
    for previous_mode in (False, True):
        for async_mode in (False, True):
            for dtype_name, current_x, current_topk_weights in inputs:
                for with_topk in (False, True):
                    dispatch_args = {
                        "x": current_x,
                        "num_tokens_per_rank": num_tokens_per_rank,
                        "is_token_in_rank": is_token_in_rank,
                        "num_tokens_per_expert": num_tokens_per_expert,
                        "config": dispatch_config,
                        "async_finish": async_mode,
                        "allocate_on_comm_stream": async_mode,
                    }
                    if with_topk:
                        dispatch_args.update(
                            {
                                "topk_idx": topk_idx,
                                "topk_weights": current_topk_weights,
                            }
                        )
                    if previous_mode:
                        dispatch_args["previous_event"] = buffer.capture()

                    dispatched = buffer.dispatch(**dispatch_args)
                    wait_event(dispatched[-1])
                    torch.npu.synchronize()
                    recv_raw, recv_topk_idx, recv_topk_weights = dispatched[:3]
                    recv_counts, handle = dispatched[3:5]
                    # Dispatch payloads are non-owning views into the shared
                    # symmetric workspace. Preserve the first-dispatch result
                    # before padded/cached dispatch or combine reuses it.
                    recv_x = per_token_cast_back(recv_raw, x.dtype).clone()
                    recv_topk_idx = recv_topk_idx.clone() if recv_topk_idx is not None else None
                    recv_topk_weights = recv_topk_weights.clone() if recv_topk_weights is not None else None
                    torch.npu.synchronize()
                    if tuple(recv_x.shape) != (num_recv_tokens, args.hidden):
                        raise AssertionError(f"{dtype_name} dispatch returned shape {tuple(recv_x.shape)}")
                    if recv_counts != expected_local_expert_counts:
                        raise AssertionError("dispatch returned unexpected per-expert receive counts")

                    if with_topk:
                        valid = recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < experts_per_rank))
                        if not bool(valid.all().item()):
                            raise AssertionError("dispatch returned invalid local expert IDs")
                        for expert, count in enumerate(recv_counts):
                            if int(recv_topk_idx.eq(expert).sum().item()) != count:
                                raise AssertionError("dispatch top-k IDs disagree with expert counts")

                        padded_args = dict(dispatch_args)
                        padded_args["num_worst_tokens"] = args.num_tokens * num_ranks
                        padded = buffer.dispatch(**padded_args)
                        wait_event(padded[-1])
                        torch.npu.synchronize()
                        padded_x = per_token_cast_back(padded[0], x.dtype)
                        if padded_x.size(0) != args.num_tokens * num_ranks:
                            raise AssertionError("num_worst_tokens was not preserved")
                        torch.testing.assert_close(
                            padded_x[:num_recv_tokens],
                            recv_x,
                            rtol=args.rtol,
                            atol=args.atol,
                        )
                        torch.testing.assert_close(padded[1][:num_recv_tokens], recv_topk_idx)
                        torch.testing.assert_close(padded[2][:num_recv_tokens], recv_topk_weights)
                        if not bool(padded[1][num_recv_tokens:].eq(-1).all().item()):
                            raise AssertionError("padded dispatch top-k IDs must use -1 sentinels")
                    else:
                        if recv_topk_idx is not None or recv_topk_weights is not None:
                            raise AssertionError("dispatch without top-k returned top-k metadata")
                        cached_args = {
                            "x": current_x,
                            "handle": handle,
                            "config": dispatch_config,
                            "async_finish": async_mode,
                            "allocate_on_comm_stream": async_mode,
                        }
                        if previous_mode:
                            cached_args["previous_event"] = buffer.capture()
                        cached = buffer.dispatch(**cached_args)
                        wait_event(cached[-1])
                        torch.npu.synchronize()
                        torch.testing.assert_close(
                            per_token_cast_back(cached[0], x.dtype),
                            recv_x,
                            rtol=args.rtol,
                            atol=args.atol,
                        )

                    combine_args = {
                        "x": recv_x,
                        "handle": handle,
                        "topk_weights": recv_topk_weights if with_topk else None,
                        "config": combine_config,
                        "async_finish": async_mode,
                        "allocate_on_comm_stream": async_mode,
                    }
                    if previous_mode:
                        combine_args["previous_event"] = buffer.capture()
                    combined_x, combined_topk_weights, combined_event = buffer.combine(**combine_args)
                    wait_event(combined_event)
                    torch.npu.synchronize()
                    source_x = per_token_cast_back(current_x, x.dtype)
                    is_fp8 = dtype_name.startswith("FP8")
                    torch.testing.assert_close(
                        combined_x,
                        source_x * factor,
                        rtol=max(args.rtol, 5e-2 if is_fp8 else 0),
                        atol=max(args.atol, 5e-2 if is_fp8 else 0),
                    )
                    if with_topk:
                        torch.testing.assert_close(
                            combined_topk_weights,
                            current_topk_weights,
                            rtol=args.rtol,
                            atol=args.atol,
                        )
                    elif combined_topk_weights is not None:
                        raise AssertionError("combine without top-k returned top-k metadata")
                    num_cases += 1

    if rank == 0:
        print(
            f"[testing] DeepEP correctness matrix: {num_cases} cases passed",
            flush=True,
        )


def validate_launch_args(args: argparse.Namespace) -> None:
    if not args.master_addr:
        raise ValueError("--master-addr must not be empty")
    if args.master_port <= 0 or args.master_port > 65535:
        raise ValueError("--master-port must be in [1, 65535]")
    if args.ep_size < 2:
        raise ValueError("--ep-size must be at least 2")
    if args.num_local_ranks <= 0:
        raise ValueError("--nproc-per-node must be greater than 0")
    if args.ep_size % args.num_local_ranks:
        raise ValueError("--ep-size must be divisible by --nproc-per-node")
    num_nodes = args.ep_size // args.num_local_ranks
    if args.node_rank < 0 or args.node_rank >= num_nodes:
        raise ValueError(f"--node-rank must be in [0, {num_nodes})")


def effective_shmem_ip_port(args: argparse.Namespace) -> str:
    if args.shmem_ip_port:
        return args.shmem_ip_port
    shmem_port = args.master_port + 1
    if shmem_port > 65535:
        raise ValueError("--shmem-ip-port is required when --master-port is 65535")
    return f"tcp://{args.master_addr}:{shmem_port}"


def run_rank(local_rank: int, args: argparse.Namespace) -> None:
    rank = args.node_rank * args.num_local_ranks + local_rank
    device_id = args.first_device + local_rank
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(device_id)
    os.environ["WORLD_SIZE"] = str(args.ep_size)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.npu.set_device(device_id)
    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.ep_size,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("npu")
    timing_group = dist.new_group(list(range(args.ep_size)))
    dist.barrier(group=timing_group, device_ids=[device_id])
    num_ranks = args.ep_size
    debug_progress(rank, "init_dist completed")

    buffer = None
    try:
        if args.num_sms is not None:
            Buffer.set_num_sms(args.num_sms)
        default_dispatch_config = Buffer.get_dispatch_config(num_ranks)
        default_combine_config = Buffer.get_combine_config(num_ranks)
        torch.manual_seed(args.seed + rank)
        dtype = torch.bfloat16 if args.dtype in ("bf16", "fp8") else torch.float16
        x = torch.ones((args.num_tokens, args.hidden), device="npu", dtype=dtype) * rank
        x_pure_rand = torch.randn((args.num_tokens, args.hidden), device="npu", dtype=dtype)
        x_fp8 = None
        x_pure_rand_fp8 = None
        topk_idx_cpu = make_topk_idx(
            rank,
            num_ranks,
            args.num_tokens,
            args.num_topk,
            args.num_experts,
            args.route,
            args.seed,
        )
        topk_idx = topk_idx_cpu.to(device=f"npu:{device_id}", dtype=topk_idx_t)
        topk_weights = (
            torch.ones(
                (args.num_tokens, args.num_topk),
                dtype=torch.float32,
                device=f"npu:{device_id}",
            )
            * rank
        )
        topk_weights_pure_rand = torch.randn(
            (args.num_tokens, args.num_topk),
            dtype=torch.float32,
            device=f"npu:{device_id}",
        )
        selected_ops = (
            (
                "layout",
                "dispatch",
                "cached-dispatch",
                "expand",
                "cached-expand",
                "aggregate",
                "combine",
                "full-dispatch",
                "full-combine",
                "e2e",
            )
            if args.op == "all"
            else (args.op,)
        )
        needs_dispatch_setup = any(op != "layout" for op in selected_ops)
        if needs_dispatch_setup and (args.dtype == "fp8" or not args.skip_fp8):
            x_fp8 = per_token_cast_to_fp8(x)
            x_pure_rand_fp8 = per_token_cast_to_fp8(
                x_pure_rand,
                column_major_scales=False,
            )
        dispatch_x = x_fp8 if args.dtype == "fp8" else x
        if needs_dispatch_setup and dispatch_x is None:
            raise RuntimeError("FP8 dispatch input was not initialized")

        # DeepEP computes these values with a reference collective. Running it
        # before Buffer construction also completes HCCL's lazy communicator
        # initialization before CANN SHMEM establishes its symmetric VMM.
        expected_layout = dispatch_layout_reference(topk_idx_cpu, num_ranks, args.num_experts)
        global_num_tokens_per_rank = expected_layout[0].to(device=f"npu:{device_id}")
        global_num_tokens_per_expert = expected_layout[1].to(device=f"npu:{device_id}")
        debug_progress(rank, "reference all_reduce start")
        dist.all_reduce(
            global_num_tokens_per_rank,
            op=dist.ReduceOp.SUM,
            group=timing_group,
        )
        dist.all_reduce(
            global_num_tokens_per_expert,
            op=dist.ReduceOp.SUM,
            group=timing_group,
        )
        torch.npu.synchronize()
        debug_progress(rank, "reference all_reduce completed")
        num_recv_tokens = int(global_num_tokens_per_rank[rank].item())
        experts_per_rank = args.num_experts // num_ranks
        expected_local_expert_counts = global_num_tokens_per_expert.view(num_ranks, experts_per_rank)[rank].tolist()

        num_bytes = args.num_bytes
        buffer = Buffer(
            timing_group,
            num_nvl_bytes=num_bytes,
            explicitly_destroy=True,
            transport=args.transport,
            shmem_ip_port=args.shmem_ip_port,
        )
        debug_progress(rank, "Buffer initialized")
        debug_progress(rank, "layout start")
        layout = get_layout(buffer, topk_idx, args.num_experts)
        wait_event(layout[-1])
        torch.npu.synchronize()
        debug_progress(rank, "layout completed")
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = layout
        payload_bytes = num_recv_tokens * args.hidden * x.element_size()
        fp8_payload_bytes = num_recv_tokens * (args.hidden + ((args.hidden + 127) // 128) * 4)
        dispatch_payload_bytes = fp8_payload_bytes if args.dtype == "fp8" else payload_bytes
        full_algorithm_bytes = args.num_tokens * args.hidden * 2 * args.num_topk

        if not args.skip_correctness_check:
            torch.testing.assert_close(num_tokens_per_rank.cpu(), expected_layout[0])
            torch.testing.assert_close(num_tokens_per_expert.cpu(), expected_layout[1])
            torch.testing.assert_close(is_token_in_rank.cpu(), expected_layout[2])
            if needs_dispatch_setup:
                run_correctness_matrix(
                    buffer=buffer,
                    x=x,
                    x_pure_rand=x_pure_rand,
                    x_fp8=x_fp8,
                    x_pure_rand_fp8=x_pure_rand_fp8,
                    num_tokens_per_rank=num_tokens_per_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    is_token_in_rank=is_token_in_rank,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    topk_weights_pure_rand=topk_weights_pure_rand,
                    num_recv_tokens=num_recv_tokens,
                    expected_local_expert_counts=expected_local_expert_counts,
                    dispatch_config=default_dispatch_config,
                    combine_config=default_combine_config,
                    rank=rank,
                    num_ranks=num_ranks,
                    args=args,
                )

        results: List[Dict[str, Any]] = []
        tuning_results: Dict[str, Any] = {}

        if "layout" in selected_ops:
            results.append(
                benchmark_operator(
                    "layout",
                    lambda: get_layout(buffer, topk_idx, args.num_experts),
                    buffer=buffer,
                    timing_group=timing_group,
                    num_warmups=args.warmup,
                    num_iterations=args.iterations,
                    num_tokens_per_rank=args.num_tokens,
                    flush_l2=args.flush_l2_cache,
                    max_rank_latency=args.max_rank_latency,
                )
            )

        if "dispatch" in selected_ops:
            dispatch_result = benchmark_operator(
                "dispatch",
                lambda: buffer.dispatch(
                    dispatch_x,
                    num_tokens_per_rank=num_tokens_per_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    num_worst_tokens=num_recv_tokens,
                    config=default_dispatch_config,
                    async_finish=True,
                    allocate_on_comm_stream=True,
                ),
                buffer=buffer,
                timing_group=timing_group,
                num_warmups=args.warmup,
                num_iterations=args.iterations,
                num_tokens_per_rank=args.num_tokens,
                logical_bytes_per_rank=dispatch_payload_bytes,
                flush_l2=args.flush_l2_cache,
                max_rank_latency=args.max_rank_latency,
            )
            dispatch_result["dtype"] = args.dtype
            dispatch_result["tuned"] = False
            results.append(dispatch_result)

        recv_x = None
        recv_topk_idx = None
        recv_topk_weights = None
        recv_expert_counts = None
        handle = None
        needs_expand_setup = any(
            op
            in (
                "expand",
                "cached-expand",
                "aggregate",
                "full-dispatch",
                "full-combine",
            )
            for op in selected_ops
        )
        if needs_dispatch_setup:
            prepared = buffer.dispatch(
                dispatch_x,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx if needs_expand_setup else None,
                topk_weights=topk_weights if needs_expand_setup else None,
                config=default_dispatch_config,
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            wait_event(prepared[-1])
            torch.npu.synchronize()
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_expert_counts,
                handle,
            ) = prepared[:5]

        selected_dispatch_config = default_dispatch_config
        dispatch_inputs: List[Tuple[str, Any, int]] = []
        if x_fp8 is not None:
            dispatch_inputs.append(("FP8", x_fp8, fp8_payload_bytes))
        if args.dtype != "fp8":
            dispatch_inputs.append((args.dtype.upper(), x, payload_bytes))
        should_scan_dispatch = args.tune and any(op in ("cached-dispatch", "combine", "e2e") for op in selected_ops)
        if should_scan_dispatch:
            for input_index, (dtype_name, current_x, current_bytes) in enumerate(dispatch_inputs):
                best_config, best_result, candidates = tune_operator(
                    "cached-dispatch",
                    dispatch_config_candidates(default_dispatch_config),
                    lambda config, current_x=current_x: lambda: buffer.dispatch(
                        current_x,
                        handle=handle,
                        num_worst_tokens=num_recv_tokens,
                        config=config,
                        async_finish=True,
                        allocate_on_comm_stream=True,
                    ),
                    dtype_name=dtype_name,
                    logical_bytes_per_rank=current_bytes,
                    buffer=buffer,
                    timing_group=timing_group,
                    rank=rank,
                    args=args,
                )
                tuning_results[f"dispatch_{dtype_name.lower()}"] = candidates
                if "cached-dispatch" in selected_ops:
                    results.append(best_result)
                if input_index == 0:
                    # DeepEP uses rank 0's first (FP8 when available) best
                    # dispatch configuration to prepare combine inputs.
                    selected_dispatch_config = best_config
        elif "cached-dispatch" in selected_ops:
            for dtype_name, current_x, current_bytes in dispatch_inputs:
                result = benchmark_operator(
                    "cached-dispatch",
                    lambda current_x=current_x: buffer.dispatch(
                        current_x,
                        handle=handle,
                        num_worst_tokens=num_recv_tokens,
                        config=default_dispatch_config,
                        async_finish=True,
                        allocate_on_comm_stream=True,
                    ),
                    buffer=buffer,
                    timing_group=timing_group,
                    num_warmups=args.warmup,
                    num_iterations=args.iterations,
                    num_tokens_per_rank=args.num_tokens,
                    logical_bytes_per_rank=current_bytes,
                    flush_l2=args.flush_l2_cache,
                    max_rank_latency=args.max_rank_latency,
                )
                result["dtype"] = dtype_name
                result["tuned"] = False
                result["tuning_config"] = config_dict(default_dispatch_config, "default")
                results.append(result)

        if needs_dispatch_setup and selected_dispatch_config != default_dispatch_config:
            prepared = buffer.dispatch(
                dispatch_x,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx if needs_expand_setup else None,
                topk_weights=topk_weights if needs_expand_setup else None,
                config=selected_dispatch_config,
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            wait_event(prepared[-1])
            torch.npu.synchronize()
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_expert_counts,
                handle,
            ) = prepared[:5]

        expanded_x = None
        expanded_topk_weights = None
        expand_handle = None
        if needs_expand_setup:
            if recv_topk_idx is None or recv_topk_weights is None:
                raise RuntimeError("expand benchmark setup requires top-k metadata")
            if recv_expert_counts is None:
                raise RuntimeError("expand benchmark setup requires expert counts")
            expanded = buffer.expand(
                recv_x,
                recv_topk_idx,
                topk_weights=recv_topk_weights,
                num_tokens_per_expert=recv_expert_counts,
                expert_alignment=1,
                config=selected_dispatch_config,
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            wait_event(expanded[-1])
            torch.npu.synchronize()
            expanded_payload, expanded_topk_weights, expand_handle = expanded[:3]
            if isinstance(expanded_payload, tuple):
                expanded_payload_bytes = sum(value.numel() * value.element_size() for value in expanded_payload)
            else:
                expanded_payload_bytes = expanded_payload.numel() * expanded_payload.element_size()
            expanded_x = per_token_cast_back(expanded_payload, dtype)
            expanded_compute_bytes = expanded_x.numel() * expanded_x.element_size()

            if "expand" in selected_ops:
                result = benchmark_operator(
                    "expand",
                    lambda: buffer.expand(
                        recv_x,
                        recv_topk_idx,
                        topk_weights=recv_topk_weights,
                        num_tokens_per_expert=recv_expert_counts,
                        expert_alignment=1,
                        config=selected_dispatch_config,
                        async_finish=True,
                        allocate_on_comm_stream=True,
                    ),
                    buffer=buffer,
                    timing_group=timing_group,
                    num_warmups=args.warmup,
                    num_iterations=args.iterations,
                    num_tokens_per_rank=args.num_tokens,
                    logical_bytes_per_rank=(dispatch_payload_bytes + expanded_payload_bytes),
                    flush_l2=args.flush_l2_cache,
                    max_rank_latency=args.max_rank_latency,
                )
                result["dtype"] = args.dtype
                results.append(result)

            if "cached-expand" in selected_ops:
                result = benchmark_operator(
                    "cached-expand",
                    lambda: buffer.expand(
                        recv_x,
                        handle=expand_handle,
                        config=selected_dispatch_config,
                        async_finish=True,
                        allocate_on_comm_stream=True,
                    ),
                    buffer=buffer,
                    timing_group=timing_group,
                    num_warmups=args.warmup,
                    num_iterations=args.iterations,
                    num_tokens_per_rank=args.num_tokens,
                    logical_bytes_per_rank=(dispatch_payload_bytes + expanded_payload_bytes),
                    flush_l2=args.flush_l2_cache,
                    max_rank_latency=args.max_rank_latency,
                )
                result["dtype"] = args.dtype
                results.append(result)

            if "aggregate" in selected_ops:
                result = benchmark_operator(
                    "aggregate",
                    lambda: buffer.aggregate(
                        expanded_x,
                        expand_handle,
                        topk_weights=expanded_topk_weights,
                        config=selected_dispatch_config,
                        async_finish=True,
                        allocate_on_comm_stream=True,
                    ),
                    buffer=buffer,
                    timing_group=timing_group,
                    num_warmups=args.warmup,
                    num_iterations=args.iterations,
                    num_tokens_per_rank=args.num_tokens,
                    logical_bytes_per_rank=(expanded_compute_bytes + payload_bytes),
                    flush_l2=args.flush_l2_cache,
                    max_rank_latency=args.max_rank_latency,
                )
                result["dtype"] = args.dtype
                results.append(result)

        if "full-dispatch" in selected_ops:

            def full_dispatch():
                dispatched = buffer.dispatch(
                    dispatch_x,
                    num_tokens_per_rank=num_tokens_per_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    config=selected_dispatch_config,
                    async_finish=True,
                    allocate_on_comm_stream=True,
                )
                expanded = buffer.expand(
                    dispatched[0],
                    dispatched[1],
                    topk_weights=dispatched[2],
                    num_tokens_per_expert=dispatched[3],
                    expert_alignment=1,
                    config=selected_dispatch_config,
                    previous_event=dispatched[-1],
                    async_finish=True,
                    allocate_on_comm_stream=True,
                )
                return dispatched, expanded

            result = benchmark_operator(
                "full-dispatch",
                full_dispatch,
                buffer=buffer,
                timing_group=timing_group,
                num_warmups=args.warmup,
                num_iterations=args.iterations,
                num_tokens_per_rank=args.num_tokens,
                flush_l2=args.flush_l2_cache,
                max_rank_latency=args.max_rank_latency,
            )
            result["dtype"] = args.dtype
            result["bandwidth_definition"] = "algorithm"
            result["algorithm_bytes_per_rank"] = full_algorithm_bytes
            result["algorithm_bandwidth_gbps"] = full_algorithm_bytes / result["mean_us"] / 1000.0
            results.append(result)

        should_prepare_combine = any(op in ("combine", "full-combine", "e2e") for op in selected_ops)
        if should_prepare_combine:
            if recv_x is None:
                raise RuntimeError("combine benchmark setup requires dispatch X")
            combine_x, _ = materialize_combine_inputs(buffer, recv_x, dtype=dtype)
            if args.tune:
                combine_config, combine_result, candidates = tune_operator(
                    "combine",
                    combine_config_candidates(default_combine_config),
                    lambda config: lambda: buffer.combine(
                        combine_x,
                        handle,
                        config=config,
                        async_finish=True,
                        allocate_on_comm_stream=True,
                    ),
                    dtype_name=args.dtype.upper(),
                    logical_bytes_per_rank=payload_bytes,
                    buffer=buffer,
                    timing_group=timing_group,
                    rank=rank,
                    args=args,
                )
                tuning_results["combine"] = candidates
                if "combine" in selected_ops:
                    results.append(combine_result)
            else:
                combine_config = default_combine_config
                if "combine" in selected_ops:
                    combine_result = benchmark_operator(
                        "combine",
                        lambda: buffer.combine(
                            combine_x,
                            handle,
                            config=combine_config,
                            async_finish=True,
                            allocate_on_comm_stream=True,
                        ),
                        buffer=buffer,
                        timing_group=timing_group,
                        num_warmups=args.warmup,
                        num_iterations=args.iterations,
                        num_tokens_per_rank=args.num_tokens,
                        logical_bytes_per_rank=payload_bytes,
                        flush_l2=args.flush_l2_cache,
                        max_rank_latency=args.max_rank_latency,
                    )
                    combine_result["dtype"] = args.dtype
                    combine_result["tuned"] = False
                    combine_result["tuning_config"] = config_dict(combine_config, "default")
                    results.append(combine_result)
        else:
            combine_config = default_combine_config

        if "full-combine" in selected_ops:

            def full_combine():
                aggregated = buffer.aggregate(
                    expanded_x,
                    expand_handle,
                    topk_weights=expanded_topk_weights,
                    config=selected_dispatch_config,
                    async_finish=True,
                    allocate_on_comm_stream=True,
                )
                combined = buffer.combine(
                    aggregated[0],
                    handle,
                    config=combine_config,
                    previous_event=aggregated[-1],
                    async_finish=True,
                    allocate_on_comm_stream=True,
                )
                return aggregated, combined

            result = benchmark_operator(
                "full-combine",
                full_combine,
                buffer=buffer,
                timing_group=timing_group,
                num_warmups=args.warmup,
                num_iterations=args.iterations,
                num_tokens_per_rank=args.num_tokens,
                flush_l2=args.flush_l2_cache,
                max_rank_latency=args.max_rank_latency,
            )
            result["dtype"] = args.dtype
            result["bandwidth_definition"] = "algorithm"
            result["algorithm_bytes_per_rank"] = full_algorithm_bytes
            result["algorithm_bandwidth_gbps"] = full_algorithm_bytes / result["mean_us"] / 1000.0
            results.append(result)

        if "e2e" in selected_ops:

            def dispatch_combine():
                dispatched = buffer.dispatch(
                    dispatch_x,
                    num_tokens_per_rank=num_tokens_per_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    num_worst_tokens=num_recv_tokens,
                    config=selected_dispatch_config,
                    async_finish=True,
                    allocate_on_comm_stream=True,
                )
                combine_x, combine_topk_weights = materialize_combine_inputs(
                    buffer,
                    dispatched[0],
                    dispatched[2],
                    dtype=dtype,
                )
                combined = buffer.combine(
                    combine_x,
                    dispatched[4],
                    topk_weights=combine_topk_weights,
                    config=combine_config,
                    async_finish=True,
                    allocate_on_comm_stream=True,
                )
                return dispatched, combined

            results.append(
                benchmark_operator(
                    "dispatch-combine",
                    dispatch_combine,
                    buffer=buffer,
                    timing_group=timing_group,
                    num_warmups=args.warmup,
                    num_iterations=args.iterations,
                    num_tokens_per_rank=args.num_tokens,
                    logical_bytes_per_rank=(dispatch_payload_bytes + payload_bytes),
                    flush_l2=args.flush_l2_cache,
                    max_rank_latency=args.max_rank_latency,
                )
            )
            results[-1]["dtype"] = args.dtype

        report: Dict[str, Any] = {
            "config": {
                "ep_size": num_ranks,
                "num_local_ranks": args.num_local_ranks,
                "num_nodes": num_ranks // args.num_local_ranks,
                "process_group_backend": "hccl",
                "num_tokens": args.num_tokens,
                "hidden": args.hidden,
                "num_topk": args.num_topk,
                "num_experts": args.num_experts,
                "num_recv_tokens_on_reported_rank": num_recv_tokens,
                "dtype": args.dtype,
                "route": args.route,
                "transport": args.transport,
                "allow_mnnvl_ignored": args.allow_mnnvl,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "iterations_requested": args.iterations,
                "iterations_reported": args.iterations - 1,
                "latency_scope": ("max_rank" if args.max_rank_latency else "rank_local"),
                "reported_rank": 0,
                "flush_l2_cache": args.flush_l2_cache,
                "l2_cache_flush_bytes": (L2_CACHE_FLUSH_BYTES if args.flush_l2_cache else 0),
                "correctness_check": ("skipped" if args.skip_correctness_check else "passed"),
                "tuning_enabled": args.tune,
                "fp8_enabled": x_fp8 is not None,
                "num_sms": selected_dispatch_config.num_sms,
                "dispatch_nvl_send_tokens": (selected_dispatch_config.num_max_nvl_chunked_send_tokens),
                "combine_nvl_send_tokens": (combine_config.num_max_nvl_chunked_send_tokens),
                "symmetric_workspace_bytes": num_bytes,
                "nvl_recv_tokens": selected_dispatch_config.num_max_nvl_chunked_recv_tokens,
                "logical_payload_excludes_metadata": True,
                "logical_payload_bandwidth_scope": "per_rank",
            },
            "results": results,
            "tuning_candidates": tuning_results,
        }
        if rank == 0:
            if args.stdout_format in ("deepep", "both"):
                print_deepep_report(report)
            if args.stdout_format in ("json", "both"):
                print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            write_json(args.output_json, report)
        buffer.barrier()
        buffer.destroy()
        dist.barrier(group=timing_group)
    finally:
        if buffer is not None and buffer.is_available():
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DeepEP Clos Buffer operators on real NPUs")
    parser.add_argument("--op", choices=SUPPORTED_OPS, default="all")
    parser.add_argument(
        "--ep-size",
        type=int,
        default=None,
        help=("global EP rank count; without --nproc-per-node this selects the single-node process count"),
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help=("legacy DeepEP local process count; defaults to 8 when both --ep-size and --nproc-per-node are omitted"),
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="enable multi-node mode and spawn this many local NPU processes",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=None,
        help="zero-based node index in multi-node mode",
    )
    parser.add_argument("--first-device", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--num-bytes",
        type=int,
        default=None,
        help=("symmetric workspace bytes; defaults to a 2 MiB-aligned size derived from the benchmark shape"),
    )
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp8"),
        default="bf16",
        help=(
            "dispatch payload dtype; FP8 uses E4M3 data with per-128-value "
            "float32 scales and BF16 for aggregate/combine"
        ),
    )
    parser.add_argument(
        "--route",
        choices=("random", "balanced", "local", "remote"),
        default="random",
    )
    parser.add_argument(
        "--transport",
        choices=("auto", "mte", "roce", "sdma", "udma"),
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=50)
    tuning_group = parser.add_mutually_exclusive_group()
    tuning_group.add_argument(
        "--tune",
        dest="tune",
        action="store_true",
        help="scan DeepEP dispatch/combine NVL chunk configurations",
    )
    tuning_group.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="benchmark Buffer's configured defaults without claiming Best",
    )
    parser.set_defaults(tune=True)
    parser.add_argument(
        "--skip-fp8",
        action="store_true",
        help="skip DeepEP's FP8 correctness and dispatch performance cases",
    )
    flush_l2_group = parser.add_mutually_exclusive_group()
    flush_l2_group.add_argument(
        "--flush-l2-cache",
        dest="flush_l2_cache",
        action="store_true",
        help="flush device L2 with a 256 MB write before every timed iteration",
    )
    flush_l2_group.add_argument(
        "--no-flush-l2-cache",
        dest="flush_l2_cache",
        action="store_false",
        help="keep device L2 contents between timed iterations",
    )
    parser.set_defaults(flush_l2_cache=True)
    parser.add_argument(
        "--max-rank-latency",
        action="store_true",
        help="report the maximum latency across ranks instead of DeepEP-style rank-local latency",
    )
    parser.add_argument(
        "--allow-mnnvl",
        action="store_true",
        help="accepted for DeepEP CLI compatibility; ignored on Ascend",
    )
    parser.add_argument("--num-sms", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--skip-correctness-check", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--stdout-format",
        choices=("deepep", "json", "both"),
        default="deepep",
        help="rank-0 stdout format; DeepEP-compatible text is the default",
    )
    parser.add_argument("--dist-init-method", help=argparse.SUPPRESS)
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help="HCCL rendezvous host shared by every node",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "8361")),
        help="HCCL rendezvous port shared by every node",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT"),
        help=("CANN SHMEM endpoint; defaults to tcp://MASTER_ADDR:(MASTER_PORT+1)"),
    )
    args = parser.parse_args()
    if args.num_processes is not None and args.nproc_per_node is not None and args.num_processes != args.nproc_per_node:
        parser.error("--num-processes and --nproc-per-node must match")

    legacy_num_nodes = int(os.getenv("WORLD_SIZE", "1"))
    legacy_node_rank = int(os.getenv("RANK", "0"))
    explicit_local_ranks = args.nproc_per_node if args.nproc_per_node is not None else args.num_processes
    if args.ep_size is None:
        args.num_local_ranks = explicit_local_ranks or 8
        args.ep_size = legacy_num_nodes * args.num_local_ranks
        args.node_rank = legacy_node_rank if args.node_rank is None else args.node_rank
    else:
        args.num_local_ranks = explicit_local_ranks or args.ep_size
        args.node_rank = 0 if args.node_rank is None else args.node_rank

    if args.dist_init_method is not None:
        prefix = "tcp://"
        if not args.dist_init_method.startswith(prefix):
            parser.error("--dist-init-method must use tcp://HOST:PORT")
        host_port = args.dist_init_method[len(prefix) :]
        host, separator, port = host_port.rpartition(":")
        if not separator or not host or not port.isdigit():
            parser.error("--dist-init-method must use tcp://HOST:PORT")
        args.master_addr = host
        args.master_port = int(port)

    try:
        validate_launch_args(args)
        args.shmem_ip_port = effective_shmem_ip_port(args)
    except ValueError as error:
        parser.error(str(error))
    global_num_ranks = args.ep_size
    if args.num_tokens <= 0 or args.hidden <= 0:
        parser.error("--num-tokens and --hidden must be greater than zero")
    if args.num_experts <= 0 or args.num_experts % global_num_ranks:
        parser.error(
            "--num-experts must be positive and divisible by the global EP size (WORLD_SIZE * --num-processes)"
        )
    if args.num_topk <= 0 or args.num_topk > min(16, args.num_experts):
        parser.error("--num-topk must be in [1, min(16, num_experts)]")
    if args.warmup < 0 or args.iterations < 2:
        parser.error("--warmup must be nonnegative and --iterations at least two")
    if args.dtype == "fp8" and args.skip_fp8:
        parser.error("--skip-fp8 cannot be used with --dtype fp8")
    if args.num_sms is not None and (args.num_sms < 0 or args.num_sms > 64 or args.num_sms % 2):
        parser.error("--num-sms must be an even integer in [0, 64]")
    if args.num_bytes is None:
        if args.num_sms is not None:
            Buffer.set_num_sms(args.num_sms)
        dispatch_config = Buffer.get_dispatch_config(global_num_ranks)
        combine_config = Buffer.get_combine_config(global_num_ranks)
        hidden_bytes = args.hidden * 2
        args.num_bytes = max(
            dispatch_config.get_nvl_buffer_size_hint(
                hidden_bytes,
                global_num_ranks,
                num_tokens=args.num_tokens,
            ),
            combine_config.get_nvl_buffer_size_hint(
                hidden_bytes,
                global_num_ranks,
                num_tokens=args.num_tokens,
            ),
        )
    if args.num_bytes <= 0 or args.num_bytes % DEVICE_LARGE_PAGE_SIZE:
        parser.error(f"--num-bytes must be positive and aligned to {DEVICE_LARGE_PAGE_SIZE}")
    return args


def main() -> None:
    args = parse_args()
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    print(
        f"Launching EP benchmark with ep_size={args.ep_size}, "
        f"node_rank={args.node_rank}, "
        f"local_processes={args.num_local_ranks}, "
        f"global_ranks={args.node_rank * args.num_local_ranks}-"
        f"{(args.node_rank + 1) * args.num_local_ranks - 1}, "
        f"master={args.master_addr}:{args.master_port}, "
        f"shmem_ip_port={args.shmem_ip_port}",
        flush=True,
    )
    if args.num_local_ranks == 1:
        run_rank(0, args)
    else:
        mp.spawn(
            run_rank,
            args=(args,),
            nprocs=args.num_local_ranks,
            join=True,
        )


if __name__ == "__main__":
    main()


# Single node, 8 ranks:
# python benchmarks/bench_ep.py --num-processes 8
# python benchmarks/bench_ep.py --ep-size 8
# Multi-node, 64 ranks (run once per node with node-rank 0..7):
# python benchmarks/bench_ep.py --ep-size 64 --nproc-per-node 8 \
#   --node-rank 0 --master-addr 10.0.0.1 --master-port 8899 \
#   --transport udma
# Multi-node, 128 ranks (run once per node with node-rank 0..15):
# python benchmarks/bench_ep.py --ep-size 128 --nproc-per-node 8 \
#   --node-rank 0 --master-addr 10.0.0.1 --master-port 8899 \
#   --transport udma
