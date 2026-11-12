# Copyright (c) 2026, Lu Lu
"""Dispatch -> per-expert Linear/SiLU/Linear -> Combine precision ST (torchrun)."""

import argparse
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

__test__ = False


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tokens", type=int, nargs="+", default=[32, 48, 64, 72, 256])
    result.add_argument("--hidden", type=int, default=3584)
    result.add_argument("--ffn-hidden", type=int, default=256)
    result.add_argument("--topk", type=int, default=8)
    result.add_argument("--experts", type=int, default=384)
    result.add_argument("--capacity", type=int, default=256)
    result.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--repeat", type=int, default=3)
    result.add_argument("--output", type=Path, required=True)
    return result


def input_tensor(rank, tokens, hidden, dtype, seed, iteration):
    import torch

    generator = torch.Generator().manual_seed(seed + rank * 1009 + iteration * 1000003)
    x = (torch.randn(tokens, hidden, generator=generator) / 4).to(dtype)
    ids = torch.arange(tokens)
    x[:, 0], x[:, 1] = (rank // 16) / 16, (rank % 16) / 16
    x[:, 2], x[:, 3] = (ids // 16) / 16, (ids % 16) / 16
    return x


def routing_weights(rank, tokens, topk, seed, iteration):
    import torch

    generator = torch.Generator().manual_seed(seed + rank * 2017 + iteration * 2000003)
    weights = torch.rand(tokens, topk, generator=generator) + 0.25
    return weights / weights.sum(dim=1, keepdim=True)


def expert_parameters(expert, hidden, intermediate, dtype, seed):
    import torch

    generator = torch.Generator().manual_seed(seed + expert * 65537 + 7000003)
    return (
        (torch.randn(hidden, intermediate, generator=generator) / math.sqrt(hidden)).to(dtype),
        (torch.randn(intermediate, generator=generator) / 32).to(dtype),
        (torch.randn(intermediate, hidden, generator=generator) / math.sqrt(intermediate)).to(dtype),
        (torch.randn(hidden, generator=generator) / 32).to(dtype),
    )


def expert_forward(x, parameters):
    import torch.nn.functional as functional

    w1, b1, w2, b2 = parameters
    return functional.silu(x @ w1 + b1) @ w2 + b2


def reference_ffn(x, parameters):
    """CPU FP32 arithmetic, rounded at the same BF16/FP16 operator boundaries."""
    import torch
    import torch.nn.functional as functional

    w1, b1, w2, b2 = (value.float() for value in parameters)
    z = functional.linear(x.float(), w1.T).to(x.dtype)
    z = (z.float() + b1).to(x.dtype).float()
    activated = (z * torch.sigmoid(z)).to(x.dtype)
    y = functional.linear(activated.float(), w2.T).to(x.dtype)
    return (y.float() + b2).to(x.dtype)


def reference_combine(x, routes, weights, intermediate, seed):
    """Original source tokens/routes only; never consumes device results/handles."""
    import torch

    selected = {}
    for token, experts in enumerate(routes):
        for slot, expert in enumerate(experts):
            selected.setdefault(expert, []).append((token, slot))
    output = torch.zeros(x.shape, dtype=torch.float32)
    for expert, pairs in selected.items():
        tokens, slots = map(list, zip(*pairs))
        parameters = expert_parameters(expert, x.shape[1], intermediate, x.dtype, seed)
        contribution = reference_ffn(x[tokens], parameters).float()
        output.index_add_(0, torch.tensor(tokens), contribution * weights[tokens, slots, None])
    return output.to(x.dtype)


def expert_ranges(prefix, world, counts, capacity):
    if len(prefix) != world * len(counts) or any(a > b for a, b in zip([0] + prefix, prefix)):
        raise AssertionError("Invalid Dispatch cumulative counts")
    ranges, start = [], 0
    for local, count in enumerate(counts):
        end = prefix[(local + 1) * world - 1]
        if end - start != count or end > capacity:
            raise AssertionError("Dispatch expert count differs from the independent CPU routes")
        ranges.append((start, end))
        start = end
    return ranges


def check_close(actual, expected, dtype):
    import torch

    rtol, atol = (0.02, 0.002) if dtype == "bf16" else (0.005, 0.0002)
    actual, expected = actual.float(), expected.float()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    difference = (actual - expected).abs()
    return {
        "max_abs": difference.max().item() if difference.numel() else 0.0,
        "max_scaled": (difference / (atol + rtol * expected.abs())).max().item() if difference.numel() else 0.0,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= min(args.tokens) <= max(args.tokens) <= args.capacity <= 256:
        raise ValueError("Require 1 <= tokens <= capacity <= 256")
    if args.hidden < 4 or args.ffn_hidden < 1 or not 1 <= args.topk <= 16 or args.repeat < 3 or args.seed < 0:
        raise ValueError("Require H>=4, FFN hidden>=1, K=1..16, repeat>=3 and seed>=0")
    if os.environ.get("DEEPEP_LL_BACKEND", "rdma") != "rdma" or not os.environ.get("DEEPEP_SHMEM_ENDPOINT"):
        raise ValueError("Use DEEPEP_LL_BACKEND=rdma and set DEEPEP_SHMEM_ENDPOINT")
    os.environ["DEEPEP_LL_BACKEND"] = "rdma"
    world, rank, local = (int(os.environ[key]) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if int(os.environ["LOCAL_WORLD_SIZE"]) != 16 or world not in (16, 32, 64, 128, 256):
        raise ValueError("Use 16 participating ranks per host, EP16/32/64/128/256")
    if not world <= args.experts <= 1024 or args.experts % world:
        raise ValueError("Require EP<=E<=1024 and E divisible by EP")

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    torch.npu.set_device(local)
    torch.set_num_threads(1)
    dist.init_process_group("hccl", timeout=timedelta(minutes=30))
    sync = torch.zeros(1, dtype=torch.int32, device="npu")
    dist.all_reduce(sync)
    torch.npu.synchronize()  # HCCL must be materialized before SHMEM.
    options = dict(vars(args), output=str(args.output))
    peer_options = [None] * world
    dist.all_gather_object(peer_options, options)
    if any(value != options for value in peer_options):
        raise ValueError("All ranks must use the same FFN test options")
    rank_dir = args.output / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=False)
    elastic_dir = Path(__file__).resolve().parents[1] / "elastic"
    reference = SimpleNamespace(**runpy.run_path(str(elastic_dir / "test_ep_low_latency.py")))
    deep_ep = runpy.run_path(str(elastic_dir / "test_dispatch.py"))["_import_installed_deep_ep"]()
    args.group_backend = "hccl"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    buffer = deep_ep.ElasticBuffer(
        group=dist.group.WORLD,
        num_bytes=8 << 30,
        num_max_tokens_per_rank=args.capacity,
        hidden=0,
        num_topk=0,
        explicitly_destroy=True,
        machine_type="mesh_clos",
    )
    compute, consumer = torch.npu.Stream(), torch.npu.Stream()
    local_experts = args.experts // world
    cpu_parameters = [
        expert_parameters(rank * local_experts + e, args.hidden, args.ffn_hidden, dtype, args.seed)
        for e in range(local_experts)
    ]
    with torch.npu.stream(compute):
        device_parameters = [tuple(value.npu() for value in values) for values in cpu_parameters]
    if rank == 0:
        print(
            f"UBX FFN E2E EP={world} H={args.hidden} FFN={args.ffn_hidden} K={args.topk} E={args.experts} "
            f"dtype={args.dtype} repeat={args.repeat} combine_calls_per_dispatch=2 "
            "reference=CPU-FP32-with-dtype-rounding precision-only",
            flush=True,
        )
    with torch.no_grad():
        for bs in args.tokens:
            retained = []
            for iteration in range(args.repeat):
                if rank == 0:
                    print(f"BS={bs} round={iteration + 1}/{args.repeat} Dispatch -> FFN -> Combine", flush=True)
                routes = reference.make_routes(world, bs, args.topk, args.experts, args.seed + iteration * 1000003)
                entries = reference.expert_entries(routes, args.experts, world, rank)
                cpu_inputs = [input_tensor(r, bs, args.hidden, dtype, args.seed, iteration) for r in range(world)]
                weights = routing_weights(rank, bs, args.topk, args.seed, iteration)
                with torch.npu.stream(compute):
                    x = cpu_inputs[rank].npu()
                    indices = torch.tensor(routes[rank], dtype=torch.int64, device="npu")
                    recv, _, _, handle, dispatched = buffer.dispatch(
                        x,
                        topk_idx=indices,
                        topk_weights=weights.npu(),
                        num_experts=args.experts,
                        num_max_tokens_per_rank=args.capacity,
                        do_expand=True,
                        do_cpu_sync=False,
                    )
                    assert dispatched.event is None and handle.is_low_latency
                    # LL exposes no public expert-count list. The test reads the
                    # actual device prefix; CPU routing independently checks it.
                    prefix = handle.send_counts.cpu().tolist()
                ranges = []
                reference.agree_check(
                    lambda: ranges.extend(expert_ranges(prefix, world, [len(v) for v in entries], recv.shape[0])),
                    args,
                    rank,
                )
                with torch.npu.stream(compute):
                    expert = torch.full_like(recv, float("nan"))
                    for (start, end), parameters in zip(ranges, device_parameters):
                        if end > start:
                            expert[start:end] = expert_forward(recv[start:end], parameters)
                    ready = buffer.capture()
                # No Host/device-wide sync or collective between the FFN and
                # Combine: its public predecessor event must order the streams.
                combined = []
                # Dispatch and Combine share a bank counter. Two consecutive
                # Combines exercise both banks; repeated D/C pairs alone do not.
                for _ in range(2):
                    with torch.npu.stream(consumer):
                        out, out_weights, done = buffer.combine(
                            expert,
                            handle,
                            previous_event=ready,
                            allocate_on_comm_stream=True,
                            async_with_compute_stream=True,
                        )
                        assert out_weights is None and done.event is not None
                        done.current_stream_wait(release_handle=True)
                        consumed = out.clone()
                        consumed_ready = buffer.capture()
                    combined.append((out, consumed))
                consumed_ready.current_stream_wait()
                actual = [snapshot.cpu() for _, snapshot in combined]
                metrics = {}

                def verify():
                    received, computed = recv[: prefix[-1]].cpu(), expert[: prefix[-1]].cpu()
                    ffn_errors = []
                    for (start, end), rows, parameters in zip(ranges, entries, cpu_parameters):
                        if not rows:
                            continue
                        block = received[start:end]
                        ids = (block[:, :4].float() * 16).round().to(torch.int64)
                        keys = (ids[:, 0] * 16 + ids[:, 1]) * bs + ids[:, 2] * 16 + ids[:, 3]
                        order = keys.argsort()
                        expected_input = torch.stack([cpu_inputs[source][token] for source, token in rows])
                        torch.testing.assert_close(block[order], expected_input, rtol=0, atol=0)
                        ffn_errors.append(
                            check_close(
                                computed[start:end][order], reference_ffn(expected_input, parameters), args.dtype
                            )
                        )
                    expected = reference_combine(cpu_inputs[rank], routes[rank], weights, args.ffn_hidden, args.seed)
                    metrics["ffn"] = {
                        name: max((value[name] for value in ffn_errors), default=0.0)
                        for name in ("max_abs", "max_scaled")
                    }
                    combine_errors = [check_close(value, expected, args.dtype) for value in actual]
                    metrics["combine"] = {
                        name: max(value[name] for value in combine_errors) for name in ("max_abs", "max_scaled")
                    }
                    for old_out, old_consumed, old_expected in retained:
                        check_close(old_out.cpu(), old_expected, args.dtype)
                        check_close(old_consumed.cpu(), old_expected, args.dtype)
                    retained.extend((value, snapshot, expected) for value, snapshot in combined)

                reference.agree_check(verify, args, rank)
                record = {"bs": bs, "round": iteration + 1, "correctness": "PASS", **metrics}
                (rank_dir / f"bs-{bs}-round-{iteration + 1}.json").write_text(json.dumps(record) + "\n")
                reports = [None] * world
                dist.all_gather_object(reports, record)
                if rank == 0:
                    maximum = {
                        stage: {key: max(value[stage][key] for value in reports) for key in ("max_abs", "max_scaled")}
                        for stage in ("ffn", "combine")
                    }
                    with (args.output / "summary.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                dict(options, world=world, bs=bs, round=iteration + 1, correctness="PASS", **maximum)
                            )
                            + "\n"
                        )
                    print(
                        f"BS={bs} round={iteration + 1}/{args.repeat} Dispatch/FFN/Combine/reuse=PASS "
                        f"max_abs(ffn/combine)={maximum['ffn']['max_abs']:.6g}/{maximum['combine']['max_abs']:.6g} "
                        f"max_scaled(ffn/combine)={maximum['ffn']['max_scaled']:.3f}/{maximum['combine']['max_scaled']:.3f}",
                        flush=True,
                    )
    buffer.destroy()
    buffer = None
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
