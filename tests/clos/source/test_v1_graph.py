"""Fixed-capacity V1 eager/NPUGraph correctness against an independent CPU oracle.

Launch with torchrun. Group/Buffer construction and CPU checks stay outside
capture. Tensor shapes stay fixed during replay; route values and payloads
change. Unequal local shapes are covered separately, including a zero-T rank.
"""

import argparse
import hashlib
import json
import os
import traceback
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from deep_ep.backends.clos import Buffer, Config
from deep_ep.backends.clos._native import load_native_module


def lengths(capacity, world, uneven, step=0):
    if not uneven:
        return [capacity] * world
    return [
        0 if (r + step) % 8 == 0 else (1 if (r + step) % 8 == 1 else max(1, capacity - (r + step) % 5))
        for r in range(world)
    ]


def routes(rank, count, step):
    t = torch.arange(count, dtype=torch.int64)[:, None]
    k = torch.arange(8, dtype=torch.int64)[None, :]
    out = (t * 3 + rank * 47 + k * 17 + step * 23) % 384
    if step % 4 == 2:
        # Replay from populated to entirely empty routes and back without
        # changing graph tensor shapes or inheriting stale notify counters.
        return torch.full_like(out, -1)
    return out.masked_fill((t + k + step) % 11 == 0, -1)


def payload(rank, tokens, hidden, step, fp8=False):
    # Exactly representable BF16 values. Bounds do not grow with EP or T.
    h = torch.arange(hidden, dtype=torch.int64)[None, :]
    value = (((tokens[:, None] * 7 + h + rank * 19 + step * 31) % 127 - 63).float() / 64).bfloat16()
    return value.to(torch.float8_e4m3fn) if fp8 else value


def scales(rank, tokens, hidden, step, integer):
    columns = torch.arange((hidden + 127) // 128)[None, :]
    value = (tokens[:, None] * 3 + columns + rank * 7 + step) % 17 + 1
    return value.int() if integer else value.float() / 8


def weights(rank, count, step):
    return ((torch.arange(count * 8).reshape(count, 8) + rank + step) % 13).float() / 16


def check(name, actual, expected):
    actual = actual.cpu()
    if actual.dtype == expected.dtype == torch.float8_e4m3fn:
        actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
        if actual.shape == expected.shape:
            bad = torch.nonzero(actual != expected)
            sample = bad[:4].tolist()
        else:
            sample = [list(actual.shape), list(expected.shape)]
        raise AssertionError(f"{name} mismatch: {sample}")


def verify(outputs, rank, ns, hidden, step, padded, inject, fp8=False, int_scales=False):
    rx, ridx, rw, counts, handle, combined, combined_w, cached, cached_combined, rs, cs = outputs
    world = len(ns)
    local_experts = 384 // world
    offset = 0
    expert_counts = torch.zeros(local_experts, dtype=torch.int64)
    for source, count in enumerate(ns):
        idx = routes(source, count, step)
        selected = (idx >= rank * local_experts) & (idx < (rank + 1) * local_experts)
        token_ids = torch.nonzero(selected.any(1)).flatten()
        mapped = torch.where(selected, idx - rank * local_experts, -1)
        expert_counts += torch.bincount(mapped[selected], minlength=local_experts)
        w = weights(source, count, step).masked_fill(~selected, 0)
        for begin in range(0, len(token_ids), 128):
            ids = token_ids[begin : begin + 128]
            end = offset + len(ids)
            expected = payload(source, ids, hidden, step, fp8)
            if inject and offset == 0 and len(ids):
                expected[0, 0] = 2
            check("recv_x", rx[offset:end], expected)
            check("cached_recv_x", cached[offset:end], expected)
            check("recv_topk_idx", ridx[offset:end], mapped[ids])
            check("recv_weights", rw[offset:end], w[ids])
            check("src_idx", handle[3][offset:end], ids.int())
            if fp8:
                expected_scales = scales(source, ids, hidden, step, int_scales)
                check("recv_scales", rs[offset:end], expected_scales)
                check("cached_scales", cs[offset:end], expected_scales)
            offset = end
    assert rx.size(0) >= offset
    if padded:
        assert counts == []
        for name, value, fill in [
            ("padding_x", rx, 0),
            ("padding_cached", cached, 0),
            ("padding_idx", ridx, -1),
            ("padding_weights", rw, 0),
        ]:
            # Scalar expected constants, independent of any kernel-produced reference.
            padding = value[offset:]
            if padding.dtype == torch.float8_e4m3fn:
                padding = padding.view(torch.uint8)
            assert not bool((padding != fill).any().item()), name
        if fp8:
            assert not bool(rs[offset:].any().item())
            assert not bool(cs[offset:].any().item())
    else:
        assert rx.size(0) == offset and counts == expert_counts.tolist()
    idx = routes(rank, ns[rank], step)
    membership = torch.zeros((ns[rank], world), dtype=torch.bool)
    for peer in range(world):
        membership[:, peer] = ((idx >= peer * local_experts) & (idx < (peer + 1) * local_experts)).any(1)
    for begin in range(0, ns[rank], 128):
        ids = torch.arange(begin, min(begin + 128, ns[rank]))
        source_x = payload(rank, ids, hidden, step, fp8).float()
        expected = torch.full_like(source_x, 0.375)
        cached_expected = torch.zeros_like(source_x)
        for peer in range(world):
            mask = membership[ids, peer]
            expected[mask] += source_x[mask]
            cached_expected[mask] += source_x[mask]
        check("combine", combined[ids[0] : ids[-1] + 1], expected.bfloat16())
        check("cached_combine", cached_combined[ids[0] : ids[-1] + 1], cached_expected.bfloat16())
    check("combined_weights", combined_w, weights(rank, ns[rank], step).masked_fill(idx < 0, 0))
    assert combined.shape == (ns[rank], hidden)


def verify_all(*args, **kwargs):
    # CPU checks run outside capture. Keep every rank at the same API boundary
    # when one check fails, rather than hiding it behind collective destruction
    # while peers enter their next dispatch.
    error = None
    try:
        verify(*args, **kwargs)
    except Exception:
        error = traceback.format_exc()
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise AssertionError(error or next(item for item in errors if item))


def capacity_guards(buffer, rank, world, hidden, config):
    # Every source sends one token to rank 0. With C=1 this would need world
    # receive rows, so num_worst_tokens=1 is a concrete insufficient bound.
    x = torch.ones((1, hidden), dtype=torch.bfloat16, device="npu")
    idx = torch.zeros((1, 1), dtype=torch.int64, device="npu")
    weights = torch.ones((1, 1), device="npu")
    nr, _, ne, member, _ = buffer.get_dispatch_layout(idx, 384)
    errors = []
    rejected = 0
    for zero_copy in (False, True):
        for bound in (1, world - 1, world + 1):
            for native in (False, True):
                try:
                    if native:
                        buffer.native.intranode_dispatch(
                            x,
                            None,
                            idx,
                            weights,
                            nr,
                            member,
                            ne,
                            0,
                            None,
                            None,
                            1,
                            bound,
                            32,
                            6,
                            256,
                            zero_copy,
                            None,
                            1,
                        )
                    else:
                        buffer.dispatch(
                            x,
                            topk_idx=idx,
                            topk_weights=weights,
                            num_tokens_per_rank=nr,
                            num_tokens_per_expert=ne,
                            is_token_in_rank=member,
                            num_worst_tokens=bound,
                            zero_copy=zero_copy,
                            config=config,
                        )
                except (ValueError, RuntimeError) as error:
                    if "fixed dispatch output requires" not in str(error):
                        errors.append(str(error))
                    else:
                        rejected += 1
                else:
                    errors.append(f"unsafe capacity accepted: native={native}, bound={bound}")
    all_errors = [None] * world
    dist.all_gather_object(all_errors, errors)
    assert not any(all_errors), all_errors
    rx, _, rw, _, handle, _ = buffer.dispatch(
        x,
        topk_idx=idx,
        topk_weights=weights,
        num_tokens_per_rank=nr,
        num_tokens_per_expert=ne,
        is_token_in_rank=member,
        num_worst_tokens=world,
        config=config,
    )
    expected = torch.full((world, hidden), 1 if rank == 0 else 0, dtype=x.dtype)
    check("guard recovery dispatch", rx, expected)
    result, result_w, _ = buffer.combine(rx, handle, topk_weights=rw, config=config)
    check("guard recovery combine", result, torch.ones((1, hidden), dtype=x.dtype))
    check("guard recovery weights", result_w, torch.ones((1, 1)))
    print("CAPACITY_GUARD_PASS " + json.dumps(dict(rank=rank, rejected=rejected, recovered=True)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["mte", "udma"], required=True)
    p.add_argument("--tokens", type=int, default=129)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--replays", type=int, default=8)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--uneven", action="store_true")
    p.add_argument("--allocate-on-comm-stream", action="store_true")
    p.add_argument("--inject-error", action="store_true")
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--int-scales", action="store_true")
    p.add_argument("--vary-tokens", action="store_true")
    p.add_argument("--default-capacity", action="store_true")
    p.add_argument("--capacity-guard", action="store_true")
    p.add_argument("--fixed-output", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    if args.graph and args.vary_tokens:
        p.error("graph replay fixes physical shapes; vary routes with -1 masks instead")
    if args.default_capacity and args.uneven:
        p.error("unequal local shapes require a common explicit capacity")
    if args.capacity_guard and (args.tokens != 1 or args.default_capacity):
        p.error("capacity-guard requires tokens=1 and explicit capacity")
    if args.benchmark and not (args.graph or args.fixed_output):
        p.error("benchmark compares graph and eager with the same fixed outputs")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.config.allow_internal_format = False
    dist.init_process_group("gloo")
    ns = lengths(args.tokens, world, args.uneven)
    config = Config(32, 6, 256, 6, 128)
    nbytes = config.get_nvl_buffer_size_hint(args.hidden * 2, world, num_tokens=args.tokens)
    buffer = Buffer(
        dist.group.WORLD,
        num_nvl_bytes=nbytes,
        transport=args.transport,
        num_max_tokens_per_rank=None if args.default_capacity else args.tokens,
        explicitly_destroy=True,
    )
    native_path = load_native_module().__file__
    with open("/proc/self/maps") as maps:
        libraries = sorted(
            {
                line.split()[-1]
                for line in maps
                if any(name in line for name in ("libshmem.so", "libascend_deepep", "deep_ep/_C"))
            }
        )
    hashes = {}
    for path in libraries:
        with open(path, "rb") as library:
            hashes[path] = hashlib.file_digest(library, "sha256").hexdigest()
    print("RUNTIME_IDENTITY " + json.dumps(dict(rank=rank, native=native_path, libraries=hashes)), flush=True)
    if args.capacity_guard:
        capacity_guards(buffer, rank, world, args.hidden, config)
        torch.npu.synchronize()
        buffer.destroy()
        dist.destroy_process_group()
        return
    x = payload(rank, torch.arange(ns[rank]), args.hidden, 0, args.fp8).npu()
    # Noncontiguous scale rows exercise the staging path. Scale values are
    # transport metadata here; combine consumes the BF16 cast of the payload.
    xs = (
        scales(rank, torch.arange(ns[rank]), args.hidden, 0, args.int_scales).npu().T.contiguous().T
        if args.fp8
        else None
    )
    idx = routes(rank, ns[rank], 0).npu()
    w = weights(rank, ns[rank], 0).npu()
    b0, b1 = (torch.full(x.shape, value, dtype=torch.bfloat16, device=x.device) for value in (0.125, 0.25))
    allocation = dict(allocate_on_comm_stream=args.allocate_on_comm_stream)
    caller = torch.npu.Stream()
    graph = None

    def operations():
        nr, _, ne, membership, layout_event = buffer.get_dispatch_layout(idx, 384, async_finish=True, **allocation)
        rx, ri, rw, counts, handle, dispatch_event = buffer.dispatch(
            (x, xs) if args.fp8 else x,
            topk_idx=idx,
            topk_weights=w,
            num_tokens_per_rank=nr,
            num_tokens_per_expert=ne,
            is_token_in_rank=membership,
            num_worst_tokens=args.tokens * world if args.graph or args.fixed_output else 0,
            previous_event=layout_event,
            async_finish=True,
            **allocation,
        )
        rs = rx[1] if args.fp8 else None
        rx = rx[0] if args.fp8 else rx
        dispatch_event.current_stream_wait()
        cx, cw, combine_event = buffer.combine(
            rx.bfloat16() if args.fp8 else rx, handle, topk_weights=rw, bias=(b0, b1), async_finish=True, **allocation
        )
        crx, _, _, _, _, cached_event = buffer.dispatch(
            (x, xs) if args.fp8 else x, handle=handle, previous_event=combine_event, async_finish=True, **allocation
        )
        cs = crx[1] if args.fp8 else None
        crx = crx[0] if args.fp8 else crx
        cached_event.current_stream_wait()
        ccx, _, finish = buffer.combine(crx.bfloat16() if args.fp8 else crx, handle, async_finish=True, **allocation)
        finish.current_stream_wait()  # Join the communication stream into capture.
        return rx, ri, rw, counts, handle, cx, cw, crx, ccx, rs, cs

    try:
        caller.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(caller):
            for _ in range(3):
                outputs = operations()
        torch.npu.synchronize()
        verify_all(outputs, rank, ns, args.hidden, 0, args.graph or args.fixed_output, False, args.fp8, args.int_scales)
        del outputs
        if args.graph:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=caller):
                outputs = operations()
            pointers = [outputs[i].data_ptr() for i in (0, 1, 2, 5, 6, 7, 8)]
        for step in range(1, args.replays + 1):
            with torch.npu.stream(caller):
                if args.vary_tokens:
                    ns = lengths(args.tokens, world, args.uneven, step)
                    x = torch.empty((ns[rank], args.hidden), dtype=x.dtype, device=x.device)
                    idx = torch.empty((ns[rank], 8), dtype=idx.dtype, device=idx.device)
                    w = torch.empty((ns[rank], 8), dtype=w.dtype, device=w.device)
                    b0, b1 = (
                        torch.full(x.shape, value, dtype=torch.bfloat16, device=x.device) for value in (0.125, 0.25)
                    )
                    if args.fp8:
                        xs = (
                            torch.empty((ns[rank], (args.hidden + 127) // 128), dtype=xs.dtype, device=xs.device)
                            .T.contiguous()
                            .T
                        )
                x.copy_(payload(rank, torch.arange(ns[rank]), args.hidden, step, args.fp8))
                if args.fp8:
                    xs.copy_(scales(rank, torch.arange(ns[rank]), args.hidden, step, args.int_scales))
                idx.copy_(routes(rank, ns[rank], step))
                w.copy_(weights(rank, ns[rank], step))
                if graph is not None:
                    graph.replay()
                else:
                    outputs = operations()
            caller.synchronize()
            verify_all(
                outputs,
                rank,
                ns,
                args.hidden,
                step,
                args.graph or args.fixed_output,
                args.inject_error and step == 1,
                args.fp8,
                args.int_scales,
            )
            if graph is not None:
                assert pointers == [outputs[i].data_ptr() for i in (0, 1, 2, 5, 6, 7, 8)]
            print(
                json.dumps(
                    dict(
                        event="REPLAY_PASS",
                        rank=rank,
                        step=step,
                        graph=args.graph,
                        local_tokens=ns[rank],
                        capacity=args.tokens,
                        transport=args.transport,
                    )
                ),
                flush=True,
            )
        if args.benchmark:
            # Includes layout, normal/cached dispatch and both combines, plus
            # their stream dependencies. No CPU golden or input copies inside
            # timing; both modes use the same padded tensors and operations.
            device_us, host_us = [], []
            for sample in range(15):
                dist.barrier()
                caller.synchronize()
                begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                wall = time.perf_counter_ns()
                with torch.npu.stream(caller):
                    begin.record(caller)
                    if graph is None:
                        outputs = operations()
                    else:
                        graph.replay()
                    end.record(caller)
                caller.synchronize()
                elapsed = (time.perf_counter_ns() - wall) / 1000
                if sample >= 5:
                    device_us.append(begin.elapsed_time(end) * 1000)
                    host_us.append(elapsed)
            verify_all(outputs, rank, ns, args.hidden, args.replays, True, False, args.fp8, args.int_scales)
            print(
                "GRAPH_PIPELINE_TIMING "
                + json.dumps(
                    dict(
                        rank=rank,
                        graph=args.graph,
                        device_us=device_us,
                        host_us=host_us,
                        device_median_us=statistics.median(device_us),
                        host_median_us=statistics.median(host_us),
                        warmup=5,
                        samples=10,
                        explicit_l2_flush=False,
                        timing="NPU events and synchronized host wall time",
                    )
                ),
                flush=True,
            )
        print(
            "GRAPH_CAPACITY_PASS "
            + json.dumps(
                dict(
                    rank=rank,
                    **vars(args),
                    shmem_bytes=nbytes,
                    torch_peak_allocated=torch.npu.max_memory_allocated(),
                    torch_peak_reserved=torch.npu.max_memory_reserved(),
                )
            ),
            flush=True,
        )
    except BaseException:
        # Preserve the original failure even if device/collective teardown stalls.
        traceback.print_exc()
        raise
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
