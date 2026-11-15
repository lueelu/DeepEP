"""Global Clos multi-Buffer precision, ownership and concurrent-stream regression.

Expected tensors come from the existing Clos CPU golden, never another Buffer.
Construct/destroy collectively in a fixed order; only operations on distinct
Buffers run concurrently. Each participating Host thread selects its NPU device.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import Barrier
import weakref

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep import Buffer, ElasticBuffer
from deep_ep.backends.direct import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as golden


def wait(result):
    if result[-1].event is not None:
        result[-1].current_stream_wait(release_handle=True)
    return result


def dispatch(buffer, args, rank, delta=0.0, handle=None):
    x = golden._to_npu(golden._make_dispatch_x(rank, args, delta))
    kwargs = dict(allocate_on_comm_stream=args.allocate_on_comm_stream)
    if args.api == "v1":
        kwargs.update(config=Config(args.num_aiv), async_finish=True, zero_copy=False)
    else:
        kwargs.update(num_sms=args.num_aiv, async_with_compute_stream=True, do_expand=args.expand)
    if handle is not None:
        out = wait(buffer.dispatch(x, handle=handle, **kwargs))
        return out[0], out[1], out[2], handle
    idx = golden._make_topk_idx(rank, args).to("npu")
    weights = golden._make_weights(rank, args).to("npu")
    experts = args.ep_size * args.experts_per_rank
    kwargs.update(topk_idx=idx, topk_weights=weights)
    if args.api == "v1":
        counts, _, per_expert, membership, event = buffer.get_dispatch_layout(
            idx, experts, async_finish=True, allocate_on_comm_stream=args.allocate_on_comm_stream
        )
        kwargs.update(
            num_tokens_per_rank=counts,
            num_tokens_per_expert=per_expert,
            is_token_in_rank=membership,
            previous_event=event,
            expert_alignment=1,
        )
    else:
        kwargs.update(num_experts=experts)
    out = wait(buffer.dispatch(x, **kwargs))
    return out[0], out[1], out[2], out[4] if args.api == "v1" else out[3]


def check_dispatch(out, args, rank, delta=0.0, cached=False):
    ref = golden._dispatch_reference(rank, args, delta=delta)
    expected_x = ref["recv_x"]
    expected_weights = ref["recv_topk_weights"]
    if args.expand:
        expanded = golden._expand_reference(expected_x, ref["recv_topk_idx"], expected_weights, ref["expert_counts"])
        expected_x = expanded["expanded_x"]
        expected_weights = expanded["expanded_topk_weights"]
    if args.inject_error:
        golden._x_tensor(expected_x).view(-1)[0] += 1
    golden._assert_equal("multi-buffer dispatch", out[0], expected_x)
    if not cached:
        if args.expand:
            assert out[1] is None
        else:
            golden._assert_equal("multi-buffer topk", out[1], ref["recv_topk_idx"])
        golden._assert_equal("multi-buffer weights", out[2], expected_weights)
        layout = out[3] if args.api == "v1" else out[3]._dispatch_layout
        golden._assert_equal("multi-buffer src_idx", layout[3], ref["recv_src_idx"])
        golden._assert_equal("multi-buffer send_head", layout[5], ref["send_head"])


def combine(buffer, out, args, rank, delta=0.0):
    kwargs = dict(topk_weights=out[2], allocate_on_comm_stream=args.allocate_on_comm_stream)
    if args.api == "v1":
        kwargs.update(config=Config(args.num_aiv), async_finish=True)
    else:
        kwargs.update(num_sms=args.num_aiv, async_with_compute_stream=True)
    value = out[0]
    if isinstance(value, tuple):
        value = golden._dequantize_x(tuple(t.cpu() for t in value), golden._compute_dtype(args)).to("npu")
    result = wait(buffer.combine(value, out[3], **kwargs))
    expected, weights = golden._combine_reference(rank, args, delta=delta)
    if args.expand:
        # Identity experts: each valid expert route returns the source token.
        # Aggregate rounds each destination's sum to the payload dtype before
        # combine accumulates destinations in FP32, in ascending rank order.
        source = golden._dequantize_x(golden._make_dispatch_x(rank, args, delta), golden._compute_dtype(args))
        routes = golden._make_topk_idx(rank, args)
        total = torch.zeros_like(source, dtype=torch.float32)
        for peer in range(args.ep_size):
            local = torch.zeros_like(total)
            for route in range(args.num_topk):
                selected = (routes[:, route] >= peer * args.experts_per_rank) & (
                    routes[:, route] < (peer + 1) * args.experts_per_rank
                )
                local[selected] += source[selected].float()
            total += local.to(source.dtype).float()
        expected = total.to(source.dtype)
    golden._assert_equal("multi-buffer combine", result[0], expected)
    if out[2] is not None:
        golden._assert_equal("multi-buffer combined weights", result[1], weights)


def run_rank(local_rank, args):
    rank = args.node_rank * (args.ep_size // args.nnodes) + local_rank
    device = args.first_device + local_rank
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(device), WORLD_SIZE=str(args.ep_size))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", init_method=args.dist_init_method, rank=rank, world_size=args.ep_size)
    group = dist.group.WORLD
    small = copy(args)
    small.num_tokens = args.small_tokens
    size_hint = max(
        Config(args.num_aiv).get_nvl_buffer_size_hint(args.hidden * 2, args.ep_size, args.num_tokens),
        ElasticBuffer.get_buffer_size_hint(
            group, args.num_tokens, args.hidden, args.num_topk, machine_type="global_clos"
        ),
    )

    def create(capacity, shape):
        kw = dict(machine_type="global_clos", explicitly_destroy=True, num_max_tokens_per_rank=shape.num_tokens)
        if args.api == "v1":
            return Buffer(group, num_nvl_bytes=capacity, **kw)
        return ElasticBuffer(
            group, capacity, hidden=args.hidden, num_topk=args.num_topk, use_fp8_dispatch=args.dtype == "fp8", **kw
        )

    print(f"MULTI_BUFFER_STAGE rank={rank} pid={os.getpid()} stage=create_old", flush=True)
    old = create(size_hint, small)
    pending = dispatch(old, small, rank)
    check_dispatch(pending, small, rank)
    old_ref = weakref.ref(old)
    cached_buffer = old
    # Exact customer lifetime: new construction happens while old outputs and
    # an unconsumed combine handle remain live.
    print(f"MULTI_BUFFER_STAGE rank={rank} stage=create_new", flush=True)
    buffer = create(2 * size_hint, args)
    assert old._native.get_comm_meta_ptr() != buffer._native.get_comm_meta_ptr()
    assert old._native.get_workspace_ptr() != buffer._native.get_workspace_ptr()
    newer = dispatch(buffer, args, rank, delta=8.0)
    check_dispatch(newer, args, rank, delta=8.0)
    combine(buffer, newer, args, rank, delta=8.0)
    del old
    gc.collect()
    assert old_ref() is cached_buffer
    del cached_buffer
    gc.collect()
    assert old_ref() is not None, "unconsumed handle must retain its Buffer"
    cached_buffer = old_ref()
    check_dispatch(pending, small, rank)
    combine(cached_buffer, pending, small, rank)
    # Wrong-owner handles must be rejected before any communication is posted.
    try:
        dispatch(buffer, args, rank, handle=pending[3])
    except (ValueError, RuntimeError) as error:
        assert "handle" in str(error).lower() or "buffer" in str(error).lower(), str(error)
    else:
        raise AssertionError("foreign Buffer handle accepted")

    ready = Barrier(2)
    torch.npu.synchronize()

    def worker(owner, shape, offset):
        torch.npu.set_device(device)
        stream = torch.npu.Stream(device=device)
        with torch.npu.stream(stream):
            ready.wait(timeout=60)
            first = dispatch(owner, shape, rank, offset)
            check_dispatch(first, shape, rank, offset)
            combine(owner, first, shape, rank, offset)
            for cycle in range(args.cycles):
                delta = offset + cycle + 1
                current = dispatch(owner, shape, rank, delta, first[3])
                check_dispatch(current, shape, rank, delta, cached=True)
                combine(owner, current, shape, rank, delta)
                check_dispatch(first, shape, rank, offset)
            stream.synchronize()
        return stream.npu_stream

    print(f"MULTI_BUFFER_STAGE rank={rank} stage=concurrent", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(worker, cached_buffer, small, 16.0)
        b = pool.submit(worker, buffer, args, 32.0)
        stream_ids = [a.result(), b.result()]
    assert stream_ids[0] != stream_ids[1]
    check_dispatch(pending, small, rank)
    # Destroy either instance independently, then reuse the freed ID while the
    # other Buffer remains capable of dispatch/combine with its old handle.
    print(f"MULTI_BUFFER_STAGE rank={rank} stage=destroy_and_recreate", flush=True)
    cached_buffer.destroy()
    check_dispatch(pending, small, rank)
    replacement = create(size_hint, small)
    latest = dispatch(replacement, small, rank, 48.0)
    check_dispatch(latest, small, rank, 48.0)
    combine(replacement, latest, small, rank, 48.0)
    # newer's handle is retained across all operations on the other instance.
    after = dispatch(buffer, args, rank, 56.0, newer[3])
    check_dispatch(after, args, rank, 56.0, cached=True)
    combine(buffer, after, args, rank, 56.0)
    buffer.destroy()
    check_dispatch(newer, args, rank, 8.0)
    again = dispatch(replacement, small, rank, 64.0, latest[3])
    check_dispatch(again, small, rank, 64.0, cached=True)
    combine(replacement, again, small, rank, 64.0)
    replacement.destroy()
    check_dispatch(latest, small, rank, 48.0)
    providers = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/") and any(n in path for n in ("libshmem", "libdeepep", "/_clos_C.", "/_C.")):
            providers[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    print(
        "MULTI_BUFFER_PASSED "
        + json.dumps(
            dict(
                rank=rank,
                api=args.api,
                cycles=args.cycles,
                tokens=[small.num_tokens, args.num_tokens],
                hidden=args.hidden,
                dtype=args.dtype,
                expand=args.expand,
                capacities=[size_hint, 2 * size_hint],
                streams=stream_ids,
                libraries=providers,
            )
        ),
        flush=True,
    )
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", choices=("v1", "v2"), required=True)
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--first-device", type=int, default=0)
    p.add_argument("--small-tokens", type=int, default=17)
    p.add_argument("--num-tokens", type=int, default=129)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--num-topk", type=int, default=8)
    p.add_argument("--experts-per-rank", type=int, default=32)
    p.add_argument("--num-aiv", type=int, default=32)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp8"), default="bf16")
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--dist-init-method", default="tcp://127.0.0.1:29780")
    p.add_argument("--allocate-on-comm-stream", action="store_true")
    p.add_argument("--inject-error", action="store_true")
    p.add_argument("--expand", action="store_true", help="V2 expert-expanded dispatch and aggregate/combine")
    args = p.parse_args()
    if args.expand and args.api != "v2":
        p.error("--expand requires --api v2")
    if args.nnodes < 1 or args.ep_size < 2 or args.ep_size % args.nnodes:
        p.error("--ep-size must be divisible by positive --nnodes")
    if not 0 <= args.node_rank < args.nnodes:
        p.error("--node-rank must be in [0, nnodes)")
    local_size = args.ep_size // args.nnodes
    if args.first_device < 0 or args.first_device + local_size > 8:
        p.error("each node must use at most eight local devices")
    if args.nnodes > 1 and args.dist_init_method == "tcp://127.0.0.1:29780":
        p.error("multi-node runs require a shared, reachable --dist-init-method")
    args.expert_alignment = 1
    mp.spawn(run_rank, args=(args,), nprocs=local_size, join=True)


if __name__ == "__main__":
    main()
