"""Basic V1/V2 BF16 precision and event validation on identical routing.

Run each API in a separate process group. Expected values use the shared CPU
reference; performance measurement lives in benchmarks/clos.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from deep_ep import EventHandle, EventOverlap
from deep_ep.backends.direct import Buffer, Config, ElasticBuffer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as golden


def buffer_entrypoints(args, group):
    if not (args.public_auto or args.public_clos):
        return Buffer, ElasticBuffer, {"transport": args.transport or "udma"}
    if args.transport is not None:
        raise ValueError("--public-auto selects the engine from topology; omit --transport")
    from deep_ep import Buffer as PublicBuffer, ElasticBuffer as PublicElasticBuffer
    from deep_ep._machine import select

    if args.public_clos:
        return PublicBuffer, PublicElasticBuffer, {"machine_type": "global_clos"}
    # This golden checks the Clos layout. Fail collectively before constructing
    # a Buffer if the node belongs to another backend; never force Clos here.
    selection = select(group=group)
    if selection["machine_type"] != "global_clos":
        raise RuntimeError(f"This precision entry requires auto-detected global_clos, got {selection}")
    return PublicBuffer, PublicElasticBuffer, {}


def wait(result):
    event = result[-1]
    if event.event is not None:
        event.current_stream_wait(release_handle=True)
    return result


def check_combine_events(buffer, owned, handle, recv_weights, bias0, bias1, rank, args, config):
    """Independent producer/consumer streams, using the existing CPU golden.

    V2 bias is produced on a stream independent of the input event and guarded
    only by previous_event_before_epilogue. CPU tests separately check that the
    wait is placed after communication, without relying on race timing here.
    """
    caller = torch.npu.current_stream()
    ready = EventHandle()
    input_stream, bias_stream, consumer = (torch.npu.Stream() for _ in range(3))
    cases = 0
    for allocation in (False, True):
        for asynchronous in (False, True):
            for weighted in (False, True):
                with torch.npu.stream(input_stream):
                    ready.current_stream_wait()
                    pending_x = owned.clone()
                    pending_weights = recv_weights.clone() if weighted else None
                    input_event = EventHandle()
                with torch.npu.stream(bias_stream):
                    ready.current_stream_wait()
                    pending_bias = (bias0.clone(), bias1.clone()) if weighted else None
                    bias_event = EventHandle()
                # Caller allocation requires the implicit caller-stream input
                # event. Explicit previous_event uses comm-stream allocation.
                if not allocation:
                    input_event.current_stream_wait()
                kwargs = dict(
                    topk_weights=pending_weights,
                    bias=pending_bias,
                    previous_event=input_event if allocation else None,
                    allocate_on_comm_stream=allocation,
                )
                if args.api == "v2":
                    result = buffer.combine(
                        pending_x,
                        handle,
                        num_sms=args.num_aiv,
                        previous_event_before_epilogue=bias_event,
                        async_with_compute_stream=asynchronous,
                        **kwargs,
                    )
                else:
                    # V1 has no epilogue event: both dependencies precede the
                    # fused communication/reduce operation.
                    input_event.current_stream_wait()
                    bias_event.current_stream_wait()
                    kwargs["previous_event"] = EventOverlap(EventHandle()) if allocation else None
                    result = buffer.combine(pending_x, handle, config=config, async_finish=asynchronous, **kwargs)
                out, out_weights, completion = result
                with torch.npu.stream(consumer):
                    if asynchronous:
                        assert completion.event is not None
                        completion.current_stream_wait(release_handle=True)
                        assert completion.extra_tensors is None
                    else:
                        assert completion.event is None
                        consumer.wait_stream(caller)
                    saved_x = out.clone()
                    saved_weights = out_weights.clone() if out_weights is not None else None
                # Drop producer references before the consumer finishes, so the
                # allocator records and returned event own the queued uses.
                del pending_x, pending_weights, pending_bias, result, out, out_weights
                consumer.synchronize()
                expected, expected_weights = golden._combine_reference(
                    rank, args, bias0.cpu() if weighted else None, bias1.cpu() if weighted else None
                )
                golden._assert_equal("event combine", saved_x, expected)
                if weighted:
                    golden._assert_equal("event combine weights", saved_weights, expected_weights)
                else:
                    assert saved_weights is None
                cases += 1
    print(f"EVENT_DEPENDENCIES_PASSED api={args.api} cases={cases}", flush=True)


def run_rank(rank, args):
    device = args.first_device + rank
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(device), WORLD_SIZE=str(args.ep_size))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", init_method=args.dist_init_method, rank=rank, world_size=args.ep_size)
    config = Config(
        args.num_aiv, udma_peers_per_worker=args.udma_peers_per_worker, udma_wqes_per_visit=args.udma_wqes_per_visit
    )
    if args.auto_policy:
        config = Config(args.num_aiv)
    num_experts = args.ep_size * args.experts_per_rank
    group = dist.group.WORLD
    buffer_type, elastic_type, engine_options = buffer_entrypoints(args, group)
    # The same workspace capacity for both APIs avoids allocation-size confounds.
    capacity = max(
        config.get_nvl_buffer_size_hint(args.hidden * 2, args.ep_size, args.num_tokens),
        elastic_type.get_buffer_size_hint(
            group,
            args.num_tokens,
            args.hidden,
            args.num_topk,
            **({"machine_type": "global_clos"} if args.public_clos else {}),
        ),
    )
    common = dict(
        explicitly_destroy=True,
        shmem_ip_port=args.shmem_ip_port,
        num_max_tokens_per_rank=args.num_tokens,
        **engine_options,
    )
    manual = dict(
        udma_schedule="peer",
        udma_auto_policy=False,
        udma_dispatch_peer_order="rank",
        udma_combine_peer_order="rank",
        udma_workers_per_peer=1,
        udma_qps_per_worker=1,
    )
    if not args.auto_policy:
        common.update(manual)
    if args.api == "v1":
        buffer = buffer_type(group, num_nvl_bytes=capacity, **common)
    else:
        buffer = elastic_type(
            group,
            num_bytes=capacity,
            hidden=args.hidden,
            num_topk=args.num_topk,
            **(
                {}
                if args.auto_policy
                else dict(
                    udma_peers_per_worker=args.udma_peers_per_worker,
                    udma_wqes_per_visit=args.udma_wqes_per_visit,
                )
            ),
            **common,
        )
    if args.public_auto or args.public_clos:
        selection = buffer.machine_selection
        print(
            "MACHINE_SELECTION "
            + json.dumps(
                dict(
                    rank=rank,
                    api=args.api,
                    selection=selection,
                    transport=buffer.transport,
                    implementation=f"{type(buffer).__module__}.{type(buffer).__name__}",
                )
            ),
            flush=True,
        )
        assert selection["machine_type"] == "global_clos"
        assert (selection["source"] == "explicit") == args.public_clos
        assert buffer.transport == "udma"
    x = golden._make_x(rank, args).to("npu")
    idx_cpu = golden._make_topk_idx(rank, args)
    idx = idx_cpu.to("npu")
    weights = golden._make_weights(rank, args).to("npu")

    def dispatch(value, handle=None, padded=False):
        if args.api == "v2":
            return buffer.dispatch(
                value,
                handle=handle,
                topk_idx=idx if handle is None else None,
                topk_weights=weights if handle is None else None,
                num_experts=num_experts,
                num_sms=args.num_aiv,
                do_cpu_sync=False if padded else None,
                async_with_compute_stream=True,
                allocate_on_comm_stream=args.allocate_on_comm_stream,
            )
        kwargs = dict(
            config=config, async_finish=True, zero_copy=False, allocate_on_comm_stream=args.allocate_on_comm_stream
        )
        if handle is not None:
            out = buffer.dispatch(value, handle=handle, **kwargs)
            return out[0], out[1], out[2], handle, out[-1]
        counts, _, experts, membership, event = buffer.get_dispatch_layout(
            idx, num_experts, async_finish=True, allocate_on_comm_stream=args.allocate_on_comm_stream
        )
        out = buffer.dispatch(
            value,
            topk_idx=idx,
            topk_weights=weights,
            num_tokens_per_rank=counts,
            num_tokens_per_expert=experts,
            is_token_in_rank=membership,
            previous_event=event,
            expert_alignment=1,
            num_worst_tokens=args.num_tokens * args.ep_size if padded else 0,
            **kwargs,
        )
        return out[0], out[1], out[2], out[4], out[-1]

    def combine(value, handle, route_weights=None, bias=None):
        kwargs = dict(topk_weights=route_weights, bias=bias, allocate_on_comm_stream=args.allocate_on_comm_stream)
        if args.api == "v1":
            return buffer.combine(value, handle, config=config, async_finish=True, **kwargs)
        return buffer.combine(value, handle, num_sms=args.num_aiv, async_with_compute_stream=True, **kwargs)

    success = False
    try:
        recv, recv_idx, recv_weights, handle, _ = wait(dispatch(x, padded=args.fixed_capacity))
        ref = golden._dispatch_reference(rank, args)
        count = ref["recv_src_idx"].numel()
        if args.inject_error and count:
            ref["recv_x"].view(-1)[0] += 1
        golden._assert_equal("dispatch payload", recv[:count], ref["recv_x"])
        golden._assert_equal("dispatch topk", recv_idx[:count], ref["recv_topk_idx"])
        golden._assert_equal("dispatch weights", recv_weights[:count], ref["recv_topk_weights"])
        if args.fixed_capacity:
            assert recv.shape[0] == args.ep_size * args.num_tokens
            golden._assert_equal("padded payload", recv[count:], torch.zeros_like(recv[count:].cpu()))
            golden._assert_equal("padded routes", recv_idx[count:], torch.full_like(recv_idx[count:].cpu(), -1))
            golden._assert_equal("padded weights", recv_weights[count:], torch.zeros_like(recv_weights[count:].cpu()))
        layout = handle if args.api == "v1" else handle._dispatch_layout
        golden._assert_equal("src_idx", layout[3][:count], ref["recv_src_idx"])
        golden._assert_equal("send_head", layout[5], ref["send_head"])
        golden._assert_equal("recv_channel_prefix", layout[2], ref["recv_channel_prefix"])
        if args.fixed_capacity and count < layout[3].numel():
            # Poison unused capacity: cached dispatch and both combines must
            # select valid rows by prefixes, never by a zero/sentinel tail.
            layout[3][count:].fill_(torch.iinfo(torch.int32).min)
            torch_npu.npu.synchronize()
        owned = recv.clone()
        bias0 = torch.full_like(x, 0.5)
        bias1 = (
            (
                torch.arange(args.num_tokens * args.hidden, dtype=torch.float32)
                .remainder_(2048)
                .reshape(args.num_tokens, args.hidden)
                * 0.001
            )
            .to(torch.bfloat16)
            .to("npu")
        )
        for weighted in (False, True):
            out, out_weights, _ = wait(
                combine(owned, handle, recv_weights if weighted else None, (bias0, bias1) if weighted else None)
            )
            expected, expected_weights = golden._combine_reference(
                rank, args, bias0.cpu() if weighted else None, bias1.cpu() if weighted else None
            )
            golden._assert_equal("combine", out, expected)
            if weighted:
                golden._assert_equal("combined weights", out_weights, expected_weights)
            else:
                assert out_weights is None
        cached_x = golden._make_x(rank, args, delta=1.0).to("npu")
        cached, _, _, _, _ = wait(dispatch(cached_x, handle))
        cached_ref = golden._dispatch_reference(rank, args, delta=1.0)
        golden._assert_equal("cached payload", cached[:count], cached_ref["recv_x"])
        golden._assert_equal("original retained", recv[:count], ref["recv_x"])
        if args.check_events:
            check_combine_events(buffer, owned, handle, recv_weights, bias0, bias1, rank, args, config)
        print(
            f"rank {rank} shared backend precision passed: api={args.api}, fixed_capacity={args.fixed_capacity}",
            flush=True,
        )
        membership = golden._layout_reference(idx_cpu, args)[2]
        record = dict(
            api=args.api,
            rank=rank,
            tokens=args.num_tokens,
            hidden=args.hidden,
            topk=args.num_topk,
            seed=args.seed,
            zero_copy=False,
            num_bytes=capacity,
            peer_counts=membership.sum(0).tolist(),
            topk_sha256=hashlib.sha256(idx_cpu.numpy().tobytes()).hexdigest(),
        )
        print("MATCHED_INPUT " + json.dumps(record), flush=True)
        dist.barrier(device_ids=[device])
        buffer.destroy()
        golden._assert_equal("original retained after destroy", recv[:count], ref["recv_x"])
        success = True
    finally:
        # A failed rank must not enter an additional collective destroy. The
        # external job watchdog can terminate this exact test's remaining ranks.
        buffer = None
        if success:
            dist.destroy_process_group()
    if args.public_auto or args.public_clos:
        print(f"PUBLIC_PRECISION_PASSED api={args.api} rank={rank}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", choices=("v1", "v2"), required=True)
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--first-device", type=int, default=0)
    p.add_argument("--num-tokens", type=int, default=129)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--num-topk", type=int, default=8)
    p.add_argument("--experts-per-rank", type=int, default=32)
    p.add_argument("--num-aiv", type=int, default=64)
    p.add_argument("--udma-peers-per-worker", type=int, default=1)
    p.add_argument("--udma-wqes-per-visit", type=int, default=8)
    p.add_argument("--transport", choices=("udma", "mte"), default=None)
    public = p.add_mutually_exclusive_group()
    public.add_argument(
        "--public-auto",
        action="store_true",
        help="Use public V1/V2 with automatic topology selection; requires a global_clos machine",
    )
    public.add_argument(
        "--public-clos", action="store_true", help="Explicit public Clos backend; does not test topology detection"
    )
    p.add_argument(
        "--auto-policy", action="store_true", help="Use the default UDMA scheduling policy and only set AIV count"
    )
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--dist-init-method", default="tcp://127.0.0.1:29500")
    p.add_argument("--shmem-ip-port", default=None)
    p.add_argument("--allocate-on-comm-stream", action="store_true")
    p.add_argument("--fixed-capacity", action="store_true")
    p.add_argument("--inject-error", action="store_true")
    p.add_argument("--check-events", action="store_true")
    args = p.parse_args()
    if (args.public_auto or args.public_clos) and args.transport is not None:
        p.error("--public-auto selects the engine from topology; omit --transport")
    args.dtype, args.expert_alignment = "bf16", 1
    mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()
