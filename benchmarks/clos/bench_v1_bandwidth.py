"""One-command-per-node V1 Buffer precision + kernel/stage bandwidth benchmark.

python benchmarks/clos/bench_v1_bandwidth.py --output ./bandwidth_v1_results

Shares EP8/EP16 orchestration, input routing and byte accounting with V2.
Uses only public V1 Buffer operations. Fused combine transfer/reduce times are
not separable: report fused_combine_us, never fabricate a communication timer.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

import bench_v2_bandwidth as shared
from bandwidth_multinode import library_identity
from route_workloads import statistics as route_statistics
from stage_metrics import byte_ledger, file_identity, route_counts, write_json

REPO = shared.REPO


def arguments(argv=None):
    args = shared.arguments(argv, description=__doc__)
    args.api = "v1"
    return args


class V1Operations:
    """Public V1 calls, keeping intermediates alive through the final event.

    Fresh timing includes get_dispatch_layout, matching V2 fresh dispatch scope.
    Expanded combine carries weights as expanded_values; topk_weights on
    Buffer.aggregate would multiply payload, which is a different workload.
    """

    def __init__(self, buffer, config, idx, weights, num_experts, expanded):
        self.buffer, self.config = buffer, config
        self.idx, self.weights = idx, weights
        self.num_experts, self.expanded = num_experts, expanded

    def dispatch(self, x, handles=None):
        options = dict(config=self.config, async_finish=True, zero_copy=False, allocate_on_comm_stream=False)
        layout = None
        if handles is None:
            layout = self.buffer.get_dispatch_layout(self.idx, self.num_experts, async_finish=True)
            dispatched = self.buffer.dispatch(
                x,
                num_tokens_per_rank=layout[0],
                num_tokens_per_expert=layout[2],
                is_token_in_rank=layout[3],
                topk_idx=self.idx,
                topk_weights=self.weights,
                expert_alignment=1,
                previous_event=layout[-1],
                **options,
            )
        else:
            dispatched = self.buffer.dispatch(x, handle=handles[0], **options)
        expanded = None
        if self.expanded:
            options = dict(
                config=self.config, async_finish=True, allocate_on_comm_stream=False, previous_event=dispatched[-1]
            )
            if handles is None:
                expanded = self.buffer.expand(
                    dispatched[0],
                    recv_topk_idx=dispatched[1],
                    topk_weights=dispatched[2],
                    num_tokens_per_expert=dispatched[3],
                    expert_alignment=1,
                    **options,
                )
            else:
                expanded = self.buffer.expand(dispatched[0], handle=handles[1], **options)
        return dispatched, expanded, layout, (expanded if self.expanded else dispatched)[-1]

    def combine(self, x, handles, weights=None, bias=None):
        aggregated = None
        options = dict(config=self.config, async_finish=True, allocate_on_comm_stream=False)
        if self.expanded:
            aggregated = self.buffer.aggregate(x, handles[1], expanded_values=weights, **options)
            x, weights = aggregated[:2]
            options["previous_event"] = aggregated[-1]
        combined = self.buffer.combine(x, handles[0], topk_weights=weights, bias=bias, **options)
        return aggregated, combined, combined[-1]


def worker(args):
    # --help and --report-only must work without torch, torch_npu or a device.
    sys.path.insert(0, str(REPO / "tests/clos/source"))
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from deep_ep import Buffer, Config, topk_idx_t
    import test_v1_intranode as golden

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if (
        int(os.environ["WORLD_SIZE"]) != args.ep_size
        or int(os.environ["LOCAL_WORLD_SIZE"]) != shared.local_world_size(args)
        or not 0 <= local_rank < shared.local_world_size(args)
        or rank != args.node_rank * shared.local_world_size(args) + local_rank
    ):
        raise RuntimeError("torchrun local/global rank mapping does not match this node")
    args.num_aiv, args.dtype, expanded = args.aiv[0], args.dtypes[0], bool(args.expand[0])
    args.expert_alignment, args.zero_copy = 1, False
    args.profile_warmup, args.profile_iters = args.warmup, args.iterations
    args.profile_sample_markers = True
    device = args.first_device + local_rank
    destination = args.output / f"rank{rank}"
    destination.mkdir(parents=True, exist_ok=False)
    write_json(
        destination / "process.json",
        dict(pid=os.getpid(), rank=rank, device=device, node_rank=args.node_rank, started=time.time()),
    )
    try:
        torch.set_num_threads(1)
        routes = [golden._make_topk_idx(p, args) for p in range(args.ep_size)]
        counts = route_counts([r.tolist() for r in routes], args.experts_per_rank)
        write_json(
            destination / "route_load.json",
            route_statistics([r.tolist() for r in routes], args.experts_per_rank, args.route_group_size),
        )
        write_json(destination / "input_routes.json", routes[rank].tolist())
        torch.npu.set_device(device)
        dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=args.rendezvous_timeout))
        config = Config(
            args.num_aiv, udma_peers_per_worker=args.peers_per_worker, udma_wqes_per_visit=args.wqes_per_visit
        )
        # Dispatch FP8 is smaller than the BF16 combine workspace; reserve for both.
        args.num_bytes = config.get_nvl_buffer_size_hint(args.hidden * 2, args.ep_size, num_tokens=args.num_tokens)
        buffer = Buffer(
            dist.group.WORLD,
            num_nvl_bytes=args.num_bytes,
            explicitly_destroy=True,
            transport="udma",
            udma_auto_policy=False,
            udma_schedule=args.udma_schedule,
            use_1g_huge_pages=args.huge_pages,
            udma_dispatch_peer_order=args.dispatch_peer_order,
            udma_combine_peer_order=args.combine_peer_order,
            udma_ranks_per_group=args.ranks_per_group,
            udma_peers_per_group_per_worker=args.peers_per_group_per_worker,
            udma_workers_per_peer=args.workers_per_peer,
            udma_qps_per_worker=args.qps_per_worker,
        )
        args.effective_udma_config = shared.effective_udma_config(buffer, config)
        if rank == 0:
            print("EFFECTIVE_UDMA_CONFIG", json.dumps(args.effective_udma_config, sort_keys=True), flush=True)
        identity = shared.loaded_identity()
        identities = [None] * args.ep_size
        dist.all_gather_object(identities, library_identity(identity))
        if any(item != identities[0] for item in identities):
            raise RuntimeError("ranks loaded different library bytes or Python modules")
        idx = routes[rank].to("npu")
        x = golden._to_npu(golden._make_dispatch_x(rank, args))
        cached_x = golden._to_npu(golden._make_dispatch_x(rank, args, delta=1))
        weights = golden._make_weights(rank, args).to("npu")
        x_parts = x if isinstance(x, tuple) else (x,)
        input_bytes = sum(t.numel() * t.element_size() for t in (*x_parts, idx, weights))
        ledger = byte_ledger(
            counts,
            rank,
            tokens=args.num_tokens,
            hidden=args.hidden,
            topk=args.num_topk,
            dtype=args.dtype,
            idx_bytes=idx.element_size(),
            op="dispatch",
        )
        if input_bytes != args.num_tokens * ledger["official_logical_bytes_per_row"]:
            raise ValueError("logical row-byte formula does not match actual dispatch input tensors")
        operations = V1Operations(buffer, config, idx, weights, args.ep_size * args.experts_per_rank, expanded)

        def wait(result):
            golden._consume_profile_result(result)
            return result

        def check_dispatch(result, delta=0, cached=False, check_storage=True):
            dispatched, expansion = result[:2]
            ref = golden._dispatch_reference(rank, args, delta=delta)
            golden._assert_equal("dispatch payload/scales", dispatched[0], ref["recv_x"])
            if check_storage:
                golden._assert_recv_storage("dispatch payload", dispatched[0], buffer, False)
            if not cached:
                golden._assert_equal("dispatch indices", dispatched[1], ref["recv_topk_idx"])
                golden._assert_equal("dispatch weights", dispatched[2], ref["recv_topk_weights"])
                golden._assert_equal(
                    "expert counts", torch.tensor(dispatched[3], dtype=torch.int32), ref["expert_counts"]
                )
                for name, value, wanted in [
                    ("rank prefix", dispatched[4][0], ref["rank_prefix"]),
                    ("local prefix", dispatched[4][1], ref["local_channel_prefix"]),
                    ("recv prefix", dispatched[4][2], ref["recv_channel_prefix"]),
                    ("src_idx", dispatched[4][3], ref["recv_src_idx"]),
                    ("send_head", dispatched[4][5], ref["send_head"]),
                ]:
                    golden._assert_equal(name, value, wanted)
            elif any(value is not None for value in dispatched[1:5]):
                raise AssertionError("payload-only cached dispatch returned metadata")
            raw_x = dispatched[0][0] if isinstance(dispatched[0], tuple) else dispatched[0]
            if raw_x.size(0) != sum(counts["rows"][p][rank] for p in range(args.ep_size)):
                raise AssertionError("independent route count disagrees with actual receive rows")
            expected = ref["recv_x"]
            if expanded:
                exp = golden._expand_reference(
                    ref["recv_x"], ref["recv_topk_idx"], ref["recv_topk_weights"], ref["expert_counts"]
                )
                expected = exp["expanded_x"]
                golden._assert_equal("expanded payload/scales", expansion[0], expected)
                eh = expansion[2]
                for name, value, wanted in [
                    ("offsets", eh.expert_offsets, exp["expert_offsets"]),
                    ("expert counts", eh.actual_expert_counts, exp["actual_expert_counts"]),
                    ("expand mapping", eh.recv_to_expanded_idx, exp["recv_to_expanded_idx"]),
                ]:
                    golden._assert_equal(name, value, wanted)
                if not cached:
                    golden._assert_equal("expanded weights", expansion[1], exp["expanded_topk_weights"])
                elif expansion[1] is not None:
                    raise AssertionError("payload-only cached expand returned weights")
            return golden._dequantize_x(expected, torch.bfloat16)

        fresh = wait(operations.dispatch(x))
        expert_ref = check_dispatch(fresh)
        dispatched, expansion = fresh[:2]
        handles = (dispatched[4], expansion[2] if expanded else None)
        recv = expansion[0] if expanded else dispatched[0]
        recv_weights = expansion[1] if expanded else dispatched[2]
        owned = golden._dequantize_x(recv, torch.bfloat16).clone()
        golden._assert_equal("expert input", owned, expert_ref)
        del expert_ref
        cached = wait(operations.dispatch(cached_x, handles))
        check_dispatch(cached, delta=1, cached=True)
        del cached
        bias0 = bias1 = bias = None
        if args.combine_bias == "two":
            bias0 = torch.full((args.num_tokens, args.hidden), 0.5, dtype=torch.bfloat16)
            bias1 = (
                torch.arange(args.num_tokens * args.hidden, dtype=torch.float32)
                .remainder_(2048)
                .reshape(args.num_tokens, args.hidden)
                * 0.001
            ).to(torch.bfloat16)
            bias = (bias0.to("npu"), bias1.to("npu"))

        def combine(weighted):
            return operations.combine(owned, handles, recv_weights if weighted else None, bias if weighted else None)

        # Independent FP32 accumulation with BF16 rounding at each aggregate boundary.
        source = golden._dequantize_x(golden._make_dispatch_x(rank, args), torch.bfloat16)
        for weighted in (False, True):
            result = wait(combine(weighted))[1]
            expected = torch.zeros_like(source, dtype=torch.float32)
            if weighted and bias is not None:
                expected += bias0.float()
                expected += bias1.float()
            for peer in range(args.ep_size):
                mask = (routes[rank] >= peer * args.experts_per_rank) & (
                    routes[rank] < (peer + 1) * args.experts_per_rank
                )
                if expanded:
                    partial = torch.zeros_like(source, dtype=torch.float32)
                    for k in range(args.num_topk):
                        partial[mask[:, k]] += source[mask[:, k]].float()
                    expected += partial.to(torch.bfloat16).float()
                else:
                    rows = mask.any(1)
                    expected[rows] += source[rows].float()
            golden._assert_equal("combine payload", result[0], expected.to(torch.bfloat16))
            if weighted:
                expected_weights = golden._make_weights(rank, args).clone()
                expected_weights[routes[rank] == -1] = 0
                golden._assert_equal("combine weights", result[1], expected_weights)
            elif result[1] is not None:
                raise AssertionError("unweighted combine returned weights")
            del result, expected
        del source, bias0, bias1
        torch.npu.synchronize()
        write_json(
            destination / "precision.json",
            dict(
                passed=True,
                fresh=True,
                cached=True,
                expanded=expanded,
                combine_unweighted=True,
                combine_weighted=True,
                combine_bias=args.combine_bias,
                combine_weighted_two_bias=args.combine_bias == "two",
            ),
        )
        print(f"PRECISION_PASSED rank={rank} api=v1", flush=True)
        operators = {}
        flush_cache = shared.L2CacheFlush(torch, buffer, args.flush_l2_cache)
        functions = dict(
            dispatch=lambda: operations.dispatch(x),
            **{"cached-dispatch": lambda: operations.dispatch(cached_x, handles)},
            combine_unweighted=lambda: combine(False),
            combine_weighted=lambda: combine(True),
        )
        for op, fn in functions.items():
            dist.barrier()
            print(f"PROFILE_BEGIN rank={rank} op={op}", flush=True)
            args.profile_dir = str(destination / op / "profiler")
            try:
                golden._profile_operator(
                    rank,
                    args,
                    op,
                    flush_cache.wrap(fn),
                    sample_callback=shared.ProfileSamples(torch),
                    guard_iters=shared.PROFILE_GUARD_ITERS,
                )
            except golden._ProfileComplete:
                pass
            profile = shared.parse_profile_collective(
                dist,
                destination / op / "profile_error.json",
                args.profile_dir,
                op,
                expanded,
                args.iterations,
                device,
                args.num_aiv,
                api="v1",
                flush_l2_cache=args.flush_l2_cache,
                sample_markers=True,
            )
            dist.barrier()
            for _ in range(args.warmup):
                flush_cache()
                wait(fn())
            torch.npu.synchronize()
            caller = []
            for _ in range(args.iterations):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                flush_cache()
                start.record()
                result = wait(fn())
                end.record()
                end.synchronize()
                elapsed = float(start.elapsed_time(end)) * 1000
                if not 0 < elapsed < float("inf"):
                    raise ValueError("invalid caller stream duration")
                caller.append(elapsed)
                del result
            operators[op] = dict(
                profile=profile,
                caller_us=caller,
                bytes=byte_ledger(
                    counts,
                    rank,
                    tokens=args.num_tokens,
                    hidden=args.hidden,
                    topk=args.num_topk,
                    dtype=args.dtype,
                    idx_bytes=torch.empty((), dtype=topk_idx_t).element_size(),
                    op=op,
                ),
            )
            write_json(destination / op / "measurement.json", operators[op])
        dist.barrier()
        if library_identity(shared.loaded_identity()) != library_identity(identity):
            raise RuntimeError("loaded libraries changed during benchmark")
        buffer.destroy()
        torch.npu.synchronize()
        # GM-owned outputs and handles survive reuse and explicit Buffer teardown.
        check_dispatch(fresh, check_storage=False)
        write_json(
            destination / "result.json",
            dict(
                rank=rank,
                device=device,
                precision_passed=True,
                destroyed=True,
                retained_after_destroy=True,
                route_counts=counts,
                identity=identity,
                operators=operators,
                input_routes=file_identity(destination / "input_routes.json"),
                config={
                    k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")
                },
            ),
        )
        dist.destroy_process_group()
        write_json(destination / "exit.json", dict(code=0, ended=time.time()))
        print(f"CASE_PASSED rank={rank} api=v1", flush=True)
    except BaseException:
        write_json(destination / "exit.json", dict(code=1, ended=time.time(), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    options = arguments()
    if options.report_only:
        shared.report(options.report_only)
    elif options.worker:
        worker(options)
    else:
        shared.launch(options, entrypoint=__file__)
