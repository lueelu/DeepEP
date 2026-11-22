"""8-AIV fresh dispatch A/B, one Buffer per rank, precision before timing."""

import argparse
import csv
from dataclasses import asdict
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import time

ENV = "DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY"


def parse_capture(directory, fused, samples, expanded=False, guards=0):
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one kernel_details.csv, found {len(paths)}")
    with paths[0].open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["Start Time(us)"]))
    names = ["cached_notify_combine_kernel"]
    if not fused:
        names.append("dispatch_layout_kernel")
    names += ["notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
    if expanded:
        names.append("expand_kernel")
    known = set(names) | {"dispatch_layout_kernel", "expand_kernel"}
    selected = [r for r in rows if r["Name"] in known]
    if [r["Name"] for r in selected] != names * (samples + 2 * guards):
        raise RuntimeError("unexpected barrier/layout/notify/comm sequence or sample count")
    result = {
        name: [float(r["Duration(us)"]) for r in selected[i :: len(names)]][guards : guards + samples]
        for i, name in enumerate(names)
    }
    if any(not math.isfinite(t) or t <= 0 for values in result.values() for t in values):
        raise RuntimeError("invalid kernel duration")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--expand", type=int, choices=(0, 1), default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--control-port", type=int, default=29871)
    p.add_argument("--shmem-port", type=int, default=29872)
    p.add_argument("--huge-pages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--flush-l2", action="store_true")
    p.add_argument("--precision-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.tokens, args.hidden, args.topk, args.experts, args.samples) <= 0 or args.warmup < 0:
        p.error("shape/sample counts must be positive; warmup must be nonnegative")
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep.backends.direct.elastic import ElasticBuffer
    from deep_ep.backends.direct import buffer as bm, elastic as em
    from deep_ep.backends.direct.dtypes import topk_idx_t
    from deep_ep.backends.direct._native import load_native_module
    from bench_fast_sweep import StoreGroup
    from stage_metrics import L2CacheFlush, prepare_v2_sample

    rank, world, device = (int(os.environ[n]) for n in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world < 2 or args.experts % world or args.topk > args.experts:
        p.error("EP must be >=2, divide experts, and topk must not exceed experts")
    torch.npu.set_device(device)
    native = load_native_module()
    if getattr(native, "fused_layout_notify_version", 0) != 1:
        raise RuntimeError("install this branch's native and device libraries before initializing Buffer")
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        args.control_port,
        world,
        rank == 0,
        timeout=datetime.timedelta(seconds=600),
        wait_for_workers=True,
    )
    group = StoreGroup(store, rank, world)
    bm._bootstrap_all_gather = lambda g, c, n, value: group.gather(value)
    em._validate_group_consensus = lambda g, name, value: group.agree([name, value])
    group.agree(vars(args) | {"output": None})
    out = args.output / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                world=world,
                native_path=native.__file__,
                torch_version=torch.__version__,
                torch_npu_version=torch_npu.__version__,
                args=vars(args) | {"output": str(args.output)},
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = []
    old_env = os.environ.get(ENV)
    buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.topk,
        explicitly_destroy=True,
        transport="udma",
        udma_auto_policy=True,
        use_1g_huge_pages=args.huge_pages,
        shmem_ip_port=f"tcp://{os.environ['MASTER_ADDR']}:{args.shmem_port}",
    )
    if rank == 0:
        print(f"BUFFER_READY EP={world} AIV=8 initializations=1", flush=True)
    flush = L2CacheFlush(torch, buffer, args.flush_l2)

    def done(result):
        if result[-1].event is not None:
            result[-1].current_stream_wait()
        torch.npu.synchronize()
        return result

    def assert_equal(a, b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def snapshot(result, local_x, local_idx, all_idx):
        recv, recv_idx, recv_w, h, _ = done(result)
        layout = [t.cpu().clone() for t in h._dispatch_layout]
        memberships = []
        for idx in all_idx:
            valid = idx >= 0
            owners = idx.clamp_min(0) // (args.experts // world)
            memberships.append(torch.stack([(valid & (owners == peer)).any(1) for peer in range(world)], 1))
        assert_equal(layout[4], memberships[rank])
        counts = torch.stack([m.sum(0) for m in memberships]).to(torch.int32)
        assert_equal(layout[0], counts.cumsum(0).to(torch.int32))
        experts = sum((torch.bincount(idx[idx >= 0], minlength=args.experts) for idx in all_idx))
        e = args.experts // world
        assert_equal(h.num_unaligned_recv_tokens_per_expert.cpu(), experts[rank * e : (rank + 1) * e].to(torch.int32))
        length = (local_idx.shape[0] + 7) // 8
        plan = h._dispatch_plan.cpu()
        stable = []
        for peer in range(world):
            total = 0
            for ch in range(8):
                begin, end = ch * length, min((ch + 1) * length, local_idx.shape[0])
                indices = torch.where(memberships[rank][begin:end, peer])[0] + begin
                total += len(indices)
                assert int(layout[1][peer, ch]) == total
                assert_equal(plan[peer, ch, : len(indices)], indices.to(torch.int32))
                if len(indices):
                    assert_equal(layout[5][indices, peer], torch.arange(len(indices), dtype=torch.int32))
                stable.append(indices)
        assert bool((layout[5][~memberships[rank]] == -1).all())
        outputs = [recv.cpu().clone()]
        if recv_idx is not None:
            outputs.append(recv_idx.cpu().clone())
        if recv_w is not None:
            outputs.append(
                recv_w.cpu().masked_fill(recv_idx.cpu() < 0, 0) if recv_idx is not None else recv_w.cpu().clone()
            )
        outputs += layout + stable
        # Exercise the handle in cached dispatch and both combine weight modes.
        cached = done(buffer.dispatch(local_x, handle=h, num_sms=8, do_expand=bool(args.expand)))
        assert_equal(cached[0], recv)
        for weighted in (False, True):
            w = recv_w if weighted else None
            combined = done(buffer.combine(recv, h, topk_weights=w, num_sms=8))
            outputs.append(combined[0].cpu().clone())
            if combined[1] is not None:
                outputs.append(combined[1].cpu().clone())
        return outputs

    try:
        gen = torch.Generator().manual_seed(20261122 + rank)
        x = (torch.randn((args.tokens, args.hidden), generator=gen) * 0.01).to(torch.bfloat16).npu()
        raw = torch.rand((args.tokens, args.experts), generator=gen).topk(args.topk, dim=1).indices
        weights = torch.rand((args.tokens, args.topk), generator=gen).npu()
        for routing in ("random", "masked", "hot", "empty", "empty_local"):
            if rank == 0:
                print(f"PRECISION routing={routing} expand={args.expand} order=0,1,1,0", flush=True)
            cpu_idx = raw.clone()
            if routing == "masked":
                cpu_idx[::3, -1] = -1
                cpu_idx[::7] = -1
            elif routing == "hot":
                cpu_idx[:] = torch.arange(args.topk)  # empty destination peers, repeated destination rank
            elif routing == "empty":
                cpu_idx.fill_(-1)
            elif routing == "empty_local" and rank == 0:
                cpu_idx = cpu_idx[:0]
            local_x, local_weights = x[: len(cpu_idx)], weights[: len(cpu_idx)]
            all_idx = [
                torch.tensor(v, dtype=torch.int64).reshape(-1, args.topk) for v in group.gather(cpu_idx.tolist())
            ]
            idx = cpu_idx.to(dtype=topk_idx_t, device="npu")
            for weighted in (False, True):
                baseline = None
                for mode in (0, 1, 1, 0):
                    torch.npu.synchronize()
                    group.agree([routing, weighted, mode])
                    os.environ[ENV] = str(mode)
                    result = buffer.dispatch(
                        local_x,
                        topk_idx=idx,
                        topk_weights=local_weights if weighted else None,
                        num_experts=args.experts,
                        num_sms=8,
                        do_expand=bool(args.expand),
                    )
                    actual = snapshot(result, local_x, cpu_idx, all_idx)
                    if baseline is None:
                        baseline = actual
                    else:
                        assert len(actual) == len(baseline)
                        for a, b in zip(actual, baseline):
                            assert_equal(a, b)
        group.agree("precision_passed")
        (out / "precision.json").write_text(json.dumps({"passed": True, "buffer_initializations": 1}), encoding="utf-8")
        if not args.precision_only:
            idx = raw.to(dtype=topk_idx_t, device="npu")
            owners = raw // (args.experts // world)
            local_rows = sum(int((owners == peer).any(1).sum()) for peer in range(world))
            mean_payload_bytes = sum(group.gather(local_rows)) * args.hidden * 2 / world
            for weighted in (False, True):
                reference_policy = None

                def invoke():
                    return done(
                        buffer.dispatch(
                            x,
                            topk_idx=idx,
                            topk_weights=weights if weighted else None,
                            num_experts=args.experts,
                            num_sms=8,
                            do_expand=bool(args.expand),
                        )
                    )

                for turn, mode in enumerate((0, 1, 1, 0)):
                    if rank == 0:
                        print(f"PERFORMANCE weights={weighted} round={turn} fused={mode}", flush=True)
                    torch.npu.synchronize()
                    group.agree(["performance", weighted, turn, mode])
                    os.environ[ENV] = str(mode)
                    check = invoke()
                    policy = asdict(check[3]._udma_policy)
                    group.agree(policy)
                    if reference_policy is not None and policy != reference_policy:
                        raise RuntimeError("dispatch policy changed during the A/B comparison")
                    reference_policy = policy
                    for _ in range(args.warmup):
                        prepare_v2_sample(buffer, flush)
                        invoke()
                    api = []
                    for _ in range(args.samples):
                        prepare_v2_sample(buffer, flush)
                        start = time.perf_counter_ns()
                        invoke()
                        api.append((time.perf_counter_ns() - start) / 1000)
                    capture = out / f"weight{int(weighted)}_round{turn}_fused{mode}"
                    prof = torch_npu.profiler
                    with prof.profile(
                        activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                        on_trace_ready=prof.tensorboard_trace_handler(str(capture)),
                        experimental_config=prof._ExperimentalConfig(
                            export_type=[prof.ExportType.Text],
                            profiler_level=prof.ProfilerLevel.Level0,
                            aic_metrics=prof.AiCMetrics.AiCoreNone,
                        ),
                    ):
                        for _ in range(args.samples + 4):
                            prepare_v2_sample(buffer, flush)
                            invoke()
                    stages = parse_capture(capture, mode, args.samples, bool(args.expand), guards=2)
                    stages["api_wall_us_unprofiled"] = api
                    stages["layout_plus_notify"] = [
                        a + b
                        for a, b in zip(
                            stages.get("dispatch_layout_kernel", [0] * args.samples), stages["notify_dispatch_kernel"]
                        )
                    ]
                    gathered = group.gather(stages)
                    summary = {
                        key: statistics.mean(max(g[key][i] for g in gathered) for i in range(args.samples))
                        for key in stages
                    }
                    rows.append(
                        dict(
                            weighted=weighted,
                            turn=turn,
                            fused=mode,
                            means_us=summary,
                            policy=policy,
                            logical_payload_bytes_per_rank=mean_payload_bytes,
                            payload_gbps=mean_payload_bytes / summary["v1_dispatch_udma_kernel"] / 1000,
                        )
                    )
                    (capture / "samples.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
        (out / "report.json").write_text(
            json.dumps(
                dict(
                    args=vars(args) | {"output": str(args.output)},
                    precision_passed=True,
                    buffer_initializations=1,
                    aiv=8,
                    results=rows,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [
                "weighted",
                "turn",
                "fused",
                "layout_us",
                "notify_us",
                "layout_notify_us",
                "comm_us",
                "expand_us",
                "api_wall_us",
                "payload_gbps",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                s = row["means_us"]
                writer.writerow(
                    dict(
                        weighted=row["weighted"],
                        turn=row["turn"],
                        fused=row["fused"],
                        layout_us=s.get("dispatch_layout_kernel", 0),
                        notify_us=s["notify_dispatch_kernel"],
                        layout_notify_us=s["layout_plus_notify"],
                        comm_us=s["v1_dispatch_udma_kernel"],
                        expand_us=s.get("expand_kernel", 0),
                        api_wall_us=s["api_wall_us_unprofiled"],
                        payload_gbps=row["payload_gbps"],
                    )
                )
    finally:
        buffer.destroy()
        if old_env is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old_env
    group.agree("complete")
    (out / "exit.json").write_text('{"exit_code": 0}', encoding="utf-8")


if __name__ == "__main__":
    main()
