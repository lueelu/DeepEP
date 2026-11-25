"""Staged rank search implementation for bench_fast_sweep --small-rank-sweep.

No separate CLI. Candidate generation/statistics remain importable without Torch.
"""

import csv
from dataclasses import replace
import datetime
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

OPS = ("fresh", "fresh_weighted", "cached", "cached_weighted", "combine", "combine_weighted")
VISITS = (4, 8, 16, 32, 64, 128)
SUPPORTED_EPS = (4, 8, 16, 32)


def validate_args(parser, args, argv):
    argv = sys.argv[1:] if argv is None else argv
    if args.ep_size not in SUPPORTED_EPS:
        parser.error("small-rank-sweep requires EP4/8/16/32")
    if args.case or args.strategies or args.replay_report or args.quick or args.smoke:
        parser.error("small-rank-sweep owns its case/candidate matrix; do not supply other selectors")
    if args.combine_self_copy_modes or args.combine_parallel_self_copy or args.combine_self_copy_after_submit:
        parser.error("small-rank-sweep preserves production self-copy behavior")
    explicit = {arg.split("=", 1)[0] for arg in argv}
    if "--huge-pages" in explicit or args.flush_l2_cache or "--visits" in explicit or args.experts != 256:
        parser.error("small-rank-sweep fixes no huge pages/no L2 flush/experts256/all six visit values")
    if "--seed" in explicit or "--ranks-per-group" in explicit:
        parser.error("small-rank-sweep uses the independent golden seed and discovers real topology")
    if args.reduce_cores < 2 or args.reduce_cores > 64 or args.reduce_cores % 2:
        parser.error("reduce-cores must be even and in [2,64]")
    args.huge_pages = False
    args.seed = 20261122
    args.case = cases()


def cases():
    # Representatives first so their winners can be transferred without rebuilding Buffer.
    representative = [(8192, h, a) for h in (2816, 6144) for a in (8, 32)]
    remaining = [
        (t, h, a) for t in (4096, 8192) for h in (2816, 4096, 6144) for a in (8, 32) if (t, h, a) not in representative
    ]
    return [dict(tokens=t, hidden=h, topk=8, aiv=a) for t, h, a in representative + remaining]


def candidate(p, w, q, visit, ep):
    return dict(
        id=f"rank_p{p}_w{w}_q{q}_v{visit}",
        order="rank",
        peers=p,
        workers=w,
        qps=q,
        visit=visit,
        payload_aivs=math.ceil((ep - 1) / p) * w,
    )


def shortlist(ep, cores, visits=(16, 32)):
    if ep not in SUPPORTED_EPS or cores not in (8, 32):
        raise ValueError("rank shortlist requires EP4/8/16/32 and 8/32 communication cores")
    # Compare single worker, dual QP and dual worker. Use the smallest legal P
    # for each, respecting both the AIV and physical QP-slot budgets. A 32-core
    # launch can use up to all 32 senders; there is no eight-sender restriction.
    triples = []
    for w, q in ((1, 1), (1, 2), (2, 1)):
        p = next(p for p in (1, 2, 4, 8) if math.ceil((ep - 1) / p) * w <= min(cores, 32 // q))
        triples.append((p, w, q))
    return [candidate(p, w, q, v, ep) for p, w, q in triples for v in visits]


def unique(configs):
    return list({c["id"]: c for c in configs}.values())


def transfer_candidate(config, ep):
    # Even an auto winner must be frozen: the held-out shape may select a different auto plan.
    return candidate(config["peers"], config["workers"], config["qps"], config["visit"], ep)


def policy_config(plan, ep, auto=False):
    if plan.peer_order != 0:
        raise ValueError("default policy is not rank; refusing to silently change the baseline")
    c = candidate(plan.peers_per_worker, plan.workers_per_peer, plan.qps_per_worker, plan.wqes_per_visit, ep)
    return c | {"id": "auto", "auto": True} if auto else c


def op_kind(op):
    return "weighted" if op == "combine_weighted" else "combine" if op == "combine" else "dispatch"


class SweepPolicy:
    """Override only scheduling fields; preserve the installed auto policy's other choices."""

    def __init__(self, base):
        self.base, self.config, self.operation = base, None, None

    def for_launch(self, num_aiv, row_bytes, operation, **kwargs):
        plan = self.base.for_launch(num_aiv, row_bytes, operation, **kwargs)
        c = self.config
        if c is not None and not c.get("auto") and operation == self.operation:
            plan = replace(
                plan,
                peer_order=0,
                peers_per_worker=c["peers"],
                workers_per_peer=c["workers"],
                qps_per_worker=c["qps"],
                wqes_per_visit=c["visit"],
            )
        plan.validate_num_aiv(num_aiv)
        return plan


def stats(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("invalid samples")
    ordered = sorted(values)
    return dict(
        mean=statistics.mean(values),
        p50=statistics.median(values),
        p95=ordered[math.ceil(len(values) * 0.95) - 1],
        maximum=max(values),
    )


def slowest(gathered):
    keys = set(gathered[0])
    if any(set(row) != keys for row in gathered):
        raise ValueError("rank stage mismatch")
    result = {}
    for key in keys:
        arrays = [row[key] for row in gathered]
        if not arrays[0] or len({len(v) for v in arrays}) != 1:
            raise ValueError("rank sample count mismatch")
        result[key] = [max(values) for values in zip(*arrays)]
    return result


def finalists(rows, count):
    # Include both objectives, rather than pruning an API winner by bandwidth alone.
    api = sorted(rows, key=lambda r: r["metrics"]["api"]["mean"])
    comm = sorted(rows, key=lambda r: r["metrics"]["comm"]["mean"])
    selected = unique([r["config"] for pair in zip(api, comm) for r in pair if not r["config"].get("auto")])[:count]
    return selected + [next(r["config"] for r in rows if r["config"].get("auto"))]


def parse_profile(directory, op, configs, warmup, samples):
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one kernel_details.csv, found {len(paths)}")
    with paths[0].open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["Start Time(us)"]))
    barrier = "cached_notify_combine_kernel"
    if op.startswith("combine"):
        sequence = [barrier, barrier, "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"]
        labels = [None, "notify", "comm", "reduce"]
    else:
        sequence, labels = [barrier], [None]
        if op.startswith("fresh"):
            layout = (
                "dispatch_layout_simt_kernel"
                if any(r["Name"] == "dispatch_layout_simt_kernel" for r in rows)
                else "dispatch_layout_kernel"
            )
            sequence += [layout, "notify_dispatch_kernel"]
            labels += ["layout", "notify"]
        else:
            sequence += ["cached_notify_dispatch_kernel"]
            labels += ["notify"]
        sequence += ["v1_dispatch_udma_kernel"]
        labels += ["comm"]
    recognized = set(sequence) | {
        "cached_dispatch_plan_kernel",
        "cached_notify_dispatch_packed_kernel",
        "notify_dispatch_kernel",
        "dispatch_layout_kernel",
        "dispatch_layout_simt_kernel",
    }
    selected = [r for r in rows if r["Name"] in recognized]
    width = warmup + samples + 2
    if [r["Name"] for r in selected] != sequence * (len(configs) * width + 4):
        raise RuntimeError("unexpected kernel sequence; retain capture, do not infer missing stages")
    output = []
    for i in range(len(configs)):
        start = 2 + i * width + warmup
        stages = {
            k: [float(selected[j * len(sequence) + offset]["Duration(us)"]) for j in range(start, start + samples)]
            for offset, k in enumerate(labels)
            if k
        }
        if any(not math.isfinite(v) or v <= 0 for values in stages.values() for v in values):
            raise ValueError("invalid device timing")
        output.append({k: stages.get(k, [0.0] * samples) for k in ("layout", "notify", "comm", "reduce")})
    return output


def require_native(native):
    for name in (
        "last_dispatch_channel_metadata",
        "last_dispatch_metadata_options",
        "last_combine_launch",
        "last_dispatch_launch",
        "last_dispatch_overlap_options",
    ):
        if not callable(getattr(native, name, None)):
            raise RuntimeError(f"loaded extension lacks {name}; rebuild and install this revision")


def check_launch(native, result, plan, op, cores, reduce):
    if op.startswith("combine"):
        actual = list(native.last_combine_launch())
        expected = [
            0,
            plan.peers_per_worker,
            plan.workers_per_peer,
            plan.qps_per_worker,
            plan.wqes_per_visit,
            cores,
            reduce,
            int(op.endswith("weighted")),
            1,
        ]
        if actual != expected:
            raise RuntimeError(f"combine native launch mismatch: {actual} != {expected}")
        return dict(combine_launch=actual)
    # Cached calls reuse the original handle; its stored policy describes handle creation,
    # not this send. Observe the actual Host tiling instead.
    actual = list(native.last_dispatch_launch())
    expected = [0, plan.peers_per_worker, plan.workers_per_peer, plan.qps_per_worker, plan.wqes_per_visit, cores, 1]
    if actual != expected:
        raise RuntimeError("dispatch submitted policy differs from candidate")
    channel = bool(native.last_dispatch_channel_metadata())
    options = int(native.last_dispatch_metadata_options())
    expected_channel = op.startswith("fresh") or op == "cached_weighted"
    overlap = int(native.last_dispatch_overlap_options())
    if channel != expected_channel or (op == "cached_weighted" and options != 7):
        raise RuntimeError("dispatch did not use the requested continuous channel metadata path")
    if op.startswith("fresh") and (not options & 2 or overlap & 12 != 12):
        raise RuntimeError("fresh direct/skip-init/double-buffer defaults are not active")
    return dict(dispatch_launch=actual, channel_metadata=channel, metadata_options=options, overlap_options=overlap)


def measure(session, shape, op, configs, invoke, logical_bytes, phase, samples):
    from stage_metrics import profile_combine_notify

    s = session
    prof = s.torch_npu.profiler
    tag = f"t{shape['tokens']}_h{shape['hidden']}_a{shape['aiv']}"
    directory = s.out / tag / op / phase
    s.write(directory / "manifest.json", dict(configs=configs, samples=samples, warmup=s.args.warmup))
    s.group.agree([tag, op, phase, configs, samples])
    if s.rank == 0:
        print(f"PHASE {tag}/{op}/{phase} candidates={len(configs)}", flush=True)
    api, observations = [], []
    with profile_combine_notify(False):
        for config in configs:
            s.activate(config, op)
            s.prepare()
            result = s.done(invoke())
            plan = s.policy.for_launch(shape["aiv"], shape["hidden"] * 2, op_kind(op), bf16=True, expanded=False)
            observation = check_launch(s.native, result, plan, op, shape["aiv"], s.args.reduce_cores)
            observations.append(observation)
            del result
            for _ in range(s.args.warmup):
                s.prepare()
                s.done(invoke())
            values = []
            for _ in range(samples):
                s.prepare()
                started = time.perf_counter_ns()
                result = s.done(invoke())
                values.append((time.perf_counter_ns() - started) / 1000)
                del result
            api.append(values)
    # Extra combine notify exists only in this independent profiler pass, never in API timing.
    with (
        profile_combine_notify(op.startswith("combine")),
        prof.profile(
            activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
            on_trace_ready=prof.tensorboard_trace_handler(str(directory)),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=prof._ExperimentalConfig(
                export_type=[prof.ExportType.Text],
                profiler_level=prof.ProfilerLevel.Level0,
                aic_metrics=prof.AiCMetrics.AiCoreNone,
            ),
        ),
    ):

        def call():
            s.prepare()
            s.done(invoke())

        s.activate(configs[0], op)
        for _ in range(2):
            call()
        for config in configs:
            s.activate(config, op)
            for _ in range(s.args.warmup + samples + 2):
                call()
        for _ in range(2):
            call()
    local = parse_profile(directory, op, configs, s.args.warmup, samples)
    for i, stages in enumerate(local):
        stages["api"] = api[i]
    s.write(
        directory / "rank_samples.json", dict(stages=local, launches=observations, logical_payload_bytes=logical_bytes)
    )
    gathered = s.group.gather(dict(stages=local, bytes=logical_bytes))
    payload = sum(r["bytes"] for r in gathered) / s.group.size()
    rows = []
    for i, config in enumerate(configs):
        raw = slowest([r["stages"][i] for r in gathered])
        metrics = {k: stats(v) for k, v in raw.items()}
        rows.append(
            dict(
                config=config,
                metrics=metrics,
                samples=raw,
                GBps=payload / metrics["comm"]["mean"] / 1000,
                payload_bytes_per_rank=payload,
                precision_checked=False,
            )
        )
    s.write(directory / "summary.json", rows)
    return rows


def precision(session, shape, op, config, invoke, cpu, inputs):
    from bench_dispatch_stages import check_cpu_reference

    s, torch = session, session.torch
    args = SimpleNamespace(tokens=shape["tokens"], hidden=shape["hidden"], topk=8, experts=256)
    s.activate(config, op)
    x, idx, weights = cpu
    # Independent payload sum, one contribution per destination rank (not per expert).
    peers = idx // (256 // s.group.size())
    expected_x = torch.zeros_like(x, dtype=torch.float32)
    for peer in range(s.group.size()):
        expected_x += x.float() * (peers == peer).any(1).unsqueeze(1)
    for generation in (0, 1, 2):
        inputs["generation"] = generation
        # One addition from the original weight in both the input and golden.
        inputs["weight"] = (weights + generation).npu()
        if op == "combine_weighted":
            inputs["combine_weight"] = (inputs["original_combine_weight"] + generation).masked_fill(
                inputs["invalid_combine_weight"], 0
            )
        s.prepare()
        result = s.done(invoke())
        if op.startswith("combine"):
            torch.testing.assert_close(result[0].cpu(), expected_x.to(x.dtype), rtol=0, atol=0)
            if op.endswith("weighted"):
                torch.testing.assert_close(result[1].cpu(), weights + generation, rtol=0, atol=0)
        else:
            actual = [v.cpu() if v is not None else None for v in result[:3]]
            check_cpu_reference(
                torch,
                actual,
                args,
                s.rank,
                s.group.size(),
                cached=op.startswith("cached"),
                src_idx=result[3]._dispatch_layout[3].cpu(),
                handle=result[3],
                weight_offset=generation,
            )
    inputs["generation"] = 0
    inputs["weight"] = weights.npu()
    inputs["combine_weight"] = inputs["original_combine_weight"]


def run_case(s, shape, winners):
    from bench_dispatch_stages import cpu_inputs, check_cpu_reference
    from deep_ep.backends.direct.dtypes import topk_idx_t

    torch = s.torch
    cores, hidden = shape["aiv"], shape["hidden"]
    args = SimpleNamespace(tokens=shape["tokens"], hidden=hidden, topk=8, experts=256)
    cpu = cpu_inputs(torch, args, s.rank)
    cx, ci, cw = cpu
    dx, di = cx.npu(), ci.to(dtype=topk_idx_t, device="npu")
    inputs = dict(weight=cw.npu(), generation=0)
    auto_dispatch = policy_config(s.base.for_launch(cores, hidden * 2, "dispatch", bf16=True), s.group.size(), True)
    s.activate(auto_dispatch, "fresh_weighted")
    s.prepare()
    fresh = s.done(
        s.buffer.dispatch(
            dx, topk_idx=di, topk_weights=inputs["weight"], num_experts=256, num_sms=cores, do_expand=False
        )
    )
    # Validate the preparation handle independently; never trust it as a golden producer.
    check_cpu_reference(
        torch,
        [v.cpu() for v in fresh[:3]],
        args,
        s.rank,
        s.group.size(),
        src_idx=fresh[3]._dispatch_layout[3].cpu(),
        handle=fresh[3],
    )
    handle, owned, owned_weights = fresh[3], fresh[0].clone(), fresh[2].clone()
    inputs.update(
        original_combine_weight=owned_weights, combine_weight=owned_weights, invalid_combine_weight=fresh[1] < 0
    )
    torch.npu.synchronize()
    peers = ci // (256 // s.group.size())
    send_rows = sum(int((peers == p).any(1).sum()) for p in range(s.group.size()))
    logical_bytes = send_rows * hidden * 2
    representative = shape["tokens"] == 8192 and hidden in (2816, 6144)
    outputs = []
    for op in OPS:
        weighted = op.endswith("weighted")

        def invoke():
            if op.startswith("combine"):
                return s.buffer.combine(
                    owned,
                    handle,
                    topk_weights=inputs["combine_weight"] if weighted else None,
                    num_sms=cores,
                    reduce_num_sms=s.args.reduce_cores,
                )
            cached = op.startswith("cached")
            return s.buffer.dispatch(
                dx,
                handle=handle if cached else None,
                topk_idx=None if cached else di,
                topk_weights=inputs["weight"] if weighted else None,
                num_experts=256,
                num_sms=cores,
                do_expand=False,
            )

        auto = policy_config(s.base.for_launch(cores, hidden * 2, op_kind(op), bf16=True), s.group.size(), True)
        if representative:
            coarse = measure(
                s, shape, op, shortlist(s.group.size(), cores) + [auto], invoke, logical_bytes, "coarse", s.args.samples
            )
            best = min((r for r in coarse if not r["config"].get("auto")), key=lambda r: r["metrics"]["api"]["mean"])[
                "config"
            ]
            extra = [candidate(best["peers"], best["workers"], best["qps"], v, s.group.size()) for v in (4, 8, 64, 128)]
            refined = measure(s, shape, op, extra, invoke, logical_bytes, "visit", s.args.samples)
            selected = finalists(coarse + refined, s.args.top)
        else:
            # H4096 deliberately uses the H6144 winner: a held-out generalization test.
            source_hidden = 2816 if hidden == 2816 else 6144
            selected = unique([winners[cores, source_hidden, op], auto])
        for config in selected:
            precision(s, shape, op, config, invoke, cpu, inputs)
        s.group.agree(["precision_passed", shape, op, selected])
        turns = [
            measure(s, shape, op, order, invoke, logical_bytes, f"final{turn}", s.args.final_samples)
            for turn, order in enumerate((selected, list(reversed(selected))))
        ]
        results = []
        for config in selected:
            rounds = [next(r for r in turn if r["config"]["id"] == config["id"]) for turn in turns]
            pooled = {k: sum((r["samples"][k] for r in rounds), []) for k in rounds[0]["samples"]}
            metrics = {k: stats(v) for k, v in pooled.items()}
            results.append(
                dict(
                    config=config,
                    metrics=metrics,
                    rounds=rounds,
                    precision_checked=True,
                    GBps=rounds[0]["payload_bytes_per_rank"] / metrics["comm"]["mean"] / 1000,
                )
            )
        # Prefer lower worst-round completed API; retain communication winner independently.
        best = min(results, key=lambda r: max(v["metrics"]["api"]["mean"] for v in r["rounds"]))
        if representative:
            winners[cores, hidden, op] = transfer_candidate(best["config"], s.group.size())
        outputs.append(
            dict(shape=shape, operator=op, representative=representative, winner=best["config"], results=results)
        )
        s.write(s.out / "progress.json", dict(completed=s.completed + outputs, buffer_initializations=1))
    return outputs


def write_summary(s):
    rows = []
    for item in s.completed:
        baseline = next(r for r in item["results"] if r["config"].get("auto"))
        for r in item["results"]:
            c = r["config"]
            row = dict(
                EP=s.group.size(),
                **item["shape"],
                operator=item["operator"],
                config=c["id"],
                P=c["peers"],
                W=c["workers"],
                Q=c["qps"],
                visit=c["visit"],
                sender_cores=c["payload_aivs"],
                reduce_cores=s.args.reduce_cores,
                GBps_per_rank=r["GBps"],
                precision="zero_tolerance_passed",
                winner=c["id"] == item["winner"]["id"],
            )
            for stage, values in r["metrics"].items():
                row.update({f"{stage}_{k}_us": v for k, v in values.items()})
            row["api_change_vs_auto"] = r["metrics"]["api"]["mean"] / baseline["metrics"]["api"]["mean"] - 1
            row["bandwidth_change_vs_auto"] = r["GBps"] / baseline["GBps"] - 1
            row["api_worse_in_either_round"] = any(
                a["metrics"]["api"]["mean"] > b["metrics"]["api"]["mean"]
                for a, b in zip(r["rounds"], baseline["rounds"])
            )
            rows.append(row)
    with (s.args.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    from bench_fast_sweep import StoreGroup, qp_plan_spec, write

    reservation = [dict(operations={"all": shortlist(args.ep_size, a, VISITS)}) for a in (8, 32)]
    if args.plan_only:
        import json

        print(
            json.dumps(
                dict(
                    ep=args.ep_size,
                    cases=args.case,
                    operators=OPS,
                    candidates=reservation,
                    qp_plans=qp_plan_spec(reservation),
                    huge_pages=False,
                    reduce_cores=args.reduce_cores,
                    note="runtime adds actual auto and preparation plans after topology discovery",
                ),
                indent=2,
            )
        )
        return
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep.backends.direct.elastic import ElasticBuffer
    from deep_ep.backends.direct import buffer as buffer_module, elastic as elastic_module
    from deep_ep.backends.direct.udma_policy import resolve_udma_policy
    from bench_fused_notify import store_topology_collective
    from stage_metrics import L2CacheFlush, prepare_v2_sample

    unexpected = {k: v for k, v in os.environ.items() if k.startswith("DEEPEP_CLOS_") and v}
    if unexpected:
        raise RuntimeError(f"clear DEEPEP_CLOS_* overrides for the default comparison: {unexpected}")
    rank, world, device = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != args.ep_size:
        raise RuntimeError("WORLD_SIZE must match EP")
    torch.npu.set_device(device)
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        args.control_port,
        world,
        rank == 0,
        datetime.timedelta(seconds=args.timeout),
        wait_for_workers=True,
    )
    group = StoreGroup(store, rank, world)
    buffer_module._bootstrap_all_gather = lambda g, c, n, v: group.gather(v)
    elastic_module._validate_group_consensus = lambda g, name, value: group.agree([name, value])
    with store_topology_collective(group):
        base = resolve_udma_policy(True, world, "udma", None, None, None, None, None).bind_group(group)
    defaults = [
        policy_config(base.for_launch(c["aiv"], c["hidden"] * 2, op_kind(op), bf16=True), world)
        for c in args.case
        for op in OPS
    ]
    # Include constructor defaults and all preparation launches in the same reservation union.
    defaults.append(policy_config(base, world))
    reservation.append(dict(operations={"default": defaults}))
    spec = qp_plan_spec(reservation)
    group.agree([args.case, defaults, spec, args.reduce_cores, args.samples, args.final_samples])
    native_path = Path(importlib.util.find_spec("deep_ep._clos_C").origin)
    identity_paths = [
        Path(__file__),
        Path(__file__).with_name("bench_fast_sweep.py"),
        Path(__file__).with_name("bench_dispatch_stages.py"),
        Path(__file__).with_name("stage_metrics.py"),
        *Path(elastic_module.__file__).parent.glob("*.py"),
        native_path,
        native_path.with_name("libdeepep_clos_kernels.so"),
        *native_path.parent.glob(".libs/*.so"),
    ]
    if not list(native_path.parent.glob(".libs/*.so")):
        identity_paths.append(Path(os.environ["SHMEM_ROOT"]) / "lib/libshmem.so")
    identities = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in identity_paths}
    group.agree(identities)
    from deep_ep.backends.direct._native import load_native_module

    native = load_native_module()
    require_native(native)
    os.environ["DEEPEP_CLOS_TUNING_SESSION"] = "1"
    os.environ["DEEPEP_CLOS_QP_PLANS"] = spec
    capacity = max(c["tokens"] for c in args.case)
    workspace = max(ElasticBuffer.get_buffer_size_hint(group, capacity, c["hidden"], 8) for c in args.case)
    out = args.output / f"rank{rank}"
    if not all(group.gather(not out.exists())):
        raise RuntimeError("output contains a previous rank directory; use a new output path")
    write(
        out / "session.json",
        dict(
            identities=identities,
            qp_plans=spec,
            cases=args.case,
            huge_pages=False,
            flush_l2=False,
            seed=20261122,
            topology=base.ranks_per_group,
            topology_known=base.topology_known,
            baseline_plans=defaults,
            reduce_cores=args.reduce_cores,
        ),
    )
    if rank == 0:
        print(f"BUFFER_INIT EP={world} capacity={capacity} bytes={workspace}", flush=True)
    with store_topology_collective(group):
        buffer = ElasticBuffer(
            group,
            num_bytes=workspace,
            num_max_tokens_per_rank=capacity,
            hidden=0,
            num_topk=0,
            explicitly_destroy=True,
            use_1g_huge_pages=False,
            transport="udma",
            shmem_ip_port=f"tcp://{os.environ['MASTER_ADDR']}:{args.shmem_port}",
        )
    if buffer.udma_auto_policy != base:
        raise RuntimeError("constructor topology/default policy changed after reservation")
    actual_huge_pages = bool(buffer.runtime.use_1g_huge_pages())
    if any(group.gather(actual_huge_pages)):
        raise RuntimeError("native Buffer unexpectedly enabled 1GiB pages")
    write(out / "buffer.json", dict(initializations=1, actual_huge_pages=actual_huge_pages))
    policy = SweepPolicy(base)
    buffer._udma_auto_policy = policy
    flush = L2CacheFlush(torch, buffer, False)

    def done(result):
        if result[-1].event is not None:
            result[-1].current_stream_wait()
        torch.npu.synchronize()
        return result

    def activate(config, op):
        torch.npu.synchronize()
        policy.config, policy.operation = config, op_kind(op)

    session = SimpleNamespace(
        torch=torch,
        torch_npu=torch_npu,
        native=native,
        buffer=buffer,
        args=args,
        group=group,
        rank=rank,
        out=out,
        policy=policy,
        base=base,
        write=write,
        done=done,
        activate=activate,
        prepare=lambda: prepare_v2_sample(buffer, flush),
        completed=[],
    )
    winners = {}
    # No finally/destroy on failure: the CLI records the error and exits the worker immediately.
    for shape in args.case:
        session.completed.extend(run_case(session, shape, winners))
        if rank == 0:
            write_summary(session)
        torch.npu.synchronize()
        torch.npu.empty_cache()
    group.agree("successful_ready_to_destroy")
    buffer.destroy()
    write(out / "exit.json", dict(code=0, complete=True, buffer_initializations=1, precision_checked=True))
    group.agree("all_exit_files_written")
    if rank == 0:
        write(
            args.output / "report.json",
            dict(
                ep=world,
                results=session.completed,
                buffer_initializations=1,
                huge_pages=False,
                precision_checked=True,
                bandwidth="deduplicated BF16 bytes including self; excludes weights",
                aggregation="per-sample slowest rank; independent API; nearest-rank P95",
                limitation="staged candidate best, not exhaustive optimum; H4096 inherits H6144 candidate",
            ),
        )
        print(f"SMALL_RANK_SWEEP_COMPLETE EP={world} output={args.output}", flush=True)
