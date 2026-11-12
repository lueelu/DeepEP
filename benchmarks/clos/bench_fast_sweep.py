"""One SHMEM initialization, existing full-Clos peer schedules, performance only.

Launch on each node with torchrun. --plan-only needs no torch/NPU installation.
No numerical validation is performed; every report explicitly records that.
"""

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time
import zlib

ORDERS = ("rank", "group", "group_staggered", "group_round_robin")
COMBINE_SELF_COPY_ENV = "DEEPEP_CLOS_COMBINE_PARALLEL_SELF_COPY"
COMBINE_AFTER_SUBMIT_ENV = "DEEPEP_CLOS_COMBINE_SELF_COPY_AFTER_SUBMIT"
SELF_COPY_MODES = ("default", "parallel_self_copy", "self_copy_after_submit")
SHAPES = ((8192, 2816, 8, 32), (16384, 7168, 6, 64))


def candidates(ep, aiv, group_size, visits=(4, 8, 16, 32), quick=False):
    result = []
    for order in ORDERS:
        grouped = order == "group_round_robin"
        for peers, (workers, qps), visit in itertools.product(
            (1, 2, 4) if grouped else (1, 2, 4, 8), ((1, 1), (1, 2), (2, 1)), visits
        ):
            owners = math.ceil((group_size if grouped else ep - 1) / peers)
            if owners * workers > aiv or owners * workers * qps > 32:
                continue
            if grouped and math.ceil(group_size / owners) * (ep // group_size) > 64:
                continue
            if quick and order not in ("rank", "group_round_robin"):
                continue
            item = dict(order=order, peers=peers, workers=workers, qps=qps, visit=visit)
            item["id"] = f"{order}_p{peers}_w{workers}_q{qps}_v{visit}"
            item["payload_aivs"] = owners * workers
            result.append(item)
    # Put the production automatic configuration first, as a measured baseline.
    workers = 2 if ep <= 8 else 1
    peers, visit = (1, 8) if ep <= 16 else (4, 32)
    while math.ceil((ep - 1) / peers) * workers > min(aiv, 32) and peers < 8:
        peers *= 2
    baseline = f"rank_p{peers}_w{workers}_q1_v{visit}"
    result.sort(key=lambda c: c["id"] != baseline)
    return result


def parse_shape(value):
    try:
        tokens, hidden, topk, aiv = map(int, value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be TOKENS,HIDDEN,TOPK,AIV") from exc
    if min(tokens, hidden, topk, aiv) <= 0 or hidden % 32 or topk > 128 or aiv % 2 or aiv > 64:
        raise argparse.ArgumentTypeError("positive case required: hidden multiple of 32, topk<=128, even AIV<=64")
    return dict(tokens=tokens, hidden=hidden, topk=topk, aiv=aiv)


def combine_candidates(configs, modes, aiv, weighted=False):
    """Mode is part of candidate identity, including in confirmations/replay."""
    return [
        c | {"id": f"{c['id']}__{mode}", "self_copy_mode": mode}
        for c in configs
        for mode in modes
        if not (weighted and mode == "parallel_self_copy")
        if not (mode == "parallel_self_copy" and "default" in modes and c["payload_aivs"] >= aiv)
    ]


def replay_config(config, op, report, legal, mode_override=None):
    # Old reports stored the mode at report level; retain that behavior on replay.
    mode = config.get("self_copy_mode")
    if mode is not None:
        if op not in ("combine_unweighted", "combine_weighted") or mode not in SELF_COPY_MODES:
            raise ValueError("replay contains an invalid candidate mode")
        suffix = "__" + mode
        if not config["id"].endswith(suffix):
            raise ValueError("replay contains an invalid candidate id")
        base = {k: v for k, v in config.items() if k != "self_copy_mode"}
        base["id"] = base["id"][: -len(suffix)]
    else:
        base = config
        parallel = report.get("combine_parallel_self_copy", False)
        after = report.get("combine_self_copy_after_submit", False)
        if parallel and after:
            raise ValueError("replay contains conflicting self-copy modes")
        mode = "parallel_self_copy" if parallel else "self_copy_after_submit" if after else "default"
    if base != legal.get(base["id"]):
        raise ValueError("replay contains an invalid candidate")
    if mode_override is not None:
        mode = mode_override
    if op == "combine_weighted" and mode == "parallel_self_copy":
        raise ValueError("weighted combine does not support parallel_self_copy")
    return combine_candidates([base], [mode], 64)[0] if op != "cached_dispatch" else base


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ep-size", type=int, choices=(2, 4, 8, 16, 32, 64, 128, 256), default=64)
    p.add_argument("--ranks-per-group", type=int, default=None)
    p.add_argument("--case", type=parse_shape, action="append", help="repeat TOKENS,HIDDEN,TOPK,AIV")
    p.add_argument("--output", type=Path, default=Path("fast_sweep"))
    p.add_argument("--control-port", type=int, default=29610)
    p.add_argument("--shmem-port", type=int, default=29620)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--final-samples", type=int, default=30)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--visits", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--quick", action="store_true", help="rank and group_round_robin search")
    p.add_argument("--strategies", nargs="+", help="restrict to exact strategy IDs for a matched A/B run")
    p.add_argument("--include-weighted-combine", action="store_true", help="also measure returned FP32 top-k weights")
    p.add_argument("--huge-pages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--flush-l2-cache", action=argparse.BooleanOptionalAction, default=False)
    self_copy = p.add_mutually_exclusive_group()
    self_copy.add_argument(
        "--combine-parallel-self-copy", action="store_true", help="use spare AIVs for combine self-copy"
    )
    self_copy.add_argument(
        "--combine-self-copy-after-submit", action="store_true", help="submit remote combine before self-copy"
    )
    self_copy.add_argument(
        "--combine-self-copy-modes",
        nargs="+",
        choices=SELF_COPY_MODES,
        help="sweep these combine modes in the same Buffer session",
    )
    p.add_argument("--replay-report", type=Path, help="rerun each operator's winner from a prior report")
    p.add_argument("--smoke", action="store_true", help="one legal candidate per peer order")
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args(argv)
    if args.replay_report and args.combine_self_copy_modes and len(args.combine_self_copy_modes) != 1:
        p.error("replay accepts only one explicit self-copy mode")
    args.replay_mode_override = (
        args.combine_self_copy_modes[0]
        if args.combine_self_copy_modes and args.replay_report
        else "parallel_self_copy"
        if args.combine_parallel_self_copy
        else "self_copy_after_submit"
        if args.combine_self_copy_after_submit
        else None
    )
    if args.combine_self_copy_modes is None:
        args.combine_self_copy_modes = [
            "parallel_self_copy"
            if args.combine_parallel_self_copy
            else "self_copy_after_submit"
            if args.combine_self_copy_after_submit
            else "default"
        ]
    if len(set(args.combine_self_copy_modes)) != len(args.combine_self_copy_modes):
        p.error("duplicate self-copy modes")
    if args.include_weighted_combine and args.combine_self_copy_modes == ["parallel_self_copy"]:
        p.error("weighted combine does not support parallel_self_copy; include default or self_copy_after_submit")
    if args.strategies and (args.replay_report or args.quick or args.smoke):
        p.error("strategies cannot be combined with replay-report, quick or smoke")
    if args.strategies and len(set(args.strategies)) != len(args.strategies):
        p.error("duplicate strategy IDs")
    args.ranks_per_group = min(8, args.ep_size) if args.ranks_per_group is None else args.ranks_per_group
    if args.ranks_per_group < 1 or args.ep_size % args.ranks_per_group:
        p.error("ranks-per-group must be a positive EP divisor")
    args.case = args.case or [dict(tokens=t, hidden=h, topk=k, aiv=a) for t, h, k, a in SHAPES]
    if args.experts % args.ep_size or args.experts < max(c["topk"] for c in args.case):
        p.error("experts must be divisible by EP and cover topk")
    if args.warmup < 0 or min(args.samples, args.final_samples, args.top) < 1:
        p.error("invalid sampling parameters")
    if not args.visits or not set(args.visits) <= {4, 8, 16, 32}:
        p.error("visits must be 4/8/16/32")
    if len({tuple(c.values()) for c in args.case}) != len(args.case):
        p.error("duplicate cases")
    return args


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf8")
    temporary.replace(path)


def pack(value):
    if isinstance(value, tuple):
        return {"__tuple__": [pack(x) for x in value]}
    if isinstance(value, list):
        return [pack(x) for x in value]
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    return value


def unpack(value):
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(unpack(x) for x in value["__tuple__"])
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(x) for x in value]
    return value


class StoreGroup:
    """Small JSON control collectives over TCPStore, with no Gloo group."""

    def __init__(self, store, rank, size):
        self.store, self._rank, self._size, self.seq = store, rank, size, 0

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def gather(self, obj):
        key = str(self.seq)
        self.seq += 1

        def encode(value):
            return zlib.compress(json.dumps(value, separators=(",", ":")).encode(), 1)

        def decode(value):
            return json.loads(zlib.decompress(value))

        self.store.set(f"{key}/{self._rank}", encode(pack(obj)))
        if self._rank == 0:
            rows = [decode(self.store.get(f"{key}/{i}")) for i in range(self._size)]
            self.store.set(key + "/all", encode(rows))
        result = unpack(decode(self.store.get(key + "/all")))
        self.store.set(f"{key}/read/{self._rank}", "1")
        if self._rank == 0:
            for i in range(self._size):
                self.store.get(f"{key}/read/{i}")
        return result

    def agree(self, obj):
        # Plans can contain hundreds of configurations. Compare canonical
        # hashes rather than broadcasting EP identical full plans to everyone.
        digest = hashlib.sha256(json.dumps(pack(obj), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        rows = self.gather(digest)
        if any(row != rows[0] for row in rows):
            raise RuntimeError("rank configuration mismatch")


def configure(buffer, config):
    from deep_ep.backends.direct.config import Config

    order = ORDERS.index(config["order"])
    group_peers = config["peers"] if order >= 3 else 0
    wanted = [order, order, group_peers, config["workers"], config["qps"]]
    actual = list(buffer.runtime._configure_tuning(*wanted))
    if actual != wanted:
        raise RuntimeError(f"native tuning mismatch: {actual} != {wanted}")
    buffer._udma_config = Config(udma_peers_per_worker=config["peers"], udma_wqes_per_visit=config["visit"])
    buffer._udma_dispatch_peer_order = buffer._udma_combine_peer_order = config["order"]
    buffer._udma_peers_per_group_per_worker = group_peers or None
    buffer._udma_workers_per_peer, buffer._udma_qps_per_worker = config["workers"], config["qps"]


def activate_candidate(torch, buffer, config):
    # Drain all previous device work before changing host launch flags or QP routing.
    # The next call's Buffer barrier aligns ranks before the measured invocation.
    torch.npu.synchronize()
    mode = config.get("self_copy_mode", "default")
    if mode not in SELF_COPY_MODES:
        raise ValueError(f"unknown self-copy mode: {mode}")
    os.environ[COMBINE_SELF_COPY_ENV] = "0"
    os.environ[COMBINE_AFTER_SUBMIT_ENV] = "0"
    if mode == "parallel_self_copy":
        os.environ[COMBINE_SELF_COPY_ENV] = "1"
    elif mode == "self_copy_after_submit":
        os.environ[COMBINE_AFTER_SUBMIT_ENV] = "1"
    configure(buffer, config)


def parse_capture(directory, op, configs, warmup, samples):
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one profiler CSV, found {len(paths)}")
    with paths[0].open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda x: float(x["Start Time(us)"].strip()))
    # buffer.barrier() and pre-combine notify share a kernel name. Match the
    # complete ordered call, so the outer barrier is never charged to notify.
    names = (
        ["cached_notify_combine_kernel", "cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
        if op == "cached_dispatch"
        else [
            "cached_notify_combine_kernel",
            "cached_notify_combine_kernel",
            "v1_combine_udma_comm_kernel",
            "v1_combine_udma_reduce_kernel",
        ]
    )
    width = warmup + samples + 2
    required = len(configs) * width + 4  # global leading/trailing guards
    selected = [r for r in rows if r["Name"] in names]
    if len(selected) != required * len(names):
        raise RuntimeError(f"incomplete capture: {len(selected)} stage records; expected {required * len(names)}")
    sequence = [r["Name"] for r in selected]
    if sequence != names * required:
        raise RuntimeError("kernel order differs from expected barrier/notify/communication/reduce sequence")
    stages = [[float(r["Duration(us)"]) for r in selected[i :: len(names)]] for i in range(len(names))]
    result = []
    for index, config in enumerate(configs):
        start = 2 + index * width + warmup
        values = [v[start : start + samples] for v in stages]
        if any(not math.isfinite(x) or x <= 0 for v in values for x in v):
            raise RuntimeError("invalid kernel durations")
        result.append(
            dict(
                config=config,
                samples=samples,
                mean_us=statistics.mean(values[2]),
                barrier_us=statistics.mean(values[0]),
                notify_us=statistics.mean(values[1]),
                reduce_us=statistics.mean(values[3]) if len(names) == 4 else 0,
                communication_samples_us=values[2],
            )
        )
    return result


def measure(torch, torch_npu, buffer, group, args, directory, op, configs, invoke, logical_bytes, samples):
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    group.agree([op, configs, samples])
    if group.rank() == 0:
        print(
            f"PHASE_BEGIN {directory.parent.name}/{directory.name} configurations={len(configs)} samples={samples}",
            flush=True,
        )
    write(
        directory / "manifest.json",
        dict(
            op=op,
            configs=configs,
            samples=samples,
            warmup=args.warmup,
            precision_checked=False,
            logical_bytes=logical_bytes,
        ),
    )

    def call():
        buffer.barrier(with_cpu_sync=True)
        flush_cache()
        result = invoke()
        result[-1].current_stream_wait()
        torch.npu.synchronize()

    from stage_metrics import L2CacheFlush, profile_combine_notify

    flush_cache = L2CacheFlush(torch, buffer, args.flush_l2_cache)
    # Start/stop the profiler once for the entire strategy list, not per strategy.
    profiler = torch_npu.profiler
    with (
        profile_combine_notify(op in ("combine_unweighted", "combine_weighted")),
        profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=profiler._ExperimentalConfig(
                export_type=[profiler.ExportType.Text],
                profiler_level=profiler.ProfilerLevel.Level0,
                aic_metrics=profiler.AiCMetrics.AiCoreNone,
            ),
        ),
    ):
        activate_candidate(torch, buffer, configs[0])
        for _ in range(2):
            call()
        for index, config in enumerate(configs):
            activate_candidate(torch, buffer, config)
            for _ in range(args.warmup + samples + 2):
                call()
            if group.rank() == 0 and (index % 20 == 0 or index == len(configs) - 1):
                print(f"SWEEP {directory.parent.name}/{op} {index + 1}/{len(configs)} {config['id']}", flush=True)
        for _ in range(2):
            call()
    local = None
    error = None
    try:
        local = parse_capture(directory, op, configs, args.warmup, samples)
        write(directory / "samples.json", local)
    except Exception as exc:
        error = str(exc)
    errors = group.gather(error)
    if any(errors):
        raise RuntimeError(f"profiler parsing failed: {[(i, e) for i, e in enumerate(errors) if e]}")
    gathered = group.gather(
        dict(bytes=logical_bytes, rows=[{k: r[k] for k in ("mean_us", "notify_us", "reduce_us")} for r in local])
    )
    result = []
    for index, config in enumerate(configs):
        rows = [r["rows"][index] for r in gathered]
        times = [r["mean_us"] for r in rows]
        bandwidths = [g["bytes"] / r["mean_us"] / 1000 for g, r in zip(gathered, rows)]
        result.append(
            dict(
                config=config,
                GBps=sum(g["bytes"] for g in gathered) / sum(times) / 1000,
                GBps_range=[min(bandwidths), max(bandwidths)],
                mean_us=statistics.mean(times),
                time_range_us=[min(times), max(times)],
                notify_us_range=[min(r["notify_us"] for r in rows), max(r["notify_us"] for r in rows)],
                reduce_us_range=[min(r["reduce_us"] for r in rows), max(r["reduce_us"] for r in rows)],
                samples_per_rank=samples,
                precision_checked=False,
            )
        )
    result.sort(key=lambda r: r["GBps"], reverse=True)
    if group.rank() == 0:
        write(directory / "ranking.json", result)
        print(
            f"PHASE_DONE {directory.parent.name}/{directory.name} elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )
    return result


def main():
    args = parse_args()
    shapes = args.case
    plans = []
    replay = json.loads(args.replay_report.read_text()) if args.replay_report else None
    if replay and any(
        replay.get(k) != v
        for k, v in (
            ("ep", args.ep_size),
            ("experts", args.experts),
            ("seed", args.seed),
            ("ranks_per_group", args.ranks_per_group),
            ("huge_pages", args.huge_pages),
            ("flush_l2", args.flush_l2_cache),
        )
    ):
        raise ValueError("replay EP/experts/seed/group size/L2/page settings must match report")
    for shape in shapes:
        configs = candidates(args.ep_size, shape["aiv"], args.ranks_per_group, args.visits, args.quick)
        if not configs:
            raise ValueError(f"no legal candidates for {shape}")
        if args.smoke:
            configs = [next(c for c in configs if c["order"] == o) for o in dict.fromkeys(c["order"] for c in configs)]
        if args.strategies:
            legal = {c["id"]: c for c in configs}
            missing = set(args.strategies) - legal.keys()
            if missing:
                raise ValueError(f"invalid strategies for {shape}: {sorted(missing)}")
            configs = [legal[name] for name in args.strategies]
        operations = {
            "cached_dispatch": configs,
            "combine_unweighted": combine_candidates(configs, args.combine_self_copy_modes, shape["aiv"]),
        }
        if args.include_weighted_combine:
            operations["combine_weighted"] = combine_candidates(
                configs, args.combine_self_copy_modes, shape["aiv"], weighted=True
            )
        if replay:
            prior = next(r for r in replay["results"] if r["shape"] == shape)
            required = {"cached_dispatch", "combine_unweighted"}
            if args.include_weighted_combine:
                required.add("combine_weighted")
            if not required <= prior["operations"].keys() or prior["operations"].keys() - {
                "cached_dispatch",
                "combine_unweighted",
                "combine_weighted",
            }:
                raise ValueError("replay must include the requested measured operators")
            legal = {c["id"]: c for c in candidates(args.ep_size, shape["aiv"], args.ranks_per_group)}
            operations = {
                op: [replay_config(r["winner"]["config"], op, replay, legal, args.replay_mode_override)]
                for op, r in prior["operations"].items()
            }
            configs = list({c["id"]: c for cs in operations.values() for c in cs}.values())
        plans.append(dict(shape=shape, configs=configs, operations=operations))
    effective_modes = list(
        dict.fromkeys(c["self_copy_mode"] for p in plans for c in p["operations"]["combine_unweighted"])
    )
    if args.plan_only:
        print(json.dumps(dict(ep=args.ep_size, plans=plans), indent=2))
        return
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep.backends.direct.elastic import ElasticBuffer
    from deep_ep.backends.direct import buffer as buffer_module, elastic as elastic_module

    rank, world, device = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != args.ep_size:
        raise RuntimeError("WORLD_SIZE must match EP")
    torch.npu.set_device(device)
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        args.control_port,
        world,
        rank == 0,
        timeout=datetime.timedelta(seconds=args.timeout),
        wait_for_workers=True,
    )
    group = StoreGroup(store, rank, world)
    # Narrow benchmark adapters: keep all bootstrap/consensus checks, replacing
    # only their transport with TCPStore; production distributed APIs are untouched.
    buffer_module._bootstrap_all_gather = lambda g, c, n, value: group.gather(value)
    elastic_module._validate_group_consensus = lambda g, name, value: group.agree([name, value])
    group.agree([vars(args) | {"output": None, "case": shapes, "replay_report": None}, plans])
    os.environ["DEEPEP_CLOS_TUNING_SESSION"] = "1"
    os.environ[COMBINE_SELF_COPY_ENV] = "0"
    os.environ[COMBINE_AFTER_SUBMIT_ENV] = "0"
    for name, value in os.environ.items():
        if (
            name.startswith("DEEPEP_CLOS_")
            and value not in ("", "0")
            and name not in ("DEEPEP_CLOS_TUNING_SESSION", COMBINE_SELF_COPY_ENV, COMBINE_AFTER_SUBMIT_ENV)
        ):
            raise RuntimeError(f"conflicting experimental environment: {name}")
    group.agree(plans)
    import importlib.util

    native = Path(importlib.util.find_spec("deep_ep._clos_C").origin)
    identity_paths = [
        Path(__file__),
        Path(__file__).with_name("stage_metrics.py"),
        *sorted(Path(elastic_module.__file__).parent.glob("*.py")),
        native,
        native.with_name("libdeepep_clos_kernels.so"),
        *sorted(native.parent.glob(".libs/*.so")),
    ]
    if not list(native.parent.glob(".libs/*.so")):
        identity_paths.append(Path(os.environ["SHMEM_ROOT"]) / "lib/libshmem.so")
    identity = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in identity_paths}
    group.agree(identity)
    capacity = max(s["tokens"] for s in shapes)
    workspace = max(ElasticBuffer.get_buffer_size_hint(group, capacity, s["hidden"], s["topk"]) for s in shapes)
    out = args.output / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    write(out / "identity.json", identity)
    write(
        out / "plan.json",
        dict(
            ep=world,
            experts=args.experts,
            seed=args.seed,
            plans=plans,
            workspace_bytes=workspace,
            huge_pages=args.huge_pages,
            flush_l2=args.flush_l2_cache,
            combine_parallel_self_copy=effective_modes == ["parallel_self_copy"],
            combine_self_copy_after_submit=effective_modes == ["self_copy_after_submit"],
            combine_self_copy_modes=effective_modes,
            pre_combine_notify=True,
            precision_checked=False,
        ),
    )
    if rank == 0:
        print(f"BUFFER_INIT bytes={workspace} cases={[len(p['configs']) for p in plans]}", flush=True)
    init_started = time.monotonic()
    buffer = ElasticBuffer(
        group,
        num_bytes=workspace,
        num_max_tokens_per_rank=capacity,
        hidden=0,
        num_topk=0,
        explicitly_destroy=True,
        use_1g_huge_pages=args.huge_pages,
        shmem_ip_port=f"tcp://{os.environ['MASTER_ADDR']}:{args.shmem_port}",
        transport="udma",
        udma_auto_policy=False,
        udma_schedule="peer",
        udma_dispatch_peer_order="rank",
        udma_combine_peer_order="rank",
        udma_ranks_per_group=args.ranks_per_group,
        udma_workers_per_peer=1,
        udma_qps_per_worker=1,
        udma_peers_per_worker=8,
        udma_wqes_per_visit=8,
    )
    init_seconds = time.monotonic() - init_started
    if rank == 0:
        print(f"BUFFER_READY initializations=1 elapsed_s={init_seconds:.1f}", flush=True)
    result = []
    try:
        for plan in plans:
            shape, configs = plan["shape"], plan["configs"]
            t, h, k, a = (shape[n] for n in ("tokens", "hidden", "topk", "aiv"))
            tag = f"t{t}_h{h}_k{k}_aiv{a}"
            gen = torch.Generator().manual_seed(args.seed + rank)
            routes = torch.rand((t, args.experts), generator=gen).topk(k, dim=1).indices
            ranked = (routes // (args.experts // world)).sort(dim=1).values
            send_rows = int((ranked[:, 1:] != ranked[:, :-1]).sum()) + t
            x = torch.randn((t, h), generator=gen, dtype=torch.bfloat16).npu()
            from deep_ep.backends.direct.dtypes import topk_idx_t

            idx = routes.to(dtype=topk_idx_t, device="npu")
            weighted = "combine_weighted" in plan["operations"]
            weights = torch.rand((t, k), generator=gen, dtype=torch.float32).npu() if weighted else None
            activate_candidate(torch, buffer, plan["operations"]["cached_dispatch"][0])
            fresh = buffer.dispatch(
                x,
                topk_idx=idx,
                topk_weights=weights,
                num_experts=args.experts,
                num_sms=a,
                do_expand=False,
                async_with_compute_stream=True,
            )
            fresh[-1].current_stream_wait()
            torch.npu.synchronize()
            recv, _, recv_weights, handle, _ = fresh
            owned = recv.clone()
            if weighted and recv_weights is None:
                raise RuntimeError("fresh dispatch did not return top-k weights")
            owned_weights = recv_weights.clone() if weighted else None
            torch.npu.synchronize()
            invocations = {
                "cached_dispatch": lambda x=x, handle=handle, a=a: buffer.dispatch(
                    x, handle=handle, num_sms=a, do_expand=False, async_with_compute_stream=True
                ),
                "combine_unweighted": lambda owned=owned, handle=handle, a=a: buffer.combine(
                    owned, handle, num_sms=a, async_with_compute_stream=True
                ),
            }
            if weighted:
                invocations["combine_weighted"] = (
                    lambda owned=owned, handle=handle, a=a, w=owned_weights: buffer.combine(
                        owned, handle, topk_weights=w, num_sms=a, async_with_compute_stream=True
                    )
                )
            item = dict(shape=shape, operations={})
            for op, invoke in invocations.items():
                row_bytes = h * 2 + (k * 4 if op == "combine_weighted" else 0)
                logical_bytes = (handle.num_recv_tokens if op == "cached_dispatch" else send_rows) * row_bytes
                op_configs = plan["operations"][op]
                coarse = measure(
                    torch,
                    torch_npu,
                    buffer,
                    group,
                    args,
                    out / tag / (op + "_coarse"),
                    op,
                    op_configs,
                    invoke,
                    logical_bytes,
                    args.samples,
                )
                finalists = [r["config"] for r in coarse[: args.top]]
                rounds = []
                for turn, order in enumerate((finalists, list(reversed(finalists)))):
                    rounds.append(
                        measure(
                            torch,
                            torch_npu,
                            buffer,
                            group,
                            args,
                            out / tag / f"{op}_final{turn}",
                            op,
                            order,
                            invoke,
                            logical_bytes,
                            args.final_samples,
                        )
                    )
                final = []
                for config in finalists:
                    rows = [next(r for r in round_ if r["config"]["id"] == config["id"]) for round_ in rounds]
                    final.append(dict(config=config, score_GBps=min(r["GBps"] for r in rows), repeats=rows))
                final.sort(key=lambda r: r["score_GBps"], reverse=True)
                item["operations"][op] = dict(
                    winner=final[0],
                    finalists=final,
                    coarse=coarse,
                    logical_row_bytes=row_bytes,
                    bf16_payload_row_bytes=h * 2,
                )
                if rank == 0:
                    print("WINNER", tag, op, json.dumps(final[0]), flush=True)
            result.append(item)
            invoke = None  # release the last closure before freeing this shape
            write(
                out / "results.json", dict(ep=world, buffer_initializations=1, precision_checked=False, results=result)
            )
            del invocations, fresh, recv, recv_weights, handle, owned, owned_weights, weights, x, idx
            torch.npu.synchronize()
            torch.npu.empty_cache()
    finally:
        buffer.destroy()
        torch.npu.synchronize()
    group.agree("complete")
    write(
        out / "exit.json",
        dict(
            code=0,
            buffer_initializations=1,
            initialization_seconds=init_seconds,
            precision_checked=False,
            complete=True,
        ),
    )
    group.agree("exit files written")
    if rank == 0:
        write(
            args.output / "report.json",
            dict(
                ep=world,
                experts=args.experts,
                dtype="bf16",
                expand=False,
                ranks_per_group=args.ranks_per_group,
                huge_pages=args.huge_pages,
                flush_l2=args.flush_l2_cache,
                combine_parallel_self_copy=effective_modes == ["parallel_self_copy"],
                combine_self_copy_after_submit=effective_modes == ["self_copy_after_submit"],
                combine_self_copy_modes=effective_modes,
                pre_combine_notify=True,
                seed=args.seed,
                bandwidth_formula="sum(rank deduplicated payload + optional FP32 weight bytes including self) / sum(rank mean communication us) / 1000",
                identity=identity,
                buffer_initializations=1,
                precision_checked=False,
                results=result,
            ),
        )
        best = [dict(shape=r["shape"], operator=op, **v["winner"]) for r in result for op, v in r["operations"].items()]
        write(args.output / "best_configs.json", best)
        with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [
                "tokens",
                "hidden",
                "topk",
                "aiv",
                "operator",
                "config",
                "self_copy_mode",
                "GBps_min",
                "GBps_max",
                "comm_us_min",
                "comm_us_max",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for item in best:
                runs = item["repeats"]
                writer.writerow(
                    dict(
                        **item["shape"],
                        operator=item["operator"],
                        config=item["config"]["id"],
                        self_copy_mode=item["config"].get("self_copy_mode", "not_applicable"),
                        GBps_min=min(r["GBps"] for r in runs),
                        GBps_max=max(r["GBps"] for r in runs),
                        comm_us_min=min(r["mean_us"] for r in runs),
                        comm_us_max=max(r["mean_us"] for r in runs),
                    )
                )
        print("FAST_SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import sys
        import traceback

        if "--output" in sys.argv and "RANK" in os.environ:
            destination = Path(sys.argv[sys.argv.index("--output") + 1]) / f"rank{os.environ['RANK']}" / "exit.json"
            write(destination, dict(code=1, complete=False, traceback=traceback.format_exc()))
        raise
