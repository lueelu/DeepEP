"""Fixed-traffic UDMA diagnostics; all cases share one native session per rank."""

import argparse
import csv
from datetime import timedelta
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys

MODES = {"pair": 0, "permutation": 1, "alltoall": 2}


class Control:
    """JSON control records over one TCPStore; no Gloo/HCCL process group."""

    def __init__(self, rank, world, port, timeout):
        from torch.distributed import TCPStore

        self.rank, self.world, self.round = rank, world, 0
        port = port or int(os.environ["MASTER_PORT"]) + 1
        if not 1 <= port <= 65535 or port == int(os.environ["MASTER_PORT"]):
            raise ValueError("control port must be valid and distinct from torchrun's port")
        self.store = TCPStore(os.environ["MASTER_ADDR"], port, world, rank == 0, timeout=timedelta(seconds=timeout))

    def key(self):
        self.round += 1
        return f"udma/{self.round}"

    def all_gather_object(self, output, value):
        key = self.key()
        self.store.set(f"{key}/{self.rank}", json.dumps(value))
        output[:] = [json.loads(self.store.get(f"{key}/{rank}")) for rank in range(self.world)]

    def broadcast_object_list(self, output, src):
        key = self.key()
        if self.rank == src:
            self.store.set(key, json.dumps(output))
        output[:] = json.loads(self.store.get(key))

    def gather_object(self, value, output, dst):
        key = self.key()
        self.store.set(f"{key}/{self.rank}", json.dumps(value))
        if self.rank == dst:
            output[:] = [json.loads(self.store.get(f"{key}/{rank}")) for rank in range(self.world)]
            for rank in range(self.world):
                self.store.delete_key(f"{key}/{rank}")

    def barrier(self):
        self.all_gather_object([], True)

    def finish(self):
        # Keep rank0's server alive until all clients have finished their last
        # reads. After publishing this acknowledgement clients never read again.
        self.store.set(f"udma/closed/{self.rank}", "1")
        if self.rank == 0:
            self.store.wait([f"udma/closed/{rank}" for rank in range(self.world)])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ep-size", type=int, default=int(os.environ.get("WORLD_SIZE", "32")))
    p.add_argument("--native-dir", type=Path, default=Path("build/udma-microbench"))
    p.add_argument("--output", type=Path, default=Path("benchmark-runs/udma-microbench"))
    p.add_argument("--first-device", type=int, default=0)
    p.add_argument("--aiv", type=int, default=32)
    p.add_argument("--peers-per-worker", type=int, default=4, choices=[1, 2, 4, 8])
    p.add_argument("--wqes-per-visit", type=int, default=32, choices=[4, 8, 16, 32])
    p.add_argument("--bytes-per-peer", type=int, nargs="+", default=[16 * 1024**2])
    p.add_argument("--chunk-bytes", type=int, nargs="+", default=[262144, 5632])
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    p.add_argument("--pair-scope", choices=["spokes", "all"], default="spokes")
    p.add_argument("--deltas", type=int, nargs="+")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--huge-pages", action="store_true")
    p.add_argument("--timeout-seconds", type=int, default=600)
    p.add_argument("--control-port", type=int, default=0, help="TCPStore port; default MASTER_PORT+1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--check-validator",
        action="store_true",
        help="inject one bad word per active receiver before the suite; require detection",
    )
    return p


def cases(args):
    world = args.ep_size
    if not 2 <= world <= 256 or not 1 <= args.aiv <= 64:
        raise ValueError("EP must be 2..256 and AIV must be 1..64")
    if math.ceil((world - 1) / args.peers_per_worker) > min(args.aiv, 32):
        raise ValueError("peer ownership exceeds available AIV/QP lanes")
    if args.warmup < 0 or args.iterations < 1 or args.timeout_seconds < 1 or args.first_device < 0:
        raise ValueError("invalid iteration, timeout or device count")
    if not 0 <= args.control_port <= 65535:
        raise ValueError("control port must be 0..65535")
    for values, limit in ((args.bytes_per_peer, 1 << 30), (args.chunk_bytes, 1 << 20)):
        if len(values) != len(set(values)) or any(n <= 0 or n > limit or n % 32 for n in values):
            raise ValueError("byte sizes must be unique positive multiples of 32 within capacity limits")
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("duplicate modes")
    deltas = args.deltas if args.deltas is not None else list(range(1, world))
    if len(deltas) != len(set(deltas)) or any(not 0 < d < world for d in deltas):
        raise ValueError("deltas must be unique and in 1..EP-1")
    patterns = []
    for mode in args.modes:
        if mode == "pair":
            pairs = (
                (src, dst)
                for src in range(world)
                for dst in range(world)
                if src != dst and (args.pair_scope == "all" or src == 0 or dst == 0)
            )
            patterns.extend(dict(mode=mode, source=src, destination=dst, delta=0) for src, dst in pairs)
        elif mode == "permutation":
            patterns.extend(dict(mode=mode, source=0, destination=0, delta=d) for d in deltas)
        else:
            patterns.append(dict(mode=mode, source=0, destination=0, delta=0))
    return [
        dict(pattern, bytes=n, chunk=chunk)
        for pattern in patterns
        for n in args.bytes_per_peer
        for chunk in args.chunk_bytes
    ]


def edges(case, world):
    if case["mode"] == "pair":
        return [(case["source"], case["destination"])]
    if case["mode"] == "permutation":
        return [(r, (r + case["delta"]) % world) for r in range(world)]
    return [(src, dst) for src in range(world) for dst in range(world) if src != dst]


def sample_metrics(raw):
    ticks = raw["ticks"]
    if not ticks or len(ticks) % 8:
        raise ValueError("invalid native timing record")
    cores = [ticks[i : i + 8] for i in range(0, len(ticks), 8)]
    if any(any(row[i + 1] < row[i] for i in range(5)) for row in cores):
        raise ValueError("non-monotonic device timestamps")
    # Only timestamps on the same rank/device are subtracted. Different cards'
    # system counters are NOT assumed to share an epoch.
    elapsed = (max(r[5] for r in cores) - min(r[0] for r in cores)) / 1000.0
    if elapsed <= 0:
        raise ValueError("non-positive device elapsed time")
    names = ["setup_us", "post_us", "quiet_us", "incoming_us", "local_join_us"]
    result = {name: max((r[i + 1] - r[i]) / 1000.0 for r in cores) for i, name in enumerate(names)}
    result.update(elapsed_us=elapsed, mismatched_words=raw["mismatched_words"])
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(case, world, ranks):
    if len(ranks) != world or {r["rank"] for r in ranks} != set(range(world)):
        raise ValueError("missing or duplicate ranks")
    n = len(ranks[0]["samples"])
    if not n or any(len(r["samples"]) != n for r in ranks):
        raise ValueError("ranks have inconsistent sample counts")
    graph = edges(case, world)
    active = {r for edge in graph for r in edge}
    active_rows = [r for r in ranks if r["rank"] in active]
    durations = [max(r["samples"][i]["elapsed_us"] for r in active_rows) for i in range(n)]
    errors = sum(s["mismatched_words"] for r in ranks for s in r["samples"])
    mean_us = statistics.mean(durations)
    total_bytes = len(graph) * case["bytes"]
    senders = len({src for src, _ in graph})
    result = dict(
        case,
        ep_size=world,
        active_senders=senders,
        directed_edges=len(graph),
        total_remote_bytes=total_bytes,
        mean_us=mean_us,
        median_us=statistics.median(durations),
        p95_us=percentile(durations, 0.95),
        max_us=max(durations),
        aggregate_GBps=total_bytes / mean_us / 1000.0,
        per_sender_GBps=total_bytes / senders / mean_us / 1000.0,
        mismatched_words=errors,
    )
    # Pair-mode bandwidth is link throughput, not divided by the idle EP group.
    result["slowest_rank"] = max(active_rows, key=lambda r: statistics.mean(s["elapsed_us"] for s in r["samples"]))[
        "rank"
    ]
    return result


def fingerprint(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def runtime_libraries():
    """Record actual loaded files, not an inferred LD_LIBRARY_PATH selection."""
    names = ("libshmem", "libhcomm", "liburma", "libascendcl", "libruntime", "libhccl", "aclshmem_bootstrap")
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            if path.name.startswith(names) and path.is_file():
                paths.add(path)
    return [dict(path=str(path), sha256=fingerprint(path)) for path in sorted(paths)]


def main(argv=None):
    args = parser().parse_args(argv)
    suite = cases(args)
    config = {k: v for k, v in vars(args).items() if k not in ("native_dir", "output", "dry_run")}
    plan = dict(
        config=config,
        cases=suite,
        case_count=len(suite),
        initializations_per_rank=1,
        capacity_bytes_per_peer=max(args.bytes_per_peer),
        workspace_bytes=args.ep_size * (512 + 2 * max(args.bytes_per_peer)),
        qp_slots=32,
        qps_per_worker=1,
        shared_jetty=True,
        timing="max active-rank device elapsed per sample; entry/exit global barriers excluded",
        bandwidth="actual remote payload bytes / time; decimal GB/s; self excluded",
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    import torch
    import torch_npu  # noqa: F401

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    device = args.first_device + int(os.environ["LOCAL_RANK"])
    if world != args.ep_size:
        raise ValueError("--ep-size must equal torchrun WORLD_SIZE")
    torch.npu.set_device(device)
    control = Control(rank, world, args.control_port, args.timeout_seconds)
    native_dir = args.native_dir.resolve()
    sys.path.insert(0, str(native_dir))
    native = importlib.import_module("_udma_microbench")
    metadata = dict(
        rank=rank,
        host=socket.gethostname(),
        logical_device=device,
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        device_name=torch.npu.get_device_name(device),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        native_sha256=fingerprint(native.__file__),
        kernel_sha256=fingerprint(native_dir / "device/libudma_bench_kernel.so"),
        shmem_root=os.environ.get("SHMEM_ROOT"),
        ascend_home_path=os.environ.get("ASCEND_HOME_PATH"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    local_path = args.output / f"rank{rank}.jsonl"
    states = [None] * world
    control.all_gather_object(states, dict(config=config, metadata=metadata, exists=local_path.exists()))
    if any(s["config"] != config for s in states):
        raise ValueError("rank configurations differ")
    for key in ("native_sha256", "kernel_sha256"):
        if len({s["metadata"][key] for s in states}) != 1:
            raise ValueError(f"different {key} across ranks")
    if any(s["exists"] for s in states):
        raise FileExistsError("choose a new --output directory; existing rank logs are preserved")
    if rank == 0:
        plan["ranks"] = [s["metadata"] for s in states]
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    uid = [native.unique_id() if rank == 0 else None]
    control.broadcast_object_list(uid, src=0)
    print(f"INITIALIZING rank={rank} device={device} cases={len(suite)}", flush=True)
    session = native.Session(
        rank,
        world,
        max(args.bytes_per_peer),
        args.aiv,
        args.peers_per_worker,
        args.wqes_per_visit,
        uid[0],
        args.huge_pages,
    )
    print(f"INITIALIZED rank={rank} count=1", flush=True)
    metadata["runtime_libraries"] = runtime_libraries()
    environments = [None] * world
    control.all_gather_object(environments, metadata)
    if rank == 0:
        plan["ranks"] = environments
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    epoch = 0
    summaries = []
    with local_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps(dict(metadata=metadata, initializations=1)) + "\n")
        if args.check_validator:
            epoch += 1
            raw = session.sample(2, 0, 0, 0, 32, 32, epoch, True)
            detected = [None] * world
            control.all_gather_object(detected, raw["mismatched_words"])
            if detected != [1] * world:
                raise RuntimeError(f"validator injection failed: {detected}")
            print(f"VALIDATOR_PASSED rank={rank}", flush=True)
        for case_id, case in enumerate(suite):
            samples = []
            if rank == 0:
                print(f"CASE_BEGIN {case_id + 1}/{len(suite)} {case}", flush=True)
            for iteration in range(args.warmup + args.iterations):
                epoch += 1
                raw = session.sample(
                    MODES[case["mode"]],
                    case["source"],
                    case["destination"],
                    case["delta"],
                    case["bytes"],
                    case["chunk"],
                    epoch,
                )
                metrics = sample_metrics(raw)
                record = dict(case_id=case_id, case=case, iteration=iteration - args.warmup, rank=rank, **raw)
                record.update(metrics)
                log.write(json.dumps(record) + "\n")
                log.flush()
                if metrics["mismatched_words"]:
                    raise RuntimeError(f"payload mismatch rank={rank} case={case_id}; see {local_path}")
                if iteration >= args.warmup:
                    samples.append(metrics)
            gathered = [None] * world if rank == 0 else None
            control.gather_object(dict(rank=rank, samples=samples), gathered, dst=0)
            if rank == 0:
                summary = dict(case_id=case_id, **summarize(case, world, gathered))
                summaries.append(summary)
                with (args.output / "samples.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(case_id=case_id, case=case, ranks=gathered)) + "\n")
                with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(summary))
                    writer.writeheader()
                    writer.writerows(summaries)
                print(
                    f"CASE_PASSED {case_id + 1}/{len(suite)} "
                    f"per_sender_GBps={summary['per_sender_GBps']:.3f} "
                    f"mean_us={summary['mean_us']:.3f} p95_us={summary['p95_us']:.3f}",
                    flush=True,
                )
    session.close()
    control.barrier()
    if rank == 0:
        (args.output / "report.json").write_text(
            json.dumps(
                dict(plan=plan, summaries=summaries, precision_passed=True, initializations_per_rank=1, destroyed=True),
                indent=2,
            )
            + "\n"
        )
    control.finish()
    print(f"SUITE_PASSED rank={rank} cases={len(suite)} initializations=1", flush=True)


if __name__ == "__main__":
    main()
