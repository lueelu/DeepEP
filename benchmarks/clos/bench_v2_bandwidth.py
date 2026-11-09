"""One-command-per-node ElasticBuffer accuracy + stage/bandwidth benchmark.

python benchmarks/clos/bench_v2_bandwidth.py --output ./bandwidth_results

Each dtype/expand/AIV configuration gets a fresh torchrun group. No kernel or
public API instrumentation is installed. Precision is checked before profiling.
Both APIs use the default library scheduling policy; this entrypoint exposes
workload options, not manual scheduling knobs.
Defaults cover BF16/FP8 and expanded/non-expanded operations at 32 AIVs.
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

from stage_metrics import L2CacheFlush, L2_CACHE_FLUSH_BYTES, ProfileSamples, PROFILE_GUARD_ITERS

from route_workloads import (
    DEFAULTS as ROUTE_DEFAULTS,
    add_arguments as add_route_arguments,
    validate as validate_routes,
    statistics as route_statistics,
)

from stage_metrics import (
    LIMITS,
    V1_LIMITS,
    byte_ledger,
    file_identity,
    parse_profile,
    route_counts,
    summarize,
    write_csv,
    write_json,
)
from bandwidth_multinode import PeerGroup, collection_manifest, content_identity, library_identity, safe_path

REPO = Path(__file__).resolve().parents[2]


def arguments(argv=None, *, description=None):
    p = argparse.ArgumentParser(
        description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.set_defaults(udma_auto_policy=True)
    p.add_argument(
        "--flush-l2-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evict L2 with a reusable 256 MB buffer before every timed API call (default: enabled)",
    )
    p.add_argument(
        "--combine-bias",
        choices=("none", "two"),
        default="none",
        help="bias inputs for weighted combine only (default: none; two reproduces older benchmarks)",
    )
    p.add_argument("--ep-size", type=int, choices=(2, 4, 8, 16, 32, 64), default=8)
    p.add_argument(
        "--nnodes",
        type=int,
        choices=(1, 2, 4, 8),
        default=None,
        help="equal local rank count on each node; at most 16 devices per node",
    )
    p.add_argument("--node-rank", type=int, choices=range(8), default=0)
    p.add_argument("--master-addr", help="node 0 reachable IPv4 address/hostname; required for multi-node execution")
    p.add_argument("--master-port", type=int, default=29500, help="node coordination port; torchrun uses the next port")
    p.add_argument("--run-id", help="same unique suite name on all nodes; required for multi-node execution")
    p.add_argument("--rendezvous-timeout", type=int, default=300)
    p.add_argument("--collection-timeout", type=int, default=600)
    p.add_argument("--first-device", type=int, default=0)
    p.add_argument(
        "--aiv",
        type=int,
        choices=range(2, 65, 2),
        nargs="+",
        default=[32],
    )
    p.add_argument("--dtypes", choices=("bf16", "fp8"), nargs="+", default=["bf16", "fp8"])
    p.add_argument("--expand", type=int, choices=(0, 1), nargs="+", default=[0, 1])
    for name, default in [
        ("num-tokens", 8192),
        ("hidden", 6144),
        ("num-topk", 8),
        ("experts-per-rank", 32),
        ("warmup", 10),
        ("iterations", 30),
        ("seed", 20260823),
        ("case-timeout", 1800),
    ]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument(
        "--huge-pages",
        action="store_true",
        help="request 1 GiB SHMEM pages; eligible FP8 dispatches also use packed scale staging",
    )
    p.add_argument(
        "--output", type=Path, default=Path("bandwidth_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    p.add_argument("--report-only", type=Path, help="recompute an existing run; never launches NPU work")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    add_route_arguments(p)
    args = p.parse_args(argv)
    if args.nnodes is None:
        args.nnodes = max(1, args.ep_size // 16)
    try:
        validate_routes(args)
    except ValueError as exc:
        p.error(str(exc))
    if any(
        getattr(args, n) <= 0
        for n in (
            "num_tokens",
            "hidden",
            "num_topk",
            "experts_per_rank",
            "iterations",
            "case_timeout",
            "rendezvous_timeout",
            "collection_timeout",
        )
    ):
        p.error("shape, iterations and timeout must be positive")
    if args.warmup < 0 or args.first_device < 0 or args.hidden % 128:
        p.error("warmup/first-device must be nonnegative; hidden must be a multiple of 128")
    if args.num_topk > min(128, args.ep_size * args.experts_per_rank):
        p.error("num-topk exceeds available experts or the supported bound")
    if any(len(v) != len(set(v)) for v in (args.aiv, args.dtypes, args.expand)):
        p.error("duplicate configurations are not allowed")
    if not args.report_only:
        if args.ep_size % args.nnodes or not 1 <= local_world_size(args) <= 16 or args.node_rank >= args.nnodes:
            p.error("nnodes must divide ep-size with 1..16 local ranks; node-rank must be below nnodes")
        if not 1 <= args.master_port <= 65534:
            p.error("master-port must leave room for master-port + 1")
        if args.nnodes > 1:
            if (
                not args.master_addr
                or args.master_addr.strip().lower() in ("localhost", "0.0.0.0", "::1")
                or args.master_addr.startswith("127.")
            ):
                p.error("multi-node execution needs a reachable node 0 --master-addr, not loopback")
            if not args.run_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.run_id):
                p.error("multi-node execution needs the same --run-id (1..64 letters/digits/_/-) on all nodes")
    return args


def local_world_size(args):
    return args.ep_size // args.nnodes


def suite_contract(args):
    return {
        k: v for k, v in vars(args).items() if k not in ("node_rank", "first_device", "output", "report_only", "worker")
    }


def worker_command(args, case, path, *, entrypoint=None):
    command = [sys.executable, "-m", "torch.distributed.run", "--max-restarts=0"]
    if args.nnodes == 1:
        command.append("--standalone")
    else:
        command += [
            f"--nnodes={args.nnodes}",
            f"--node-rank={args.node_rank}",
            "--rdzv-backend=static",
            f"--rdzv-id={args.run_id}_{case['name']}",
            f"--master-addr={args.master_addr}",
            f"--master-port={args.master_port + 1}",
            f"--rdzv-conf=timeout={args.rendezvous_timeout}",
        ]
    command += [
        f"--nproc-per-node={local_world_size(args)}",
        str(Path(entrypoint or __file__).resolve()),
        "--worker",
        "--aiv",
        str(case["aiv"]),
        "--dtypes",
        case["dtype"],
        "--expand",
        str(case["expand"]),
        "--output",
        str(path),
    ]
    for key in (
        "ep_size",
        "nnodes",
        "node_rank",
        "first_device",
        "num_tokens",
        "hidden",
        "num_topk",
        "experts_per_rank",
        "warmup",
        "iterations",
        "seed",
        "case_timeout",
        "rendezvous_timeout",
        "collection_timeout",
        "master_port",
    ):
        value = getattr(args, key)
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    for key in ROUTE_DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    if args.nnodes > 1:
        command += ["--master-addr", args.master_addr, "--run-id", args.run_id]
    if args.huge_pages:
        command.append("--huge-pages")
    command.append("--flush-l2-cache" if args.flush_l2_cache else "--no-flush-l2-cache")
    command += ["--combine-bias", args.combine_bias]
    return command


def git_identity():
    def git(*args):
        r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return dict(
        revision=git("rev-parse", "HEAD"),
        dirty=git("status", "--porcelain"),
        golden=file_identity(REPO / "tests/clos/common.py"),
        sources=[
            file_identity(p)
            for folder in ("deep_ep", "csrc", "kernels", "include", "benchmarks")
            for p in sorted((REPO / folder).rglob("*"))
            if p.is_file() and p.suffix in (".py", ".cpp", ".h")
        ],
    )


def loaded_identity():
    from deep_ep.backends import direct as clos
    import torch
    import torch_npu

    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split()
        if (
            len(parts) >= 6
            and ".so" in parts[-1]
            and any(
                tag in parts[-1]
                for tag in ("deep_ep", "deepep", "shmem", "libascendcl", "libruntime", "libhcomm", "libhccl")
            )
        ):
            path = Path(parts[-1])
            if not path.is_file():
                raise RuntimeError(f"loaded library disappeared: {path}")
            paths.add(path.resolve())
    if not any("kernel" in p.name for p in paths) or not any("shmem" in p.name for p in paths):
        raise RuntimeError("cannot identify loaded kernel/SHMEM libraries")
    return dict(
        pid=os.getpid(),
        python=sys.version,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        python_module=file_identity(clos.__file__),
        libraries=[file_identity(p) for p in sorted(paths)],
    )


def check_idle(text, first_device, ep_size):
    """Require explicit health and empty process-table entries; unknown is not idle."""
    health = {
        int(m[0]): m[1] for m in re.findall(r"^\|\s*(\d+)\s*\|\s*Ascend[^|]*\|\s*(OK|Warning|Alarm)\s*\|", text, re.M)
    }
    idle = {int(n) for n in re.findall(r"No running processes found in NPU\s+(\d+)", text)}
    for device in range(first_device, first_device + ep_size):
        if device not in idle or health.get(device) not in ("OK", "Warning"):
            raise RuntimeError(f"NPU {device} is busy, unhealthy or cannot be verified; no job started")


def effective_udma_config(buffer, config=None, *, num_aiv=None):
    """Report resolved runtime values rather than caller defaults (V1 uses Config)."""
    result = {
        name: getattr(buffer, name)
        for name in (
            "udma_schedule",
            "udma_dispatch_peer_order",
            "udma_combine_peer_order",
            "udma_ranks_per_group",
            "udma_peers_per_group_per_worker",
            "udma_workers_per_peer",
            "udma_qps_per_worker",
        )
    }
    policy = buffer.udma_auto_policy
    if policy is not None:
        if num_aiv is None:
            num_aiv = config.num_aiv if config is not None else 32
        policy = policy.for_num_aiv(num_aiv)
    for name in ("udma_peers_per_worker", "udma_wqes_per_visit"):
        result[name] = (
            getattr(policy, name.removeprefix("udma_"))
            if policy is not None
            else getattr(buffer if config is None else config, name)
        )
    result["udma_auto_policy"] = buffer.udma_auto_policy is not None
    return result


def parse_profile_collective(dist, error_path, *args, **kwargs):
    """Every live rank reports parsing success/failure before caller timing.

    This handles Python parsing errors, not a lost process/device. Preserve the
    original traceback locally and report all failing ranks to every participant.
    """
    profile, error = None, None
    try:
        profile = parse_profile(*args, **kwargs)
    except Exception:
        error = traceback.format_exc()
        try:
            write_json(error_path, dict(rank=dist.get_rank(), traceback=error))
        except Exception:
            error += "\nFailed to save parse error:\n" + traceback.format_exc()
    statuses = [None] * dist.get_world_size()
    dist.all_gather_object(statuses, dict(rank=dist.get_rank(), error=error))
    failures = [s for s in statuses if s["error"]]
    if failures:
        details = "\n".join(f"rank {s['rank']}: {s['error']}" for s in failures)
        raise RuntimeError("profiler parsing failed across ranks; caller timing cancelled:\n" + details)
    return profile


def worker(args):
    # Imports follow argument handling so --help/report-only need no NPU runtime.
    # Use the installed wheel; prepending the checkout can hide its native
    # extensions. Only the independent CPU reference comes from this tree.
    sys.path.insert(0, str(REPO / "tests/clos"))
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from deep_ep.backends.global_clos import ElasticBuffer
    from deep_ep import topk_idx_t
    import common as golden
    import profiling

    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != args.ep_size:
        raise RuntimeError("torchrun world size does not match EP size")
    local_rank = int(os.environ["LOCAL_RANK"])
    if (
        int(os.environ["LOCAL_WORLD_SIZE"]) != local_world_size(args)
        or not 0 <= local_rank < local_world_size(args)
        or rank != args.node_rank * local_world_size(args) + local_rank
    ):
        raise RuntimeError("torchrun local/global rank mapping does not match this node")
    args.num_aiv, args.dtype, expanded = args.aiv[0], args.dtypes[0], bool(args.expand[0])
    args.expert_alignment = 1
    args.profile_warmup, args.profile_iters = args.warmup, args.iterations
    args.profile_sample_markers = True
    args.combine_notify = True
    device = args.first_device + local_rank
    destination = args.output / f"rank{rank}"
    destination.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write_json(
        destination / "process.json",
        dict(pid=os.getpid(), rank=rank, device=device, node_rank=args.node_rank, started=started),
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
        buffer = ElasticBuffer(
            dist.group.WORLD,
            num_max_tokens_per_rank=args.num_tokens,
            hidden=args.hidden,
            num_topk=args.num_topk,
            use_fp8_dispatch=args.dtype == "fp8",
            explicitly_destroy=True,
            transport="udma",
            use_1g_huge_pages=args.huge_pages,
        )
        args.effective_udma_config = effective_udma_config(buffer, num_aiv=args.num_aiv)
        if rank == 0:
            print("EFFECTIVE_UDMA_CONFIG", json.dumps(args.effective_udma_config, sort_keys=True), flush=True)
        if getattr(args, "udma_auto_policy", False):
            args.resolved_udma_policy = asdict(buffer.udma_auto_policy.for_num_aiv(args.num_aiv))
            if rank == 0:
                print("AUTO_POLICY", json.dumps(args.resolved_udma_policy, sort_keys=True), flush=True)
        identity = loaded_identity()
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
        dispatch_ledger = byte_ledger(
            counts,
            rank,
            tokens=args.num_tokens,
            hidden=args.hidden,
            topk=args.num_topk,
            dtype=args.dtype,
            idx_bytes=idx.element_size(),
            op="dispatch",
        )
        if input_bytes != args.num_tokens * dispatch_ledger["official_logical_bytes_per_row"]:
            raise ValueError("logical row-byte formula does not match actual dispatch input tensors")

        def wait(result):
            profiling._consume_profile_result(result)
            return result

        def dispatch(value, handle=None):
            return buffer.dispatch(
                value,
                topk_idx=idx if handle is None else None,
                topk_weights=weights if handle is None else None,
                handle=handle,
                num_experts=args.ep_size * args.experts_per_rank,
                num_sms=args.num_aiv,
                do_expand=expanded,
                expert_alignment=1,
                async_with_compute_stream=True,
            )

        def check_dispatch(result, delta=0, cached=False):
            recv, recv_idx, recv_weights, handle, _ = result
            ref = golden._dispatch_reference(rank, args, delta=delta)
            if expanded:
                exp = golden._expand_reference(
                    ref["recv_x"], ref["recv_topk_idx"], ref["recv_topk_weights"], ref["expert_counts"]
                )
                expected = exp["expanded_x"]
                expected_weights = exp["expanded_topk_weights"]
                if recv_idx is not None:
                    raise AssertionError("expanded output unexpectedly has rank topk indices")
                offsets, actual, mapping, aligned = handle._expand_layout
                for name, value, wanted in [
                    ("offsets", offsets, exp["expert_offsets"]),
                    ("expert counts", actual, exp["actual_expert_counts"]),
                    ("expand mapping", mapping, exp["recv_to_expanded_idx"]),
                ]:
                    golden._assert_equal(name, value, wanted)
                if aligned != ref["expert_counts"].tolist():
                    raise AssertionError("aligned expert counts mismatch")
            else:
                expected, expected_weights = ref["recv_x"], ref["recv_topk_weights"]
                if not cached:
                    golden._assert_equal("dispatch indices", recv_idx, ref["recv_topk_idx"])
                elif recv_idx is not None:
                    raise AssertionError("payload-only cached dispatch returned topk")
            golden._assert_equal("dispatch payload/scales", recv, expected)
            if not cached:
                golden._assert_equal("dispatch weights", recv_weights, expected_weights)
            elif recv_weights is not None:
                raise AssertionError("payload-only cached dispatch returned weights")
            for name, value, wanted in [
                ("src_idx", handle._dispatch_layout[3], ref["recv_src_idx"]),
                ("send_head", handle._dispatch_layout[5], ref["send_head"]),
                ("recv prefix", handle._dispatch_layout[2], ref["recv_channel_prefix"]),
            ]:
                golden._assert_equal(name, value, wanted)
            expected_rows = sum(counts["rows"][p][rank] for p in range(args.ep_size))
            if handle.num_recv_tokens != expected_rows:
                raise AssertionError("independent route count disagrees with actual receive rows")
            # Return independent CPU expert input, never use kernel output as golden.
            return golden._dequantize_x(expected, torch.bfloat16)

        fresh = wait(dispatch(x))
        expert_ref = check_dispatch(fresh)
        recv, _, recv_weights, handle, _ = fresh
        owned = golden._dequantize_x(recv, torch.bfloat16).clone()
        golden._assert_equal("expert input", owned, expert_ref)
        del expert_ref
        cached = wait(dispatch(cached_x, handle))
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
            return buffer.combine(
                owned,
                handle,
                topk_weights=recv_weights if weighted else None,
                bias=bias if weighted else None,
                num_sms=args.num_aiv,
                async_with_compute_stream=True,
            )

        # Reproduce BF16 rounding at the expert aggregate boundary before rank reduction.
        source = golden._dequantize_x(golden._make_dispatch_x(rank, args), torch.bfloat16)
        for weighted in (False, True):
            result = wait(combine(weighted))
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
                combine_unweighted=True,
                combine_weighted=True,
                combine_bias=args.combine_bias,
                combine_weighted_two_bias=args.combine_bias == "two",
            ),
        )
        print(f"PRECISION_PASSED rank={rank}", flush=True)
        operators = {}
        flush_cache = L2CacheFlush(torch, buffer, args.flush_l2_cache)
        functions = dict(
            dispatch=lambda: dispatch(x),
            **{"cached-dispatch": lambda: dispatch(cached_x, handle)},
            combine_unweighted=lambda: combine(False),
            combine_weighted=lambda: combine(True),
        )

        def align_combine_ranks():
            # Host SHMEM rendezvous after local device completion; outside all timers.
            torch.npu.synchronize()
            buffer.runtime.barrier()

        for op, fn in functions.items():
            dist.barrier()
            before_iteration = align_combine_ranks if op.startswith("combine") else None
            print(f"PROFILE_BEGIN rank={rank} op={op} host_sync_each_iteration=True", flush=True)
            args.profile_dir = str(destination / op / "profiler")
            try:
                profiling._profile_operator(
                    rank,
                    args,
                    op,
                    flush_cache.wrap(fn),
                    sample_callback=ProfileSamples(torch),
                    guard_iters=PROFILE_GUARD_ITERS,
                    synchronize_each_iteration=True,
                    before_iteration=before_iteration,
                )
            except profiling._ProfileComplete:
                pass
            profile = parse_profile_collective(
                dist,
                destination / op / "profile_error.json",
                args.profile_dir,
                op,
                expanded,
                args.iterations,
                device,
                args.num_aiv,
                flush_l2_cache=args.flush_l2_cache,
                sample_markers=True,
                require_combine_notify=True,
            )
            # Independent caller-stream samples; do not subtract from profiler kernel times.
            profile["sampling"]["host_sync_each_iteration"] = True
            dist.barrier()
            for _ in range(args.warmup):
                if before_iteration is not None:
                    before_iteration()
                flush_cache()
                wait(fn())
                torch.npu.synchronize()
            torch.npu.synchronize()
            caller = []
            for _ in range(args.iterations):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                if before_iteration is not None:
                    before_iteration()
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
        if library_identity(loaded_identity()) != library_identity(identity):
            raise RuntimeError("loaded libraries changed during benchmark")
        buffer.destroy()
        torch.npu.synchronize()
        write_json(
            destination / "result.json",
            dict(
                rank=rank,
                device=device,
                precision_passed=True,
                destroyed=True,
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
        print(f"CASE_PASSED rank={rank}", flush=True)
    except BaseException:
        write_json(destination / "exit.json", dict(code=1, ended=time.time(), traceback=traceback.format_exc()))
        raise


def rank_directory(case_path, rank, plan):
    node = rank // (plan["arguments"]["ep_size"] // plan["arguments"].get("nnodes", 1))
    if node:
        return case_path / f"peer_node{node}" / f"rank{rank}"
    return case_path / f"rank{rank}"


def verify_collection(root):
    manifest = json.loads((root / "collection.json").read_text(encoding="utf-8"))
    if not manifest["passed"] or not manifest["files"]:
        raise ValueError("peer collection incomplete")
    seen = set()
    for item in manifest["files"]:
        if item["relative"] in seen:
            raise ValueError("duplicate peer artifact")
        seen.add(item["relative"])
        actual = file_identity(safe_path(root, item["relative"]))
        if any(actual[k] != item[k] for k in ("size", "sha256")):
            raise ValueError(f"peer artifact changed: {item['relative']}")


def report(root):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    api = plan.get("api", "v2")
    if api not in ("v1", "v2"):
        raise ValueError("unknown benchmark API")
    limits = V1_LIMITS if api == "v1" else LIMITS
    nnodes = plan["arguments"].get("nnodes", 1)
    dual = nnodes > 1
    if dual and plan["arguments"]["node_rank"] != 0:
        raise ValueError("the complete report is in the node0 output directory")
    rows, raw_samples, kernels, artifacts, completed = [], [], [], [], []
    workloads, policies, effective_configs = {}, {}, {}
    expected_route_hash = None
    for case in plan["cases"]:
        path = root / case["name"]
        for node in range(1, nnodes):
            verify_collection(path / f"peer_node{node}")
        independent_counts = route_counts(
            [
                json.loads((rank_directory(path, rank, plan) / "input_routes.json").read_text(encoding="utf-8"))
                for rank in range(plan["arguments"]["ep_size"])
            ],
            plan["arguments"]["experts_per_rank"],
        )
        raw_routes = [
            json.loads((rank_directory(path, rank, plan) / "input_routes.json").read_text(encoding="utf-8"))
            for rank in range(plan["arguments"]["ep_size"])
        ]
        workload = route_statistics(
            raw_routes, plan["arguments"]["experts_per_rank"], plan["arguments"].get("route_group_size")
        )
        workloads[case["name"]] = workload
        if "route_mode" in plan["arguments"]:
            if expected_route_hash is not None and workload["route_sha256"] != expected_route_hash:
                raise ValueError("routing changed between dtype/expand/AIV cases")
            expected_route_hash = workload["route_sha256"]
        records = []
        for rank in range(plan["arguments"]["ep_size"]):
            rank_dir = rank_directory(path, rank, plan)
            record = json.loads((rank_dir / "result.json").read_text(encoding="utf-8"))
            ex = json.loads((rank_dir / "exit.json").read_text(encoding="utf-8"))
            if ex["code"] != 0 or not record["precision_passed"] or not record["destroyed"] or record["rank"] != rank:
                raise ValueError(f"{rank_dir}: failed rank")
            config = record["config"]
            effective = config.get("effective_udma_config")
            if effective is not None:
                if case["name"] in effective_configs and effective_configs[case["name"]] != effective:
                    raise ValueError("ranks resolved different effective UDMA configurations")
                effective_configs[case["name"]] = effective
            for key in ROUTE_DEFAULTS:
                if config.get(key) != plan["arguments"].get(key):
                    raise ValueError(f"route configuration mismatch: {key}")
            if config.get("api", "v2") != api:
                raise ValueError("mixed V1/V2 rank results")
            if (config["num_aiv"], config["dtype"], config["expand"][0]) != (
                case["aiv"],
                case["dtype"],
                case["expand"],
            ):
                raise ValueError("case configuration mismatch")
            for key, default in (
                ("peers_per_worker", None),
                ("wqes_per_visit", None),
                ("dispatch_peer_order", "rank"),
                ("combine_peer_order", "rank"),
                ("ranks_per_group", None),
                ("peers_per_group_per_worker", None),
                ("workers_per_peer", 1),
                ("qps_per_worker", 1),
                ("udma_schedule", "peer"),
                ("sweep_plan_sha256", None),
                ("udma_auto_policy", False),
                ("flush_l2_cache", False),
                ("combine_bias", "two"),
                ("profile_sample_markers", False),
                ("combine_notify", False),
            ):
                if config.get(key, default) != plan["arguments"].get(key, default):
                    raise ValueError(f"peer order configuration mismatch: {key}")
            if plan["arguments"].get("udma_auto_policy", False):
                resolved = config.get("resolved_udma_policy")
                if not resolved or resolved.get("world_size") != plan["arguments"]["ep_size"]:
                    raise ValueError("missing or invalid resolved automatic policy")
                if case["name"] in policies and policies[case["name"]] != resolved:
                    raise ValueError("ranks resolved different automatic policies")
                policies[case["name"]] = resolved
            for key in (
                "ep_size",
                "num_tokens",
                "hidden",
                "num_topk",
                "experts_per_rank",
                "iterations",
                "warmup",
                "seed",
                "huge_pages",
            ):
                if config[key] != plan["arguments"][key]:
                    raise ValueError(f"rank configuration mismatch: {key}")
            if dual:
                local_size = plan["arguments"]["ep_size"] // nnodes
                node = rank // local_size
                peer_plan = root / ("peer_plan.json" if node == 1 else f"peer_plan_node{node}.json")
                node_args = (plan if node == 0 else json.loads(peer_plan.read_text(encoding="utf-8")))["arguments"]
                if (
                    config["node_rank"] != node
                    or config["run_id"] != plan["arguments"]["run_id"]
                    or record["device"] != node_args["first_device"] + rank % local_size
                ):
                    raise ValueError("node/rank/device identity mismatch")
            actual_routes = file_identity(rank_dir / "input_routes.json")
            if any(actual_routes[k] != record["input_routes"][k] for k in ("size", "sha256")):
                raise ValueError("input routes changed")
            if record["route_counts"] != independent_counts:
                raise ValueError("byte counts disagree with independently reconstructed input routing")
            if set(record["operators"]) != {"dispatch", "cached-dispatch", "combine_unweighted", "combine_weighted"}:
                raise ValueError("missing operator measurement")
            for data in record["operators"].values():
                if len(data["caller_us"]) != config["iterations"] or not all(
                    0 < v < float("inf") for v in data["caller_us"]
                ):
                    raise ValueError("invalid caller-stream samples")
            if records and record["route_counts"] != records[0]["route_counts"]:
                raise ValueError("rank route ledgers disagree")
            if records and library_identity(record["identity"]) != library_identity(records[0]["identity"]):
                raise ValueError("ranks loaded different libraries")
            for op, data in record["operators"].items():
                for artifact in data["profile"]["artifacts"]:
                    # Resolve under the current report root, so copied bundles can be re-reviewed.
                    original = Path(artifact["path"]).resolve()
                    suffix = original.relative_to((Path(config["output"]) / f"rank{rank}").resolve())
                    local = safe_path(rank_dir, suffix.as_posix())
                    actual = file_identity(local)
                    if (actual["size"], actual["sha256"]) != (artifact["size"], artifact["sha256"]):
                        raise ValueError(f"raw artifact changed: {local}")
                    artifacts.append(dict(case=case["name"], rank=rank, op=op, **actual))
                data["profile"] = parse_profile(
                    rank_dir / op / "profiler",
                    op,
                    bool(case["expand"]),
                    config["iterations"],
                    record["device"],
                    case["aiv"],
                    api=api,
                    flush_l2_cache=config.get("flush_l2_cache", False),
                    sample_markers=config.get("profile_sample_markers", False),
                    require_combine_notify=config.get("combine_notify", False),
                )
                import_size = plan["topk_idx_bytes"]
                expected = byte_ledger(
                    record["route_counts"],
                    rank,
                    tokens=config["num_tokens"],
                    hidden=config["hidden"],
                    topk=config["num_topk"],
                    dtype=case["dtype"],
                    idx_bytes=import_size,
                    op=op,
                )
                recorded_contract = expected
                if plan.get("format_version", 1) == 1:
                    recorded_contract = {k: v for k, v in expected.items() if not k.startswith("official_")}
                if data["bytes"] != recorded_contract:
                    raise ValueError("byte ledger changed")
                # The v1 report format predated the official logical numerator;
                # derive it from verified raw routes, without rerunning NPU work.
                data["bytes"] = expected
                raw_samples.extend(dict(case=case["name"], rank=rank, op=op, **s) for s in data["profile"]["samples"])
                kernels.extend(dict(case=case["name"], rank=rank, op=op, **s) for s in data["profile"]["kernels"])
            records.append(record)
        for launcher in [path] + [path / f"peer_node{node}" for node in range(1, nnodes)]:
            status = json.loads((launcher / "launcher_exit.json").read_text(encoding="utf-8"))
            if status["code"] != 0 or status.get("suite_aborted"):
                raise ValueError("nonzero launcher exit")
        rows.extend(
            dict(
                case=case["name"],
                aiv=case["aiv"],
                dtype=case["dtype"],
                expanded=case["expand"],
                flush_l2_cache=plan["arguments"].get("flush_l2_cache", False),
                l2_flush_bytes=L2_CACHE_FLUSH_BYTES if plan["arguments"].get("flush_l2_cache", False) else 0,
                combine_bias=plan["arguments"].get("combine_bias", "two"),
                bias_count=(
                    2 if plan["arguments"].get("combine_bias", "two") == "two" and r["op"] == "combine_weighted" else 0
                ),
                route_sha256=workload["route_sha256"],
                recv_max_over_mean=workload["recv_max_over_mean"],
                single_contribution_fraction=workload["single_contribution_fraction"],
                local_group_row_fraction=workload["local_group_row_fraction"],
                **r,
            )
            for r in summarize(records)
        )
        completed.append(case["name"])
    write_csv(root / "summary.csv", rows)
    write_csv(root / "samples.csv", raw_samples)
    write_csv(root / "kernels.csv", kernels)
    write_csv(root / "artifacts.csv", artifacts)
    write_json(
        root / "report.json",
        dict(
            passed=True,
            api=api,
            completed=completed,
            limits=limits,
            profile_sample_markers=plan["arguments"].get("profile_sample_markers", False),
            flush_l2_cache=plan["arguments"].get("flush_l2_cache", False),
            l2_flush_bytes=L2_CACHE_FLUSH_BYTES if plan["arguments"].get("flush_l2_cache", False) else 0,
            combine_bias=plan["arguments"].get("combine_bias", "two"),
            summary=rows,
            route_statistics=workloads,
            resolved_udma_policies=policies,
            effective_udma_configs=effective_configs,
        ),
    )
    for name, workload in workloads.items():
        print(
            "ROUTE_LOAD",
            name,
            json.dumps(
                {
                    k: workload[k]
                    for k in (
                        "route_sha256",
                        "recv_max_over_mean",
                        "single_contribution_fraction",
                        "local_group_row_fraction",
                        "rank_over_other_mean",
                        "group_over_other_mean",
                    )
                },
                sort_keys=True,
            ),
        )
    print(
        f"L2_CACHE_FLUSH enabled={plan['arguments'].get('flush_l2_cache', False)} "
        f"bytes={L2_CACHE_FLUSH_BYTES if plan['arguments'].get('flush_l2_cache', False) else 0}; "
        "before each full API call; excluded from all reported API times/bandwidth"
    )
    print(f"COMBINE_BIAS weighted={plan['arguments'].get('combine_bias', 'two')} unweighted=none")
    print(
        "Numerator: DeepEP-Ascend logical receive bytes INCLUDING self. "
        "Denominator: this implementation comm kernel (not identical official profiling)."
    )
    if api == "v1":
        print(
            "V1 combine uses a fused transfer+reduce kernel. Its comm/reduce-only fields are N/A; "
            "logical-fused GB/s INCLUDES reduce and is not official communication bandwidth."
        )
    columns = [
        "comm_us",
        "notify_us",
        "expand_us",
        "aggregate_us",
        "reduce_us",
        "official_logical_GBps",
        "remote_payload_GBps",
        "device_sum_us",
        "algorithm_device_sum_GBps",
    ]
    if api == "v1":
        columns += ["layout_us", "fused_combine_us", "combine_with_reduce_us", "logical_fused_combine_GBps"]
    print("rank (0=single rank, all=whole EP group) / case / operator / " + " / ".join(columns))
    for row in rows:
        if row["rank"] in (0, "all"):
            print(
                f"{row['rank']} / {row['case']} / {row['op']} / "
                + " / ".join("N/A" if row[k] is None else f"{row[k]:.3f}" for k in columns)
            )
    if policies:
        print_auto_summary(rows, plan["arguments"]["ep_size"], policies)
    print(f"REPORT_PASSED {root / 'report.json'}")


def print_auto_summary(rows, ep_size, policies):
    print("\nAUTO V2: logical GB/s per rank equivalent = whole-EP logical GB/s / EP; includes self.")
    print(
        "Bandwidth denominator is comm_us only; notify/reduce are separate. "
        "Times are mean of per-sample max-rank durations, in us; absent stages = 0."
    )
    for case, policy in policies.items():
        print("AUTO_POLICY", case, json.dumps(policy, sort_keys=True))
    print("case / operator / GBps_per_rank / comm_us / notify_us / reduce_us / layout_us / aggregate_us / expand_us")
    for row in rows:
        if row["rank"] != "all":
            continue
        bw = row["official_logical_GBps"]
        values = [None if bw is None else bw / ep_size] + [
            row[k] for k in ("comm_us", "notify_us", "reduce_us", "layout_us", "aggregate_us", "expand_us")
        ]
        print(
            f"{row['case']} / {row['op']} / "
            + " / ".join("N/A" if value is None else f"{value:.3f}" for value in values)
        )


def stop_child(child):
    """Only the process group created by this launcher; never match process names."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The group can still contain workers after its torchrun leader has exited.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def require_ready(peer, stage, error=None):
    if peer:
        peer.require_both(stage, error)
    elif error:
        raise RuntimeError(f"{stage}: {error}")


def run_case(args, case, path, environment, peer, *, entrypoint=None):
    command = worker_command(args, case, path, entrypoint=entrypoint)
    write_json(path / "command.json", dict(argv=command, started=time.time()))
    print(f"RUN {case['name']} node={args.node_rank} (log: {path / 'run.log'})", flush=True)
    child = None
    try:
        with (path / "run.log").open("w", encoding="utf-8") as log:
            error = None
            try:
                child = subprocess.Popen(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            except Exception:
                error = traceback.format_exc()
            require_ready(peer, case["name"] + "/launch", error)
            started = time.monotonic()
            tick = 0
            while True:
                code = child.poll()
                state = dict(code=code, timed_out=time.monotonic() - started >= args.case_timeout)
                other = peer.exchange(f"{case['name']}/poll/{tick}", state) if peer else state
                if any(s["timed_out"] or s["code"] not in (None, 0) for s in (state, other)):
                    raise RuntimeError(f"case failed or timed out: local={state}, peer={other}")
                if code == 0 and other["code"] == 0:
                    break
                time.sleep(0.5)
                tick += 1
    except BaseException:
        if child is not None:
            stop_child(child)
        write_json(
            path / "launcher_exit.json",
            dict(
                code=None if child is None else child.returncode,
                error=traceback.format_exc(),
                ended=time.time(),
                suite_aborted=True,
            ),
        )
        raise
    write_json(path / "launcher_exit.json", dict(code=0, ended=time.time()))


def launch(args, *, entrypoint=None):
    if os.name != "posix":
        raise RuntimeError("NPU execution requires Linux; --report-only is platform independent")
    if os.environ.get("SHMEM_UID_SESSION_ID"):
        raise RuntimeError("unset SHMEM_UID_SESSION_ID for automatic per-job UID endpoints")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("unset ASCEND_RT_VISIBLE_DEVICES; use local physical IDs via --first-device")

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("benchmark interrupted")

    signal.signal(signal.SIGTERM, terminate)
    # Different output paths cannot accidentally create two benchmark suites in this checkout.
    import fcntl
    import tempfile
    import hashlib

    lock_name = hashlib.sha256(str(REPO).encode()).hexdigest()[:16]
    lock = open(Path(tempfile.gettempdir()) / f"deepep_bandwidth_{lock_name}.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output = args.output.resolve()
    if args.nnodes > 1:
        args.output = args.output / f"node{args.node_rank}"
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from deep_ep import topk_idx_t

    cases = [
        dict(name=f"{dtype}_expand{expand}_aiv{aiv}", dtype=dtype, expand=expand, aiv=aiv)
        for dtype, expand, aiv in itertools.product(args.dtypes, args.expand, args.aiv)
    ]
    api = getattr(args, "api", "v2")
    args.profile_sample_markers = True
    args.combine_notify = api == "v2"
    plan = dict(
        format_version=2,
        api=api,
        cases=cases,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        identity=git_identity(),
        topk_idx_bytes=torch.empty((), dtype=topk_idx_t).element_size(),
        limits=V1_LIMITS if api == "v1" else LIMITS,
    )
    plan["source_identity"] = content_identity(plan["identity"], REPO)
    write_json(args.output / "plan.json", plan)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    peer = None
    try:
        if args.nnodes > 1:
            print(
                f"WAIT node={args.node_rank} run={args.run_id} coordinator={args.master_addr}:{args.master_port}",
                flush=True,
            )
            peer = PeerGroup.connect(
                args.master_addr, args.master_port, args.node_rank, args.rendezvous_timeout, args.nnodes
            )
            hello = dict(
                node_rank=args.node_rank,
                contract=suite_contract(args),
                source=plan["source_identity"],
                topk_idx_bytes=plan["topk_idx_bytes"],
            )
            hellos = peer.exchange_all("hello", hello)
            for node, other in enumerate(hellos):
                if other["node_rank"] != node:
                    raise ValueError("node identity mismatch")
                for key in ("contract", "source", "topk_idx_bytes"):
                    if json.dumps(hello[key], sort_keys=True) != json.dumps(other[key], sort_keys=True):
                        raise ValueError(f"node {node} {key} mismatch; no NPU job started")
            plans = peer.exchange_all("plan", plan)
            for node, other_plan in enumerate(plans):
                if node != args.node_rank:
                    write_json(
                        args.output / ("peer_plan.json" if node == 1 else f"peer_plan_node{node}.json"), other_plan
                    )
        for case in cases:
            path = args.output / case["name"]
            error = None
            try:
                path.mkdir()
                resource = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30, check=True)
                (path / "resources.before.txt").write_text(resource.stdout, encoding="utf-8")
                check_idle(resource.stdout, args.first_device, local_world_size(args))
            except Exception:
                error = traceback.format_exc()
            # Both machines must be ready before either launches NPU workers.
            require_ready(peer, case["name"] + "/preflight", error)
            run_case(args, case, path, environment, peer, entrypoint=entrypoint)
            if peer:
                peer.connection.settimeout(args.collection_timeout)
                error, manifest = None, None
                try:
                    start = args.node_rank * local_world_size(args)
                    manifest = collection_manifest(path, range(start, start + local_world_size(args)))
                except Exception:
                    error = traceback.format_exc()
                require_ready(peer, case["name"] + "/collection_ready", error)
                peer.collect(path, manifest, args.collection_timeout)
                peer.require_both(case["name"] + "/collected")
                peer.connection.settimeout(30)
        error = None
        if peer:
            peer.connection.settimeout(args.collection_timeout)
        try:
            if args.node_rank == 0:
                report(args.output)
        except Exception:
            error = traceback.format_exc()
        require_ready(peer, "final_report", error)
        write_json(
            args.output / "suite_exit.json", dict(code=0, completed=[c["name"] for c in cases], ended=time.time())
        )
        if args.node_rank != 0:
            print("SUITE_PASSED: node 0 has the complete report and copied raw artifacts", flush=True)
    except BaseException:
        write_json(args.output / "suite_exit.json", dict(code=1, traceback=traceback.format_exc(), ended=time.time()))
        raise
    finally:
        if peer:
            peer.close()
        lock.close()


if __name__ == "__main__":
    options = arguments()
    if options.report_only:
        report(options.report_only)
    elif options.worker:
        worker(options)
    else:
        launch(options)
