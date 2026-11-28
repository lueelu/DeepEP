"""Matched fresh/cached dispatch: one Buffer, channel A/B and separate traces.

Run with torchrun. This is a dispatch diagnostic, not a MoE end-to-end benchmark.
"""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, replace
import datetime
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

from bench_fused_notify import store_topology_collective
from stage_metrics import L2CacheFlush, prepare_v2_sample

TRACE_ENV = "DEEPEP_CLOS_DISPATCH_DEVICE_TIMING"
CHANNEL_ENV = "DEEPEP_CLOS_CACHED_CHANNEL"
COMPACT_ENV = "DEEPEP_CLOS_COMBINE_COMPACT_SELF"
METADATA_ENV = "DEEPEP_CLOS_DISPATCH_CHANNEL_METADATA"
METADATA_TILE_ENV = "DEEPEP_CLOS_NOTIFY_METADATA_TILE_TOKENS"


class DispatchBenchmarkPolicy:
    """Per-launch benchmark override; never changes the library's default policy."""

    def __init__(self, base):
        self.base = base
        self.variant = "auto"

    def for_launch(self, num_aiv, row_bytes, operation, **kwargs):
        plan = self.base.for_launch(num_aiv, row_bytes, operation, **kwargs)
        if self.variant == "auto":
            return plan
        if (
            operation != "dispatch"
            or self.base.world_size != 8
            or num_aiv != 8
            or row_bytes != 5632
            or not self.base.topology_known
            or self.base.ranks_per_group != 8
            or not kwargs.get("bf16")
            or kwargs.get("expanded")
            or not kwargs.get("v2", True)
        ):
            raise ValueError("zeng dispatch comparison requires contiguous EP8/AIV8/H2816/BF16/non-expanded V2")
        plan = replace(
            plan,
            peer_order=0,
            peers_per_worker=2,
            workers_per_peer=1,
            qps_per_worker=1,
            wqes_per_visit=32,
            self_copy=0,
            weight_mode=0,
        )
        plan.validate_num_aiv(num_aiv)
        return plan


OVERLAP_ENVS = (
    "DEEPEP_CLOS_DISPATCH_OVERLAP_SELF",
    "DEEPEP_CLOS_DISPATCH_EARLY_SLOTS",
    "DEEPEP_CLOS_DISPATCH_SKIP_TOPK_INIT",
    "DEEPEP_CLOS_NOTIFY_METADATA_DOUBLE_BUFFER",
)
# "off" is retained for older commands, but only disables optional work.
# Send-before-self is mandatory in every mode, including this legacy alias.
OVERLAP_MODES = {"off": 1, "self": 1, "slots": 3, "send": 3, "init": 5, "notify": 9, "all": 15}


def set_overlap_mode(mode):
    mask = OVERLAP_MODES[mode]
    for bit, name in enumerate(OVERLAP_ENVS):
        os.environ[name] = str((mask >> bit) & 1)


METADATA_OUTPUT_ENVS = (
    "DEEPEP_CLOS_CACHED_WEIGHT_PACK",
    "DEEPEP_CLOS_DISPATCH_METADATA_DIRECT",
    "DEEPEP_CLOS_CACHED_OMIT_TOPK_OUTPUT",
)
METADATA_OUTPUT_MODES = {"off": 0, "pack": 1, "direct": 2, "omit": 4, "pack_direct": 3, "all": 7}


def set_metadata_output_mode(mode):
    mask = METADATA_OUTPUT_MODES[mode]
    for bit, name in enumerate(METADATA_OUTPUT_ENVS):
        os.environ[name] = str((mask >> bit) & 1)


def offset_weights(weights, offset):
    """One tensor addition, identical on device inputs and the CPU reference."""
    return None if weights is None else weights + offset


@contextmanager
def benchmark_session(buffer, group):
    """Collectively destroy only after successful work on every rank."""
    try:
        yield
    finally:
        set_metadata_output_mode("off")
        set_overlap_mode("off")
        os.environ[TRACE_ENV] = "0"
        os.environ["DEEPEP_CLOS_COMBINE_DEVICE_TIMING"] = "0"
        os.environ[COMPACT_ENV] = "0"
        os.environ[METADATA_ENV] = "0"
        os.environ[METADATA_TILE_ENV] = "0"
    # An exception thrown through yield exits after finally, skipping this
    # collective and destroy. Never introduce a second device wait on failure.
    group.agree("ready_to_destroy")
    buffer.destroy()


def run_worker(entrypoint):
    """Preserve the first failure and let torchrun tear down this failed job.

    Raising normally can run native Buffer/profiler destructors or atexit hooks
    while peers are in a different collective. Keep the traceback (and its live
    frame resources), flush it, then exit this worker without Python teardown.
    This is a CLI failure path only; successful runs explicitly destroy Buffer.
    """
    try:
        entrypoint()
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            raise
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        if isinstance(error, SystemExit) and isinstance(error.code, int) and 0 < error.code < 256:
            code = error.code
        try:
            traceback.print_exc()
            sys.stderr.flush()
            sys.stdout.flush()
        finally:
            os._exit(code)


def notify_stage(row):
    if "notify_total_us" in row["stats_us"]:
        return "notify_total_us"
    return "cached_notify_dispatch_kernel" if row["op"].startswith("cached") else "notify_dispatch_kernel"


TIMELINE = (
    "init_done",
    "begin_done",
    "metadata_done",
    "payload_queued",
    "published",
    "incoming_done",
    "quiet_done",
    "sync_done",
    "output_done",
    "handle_done",
    "work_done",
)


def cpu_inputs(torch, args, source, tokens=None, routing="random"):
    tokens = args.tokens if tokens is None else tokens
    gen = torch.Generator().manual_seed(20261122 + source)
    x = (torch.randn((tokens, args.hidden), generator=gen) * 0.01).to(torch.bfloat16)
    idx = torch.rand((tokens, args.experts), generator=gen).topk(args.topk, dim=1).indices
    weights = torch.rand((tokens, args.topk), generator=gen)
    if routing == "masked":
        idx[::7] = -1
        idx[:, -1] = -1
    elif routing == "invalid":
        idx.fill_(-1)
    elif routing == "skew":
        idx = torch.arange(args.topk).remainder(args.experts).expand(tokens, -1).clone()
    return x, idx, weights


def check_cpu_reference(
    torch,
    actual,
    args,
    rank,
    world,
    sizes=None,
    routing="random",
    cached=False,
    src_idx=None,
    handle=None,
    weight_offset=0,
):
    """Derive receive rows from original routes, never from Notify/handle output."""
    recv, recv_idx, recv_weights = actual
    local_experts = args.experts // world
    first = 0
    source_metadata = None if handle is None else handle.recv_src_metadata.cpu()
    for source in range(world):
        x, idx, weights = cpu_inputs(torch, args, source, None if sizes is None else sizes[source], routing)
        hits = (idx >= rank * local_experts) & (idx < (rank + 1) * local_experts)
        selected = hits.any(1)
        n = int(selected.sum())
        torch.testing.assert_close(recv[first : first + n], x[selected], rtol=0, atol=0)
        if src_idx is not None:
            torch.testing.assert_close(
                src_idx[first : first + n].to(torch.int64), torch.where(selected)[0], rtol=0, atol=0
            )
        expected_idx = (idx[selected] - rank * local_experts).masked_fill(~hits[selected], -1)
        if not cached:
            torch.testing.assert_close(recv_idx[first : first + n].to(torch.int64), expected_idx, rtol=0, atol=0)
        if recv_weights is not None:
            torch.testing.assert_close(
                recv_weights[first : first + n],
                offset_weights(weights[selected], weight_offset).masked_fill(~hits[selected], 0),
                rtol=0,
                atol=0,
            )
        if source_metadata is not None:
            expected_source = torch.full((n, args.topk + 2), -1, dtype=source_metadata.dtype)
            expected_source[:, 0] = source * handle.num_max_tokens_per_rank + torch.where(selected)[0]
            expected_source[:, 1] = source * args.topk + hits[selected].to(torch.int64).argmax(1)
            torch.testing.assert_close(source_metadata[first : first + n], expected_source, rtol=0, atol=0)
        first += n
    if handle is not None:
        _, local_idx, _ = cpu_inputs(torch, args, rank, None if sizes is None else sizes[rank], routing)
        expected_slots = torch.full(local_idx.shape, -1, dtype=torch.int32)
        peer_ordinals = [0] * world
        for token, experts in enumerate(local_idx.tolist()):
            seen = set()
            for column, expert in enumerate(experts):
                if not 0 <= expert < args.experts:
                    continue
                peer = expert // local_experts
                if peer in seen:
                    continue
                seen.add(peer)
                expected_slots[token, column] = rank * handle.num_max_tokens_per_rank + peer_ordinals[peer]
                peer_ordinals[peer] += 1
        torch.testing.assert_close(handle.dst_buffer_slot_idx.cpu(), expected_slots, rtol=0, atol=0)
    if first != len(recv):
        raise AssertionError("receive row count differs from the independent routing reference")


def channel_comparison(results):
    """Summarize matched off/on turns; never add independently aggregated stages."""
    comparisons = []

    def comparison_key(r):
        return (
            r["op"],
            r["cores"],
            r["channel"],
            r.get("metadata_tile_requested", 32),
            r.get("dispatch_plan", "auto"),
            r.get("overlap_mode", "off"),
            r.get("metadata_output_mode", "off"),
        )

    for key in sorted({comparison_key(r) for r in results}):
        selected = [r for r in results if comparison_key(r) == key]
        off = [r for r in selected if r["metadata_requested"] == 0]
        on = [r for r in selected if r["metadata_requested"] == 1]
        if not off or not on:
            continue
        if len(off) != len(on) or any(r["policy"] != off[0]["policy"] for r in selected):
            raise ValueError("unmatched metadata comparison")
        row = dict(
            op=key[0],
            cores=key[1],
            cached_channel=key[2],
            metadata_tile_requested=key[3],
            dispatch_plan=key[4],
            overlap_mode=key[5],
            metadata_output_mode=key[6],
            pairs=len(off),
        )
        notify = notify_stage(off[0])
        for label, stage in (("notify", notify), ("comm", "direct_dispatch_udma_kernel"), ("api", "api_wall_us")):
            a = [r["stats_us"][stage]["mean"] for r in off]
            b = [r["stats_us"][stage]["mean"] for r in on]
            row.update(
                {
                    label + "_off_us": statistics.mean(a),
                    label + "_on_us": statistics.mean(b),
                    label + "_on_min_us": min(b),
                    label + "_on_max_us": max(b),
                    label + "_time_change_pct": (statistics.mean(b) / statistics.mean(a) - 1) * 100,
                    label + "_pair_delta_min_us": min(y - x for x, y in zip(a, b)),
                    label + "_pair_delta_max_us": max(y - x for x, y in zip(a, b)),
                }
            )
        payload = off[0]["payload_bytes_per_rank"]
        row["off_gbps"] = payload / row["comm_off_us"] / 1000
        row["on_gbps"] = payload / row["comm_on_us"] / 1000
        comparisons.append(row)
    return comparisons


def overlap_comparison(results, mode_key="overlap_mode", effective_key="overlap_effective"):
    comparisons = []

    other_mode = "metadata_output_mode" if mode_key == "overlap_mode" else "overlap_mode"
    keys = ("op", "cores", "channel", "metadata_requested", "metadata_tile_requested", "dispatch_plan", other_mode)

    def key(row):
        return tuple(row.get(name, "off") for name in keys)

    for config in sorted({key(row) for row in results}):
        selected = [row for row in results if key(row) == config]
        off = [row for row in selected if row.get(mode_key, "off") == "off"]
        if not off:
            continue
        for mode in sorted({row.get(mode_key, "off") for row in selected} - {"off"}):
            on = [row for row in selected if row[mode_key] == mode]
            if len(on) != len(off) or any(row["policy"] != off[0]["policy"] for row in off + on):
                raise ValueError("unmatched overlap comparison")
            row = dict(zip(keys, config))
            row.update(mode=mode, pairs=len(on), effective_masks=sorted({r[effective_key] for r in on}))
            notify = notify_stage(off[0])
            for label, stage in (("notify", notify), ("comm", "direct_dispatch_udma_kernel"), ("api", "api_wall_us")):
                a = [r["stats_us"][stage]["mean"] for r in off]
                b = [r["stats_us"][stage]["mean"] for r in on]
                row[label + "_off_us"] = statistics.mean(a)
                row[label + "_on_us"] = statistics.mean(b)
                row[label + "_time_change_pct"] = (statistics.mean(b) / statistics.mean(a) - 1) * 100
                row[label + "_pair_delta_min_us"] = min(y - x for x, y in zip(a, b))
                row[label + "_pair_delta_max_us"] = max(y - x for x, y in zip(a, b))
            payload = off[0]["payload_bytes_per_rank"]
            row["off_gbps"] = payload / row["comm_off_us"] / 1000
            row["on_gbps"] = payload / row["comm_on_us"] / 1000
            comparisons.append(row)
    return comparisons


def decode_trace(raw, fields, rank, cores, cached, whole_channel, cycles_per_us, operation=0):
    if len(raw) != cores or cycles_per_us <= 0:
        raise ValueError("trace core count/clock mismatch")
    result = []
    for core, values in enumerate(raw):
        if len(values) < len(fields):
            raise ValueError("truncated trace")
        row = dict(zip(fields, values))
        if row["version"] not in (1, 2) or [row[k] for k in ("rank", "core", "cached", "whole_channel")] != [
            rank,
            core,
            int(cached),
            int(whole_channel),
        ]:
            raise ValueError("trace identity or effective whole-channel gate mismatch")
        if row["version"] == 2 and row.get("operation") != operation:
            raise ValueError("trace operation mismatch")
        # Send-first dispatch publishes SIMT metadata after payload submission.
        # Retain both legal orders; do not silently accept other timestamp inversions.
        timeline = list(TIMELINE)
        metadata_after_payload = operation == 0 and row["metadata_done"] > row["payload_queued"]
        if metadata_after_payload:
            timeline[2], timeline[3] = timeline[3], timeline[2]
        stamps = [row[k] for k in timeline]
        if stamps != sorted(stamps) or stamps[0] < 0 or not row["entry_cycles"]:
            raise ValueError("invalid trace timeline")
        previous = 0
        intervals = {}
        for name in timeline:
            intervals[name + "_interval_us"] = (row[name] - previous) / cycles_per_us
            previous = row[name]
        intervals.update(
            {
                key + "_us": row[key] / cycles_per_us
                for key in ("src_cycles", "topk_cycles", "self_cycles", "list_cycles")
            }
        )
        queue = {}
        if row["version"] == 2:
            for name in (
                "descriptor_cycles",
                "wqe_build_cycles",
                "wqe_submit_cycles",
                "credit_cycles",
                "credit_blocked_cycles",
                "credit_max_cycles",
            ):
                intervals[name + "_us"] = row[name] / cycles_per_us
            if (
                row["credit_blocked"] > row["credit_checks"]
                or row["short_flushes"] > row["flushes"]
                or row["metadata_wqes"] > row["wqes"]
                or row["metadata_flushes"] > row["flushes"]
                or row["wqes"] > row["flushes"] * 32
                or row["wqes"] < row["flushes"]
                or row["credit_blocked_cycles"] > row["credit_cycles"]
            ):
                raise ValueError("inconsistent queue counters")
            if row["self_end"] < row["self_begin"] or row["self_end"] > row["work_done"]:
                raise ValueError("invalid local-copy interval")
            if not row["published"] <= row["incoming_begin"] <= row["incoming_done"]:
                raise ValueError("invalid incoming-wait interval")
            intervals["incoming_wait_us"] = (row["incoming_done"] - row["incoming_begin"]) / cycles_per_us
            intervals["quiet_wait_us"] = (row["quiet_done"] - row["incoming_done"]) / cycles_per_us
            intervals["final_sync_us"] = (row["sync_done"] - row["quiet_done"]) / cycles_per_us
            intervals["self_begin_us"] = row["self_begin"] / cycles_per_us
            intervals["self_end_us"] = row["self_end"] / cycles_per_us
            for name in ("topk_init_cycles", "source_cycles", "slots_cycles", "final_quiet_cycles"):
                if name in row:
                    intervals[name + "_us"] = row[name] / cycles_per_us
            queue = dict(
                metadata_after_payload=metadata_after_payload,
                mean_batch_wqes=row["wqes"] / row["flushes"] if row["flushes"] else 0,
                short_batch_fraction=row["short_flushes"] / row["flushes"] if row["flushes"] else 0,
                blocked_admission_fraction=row["credit_blocked"] / row["credit_checks"] if row["credit_checks"] else 0,
                mean_software_sq_used=row["sq_used_sum"] / row["credit_checks"] if row["credit_checks"] else 0,
                sender=bool(row["payload_peers"]),
            )
        result.append(dict(raw=row, intervals=intervals, queue=queue))
    return result


def capture_stages(directory, samples, cached, cached_weighted=False, cached_pack=False):
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError("expected exactly one kernel_details.csv")
    with paths[0].open(encoding="utf-8-sig", newline="") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: float(row["Start Time(us)"]))
    from stage_metrics import normalize_kernel_rows

    rows = normalize_kernel_rows(rows)
    names = ["cached_notify_combine_kernel"]  # out-of-measurement alignment
    layout = (
        "dispatch_layout_simt_kernel"
        if any(row["Name"] == "dispatch_layout_simt_kernel" for row in rows)
        else "dispatch_layout_kernel"
    )
    if not cached:
        names.append(layout)
    notify = "cached_notify_dispatch_kernel" if cached else "notify_dispatch_kernel"
    # Version 3 prepares metadata in dispatch after payload submission. Cached
    # weighted packing therefore uses the same alignment-only notify kernel.
    names.append(notify)
    if cached_weighted and not cached_pack:
        names.append("cached_dispatch_plan_kernel")
    names.append("direct_dispatch_udma_kernel")
    selected = [
        row
        for row in rows
        if row["Name"]
        in set(names)
        | {
            "cached_dispatch_plan_kernel",
            "cached_notify_dispatch_packed_kernel",
            "cached_notify_dispatch_kernel",
            "dispatch_layout_kernel",
            "dispatch_layout_simt_kernel",
        }
    ]
    if [row["Name"] for row in selected] != names * (samples + 4):
        raise RuntimeError("unexpected kernel sequence/sample count; retain raw capture for inspection")
    result = {
        name: [float(row["Duration(us)"]) for row in selected[i :: len(names)]][2:-2]
        for i, name in enumerate(names)
        if i
    }
    if any(not math.isfinite(t) or t <= 0 for values in result.values() for t in values):
        raise ValueError("invalid profiler duration")
    if layout == "dispatch_layout_simt_kernel" and not cached:
        result["dispatch_layout_kernel"] = result.pop(layout)
    # Aggregate within each rank/sample before slow-rank aggregation, including legacy head rebuilding.
    result["notify_total_us"] = [
        value + result.get("cached_dispatch_plan_kernel", [0] * samples)[i] for i, value in enumerate(result[notify])
    ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--cores", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--cached-channel-modes", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--metadata-modes", type=int, nargs="+", choices=[0, 1], default=[0])
    parser.add_argument(
        "--metadata-tile-tokens",
        type=int,
        nargs="+",
        choices=[0, 32, 64, 128, 256],
        default=[0],
        help="0: automatic 128-token tile with UB fallback; reuse one Buffer across choices",
    )
    parser.add_argument(
        "--dispatch-plans",
        nargs="+",
        choices=["auto", "zeng"],
        default=["auto"],
        help="zeng: EP8/AIV8/H2816 rank/P2/W1/Q1/visit32; metadata path unchanged",
    )
    parser.add_argument(
        "--overlap-modes",
        nargs="+",
        choices=list(OVERLAP_MODES),
        default=["send"],
        help="send-first in every mode; off only disables optional slots/init/notify optimizations",
    )
    parser.add_argument("--metadata-output-modes", nargs="+", choices=list(METADATA_OUTPUT_MODES), default=["off"])
    parser.add_argument("--trace-warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--trace-samples", type=int, default=3, help="0 for formal timing only (without combine diagnostics)"
    )
    parser.add_argument(
        "--combine-diagnostics", action="store_true", help="also trace ordinary combine in the same Buffer"
    )
    parser.add_argument("--reduce-cores", type=int, default=48)
    parser.add_argument("--combine-only", action="store_true", help="skip dispatch timing; reuse its precision setup")
    parser.add_argument("--compact-weight-modes", type=int, nargs="+", choices=[0, 1], default=[0])
    parser.add_argument(
        "--compact-self-modes",
        type=int,
        nargs="+",
        choices=[0, 1],
        default=[1],
        help="legacy local-copy modes; ordinary V2 UDMA now reads self input directly in both modes",
    )
    parser.add_argument("--flush-l2", action="store_true")
    parser.add_argument("--huge-pages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-port", type=int, default=29881)
    parser.add_argument("--shmem-port", type=int, default=29882)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.combine_only:
        args.combine_diagnostics = True
    if (
        min(args.tokens, args.hidden, args.topk, args.experts, args.samples, args.reduce_cores, *args.cores) <= 0
        or min(args.warmup, args.trace_samples, args.trace_warmup) < 0
    ):
        parser.error("invalid shape/core/sample count")
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep.backends.direct import buffer as bm, elastic as em
    from deep_ep.backends.direct.elastic import ElasticBuffer
    from deep_ep.backends.direct.dtypes import topk_idx_t
    from deep_ep.backends.direct._native import load_native_module
    from bench_fast_sweep import StoreGroup
    from bench_v2_bandwidth import loaded_identity
    from bandwidth_multinode import library_identity

    rank, world, device = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if not 2 <= world <= 256 or args.experts % world or args.topk > args.experts:
        parser.error("this experiment requires EP2..256 and experts divisible by EP")
    if "zeng" in args.dispatch_plans and (world != 8 or args.cores != [8] or args.hidden != 2816):
        parser.error("zeng dispatch comparison requires EP8, --cores 8 and --hidden 2816")
    torch.npu.set_device(device)
    native = load_native_module()
    if getattr(native, "clos_sender_pipeline_version", 0) != 4:
        raise RuntimeError("rebuild version-4 host and kernel libraries for cached-plan channel packing")
    if args.trace_samples and getattr(native, "dispatch_device_timing_version", 0) != 2:
        raise RuntimeError("rebuild host and kernel libraries for device traces")
    if args.combine_diagnostics and args.trace_samples and getattr(native, "combine_device_timing_version", 0) != 2:
        raise RuntimeError("combine device traces require version-2 host/kernel libraries")
    if 1 in args.metadata_modes and getattr(native, "dispatch_channel_metadata_version", 0) != 4:
        raise RuntimeError("rebuild host and kernel libraries for channel metadata")
    if any(mode != "off" for mode in args.overlap_modes) and getattr(native, "dispatch_overlap_version", 0) != 1:
        raise RuntimeError("rebuild host and kernel libraries for overlap experiments")
    if (
        any(mode != "off" for mode in args.metadata_output_modes)
        and getattr(native, "dispatch_metadata_output_version", 0) != 2
    ):
        raise RuntimeError("rebuild host and kernels for metadata output experiments")
    set_overlap_mode("off")
    set_metadata_output_mode("off")
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
    # Explicitly isolate these changes from earlier experimental metadata/notify options.
    for key in (
        TRACE_ENV,
        "DEEPEP_CLOS_COMBINE_DEVICE_TIMING",
        "DEEPEP_CLOS_COMBINE_COMPACT_WEIGHTS",
        CHANNEL_ENV,
        COMPACT_ENV,
        METADATA_ENV,
        METADATA_TILE_ENV,
        "DEEPEP_CLOS_DISPATCH_SRC_IDX_256",
        "DEEPEP_CLOS_DISPATCH_TOPK_256",
        "DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY",
        "DEEPEP_CLOS_NOTIFY_HOST_PREP",
        "DEEPEP_CLOS_NOTIFY_DEFER_PLAN",
        "DEEPEP_CLOS_DISPATCH_HOST_TIMING",
    ):
        os.environ[key] = "0"
    with store_topology_collective(group):
        buffer = ElasticBuffer(
            group,
            combine_notify=True,
            num_max_tokens_per_rank=args.tokens,
            hidden=args.hidden,
            num_topk=args.topk,
            explicitly_destroy=True,
            transport="udma",
            udma_auto_policy=True,
            use_1g_huge_pages=args.huge_pages,
            shmem_ip_port=f"tcp://{os.environ['MASTER_ADDR']}:{args.shmem_port}",
        )
    flush = L2CacheFlush(torch, buffer, args.flush_l2)
    launch_policy = DispatchBenchmarkPolicy(buffer._udma_auto_policy)
    buffer._udma_auto_policy = launch_policy
    actual_huge_pages = bool(buffer.runtime.use_1g_huge_pages())
    if not all(group.gather(actual_huge_pages == args.huge_pages)):
        raise RuntimeError("native huge-page state differs from requested configuration")
    identity = loaded_identity()
    identities = group.gather(library_identity(identity))
    if any(item != identities[0] for item in identities):
        raise RuntimeError("ranks loaded different library bytes")
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                args=vars(args),
                rank=rank,
                world=world,
                buffer_initializations=1,
                identity=identity,
                actual_huge_pages=actual_huge_pages,
                bandwidth="decimal GB/s, deduplicated BF16 payload including self, excluding metadata",
                aggregation="per-iteration slowest rank, then mean/P50/P95; API and profiler separate passes",
            ),
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )

    def done(result):
        if result[-1].event is not None:
            result[-1].current_stream_wait()
        torch.npu.synchronize()
        return result

    def snapshot(result):
        recv, idx, weight = result[:3]
        return [
            recv.cpu().clone(),
            None if idx is None else idx.cpu().clone(),
            None if weight is None else weight.cpu().clone(),
        ]

    def consumer_snapshot(dx, di, dw, core, expanded):
        """Exercise owned metadata through expansion, cached reuse and combine."""
        outputs = []
        prepare_v2_sample(buffer, flush)
        fresh = done(
            buffer.dispatch(
                dx, topk_idx=di, topk_weights=dw, num_experts=args.experts, num_sms=core, do_expand=expanded
            )
        )
        handle = fresh[3]
        for generation in range(3):
            result = fresh
            if generation:
                prepare_v2_sample(buffer, flush)
                result = done(
                    buffer.dispatch(
                        dx,
                        handle=handle,
                        topk_weights=offset_weights(dw, generation),
                        num_sms=core,
                        do_expand=expanded,
                    )
                )
            # Cached non-expanded public API deliberately returns no indices.
            outputs.extend(snapshot(result))
            outputs.append(result[3].recv_src_metadata.cpu().clone())
            prepare_v2_sample(buffer, flush)
            combined = done(
                buffer.combine(
                    result[0], result[3], topk_weights=result[2], num_sms=core, reduce_num_sms=args.reduce_cores
                )
            )
            outputs.extend(None if value is None else value.cpu().clone() for value in combined[:2])
        return outputs

    with benchmark_session(buffer, group):
        # Include masked/empty/reused handles for the generic channel/copy paths,
        # even when metadata packing/direct-output experiments are disabled.
        if args.cached_channel_modes:
            precision_cases = []
            for routing in ("masked", "invalid", "skew", "empty_local"):
                sizes = [min(args.tokens, 65)] * world
                if routing == "empty_local":
                    sizes[0] = 0
                cpu_x, cpu_idx, cpu_w = cpu_inputs(torch, args, rank, sizes[rank], routing)
                dx, di, dw = cpu_x.npu(), cpu_idx.to(dtype=topk_idx_t, device="npu"), cpu_w.npu()
                for core in args.cores:
                    for weighted in (False, True):
                        for mode, tile, plan_name, overlap_mode, output_mode, channel_mode in itertools.product(
                            args.metadata_modes,
                            args.metadata_tile_tokens,
                            args.dispatch_plans,
                            args.overlap_modes,
                            args.metadata_output_modes,
                            args.cached_channel_modes,
                        ):
                            group.agree(
                                [
                                    "precision",
                                    routing,
                                    core,
                                    weighted,
                                    mode,
                                    tile,
                                    plan_name,
                                    overlap_mode,
                                    output_mode,
                                    channel_mode,
                                ]
                            )
                            set_overlap_mode(overlap_mode)
                            set_metadata_output_mode(output_mode)
                            launch_policy.variant = plan_name
                            os.environ[METADATA_ENV] = str(mode)
                            os.environ[METADATA_TILE_ENV] = str(tile)
                            os.environ[CHANNEL_ENV] = str(channel_mode)
                            prepare_v2_sample(buffer, flush)
                            result = done(
                                buffer.dispatch(
                                    dx,
                                    topk_idx=di,
                                    topk_weights=dw if weighted else None,
                                    num_experts=args.experts,
                                    num_sms=core,
                                    do_expand=False,
                                )
                            )
                            if bool(native.last_dispatch_channel_metadata()) != bool(mode):
                                raise RuntimeError("precision launch used the wrong channel-metadata mode")
                            check_cpu_reference(
                                torch,
                                snapshot(result),
                                args,
                                rank,
                                world,
                                sizes,
                                routing,
                                src_idx=result[3]._dispatch_layout[3].cpu(),
                                handle=result[3],
                            )
                            # Reuse this handle, but update weights on each cached call.
                            for generation in range(2):
                                cached_w = offset_weights(dw, generation + 1) if weighted else None
                                prepare_v2_sample(buffer, flush)
                                cached_result = done(
                                    buffer.dispatch(
                                        dx, handle=result[3], topk_weights=cached_w, num_sms=core, do_expand=False
                                    )
                                )
                                check_cpu_reference(
                                    torch,
                                    snapshot(cached_result),
                                    args,
                                    rank,
                                    world,
                                    sizes,
                                    routing,
                                    cached=True,
                                    src_idx=result[3]._dispatch_layout[3].cpu(),
                                    handle=result[3],
                                    weight_offset=generation + 1 if weighted else 0,
                                )
                            # Restore fresh state for the tile observation below.
                            prepare_v2_sample(buffer, flush)
                            result = done(
                                buffer.dispatch(
                                    dx,
                                    topk_idx=di,
                                    topk_weights=dw if weighted else None,
                                    num_experts=args.experts,
                                    num_sms=core,
                                    do_expand=False,
                                )
                            )
                            effective_tile = native.last_dispatch_channel_metadata_tile_tokens()
                            if (not mode and effective_tile != 0) or (mode and tile and effective_tile != tile):
                                raise RuntimeError("precision launch used the wrong metadata tile")
                            if output_mode != "off" or args.combine_diagnostics:
                                for expanded in (False, True) if plan_name == "auto" else (False,):
                                    set_metadata_output_mode("off")
                                    os.environ[COMPACT_ENV] = "0"
                                    os.environ["DEEPEP_CLOS_COMBINE_COMPACT_WEIGHTS"] = "0"
                                    expected = consumer_snapshot(dx, di, dw if weighted else None, core, expanded)
                                    set_metadata_output_mode(output_mode)
                                    for compact in args.compact_self_modes:
                                        os.environ[COMPACT_ENV] = str(compact)
                                        for compact_weight in args.compact_weight_modes:
                                            os.environ["DEEPEP_CLOS_COMBINE_COMPACT_WEIGHTS"] = str(compact_weight)
                                            actual = consumer_snapshot(dx, di, dw if weighted else None, core, expanded)
                                            for got, want in zip(actual, expected):
                                                if want is None:
                                                    if got is not None:
                                                        raise AssertionError("optimization changed optional outputs")
                                                else:
                                                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                                    os.environ["DEEPEP_CLOS_COMBINE_COMPACT_WEIGHTS"] = "0"
                                    os.environ[COMPACT_ENV] = "0"
                            precision_cases.append(
                                dict(
                                    routing=routing,
                                    cores=core,
                                    weighted=weighted,
                                    mode=mode,
                                    tile=tile,
                                    effective_tile=effective_tile,
                                    plan=plan_name,
                                    overlap_mode=overlap_mode,
                                    metadata_output_mode=output_mode,
                                    cached_channel=channel_mode,
                                    overlap_effective=(
                                        native.last_dispatch_overlap_options()
                                        if hasattr(native, "last_dispatch_overlap_options")
                                        else 0
                                    ),
                                )
                            )
            group.agree("channel_metadata_precision_passed")
            (out / "precision.json").write_text(json.dumps(precision_cases, indent=2), encoding="utf-8")
        cpu_x, idx, cpu_w = cpu_inputs(torch, args, rank)
        x = cpu_x.npu()
        owners = idx // (args.experts // world)
        local_rows = sum(int((owners == peer).any(1).sum()) for peer in range(world))
        payload = sum(group.gather(local_rows)) * args.hidden * 2 / world
        idx = idx.to(dtype=topk_idx_t, device="npu")
        weights = cpu_w.npu()
        results = []
        for op in () if args.combine_only else ("fresh", "fresh_weighted", "cached", "cached_weighted"):
            configs = [
                (core, flag, metadata, tile, plan_name, overlap_mode, output_mode)
                for core in args.cores
                for flag in args.cached_channel_modes
                for metadata in args.metadata_modes
                for tile in args.metadata_tile_tokens
                for plan_name in args.dispatch_plans
                for overlap_mode in args.overlap_modes
                for output_mode in args.metadata_output_modes
            ]
            if not configs:
                continue
            reference = None
            policies = {}
            for turn, (core, flag, metadata, tile, plan_name, overlap_mode, output_mode) in enumerate(
                configs + configs[::-1]
            ):
                group.agree([op, turn, core, flag, metadata, tile, plan_name, overlap_mode, output_mode])
                set_overlap_mode(overlap_mode)
                set_metadata_output_mode(output_mode)
                torch.npu.synchronize()
                launch_policy.variant = plan_name
                os.environ[CHANNEL_ENV] = str(flag)
                os.environ[METADATA_ENV] = str(metadata)
                os.environ[METADATA_TILE_ENV] = str(tile)
                handle = None

                def invoke():
                    result = done(
                        buffer.dispatch(
                            x,
                            handle=handle,
                            topk_idx=idx if handle is None else None,
                            topk_weights=weights if op.endswith("weighted") else None,
                            num_experts=args.experts,
                            num_sms=core,
                            do_expand=False,
                        )
                    )
                    if getattr(native, "dispatch_channel_metadata_version", 0) >= 1:
                        effective = bool(native.last_dispatch_channel_metadata())
                        if effective != bool(
                            (metadata and handle is None)
                            or (
                                handle is not None
                                and op == "cached_weighted"
                                and METADATA_OUTPUT_MODES[output_mode] & 1
                            )
                        ):
                            raise RuntimeError("native channel-metadata gate did not match this launch")
                        if getattr(native, "dispatch_channel_metadata_version", 0) >= 2:
                            effective_tile = native.last_dispatch_channel_metadata_tile_tokens()
                            if (not effective and effective_tile != 0) or (
                                effective and tile and effective_tile != tile
                            ):
                                raise RuntimeError("native metadata tile did not match this launch")
                    return result

                initial = invoke()
                policy = asdict(initial[3]._udma_policy)
                key = (core, flag, plan_name)
                if key in policies and policies[key] != policy:
                    raise RuntimeError("metadata toggle changed the payload policy")
                policies[key] = policy
                if op.startswith("cached"):
                    handle = initial[3]
                checked_result = invoke()
                checked = snapshot(checked_result)
                if 1 in args.metadata_modes or any(mode != "off" for mode in args.metadata_output_modes):
                    check_cpu_reference(
                        torch,
                        checked,
                        args,
                        rank,
                        world,
                        cached=op.startswith("cached"),
                        src_idx=checked_result[3]._dispatch_layout[3].cpu(),
                        handle=checked_result[3],
                    )
                if reference is None:
                    reference = checked
                else:
                    for actual, expected in zip(checked, reference):
                        if expected is None:
                            assert actual is None
                        else:
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for _ in range(args.warmup):
                    prepare_v2_sample(buffer, flush)
                    invoke()
                api = []
                for _ in range(args.samples):
                    prepare_v2_sample(buffer, flush)
                    start = time.perf_counter_ns()
                    invoke()
                    api.append((time.perf_counter_ns() - start) / 1000)
                capture = (
                    out
                    / f"{op}_round{turn}_aiv{core}_channel{flag}_metadata{metadata}_tile{tile}_{plan_name}_{overlap_mode}_{output_mode}"
                )
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
                stages = capture_stages(
                    capture,
                    args.samples,
                    op.startswith("cached"),
                    cached_weighted=op == "cached_weighted",
                    cached_pack=bool(METADATA_OUTPUT_MODES[output_mode] & 1),
                )
                for actual, expected in zip(snapshot(invoke()), reference):
                    if expected is not None:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                stages["api_wall_us"] = api
                (capture / "samples.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
                gathered = group.gather(stages)
                stats = {}
                for name in stages:
                    values = sorted(max(g[name][i] for g in gathered) for i in range(args.samples))
                    stats[name] = dict(
                        mean=statistics.mean(values),
                        p50=statistics.median(values),
                        p95=values[math.ceil(0.95 * len(values)) - 1],
                    )
                results.append(
                    dict(
                        op=op,
                        turn=turn,
                        cores=core,
                        channel=flag,
                        metadata_requested=metadata,
                        dispatch_plan=plan_name,
                        metadata_output_mode=output_mode,
                        metadata_output_effective=(
                            native.last_dispatch_metadata_options()
                            if hasattr(native, "last_dispatch_metadata_options")
                            else 0
                        ),
                        overlap_mode=overlap_mode,
                        overlap_effective=(
                            native.last_dispatch_overlap_options()
                            if hasattr(native, "last_dispatch_overlap_options")
                            else 0
                        ),
                        metadata_effective=bool(native.last_dispatch_channel_metadata()),
                        metadata_tile_requested=tile,
                        metadata_tile_effective=(
                            native.last_dispatch_channel_metadata_tile_tokens()
                            if getattr(native, "dispatch_channel_metadata_version", 0) >= 2
                            else 0
                        ),
                        policy=policy,
                        stats_us=stats,
                        payload_bytes_per_rank=payload,
                        payload_gbps=payload / stats["direct_dispatch_udma_kernel"]["mean"] / 1000,
                    )
                )
                (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                if args.trace_samples:
                    traces = []
                    diagnostic_api = []
                    fields = list(native.dispatch_device_timing_fields)
                    os.environ[TRACE_ENV] = "1"
                    try:
                        for _ in range(args.trace_warmup):
                            prepare_v2_sample(buffer, flush)
                            invoke()
                        for _ in range(args.trace_samples):
                            prepare_v2_sample(buffer, flush)
                            diagnostic_start = time.perf_counter_ns()
                            traced = invoke()
                            diagnostic_api.append((time.perf_counter_ns() - diagnostic_start) / 1000)
                            tensor = native.take_dispatch_device_timing()
                            if tensor is None:
                                raise RuntimeError("device trace missing")
                            # invoke has consumed the completion event; no asynchronous CPU read.
                            raw = tensor.cpu().tolist()
                            counters = [dict(zip(fields, row)) for row in raw]
                            effective_outputs = (
                                native.last_dispatch_metadata_options()
                                if hasattr(native, "last_dispatch_metadata_options")
                                else 0
                            )
                            eliminated = []
                            if effective_outputs & 2:
                                eliminated.extend(("src_output_bytes", "weight_output_bytes"))
                            if effective_outputs & 4:
                                eliminated.append("topk_output_bytes")
                            if any(row.get(name, 0) for row in counters for name in eliminated):
                                raise RuntimeError("metadata output copy remained enabled despite native option")
                            traces.append(
                                decode_trace(
                                    raw,
                                    fields,
                                    rank,
                                    core,
                                    op.startswith("cached"),
                                    op.startswith("cached") and flag == 1,
                                    native.dispatch_device_timing_cycles_per_us,
                                )
                            )
                            for actual, expected in zip(snapshot(traced), reference):
                                if expected is not None:
                                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    finally:
                        os.environ[TRACE_ENV] = "0"
                    (capture / "device_trace.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
                    gathered_api = group.gather(diagnostic_api)
                    diagnostic_max = [max(values[i] for values in gathered_api) for i in range(args.trace_samples)]
                    (capture / "trace_overhead.json").write_text(
                        json.dumps(
                            dict(
                                diagnostic_api_us=diagnostic_max,
                                formal_api_mean_us=stats["api_wall_us"]["mean"],
                                diagnostic_api_mean_us=statistics.mean(diagnostic_max),
                                note="Diagnostic API overhead only; never use diagnostic times for headline bandwidth.",
                            ),
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            group.agree([op, "passed"])
        comparisons = channel_comparison(results)
        if comparisons:
            with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
                writer.writeheader()
                writer.writerows(comparisons)
        overlap_rows = overlap_comparison(results)
        if overlap_rows:
            with (out / "overlap_comparison.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(overlap_rows[0]))
                writer.writeheader()
                writer.writerows(overlap_rows)
        metadata_rows = overlap_comparison(results, "metadata_output_mode", "metadata_output_effective")
        if metadata_rows:
            with (out / "metadata_output_comparison.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(metadata_rows[0]))
                writer.writeheader()
                writer.writerows(metadata_rows)
        if args.combine_diagnostics:
            from combine_queue_diagnostics import run_combine_diagnostics

            set_overlap_mode("off")
            set_metadata_output_mode("off")
            launch_policy.variant = "auto"
            os.environ[TRACE_ENV] = "0"
            os.environ[METADATA_ENV] = "0"
            run_combine_diagnostics(
                buffer,
                native,
                torch,
                torch_npu,
                group,
                args,
                flush,
                cpu_x,
                owners,
                cpu_w,
                x,
                idx,
                weights,
                payload,
                out,
                decode_trace,
            )
    group.agree("complete")
    (out / "exit.json").write_text('{"exit_code": 0, "variant_precision_passed": true}', encoding="utf-8")


if __name__ == "__main__":
    run_worker(main)
