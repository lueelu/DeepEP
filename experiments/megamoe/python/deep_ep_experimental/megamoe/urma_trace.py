# Copyright (c) 2026, Lu Lu
# Modified by SimpleBrightMan 2026

"""Validate URMA detailed trace schema 1 and export a rank-local Chrome trace.

Raw records separate submission, waiting and drain. The default display joins
unambiguously paired Cube submission/drain records, retaining their provenance.
"""

import argparse
from collections import Counter
import json
import math
import statistics
from pathlib import Path

TRACE_SCHEMA = 1
ENGINE_COUNT = 84
HEADER_WORDS = RECORD_WORDS = 16
UNKNOWN = (1 << 64) - 1
EVENTS = (
    "Pack",
    "DispatchPutX",
    "DispatchPutScale",
    "DispatchQuiet",
    "DispatchPublish",
    "SelfDispatch",
    "WaitStartPeer",
    "PublishStart",
    "WaitLocalDispatch",
    "WaitCombineReady",
    "CombinePut",
    "CombineQuiet",
    "SelfCombine",
    "FinalPeerHandshake",
    "WaitDispatch",
    "Gmm1Tile",
    "Gmm1TileDrain",
    "Gmm1Dequant",
    "WaitActivation",
    "Gmm2Tile",
    "Gmm2Dequant",
    "Gmm2Drain",
    "WaitGmm1",
    "ActivationSlice",
    "RankBarrier",
    "Reduce",
    "LocalSync",
    "Gmm1Drain",
    "InputScaleGather",
    "WaitInputScale",
    "CombinePublish",
    "WaitReduceReady",
    "ReduceDrain",
    # Only 33..39 are reserved (JSON composites use 33/34).
    # Unitflag emits Reorder/WaitReorder at native IDs 40/41.
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    "Gmm2Reorder",
    "WaitGmm2Reorder",
    "CombineSendComplete",
    "DispatchSendComplete",
    "CoarsePack",
    "CoarseLocalSyncStart",
    "CoarseGMM",
    "CoarseDequant",
    "CoarseDispatch",
    "CoarseCombine",
    "CoarseActivation",
    "CoarseReduce",
    "CoarseLocalSyncEnd",
    "CoarseDispatchSubmit",
    "PackRetire",
    "SharedGmm1Tile",
    "SharedGmm1Drain",
    "SharedActivation",
    "SharedGmm2Tile",
    "SharedGmm2Drain",
    "SharedAdd",
    "CoarseSharedCombineSubmit",
    "RoutePrepare",
    "RouteSummary",
    "WaitRouteSummary",
    "DispatchReceiveComplete",
)
METADATA_FIELDS = ("peer", "cv", "fragment", "expert", "row", "rows", "tile_m", "tile_n", "columns")
SYNC_STAGES = {
    4: "shared addition complete",
    0: "pack",
    1: "compute/return",
    2: "rank barrier",
    3: "local invocation completion",
}


class TraceOverflowError(ValueError):
    """An incomplete trace must not be used as if it contained every event."""


def event_style(event):
    if event == "RouteSummary":
        return (
            event,
            "routing.device",
            "rail_animation",
            "AIV0 source/expert phases and leader prefix; includes phase waits while AIV1 Pack/Dispatch progress",
        )
    if event == "WaitRouteSummary":
        return event, "wait.sync", "thread_state_iowait", "Wait/acquire published incoming route summary"
    if event == "RoutePrepare":
        return (
            event,
            "routing.device",
            "rail_animation",
            "Protocol6 local expert buckets/prefix/send; incoming summary is a separate stage (legacy includes it here)",
        )
    if event == "Pack":
        return (
            event,
            "compute.copy",
            "rail_animation",
            "Pack submission; includes retirement of previous segment, current ready may be deferred",
        )
    if event == "PackRetire":
        return (
            event,
            "compute.drain",
            "thread_state_iowait",
            "wait captured Pack fence and publish pending ready; Close also releases six DMA credits",
        )
    if event == "CoarseSharedCombineSubmit":
        return (
            event,
            "coarse.envelope",
            "thread_state_runnable",
            "Combine submission envelope including cooperative local work; final quiet is deferred until after routed Reduce",
        )
    if event == "CoarseDispatchSubmit":
        return (
            event,
            "coarse.envelope",
            "thread_state_runnable",
            "Dispatch submission envelope, not remote completion; includes cooperative local work between submissions; final Combine quiet retires both phases",
        )
    if event.startswith("Coarse"):
        return (
            event,
            "coarse.envelope",
            "good",
            "stage envelope including internal gaps; timestamps buffered until final local join",
        )
    if event == "Reduce":
        return (
            event,
            "compute.copy",
            "rail_animation",
            "reduction scope; may include input readiness waits and final output drain",
        )
    if event == "Gmm2Reorder":
        return (
            event,
            "compute.copy",
            "rail_animation",
            "ready-span batch with local store completion; rows/bytes cover only this batch, not the whole CV",
        )
    if event in ("DispatchPutX", "DispatchPutScale", "CombinePut"):
        return (
            event + " [PUT submit]",
            "transport.submit",
            "thread_state_runnable",
            "enqueue scope; not remote completion",
        )
    if event in ("Gmm1Tile", "Gmm2Tile"):
        return event + " [Cube submit]", "cube.submit", "good", "submission scope; not Cube completion"
    if event.endswith("Quiet"):
        return (
            event + " [completion wait]",
            "transport.quiet",
            "thread_state_iowait",
            "wait for prior peer PUT completion",
        )
    if event == "CombineSendComplete":
        return (
            event + " [outbound completion]",
            "transport.quiet",
            "thread_state_iowait",
            "final outbound-peer quiet; end confirms all prior Dispatch/Combine payload PUTs and SO flags to this peer completed",
        )
    if event == "DispatchReceiveComplete":
        return (
            event + " [peer to local complete]",
            "transport.receive_observation",
            "good",
            "first nonblocking observation of this peer's final Dispatch SO epoch in local memory; includes all preceding X/scale PUTs from peer to this rank; rows=0 marks an empty incoming edge and contributes no receive-end timestamp",
        )
    if event == "DispatchSendComplete":
        return (
            event + " [outbound completion]",
            "transport.quiet",
            "thread_state_iowait",
            "outbound Dispatch completion observation; zero-duration records are nonblocking fixed CQ-target probes, not quiet waits; timestamp is first observation and may be later than actual completion",
        )
    if event.endswith("Drain"):
        return event + " [drain]", "compute.drain", "thread_state_iowait", "explicit pipeline drain scope"
    if event.startswith("Wait") or event in ("RankBarrier", "LocalSync", "FinalPeerHandshake"):
        return event, "wait.sync", "thread_state_iowait", "dependency or synchronization wait"
    if event in ("DispatchPublish", "CombinePublish", "PublishStart"):
        return event, "signal.publish", "rail_response", "signal publication scope"
    return event, "compute.copy", "rail_animation", "recorded device scope"


def decode_trace(raw, *, rank, clock_mhz):
    """Return Chrome trace JSON. All tick subtraction uses Python integers first.

    Input is an int64/uint64 array or the little-endian raw bytes written by the
    benchmark. Signed storage is reinterpreted as uint64, including sentinels.
    Each rank is zeroed independently; joining files cannot align rank clocks.
    """
    import numpy as np

    if type(rank) is not int or not 0 <= rank < 16:
        raise ValueError("rank must be an integer in [0,16)")
    if isinstance(clock_mhz, bool) or not math.isfinite(clock_mhz) or clock_mhz <= 0:
        raise ValueError("clock_mhz must be positive and finite")
    if isinstance(raw, (bytes, bytearray, memoryview)):
        if len(raw) % 8:
            raise ValueError("trace byte length must be a multiple of 8")
        values = np.frombuffer(raw, dtype="<u8")
    else:
        values = np.asarray(raw)
        if values.dtype.kind not in "iu" or values.dtype.itemsize != 8:
            raise ValueError("trace requires 64-bit integer storage")
        values = values.view(np.dtype(values.dtype.byteorder + "u8")).reshape(-1)
    if values.size < ENGINE_COUNT * HEADER_WORDS:
        raise ValueError("trace is missing engine headers")
    if int(values[0]) != TRACE_SCHEMA:
        raise ValueError("trace schema is missing or incompatible")
    capacity = int(values[1])
    if not 1 <= capacity <= 65536:
        raise ValueError("trace capacity must be in [1,65536]")
    stride = HEADER_WORDS + capacity * RECORD_WORDS
    if values.size != ENGINE_COUNT * stride:
        raise ValueError(f"trace size disagrees with capacity={capacity}: expected {ENGINE_COUNT * stride} words")
    slices = values.reshape(ENGINE_COUNT, stride)
    counts, overflows = [], []
    for engine, data in enumerate(slices):
        schema, engine_capacity, count, dropped, engine_id = map(int, data[:5])
        if schema != TRACE_SCHEMA:
            raise ValueError(f"engine={engine}: incompatible schema={schema}")
        if engine_capacity != capacity:
            raise ValueError(f"engine={engine}: capacity={engine_capacity} differs from {capacity}")
        if engine_id != engine:
            raise ValueError(f"engine={engine}: invalid engine id={engine_id}")
        if count > capacity or dropped:
            overflows.append(f"engine={engine} count={count} capacity={capacity} dropped={dropped}")
        if any(int(value) for value in data[5:HEADER_WORDS]):
            raise ValueError(f"engine={engine}: nonzero reserved header fields")
        counts.append(count)
    if overflows:
        raise TraceOverflowError("URMA trace overflow; increase --trace-capacity; " + "; ".join(overflows))

    records = []
    for engine, (data, count) in enumerate(zip(slices, counts)):
        previous_end = None
        for ordinal in range(count):
            offset = HEADER_WORDS + ordinal * RECORD_WORDS
            row = tuple(map(int, data[offset : offset + RECORD_WORDS]))
            start, end, event_id = row[:3]
            if event_id >= len(EVENTS) or EVENTS[event_id] is None:
                raise ValueError(f"engine={engine} record={ordinal}: unknown event id={event_id}")
            if end < start or start == UNKNOWN or end == UNKNOWN:
                raise ValueError(f"engine={engine} record={ordinal}: invalid ticks start={start} end={end}")
            observation_point = event_id in (43, 65) and start == end
            if not observation_point:
                if previous_end is not None and start < previous_end:
                    raise ValueError(f"engine={engine} record={ordinal}: overlapping or out-of-order event ticks")
                previous_end = end
            if any(row[13:]):
                raise ValueError(f"engine={engine} record={ordinal}: nonzero reserved record fields")
            if row[3] != UNKNOWN and row[3] >= 16:
                raise ValueError(f"engine={engine} record={ordinal}: peer must be in [0,16) or unknown")
            records.append((start, end, engine, ordinal, row))
    if not records:
        raise ValueError("trace has no recorded events")
    origin = min(record[0] for record in records)
    last = max(record[1] for record in records)
    metadata = dict(
        schema=TRACE_SCHEMA,
        rank=rank,
        engine_count=ENGINE_COUNT,
        capacity_per_engine=capacity,
        records_per_engine=counts,
        record_count=len(records),
        dropped=0,
        raw_bytes=int(values.nbytes),
        clock_mhz=float(clock_mhz),
        clock_source="caller-supplied ticks/us; not detected from hardware",
        origin_tick=str(origin),
        last_tick=str(last),
        envelope_ticks=last - origin,
        envelope_us=(last - origin) / clock_mhz,
        clock_alignment="each rank starts at its own earliest event; cross-rank clocks are not synchronized",
        scope="diagnostic invocation; excluded from formal performance samples",
        instrumentation="coarse" if any(44 <= record[4][2] <= 53 for record in records) else "fine",
        interpretation="PUT/Cube submit bars do not measure asynchronous completion; inspect quiet/drain/wait separately",
        events=dict(Counter(EVENTS[record[4][2]] for record in records)),
    )
    events = [dict(ph="M", pid=rank, tid=0, name="process_name", args=dict(name=f"rank {rank} (independent clock)"))]
    if metadata["instrumentation"] == "coarse":
        metadata["coarse_stage_envelopes_us"] = {}
        for name in EVENTS[44:]:
            selected = [record for record in records if EVENTS[record[4][2]] == name]
            if selected:
                metadata["coarse_stage_envelopes_us"][name] = (
                    max(record[1] for record in selected) - min(record[0] for record in selected)
                ) / clock_mhz
        metadata["interpretation"] = (
            "Stage envelopes include internal idle time. GMM/Dequant start at task-loop entry, "
            "including initial readiness wait, and end after final context drain. "
            "All records are written after final LocalSync; record writeback is outside the raw envelope."
        )
    for engine in range(ENGINE_COUNT):
        block, kind = divmod(engine, 3)
        events.extend(
            [
                dict(
                    ph="M",
                    pid=rank,
                    tid=engine,
                    name="thread_name",
                    args=dict(name=f"block {block:02d} {('AIC', 'AIV0', 'AIV1')[kind]}"),
                ),
                dict(ph="M", pid=rank, tid=engine, name="thread_sort_index", args=dict(sort_index=engine)),
            ]
        )
    for start, end, engine, ordinal, row in sorted(records, key=lambda value: (value[0], value[2], value[3])):
        event = EVENTS[row[2]]
        name, category, color, meaning = event_style(event)
        if event == "DispatchSendComplete" and start == end:
            name = event + " [nonblocking observation]"
            category, color = "transport.completion_observation", "good"
        if event == "DispatchReceiveComplete" and row[8] == 0:
            name = event + " [empty incoming edge]"
        if engine % 3 == 2 and event in ("Gmm1TileDrain", "Gmm2Drain"):
            name = event + " [paired FIX completion]"
            meaning = "paired AIV1 waits for actual FIX full before consuming the UB mailbox with GMM1 activation or GMM2 reorder; not AIC execution time"
        args = {field: None if value == UNKNOWN else value for field, value in zip(METADATA_FIELDS, row[3:12])}
        if event == "LocalSync":
            stage = args.pop("cv")
            args.update(sync_stage=stage, sync_stage_name=SYNC_STAGES.get(stage, "unknown"))
        args.update(
            event_id=row[2],
            bytes=row[12],
            record_index=ordinal,
            start_tick=str(start),
            end_tick=str(end),
            elapsed_ticks=end - start,
            meaning=meaning,
        )
        events.append(
            dict(
                ph="X",
                name=name,
                cat=category,
                cname=color,
                pid=rank,
                tid=engine,
                ts=(start - origin) / clock_mhz,
                dur=(end - start) / clock_mhz,
                args=args,
            )
        )
    return dict(traceEvents=events, displayTimeUnit="us", metadata=metadata)


def filter_trace_waits(trace):
    """Hide dependency waits without moving timestamps or losing final barriers."""
    kept, hidden = [], 0
    for event in trace["traceEvents"]:
        if event.get("ph") == "X" and event.get("cat") == "wait.sync":
            name = event["name"]
            final_barrier = name in ("RankBarrier", "FinalPeerHandshake") or (
                name == "LocalSync" and event.get("args", {}).get("sync_stage") in (2, 3)
            )
            if not final_barrier:
                hidden += 1
                continue
        kept.append(event)
    metadata = dict(trace["metadata"])
    per_engine = [0] * ENGINE_COUNT
    for event in kept:
        if event.get("ph") == "X":
            per_engine[event["tid"]] += 1
    metadata.update(
        visible_record_count=sum(per_engine),
        visible_records_per_engine=per_engine,
        hidden_wait_count=metadata.get("hidden_wait_count", 0) + hidden,
        wait_filter="dependency waits and startup sync hidden; final rank barrier/handshake/local completion retained; timestamps unchanged",
    )
    return dict(trace, traceEvents=kept, metadata=metadata)


def coalesce_gmm_tiles(trace):
    """Join adjacent AIC tile/drain records without changing device synchronization.

    Group drains belong to a tile only when this engine submitted exactly one
    tile in that group. Record ordinals prevent joining across hidden waits.
    IDs 33/34 are JSON-only composite events, never raw device event IDs.
    """
    lanes = {}
    for index, event in enumerate(trace["traceEvents"]):
        if event.get("ph") == "X" and event["tid"] % 3 == 0:
            lanes.setdefault((event["pid"], event["tid"]), []).append((index, event))

    def context(event):
        return tuple(event["args"].get(key) for key in ("peer", "cv", "fragment", "expert"))

    replacements, removed = {}, set()
    for lane in lanes.values():
        counts = Counter(
            (event["args"]["event_id"], context(event)) for _, event in lane if event["args"]["event_id"] in (15, 19)
        )
        for position, (index, tile) in enumerate(lane):
            event_id = tile["args"]["event_id"]
            if event_id not in (15, 19) or tile["args"].get("fragment") is None:
                continue
            parts = [tile]
            candidates = (16, 27) if event_id == 15 else (21,)
            for expected in candidates:
                following = position + len(parts)
                if following >= len(lane):
                    break
                drain_index, drain = lane[following]
                args = drain["args"]
                if (
                    args["event_id"] != expected
                    or context(drain) != context(tile)
                    or args["record_index"] != parts[-1]["args"]["record_index"] + 1
                    or int(args["start_tick"]) < int(parts[-1]["args"]["end_tick"])
                ):
                    break
                if expected == 16:
                    if any(args.get(key) != tile["args"].get(key) for key in METADATA_FIELDS):
                        break
                elif counts[(event_id, context(tile))] != 1:
                    break
                parts.append(drain)
                removed.add(drain_index)
            if len(parts) == 1:
                continue
            start, end = int(tile["args"]["start_tick"]), int(parts[-1]["args"]["end_tick"])
            args = dict(
                tile["args"],
                event_id=33 if event_id == 15 else 34,
                end_tick=str(end),
                elapsed_ticks=end - start,
                bytes=sum(part["args"].get("bytes", 0) for part in parts),
                meaning="submission through recorded drain end; includes waits/teardown, not pure Cube execution",
                source_records=[
                    {key: part["args"][key] for key in ("event_id", "record_index", "start_tick", "end_tick")}
                    for part in parts
                ],
            )
            replacements[index] = dict(
                tile,
                name=("Gmm1Tile" if event_id == 15 else "Gmm2Tile") + " [compute + drain]",
                cat="cube.compute_drain",
                dur=(end - start) / trace["metadata"]["clock_mhz"],
                args=args,
            )
    if not removed:
        return trace
    events = [
        replacements.get(index, event) for index, event in enumerate(trace["traceEvents"]) if index not in removed
    ]
    counts = [0] * ENGINE_COUNT
    for event in events:
        if event.get("ph") == "X":
            counts[event["tid"]] += 1
    metadata = dict(
        trace["metadata"],
        coalesced_record_count=trace["metadata"].get("coalesced_record_count", 0) + len(removed),
        visible_record_count=sum(counts),
        visible_records_per_engine=counts,
        gmm_display="paired AIC tile/drain envelopes; raw counts retained; IDs 33/34 are JSON-only composites",
    )
    return dict(trace, traceEvents=events, metadata=metadata)


def export_trace(raw, output, *, rank, clock_mhz, include_waits=False, combine_gmm=True):
    """Validate every record before writing the Chrome JSON file."""
    trace = decode_trace(raw, rank=rank, clock_mhz=clock_mhz)
    if combine_gmm:
        trace = coalesce_gmm_tiles(trace)
    if not include_waits:
        trace = filter_trace_waits(trace)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return trace["metadata"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="One rank's urma_trace.bin; not the aggregate urma_timer.bin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--include-waits", action="store_true", help="Restore dependency waits from the raw binary")
    parser.add_argument("--split-gmm-events", action="store_true", help="Keep raw Cube submit/drain bars separate")
    parser.add_argument(
        "--clock-mhz", type=float, required=True, help="Counter ticks/us; use the benchmark's clock setting"
    )
    args = parser.parse_args(argv)
    try:
        metadata = export_trace(
            args.raw.read_bytes(),
            args.output,
            rank=args.rank,
            clock_mhz=args.clock_mhz,
            include_waits=args.include_waits,
            combine_gmm=not args.split_gmm_events,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"URMA trace export failed: {error}\n")
    print(json.dumps(dict(output=str(args.output), **metadata), ensure_ascii=False, allow_nan=False))


def _remote_peer_for_worker(rank, worker, world=16):
    """One remote PE per communication worker; the self PE uses local copies."""
    if not 0 <= rank < world or not 0 <= worker < world - 1:
        raise ValueError("invalid rank or remote worker")
    return worker + int(worker >= rank)


PHASES = ("Pack", "Dispatch", "Gmm1", "Activation", "Gmm2", "Combine", "Barrier", "Reduce")
TIMER_NUMEL = 28 * 3 * 8 * 16
TIMER_SCHEMA = 1
LEGACY_COMPLETED_SCOPE = 0
SUBMISSION_ONLY = 1
DISPATCH_SEND_TO_RECEIVE = 2


def communication_bandwidth(raw, rank, *, world=16, group_size=4, clock_mhz=1000.0):
    """Effective aggregate payload divided by the class's first-to-last span.

    1000 ticks/us follows the existing MegaMoE timer protocol. It is an explicit
    assumption, not a device-frequency measurement. Pass None for raw ticks.
    Record word 5 distinguishes legacy completed scopes (0) from PUT-only
    scopes (1), and Dispatch outgoing-first -> incoming-last scope (2).
    Kind2 word7 is the local receive observation tick (0 for empty inbound),
    word8 received bytes and word9 received rows; other kinds keep outbound
    completion semantics. The bandwidth numerator remains outgoing payload.
    Missing completion never turns a submission rate into bandwidth.
    Older records may include self-copy in INTRA. Detect it from recorded
    payload; direct self Pack/GMM2 writes are not communication payload.
    """
    import numpy as np

    if world not in (4, 8, 16) or not 0 <= rank < world:
        raise ValueError("world must be 4, 8 or 16 and rank must be in [0,world)")
    if type(group_size) is not int or group_size <= 0 or world % group_size:
        raise ValueError("group_size must be positive and divide world")
    if clock_mhz is not None and (not math.isfinite(clock_mhz) or clock_mhz <= 0):
        raise ValueError("clock_mhz must be positive and finite, or None")
    values = np.asarray(raw)
    if values.size != TIMER_NUMEL or values.dtype.kind not in "iu":
        raise ValueError(f"URMA timer requires {TIMER_NUMEL} integer values")
    values = values.reshape(28, 3, 8, 16)
    receive_markers = values[..., 5] == DISPATCH_SEND_TO_RECEIVE
    if np.any(receive_markers[:, :, :1]) or np.any(receive_markers[:, :, 2:]):
        raise ValueError("receive timing semantics are valid only for Dispatch")
    receive_mode = bool(np.any(values[: world - 1, 2, 1, 5] == DISPATCH_SEND_TO_RECEIVE))
    active = (values[..., 4] > 0) | receive_markers
    if np.any(values[..., 6][active] != TIMER_SCHEMA):
        raise ValueError("URMA timer schema is missing or incompatible")
    if np.any(values[..., 1][active] < values[..., 0][active]):
        raise ValueError("URMA timer end precedes start")
    if np.any(~np.isin(values[..., 5][active], (LEGACY_COMPLETED_SCOPE, SUBMISSION_ONLY, DISPATCH_SEND_TO_RECEIVE))):
        raise ValueError("URMA timer has unknown timing semantics")
    completed_submissions = active & (values[..., 5] == SUBMISSION_ONLY) & (values[..., 7] != 0)
    if np.any(values[..., 7][completed_submissions] < values[..., 1][completed_submissions]):
        raise ValueError("URMA timer completion precedes the last submission")
    result = dict(
        rank=rank,
        world=world,
        group_size=group_size,
        schema=TIMER_SCHEMA,
        clock_mhz=clock_mhz,
        clock_source="caller supplied; default 1000 follows legacy MegaMoE protocol",
        metric="single_direction_aggregate_effective_payload_gbps",
        denominator="Dispatch kind2: same-rank class max(peer receive observation) - min(outgoing first PUT); other records retain outbound completion spans",
        dispatch_endpoint="peer_to_local_receive" if receive_mode else "outbound_completion",
        includes_self_copy_in_intra=False,
        classes=[],
    )
    for phase in ("Dispatch", "Combine"):
        phase_idx = PHASES.index(phase)
        for kind in ("INTRA", "CROSS"):
            selected = []
            for owner in range(world):
                peer = _remote_peer_for_worker(rank, owner, world) if owner < world - 1 else rank
                owner_kind = "INTRA" if peer // group_size == rank // group_size else "CROSS"
                record = values[owner, 2, phase_idx]
                if owner_kind == kind and record[3] > 0:
                    if record[4] <= 0 or record[6] != TIMER_SCHEMA:
                        raise ValueError("payload without a completed timer record")
                    selected.append((owner, peer, record))
            rx_records = []
            use_receive = phase == "Dispatch" and receive_mode
            if use_receive:
                for owner in range(world - 1):
                    peer = _remote_peer_for_worker(rank, owner, world)
                    owner_kind = "INTRA" if peer // group_size == rank // group_size else "CROSS"
                    if owner_kind == kind:
                        rx_records.append((owner, peer, values[owner, 2, phase_idx]))
                if any(int(row[8]) < 0 or int(row[9]) < 0 for _, _, row in rx_records):
                    raise ValueError("negative received payload/rows")
            rx_nonempty = [
                (owner, peer, row)
                for owner, peer, row in rx_records
                if row[5] == DISPATCH_SEND_TO_RECEIVE and row[8] > 0
            ]
            rx_known = all(row[5] == DISPATCH_SEND_TO_RECEIVE for _, _, row in rx_records)
            payload = sum(int(record[3]) for _, _, record in selected)
            includes_self_copy = any(peer == rank for _, peer, _ in selected)
            result["includes_self_copy_in_intra"] |= includes_self_copy
            if selected:
                first = min(int(record[0]) for _, _, record in selected)
                recorded_last = max(int(record[1]) for _, _, record in selected)
                submission_only = all(
                    int(record[5]) in (SUBMISSION_ONLY, DISPATCH_SEND_TO_RECEIVE) for _, _, record in selected
                )
                completion_ticks = [
                    int(record[7] if record[5] == SUBMISSION_ONLY else record[1]) for _, _, record in selected
                ]
                if use_receive:
                    completion_ticks = [int(row[7]) for _, _, row in rx_nonempty]
                    complete = rx_known and bool(completion_ticks) and all(tick != 0 for tick in completion_ticks)
                else:
                    complete = all(tick != 0 for tick in completion_ticks)
                last = max(completion_ticks) if complete else None
                span = last - first if complete else None
                if span is not None and span <= 0:
                    if use_receive:
                        span = None
                        complete = False
                    else:
                        raise ValueError("nonempty payload requires a positive timer span")
                completion_status = (
                    "outbound_complete"
                    if complete and submission_only
                    else "legacy_completed_scope"
                    if complete
                    else "not_recorded"
                )
                if use_receive:
                    completion_status = (
                        "peer_dispatch_received"
                        if complete
                        else (
                            "no_inbound_payload"
                            if rx_known and not rx_nonempty
                            else "nonpositive_exchange_span"
                            if last is not None
                            else "not_recorded"
                        )
                    )
            else:
                first = last = span = 0
                recorded_last = 0
                submission_only = False
                completion_status = "empty"
            result["classes"].append(
                dict(
                    phase=phase,
                    communication_class=kind,
                    payload_bytes=payload,
                    first_tick=first,
                    last_tick=last,
                    envelope_ticks=span,
                    envelope_us=span / clock_mhz if span is not None and clock_mhz is not None else None,
                    bandwidth_gbps=payload * clock_mhz / (span * 1000) if span and clock_mhz is not None else None,
                    completion_status=completion_status,
                    missing_completion_peers=(
                        [
                            peer
                            for _, peer, row in rx_records
                            if row[5] != DISPATCH_SEND_TO_RECEIVE or (row[8] > 0 and row[7] == 0)
                        ]
                        if use_receive
                        else [peer for _, peer, record in selected if record[5] == SUBMISSION_ONLY and record[7] == 0]
                    ),
                    endpoint_semantics="outgoing_first_to_incoming_last" if use_receive else "outbound_completion",
                    received_payload_bytes=sum(int(row[8]) for _, _, row in rx_nonempty) if use_receive else None,
                    receive_peers=[peer for _, peer, _ in rx_nonempty] if use_receive else [],
                    submission_first_tick=first if submission_only else None,
                    submission_last_tick=recorded_last if submission_only else None,
                    submission_envelope_ticks=recorded_last - first if submission_only else None,
                    submission_sum_ticks=sum(int(record[2]) for _, _, record in selected) if submission_only else None,
                    workers=[owner for owner, _, _ in selected],
                    peers=[peer for _, peer, _ in selected],
                    includes_self_copy=includes_self_copy,
                )
            )
    return result


def summarize_bandwidth(records):
    """Average already-local bandwidths, without mixing clocks across ranks."""
    world = len(records)
    if (
        world not in (4, 8, 16)
        or {row["rank"] for row in records} != set(range(world))
        or any(row.get("world", world) != world for row in records)
    ):
        raise ValueError("bandwidth summary requires exactly one report for every EP4/EP8/EP16 rank")
    result = []
    for phase in ("Dispatch", "Combine"):
        for kind in ("INTRA", "CROSS"):
            rows = [
                next(row for row in record["classes"] if (row["phase"], row["communication_class"]) == (phase, kind))
                for record in records
            ]
            rates = [row["bandwidth_gbps"] for row in rows if row["bandwidth_gbps"] is not None]
            result.append(
                dict(
                    phase=phase,
                    communication_class=kind,
                    mean_gbps=statistics.mean(rates) if rates else None,
                    rank_min_gbps=min(rates) if rates else None,
                    rank_max_gbps=max(rates) if rates else None,
                    nonempty_ranks=len(rates),
                    payload_ranks=sum(row["payload_bytes"] > 0 for row in rows),
                    missing_completion_ranks=[
                        record["rank"]
                        for record, row in zip(records, rows)
                        if row.get("completion_status") == "not_recorded"
                    ],
                    bytes_per_rank=[row["payload_bytes"] for row in rows],
                    includes_self_copy=any(row.get("includes_self_copy", False) for row in rows),
                )
            )
    return result


def export_diagnostics(group, trace, directory, *, timer_ranks):
    """Export selected ranks, then propagate failures through every rank."""
    from .validation import collective_check

    failure = None
    if trace is not None:
        try:
            buffer = trace.cpu().numpy()
            if buffer.ndim != 1 or buffer.size <= TIMER_NUMEL:
                raise ValueError("Invalid combined timer buffer")
            raw, counters = buffer[:-TIMER_NUMEL], buffer[-TIMER_NUMEL:]
            raw.astype("<i8", copy=False).tofile(directory / "urma_trace.bin")
            export_trace(raw, directory / "urma_trace.json", rank=group.rank(), clock_mhz=1000.0)
            timing = communication_bandwidth(counters, group.rank(), world=group.size(), clock_mhz=1000.0)
            timing.update(diagnostic_only=True, selected_timer_ranks=timer_ranks)
            (directory / "communication_bandwidth.json").write_text(
                json.dumps(timing, indent=2, allow_nan=False), encoding="utf-8"
            )
        except Exception as error:
            failure = str(error)
    collective_check(group, "URMA trace export", failure)


if __name__ == "__main__":
    main()
