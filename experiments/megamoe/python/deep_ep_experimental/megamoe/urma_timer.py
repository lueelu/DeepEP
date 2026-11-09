"""Decode URMA schema 1; separate PUT submission from outbound completion."""

import math
import statistics

from ._urma_route_reference import remote_peer_for_worker

PHASES = ("Pack", "Dispatch", "Gmm1", "Activation", "Gmm2", "Combine", "Barrier", "Reduce")
TIMER_NUMEL = 28 * 3 * 8 * 16
TIMER_SCHEMA = 1
LEGACY_COMPLETED_SCOPE = 0
SUBMISSION_ONLY = 1


def communication_bandwidth(raw, rank, *, world=16, group_size=4, clock_mhz=1000.0):
    """Effective aggregate payload divided by the class's first-to-last span.

    1000 ticks/us follows the existing MegaMoE timer protocol. It is an explicit
    assumption, not a device-frequency measurement. Pass None for raw ticks.
    Record word 5 distinguishes legacy completed scopes (0) from PUT-only
    scopes (1); word 7 is an optional confirmed outbound completion tick.
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
    active = values[..., 4] > 0
    if np.any(values[..., 6][active] != TIMER_SCHEMA):
        raise ValueError("URMA timer schema is missing or incompatible")
    if np.any(values[..., 1][active] < values[..., 0][active]):
        raise ValueError("URMA timer end precedes start")
    if np.any(~np.isin(values[..., 5][active], (LEGACY_COMPLETED_SCOPE, SUBMISSION_ONLY))):
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
        denominator="same-rank class max(outbound completion tick) - min(first PUT tick); null without completion",
        includes_self_copy_in_intra=False,
        classes=[],
    )
    for phase in ("Dispatch", "Combine"):
        phase_idx = PHASES.index(phase)
        for kind in ("INTRA", "CROSS"):
            selected = []
            for owner in range(world):
                peer = remote_peer_for_worker(rank, owner, world) if owner < world - 1 else rank
                owner_kind = "INTRA" if peer // group_size == rank // group_size else "CROSS"
                record = values[owner, 2, phase_idx]
                if owner_kind == kind and record[3] > 0:
                    if record[4] <= 0 or record[6] != TIMER_SCHEMA:
                        raise ValueError("payload without a completed timer record")
                    selected.append((owner, peer, record))
            payload = sum(int(record[3]) for _, _, record in selected)
            includes_self_copy = any(peer == rank for _, peer, _ in selected)
            result["includes_self_copy_in_intra"] |= includes_self_copy
            if selected:
                first = min(int(record[0]) for _, _, record in selected)
                recorded_last = max(int(record[1]) for _, _, record in selected)
                submission_only = all(int(record[5]) == SUBMISSION_ONLY for _, _, record in selected)
                completion_ticks = [
                    int(record[7] if record[5] == SUBMISSION_ONLY else record[1]) for _, _, record in selected
                ]
                complete = all(tick != 0 for tick in completion_ticks)
                last = max(completion_ticks) if complete else None
                span = last - first if complete else None
                if span is not None and span <= 0:
                    raise ValueError("nonempty payload requires a positive timer span")
                completion_status = (
                    "outbound_complete"
                    if complete and submission_only
                    else "legacy_completed_scope"
                    if complete
                    else "not_recorded"
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
                    missing_completion_peers=[
                        peer for _, peer, record in selected if record[5] == SUBMISSION_ONLY and record[7] == 0
                    ],
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
