# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Rank-local pipeline intervals and overlap breakdown."""

from collections import defaultdict
import json

from . import protocol as p
from .parse_timer_csv import core_info, validate_event_lane
from .trace_parser import event_family, merge_intervals


def compact_events(raw, rank):
    """Read only written slots; no wide CSV, task metadata or trace JSON is needed."""
    if len(raw) != p.TOTAL_BUFFER_SIZE:
        raise ValueError(f"rank={rank}: timer size differs from the installed protocol")
    events = []
    for core in range(p.N_CORE_COUNT):
        group, core_type, sub = core_info(core)
        names = [(name, index) for index, name in enumerate(p.FIXED_NAMES)]
        for family, name in enumerate(p.DYNAMIC_NAMES):
            count = int(raw[p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family])
            if not 0 <= count <= p.MAX_DYNAMIC_ITER:
                raise ValueError(f"rank={rank} core={core} {name}: invalid timer count={count}")
            names.extend(
                (f"{name}_{slot}", len(p.FIXED_NAMES) + family * p.MAX_DYNAMIC_ITER + slot) for slot in range(count)
            )
        for name, slot in names:
            offset = core * p.COUNTERS_PER_CORE + slot * 2
            start, end = int(raw[offset]), int(raw[offset + 1])
            validate_event_lane(name, core, core_type, start, end)
            if end > start > 0 and name != "KERNEL_TIMING" and "WAIT" not in name:
                events.append(
                    dict(
                        name=name,
                        group_id=group,
                        core_type=core_type,
                        sub_id=sub,
                        rank=rank,
                        start_cycle=start,
                        end_cycle=end,
                    )
                )
    return events


def rank_breakdown(events, rank):
    """A visible trace envelope is not profiler E2E, nor pure Cube execution time."""
    result = dict(rank=rank, status="unavailable")
    active = [e for e in events if "WAIT" not in e["name"] and event_family(e["name"]) != "KERNEL_TIMING"]
    families = defaultdict(list)
    for event in active:
        families[event_family(event["name"])].append(event)
    required = ("AIV_DISPATCH", "AIC_GMM1", "AIC_GMM2", "AIV_UNPERMUTE")
    missing = [name for name in required if not families[name]]
    if missing:
        return dict(result, reason="missing " + ", ".join(missing))
    origin = min(e["start_cycle"] for e in active)
    dispatch = min(e["start_cycle"] for e in families["AIV_DISPATCH"])
    gmm = families["AIC_GMM1"] + families["AIC_GMM2"]
    first = min(e["start_cycle"] for e in gmm)
    last = max(e["end_cycle"] for e in gmm)
    end = max(e["end_cycle"] for e in active)
    if not origin <= dispatch <= first <= last <= end:
        return dict(result, reason="non-sequential boundaries (e.g. shared-expert pipeline)")
    # 仅分解末完成 AIC，不能把并行核的区间累计当成 rank 端到端。
    final = max(gmm, key=lambda e: (e["end_cycle"], -e["group_id"], -e["sub_id"]))
    lane = (final["group_id"], final["sub_id"])
    spans, lane_intervals = {}, []
    for family in ("AIC_GMM1", "AIC_GMM2"):
        intervals = merge_intervals(
            (e["start_cycle"], e["end_cycle"]) for e in families[family] if (e["group_id"], e["sub_id"]) == lane
        )
        spans[family] = sum(b - a for a, b in intervals)
        lane_intervals.extend(intervals)
    total = sum(spans.values())
    if sum(b - a for a, b in merge_intervals(lane_intervals)) != total:
        raise ValueError(f"rank={rank} AIC={lane}: overlapping GMM1/GMM2 intervals")

    def us(cycles):
        return cycles / p.CYCLE_TO_US

    sync = families["AIV_INPUT_CROSS_RANK_SYNC"]
    return dict(
        result,
        status="ok",
        final_aic_group=lane[0],
        head_us=us(dispatch - origin),
        launch_us=us(first - dispatch),
        gmm_window_us=us(last - first),
        tail_us=us(end - last),
        visible_span_us=us(end - origin),
        gmm1_spans_us=us(spans["AIC_GMM1"]),
        gmm2_spans_us=us(spans["AIC_GMM2"]),
        gmm_uncovered_us=us(last - first - total),
        input_sync_max_us=max((us(e["end_cycle"] - e["start_cycle"]) for e in sync), default=None),
        dispatch_at_us=us(dispatch - origin),
        first_gmm_at_us=us(first - origin),
        last_gmm_at_us=us(last - origin),
        visible_end_at_us=us(end - origin),
    )


def write_breakdown(records, output_dir, case_label):
    records = sorted(records, key=lambda r: r["rank"])
    notes = (
        "rank-local raw time; WAIT hidden, input-sync NOT compressed; not profiler E2E",
        "head=[0,dispatch); launch=[dispatch,first_gmm); GMM=[first_gmm,last_gmm); tail=[last_gmm,end)",
        "GMM1/GMM2 spans use final_aic_group only; uncovered includes its start skew and unrecorded gaps",
        "spans may contain internal waits; uncovered is NOT proven hardware idle",
    )
    rows = (
        "head_us",
        "launch_us",
        "gmm_window_us",
        "tail_us",
        "visible_span_us",
        "input_sync_max_us",
        "final_aic_group",
        "gmm1_spans_us",
        "gmm2_spans_us",
        "gmm_uncovered_us",
        "dispatch_at_us",
        "first_gmm_at_us",
        "last_gmm_at_us",
        "visible_end_at_us",
    )
    lines = [f"[TIMER-BREAKDOWN] case={case_label}", *(f"[TIMER-BREAKDOWN] {note}" for note in notes)]
    for offset in range(0, len(records), 8):
        batch = records[offset : offset + 8]
        lines.append(f"{'metric / us':<22}" + "".join(f"{'rank' + str(r['rank']):>13}" for r in batch))
        for name in rows:
            values = []
            for record in batch:
                value = record.get(name)
                values.append("N/A" if value is None else str(value) if name == "final_aic_group" else f"{value:.3f}")
            lines.append(f"{name:<22}" + "".join(f"{value:>13}" for value in values))
    lines.extend(
        f"[TIMER-BREAKDOWN] rank={r['rank']} unavailable: {r['reason']}" for r in records if r["status"] != "ok"
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "timer_breakdown.txt").write_text(text, encoding="utf-8")
    (output_dir / "timer_breakdown.json").write_text(
        json.dumps(
            dict(schema="megamoe_breakdown_v1", case=case_label, notes=notes, ranks=records),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(text, end="", flush=True)
