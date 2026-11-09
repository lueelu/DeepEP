#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026
"""Condense a single-rank URMA Chrome JSON into readable performance text.

Standard library only; no installation, Torch, or NPU is needed. The default
report is small. --detail appends exact event positions with abbreviated fields.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

KINDS = ("AIC", "AIV0", "AIV1")
WAIT_CATEGORIES = ("wait.sync", "transport.quiet", "compute.drain")
FIELDS = {
    "peer": "p",
    "cv": "c",
    "fragment": "f",
    "expert": "e",
    "row": "r",
    "rows": "l",
    "tile_m": "m",
    "tile_n": "n",
    "columns": "w",
    "sync_stage": "s",
    "bytes": "b",
}


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def intervals(events):
    """Return wall-clock union; never sum simultaneous engines as elapsed time."""
    total = 0.0
    end = None
    for event in sorted(events, key=lambda e: e["ts"]):
        start, stop = event["ts"], event["ts"] + event["dur"]
        total += max(0.0, stop - (start if end is None else max(start, end)))
        end = stop if end is None else max(stop, end)
    return total


def stats(events):
    durations = sorted(e["dur"] for e in events)
    first = min(e["ts"] for e in events)
    last = max(e["ts"] + e["dur"] for e in events)
    return dict(
        count=len(events),
        sum=math.fsum(durations),
        union=intervals(events),
        first=first,
        last=last,
        span=last - first,
        p50=durations[math.ceil(len(events) * 0.50) - 1],
        p95=durations[math.ceil(len(events) * 0.95) - 1],
        maximum=durations[-1],
    )


def engine_label(tid):
    return f"{tid // 3:02d}/{KINDS[tid % 3]}"


def short_args(event):
    return " ".join(
        f"{alias}{event['args'][key]}"
        for key, alias in FIELDS.items()
        if event["args"].get(key) is not None and (key != "bytes" or event["args"][key] != 0)
    )


def event_line(event):
    # No rebasing or removal of gaps: these are the original JSON coordinates.
    return f"{event['ts']} {event['dur']} {event['id']} {short_args(event)}".rstrip()


def read_events(document):
    if not isinstance(document, dict) or not isinstance(document.get("traceEvents"), list):
        raise ValueError("Expected a single-rank URMA Chrome JSON object with traceEvents")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("schema") != 1:
        raise ValueError("Expected URMA trace metadata schema=1")
    events, ranks, definitions = [], set(), {}
    for raw in document["traceEvents"]:
        if not isinstance(raw, dict):
            raise ValueError("Invalid trace event")
        if raw.get("ph") != "X":
            continue
        tid, pid, args = raw.get("tid"), raw.get("pid"), raw.get("args")
        if type(tid) is not int or not 0 <= tid < 84 or type(pid) is not int or not 0 <= pid < 16:
            raise ValueError("Invalid URMA engine or rank")
        if not isinstance(args, dict) or type(args.get("event_id")) is not int:
            raise ValueError("Missing URMA event_id")
        for key in FIELDS:
            value = args.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{key} must be a nonnegative integer or null")
        start, duration = number(raw.get("ts"), "ts"), number(raw.get("dur"), "dur")
        number(start + duration, "event end")
        if duration < 0:
            raise ValueError("Negative event duration")
        event_id = args["event_id"]
        name = str(raw.get("name", "")).split(" [", 1)[0]
        category = str(raw.get("cat", ""))
        if event_id < 0 or not name or not category:
            raise ValueError("Invalid event definition")
        definition = (name, category)
        if event_id in definitions and definitions[event_id] != definition:
            raise ValueError("Conflicting names/categories for one event_id")
        definitions[event_id] = definition
        ranks.add(pid)
        events.append(dict(tid=tid, ts=start, dur=duration, id=event_id, name=name, cat=category, args=args))
    if len(ranks) != 1:
        raise ValueError("Expected nonempty single-rank trace; rank clocks must not be merged")
    rank = next(iter(ranks))
    if metadata.get("rank") != rank:
        raise ValueError("Rank differs from trace metadata")
    observed = [0] * 84
    for event in events:
        observed[event["tid"]] += 1
    declared = metadata.get("record_count")
    declared_per_engine = metadata.get("records_per_engine")
    if declared is None or declared_per_engine is None:
        completeness = "UNVERIFIED"
    elif (
        metadata.get("hidden_wait_count", 0) + metadata.get("coalesced_record_count", 0) > 0
        and metadata.get("visible_record_count") == len(events)
        and metadata.get("visible_records_per_engine") == observed
        and declared == len(events) + metadata.get("hidden_wait_count", 0) + metadata.get("coalesced_record_count", 0)
        and not metadata.get("dropped", 0)
    ):
        completeness = "COALESCED_GMM" if metadata.get("coalesced_record_count", 0) else "FILTERED_WAITS"
    elif declared != len(events) or declared_per_engine != observed or metadata.get("dropped", 0):
        completeness = "INCOMPLETE"
    else:
        completeness = "COMPLETE"
    return metadata, events, definitions, completeness


def render_report(document, *, source="trace.json", cores=None, engines=None, detail=False, include_waits=False, top=8):
    metadata, original, definitions, completeness = read_events(document)
    if type(top) is not int or top < 0:
        raise ValueError("top must be nonnegative")
    if cores is not None and any(type(core) is not int or not 0 <= core < 28 for core in cores):
        raise ValueError("cores must be block numbers in [0,27]")
    if engines is not None and not set(engines) <= set(KINDS):
        raise ValueError("engines must be AIC,AIV0,AIV1")
    events = [
        e
        for e in original
        if (cores is None or e["tid"] // 3 in cores) and (engines is None or KINDS[e["tid"] % 3] in engines)
    ]
    if not events:
        raise ValueError("No events selected")
    total = stats(events)
    lines = [
        f"URMA_TRACE_TEXT_V1 source={json.dumps(str(source), ensure_ascii=False)}",
        f"rank={metadata['rank']} source_status={completeness} source_events={len(original)} "
        f"declared={metadata.get('record_count')} selected={len(events)} clock_mhz={metadata.get('clock_mhz')}",
        f"cores={','.join(map(str, sorted(cores))) if cores is not None else 'all'} "
        f"engines={','.join(sorted(engines)) if engines is not None else 'all'} detail={int(detail)} "
        f"detail_wait_sync={'keep' if include_waits else 'hide'}",
        "NOTE time=us; independent diagnostic pass; rank-local timestamps; no time compression; summaries rounded to .001us.",
        "NOTE sum overlaps across engines; union/span/gap describe recorded scopes, not hardware utilization.",
        "NOTE quiet/drain/wait can overlap in-flight work; submit bytes/duration is NOT link bandwidth.",
        "NOTE summaries include waits PRESENT in source; FILTERED_WAITS means intentionally hidden waits; "
        "COALESCED_GMM means joined tile/drain records (possibly with waits hidden); "
        "INCOMPLETE means unaccounted filtered/lost records. None means zero waiting.",
        f"SELECTED first_us={total['first']:.3f} last_us={total['last']:.3f} span_us={total['span']:.3f} "
        f"sum_us={total['sum']:.3f} union_us={total['union']:.3f}",
        "[LEGEND] id name category",
    ]
    for event_id in sorted({e["id"] for e in events}):
        lines.append(f"{event_id} {' '.join(definitions[event_id])}")
    by_op, by_core, by_peer = defaultdict(list), defaultdict(list), defaultdict(list)
    for event in events:
        by_op[event["id"]].append(event)
        by_core[event["tid"]].append(event)
        if event["args"].get("peer") is not None:
            by_peer[event["args"]["peer"]].append(event)
    lines.append("[OP] id count sum_us union_us p50_us p95_us max_us first_us last_us payload_B")
    for event_id, rows in sorted(by_op.items()):
        s = stats(rows)
        payload = sum(row["args"].get("bytes", 0) or 0 for row in rows)
        lines.append(
            f"{event_id} {len(rows)} {s['sum']:.3f} {s['union']:.3f} {s['p50']:.3f} {s['p95']:.3f} "
            f"{s['maximum']:.3f} {s['first']:.3f} {s['last']:.3f} {payload}"
        )
    lines.append("[CORE] block/engine; op=id:sum_us; gap=span-union of recorded events")
    for tid, rows in sorted(by_core.items()):
        s = stats(rows)
        sums = defaultdict(float)
        for row in rows:
            sums[row["id"]] += row["dur"]
        by_phase = ",".join(f"{key}:{value:.3f}" for key, value in sorted(sums.items()))
        lines.append(
            f"{engine_label(tid)} n={len(rows)} first_us={s['first']:.3f} last_us={s['last']:.3f} "
            f"span_us={s['span']:.3f} sum_us={s['sum']:.3f} union_us={s['union']:.3f} "
            f"gap_us={max(0.0, s['span'] - s['union']):.3f} op={by_phase}"
        )
    lines.append(
        "[PEER] peer DX_n DX_B DS_n DS_B CY_n CY_B Dcompletion_wait_us Ccompletion_wait_us; submit payload only"
    )
    for peer, rows in sorted(by_peer.items()):
        parts = [str(peer)]
        for name in ("DispatchPutX", "DispatchPutScale", "CombinePut"):
            matching = [row for row in rows if row["name"] == name]
            parts.extend((str(len(matching)), str(sum(row["args"].get("bytes", 0) or 0 for row in matching))))
        for names in (("DispatchQuiet", "DispatchSendComplete"), ("CombineQuiet", "CombineSendComplete")):
            parts.append(f"{math.fsum(row['dur'] for row in rows if row['name'] in names):.3f}")
        lines.append(" ".join(parts))
    lines.append(
        "[FIELDS] p=peer c=CV f=fragment e=local_expert r=row l=rows m=tile_m n=tile_n "
        "w=columns s=sync_stage b=bytes; unknown/zero-byte fields omitted"
    )
    for category in WAIT_CATEGORIES:
        rows = sorted((e for e in events if e["cat"] == category), key=lambda e: (-e["dur"], e["tid"], e["ts"]))
        lines.append(f"[LONG {category}] top={min(top, len(rows))} of={len(rows)}; block/engine ts_us dur_us id fields")
        lines.extend(f"{engine_label(row['tid'])} {event_line(row)}" for row in rows[:top])
    if detail:
        emitted = 0
        lines.append("[DETAIL] grouped by engine; ts_us dur_us id fields; original coordinates")
        for tid, rows in sorted(by_core.items()):
            lines.append(f"@ {engine_label(tid)}")
            for row in sorted(rows, key=lambda e: e["ts"]):
                if not include_waits and row["cat"] == "wait.sync":
                    continue
                lines.append(event_line(row))
                emitted += 1
        lines.append(f"[DETAIL_END] emitted={emitted} omitted_wait_sync={len(events) - emitted}")
    return "\n".join(lines) + "\n"


def parse_cores(value):
    result = set()
    try:
        for part in value.split(","):
            bounds = part.split("-")
            start, end = (int(bounds[0]), int(bounds[-1]))
            if len(bounds) > 2 or not 0 <= start <= end < 28:
                raise ValueError
            result.update(range(start, end + 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use block numbers 0..27, e.g. 0,1,15-17") from error
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path)
    parser.add_argument("--output", type=Path, help="Default: input name with .txt suffix")
    parser.add_argument("--detail", action="store_true", help="Append per-event rows; much larger than summary")
    parser.add_argument(
        "--include-waits", action="store_true", help="Keep wait.sync rows in detail; summaries always keep them"
    )
    parser.add_argument("--cores", type=parse_cores, help="Block indices, e.g. 0,1,15-17; default all 28")
    parser.add_argument("--engines", help="Comma-separated AIC,AIV0,AIV1; default all")
    parser.add_argument("--top", type=int, default=8, help="Longest rows per wait/quiet/drain category")
    args = parser.parse_args(argv)
    output = args.output or args.json.with_suffix(".txt")
    try:
        if args.json.resolve() == output.resolve() or (output.exists() and args.json.samefile(output)):
            raise ValueError("Output must not overwrite the source JSON")
        report = render_report(
            json.loads(args.json.read_text(encoding="utf-8-sig")),
            source=args.json,
            cores=args.cores,
            engines=set(args.engines.split(",")) if args.engines else None,
            detail=args.detail,
            include_waits=args.include_waits,
            top=args.top,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"URMA trace text export failed: {error}\n")
    print(f"{output} ({len(report.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
