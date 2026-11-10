#!/usr/bin/env python3
"""Compact shared URMA single-rank/allrank JSON; standard library, no Torch/NPU."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import urma_trace_to_text as base

STAGES = {
    "Dsubmit": {1, 2, 4, 48, 53},
    "Csubmit": {10, 30, 49, 61},
    "Dcomplete": {43},
    "Dreceive": {65},
    "Ccomplete": {42},
    "routedGMM": {15, 16, 19, 21, 27, 33, 34, 35, 36},
    "sharedGMM1": {55, 56},
    "sharedGMM2": {58, 59},
    "allGMM": {15, 16, 19, 21, 27, 33, 34, 35, 36, 46, 55, 56, 58, 59},
    "dequant": {17, 20, 37, 38, 47, 55, 56, 58, 59},
    "pack": {0, 44, 54},
    "activation": {23, 50, 57},
    "reduce": {25, 32, 51},
    "sharedAdd": {60},
    "sync": {24, 26, 45, 52},
}


def split_ranks(doc):
    if not isinstance(doc, dict) or not isinstance(doc.get("traceEvents"), list):
        raise ValueError("Expected Chrome trace JSON, not summary.json")
    meta = doc.get("metadata", {})
    per_rank = {m["rank"]: m for m in meta.get("per_rank", [])}
    grouped = defaultdict(list)
    for e in doc["traceEvents"]:
        if e.get("ph") == "X":
            grouped[e.get("pid")].append(e)
    if not grouped:
        raise ValueError("Empty trace")
    if len(grouped) > 1 and not per_rank:
        raise ValueError("allrank needs metadata.per_rank; cannot invent rank-local spans")
    result = {}
    for rank, rows in grouped.items():
        m = per_rank.get(rank, meta if len(grouped) == 1 else None)
        if m is None:
            raise ValueError(f"Missing rank {rank} metadata")
        result[rank] = base.read_events(dict(metadata=m, traceEvents=rows))
    return result


def render(doc, *, source="trace.json", ranks="slowest", detail=False, cores=None, digits=2, top=5):
    data = split_ranks(doc)

    def fmt(x):
        return "-" if x is None else f"{x:.{digits}f}"

    def span(rank):
        meta, rows, _, _ = data[rank]
        if meta.get("envelope_us") is not None:
            return base.number(meta["envelope_us"], "envelope_us")
        return base.stats(rows)["span"]

    def interval(rows):
        if not rows:
            return "-"
        s = base.stats(rows)
        return f"{fmt(s['first'])}:{fmt(s['last'])}({fmt(s['span'])})"

    def stage(rows, name):
        out = [e for e in rows if e["id"] in STAGES[name]]
        if name in ("allGMM", "routedGMM", "sharedGMM1", "sharedGMM2"):
            out = [e for e in out if e["tid"] % 3 == 0]
        if name == "Dreceive":
            out = [e for e in out if e.get("args", {}).get("rows", 0) > 0]
        if name == "dequant":
            out = [e for e in out if e["tid"] % 3 == 1]
        return out

    if ranks == "slowest":
        selected = [max(sorted(data), key=span)]
    elif ranks == "all":
        selected = sorted(data)
    else:
        try:
            selected = sorted(set(map(int, ranks.split(","))))
        except ValueError:
            raise ValueError("--ranks must be slowest / all / 0,3,...")
        if not selected or any(r not in data for r in selected):
            raise ValueError("Requested rank missing")
    lines = [
        f"URMA_SHARED_TEXT_V1 source={json.dumps(str(source), ensure_ascii=False)}",
        f"ranks={','.join(map(str, sorted(data)))} selected={','.join(map(str, selected))} detail={int(detail)} digits={digits}",
        "NOTE us; rank clocks independent; preserve gaps; keep ALL source waits, quiet, drain and final sync.",
        "NOTE Dsubmit/Csubmit are SUBMISSION windows, not network service time. No ready-time inference from missing events.",
        "NOTE coarse/fine are separate invocations: do not align their timestamps or treat differences as one run.",
        "NOTE source-hidden waits cannot be recovered; coarse cannot resolve per-CV readiness. Rounded values only for display.",
        "[RANK] rank diagnostic_us source_status instrumentation source_events hidden_waits dropped",
    ]
    for rank in sorted(data):
        m, rows, _, status = data[rank]
        lines.append(
            f"{rank} {fmt(span(rank))} {status} {m.get('instrumentation', '?')} {len(rows)} {m.get('hidden_wait_count', 0)} {m.get('dropped', 0)}"
        )
        if m.get("model") is not None:
            lines.append(
                f"MODEL rank={rank} bs={m.get('bs')} ep={m.get('world')} "
                + json.dumps(m["model"], ensure_ascii=False, separators=(",", ":"))
            )
        lines.append(
            "BOUND rank=" + str(rank) + " " + " ".join(f"{key}={interval(stage(rows, key))}" for key in STAGES)
        )
    declared = doc.get("metadata", {}).get("rank_count")
    if declared is not None and declared != len(data):
        lines.append(f"WARNING declared_rank_count={declared} actual={len(data)}")
    for rank in selected:
        meta, original, definitions, status = data[rank]
        rows = [e for e in original if cores is None or e["tid"] // 3 in cores]
        if not rows:
            raise ValueError("No events in selected cores")
        lines.append(f"[RANK_DETAIL] rank={rank} source_status={status} selected_events={len(rows)}")
        lines.append("[LEGEND] id name category")
        for id in sorted({e["id"] for e in rows}):
            lines.append(f"{id} {' '.join(definitions[id])}")
        lines.append(
            "[FIELDS] p=peer c=CV f=fragment e=expert r=row l=rows m=tile_m n=tile_n w=columns s=sync_stage b=bytes"
        )
        lanes = defaultdict(list)
        peers = defaultdict(list)
        for e in rows:
            lanes[e["tid"]].append(e)
            if e["args"].get("peer") is not None:
                peers[e["args"]["peer"]].append(e)
        lines.append("[LANE] block/engine first:last(span) union_us gaps_us; op=id:count/sum_us")
        for tid, lane in sorted(lanes.items()):
            ss = base.stats(lane)
            ops = defaultdict(list)
            for e in lane:
                ops[e["id"]].append(e)
            lines.append(
                f"{base.engine_label(tid)} {interval(lane)} union={fmt(ss['union'])} gaps={fmt(max(0, ss['span'] - ss['union']))} "
                + " ".join(f"{id}:{len(es)}/{fmt(sum(e['dur'] for e in es))}" for id, es in sorted(ops.items()))
            )
        lines.append(
            "[PEER] p stage put_count payload_B first:last(span); longest gaps prev_end->next_start:us,next_cv"
        )
        for peer, events in sorted(peers.items()):
            for name, ids in [("DX", {1}), ("DS", {2}), ("CY", {10})]:
                puts = sorted([e for e in events if e["id"] in ids], key=lambda e: e["ts"])
                if not puts:
                    continue
                gaps = [
                    (max(0, b["ts"] - a["ts"] - a["dur"]), a["ts"] + a["dur"], b["ts"], b["args"].get("cv"))
                    for a, b in zip(puts, puts[1:])
                ]
                lines.append(
                    f"{peer} {name} {len(puts)} {sum(e['args'].get('bytes', 0) or 0 for e in puts)} {interval(puts)} "
                    + " ".join(f"{fmt(a)}->{fmt(b)}:{fmt(g)},c{cv}" for g, a, b, cv in sorted(gaps, reverse=True)[:top])
                )
        if detail:
            lines.append("[EVENTS] grouped by lane; start_us duration_us id fields (waits kept)")
            for tid, lane in sorted(lanes.items()):
                lines.append("@ " + base.engine_label(tid))
                for e in sorted(lane, key=lambda x: x["ts"]):
                    lines.append(f"{fmt(e['ts'])} {fmt(e['dur'])} {e['id']} {base.short_args(e)}".rstrip())
                    sources = e["args"].get("source_records")
                    if sources:
                        clock = meta.get("clock_mhz")
                        origin = e["args"].get("start_tick")
                        if clock and origin is not None:
                            parts = [
                                f"{part['event_id']}:{fmt(e['ts'] + (int(part['start_tick']) - int(origin)) / clock)}:{fmt((int(part['end_tick']) - int(part['start_tick'])) / clock)}"
                                for part in sources
                            ]
                            lines.append(" src(id:start:dur)=" + " ".join(parts))
                        else:
                            lines.append(" src_ticks=" + json.dumps(sources, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def render_compact(doc, *, source="trace.json", limit=60000):
    """A bounded diagnosis, not an event dump. ASCII byte/character limit."""
    if not 4096 <= limit <= 60000:
        raise ValueError("--max-chars must be in [4096,60000]")
    data = split_ranks(doc)
    lines = []
    used = 0
    omitted = defaultdict(int)

    # Reserve space for omission counts and end marker. Never split an event/line.
    def add(line, section):
        nonlocal used
        line = str(line).encode("ascii", "backslashreplace").decode("ascii")
        if used + len(line) + 1 > limit - 1024:
            omitted[section] += 1
            return
        lines.append(line)
        used += len(line) + 1

    def f(x):
        return "-" if x is None else f"{x:.2f}"

    def edge(rows):
        return (min(e["ts"] for e in rows), max(e["ts"] + e["dur"] for e in rows)) if rows else (None, None)

    def select(rows, ids, engine=None):
        return [e for e in rows if e["id"] in ids and (engine is None or e["tid"] % 3 == engine)]

    def pos(rows):
        return "/".join(f(x) for x in edge(rows))

    add("URMA_COMMENT_V1 source=" + json.dumps(Path(source).name, ensure_ascii=True), "header")
    add("units=us,B; all timestamps rank-local; no cross-rank alignment. start/end pairs, NOT durations.", "header")
    add(
        "Submission!=completion. Reduce end!=send completion. No source waits reconstructed. No per-event dump.",
        "header",
    )
    add("D=DX+DS; C=CY. WaitDispatch may include Pack readiness or shared-GMM1 throttle.", "header")
    add("PEER windows are submissions; Q=final quiet end. R=rank latest Reduce end (proxy only).", "header")
    models = set()
    gaps = []
    lanes = {}
    peer_rows = []
    add("[RANK] r mode status n hidden dropped diag Dsubmit Csubmit Qend GMM sharedG1 sharedG2 Reduce Add", "rank")
    for rank, (meta, rows, definitions, status) in sorted(data.items()):
        model = json.dumps(
            dict(bs=meta.get("bs"), ep=meta.get("world"), model=meta.get("model")),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        if model not in models:
            add("MODEL " + model, "model")
            models.add(model)
        diag = meta.get("envelope_us", base.stats(rows)["span"])
        ds = select(rows, {1, 2, 4, 48, 53})
        cs = select(rows, {10, 30, 49, 61})
        q = edge(select(rows, {42}))[1]
        red = edge(select(rows, {25, 32, 51}))[1]
        add(
            f"{rank} {meta.get('instrumentation', '?')} {status} {len(rows)} {meta.get('hidden_wait_count', 0)} {meta.get('dropped', 0)} {f(diag)} "
            + " ".join(
                [
                    pos(ds),
                    pos(cs),
                    f(q),
                    pos(select(rows, STAGES["allGMM"], 0)),
                    pos(select(rows, {55, 56}, 0)),
                    pos(select(rows, {58, 59}, 0)),
                    pos(select(rows, {25, 32, 51})),
                    pos(select(rows, {60})),
                ]
            ),
            "rank",
        )
        peers = defaultdict(list)
        ls = defaultdict(list)
        for e in rows:
            ls[e["tid"]].append(e)
            if e["args"].get("peer") is not None:
                peers[(e["tid"], e["args"]["peer"])].append(e)
        for tid, es in ls.items():
            lanes[(rank, tid)] = sorted(es, key=lambda e: e["ts"])
        for (tid, peer), es in sorted(peers.items()):
            dx = sorted(select(es, {1}), key=lambda e: e["ts"])
            scale = select(es, {2})
            cy = sorted(select(es, {10}), key=lambda e: e["ts"])
            if not dx and not cy:
                continue

            def payload(items):
                return sum(e["args"].get("bytes", 0) or 0 for e in items)

            def sizes(items):
                values = sorted(e["args"].get("rows") for e in items if e["args"].get("rows") is not None)
                return "-/-/-" if not values else f"{values[0]}/{values[len(values) // 2]}/{values[-1]}"

            stage_gaps = []
            for name, puts in [("D", dx), ("C", cy)]:
                local = []
                for a, b in zip(puts, puts[1:]):
                    start = a["ts"] + a["dur"]
                    end = b["ts"]
                    duration = max(0, end - start)
                    if duration > 0:
                        local.append((duration, rank, tid, peer, name, start, end, a, b))
                gaps.extend(sorted(local, key=lambda x: x[0], reverse=True)[:3])
                stage_gaps.append(max((g[0] for g in local), default=0))
            peer_rows.append(
                f"{rank} {peer} {base.engine_label(tid)} {len(dx)}/{len(cy)} {payload(dx) + payload(scale)}/{payload(cy)} {pos(dx)} {pos(cy)} {f(edge(select(es, {42}))[1])} {f(red)} {sizes(dx)} {sizes(cy)} "
                + "/".join(f(x) for x in stage_gaps)
            )
    # A coarse file has no per-peer PUTs; keep all its lane boundaries instead.
    if not peer_rows:
        add("[COARSE_LANES] r lane id:name first/end sum count", "coarse")
        for (rank, tid), es in sorted(lanes.items()):
            ops = defaultdict(list)
            for e in es:
                ops[e["id"]].append(e)
            for id, items in sorted(ops.items()):
                add(
                    f"{rank} {base.engine_label(tid)} {id}:{items[0]['name']} {pos(items)} {f(sum(e['dur'] for e in items))} {len(items)}",
                    "coarse",
                )
    else:
        add(
            "[PEER] r p lane Dn/Cn DB/CB Dstart/end Cstart/end Qend Rend Drows(min/median/max) Crows(min/median/max) DgapMax/CgapMax",
            "peer",
        )
        for line in peer_rows:
            add(line, "peer")
        add(
            "[GAPS] largest 12 observed inter-PUT gaps (max3 per peer/stage); clipped SAME-lane operation union inside gap",
            "gaps",
        )
        chosen = sorted(gaps, key=lambda x: x[0], reverse=True)[:12]
        for n, (duration, rank, tid, peer, name, start, end, a, b) in enumerate(chosen):
            add(
                f"G{n} r={rank} p={peer} lane={base.engine_label(tid)} stage={name} gap={f(duration)} from={f(start)} to={f(end)} prevCV={a['args'].get('cv')} nextCV={b['args'].get('cv')}",
                "gaps",
            )
            local = lanes[(rank, tid)]
            inside = [e for e in local if e["ts"] < end and e["ts"] + e["dur"] > start]
            ops = defaultdict(list)
            for e in inside:
                lo = max(start, e["ts"])
                hi = min(end, e["ts"] + e["dur"])
                ops[(e["id"], e["name"])].append(dict(ts=lo, dur=hi - lo))
            for (id, op), items in sorted(ops.items()):
                add(f" G{n} op={id}:{op} union={f(base.intervals(items))} n={len(items)}", "gap_ops")
            add(
                f" G{n} no_recorded_scope={f(max(0, duration - base.intervals([x for es in ops.values() for x in es])))} (not proof of hardware idle)",
                "gap_ops",
            )
            # Include edge events, never dump all intervening tokens/tiles.
            context = inside[:2] + inside[-2:] + [a, b]
            seen = set()
            for e in sorted(context, key=lambda x: x["ts"]):
                key = (e["ts"], e["dur"], e["id"])
                if key in seen:
                    continue
                seen.add(key)
                add(f" G{n} event {f(e['ts'])} {f(e['dur'])} {e['id']}:{e['name']} {base.short_args(e)}", "context")
        add(f"GAP_SELECTION candidates={len(gaps)} shown={len(chosen)}; lower-ranked gaps omitted by design", "gaps")
        # Slowest diagnostic rank: per-core GMM/dequant/reduce boundaries reveal
        # producer imbalance without thousands of tile events.
        slow = max(sorted(data), key=lambda r: data[r][0].get("envelope_us", base.stats(data[r][1])["span"]))
        add(f"[SLOW_RANK_LANES] rank={slow}; lane GMM dequant Reduce last_record_end", "lanes")
        for (rank, tid), es in sorted(lanes.items()):
            if rank != slow:
                continue
            add(
                f"{base.engine_label(tid)} {pos(select(es, STAGES['allGMM'], 0))} {pos(select(es, STAGES['dequant'], 1))} {pos(select(es, STAGES['reduce']))} {f(edge(es)[1])}",
                "lanes",
            )
    footer = "OMITTED_BY_BYTE_BUDGET " + json.dumps(dict(omitted), sort_keys=True, separators=(",", ":"))
    lines.extend([footer, "END URMA_COMMENT_V1 (full source JSON retained locally)"])
    result = "\n".join(lines) + "\n"
    assert len(result.encode("utf-8")) <= limit and len(result) <= limit
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--ranks", default="slowest", help="slowest (diagnostic span), all, or comma-separated ranks")
    p.add_argument(
        "--compact", action="store_true", help="Comment-safe diagnosis, <=60000 ASCII characters/bytes; no event dump"
    )
    p.add_argument("--max-chars", type=int, default=60000, help="Compact limit, 4096..60000")
    p.add_argument("--detail", action="store_true", help="Append all selected events; waits are always kept")
    p.add_argument("--cores", type=base.parse_cores, help="Optional 0,1,15-17; all lanes by default")
    p.add_argument("--digits", type=int, choices=range(0, 7), default=2)
    p.add_argument("--top", type=int, default=5)
    a = p.parse_args()
    out = a.output or a.json.with_suffix(".shared.txt")
    try:
        if a.top < 0:
            raise ValueError("--top must be nonnegative")
        if out.resolve() == a.json.resolve() or (out.exists() and out.samefile(a.json)):
            raise ValueError("Cannot overwrite input JSON")
        document = json.loads(a.json.read_text(encoding="utf-8-sig"))
        text = (
            render_compact(document, source=a.json, limit=a.max_chars)
            if a.compact
            else render(
                document, source=a.json, ranks=a.ranks, detail=a.detail, cores=a.cores, digits=a.digits, top=a.top
            )
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        p.exit(2, f"Export failed: {error}\n")
    print(f"{out} ({len(text.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
