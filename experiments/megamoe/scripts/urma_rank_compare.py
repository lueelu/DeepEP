#!/usr/bin/env python3
"""Fastest/slowest URMA rank diagnosis, <=60000 ASCII characters/bytes. No Torch."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import urma_shared_trace_to_text as shared

G1 = {15, 16, 27}
G2 = {19, 21}
SHARED1 = {55, 56}
SHARED2 = {58, 59}
GMM = G1 | G2 | SHARED1 | SHARED2 | {33, 34, 35, 36, 46}
WAIT = {14, 18, 22, 29, 9, 6, 8, 31, 41}
STAGES = [
    ("GMM", GMM, 0),
    ("SharedG1", SHARED1, 0),
    ("SharedG2", SHARED2, 0),
    ("RouteG1", G1, 0),
    ("RouteG2", G2, 0),
    ("Dequant", {17, 20, 37, 38, 47, 55, 56, 58, 59}, 1),
    ("Pack", {0, 44, 54}, None),
    ("ScaleGather", {28}, None),
    ("Activation", {23, 50}, None),
    ("SharedActivation", {57}, None),
    ("Dsubmit", {1, 2, 4, 48, 53}, None),
    ("Dreceive", {65}, None),
    ("Csubmit", {10, 30, 49, 61}, None),
    ("Reorder", {40}, None),
    ("Quiet", {3, 11, 42, 43}, None),
    ("Reduce", {25, 32, 51}, None),
    ("SharedAdd", {60}, None),
    ("Sync", {24, 26, 45, 52}, None),
]
SOURCE_NAMES = {
    15: "Gmm1Tile",
    16: "Gmm1TileDrain",
    27: "Gmm1Drain",
    19: "Gmm2Tile",
    21: "Gmm2Drain",
    14: "WaitDispatch",
    18: "WaitActivation",
    29: "WaitInputScale",
    17: "Gmm1Dequant",
    20: "Gmm2Dequant",
}


def expand(rows, meta):
    result = []
    unresolved = 0
    for e in rows:
        parts = e["args"].get("source_records")
        if not parts:
            result.append(e)
            continue
        clock = meta.get("clock_mhz")
        start = e["args"].get("start_tick")
        if not isinstance(clock, (int, float)) or not math.isfinite(clock) or clock <= 0 or start is None:
            result.append(e)
            unresolved += 1
            continue
        for part in parts:
            t0 = int(part["start_tick"])
            t1 = int(part["end_tick"])
            if t1 < t0:
                raise ValueError("Invalid composite source interval")
            args = {**e["args"], **part}
            args.pop("source_records", None)
            result.append(
                dict(
                    e,
                    id=part["event_id"],
                    name=SOURCE_NAMES.get(part["event_id"], e["name"]),
                    ts=e["ts"] + (t0 - int(start)) / clock,
                    dur=(t1 - t0) / clock,
                    args=args,
                )
            )
    return result, unresolved


def render(document, source="allrank.json", limit=60000):
    if not 4096 <= limit <= 60000:
        raise ValueError("--max-chars must be4096..60000")
    ranks = shared.split_ranks(document)
    if len(ranks) < 2:
        raise ValueError("Use allrank.json with at least two ranks")
    declared = document.get("metadata", {}).get("rank_count")
    if declared is not None and declared != len(ranks):
        raise ValueError("Missing ranks: cannot select global fastest/slowest")

    def f(x):
        return "-" if x is None else f"{x:.2f}"

    def span(rank):
        meta, rows, _, _ = ranks[rank]
        value = meta.get("envelope_us")
        return shared.base.number(value, "envelope_us") if value is not None else shared.base.stats(rows)["span"]

    order = sorted(ranks, key=lambda r: (span(r), r))
    fast = order[0]
    slow = order[-1]
    selected = [fast, slow]
    out = []
    used = 0
    omitted = defaultdict(int)

    def add(line, section="summary"):
        nonlocal used
        line = str(line).encode("ascii", "backslashreplace").decode("ascii")
        if used + len(line) + 1 > limit - 1024:
            omitted[section] += 1
            return
        out.append(line)
        used += len(line) + 1

    def choose(rows, ids, engine=None):
        return [e for e in rows if e["id"] in ids and (engine is None or e["tid"] % 3 == engine)]

    def edge(rows):
        return (min(e["ts"] for e in rows), max(e["ts"] + e["dur"] for e in rows)) if rows else (None, None)

    def window(rows):
        return "/".join(f(v) for v in edge(rows))

    def union(rows):
        return shared.base.intervals(rows) if rows else 0

    def longest(rows):
        lanes = defaultdict(list)
        for e in rows:
            lanes[e["tid"]].append(e)
        return max((union(es) for es in lanes.values()), default=0)

    def brief(e):
        return f"{shared.base.engine_label(e['tid'])} {f(e['ts'])} {f(e['dur'])} {e['id']}:{e['name']} {shared.base.short_args(e)}".rstrip()

    add("URMA_RANK_PAIR_V1 source=" + json.dumps(Path(source).name, ensure_ascii=True))
    add(f"FAST={fast} SLOW={slow} selected_by=diagnostic_envelope_us delta={f(span(slow) - span(fast))}")
    add("NOTE us,B; rank-local independent clocks. Compare intervals/durations, never cross-rank absolute timestamps.")
    add(
        "NOTE coarse and fine are DIFFERENT invocations. Filtered source waits cannot be recovered; missing wait rows do NOT mean zero waits."
    )
    add(
        "NOTE GMM scopes include software overhead. PUT submit != arrival. Retire-end precedes flag publication; not exact consumer-ready time."
    )
    add("FIELDS p=peer c=CV f=fragment e=expert r=row l=rows m=tile_m n=tile_n w=columns b=bytes")
    add("[ALL_RANKS] rank diagnostic_us source_events hidden_waits dropped status")
    for rank in sorted(ranks):
        meta, rows, _, status = ranks[rank]
        if meta.get("dropped", 0):
            raise ValueError(f"rank{rank}: dropped trace records")
        add(f"{rank} {f(span(rank))} {len(rows)} {meta.get('hidden_wait_count', 0)} {meta.get('dropped', 0)} {status}")
    data = {}
    lanes = {}
    for rank in selected:
        meta, original, _, status = ranks[rank]
        rows, unresolved = expand(original, meta)
        data[rank] = rows
        lane = defaultdict(list)
        for e in rows:
            lane[e["tid"]].append(e)
        lanes[rank] = {t: sorted(es, key=lambda e: e["ts"]) for t, es in lane.items()}
        add(
            f"MODEL rank={rank} mode={meta.get('instrumentation', '?')} bs={meta.get('bs')} ep={meta.get('world')} status={status} unresolved_composites={unresolved} "
            + json.dumps(meta.get("model"), ensure_ascii=True, separators=(",", ":"))
        )
        if meta.get("hidden_wait_count", 0):
            add(
                f"WARNING rank={rank}: waits were hidden in source; reported wait sums are only available recorded waits."
            )
    add("[STAGES] stage FAST(start/end longest_core_sum count) SLOW(start/end longest_core_sum count)")
    for name, ids, engine in STAGES:
        parts = []
        for rank in selected:
            es = choose(data[rank], ids, engine)
            if name == "Dreceive":
                es = [e for e in es if e.get("args", {}).get("rows", 0) > 0]
            parts.append(f"{window(es)} {f(longest(es)) if es else '-'} {len(es)}")
        add(name + " | " + " | ".join(parts))
    add("[WAITS] stage FAST(longest_core_sum max_record count) SLOW(...); missing=-, not0")
    for label, ids, engine in [
        ("AIC-input", {29}, 0),
        ("AIC-activation", {18}, 0),
        ("peer-start", {6}, 2),
        ("dispatch-ready", {14}, 2),
        ("combine-ready", {9}, 2),
        ("activation-helper", {22}, 2),
        ("reduce-ready", {31}, None),
        ("reorder-ready", {41}, 2),
    ]:
        parts = []
        for rank in selected:
            es = choose(data[rank], ids, engine)
            parts.append(f"{f(longest(es)) if es else '-'} {f(max((e['dur'] for e in es), default=None))} {len(es)}")
        add(label + " | " + " | ".join(parts))
    # Interleave pair rows: when a tighter budget is requested neither rank gets all space.
    add("[CORE_PAIR] rank lane GMM Dequant Reduce; G1_submit/G2_submit/Drain/InputWait/ActWait sums; unrecorded_gap_us")
    for tid in range(84):
        for rank in selected:
            es = lanes[rank].get(tid, [])
            if not es:
                continue
            total = edge(es)
            gaps = max(0, total[1] - total[0] - union(es))
            sums = [
                union(choose(es, ids)) if choose(es, ids) else None
                for ids in [{15}, {19}, {16, 21, 27, 56, 59}, {29}, {18}]
            ]
            add(
                f"{rank} {shared.base.engine_label(tid)} {window(choose(es, GMM, 0))} {window(choose(es, {17, 20, 37, 38, 47, 55, 56, 58, 59}, 1))} {window(choose(es, {25, 32, 51}))} "
                + "/".join(f(v) for v in sums)
                + f" gap={f(gaps)}",
                "core",
            )
    peer_maps = {}
    gap_lists = {}
    for rank in selected:
        peers = defaultdict(list)
        gaps = []
        for e in data[rank]:
            if e["args"].get("peer") is not None:
                peers[e["args"]["peer"]].append(e)
        peer_maps[rank] = peers
        for peer, es in peers.items():
            for name, ids in [("DX", {1}), ("CY", {10})]:
                puts = sorted(choose(es, ids), key=lambda e: e["ts"])
                for a, b in zip(puts, puts[1:]):
                    gap = b["ts"] - a["ts"] - a["dur"]
                    if gap > 0:
                        gaps.append((gap, peer, name, a, b))
        gap_lists[rank] = sorted(gaps, key=lambda x: x[0], reverse=True)
    add("[PEERS] rank peer Dstart/end Cstart/end Qend DXn/CYn DX+DS_B/CY_B CYrows(min/median/max)")
    for peer in sorted(set(peer_maps[fast]) | set(peer_maps[slow])):
        for rank in selected:
            es = peer_maps[rank].get(peer, [])
            dx = choose(es, {1})
            cy = choose(es, {10})
            ds = choose(es, {2})
            if not dx and not cy:
                continue
            sizes = sorted(e["args"]["rows"] for e in cy if e["args"].get("rows") is not None)
            size = "-/-/-" if not sizes else f"{sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]}"

            def byte(ev):
                return sum(e["args"].get("bytes", 0) or 0 for e in ev)

            add(
                f"{rank} {peer} {window(dx)} {window(cy)} {f(edge(choose(es, {42}))[1])} {len(dx)}/{len(cy)} {byte(dx + ds)}/{byte(cy)} {size}",
                "peer",
            )
    # Expert progression and group dependency distances. Uses observed scopes only.
    expert_maps = {}
    group_maps = {}
    for rank in selected:
        experts = defaultdict(list)
        groups = defaultdict(list)
        for e in data[rank]:
            if e["id"] not in G1 | G2 | {23, 28}:
                continue
            if e["args"].get("expert") is not None and e["args"]["expert"] < 384:
                experts[e["args"]["expert"]].append(e)
            if e["args"].get("fragment") is not None:
                groups[e["args"]["fragment"]].append(e)
        expert_maps[rank] = experts
        group_maps[rank] = groups
    add(
        "[EXPERT_PROGRESS] rank expert groups G1start/end ActStart/end G2start/end; same expert IDs across ranks are different weights"
    )
    for expert in sorted(set(expert_maps[fast]) | set(expert_maps[slow])):
        for rank in selected:
            es = expert_maps[rank].get(expert, [])
            if not es:
                continue
            groups = {e["args"]["fragment"] for e in es if e["args"].get("fragment") is not None}
            add(
                f"{rank} {expert} {len(groups)} {window(choose(es, G1, 0))} {window(choose(es, {23}))} {window(choose(es, G2, 0))}",
                "expert",
            )
    add(
        "[GROUP_DELAYS] top8 per rank: observed ActStart-G1end and G2Start-ActEnd; negative means overlapping scopes, not a ready timestamp"
    )
    for rank in selected:
        candidates = []
        for group, es in group_maps[rank].items():
            g1 = edge(choose(es, G1, 0))
            act = edge(choose(es, {23}))
            g2 = edge(choose(es, G2, 0))
            da = act[0] - g1[1] if act[0] is not None and g1[1] is not None else None
            dg = g2[0] - act[1] if g2[0] is not None and act[1] is not None else None
            if da is None and dg is None:
                continue
            candidates.append((max(v for v in [da, dg] if v is not None), group, es, da, dg))
        for _, group, es, da, dg in sorted(candidates, key=lambda x: x[0], reverse=True)[:8]:
            add(
                f"{rank} f{group} G1={window(choose(es, G1, 0))} Act={window(choose(es, {23}))} G2={window(choose(es, G2, 0))} Act-G1={f(da)} G2-Act={f(dg)}",
                "group",
            )
    add("[LONG_WAITS] top8 AIC waits per rank: lane start dur id:name fields")
    for rank in selected:
        for e in sorted(choose(data[rank], {18, 29}, 0), key=lambda e: e["dur"], reverse=True)[:8]:
            add(f"{rank} " + brief(e), "wait")
    add("[PUT_GAPS] top6 per rank: same-lane clipped scopes, then2 edge records; no recorded scope != hardware idle")
    for index in range(6):
        for rank in selected:
            if index >= len(gap_lists[rank]):
                continue
            gap, peer, name, a, b = gap_lists[rank][index]
            lo = a["ts"] + a["dur"]
            hi = b["ts"]
            add(
                f"{rank} p{peer} {name} gap={f(gap)} {f(lo)}->{f(hi)} c{a['args'].get('cv')}->c{b['args'].get('cv')}",
                "gap",
            )
            ops = defaultdict(list)
            for e in lanes[rank].get(a["tid"], []):
                if e["ts"] < hi and e["ts"] + e["dur"] > lo:
                    start = max(lo, e["ts"])
                    end = min(hi, e["ts"] + e["dur"])
                    ops[(e["id"], e["name"])].append(dict(ts=start, dur=end - start))
            add(" scopes " + " ".join(f"{id}:{name}={f(union(es))}" for (id, name), es in sorted(ops.items())), "gap")
            add(f" unrecorded={f(max(0, gap - union([x for es in ops.values() for x in es])))}", "gap")
            add(" prev " + brief(a), "gap")
            add(" next " + brief(b), "gap")
    out += [
        "OMITTED_BY_BUDGET " + json.dumps(dict(omitted), separators=(",", ":")),
        "END URMA_RANK_PAIR_V1; original JSON unchanged",
    ]
    result = "\n".join(out) + "\n"
    assert len(result) <= limit and len(result.encode("utf-8")) <= limit
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--max-chars", type=int, default=60000)
    a = p.parse_args()
    out = a.output or a.json.with_suffix(".rank-pair.txt")
    try:
        if out.resolve() == a.json.resolve() or (out.exists() and out.samefile(a.json)):
            raise ValueError("Output cannot overwrite input JSON")
        result = render(json.loads(a.json.read_text(encoding="utf-8-sig")), str(a.json), a.max_chars)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        p.exit(2, f"Export failed: {error}\n")
    print(f"{out}: {len(result)} characters / {len(result.encode('utf-8'))} bytes; limit {a.max_chars}")


if __name__ == "__main__":
    main()
