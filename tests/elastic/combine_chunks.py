"""Compact Combine chunk diagnostics and Perfetto/SVG/CSV timeline export."""

import argparse
import csv
import html
import json
from pathlib import Path

EP8_SINGLE_QP_FLAG = 0x45505131
VERSION, MAGIC, WORDS, TICKS_PER_US = 2, 0x43484B50, 36, 1000
FIELDS = (
    "Magic Version Sample KindId Core Group Chunk Peer Begin WaitBegin Ready WorkBegin WorkEnd End Tokens Rows "
    "DirectRows ReducedRows FirstSend0 LastSend0 FirstSend1 LastSend1 FirstDirect LastDirect FirstReduced "
    "LastReduced PayloadWqes Sos PeerMask WindowEnd Flags Reserved BarrierBegin BarrierEnd Lane0Rows Lane1Rows"
).split()
KINDS = {1: "gather", 2: "gateway_reduce", 3: "cross_send", 4: "local_return", 5: "final_reduce"}
SEND_FIELDS = FIELDS[18:26]
TIME_FIELDS = FIELDS[8:14] + SEND_FIELDS + ["BarrierBegin", "BarrierEnd"]
NOTE = (
    "Rank-local system-counter timestamps; devices are not clock-aligned. "
    "Cross send times are measured immediately after payload SQ doorbell updates, not UB staging or remote arrival. "
    "A mixed batch can contain both direct and reduced rows, sharing a submission timestamp. "
    "Lane row counts measure submitted payload; multiply by 14336 for bytes. Cross group barriers are "
    "rank-local synchronization of the 16 send owners, after WorkEnd and before End. "
    "Gateway reduce ends after existing output-store drains and reduce_done publication; tokens=0 means no reduction. "
    "Final reduce adds diagnostic-only per-chunk local drains; incoming readiness is still observed per SO window. "
    "Gather/local-return work_end describes issue/publication, not an independent remote-arrival measurement. "
    "Use the separate uninstrumented profiling invocation for performance."
)


def shape(world, tokens, chunk_tokens=256):
    if chunk_tokens not in (256, 512, 1024):
        raise ValueError("chunk_tokens must be 256, 512 or 1024")
    if world not in (2, 4, 8, 16, 32, 64, 128, 256, 512) or not 1 <= tokens <= 16384:
        raise ValueError("Invalid Combine chunk shape")
    groups = world // min(world, 64)
    chunks = (tokens + chunk_tokens - 1) // chunk_tokens + min(4, tokens, chunk_tokens) - 1
    return 64, (groups + 2) * chunks, WORDS


def token_range(chunk, tokens, chunk_tokens=256):
    splits = min(4, tokens)
    first = min(chunk_tokens, tokens)
    if chunk < splits:
        # Same ceil partition as ChunkTokenEnd in combine.cpp.
        return (first * chunk + splits - 1) // splits, (first * (chunk + 1) + splits - 1) // splits
    begin = (chunk - splits + 1) * chunk_tokens
    return begin, min(tokens, begin + chunk_tokens)


def decode(records, *, rank, world, tokens, sample, chunk_tokens=256, ep8_rank_pipeline=False):
    if ep8_rank_pipeline and world != 8:
        raise ValueError("EP8 return pipeline diagnostics require EP8")
    expected = shape(world, tokens, chunk_tokens)
    local = 32 if ep8_rank_pipeline else min(world, 8)
    gateway_begin = 32 if ep8_rank_pipeline else 24 if world > 8 else 8
    groups = world // min(world, 64)
    chunks = expected[1] // (groups + 2)
    if len(records) != 64 or any(len(core) != expected[1] for core in records):
        raise ValueError("Invalid chunk tensor shape")
    compact = ep8_rank_pipeline and any(
        len(row) == WORDS and row[3] == 4 and row[30] == EP8_SINGLE_QP_FLAG for row in records[0]
    )
    window_pool = {}
    if compact:
        local = 11
        # Core 63 participates in every window. Require the final schedule,
        # rather than accepting experimental fixed-width or 32-AIV policies.
        for row in records[63]:
            if len(row) == WORDS and row[:3] == [MAGIC, VERSION, sample] and row[3] == 5:
                begin = 11 if row[29] == chunks - 1 else 32
                if row[30] != 0x45600000 | (64 << 8) | (64 - begin):
                    raise ValueError("Invalid window source-pool marker")
                window_pool[row[6]] = begin
        if set(window_pool) != set(range(chunks)):
            raise ValueError("Missing window source-pool plan")
    events = []
    final_seen, gateway_seen, cross_seen, local_seen = set(), set(), set(), set()
    for core, rows in enumerate(records):
        for index, words in enumerate(rows):
            if len(words) != WORDS or any(type(v) is not int or not 0 <= v < 2**63 for v in words):
                raise ValueError("Invalid chunk record words")
            if not any(words):
                continue  # Unused role/slot; producer record is optional only for empty gather.
            v = dict(zip(FIELDS, words))
            kind, group, chunk = v["KindId"], v["Group"], v["Chunk"]
            if words[:3] != [MAGIC, VERSION, sample] or v["Core"] != core or kind not in KINDS:
                raise ValueError("Missing/stale chunk record identity")
            if not 0 <= group < groups or not 0 <= chunk < chunks:
                raise ValueError("Invalid chunk group/index")
            expected_index = ((groups + 1) if kind == 5 else groups if kind == 4 else group) * chunks + chunk
            if index != expected_index:
                raise ValueError("Chunk record stored in another owner's slot")
            if not 0 < v["Begin"] <= v["End"]:
                raise ValueError("Invalid chunk envelope")
            if any(value and value > v["End"] for key, value in v.items() if key in TIME_FIELDS):
                raise ValueError("Chunk time exceeds its envelope")
            for a, b in (
                ("WorkBegin", "WorkEnd"),
                ("WaitBegin", "Ready"),
                ("FirstSend0", "LastSend0"),
                ("FirstSend1", "LastSend1"),
                ("FirstDirect", "LastDirect"),
                ("FirstReduced", "LastReduced"),
            ):
                if v[a] and (not v[b] or v[b] < v[a]):
                    raise ValueError(f"Invalid chunk interval {a}/{b}")
            if kind == 5:
                if compact and (
                    core < window_pool[chunk] or v["Flags"] != 0x45600000 | (64 << 8) | (64 - window_pool[chunk])
                ):
                    raise ValueError("Invalid EP8 source-pool owner/marker")
                if group != 0 or not chunk <= v["WindowEnd"] < chunks:
                    raise ValueError("Invalid final-reduce window")
                final_seen.add((core, chunk))
            elif kind == 2:
                if core < gateway_begin:
                    raise ValueError("Invalid gateway core")
                gateway_seen.add((core, group, chunk))
            elif kind == 3:
                if world <= 8 or not 8 <= core < 24:
                    raise ValueError("Invalid cross core")
                if v["PayloadWqes"] != v["DirectRows"] + v["ReducedRows"]:
                    raise ValueError("Chunk WQE/row count mismatch")
                if bool(v["DirectRows"]) != bool(v["FirstDirect"]) or bool(v["ReducedRows"]) != bool(v["FirstReduced"]):
                    raise ValueError("Missing payload doorbell timestamp")
                if v["Lane0Rows"] + v["Lane1Rows"] != v["PayloadWqes"]:
                    raise ValueError("Chunk lane payload mismatch")
                for lane in range(2):
                    if bool(v[f"Lane{lane}Rows"]) != bool(v[f"FirstSend{lane}"]):
                        raise ValueError("Missing lane doorbell timestamp")
                if group + 1 < groups:
                    if not v["WorkEnd"] <= v["BarrierBegin"] <= v["BarrierEnd"] <= v["End"]:
                        raise ValueError("Invalid cross group barrier interval")
                elif v["BarrierBegin"] or v["BarrierEnd"]:
                    raise ValueError("Unexpected final group barrier")
                cross_seen.add((core, group, chunk))
            elif kind == 4:
                if compact and v["Flags"] != EP8_SINGLE_QP_FLAG:
                    raise ValueError("Invalid EP8 return marker")
                local_seen.add((core, chunk))
            peer = (
                ((core if core < rank else core + 1) if core < 7 else rank)
                if compact
                else core // 4
                if ep8_rank_pipeline
                else rank // 8 * 8 + core
            )
            if ep8_rank_pipeline and kind == 1:
                raise ValueError("EP8 return pipeline has no gather tasks")
            if kind in (1, 4) and (core >= local or v["Peer"] != peer):
                raise ValueError("Invalid local peer/core")
            begin, end = token_range(chunk, tokens, chunk_tokens)
            events.append(
                dict(
                    rank=rank,
                    core=core,
                    kind=KINDS[kind],
                    group=group,
                    chunk=chunk,
                    token_begin=begin,
                    token_end=end,
                    counters={
                        name: v[name]
                        for name in (
                            "Tokens",
                            "Rows",
                            "DirectRows",
                            "ReducedRows",
                            "PayloadWqes",
                            "Sos",
                            "Lane0Rows",
                            "Lane1Rows",
                        )
                    },
                    times_ticks={name: v[name] for name in TIME_FIELDS if v[name]},
                    peer=None if v["Peer"] == 2**32 - 1 else v["Peer"],
                    window_end=v["WindowEnd"],
                )
            )
    if final_seen != {(c, k) for k in range(chunks) for c in range(window_pool[k] if compact else 0, 64)}:
        raise ValueError("Missing final-reduce chunk records")
    if gateway_seen != {(c, g, k) for c in range(gateway_begin, 64) for g in range(groups) for k in range(chunks)}:
        raise ValueError("Missing gateway chunk records")
    if cross_seen != (
        {(c, g, k) for c in range(8, 24) for g in range(groups) for k in range(chunks)} if world > 8 else set()
    ):
        raise ValueError("Missing cross-send chunk records")
    if local_seen != {(c, k) for c in range(local) for k in range(chunks)}:
        raise ValueError("Missing local-return chunk records")
    return events


def perfetto_trace(trace, source_name):
    """Chrome JSON trace, with separate AIV/kind/group tracks and one rank clock."""
    if (
        trace["schema_version"] not in (1, VERSION)
        or (trace["ticks_per_us"], trace["precision_checked"]) != (TICKS_PER_US, True)
        or trace["kernel"]
        not in ("combine_profile_kernel", "combine_ep8_profile_kernel", "combine_ep8_mte_profile_kernel")
        or (trace["kernel"] != "combine_profile_kernel" and trace["world_size"] != 8)
    ):
        raise ValueError("Unsupported or unvalidated chunk format")
    events, rank = trace["events"], trace["rank"]
    if not events:
        raise ValueError("Empty chunk data")
    origin = min(t for e in events for t in e["times_ticks"].values())
    kinds = ("gather", "local_return", "cross_send", "gateway_reduce", "final_reduce")
    lanes = sorted({(e["core"], e["kind"], e["group"]) for e in events}, key=lambda x: (x[0], kinds.index(x[1]), x[2]))
    tids = {lane: i + 1 for i, lane in enumerate(lanes)}
    pid = rank + 1
    metadata = [
        dict(
            ph="M",
            name="process_name",
            pid=pid,
            tid=0,
            args=dict(name=f"Rank {rank:03d} | EP{trace['world_size']} T{trace['tokens']} | chunk diagnostics"),
        )
    ]
    for (core, kind, group), tid in tids.items():
        metadata.extend(
            [
                dict(
                    ph="M",
                    name="thread_name",
                    pid=pid,
                    tid=tid,
                    args=dict(name=f"AIV {core:02d} | {kind} | group {group}"),
                ),
                dict(ph="M", name="thread_sort_index", pid=pid, tid=tid, args=dict(sort_index=tid)),
            ]
        )
    semantics = {
        "gather": "WorkEnd is issue/publication, not an independent remote-arrival measurement.",
        "local_return": "WorkEnd is issue/publication, not an independent remote-arrival measurement.",
        "cross_send": "Send markers are payload SQ doorbell timestamps, not remote arrival or completion.",
        "gateway_reduce": "Nonempty WorkEnd follows output-store drains and reduce_done publication; Tokens=0 means no reduction.",
        "final_reduce": "WorkEnd includes diagnostic-only chunk-local drains; incoming readiness is observed per SO window.",
    }
    timed, seen, track_slices = [], set(), {tid: [] for tid in tids.values()}
    for e in events:
        core, kind, group, chunk = e["core"], e["kind"], e["group"], e["chunk"]
        identity = core, kind, group, chunk
        if identity in seen or e["rank"] != rank:
            raise ValueError(f"Duplicate/mismatched chunk identity: {identity}")
        seen.add(identity)
        t, counters = e["times_ticks"], e["counters"]
        if any(type(v) is not int or v <= 0 for v in t.values()):
            raise ValueError("Invalid chunk timestamp")
        has_work = "WorkBegin" in t
        if not has_work and (kind != "cross_send" or any(counters.values()) or any(f in t for f in SEND_FIELDS)):
            raise ValueError(f"Missing chunk work timestamp: {identity}")
        work_begin = t.get("WorkBegin", t["WorkEnd"])
        if not t["Begin"] <= work_begin <= t["WorkEnd"] <= t["End"]:
            raise ValueError(f"Invalid chunk work interval: {identity}")
        if any(v > t["End"] for v in t.values()):
            raise ValueError(f"Timestamp exceeds chunk end: {identity}")
        tid = tids[core, kind, group]
        base = {
            key: e[key]
            for key in ("rank", "core", "kind", "group", "chunk", "token_begin", "token_end", "peer", "window_end")
        }

        def emit(name, begin, end=None, extra=None):
            attrs = dict(base)
            if extra:
                attrs.update(extra)
            event = dict(
                name=name, cat="combine.chunk." + kind, pid=pid, tid=tid, ts=(begin - origin) / TICKS_PER_US, args=attrs
            )
            if end is None:
                event.update(ph="I", s="t")
                duration = 0
            else:
                if end < begin:
                    raise ValueError(f"Invalid chunk interval: {identity} {name}")
                duration = end - begin
                event.update(ph="X", dur=duration / TICKS_PER_US)
                track_slices[tid].append((begin, end, name))
            timed.append((begin, -duration, tid, len(timed), event))

        # SO-window waiting can precede the first final-reduce chunk's Begin.
        # Later chunks reuse Ready; do not invent another wait for those chunks.
        start = min(t["Begin"], t.get("WaitBegin", t["Begin"]))
        emit(
            f"chunk {chunk:02d} | tokens [{e['token_begin']},{e['token_end']})",
            start,
            t["End"],
            dict(
                counters=counters,
                source_times_ticks=t,
                times_us_from_rank_origin={name: (tick - origin) / TICKS_PER_US for name, tick in t.items()},
                semantics=semantics[kind],
                source_file=source_name,
                envelope_note="Display envelope includes this record's explicit wait, which can precede Begin.",
            ),
        )
        if "WaitBegin" in t:
            if not t["WaitBegin"] <= t["Ready"] <= work_begin:
                raise ValueError(f"Invalid chunk readiness interval: {identity}")
            wait_name = (
                f"SO_window_wait | chunks {chunk:02d}..{e['window_end']:02d}"
                if kind == "final_reduce"
                else f"wait | chunk {chunk:02d}"
            )
            emit(wait_name, t["WaitBegin"], t["Ready"], dict(start_field="WaitBegin", end_field="Ready"))
        reduction = kind in ("gateway_reduce", "final_reduce") and counters["Tokens"] > 0
        work_name = kind if not kind.endswith("reduce") or reduction else "no_reduce"
        if has_work:
            emit(
                f"{work_name} | chunk {chunk:02d}",
                t["WorkBegin"],
                t["WorkEnd"],
                dict(counters=counters, start_field="WorkBegin", end_field="WorkEnd", semantics=semantics[kind]),
            )
        if reduction:
            emit(
                f"{kind}_done | chunk {chunk:02d}",
                t["WorkEnd"],
                extra=dict(timestamp_field="WorkEnd", semantics=semantics[kind]),
            )
        if "BarrierBegin" in t or "BarrierEnd" in t:
            if (
                kind != "cross_send"
                or not t["WorkEnd"] <= t.get("BarrierBegin", 0) <= t.get("BarrierEnd", 0) <= t["End"]
            ):
                raise ValueError(f"Invalid group barrier: {identity}")
            emit(f"cross_group_barrier | chunk {chunk:02d}", t["BarrierBegin"], t["BarrierEnd"])
        for field in SEND_FIELDS:
            if field in t:
                if not t["WorkBegin"] <= t[field] <= t["WorkEnd"]:
                    raise ValueError(f"Send marker outside chunk work: {identity} {field}")
                emit(
                    f"{field} | chunk {chunk:02d}",
                    t[field],
                    extra=dict(timestamp_field=field, counters=counters, semantics=semantics["cross_send"]),
                )
    # Chrome synchronous slices must nest or be disjoint on each track.
    for tid, slices in track_slices.items():
        stack = []
        for begin, end, name in sorted(slices, key=lambda s: (s[0], -s[1])):
            while stack and begin >= stack[-1]:
                stack.pop()
            if stack and end > stack[-1]:
                raise ValueError(f"Partially overlapping chunk slices: rank={rank} tid={tid} {name}")
            if end > begin:
                stack.append(end)
    timed.sort(key=lambda row: row[:4])
    return dict(
        traceEvents=metadata + [row[-1] for row in timed],
        displayTimeUnit="ns",
        otherData=dict(
            source_file=source_name,
            kernel_source_sha256=trace["source_sha256"],
            rank=rank,
            time_origin_tick=origin,
            ticks_per_us=TICKS_PER_US,
            note=trace["note"],
            clock_scope="One rank only; files from different devices are not clock-aligned.",
            performance="Instrumented invocation. Use the separate clean combine profiling summary for performance.",
        ),
    )


def write_timeline(path, trace):
    """Same core/group/chunk timeline organization as the reference SVG tool."""
    path = Path(path)
    perfetto = path.with_suffix(".perfetto.json")
    temporary = perfetto.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(perfetto_trace(trace, path.name), separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(perfetto)
    events = trace["events"]
    origin = min(e["times_ticks"]["Begin"] for e in events)
    finish = max(e["times_ticks"]["End"] for e in events)
    span = max(1, finish - origin)
    lanes = sorted({(e["core"], e["kind"], e["group"]) for e in events})
    lane_y = {lane: 105 + i * 22 for i, lane in enumerate(lanes)}
    left, width = 245, 2400

    def x(tick):
        return left + (tick - origin) / span * width

    escape = html.escape
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="2700" height="{130 + len(lanes) * 22}">',
        "<style>text{font:12px monospace}rect:hover{stroke:#000;stroke-width:2}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="12" y="20">Combine rank {trace["rank"]} EP{trace["world_size"]} T{trace["tokens"]} chunk timeline</text>',
        '<text x="12" y="40">gray=wait; blue=issue; purple=SQ submit; orange=group barrier; green=reduce; pale=no reduce; hover for details</text>',
        '<text x="12" y="60">Diagnostic call only. Send is submission, not remote completion. Time origin is rank-local.</text>',
    ]
    for i in range(11):
        xx = left + width * i / 10
        parts.append(f'<text x="{xx:.2f}" y="84">{span * i / 10 / TICKS_PER_US:.1f} us</text>')
    for lane, y in lane_y.items():
        parts.extend(
            [
                f'<text x="8" y="{y + 12}">AIV {lane[0]:02d} {lane[1]} g{lane[2]}</text>',
                f'<path d="M{left} {y + 18} H{left + width}" stroke="#eee"/>',
            ]
        )
    for e in events:
        t = e["times_ticks"]
        y = lane_y[e["core"], e["kind"], e["group"]]
        details = {**e, "times_us_from_rank_origin": {k: (v - origin) / TICKS_PER_US for k, v in t.items()}}
        title = escape(json.dumps(details, ensure_ascii=False))

        def bar(begin, end, color, yy=y, height=15):
            if t.get(begin) and t.get(end):
                parts.append(
                    f'<rect x="{x(t[begin]):.3f}" y="{yy}" width="{max(0.5, x(t[end]) - x(t[begin])):.3f}" height="{height}" fill="{color}"><title>{title}</title></rect>'
                )

        bar("WaitBegin", "Ready", "#d3d8df")
        reduce = e["kind"] in ("gateway_reduce", "final_reduce")
        color = ("#21a675" if e["counters"]["Tokens"] else "#d5eee4") if reduce else "#5091d4"
        bar("WorkBegin", "WorkEnd", color)
        bar("BarrierBegin", "BarrierEnd", "#eb9634")
        for lane in range(2):
            bar(f"FirstSend{lane}", f"LastSend{lane}", "#8648bd", y + lane * 8, 6)
    parts.append("</svg>")
    path.with_suffix(".svg").write_text("\n".join(parts), encoding="utf-8")
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as out:
        columns = ["rank", "core", "kind", "group", "chunk", "token_begin", "token_end", "peer", "window_end"]
        counters = list(events[0]["counters"])
        writer = csv.DictWriter(out, fieldnames=columns + counters + [n + "_us" for n in TIME_FIELDS])
        writer.writeheader()
        for e in events:
            writer.writerow(
                {
                    **{k: e[k] for k in columns},
                    **e["counters"],
                    **{k + "_us": (v - origin) / TICKS_PER_US for k, v in e["times_ticks"].items()},
                }
            )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("json", type=Path)
    args = cli.parse_args()
    write_timeline(args.json, json.loads(args.json.read_text(encoding="utf-8")))
