"""Receive flag observations and sender SO submissions, with rank-local clocks."""

import json

VERSION, MAGIC, WORDS = 1, 0x43505250, 16
FIELDS = (
    "Magic Version Sample Kind Rank Core Peer Chunk Lane Qp Begin Observed FlagsReady SyncEnd Polls Flags"
).split()
NOTE = (
    "A separate diagnostic invocation polls outstanding flags round-robin. Observed is an upper bound on "
    "arrival, not a NIC timestamp. Ready on first poll is left-censored: arrival may have happened earlier. "
    "SO submission is not remote completion. Rank clocks are not aligned; never subtract clocks across ranks. "
    "Use clean profiling for performance."
)


def shape(world, tokens, chunk_tokens=256):
    if chunk_tokens not in (256, 512, 1024):
        raise ValueError("chunk_tokens must be 256, 512 or 1024")
    if world not in (2, 4, 8, 16, 32, 64, 128, 256, 512) or not 1 <= tokens <= 16384:
        raise ValueError("Invalid Combine peer shape")
    return 2, world, (tokens + chunk_tokens - 1) // chunk_tokens + min(4, tokens) - 1, 2, WORDS


def owner(sender, peer, world, chunk, tokens):
    """Host interpretation of common.hpp's CombineOwner, validated by CPU tests."""
    width = min(world, 16)

    def order(p):
        if world == 128:
            side, peer_side = sender % 8 // 4, p % 8 // 4
            group = ((sender // 64) ^ (p // 64) ^ side) * 4
            group += ((sender % 64 // 32) ^ (p % 64 // 32)) * 2 + (side ^ peer_side)
            index = (p // 8 % 4) * 4 + p % 4
        else:
            half, half_width = world // 2, width // 2
            group = (p % half // half_width + world // width - sender % half // half_width) % (world // width)
            index = p // half * half_width + p % half_width
        return group * width + index

    ordinal = order(peer)
    ordinal -= sum(order(p) < order(peer) for p in range(sender // 8 * 8, sender // 8 * 8 + 8))
    ordinal = ordinal // 8 * 8 + (ordinal % 8 + 8 - sender % 8) % 8
    logical = max(0, chunk - min(4, tokens) + 1)
    if world == 64 and ordinal >= 48:
        return ordinal % 8 + logical % 2 * 8
    return (ordinal + 8) % 16


def window_ends(tokens, chunk_tokens=256):
    chunks, split = (tokens + chunk_tokens - 1) // chunk_tokens, min(4, tokens) - 1
    count = min(3, chunks)
    if count == 3 and chunks >= 4:
        first = (chunks + 1) // 2
        return [first - 1 + split, first + (chunks - first + 1) // 2 - 1 + split, chunks - 1 + split]
    return [((w + 1) * chunks + count - 1) // count - 1 + split for w in range(count)]


def expected_records(rank, world, tokens, chunk_tokens=256):
    """Map (direction, peer, actual SO chunk, lane) to its unique writer and QP."""
    expected = {}
    for direction in range(2):
        for peer in range(world):
            remote = peer // 8 != rank // 8
            if direction and not remote:
                continue
            sender, receiver = (rank, peer) if direction else (peer, rank)
            previous_end = -1
            for end in window_ends(tokens, chunk_tokens):
                actual = [end]
                if (
                    remote
                    and end - 1 > previous_end
                    and owner(sender, receiver, world, end - 1, tokens) != owner(sender, receiver, world, end, tokens)
                ):
                    actual.insert(0, end - 1)
                previous_end = end
                for chunk in actual:
                    for lane in range(2 if remote else 1):
                        qp = 2 * owner(sender, receiver, world, chunk, tokens) + lane if remote else 2**32 - 1
                        core = 8 + qp // 2 if direction else peer % 64
                        expected[direction, peer, chunk, lane] = core, qp, end
    return expected


def decode(records, *, rank, world, tokens, sample, chunk_tokens=256):
    dimensions = shape(world, tokens, chunk_tokens)
    expected = expected_records(rank, world, tokens, chunk_tokens)
    compact = (
        world == 8
        and len(records) == 2
        and any(
            len(words) == WORDS and any(words) and words[5] == 11 + peer
            for peer, rows in enumerate(records[0])
            for lanes in rows
            for words in lanes
        )
    )
    if compact:
        ends = window_ends(tokens, chunk_tokens)
        expected = {key: ((11 if end == ends[-1] else 32) + key[1], qp, end) for key, (_, qp, end) in expected.items()}
    events, seen = [], set()
    if len(records) != 2 or not 0 <= rank < world or sample < 1:
        raise ValueError("Invalid peer tensor identity")
    for direction, peer_rows in enumerate(records):
        if len(peer_rows) != world:
            raise ValueError("Invalid peer tensor world")
        for peer, rows in enumerate(peer_rows):
            if len(rows) != dimensions[2]:
                raise ValueError("Invalid peer tensor chunks")
            for chunk, lanes in enumerate(rows):
                if len(lanes) != 2:
                    raise ValueError("Invalid peer tensor lanes")
                for lane, words in enumerate(lanes):
                    if len(words) != WORDS or any(type(v) is not int or not 0 <= v < 2**63 for v in words):
                        raise ValueError("Invalid peer record words")
                    if not any(words):
                        continue
                    key = direction, peer, chunk, lane
                    if key not in expected:
                        raise ValueError("Unexpected peer completion slot")
                    core, qp, end = expected[key]
                    if words[:10] != [MAGIC, VERSION, sample, direction, rank, core, peer, chunk, lane, qp]:
                        raise ValueError("Stale peer record or wrong owner/QP")
                    v = dict(zip(FIELDS, words))
                    if not 0 < v["Begin"] <= v["Observed"]:
                        raise ValueError("Invalid peer timestamp order")
                    if direction == 0:
                        if not v["Observed"] <= v["FlagsReady"] <= v["SyncEnd"] or v["Polls"] < 1:
                            raise ValueError("Invalid receive observation")
                        if v["Flags"] != int(v["Polls"] == 1):
                            raise ValueError("Invalid already-ready marker")
                    elif any(v[name] for name in ("FlagsReady", "SyncEnd", "Polls", "Flags")):
                        raise ValueError("Send record cannot describe receiver completion")
                    events.append(
                        dict(
                            rank=rank,
                            core=core,
                            peer=peer,
                            chunk=chunk,
                            lane=lane,
                            qp=qp if remote_peer(rank, peer) else None,
                            kind="so_submit" if direction else "receive_observation",
                            window_end=end,
                            times_ticks={name: v[name] for name in FIELDS[10:14] if v[name]},
                            polls=v["Polls"],
                            ready_on_first_poll=bool(v["Flags"] & 1),
                        )
                    )
                    seen.add(key)
    if seen != set(expected):
        raise ValueError(f"Missing peer records: {len(expected) - len(seen)}")
    return events


def remote_peer(rank, peer):
    return rank // 8 != peer // 8


def summarize(events, origin):
    pending = [e for e in events if e["kind"] == "receive_observation" and not e["ready_on_first_poll"]]
    latest = sorted(pending, key=lambda e: e["times_ticks"]["Observed"], reverse=True)[:24]
    return dict(
        pending_observations=len(pending),
        ready_on_first_poll=sum(e["ready_on_first_poll"] for e in events),
        latest_pending=[
            dict(
                peer=e["peer"],
                qp=e["qp"],
                lane=e["lane"],
                core=e["core"],
                chunk=e["chunk"],
                observed_us=(e["times_ticks"]["Observed"] - origin) / 1000,
                poll_interval_us=(e["times_ticks"]["Observed"] - e["times_ticks"]["Begin"]) / 1000,
                polls=e["polls"],
            )
            for e in latest
        ],
    )


def write_trace(
    path, *, events, rank, world, tokens, origin, sample, chunk_tokens=256, kernel="combine_profile_kernel"
):
    """Write source data and a ready-to-open Perfetto trace; no clock alignment."""
    if not events:
        raise ValueError("Empty peer trace")
    result = dict(
        schema_version=VERSION,
        rank=rank,
        world_size=world,
        tokens=tokens,
        chunk_tokens=chunk_tokens,
        sample=sample,
        kernel=kernel,
        precision_checked=True,
        ticks_per_us=1000,
        time_origin_tick=origin,
        note=NOTE,
        events=events,
        summary=summarize(events, origin),
    )
    path.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    tracks = sorted({(e["kind"], e["peer"], e["lane"]) for e in events})
    tids = {key: index + 1 for index, key in enumerate(tracks)}
    trace = [
        dict(
            ph="M",
            name="process_name",
            pid=rank + 1,
            tid=0,
            args=dict(name=f"Rank {rank} | peer observations (rank-local clock)"),
        )
    ]
    for (kind, peer, lane), tid in tids.items():
        trace.append(
            dict(
                ph="M",
                name="thread_name",
                pid=rank + 1,
                tid=tid,
                args=dict(name=f"{kind} | peer {peer:03d} | lane {lane}"),
            )
        )
    for e in events:
        t = e["times_ticks"]
        name = (
            "SO submit"
            if e["kind"] == "so_submit"
            else (
                "ready on first poll (arrival earlier or equal)"
                if e["ready_on_first_poll"]
                else "pending until observed"
            )
        )
        attrs = {k: v for k, v in e.items() if k != "times_ticks"}
        attrs["source_times_ticks"] = t
        tid = tids[e["kind"], e["peer"], e["lane"]]
        trace.append(
            dict(
                ph="X",
                cat="combine.peer",
                name=f"{name} | chunk {e['chunk']}",
                pid=rank + 1,
                tid=tid,
                ts=(t["Begin"] - origin) / 1000,
                dur=(t["Observed"] - t["Begin"]) / 1000,
                args=attrs,
            )
        )
    # The same core/window barrier is carried by multiple peer records. Export it once.
    barriers = {}
    for e in events:
        if e["kind"] == "receive_observation":
            key = e["core"], e["window_end"]
            value = e["times_ticks"]["FlagsReady"], e["times_ticks"]["SyncEnd"]
            if key in barriers and barriers[key] != value:
                raise ValueError("Inconsistent core/window barrier records")
            barriers[key] = value
    for core in sorted({k[0] for k in barriers}):
        trace.append(
            dict(
                ph="M",
                name="thread_name",
                pid=rank + 1,
                tid=len(tracks) + core + 1,
                args=dict(name=f"AIV {core:02d} | existing SyncAll"),
            )
        )
    for (core, window), (begin, end) in barriers.items():
        trace.append(
            dict(
                ph="X",
                cat="combine.barrier",
                name=f"SyncAll | window {window}",
                pid=rank + 1,
                tid=len(tracks) + core + 1,
                ts=(begin - origin) / 1000,
                dur=(end - begin) / 1000,
            )
        )
    output = path.with_suffix(".perfetto.json")
    output.write_text(
        json.dumps(
            dict(traceEvents=trace, displayTimeUnit="us", otherData=dict(note=NOTE, time_origin_tick=origin)),
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return dict(file=str(path), perfetto=str(output), events=len(events), summary=result["summary"])
