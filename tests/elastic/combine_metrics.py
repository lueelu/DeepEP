"""Decode fixed-size Combine AIV diagnostics; all durations use one core's clock."""

import argparse
import json
from pathlib import Path

VERSION, WORDS = 4, 72
MAGIC = 0x434D4250
TICKS_PER_US = 1000  # Ascend 950 system counter, not the AIV clock frequency.
FIELDS = (
    "Magic Version Rank Core World Tokens Role WithWeights Sample "
    "Begin InitDone ProducerBegin ProducerEnd SourceBegin SourceEnd "
    "QuietBegin QuietEnd ReturnsBegin ReturnsEnd WeightWaitBegin WeightWaitEnd End "
    "WeightSendBegin WeightSendEnd GatherWaitTicks GatewayReduceTicks LocalReduceWaitTicks "
    "CrossReduceWaitTicks CrossSubmitTicks CrossBarrierTicks SourceWaitTicks "
    "GatewayTokens GatewayRows DirectRows ReducedRows PayloadWqes CompletionSos "
    "GatewayReadBytes GatewayWriteBytes FinalTokens FinalRows FinalReadBytes FinalWriteBytes "
    "ControlLoadTicks Groups Chunks ChunkTokens CrossCores Topk FirstDirectIssue FirstReducedIssue "
    "GatherEnd LocalReturnBegin LocalReturnEnd "
    "SourceFlagsWaitTicks SourceBarrierTicks PeerExportTicks SqPressureCount CreditSos CreditQuietCount CreditWaitTicks "
    "CrossPeerSetupTicks CrossSoReserveTicks "
    "Lane0PayloadRows Lane1PayloadRows Lane0CreditWaitTicks Lane1CreditWaitTicks "
    "Lane0QuietTicks Lane1QuietTicks Lane0SqPressureCount Lane1SqPressureCount"
).split()
ROLES = ("local", "cross", "gateway", "idle")
NOTE = (
    "Separate instrumented invocation of the same scheduler; use clean profiling for performance. "
    "Scalar phase boundaries do not add pipeline fences. Source includes existing waits/drains; "
    "gateway reduction includes metadata, memory work and readiness publication. "
    "Cross submit includes batch flush, SQ pressure handling and SO submission, not remote arrival. "
    "Payload WQEs exclude SO/credit WQEs; completion SOs exclude credit checkpoints. "
    "Source wait splits flag polling, existing SyncAll and diagnostic record export. "
    "SQ pressure counters describe software credit reclamation, not physical SQ occupancy. "
    "Lane payload bytes equal submitted BF16 rows times H*2, excluding SO and credit WQEs. "
    "Lane quiet durations are sequential local waits (lane 0 then lane 1), not comparable wire-service times. "
    "Phases overlap across cores; their maxima must not be added. "
    "Clocks are never aligned across devices. Kernel records exclude prepare, entry/exit barriers, "
    "input/output copies and the final profile-record store."
)


def decode_profile(
    records, *, rank, world, tokens, topk, weights, sample, chunk_tokens=256, hidden=7168, ep8_rank_pipeline=False
):
    if ep8_rank_pipeline and world != 8:
        raise ValueError("EP8 return pipeline diagnostics require EP8")
    if len(records) != 64:
        raise ValueError("Expected 64 Combine AIV records")
    cores = []
    version = records[0][1] if len(records[0]) > 1 else 0
    words_per_record = {3: 64, VERSION: WORDS}.get(version)
    cross = 16 if world > 8 else 0
    compact = ep8_rank_pipeline and len(records[11]) == WORDS and records[11][6] == 3
    local = (11 if compact else 32) if ep8_rank_pipeline else min(world, 8)
    for core, words in enumerate(records):
        if len(words) != words_per_record or any(type(v) is not int or not 0 <= v < 2**63 for v in words):
            raise ValueError("Invalid Combine summary record width/version")
        role = (
            0
            if core < local
            else 1
            if 8 <= core < 8 + cross
            else 2
            if core >= (32 if ep8_rank_pipeline else 8 + cross)
            else 3
        )
        if words[:9] != [MAGIC, version, rank, core, world, tokens, role, int(weights), sample]:
            raise ValueError(f"Missing/stale Combine profile rank={rank} core={core}")
        v = dict(zip(FIELDS if version == VERSION else FIELDS[:63], words))
        if (v["Groups"], v["Chunks"], v["ChunkTokens"], v["CrossCores"], v["Topk"]) != (
            world // min(world, 64),
            (tokens + chunk_tokens - 1) // chunk_tokens + min(4, tokens, chunk_tokens) - 1,
            chunk_tokens,
            cross,
            topk,
        ):
            raise ValueError("Combine profile configuration mismatch")
        ordered = [
            v[name]
            for name in (
                "Begin",
                "InitDone",
                "WeightSendBegin",
                "WeightSendEnd",
                "ProducerBegin",
                "ProducerEnd",
                "SourceBegin",
                "SourceEnd",
                "QuietBegin",
                "QuietEnd",
                "ReturnsBegin",
                "ReturnsEnd",
                "WeightWaitBegin",
                "WeightWaitEnd",
                "End",
            )
        ]
        if not ordered[0] or any(b < a for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Invalid Combine profile timestamp order")

        def interval(begin, end):
            if not v[begin] or not v[begin] <= v[end] <= v["End"]:
                raise ValueError(f"Invalid Combine interval {begin}/{end}")
            return (v[end] - v[begin]) / TICKS_PER_US

        def ticks(name, bound):
            value = v[name] / TICKS_PER_US
            if value > bound:
                raise ValueError(f"Combine accumulated interval exceeds its envelope: {name}")
            return value

        phases = {
            "setup": interval("Begin", "InitDone"),
            "producer": interval("ProducerBegin", "ProducerEnd"),
            "source_total": interval("SourceBegin", "SourceEnd"),
            "outgoing_quiet": interval("QuietBegin", "QuietEnd"),
            "incoming_drain": interval("ReturnsBegin", "ReturnsEnd"),
            "kernel_envelope": interval("Begin", "End"),
        }
        phases["source_wait"] = ticks("SourceWaitTicks", phases["source_total"])
        for name, field in (
            ("source_flags_wait", "SourceFlagsWaitTicks"),
            ("source_barrier", "SourceBarrierTicks"),
            ("peer_export", "PeerExportTicks"),
        ):
            phases[name] = ticks(field, phases["source_wait"])
        if (
            sum(v[name] for name in ("SourceFlagsWaitTicks", "SourceBarrierTicks", "PeerExportTicks"))
            > v["SourceWaitTicks"]
        ):
            raise ValueError("Source wait parts exceed the measured envelope")
        phases["source_after_wait"] = phases["source_total"] - phases["source_wait"]
        if weights:
            phases["weight_wait"] = interval("WeightWaitBegin", "WeightWaitEnd")
            if role == 2:
                phases["weight_send"] = interval("WeightSendBegin", "WeightSendEnd")
        if role == 0:
            if (
                not v["ProducerBegin"]
                <= v["GatherEnd"]
                <= v["LocalReturnBegin"]
                <= v["LocalReturnEnd"]
                <= v["ProducerEnd"]
            ):
                raise ValueError("Invalid local Combine phase order")
            phases.update(
                gather=interval("ProducerBegin", "GatherEnd"),
                local_return=interval("LocalReturnBegin", "LocalReturnEnd"),
            )
            phases["local_reduce_wait"] = ticks("LocalReduceWaitTicks", phases["local_return"])
        elif role == 1:
            for name, field in (
                ("cross_peer_setup", "CrossPeerSetupTicks"),
                ("cross_so_reserve", "CrossSoReserveTicks"),
                ("cross_reduce_wait", "CrossReduceWaitTicks"),
                ("cross_submit", "CrossSubmitTicks"),
                ("cross_group_barrier", "CrossBarrierTicks"),
            ):
                phases[name] = ticks(field, phases["producer"])
            phases["credit_wait"] = ticks("CreditWaitTicks", phases["cross_submit"])
            if v["PayloadWqes"] != v["DirectRows"] + v["ReducedRows"]:
                raise ValueError("Inconsistent Combine payload WQE counters")
            for field, count in (("FirstDirectIssue", "DirectRows"), ("FirstReducedIssue", "ReducedRows")):
                if (not v[field]) != (not v[count]) or (
                    v[field] and not v["ProducerBegin"] <= v[field] <= v["ProducerEnd"]
                ):
                    raise ValueError("Invalid Combine first issue timestamp")
            first_issue = min((v[f] for f in ("FirstDirectIssue", "FirstReducedIssue") if v[f]), default=0)
            if first_issue:
                phases["cross_first_payload"] = (first_issue - v["ProducerBegin"]) / TICKS_PER_US
        elif role == 2:
            phases["gather_wait"] = ticks("GatherWaitTicks", phases["producer"])
            phases["gateway_reduce"] = ticks("GatewayReduceTicks", phases["producer"])
            phases["control_load"] = ticks("ControlLoadTicks", phases["kernel_envelope"])
        counters = {name: v[name] for name in FIELDS[31:43]}
        counters.update({name: v[name] for name in ("SqPressureCount", "CreditSos", "CreditQuietCount")})
        if version == VERSION:
            for lane in range(2):
                prefix = f"Lane{lane}"
                rows = v[prefix + "PayloadRows"]
                counters[prefix + "PayloadRows"] = rows
                counters[prefix + "PayloadBytes"] = rows * hidden * 2
                counters[prefix + "SqPressureCount"] = v[prefix + "SqPressureCount"]
                phases[f"lane{lane}_credit_wait"] = ticks(prefix + "CreditWaitTicks", phases["producer"])
                phases[f"lane{lane}_quiet"] = ticks(prefix + "QuietTicks", phases["outgoing_quiet"])
            if (
                v["Lane0PayloadRows"] + v["Lane1PayloadRows"] != v["PayloadWqes"]
                or v["Lane0CreditWaitTicks"] + v["Lane1CreditWaitTicks"] != v["CreditWaitTicks"]
                or v["Lane0SqPressureCount"] + v["Lane1SqPressureCount"] != v["SqPressureCount"]
                or v["Lane0QuietTicks"] + v["Lane1QuietTicks"] > v["QuietEnd"] - v["QuietBegin"]
            ):
                raise ValueError("Inconsistent Combine lane counters/times")
        cores.append(dict(rank=rank, core=core, role=ROLES[role], phases_us=phases, counters=counters, raw_words=words))
    if sum(c["counters"]["FinalTokens"] for c in cores) != tokens:
        raise ValueError("Incomplete Combine final token coverage")
    return cores


def summarize(reports):
    if not reports:
        raise ValueError("No Combine rank reports")
    world = reports[0]["world_size"]
    if len(reports) != world or {r["rank"] for r in reports} != set(range(world)):
        raise ValueError("Missing/duplicate Combine rank reports")
    expected = None
    maxima, totals = {}, {}
    for report in reports:
        diagnostics = report.get("combine_diagnostics", [])
        if not report.get("passed") or len(diagnostics) != 1:
            raise ValueError("Expected one validated Combine diagnostic per rank")
        d = diagnostics[0]
        if any(
            d.get(k) != report.get(k)
            for k in ("world_size", "tokens", "topk", "experts", "seed", "with_weights", "source_sha256")
        ):
            raise ValueError("Combine diagnostics do not match the precision report")
        identity = tuple(
            d.get(k)
            for k in (
                "schema_version",
                "world_size",
                "tokens",
                "topk",
                "hidden",
                "experts",
                "scenario",
                "chunk_tokens",
                "seed",
                "sample",
                "with_weights",
                "source_sha256",
                "ticks_per_us",
                "precision_checked",
                "kernel",
            )
        )
        expected = identity if expected is None else expected
        if identity != expected or d["world_size"] != world or d["rank"] != report["rank"]:
            raise ValueError("Mismatched Combine diagnostic configuration")
        if (
            d["schema_version"] not in (3, VERSION)
            or (d["ticks_per_us"], d["precision_checked"]) != (TICKS_PER_US, True)
            or d["kernel"]
            not in ("combine_profile_kernel", "combine_ep8_profile_kernel", "combine_ep8_mte_profile_kernel")
        ):
            raise ValueError("Invalid Combine diagnostic schema/precision/kernel")
        cores = decode_profile(
            [c["raw_words"] for c in d["cores"]],
            rank=d["rank"],
            world=world,
            tokens=d["tokens"],
            topk=d["topk"],
            weights=d["with_weights"],
            sample=d["sample"],
            chunk_tokens=d.get("chunk_tokens", 256),
            hidden=d["hidden"],
            ep8_rank_pipeline=d["kernel"] != "combine_profile_kernel",
        )
        for c in cores:
            for name, value in c["phases_us"].items():
                key = c["role"] + "/" + name
                if key not in maxima or value > maxima[key]["us"]:
                    maxima[key] = dict(us=value, rank=c["rank"], core=c["core"])
            for name, value in c["counters"].items():
                totals[name] = totals.get(name, 0) + value
    return dict(schema_version=VERSION, world_size=world, phase_maxima_us=maxima, counters_sum=totals, note=NOTE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    result = summarize([json.loads(p.read_text()) for p in args.results.glob("rank-*.json")])
    output = args.results / "combine-diagnostics.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
