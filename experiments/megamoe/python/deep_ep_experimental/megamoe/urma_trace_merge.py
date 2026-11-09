"""Merge rank-local URMA Chrome traces without pretending their clocks are aligned."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from .urma_trace import ENGINE_COUNT, TRACE_SCHEMA, coalesce_gmm_tiles, filter_trace_waits


def _validate_trace(trace, rank, expected_metadata=None):
    metadata = trace["metadata"]
    if (
        metadata.get("schema") != TRACE_SCHEMA
        or type(metadata.get("rank")) is not int
        or metadata.get("rank") != rank
        or metadata.get("engine_count") != ENGINE_COUNT
        or metadata.get("dropped") != 0
    ):
        raise ValueError(f"rank={rank}: incompatible, incomplete, or wrong-rank URMA trace")
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError(f"rank={rank}: trace metadata differs from this run's export")
    clock = metadata.get("clock_mhz")
    if isinstance(clock, bool) or not isinstance(clock, (int, float)) or not math.isfinite(clock) or clock <= 0:
        raise ValueError(f"rank={rank}: invalid clock_mhz")
    count = 0
    for event in trace["traceEvents"]:
        if (
            type(event.get("pid")) is not int
            or event.get("pid") != rank
            or type(event.get("tid")) is not int
            or not 0 <= event["tid"] < ENGINE_COUNT
        ):
            raise ValueError(f"rank={rank}: invalid event rank/engine identity")
        if event.get("ph") == "X":
            count += 1
            for field in ("ts", "dur"):
                value = event.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"rank={rank}: invalid event {field}")
        elif event.get("ph") != "M":
            raise ValueError(f"rank={rank}: unexpected event type")
    visible = metadata.get("visible_record_count", metadata.get("record_count"))
    hidden = metadata.get("hidden_wait_count", 0)
    coalesced = metadata.get("coalesced_record_count", 0)
    if (
        type(visible) is not int
        or type(hidden) is not int
        or hidden < 0
        or type(coalesced) is not int
        or coalesced < 0
        or count != visible
        or count + hidden + coalesced != metadata.get("record_count")
    ):
        raise ValueError(f"rank={rank}: record count mismatch; refusing a partial trace")
    return metadata


def merge_trace_files(paths, output, *, world=16, include_waits=False, expected_metadata=None, combine_gmm=True):
    """Read at most one rank's JSON at a time; atomically publish a complete merge."""
    if type(world) is not int or not 1 <= world <= 16 or len(paths) != world:
        raise ValueError("one trace per rank is required, with world in [1,16]")
    output = Path(output)
    paths = [Path(path) for path in paths]
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("merged output must not overwrite a rank's input")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".allrank-", suffix=".json.tmp", dir=output.parent)
    try:
        per_rank, first, clock = [], True, None
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write('{"traceEvents":[')
            for rank, path in enumerate(paths):
                trace = json.loads(path.read_text(encoding="utf-8"))
                original = _validate_trace(trace, rank, None if expected_metadata is None else expected_metadata[rank])
                if clock is not None and clock != original["clock_mhz"]:
                    raise ValueError("rank traces use different clock_mhz settings")
                clock = original["clock_mhz"]
                if combine_gmm:
                    trace = coalesce_gmm_tiles(trace)
                if not include_waits:
                    trace = filter_trace_waits(trace)
                for event in trace["traceEvents"]:
                    if not first:
                        stream.write(",")
                    first = False
                    json.dump(event, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                # Explicit process ordering, with the original pid/tid retained.
                if not first:
                    stream.write(",")
                first = False
                json.dump(dict(ph="M", name="process_sort_index", pid=rank, tid=0, args=dict(sort_index=rank)), stream)
                per_rank.append(trace["metadata"])
                del trace
            metadata = dict(
                schema=TRACE_SCHEMA,
                kind="urma_allrank",
                ranks=list(range(world)),
                rank_count=world,
                engine_count_per_rank=ENGINE_COUNT,
                record_count=sum(item["record_count"] for item in per_rank),
                visible_record_count=sum(item.get("visible_record_count", item["record_count"]) for item in per_rank),
                hidden_wait_count=sum(item.get("hidden_wait_count", 0) for item in per_rank),
                coalesced_record_count=sum(item.get("coalesced_record_count", 0) for item in per_rank),
                clock_mhz=clock,
                clock_alignment="each rank starts at its own earliest event; cross-rank clocks are not synchronized",
                scope="one BS diagnostic invocation; no timestamp shifts or gap compression",
                per_rank=per_rank,
            )
            stream.write('],"displayTimeUnit":"us","metadata":')
            json.dump(metadata, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("}")
        os.replace(temporary, output)
        return metadata
    finally:
        Path(temporary).unlink(missing_ok=True)


def merge_trace_collective(group, records, output, *, bs, include_waits=False):
    """Use shared files when available, otherwise collect through bounded Store chunks."""
    rank, world = group.rank(), group.size()
    output = Path(output)
    staging, paths, setup_error = None, None, None
    shared = False
    if rank == 0:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            paths = [Path(record["chrome_path"]) for record in records]
            shared = all(path.is_file() for path in paths)
            if not shared:
                staging = tempfile.TemporaryDirectory(prefix=".rank-traces-", dir=output.parent)
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
    try:
        setup = group.all_gather(dict(shared=shared, error=setup_error), f"URMA trace merge setup BS={bs}")[0]
        if setup["error"]:
            raise RuntimeError(f"URMA allrank setup failed: {setup['error']}")
        if not setup["shared"]:
            paths = group.gather_file(
                records[rank]["chrome_path"], staging.name if staging else None, f"URMA trace files BS={bs}"
            )
        report = dict(error=None, path=str(output), metadata=None)
        if rank == 0:
            try:
                report["metadata"] = merge_trace_files(
                    paths,
                    output,
                    world=world,
                    include_waits=include_waits,
                    expected_metadata=[record["metadata"] for record in records],
                )
                # Keep the collective status small; the JSON contains full per-rank metadata.
                report["metadata"] = {key: value for key, value in report["metadata"].items() if key != "per_rank"}
            except Exception as error:
                report["error"] = f"{type(error).__name__}: {error}"
        merged = group.all_gather(report, f"URMA trace merge result BS={bs}")[0]
        if merged["error"]:
            raise RuntimeError(f"URMA allrank export failed: {merged['error']}")
        return merged
    finally:
        if staging is not None:
            staging.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Run directory containing rank_0 ... rank_15")
    parser.add_argument("--bs-values", help="Comma-separated BS values; default discovers rank_0/bs_*")
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--include-waits", action="store_true", help="Keep waits that are present in the input JSON")
    parser.add_argument("--split-gmm-events", action="store_true", help="Do not join separate GMM bars in input JSON")
    args = parser.parse_args(argv)
    try:
        batches = (
            [int(value) for value in args.bs_values.split(",")]
            if args.bs_values
            else sorted(int(path.name[3:]) for path in (args.directory / "rank_0").glob("bs_*") if path.is_dir())
        )
        if not batches or any(bs <= 0 for bs in batches):
            raise ValueError("no valid BS cases found")
        for bs in batches:
            paths = [
                args.directory / f"rank_{rank}" / f"bs_{bs}" / "urma_trace.json" for rank in range(args.world_size)
            ]
            output = args.directory / f"bs_{bs}" / "allrank.json"
            metadata = merge_trace_files(
                paths,
                output,
                world=args.world_size,
                include_waits=args.include_waits,
                combine_gmm=not args.split_gmm_events,
            )
            print(
                f"[URMA ALLRANK] BS={bs} ranks={metadata['rank_count']} events={metadata['visible_record_count']}; {output}"
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"URMA allrank export failed: {error}\n")


if __name__ == "__main__":
    main()
