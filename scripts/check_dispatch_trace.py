"""Verify a completed single-node dispatch trace run, including every expected rank."""

import argparse
import json
from pathlib import Path


def check(directory, ranks):
    observed = set()
    completed = 0
    paths = sorted(directory.glob("dispatch-*.jsonl"))
    if not paths:
        raise ValueError(f"no trace files in {directory}")
    for path in paths:
        offset = None
        launches = {}
        ended = set()
        finished = False
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                kind = row["event"]
                where = f"{path.name} line {number}"
                if finished or (offset is None and kind != "calibration"):
                    raise ValueError(f"{where}: unexpected event order")
                if kind == "calibration":
                    if offset is not None or row["tick_hz"] != 1_000_000_000 or row["tick"] <= 0:
                        raise ValueError(f"{where}: invalid calibration")
                    offset = row["offset_ns"]
                elif kind == "launch":
                    gid = row["gid"]
                    if gid in launches or row["status"] or row["kernel"] != "v1_dispatch_udma_kernel":
                        raise ValueError(f"{where}: invalid launch")
                    launches[gid] = (row["rank"], row["local_instance"], row["launch_ns"])
                elif kind == "end":
                    gid = row["gid"]
                    identity = (row["rank"], row["local_instance"], row["launch_ns"])
                    if launches.get(gid) != identity or gid in ended or row["kernel"] != "v1_dispatch_udma_kernel":
                        raise ValueError(f"{where}: unmatched end")
                    if not (
                        0 < row["start_tick"] <= row["end_tick"]
                        and row["start_ns"] == row["start_tick"] + offset
                        and row["end_ns"] == row["end_tick"] + offset
                        and row["elapsed_ns"] == row["end_tick"] - row["start_tick"]
                    ):
                        raise ValueError(f"{where}: inconsistent device timing")
                    observed.add(row["rank"])
                    ended.add(gid)
                elif kind == "summary":
                    if row["dropped"] or row["incomplete"] or launches.keys() != ended or row["completed"] != len(ended):
                        raise ValueError(f"{where}: incomplete trace: {row}")
                    finished = True
                else:
                    raise ValueError(f"{where}: {kind}")
        if not finished:
            raise ValueError(f"{path}: missing final summary")
        completed += len(ended)
    if observed != set(range(ranks)):
        raise ValueError(f"expected ranks 0..{ranks - 1}, got {sorted(observed)}")
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    args = parser.parse_args()
    print(f"verified {check(args.directory, args.ranks)} completed dispatch records")
