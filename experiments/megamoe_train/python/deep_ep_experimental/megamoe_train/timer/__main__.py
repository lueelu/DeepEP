# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""Offline export of selected training ranks; raw files are never modified."""

import argparse
import json
from pathlib import Path

from .trace_parser import export_rank


def parse_ranks(value):
    ranks = []
    for part in value.split(","):
        bounds = part.split("-")
        if len(bounds) == 1:
            ranks.append(int(part))
        elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
            ranks.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            raise ValueError("Use ranks such as 0-7 or 0,3,7")
    if not ranks or min(ranks) < 0 or max(ranks) >= 1024 or len(set(ranks)) != len(ranks):
        raise ValueError("Timer ranks must be unique and within [0,1023]")
    return ranks


def export(directory, ranks, *, include_wait=False, merge=True):
    directory = Path(directory)
    combined = []
    for rank in ranks:
        combined.extend(export_rank(directory, rank, include_wait=include_wait))
    if merge:
        for phase in ("forward", "backward", None):
            selected = combined if phase is None else [e for e in combined if e["args"]["phase"] == phase]
            suffix = "" if phase is None else f"_{phase}"
            (directory / f"all_rank{suffix}_trace.json").write_text(
                json.dumps(selected, ensure_ascii=False), encoding="utf-8"
            )
    return len(combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=parse_ranks, default=parse_ranks("0-7"))
    parser.add_argument("--include-wait", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    count = export(args.directory, args.ranks, include_wait=args.include_wait, merge=args.merge)
    print(f"[TIMER] exported {count} events; rank-local clocks, common forward/backward origin per rank")


if __name__ == "__main__":
    main()
