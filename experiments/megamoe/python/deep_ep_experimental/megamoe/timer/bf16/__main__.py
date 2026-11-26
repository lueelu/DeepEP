# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""Offline export of selected training ranks; raw files are never modified."""

import argparse
from pathlib import Path

from .. import parse_rank_list
from .trace_parser import export_rank, write_trace


def parse_ranks(value):
    return parse_rank_list(value, 1024)


def export(directory, ranks, *, include_wait=False, merge=True, output_dir=None):
    directory = Path(directory)
    output = Path(output_dir) if output_dir is not None else directory
    combined = []
    for rank in ranks:
        combined.extend(export_rank(directory, rank, include_wait=include_wait, output_dir=output))
    if merge:
        for phase in ("forward", "backward", None):
            selected = combined if phase is None else [e for e in combined if e["args"]["phase"] == phase]
            suffix = "" if phase is None else f"_{phase}"
            write_trace(output / f"all_rank{suffix}_trace.json", selected)
    return len(combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output-dir", type=Path, help="write traces here (default: alongside the raw buffers)")
    parser.add_argument("--ranks", type=parse_ranks, default=parse_ranks("0-7"))
    parser.add_argument("--include-wait", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    count = export(
        args.directory, args.ranks, include_wait=args.include_wait, merge=args.merge, output_dir=args.output_dir
    )
    print(f"[TIMER] exported {count} events; rank-local clocks, common forward/backward origin per rank")


if __name__ == "__main__":
    main()
