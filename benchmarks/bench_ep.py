# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""python -m benchmarks.bench_ep --dry-run (no framework or device required)."""

import argparse
import json
from pathlib import Path
import sys

from ._config import Config, load_config
from ._utils import measure, measured_result, plan, write_result

PRESETS = {
    "smoke": "smoke.json",
    "upstream-bf16": "upstream_bf16.json",
    "upstream-fp8": "upstream_fp8.json",
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, help="Strict JSON workload; CLI fields override it")
    result.add_argument("--preset", choices=PRESETS)
    result.add_argument("--list-scenarios", action="store_true")
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; never emit measured latency/bandwidth",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; existing output is never overwritten",
    )
    result.add_argument("--revision", help="Full source Git SHA; required for measurement")
    for name in ("api", "operation", "dtype", "layout", "dispatch-mode", "route"):
        result.add_argument("--" + name, default=argparse.SUPPRESS)
    for name in (
        "world-size",
        "hidden",
        "topk",
        "experts",
        "alignment",
        "seed",
        "warmup",
        "iterations",
    ):
        result.add_argument("--" + name, type=int, default=argparse.SUPPRESS)
    result.add_argument(
        "--tokens-per-rank",
        default=argparse.SUPPRESS,
        help="Comma-separated actual counts, one per rank",
    )
    result.add_argument(
        "--with-weights",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    return result


def main(argv=None):
    args = vars(parser().parse_args(argv))
    if args.pop("list_scenarios"):
        print(
            json.dumps(
                {name: load_config(Path(__file__).parent / "configs" / file) for name, file in PRESETS.items()},
                indent=2,
            )
        )
        return 0
    config_file, preset = args.pop("config"), args.pop("preset")
    dry_run, output, revision = (
        args.pop("dry_run"),
        args.pop("output_dir"),
        args.pop("revision"),
    )
    try:
        if config_file and preset:
            raise ValueError("choose either --config or --preset")
        data = (
            load_config(config_file)
            if config_file
            else (load_config(Path(__file__).parent / "configs" / PRESETS[preset]) if preset else {})
        )
        if not isinstance(data, dict):
            raise ValueError("configuration must be a JSON object")
        if "tokens_per_rank" in args:
            args["tokens_per_rank"] = tuple(int(value) for value in args["tokens_per_rank"].split(","))
        config = Config.from_dict(data | args)
        if dry_run:
            report = plan(config)
            if output:
                write_result(output, report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        # The capability check is before rank/device setup; API declarations
        # cannot manufacture measurements even with valid torchrun variables.
        from ._backend import resolve_backend

        factory = resolve_backend(config)
        import re

        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("measurement requires --revision with a full lowercase Git SHA")
        if output is None or output.exists():
            raise ValueError("measurement requires --output-dir pointing to a new directory")
        # Native adapter is a context manager; partial construction and error
        # teardown must be bounded and must not touch a caller-owned group.
        with factory(config.to_dict()) as runtime:
            measurements = measure(config, runtime)
            report = measured_result(config, measurements, revision=revision, metadata=runtime.metadata)
            is_writer = runtime.rank == 0
        # Failed teardown must not leave a successful performance artifact.
        if is_writer:
            write_result(output, report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        # Do not embed exception text (possibly internal paths) in result files.
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "error_type": type(error).__name__,
                }
            )
        )
        print(f"benchmark: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
