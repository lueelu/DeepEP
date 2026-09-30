# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Run the six functional smoke scenarios; this does not qualify a device."""

import argparse
import json
from time import perf_counter

from .functional_stub import create_backend
from .workflow import next_iteration, run_roundtrip, smoke_cases


def run_suite(*, backend):
    if not __debug__:
        raise RuntimeError("precision validation requires Python assertions; do not use -O")
    worker = create_backend(backend)
    started = perf_counter()
    results = []
    try:
        for case in smoke_cases():
            result = run_roundtrip(case, worker)
            if case.name == "fresh_iterations":
                result["second_iteration"] = run_roundtrip(next_iteration(case), worker)
            results.append(result)
    finally:
        worker.destroy()
    return {
        "schema_version": 1,
        "backend": backend,
        "api_semantics": "v1-ordinary",
        "execution": "single-process-cpu-simulation",
        "dtype": "bfloat16",
        "device_communication_verified": False,
        "wheel_api_verified": False,
        "device_reference_captured": False,
        "second_reference": "cpu-bf16-fp32-path-simulation",
        "case_count": len(results),
        "results": results,
        "suite_wall_seconds": perf_counter() - started,
        "passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("functional-stub",))
    args = parser.parse_args(argv)
    # Exceptions remain failures. No exception-to-skip or fallback conversion.
    print(json.dumps(run_suite(backend=args.backend), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
