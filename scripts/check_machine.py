"""Check Buffer's machine selection without importing Torch or initializing an NPU."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.machinery import PathFinder
import importlib.util
import json
from pathlib import Path
import sys


def load_selector(implementation):
    if implementation == "source":
        path = Path(__file__).resolve().parents[1] / "deep_ep" / "_machine.py"
    else:
        # Resolve the active Python installation without executing deep_ep.__init__.
        spec = PathFinder.find_spec("deep_ep")
        if spec is None or not spec.origin:
            raise RuntimeError(
                "deep_ep is not installed in this Python; use --implementation source for a source check"
            )
        path = Path(spec.origin).resolve().with_name("_machine.py")
    if not path.is_file():
        raise RuntimeError(f"No machine selector at {path}; check the active Python installation")
    spec = importlib.util.spec_from_file_location("_deepep_machine_check", path)
    selector = importlib.util.module_from_spec(spec)
    # Standalone import: _machine uses a CPU lock and does not import deep_ep/Torch.
    spec.loader.exec_module(selector)
    if not callable(getattr(selector, "detect_physical", None)):
        raise RuntimeError(
            f"Selector at {path} lacks detect_physical; update the package or check --implementation source"
        )
    return selector, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def physical_id(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("physical device IDs must be integers") from exc
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("physical device IDs must be in [0, 2**32)")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-devices",
        "--physical-device",
        type=physical_id,
        nargs="+",
        required=True,
        help="Physical chip IDs to check, not LOCAL_RANK or visible logical device IDs",
    )
    parser.add_argument(
        "--expect",
        choices=("mesh_clos", "global_clos", "mesh_pcie"),
        help="Fail if auto-detection differs",
    )
    parser.add_argument("--rootinfo", type=Path, help="Explicit rootinfo; missing/invalid files fail without fallback")
    parser.add_argument(
        "--implementation",
        choices=("installed", "source"),
        default="installed",
        help="Check the active Python package (default), or this checkout without installation",
    )
    parser.add_argument("--json", type=Path, help="Write a new JSON report; existing files are never overwritten")
    args = parser.parse_args(argv)
    if len(set(args.physical_devices)) != len(args.physical_devices):
        parser.error("physical device IDs must be distinct")
    if args.json is not None and args.json.exists():
        parser.error(f"report already exists: {args.json}")

    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "implementation": args.implementation,
        "expected_machine": args.expect,
        "physical_devices": args.physical_devices,
        "rootinfo_path": str(args.rootinfo or "/etc/hccl_rootinfo.json"),
        "selector": None,
        "results": [],
        "errors": [],
        "passed": False,
        "scope": "Topology classification only; logical-device mapping, SHMEM initialization and link health untested",
    }
    try:
        selector, report["selector"] = load_selector(args.implementation)
        print(f"Selector: {report['selector']['path']}", flush=True)
        for physical in args.physical_devices:
            print(f"Checking physical={physical} (DCMI query, if needed: 5s timeout)...", flush=True)
            record = {"physical_device": physical, "selection": None, "passed": False, "error": None}
            try:
                selection = selector.detect_physical(physical, args.rootinfo)
                record["selection"] = selection
                if args.expect is not None and selection["machine_type"] != args.expect:
                    raise ValueError(f"expected {args.expect}, detected {selection['machine_type']}")
                record["passed"] = True
                print(
                    f"physical={physical} machine_type={selection['machine_type']} source={selection['source']} OK",
                    flush=True,
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"physical={physical} FAILED: {record['error']}", flush=True)
            report["results"].append(record)
        kinds = {r["selection"]["machine_type"] for r in report["results"] if r["selection"] is not None}
        if len(kinds) > 1:
            report["errors"].append(f"Selected physical devices disagree on machine type: {sorted(kinds)}")
        report["passed"] = all(r["passed"] for r in report["results"]) and not report["errors"]
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")

    for error in report["errors"]:
        print(f"FAILED: {error}", flush=True)
    print(report["scope"], flush=True)
    if args.json is not None:
        with args.json.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(f"JSON: {args.json.resolve()}", flush=True)
    print("MACHINE_CHECK_PASSED" if report["passed"] else "MACHINE_CHECK_FAILED", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
