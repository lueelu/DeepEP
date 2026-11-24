# Copyright (c) 2026, Lu Lu

"""AOT/JIT clock-alignment benchmark over one shared model and wait ABI.

Run the benchmark once per installed wheel (AOT, then JIT) with identical
arguments and topology, then compare the two reports:

    python tests/elastic/ll_clock_benchmark.py --label aot --report /tmp/aot.json <launch args>
    python tests/elastic/ll_clock_benchmark.py --label jit --report /tmp/jit.json <launch args>
    python tests/elastic/ll_clock_benchmark.py --compare /tmp/aot.json /tmp/jit.json

The device run delegates to the LL ST graph entry with the formal Exp prefix, so
both builds share one clock model, one admission rule, one wait launcher and one
record/output verification path. The report records the package identity, the
shared model digest and the wait ABI; the comparison refuses to produce a delta
when the two runs did not use the same protocol.

The comparison is CPU-only. It compares only Dispatch durations: wait kernels,
the probe replay, calibration and the Exp prefix are excluded by construction.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

PROTOCOL = "deepep-clock-benchmark.v1"
EXPECTED_EXP_PREFIX = 100
PROTOCOL_FIELDS = (
    "iterations", "calibration_windows", "calibration_gap_secs", "lead_us", "interval_us",
    "warmup", "exp_prefix_repeats", "samples", "record_words", "record_stride_words",
    "record_stride_bytes", "cores",
)


def st_module():
    """Load the LL ST module; the benchmark reuses its clock protocol."""
    module_path = Path(__file__).with_name("test_ep_low_latency.py")
    spec = importlib.util.spec_from_file_location("ll_st_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_module():
    """Load the shared clock model, falling back to this checkout.

    An older AOT wheel may predate the model module. The benchmark then loads the
    repository copy explicitly instead of re-implementing the math, so both
    builds still compare the same model code.
    """
    try:
        from deep_ep.utils import ll_clock_model

        return ll_clock_model
    except ImportError:
        path = Path(__file__).resolve().parents[2] / "deep_ep" / "utils" / "ll_clock_model.py"
        spec = importlib.util.spec_from_file_location("deep_ep_ll_clock_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def model_module_digest():
    """Identity of the shared clock model; both builds must load the same file."""
    model = load_model_module()
    path = Path(model.__file__).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "schema": model.MODEL_SCHEMA}


def package_identity():
    import deep_ep

    package = Path(deep_ep.__file__).resolve()
    native = sorted(package.parent.glob("_C*.so"))
    return {
        "package": str(package),
        "native_sha256": sha256_file(native[0]) if native else None,
        "native_name": native[0].name if native else None,
    }


def wait_abi(native, st):
    """Wait ABI actually launched with; schema 2 and a 128-byte slot are required."""
    words, stride_words = st.clock_wait_table_contract(native)
    return {
        "schema": getattr(native, "low_latency_clock_wait_schema", None),
        "words": words,
        "stride_words": stride_words,
        "stride_bytes": getattr(native, "low_latency_clock_wait_record_stride_bytes", None),
        "alignment_bytes": getattr(native, "low_latency_clock_wait_record_alignment_bytes", None),
        "arg_bytes": getattr(native, "low_latency_clock_wait_arg_bytes", None),
        "record_fields": [str(name) for name in
                          getattr(native, "low_latency_clock_wait_record_fields", [])],
    }


def protocol_fields(clock, config):
    """Every sampling condition that must match between the two builds."""
    values = {field: clock.get(field) for field in PROTOCOL_FIELDS if field != "warmup"}
    values["warmup"] = config.get("warmup")
    return values


def benchmark_case(result):
    """Reduce one ST result to the fields a comparison needs."""
    clock = result.get("clock_alignment")
    if not clock:
        raise RuntimeError("The run produced no clock graph summary; check the graph entry")
    replay = clock.get("replay") or {}
    model = clock.get("model") or {}
    performance = clock.get("performance")
    if performance is None:
        raise RuntimeError(
            "The run produced no formal replay performance; do not pass --skip-perf-test")
    if replay.get("accuracy_status") == "threshold_exceeded":
        raise RuntimeError("The run exceeded its configured accuracy threshold")
    return {
        "case_id": result.get("case_id"),
        "config": result.get("config"),
        "route": result.get("route"),
        "cores": clock.get("cores"),
        "replays": clock.get("replays"),
        "iterations": clock.get("iterations"),
        "exp_prefix_repeats": clock.get("exp_prefix_repeats"),
        "accuracy_status": replay.get("accuracy_status"),
        "pair_error_bound_ns_max": replay.get("pair_error_bound_ns_max"),
        "local_error_bound_ns_max": replay.get("local_error_bound_ns_max"),
        "residual_cycles": model.get("residual_cycles"),
        "residual_validated": model.get("residual_validated"),
        "rate_ppm": model.get("rate_ppm"),
        "alpha_error_bound_ppm": model.get("alpha_error_bound_ppm"),
        "calibration_age_secs": clock.get("calibration_age_secs"),
        "acceptance": clock.get("acceptance"),
        "performance": performance,
    }


def run_device(arguments, report_path, label):
    """Run the shared clock-graph protocol once and save the benchmark report."""
    st = st_module()
    report_path = Path(report_path)
    raw_path = report_path.with_name(report_path.stem + ".raw.json")
    if report_path.exists() or raw_path.exists():
        raise ValueError(f"Refusing to overwrite {report_path} or {raw_path}")
    forced = ("--clock-align", "--clock-align-mode", "--dispatch-only", "--do-cpu-sync",
              "--clock-align-graph-exp-prefix")
    overridden = [flag for flag in forced
                  if any(item == flag or item.startswith(flag + "=") for item in arguments)]
    if overridden:
        raise ValueError(f"The benchmark protocol owns {overridden}; remove them from the launch args")
    argv = list(arguments) + [
        "--clock-align", "--clock-align-mode", "graph", "--dispatch-only", "--do-cpu-sync", "0",
        "--clock-align-graph-exp-prefix", str(EXPECTED_EXP_PREFIX), "--output", str(raw_path),
    ]
    st.main(argv)
    raw = json.loads(raw_path.read_text())
    results = raw.get("results") or []
    if len(results) != 1:
        raise ValueError("The benchmark samples exactly one case per run")
    case = benchmark_case(results[0])
    if case["exp_prefix_repeats"] != EXPECTED_EXP_PREFIX:
        raise RuntimeError(
            f"The replay did not run behind the {EXPECTED_EXP_PREFIX}-iteration Exp prefix")
    from deep_ep._native import require_native

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": PROTOCOL,
        "label": label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "package": package_identity(),
        "model_module": model_module_digest(),
        "wait_abi": wait_abi(require_native("LL clock benchmark"), st),
        "protocol_fields": protocol_fields(results[0]["clock_alignment"], raw.get("config") or {}),
        "cases": [case],
        "notes": [
            "One contemporaneous A/B only; this is not a stable speedup claim.",
            "Dispatch durations only: wait, probe replay, calibration and Exp are excluded.",
            "Model-domain budgets are reported separately from the release skew.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare_reports(left, right):
    """Compare two benchmark reports only when every protocol field matches."""
    for name, report in (("left", left), ("right", right)):
        if report.get("protocol") != PROTOCOL:
            raise ValueError(f"{name} is not a {PROTOCOL} report")
    if left["protocol_fields"] != right["protocol_fields"]:
        raise ValueError(
            f"Sampling protocols differ; refusing to compare: "
            f"{left['protocol_fields']} vs {right['protocol_fields']}")
    if left["model_module"]["sha256"] != right["model_module"]["sha256"]:
        raise ValueError("The two runs used different clock model modules")
    if left["wait_abi"] != right["wait_abi"]:
        raise ValueError(f"The two runs used different wait ABIs: "
                         f"{left['wait_abi']} vs {right['wait_abi']}")
    rows = []
    for left_case, right_case in zip(left["cases"], right["cases"]):
        if left_case["config"] != right_case["config"]:
            raise ValueError("Case configurations differ; compare the same case only")
        for name, case in (("left", left_case), ("right", right_case)):
            if not case.get("performance"):
                raise ValueError(f"{name} has no formal replay performance")
            if case.get("accuracy_status") == "threshold_exceeded":
                raise ValueError(f"{name} exceeded its accuracy threshold")
        left_perf, right_perf = left_case["performance"], right_case["performance"]
        base = left_perf["max_rank_mean_kernel_us"]
        delta = right_perf["max_rank_mean_kernel_us"] - base
        rows.append({
            "case_id": left_case["case_id"],
            "left_slowest_rank_mean_us": base,
            "right_slowest_rank_mean_us": right_perf["max_rank_mean_kernel_us"],
            "delta_us": delta,
            "relative": None if not base else delta / base,
            "left_mean_us": left_perf["mean_kernel_us"],
            "right_mean_us": right_perf["mean_kernel_us"],
            "left_p99_us": left_perf["p99_kernel_us"],
            "right_p99_us": right_perf["p99_kernel_us"],
            "accuracy_status": {"left": left_case.get("accuracy_status"),
                                "right": right_case.get("accuracy_status")},
        })
    return {
        "protocol": PROTOCOL,
        "cases": rows,
        "notes": [
            "One contemporaneous A/B; do not report it as a stable speedup.",
            "Difference is only comparable under the identical protocol fields recorded here.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"),
                        help="Compare two benchmark reports on CPU")
    parser.add_argument("--report", help="Benchmark report path for a device run")
    parser.add_argument("--label", default="wheel", help="Short label such as aot or jit")
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="Launch arguments forwarded to the LL ST entry")
    options = parser.parse_args(argv)
    if options.compare:
        left = json.loads(Path(options.compare[0]).read_text())
        right = json.loads(Path(options.compare[1]).read_text())
        print(json.dumps(compare_reports(left, right), indent=2))
        return
    if not options.report:
        parser.error("--report is required for a device run")
    print(json.dumps(run_device(options.arguments, options.report, options.label), indent=2))


if __name__ == "__main__":
    sys.exit(main())
