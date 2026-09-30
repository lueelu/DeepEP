# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""Profiler commands use published interfaces; no changes to operator kernels."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import shlex
import subprocess

KERNELS = {
    "alltoallv_gmm": ("Ascend950AllToAllVGroupedMatmul", "AllToAllVGroupedMatMulUdma"),
    "gmm_alltoallv": ("GroupedMatmulAllToAllV", "GroupedMatMulAllToAllVUdma"),
}


@dataclass
class Profiler:
    argv: list
    help: str
    version: str

    def record(self):
        return asdict(self)


def discover(metric="PipeTimeline"):
    errors = []
    candidates = (("msprof", []),) if metric == "op_summary" else (("msopprof", []), ("msprof", ["op"]))
    for name, suffix in candidates:
        path = shutil.which(name)
        if path is None:
            continue
        command = [path, *suffix]
        help_result = subprocess.run([*command, "--help"], text=True, capture_output=True, timeout=30)
        help_text = help_result.stdout + help_result.stderr
        required = ("--application", "--output", "--task-time") if metric == "op_summary" else ("--aic-metrics",)
        if help_result.returncode == 0 and all(option in help_text for option in required):
            version = subprocess.run([path, "--version"], text=True, capture_output=True, timeout=30)
            return Profiler(command, help_text, version.stdout + version.stderr)
        errors.append(help_text)
    raise RuntimeError(f"no usable profiler found for {metric}: " + "\n".join(errors))


def command(profiler, metric, case, output, application, warmup, iterations, pipes=None):
    if metric == "op_summary":
        if pipes:
            raise ValueError("pipe filtering is only available for InstrTimeline")
        required = ("--application", "--output", "--task-time")
        if any(option not in profiler.help for option in required):
            raise RuntimeError("msprof lacks required application/task-time options")
        return [
            *profiler.argv,
            f"--output={output}",
            "--task-time=on",
            "--application=" + shlex.join(application),
        ]
    if metric not in ("PipeTimeline", "InstrTimeline"):
        raise ValueError("select PipeTimeline or InstrTimeline in separate captures")
    candidates = (metric,) if metric == "PipeTimeline" else ("InstrTimeline", "instrTimeLine", "InstrTimeLine")
    actual = next((x for x in candidates if x in profiler.help), None)
    if actual is None:
        raise RuntimeError(f"installed profiler help does not advertise {metric}; see saved help/version")
    if case.operator not in KERNELS:
        raise ValueError("profile the two component operators separately")
    required = ("--kernel-name", "--launch-count", "--launch-skip", "--kill", "--output")
    if any(option not in profiler.help for option in required):
        raise RuntimeError("profiler lacks required kernel selection/normal-exit options")
    args = [
        *profiler.argv,
        f"--aic-metrics={actual}",
        f"--output={output}",
        "--kernel-name=" + "|".join(KERNELS[case.operator]),
        f"--launch-skip={warmup + 1}",
        f"--launch-count={iterations}",
        "--kill=off",
    ]
    if pipes:
        selected = pipes.split("|")
        if metric != "InstrTimeline" or "--instr-timeline-pipe" not in profiler.help:
            raise ValueError("pipe filtering requires InstrTimeline and --instr-timeline-pipe support")
        if not set(selected) <= {"cube", "fixp", "vector", "mte1", "mte2", "mte3"}:
            raise ValueError("unknown instruction pipe")
        args.append(f"--instr-timeline-pipe={pipes}")
    return args + list(application)


def inspect_capture(path, operator, *, metric="PipeTimeline", warmup=0, iterations=1):
    """Reject empty/mismatched captures; retain observed scope without claiming completeness."""
    root = Path(path)
    if metric == "op_summary":
        from .msprof_parser import parse_msprof_case

        names = {"alltoallv_gmm": "all_to_allv_grouped_mat_mul", "gmm_alltoallv": "grouped_mat_mul_all_to_allv"}
        return parse_msprof_case(
            root,
            {
                "id": "capture",
                "operator": names[operator],
                "world_size": 1,
                "performance": {"warmup": warmup + 1, "repeats": 1, "align_iterations": 0, "iterations": iterations},
            },
        )
    traces = list(root.rglob("trace.json"))
    binaries = [p for p in root.rglob("visualize_data.bin") if p.stat().st_size]
    if not traces or not binaries:
        raise ValueError("capture is missing nonempty trace.json / visualize_data.bin")
    names, tracks, events, files = set(), set(), 0, []
    kernel_seen = False
    for trace in traces:
        data = json.loads(trace.read_text(encoding="utf-8"))
        serialized = json.dumps(data)
        kernel_seen |= any(k.lower() in (str(trace) + serialized).lower() for k in KERNELS[operator])
        rows = data.get("traceEvents", []) if isinstance(data, dict) else data
        for event in rows:
            if isinstance(event, dict):
                name = str(event.get("name", ""))
                names.add(name)
                tracks.add((str(event.get("pid", "")), str(event.get("tid", ""))))
                if event.get("ph") in ("X", "B", "E"):
                    events += 1
        files.append({"path": str(trace), "sha256": hashlib.sha256(trace.read_bytes()).hexdigest()})
    if not kernel_seen or events == 0:
        raise ValueError("no matching fused kernel or no timeline events in capture")
    pipes = sorted({pipe for name in names for pipe in re.findall(r"CUBE|FIXP|VECTOR|MTE[123]|SCALAR", name.upper())})
    return {
        "passed": True,
        "trace_files": files,
        "binary_files": [str(p) for p in binaries],
        "events": events,
        "observed_tracks": sorted(tracks),
        "observed_pipes": pipes,
        "scope": "observed profiler events; full core/instruction coverage is not assumed",
    }
