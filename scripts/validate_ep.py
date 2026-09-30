# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Run installed native EP or explicit NPU Tensor stubs without rebuilding.

Requires two explicitly reserved devices. Tests are staged outside the source
tree; no package installation, fallback, automatic device choice or reset.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from time import monotonic


CASE_NAMES = ("mixed_k1", "cross_rank", "same_rank", "hotspot", "masked_zero", "fresh_iterations")


def validate_devices(value):
    if not isinstance(value, str) or len(value) > 15 or not re.fullmatch(r"[0-9]+,[0-9]+", value):
        raise ValueError("Explicitly reserve two devices with --devices or ASCEND_RT_VISIBLE_DEVICES (e.g. 0,1)")
    devices = list(map(int, value.split(",")))
    if devices[0] == devices[1] or any(device > 255 for device in devices):
        raise ValueError("Devices must be distinct IDs in [0, 255]")
    return ",".join(map(str, devices))


def launch(command, *, cwd, env, timeout):
    with subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True) as process:
        try:
            process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def worker_command(python, backend, case, directory):
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=0",
        "--log-dir",
        str(directory / "logs"),
        "--tee=3",
        "-m",
        "tests.integration.npu_roundtrip",
        "--backend",
        backend,
        "--case",
        case,
        "--result-dir",
        str(directory),
    ]


def read_results(directory, *, backend, names):
    reports = []
    for rank in range(2):
        path = directory / f"rank-{rank}.json"
        if path.is_symlink() or path.stat().st_size > 1 << 20:
            raise RuntimeError("Invalid or oversized rank report")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != 1
            or report.get("rank") != rank
            or report.get("world_size") != 2
            or report.get("backend") != backend
            or report.get("passed") is not True
            or report.get("npu_tensor_roundtrip_tested") is not True
            or report.get("native_ep_verified") is not (backend == "native")
            or report.get("wheel_api_verified") is not (backend == "native")
            or report.get("device_reference_captured") is not False
            or report.get("second_reference") != "cpu-bf16-fp32-path-simulation"
            or report.get("data_transport")
            != ("installed-deepep" if backend == "native" else "hccl-reference-collectives")
        ):
            raise RuntimeError("Rank report identity/qualification mismatch")
        records = report.get("results", [])
        if [record.get("name") for record in records] != list(names):
            raise RuntimeError("Rank report did not execute every requested case")
        for record in records:
            checks = [record]
            if record["name"] == "fresh_iterations":
                checks.append(record.get("second_iteration", {}))
            for check in checks:
                if (
                    check.get("passed") is not True
                    or check.get("npu_tensors_checked") is not True
                    or check.get("dispatch_checked") is not True
                    or check.get("precision", {}).get("passed") is not True
                ):
                    raise RuntimeError("A required NPU stage or precision check did not pass")
        reports.append(report)
    if backend == "native":
        identities = [report.get("build_identity_sha256") for report in reports]
        if (
            not all(isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity) for identity in identities)
            or identities[0] != identities[1]
        ):
            raise RuntimeError("Native ranks used different or missing build identities")
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("npu-stub", "native"))
    parser.add_argument("--devices", default=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    parser.add_argument("--case", choices=("all", *CASE_NAMES), default="all")
    parser.add_argument(
        "--output", type=Path, required=True, help="New dedicated directory for staged tests and results"
    )
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per torchrun job, including initialization")
    args = parser.parse_args(argv)
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
        parser.error("NPU validation requires Python assertions; unset PYTHONOPTIMIZE and do not use -O")
    if sys.platform != "linux":
        parser.error("NPU validation requires Linux")
    try:
        devices = validate_devices(args.devices)
    except ValueError as error:
        parser.error(str(error))
    if not 30 <= args.timeout <= 1800:
        parser.error("timeout must be in [30, 1800] seconds")
    if args.backend == "native" and not os.environ.get("DEEPEP_SHMEM_ENDPOINT"):
        parser.error("Native EP requires an explicit DEEPEP_SHMEM_ENDPOINT")
    source = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == source or source.is_relative_to(output) or output.is_relative_to(source):
        parser.error("output must be outside, and must not contain, the source tree")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "tests", output / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=devices, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("HCCL_CONNECT_TIMEOUT", "120")
    env.setdefault("HCCL_EXEC_TIMEOUT", "120")
    names = CASE_NAMES if args.case == "all" else (args.case,)
    # Native buffers have not yet qualified safe reinitialization. Isolate
    # cases until they do; no new wheel or environment per case. The NPU stub
    # owns no SHMEM resource and can batch the full suite in one job.
    batches = ((args.case, names),) if args.backend == "npu-stub" else tuple((name, (name,)) for name in names)
    started, results = monotonic(), []
    for name, selected in batches:
        directory = output / name
        directory.mkdir()
        print(f"NPU roundtrip: backend={args.backend}, cases={','.join(selected)}", flush=True)
        launch(worker_command(sys.executable, args.backend, name, directory), cwd=output, env=env, timeout=args.timeout)
        results.extend(read_results(directory, backend=args.backend, names=selected))
    summary = {
        "schema_version": 1,
        "backend": args.backend,
        "world_size": 2,
        "cases": list(names),
        "npu_tensor_roundtrip_tested": True,
        "native_ep_verified": args.backend == "native",
        "wheel_api_verified": args.backend == "native",
        "suite_wall_seconds": monotonic() - started,
        "rank_reports": results,
        "passed": True,
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"NPU Tensor roundtrip passed; native_ep_verified={summary['native_ep_verified']}", flush=True)


if __name__ == "__main__":
    main()
