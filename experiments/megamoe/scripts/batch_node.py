# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Remote stages for batch_launch.py; paths/configuration come from the shared run manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def interface_for(host, interfaces):
    names = {
        item["ifname"]
        for item in interfaces
        for addr in item.get("addr_info", [])
        if addr.get("family") == "inet" and addr.get("local") == host
    }
    if len(names) != 1:
        raise ValueError(f"Expected one local interface for {host}, found {sorted(names)}")
    return names.pop()


def wheel_info(path):
    with zipfile.ZipFile(path) as wheel:
        return json.loads(wheel.read("deep_ep/_build_info.json"))


def existing_wheel(root, installed):
    candidates = []
    for path in (root / "dist").glob("build-*/ascend_deepep-*.whl"):
        if wheel_info(path) == installed:
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Found {len(candidates)} wheels matching master's installed build; "
            "select --wheel explicitly or run without --skip-build"
        )
    return candidates[0]


def benchmark_command(plan, run_dir, rank, bs):
    command = [
        sys.executable,
        "-I",
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(len(plan["hosts"])),
        "--nproc-per-node",
        "8",
        "--node-rank",
        str(rank),
        "--master-addr",
        plan["hosts"][0],
        "--master-port",
        str(plan["master_port"]),
        "--max-restarts",
        "0",
        "--module",
        "deep_ep_experimental.megamoe.benchmark",
        "--config",
        str(run_dir / "case.json"),
        "--bs-values",
        str(bs),
        "--mode",
        plan["mode"],
        "--input",
        plan["input"],
        "--routing",
        plan["routing"],
        "--output-dir",
        str(run_dir / f"bs_{bs}"),
    ]
    if plan["mode"] == "accuracy":
        command += ["--check-tokens", "0", "--golden-dir", str(Path(plan["root"]) / "benchmark-golden")]
    if plan["timer"]:
        command += ["--timer", "--timer-ranks", "0-7"]
    return command


def golden_command(plan, run_dir, rank, bs):
    return [
        sys.executable,
        "-I",
        "-m",
        "deep_ep_experimental.megamoe.golden",
        "--config",
        str(run_dir / "case.json"),
        "--bs-values",
        str(bs),
        "--ep-size",
        str(len(plan["hosts"]) * 8),
        "--ranks",
        f"{rank * 8}-{rank * 8 + 7}",
        "--input",
        plan["input"],
        "--routing",
        plan["routing"],
        "--golden-dir",
        str(Path(plan["root"]) / "benchmark-golden"),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "prepare", "install", "verify", "golden", "run", "parse"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--bs", type=int)
    args = parser.parse_args(argv)
    run_dir, rank = args.run_dir.resolve(strict=True), args.node_rank
    plan = read_json(run_dir / "launch.json")
    root = Path(plan["root"])
    if not 0 <= rank < len(plan["hosts"]) or run_dir.name != plan["run_id"]:
        raise ValueError("Run identity/rank mismatch; verify the shared mount")
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "experiments/megamoe/scripts"))
    from build_shmem import cann_version

    cann = cann_version(read_json(root / "dependencies.lock.json")["cann_version"])

    def run(command, capture=False, cwd=None):
        result = subprocess.run(
            command, cwd=cwd or "/tmp", check=True, text=True, stdout=subprocess.PIPE if capture else None
        )
        return result.stdout

    def git(*values):
        return run(["git", "-C", str(root), *values], capture=True).strip()

    def probe():
        target = run_dir / f"node_{rank}_verify.json"
        code = """import json, sys, torch, torch_npu
from importlib.resources import files
from deep_ep._native import require_native
require_native("cluster preflight")
import deep_ep_experimental.megamoe._C
info = json.loads(files("deep_ep").joinpath("_build_info.json").read_text())
with open(sys.argv[1], "w") as f:
    json.dump(dict(build=info, torch=torch.__version__, torch_npu=torch_npu.__version__,
                   python=list(sys.version_info[:2])), f)
"""
        run([sys.executable, "-I", "-c", code, str(target)])
        return read_json(target)

    interfaces = json.loads(run(["ip", "-j", "-4", "addr", "show", "up"], capture=True))
    interface = interface_for(plan["hosts"][rank], interfaces)
    os.environ.update(
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        DEEPEP_SHMEM_ENDPOINT=f"tcp://{plan['hosts'][0]}:{plan['shmem_port']}",
    )
    if args.stage == "preflight":
        if git("status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Shared source/submodule has tracked modifications; preserve/review before launching")
        run(
            [
                sys.executable,
                "-I",
                "-c",
                "import numpy, torch, torch_npu; "
                "print(torch.__version__, torch_npu.__version__); "
                "assert torch.npu.device_count() >= 8, 'Need eight visible NPU devices'",
            ]
        )
        run(["npu-smi", "info"])
        write_json(
            run_dir / f"node_{rank}_preflight.json",
            dict(
                run_id=plan["run_id"],
                host=plan["hosts"][rank],
                interface=interface,
                cann=cann,
                commit=git("rev-parse", "HEAD"),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            ),
        )
        print(f"[PREFLIGHT] node={rank} ip={plan['hosts'][rank]} ifname={interface}", flush=True)
        return
    artifacts_path = run_dir / "artifacts.json"
    if args.stage == "prepare":
        if rank != 0:
            raise ValueError("Only rank0 prepares shared artifacts")
        wheel = Path(plan["wheel"]) if plan["wheel"] else None
        if wheel is None and plan["skip_build"]:
            wheel = existing_wheel(root, probe()["build"])
        if wheel is None:
            # 集群仅使用这一份仓内 SDK，避免每台机器隐式选择不同的外部路径。
            for name in ("SHMEM_ROOT", "CATLASS_ROOT"):
                os.environ.pop(name, None)
            os.environ.update(DEEPEP_BUILD_MEGAMOE="ON", DEEPEP_MEGAMOE_TIMER="ON")
            from build import build

            wheel = build(root)
        info = wheel_info(wheel)
        if not info["megamoe"]["enabled"] or (plan["timer"] and not info["megamoe"]["timer"]):
            raise ValueError("Selected wheel does not support the requested MegaMoE/timer mode")
        if info["cann"] != {"version": cann["Version"], "timestamp": cann.get("timestamp", "unknown")}:
            raise ValueError("Selected wheel was built with a different CANN build")
        write_json(
            artifacts_path,
            dict(wheel=str(wheel), wheel_sha256=sha256(wheel), build=info, commit=git("rev-parse", "HEAD"), cann=cann),
        )
        print(f"[ARTIFACT] {wheel}", flush=True)
        return
    artifacts = read_json(artifacts_path)
    if artifacts["commit"] != git("rev-parse", "HEAD") or cann != artifacts["cann"]:
        raise RuntimeError("Source commit or CANN build differs from master's recorded artifacts")
    wheel = Path(artifacts["wheel"])
    if sha256(wheel) != artifacts["wheel_sha256"]:
        raise RuntimeError("Shared wheel changed after preparation")
    if args.stage == "install":
        run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)])
    elif args.stage == "verify":
        result = probe()
        if result["build"] != artifacts["build"]:
            raise RuntimeError("Installed package does not match selected wheel")
    elif args.stage in ("golden", "run"):
        if args.bs not in plan["config"]["bs_values"]:
            raise ValueError("BS not in launch plan")
        factory = golden_command if args.stage == "golden" else benchmark_command
        command = factory(plan, run_dir, rank, args.bs)
        print(f"[{args.stage.upper()}] node={rank} BS={args.bs} ifname={interface}", flush=True)
        # 替换本进程，torchrun/worker 留在远端 shell 管理的本次进程组内。
        os.execv(sys.executable, command)
    elif args.stage == "parse":
        if rank != 0:
            raise ValueError("Only the master parses the selected timer ranks")
        for bs in plan["config"]["bs_values"]:
            run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "deep_ep_experimental.megamoe.timer",
                    str(run_dir / f"bs_{bs}/bs_{bs}/timer"),
                    "--ranks",
                    "0-7",
                ]
            )


if __name__ == "__main__":
    main()
