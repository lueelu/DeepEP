# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Build wheel/sdist, install into a fresh venv, and test outside the checkout.

No packages are installed into the caller's environment. --system-site-packages
explicitly reuses a provisioned framework/SDK Python environment, not a claim
of complete dependency isolation. CANN/SHMEM remain external system SDKs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import venv
import zipfile


def run(command, cwd, env, timeout=1800):
    print("Running:", " ".join(map(str, command)), flush=True)
    with subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, start_new_session=True) as process:
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            # Kill only the process group created for this command, including
            # torchrun workers. Never leave orphaned device users after timeout.
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
        if status:
            raise subprocess.CalledProcessError(status, command)


def one(directory, pattern):
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} in {directory}; found {len(matches)}")
    return matches[0]


def inspect_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len([n for n in names if n.startswith("deep_ep/_C.") and n.endswith(".so")]) != 1:
            raise RuntimeError("Wheel must contain exactly one native extension")
        if "deep_ep/_build_info.json" not in names:
            raise RuntimeError("Missing build identity")
        if any(n.startswith(("tests/", "benchmarks/", "csrc/")) for n in names):
            raise RuntimeError("Unexpected package content")
        if len([n for n in names if n.endswith("deepseek-deepep.LICENSE")]) != 1:
            raise RuntimeError("Wheel must include the adopted API's third-party license")
        info = json.loads(archive.read("deep_ep/_build_info.json"))
        beta = info.get("megamoe", {}).get("enabled", False)
        beta_names = [name for name in names if name.startswith("deep_ep_experimental/megamoe/")]
        if beta:
            required = {
                "deep_ep_experimental/megamoe/api.py",
                "deep_ep_experimental/megamoe/benchmark.py",
                "deep_ep_experimental/megamoe/configs/smoke.json",
                "deep_ep_experimental/megamoe/libmegamoe_kernel.so",
            }
            extensions = [
                n for n in beta_names if n.startswith("deep_ep_experimental/megamoe/_C.") and n.endswith(".so")
            ]
            if len(extensions) != 1 or not required.issubset(names):
                raise RuntimeError("Incomplete MegaMoE Beta wheel")
            if not any(name.endswith("cann.LICENSE") for name in names):
                raise RuntimeError("Missing CANN license for MegaMoE")
        elif beta_names:
            raise RuntimeError("MegaMoE Beta payload in an OFF wheel; use a fresh build directory")
        component = "deep_ep_experimental/ep_gmm_fused/"
        if any(component in n and not n.startswith(component) for n in names):
            raise RuntimeError("Wheel contains nested EP_GMM_FUSED staging artifacts")
        if info.get("ep_gmm_fused", {}).get("enabled", False):
            if len([n for n in names if n.startswith(component + "_C.") and n.endswith(".so")]) != 1:
                raise RuntimeError("EP_GMM_FUSED wheel must contain exactly one component extension")
            required = {
                "__init__.py",
                "_build_info.json",
                "NOTICE.md",
                "upstream.json",
                "licenses/shmem/LICENSE",
                "licenses/catccos/LICENSE",
                "licenses/catlass/LICENSE",
            }
            missing = sorted(component + name for name in required if component + name not in names)
            if missing:
                raise RuntimeError(f"Incomplete EP_GMM_FUSED wheel: {missing}")
            if any(n.startswith(component + ".libs/") for n in names):
                raise RuntimeError("EP_GMM_FUSED must use the external SHMEM SDK; bundled libraries are unsupported")
            component_info = json.loads(archive.read(component + "_build_info.json"))
            if (
                not info.get("shmem", {}).get("revision")
                or component_info.get("shmem_revision") != info["shmem"]["revision"]
            ):
                raise RuntimeError("EP_GMM_FUSED and the core runtime must use the same SHMEM revision")
        elif any(n.startswith(component) for n in names):
            raise RuntimeError("Core wheel contains disabled EP_GMM_FUSED artifacts")

        kernel_capabilities = {
            "elastic-notify-metadata-v6",
            "elastic-dispatch-expanded-v1",
            "elastic-combine-expanded-v3",
        }
        if kernel_capabilities.intersection(info.get("capabilities", [])):
            if "deep_ep/libdeepep_kernels.so" not in names:
                raise RuntimeError("Kernel capabilities require their packaged Device library")
        if "deep_ep/libdeepep_notify.so" in names:
            raise RuntimeError("Wheel contains a stale libdeepep_notify.so; rebuild the package")
        operations = info.get("communication_operations")
        known = {
            "Buffer.get_dispatch_layout",
            "Buffer.dispatch",
            "Buffer.combine",
            "ElasticBuffer.dispatch",
            "ElasticBuffer.combine",
        }
        if (
            not isinstance(operations, list)
            or any(not isinstance(operation, str) or operation not in known for operation in operations)
            or len(operations) != len(set(operations))
        ):
            raise RuntimeError("Invalid or unknown communication_operations build metadata")
        return info


def resolve_component_paths(env, source):
    # A rebuild from an unpacked sdist must use the same provisioned SDKs.
    if env.get("DEEPEP_BUILD_EP_GMM_FUSED", "OFF").upper() not in {"ON", "1", "TRUE"}:
        return
    for name, default in (
        ("CATCCOS_SOURCE_PATH", source / "third-party/catccos"),
        ("SHMEM_SOURCE_PATH", source / "third-party/shmem"),
    ):
        env[name] = str(Path(env.get(name, default)).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New, dedicated artifact directory")
    parser.add_argument("--system-site-packages", action="store_true")
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Use only pre-downloaded build/test wheels (offline)",
    )
    parser.add_argument("--runtime-ranks", type=int, choices=[0, 2, 4, 8], default=0)
    parser.add_argument(
        "--ep-roundtrip",
        choices=("npu-stub", "native"),
        help="After installation, run the EP2 NPU Tensor dispatch/expert/combine suite (explicit backend)",
    )
    args = parser.parse_args()
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
        parser.error("Package validation requires Python assertions; unset PYTHONOPTIMIZE and do not use -O")
    source = Path(__file__).resolve().parents[1]
    if sys.platform != "linux":
        parser.error("Native package validation requires Linux")
    output = args.output.resolve()
    if output == source or source.is_relative_to(output):
        parser.error("output cannot contain the source tree")
    if args.ep_roundtrip and output.is_relative_to(source):
        parser.error("EP roundtrip output must be outside the source tree")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    # sdist 不包含子模块源码；两次构建必须使用原 checkout 的同一份依赖。
    env.setdefault("SHMEM_ROOT", str(source / "third-party/shmem/install/shmem"))
    env.setdefault("CATLASS_ROOT", str(source / "third-party/catlass"))
    env.pop("PYTHONPATH", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    resolve_component_paths(env, source)
    if args.ep_roundtrip:
        if __package__:
            from .validate_ep import validate_devices
        else:
            from validate_ep import validate_devices

        try:
            validate_devices(env.get("ASCEND_RT_VISIBLE_DEVICES"))
        except ValueError as error:
            parser.error(str(error))
        if args.runtime_ranks not in (0, 2):
            parser.error("EP roundtrip uses exactly two devices; runtime-ranks must be 0 or 2")
        if args.ep_roundtrip == "native" and not env.get("DEEPEP_SHMEM_ENDPOINT"):
            parser.error("Native EP roundtrip requires DEEPEP_SHMEM_ENDPOINT")
    if args.runtime_ranks:
        devices = env.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if (
            not env.get("DEEPEP_SHMEM_ENDPOINT")
            or len(devices) != args.runtime_ranks
            or any(not device.isdecimal() for device in devices)
            or len(set(devices)) != len(devices)
        ):
            parser.error("Runtime tests require an explicit, distinct device per rank and DEEPEP_SHMEM_ENDPOINT")
    virtual = output / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=args.system_site_packages).create(virtual)
    python = virtual / "bin/python"
    dependencies = [
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--timeout",
        "20",
        "--retries",
        "1",
        "-r",
        source / "requirements-build.txt",
        "-r",
        source / "requirements-dev.txt",
    ]
    if args.wheelhouse:
        dependencies.extend(["--no-index", "--find-links", args.wheelhouse.resolve()])
    run(dependencies, output, env)
    run(
        [
            python,
            "-m",
            "pytest",
            "--require-framework",
            "--junitxml=" + str(output / "source-tests.xml"),
        ],
        source,
        env,
    )
    run(
        [
            python,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            output / "dist",
            source,
        ],
        output,
        env,
    )
    wheel = one(output / "dist", "*.whl")
    info = inspect_wheel(wheel)
    unpacked = output / "sdist"
    unpacked.mkdir()
    with tarfile.open(one(output / "dist", "*.tar.gz")) as archive:
        for member in archive.getmembers():
            target = (unpacked / member.name).resolve()
            if not target.is_relative_to(unpacked) or not (member.isfile() or member.isdir()):
                raise RuntimeError("Unsafe source archive member")
        archive.extractall(unpacked, filter="data")
    rebuild_source = one(unpacked, "*")
    run(
        [
            python,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            output / "rebuilt",
            rebuild_source,
        ],
        output,
        env,
    )
    rebuilt = one(output / "rebuilt", "*.whl")
    if inspect_wheel(rebuilt) != info:
        raise RuntimeError("sdist rebuild has a different capability/build identity")
    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(rebuilt) as reconstructed:
        original_files = {
            name: original.read(name)
            for name in original.namelist()
            if name.startswith(("deep_ep/", "deep_ep_experimental/")) and ".so" not in Path(name).suffixes
        }
        reconstructed_files = {
            name: reconstructed.read(name)
            for name in reconstructed.namelist()
            if name.startswith(("deep_ep/", "deep_ep_experimental/")) and ".so" not in Path(name).suffixes
        }
        if original_files != reconstructed_files:
            raise RuntimeError("sdist rebuild changed installed Python/metadata content")
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "--no-deps",
            "--no-index",
            wheel,
        ],
        output,
        env,
    )
    probe = (
        "import pathlib,sys,json; import deep_ep; from deep_ep._native import require_native; "
        "root=pathlib.Path(sys.prefix).resolve(); p=pathlib.Path(deep_ep.__file__).resolve(); "
        "assert p.is_relative_to(root), 'Package shadowed by another installation'; "
        "m=require_native('installation check'); assert pathlib.Path(m.__file__).resolve().is_relative_to(root); "
        "import torch,torch_npu; assert not torch.npu.is_initialized(); print('Installed native package verified')"
    )
    if info.get("ep_gmm_fused", {}).get("enabled", False):
        # Both extensions must coexist and load exactly the same external SDK.
        probe += (
            "; import deep_ep_experimental.ep_gmm_fused._C as component; "
            "assert pathlib.Path(component.__file__).resolve().is_relative_to(root); "
            "loaded={pathlib.Path(line.split(maxsplit=5)[5]).resolve() "
            "for line in pathlib.Path('/proc/self/maps').read_text().splitlines() "
            "if len(line.split(maxsplit=5)) == 6 "
            "and pathlib.Path(line.split(maxsplit=5)[5]).name.startswith('libshmem.so')}; "
            "assert loaded == {pathlib.Path(m.shmem_library_path()).resolve()}, loaded; "
            "assert not torch.npu.is_initialized(); print('Installed shared EP_GMM_FUSED runtime verified')"
        )
    run([python, "-I", "-c", probe], output, env, 120)
    if info.get("megamoe", {}).get("enabled"):
        probe_beta = (
            "import pathlib,sys; from deep_ep_experimental.megamoe.api import _extension; "
            "m=_extension(); assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()); "
            "assert callable(m.forward) and callable(m.buffer_size); print('Installed MegaMoE extension loaded; no device call')"
        )
        run([python, "-I", "-c", probe_beta], output, env, 120)
    # Copy tests, not the Python package: sys.path cannot select checkout code.
    shutil.copytree(source / "tests", output / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        source / "benchmarks",
        output / "benchmarks",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(source / "pytest.ini", output / "pytest.ini")
    run(
        [
            python,
            "-m",
            "pytest",
            "--require-framework",
            "--junitxml=installed-tests.xml",
            "tests",
        ],
        output,
        env,
    )
    if args.runtime_ranks:
        run(
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node",
                args.runtime_ranks,
                output / "tests/runtime/device_probe.py",
            ],
            output,
            env,
            600,
        )
    if args.ep_roundtrip:
        run(
            [
                python,
                source / "scripts/validate_ep.py",
                "--backend",
                args.ep_roundtrip,
                "--output",
                output / "ep-roundtrip",
            ],
            output,
            env,
            3700,
        )
    result = {
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "build_info": info,
        "runtime_ranks": args.runtime_ranks,
        "framework_dependencies_reused": args.system_site_packages,
        "ep_backend": args.ep_roundtrip,
        "npu_tensor_roundtrip_tested": args.ep_roundtrip is not None,
        "ep_communication_tested": args.ep_roundtrip == "native",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
