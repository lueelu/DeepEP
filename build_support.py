# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""One native configure/build/install path; no network or device side effects."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import warnings

from setuptools.command.build_ext import build_ext
from setuptools.command.build import build
from setuptools.command.egg_info import egg_info
from setuptools.command.install_lib import install_lib

ROOT = Path(__file__).resolve().parent

# Runtime closure of the pinned SDK; CANN and driver libraries remain external.
SHMEM_RUNTIME_LIBRARIES = (
    "libshmem_utils.so",
    "aclshmem_bootstrap_config_store.so",
    "aclshmem_bootstrap_uid.so",
    "libshmem.so",
)


def bundle_shmem_runtime(sdk, destination):
    """Copy SDK runtime files, following SDK links into a self-contained wheel."""
    sources = [Path(sdk) / "lib" / name for name in SHMEM_RUNTIME_LIBRARIES]
    for path in sources:
        if not path.is_file():
            raise RuntimeError(f"Incomplete SHMEM runtime: missing {path}")
    directory = destination / ".libs"
    if directory.is_symlink():
        raise RuntimeError(f"Refusing to replace linked SHMEM staging: {directory}")
    # This directory belongs to build staging; discard stale SDK payloads.
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()
    hashes = {}
    for source in sources:
        target = directory / source.name
        shutil.copy2(source, target)
        hashes[source.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


class BuildEggInfo(egg_info):
    def finalize_options(self):
        # Source metadata must not shadow the installed distribution in pip.
        # Preserve an explicit --egg-base from the PEP 517 metadata hook.
        if self.egg_base is None:
            self.egg_base = "build/metadata"
            Path(self.egg_base).mkdir(parents=True, exist_ok=True)
        super().finalize_options()


def enabled(name, environment=None):
    env = os.environ if environment is None else environment
    value = env.get(name, "OFF").upper()
    if value not in {"ON", "OFF", "1", "0", "TRUE", "FALSE"}:
        raise ValueError(f"{name} must be ON or OFF, got {value!r}")
    return value in {"ON", "1", "TRUE"}


class ConfiguredBuild(build):
    def finalize_options(self):
        default_lib = (
            self.build_lib is None
            and (self.build_platlib if self.distribution.has_ext_modules() else self.build_purelib) is None
        )
        default_temp = self.build_temp is None
        super().finalize_options()
        if not enabled("DEEPEP_BUILD_EP_GMM_FUSED"):
            return
        # Registered only for the experiment; keep its default staging separate
        # from the core build while respecting explicit output directories.
        variant = "ep-gmm-fused-shared"
        if enabled("DEEPEP_LINEINFO"):
            variant += "-lineinfo"
        if default_lib:
            self.build_lib = str(Path(self.build_lib).with_name(Path(self.build_lib).name + "-" + variant))
        if default_temp:
            self.build_temp = str(Path(self.build_temp).with_name(Path(self.build_temp).name + "-" + variant))


class CheckedInstallLib(install_lib):
    def install(self):
        # install_lib recursively copies staging, including stale native files
        # that build_py does not know about. Check even with bdist_wheel --skip-build.
        component_enabled = enabled("DEEPEP_BUILD_EP_GMM_FUSED")
        for directory in (self.build_dir, self.install_dir):
            root = Path(directory)
            # Renamed internal implementations must not survive in a reused
            # wheel staging directory alongside the canonical topology packages.
            for old in ("backends/clos", "backends/pod", "_clos_adapter.py", "_card_adapter.py"):
                obsolete = root / "deep_ep" / old
                files = obsolete.rglob("*") if obsolete.is_dir() else (obsolete,)
                if any(path.is_file() and "__pycache__" not in path.parts for path in files):
                    raise RuntimeError(
                        f"Obsolete backend implementation at {obsolete}. "
                        "Use a clean build/install directory for the topology package layout."
                    )
            compatibility = root / "ascend_deepep"
            # Only ship the forwarding files from this source tree, never an
            # old implementation or its native libraries left in build staging.
            for path in compatibility.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                source = ROOT / "ascend_deepep" / path.relative_to(compatibility)
                if not source.is_file() or path.read_bytes() != source.read_bytes():
                    raise RuntimeError(
                        f"Obsolete ascend_deepep package file at {path}. "
                        "Use a clean build/install directory; only the current compatibility aliases are supported."
                    )
            for component in root.glob("**/deep_ep_experimental/ep_gmm_fused"):
                relative = component.relative_to(root).as_posix()
                if not component_enabled or relative != "deep_ep_experimental/ep_gmm_fused":
                    raise RuntimeError(
                        f"Stale EP_GMM_FUSED staging at {component}. "
                        f"Move {root} outside the build tree and rebuild; "
                        "use separate --build-lib paths for different component configurations."
                    )
        build_root = Path(self.build_dir)
        install_root = Path(self.install_dir)
        build_info_path = build_root / "deep_ep" / "_build_info.json"
        build_info = json.loads(build_info_path.read_text(encoding="utf-8")) if build_info_path.is_file() else None
        if build_info is not None and install_root.resolve() != build_root.resolve():
            # Remove the old optional extension and header tree from a reused
            # install tree before copying the current build.
            package = install_root / "deep_ep"
            for path in package.glob("_jit_C*"):
                path.unlink()
            kernel_src = package / "_kernel_src"
            if kernel_src.is_symlink():
                kernel_src.unlink()
            elif kernel_src.exists():
                shutil.rmtree(kernel_src)
        return super().install()


def source_fingerprint():
    files = [
        ROOT / name
        for name in (
            "VERSION",
            "setup.py",
            "pyproject.toml",
            "build_support.py",
            "CMakeLists.txt",
            "dependencies.lock.json",
            "MANIFEST.in",
            "requirements-dev.txt",
            "requirements-build.txt",
            ".gitmodules",
            "experiments/ep_gmm_fused/tests/requirements.txt",
            "third-party/deepseek-deepep.LICENSE",
            "third-party/cann.LICENSE",
        )
    ]
    for name in (
        "deep_ep",
        "csrc",
        "cmake",
        "scripts",
        "tests",
        "benchmarks",
        "experiments/megamoe",
        "experiments/ep_gmm_fused",
        "third-party/deep_jit/include",
        "third-party/fmt/include",
    ):
        files.extend(
            path
            for path in (ROOT / name).rglob("*")
            if (path.suffix in {".sh", ".py", ".cpp", ".hpp", ".h", ".cmake", ".json"} or path.name == "CMakeLists.txt")
            and path.name != "_build_info.json"
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def checked_run(argv, **kwargs):
    subprocess.run([str(arg) for arg in argv], check=True, timeout=1800, **kwargs)


def bundle_jit_shmem_headers(sdk, destination):
    """Package the SHMEM Device headers needed by an offline JIT compile."""
    source = Path(sdk)
    target = Path(destination) / "shmem"
    copied = 0
    for tree in ("include", "src"):
        root = source / tree
        if not root.is_dir():
            raise RuntimeError(f"SHMEM SDK is missing the JIT header tree: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".inc"}:
                continue
            relative = path.relative_to(source)
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            copied += 1
    if copied == 0:
        raise RuntimeError(f"SHMEM SDK has no JIT headers under {source}")
    manifest = source / "deepep-sdk.json"
    if manifest.is_file():
        shutil.copy2(manifest, target / manifest.name)
    return target


def bundle_megamoe_headers(destination, catlass_root=None):
    """Copy experiment/CATLASS/Tensor API headers used by the JIT source."""
    target = Path(destination) / "megamoe"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(ROOT / "experiments/megamoe/csrc", target / "csrc")
    catlass = Path(catlass_root or os.environ.get("CATLASS_ROOT", ROOT / "third-party/catlass"))
    if (catlass / "include").is_dir():
        shutil.copytree(catlass / "include", target / "catlass/include")
    if (catlass / "examples/common").is_dir():
        shutil.copytree(catlass / "examples/common", target / "catlass/examples/common")
    common = os.environ.get("ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR", "")
    candidates = [
        common,
        str(Path(os.environ.get("ASCEND_HOME_PATH", "")) / "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common"),
        str(Path(os.environ.get("ASCEND_HOME_PATH", "")) / "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "tensor_api").is_dir():
            shutil.copytree(Path(candidate) / "tensor_api", target / "tensor_api")
            break
    else:
        raise RuntimeError(
            "MegaMoE JIT requires the CANN Tensor API headers; set "
            "ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR to the common package root"
        )
    return target


def validate_shmem_sdk(root, expected):
    manifest = Path(root) / "deepep-sdk.json"
    if not manifest.is_file():
        raise RuntimeError("SHMEM SDK needs deepep-sdk.json; see scripts/record_shmem_sdk.py")
    sdk = json.loads(manifest.read_text(encoding="utf-8"))
    if sdk.get("revision") != expected["revision"]:
        raise RuntimeError("SHMEM SDK source revision differs from dependencies.lock.json")
    for relative, digest in expected["header_sha256_lf"].items():
        data = (manifest.parent / "include" / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"SHMEM public API differs from the pinned SDK: {relative}")
    for relative, digest in sdk["sha256"].items():
        path = (manifest.parent / relative).resolve()
        if not path.is_relative_to(manifest.parent.resolve()) or not path.is_file():
            raise RuntimeError("Invalid SHMEM SDK manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"SHMEM SDK checksum mismatch: {relative}")
    if "lib/libshmem.so" not in sdk["sha256"] or not any(key.startswith("include/") for key in sdk["sha256"]):
        raise RuntimeError("SHMEM SDK manifest must cover headers and libshmem.so")
    return sdk


def build_cache_key(configuration):
    """Keep object files across source edits; isolate incompatible toolchains."""
    identity = {key: value for key, value in configuration.items() if key not in {"source_sha256", "source_version"}}
    identity["source_root"] = str(ROOT.resolve())
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def find_host_compiler(env):
    """Select an executable without resolving compiler-wrapper symlinks."""
    explicit = env.get("CXX")
    candidates = [explicit] if explicit else ["c++", "g++", "clang++", "bisheng++", "bisheng"]
    if not explicit and env.get("ASCEND_HOME_PATH"):
        directory = Path(env["ASCEND_HOME_PATH"]) / "tools/bisheng_compiler/bin"
        candidates.extend(str(directory / name) for name in ("bisheng++", "bisheng"))
    failures = []
    for name in candidates:
        compiler = shutil.which(name, path=env.get("PATH"))
        if compiler is None:
            failures.append(f"{name}: not found")
            continue
        # c++ -> ccache is valid; selecting bare ccache is not a compiler.
        if Path(compiler).name in {"ccache", "sccache"}:
            failures.append(f"{name}: select a C++ compiler, not a standalone cache launcher")
            continue
        compiler = str(Path(compiler).absolute())
        try:
            version = subprocess.check_output([compiler, "--version"], stderr=subprocess.STDOUT, timeout=30)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{name}: --version failed ({type(exc).__name__})")
            continue
        return compiler, version
    prefix = "Explicit CXX is unusable" if explicit else "No runnable Host C++ compiler found"
    raise RuntimeError(f"{prefix}. Set CXX to a C++20 compiler executable. Tried: " + "; ".join(failures))


def build_configuration(environment=None):
    env = os.environ if environment is None else environment
    if platform.system() != "Linux":
        raise RuntimeError("Native wheels require Linux with an Ascend SDK")
    if env.get("DEEPEP_NPU_ARCH") != "Ascend950":
        raise RuntimeError("DEEPEP_NPU_ARCH must be Ascend950; other targets are unqualified")
    beta = env.get("DEEPEP_BUILD_MEGAMOE", "OFF")
    timer = env.get("DEEPEP_MEGAMOE_TIMER", "OFF")
    if beta not in {"ON", "OFF"} or timer not in {"ON", "OFF"}:
        raise RuntimeError("MegaMoE build options accept ON or OFF only")
    if timer == "ON" and beta != "ON":
        raise RuntimeError("MegaMoE timer requires DEEPEP_BUILD_MEGAMOE=ON")
    ep_gmm_fused = enabled("DEEPEP_BUILD_EP_GMM_FUSED", env)
    lineinfo = ep_gmm_fused and enabled("DEEPEP_LINEINFO", env)
    result = {"arch": "Ascend950", "python": sys.version, "machine": platform.machine()}
    default_shmem = ROOT / "third-party/shmem/install/shmem"

    result["build_type"] = env.get("DEEPEP_BUILD_TYPE", "RelWithDebInfo" if lineinfo else "Release")
    if result["build_type"] not in {"Release", "RelWithDebInfo", "Debug"}:
        raise RuntimeError("DEEPEP_BUILD_TYPE must be Release, RelWithDebInfo or Debug")
    for name, header in (
        ("ASCEND_HOME_PATH", "include/acl/acl.h"),
        ("SHMEM_ROOT", "include/host/shmem_host_def.h"),
    ):
        value = env.get(name, str(default_shmem) if name == "SHMEM_ROOT" else None)
        if not value or not (Path(value) / header).is_file():
            hint = "; for bundled SHMEM run python scripts/build_shmem.py first" if name == "SHMEM_ROOT" else ""
            raise RuntimeError(f"{name} must name an installed SDK containing {header}{hint}")
        result[name] = str(Path(value).resolve())
    manifest = Path(result["SHMEM_ROOT"]) / "deepep-sdk.json"
    lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
    sdk = {}
    try:
        if not manifest.is_file():
            raise RuntimeError("SHMEM SDK has no deepep-sdk.json")
        sdk = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(sdk, dict):
            raise ValueError("Invalid SHMEM SDK manifest")
        if sdk.get("revision") != lock["shmem"]["revision"]:
            raise RuntimeError("SHMEM SDK source revision differs from dependencies.lock.json")
        for relative, expected in lock["shmem"]["header_sha256_lf"].items():
            data = (manifest.parent / "include" / relative).read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"SHMEM public API differs from the pinned SDK: {relative}")
        hashes = sdk.get("sha256")
        if not isinstance(hashes, dict):
            raise ValueError("Invalid SHMEM SDK manifest checksums")
        for relative, expected in hashes.items():
            path = (manifest.parent / relative).resolve()
            if not path.is_relative_to(manifest.parent.resolve()) or not path.is_file():
                raise RuntimeError("Invalid SHMEM SDK manifest path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"SHMEM SDK checksum mismatch: {relative}")
        if "lib/libshmem.so" not in hashes or not any(key.startswith("include/") for key in hashes):
            raise RuntimeError("SHMEM SDK manifest must cover headers and libshmem.so")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        if beta == "ON":
            raise
        warnings.warn(f"{exc}; continuing with SHMEM_ROOT={manifest.parent}", RuntimeWarning, stacklevel=2)
    library = manifest.parent / "lib/libshmem.so"
    if not library.is_file():
        raise RuntimeError(f"SHMEM SDK is missing {library}")
    files = sorted(
        path for path in (manifest.parent / "include").rglob("*") if path.suffix in {".h", ".hpp"} and path.is_file()
    )
    files.append(library)
    # Record actual inputs so replacing an unpinned SDK invalidates the build cache.
    result["shmem"] = {
        "revision": sdk.get("revision", "unknown") if isinstance(sdk, dict) else "unknown",
        "sha256": {
            path.relative_to(manifest.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    result["pybind11"] = __import__("pybind11").__version__
    if result["pybind11"] != lock["python_build"]["pybind11"]:
        raise RuntimeError("pybind11 version differs from dependencies.lock.json")
    version_file = Path(result["ASCEND_HOME_PATH"]) / "compiler/version.info"
    versions = dict(line.split("=", 1) for line in version_file.read_text().splitlines() if "=" in line)
    actual = versions.get("Version", "")
    release = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:[.+-].*)?", actual)
    minimum = tuple(int(part) for part in lock["cann_version"].split("."))
    if release is None or tuple(int(part or 0) for part in release.groups()) < minimum:
        raise RuntimeError(f"This runtime build requires CANN >= {lock['cann_version']}; detected {actual!r}")
    result["cann"] = {
        "version": versions["Version"],
        "timestamp": versions.get("timestamp", "unknown"),
    }
    result["source_version"] = (ROOT / "VERSION").read_text().strip()
    result["source_sha256"] = source_fingerprint()
    import torch
    import torch_npu

    result["direct"] = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "topk_idx_bits": 64,
    }
    result["ll_jit"] = {"enabled": True}
    # DeepJIT is mandatory, like DeepEP-Ascend. The main _C module compiles and
    # links against torch_npu and the pinned DeepJIT headers.
    result["TORCH_NPU_PATH"] = str(Path(torch_npu.__file__).parent.resolve())
    result["Torch_DIR"] = str(Path(torch.__file__).parent / "share/cmake/Torch")
    deepjit = Path(env.get("DEEPJIT_ROOT", ROOT / "third-party/deep_jit")).resolve()
    if not (deepjit / "include/deep_jit").is_dir():
        raise RuntimeError(
            "DEEPJIT_ROOT must contain include/deep_jit (run git submodule update --init --recursive "
            "or unpack the sdist)"
        )
    result["DEEPJIT_ROOT"] = str(deepjit)
    if (deepjit / ".git").exists():
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(deepjit), "rev-parse", "HEAD"], text=True, timeout=30
            ).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Cannot identify the locked DeepJIT checkout revision") from exc
        if revision != lock["deepjit"]["revision"]:
            raise RuntimeError("DeepJIT revision differs from dependencies.lock.json")
    fmt_root = env.get("DEEPEP_DEEPJIT_FMT_ROOT", "")
    if fmt_root:
        fmt = Path(fmt_root).resolve()
        if not (fmt / "fmt/format.h").is_file():
            raise RuntimeError("DEEPEP_DEEPJIT_FMT_ROOT must contain fmt/format.h")
        result["DEEPEP_DEEPJIT_FMT_ROOT"] = str(fmt)
    compiler, compiler_version = find_host_compiler(env)
    result["compiler_path"] = compiler
    result["megamoe"] = {"enabled": beta == "ON", "timer": timer == "ON"}
    if beta == "ON":
        catlass = Path(env.get("CATLASS_ROOT", str(ROOT / "third-party/catlass")))
        if not (catlass / "include/catlass/gemm/tile/copy_l1_to_l0a.hpp").is_file():
            raise RuntimeError(
                "MegaMoE requires CATLASS: run git submodule update --init --recursive, "
                "or set CATLASS_ROOT; see experiments/megamoe/README.md"
            )
        revision = subprocess.check_output(["git", "-C", str(catlass), "rev-parse", "HEAD"], text=True).strip()
        if revision != lock["megamoe"]["catlass_revision"]:
            raise RuntimeError("MegaMoE CATLASS revision differs from dependencies.lock.json")
        import torch
        import torch_npu

        result["CATLASS_ROOT"] = str(catlass.resolve())
        result["megamoe"].update(torch=torch.__version__, torch_npu=torch_npu.__version__, catlass=revision)
    result["compiler_sha256"] = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
    result["compiler_version_sha256"] = hashlib.sha256(compiler_version).hexdigest()
    print(f"Host C++ compiler: {compiler}", flush=True)
    cmake = shutil.which("cmake", path=env.get("PATH"))
    if cmake is None:
        raise RuntimeError("CMake must be installed and available on PATH")
    result["cmake_path"] = str(Path(cmake).resolve())
    result["cmake_version"] = subprocess.check_output([cmake, "--version"], text=True, timeout=30).splitlines()[0]
    # Keep the virtualenv path even when its executable symlinks to system Python.
    result["python_executable"] = str(Path(sys.executable).absolute())
    result["build_environment"] = {
        name: env.get(name, "")
        for name in (
            "CXXFLAGS",
            "LDFLAGS",
            "CPPFLAGS",
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "LIBRARY_PATH",
            "CMAKE_GENERATOR",
            "CMAKE_GENERATOR_PLATFORM",
            "CMAKE_GENERATOR_TOOLSET",
            "CMAKE_PREFIX_PATH",
        )
    }
    device_compiler = Path(result["ASCEND_HOME_PATH"]) / "tools/bisheng_compiler/bin/bisheng"
    if not device_compiler.is_file():
        raise RuntimeError("Device kernels require CANN Bisheng with -xasc support")
    result["device_compiler_sha256"] = hashlib.sha256(device_compiler.read_bytes()).hexdigest()
    sdk_root = Path(result["SHMEM_ROOT"])
    for relative in ("include/shmem.h", "src/device/gm2gm/shmemi_device_cc.h"):
        if not (sdk_root / relative).is_file():
            raise RuntimeError(f"Kernels require the full installed SHMEM Device SDK: missing {relative}")
    result["shmem_device_headers"] = {
        p.relative_to(sdk_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((sdk_root / "src").rglob("*"))
        if p.is_file() and p.suffix in {".h", ".hpp"}
    }
    result["ep_gmm_fused"] = {
        "enabled": ep_gmm_fused,
        "lineinfo": lineinfo,
    }
    if result["ep_gmm_fused"]["enabled"]:
        import torch
        import torch_npu

        result["ep_gmm_fused"].update(
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            torch_cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            shmem_revision=lock["shmem"]["revision"],
        )
        source = Path(env.get("SHMEM_SOURCE_PATH", ROOT / "third-party/shmem"))
        result["ep_gmm_fused_paths"] = {
            "CATCCOS_SOURCE_PATH": str(Path(env.get("CATCCOS_SOURCE_PATH", ROOT / "third-party/catccos")).resolve()),
            "SHMEM_SOURCE_PATH": str(source.resolve()),
            "Torch_DIR": str(Path(torch.__file__).parent / "share/cmake/Torch"),
            "TORCH_NPU_PATH": str(Path(torch_npu.__file__).parent),
        }
    return result


class VariantBuild(ConfiguredBuild):
    def finalize_options(self):
        # 隔离 ON/OFF 产物，避免连续构建把旧 Beta 模块打入基础 wheel。
        beta = os.environ.get("DEEPEP_BUILD_MEGAMOE", "OFF") == "ON"
        timer = os.environ.get("DEEPEP_MEGAMOE_TIMER", "OFF") == "ON"
        variant = "megamoe-timer" if beta and timer else ("megamoe" if beta else "core")
        self.build_base = str(Path(self.build_base) / variant)
        super().finalize_options()


def publish_source_native(destination):
    """Expose a complete build to imports from the checkout, without mixing wheels.

    Keep a private, persistent copy: setuptools staging can be temporary or be
    overwritten by a later build. Switch all existing links through one anchor.
    This does not run when installing a prebuilt wheel.
    """
    package = ROOT / "deep_ep"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    names = [
        "_C" + suffix,
        "_clos_C" + suffix,
        "libdeepep_jit.so",
        "_build_info.json",
        "_kernel_src",
    ]
    for name in names:
        path = destination / name
        if name == "_kernel_src":
            if not (path / "csrc").is_dir():
                raise RuntimeError(f"Incomplete native build: {path}")
        elif not path.is_file():
            raise RuntimeError(f"Incomplete native build: {path}")
    for name in SHMEM_RUNTIME_LIBRARIES:
        if not (destination / ".libs" / name).is_file():
            raise RuntimeError(f"Incomplete native build: {destination / '.libs' / name}")
    names.append(".libs")
    directory_names = {".libs", "_kernel_src"}
    anchor = package / ".native-build"
    storage = ROOT / "build" / "source-native"
    if anchor.exists() and not anchor.is_symlink():
        raise RuntimeError(f"Refusing to replace non-link source artifact: {anchor}")
    if anchor.is_symlink() and anchor.resolve().parent != storage.resolve():
        raise RuntimeError(f"Refusing to replace unmanaged source anchor: {anchor}")
    managed = {}
    for path in package.iterdir():
        if path.is_symlink() and path.readlink() == Path(".native-build") / path.name:
            managed[path.name] = path
    for name in names:
        path = package / name
        if (path.exists() or path.is_symlink()) and name not in managed:
            raise RuntimeError(f"Refusing to replace unmanaged source artifact: {path}")
    storage.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix="build-", dir=storage)).resolve()
    for name in names:
        if name in directory_names:
            shutil.copytree(destination / name, snapshot / name)
        else:
            shutil.copy2(destination / name, snapshot / name)
    # Install new names before publishing the completed snapshot. Old names use
    # the previous snapshot until the anchor is atomically replaced.
    for name in names:
        if name not in managed:
            (package / name).symlink_to(Path(".native-build") / name, target_is_directory=name in directory_names)
    pending = snapshot / "anchor"
    pending.symlink_to(os.path.relpath(snapshot, package), target_is_directory=True)
    os.replace(pending, anchor)
    for name, path in managed.items():
        if name not in names:
            path.unlink()
    print(f"Prepared source native links in {package}")


class CMakeBuild(build_ext):
    def run(self):
        # Check before setuptools temporarily clears inplace during run().
        if enabled("DEEPEP_BUILD_EP_GMM_FUSED") and (self.inplace or self.editable_mode):
            raise RuntimeError("EP_GMM_FUSED requires a wheel; editable/inplace builds are not supported")
        self._publish_source_links = not (self.inplace or self.editable_mode)
        super().run()

    def build_extension(self, ext):
        import pybind11

        configuration = build_configuration()
        megamoe = configuration.get("megamoe", {})
        ep_gmm_fused = configuration.get("ep_gmm_fused", {}).get("enabled", False)
        fingerprint = build_cache_key(configuration)
        directory = Path(self.build_temp).resolve() / fingerprint
        destination = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        default_jobs = os.environ.get("MAX_JOBS", "4") if ep_gmm_fused else "4"
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", default_jobs)
        if not jobs.isdecimal() or not 1 <= int(jobs) <= 64:
            raise RuntimeError("CMAKE_BUILD_PARALLEL_LEVEL must be in [1, 64]")
        cmake = configuration["cmake_path"]
        directory.mkdir(parents=True, exist_ok=True)
        arguments = [
            cmake,
            "-S",
            ROOT,
            "-B",
            directory,
            f"-DCMAKE_BUILD_TYPE={configuration['build_type']}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DCMAKE_CXX_COMPILER={configuration['compiler_path']}",
            f"-DASCEND_HOME_PATH={configuration['ASCEND_HOME_PATH']}",
            f"-DSHMEM_ROOT={configuration['SHMEM_ROOT']}",
            f"-DDEEPEP_BUILD_MEGAMOE={'ON' if megamoe.get('enabled', False) else 'OFF'}",
            f"-DDEEPEP_MEGAMOE_TIMER={'ON' if megamoe.get('timer', False) else 'OFF'}",
            f"-DCATLASS_ROOT={configuration.get('CATLASS_ROOT', '')}",
            f"-DDEEPEP_BUILD_EP_GMM_FUSED={'ON' if ep_gmm_fused else 'OFF'}",
            f"-DCMAKE_INSTALL_PREFIX={destination}",
        ]
        arguments.extend(
            [
                f"-DTORCH_NPU_PATH={configuration['TORCH_NPU_PATH']}",
                f"-DTorch_DIR={configuration['Torch_DIR']}",
                f"-DDEEPJIT_ROOT={configuration['DEEPJIT_ROOT']}",
            ]
        )
        if configuration.get("DEEPEP_DEEPJIT_FMT_ROOT"):
            arguments.append(f"-DDEEPEP_DEEPJIT_FMT_ROOT={configuration['DEEPEP_DEEPJIT_FMT_ROOT']}")
        if ep_gmm_fused:
            arguments.extend(
                [
                    f"-DPython3_EXECUTABLE={sys.executable}",
                    f"-DDEEPEP_EXTENSION_SUFFIX={sysconfig.get_config_var('EXT_SUFFIX')}",
                    *(f"-D{name}={value}" for name, value in configuration["ep_gmm_fused_paths"].items()),
                ]
            )
        checked_run(arguments)
        checked_run([cmake, "--build", directory, "--parallel", jobs])
        # A reused build_lib may still contain the old optional extension from
        # an earlier wheel; JIT now lives in _C.
        for path in destination.glob("_jit_C*"):
            path.unlink()
        checked_run([cmake, "--install", directory])
        kernel_src = destination / "_kernel_src"
        # DeepJIT compiles the LL entries on first launch, so the installed
        # package carries the same device headers this build configured from.
        if kernel_src.is_symlink():
            kernel_src.unlink()
        elif kernel_src.exists():
            shutil.rmtree(kernel_src)
        shutil.copytree(ROOT / "csrc", kernel_src / "csrc")
        bundle_jit_shmem_headers(configuration["SHMEM_ROOT"], kernel_src)
        if megamoe.get("enabled", False):
            bundle_megamoe_headers(kernel_src, configuration.get("CATLASS_ROOT"))
        # A reused setuptools build_lib may still contain the former library.
        (destination / "libdeepep_notify.so").unlink(missing_ok=True)
        # Elastic implementation/layout modules now live under buffers/.
        for operation in ("notify", "dispatch", "combine"):
            for suffix in (".py", "_layout.py"):
                (destination / f"_{operation}{suffix}").unlink(missing_ok=True)
        # Deployment metadata contains identities/hashes, never build-machine paths.
        info = {
            key: value
            for key, value in configuration.items()
            if key
            not in {
                "ASCEND_HOME_PATH",
                "SHMEM_ROOT",
                "CATLASS_ROOT",
                "TORCH_NPU_PATH",
                "Torch_DIR",
                "DEEPJIT_ROOT",
                "DEEPEP_DEEPJIT_FMT_ROOT",
                "compiler_path",
                "ep_gmm_fused_paths",
                "cmake_path",
                "python_executable",
                "build_environment",
            }
        }
        info["compiler"] = json.loads((directory / "compiler-info.json").read_text())
        info.update(json.loads((directory / "capabilities.json").read_text(encoding="utf-8")))
        if "ll_jit" in info:
            deepjit_info = directory / "deepjit-info.json"
            if deepjit_info.is_file():
                info["ll_jit"].update(json.loads(deepjit_info.read_text(encoding="utf-8")))
        info["shmem_runtime"] = bundle_shmem_runtime(configuration["SHMEM_ROOT"], destination)
        (destination / "_build_info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if ep_gmm_fused:
            experiment = ROOT / "experiments/ep_gmm_fused"
            component = destination.parent / "deep_ep_experimental/ep_gmm_fused"
            for name in ("NOTICE.md", "upstream.json"):
                shutil.copy2(experiment / name, component / name)
            (component / "LICENSE").unlink(missing_ok=True)
            component_info = dict(
                configuration["ep_gmm_fused"],
                npu_arch="dav-3510",
                python=sys.version,
                platform=platform.platform(),
                upstream=json.loads((experiment / "upstream.json").read_text())["commit"],
            )
            (component / "_build_info.json").write_text(
                json.dumps(component_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        # Run only after every native artifact and its build identity is ready.
        if getattr(self, "_publish_source_links", not (self.inplace or self.editable_mode)):
            publish_source_native(destination)
