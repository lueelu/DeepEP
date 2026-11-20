# Copyright (c) 2026, Lu Lu

"""Exercise SDK architecture selection with real CMake, without a CANN compiler."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")
pytestmark = pytest.mark.skipif(CMAKE is None, reason="CMake is required")
CONTAMINATED = (
    "_bin_sh: _root_miniconda3_envs_py313_zmz_lib_libtinfo.so.6: "
    "no version information available (required by _bin_sh)\nx86_64"
)


@pytest.mark.parametrize(
    "detected,override,valid",
    [
        ("x86_64", None, True),
        ("aarch64", None, True),
        (CONTAMINATED, "x86_64", True),
        (CONTAMINATED, "aarch64", True),
        (CONTAMINATED, None, False),
        ("x86_64", "", False),
        ("x86_64", CONTAMINATED, False),
        ("x86_64", "../x86_64", False),
    ],
)
def test_host_processor_validation(tmp_path, detected, override, valid):
    script = tmp_path / "architecture.cmake"
    script.write_text(
        f"set(CMAKE_SYSTEM_PROCESSOR [=[{detected}]=])\n"
        + (f"set(DEEPEP_HOST_PROCESSOR [=[{override}]=])\n" if override is not None else "")
        + f'include("{ROOT.as_posix()}/cmake/HostArchitecture.cmake")\n'
        + 'message(STATUS "SDK_ARCH=${DEEPEP_HOST_PROCESSOR}")\n',
        encoding="utf-8",
    )
    result = subprocess.run([CMAKE, "-P", str(script)], capture_output=True, encoding="utf-8", errors="replace")
    if valid:
        assert result.returncode == 0, result.stderr
        assert f"SDK_ARCH={override or detected}\n" in result.stdout
    else:
        assert result.returncode != 0
        assert "Invalid host processor for CANN SDK paths" in result.stderr


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_device_makefile_ignores_contaminated_processor(tmp_path, architecture):
    cann, shmem = tmp_path / "cann", tmp_path / "shmem"
    arch_dir = cann / f"{architecture}-linux"
    for path in (
        arch_dir / "lib64/libascendcl.so",
        arch_dir / "lib64/libruntime.so",
        arch_dir / "pkg_inc/runtime/rt.h",
        shmem / "lib/libshmem.so",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (shmem / "deepep-sdk.json").write_text(
        '{"build":{"source_patches":[],"target":"Ascend950","cann_version":"9.2","cann_timestamp":"fixture",'
        '"rdma_backend":"HNS_1825","hns_1825_pi_on_chip":1}}'
    )
    toolchain = tmp_path / "toolchain.cmake"
    toolchain.write_text(
        "set(CMAKE_SYSTEM_NAME Linux)\n"
        f"set(CMAKE_SYSTEM_PROCESSOR [=[{CONTAMINATED}]=])\n"
        f'set(CMAKE_CXX_COMPILER "{Path(CMAKE).as_posix()}")\n'
        "set(CMAKE_CXX_COMPILER_ID GNU)\n"
        "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n",
        encoding="utf-8",
    )
    directory = tmp_path / "build"
    command = [
        CMAKE,
        "-S",
        str(ROOT / "cmake/device"),
        "-B",
        str(directory),
        "-G",
        "MinGW Makefiles" if sys.platform == "win32" else "Unix Makefiles",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}",
        f"-DCMAKE_MAKE_PROGRAM={Path(CMAKE).as_posix()}",
        f"-DDEEPEP_SOURCE_DIR={ROOT.as_posix()}",
        f"-DASCEND_HOME_PATH={cann.as_posix()}",
        f"-DSHMEM_ROOT={shmem.as_posix()}",
        f"-DDEEPEP_HOST_PROCESSOR={architecture}",
    ]
    # Configure twice to cover reuse of a build tree with polluted system data.
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace")
        assert result.returncode == 0, result.stdout + result.stderr
        flags = (directory / "CMakeFiles/deepep_kernels.dir/flags.make").read_text(encoding="utf-8")
        includes = next(line for line in flags.splitlines() if line.startswith("CXX_INCLUDES =")).replace("\\", "/")
        assert f"{architecture}-linux/include" in includes
        assert f"{architecture}-linux/pkg_inc/runtime" in includes
        assert "libtinfo" not in flags
        cache = (directory / "CMakeCache.txt").read_text(encoding="utf-8")
        assert f"ASCENDCL_LIBRARY:FILEPATH={arch_dir.as_posix()}/lib64/libascendcl.so" in cache


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_shmem_hook_repairs_sdk_include_paths_after_project(tmp_path, monkeypatch, architecture):
    ninja = shutil.which("ninja")
    make = ninja or shutil.which("make")
    if make is None:
        pytest.skip("Ninja or Make is required")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("shmem_arch_build", ROOT / "scripts/build_shmem.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "shmem source"
    source.mkdir()
    cann = tmp_path / "cann"
    include = cann / f"{architecture}-linux/pkg_inc/runtime"
    (include / "runtime").mkdir(parents=True)
    (include / "runtime/kernel.h").write_text("// CANN Runtime header\n")
    (source / "probe.cpp").write_text('#include "runtime/kernel.h"\n')
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(FIXTURE NONE)\n"
        # Initialize the native platform before injecting the Linux diagnostic.
        f"set(CMAKE_SYSTEM_PROCESSOR [=[{CONTAMINATED}]=])\n"
        "project(SHMEM NONE)\n"
        f'set(ASCEND_HOME_PATH "{cann.as_posix()}")\n'
        # Match upstream's unquoted architecture-dependent include expression.
        "include_directories(${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime)\n"
        "get_directory_property(includes INCLUDE_DIRECTORIES)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/includes.txt" "${includes}")\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/platform.txt" "${CMAKE_SYSTEM_PROCESSOR}|${CMAKE_CROSSCOMPILING}")\n',
        encoding="utf-8",
    )

    def configure(directory):
        command = [
            CMAKE,
            "-S",
            str(source),
            "-B",
            str(directory),
            "-G",
            "Ninja" if ninja else "Unix Makefiles",
            f"-DCMAKE_MAKE_PROGRAM={Path(make).as_posix()}",
        ]
        result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace")
        assert result.returncode == 0, result.stdout + result.stderr
        return (directory / "includes.txt").read_text(encoding="utf-8")

    # Prove that the fixture reproduces the server's broken path without the hook.
    assert "libtinfo" in configure(source / "baseline")
    module.seed_cmake_cache(source, architecture, {"hns_1825_pi_on_chip": 1})
    directory = source / "build"
    for _ in range(2):
        command = configure(directory).replace("\\", "/")
        assert command == include.as_posix()
        assert "libtinfo" not in command
        assert (directory / "platform.txt").read_text() == f"{architecture}|FALSE"


@pytest.mark.parametrize("processor", ["", CONTAMINATED, "../x86_64", "x86_64;other"])
def test_shmem_hook_rejects_invalid_architecture_before_writing(tmp_path, monkeypatch, processor):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("shmem_arch_build", ROOT / "scripts/build_shmem.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Invalid host processor"):
        module.seed_cmake_cache(tmp_path, processor, {})
    assert not (tmp_path / "build").exists()
