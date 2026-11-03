"""Exercise the real parent/child CMake scheduling without a CANN compiler."""

from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def tool(name):
    executable = shutil.which(name)
    if executable:
        return executable
    suffix = ".exe" if os.name == "nt" else ""
    bundled = ROOT / ".cache/testdeps" / name / "data/bin" / (name + suffix)
    if bundled.is_file():
        return str(bundled)
    pytest.skip(f"{name} required for CPU CMake scheduling test")


@pytest.fixture
def project(tmp_path):
    cmake, ninja = tool("cmake"), tool("ninja")
    source = tmp_path / "checkout with spaces"
    child = source / "cmake/device"
    child.mkdir(parents=True)
    shutil.copyfile(ROOT / "cmake/Device.cmake", source / "cmake/Device.cmake")
    sdk = source / "sdk"
    sdk.mkdir()
    (sdk / "deepep-sdk.json").write_text("{}")
    (sdk / "shared.h").write_text("first")
    (child / "rules.cmake").write_text("# tracked child CMake input\n")
    (child / "CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.20)
project(fake_device LANGUAGES NONE)
include(rules.cmake)
file(APPEND "${CMAKE_BINARY_DIR}/configures.log" "configure\\n")
if(DEEPEP_LL_STAGE_TIMER STREQUAL "FAIL")
  message(FATAL_ERROR "deliberate configuration failure")
endif()
add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/libdeepep_kernels.so"
  COMMAND "${CMAKE_COMMAND}" -E copy "${SHMEM_ROOT}/shared.h"
    "${CMAKE_BINARY_DIR}/libdeepep_kernels.so"
  DEPENDS "${SHMEM_ROOT}/shared.h" VERBATIM)
add_custom_target(fake_library ALL DEPENDS "${CMAKE_BINARY_DIR}/libdeepep_kernels.so")
''')
    (source / "CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.20)
project(fake_parent LANGUAGES NONE)
set(ASCEND_HOME_PATH "${CMAKE_SOURCE_DIR}/sdk")
set(SHMEM_ROOT "${CMAKE_SOURCE_DIR}/sdk")
include(cmake/Device.cmake)
''')
    build = tmp_path / "build with spaces"

    def configure(timer="OFF", check=True):
        return subprocess.run(
            [cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
             f"-DCMAKE_MAKE_PROGRAM={ninja}", f"-DDEEPEP_LL_STAGE_TIMER={timer}"],
            cwd=tmp_path, check=check, capture_output=True, text=True,
        )

    def compile():
        return subprocess.run(
            [cmake, "--build", str(build), "--target", "device_build"],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        )

    configure()
    return source, build, configure, compile


def test_noop_build_skips_child_configure_and_preserves_artifact(project):
    source, build, configure, compile = project
    compile()
    artifact = build / "device/libdeepep_kernels.so"
    stamp = artifact.stat().st_mtime_ns
    compile()
    assert artifact.stat().st_mtime_ns == stamp
    assert (build / "device/configures.log").read_text().splitlines() == ["configure"]


def test_child_reconfiguration_and_dependency_build_are_not_skipped(project):
    source, build, configure, compile = project
    compile()
    rules = source / "cmake/device/rules.cmake"
    rules.write_text("# changed child configure input\n")
    compile()
    assert len((build / "device/configures.log").read_text().splitlines()) == 2
    (source / "sdk/shared.h").write_text("changed SDK header")
    compile()
    assert (build / "device/libdeepep_kernels.so").read_text() == "changed SDK header"
    assert len((build / "device/configures.log").read_text().splitlines()) == 2


def test_parent_options_and_manifest_trigger_child_configuration(project):
    source, build, configure, compile = project
    compile()
    configure("ON")
    compile()
    assert "DEEPEP_LL_STAGE_TIMER:UNINITIALIZED=ON" in (build / "device/CMakeCache.txt").read_text()
    (source / "sdk/deepep-sdk.json").write_text('{"identity": "changed"}')
    compile()
    assert len((build / "device/configures.log").read_text().splitlines()) == 3


def test_child_configuration_failure_stops_parent(project):
    source, build, configure, compile = project
    result = configure("FAIL", check=False)
    assert result.returncode != 0
    assert "Device CMake configuration failed" in result.stderr


def test_stage_timer_definition_is_scoped_to_ll_sources():
    text = (ROOT / "cmake/device/CMakeLists.txt").read_text()
    block = text.split("set(_ll_sources", 1)[1].split(")", 1)[0]
    assert '"${DEEPEP_SOURCE_DIR}/csrc/kernels/dispatch/low_latency.cpp"' in block
    assert '"${DEEPEP_SOURCE_DIR}/csrc/kernels/combine/low_latency.cpp"' in block
    assert "target_compile_definitions(deepep_kernel_ll_timer PRIVATE DEEPEP_LL_STAGE_TIMER=1)" in text
    assert "target_compile_definitions(deepep_kernels PRIVATE DEEPEP_LL_STAGE_TIMER" not in text


def test_ndr_build_excludes_ht_bindings_and_declares_ll_only():
    import json
    parent = (ROOT / "CMakeLists.txt").read_text()
    sources = parent.split("pybind11_add_module(_C ", 1)[1].split(")", 1)[0].split()
    assert sources == ["csrc/bindings/module.cpp", "csrc/runtime/session.cpp", "csrc/ops/low_latency.cpp"]
    device = (ROOT / "cmake/device/CMakeLists.txt").read_text()
    assert "deepep_kernel_common" not in device
    capabilities = json.loads(parent.split('capabilities.json" CONTENT [=[', 1)[1].split(']=]', 1)[0])
    assert "ll-only" in capabilities["capabilities"]
    assert "runtime-urma" not in capabilities["capabilities"]
    assert not any(name.startswith(("elastic-notify", "elastic-dispatch-expanded", "elastic-combine-expanded"))
                   for name in capabilities["capabilities"])
    bindings = (ROOT / "csrc/bindings/module.cpp").read_text()
    assert 'module.attr("high_throughput_enabled") = false' in bindings
    assert 'for (const char* operation : {"notify", "dispatch", "combine"})' in bindings
    assert 'module.def(operation, [](py::args, py::kwargs)' in bindings
    assert 'throw std::runtime_error("HT operations are unavailable in this LL-only NDR build")' in bindings
    assert "deepep::bind_low_latency(module)" in bindings
    for name in ("notify", "dispatch", "combine"):
        assert f"deepep::bind_{name}(module)" not in bindings


def test_real_device_rules_recompile_only_ll_when_timer_changes(tmp_path):
    """Use the actual device CMake file and an output-producing fake GNU tool.

    This tests generated dependency/command rules, not Bisheng correctness.
    """
    cmake, ninja = tool("cmake"), tool("ninja")
    source = tmp_path / "checkout with spaces"
    child = source / "cmake/device"
    child.mkdir(parents=True)
    shutil.copyfile(ROOT / "cmake/device/CMakeLists.txt", child / "CMakeLists.txt")
    shutil.copyfile(ROOT / "cmake/Dependencies.cmake", source / "cmake/Dependencies.cmake")
    units = ["dispatch/low_latency.cpp", "combine/low_latency.cpp"]
    for unit in units:
        path = source / "csrc/kernels" / unit
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fake translation unit\n")
    sdk = source / "sdk"
    for path in [sdk / "lib64/libascendcl.so", sdk / "lib64/libruntime.so", sdk / "lib/libshmem.so"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake SDK library")
    (sdk / "deepep-sdk.json").write_text(
        '{"build":{"rdma_backend":"HNS_1825","hns_1825_pi_on_chip":0,"cann_version":"test","cann_timestamp":"test","target":"Ascend950","source_patches":[]}}')
    shared_header = sdk / "shared.h"
    shared_header.write_text("// fake dependency used by LL dispatch\n")
    compiler = tmp_path / "fake_compiler.py"
    compiler.write_text('''
import json
from pathlib import Path
import sys
args = sys.argv[1:]
if "-o" in args:
    output = Path(args[args.index("-o") + 1])
    if str(output) == "-":
        sys.exit(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("fake compiler artifact")
    unit = next((arg for arg in args if arg.endswith(".cpp")), "link")
    with (Path(__file__).parent / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(unit) + "\\n")
    with (Path(__file__).parent / "command_args.jsonl").open("a") as stream:
        stream.write(json.dumps(args) + "\\n")
    if "-MF" in args:
        def escape(value):
            return str(value).replace("\\\\", "/").replace(" ", "\\\\ ").replace(":", "\\\\:")
        dependencies = [unit]
        if unit.replace("\\\\", "/").endswith("/dispatch/low_latency.cpp"):
            dependencies.append(Path(__file__).parent / "checkout with spaces/sdk/shared.h")
        Path(args[args.index("-MF") + 1]).write_text(
            escape(output) + ": " + " ".join(escape(item) for item in dependencies) + "\\n")
''')
    build = tmp_path / "device build"
    commands = tmp_path / "commands.jsonl"

    def run(timer):
        subprocess.run(
            [cmake, "-S", str(child), "-B", str(build), "-G", "Ninja",
             f"-DCMAKE_MAKE_PROGRAM={ninja}", f"-DCMAKE_CXX_COMPILER={sys.executable}",
             f'-DCMAKE_CXX_COMPILER_ARG1={compiler.as_posix()}',
             "-DCMAKE_CXX_COMPILER_ID=GNU", "-DCMAKE_CXX_COMPILER_FORCED=TRUE",
             "-DCMAKE_CXX_COMPILER_VERSION=12.0", "-DCMAKE_SYSTEM_NAME=Linux",
             "-DCMAKE_BUILD_TYPE=Release", f"-DDEEPEP_SOURCE_DIR={source}",
             f"-DASCEND_HOME_PATH={sdk}", f"-DSHMEM_ROOT={sdk}",
             f"-DDEEPEP_LL_STAGE_TIMER={timer}"],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        )
        before = commands.read_text().splitlines() if commands.exists() else []
        subprocess.run([cmake, "--build", str(build)], cwd=tmp_path, check=True, capture_output=True, text=True)
        import json
        return [json.loads(line) for line in commands.read_text().splitlines()[len(before):]]

    initial = run("OFF")
    assert len(initial) == 3  # Two LL objects plus the shared library; no HT sources.
    changed = run("ON")
    assert len(changed) == 3
    expected_ll = {"dispatch/low_latency.cpp", "combine/low_latency.cpp"}
    assert {Path(unit).relative_to(source / "csrc/kernels").as_posix()
            for unit in changed if unit != "link"} == expected_ll
    for timer in ["OFF", "ON", "OFF"]:
        assert run(timer) == ["link"]
    assert run("OFF") == []
    import json
    for line in (tmp_path / "command_args.jsonl").read_text().splitlines():
        args = json.loads(line)
        if "-c" in args:
            output = args[args.index("-o") + 1]
            assert ("-DDEEPEP_LL_STAGE_TIMER=1" in args) == ("deepep_kernel_ll_timer.dir" in output)
            assert "-fPIC" in args
            assert "-Ddeepep_kernels_EXPORTS" in args
            assert "-DACLSHMEMI_HNS_1825_PI_ON_CHIP=0" in args
    shared_header.write_text("// changed SDK dependency\n")
    changed = run("OFF")
    assert len(changed) == 2
    assert any(unit.endswith("/dispatch/low_latency.cpp") for unit in changed)
    assert "link" in changed
    (source / "csrc/kernels/combine/low_latency.cpp").write_text("// changed LL source\n")
    changed = run("OFF")
    assert len(changed) == 2
    assert any(unit.endswith("/combine/low_latency.cpp") for unit in changed)
    assert "link" in changed
    # The inactive timer variant must catch both edits when selected again.
    changed = run("ON")
    assert len(changed) == 3
    assert sorted(Path(unit).parent.name for unit in changed if unit != "link") == ["combine", "dispatch"]
    assert run("OFF") == ["link"]
    assert run("OFF") == []
    # SDK and inline LL headers must select the same doorbell mode. Changing
    # that recorded mode changes real compiler commands for every device unit.
    identity_path = sdk / "deepep-sdk.json"
    identity = json.loads(identity_path.read_text())
    identity["build"]["hns_1825_pi_on_chip"] = 1
    identity_path.write_text(json.dumps(identity))
    previous_commands = len((tmp_path / "command_args.jsonl").read_text().splitlines())
    assert len(run("ON")) == 3
    for line in (tmp_path / "command_args.jsonl").read_text().splitlines()[previous_commands:]:
        args = json.loads(line)
        if "-c" in args:
            assert "-DACLSHMEMI_HNS_1825_PI_ON_CHIP=1" in args
            assert "-DACLSHMEMI_HNS_1825_PI_ON_CHIP=0" not in args
