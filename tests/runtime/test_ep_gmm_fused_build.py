# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Additional packaging coverage for source metadata and the optional component."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

from tests.runtime.test_build import ROOT, load


@pytest.fixture
def build(monkeypatch):
    module = load(ROOT / "build_support.py")
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    return module


@pytest.mark.parametrize("ep_gmm_fused", [None, "OFF", "ON"])
def test_source_metadata_location_follows_component_flag(tmp_path, monkeypatch, ep_gmm_fused):
    import shutil

    if ep_gmm_fused is None:
        monkeypatch.delenv("DEEPEP_BUILD_EP_GMM_FUSED", raising=False)
    else:
        monkeypatch.setenv("DEEPEP_BUILD_EP_GMM_FUSED", ep_gmm_fused)
    for name in ("setup.py", "build_support.py", "VERSION"):
        shutil.copy2(ROOT / name, tmp_path / name)
    packages = ["deep_ep"]
    if ep_gmm_fused == "ON":
        packages += [
            "experiments/ep_gmm_fused/python/deep_ep_experimental",
            "experiments/ep_gmm_fused/python/deep_ep_experimental/ep_gmm_fused",
        ]
    for name in packages:
        package = tmp_path / name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    result = subprocess.run(
        [sys.executable, "setup.py", "egg_info"], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    root_metadata = tmp_path / "ascend_deepep.egg-info/PKG-INFO"
    build_metadata = tmp_path / "build/metadata/ascend_deepep.egg-info/PKG-INFO"
    assert root_metadata.is_file() is (ep_gmm_fused != "ON")
    assert build_metadata.is_file() is (ep_gmm_fused == "ON")
    # The PEP 517 metadata directory must take precedence over our default.
    metadata = tmp_path / "pep517"
    metadata.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import prepare_metadata_for_build_wheel as f; f('pep517')"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert list(metadata.glob("*.dist-info/METADATA"))
    assert root_metadata.is_file() is (ep_gmm_fused != "ON")
    assert build_metadata.is_file() is (ep_gmm_fused == "ON")


@pytest.mark.parametrize(
    "ep_gmm_fused,max_jobs,cmake_jobs,expected_jobs",
    [
        (False, "128", None, "4"),
        (True, "8", None, "8"),
        (False, "128", "6", "6"),
        (True, "128", "6", "6"),
        (True, "128", None, None),
    ],
)
def test_packaging_uses_cmake_capabilities_and_removes_old_library(
    build, monkeypatch, tmp_path, ep_gmm_fused, max_jobs, cmake_jobs, expected_jobs
):
    from setuptools import Distribution, Extension

    # This CPU-only test mocks the build tools, including their discovery paths.
    pybind11_cmake_dir = str(tmp_path / "pybind11 cmake")
    pybind11 = ModuleType("pybind11")
    pybind11.get_cmake_dir = lambda: pybind11_cmake_dir
    monkeypatch.setitem(sys.modules, "pybind11", pybind11)

    configuration = {
        "source_sha256": "current",
        "source_version": "1",
        "cmake_path": "/tools/cmake",
        "compiler_path": "/tools/c++",
        "ASCEND_HOME_PATH": "/sdk/cann",
        "SHMEM_ROOT": "/sdk/shmem",
        "build_type": "RelWithDebInfo",
        "python_executable": "/private/python",
        "build_environment": {"CXXFLAGS": "-I/private/include"},
        "ep_gmm_fused": {"enabled": ep_gmm_fused, "lineinfo": False},
    }
    if ep_gmm_fused:
        configuration["ep_gmm_fused_paths"] = {"SHMEM_SOURCE_PATH": "/private/shmem"}
    monkeypatch.setattr(build, "build_configuration", lambda: configuration)
    monkeypatch.setenv("MAX_JOBS", max_jobs)
    if cmake_jobs is None:
        monkeypatch.delenv("CMAKE_BUILD_PARALLEL_LEVEL", raising=False)
    else:
        monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", cmake_jobs)
    ext = Extension("deep_ep._C", sources=[])
    command = build.CMakeBuild(Distribution({"ext_modules": [ext]}))
    command.ensure_finalized()
    command.build_temp = str(tmp_path / "temp")
    command.build_lib = str(tmp_path / "lib")
    destination = Path(command.get_ext_fullpath(ext.name)).parent
    destination.mkdir(parents=True)
    component = destination.parent / "deep_ep_experimental/ep_gmm_fused"
    (destination / "libdeepep_notify.so").write_bytes(b"old")
    obsolete_modules = [
        f"_{op}{suffix}" for op in ("notify", "dispatch", "combine") for suffix in (".py", "_layout.py")
    ]
    for name in obsolete_modules:
        (destination / name).write_text("old module\n")
    calls = []
    capabilities = {"capabilities": ["test-built-target"], "communication_operations": []}

    def run(argv):
        calls.append(list(map(str, argv)))
        if "-B" in argv:
            directory = Path(argv[argv.index("-B") + 1])
            (directory / "compiler-info.json").write_text('{"id":"test","version":"1"}')
            (directory / "capabilities.json").write_text(json.dumps(capabilities))
        if "--install" in argv:
            (destination / "libdeepep_kernels.so").write_bytes(b"new")
            if ep_gmm_fused:
                component.mkdir(parents=True)

    monkeypatch.setattr(build, "checked_run", run)
    if expected_jobs is None:
        with pytest.raises(RuntimeError, match="must be in"):
            command.build_extension(ext)
        assert not calls
        return
    command.build_extension(ext)
    assert "-DCMAKE_BUILD_TYPE=RelWithDebInfo" in calls[0]
    assert f"-DDEEPEP_DEVICE_JOBS={expected_jobs}" in calls[0]
    assert f"-Dpybind11_DIR={pybind11_cmake_dir}" in calls[0]
    assert f"-DDEEPEP_BUILD_EP_GMM_FUSED={'ON' if ep_gmm_fused else 'OFF'}" in calls[0]
    assert calls[1][-2:] == ["--parallel", expected_jobs]
    assert not (destination / "libdeepep_notify.so").exists()
    assert all(not (destination / name).exists() for name in obsolete_modules)
    info = json.loads((destination / "_build_info.json").read_text())
    assert info["capabilities"] == capabilities["capabilities"]
    assert info["source_sha256"] == "current"
    assert not {"cmake_path", "compiler_path", "python_executable", "build_environment"}.intersection(info)
    assert "ep_gmm_fused_paths" not in info
    assert info["ep_gmm_fused"]["enabled"] is ep_gmm_fused
    if ep_gmm_fused:
        assert not (component / "LICENSE").exists()
        assert (component / "NOTICE.md").is_file()
        upstream = json.loads((component / "upstream.json").read_text())
        component_info = json.loads((component / "_build_info.json").read_text())
        assert component_info["upstream"] == upstream["commit"]
        assert component_info["enabled"] is True


@pytest.mark.parametrize("ep_gmm_fused", [False, True])
@pytest.mark.parametrize("option", [None, "inplace", "editable_mode"])
def test_editable_restriction_only_applies_to_component(build, monkeypatch, ep_gmm_fused, option):
    from setuptools import Distribution, Extension

    monkeypatch.setenv("DEEPEP_BUILD_EP_GMM_FUSED", "ON" if ep_gmm_fused else "OFF")
    command = build.CMakeBuild(Distribution({"ext_modules": [Extension("deep_ep._C", sources=[])]}))
    command.ensure_finalized()
    if option:
        setattr(command, option, True)
    calls = []
    monkeypatch.setattr(build.build_ext, "run", lambda self: calls.append(self))
    if ep_gmm_fused and option:
        with pytest.raises(RuntimeError, match="EP_GMM_FUSED requires a wheel"):
            command.run()
        assert not calls
    else:
        command.run()
        assert calls == [command]


@pytest.mark.parametrize("ep_gmm_fused", [False, True])
@pytest.mark.parametrize("option", [None, "build_lib", "build_temp", "build_platlib"])
def test_build_directories_preserve_core_defaults_and_explicit_paths(build, monkeypatch, ep_gmm_fused, option):
    import setuptools

    monkeypatch.setenv("DEEPEP_BUILD_EP_GMM_FUSED", "ON" if ep_gmm_fused else "OFF")
    # Even an invalid experiment-only value must be ignored by the core build.
    monkeypatch.setenv("DEEPEP_LINEINFO", "ON" if ep_gmm_fused else "invalid")
    monkeypatch.setitem(sys.modules, "build_support", build)
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(ROOT / "setup.py"))
    distribution = setuptools.Distribution(captured)
    command = distribution.get_command_obj("build")
    baseline = build.build(distribution)
    if option:
        setattr(command, option, "explicit-output")
        setattr(baseline, option, "explicit-output")
    command.ensure_finalized()
    baseline.ensure_finalized()
    for name in ("build_lib", "build_temp"):
        expected = getattr(baseline, name)
        explicit = option == name or (name == "build_lib" and option == "build_platlib")
        if ep_gmm_fused and not explicit:
            expected = str(Path(expected).with_name(Path(expected).name + "-ep-gmm-fused-shared-lineinfo"))
        assert getattr(command, name) == expected


@pytest.fixture
def packaging_source(tmp_path):
    for name in ("setup.py", "build_support.py", "VERSION", "pyproject.toml"):
        shutil.copy2(ROOT / name, tmp_path / name)
    for directory in ("deep_ep", "experiments/ep_gmm_fused/python"):
        shutil.copytree(ROOT / directory, tmp_path / directory, ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def run_packaging(source, enabled, lineinfo, *arguments):
    env = dict(os.environ, DEEPEP_BUILD_EP_GMM_FUSED=enabled, DEEPEP_LINEINFO=lineinfo)
    return subprocess.run(
        [sys.executable, "setup.py", *arguments], cwd=source, env=env, capture_output=True, text=True, timeout=60
    )


def test_switching_component_and_lineinfo_produces_clean_wheels(packaging_source):
    # Exercise real setuptools staging and install_lib, without requiring CANN.
    for enabled, lineinfo in (("ON", "OFF"), ("ON", "ON"), ("OFF", "OFF"), ("ON", "OFF"), ("OFF", "OFF")):
        result = run_packaging(packaging_source, enabled, lineinfo, "build_py")
        assert result.returncode == 0, result.stderr
        result = run_packaging(packaging_source, enabled, lineinfo, "bdist_wheel", "--skip-build")
        assert result.returncode == 0, result.stderr
        with zipfile.ZipFile(next((packaging_source / "dist").glob("*.whl"))) as archive:
            names = archive.namelist()
        component = "deep_ep_experimental/ep_gmm_fused/"
        assert any(n.startswith(component) for n in names) is (enabled == "ON")
        assert not any(component in n and not n.startswith(component) for n in names)
        assert not any("ep-gmm-fused-shared" in n for n in names)


@pytest.mark.parametrize("prefix", ["ep-gmm-fused-shared", "ep-gmm-fused-shared-lineinfo", ""])
def test_old_or_explicitly_reused_staging_cannot_leak_into_core_wheel(packaging_source, prefix):
    result = run_packaging(packaging_source, "OFF", "OFF", "build_py")
    assert result.returncode == 0, result.stderr
    staging = next((packaging_source / "build").glob("lib*"))
    stale = staging / prefix / "deep_ep_experimental/ep_gmm_fused/_C.stale.so"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale native extension")
    result = run_packaging(packaging_source, "OFF", "OFF", "bdist_wheel", "--skip-build")
    assert result.returncode != 0
    assert "Stale EP_GMM_FUSED staging" in result.stderr
    assert stale.is_file()  # No implicit recursive deletion of user caches.
    assert not list((packaging_source / "dist").glob("*.whl"))


@pytest.fixture
def sdk_environment(build, monkeypatch, tmp_path):
    cann, sdk, source = tmp_path / "cann", tmp_path / "sdk", tmp_path / "source"
    files = {
        cann / "include/acl/acl.h": "acl",
        cann / "compiler/version.info": "Version=9.2.0\ntimestamp=test\n",
        cann / "tools/bisheng_compiler/bin/bisheng": "device compiler",
        sdk / "include/host/shmem_host_def.h": "header",
        sdk / "include/shmem.h": "device header",
        sdk / "src/device/gm2gm/shmemi_device_cc.h": "device implementation",
        sdk / "lib/libshmem.so": "library",
        source / "VERSION": "0.1.0",
        tmp_path / "compiler": "host compiler",
    }
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    lock["shmem"]["header_sha256_lf"] = {"host/shmem_host_def.h": hashlib.sha256(b"header").hexdigest()}
    (source / "dependencies.lock.json").write_text(json.dumps(lock))
    (sdk / "deepep-sdk.json").write_text(
        json.dumps(
            {
                "revision": lock["shmem"]["revision"],
                "sha256": {
                    path.relative_to(sdk).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files
                    if path.is_relative_to(sdk)
                },
            }
        )
    )
    monkeypatch.setattr(build, "ROOT", source)
    monkeypatch.setattr(build, "source_fingerprint", lambda: "test")
    monkeypatch.setattr(build, "find_host_compiler", lambda env: (str(tmp_path / "compiler"), b"test"))
    monkeypatch.setattr(build.shutil, "which", lambda *args, **kwargs: str(tmp_path / "cmake"))
    monkeypatch.setattr(build.subprocess, "check_output", lambda *args, **kwargs: "cmake version test")
    monkeypatch.setitem(sys.modules, "pybind11", SimpleNamespace(__version__=lock["python_build"]["pybind11"]))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="test",
            __file__=str(tmp_path / "torch/__init__.py"),
            _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            __version__="test",
            __file__=str(tmp_path / "torch_npu/__init__.py"),
        ),
    )
    return {"DEEPEP_NPU_ARCH": "Ascend950", "ASCEND_HOME_PATH": str(cann), "SHMEM_ROOT": str(sdk)}


@pytest.mark.parametrize(
    "ep_gmm_fused,lineinfo,explicit,expected",
    [
        (False, "ON", None, "Release"),
        (False, "invalid", None, "Release"),
        (True, "ON", None, "RelWithDebInfo"),
        (True, "ON", "Debug", "Debug"),
        (True, "OFF", None, "Release"),
        (True, "invalid", None, None),
    ],
)
def test_lineinfo_only_affects_component_builds(build, sdk_environment, ep_gmm_fused, lineinfo, explicit, expected):
    env = {**sdk_environment, "DEEPEP_BUILD_EP_GMM_FUSED": "ON" if ep_gmm_fused else "OFF", "DEEPEP_LINEINFO": lineinfo}
    if explicit:
        env["DEEPEP_BUILD_TYPE"] = explicit
    if expected is None:
        with pytest.raises(ValueError, match="DEEPEP_LINEINFO"):
            build.build_configuration(env)
    else:
        configuration = build.build_configuration(env)
        assert configuration["build_type"] == expected
        assert configuration["ep_gmm_fused"]["lineinfo"] is (ep_gmm_fused and lineinfo == "ON")


@pytest.mark.parametrize("component", ["ep_gmm_fused", "other", "nested"])
def test_disabled_component_check_leaves_other_experiments_alone(tmp_path, component):
    validator = load(ROOT / "scripts/validate_package.py")
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("deep_ep/_C.fake.so", b"test")
        archive.writestr("deep_ep/deepseek-deepep.LICENSE", b"test")
        archive.writestr("deep_ep/_build_info.json", json.dumps({"communication_operations": []}))
        archive.writestr("deep_ep_experimental/__init__.py", "")
        name = (
            "ep-gmm-fused-shared/deep_ep_experimental/ep_gmm_fused/__init__.py"
            if component == "nested"
            else f"deep_ep_experimental/{component}/__init__.py"
        )
        archive.writestr(name, "")
    if component == "ep_gmm_fused":
        with pytest.raises(RuntimeError, match="disabled EP_GMM_FUSED"):
            validator.inspect_wheel(wheel)
    elif component == "nested":
        with pytest.raises(RuntimeError, match="nested EP_GMM_FUSED"):
            validator.inspect_wheel(wheel)
    else:
        validator.inspect_wheel(wheel)
