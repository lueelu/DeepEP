# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Artifact shape tests, not native binary compilation."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest
from setuptools import Distribution

ROOT = Path(__file__).resolve().parents[3]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_variants_have_disjoint_python_and_native_outputs(monkeypatch):
    build = load(ROOT / "build_support.py")
    outputs = []
    for beta, timer in (("OFF", "OFF"), ("ON", "OFF"), ("ON", "ON")):
        monkeypatch.setenv("DEEPEP_BUILD_MEGAMOE", beta)
        monkeypatch.setenv("DEEPEP_MEGAMOE_TIMER", timer)
        command = build.VariantBuild(Distribution({"ext_modules": []}))
        command.ensure_finalized()
        outputs.append((command.build_lib, command.build_temp))
    assert len({row[0] for row in outputs}) == 3
    assert len({row[1] for row in outputs}) == 3


def wheel(tmp_path, enabled, payload, ep_gmm_fused=False):
    entries = {
        "deep_ep/_C.fake.so": b"not-an-executable",
        "deep_ep/_build_info.json": json.dumps(
            {
                "communication_operations": [],
                "megamoe": {"enabled": enabled},
                "ep_gmm_fused": {"enabled": ep_gmm_fused},
                "shmem": {"revision": "test-sdk"},
            }
        ).encode(),
        "test.dist-info/licenses/deepseek-deepep.LICENSE": b"fixture",
    }
    if payload:
        for name in ("api.py", "benchmark.py", "configs/smoke.json", "libmegamoe_kernel.so", "_C.fake.so"):
            entries["deep_ep_experimental/megamoe/" + name] = b"fixture"
        entries["test.dist-info/licenses/cann.LICENSE"] = b"fixture"
    if ep_gmm_fused:
        prefix = "deep_ep_experimental/ep_gmm_fused/"
        for name in (
            "__init__.py",
            "_C.fake.so",
            "NOTICE.md",
            "upstream.json",
            "licenses/shmem/LICENSE",
            "licenses/catccos/LICENSE",
            "licenses/catlass/LICENSE",
        ):
            entries[prefix + name] = b"fixture"
        entries[prefix + "_build_info.json"] = json.dumps({"shmem_revision": "test-sdk"}).encode()
    path = tmp_path / "synthetic.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


@pytest.mark.parametrize("enabled,payload", [(False, False), (True, True)])
def test_wheel_expected_payload(tmp_path, enabled, payload):
    validate = load(ROOT / "scripts/validate_package.py")
    assert validate.inspect_wheel(wheel(tmp_path, enabled, payload))["megamoe"]["enabled"] == enabled


@pytest.mark.parametrize("megamoe,ep_gmm_fused", [(False, False), (False, True), (True, False), (True, True)])
def test_wheel_components_are_validated_independently(tmp_path, megamoe, ep_gmm_fused):
    validate = load(ROOT / "scripts/validate_package.py")
    info = validate.inspect_wheel(wheel(tmp_path, megamoe, megamoe, ep_gmm_fused))
    assert info["megamoe"]["enabled"] is megamoe
    assert info["ep_gmm_fused"]["enabled"] is ep_gmm_fused


def test_component_switches_preserve_python_packages(tmp_path):
    # Exercise actual setuptools staging and wheel assembly without CANN compilation.
    for name in ("setup.py", "build_support.py", "VERSION", "pyproject.toml"):
        shutil.copy2(ROOT / name, tmp_path / name)
    for name in ("deep_ep", "experiments/megamoe/python", "experiments/ep_gmm_fused/python"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    configurations = [
        (True, False, False),
        (True, True, False),
        (False, True, False),
        (True, True, True),
        (True, False, True),
        (False, False, False),
    ]
    for megamoe, ep_gmm_fused, timer in configurations:
        env = dict(os.environ)
        for name, flag in (
            ("DEEPEP_BUILD_MEGAMOE", megamoe),
            ("DEEPEP_BUILD_EP_GMM_FUSED", ep_gmm_fused),
            ("DEEPEP_MEGAMOE_TIMER", timer),
            ("DEEPEP_LINEINFO", False),
        ):
            env[name] = "ON" if flag else "OFF"
        for args in (("build_py",), ("bdist_wheel", "--skip-build")):
            result = subprocess.run(
                [sys.executable, "setup.py", *args],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr
        (archive_path,) = (tmp_path / "dist").glob("*.whl")
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert names.count("deep_ep_experimental/__init__.py") == int(megamoe or ep_gmm_fused)
            for component, present, module in (
                ("megamoe", megamoe, "api.py"),
                ("ep_gmm_fused", ep_gmm_fused, "ops.py"),
            ):
                prefix = f"deep_ep_experimental/{component}/"
                assert any(name.startswith(prefix) for name in names) is present
                if present:
                    original = ROOT / "experiments" / component / "python" / prefix / module
                    assert archive.read(prefix + module) == original.read_bytes()


@pytest.mark.parametrize("enabled,payload", [(False, True), (True, False)])
def test_stale_or_missing_beta_rejected(tmp_path, enabled, payload):
    validate = load(ROOT / "scripts/validate_package.py")
    with pytest.raises(RuntimeError, match="Beta"):
        validate.inspect_wheel(wheel(tmp_path, enabled, payload))


@pytest.mark.parametrize("fault", ["missing", "json", "list", "revision", "header", "checksum"])
def test_megamoe_build_keeps_strict_sdk_validation(tmp_path, fault):
    build = load(ROOT / "build_support.py")
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())["shmem"]
    cann, sdk = tmp_path / "cann", tmp_path / "sdk"
    (cann / "include/acl").mkdir(parents=True)
    (cann / "include/acl/acl.h").touch()
    header = sdk / "include/host/shmem_host_def.h"
    header.parent.mkdir(parents=True)
    header.write_bytes(b"header")
    library = sdk / "lib/libshmem.so"
    library.parent.mkdir()
    library.write_bytes(b"library")
    lock["header_sha256_lf"] = {"host/shmem_host_def.h": hashlib.sha256(header.read_bytes()).hexdigest()}
    identity = {
        "revision": lock["revision"],
        "sha256": {
            path.relative_to(sdk).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (header, library)
        },
    }
    (tmp_path / "dependencies.lock.json").write_text(json.dumps({"shmem": lock}))
    build.ROOT = tmp_path
    if fault == "revision":
        identity["revision"] = "unexpected"
    elif fault == "header":
        header.write_bytes(b"changed header")
    elif fault == "checksum":
        library.write_bytes(b"changed library")
    if fault != "missing":
        (sdk / "deepep-sdk.json").write_text(
            "invalid json" if fault == "json" else "[]" if fault == "list" else json.dumps(identity)
        )
    with pytest.raises((RuntimeError, ValueError), match="SHMEM|JSON|Expecting"):
        build.build_configuration(
            {
                "DEEPEP_NPU_ARCH": "Ascend950",
                "DEEPEP_BUILD_MEGAMOE": "ON",
                "ASCEND_HOME_PATH": str(cann),
                "SHMEM_ROOT": str(sdk),
            }
        )
