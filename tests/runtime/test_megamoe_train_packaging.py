# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026

"""Training wheel contents and component switching, without native compilation."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPONENT = "deep_ep_experimental/megamoe_train/"
TRAINING_FILES = ("__init__.py", "bf16.py", "api.py", "_C.fake.so", "libdeepep_megamoe_train_bf16.so")


def inspect_wheel(path):
    spec = importlib.util.spec_from_file_location("validate_package", ROOT / "scripts/validate_package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inspect_wheel(path)


def training_wheel(tmp_path, shmem_wheel_payload, enabled, files, inference=False):
    path = tmp_path / "training.whl"
    with zipfile.ZipFile(path, "w") as archive:
        runtime = shmem_wheel_payload(archive)
        info = {
            "communication_operations": [],
            "megamoe": {"enabled": inference},
            "megamoe_train": {"enabled": enabled},
            "shmem_runtime": runtime,
        }
        archive.writestr("deep_ep/_C.fake.so", "fixture")
        archive.writestr("deep_ep/_build_info.json", json.dumps(info))
        archive.writestr("licenses/deepseek-deepep.LICENSE", "fixture")
        if inference:
            for name in ("api.py", "benchmark.py", "configs/smoke.json", "_C.fake.so", "libmegamoe_kernel.so"):
                archive.writestr("deep_ep_experimental/megamoe/" + name, "fixture")
        for name in files:
            archive.writestr(name, "fixture")
    return path


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_training_wheel_expected_payload(tmp_path, shmem_wheel_payload, enabled, inference):
    files = [COMPONENT + name for name in TRAINING_FILES] if enabled else []
    info = inspect_wheel(training_wheel(tmp_path, shmem_wheel_payload, enabled, files, inference))
    assert info["megamoe_train"]["enabled"] is enabled
    assert info["megamoe"]["enabled"] is inference


@pytest.mark.parametrize("missing", TRAINING_FILES)
def test_training_wheel_rejects_missing_files(tmp_path, shmem_wheel_payload, missing):
    files = [COMPONENT + name for name in TRAINING_FILES if name != missing]
    with pytest.raises(RuntimeError, match="Incomplete MegaMoE training wheel"):
        inspect_wheel(training_wheel(tmp_path, shmem_wheel_payload, True, files))


def test_training_wheel_rejects_multiple_extensions(tmp_path, shmem_wheel_payload):
    files = [COMPONENT + name for name in (*TRAINING_FILES, "_C.old.so")]
    with pytest.raises(RuntimeError, match="Incomplete MegaMoE training wheel"):
        inspect_wheel(training_wheel(tmp_path, shmem_wheel_payload, True, files))


@pytest.mark.parametrize("leftover", ["bf16.py", "libdeepep_megamoe_train_bf16.so"])
def test_disabled_training_wheel_rejects_leftovers(tmp_path, shmem_wheel_payload, leftover):
    with pytest.raises(RuntimeError, match="MegaMoE training payload in an OFF wheel"):
        inspect_wheel(training_wheel(tmp_path, shmem_wheel_payload, False, [COMPONENT + leftover]))


@pytest.mark.parametrize("enabled", [False, True])
def test_training_wheel_rejects_nested_staging(tmp_path, shmem_wheel_payload, enabled):
    files = [COMPONENT + name for name in TRAINING_FILES] if enabled else []
    files.append("build/stale/" + COMPONENT + "_C.old.so")
    with pytest.raises(RuntimeError, match="nested MegaMoE training staging"):
        inspect_wheel(training_wheel(tmp_path, shmem_wheel_payload, enabled, files))


def test_training_switches_produce_clean_wheels(tmp_path):
    for name in ("setup.py", "build_support.py", "VERSION", "pyproject.toml"):
        shutil.copy2(ROOT / name, tmp_path / name)
    for name in ("deep_ep", "ascend_deepep", "experiments/megamoe_train/python"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    package = tmp_path / "experiments/megamoe_train/python" / COMPONENT
    # Exercise SO inclusion as package data; native compilation is skipped.
    for name in TRAINING_FILES:
        if name.endswith(".so"):
            (package / name).write_bytes(name.encode())
    env = dict(
        os.environ,
        DEEPEP_BUILD_MEGAMOE="OFF",
        DEEPEP_BUILD_EP_GMM_FUSED="OFF",
        DEEPEP_BF16_FORCE_EXPERT_READY="OFF",
        PYTHONIOENCODING="utf-8",
    )
    for enabled in (True, False, True):
        env["DEEPEP_BUILD_MEGAMOE_TRAIN"] = "ON" if enabled else "OFF"
        for args in (("build_py",), ("bdist_wheel", "--skip-build")):
            result = subprocess.run(
                [sys.executable, "setup.py", *args],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            assert result.returncode == 0, result.stdout + result.stderr
        (wheel,) = (tmp_path / "dist").glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            assert names.count("deep_ep_experimental/__init__.py") == int(enabled)
            assert any(name.startswith(COMPONENT) for name in names) is enabled
            if enabled:
                for name in TRAINING_FILES:
                    assert names.count(COMPONENT + name) == 1
                    assert archive.read(COMPONENT + name) == (package / name).read_bytes()
