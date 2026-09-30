# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Metadata hooks do not import frameworks, probe devices or configure CMake."""

from pathlib import Path
import os
import sys

from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.resolve()))
# PEP 517 setup execution must add the project root before this local import.
from build_support import (  # noqa: E402
    BuildEggInfo,
    CheckedInstallLib,
    CMakeBuild,
    ConfiguredBuild,
    VariantBuild,
    enabled,
)

BETA = "experiments/megamoe/python"
beta_enabled = os.environ.get("DEEPEP_BUILD_MEGAMOE", "OFF") == "ON"
ep_gmm_fused = enabled("DEEPEP_BUILD_EP_GMM_FUSED")
packages = find_packages(include=["deep_ep", "deep_ep.*"])
package_dir = {}
cmdclass = {"build_ext": CMakeBuild, "install_lib": CheckedInstallLib}
if ep_gmm_fused:
    packages += ["deep_ep_experimental", "deep_ep_experimental.ep_gmm_fused"]
    package_dir["deep_ep_experimental"] = "experiments/ep_gmm_fused/python/deep_ep_experimental"
    cmdclass["build"] = ConfiguredBuild
    cmdclass["egg_info"] = BuildEggInfo
if beta_enabled:
    packages += [name for name in find_packages(where=BETA) if name not in packages]
    package_dir.setdefault("deep_ep_experimental", BETA + "/deep_ep_experimental")
    package_dir["deep_ep_experimental.megamoe"] = BETA + "/deep_ep_experimental/megamoe"
    cmdclass["build"] = VariantBuild

setup(
    name="ascend-deepep",
    version=(ROOT / "VERSION").read_text(encoding="utf-8").strip(),
    description="DeepEP for Ascend: native runtime and optional communication/GMM operators",
    license_files=["LICENSE", "third-party/deepseek-deepep.LICENSE"]
    + (["third-party/cann.LICENSE"] if beta_enabled else []),
    python_requires=">=3.11",
    packages=packages,
    package_dir=package_dir,
    package_data={
        "deep_ep_experimental.ep_gmm_fused": ["NOTICE.md", "upstream.json"],
        "deep_ep_experimental.megamoe": ["*.so", "configs/*.json"],
    },
    ext_modules=[Extension("deep_ep._C", sources=[])],
    cmdclass=cmdclass,
    zip_safe=False,
)
