# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Prepare pinned dependencies, build and install one wheel in the active Python environment."""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from build_shmem import cann_version, prepare  # noqa: E402


def build(root=ROOT, *, rebuild_shmem=False):
    root = Path(root).resolve()
    if platform.system() != "Linux" or sys.version_info < (3, 11):
        raise RuntimeError("Build requires Linux and Python >= 3.11")
    lock = json.loads((root / "dependencies.lock.json").read_text(encoding="utf-8"))
    cann_version(lock["cann_version"])
    env = dict(os.environ)
    env.setdefault("DEEPEP_NPU_ARCH", lock["target"])
    env.setdefault("DEEPEP_BUILD_MEGAMOE", "ON")
    env.setdefault("DEEPEP_MEGAMOE_TIMER", env["DEEPEP_BUILD_MEGAMOE"])
    beta, timer = env["DEEPEP_BUILD_MEGAMOE"], env["DEEPEP_MEGAMOE_TIMER"]
    if beta not in {"ON", "OFF"} or timer not in {"ON", "OFF"} or (beta == "OFF" and timer == "ON"):
        raise ValueError("Build options must be ON/OFF; timer requires MegaMoE")
    if env["DEEPEP_NPU_ARCH"] != lock["target"]:
        raise ValueError(f"This build requires {lock['target']}")
    for tool in ("git", "bash", "cmake", env.get("CXX", "c++")):
        if not shutil.which(tool):
            raise RuntimeError(f"Required build tool not found: {tool}")
    for name in ("SHMEM_ROOT", "CATLASS_ROOT"):
        if name in env:
            if not env[name] or not Path(env[name]).is_dir():
                raise ValueError(f"{name} must be an existing dependency directory")
            env[name] = str(Path(env[name]).resolve())
    if rebuild_shmem and "SHMEM_ROOT" in env:
        raise ValueError("--rebuild-shmem only rebuilds the bundled SDK; unset SHMEM_ROOT first")

    # Git 的 HTTPS helper 使用系统库，避免加载 Python/CANN 环境中的 OpenSSL 等动态库。
    git_env = dict(env)
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        git_env.pop(name, None)

    def run(command, *, cwd=root):
        print("[BUILD] " + shlex.join(map(str, command)), flush=True)
        subprocess.run(command, cwd=cwd, env=git_env if command[0] == "git" else env, check=True)

    # 提前发现框架缺失；不安装或升级用户的 Torch/CANN 环境。
    if beta == "ON":
        run([sys.executable, "-c", "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"])

    def submodule(name, revision):
        source = root / "third-party" / name
        if not (source / ".git").exists():
            run(["git", "submodule", "update", "--init", "--recursive", "--", f"third-party/{name}"])

        # 不对已初始化子模块自动 checkout，避免覆盖用户修改或隐藏版本漂移。
        def git(*args):
            return subprocess.check_output(["git", "-C", str(source), *args], env=git_env, text=True).strip()

        if git("rev-parse", "HEAD") != revision or git("status", "--porcelain", "--untracked-files=normal"):
            raise RuntimeError(f"{source} must be clean and pinned to {revision}; review it before updating")

    if "SHMEM_ROOT" not in env:
        submodule("shmem", lock["shmem"]["revision"])
    if beta == "ON" and "CATLASS_ROOT" not in env:
        submodule("catlass", lock["megamoe"]["catlass_revision"])
    run([sys.executable, "-m", "pip", "install", "-r", str(root / "requirements-build.txt")])
    if "SHMEM_ROOT" not in env:
        prepare(root, rebuild=rebuild_shmem, if_needed=True)
        env["SHMEM_ROOT"] = str(root / "third-party/shmem/install/shmem")
    env.setdefault("CATLASS_ROOT", str(root / "third-party/catlass"))
    env["LD_LIBRARY_PATH"] = str(Path(env["SHMEM_ROOT"]) / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    # 复用与外部 SDK 同样走原有完整预检，主构建仍只有一个 CMake/wheel 路径。
    sys.path.insert(0, str(ROOT))
    from build_support import build_configuration

    build_configuration(env)
    (root / "dist").mkdir(exist_ok=True)
    wheel_dir = Path(tempfile.mkdtemp(prefix="build-", dir=root / "dist"))
    run([sys.executable, "-m", "pip", "wheel", ".", "--no-build-isolation", "--no-deps", "-w", str(wheel_dir)])
    wheels = list(wheel_dir.glob("ascend_deepep-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one ascend-deepep wheel in {wheel_dir}, found {len(wheels)}")
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0])])
    probe = "from deep_ep._native import require_native; require_native('build smoke')"
    if beta == "ON":
        probe += "; import torch, torch_npu; import deep_ep_experimental.megamoe._C"
    # -I 不从源码目录/PYTHONPATH 导入；这里只加载扩展，不分配 NPU 或启动通信。
    run([sys.executable, "-I", "-c", probe], cwd=wheel_dir)
    print(f"[BUILD] Installed: {wheels[0]}", flush=True)
    print("[BUILD] Before running tests in your shell (bash cannot export to its parent):")
    print(f'export LD_LIBRARY_PATH={shlex.quote(str(Path(env["SHMEM_ROOT"]) / "lib"))}:"${{LD_LIBRARY_PATH:-}}"')
    return wheels[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-shmem",
        action="store_true",
        help="Regenerate bundled SHMEM build/install; stop all users of that SDK first",
    )
    args = parser.parse_args()
    build(rebuild_shmem=args.rebuild_shmem)
