# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Explicitly build the pinned SHMEM submodule; pip/wheel never invokes this script."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

from record_shmem_sdk import record

ROOT = Path(__file__).resolve().parents[1]


def cann_version(expected):
    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann or not (Path(cann) / "compiler/version.info").is_file():
        raise RuntimeError("Source the CANN set_env.sh first (ASCEND_HOME_PATH is missing or invalid)")
    version = dict(
        line.split("=", 1) for line in (Path(cann) / "compiler/version.info").read_text().splitlines() if "=" in line
    )
    if version.get("Version") != expected:
        raise RuntimeError(f"This build requires CANN {expected}")
    return version


def source_patches(source):
    """Fingerprint tracked edits and untracked inputs without modifying Git."""

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    if not git("status", "--porcelain", "--untracked-files=normal").strip():
        return []
    digest = hashlib.sha256(git("diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv"))
    for name in sorted(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
        if not name:
            continue
        path = source / os.fsdecode(name)
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise RuntimeError(f"Untracked SHMEM input must be a local regular file: {path}")
        digest.update(name + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return [{"kind": "local-worktree", "sha256": digest.hexdigest()}]


def prepare(root=ROOT, *, rebuild=False, if_needed=False, allow_local_changes=False):
    if platform.system() != "Linux":
        raise RuntimeError("SHMEM compilation requires Linux and the pinned CANN SDK")
    root = Path(root).resolve()
    lock = json.loads((root / "dependencies.lock.json").read_text(encoding="utf-8"))
    source = root / "third-party/shmem"
    if not (source / ".git").exists() or not (source / "scripts/build.sh").is_file():
        raise RuntimeError("Initialize dependencies first: git submodule update --init --recursive")
    if source.resolve() != source or source.parent.resolve() != source.parent:
        raise RuntimeError("SHMEM source must not be a symlink to another checkout")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
        raise RuntimeError("third-party/shmem must be its own initialized Git submodule")
    revision = git("rev-parse", "HEAD")
    if revision != lock["shmem"]["revision"]:
        raise RuntimeError("SHMEM revision differs from dependencies.lock.json; do not build a moving branch")
    patches = source_patches(source)
    if patches and not allow_local_changes:
        raise RuntimeError(
            "SHMEM has local changes; pass --allow-local-changes to build them explicitly "
            "(cluster launcher: --allow-local-shmem)"
        )

    version = cann_version(lock["cann_version"])
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    if not jobs.isdecimal() or not 1 <= int(jobs) <= 64:
        raise ValueError("CMAKE_BUILD_PARALLEL_LEVEL must be in [1, 64]")

    # 上游脚本会清空 build/install，并生成 ci 打包产物；禁止跳转到其他目录。
    for name in ("build", "install", "ci", "3rdparty"):
        path = source / name
        if path.resolve() != path:
            raise RuntimeError(f"Refusing SHMEM build through a symlink: {path}")
    manifest = source / "install/shmem/deepep-sdk.json"
    if if_needed and not rebuild and manifest.is_file():
        identity = json.loads(manifest.read_text(encoding="utf-8"))
        expected_build = {
            "cann_version": version["Version"],
            "cann_timestamp": version.get("timestamp", "unknown"),
            "target": lock["target"],
            "source_patches": patches,
        }
        if identity.get("revision") != revision or identity.get("build") != expected_build:
            raise RuntimeError(
                "Existing SHMEM build identity changed; stop SDK users and rebuild explicitly "
                "with --rebuild (cluster launcher: --rebuild-shmem)"
            )
        # 复用也校验全部头文件/库哈希，不把存在 manifest 当成构建成功。
        import sys

        sys.path.insert(0, str(ROOT))
        from build_support import validate_shmem_sdk

        validate_shmem_sdk(manifest.parent, lock["shmem"])
        print(f"[SHMEM] Reusing validated SDK: {manifest.parent}", flush=True)
        return manifest
    if not rebuild and any((source / name).exists() for name in ("build", "install")):
        raise FileExistsError(
            "SHMEM build/install already exists; reuse its SDK, or stop users of that SDK "
            "and pass --rebuild to regenerate it (then rebuild dependent wheels)"
        )
    print(f"[SHMEM] source={revision} CANN={version['Version']} target={lock['target']}", flush=True)
    if patches:
        print(f"[SHMEM] Building local source modifications: {json.dumps(patches)}", flush=True)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("Git is required to prepare SHMEM dependencies")
    # 上游脚本会调用 Git 下载 json 等依赖；只隔离 Git 的库路径，保留编译器所需的 CANN 环境。
    with tempfile.TemporaryDirectory(prefix="deepep-shmem-git-") as git_dir:
        wrapper = Path(git_dir) / "git"
        wrapper.write_text(
            "#!/bin/sh\nunset LD_LIBRARY_PATH LD_PRELOAD\n"
            f'exec {shlex.quote(str(Path(git_executable).resolve()))} "$@"\n'
        )
        wrapper.chmod(0o700)
        build_env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=jobs)
        build_env["PATH"] = git_dir + os.pathsep + build_env.get("PATH", os.defpath)
        # -e 防止上游中间命令失败后继续打包，随后误登记旧库为成功产物。
        subprocess.run(
            ["bash", "-e", "scripts/build.sh", "-soc_type", lock["target"]],
            cwd=source,
            env=build_env,
            check=True,
        )
    sdk = source / "install/shmem"
    if git("rev-parse", "HEAD") != revision or source_patches(source) != patches:
        raise RuntimeError("SHMEM source changed during compilation; SDK identity was not recorded")
    manifest = record(
        sdk,
        revision,
        build_info={
            "cann_version": version["Version"],
            "cann_timestamp": version.get("timestamp", "unknown"),
            "target": lock["target"],
            "source_patches": patches,
        },
    )
    print(f"[SHMEM] SDK identity: {manifest}", flush=True)
    print("[SHMEM] Add third-party/shmem/install/shmem/lib to LD_LIBRARY_PATH before loading the wheel.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Regenerate this submodule's build/install artifacts")
    parser.add_argument("--if-needed", action="store_true", help="Reuse a verified SDK from this CANN build")
    parser.add_argument(
        "--allow-local-changes",
        action="store_true",
        help="Build local edits on the pinned commit and record their source fingerprint",
    )
    args = parser.parse_args()
    prepare(rebuild=args.rebuild, if_needed=args.if_needed, allow_local_changes=args.allow_local_changes)
