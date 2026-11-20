# Copyright (c) 2026, Lu Lu

"""Prepare an old commit in a new worktree with the SHMEM !944 compatibility overlay."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
# The !944 overlay belongs to the original EP32 history and its SDK.
LEGACY_REF = "e2199e611948056044525413d80fcfd26bbd75a4"
TEXT_PATHS = (
    "dependencies.lock.json",
    ".gitmodules",
    "csrc/runtime/session.cpp",
    "cmake/Dependencies.cmake",
    "experiments/megamoe/CMakeLists.txt",
    "build_support.py",
    "csrc/kernels/dispatch/dispatch.cpp",
    "csrc/kernels/dispatch/fm_primary.hpp",
    "csrc/kernels/dispatch/fm_simt.hpp",
    "csrc/kernels/dispatch/wqe_cache.hpp",
    "csrc/kernels/combine/combine.cpp",
)
COPY_PATHS = (
    "scripts/build_shmem.py",
    "scripts/record_shmem_sdk.py",
    "cmake/ShmemDoorbell.cmake",
    "csrc/kernels/dispatch/udma_live_address.hpp",
)


def adapt_udma_calls(original):
    """Keep live framework addresses distinct from SHMEM's new symmetric-MR inputs."""
    result = dict(original)
    renames = {
        "aclshmemi_udma_fill_sqe_base_ctx<": "ascend_deepep::FillLiveUdmaSqeBase<",
        "aclshmemi_udma_stage_send_wqe<": "ascend_deepep::StageLiveUdmaWqe<",
        "aclshmemi_udma_submit_send_wqes<": "ascend_deepep::SubmitLiveUdmaWqes<",
    }
    for path in TEXT_PATHS:
        if path.startswith("csrc/kernels/dispatch/"):
            text = result[path]
            for before, after in renames.items():
                text = text.replace(before, after)
            if path.endswith("/dispatch.cpp") and '#include "udma_live_address.hpp"' not in text:
                text = replace_once(text, '#include "shmem.h"', '#include "shmem.h"\n#include "udma_live_address.hpp"')
            result[path] = text
    path = "csrc/kernels/combine/combine.cpp"
    before = "auto* remote = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(dst, target));"
    after = "auto* remote = reinterpret_cast<__gm__ uint8_t*>(dst);"
    if after not in result[path]:
        result[path] = replace_once(result[path], before, after)
    return result


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Unrecognized historical source; expected one occurrence of {before!r}")
    return text.replace(before, after, 1)


def overlay(original, payload):
    """Transform dependency/build integration only; do not copy newer algorithms."""
    result = adapt_udma_calls(original)
    lock = json.loads(result["dependencies.lock.json"])
    lock["shmem"] = json.loads(payload["dependencies.lock.json"])["shmem"]
    result["dependencies.lock.json"] = json.dumps(lock, indent=2) + "\n"
    result[".gitmodules"] = result[".gitmodules"].replace("branch = dev", "branch = master")
    session = result["csrc/runtime/session.cpp"]
    old = 'check(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, true), "aclshmemx_set_shared_jetty");'
    new = "attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED;"
    if new not in session:
        session = replace_once(session, old, new)
    result["csrc/runtime/session.cpp"] = session
    cmake = result["cmake/Dependencies.cmake"]
    include = 'include("${CMAKE_CURRENT_LIST_DIR}/ShmemDoorbell.cmake")'
    if include not in cmake:
        cmake += "\n" + include + "\n"
    result["cmake/Dependencies.cmake"] = cmake
    cmake = result["experiments/megamoe/CMakeLists.txt"]
    flags = "list(APPEND kernel_flags ${DEEPEP_SHMEM_DOORBELL_FLAGS})"
    if flags not in cmake:
        cmake = replace_once(
            cmake, "foreach(path IN LISTS beta_include)", flags + "\nforeach(path IN LISTS beta_include)"
        )
    result["experiments/megamoe/CMakeLists.txt"] = cmake
    build = result["build_support.py"]
    if '"hns_1825_pi_on_chip":' not in build:
        build = replace_once(
            build,
            'result["shmem"] = {',
            'result["shmem"] = {\n        "hns_1825_pi_on_chip": '
            'sdk.get("build", {}).get("hns_1825_pi_on_chip") if isinstance(sdk, dict) else None,',
        )
    result["build_support.py"] = build
    for path in COPY_PATHS:
        result[path] = payload[path]
    return result


def plan(ref, root=ROOT):
    revision = git("rev-parse", "--verify", f"{ref}^{{commit}}", root=root)
    subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", revision, LEGACY_REF], check=True)
    original = {path: git("show", f"{revision}:{path}", root=root) + "\n" for path in TEXT_PATHS}
    payload = {path: git("show", f"{LEGACY_REF}:{path}", root=root) + "\n" for path in (*COPY_PATHS, "dependencies.lock.json")}
    return revision, overlay(original, payload)


def prepare(ref, output, root=ROOT):
    root = Path(root).resolve()
    revision, files = plan(ref, root)
    output = Path(os.path.abspath(Path(output).expanduser()))
    # The caller chooses a new directory. Never reuse/reset an existing checkout.
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("--output must be a new directory beneath an existing parent")
    if output.parent.resolve() != output.parent:
        raise ValueError("--output parent must not contain symlinks")
    lock = json.loads(files["dependencies.lock.json"])
    shmem_revision = lock["shmem"]["revision"]
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", str(output), revision], check=True)
    for relative, content in files.items():
        path = output / relative
        if path.resolve() != path:
            raise ValueError(f"Refusing to write through a symlink: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    # Submodule update reads the index, so use the new pin before initialization.
    subprocess.run(
        ["git", "-C", str(output), "update-index", "--cacheinfo", f"160000,{shmem_revision},third-party/shmem"],
        check=True,
    )
    report = {
        "original_commit": revision,
        "shmem_revision": shmem_revision,
        "overlay_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()},
        "note": "Dependency/build overlay only. Original algorithms and benchmark scripts retained. Rebuild before running.",
    }
    (output / "shmem-944-migration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {revision} + SHMEM {shmem_revision} in {output}")
    print("Review git diff and git diff --cached, initialize submodules, then rebuild and run that revision's tests.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", help="Historical commit from the original EP32 branch (requires its Git history)")
    parser.add_argument("--output", required=True, type=Path, help="New worktree directory")
    args = parser.parse_args()
    prepare(args.ref, args.output)
