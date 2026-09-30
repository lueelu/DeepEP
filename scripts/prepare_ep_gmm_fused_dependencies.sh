#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=${PROJECT_DIR}
PYTHON=${PYTHON:-python3}
CATCCOS_REPOSITORY=https://gitcode.com/cann/catccos.git
CATCCOS_COMMIT=878c2e0a504be8c49c03b5e1157a7c714a628e29
CATCCOS_DIR=${CATCCOS_SOURCE_PATH:-"${REPO_ROOT}/third-party/catccos"}
SHMEM_REPOSITORY=https://gitcode.com/cann/shmem.git
SHMEM_COMMIT=$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["shmem"]["revision"])' "${REPO_ROOT}/dependencies.lock.json")
SHMEM_DIR=${SHMEM_SOURCE_PATH:-"${REPO_ROOT}/third-party/shmem"}

if [[ ! -d "${CATCCOS_DIR}/.git" ]]; then
    if [[ -e "${CATCCOS_DIR}" ]]; then
        echo "error: ${CATCCOS_DIR} exists but is not a Git checkout" >&2
        exit 1
    fi
    git clone "${CATCCOS_REPOSITORY}" "${CATCCOS_DIR}"
    git -C "${CATCCOS_DIR}" checkout --detach "${CATCCOS_COMMIT}"
else
    ACTUAL_COMMIT=$(git -C "${CATCCOS_DIR}" rev-parse HEAD)
    if [[ "${ACTUAL_COMMIT}" != "${CATCCOS_COMMIT}" ]]; then
        echo "error: CATCCOS_SOURCE_PATH is at ${ACTUAL_COMMIT}, expected ${CATCCOS_COMMIT}" >&2
        echo "Use a clean checkout at the pinned commit or unset CATCCOS_SOURCE_PATH." >&2
        exit 1
    fi
fi

# Only CATLASS is needed from CatCCOS submodules; SHMEM follows the main lock.
git -C "${CATCCOS_DIR}" submodule update --init --recursive 3rdparty/catlass

if [[ ! -e "${SHMEM_DIR}/.git" ]]; then
    if [[ -e "${SHMEM_DIR}" ]]; then
        echo "error: ${SHMEM_DIR} exists but is not a Git checkout" >&2
        exit 1
    fi
    git clone "${SHMEM_REPOSITORY}" "${SHMEM_DIR}"
fi

# Keep user edits intact and never label a modified checkout as the official
# binary. Gitignored build/install products do not make the checkout dirty.
SHMEM_WORKTREE_STATUS=$(git -C "${SHMEM_DIR}" status --porcelain --untracked-files=normal)
if [[ -n "${SHMEM_WORKTREE_STATUS}" ]]; then
    echo "error: SHMEM checkout has local changes; commit or move them before preparing dependencies" >&2
    printf '%s\n' "${SHMEM_WORKTREE_STATUS}" >&2
    exit 1
fi
ACTUAL_SHMEM_COMMIT=$(git -C "${SHMEM_DIR}" rev-parse HEAD)
if [[ "${ACTUAL_SHMEM_COMMIT}" != "${SHMEM_COMMIT}" ]]; then
    if ! git -C "${SHMEM_DIR}" cat-file -e "${SHMEM_COMMIT}^{commit}" 2>/dev/null; then
        git -C "${SHMEM_DIR}" fetch --no-tags "${SHMEM_REPOSITORY}" "${SHMEM_COMMIT}"
    fi
    git -C "${SHMEM_DIR}" checkout --detach "${SHMEM_COMMIT}"
fi

# Resolve paths before changing directories and before printing shell commands.
CATCCOS_DIR=$(cd -- "${CATCCOS_DIR}" && pwd)
SHMEM_DIR=$(cd -- "${SHMEM_DIR}" && pwd)
DEFAULT_SDK_DIR="${SHMEM_DIR}/install/shmem"
SDK_DIR=$(realpath -m -- "${SHMEM_ROOT:-${DEFAULT_SDK_DIR}}")
DEFAULT_SDK_DIR=$(realpath -m -- "${DEFAULT_SDK_DIR}")

if [[ -e "${SDK_DIR}/deepep-sdk.json" ]]; then
    # Never silently rebuild or relabel a mismatched SDK.
    "${PYTHON}" "${SCRIPT_DIR}/record_shmem_sdk.py" --sdk "${SDK_DIR}" --check
    echo "Reusing verified SHMEM SDK: ${SDK_DIR}"
elif [[ "${SDK_DIR}" != "${DEFAULT_SDK_DIR}" ]]; then
    echo "error: SHMEM_ROOT=${SDK_DIR} has no deepep-sdk.json; provide a recorded SDK or unset SHMEM_ROOT to build ${DEFAULT_SDK_DIR}" >&2
    exit 1
else
    if [[ ! -f "${ASCEND_HOME_PATH:-}/include/acl/acl.h" ]]; then
        echo "error: load the CANN set_env.sh before building SHMEM" >&2
        exit 1
    fi
    echo "Building locked SHMEM ${SHMEM_COMMIT} for Ascend950"
    (
        cd -- "${SHMEM_DIR}"
        # The upstream script does not enable errexit itself.
        bash -e -o pipefail scripts/build.sh -soc_type Ascend950
    )
    # Only attest a successful build from the clean, locked source tree.
    ACTUAL_SHMEM_COMMIT=$(git -C "${SHMEM_DIR}" rev-parse HEAD)
    SHMEM_WORKTREE_STATUS=$(git -C "${SHMEM_DIR}" status --porcelain --untracked-files=normal)
    if [[ "${ACTUAL_SHMEM_COMMIT}" != "${SHMEM_COMMIT}" || -n "${SHMEM_WORKTREE_STATUS}" ]]; then
        echo "error: SHMEM source changed during the build; SDK identity was not recorded" >&2
        exit 1
    fi
    "${PYTHON}" "${SCRIPT_DIR}/record_shmem_sdk.py" --sdk "${SDK_DIR}" --revision "${SHMEM_COMMIT}"
    "${PYTHON}" "${SCRIPT_DIR}/record_shmem_sdk.py" --sdk "${SDK_DIR}" --check
fi

echo "CatCCOS: ${CATCCOS_DIR}"
echo "SHMEM source: ${SHMEM_DIR} (${SHMEM_COMMIT})"
echo "Run these commands in the calling shell before building or testing DeepEP:"
printf 'export CATCCOS_SOURCE_PATH=%q\n' "${CATCCOS_DIR}"
printf 'export SHMEM_SOURCE_PATH=%q\n' "${SHMEM_DIR}"
printf 'export SHMEM_ROOT=%q\n' "${SDK_DIR}"
printf 'export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}\n' "${SDK_DIR}/lib"
