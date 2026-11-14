#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_DIR=$(cd -- "${EXAMPLE_DIR}/../.." && pwd)
: "${ASCEND_HOME_PATH:?Source the CANN set_env.sh first}"
# Preserve an existing CatCCOS-based setup, but only use its Catlass dependency.
if [ -z "${CATLASS_SOURCE_PATH:-}" ]; then
    if [ -n "${CATCCOS_SOURCE_PATH:-}" ]; then
        CATLASS_SOURCE_PATH="$CATCCOS_SOURCE_PATH/3rdparty/catlass"
    else
        CATLASS_SOURCE_PATH="$REPO_DIR/third-party/catlass"
    fi
fi
export CATLASS_SOURCE_PATH
: "${SHMEM_ROOT:?Set SHMEM_ROOT to the installed UDMA+RDMA SDK containing include/src/lib}"
export GROUP_SIZE=${GROUP_SIZE:-16}
export COMM_BLOCK_M=${COMM_BLOCK_M:-64}
export SHMEM_RDMA_BACKEND=${SHMEM_RDMA_BACKEND:-HNS_1825}
export SHMEM_HNS_1825_PI_ON_CHIP=${SHMEM_HNS_1825_PI_ON_CHIP:-}
case "$GROUP_SIZE" in 1|2|4|8|16) ;; *) echo 'GROUP_SIZE must be 1,2,4,8,16' >&2; exit 1 ;; esac
case "$COMM_BLOCK_M" in ''|*[!0-9]*|0) echo 'COMM_BLOCK_M must be positive' >&2; exit 1 ;; esac
case "$SHMEM_RDMA_BACKEND" in
    HNS_1825) case "$SHMEM_HNS_1825_PI_ON_CHIP" in 0|1) ;; *) echo 'Set SHMEM_HNS_1825_PI_ON_CHIP=0 or 1 to match SDK/firmware' >&2; exit 1 ;; esac ;;
    XSCALE) if [ -n "$SHMEM_HNS_1825_PI_ON_CHIP" ]; then echo 'Unset SHMEM_HNS_1825_PI_ON_CHIP for XSCALE' >&2; exit 1; fi ;;
    *) echo 'SHMEM_RDMA_BACKEND must be HNS_1825 or XSCALE' >&2; exit 1 ;;
esac
BUILD_DIR=${BUILD_DIR:-${REPO_DIR}/build/hierarchical_alltoall}
if [ -z "${CXX:-}" ]; then
    for compiler in \
        "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
        "$ASCEND_HOME_PATH/compiler/ccec_compiler/bin/bisheng" \
        "$ASCEND_HOME_PATH/$(uname -m)-linux/bin/bisheng"; do
        if [ -x "$compiler" ]; then
            CXX=$compiler
            break
        fi
    done
    [ -n "${CXX:-}" ] || { echo "Missing Bisheng under $ASCEND_HOME_PATH; set CXX to its executable path" >&2; exit 1; }
fi
PYTHON=${PYTHON:-python3}
compiler_path=$(command -v -- "$CXX") && [ -x "$compiler_path" ] || { echo "Missing Bisheng: $CXX" >&2; exit 1; }
CXX=$compiler_path
[ -f "$SHMEM_ROOT/lib/libshmem.so" ] || { echo "SHMEM_ROOT must point to install/shmem, not the source tree: $SHMEM_ROOT" >&2; exit 1; }
cmake -S "$EXAMPLE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSHMEM_ROOT="$SHMEM_ROOT" \
    -DCATLASS_SOURCE_PATH="$CATLASS_SOURCE_PATH" \
    -DGROUP_SIZE="$GROUP_SIZE" -DCOMM_BLOCK_M="$COMM_BLOCK_M" \
    -DRDMA_BACKEND="$SHMEM_RDMA_BACKEND" -DHNS_PI_ON_CHIP="$SHMEM_HNS_1825_PI_ON_CHIP"
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-8}"
"$PYTHON" "$SCRIPT_DIR/config.py" record "$BUILD_DIR"
echo "Built: $BUILD_DIR/hierarchical_alltoall"
