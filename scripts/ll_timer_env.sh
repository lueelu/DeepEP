#!/usr/bin/env bash
# Source from any directory. External CANN location is explicit; no device selection here.
LL_TIMER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export TMPDIR="$LL_TIMER_ROOT/.tmp" TMP="$LL_TIMER_ROOT/.tmp" TEMP="$LL_TIMER_ROOT/.tmp"
export XDG_CACHE_HOME="$LL_TIMER_ROOT/.cache" PIP_CACHE_DIR="$LL_TIMER_ROOT/.cache/pip"
export TORCH_EXTENSIONS_DIR="$LL_TIMER_ROOT/.cache/torch_extensions" TRITON_CACHE_DIR="$LL_TIMER_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$LL_TIMER_ROOT/.cache/cuda" CCACHE_DIR="$LL_TIMER_ROOT/.cache/ccache"
export ASCEND_CACHE_PATH="$LL_TIMER_ROOT/.cache/ascend"
export ASCEND_WORK_PATH="$LL_TIMER_ROOT/.logs/ascend" ASCEND_PROCESS_LOG_PATH="$LL_TIMER_ROOT/.logs/ascend"
export SHMEM_LOG_PATH="$LL_TIMER_ROOT/.logs/shmem" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH" "$SHMEM_LOG_PATH"
if [[ -n "${CANN_ENV:-}" ]]; then
    ll_cann_env="$CANN_ENV"
    if [[ "$ll_cann_env" != /* ]]; then ll_cann_env="$LL_TIMER_ROOT/$ll_cann_env"; fi
    source "$ll_cann_env"
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo 'Load the CANN environment or explicitly set CANN_ENV before this script.' >&2
    return 1
fi
export LD_LIBRARY_PATH="$LL_TIMER_ROOT/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
