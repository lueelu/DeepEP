#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# 在 Python 启动前优先使用当前 Conda 的运行库，避免加载过旧的系统 libstdc++。
if [[ -n ${CONDA_PREFIX:-} ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "${PYTHON:-python3}" "$ROOT/build.py" "$@"
