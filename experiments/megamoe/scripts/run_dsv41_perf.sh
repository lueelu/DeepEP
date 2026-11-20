#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/dsv41_perf.py" "$@"
