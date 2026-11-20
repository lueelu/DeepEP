#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu

# Run from an existing checkout; never clone or update the repository here.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" -I "$SCRIPT_DIR/bs32.py" "$@"
