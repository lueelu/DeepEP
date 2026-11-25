#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
export BUILD_DIR=${BUILD_DIR:-"$REPO_DIR/build/hierarchical_alltoall"}
base_id=${RUN_ID:?Set RUN_ID to the same value on both nodes}
case "$base_id" in
    .|..|*[!a-zA-Z0-9_.-]*) echo 'Invalid RUN_ID' >&2; exit 1;;
esac
export NNODES=${NNODES:-2}
export TEST_MODE=perf N=1024 WINDOW_M=${WINDOW_M:-256}
base_port=${PORT:-8666}
if [[ ! "$base_port" =~ ^[1-9][0-9]{0,4}$ ]] || ((base_port > 65531)); then
    echo 'PORT must be between 1 and 65531 (the sweep uses five consecutive ports)' >&2
    exit 1
fi
export LOG_ROOT="${LOG_ROOT:-"$BUILD_DIR/logs"}/$base_id"
offset=0
for rows in 4 64 1024 2048 4096; do
    bytes=$((rows * N))
    export M=$rows PORT=$((base_port + offset)) RUN_ID="data-$bytes"
    bash "$SCRIPT_DIR/run.sh"
    offset=$((offset + 1))
done
if [[ "$NNODES" == 1 ]]; then
    output="$LOG_ROOT/summary.csv"
    summary=$("${PYTHON:-python3}" "$SCRIPT_DIR/perf_summary.py" "$LOG_ROOT")
    printf '%s\n' "$summary" | tee "$output"
else
    printf 'Collect both nodes and summarize: python3 %s/perf_summary.py NODE0_LOG_ROOT NODE1_LOG_ROOT\n' "$SCRIPT_DIR"
fi
