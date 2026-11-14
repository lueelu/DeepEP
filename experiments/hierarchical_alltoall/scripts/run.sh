#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
BUILD_DIR=${BUILD_DIR:-${REPO_DIR}/build/hierarchical_alltoall}
PYTHON=${PYTHON:-python3}
config=$("$PYTHON" "$SCRIPT_DIR/config.py" check "$BUILD_DIR")
GROUP_SIZE=$(printf '%s\n' "$config" | sed -n '1p')
SHMEM_ROOT=$(printf '%s\n' "$config" | sed -n '2p')
ASCEND_HOME_PATH=$(printf '%s\n' "$config" | sed -n '3p')
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-2}
QP_NUM=${QP_NUM:-1}
M=${M:-257}
N=${N:-1024}
WINDOW_M=${WINDOW_M:-64}
VERIFY_READY=${VERIFY_READY:-1}
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
PORT=${PORT:-8666}
# No process substitution: also parses under bash +o posix in inherited POSIX environments.
IFS=',' read -r -a devices <<< "$DEVICES"
local_size=${#devices[@]}
if [ "$NNODES" != 1 ] && [ "$NNODES" != 2 ]; then echo 'NNODES must be 1 or 2' >&2; exit 1; fi
if [[ ! "$NODE_RANK" =~ ^[01]$ ]] || (( NODE_RANK >= NNODES )); then echo 'Invalid NODE_RANK' >&2; exit 1; fi
if (( local_size < 1 || local_size > GROUP_SIZE )) || { (( NNODES == 2 )) && (( local_size != GROUP_SIZE )); }; then
    echo 'Two nodes require exactly GROUP_SIZE devices per node; one node allows at most GROUP_SIZE' >&2; exit 1
fi
if (( NNODES == 2 )); then : "${RANK0_IP:?Set RANK0_IP to node 0 IPv4 on both servers}"; else RANK0_IP=${RANK0_IP:-127.0.0.1}; fi
export NNODES NODE_RANK QP_NUM M N WINDOW_M VERIFY_READY DEVICES PORT RANK0_IP
# Validate every option before starting any ranks (including duplicate/trailing device IDs).
"$PYTHON" - <<'PY'
import ipaddress, os
ipaddress.IPv4Address(os.environ['RANK0_IP'])
def require(condition, message):
    if not condition:
        raise SystemExit(message)
values = {}
for key in ('QP_NUM','M','N','WINDOW_M','VERIFY_READY','PORT'):
    text = os.environ[key]
    if not text.isascii() or not text.isdecimal() or int(text) > 0xffffffff:
        raise SystemExit('Invalid unsigned integer: ' + key)
    values[key] = int(text)
require(values['QP_NUM'] in (1,2,4), 'QP_NUM must be 1/2/4')
require(values['VERIFY_READY'] in (0,1), 'VERIFY_READY must be 0/1')
require(values['M'] > 0 and values['N'] > 0 and values['N'] % 32 == 0 and values['WINDOW_M'] > 0, 'Invalid M/N/WINDOW_M')
require(0 < values['PORT'] < 65536, 'Invalid PORT')
ids = os.environ['DEVICES'].split(',')
require(all(x.isascii() and x.isdecimal() and int(x) < 2**31 for x in ids), 'Invalid DEVICES')
require(len(set(map(int, ids))) == len(ids), 'Duplicate device IDs')
PY
world=$((NNODES * local_size))
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-qp${QP_NUM}}
case "$RUN_ID" in ''|*[!a-zA-Z0-9_.-]*) echo 'RUN_ID must use letters, digits, dot, dash or underscore' >&2; exit 1 ;; esac
LOG_ROOT=${LOG_ROOT:-${BUILD_DIR}/logs}
mkdir -p "$LOG_ROOT"
case_dir="$LOG_ROOT/${RUN_ID}-node${NODE_RANK}"
mkdir "$case_dir" # Refuse to overwrite previous results.
cp "$BUILD_DIR/build_config.json" "$case_dir/"
printf 'world=%s node=%s group=%s qp=%s m=%s n=%s window_m=%s logs=%s\n' \
    "$world" "$NODE_RANK" "$GROUP_SIZE" "$QP_NUM" "$M" "$N" "$WINDOW_M" "$case_dir"
pids=()
for ((local_rank=0; local_rank<local_size; ++local_rank)); do
    rank=$((NODE_RANK * local_size + local_rank))
    "$BUILD_DIR/hierarchical_alltoall" "$world" "$rank" "tcp://${RANK0_IP}:${PORT}" \
        "$M" "$N" "$WINDOW_M" "${devices[$local_rank]}" "$QP_NUM" "$VERIFY_READY" \
        > "$case_dir/rank${rank}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else status=1; fi
done
for ((local_rank=0; local_rank<local_size; ++local_rank)); do
    rank=$((NODE_RANK * local_size + local_rank))
    tail -n 4 "$case_dir/rank${rank}.log"
done
exit "$status"
