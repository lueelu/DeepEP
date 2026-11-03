#!/usr/bin/env bash
# Compile all production NDR LL translation units with tracing absent; no NPU execution.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/../../scripts/ll_timer_env.sh"
set -u
OUT=$(mktemp -d "$LL_TIMER_ROOT/.tmp/ll-timer-off-XXXXXXXX")
SDK="$LL_TIMER_ROOT/third-party/shmem/install/shmem"
test -f "$SDK/lib/libshmem.so"
HNS_PI_ON_CHIP=$(python3 "$LL_TIMER_ROOT/scripts/shmem_sdk_identity.py" "$SDK")
SOURCES=(dispatch/low_latency.cpp combine/low_latency.cpp)
# NDR has one compiled transport capability. Runtime MTE is selected at runtime,
# so compiling with RDMA absent would not test the production configuration.
for src in "${SOURCES[@]}"; do
  bisheng -O2 -std=c++17 -fPIC -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false \
    -DCATLASS_ARCH=3510 -DUSE_SIMT -DL2_CACHE_HINT -DDEEPEP_LL_RDMA=1 -DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1 \
    "-DACLSHMEMI_HNS_1825_PI_ON_CHIP=$HNS_PI_ON_CHIP" \
    -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/asc/include" \
    -I"$SDK/include" -I"$SDK/src" -I"$SDK/src/device" -I"$SDK/src/device_simt" \
    -c "$LL_TIMER_ROOT/csrc/kernels/$src" -o "$OUT/${src//\//_}.o"
done
printf 'PASS all %s production NDR LL translation units compile with timer OFF: %s\n' "${#SOURCES[@]}" "$OUT"
