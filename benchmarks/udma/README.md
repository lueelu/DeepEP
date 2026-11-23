# UDMA environment microbenchmark

This diagnostic isolates fixed-size remote writes from MoE routing, notify,
aggregation, reduction and dynamic output allocation. It uses the **original
global Clos** `ShmemInstance` and `SenderT`, including 32 shared-Jetty QP slots,
peer ownership, SQ admission, per-batch CQE requests and SO completion markers.
It does not include the POD 3:1 adaptation or change any production operator.
It is an Ascend950/CANN 9.2.x benchmark, not a generic HCCL alltoall.

One process per rank creates **one session, one set of QPs and one maximum-sized
workspace** for the complete suite. Changing the mode, message size or WQE size
does not reinitialize SHMEM. The standalone build does not reinstall DeepEP.

## Build

Use the same CANN environment, matching Torch/torch_npu, pybind11 and SHMEM SDK
as the original global Clos installation. `SHMEM_ROOT` must identify the SDK
with its headers and shared libraries, not just a wheel's runtime libraries.

```bash
# Source the environment's CANN set_env.sh first.
export SHMEM_ROOT=/path/to/shmem/sdk
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
cmake -S benchmarks/udma -B build/udma-microbench \
    -DPython_EXECUTABLE="$(command -v python)" \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSHMEM_ROOT="$SHMEM_ROOT"
cmake --build build/udma-microbench -j2
```

Deploy the same source and build artifacts on all nodes. The runner checks
native/kernel hashes and suite parameters across ranks before SHMEM init.

## Run all scenarios with one initialization

For EP32 on two 16-device hosts, run this on both hosts with the appropriate
`NODE_RANK`. For four 8-device hosts, use `NNODES=4` and `LOCAL_RANKS=8` instead.
The hosts need access to node0's torchrun port and the next port (`MASTER_PORT+1`,
or an explicit `--control-port`). Parameters, SHMEM UID and result JSON use one
TCPStore server. No Gloo/HCCL process group is created; measured payload uses
UDMA exclusively. This avoids requiring all-to-all TCP connectivity for control.

```bash
set -euo pipefail
EP=32
NNODES=2
LOCAL_RANKS=16
NODE_RANK=0                     # Set separately on each host.
MASTER_ADDR=192.0.2.10          # Replace with node0's reachable address.
MASTER_PORT=29600
RUN_ID=udma_env_01              # Same on every host; change for another run.
test "$EP" -eq "$((NNODES * LOCAL_RANKS))"
mkdir -p "benchmark-runs/$RUN_ID"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m torch.distributed.run \
    --nnodes "$NNODES" --nproc-per-node "$LOCAL_RANKS" --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    benchmarks/udma/bench.py --ep-size "$EP" \
    --native-dir build/udma-microbench \
    --aiv 32 --peers-per-worker 4 --wqes-per-visit 32 \
    --bytes-per-peer 16777216 --chunk-bytes 262144 5632 \
    --modes pair permutation alltoall --pair-scope spokes \
    --warmup 3 --iterations 10 --check-validator \
    --output "benchmark-runs/$RUN_ID" \
    2>&1 | tee "benchmark-runs/${RUN_ID}_node${NODE_RANK}.log"
```

The default EP32 suite has **188 cases**:

| Mode | Traffic | Patterns before byte/chunk size combinations |
|---|---|---:|
| `pair` | One directed pair at a time: rank0 to every peer, then every peer to rank0 | 62 |
| `permutation` | Each rank sends to `(rank + delta) % EP`; one sender per receiver | 31 |
| `alltoall` | Every rank sends the same bytes to each remote rank | 1 |

Both chunk sizes transfer exactly 16 MiB per active edge, including a shortened
final WQE. 5632 bytes corresponds to one BF16 row at hidden=2816; this benchmark
has no tokens, experts or topk. These are transport bytes, not MoE logical bytes.
`--pair-scope all` checks every directed pair (992 for EP32), making the suite
2048 cases with the default two chunk sizes. `--deltas 1 8 16` limits permutation
patterns. `--bytes-per-peer` accepts multiple sizes within the same allocation.
Use `--dry-run` to print the complete plan without Torch/NPU initialization.

The default heap uses normal SHMEM allocation. `--huge-pages` selects the same
1 GiB workspace allocation path as global Clos. Match this setting between
environments; it is fixed for the lifetime of a session.

## Timing, correctness and outputs

Each iteration prepares fresh deterministic source values and poisons active
receive slots outside the timer. The value depends on source, destination,
iteration and word offset. Every received 64-bit word is verified after transfer;
32-byte alignment ensures there is no unchecked byte tail. `--check-validator`
first flips one received word on every rank in a separate alltoall and requires
exactly one mismatch on each rank. All following samples must have zero errors.

The transfer kernel performs:

1. Reset incoming markers and run the original all-rank/all-AIV entry barrier.
2. **Start timing**, initialize sender headers, submit payload and SO markers.
3. Drain outgoing QPs (`Quiet`), wait for active incoming markers, join local AIVs.
4. **Stop timing**, copy timing records, then run the exit all-rank barrier.

Preparation, verification, Python/TCPStore and both global barriers are excluded.
Header setup, payload posting, final outgoing completion, incoming waits and
the local AIV join are included. This is completed-transfer throughput, not
doorbell submission bandwidth. Preparation touches the active payload; no L2
flush is performed. It is not a cold-cache measurement.

Each device's elapsed time is `max(core_end) - min(core_start)`. Each sample uses
the maximum elapsed time among active ranks; card clocks are never subtracted
across devices. The entry barrier aligns starts but is not a synchronized global
clock. Mean latency is the mean of these per-sample maxima. Aggregate bandwidth
is total actual remote payload bytes divided by mean latency; per-sender bandwidth
divides that by the number of active senders. Pair-mode per-sender bandwidth is
the single directed link's rate, with no division by idle ranks. Alltoall
per-sender bandwidth is `(EP - 1) * bytes_per_peer / time / 1e9` (time in seconds).
No self bytes, marker bytes or synthetic topk multiplier enter the numerator.

On rank0's host:

- `plan.json`: parameters, allocation, device/rank mapping and binary hashes.
- `summary.csv`: per-case mean/median/P95/max latency, aggregate and per-sender
  GB/s, slowest rank and validation errors. Updated after each completed case.
- `samples.jsonl`: per-rank stage and total timings for every measured iteration.
- `report.json`: complete summary, written only after successful verification
  and session destruction.

On each host, `rankN.jsonl` contains raw per-core timestamps, active-sender flags,
all warmup/measured samples and mismatch counts. Stage maxima may come from
different cores and **must not be added together**. Incoming wait starts after
outgoing quiet here; it is not the full incoming transfer duration. Empty cores
and idle ranks still join all collectives, but idle ranks do not dilute pair
bandwidth. Existing rank logs are never overwritten.

Compare the same shapes, binary/SDK versions, heap setting and rank/device map
in both environments. Pair degradation points toward a specific path; degradation
only under permutation/alltoall suggests contention; a small-WQE-only difference
points toward submission/completion overhead. These are diagnostic clues, not
proof of a hardware fault.

## Memory and synchronization contract

Let `P=EP` and `C=max(bytes_per_peer)` (multiple of 32). Every rank allocates:

| Byte range | Owner / use |
|---|---|
| `[0, P*512)` | Incoming completion markers, one 512-byte slot per source |
| `[P*512, P*512 + P*C)` | Local source rows, one capacity-sized slot per destination |
| `[P*512 + P*C, P*512 + 2*P*C)` | Receive rows, one capacity-sized slot per source |

Source `s` alone writes receiver `d`'s receive slot `s`. Existing rank-order
worker ownership gives each edge one writer and one QP lane. AIV count and peer
grouping are fixed across cases. The original sender's UB arena is reused:
2048 bytes for 32 WQEs, 1024 for request descriptors, 1024 for two header templates
per peer (maximum eight peers). The later 64-byte timing record reuses this arena
only after WQE and completion-marker reads have drained. Payload itself stays
in registered GM; no per-WQE staging copy is introduced.

Every launched AIV participates in entry/exit collectives, including pair-test
idle ranks. SO markers establish payload arrival; outgoing quiet releases source
memory. A post-verification host barrier prevents a fast rank preparing the next
iteration while another rank still validates the old receive data. Standalone
host timing/error arrays are allocated once, with disjoint per-core/lane owners.
Native capacity bounds keep all address products within checked 64-bit limits.

Offline tests: `python -m pytest tests/test_udma_microbench.py -q`.
