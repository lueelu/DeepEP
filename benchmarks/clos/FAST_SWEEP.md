# Full-Clos single-initialization strategy sweep

Build and install this branch on every participating machine first. The private
Host tuning interface requires the rebuilt `_clos_C`; replacing only the Python
script is insufficient. Use the SDK pinned in `dependencies.lock.json`.

This performance-only runner uses one Buffer/SHMEM initialization per rank across
all cases. It measures BF16, random distinct experts per token, expand=0,
cached dispatch and unweighted combine; `--include-weighted-combine` also measures
combine returning FP32 top-k weights. No numerical checks are performed;
reports explicitly contain `precision_checked: false`. The ordinary
`bench_v2_bandwidth.py` checks its default policy; qualification of a manually
selected winner also requires accuracy checks using that exact configuration.

Example: run this on every machine, with identical settings except `NODE_RANK`.
Set `NNODES` and `LOCAL_RANKS` to match the actual machines. EP is their product.
`ranks-per-group` describes logical contiguous rank groups; verify it matches
the grouping intended for the experiment. It defaults to min(8, EP).

```bash
set -euo pipefail
NNODES=8
LOCAL_RANKS=8
NODE_RANK=0
MASTER_ADDR="node0_reachable_ip"
EP=$((NNODES * LOCAL_RANKS))
RUN_ID=clos_sweep_01                 # same on all machines; use a new directory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SHMEM_UDMA_CONNECT_TIMEOUT_MS=600000

python -u -m torch.distributed.run \
  --nnodes "$NNODES" --nproc-per-node "$LOCAL_RANKS" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29600 \
  benchmarks/clos/bench_fast_sweep.py \
  --ep-size "$EP" --ranks-per-group 8 --experts 256 \
  --case 8192,2816,8,32 --case 16384,7168,6,64 \
  --control-port 29610 --shmem-port 29620 \
  --huge-pages --no-flush-l2-cache --quick \
  --output "$RUN_ID/node$NODE_RANK" 2>&1 | tee "${RUN_ID}_node${NODE_RANK}.log"
```

Repeat `--case TOKENS,HIDDEN,TOPK,AIV` to add shapes. Defaults are the two cases
above. `--plan-only` prints candidates without torch/NPU/SHMEM; `--smoke` picks
one legal candidate for each enabled order. Remove `--quick` for all four
existing orders: rank, group, group_staggered, group_round_robin. Quick mode uses
rank and group_round_robin. Neither mode adds a new Device scheduling algorithm.
Candidates cover legal peer counts, W/Q=1/1,1/2,2/1 and visit=4/8/16/32. There is
no group_window or adjustable lane ratio in this full-Clos version. The existing
automatic configuration is measured first when legal and included by the visit
filter. The default candidate list depends on EP, AIV and group size.

`--flush-l2-cache` enables the same 256 MB eviction helper used by the ordinary
Clos benchmark. Default: no eviction. Default: 1 GiB pages; use
`--no-huge-pages` if the environment is not configured for them. All ranks must
use the same settings. Remove inherited experimental `DEEPEP_CLOS_*` overrides.
For an A/B run, replay the same strategy with and without
`--combine-parallel-self-copy`, using separate output directories. The flag
passes `DEEPEP_CLOS_COMBINE_PARALLEL_SELF_COPY=1` for unweighted V2 peer combine.
When the strategy leaves spare AIVs, they take all self-copy channels. When
there are no spare AIVs, each core keeps its original channel.
`--combine-self-copy-after-submit` keeps each AIV's original self-copy channel
and starts it after posting remote requests and terminal markers. This mode
uses `DEEPEP_CLOS_COMBINE_SELF_COPY_AFTER_SUBMIT=1` and is exclusive with the
spare-AIV mode. It supports ordinary weighted and unweighted V2 peer combine;
token scheduling, fused aggregate, expanded inputs and pipeline paths are excluded.
Both switches default off. Spare-AIV self-copy remains unweighted-only.

To sweep all three self-copy modes without reinitializing SHMEM, replace either
single-mode switch with:

```bash
--combine-self-copy-modes default parallel_self_copy self_copy_after_submit \
--case 8192,2816,8,8 --case 8192,2816,8,20
```

Launch once per EP. Both AIV settings, all sending strategies, all three combine
modes and the confirmation rounds share one Buffer per rank. EP4/8/16/32 still
require four separate launches because communication group membership changes.
Omit `--quick` to include all four peer orders. Cached dispatch is measured only
once per sending strategy; combine candidates include the self-copy mode in their
IDs. If no spare AIV exists and default mode is included, the equivalent
parallel-self-copy candidate is skipped. Each switch drains previous device work
before changing the host flags; the next invocation starts with the usual Buffer
barrier. This uses the existing per-call native flags and does not change kernels.
Mixed-mode runs require the Host and Device build containing both modes
(`371039b` or later); copying this script onto an older build is insufficient.
The top candidates are selected across modes together. `best_configs.json` and
`summary.csv` record the winning self-copy mode as well as the sending strategy.

For the weighted ordering experiment, rebuild/install the updated Host and Device
code, then use the existing launcher with EP8 and the following runner arguments:

```bash
--ep-size 8 --ranks-per-group 8 --experts 256 \
--case 8192,2816,8,32 --strategies group_p1_w1_q1_v32 \
--include-weighted-combine \
--combine-self-copy-modes default self_copy_after_submit \
--huge-pages --no-flush-l2-cache --warmup 2 --samples 8 --final-samples 30 --top 2
```

The two copy modes and both combines share one Buffer; cached dispatch is measured
once. Exact `--strategies` IDs prevent changing the sender configuration in an A/B
comparison. They cannot be combined with quick/smoke/replay selection. Weighted
combine omits the unsupported `parallel_self_copy` candidate rather than reporting
it as effective. Replay can include the optional weighted operator and restores its
saved copy mode. The runner remains performance-only: validate both modes against
an independent reference for BF16 payload and FP32 weights before accepting gains.

Weighted after-submit keeps the original self-copy channel ownership. Remote SIMT
weights finish their per-AIV V_S wait before that AIV copies self; payload AIVs post
UDMA first. Self stores target this rank's `(rank * token_capacity + token)` slot,
disjoint from incoming contributors' slots. Input GM stays live through Quiet.
No workspace offsets, strides, alignment or capacity change. Self-copy reuses the
existing UB region after the MTE3-to-MTE2 fence; its payload/weight slots and
metadata tile retain their original offsets and event drains. All launched AIVs,
including empty workers, still reach WaitIncoming and the existing final collective
for SIMT weights. This experiment does not remove a barrier or add stage timers.

The script scopes the separate pre-combine notify to profile capture, preserving
the PR's profiling-only behavior. Every call starts with a device Buffer barrier;
L2 eviction, when enabled, follows the barrier and precedes the measured API.

Each candidate gets 2 warmups, 8 measured calls and trailing guards. The top 3
per operator are rerun forward and reverse with 30 measured calls. Winners
maximize the lower bandwidth of the two confirmation runs. These are measured
set winners, not proof of a global optimum. Tune counts with `--warmup`,
`--samples`, `--final-samples`, `--top`, and restrict visits with `--visits`.

Outputs under node0's output directory:

- `report.json`: complete rankings and confirmation results for every case.
- `best_configs.json`: each operator's selected strategy and both repeats.
- `summary.csv`: shape, strategy, confirmation bandwidth and communication time.
- `rank*/`: plans, library hashes, profiler CSV/trace, raw samples and exit status.

Bandwidth is sum(rank-deduplicated payload bytes including self, plus FP32 top-k
weights for weighted combine) divided by
sum(rank mean communication microseconds), divided by 1000, in decimal GB/s.
Notify and final reduce are recorded separately and excluded. The communication
kernel includes its local work and completion waits; this is not isolated link
bandwidth. The numerator excludes routing indices and protocol overhead. Weighted
combine adds `topk * 4` bytes per row; `logical_row_bytes` and
`bf16_payload_row_bytes` record the distinction. Weights are returned metadata,
not multipliers of the BF16 payload. There is no
aggregate stage in these expand=0 calls. Reported rank bandwidth ranges and
forward/reverse repeat ranges are distinct fields.

For replay, copy node0's `report.json` to every machine and add
`--replay-report /path/to/report.json` to the same command with a fresh RUN_ID.
Keep the cases, EP, experts, seed, group size, L2 and page options unchanged.
Only each operator's winning configuration is rerun, still with one initialization.
Replay restores the saved combine mode, including the report-level flags in older
reports. To compare the same strategy in another mode, explicitly pass one mode
with `--combine-self-copy-modes MODE` (or either legacy single-mode flag).
Multi-mode scanning and winner replay are separate operations.
Accept a run only when every rank has exit code 0, complete=true and
buffer_initializations=1. Missing/reordered kernels fail the run.

The tuning session prebuilds a QP-mask union; this can make its first initialization
slower and consume more channel resources than the ordinary path. It avoids
repeating that cost per candidate. Only one Buffer may be active per process;
concurrent API use during strategy switching is unsupported. Routing handles
are rebuilt per shape/AIV and released after all device work finishes. The
symmetric workspace reserves the maximum size hint across all cases. Production
defaults and kernel completion protocols are unchanged.
