# Fused layout/notify experiment

Opt-in fresh V2 UDMA dispatch experiment. Default dispatch is unchanged:

```bash
export DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY=0  # separate layout, then notify
export DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY=1  # layout inside notify
```

Set the same value on every rank before a fresh dispatch. The switch is read
per invocation; changing it does not require rebuilding Buffer or SHMEM.
Cached dispatch reuses its handle and does not rebuild layout in either mode.
V1 and standalone `get_dispatch_layout` retain their original implementation.
The opt-in path rejects non-UDMA V2 calls. Rebuild **both native and device
libraries**: the dispatch tiling ABI changes in this branch.

## Implementation contract

- Notify launches `num_sms` AIVs, exactly as dispatch. One AIV owns one existing
  channel and its contiguous token interval. Empty channels participate in all
  original local and cross-rank rendezvous.
- AIV/MTE2 loads top-k tiles to UB. SIMT builds rank membership bytes, stable
  per-peer plan entries and per-token send heads. Private expert histograms
  avoid competing increments for hot experts; their sums are published per channel.
- Each rank reduces channel counts into the original rank/expert count tensors,
  then uses the existing notify exchange, prefix construction and MTE3 host-count
  publication. Send policies, payload transport and combine kernels are unchanged.
- Membership bytes are still written to GM for handle/cached compatibility.
  The small count tensors are still materialized and read by existing publication
  code. This experiment removes the separate layout launch and route-mask reread
  during plan construction, not every metadata copy.
- Per-channel expert scratch is `num_sms * align8(experts) * 4` bytes in ordinary
  GM, owned by the dispatch call and recorded on its execution stream. It does
  not change the SHMEM workspace layout.

UB ledger (byte offsets; reused by later notify stages):

| Region | Begin | Capacity |
|---|---:|---:|
| Rank counts | 0 | 4096 |
| Rank membership tile | 4096 | 65536 |
| Existing eight-token chunk prefixes | 69632 | 36864 |
| Top-k tile | 106496 | 32768 |
| Private expert histograms | 139264 | remaining platform UB |

Tile size is capped at 128 tokens, rounded down to eight tokens, and bounded by
top-k bytes and world size. The final tile is bounded to actual tokens. Histogram
shards are capped at 32 and shrink with expert count. Unsupported histogram sizes
fail on the host before launching notify. The enable bit is separate from the
top-k pointer so zero-token ranks still execute fused count generation.

## Build and run

Use the toolchain and pinned SHMEM dependency in [build instructions](../../docs/build.md).
From this branch, with CANN and matching Torch/torch_npu active:

```bash
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

Single-machine EP8, exactly eight AIVs per dispatch/notify:

```bash
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_fused_notify.py \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 --expand 0 \
  --warmup 10 --samples 30 --flush-l2 --output /tmp/fused_notify_ep8
```

The script defaults to 1 GiB huge pages; use `--no-huge-pages` only when necessary
and keep it identical for both modes. L2 flushing is opt-in; when enabled it
finishes **before** the per-sample all-rank device barrier on the Buffer's existing
stream. The outer barrier is excluded from layout/notify/communication timing.
Use a new output directory for each run. Multi-node runs use the normal torchrun
`--nnodes`, `--node_rank`, `--nproc_per_node`, `--master_addr`, `--master_port`
arguments. The separate control and SHMEM ports default to 29871 and 29872.

Each process creates Buffer once, checks precision, then measures both weight
modes with switch order **0, 1, 1, 0**. Each phase keeps the same routing and
automatic send policy; the policy is recorded and checked for A/B consistency.
Baseline standalone layout retains its original core count; only the fused
layout phase is constrained to the eight notify cores.

Precision checks compare against the original path and independently check
CPU-derived rank memberships, rank prefixes, expert counts, channel counts,
stable plan entries and send heads. They also compare payload/weights and reuse
the resulting handle for cached dispatch and combine. Routing fixtures cover
random, masked `-1`, hot experts, all-invalid routes, and a zero-token rank.
To check token tails and expanded dispatch, run separately, each with one Buffer:

```bash
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_fused_notify.py \
  --tokens 4099 --expand 1 --precision-only --output /tmp/fused_notify_tail_expand
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_fused_notify.py \
  --tokens 3 --precision-only --output /tmp/fused_notify_empty_channels
```

This script covers BF16, FP32 routing weights, and expert alignment 1. FP8/scales,
graph capture and asynchronous fixed-capacity dispatch need separate device
qualification before enabling this experimental path for those uses.

## Reports and interpretation

Each rank writes `precision.json`, `summary.csv`, `report.json`, raw profiler
exports and per-phase `samples.json`. `exit.json` with code 0 is written only
after all ranks finish. Collect every rank's directory and torchrun exit code.
Missing kernels, extra layout launches in fused mode, invalid durations and
wrong sample counts fail the run. Two leading/trailing profile guards are
excluded from the requested sample count.

- Primary comparison: separate **layout + notify** versus **fused notify**, plus
  the unprofiled complete API wall time. Fused notify includes layout work, so
  notify-only comparison with baseline is misleading.
- `layout_notify_us` adds stages on each rank/sample **before** taking that
  sample's slowest rank, then averages samples. Other stage means likewise use
  per-sample slowest rank; their separately aggregated values cannot simply be
  added to reconstruct API time.
- `payload_gbps` = all ranks' rank-deduplicated BF16 payload bytes / EP /
  communication-kernel time / 1000 (time in microseconds). It includes self,
  excludes indices/weights and uses decimal GB/s. Expand is reported separately.
- API wall timing and kernel profiling use separate passes; the API wall metric
  includes host submission and completion waits, but excludes the outer barrier
  and L2 flush. It is not a device-only API span.

No NPU performance or precision claim follows from local CPU checks alone.
