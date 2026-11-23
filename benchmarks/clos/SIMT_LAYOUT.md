# V2 separate SIMT layout

`DEEPEP_CLOS_V2_SIMT_LAYOUT=1` selects the new independent layout kernel for
fresh `ElasticBuffer.dispatch`. Its AIV count follows the resolved `num_sms`
(including the existing `num_sms=0` resolution to 32). This is now the default
for separate V2 layout. Set `DEEPEP_CLOS_V2_SIMT_LAYOUT=0` to use the original
layout and its original launch sizing. V1, cached dispatch, notify/dispatch
schedules and combine are unchanged. Explicitly enabling
`DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY=1` uses the fused path when the SIMT switch is
unset; explicitly setting both switches to `1` remains an error.

Rebuild both native and device libraries. The new native entry is
`get_dispatch_layout_v2_simt`, guarded by `v2_simt_layout_version=1`; the existing
layout tiling ABI is unchanged. Set the same switch on every rank before calling
dispatch. It is read per call, so a comparison can reuse one Buffer per rank.

## Data and ownership

- Each AIV owns a contiguous, disjoint token interval. It DMA-loads top-k tiles
  into two UB buffers; SIMT reads indices, deduplicates rank membership and
  accumulates private per-thread expert/rank histograms entirely in UB. Expert
  counts include every valid top-k entry; rank counts include each token once.
- Mask tiles are DMA-stored contiguously to their original GM output rows.
  Per-AIV histogram sums are DMA-stored into the original scratch row layout.
  Empty AIVs write zero counts and participate in `SyncAll<true>()`.
- After the local barrier, AIVs own disjoint aligned output count blocks. One
  strided DMA gathers each block of partial rows into UB; SIMT reduces them,
  then DMA stores the count vector. SIMT has no GM pointer or GM atomic access.
- Input and mask ping/pong events are drained only when produced. Histogram
  regions are disjoint from both pipelines; scratch/output reuse waits for
  completed readers/stores. The kernel has `__schedmode__(1)`.

For tile `T`, histogram lanes `L`, ranks `P`, experts `E` and top-k `K`:

| UB region | Bytes |
|---|---:|
| Two input tiles | `2 * align32(T*K*sizeof(topk_idx_t))` |
| Two mask tiles | `2 * align32(T*P)` |
| Private histograms | `L * (align8(P)+align8(E)) * 4` |
| Local reduced counts | `(align8(P)+align8(E)) * 4` |

The host and kernel share the capacity calculation. Tile and active lane counts
are powers of two up to 128, reduced to fit the platform UB budget; an unsupported
shape raises an error instead of silently changing the AIV count. The reduction
phase reuses the arena after all first-phase accesses complete. Padding belongs
to count storage; mask tails only write valid bytes.

## A/B run

The existing benchmark now accepts `--simt-layout-sweep --aiv N`. It runs
old/SIMT/SIMT/old, with fusion off and one Buffer per rank across precision and
timing. Both sides use the same dispatch core count and sender policy. Only the
layout implementation and its launch sizing change. No 1 GiB pages are enabled
with the following command (adapt the torchrun node/rendezvous arguments):

```bash
torchrun --nproc_per_node=8 benchmarks/clos/bench_fused_notify.py \
  --simt-layout-sweep --aiv 8 --no-huge-pages --flush-l2 \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 \
  --warmup 10 --samples 30 --output /tmp/v2_layout_ep8_aiv8
```

Use the existing EP16/32 launcher with the same script for other EP sizes, and
repeat with `--aiv 16` or `--aiv 32` to check core-count scaling. Each independent
script run creates one Buffer per rank. Precision covers random/masked/hot/empty
routes and handle reuse through cached dispatch/combine, with zero tolerance.
Use `--expand 1` in a separate run to check expanded consumers.

`summary.csv` separates layout, notify, communication and unprofiled complete
API mean; each capture's `samples.json` retains per-rank samples for per-iteration
slowest-rank aggregation and P95 analysis, while `report.json` records summaries.
New profiler captures must contain
`dispatch_layout_simt_kernel` for the SIMT mode; the benchmark rejects a missing
or mixed layout sequence. L2 eviction and rank alignment precede measurement.
Do not infer API gains by subtracting independently sampled kernel times.

CPU tests emulate the production VF/DMA bodies and check masks, counts, tails,
UB bounds and event credit balance. They do not validate CANN compilation,
asynchronous hardware ordering or NPU performance; those need the remote run.
