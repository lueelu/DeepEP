# Clos performance and diagnostics

This guide describes the current implementation and retained comparison tools.
For the independent SIMT layout, see [SIMT_LAYOUT.md](SIMT_LAYOUT.md); for SQ
ownership, sparse CQEs and completion guarantees, see
[SPARSE_CQE.md](../../tests/clos/SPARSE_CQE.md).

## Current defaults

- V2 fresh dispatch uses independent SIMT layout, following resolved `num_sms`.
  `DEEPEP_CLOS_V2_SIMT_LAYOUT=0` selects the original layout baseline.
- Supported peer-UDMA paths enable fresh channel metadata, double buffering,
  skip-init, direct output and early slots. Cached weighted dispatch enables
  pack/direct/omit; omission applies only to unused non-expanded top-k output.
- Payload submission precedes sender-local copying, including when the legacy
  overlap switch is `0`. Compact combine self-copy is enabled on supported paths.
- Layout/notify fusion, host preparation, deferred plans and timing diagnostics
  remain off. The library does not enable 1 GiB huge pages. Callers independently
  choose `reduce_num_sms`; zero inherits the resolved communication core count.
- Automatic metadata selection retains supported fallbacks for graph capture,
  token scheduling, shape limits and cached handles without private routing data.
  Explicit unsupported opt-ins fail rather than silently changing the request.

Set matching options on all ranks. Benchmarks can explicitly override library
settings: record effective flags, launch policy, kernel names and core counts.
Rebuild **both Host and device libraries** after updating this branch; tiling,
output-publication and diagnostic layouts must match the loaded kernels.

## Data ownership and completion

Channel metadata is packed from existing plan/send-head data after payload
submission, then sent by channel and field. Direct output writes source indices,
retained top-k indices and weights to final tensors using the existing address
publication round. Unused cached top-k output may be omitted, but internal routing
indices and format conversion remain. Sparse output slots retain required
initialization unless the selected path proves full coverage.

Payload WQE preparation reads staged UB indices. Pending short spans retain their
indices before a refill; a refill does not force an extra flush or change visit,
QP ownership, batch capacity or peer order. Cached whole-channel traversal uses
exact channel row counts. Token scheduling retains its separate GM-index path.
Source-handle construction remains the single-pass implementation.

Local-copy UB is separate from WQE storage. Early slots only advance work whose
inputs are already available; receiving source data and building the final Source
handle still respect completion. Compact combine self-copy changes local layout,
not the caller's independent reduce-core choice. Final SO flags, quiet, receive
waits and SIMT completion remain in force; posting a WQE does not release its GM
source. See the sparse-CQE document for pressure checkpoints and shared-SQ rules.

## Comparison entry points

Use the pinned dependency and toolchain in [build instructions](../../docs/build.md).
All examples use a fresh result directory, disable 1 GiB pages and flush L2.
For EP16/32, replace the torchrun launcher arguments with the site's topology;
keep the same script arguments on every rank.

Independent old/SIMT layout, same communication core count, one Buffer per rank:

```bash
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_fused_notify.py \
  --simt-layout-sweep --aiv 8 --no-huge-pages --flush-l2 \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 \
  --warmup 10 --samples 30 --output /tmp/layout_abba
```

For separate-layout/fused-notify comparison, omit `--simt-layout-sweep`.
`--pipeline-sweep` additionally compares fused host preparation and deferred
plans. The baseline in this comparison explicitly selects the original layout.
`--host-timing` adds a separate Host diagnostic pass. `--expand 1` and
`--precision-only` support expanded and tail-routing checks.

Fresh/cached channel-metadata comparison, with one Buffer per rank:

```bash
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_dispatch_stages.py \
  --cores 8 --cached-channel-modes 0 --metadata-modes 0 1 \
  --trace-samples 0 --flush-l2 --no-huge-pages \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 \
  --warmup 10 --samples 30 --output /tmp/channel_abba
```

This replaces the removed `bench_channel_metadata.py` forwarding wrapper with
its exact preset arguments. Keep any additional options after those presets,
as before. The following options remain available on the shared entry point:

| Option | Purpose |
|---|---|
| `--cores 8 32` | Compare communication core counts |
| `--cached-channel-modes 0 1` | Compare cached window/whole-channel traversal |
| `--metadata-modes 0 1` | Compare legacy/channel metadata |
| `--metadata-tile-tokens 0 32 64 128 256` | Automatic or explicit packing tiles |
| `--overlap-modes send all` | Compare optional slots, initialization and buffering; both send first |
| `--metadata-output-modes off pack direct omit pack_direct all` | Independent cached-pack/direct/omit combinations |
| `--dispatch-plans auto zeng` | Retained constrained EP8 policy experiment |
| `--trace-samples 5 --combine-diagnostics --reduce-cores 48` | Separate dispatch/combine diagnostic passes |
| `--compact-self-modes 0 1` | Compare scatter/compact combine local copies |

These presets are ablations, not an assertion that every production default is
active. Use `--help` for constraints and defaults. `combine_queue_diagnostics.py`
is the shared script's combine helper, not a separate launch entry point.

## Interpreting results

The scripts run precision checks and forward/reverse comparisons with one Buffer
per rank. Both weight modes use the same routing across variants. L2 eviction
finishes before each sample's all-rank barrier on the existing Buffer stream;
neither preparation step is part of the measured API or communication duration.

- Formal API wall time is measured without profiler, through output completion.
  Kernel durations come from a separate profiler pass. Do not subtract these
  independently sampled measurements to infer Host or synchronization cost.
- Aggregate each stage by the slowest rank **per iteration**, then report mean,
  P50/P95 and both round results. For layout+notify, sum within each rank/sample
  before aggregation. Independently aggregated stages cannot reconstruct API.
- Logical bandwidth is rank-deduplicated BF16 payload bytes / EP / communication
  time / 1000 for microseconds and decimal GB/s. It includes self and excludes
  index/weight bytes. Expanded local work is reported separately.
- Device diagnostics default off. Their counters and synchronization can perturb
  overlap; use them to locate WQE preparation, queue waits, local copies and
  handle work, not to claim formal bandwidth. `--host-timing` similarly isolates
  Python/native preparation, count waits, output allocation and submission.
- Retain raw per-rank samples and both rounds, including long tails. Missing or
  mixed kernel sequences and invalid durations fail profiling validation.

Each rank exports precision status, summaries/reports, profiler captures and raw
samples; diagnostic modes add trace/overhead records. Collect all rank outputs
and launcher exit codes. The stage benchmark's failed-worker handler retains
its traceback and avoids collective teardown that could mask the original error.

## Validation

Regression tests under `tests/clos` cover production VF/DMA bodies, sparse
routes, direct output addresses, pending-index retention, local-copy ownership,
UB capacity, event credits, queue completion and profiler accounting. API tests
cover launch policy and default routing. Use the focused tests for the changed
path, preserving zero tolerance and intentionally corrupted-reference checks.

CPU emulation and source checks do not establish CANN compilation, asynchronous
hardware ordering or NPU performance. Device qualification should include empty
and tail routes, changed weights, continuous reuse, both index widths, local and
remote outputs, cached handles and expanded consumers. Diagnostics have narrower
supported paths: combine tracing rejects token scheduling, early-single/pipeline,
integrated expanded inputs and graph capture. Validate FP8/scales, graph capture
and async fixed-capacity paths separately before drawing conclusions about them.
