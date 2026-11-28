# Clos performance and diagnostics

This guide describes the current implementation and retained comparison tools.
For the independent SIMT layout, see [SIMT_LAYOUT.md](SIMT_LAYOUT.md); for SQ
ownership, sparse CQEs and completion guarantees, see
[SPARSE_CQE.md](../../tests/clos/SPARSE_CQE.md).

## Current defaults

- EP64 automatic plans with 8--64 AIVs use rank order. The G16 weighted-combine profile at
  32--64 AIVs and 8192--12288 payload bytes per row uses P4/W1/Q1/visit8
  (16 payload senders), replacing group-round-robin P1 without changing visit,
  weight preference or local-copy mode. Eight-core plans remain P8/W1/Q1/visit32.
- EP128 with exactly 8 communication AIVs uses rank/P16/W1/Q1/visit32.
  Other EP128 core budgets retain their previous plans. P16 is supported only
  for rank order; existing P<=8 and token sender windows retain their sizes.
  Fresh/cached dispatch and combine select a separate 16-peer sender. Its Q1
  live UB prefix is 23,200 bytes for dispatch and at most 21,664 bytes for
  combine (including merged-row scratch), followed by existing local-work
  storage. Output row mapping and final completion semantics are unchanged.
  All launched AIVs still participate in entry/final sync.
  To compare with the former G16 8-core plan in one tuning Buffer, reserve
  `rank:16:1:1;group_round_robin:2:1:1` before initialization and compare
  `rank_p16_w1_q1_v32` with `group_round_robin_p2_w1_q1_v32` using
  `bench_fast_sweep.py --strategies`. Keep the actual metadata/weight modes
  identical and validate outputs before measuring; this is not a performance
  guarantee and requires rebuilding Host and device libraries.
- V2 fresh dispatch uses independent SIMT layout, following resolved `num_sms`.
  `DEEPEP_CLOS_V2_SIMT_LAYOUT=0` selects the original layout baseline.
- Automatic V2 non-expanded BF16 plans for EP8/16/32 use rank ordering in
  the existing low-core tier (`8 <= num_sms < 16`). P/W/Q, visit and selected
  copy/weight modes remain unchanged; EP8 weighted combine retains UDMA at
  8 cores even though rank ordering frees one worker slot. Other core tiers,
  EP sizes and explicit benchmark/manual policies retain their own selection.
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
| `--compact-self-modes 0 1` | Legacy local-copy comparison; ordinary V2 UDMA combine now reads self input directly |

These presets are ablations, not an assertion that every production default is
active. Use `--help` for constraints and defaults. `combine_queue_diagnostics.py`
is the shared script's combine helper, not a separate launch entry point.

### Combine preparation and copy A/B

Ordinary V2 peer-schedule UDMA combine now always reads local payload and weight
directly from its input tensors during reduction (`combine_direct_self=1`).
It does not stage either local field in SHMEM. Reduce cores remain independent
of dispatch/combine cores. There is no opt-out switch; compare revisions for
this change. Early-single, aggregate pipeline, token scheduling and legacy
monolithic paths retain their existing behavior. Ordinary expanded combine is
eligible only after aggregation has fully materialized its input.

On the direct-self path, `COMPACT_SELF` and `SELF_DOUBLE_BUFFER` settings do not
restore local copies; effective `compact_self_rows` and
`combine_self_double_buffer` are zero. Rebuild Host and kernel together.

Two independent switches default to `0` and are read on each combine call:

| Environment variable | When set to `1` |
| --- | --- |
| `DEEPEP_CLOS_COMBINE_WQE_FAST` | Specialize ordinary contiguous-row WQE construction; single-span batches skip lookup, multi-span batches use binary search |
| `DEEPEP_CLOS_COMBINE_SELF_DOUBLE_BUFFER` | Pipeline legacy compact local copies through two 64 KiB UB slots; inactive with direct self reads |

Compare `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` in forward and reverse order with
one Buffer per rank. Rebuild both Host and kernel because the tiling grows.
Verify the effective `combine_wqe_fast` and `combine_self_double_buffer` fields;
unsupported paths retain the baseline. WQE fast mode excludes token scheduling,
early-single and aggregate pipeline; double buffering requires compact self rows.
Neither switch changes sender policy, P/W/Q, visit, WQE batch size, weight
transport, local row ownership, final flags or quiet. Do not change compact-self
or reduce cores between groups. Run zero-tolerance precision before timing;
report communication and full API mean/P50/P95 separately, with diagnostics off.
Use an independent diagnostic pass for WQE build/submit, local copy, incoming
wait and quiet. CPU equivalence/event tests do not establish NPU ordering or
performance; both still require device validation.

## Interpreting results

### Combine dense payload batches

`DEEPEP_CLOS_COMBINE_PAYLOAD_MAX_ROWS` is read on each combine call. When unset,
the maximum is `4` for payload rows of at most 8192 bytes (inclusive), `2` for
rows above 8192 and below 32768 bytes, and `1` for rows of at least 32768 bytes. The byte count
is hidden size times payload element size, excluding weights and metadata.
For BF16 this means H<=4096 uses 4, 4096<H<16384 uses 2, and H>=16384 uses 1.
Explicit integers from `1` through
`32` override this default; set `1` to compare with the single-row builder.
Values above one enable dense remote payload slots for ordinary non-expanded
V2 peer-scheduled combine with aligned payload strides and original dispatch
send-channel prefixes. Each contributor retains its capacity-sized SHMEM
partition, but destinations use peer-local row ordinals rather than original
token numbers. The source peer interval is already contiguous: no payload
repacking or src_idx DMA is needed. SIMT builds batches from these continuous
intervals even when the original token numbers are sparse. Each WQE
is capped at 64 KiB, or one unchanged row if a row itself exceeds that size.
It never merges across descriptor spans, visits, peers/QPs, worker partitions,
the original 32-logical-row submission batch. Unsupported paths
retain individual-row WQEs, including interleaved legacy payload/weight WQEs,
native callers without send prefixes, expanded input, early-single, pipeline
and token schedules. Their effective row limit is reported as 1.
The default compact-weight path is supported; weight transfers are unchanged.

With coalescing enabled this builder takes precedence over `COMBINE_WQE_FAST`.
P/W/Q, peer order and logical visit budgets are unchanged. SQ occupancy, head
updates and diagnostic WQE counts use the reduced physical count. Final flags,
quiet and direct self reads are unchanged. Reduce loads the original send-prefix
table contiguously into UB, then maps token/channel ordinals to dense payload
slots without changing contributor accumulation order. Dispatch and reduce core
counts remain independent. The payload reduction tile reserves space for this
aligned table; weight reduction reuses UB after payload reduction drains.
The count retains its separate 32-byte UB reservation; sending never stages
payload in UB. SHMEM sizing, offsets and public handle format are unchanged.
Rebuild Host and kernel together; the existing row-limit field now selects the
dense layout for values above one. Rows=1 retains the old token-addressed path.

Compare `1/2/4/8/16/32` using the same Buffer, routes and other settings; setting
the environment before each call is sufficient outside graph capture. Validate
zero-tolerance output first, then measure communication and full API separately.
Use an independent diagnostic pass to compare physical WQE counts and build
time, plus reduce time and complete API time. Prefix staging and changed reduce
locality can offset sender savings; no performance gain is assumed. H6144 now
exercises dense payloads by default with two rows per WQE; set the limit to 1
for the token-addressed baseline.

The existing `bench_fast_sweep.py` entry point also accepts a runtime-only
combine sweep (do not combine this with `--small-rank-sweep`):

```bash
# Use the existing torchrun/multi-node launcher with matching WORLD_SIZE.
python benchmarks/clos/bench_fast_sweep.py --ep-size 16 \
  --case 4096,2816,8,8 --case 4096,6144,8,8 \
  --combine-payload-max-rows 1 2 4 8 16 32 \
  --reduce-cores 48 --warmup 3 --final-samples 30 --output payload_sweep_ep16
```

Use `--plan-only` to inspect the matrix without a device. EP4/8/16/32 and
8/32 communication cores are supported; repeat `--case TOKENS,HIDDEN,TOPK,AIV`
to restrict shapes (topk8, experts256, BF16, expand0). With no `--case`, use
the existing T4096/8192, H2816/4096/6144, 8/32-core matrix. This mode measures
only unweighted and weighted combine, freezes default rank/P/W/Q, disables
1 GiB pages and L2 flushing, and initializes one Buffer per rank for the entire
EP session. Preparation dispatch is not a measured candidate. Only explicitly
requested limits are measured; include `1` for a single-row baseline. Each
candidate gets zero-tolerance checks and
forward/reverse performance rounds; notify/communication/reduce profiler
samples are separate from full API samples. `summary.csv` records the limit,
bandwidth, mean/P50/P95 and changes versus rows=1/default-visit/fast=0, if that
candidate is present; otherwise delta columns are blank. Per-rank observations
check the requested limit, visit and WQE fast against the Host tiling fields.

By default, visit is unchanged and WQE fast is off. To explore single-row
preparation as well as merging without restarting Buffer, add:

```bash
--combine-payload-max-rows 1 2 4 --combine-visits 8 32 --combine-wqe-fast 0 1
```

`--combine-visits` accepts 4/8/16/32/64/128. WQE fast applies only to rows=1;
merged-row candidates run once per visit with fast=0, because the coalescing
builder takes precedence. The example measures eight configurations per shape
and weight mode, retaining the same rank/P/W/Q and all other production settings.

Install the matching Host/kernel build once (including
`last_combine_payload_max_rows` and `last_combine_wqe_fast` for audit). Switching limits only updates the
environment/policy between completed calls, outside timing; it does **not** rebuild
the library, restart workers, recreate Buffer or change QP reservations. Graph
capture is not used. Host observation verifies the submitted limit, not how
many rows actually merged; use the independent WQE diagnostic pass for that.

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
