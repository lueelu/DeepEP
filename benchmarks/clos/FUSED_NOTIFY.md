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
The benchmark's TCPStore group also carries the automatic policy's host-identity
collective during Buffer construction. This adapter is scoped to that group and
restored after construction (including failures); physical topology inference
and the default send policy remain unchanged. No Gloo/HCCL process group is added.

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

## Host overlap experiment

These independent per-call switches default to **0** and require fused layout
notify. Set identical values on every rank. Cached dispatch ignores them.

| Switch | Effect |
|---|---|
| `DEEPEP_CLOS_NOTIFY_HOST_PREP=1` | Allocate layout metadata through native code using one aligned count arena plus the route mask. Defer the communication-only `dst_slots` allocation until after notify submission. |
| `DEEPEP_CLOS_NOTIFY_DEFER_PLAN=1` | Produce route/count metadata first. After the existing count publication and prefix work, build stable plan/send-head entries using the original plan builder. |
| `DEEPEP_CLOS_DISPATCH_HOST_TIMING=1` | Collect calling-thread Host timestamps; no device synchronization or logging is added by the hooks. |

The host-preparation path owns fresh metadata on every invocation; it does not
reuse storage held by an old handle. Notify receives its tiling by value and
never reads `dst_slots`; only the later communication launch receives that
pointer. Stream-aware allocation and lifetime tracking remain in place.

The deferred-plan path retains every existing local/cross-rank synchronization
and the MTE3 mapped-host publication. Each AIV still owns its original channel.
UB is reused only after existing consumers drain. Count publication is **not**
dispatch completion: the communication kernel remains ordered after the full
notify kernel on the same stream. Host output allocation can overlap the tail
of notify, without changing output capacity or the public return contract.
This option rereads the route mask and recomputes plan prefixes, so total kernel
time may grow; a complete API improvement must be measured, not assumed.

Use one Buffer initialization to test all five modes in forward/reverse order:

```bash
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_fused_notify.py \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 --expand 0 \
  --warmup 10 --samples 30 --flush-l2 --no-huge-pages \
  --pipeline-sweep --host-timing --output /tmp/fused_notify_pipeline_ep8
```

`--no-huge-pages` here matches the preceding remote comparison; keep the setting
identical across modes. The sweep is `baseline, fused, host, plan, both`, then
the reverse, with/without FP32 weights. Baseline uses standalone layout; the
other four use fused notify with neither, Host preparation only, deferred plan
only, or both optimizations. Notify/communication remain eight AIVs and the
script checks that the automatic send policy is identical. Both libraries must
be rebuilt; the script requires `fused_notify_pipeline_version == 1`.

Precision runs first for all modes and routing fixtures. For CPU source-body
checks before device runs, use `python -m pytest tests/clos/test_fused_notify.py -q`.
The extracted production builders are compared to an independent stable-order
oracle, including empty channels, tails, skew, untouched early plan storage,
and overwritten UB before deferred construction. These tests do not establish
device event ordering or NPU precision.

Formal unprofiled API timing and profiler passes run with Host timing **off**.
`--host-timing` adds a separately warmed diagnostic pass and writes per-rank
`host_timing.json` with raw timestamps plus `host_summary.csv` with intervals:

- `api_to_notify_submit`: API entry through notify submission; `python_front`
  includes Python work and any native layout/allocation helpers before the
  shared V1 dispatch body. `native_prepare` and `notify_launch` split its work.
- `post_launch_prepare`: notify submission through starting the count wait.
- `wait_total` / `wait_remaining_counts`: when CPU observes total receive count,
  then all expert counts. These are observation times, not device write times.
  The API still waits for all counts before allocating receive outputs.
- `count_to_outputs`: count validation/list preparation and receive allocation.
- `comm_launch`, `comm_submit_to_complete`, `api_host_return`, and
  `api_complete_diagnostic`: launch, completion tail and API boundary intervals.

All timestamps use the same native steady clock. The raw `native_return` mark
is the end of the shared V1 dispatch body; public-handle wrapping and Python
return work are captured before `api_return`. Diagnostic summaries use each
iteration's slowest rank, then the mean; independently aggregated intervals
cannot be added to reconstruct the API. Use raw same-call rows for attribution.
Do not subtract profiler timings from diagnostic API timings.

Check `mapped_counts_all_samples` in the report. If false, the `.cpu()` fallback
waits for stream completion and cannot demonstrate early mapped-count overlap.
No new `.cpu()` fallback or device-wide synchronization is introduced here.

## Eight-AIV copy and cached-path transplant

The branch also contains the compact self-copy and cached dispatch changes from
[optimize_8aiv_kernel, commit 8d98be0](https://github.com/ZiHa0-Zeng/DeepEP/commit/8d98be0e7fb93420e55a62cb8c3a69f569a60cc9).
Automatic peer order, P/W/Q, visit, weight transport, and self-copy scheduling
are unchanged; the source branch's policy and weight-sender changes are not included.

- Ordinary UDMA V2 combine uses compact local rows for EP8, eight communication
  AIVs and eight handle channels, with BF16 payload and dense aligned row strides.
  It copies payload and optional weights in 64 KiB MTE chunks, partitioning rows
  across the existing self-copy AIVs. Remote slots remain token-indexed; reduce
  locates local rows through the receive channel prefix and send-head ordinal.
  Independent reduction core counts keep that same mapping. Early-single,
  aggregate-pipeline and token scheduling retain their original path.
- Cached UDMA dispatch with EP8/eight AIVs, two-byte elements and no scales uses
  the handle's full per-channel counts, skipping repeated 256-token list-window
  loads and scans. Per-peer visit budgets and completion synchronization remain
  unchanged. Fresh dispatch and unsupported layouts retain the old loop.

CPU ownership/addressing checks execute extracted production functions:

```bash
python -m pytest tests/clos/test_eight_aiv_copy.py tests/clos/test_fused_notify.py tests/api/test_clos_compatibility.py -q
```

The CPU mocks do not validate device ordering or performance. In addition to the
fused on/off precision commands above, check independent reduction cores against
the public CPU golden on an idle EP8 machine after rebuilding:

```bash
for reduce in 8 32; do
    python tests/clos/source/test_v1_v2_shared_backend.py \
        --api v2 --public-auto --auto-policy --ep-size 8 \
        --num-tokens 129 --hidden 2816 --num-topk 8 --experts-per-rank 32 \
        --num-aiv 8 --combine-num-sms 8 --reduce-num-sms "$reduce" --check-events
done
```

Rebuild the native library as well as Python: the internal combine tiling ABI
now carries the local-layout flag.

## Opt-in metadata thread comparison

Two independent native dispatch options accept only `0` (default/unset) or `1`:

| Mode | `DEEPEP_CLOS_DISPATCH_SRC_IDX_256` | `DEEPEP_CLOS_DISPATCH_TOPK_256` |
| --- | --- | --- |
| Baseline | 0 | 0 |
| Top-k only | 0 | 1 |
| Source indices only | 1 | 0 |
| Both | 1 | 1 |

These options require UDMA. They are read at each native dispatch call, so a
drained Buffer can be reused across modes without SHMEM reinitialization.
Keep all ranks in the same mode. Captured graphs retain their captured tiling;
use ordinary uncaptured calls for this comparison.

The default remains 128 threads for each VF and 16 threads per peer for source
indices. The source-index option selects 256 threads, with 32 threads per peer
for EP <= 8 and 16 for larger EP (including EP16). The top-k option selects 256
threads for the existing token/top-k slot loop, including optional FP32 weights.
Neither option changes peer scheduling, P/W/Q, visit, payload WQE batches,
self-copy placement, UB allocation or completion synchronization. Source-index
publication remains fresh-only; top-k publication still requires top-k inputs.
Cached calls without top-k inputs do not exercise either VF.

After rebuilding, inspect native `DispatchTilingData.metadata_src_threads`,
`metadata_src_lanes`, and `metadata_topk_threads` during precision checks to
verify the actual configuration. Remove diagnostic probes before timing.
The tiling ABI changed, so rebuilding only Python is insufficient.

Run `python -m pytest tests/clos/test_dispatch_metadata_threads.py -q` with a
C++17 compiler for production-body CPU checks (sparse/empty routes, both weight
modes, per-peer ownership and host option validation). These mocks do not
validate SIMT execution, resource usage or remote-store completion on device.
For device comparisons use EP8/16, eight communication AIVs, reduce48,
T4096/H2816/K8/E256/BF16/expand0, with unchanged routes and huge-page/L2 settings.
Compare modes 0/1/2/3 then 3/2/1/0, checking precision before timing fresh dispatch
with and without weights; use cached dispatch and combine as regression controls.

## Dispatch stage diagnostic and EP16 cached-channel experiment

`DEEPEP_CLOS_CACHED_CHANNEL_EP16=1` enables the existing whole-channel list
schedule for **EP16, eight communication AIVs, cached, two-byte payload, no
scales**. Its default is zero. EP8's existing eight-AIV fast path is unchanged;
fresh dispatch and all other core/EP combinations retain their prior path.
Each channel uses its exact compact per-peer count, so the scalar input-token
window scan and GM-to-UB list refill are skipped. Destination rows, visit/WQE
budget, sender ownership and all completion operations remain unchanged.
The longer per-peer channel interval can change inter-peer progress, so this
is an experiment, not a promised improvement. Keep the flag identical on all
ranks; change it only between drained, uncaptured launches.

`DEEPEP_CLOS_DISPATCH_DEVICE_TIMING=1` selects a separate diagnostic kernel.
The default kernel specializes all trace operations away and has no trace
array, clock reads or diagnostic DMA. Both native and kernel libraries must
be rebuilt together because `DispatchTilingData` changed. The diagnostic
currently supports peer-scheduled dispatch only.

Each AIV owns one 512-byte GM row allocated by the stream-aware allocator.
It accumulates scalar timestamps/counters privately and exports through
UB `[0,512)` after the original users finish; no trace region overlaps live
sender staging. The export uses MTE3, with MTE3-to-scalar then scalar-to-MTE3
ordering. There are no extra cross-rank collectives. Existing V-to-scalar
completion edges bound source-index and top-k VFs; self-copy timing follows
its existing final drain. A diagnostic-only MTE3-to-scalar edge bounds output
copy completion. Formal bandwidth must be measured with diagnostics disabled.

The native `take_dispatch_device_timing()` returns the latest launch tensor
on the calling host thread and clears that retained reference. Consume the
dispatch completion event before copying it to CPU; retrieve each launch
before submitting the next. Rows have stride 64 int64 elements; interpret only
`dispatch_device_timing_fields`. `entry_cycles` is an absolute local-device
clock, other timeline fields are per-AIV entry offsets. Convert cycle fields
using `dispatch_device_timing_cycles_per_us`; never align different devices
using their absolute clock values.

`payload_queued` means the final PutRows work has been queued, not that its
last deferred batch or network transfer has completed. `published` follows
Publish (including its existing flush), `incoming_done` follows incoming
markers, `quiet_done` follows sender completion, and `sync_done` follows the
existing all-rank collective. `src_cycles`, `topk_cycles`, `self_cycles` and
`list_cycles` accumulate substage durations over channels. They overlap the
timeline intervals and other AIVs: **do not sum their maxima into API time**.
`work_done` excludes trace-export overhead and is diagnostic, not bandwidth.

Run the following through the site's normal torchrun launcher, once for EP8
and once for EP16 (same shape, routes, topology and libraries as the baseline):

```bash
python benchmarks/clos/bench_dispatch_stages.py \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 \
  --cores 8 32 --cached-channel-modes 0 1 \
  --warmup 10 --samples 30 --trace-samples 3 --flush-l2 --huge-pages \
  --output /path/to/results
```

The script initializes Buffer once per rank and runs fresh without/with
weights and cached dispatch, forward then reverse. It checks cross-variant
output equality, requested/native huge-page state and all-rank library hashes.
Formal API wall and kernel profiling are separate passes with traces off;
L2 eviction finishes before each alignment barrier. Diagnostic traces follow
in a separate pass and validate the effective whole-channel gate per core.
Per-rank raw reports and `summary.json` contain per-iteration slowest-rank
mean/P50/P95; bandwidth uses decimal GB/s, deduplicated payload including self.
`device_trace.json` retains every diagnostic core/sample rather than hiding
slow-core differences in one average.

For an older installed baseline, use this script from the new checkout with
`--cores 32 --cached-channel-modes 0 --trace-samples 0`; it imports the installed
wheel rather than placing the checkout package on `sys.path`. Use a separate
process/environment for each installed version. Cross-variant equality is not
an independent golden: also run the standard device precision suite, including
empty/skewed routes and tail tokens. Run the existing ABC harness with reduce48
for combine and end-to-end regression; this script measures dispatch APIs only
and contains no expert compute.

CPU checks:

```bash
python -m pytest tests/clos/test_eight_aiv_copy.py \
  tests/clos/test_dispatch_metadata_threads.py tests/clos/test_dispatch_stages.py -q
```

The production-body C++ oracle covers EP8/16/32, on/off, cached/fresh,
8/16 cores, visit4/8/16/32, empty channels and non-divisible token counts.
It checks exact token/destination rows and per-visit budget, not network
completion or device performance. CANN compilation and NPU validation remain
required before treating the experiment as deployable.

## Opt-in channel metadata experiment

`DEEPEP_CLOS_DISPATCH_CHANNEL_METADATA=1` enables **fresh peer-scheduled UDMA
dispatch only**. The default is 0. Cached dispatch ignores the switch and is
the negative control; graph capture is rejected when the fresh experiment is
requested. All ranks must use the same switch. Rebuild and install both the host
extension and kernel library because `DispatchTilingData` has new fields.

Notify packs top-k indices and optional FP32 weights in fixed
`[peer, channel, aligned_channel_tokens, topk]` slots. It does not compact across
channels. Original top-k positions are retained, with invalid entries -1/0.
The existing src_idx plan is already compact within a channel and is not copied.
Each notify AIV owns one channel. It loads token-major indices, weights and
send-head ordinals through MTE2 to UB; SIMT reads those UB inputs and writes
the packed GM slots. No SIMT input routing scan through GM is added.

Packing overlays notify's released UB at the tail (after deferred-plan building
if enabled). The input tile uses actual EP/K/index width, with each region
rounded up to 32 bytes. For tile length B:

| Region | Bytes |
| --- | --- |
| Send heads, starting at UB byte 0 | align32(B x EP x 4) |
| Top-k indices, immediately after heads | align32(B x K x sizeof(topk_idx_t)) |
| Optional FP32 weights, immediately after indices | align32(B x K x 4), or zero without weights |

`DEEPEP_CLOS_NOTIFY_METADATA_TILE_TOKENS=0` (unset/default) selects 128 tokens,
halving to 64 or 32 only when the complete tile exceeds the architecture's
available kernel UB. Explicit 32/64/128/256 settings fail on the host if they
do not fit. EP/K remain bounded by 256/128. EP8/K8/64-bit indices with weights
uses 16 KiB at B=128; EP16 uses 20 KiB. This reduces per-channel load/VF/event
rounds from 16 to 4 for T4096/eight channels, without changing the 128 SIMT
threads, route ownership, output layout or completion rules. Device performance
must still be measured; the channel-metadata experiment remains opt-in.

Loads complete with MTE2_V; V_MTE2 drains readers before the next tile load.
V_S and same-stream kernel completion gate dispatch. Packed tensors use the
stream-aware allocator and remain valid through dispatch/UDMA completion. For
T4096/K8/64-bit indices, staging is 3 MiB/rank at EP8 and 6 MiB/rank at EP16
with weights (2/4 MiB without weights). Allocation is included in API timing.

Payload peer order, P/W/Q, visit and self-payload copy order stay unchanged.
After payload submission, each sender handles only its existing peer/channel
ownership (`channel % workers_per_peer`). AIV prepares source/destination/length
descriptors; each SIMT lane constructs one channel/field WQE using the existing
32-WQE builder. UDMA moves the bytes. Only valid rows are sent, never padding
between channels. At eight channels/W1, a nonempty peer needs at most 16 WQEs
without weights or 24 with weights, excluding payload. Actual flush counts still
depend on SQ ownership and available credits. Payload spans are flushed before
the descriptor UB is reused; metadata uses the same owned QPs and terminal
markers, with the original incoming wait, quiet and all-AIV/rank collective.

Self metadata stays local: its existing channel workers load packed blocks to
UB and use exact-word local stores, avoiding padded MTE writes across compact
channel boundaries. This does not change self **payload** copying. The experiment
therefore includes notify packing, bulk remote metadata and local metadata copy;
it is not a payload-scheduling change. With device tracing enabled, `self_cycles`
also includes the local metadata copy, and `payload_queued` includes metadata
descriptor submission; formal performance runs keep tracing off.

Use the site's torchrun launcher for **EP8 and EP16 in separate jobs**:

```bash
# EP8 on one node; use --nnodes=2 --node_rank=$NODE_RANK and the site's
# rendezvous arguments for EP16 across two eight-rank nodes.
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_channel_metadata.py \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 \
  --warmup 10 --samples 30 --huge-pages --output /path/to/channel_metadata_ep8

# One Buffer: compare 32/64/128/256-token packing under unchanged auto payload
# policy and Zeng's contiguous EP8/AIV8/H2816 rank/P2/W1/Q1/visit32 policy.
# Keep weights prepacked in notify for every measured fresh dispatch.
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_channel_metadata.py \
  --metadata-modes 1 --metadata-tile-tokens 32 64 128 256 \
  --dispatch-plans auto zeng --warmup 10 --samples 30 --huge-pages \
  --output /path/to/channel_tile_zeng_ep8
```

The second command visits every tile/plan combination and then reverses the
order, reusing a single Buffer throughout. `--dispatch-plans zeng` is restricted
to contiguous EP8/AIV8/H2816/BF16/non-expanded dispatch; it is a benchmark-only
per-launch override and leaves the production default policy untouched. It does
not change the prepacked weight path or enable Zeng's combine weight policy.
For EP16 tile testing use `--dispatch-plans auto`; no EP16 Zeng policy is assumed.
Every summary includes the requested/effective tile and actual native launch
policy. Native `last_dispatch_channel_metadata_tile_tokens()` returns zero when
packing is inactive (including cached). Rebuild host and kernel together; this
benchmark requires `dispatch_channel_metadata_version=2` for enabled packing.

The wrapper defaults to 8 AIV, BF16, expand=0, L2 flush then all-rank alignment,
1 GiB pages, and **off/on/on/off**. It initializes Buffer once per rank per EP,
including precision checks and all three operators: fresh without weights,
fresh with weights, and cached. It fixes other experimental flags to 0 and
checks that the payload policy is identical across off/on. Native
`last_dispatch_channel_metadata()` is checked for each invocation, not just the
requested environment value. Native huge-page state and all-rank library hashes
are recorded; retain the site's physical huge-page allocation audit as well.

Precision precedes performance: an independent CPU oracle reconstructs receive
rows from source routes, covering masked/tail routes, all-invalid, skew and an
empty source rank. The normal 4096-token case is also checked independently and
against the other variant, with another equality check after sampling. Each rank
keeps raw profiler exports, `samples.json`, `precision.json`, `manifest.json`,
`summary.json`, `comparison.csv` and final `exit.json`. Require every rank's exit
file **and** the launcher exit code to be zero.

`comparison.csv` separates notify, communication and unprofiled complete API
wall time, reports paired deltas and repeat-mean ranges, and derives decimal
payload GB/s from communication time. It never sums independently aggregated
stage times into API latency. Cached must stay on the old path; use its variation
to detect drift. This script has no combine/reduce or expert compute, so it does
not establish MoE end-to-end performance.

Local checks (CANN/NPU validation is still required remotely):

```bash
python -m pytest tests/clos/test_channel_metadata.py tests/clos/test_dispatch_stages.py \
  tests/clos/test_dispatch_metadata_threads.py tests/clos/test_eight_aiv_copy.py -q
```

## Report files

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

## Queue and local-copy diagnostics

Version-2 device traces add per-rank, per-AIV accounting to fresh/cached dispatch
and ordinary split peer-scheduled combine. Rebuild **both host and kernels**.
Normal kernels use the disabled trace specialization; no policy, visit, batch
capacity or payload transport is changed. The switches default off:

- `DEEPEP_CLOS_DISPATCH_DEVICE_TIMING=1`
- `DEEPEP_CLOS_COMBINE_DEVICE_TIMING=1`

The benchmark controls these switches itself. Formal API timing and kernel
profiling run with tracing off; a separate diagnostic pass enables tracing.
`trace_overhead.json` compares instrumented and normal API samples. Diagnostic
V_S/MTE2_S edges and counters can perturb overlap, so never publish bandwidth
computed from diagnostic timings. Combine tracing rejects token scheduling,
early-single/aggregate pipeline, integrated expanded inputs and graph capture.

Example, once per EP8/16 with the site's usual torchrun launch arguments:

```bash
torchrun <site-launch-arguments> benchmarks/clos/bench_channel_metadata.py \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 --cores 8 \
  --metadata-modes 1 --metadata-tile-tokens 128 --dispatch-plans auto \
  --cached-channel-modes 1 --combine-diagnostics --reduce-cores 48 \
  --warmup 10 --samples 30 --trace-samples 5 --flush-l2 --huge-pages \
  --output <shared-result-directory>
```

One Buffer per rank is reused across precision checks, forward/reverse formal
sampling and diagnostic passes, including weighted/unweighted combine. BF16,
expand=0 and routing are fixed. Fresh metadata keeps the notify-prepacked UDMA
path. Combine uses its existing automatic policy, independent of the dispatch
plan override. For EP8 only, `--dispatch-plans auto zeng` additionally compares
the existing Q2 plan with rank/P2/W1/Q1/visit32; no default strategy is changed.
Use `--cores 8 32 --dispatch-plans auto` to include a same-build 32-core control
without another Buffer initialization. Reduce remains independently fixed at 48.

Per-rank output retains the original formal dispatch `summary.json`, adds
`combine_summary.json`, and writes `device_trace.json` in each capture directory.
Every trace row includes these measurements:

| Field | Interpretation |
| --- | --- |
| `list_cycles` | Load/scan the dispatch index window, or load the combine index window |
| `descriptor_cycles` | Scalar descriptor/span preparation; excludes nested submission |
| `wqe_build_cycles` | Completed SIMT WQE construction, including synchronization |
| `wqe_submit_cycles` | Inclusive SHMEM batch submission; SDK-internal CQ polling is included |
| `credit_cycles` / `credit_blocked_cycles` | Explicit SQ admission checks / checks that initially lacked credit |
| `credit_checks`, `credit_blocked`, `credit_max_cycles` | Admission count, blocked count and maximum check duration |
| `wqes`, `flushes`, `short_flushes`, `spans` | Aggregate sender WQEs, batches, batches smaller than 32, and spans |
| `state_flushes`, `qp_flushes`, `index_flushes` | Nonempty batches flushed for state, QP or combine index-window changes |
| `metadata_wqes`, `metadata_flushes` | Subset submitted through the generic descriptor sender, including channel metadata |
| `submit_no_credit` | SDK submissions observed without sufficient credit immediately before the call |
| `sq_used_max`, `sq_used_sum`, `sq_empty_checks` | Software `sq_head-sq_tail` snapshots at admission |
| `self_cycles`, `self_begin`, `self_end` | Local-copy duration and its interval relative to that AIV's entry |
| `src_cycles`, `topk_cycles` | Direct SIMT metadata/weight work when that path is active |

The decoder supplies microseconds, mean WQEs per batch, short-batch fraction,
blocked-admission fraction, and separate incoming-flag, quiet and final-sync
intervals. `incoming_wait_us` starts **after** any after-submit local copy;
the older `incoming_done_interval_us` may contain that copy and is not a pure
wait metric. Sender publication, including terminal flags, has its own timeline
interval. Direct SIMT metadata/weights and terminal flags are not aggregate
WQEs and are not included in the WQE counters.

Software head/tail snapshots are not hardware queue occupancy, and an empty
snapshot alone does not prove a starved link. Explicit admission wait also does
not capture SDK-internal waits; inspect inclusive submission and
`submit_no_credit` together. Same-core nested counters and stages overlap;
different AIVs overlap as well. Do not sum stage maxima into kernel/API time,
or compare absolute cycle timestamps across devices. Use formal slow-rank
samples and per-AIV traces to identify preparation, credit pressure, local-copy
or completion tails before selecting the next optimization. Notify is profiled
for dispatch; combine's outer alignment notify is excluded from its measured API.

## Dispatch overlap experiments

These four per-launch switches default to `0`. Rebuild both host and kernel
libraries; the tiling ABI has changed. Peer order, P/W/Q, visit, payload UDMA,
combine and reduce policies are unchanged.

| Benchmark mode | Environment variable set to `1` | Effective scope |
| --- | --- | --- |
| `self` | `DEEPEP_CLOS_DISPATCH_OVERLAP_SELF` | Peer dispatch without auxiliary metadata/copy AIVs or scales, with sufficient private UB: submit remote payload/metadata and terminal flags, then copy self before incoming wait/quiet |
| `slots` | `DEEPEP_CLOS_DISPATCH_EARLY_SLOTS` | Fresh peer dispatch: build local V2 destination slots after submission/self work; Source still waits for received metadata |
| `init` | `DEEPEP_CLOS_DISPATCH_SKIP_TOPK_INIT` | Fresh channel-metadata path only: omit valid-row top-k prefill because packed metadata writes every slot, including invalid `-1` entries |
| `notify` | `DEEPEP_CLOS_NOTIFY_METADATA_DOUBLE_BUFFER` | Channel-metadata pack: two independent input UB tiles and event pairs; GM output layout is unchanged |
| `all` / `off` | All four / none | Matched combined treatment / control |

Auxiliary AIVs retain parallel self-copy. A live sender's descriptors, headers,
completion records and index windows are reserved throughout post-submit work;
local-copy scratch starts after that region. No incoming wait, quiet, collective,
output padding initialization or completion publication is removed. Double-buffer
tile selection checks **two** tiles against UB capacity; an explicit oversized
tile fails instead of silently changing the experiment.

Run once per EP8/16 using the site's launch arguments, with one Buffer per rank
across all variants and both core counts:

```bash
torchrun <site-launch-arguments> benchmarks/clos/bench_channel_metadata.py \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 --cores 8 32 \
  --metadata-modes 1 --metadata-tile-tokens 128 --dispatch-plans auto \
  --cached-channel-modes 1 --overlap-modes off self slots init notify all \
  --warmup 10 --samples 30 --trace-warmup 5 --trace-samples 5 \
  --flush-l2 --huge-pages --output <shared-result-directory>
```

Fresh, fresh weighted and cached dispatch run in forward/reverse order. Precision
checks also compare public Source/Slots against original CPU routes, including
invalid routes, duplicate peer routes, empty ranks and tails. Timing remains
L2 flush → rank alignment → measured call. Diagnostic kernels get separate warmup.
The former cached-channel filter no longer drops fresh/32-core control groups.

`overlap_comparison.csv` reports matched notify, communication and complete API
means and pair deltas; it never mixes core counts, metadata modes or payload
policies. `summary.json` also records the native effective option mask:
bit 0 = after-submit self, bit 1 = early slots, bit 2 = skipped prefill,
bit 3 = double-buffer pack. Cached or auxiliary-worker paths legitimately mask
out inapplicable options. Traces append `topk_init_cycles`, `source_cycles` and
`slots_cycles`; early slots occur before incoming wait and must not be counted
again as a completion-tail cost. CPU mocks check indexing, UB boundaries and
event credit pairing, but do not establish device ordering or NPU performance.

### Cached weights and direct metadata outputs (opt-in)

Three independent flags default to `0`; rebuild both host and kernels before
using them, with identical settings on every rank. The default payload schedule, core count, visit and completion
protocol are unchanged.

| Flag | Effect |
| --- | --- |
| `DEEPEP_CLOS_CACHED_WEIGHT_PACK=1` | Repack this cached invocation's weights in cached notify, using the handle-owned head and plan. Send channel top-k/weight blocks with UDMA. |
| `DEEPEP_CLOS_DISPATCH_METADATA_DIRECT=1` | For channel UDMA metadata only, send fresh source indices and fresh/cached weights directly into owned output tensors, including self. SIMT metadata retains its SHMEM path. |
| `DEEPEP_CLOS_CACHED_OMIT_TOPK_OUTPUT=1` | Skip the unused top-k output allocation/copy for cached weighted non-expanded public dispatch. Internal routing indices and expanded consumers remain intact. |

Fresh channel packing still requires `DEEPEP_CLOS_DISPATCH_CHANNEL_METADATA=1`.
Cached packing requires the current handle's private plan/head, and shares the
existing channel-packing capacity and graph-capture restrictions. It repacks
every call: changing weights does not change routing or mutate the handle.

Direct output addresses occupy the two spare words of the existing payload
record. Capacity checks and epoch publication remain in place; no additional
address-exchange round or collective is added. Output tensors exist before
publication and use the existing stream-aware allocator. Source conversion
still runs, reading the final source-index tensor. Internal top-k remains in
SHMEM. Invalid weight slots are initialized in the actual final destination.

Run separately for EP8 and EP16, with one Buffer per rank throughout each run:

```bash
torchrun <site-launch-arguments> benchmarks/clos/bench_channel_metadata.py \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 --cores 8 32 \
  --metadata-modes 1 --metadata-tile-tokens 128 --dispatch-plans auto \
  --cached-channel-modes 1 --overlap-modes off \
  --metadata-output-modes off pack direct omit pack_direct all \
  --reduce-cores 48 --warmup 10 --samples 30 --trace-warmup 5 --trace-samples 5 \
  --flush-l2 --huge-pages --output <shared-result-directory>
```

`cached_weighted` is measured separately. Precision checks read raw weights
(including invalid zeros), change cached weights across two handle reuses,
and compare expansion and subsequent combine against options-off controls.
They cover masked/invalid/skewed routes, empty ranks and partial tiles. Formal
timing disables diagnostic instrumentation. Traces separately report
`src_output_bytes`, `topk_output_bytes` and `weight_output_bytes` for valid-row
SHMEM-to-output copies. Direct mode must eliminate source/weight copies;
cached omit mode must eliminate the top-k copy.

`metadata_output_comparison.csv` groups matched options and reports notify,
communication and complete API timing separately. The legacy cached weighted
plan-rebuild kernel is reported separately and included in `notify_total_us`
within each rank/sample before rank aggregation. The effective metadata mask is
bit 0 = cached pack, bit 1 = direct output, bit 2 = omitted top-k output.
The older overlap experiments remain off in this comparison. CPU tests validate
addressing and control flow; CANN compilation and NPU precision/performance
must be checked on the target environment before enabling these options.
