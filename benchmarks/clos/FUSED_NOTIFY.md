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
