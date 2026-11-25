# Fresh dispatch composed Host entry

Rebuilt Direct/Global Clos wheels expose `dispatch_host_entry_version=1` and use
one native Host entry for ordinary fresh dispatch. The entry launches the selected
existing layout kernel, calls the existing dispatch-with-handle implementation,
and returns the original outputs/handle plus the layout rank-membership mask.
The Python dispatch return signature and handle fields are unchanged.

This removes the intermediate Python round trip and boxing/unboxing of layout
outputs. It does not fuse device kernels or change sending, count publication,
allocation streams, clone, event completion or resource ownership. Notify buffers
are still allocated by the existing implementation after layout submission.
Thus the previous layout-to-notify bubble is not expected to disappear completely.
Moving/combining allocations is separate work.

## A/B on one installed build

Rebuild and install the branch using the repository build procedure. Record actual
loaded Python/native library paths, hashes, commit and SHMEM revision. The native
module returned by `deep_ep.backends.direct._native.load_native_module()` must have
`dispatch_host_entry_version == 1`.

- `DEEPEP_CLOS_DISPATCH_HOST_ENTRY=0`: original two-native-entry submission.
- `DEEPEP_CLOS_DISPATCH_HOST_ENTRY=1` (default): composed Host entry.
- `DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY=0`: keep independent device kernels in both arms.
- Fix `DEEPEP_CLOS_V2_SIMT_LAYOUT=1` in both arms to avoid changing layout kernels.

Older wheels without the capability retain the original two-entry behavior. Cached
dispatch and the explicit device layout/notify fusion path retain their existing
entries regardless of this setting. Verify capability before accepting A/B results.

Use one Buffer per rank and change the Host-entry environment setting between
drained sampling blocks, identically on every rank. Preserve routing, seed, sender
policy, huge-page mode, L2 handling and input/handle lifetime in both blocks.
Never introduce a synchronize between dispatch and independent shared FC1.

Recommended customer-shaped comparison: EP8, tokens4096/rank, hidden2816, topk8,
experts256, BF16, FP32 weights, AIV32, expand0,
`allocate_on_comm_stream=True`, shared FC1 [4096,2816] x [2816,1536].
Use the customer's fixed balanced routing in both arms. Warmup10, measure30 calls
per arm, first0 then1. Run four profiler samples per arm separately from formal
timing. Check fresh outputs, routing weights, expert counts and reuse of the handle
in cached dispatch/combine against an independent reference outside timing.

Record these separate boundaries:

1. Exact public `buffer.dispatch()` entry/return, with the caller's old-result
   unpacking/release outside the range. Retain normal old-output lifetimes in both arms.
2. Layout and notify Host submissions and device execution/gap, matched by connection.
3. Payload Host submission and interface return tail.
4. Dispatch return to independent FC1 Host submission; pair completion including
   both streams, and cleanup/window cost separately.

Aggregate each matched call's maximum over ranks, then mean/median/P95/max. Retain
outliers and per-rank samples. Profiler Host/device exported clocks can be offset:
use Host-to-Host and device-to-device differences, not uncalibrated cross-clock gaps.
Do not treat kernel-only improvement or a shifted allocation as full-API improvement.

Local verification covers CPU routing, output/handle ownership forwarding and a
source-derived native composition oracle. CANN compilation and device precision,
asynchronous lifetime behavior and performance require remote validation.
