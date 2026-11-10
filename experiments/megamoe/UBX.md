# UBX consumer integration

This branch starts at `lueelu/DeepEP` main `89e1939` and imports the UBX
consumer from `SimpleBrightMan/DeepEP` `2615922`. It does not merge the older
fork's ordinary communication backends over main.

## Public API and selection

Use the existing `SymmBuffer`, `transform_weights_for_mega_moe`, `mega_moe`
and `fp8_fp4_mega_moe` signatures. At Buffer construction the existing DCMI
physical-device query identifies board `0x44` as UBX. Other products or failed
product detection select the existing POD implementation. Ranks agree on the
backend before initialization. The chosen implementation still checks its
shape and runtime requirements.

The POD kernel sources under `csrc/kernels/mega_moe` are unchanged. Consumer
device code and its required local headers are isolated under
`csrc/kernels/ubx`, with a separate device library, namespace and hidden helper
symbols. This deliberately preserves POD template defaults; deduplication of
those headers is not a prerequisite for this integration.

Both products use one internal `_MegaRuntime` and the core binding
`create_megamoe_runtime(rank, size, bytes, endpoint, timeout, backend="pod")`.
The existing five-argument call keeps POD defaults. Backend selection is
internal; the public `SymmBuffer` signature does not gain a backend argument.
Native `RuntimeProfile` configuration controls transport, queue ownership,
capacity and required peer access. Shared state is queried through
`Runtime.peer_access_error`, not a URMA-specific property or factory.

There is no multi-host opt-in flag. Host identity and visible-device consistency
remain checked, and backends requiring peer mappings validate actual MTE/UDMA
access collectively before device communication. Cross-host participation alone
does not reject a supported group or bypass these checks. Rebuild the complete
wheel when changing this runtime ABI; do not mix the new Python code with an old
native extension.

UBX supports EP4/8/16, equal positive BS on all ranks, capacity up to 16384,
H256..8192 and I256..16384 aligned to 256, TopK1..min(8,E), and equal contiguous
expert partitions with 1..128 routed experts per rank. Expanded receive
capacity is computed from EP, capacity and TopK, not device data. Large shapes
can fail allocation; no route truncation or Host count-based allocation is used.

UBX uses prequantized FP8 E4M3/E8M0 input, routed FP4 weights and SwiGLU. S0/S1
are supported; S1 uses contiguous ND FP8 E4M3 shared weights with E8M0 scales.
Shared I is inferred from the supplied weight shapes and can differ from routed
I. POD's existing dtype and shared-expert support is unchanged. UBX requires
28 AIC/56 AIV and at least 248 KiB usable UB, plus actual MTE/UDMA peer mappings.

| UBX shape | Internal CV |
| --- | --- |
| EP16 and actual BS exactly 1024 or 2048 | 128 |
| All other supported shapes | 256 |

There are no H/I/TopK gates on this CV rule and no public CV/algorithm knob.
Control-region offsets are reserved for CV128 at construction and stay fixed
when logical CV changes. A CV/shared-shape change retires previous work and
coordinates the resource transition; steady calls reuse resources. Every
kernel rebuilds routes from the current top-k input without D2H.

`mega_moe` returns `(output, expert_counts)` on both products. Consumer exports
the existing route summary totals into a fresh NPU int32 tensor, not a workspace
view. The inplace API returns None and supports the original cumulative stats
argument. No CPU route count or debug snapshot participates in forward.

## SHMEM

The dependency lock remains main's `836d3fd6`. Consumer's PR944 dependency is a
divergent history, not a strict successor. The integration uses the locked
SDK's Host shared-jetty setter and defines the missing RO shorthand using its
existing config type and RO ordering value. RO payloads and SO flag atomics
share the same peer/QP0; peer owners do not share mutable queues. Final quiet
and source lifetime rules remain mandatory. Build against the locked headers
and runtime together.

## Accuracy, performance and trace

The main runner dispatches internally to a UBX driver using the public API:

```bash
python -m deep_ep_experimental.megamoe.benchmark \
  --config urma --input fp8 --mode accuracy --golden-dir benchmark-golden \
  --output-dir benchmark-runs/ubx-accuracy
```

Launch with the repository's usual torchrun/SHMEM environment. Generate Golden
with the same installed version, configuration, routing and BS values first.
`--dual-reference` adds the NPU cascade error-budget reference; without it the
existing CPU Golden check remains the default. Shared benchmark fixtures expand
FP4 codes to FP8 exactly before timing; this is not an operator conversion.
Native FP8 shared weights outside that fixture subset still require device
qualification.

`--mode perf` uses the config's alignment/warmup/iters. Device RoutePrepare is
included in the fused-kernel duration; resource reuse is not cached routing.
`--timer` runs a separate fine diagnostic, respecting `--timer-ranks`; no wait
events are recorded, and real synchronization/credit waits remain. Trace tick
conversion follows the existing 1000 MHz protocol assumption, not a claimed
hardware frequency probe. Overflow is an explicit error.

Dispatch diagnostics include a nonblocking observation of each peer's last
incoming Dispatch CV SO epoch in local memory. This is a zero-duration completion
point, not a wait interval; it adds no CQ access, quiet, remote operation or
correctness dependency. Timer/trace-off calls do not read these diagnostic flags.
The matching `diagnostic-peer-dispatch-receive-v2` capability requires rebuilding
Python/native/device components together.

For each selected rank and peer class, `communication_bandwidth.json` reports
the last nonempty incoming receive observation minus the first outgoing PUT,
using outgoing X+scale bytes as numerator. Receive-only peers affect the end;
empty incoming edges do not. Missing endpoints or nonpositive spans produce no
bandwidth value. This is an effective exchange-span rate including observation
delay, not pure outbound link throughput. Both endpoints use the same rank's
clock. Combine retains its existing semantics. This update imports the final
Dispatch diagnostic from consumer `07cd68f`, not its superseded CQ experiment or
the standalone hardware-capture scripts.

Only input/routing, precision, performance and trace utilities are imported;
the source branch's standalone CPU simulators and tuning CLI are not imported.

## Validation status

Windows CPU checks cover public POD API regression, exact CV selection,
product fallback, numeric references, runtime and timer/performance helpers.
A targeted run passed 219 tests, with 20 real-Torch cases skipped because Torch
is unavailable and 6 Linux/SDK-only cases deselected. The repository-wide
pre-commit checks passed using the existing local hook cache (the first attempt
to download hooks was interrupted by a GitHub connection failure).
A separate native CPU check exercised the actual route summary headers for
all local-expert counts 1..128, EP4/8/16 and CV128/256 (768 combinations),
including exact receive counts, group counts, mailbox size and bounded bucket
storage; all passed. This is not device DMA or concurrency qualification.

On 2026-11-10, runtime unification passed 144 local CPU/API checks (20 skipped
without Torch). A complete wheel was then built on an aarch64 host in the
`tpr` conda environment: Python 3.12.14, Torch 2.12.0+cpu, torch_npu 2.12.0.post2,
CANN 9.2.0, SHMEM `836d3fd6` and the locked CATLASS revision. Core, Direct,
POD MegaMoE and UBX consumer extensions/device libraries all compiled and
linked. The initial device build exposed malformed residual statements from
the earlier wait-trace removal; these were corrected without removing the
underlying waits or synchronization, and the rebuilt wheel succeeded.
After explicit `conda activate tpr`, all 24 runtime mock tests passed without
skips. Loading the wheel from an isolated target directory confirmed the core
and MegaMoE extensions load, the unified factory defaults to `backend="pod"`,
the old URMA factory/properties are absent, and `Runtime.peer_access_error` is
available. These checks did not create a Session or launch a kernel.

The isolated build directory is
`/home/codex-deepep/megamoe-runtime-unified-20261110-b89e367-check1`.
The build did not replace any existing installed package or run NPU operators.
NPU accuracy/performance for this revision remains unverified: run POD
regression plus UBX S0/S1, CV-boundary/transition, empty/hot expert and E-limit
accuracy cases before qualification. Performance and RO/SO hardware ordering
require device results; compilation and CPU checks do not establish them.
