# Full Clos sparse CQE validation

Same-version baseline: `8cb5689b9e3360796b0039524cb74cd653c88ae7`.
SHMEM remains `7af7bf48410abca964a5edd246d243600d634b66` from
`dependencies.lock.json`. No strategy, visit, batch-size or feature-toggle changes.

The submission design follows the physical-SQ credit and low-level WQE/doorbell
path in [peer_transport.hpp](https://github.com/lueelu/DeepEP/blob/83252a54d2363734a706a7dd8368bebc3094af5a/csrc/kernels/dispatch/peer_transport.hpp)
and [combine.cpp](https://github.com/lueelu/DeepEP/blob/83252a54d2363734a706a7dd8368bebc3094af5a/csrc/kernels/combine/combine.cpp).

## Ownership and completion

| Region | Layout | Lifetime / owner |
|---|---|---|
| UB WQEs | `[0, 2048)` | Original 32-WQE batch, VF produces, MTE3 consumes |
| UB descriptors | `[2048, 3072)` | Original 32 four-word requests |
| UB headers | `3072 + peer/QP index * 128` | First 64 bytes are immutable unsignaled WRITE template |
| UB pressure WQE | `[3136, 3200)` | Unused second header of peer zero; never read by data VF |
| GM readiness | `workspace + source_rank * 512 + [0, 256)` | Original peer/worker/QP final flags |
| GM pressure sink | `workspace + source_rank * 512 + 256 + qp * 8` | Distinct source rank and physical QP; never polled by receivers |

The existing UB suffix used for indices, local copy and pipeline aggregation is
unchanged. Full Clos host layouts place completion signals at offset zero. All
current marker counts are at most 32 (including token/aggregate pipelines);
`Begin` enforces that bound. The own-rank readiness word is initialized to one
and provides a stable pressure source. It is not a pressure destination.

The pinned SHMEM Host maps `(jfsID, sqVa, dbVa)` to one `state_addr`, binds SQ and
CQ to that state and rejects cross-QP ownership of one physical queue. Same-QP
cross-peer sharing is supported. Credit therefore uses the actual shared
`sq_head - sq_tail`, not a per-peer shadow. One WRITE occupies one WQEBB, checked
at compile time. Admission reserves 32 data WQEBBs plus one checkpoint, including
the SDK's strict guard. On pressure, quiet first reclaims any existing CQEs;
an SO+CQE checkpoint is appended only if the unsignaled suffix still blocks
admission. Its quiet reclaims through the SDK CQ entry, never by assigning tail
to head. Final per-peer/QP SO flags retain their original destinations and CQEs.

Data batches copy both wrap spans before their doorbell. The SDK's MTE3_S drains
UB reads, allowing descriptor/header reuse; this does **not** release outgoing
GM payload/metadata. Existing final quiet, incoming waits and SIMT completion
collectives remain responsible for those lifetimes. Pressure SO ordering across
shared-SQ destinations follows the reference design and still needs device stress
validation on the installed HCOMM/firmware.

## Coverage

`SenderT`, `DispatchSenderT` and `CombineSenderT` use this submission path:
fresh/cached dispatch, payload/scales, channel-packed metadata (including direct
outputs), weighted/unweighted combine, ordinary expanded/non-expanded combine,
token scheduling, early-single and aggregate-pipeline UDMA sends.

Independent SIMT src/top-k/weight stores and output-address publication are
unchanged, as are their completion collectives. Notify and non-Clos transports,
and the separate `csrc/kernels` backend, are not modified. No wait kernel is split.

## Local checks versus device checks

```bash
python -m pytest -q tests/clos/test_sparse_cqe.py tests/clos/test_queue_diagnostics.py \
  tests/clos/test_channel_metadata.py tests/clos/test_eight_aiv_copy.py
```

The credit test compiles the production admission, submission, final-publication
and quiet methods into constexpr SDK simulations. It covers distinct/shared SQs,
two QP lanes, empty sends, 1/32-WQE batches, short tails, physical SQ ring and uint32
SQ counter wrap, repeated calls, traffic beyond SQ capacity, partial CQ reclamation,
and many final flags sharing one SQ. It checks that pressure does not overwrite
pending WQEs or readiness. A deliberately corrupted sink must fail compilation.
The production payload WQE builders are also constant-evaluated for every count
from 0 to 32, with/without auxiliary fields and with early-single source mapping.
These tests do not emulate transport ordering or replace CANN/NPU verification.

At implementation time the Windows host has no CANN/NPU toolchain, and configured
SSH targets are unreachable. CANN host/kernel compilation, NPU precision and
continuous-call stress, performance and dual-stream end-to-end results remain
**unverified**. Do not promote CQE reduction as measured performance improvement.

## Remote qualification and matched performance

Build baseline and candidate separately against the **same pinned SHMEM**, CANN,
firmware and build flags. Verify loaded wheel and shared-library paths/hashes on
every rank. Do not compare with an older SHMEM or change scheduling to compensate.

Use the existing dispatch-stage benchmark for EP8/16, T4096/H2816/K8/E256/BF16,
communication cores 8/32, reduce 48. On each node, supply the normal torchrun
rendezvous/node arguments; for example (single-node EP8):

```bash
torchrun --standalone --nproc-per-node=8 benchmarks/clos/bench_dispatch_stages.py \
  --tokens 4096 --hidden 2816 --topk 8 --experts 256 --cores 8 32 \
  --reduce-cores 48 --cached-channel-modes 0 --metadata-modes 0 \
  --overlap-modes off --metadata-output-modes off --combine-diagnostics \
  --warmup 10 --samples 30 --trace-warmup 5 --trace-samples 3 \
  --huge-pages --output /path/to/results/REVISION_ep8
```

Keep existing optimization switches identical in both versions. Run a separate
matched pair with `--metadata-modes 1 --metadata-output-modes all` to exercise
packed UDMA metadata and cached-weight repacking. Each command reuses one Buffer
per EP across its cases. The script separates uninstrumented timing from trace
passes; retain its precision checks. Run baseline/candidate/candidate/baseline,
with identical seeds, topology, L2 settings, warmup, sampling and rank aggregation.

Additionally run device correctness/stress with:

- Fresh/cached, with/without weights, expand 0/1, FP32 weight return checks.
- All-invalid routes, zero-receive peers, empty senders, skew, and token tails
  around 31/32/33/65; repeated Buffer and handle reuse with changed input values.
- One/two QP lanes; actual shared-SQ aliases confirmed from SHMEM logs; enough
  traffic and repetitions to wrap SQ/CQ and exceed SQ capacity many times.
- Token, early-single and aggregate-pipeline modes where enabled by existing
  constraints; local/remote payload and metadata outputs, not only counts.

This is a required device matrix, not a claim that the above one command covers
every case. Current combine trace support is ordinary non-expanded peer mode;
expanded/pipeline paths need separate precision and profiler runs.

Trace JSON exports `cqe_requested`, `credit_checkpoints`, `credit_quiet_calls`,
`final_quiet_calls`, `final_quiet_cycles` and `sq_wraps`, alongside existing
credit wait, WQE preparation/submission and local-copy counters. CQE requests are
counted after real pressure/final posts; they are not polled-completion counts.
For the unchanged baseline, ordinary batch CQEs equal `flushes`; final CQEs equal
the number of peer/QP publications from the effective launch plan. Label that
baseline number as source-derived unless measurement-only counters are backported.

Report per-case and per-rank CQEs, credit wait and final quiet separately from
uninstrumented communication, notify, reduce and full API mean/P95. Do not sum
independently aggregated stage maxima into API time. Retain raw traces and kernel
exports. If a dual-stream workload is available, compare its end-to-end time and
correctness without claiming this single-stream diagnostic proves equivalence.
