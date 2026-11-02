<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Modified by zhu-mingzhe71 2026
-->

# CatCCOS UDMA overlay

The headers under `include/catccos` provide the all-to-all-v UDMA implementation
required by these operators and take precedence over the CatCCOS dependency's
include directory. They implement UDMA peer-serial scheduling and layout-aware
two-dimensional copies. The build validates the CatCCOS, CATLASS and SHMEM
dependency versions before compilation.

The overlay supports optional AllToAllV export and prevents deadlocks on
non-communication cores in GroupedMatmul-AllToAllV.
The export path is pipelined: AIV subcore 0 exports the previous staging buffer while
subcore 1 submits the current local MTE and remote UDMA communication.
The GroupedMatMul-AllToAllV overlay additionally carries separate input and
output row extents so unbalanced routes can consume `sum(global_tokens)` rows
and return `sum(local_tokens)` rows without requiring those totals to match.
The target-order AIC/AIV schedulers pack complete N-width row ranges
in destination order, reducing each source-rank/round transfer to one
contiguous UDMA request.
The overlaid problem-shape ABI reads both route-count tables from contiguous
signed 64-bit storage, matching the public PyTorch `torch.int64` interface.

Both kernels support physical tiling: explicit staging-row
capacity, two/three-buffer kernel templates, configurable export/self-copy AIV
worker pools, and the `256x256x256` GMM-AllToAllV template.
AllToAllV-GroupedMatmul also aggregates all source-rank fragments of one
expert into a single AIC task domain and supports the `mSplit` M-axis swizzle.
The optional route-metadata ABI remains nullable in this PyTorch integration so
device route tensors do not force a synchronous host round trip per invocation.
The implementation handles dynamic output extents, publishes UDMA completion
after queue quiet, and guards workspace-loop subtraction.

Both directions instantiate `128x256x256`,
`256x128x256` and `256x256x256`; the inverse supports two or three workspace
stages, explicit stage-row capacity and configurable self-copy workers.
Each remote peer retains one completion owner. Only the primary self worker
publishes receive completion; all self workers finish their MTE block loops
and join the AIV barrier before workspace reuse. Peer-scoped UDMA completion
and `aclshmemx_mte_quiet()` remain in place.
Legacy five-field inverse tilings retain derived staging, two workspace stages
and one self-copy worker. The EP8/256-expert defaults are unchanged; new EP4/
28-expert profiles are selected by the host for the documented shapes.
Host parsing and Python search reject all other MMAD shapes.
Local copy staging uses the pinned CATLASS Ascend950 allocation of 248 KiB,
including a 256-byte UDMA scratch reservation, rather than physical 256 KiB UB.
