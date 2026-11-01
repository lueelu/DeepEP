# Ascend Clos backend provenance

Imported from https://gitcode.com/lishaoxun/ascend_deepep, initially branch `deepep`,
commit `c09040328c853a011dfe37c74463c44723db4476`; compute, scheduling and benchmark
updates now track branch `v1-shared-jetty-scheduler`, commit
`6f9d8c9ef9403a302724ce42cf55810843ffebc3`.
The imported implementation is licensed under CANN Open Software License
Agreement Version 2.0; see `third-party/ascend-deepep.LICENSE` at repository root.
Retained copyright notices apply to their respective files.

`upstream.json` records the original kernel/include hashes and the minimal
CATLASS/TLA resource-header dependency closure. The latter is pinned to
`48f7461afcff7c9282c6372c19d8768d9c6e0d9d` and retains its CANN license notices.
It does not replace the existing MegaMoE CATLASS submodule.

Device sources and layouts match that updated source revision. Both backends
use the target repository's SHMEM `836d3fd6f9c2f3b10a6acac36ccfd531b104d4a7`,
including the MIX barrier fix, instead of the source revision's `4dd80e2`.
Host integration adds
the cross-backend SHMEM ownership check. Python adaptation adds platform keyword
aliases, backend selection, event compatibility and a shared lifetime lock.
