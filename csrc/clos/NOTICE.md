# Ascend Clos backend provenance

Imported from https://gitcode.com/lishaoxun/ascend_deepep, initially branch `deepep`,
commit `c09040328c853a011dfe37c74463c44723db4476`; compute, scheduling and benchmark
updates now track branch `v1-shared-jetty-scheduler`, commit
`6f9d8c9ef9403a302724ce42cf55810843ffebc3`.
The imported implementation is licensed under CANN Open Software License
Agreement Version 2.0; see `third-party/ascend-deepep.LICENSE` at repository root.

`upstream.json` records the original kernel/include hashes. Local adaptations
replace the imported resource wrapper with native AscendC UB allocation and
rename the architecture build define; UB offsets and capacity are preserved.
Local header comments also differ; recorded hashes describe upstream originals.

Both backends use the target repository's SHMEM
`836d3fd6f9c2f3b10a6acac36ccfd531b104d4a7`, including the MIX barrier fix,
instead of the source revision's `4dd80e2`.
Host integration adds
the cross-backend SHMEM ownership check. Python adaptation adds platform keyword
aliases, backend selection, event compatibility and a shared lifetime lock.
