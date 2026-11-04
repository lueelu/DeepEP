# 生产 LL 性能协议

此维护分支的完整 LL 采样实现在 `tests/elastic/test_ep_low_latency.py`。HNS_1825 显式部署入口为 `tests/ll_rdma/run_ll.py`，完整步骤见 [RDMA 验收指南](../ll_rdma/README.md)。它不依赖实验 checkout 的生成 snapshot 或仓外 example 模块。

2026-10-27按用户要求采用`initial-reduce-exp200-batch-v3`：每个操作的采样批次开始只做一次
scalar all_reduce；默认预热10次并同步后，连续提交固定200次Exp，再执行进入active的
`profiler.step()`，随后连续调用算子100次。采样循环内不再插入Exp或fill。
不在Exp前缀之后额外同步，Exp次数不随算子迭代次数变化。尾50统计和profiler kernel Duration口径保持。

当前每次 dispatch/combine 调用前都插入一次标量 all_reduce（异步提交后 wait，`aligned()`），用于拉齐快慢卡：
所有 rank 到齐后才进入 kernel，避免仍在入队的 rank 让对端空转。该集合通信位于测量循环内，
开销计入每次调用，因此数值是“对齐后的端到端时延”，不再等于纯 kernel 时间；对齐用的 all_reduce
kernel 名不匹配目标 kernel，不进入尾50统计。

独立timer/reference采样同样在初始all_reduce、预热和同步后提交200次Exp，再连续执行
稳定调用及打点调用。若在perf结束后追加打点，则重新提交200次Exp作为该诊断批次前缀，
共用perf开始时的all_reduce，不再增加一次；两批诊断和性能数据分开统计。
正确性检查、批次边界及算子内部的必要复用协调保留，不属于被删除的逐轮采样all_reduce。

Exp默认118260×30720 FP32，约13.53GiB/卡。保留所有原始task和trace，Event/API时间
与kernel时间分开。新旧采样协议不可直接作为同条件A/B；比较器检查协议标识。
旧`--delay-exp-before/after`兼容参数仅接受200/0，也可直接省略；执行次数固定在代码中。

A5 只验证脚本/构建/允许的双卡原语；不伪造 LL EP16，不作目标性能结论。完整 EP16/32 数值、多机和性能由用户在目标机执行。没有匹配的目标机实测 MTE 基线时，不报告加速比。
