<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 性能测量与复现

仓库包含 HT、LL kernel 测量入口和通用 benchmark 协议，计时和统计口径各不相同。运行方式见[测试入口](../tests/README.md)与 [benchmark 工具](../benchmarks/README.md)。

## 测量入口与范围

| 入口 | 计时范围 | 跨 rank 汇总 |
| --- | --- | --- |
| HT profiler 入口 | torch_npu profiler 记录的 Notify、cached dispatch 或 combine kernel 耗时 | 保存各 rank 统计，不自动汇总 |
| HT peer-table 入口 | NPU Event 计时；另有独立插桩采样 | 按每次采样取 rank MAX，再计算均值与中位数 |
| LL 系统测试 | torch_npu profiler 记录的 dispatch/combine kernel task 耗时 | 先算各 rank 均值，再报告最大 rank 均值 |
| 通用 benchmark 协议 | 同步 Host 墙钟，包含调用至结果可消费的等待 | 逐 iteration 取 rank MAX，再求均值与分位数 |

HT、LL 入口可以执行设备测量；通用入口当前支持配置规划，其 native 测量适配器未接入。通用协议的 CPU 测试和 dry-run 不产生设备性能结论。

## HT kernel 耗时

`test_dispatch.py` 的 `--profile-op notify` 单独测量 Notify kernel；`--profile-op dispatch` 使用已生成的 handle 测量 cached dispatch，不包含 fresh 调用的 Notify。`test_combine.py --profile-op combine` 测量 combine kernel，单独记录 prepare kernel，后者不计入 combine 耗时。

每个 rank 在 `output/profile/<op>/<scenario>/rank_<rank>/<timestamp>/` 保存配置 `capture.json`、profiler 记录和 `summary.json`。预热后采集指定次数，检查样本完整性及输出正确性，再报告逐 rank 的原始样本、均值、中位数、最小值和最大值，单位为微秒。没有自动生成跨 rank 汇总；不得将某个 rank 的结果标为全作业耗时，也不能将 kernel 时间标为完整 API 或 MoE 前向时延。

## HT peer-table Event 计时

`run_peer_table_check.py` 使用 NPU Event 记录 dispatch 设备区间，`--kernel-profile` 另外采集插桩结果，`--routing-profile` 增加 fresh handle 的 Notify/建表采样。Notify 总时间已经包含建表子区间，不能重复相加。Host join、冷初始化以及 Python 分配/校验不在对应设备 Event 区间内。

`dispatch_metrics.py` 输出 `kernel-summary.json`，保留逐次跨 rank 最大值并计算均值与中位数。各阶段的 rank 最大值不能相加解释为端到端耗时。命令、区间划分及跨轮同步约束见 [peer-table 测试说明](../tests/README.md#ht-peer-table-检查与计时)。

## LL kernel 耗时

dispatch 和 combine 分别采集 profiler 记录，combine 所需的 dispatch 和专家输入准备在其采集区间之外。统计只使用指定 kernel 的正有限耗时样本，单位为微秒，不包含整个 Python API 或 MoE 前向的全部工作。

每个 rank 保存原始样本。`--perf-tail=N` 使用最后 N 个样本统计，0 表示全部。各 rank 分别计算 mean/min/median/max/p95/p99；分位数在线性位置 `(N-1)*p` 上插值。汇总字段 `max_rank_mean_kernel_us` 为各 rank 均值的最大值，不是逐轮 rank MAX 的均值。

默认测量前检查 dispatch、combine 和重复调用的正确性。`--skip-check` 会标记跳过精度，不能将该结果写成精度通过。发布记录应保留配置、源码和依赖版本、各 rank 样本、profiler 输出及正常退出记录。`summary.json` 中的配置和路径可能包含本地信息，公开前检查导出内容。

LL ST 不直接输出逻辑带宽，也不测量完整专家计算。不得将其 kernel 时间标为端到端时延，或把独立测量的 dispatch/combine 简单相加当作实测 roundtrip。

## 通用 benchmark 协议

以下定义对应现有通用计时与结果代码，便于解释 dry-run 的计划及协议测试，不表示 CLI 已能执行 native 测量。

| 操作 | 计时工作 | 区间外准备 |
| --- | --- | --- |
| dispatch | 调用至输出可安全消费；fresh 包含布局/通知 | 输入和路由，cached handle |
| combine | 调用至 BF16 输出可安全消费 | dispatch 生成匹配 handle 和专家输入 |
| roundtrip | dispatch → 同一 handle 的 combine，不含专家 GEMM | 输入和路由；FP8 恢复 BF16 位于计时内 |

计时使用 `perf_counter_ns`。每轮先 barrier 和同步，再记录开始时间；调用后等待并同步，记录结束时间。API 内部分配计入，输入生成、正确性检查、结果释放和样本收集不计入。预热不进入样本，重复使用输入，不主动清 cache。

全 rank 样本按 iteration 对齐，逐轮取 MAX 后计算 mean/min/max 和 nearest-rank p50/p95：排序数组下标为 `ceil(p*N)-1`。前后正确性检查、全 rank 样本与会话退出均成功才可生成 `measured` 记录。

### 逻辑字节与带宽

`effective_bytes` 是全 rank 有效 hidden payload 的逻辑字节数：

- 普通布局按每个源 token 的有效目标 rank 数计行；expanded 按有效路由槽位计行，重复槽位分别计数。
- BF16 dispatch 每元素 2 字节，FP8 dispatch 每元素 1 字节；combine 每元素 2 字节；roundtrip 累加两方向字节。
- 包含本 rank 路由，不包含 padding、scale、索引、权重、协议开销或重传。

`bandwidth_gbps = effective_bytes / (mean_rank_max_us * 1000)`，使用十进制 GB/s。这不是物理链路带宽、单 rank 带宽或互联利用率。全无效路由可以有零有效字节，但耗时样本仍须为正有限值。

## 比较与发布记录

比较结果需保持版本、dtype/量化、负载、拓扑、路由、缓存策略、计时边界和统计方法一致。没有匹配的实测基线就不填写加速比；未测量不填 0。各测试的实际精度判据见[精度说明](precision.md)，不能用统一的“L2 通过”覆盖不同入口。

通用结果格式见[结果说明](../benchmarks/results/README.md)；HT、LL 分别使用各自的 summary、rank 统计和 profiler 文件。MegaMoE 记录完整融合前向，与相同语义的非融合流程比较，不与单独 EP kernel 时间混用。
