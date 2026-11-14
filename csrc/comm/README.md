# 公共分层通信模板

命名空间为 `deepep::comm`，实现基于 CatCCOS `2ab9a31` 迁移，许可证见 [LICENSE.catccos](LICENSE.catccos)。
本目录不引用 `experiments/`、测试数据规则或外部 `catccos/` 头文件。
依赖为 AscendC、Catlass 和 SHMEM；当前设备路径面向 Ascend950。

| 层级 | 接口 | 职责 |
| --- | --- | --- |
| block | `HierarchicalCommSwizzle<CoreCount, GroupSize, CommBlockM>` | 一个 AIV 对应一个 peer；选择 MTE/UDMA/RDMA 和当前窗口的通信块 |
| block | `HierarchicalCopy<ArchTag, SrcType, DstType, TileCopy, TileSwizzle, UbStages>` | RowMajor Put；MTE 按 UB 容量切 tile，网络数据切分到 QP1/2/4 |
| tile | `TileMteCopy<SrcType, DstType>` | GM→UB→目标 GM，支持逻辑行宽及原始 GM stride |
| tile | `TileHierarchicalSignals` | 基于 SHMEM 的标记、等待、QP quiet 和缓存可见性操作 |
| tile | `TileHierarchicalReady<Signals>` | 每轮 Notify、WaitAll、WaitReady；支持显式 Deferred RDMA 通知 |
| tile | `FillUdmaRemoteSqe` / `CopyUdmaWqe` | SHMEM v1.8.0 原始远端 VA WQE 适配与阻塞 UB→SQ 拷贝，供已有 EP 快速路径使用 |
| tile | `ReadyLayout{world, qps, windows}` | 独立的控制区布局，无矩阵形状或 dtype 依赖 |
| kernel | `HierarchicalAllToAll<BlockData, BlockScale, Scheduler, Consumer, ScaleGroup>` | 窗口循环、data/scale 发送及 AIC0 消费回调 |
| kernel | `AllToAllLayout<DataElementBytes, ScaleElementBytes, ScaleGroup>` | 每 peer 矩阵、scale、行窗口和分配字节数 |
| kernel | `AllToAllArguments<Layout, ConsumerArguments>` | 通信指针、rank、布局和调用方的消费参数 |
| kernel | `HierarchicalBarrier` | 输入就绪、最终 drain 后的全卡 dissemination barrier |

`SrcType`/`DstType` 提供 `Element` 和 `Layout` 别名，可使用 Catlass GemmType。
`HierarchicalCopy` 当前只支持同类型、RowMajor、非 sub-byte 的 Put，不提供 Get 语义。
block 默认 `RdmaCompletion::Wait`，返回前完成 MTE/RDMA；UDMA 仍由调用方 quiet。
流水线调用显式传入 `RdmaCompletion::Deferred`，RDMA 仅提交，由末尾的
`HierarchicalCopy::QuietRdma(peer, qpNum)` 统一完成 data、scale 和标记的 drain。

`GroupSize` 是连续 rank 属于同一个 UDMA 域的部署约定，host 必须验证路由。
`CommBlockM` 只影响 MTE 行块；UDMA/RDMA 每轮提交完整 peer 窗口，内部再按 QP 划分。
每条 peer/QP 必须只有一个提交者；禁用 shared jetty，或由其他调用方实现等价的独占约束。

## 多轮完成协议

- MTE：完成本轮 tile 搬运后写本地 ready。
- UDMA：quiet 所有本轮数据 QP，写标记，再 quiet 标记，避免跨轮乱序覆盖。
- RDMA：多轮 kernel 在 block 和 `Notify` 两处显式选择 Deferred，每条 QP 内按 data→scale→flag 提交，
  所有窗口提交后通过 block 的 `QuietRdma` 对每条 QP 最终 quiet 一次。
  `Notify` 的默认 Wait 路径会先 quiet 数据，再发标记并完成标记，避免公共接口隐式依赖保序。
- 多 QP 不共享完成保证；接收方等待每个来源的每条 QP 标记，包括无数据分片的 QP。
- host 将 `ReadyLayout::ValueOffset(window)` 初始化为 `uint32_t(window+1)`，整个 kernel 期间保持不变。
  ready/frontier 初值为 0，可写槽独占 64 字节缓存行，workspace 起始地址按 64 字节对齐。
  只读发送值为紧凑的 uint32 数组，所有 peer 共享，每轮占 4 字节，无需逐项填充到 64 字节。
- 独立协调 AIV（physical index = world）等所有 incoming epoch `>= window+1`，再发布本地 frontier。AIC0 等 frontier 后消费。
  这表示本卡当前窗口可读，不表示其他卡消费结束，也不授权重用缓冲区。
- 不同窗口写入不同输出区间；输入、输出、控制区需保留至 kernel、最终 drain 和消费者全部结束。
  跨次调用复用控制区，必须先完成所有参与者，再共同重新初始化；当前 Arguments 不包含跨次 epoch 基址。

RDMA 优化要求同一 QP 的 RDMA WRITE 按提交顺序在远端可见，不能使用破坏该保证的 relaxed ordering 配置。
SQ/CQ 满时 SHMEM 仍可内部 poll；不承诺所有提交调用都不阻塞。

## PR #239 对齐范围

参考 [CatCCOS PR #239](https://gitcode.com/cann/catccos/pull/239) 的
`76a213b6ff40d3708d47ff7a7a7ea263d2aeeb99`：显式 Wait/Deferred、block 最终 QuietRdma、
只读 epoch 数组和独立协调核。该 PR 的免逐轮 quiet 快路径只使用 QP0；这里保持 QP1/2/4，
以每条 QP 的独立标记拓展相同的顺序保证，不用 QP0 代表其他 QP 已完成。
UDMA 始终按每轮 quiet 后通知；未将 RDMA 的保序假设用于 UDMA。

## 组合与调用约束

`HierarchicalAllToAll` 的默认消费者是 `WaitOnlyConsumer`。自定义消费者需提供：

```cpp
struct Consumer {
    struct Arguments { /* host 可传递的 POD 参数 */ };
    template <class Params>
    __aicore__ void operator()(const Params& params, uint32_t window);
};
```

回调仅在 **AIC0** 上执行，进入前当前窗口的所有来源 data/scale 已就绪。
多 AIC 计算的分发和同步需要调用方进一步组织；回调不可直接等待未参与的 AIC。
示例 checksum 消费者位于 `experiments/hierarchical_alltoall/checksum.h`，公共 kernel 不包含任何校验逻辑。

host 在下发前验证 M/N、ScaleGroup 整除、windowM、rank/world、QP 数、可用核数和字节数不溢出。
world 不大于 Scheduler::CORE_COUNT，peer 张量每轮元素数须适合 uint32，数据布局及 QP 数在各 rank 一致。
以 MIX 1:2 启动，至少需要 `world+1` 个 AIV、`ceil((world+1)/2)` 个 AIC。
physical AIV index `[0, world)` 对应 peer，AIV `world` 只协调接收，不阻塞任何 sender 的后续窗口。
`scratch` 和 event ID 为当前 AIV 独占，进入 ready 接口前不得有未完成的 UB 使用。

CMake 使用：设置 `CATLASS_SOURCE_PATH`，加入 `add_subdirectory(.../csrc/comm ...)`，
将消费目标链接到 `deepep_comm`。该 INTERFACE target 提供 `csrc` 和 Catlass include 目录；
CANN/SHMEM 路径、backend 宏、MIX 编译选项及链接库由调用方配置。
公共头文件使用 `#include "comm/block/..."` 等形式；不会改变 DeepEP 已有算子的构建或协议。

功能运行命令与 CPU 检查见 [独立样例 README](../../experiments/hierarchical_alltoall/README.md)。
