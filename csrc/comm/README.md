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
| tile | `TileHierarchicalReady<Signals, Mode>` | 三种 ready 策略；提供 NotifyWindow、FinishPeer、WaitAll、WaitReady |
| kernel | `HierarchicalAllToAll<BlockData, BlockScale, Scheduler, ScaleGroup, Mode>` | 窗口循环、data/scale 发送及接收完成同步 |
| kernel | `HierarchicalAllToAll::Arguments / Params` | Host 入参和设备侧参数，通过 `ToUnderlyingArguments` 转换 |
| kernel | `HierarchicalBarrier` | 输入就绪、最终 drain 后的全卡 dissemination barrier |

`SrcType`/`DstType` 提供 `Element` 和 `Layout` 别名，可使用 Catlass GemmType。
`HierarchicalCopy` 当前只支持同类型、RowMajor、非 sub-byte 的 Put，不提供 Get 语义。
block 默认 `RdmaCompletion::Wait`，返回前完成 MTE/RDMA；UDMA 仍由调用方 quiet。
流水线 kernel 使用 `Ready::RDMA_COMPLETION` 选择 block 完成模式，由 Ready 层负责延迟的 quiet。

`GroupSize` 是连续 rank 属于同一个 UDMA 域的部署约定，host 必须验证路由。
`CommBlockM` 只影响 MTE 行块；UDMA/RDMA 每轮提交完整 peer 窗口，内部再按 QP 划分。
每条 peer/QP 必须只有一个提交者；禁用 shared jetty，或由其他调用方实现等价的独占约束。

## 多轮完成协议

- `PerWindow`（Ready 默认）：每窗先完成 payload 再写 ready；网络通知也等待完成，防止跨轮乱序覆盖。
- `PerWindowOrderedRdma`：RDMA 在每条 QP 上按 data→scale→ready 提交，`FinishPeer` 排空所有数据 QP。
  UDMA 和 MTE 保留逐窗完成等待。该策略参考 [CatCCOS PR #239](https://gitcode.com/cann/catccos/pull/239)，
  DeepEP 为每条 QP 使用独立的 ready 槽，包括无数据分片的 QP。
- `FinalOnly`：`FinishPeer` 排空该 peer 的全部 UDMA 或 RDMA QP，通过 QP0 发布唯一的最终 ready。
  仅适用于中途不消费输出、TX 不提前复用的调用；接收端只等待每个来源的 QP0 标记。
- host 将 `GetReadyValueOffset(rankSize, qpNum, commIdx)` 初始化为 `uint32_t(commIdx + 1)`，整个 kernel 期间保持不变。
  ready/frontier 初值为 0，可写槽独占 64 字节缓存行，workspace 起始地址按 64 字节对齐。
  只读发送值为紧凑的 uint32 数组，所有 peer 共享，每轮占 4 字节，无需逐项填充到 64 字节。
- 独立协调 AIV（physical index = world）等所有 incoming epoch `>= window+1`，再发布本地 frontier。通算融合的调用方可通过 `WaitReady` 等待本轮数据就绪。
  这表示本卡当前窗口可读，不表示其他卡消费结束，也不授权重用缓冲区。
- 不同窗口写入不同输出区间；输入、输出、控制区需保留至 kernel、最终 drain 和所有数据使用方结束。
  跨次调用复用控制区，必须先完成所有参与者，再共同重新初始化；当前 Arguments 不包含跨次 epoch 基址。

RDMA 优化要求同一 QP 的 RDMA WRITE 按提交顺序在远端可见，不能使用破坏该保证的 relaxed ordering 配置。
SQ/CQ 满时 SHMEM 仍可内部 poll；不承诺所有提交调用都不阻塞。

`HierarchicalAllToAll` 在 HNS_1825 上默认选择 `PerWindowOrderedRdma`，其他后端使用
`PerWindow`。显式选择 `FinalOnly` 时，协调 AIV 等待最终窗口再交付输出。

## 组合与调用约束

host 在下发前验证 M/N、ScaleGroup 整除、commSizeM、rankIdx/rankSize、QP 数、可用核数和字节数不溢出。
world 不大于 Scheduler::CORE_COUNT，peer 张量每轮元素数须适合 uint32，数据布局及 QP 数在各 rank 一致。
以 MIX 1:2 启动，至少需要 `world+1` 个 AIV、`ceil((world+1)/2)` 个 AIC。
physical AIV index `[0, world)` 对应 peer，AIV `world` 只协调接收，不阻塞任何 sender 的后续窗口。
`scratch` 和 event ID 为当前 AIV 独占，进入 ready 接口前不得有未完成的 UB 使用。

CMake 使用：设置 `CATLASS_SOURCE_PATH`，加入 `add_subdirectory(.../csrc/comm ...)`，
将消费目标链接到 `deepep_comm`。该 INTERFACE target 提供 `csrc` 和 Catlass include 目录；
CANN/SHMEM 路径、backend 宏、MIX 编译选项及链接库由调用方配置。
公共头文件使用 `#include "comm/block/..."` 等形式；不会改变 DeepEP 已有算子的构建或协议。

功能测试命令见 [独立样例 README](../../experiments/hierarchical_alltoall/README.md)。
