<!--
Copyright (c) 2026, Lu Lu
Modified by SimpleBright_Man 2026
-->

# MegaMoE A8W4 CATLASS GMM

GMM 使用 CATLASS 的 `PackedMxA8W4TileCopyTla`、MX MMAD 和 NZ 权重反量化。
`mega_moe_grouped_matmul.hpp` 接受一个专家 slice 的地址和形状，由 stage 层提供 tile 顺序及输入/输出完成同步。

- `../common/mega_moe_block_scheduler.hpp` 使用 swizzle=3、跨专家轮转和共享专家调度。
- `block_mmad_a8w4.hpp`：AIC 使用独立的 A/B K 窗口、L0 K128、4096-K scale 窗口，只等待 AIV0。
- `a8w4_k_plan.hpp`：AIC/AIV0 共用的分块规则。小 M 时按 N/(2×M对齐值) 扩大 A 窗口；slice M<256 且实际 N≤128 时 B 使用 K512，其余使用 K256。
- `block_prologue_a8w4.hpp`：AIV0 将每个 B 窗口拆成两个半块，使用四个 UB stage；每个 stage 使用 16 KiB FP4 输入和 32 KiB FP8 输出，共 192 KiB。两个半块写完后才发布一次权重就绪。
- `resource.hpp`：本地存储的非拥有视图；不创建 TPipe，不重置 Wave 中其他阶段的事件。

AIC 以 mode 4 的 flag 8/9 发布 L1 B 空闲，AIV0 以 flag 6/7 发布权重就绪。
AIC 不发送或等待 AIV1 的 `+16` 标志。AIV0 在 slice 结束时排空两个空闲标志，AIC/Vector block 在 `End()` 中回收内部事件。
路由 GMM1 的 `gmm1CatlassContext_` 由 `MegaMoeA8W4Wave` 持有，跨专家切片和 Wave 复用同一个
`CatlassGmm1` 及其 `BlockMmad`/Prologue 成员。固定缓冲区视图和 event ID 只在构造时设置一次；
专家切换通过 `UpdateProblem()` 更新 M/N/K、scaleK、GM 地址和权重布局。
`Begin()`/`End()` 管理每个 slice 的事件启动/排空、MM layout 切换和 ping-pong 起点，
因此 GMM2 可以继续使用同一组 L1/L0/UB 和 event ID；对象存活不等于持续占用硬件事件。
AIV1 的 context 保持休眠，不建立 prologue 缓冲区视图、不操作 CATLASS 同步标志，继续执行 Dispatch、SwiGLU 和 Combine。
GMM2 使用独立的 `CatlassGmm2` 局部对象。GMM1/GMM2 只共用内部的 AIC/AIV0 pipeline；
gate/up 合并投影及输出偏移由 `CatlassGmm1` 管理，单投影和行主序输出由 `CatlassGmm2` 管理。
L1 分配为两组 B64KiB、两组 A128KiB、两组 scaleA32KiB 和两组 scaleB32KiB，共 512 KiB。
L1 地址分配：B 在 0/384 KiB，A 在 128/256 KiB，scaleA 在 64/448 KiB，scaleB 在 96/480 KiB。
UB 的 FP4 输入在 0/16/32/48 KiB；FP8 输出从 64 KiB 开始，每个 256 字节向量按四个 stage 交错，步长 1024 字节。
VF 和 UB→L1 DMA 使用相同的交错步长，DMA 将当前 stage 的向量收集为连续 NZ 数据。
AIV0 在第一半块反量化前等待对应 L1 B 槽位空闲。
AIC 先发起 scale 再发起 A 搬运，每个 K 窗口使用一个 MTE2→MTE1 完成事件；L0A/B 同步轮转并共用一个 M→MTE1 空闲事件。
scale L1 视图按实际窗口的成对 scale 数量创建，保持完整 K4096 窗口及尾窗口的紧凑布局。
GMM1 发布 tile 完成标志前已有 FIX_S 等待，End(true) 跳过 slice 末尾重复的 FIX_S。
CATLASS 行主序写回时执行 MM layout 状态切换、跨核 token 排空及尾 K 补零。

A 与 scale 的事件独立保护各自窗口，B 由跨核标志保护。64 对齐的 B 块在 MTE1 等待就绪；不足 64 对齐的尾块在 MTE2 等待并补零，再通知 MTE1。

路由 GMM1 按实际 BS 选择布局和任务粒度，AIC、AIV0、AIV1 和 GMM2 ready 使用同一个 `FuseGateUp` 特化：

- BS≤256：GM 权重和 scale 离线重排为 `[gate128, up128]`。逻辑 N128 tile 只调用一次 `ComputeProjection`，以 `2*n` 为列偏移计算物理 N256；AIV0 同样只调用一次反量化，结果写回交错 workspace。
- BS>256：GM 权重和 scale 保留 `[gate | up]`。逻辑 N256 tile 分别在 `n`、`n+I` 进行 gate/up 投影及反量化，结果写回 planar workspace；两次投影完成后发布一个逻辑 tile 标志。尾 tile 按实际 N 处理。

SwiGLU 根据同一布局取数、写回逻辑 N 列；路由 GMM2 就绪计数分别按 N128/N256 计算，不改变 GMM2 的任务粒度。
两种权重布局在 Python 准备阶段生成，运行时不转换。共享专家使用 planar 权重，decode 使用 GMM1 N192 / GMM2 N224，prefill 使用 N256 配对投影；共享 GMM1 使用局部 kernel 和独立完成通知。
A8W4 使用独立 A 窗口、条件 K512 B 窗口及半块四缓冲；实际性能仍需同 shape 上机测量。
GMM2 写回行主序结果。共享专家的 A8W4 调用也复用此 kernel。

通用 A8W8/A4W4 GMM 使用 `mega_moe_mx_matmul.hpp` 和 `block_mmad_mx.hpp` 的 CATLASS MX 流水。
适配层保留 tensor-API slice 起始地址和父矩阵行距，支持 ND/NZ 权重、GM 输出及指定 AIV0 的 UB 输出。
通用路径固定使用 K256 数据窗口及 K512 scale 窗口。
所有 GMM 后端均使用 CATLASS。
后处理位于 `../epilogue`。

设备验收需要覆盖完整 K/尾 K、gate/up 列偏移、跨专家复用、共享专家和不同 BS。
CPU 接口及数值参考测试见 [MegaMoE 使用说明](../../../../README.md#4-测试与验证范围)，
不能替代 Ascend950 的设备编译、FP8/FP4 精度、跨核同步和性能验证。
