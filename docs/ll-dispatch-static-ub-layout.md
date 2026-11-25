# LL Dispatch：对齐 DeepEP-Ascend 的静态布局与代码形态

## 1. 目标与范围

把 LL dispatch 改造成 DeepEP-Ascend 的形态：

- 除 `num_tokens` 外，所有形状都是 JIT 模板轴；
- UB 采用**两层静态结构**：纯状态结构 + `UBLayoutWithBuffer`（内含类型化数组）；
- 尺寸由 `static constexpr` 公式与语义断言决定，取消手写 offset 与 bump 分配器；
- 三段有状态类（Counts/Sender/Receiver）退化为“纯数据 struct + 阶段代码”，向 Ascend 的扁平 `dispatch_impl` 靠拢；
- 运行时入参只剩：`num_tokens`、`rank_idx`、`magic`、`profile*`、各 GM 指针、`core` 上下文、`physical_ranks_per_host`、`peer_mem_bytes`、`stage_block_num`、`ffts_val`。

范围仅 LL dispatch（含 Host JIT 生成与容量准入）；combine/ubx 不动。

## 2. 与 DeepEP-Ascend 的对照

| 维度 | DeepEP-Ascend | 本项目目标 |
|---|---|---|
| 结构层次 | 两层：`DispatchUBLayout`（状态/scratch）+ `UBLayoutWithBuffer`（state + 缓冲） | 相同 |
| 成员形态 | 类型化数组/标量，如 `int expert_histogram[kNumExperts]`、`volatile int ready` | 相同，禁止 offset 表/getter |
| 尺寸来源 | `static constexpr` 公式 + `math::min/ceil_div` | 相同 |
| 容量校验 | `EP_STATIC_ASSERT(kNumAvailableMetadataBytes >= 4 * kNumMetadataTokenBytes)` | 同风格语义断言 + `sizeof` 上界 |
| 条件缓冲 | 模板 `bool` 参与公式，长度可为 0 | 相同，不用 `std::conditional_t` |
| 代码形态 | 单个 `dispatch_impl` 函数 + 少量纯数据 struct | 阶段代码合并，状态收敛到 `DispatchContext` |
| 跨阶段 UB 预留 | 无 | 保留 `kCountTxBytes`（见 §6），这是唯一例外 |

### 2.1 实现细节差异审计（以 Ascend 为准）

| 项目 | Ascend 写法 | 我们的修正 |
|---|---|---|
| 尺寸/参数类型 | `int`、`static constexpr int` | 布局与公式统一用 `int`，不再 `uint32_t` 混用 |
| UB 对齐常量 | `compiled.hpp` 里 `static constexpr int kNumUbAlignmentBytes = 32` | 在 `compiled.hpp` 增加同名常量，删掉代码里的 `32U` 字面量 |
| 容量断言 | `EP_STATIC_ASSERT(kNumAvailableMetadataBytes >= 4 * kNumMetadataTokenBytes, ...)` | 用语义下限断言（如 `>= 4 * kPayloadBytes`），而非仅 `sizeof <= kUbBytes` |
| stage 数公式 | `math::min(8, math::ceil_div(...))` + 除法取可用块数 | 相同写法，显式 `math::ceil_div`，不再手写三元 |
| 成员对齐 | 每个缓冲成员 `alignas(kNumUbAlignmentBytes)` | 相同 |
| 就绪标志 | `volatile int local_copy_ready, metadata_ready` | 用 `volatile int` 状态字，替代 event 状态成员 |
| 条件区域 | 模板 `bool` 参与公式、可 0 长度 | 删除 `std::conditional_t` 与运行时 `if` 改布局 |
| 访问方式 | 直接成员访问（`layout.expert_histogram`） | 删除 getter 与 `kXxxOffset`；内核内不再 `reinterpret_cast<__ubuf__ T*>(offset)`，一律通过成员引用 |
| token 布局 | `layout::TokenLayout(0, sfBytes, topk, withMeta).get_num_bytes()` 直接进 `static constexpr` | `DispatchServerDedupDataLayout` 改成同形态的 constexpr 类，提供 `get_num_bytes()/get_x_ptr()` |
| 状态组织 | `handle::HcommJettyInfo`、`handle::HcommPeerInfo` 等 `handle::` 结构 | 传输/RDMA 元数据收敛到 `comm::` 下的小型纯数据结构，放进状态 struct |

## 3. 轴与派生量

编译期轴（沿用现有 11 个 JIT 形状轴）：

`kNumRanks, kNumMaxTokensPerRank, kNumHiddenElems, kNumTopk, kNumExperts, kNumVecCores, kNumUbBytes, kNumRanksPerGroup, kPrequantized, kEnableDedup, kRdma`

派生常量：

| 名称 | 表达式 |
|---|---|
| `kInputMode` | `kPrequantized ? PrequantizedFp8Packs : Plain16` |
| `kHiddenBytes` | `kPrequantized ? kNumHiddenElems : kNumHiddenElems * 2` |
| `kLocalExperts` | `kNumExperts / kNumRanks` |
| `kSendCores / kRecvCores` | `kNumVecCores / 2` |
| `kData` | `DispatchServerDedupDataLayout(kNumMaxTokensPerRank, kNumHiddenElems, kNumTopk, kNumExperts)`（见 §5） |
| `kCountTxBytes` | `DispatchServerDedupCountLayout(kNumExperts, kNumVecCores).statusBytes` 的等价值（见 §6） |

## 4. 两层静态布局（Ascend 风格）

### 4.1 纯状态结构

```cpp
template <uint32_t kRanks, uint32_t kExperts, uint32_t kMaxTokens, uint32_t kTopk,
          uint32_t kVecCores, bool kRdma>
struct alignas(kNumUbAlignmentBytes) DispatchUBLayout {
    static constexpr uint32_t kMaxPerCore = math::ceil_div(kMaxTokens, kVecCores / 2U);
    static constexpr uint32_t kMaxServerWords = kRanks / 16U + 16U;

    int32_t  routes[kMaxTokens * kTopk];
    uint32_t qdst[kMaxTokens * kTopk];
    float    route_weights[kMaxPerCore * kTopk];
    float    source_weights[kMaxPerCore * kTopk];
    uint32_t source_masks[kMaxPerCore];
    uint32_t direct_tasks[kMaxPerCore * kTopk];
    uint32_t server_state[kMaxServerWords];
    int32_t  prefix[kExperts];
    // count 阶段状态（与 §5.2 对应）
    int      expert_histogram[kExperts];
    int      rank_histogram[kRanks];
    int      rank_psum[kRanks];
    int      cumsum_state[kRanks];
    volatile int32_t stage_ready;             // 取代运行时 event 状态
    // RDMA 专属状态（kRdma=false 时长度为 0）
    uint8_t  rdma_state[kRdma ? kRdmaStateBytes : 0];
};
```

### 4.2 状态 + 缓冲

```cpp
template <typename layout_t, uint32_t kTopk, uint32_t kHiddenBytes, uint32_t kVecCores,
          uint32_t kUbBytes, bool kRdma, bool kPrequantized>
struct UBLayoutWithBuffer {
    static constexpr uint32_t kCountTxBytes  = ...;                       // §6
    static constexpr uint32_t kPayloadBytes  = kHiddenBytes + ...;        // 与 kData 一致
    static constexpr uint32_t kFixedBytes    = sizeof(layout_t) + kCountTxBytes;
    static constexpr uint32_t kTokenStride   = ...;                       // 来自 kData
    static constexpr uint32_t kNumLanes      = math::min<uint32_t>(8U, (kUbBytes - kFixedBytes) / kTokenStride);
    static constexpr uint32_t kNumHiddenStages = math::min<uint32_t>(8U,
        (kUbBytes - kFixedBytes - kNumLanes * kTokenStride) / kPayloadBytes);
    EP_STATIC_ASSERT(kNumLanes > 0, "Insufficient UB for dispatch lanes");
    EP_STATIC_ASSERT(kNumHiddenStages > 0, "Insufficient UB for hidden stages");

    uint8_t count_tx[kCountTxBytes];
    layout_t workspace;
    alignas(kNumUbAlignmentBytes) uint8_t wire[kNumLanes * kTokenStride];
    alignas(kNumUbAlignmentBytes) uint8_t hidden[kNumHiddenStages][kPayloadBytes];
    alignas(kNumUbAlignmentBytes) uint8_t rdma_ws[kRdma ? kRdmaWorkspaceBytes : 0];
};
static_assert(sizeof(UBLayoutWithBuffer<...>) <= kUbBytes, "UB capacity");
```

要点：

1. 偏移由成员顺序与 `sizeof` 隐含；不再出现 `kXxxOffset` 或 getter。
2. 条件区域靠模板 `bool` 给 0 长度，不用运行时 `if` 改变布局。
3. 容量按 `kNumMaxTokensPerRank` 计，而不是实际 `num_tokens`（Ascend 同做法）。
4. 所有语义下限用 `EP_STATIC_ASSERT`（如 `>= 4 * kPayloadBytes`），与 Ascend 一致。

### 4.3 容量化对 UB 的实际影响（按 256 容量核算）

现实现里只有 4 个小数组按“每核实际 token 数 `count_`”分配，其余（`routes/qDst/weights 前缀` 等）本来就是按 `kNumMaxTokensPerRank` 容量分配：

| 区域 | 现尺寸 | 容量化尺寸（每核 256/sendCores） | 最大增量 |
|---|---|---|---|
| `routeWeights` | `count_ * kTopk * 4` | `8 * 8 * 4 = 256` | +256 |
| `sourceWeights` | 同上 | 256 | +256 |
| `sourceMasks` | `count_ * 4` | 32 | +32 |
| `directTasks` | `count_ * kTopk * 16` | 1024 | +1024 |

（按 `kNumVecCores=64`、`sendCores=32`、`kTopk=8`、`kNumMaxTokensPerRank=256` 计。）

结论：

1. `num_tokens = 256` 时，`count_` 恰好等于每核容量 8，**布局与原实现逐字节相同**；
2. `num_tokens < 256` 时，这 4 个数组最多多占 **1568 字节**（`= 8*8*4*2 + 8*4 + 8*8*16`），相对于 `tokenStride`（7168 维 BF16 时 15872B）不到 10%；
3. `lanes = min(8, 可用/stride)` 有 8 的硬上限，除非“可用”恰好落在 `8*stride ± 1568` 的窄带内，否则 lanes 不变；满 256 时必然不变。

## 5. `data_` constexpr 化

现状：每个 Sender/Receiver 持有 `data_`（`EpServerDedupDataLayout`）以及 `h_/k_/experts_/hiddenBytes_`，均由 `Init` 用运行参数算出。

由于这些值只由形状决定，改为：

```cpp
static constexpr auto kData = DispatchServerDedupDataLayout(
    kNumMaxTokensPerRank, kNumHiddenElems, kNumTopk, kNumExperts);
```

- 内核不再保存 `data_` 与 `h_/k_/experts_`，直接用 `kData.tokenStride` 等常量；
- 前提：`DispatchServerDedupDataLayout` 及相关函数改成 `constexpr`（去掉 `trap`、不依赖指针）；
- `tokenStride % 512`、`scaleBytes` 对齐等检查改为 `static_assert`；
- 效果：`Init` 只剩指针/`view`/`rank`/`magic`/`num_tokens`，与 Ascend 初始化形态一致。

### 5.2 count 阶段同样 constexpr 化

`ServerDedupCounts::Init` 现在接收 `num_tokens, topK, experts, expertTokenNumsType, magic, core`，但其中只有 `num_tokens/magic` 是真运行时：

| 现成员 | 来源 | 处理 |
|---|---|---|
| `layout_ = DispatchServerDedupCountLayout(experts, coreNum_)` | `kNumExperts, kNumVecCores` | `static constexpr kCountLayout` |
| `epWorldSize_ / moeExpertNumPerRank_` | `kNumRanks / kNumExperts` | 常量 |
| `ubRouteCount_ = kNumMaxTokensPerRank * topK` | 模板轴 | 常量 |
| `useAllCoreCount_` | `kNumMaxTokensPerRank` 与阈值 | 常量 |
| `packedFrames_ / directRanks_` | `kRdma / kNumRanks` | 常量 |
| `statusCntAlign_` | `CountWireBytes(kMaxTokens*kTopk)/32` 或 `alignedRecords` | 常量 → `kCountTxBytes` |
| `startStatusIndex_/endStatusIndex_/recStatusNumPerCore_` | `kNumExperts, kNumVecCores` | 常量 |
| `expertIdsCnt_`、count 工作分片 | `num_tokens` | 运行时（仅循环范围，不参与布局） |

count 的 UB 状态（`histogram_`、累加器、cumsum 状态、`prefix`）并入 `DispatchUBLayout` 的成员数组；`retirement_` 保留为小型运行时状态（pending 位），放进 `DispatchContext`。这样 count 阶段的 UB 占用也与 `num_tokens` 无关。

## 6. `kCountTxBytes`（原 `kCountReserve`）

现状：`ReservedBytes() = statusCntAlign_ * kEpServerDedupCountRecordBytes`，是每核 count **帧发送的 UB 源区**，位于 `[0, ReservedBytes)`。

为什么必须保留：

- `comm::CountFrame(..., records, wireBytes, count)` 直接从该 UB 区把 count 帧写到 peer（[transport.hpp:212](file:///c:/Users/liyan/Project/DeepEP-Huawei/deep_ep/include/deep_ep/comm/transport.hpp#L212)）；
- RDMA 模式下 TX 完成由 `CountRetirement` 跟踪，quiet 延迟到 `rdmaBatch_` 收尾（[dispatch.hpp:1490](file:///c:/Users/liyan/Project/DeepEP-Huawei/deep_ep/include/deep_ep/impls/ep/dispatch.hpp#L1490)），在途期间源区不可覆盖；
- 因此 sender 的 UB 起点必须抬到该前缀之后。

注意区分：后段 quiet 的 staging 用的是 `histogram_`（[dispatch.hpp:773](file:///c:/Users/liyan/Project/DeepEP-Huawei/deep_ep/include/deep_ep/impls/ep/dispatch.hpp#L773)），不是这块 records 区。

能否移除：

- **可以，但有代价**：若在 sender 分配前完成 count 退休（非 RDMA 已这样走 `else counts.RetireOwnScratch();`，[dispatch.hpp:2303](file:///c:/Users/liyan/Project/DeepEP-Huawei/deep_ep/include/deep_ep/impls/ep/dispatch.hpp#L2303)），即先等待在途 TX/quiet 完成，则前缀可以去掉；
- 代价是 count 与 dispatch 不再重叠，RDMA 模式下会插入一次等待（首包延迟与吞吐都受影响）；
- 折中：前缀不一定按整帧大小，而是“每核一帧”的最小 staging；这需要传输层支持按记录发送，属额外改动。

结论：本方案**保留前缀但常量化**（`kCountTxBytes`），并把“移除前缀”列为可选优化（§12），默认不做，避免牺牲重叠。

## 7. N 级 hidden stage 环

现在 `input_[0]/input_[1]` 是两行全量 payload 双缓冲。Ascend 用 `kNumHiddenMTEStages = min(8, 可用/hiddenStageBytes)` 的 stage 环。

可改为：

- `kNumHiddenStages` 由 §4.2 公式给出（`static constexpr`）；
- `hidden[kNumHiddenStages][kPayloadBytes]`，每 stage 一个 event，`(idx + 1) % kNumHiddenStages` 轮转；
- 只影响隐藏维预取深度，不改变 lane 语义；
- 风险：事件数上限（≤8）、在线 E4M3 量化与 stage 环的同步、profile 计数点位置。

## 8. 代码形态：向 `dispatch_impl` 靠拢

目标形态（Ascend 是单函数 + 纯数据 struct）：

- `ServerDedupCounts` → 阶段代码，状态并入 `DispatchUBLayout`（histogram/retirement）；
- `ServerDedupSender/Receiver` → 阶段代码；
- 跨阶段共享量收敛为：

```cpp
struct DispatchContext {
    TransportView view;
    __gm__ uint8_t *x, *ids, *weights, *expand_x, ...;
    uint32_t rank_idx, num_tokens;
    int64_t magic, peer_mem_bytes, ffts_val;
    uint32_t stage_block_num;
    CoreContext core;
    profile::DeviceProfile* profile;
    comm::DispatchRdmaBatch batch;   // 有状态传输对象保留
};
```

- `Init/Process` 生命周期取消，`dispatch_impl` 开头解包指针与标量，随后顺序执行 count → send → recv 阶段；
- `dispatch_impl_sk<slot>` 复用同一实现，`slot` 只影响 core index 映射。

代价：函数体变长（Ascend 约 400 行，我们因 RDMA/count/profile 会更长），需用注释分节。

## 9. 分阶段计划

| 阶段 | 内容 | 验收 |
|---|---|---|
| P1 | `DispatchServerDedupCountLayout` / `DataLayout` 改 `constexpr` 可求值 | Host/设备均能 `static_assert` |
| P2 | 去掉 `h/k/experts/inputMode` 运行参；`data_` 等改常量 | 设备 TU 编译通过 |
| P3 | 落地 §4 两层结构；count 状态并入 `DispatchUBLayout`；容量化所有区域；`EP_STATIC_ASSERT` + `sizeof` 上界 | 断言全绿 |
| P4 | 删除 `UbArena`、运行时 lanes 推导；`if constexpr` 条件区 | lanes/stages 为常量 |
| P5 | （可选）N 级 hidden stage 环 | 流水正确 + 性能对比 |
| P6 | 合并三段类为 `DispatchContext` + 阶段代码（§8） | 行为不变，设备 TU 通过 |
| P7 | Host/JIT 收敛：去掉形状实参；容量准入用同一布局 | OFF wheel + NPU golden |
| P8 | 代码简化（见 §10） | 单独提交，逐批验证 |

## 10. 后续简化清单（P8）

1. 消失项：`h_/k_/experts_/hiddenBytes_/data_/wireFp8_`、`if (!h || !k)` 校验、运行时 `lanes_` 取小。
2. 重复合并：[dispatch.hpp:2008](file:///c:/Users/liyan/Project/DeepEP-Huawei/deep_ep/include/deep_ep/impls/ep/dispatch.hpp#L2008) 与 [L2035](file:///c:/Users/liyan/Project/DeepEP-Huawei/deep_ep/include/deep_ep/impls/ep/dispatch.hpp#L2035) 的 `expandX/dynamicScales` 输出两段；mask/index/relay 复制样板。
3. count 布局与 combine 侧的重复公式合并为一份 `constexpr`。
4. group 4/16 两套路径合并为同一模板代码。
5. `EpServerDedupInputMode` 两态枚举可删，直接用 `kPrequantized`。
6. profile 调用点用 `if constexpr (kProfileEnabled)` 裁剪。
7. `dispatch_impl` 与 `dispatch_impl_sk` 去重。
8. 死代码盘点：`LlTilingKey`/`LlTilingKeyValid`、`EpServerDedupEnabled`、`kLlCoreProtocolVersion`。

## 11. 风险

1. **UB 放大**：容量化后固定区变大，`kNumLanes` 可能不足；靠 stage 复用与 N 级环调节。
2. **偏移漂移不可编译期发现**：禁止手写 offset；必需边界全部由成员顺序 + `sizeof` 断言覆盖。
3. **Host/设备一致**：`kCountTxBytes` 与 `get_dispatch_buffer_size`、combine 的 count 布局必须共用同一 `constexpr`。
4. **命名统一**：模板轴 `kNumHiddenElems` 与 JIT 源里的 `num_hidden_bytes` 要统一为 elems。
5. **RDMA 生命周期**：count 前缀缩容需先确认 quiet 时序，否则在途 TX 读写冲突。
6. **单函数化的可维护性**：P6 后函数体变长，需靠分节注释与 `if constexpr` 分区，避免变成不可读的巨型函数。

## 12. 可选优化：移除 `kCountTxBytes` 前缀

默认不做，仅在确认性能可接受时评估：

1. 在 sender 分配 UB 之前调用 `RetireOwnScratch()`（RDMA 模式也一样），等待 count TX/quiet 完成；
2. 布局起点归零，`UBLayoutWithBuffer` 去掉 `count_tx` 成员、`kFixedBytes` 减少；
3. 验收：当前环境只能编译，先保证设备 TU 通过；首包延迟/吞吐对比**等可用 NPU 环境后再执行**。

若未来传输层支持按记录发送 count（而不是整帧从 UB 直发），可把前缀缩到一个记录的 32B，此时即使保留前缀也几乎不占 UB。

## 13. 验证策略（当前环境只能编译、不能运行）

现环境没有可执行的 NPU，因此：

- **可立即执行**：bisheng 设备 TU 编译、OFF wheel 构建、`EP_STATIC_ASSERT`/`static_assert`、Host `g++` 语法检查。
- **可用编译期代理指标**：
  - `kNumLanes`、`kNumHiddenStages`、`sizeof(UBLayoutWithBuffer)`、`kCountTxBytes` 全部是常量，可在编译期断言；
  - 由于不能运行程序，等价性以**编译期断言锁定期望值**为准：把当前实现实际达到的 `kNumLanes/kNumHiddenStages/sizeof` 记成期望常量，写成 `static_assert(kNumLanes == 8)`、`static_assert(kNumHiddenStages == N)`，改造过程中任何偏离都会编译失败；
  - 关键断言示例：`static_assert(kNumLanes >= 4, "lanes 下限")`、`static_assert(sizeof(...) + kCountTxBytes <= kNumUbBytes)`。
- **不可执行**：msprof、端到端吞吐、首包延迟，以及 §12 的前缀移除取舍。这些统一标记为“待 NPU 环境”验证项，方案中不作为本次验收门槛。
- **风险处置**：容量化（P3）的 UB 差异已核算，见 §4.3；满 `kNumMaxTokensPerRank=256` 时与现实现完全一致，不做“lanes 必须一致”的额外硬约束。
