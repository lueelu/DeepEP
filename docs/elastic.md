<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# Elastic dispatch / combine

`ElasticBuffer` 通过 `dispatch()` 分发 token，通过 `combine()` 归并专家输出。两者使用匹配的 `EPHandle`。本页说明当前代码的调用约束；资源生命周期见 [API 指南](api.md)，设备运行命令见[测试入口](../tests/README.md)。

记 P 为 EP rank 数，B/T 为每 rank 实际 token 数，M 为 token 容量上限，H 为 hidden，K 为 top-k，E 为全局专家数，R 为本 rank 的有效接收行数。

## 路径选择

fresh dispatch 按调用参数、非零 Buffer 上限、实际 B 的顺序解析 `M=num_max_tokens_per_rank`。当 `0 < B ≤ M ≤ 256` 且 dtype、EP、权重及调用选项符合 LL 要求时，优先选择低时延路径。否则仅在 HT 支持完整调用时使用 HT；不支持的组合报错。

LL 选中后要求工作区**恰好 8 GiB**。容量错误直接报错，不因此切换到 HT。构造 Buffer 不选择算法，也不在 dispatch 时扩容、重新清零或创建 Session。`get_buffer_size_hint()` 只计算 HT 需求，不能用于计算 LL 容量。

combine 根据 `handle.is_low_latency` 执行匹配路径，不改变 handle 的后端。不要在同一个 Buffer 中交错使用 HT 和 LL。小 BS 不保证走 LL；例如 M=257 的调用不满足 LL 条件。

## 共同要求

- 环境为 Ascend950、CANN 9.2.0，SHMEM 版本由根目录 `dependencies.lock.json` 固定；HT 依赖 SHMEM 的 shared-jetty 支持，准备步骤见[构建指南](build.md)。
- 各 rank 使用相同的 B/T、K、E、dtype、容量和调用顺序，专家均匀分配，E 可被 P 整除。
- 所有 Tensor 位于创建 Buffer 的 NPU，连续存储，不使用 autograd。调用方负责设备绑定，不能依赖 LL 包装层拒绝所有错误设备输入。
- 所有 peer 的 SHMEM 工作区可通过 MTE 映射；连续 8 rank 为一个逻辑 server。逻辑 server 不等同于物理主机，也不是任意可配置的分组。
- 使用 expanded 布局，`do_expand=None/True`、`expert_alignment=None/1`。普通布局和 padding 不在支持范围内。

## 高吞吐路径

### 支持范围

| 项目 | 约束 |
| --- | --- |
| EP | dispatch / combine：2、4、8、16、32、64、128、256 |
| Shape | H=7168，1≤T≤10240，1≤K≤16，P≤E≤1024；EP256 combine 在默认 chunk=256 下要求 T≤4608（44 KiB 常驻控制区），此 UB 限制不适用于纯 dispatch |
| 编码与容量 | `P*P*T*K < 2**31`（EP256 要求 `T*K < 32768`），地址编码和实际接收/gather 行数还须通过布局检查 |
| 输入 | BF16 `[T,7168]`，或 `(FP8 E4M3FN [T,7168], FP32 scales [T,56])` |
| 路由 | int64 `topk_idx [T,K]`；可选 FP32 `topk_weights [T,K]` |
| 调度 | BF16 dispatch：`num_compute_units=0/32`；FP8 dispatch 和 combine：`0/64`；`num_qps=0/2` |
| 构造配置 | `num_allocated_qps=0/6`；0 使用固定配置 |
| 同步 | `do_cpu_sync=None/True`；支持 `previous_event` 和 `async_with_compute_stream=True` |

HT 不支持 epilogue event、累积统计、列主序 scales 或 `allocate_on_comm_stream=True`。异步返回仍可能包含计数读取和容量检查所需的 Host 同步。各 rank 的 weights 选项和 fresh/cached 模式须一致。

### Dispatch 与 handle 复用

返回 `(recv_x, None, recv_weights, handle, event)`。`recv_x` 为 `[R,7168]`；FP8 时为 `(payload [R,7168], scales [R,56])`。`recv_weights` 为 FP32 `[R]` 或 `None`。

输出按 expert/source/token/slot 展开。无效 expert ID（负数或 ≥E）不产生行；重复 expert 槽位分别保留。R=0 时返回空 Tensor。payload、scales 和权重分别搬运，dispatch 不对 hidden 乘权，也不执行量化。

fresh 调用执行 Notify 后搬运数据。cached 调用传入 `handle=handle` 并省略 `topk_idx`，可以更换 payload 和 weights，但必须保持原 T/K/E、路由和布局。默认复制输入索引；指定 `do_handle_copy=False` 时保持原始索引不变。handle 只能由创建它的 Buffer 使用，内部路由 Tensor 不得修改。

### Combine 与乘权

输入为 BF16 `[handle.num_expanded_tokens,7168]`，保持 dispatch 的行顺序。返回 `(out, out_weights, event)`，其中 `out` 为 BF16 `[T,7168]`。FP8 dispatch 后也使用 BF16 专家输出。空接收 rank 传 `[0,7168]`，仍须参与通信。

`topk_weights=None` 时，`out_weights` 为 `None`。也可传入与展开行一一对应的 FP32 `topk_weights [R]`，通过独立通道回传为源 rank 的 FP32 `[T,K]`，无效路由槽位填 0。`bias` 仅支持 `None`。

**独立权重回传不会对 hidden 乘 gate；HT 模型侧仍须在专家计算或其后处理中应用一次路由权重。** 同一个 handle 可以重复 combine 或用于 cached dispatch。

归约先在逻辑 server 内以 FP32 累加并写出 BF16 partial，再在源端以 FP32 累加 partial 并写 BF16。舍入结果可能与一次性全 FP32 求和不同。

HT combine 默认使用三个 SO 窗口；T=4096、chunk=256 时按 8/4/4 个逻辑 chunk 划分，逻辑 chunk 不足三个时减少窗口数。

### 调用片段

以下片段使用当前 HT 测试入口的显式 rank/world_size 方式，不需要 Gloo/HCCL 进程组。假定 NPU、endpoint、rank、输入 Tensor 和专家计算函数已由调用方准备。M=1024 使用 HT；FP8 输入需要在构造时配置对应的容量预算。

```python
from deep_ep import ElasticBuffer

buffer = ElasticBuffer(
    group=None,
    rank=rank,
    world_size=world_size,
    num_max_tokens_per_rank=1024,
    hidden=7168,
    num_topk=6,
    explicitly_destroy=True,
)
try:
    recv, _, weights, handle, event = buffer.dispatch(
        x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_experts=1024, do_expand=True,
    )
    event.current_stream_wait()
    expert_out = run_experts(recv, handle)  # 返回 BF16，保持专家内行顺序。
    if weights is not None:
        expert_out = (expert_out.float() * weights[:, None]).to(expert_out.dtype)
    out, _, event = buffer.combine(expert_out, handle)
    event.current_stream_wait()
finally:
    buffer.destroy()
```

该片段的 `run_experts` 不预先乘 gate。专家计算若已经包含乘权，则省略片段中的重复乘法。

## 低时延路径

### 输入与选项

LL 按容量上限 M 选择传输路径：M≤32 不执行 server 级去重，32<M≤256 执行 server 级去重。路径由 M 决定，而非实际 B；例如 B=32、M=96 使用去重路径。dispatch 将 M 保存在 handle 中，combine 使用相同的选择，实际 Tensor shape 仍由 B 决定。

| 项目 | 约束 |
| --- | --- |
| EP、token | P=16/32/64/128，`0 < B ≤ M ≤ 256` |
| 容量 | 构造时显式 `num_bytes=8 << 30`，各 rank 相同 |
| 输入 | BF16/FP16 `[B,H]`，或 FP8 E4M3FN payload 与 scales 元组 |
| 路由 | int64 `topk_idx [B,K]`；必须传入 FP32 `topk_weights [B,K]` |
| Shape | H>0，1≤K≤32，P≤E≤1024，E 可被 P 整除；组合还须满足 Host 布局、UB 和工作区校验 |
| 调度 | 默认固定调度，`num_compute_units=0`、`num_qps=0` |
| 流与复用 | 默认流选项；不支持 cached dispatch、previous/epilogue event 或异步请求 |

构造时 `hidden` 只接受 0 或 7168，`num_topk` 接受 0～16；0 表示不预绑定该维度，dispatch 仍执行实际 shape 校验。LL ST 的 top-k 范围为 1～16，不能据此宣称已验证 kernel 布局接受的全部组合。

FP8 scales 为连续 `[B,ceil(H/128)]`：INT32 存放四个 per-32 E8M0 scale 的打包值，FP32 表示一个 per-128 scale。dispatch 按位搬运并保留 scales dtype，不转换量化分组。调用方保证 payload、scale 和下游专家计算格式匹配。

LL 同样可以使用显式 rank/world_size，并将工作区设置为 8 GiB。此方式返回容量布局 `[C,H]`：

```python
buffer = ElasticBuffer(
    group=None,
    rank=rank,
    world_size=world_size,
    num_bytes=8 << 30,
    num_max_tokens_per_rank=96,
    hidden=7168,
    num_topk=6,
    explicitly_destroy=True,
)
```

需要精确接收行数 `[R,H]` 时，将上例的 `group=None` 和 `rank/world_size` 改为 `group=ep_group`，并在 dispatch 时使用 `do_cpu_sync=None/True`。`ep_group` 是已初始化的 Gloo/HCCL EP 进程组，由它确定 rank 顺序与规模；两种身份配置不能同时传入。该组负责计数协调，payload 仍由 SHMEM 和 LL kernel 传输。使用结束后，各 rank 集体调用 `buffer.destroy()`。

### 返回与有效区域

dispatch 返回 `(recv_x, None, None, handle, event)`，权重由 handle 保留，不作为第三项输出。接收行数取决于同步选项：

| 条件 | 输出行数 |
| --- | --- |
| `group=None`，使用显式 rank/world_size | 返回 `[C,H]`，`C=P*M*min(K,E/P)`；`do_cpu_sync` 不改变此结果 |
| 传入 EP group，`do_cpu_sync=False` | 仍返回 `[C,H]`，不通过该组汇总精确接收行数 |
| 传入 EP group，`do_cpu_sync=None/True` | 使用该组 all_reduce 获取精确 R，返回 `[R,H]` |

容量布局中仅有效路由对应的行可供消费，容量尾部不是有效 token。

FP8 输出为 payload 与同 dtype scales 的元组。LL 的 `expert_recv_counts`、`num_unaligned_recv_tokens_per_expert`、`psum_num_recv_tokens_per_expert` 均为 `None`；内部 kernel 计数不属于可读取的公开专家计数接口。未使用行或槽位不承诺为零或 -1，不得把容量缓冲区的全部行当作有效专家输入。

combine 接受与 handle 行数一致的连续 BF16/FP16 专家输出，返回 `(out, None, event)`，out 为同 dtype `[B,H]`。精确 R 路径无需扩容至 C；FP8 dispatch 后的专家输出仍须为 BF16/FP16。

**LL combine 在内部应用 handle 保存的路由权重，专家输出不得预乘 gate。** `topk_weights` 和 `bias` 均传 `None`。LL handle 的 combine 不回退 HT；可用返回和重复调用示例见 [LL 系统测试](../tests/elastic/test_ep_low_latency.py)。

## 工作区与输出所有权

HT 的 `num_bytes` 包含 Notify/dispatch 和 combine 两个固定半区，按 2 MiB 对齐，最高 32 GiB。size hint 的接收/gather 行预算为 `min(P,4)*T*K`，不是最坏路由上界。严重热点需要更大的显式容量；超过实际容量时在 payload 启动前报错，不自动扩容。

EP256 shape 超出默认 combine 的 UB 上限时，`get_buffer_size_hint()` 仅估算 dispatch；返回容量不代表该 shape 可用于 combine。实际 combine 调用仍检查 UB 和工作区容量。

shape 合法不保证所有路由都满足容量限制。例如 EP64、T=1024、K=16、E=512、BF16 的所有槽位指向同一专家时，目标 rank 展开为 1,048,576 行，当前固定布局需要超过 32 GiB 的工作区，超过 Session 上限。

LL 的 8 GiB 是对称工作区容量。两条路径都另行分配路由元数据和输出 Tensor，因此工作区容量不等于总显存用量。输出由框架管理，后续通信或 Buffer 销毁不会将其存储作为新的工作区覆盖。

原生布局与路由编码是内部实现细节，不作为应用解析 handle 的协议。调试和测试入口见[测试说明](../tests/README.md)，性能统计见[性能说明](performance.md)。

## 随机路由测试

HT dispatch 和 combine 测试入口均支持 `--route random --seed 1`，`--route all` 也包含 random。
随机路由使用确定性的 32 位伪随机序列，
按 seed/source/token 生成专家槽位；允许重复专家，不做去重或强制均衡。所有 rank 必须使用相同的 seed，默认值为 1。
重复调用保持路由不变以验证 cached handle，payload 随迭代变化；更换 seed 后重新启动作业。
seed 记录在阶段日志、每 rank 结果和 profiler 元数据中，其余路由模式不受 seed 影响。
random 仅指随机路由，hidden/权重仍使用确定性精度输入，不代表随机浮点数值测试。

单独使用 random 时按实际路由计算容量；dispatch 的 `--route all` 还包含 hotspot，保留全局热点容量要求。
其他路由和运行方式见[测试说明](../tests/README.md)。
