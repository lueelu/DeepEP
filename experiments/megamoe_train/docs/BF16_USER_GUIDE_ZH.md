# MegaMoE BF16 训练算子用户指南

本文面向将 `MegaMoeBF16Train` 接入 PyTorch 训练框架的使用者，说明安装、公开接口、路由与权重适配、SHMEM 通信窗口、容量规划和使用限制。更新日期：2026-11-04。

## 1. 功能与适用环境

`MegaMoeBF16Train` 是一个 `torch.nn.Module`，融合 EP MoE 的 routing、dispatch、GMM1、SwiGLU、GMM2、combine，并通过 autograd 计算 token、路由权重和两层专家权重的梯度。

接入位置如下；框架完成 router / TopK 选择，算子完成专家计算及对应通信：

```text
训练框架：router / TopK → x、topk_ids、topk_weights、本地专家权重
                                ↓
MegaMoE：dispatch → GMM1 → SwiGLU → GMM2 → combine
                                ↓
训练框架：后续网络 → loss.backward() → 优化器
```

当前融合的专家网络是无 bias 的两层 SwiGLU。路由权重乘在 SwiGLU 之后、GMM2 之前。router、负载均衡辅助损失、专家参数注册、优化器及额外的数据并行梯度归约由训练框架管理。

| 项目 | 要求 |
| --- | --- |
| 设备与系统 | Linux、Ascend950；当前设备验收范围为单节点 |
| 环境版本 | 平台、CANN、SHMEM/CATLASS 与构建工具的版本要求见[仓库 README 的环境要求](../../../README.md#环境要求)和[构建指南](../../../docs/build.md)；本节不再重复 |
| 训练组件 | 构建时启用 `DEEPEP_BUILD_MEGAMOE_TRAIN=ON`，该开关默认关闭 |
| EP 身份 | 调用方创建已有 ProcessGroup 或 StoreGroup；ProcessGroup 初始化支持 Gloo / HCCL |
| 数据传输 | 训练算子使用 SHMEM 的 MTE/default 路径；传入 HCCL group 只提供身份与初始化协调，不切换 token 数据面 |

训练组件还要求：根 `deep_ep` Runtime、训练扩展和设备库来自**同一次构建**，各 rank 安装同一份 wheel。wheel 自带构建所用的 SHMEM 运行库，根加载器校验哈希并按包内绝对路径加载；运行时只需加载匹配的 CANN 环境，不需设置 `SHMEM_ROOT` 或添加外部 SHMEM 的动态库路径，也不能在同一进程预加载另一套 SHMEM。在目标运行镜像中构建；转移 wheel 时需匹配 Python/Torch ABI 及 glibc 等系统库。源码构建与安装步骤见[模块 README](../README.md#build-and-installation)，固定依赖见[根锁文件](../../../dependencies.lock.json)。

BF16/FP32 p 共用 `MegaMoeBF16Train`、训练 `_C` 扩展和 `libdeepep_megamoe_train_bf16.so` 名称；旧 BF16 调用无需改名或增加必填参数。但 typed tiling 与 p/dp 字节布局需要配套 Host/Device，不能只替换旧安装中的一个 SO。

首次检查安装时，在源码目录以外按[构建检查步骤](../../../.agent/skills/megamoe_train/build-megamoe-train/SKILL.md#4-检查安装与原生扩展加载)先调用根 `require_native()` 校验并预加载包内 SHMEM，再导入训练原生扩展；不要直接导入扩展绕过根加载器。正常构造 `MegaMoeBF16Train` 会执行这一步，无需框架侧额外调用。检查 `PYTHONPATH` 中的空项或末尾冒号，避免当前源码目录覆盖已安装包；完整 wheel 安装后再进行本文示例。

## 2. 最小示例

可运行示例、参数说明、容量取值与常见错误见[快速开始](BF16_QUICKSTART_ZH.md)。本节补充该示例涉及的
两项外部配置：SHMEM 端点与窗口容量，二者在多节点与多层接入时同样适用。独立精度对照与性能测试见
[测试入口](../tests/bf16/README.md)。

### 2.1 多节点端点配置

`DEEPEP_SHMEM_ENDPOINT` 是 DeepEP 读取的环境变量，随后作为 SHMEM 默认初始化模式的 `ip_port` 传入 SDK，用于初始化和控制协调。SDK 由 EP 组内 rank 0 创建 TCP 监听，其余 rank 连接该端点；应用无需单独启动一个自建 TCP 服务。

`127.0.0.1` 指当前进程所在机器的回环地址，只适用于本文单节点示例。多机时，选择 **EP 组内 rank 0 所在机器可监听、且所有参与 rank 都可达的 IPv4 地址**，同一 EP 组的所有机器和进程都设置同一个值。

例如机器 A 为 `10.0.0.10`、机器 B 为 `10.0.0.11`，且 EP rank 0 在 A；两台机器都执行：

```bash
export DEEPEP_SHMEM_ENDPOINT=tcp://10.0.0.10:29862
```

这里的 IP 是示例，需替换成实际地址；29862 是任选的未占用端口，不是固定服务端口。不要在 B 上改成 B 自己的 IP，也不要在每台机器分别使用回环地址。容器部署时，应保证 EP rank 0 能在该地址监听，其他容器/节点能访问对应 TCP 端口。

| 配置 | 负责的协调入口 |
| --- | --- |
| `MASTER_ADDR` / `MASTER_PORT` | PyTorch 启动和进程组 rendezvous；具体由框架/启动器设置 |
| `DEEPEP_SHMEM_ENDPOINT` | SHMEM 初始化和控制协调，使用独立的地址/端口配置 |

两个入口可以使用同一机器的 IP，但不能争用同一个监听端口，例如 PyTorch 使用 29500、SHMEM 使用 29862。EP rank 0 不一定是整个训练任务的 global rank 0，应按实际 EP 组成员选择主机。

endpoint 不是窗口内存地址，也不决定 NPU 数据通过哪种引擎传输。当前训练 Session 固定使用 MTE/default；把 endpoint 改成远端 IP 不会自动启用 UDMA/RDMA。当前算子设备验收仍限于单节点 Ascend950，多机还需验证对应硬件拓扑下的 peer 映射、MTE 数据可达性和同步行为，不能将 TCP 初始化成功视为多机训练已经通过验证。

该配置路径见 [DeepEP Runtime](../../../deep_ep/_runtime.py) 和 [Session](../../../csrc/runtime/session.cpp)；默认 bootstrap 的服务端选择见[固定版本 SHMEM 实现](https://gitcode.com/cann/shmem/blob/836d3fd6f9c2f3b10a6acac36ccfd531b104d4a7/src/host/bootstrap/shmemi_bootstrap_config_store.cpp)。

## 3. 公开接口

### 3.1 构造与关闭

构造签名为 `MegaMoeBF16Train(ep_group, local_mem_size, *, timeout=120, debug_collectives=False)`。

| 参数 | 类型 / 单位 | 说明 |
| --- | --- | --- |
| `ep_group` | 已创建的 ProcessGroup 或 StoreGroup | 提供 EP 组内 rank/size 和初始化协调，不能为 None；group 由调用方持有 |
| `local_mem_size` | Python int，字节 | 每 rank 的 SHMEM 窗口容量，至少 2 MiB、按 2 MiB 对齐，各 rank 相同；构造后固定 |
| `timeout` | Python int，秒，1..3600 | SHMEM 初始化和控制操作超时；group 的超时另由调用方设置，不保证覆盖设备内核中的全部自旋等待 |
| `debug_collectives` | bool，默认 False | 构造时各 rank 设置一致；True 时在每次原生前后向前集体检查 p dtype 和前向/反向阶段 |

构造前设置当前 NPU 和 `DEEPEP_SHMEM_ENDPOINT=tcp://<IPv4>:<unused-port>`。端口范围为 1024..65535，各 rank 必须使用相同容量、超时和 endpoint。

初始化会协调 `debug_collectives` 配置；不一致在创建 SHMEM 资源前被拒绝。默认 False 时保留本地 dtype、流及生命周期检查，不增加逐次 dtype 集体操作；调用方仍须保证同次调用的 p dtype、其他配置和调用顺序在各 rank 一致。True 的逐次检查用于定位 dtype/阶段不一致，含 Host 协调和等待，应记录到性能配置中；它不检查计算图身份，也不构成完整操作顺序协议。

`op.close()` 集体释放本模块的 SHMEM Session 和窗口。所有 rank 完成所需 backward 后，以相同顺序调用，再销毁 EP group；成功后重复 close 直接返回。Python GC 不执行集体通信清理，不能代替显式关闭。

### 3.2 前向

通过 `op(...)` 调用前向；`swiglu_limit` 和 `save_gmm1_output` 仅接受关键字传参：

```python
y, counts = op(
    x, topk_ids, topk_weights, w1, w2, max_recv_tokens,
    swiglu_limit=0.0, save_gmm1_output="bf16",
)
```

令 M 为本 rank 的输入 token 数、K 为 hidden、F 为专家中间维、T 为 TopK、E 为每 rank 的本地专家数、EP 为 EP 组大小。

| 参数 / 输出 | 类型与形状 | 含义 |
| --- | --- | --- |
| `x` | BF16 `[M,K]` | 当前源 rank 的输入 token |
| `topk_ids` | int32 `[M,T]` | 当前 EP 组内的专家编号，范围为 `[0,EP*E)`；同 token 内不重复 |
| `topk_weights` | BF16 或 FP32 `[M,T]` | 每条路由的权重，可参与求导；同次调用的所有 EP rank dtype 相同 |
| `w1` | BF16 `[E,2F,K]` | 本 rank 的 Gate/Up 参数，前 F 行为 Gate、后 F 行为 Up |
| `w2` | BF16 `[E,K,F]` | 本 rank 的 Down 参数 |
| `max_recv_tokens` | Python int，1..int32 最大值 | 本 rank 所有本地专家合计接收的路由行容量 |
| `swiglu_limit` | 可表示为 float32 的有限数值 | <=0 关闭 clipping；正值 L 对 Gate 做上界 L 裁剪，对 Up 做 [-L,L] 裁剪 |
| `save_gmm1_output` | `bf16 / none / mxfp8 / auto` | 供反向使用的 Gate/Up 保存形式 |
| `y` | BF16 `[M,K]` | 已 combine 并恢复本 rank token 顺序的输出 |
| `counts` | int32 `[E]` | 本地专家接收行数统计，不参与求导 |

Tensor 输入必须 contiguous，全部位于构造时选定的同一 NPU。也支持 `w1=[E,K,2F]`、`w2=[E,F,K]` 的另一种布局；权重梯度保持输入布局。

专家编号按 `ep_rank*E+local_expert` 解释。这里的 ep_rank 是 EP 组内 rank，适配层应据此排列本地专家权重，不能直接使用整个分布式任务的全局 rank 编号。

### 3.3 自动求导与保存模式

`loss.backward()` 自动计算 x、topk_weights、w1、w2 的梯度；对应模型参数需保留 requires_grad 及梯度连接。`dx/dw1/dw2` 为 BF16，`dtopk_weights` 跟随本次 p 的 BF16 或 FP32 类型。FP32 p 在搬运和激活计算前不降为 BF16；dp 乘积及部分和始终 FP32，仅 BF16 返回分支在末端舍入。算子本身不注册专家参数，优化器应管理模型的 router 和专家 `nn.Parameter`。

topk_ids、counts 和容量/模式等非 Tensor 参数不参与求导。当前不支持二阶求导，内核不按单个输入是否 requires_grad 切换另一条梯度路径。

| 模式 | 反向使用的 Gate/Up | 取舍 |
| --- | --- | --- |
| `bf16`，默认 | 前向保存的 BF16 值 | 首次接入可先用此路径对照数值 |
| `none` | 根据保留的 packed X 和 W1 重算 | 少保存 Gate/Up，增加重算；仍保留路由和 packed 输入 |
| `mxfp8` | 压缩值、scale，启用 clipping 时还有三个 mask 位平面 | 减少保存激活空间，引入压缩误差 |
| `auto` | 关闭 clipping 时选 MXFP8，开启时选 BF16 | 按裁剪配置选择保存形式 |

四种保存模式均可使用 BF16 或 FP32 p，p 保留实际类型，与 Gate/Up 保存格式独立，MXFP8 不压缩 p。`mxfp8` 与 `auto` 仍为实验性质：MXFP8 压缩会引入独立误差，重建的 H 与未压缩前向不逐位一致。典型模型用例使用 BF16 保存，dtype 专项另含其他保存模式及 clipping 组合；列入用例不代表已通过所有设备验收。用例和判据见[测试规范摘要](../tests/bf16/TEST_SPEC.md#1-范围与模型参数)。

MXFP8 是激活保存格式，GEMM 仍使用 BF16。BF16 舍入位置、裁剪掩码和 TopK 权重参与位置属于数值契约；外部参考需匹配这些语义。详见[前后向数值约定](mega_moe_bf16_design.md#13-数学关系与舍入边界)及[README 的保存模式](../README.md#gateup-save-modes)。

## 4. 接入已有训练框架

### 4.1 框架适配层负责什么

| 框架侧工作 | 接入要求 |
| --- | --- |
| router / TopK | 保持原有路由选择和辅助损失；将 IDs 转为 int32，p 保留 BF16/FP32 实际类型，并保证连续性及 p 的上游梯度连接 |
| 专家参数 | 适配 Gate/Up 排列和支持的权重布局；参数、优化器状态和更新由模型持有 |
| EP 通信 | 复用已有 EP group；算子已包含 token dispatch/combine |
| DP / TP / SP | 外层安排参数梯度归约、张量或序列切分，使本地 Tensor 符合接口契约；算子不自动推导这些并行策略 |
| PP / checkpoint | 各 EP rank 的前向、重放与反向调用顺序一致，调用使用构造流 |
| 生命周期 | 模型/训练任务持有共享 op，在所有所需反向结束后统一关闭 |

例如 router 输出为 FP32 权重、int64 IDs 时，适配层可保留 FP32 p，仅转换 IDs；实际路由合法性仍由框架保证：

```python
topk_ids = router_ids.to(torch.int32).contiguous()
topk_weights = router_weights.contiguous()  # router_weights 已为 torch.float32
```

若框架明确选择 BF16 p，可显式使用 `router_weights.to(torch.bfloat16).contiguous()`；这会先量化 p，后续将梯度升为 FP32 不能恢复已丢失的有效位。dtype 转换需保留 autograd 连接，不能通过 detach 得到算子输入后还期待 router 接收梯度。权重的布局转换也需保持参数与计算图的连接；所需 backward 完成前不原地更新权重。

### 4.2 多层和多个 microbatch

当前同一进程同一时刻只允许一个存活的 SHMEM Session。多层 MoE 应共享同一个 op、同一 EP 组，在构造流上串行调用，逐次传入各层参数和容量。模块绑定的 EP 组、设备和构造流固定。

每次前向独立保留路由、packed 输入和可选 Gate/Up，允许按一致顺序排队多个前向，再执行对应反向。不同调用可以切换 BF16/FP32 p，各计算图保留自己的 p 类型和保存模式；同次调用各 rank 仍需一致。尚未结束的计算图越多，保存状态占用越大；通信窗口仍是同一个。

当前公开接口不接受外部 Runtime 或 SHMEM 裸地址，也不支持同时持有另一个 DeepEP buffer、推理 Session 或其他已初始化的 SHMEM Session。不同层需要不同 EP 组的接入方式不能直接由这个单 Session 接口覆盖。

跨流并发调用不受支持。checkpoint 重放、梯度累积、PP microbatch、图编译和图捕获需要针对实际框架分别验证；已有算子用例通过不代表这些整网场景已经验收。

## 5. SHMEM 窗口怎样使用

每个 rank 的 op 创建一块 NPU 通信窗口。各 rank 使用相同布局和偏移，实际基地址可以不同；Runtime/Session 负责申请、初始化、建立 peer 地址映射和最终释放。应用只传容量，无需自行写窗口、传裸地址或调用 SHMEM free。

当前普通 BF16 窗口布局如下：

```text
本 rank 的窗口
  ├─ 72 KiB 同步头：peer 到达信号与同步代数
  ├─ 可选 compact 计数交换区
  ├─ dense token 计数表
  ├─ 发布区 ptrA0：前向发布排序后的 X，反向发布展开后的 dY
  ├─ 路由权重区：与 X 同序的 p，每元素按实际类型占 2 或 4 字节
  └─ 返回区 ptrD：前向接收专家输出，反向先接收 dX、再复用为 dp
```

以 rank 0 的 token 路由到 rank 1 的专家为例：

1. 前向：rank 0 发布 X/p，rank 1 从 peer 窗口拉入本地 packed Tensor，计算后把专家输出写回 rank 0；rank 0 恢复 token 顺序并合并为 y。
2. 反向：rank 0 发布 dY，rank 1 拉入为 dO，计算并回传 dX/dp；rank 0 得到 dx/dtopk。权重梯度保留在专家 rank。
3. 内核通过就绪信号、流水排空和跨 rank 屏障保护读写完成，再复用窗口中的区域。

通信窗口与计算图保存状态具有不同生命周期：

| 内存 | 生命周期 | 内容 |
| --- | --- | --- |
| SHMEM 窗口 | 构造至集体 close，多次调用复用 | 发布/返回数据、计数、同步信号 |
| 每次前向的保存状态 | 对应计算图需要它的期间 | 路由映射、计数快照、packed X/p、可选 Gate/Up |
| 前后向临时工作区 | 当前异步调用期间 | Hp、梯度中间量、部分和、控制计数 |

因此，local_mem_size 只覆盖通信窗口；模型、优化器、保存激活、输出和临时工作区还需要额外显存。

## 6. 容量怎么规划

### 6.1 先区分两个容量

| 参数 | 单位 | 决定什么 | 不足时 |
| --- | --- | --- | --- |
| `local_mem_size` | 每 rank 字节数 | 固定的通信窗口容量 | 实际调用时 Host 检查完整布局并报错，不自动扩容 |
| `max_recv_tokens` | 路由行数 | 当前 rank 所有本地专家合计接收容量，影响 packed 状态及计算工作区 | 设备可能截断，无法得到完整结果；Host 不根据实际路由提前保证充足 |

增加窗口不会补救 max_recv_tokens 过小；增加接收行容量也不代表窗口已足够。构造时还未传入 M/T/K，成功创建 op 不等于任意后续 shape 都能放入。

### 6.2 max_recv_tokens

在每 token 专家 ID 唯一、各源 rank 都有 M 个 token 时，可保守设置：

```text
max_recv_tokens >= EP * M * min(E, T)
```

这约束最繁忙目标 rank 的总接收行数。不要仅用平均接收行数替代；若使用更小容量，框架应保证实际路由需求。counts 是调用后的统计，不是容量不足前的协商接口。

双卡示例 `EP=2、M=129、E=2、T=4` 的保守值为 516。这里不是每专家各分配 516 行，而是本 rank 所有本地专家共享 516 行容量。

### 6.3 local_mem_size

当前普通 BF16 训练的发布/返回区按源 rank 的 `M*T` 条路由规划，专家接收端的 packed 扩张主要放在独立 Torch Tensor 中。令 b 为 p 的元素字节数：BF16 取 2，FP32 取 4。

记 `A_a(v)` 为向上对齐到 a 的整数倍。窗口布局需要的字节数为：

```text
Q = 0                                                   # dense 计数交换
    或 A_512(64*EP) + A_512(72*4)                         # compact 附加区
C = 72*1024 + Q + 4*EP*A_128(EP*E)                       # 同步与计数区
B = C + 2*M*T*K + A_512(b*M*T) + 2*M*T*K                 # 发布、p、返回区
local_mem_size >= A_(2*1024*1024)(B)
```

以下是按本版本布局规划容量的示例辅助函数，**不是库导出的 API**。它保守计入可选 compact 区，覆盖 dense/compact 两种布局；输入需满足第 7 节约束。

```python
def plan_bf16_window_bytes(m, hidden, topk, ep_size, experts_per_rank,
                           topk_weight_bytes=2):
    if topk_weight_bytes not in (2, 4):
        raise ValueError("topk_weight_bytes must be 2 (BF16) or 4 (FP32)")

    def align_up(value, alignment):
        return (value + alignment - 1) // alignment * alignment

    optional_counts = align_up(64 * ep_size, 512) + align_up(72 * 4, 512)
    control = (
        72 * 1024
        + optional_counts
        + 4 * ep_size * align_up(ep_size * experts_per_rank, 128)
    )
    payload = 4 * m * topk * hidden + align_up(topk_weight_bytes * m * topk, 512)
    return align_up(control + payload, 2 * 1024 * 1024)
```

函数返回字节数，省略最后参数按 BF16 p 的 2 字节规划，FP32 p 传 `topk_weight_bytes=4`。混用两种 dtype 的实例按 4 字节及所有层允许 shape 的最大规划值选择固定窗口，各 rank 使用同一容量。返回区仍按 BF16 专家输出/dX 的矩阵容量保留，P6 在旧读者退出后复用它存放 2/4 字节 dp，不另增一个 FP32 返回面。实际检查以当前构建的 Host 通信布局为准；布局公式及此函数不包含模型、计算图状态、scratch 和运行时额外显存。评估总显存时应一起计入。

保存模式控制的主要额外显存包括：BF16 Gate/Up 每个平面约 `2*R*F` 字节；MXFP8 每个数据面约 `R*F`，还需 scale 和可选 mask；none 仍有 packed X/p、路由元数据和反向重算空间。packed p 按 `b*R` 字节而非固定 BF16 规划。这里 R 为 max_recv_tokens，所有区域还存在对齐及控制开销。

## 7. 限制与验收边界

### 7.1 接口和资源要求

| 条件 | 当前要求 |
| --- | --- |
| EP 大小 | 训练 Runtime 接受 2..1024；StoreGroup 自身接受的 size 不超过 128 |
| 本地 / 总专家 | `1<=E<=32`，`EP*E<=2048` |
| token 与维度 | `1<=M<=16384`，K/F 为正数且均是 256 的倍数；仍受实际 UB 预算检查 |
| TopK | `1<=T<=EP*E`；IDs 合法且同 token 内唯一 |
| Tensor | 连续、指定 dtype、同一构造 NPU；不能假定所有错误设备组合都已被 Host 全量拦截 |
| rank 间配置 | 构造时 debug_collectives 一致；同次调用的 p dtype、M/K/F/T/E、接收容量、保存模式、clipping、构建策略和操作顺序一致；实际输入数据及路由内容可以不同 |
| 流 / Session | 构造流串行调用、每进程一个存活 Session；不支持并发换流或外部窗口注入 |
| 求导 | x/p/w1/w2 一阶梯度；不支持二阶，IDs/counts 不可微 |
| 关闭 | 完成所需 backward 或丢弃不用的图后集体 close；retain_graph 后仍需反向时保持 op 开启 |

部分条件由入口硬检查，路由数据合法性、实际接收容量、rank 调用顺序等仍由调用方保证。默认关闭逐次集体诊断时，跨 rank p dtype 一致性也由框架保证；某 rank 本地校验失败或提前退出时，框架应终止整个任务，不能让其他 rank 继续进入设备等待。代码接受的 EP 上限不等于已经通过同等规模设备验证。

### 7.2 部署前另行验证的场景

当前设备验收边界为单节点 Ascend950。实际框架使用的 HCCL ProcessGroup、EP 子组、更大 EP、跨节点、TP/SP/PP 组合、checkpoint、图模式和整网训练收敛，应分别验证。传入接口能够接受的 group，不构成这些部署路径已通过验收的证明。

建议首次接入按 BF16 保存模式，对照前向输出、counts 和四类梯度；再覆盖目标保存模式、clipping、偏斜路由、空专家及连续多轮调用，最后用实际模型短训验证 loss/梯度行为。设备精度与性能的命令、口径和报告结构见[测试入口](../tests/bf16/README.md)，用例与判据见[测试规范](../tests/bf16/TEST_SPEC.md)。

### 7.3 实现上限与调度阈值

EP、专家数和 token 的硬上限来自当前 Host 检查及设备布局；K/F 的对齐要求来自现有 GEMM tile 和搬运路径。这些上限与设备功能验收范围分别判断，入口检查见 [training.cpp](../csrc/bf16/training.cpp)，平台与 UB 预算见 [tiling.cpp](../csrc/bf16/tiling.cpp)。

compact 计数交换、行进度调度的阈值影响所选执行路径：

| 条件 | 当前行为 |
| --- | --- |
| compact 计数交换 | 支持本地专家数不超过 8；默认仅 EP 至少 512 且满足布局条件时选择，其余使用 dense |
| 前向行就绪 | 满足形状、实际接收行数和元数据容量条件时使用 Ready384；否则等待整专家就绪 |
| 反向行就绪 | 默认在保存模式为 BF16/MXFP8、本地专家数不超过 7、专家有效行数大于 128 时使用；其他条件按策略回退 |
| 强制整专家构建 | `DEEPEP_BF16_FORCE_EXPERT_READY=ON` 同时关闭前后向行进度细分，优先于反向运行时覆盖 |

更改构建策略需要完整重建、重新安装配套的 Host/Device wheel，各 EP rank 保持一致。具体选择与同步关系见[Dispatch 设计](mega_moe_bf16_design.md#37-前后向-dispatch-就绪与同步协议)。

## 8. 常见接入问题

| 现象 | 优先检查 |
| --- | --- |
| Runtime 或训练扩展不可用 | 是否安装启用训练的完整 wheel、加载了匹配 CANN 环境；包内 SHMEM 是否缺失、哈希不符或与已加载的外部 SDK 冲突；当前目录和 PYTHONPATH 是否遮蔽安装包 |
| 已存在 SHMEM runtime | 是否给多层各创建了 op，或其他 DeepEP/SHMEM Session 尚未关闭 |
| BF16 / contiguous 检查报错 | router IDs/p 的类型、权重重排后的连续性，以及全部 Tensor 所在设备 |
| local_mem_size too small | 按当前调用或全模型最大 shape 重新规划窗口；增大后需重新创建 op |
| 调用无法继续 | 检查是否有 rank 提前退出、endpoint/端口冲突，以及各 rank 的前后向/重放顺序和配置是否一致 |
| construction stream / creating NPU 报错 | 构造后是否切换设备或从其他流调用了前后向 |
| 关闭后反向报错 | 模块是否在计算图仍需执行 backward 时已 close |

设备或集体操作失败后，应保留日志并重启分布式任务；不能假设原 Session 仍可安全复用。构造 timeout 也不是设备侧全部等待循环的统一截止时间。

实现层的调用关系、状态所有权、前后向计算、内存复用和同步协议统一见[前后向设计](mega_moe_bf16_design.md)。源码导航另见[交互式导览](visual/README.md)。
