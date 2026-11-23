<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# MegaMoE

MegaMoE 在 Ascend950 上融合专家路由、两次分组矩阵乘、SwiGLU 和结果归并。三条路径共用 `deep_ep_experimental.megamoe` 包和 `SymmBuffer`，由 `mma_type` 选择计算类型。 构建和多机批量测试脚本位于 [scripts/](scripts/)。

| 路径 | `mma_type` | 公开接口 | 共享专家 |
| --- | --- | --- | --- |
| W4A8 推理 | `"fp8xfp4"` | `fp8_fp4_mega_moe` | 每卡 0～8 个 |
| W8A8 推理 | `"fp8xfp8"` | `fp8_fp4_mega_moe` | 不支持 |
| BF16 推理与训练 | `"bf16xbf16"` | `bf16_mega_moe` | 不支持 |

## 1. 输入、输出与公开接口

下文用 `B` 表示每卡输入 token 数，`H` 表示隐藏维度，`I` 表示 SwiGLU 输出宽度（BF16 测试中的 `--ffn`），`K` 表示 TopK。`E` 为全局路由专家数，`L = E / EP` 为每卡路由专家数，`S` 为每卡共享专家数。两个接口均原地写入调用方分配的 BF16 输出 `y[B,H]`，返回 `None`。

### 量化推理接口（W4A8、W8A8）

```python
# 公开调用签名；原位写入 y，返回 None。
def fp8_fp4_mega_moe(
    y, l1_weights, l2_weights, sym_buffer,
    shared_l1_weights=None, shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    recipe=(1, 1, 32), activation="swiglu",
    activation_clamp=None, fast_math=True,
    *, x=None, timer=None,
) -> None: ...
```

| 参数/输入 | 类型、形状与用途 |
| --- | --- |
| `y` | 调用方预分配的 BF16 `[B,H]` 输出，不能与输入重叠 |
| `sym_buffer.x / x_sf` | 默认输入：FP8 E4M3 `[B,H]` + E8M0 `[B,H/32]`，调用前填入 buffer 的前 B 行 |
| `x` | 可选 BF16 `[B,H]`，覆盖 buffer 中的 FP8 输入；内部量化为 FP8，**不是 BF16 GEMM** |
| `sym_buffer.topk_idx` | int32 `[B,K]`，全局专家 ID，同 token 不重复且不能为 -1 |
| `sym_buffer.topk_weights` | FP32 `[B,K]` 路由权重；BF16/FP8 输入均从 buffer 读取路由 |
| `l1_weights / l2_weights` | 两对 `(weight, scale)`，W4A8 使用 FP4 E2M1，W8A8 使用 FP8 E4M3，两者的 scale 均为 E8M0 |
| `shared_l1_weights / shared_l2_weights` | 仅 W4A8 支持。共享专家的两对权重，必须同时传入，数量与 buffer 的 S 一致；默认不使用 |
| `cumulative_local_expert_recv_stats` | 可选 int32 `[L]`，在原值上累加专家接收 token 数 |
| `timer` | 可选，仅 W4A8 支持。单独诊断时传入新分配的 `buffer.allocate_timer()` |

`recipe/activation/activation_clamp/fast_math` 当前仅支持签名中的默认值。所有张量在 buffer 所在 NPU 上。
量化输入 buffer 是普通 NPU 存储。传入已有的 FP8 数据和 scale 时，`AIV_QUANT` 负责打包到通信区，不会再次量化。

在测试或推理前调用 `w1, w2 = transform_weights_for_mega_moe(raw_w1, raw_w2)`，从原始 ND 权重准备输入。这一步不计入算子耗时。两种量化格式的原始布局如下。

| 权重 | W4A8，uint8，每字节打包两个 FP4 值 | W8A8，FP8 E4M3 | E8M0 scale |
| --- | --- | --- | --- |
| GMM1 | `[L,2I,H/2]` | `[L,2I,H]` | `[L,2I,H/64,2]` |
| GMM2 | `[L,H,I/2]` | `[L,H,I]` | `[L,H,I/64,2]` |

原始 GMM1 权重沿输出维先放 Gate，再放 Up。W4A8 转换为 Ascend NZ 格式，路由 GMM1 同时保留两套权重和 scale：实际 B≤256 使用 `[gate128, up128]` 融合布局，B>256 使用 `[gate | up]` 配对布局。GMM2 只保留一份。W8A8 的 FP8 权重保持原布局。

请完整保留返回的 `w1`，不要解包重组、原地修改或单独搬运其中的张量。更新权重或更换设备时，从原始 ND 重新转换；已经过 CUDA 专用重排的权重不能直接传入。W4A8 共享专家将首维 L 换成 S，并传入 `shared=True`，共享 GMM1 使用 planar 布局。

下面以 W4A8 为例，`SymmBuffer` 中的模型参数与测试 JSON 含义相同。各 rank 在设置本机 NPU 后建立通信组。

```python
from deep_ep_experimental.megamoe import (
    StoreGroup, SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe,
)

# torchrun 启动后先 torch.npu.set_device(local_rank)。
group = StoreGroup.from_env()  # 主机协调用 TCPStore，算子通信用 SHMEM + MTE
buffer = SymmBuffer(
    group, num_experts=384, num_max_tokens_per_rank=4096, num_topk=6,
    hidden=7168, intermediate_hidden=3072,
    num_shared_experts=0, mma_type="fp8xfp4", activation="swiglu",
    ranks_per_node=8, timeout=120,
)
```

填入输入和路由后，依次调用 `buffer.validate_inputs(B)`、`buffer.prepare(B)` 和 `fp8_fp4_mega_moe(...)`。直接传 BF16 激活时，预检和算子调用都传 `x=x_bf16`。各 rank 按相同顺序、相同 B 调用；切换 B 时在计时前集体调用 `prepare(B)`。它会核对各 rank 的 B，确保使用同一握手策略。所有 rank 使用结束后同序调用 `buffer.destroy()`、`group.close()`。完整调用及异常处理见 [example.py](python/deep_ep_experimental/megamoe/example.py)。

量化推理支持 EP=2～128，E 须能被 EP 整除，每卡最多 128 个路由专家。B=1～65536，H=1024～8192 且为 64 的倍数，I=256～4096 且为 128 的倍数，1≤K≤min(32,E)。当前仅支持推理，各 rank 的 B 必须相同，暂不支持图捕获。

量化接口签名参照 DeepGEMM `78b6900`。CPU 参考与算子均在 GMM2 后施加路由权重；将乘权移到中间量化前会改变舍入位置，两种结果可能不同。兼容入口 `mega_moe(x, ids, weights, w1, w2, buffer, ...)` 返回 `(y, per_call_counts)`。

#### 共享专家策略

W4A8 路由专家沿用 Prefill 的 Wave 调度。当 S>0 时，共享专家按本次实际 B 选择小 BS 或 Prefill 策略，容量 `num_max_tokens_per_rank` 不参与判断。

| 项目 | B<=256 | B>256 |
| --- | --- | --- |
| 输入 | 双 AIV Quant → AIV 同步 → Route → reset → 共享输入准备 | 双 AIV Quant → reset → 共享输入准备 |
| 本卡全核屏障后 | AIC/AIV0 共享 GMM1，AIV1 count/prefix/Dispatch | AIC/AIV0 共享 GMM1，AIV1 Route/子组及跨卡同步/count/Dispatch |
| 共享 GMM1 | 完整 gate/up 输出 N192 独立任务 | N256 gate/up 配对任务 |
| 共享 GMM2 | N224 | N256 |

两条路径均使用 CATLASS 后端、M256 和 N256 共享 SwiGLU。路由专家在 B≤256 使用逻辑 N128/物理 N256 融合；B>256 使用 gate256/up256 配对任务。
Token Combine 按实际单卡 B 选择：B≤256 使用六槽，B>256 使用两槽，与是否开启共享专家无关。
小 BS 使用逐槽 epoch/count 就绪协议，每个物理 AIV 本地推进一次 epoch，
输出握手再推进一次；大 BS 使用输入/输出跨卡握手。共享 GMM1 后无需全核屏障。
共享 SwiGLU 在 AIV1 上优先于 Combine，共享 GMM2 按本核路由完成及共享激活就绪推进。
S=0 的 W4A8 小 BS 也使用上述量化完成/count 就绪协议；W8A8 使用输入/输出跨卡握手。

W4A8 路由专家（S=0）的接收容量超过 8 个 Wave 时，Dispatch、量化激活及 GMM 输出使用 8 个循环槽。每个槽在该 Wave 的所有 Combine 发送完成后归还。专家计数、路由映射和通信窗口按原容量分配。量化路径的 tiling ABI 为 224 字节。

W4A8 在 B≤256 时使用较小 wave、rail 路由中转和打包 count。
rail 按每8个 rank 一组，EP≤8 或完整8卡分组启用，其余 EP 回退直发路由。
B>256 保留 Prefill 输入协议，所有路径保留输出跨卡同步。

量化与 BF16 前向共用 Wave 预算公式，各路径按自己的 GMM tile 数确定分组。CATLASS GMM 的布局和同步机制见 [GMM 说明](csrc/kernels/mega_moe/catlass_gmm/README.md)。

<a name="bf16前向与训练"></a>

### BF16 接口

推理和训练调用同一个 `bf16_mega_moe`。`SymmBuffer.need_grad` 决定路由权重的乘法位置和反向语义。BF16 只计算路由专家，`num_shared_experts` 取 0，两个共享权重参数均留空。

```python
# 公开调用签名；原位写入 y，返回 None。
def bf16_mega_moe(
    y, l1_weights, l2_weights, sym_buffer,
    shared_l1_weights=None, shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    activation="swiglu", activation_clamp=None, fast_math=True,
) -> None: ...
```

| 参数 | 类型、形状与用途 |
| --- | --- |
| `sym_buffer.x` / `y` | 连续 BF16 `[B,H]`。`y` 使用独立存储，调用前 `requires_grad=False`。传给 `y.backward(grad_y)` 的 `grad_y` 也为 BF16 `[B,H]` |
| `sym_buffer.topk_idx` | 连续 int32 `[B,K]`，存放全局专家 ID |
| `sym_buffer.topk_weights` | 连续 BF16 或 FP32 `[B,K]`。每个 TopK slot 对应一个权重 `p`，返回的 `dP` 与 `p` 同类型 |
| `l1_weights` / `l2_weights` | 连续 BF16 张量，成对使用 `[L,2I,H]` / `[L,H,I]` 或 `[L,H,2I]` / `[L,I,H]` |
| `cumulative_local_expert_recv_stats` | 可选的 int32 `[L]`，在原值上累加各本地专家接收的行数 |
| `activation_clamp` | `None` 或 0 表示不截断；正值对 Gate 作上界截断、对 Up 作双侧截断。激活使用 `"swiglu"`，`fast_math=True` |

| `need_grad` | 前向乘权位置 | 反向 |
| --- | --- | --- |
| `False`（默认，推理） | SwiGLU 输出以 BF16 进入 GMM2，源 rank 按 TopK slot 在 FP32 中乘 `p` 并累加 | 仅前向；开启 autograd 且任一输入要求梯度时会报错 |
| `True`（训练） | `Hp = BF16(SiLU_FP32(Gate) × Up × FP32(p))`，随后进入 GMM2 | 通过 `y.backward(grad_y)` 计算所需的 `dX`、`dP`、`dW1`、`dW2` |

各 rank 先设置当前 NPU、使用相同的 SHMEM endpoint，再传入已初始化的 PyTorch EP 通信组，或使用 `StoreGroup.from_env(bf16=True)`。下面以 EP8、每卡 4 个专家为例。传入原始 `x`、`topk_weights`、`w1`、`w2` 张量，每个张量可独立设置 `requires_grad`。示例在前向前对需要梯度的张量设置 `requires_grad=True`。

```python
import torch
from deep_ep_experimental.megamoe import SymmBuffer, bf16_mega_moe

buffer = SymmBuffer(
    ep_group, num_experts=32, num_max_tokens_per_rank=4096,
    num_topk=6, hidden=7168, intermediate_hidden=3072,
    mma_type="bf16xbf16", need_grad=True,
    gmm1_save_type="bf16", max_recv_tokens=27648,
)
buffer.x = x
buffer.topk_idx = topk_idx
buffer.topk_weights = topk_weights
buffer.validate_inputs(x.shape[0])  # 预检，在计时区间外执行
buffer.prepare(x.shape[0])         # 可选：提前准备临时 workspace

y = torch.empty_like(x, requires_grad=False)
bf16_mega_moe(y, w1, w2, buffer)
y.backward(grad_y)
buffer.destroy()                  # 所有反向完成后，所有 rank 同序释放
```

推理时构造 `need_grad=False` 的 buffer，填入相同的输入和权重，调用 `bf16_mega_moe(y, w1, w2, buffer)` 即可。训练 buffer 在 `torch.no_grad()` 下仍按训练顺序乘权，同时跳过构图和跨调用保存。

BF16 支持 EP=2～1024，E 须能被 EP 整除，每卡 1～32 个路由专家，E≤2048。B=1～16384，1≤K≤min(32,E)，H 和 I 均为 256 的倍数，具体形状还须满足 Dispatch 和激活的 UB 容量。`max_recv_tokens` 按单个 rank 接收的路由行数配置。训练必须显式传入；推理和 W4A8 使用相同的默认上限 `num_max_tokens_per_rank × EP × min(K,L)`。每个 token 的 TopK 专家互不重复，最多有 `min(K,L)` 条路由落到同一 rank，因此该值覆盖路由集中到热卡的情况。已知峰值时可指定较紧的容量。

#### 训练保存状态

`gmm1_save_type` 决定 Gate/Up 的来源，默认保存 BF16，也可选 `mxfp8` 量化保存或 `none` 在反向重算，见[反向调度](#bf16-反向调度)。dP 在 FP32 中计算并归约，送入 GMM 的 Hp 和 dZ 使用 BF16。SHMEM 窗口由运行时持有，Host Plan 借用该窗口，按输入形状、类型和保存配置复用 tiling 及前向临时 workspace；权重布局在每次调用时确定。Host 和设备使用同一套缓冲区偏移公式。

每张源卡在 SHMEM 中按 token 存一份前向 X，反向在同一区域存一份 dY。专家卡按路由索引取行。
路由索引、前乘所需的 p 和 Combine 回传槽按每卡源 token 数乘以 TopK 分配。 每次反向读取对应前向保存的索引，因此多个 microbatch 可以连续前向后逆序反向。

训练时 `SymmBuffer(..., save_dispatched_x=True)` 默认将专家端 X 和 TopK weight 保存在本次前向的 autograd context 中。行序依次为本地专家、源 EP rank、源内 token/TopK slot。BF16、MXFP8 的 Gate/Up 也按此顺序保存，MXFP8 的 scale 和 clipping 掩码与对应行对齐。

框架自行重建时设置 `save_dispatched_x=False`，并在每次前向前设置 `sym_buffer.dispatch_rebuilder` 为该 microbatch 的无参函数。反向调用该函数取得连续的设备端 UInt8 缓冲。缓冲依次存放 `[max_recv_tokens, H]` BF16 X 和 `[max_recv_tokens]` 原 dtype 的 p，两段分别按 512 字节对齐。异步重建时，回调须让调用流等待生产端事件。每个前向独立记录回调，支持多前向后逆序反向。直接调用原生 Plan 的框架可以在 `plan.forward(..., capture_timer=False, save_dispatched_x=False)` 后将重建张量传入 `backward`。

同一 EP 组的调用顺序、配置和 p 的 dtype 须一致。跨 rank 的 dtype 和阶段诊断默认关闭，设置 `debug_collectives=True` 可开启。

#### BF16 前向缓冲

前向按 Wave 从多个源 rank 取行，供 GMM1 和 SwiGLU 连续计算。推理固定使用模式 0，训练默认也使用模式 0；训练构建时设置 `DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE=1` 可切到模式 1。接收容量超过 8 个 Wave 时，临时 X 和 GMM2 输出共用 8 个循环槽，可借用的 Y 空间计入槽容量。每个槽容纳一个 Wave 的全部专家切片。

量化与 BF16 共用专家内的路由索引、peer 映射和行区间计算。量化路径搬运 token 和 scale，BF16 训练搬运 X 和配对的路由权重 p；源 rank 按原 TopK slot 顺序用 FP32 累加各路结果。两者的 SwiGLU 共用数学公式，量化路径使用默认除法，BF16 使用高精度除法。

| 调度 | Hp 空间 | 同一槽何时可覆盖 |
| --- | --- | --- |
| 模式 0，GMM1/GMM2 交错 | 与临时 X 等行数，独立存放 | 上一轮使用该槽的 Combine 全部发完，Dispatch 才能写下一轮 X。 |
| 模式 1，GMM1/GMM2 分阶段 | 按完整接收容量分配 | 上一轮整行 Hp 就绪后可以覆盖 X。全部 SwiGLU tile 完成后 GMM2 接管循环槽，每次回绕等待上一轮 Combine 发完。 |

#### BF16 反向调度

| `gmm1_save_type` | Gate/Up 来源与计算方式 |
| --- | --- |
| `bf16` | 读取保存的 BF16 Gate/Up。整行路径一次读取 dHp、Gate、Up，计算 dP、Hp 和 dZ。I 为 64 的倍数且 I ≤ 16384 时使用反向 Wave。 |
| `mxfp8` | 按行读取每 32 列的量化值和 scale，以 FP32 解码后舍入到 BF16。启用 clipping 时读取同行的掩码。支持 32 列分组的路径使用反向 Wave。 |
| `none` | 从前向保存或框架重建的 X 中取每个专家的连续行，按整专家重算 GMM1，随后计算梯度。 |

反向 Wave 将每个专家的 dO 按规范行序划为 128 行块，尾块取实际行数。每个专家只选一次块序，Dispatch、dHp、SwiGLUGrad 共用。令 `Cmax` 为接收容量截断后单个源 rank 发给该专家的最大行数。专家行数一波可覆盖、最长源段不足一波，或所有块只占一组时，从本 rank 的起始块依次访问并回绕。其余情况每 `ceil(Cmax / 128)` 块一组，先访问各组第一个块，再访问各组第二个块。访问顺序只影响搬运和计算的先后，各块始终写回自己的规范行地址。

AIV 写完本块 dO 后发布 ready。AIC 等本块就绪再算 dHp；SwiGLUGrad 等对应 dHp 列就绪。AIC 等 dO 时可为已有完整 dO 和 Hp 的前序专家算一个 dW2 输出 tile，算完重查 dO。后续 dW2 跳过已完成的 tile。dW1、dW2 的每个输出 tile 始终沿整专家的行做归约。各 rank 设置 `MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS=0` 可切换到等整专家 dO 就绪，dW2 全部在原阶段计算。

本行 dP 算完后 Hp 才可覆盖 dHp；dHp 和 dW2 读完该专家的 dO 后 dX 才可覆盖 dO。dP 按全 rank 的路由行数和全部 AIV 划分归约任务。各专家按源 rank 前缀将结果写入独立 SHMEM 回传区，每条 route 的 dP 与输入 p 同类型。跨卡发送核回读确认后进入屏障，源卡再恢复原 token 和 TopK slot 顺序。

## 2. 编译安装

W4A8、W8A8 和 BF16 使用同一个 wheel，包含一个 MegaMoE Host 扩展和设备库 `libmegamoe_kernel.so`，沿用根仓选定的 SHMEM 运行库。先准备 Linux、Ascend950、CANN 9.2.0、Python ≥3.11 及匹配的 Torch/torch_npu 环境，再在该环境中运行：

```bash
git clone https://github.com/lueelu/DeepEP.git
cd DeepEP
source /path/to/cann/set_env.sh
bash experiments/megamoe/scripts/build.sh
python -m pip show ascend-deepep
```

`build.sh` 默认开启 `DEEPEP_BUILD_MEGAMOE` 和 `DEEPEP_MEGAMOE_TIMER`，自动初始化锁定版本的 SHMEM/CATLASS 子模块，校验已有 SHMEM 或按需构建，再构建并安装 wheel。`PYTHON=/path/to/python` 可指定解释器，`DEEPEP_MEGAMOE_TIMER=OFF` 可关闭设备打点。已激活 Conda 时，脚本优先使用该环境的运行库。

已有 SDK 可通过 `SHMEM_ROOT`、`CATLASS_ROOT` 指定。仓内子模块须与[依赖锁文件](../../dependencies.lock.json)一致，且工作区干净。需要重建仓内 SHMEM 时，停止使用该 SDK 的作业后传 `--rebuild-shmem`。wheel 位于 `dist/build-*/`，运行前按脚本输出设置 SHMEM 的 `LD_LIBRARY_PATH`。

直接从根目录构建时，基础包的 `pip wheel` 默认关闭 MegaMoE。按[构建文档](../../docs/build.md)准备好 SHMEM、CATLASS 和 Python 构建依赖后，用以下命令启用量化推理和 BF16 前反向。安装时选择本次生成的 wheel（下例假设 `dist/` 中只有这一个 wheel）。

```bash
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_BUILD_MEGAMOE=ON
python -m build --wheel --no-isolation
python -m pip install --force-reinstall --no-deps dist/*.whl
```

## 3. 多机测试

各节点使用同一 wheel，并加载相同的 CANN 和 SHMEM 环境。W4A8 的批量脚本从控制机启动；BF16 的推理和训练使用同一测试脚本，在每个节点运行 `torchrun`。W8A8 可通过量化接口调用，当前批量脚本固定生成 FP4 权重，没有权重 dtype 开关。`--input bf16` 只切换 W4A8 的激活输入。

### W4A8 批量测试

控制机通过 SSH 访问各节点，每个节点需要 8 张空闲 NPU。各节点使用相同绝对路径的共享 checkout 和匹配的 Conda/CANN 环境。在仓库外创建机器配置，将示例地址替换为分配到的节点，通过 `--hosts-file` 传入。

```json
{
  "hosts": ["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.4"],
  "master": "192.0.2.1"
}
```

EP32/64/128 分别需要4/8/16个不同节点。`master` 可省略，默认使用列表首节点。
主节点排在 rank0～7，其余节点保持列表顺序。仓库不提供实际机器池。
在一台控制机设置远端环境后启动：

```bash
export CONDA_BASE=/path/to/conda
export CONDA_ENV=your_env
export ASCEND_ENV=/path/to/cann/set_env.sh
export HOSTS_FILE=/path/outside/checkout/hosts.json
# REMOTE_USER 默认为控制机当前用户，可按需覆盖；默认使用 SSH 密钥。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 32 --plan
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 32
```

入口编译一次、逐节点安装同一 wheel，再运行测试及解析 timer；不会更新 Git 分支。
`CONDA_BASE`、`CONDA_ENV`、`ASCEND_ENV` 必须显式设置，路径对应每台远端机器。
密码登录可设置 `SSH_AUTH=password` 并安装 sshpass，密码通过交互输入或 `SSHPASS` 提供。
实际机器配置、凭据和输出报告不要加入版本控制。

#### 常用参数

| 参数 | 默认 / 作用 |
| --- | --- |
| `--mode perf\|accuracy` | 默认 perf 执行性能测试；accuracy 检查全量输出和专家计数 |
| `--input fp8\|bf16` | 默认 fp8；只改变输入类型，权重仍为 FP4 |
| `--config NAME或路径` | 默认 `deepseek_v4_pro`；可传模型 JSON 路径 |
| `--bs-values 96,4096` | 覆盖 JSON 中 BS 列表；每个 BS 单独启动 |
| `--routing balanced\|random\|skewed` | 默认 balanced；random/skewed 用于精度压力测试 |
| `--no-timer` | 关闭打点和解析；默认自动保存、解析主节点 rank0～7 |
| `--no-parse-timer` | 跳过自动流水解析，保留原始 timer 和耗时拆分 |
| `--ep-size 32\|64\|128` | 默认64；分别为4/8/16台八卡机器 |
| `--hosts-file 路径 / --master-ip IP` | 必填机器列表 JSON / 覆盖列表内的主节点 |
| `--master-port / --shmem-port` | 默认29654 / 26733，必须不同且未占用 |
| `--skip-build / --wheel 路径` | 复用已安装身份匹配的唯一 wheel / 指定共享 wheel；不再编译，但仍向各机安装 |
| `--prepare-only / --plan` | 只部署并检查 / 只预览 |
| `--timeout 秒` | 每阶段、每 BS 默认7200秒 |

`balanced` 路由按全局 `[token, top-k]` 顺序轮转目标 rank，再轮转卡内 expert。
精度使用相同路由生成 CPU Golden。修改算子或包内 Python 后，需要重新构建安装，
首次运行时省略 `--skip-build`；修改文档或启动脚本无需重编算子。

以下命令使用8节点机器文件，执行 EP64：

```bash
# 全量精度测试，关闭设备打点。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --mode accuracy --bs-values 96,4096 --no-timer

# W4A8 使用 BF16 激活输入，测性能并解析流水。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --input bf16 --bs-values 96

# EP128：将 HOSTS_FILE 改为包含16个节点的配置。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 128
```

#### 模型和共享专家配置

默认模型：H=7168、I=3072（gate+up 总宽6144）、全局384路由专家、TopK=6、S=0。
EP32/64/128 分别每卡12/6/3个路由专家；共享专家不计入384，`num_shared_experts` 表示**每卡**数量。

已提供两个配置，只有 `num_shared_experts` 不同，其余模型、BS、seed 和轮次一致：

- [`deepseek_v4_pro`](python/deep_ep_experimental/megamoe/configs/deepseek_v4_pro.json)：无共享专家，默认配置。
- [`deepseek_v4_pro_shared1`](python/deep_ep_experimental/megamoe/configs/deepseek_v4_pro_shared1.json)：每卡额外1个共享专家，不减少原有路由专家。

```bash
# 每卡1个共享专家，全量精度检查 BS96/4096。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --config deepseek_v4_pro_shared1 --mode accuracy --no-timer

# 使用相同配置执行性能测试，并自动采集、解析主节点8卡流水。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --config deepseek_v4_pro_shared1
```

需要其他数量可复制 JSON、修改 `num_shared_experts` 后使用 `--config /path/to/case.json`。
脚本按配置生成共享专家权重，算子和 CPU Golden 使用同一组权重。共享专家数由 JSON 的 `num_shared_experts` 指定，命令行没有 `--shared-experts` 开关。warmup、iters 和 seed 也在 JSON 中设置。

### BF16 多机测试

在每台机器上运行下面的命令。以两台八卡机器为例，`NODE_RANK` 分别设为 0、1，`MASTER_ADDR`、`MASTER_PORT`、`SHMEM_PORT` 和共享输出目录 `REPORT_DIR` 取相同值。预留三个互不冲突的端口：`MASTER_PORT`、`SHMEM_PORT` 及 `SHMEM_PORT + 2`。设备列表使用本机卡号，EP 取全局 `WORLD_SIZE`。

示例运行 EP16、每卡 3 个路由专家的训练前反向。`--expert-per-rank` 只统计路由专家。

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  experiments/megamoe/tests/bf16/test_mega_moe.py 0,1,2,3,4,5,6,7 \
  --seq 8192 --hidden 5120 --ffn 2304 --topk 6 --expert-per-rank 3 \
  --topk-weight-dtype bf16 --data-profile equal_weights --routed-scaling-factor 1 \
  --routing-mode balanced --swiglu-limit 10 --save-gmm1-output bf16 \
  --shmem-ip-port "tcp://${MASTER_ADDR}:${SHMEM_PORT}" --output "$REPORT_DIR"
```

在同一条命令上加 `--inference` 即可测 BF16 推理前向。训练时加 `--timer` 可采集前反向，`--timer-ranks` 选择采集的全局 rank。`--output` 使用新目录；跨节点收集打点时，各节点须能读取同一共享目录。

## 4. 测试与验证范围

组件 [tests/](tests/) 下的公共回归覆盖 Python 接口、运行时生命周期、wheel 和跨 rank 计时统计。W4A8 与 BF16 的专项用例分别位于 [tests/w4a8/](tests/w4a8/) 和 [tests/bf16/](tests/bf16/)。以下 pytest 命令运行 CPU 回归，不启动 NPU kernel。在仓库根目录运行：

```bash
python -m pip install -r experiments/megamoe/tests/requirements.txt
python -m pytest experiments/megamoe/tests -q
```

| 设备验证路径 | 精度对照 | 性能和打点 |
| --- | --- | --- |
| W4A8 推理 | 同输入的 CPU Golden，对比全量输出和专家计数，包含配置的共享专家 | profiler 统计前向 kernel 耗时；Timer 单独采集 |
| W8A8 推理 | W4A8 批量脚本只覆盖 FP4 权重，W8A8 需使用 FP8 权重单独验证 | 当前量化 Timer 仅支持 W4A8 |
| BF16 推理 | CPU FP64 真值和设备小算子，对比后乘权的 Y 和专家计数 | 连续下发前向，统计稳态耗时 |
| BF16 训练 | CPU FP64 真值和设备小算子，检查 Y、专家计数、dX、dP、dW1、dW2 | 分别测前向、反向和完整前反向，可单独采集训练 Timer |

Ascend950 设备验证按配置分别记录：

- 构建安装检查基础包与 MegaMoE wheel，以及 MegaMoE 启用时 Timer 开、关两种构建。
- 精度覆盖各路径支持的输入类型、路由分布、接收容量边界和多次调用；共享专家用例只在 W4A8 下运行。
- 生命周期检查同一 buffer 切换 B、多流使用、初始化与销毁，以及多机长时间运行。
- 各 EP 规模分别核验。性能对照固定输入、软件版本和计时范围，分别记录各测试入口的性能指标。

### W4A8 测试结果

W4A8 精度覆盖 FP8 和 BF16 两种激活输入、均衡及非均衡路由。小 BS 和 Prefill 的切换可用 B=255/256/257/512、S=0/1/2 检查；同一 buffer 反复切换 `256 → 257 → 256` 可检查同步协议和槽位复用。连续调用还需覆盖 epoch 回绕。CPU Golden 缓存在 `benchmark-golden/`，相同配置可复用。

`batch_launch.py` 在终端输出各 BS 的 `[PERF]` 或 `[ACCURACY SUMMARY]`，报告保存到 `benchmark-runs/ep<EP>_<时间>_<编号>/`。性能来自关闭 Timer 时 profiler 采集的 kernel Duration；原始打点来自单独的采集调用。 流水分析默认解析主节点 rank 0～7，导出时间线 JSON 和各阶段耗时。

| 目录内文件 | 内容 |
| --- | --- |
| `summary.txt` | 各 BS 的性能/精度汇总 |
| `logs/node_<n>_<stage>*.log` | 各节点的构建、Golden、运行、解析日志 |
| `bs_<B>/rank_0/bs_<B>/accuracy.json` | 全部 rank 的精度报告 |
| `bs_<B>/rank_<r>/bs_<B>/profiling.json` | 性能汇总；同目录保留 `kernel_samples.json` 和 `profiler/` |
| `bs_<B>/bs_<B>/timer/` | 主节点8卡原始 `.bin`、`rank_<r>_npu_time.json`、CSV 和 `timer_breakdown.txt/json` |

共享专家的 Timer 分别标记 GMM1、GMM2、两者的前处理以及 SwiGLU，共五类事件。N192/N224 的 tile 元数据记录实际 N 分块，沿用现有 JSON 格式；没有独立的 count 诊断事件。原始 Timer 数据可离线解析。

```bash
python -m deep_ep_experimental.megamoe.timer /实际结果目录/bs_96/bs_96/timer --ranks 0-7
```

所有 BS 执行成功后，终端输出 `[DONE] all nodes completed`；执行或报告校验失败时返回非零退出码。

<a name="bf16测试"></a>

### BF16 测试

以下单机八卡示例覆盖训练前向和反向，TopK weight 为 FP32，Gate/Up 保存为 BF16，SwiGLU 不截断。多机使用[同一入口](#bf16-多机测试)。 输入生成、CPU FP64 真值、设备小算子参考和精度评价都在同一测试入口。使用 `--help` 可查看精度、性能和 Timer 参数。

```bash
python experiments/megamoe/tests/bf16/test_mega_moe.py 0,1,2,3,4,5,6,7 \
  --seq 4096 --hidden 7168 --ffn 3072 --topk 6 --expert-per-rank 4 \
  --topk-weight-dtype fp32 --save-gmm1-output bf16 --swiglu-limit 0
```

| 在同一测试命令上选择 | 检查内容或计时口径 |
| --- | --- |
| `--inference --skip performance --bitwise` | 推理前向。Y 和专家计数与设备小算子逐 bit 对照 |
| `--skip performance --bitwise` | 按所选保存模式，将每个 rank 的 Y、dX、dP、dW1、dW2 与设备小算子全张量逐 bit 比较，另核对专家计数 |
| `--skip performance --bitwise-bigop` | 在双标杆检查之外，逐元素比较上述五项输出与未压缩 BigOp 的原始字节。即使选择 MXFP8 保存，对照仍为未压缩 BigOp。counts 单独按整数值检查；报告列出各项的差异元素数和首个差异的展平索引，有差异即判失败 |
| `--skip precision` | 训练每轮取 EP 组最慢 rank，分别报告前向、反向、完整前反向的 P50 和线性插值 P95，见 `summary.json` 的 `ep_performance`。加 `--profile-forward-kernel` 单独采集设备 kernel |
| `--inference --skip precision` | 连续下发推理前向，统计稳态耗时 |
| `--timer` | 单独采集训练前反向的设备时间线，解析方式见[核内打点](#bf16-核内打点) |

训练可通过 `--save-gmm1-output` 选择 `bf16`、`mxfp8` 或 `none`，并切换 BF16/FP32 路由权重和 balanced/random 路由。各 rank 的 `shmem_bytes` 为实际分配的对称窗口字节数。`event_tail_half_rank_max_mean_ms` 取正式轮次的后半段，逐轮取各 rank 最长 Event 耗时，再求平均。

**与 BigOp、MC 对照。** 同机使用 `--compare-forward` 或 `--compare-backward`，同时构建 MegaMoE 和 EP_GMM_FUSED。对照时固定形状、路由、TopK weight 类型和 seed。三条路径依次启动，逐 rank 核对输入指纹。

- `--compare-forward` 比较三种前向，MegaMoE 按指定保存模式执行训练前向。MC 使用 `alltoallv_gmm` 和 `gmm_alltoallv`，与 BigOp 共用乘权、激活和 TopK 归并。路由计划和权重布局提前准备，计时包含每次调用的数据重排、通信和计算；MC 关闭 AllToAllV 输出导出。
- `--compare-backward` 使用 `save=bf16`，分别测训练前向和完整反向。MC 前向导出专家端 X，反向用 AllToAllV-GMM 计算 dHp 并导出 dO，再用 GMM-AllToAllV 回传各路 dX。三路使用相同的输入和上游梯度 dY，逐 rank 核对 Y、dX、dP、dW1、dW2，反向 Event 和 kernel 计时均从前向建图完成后开始。

`summary.json` 中的三路性能报告分别列出 NPU Event 和设备 kernel 耗时（ms）。Event 单独采样。kernel 耗时按 rank 和轮次统计，分别对一次前向或反向中的计算、通信区间取并集，重叠部分只计一次。两列均取正式轮次的后半段，逐轮取最慢 rank 再求平均。各 rank 的原始样本和 profiler 导出保存在 `--output` 指定的新目录。

**多 microbatch 生命周期。** 两卡回归在同一 `SymmBuffer` 上连续前向、逆序反向，逐字节比较各 microbatch 与串行执行的输出和梯度。加 `--stress-iterations 32` 可检查连续复用。

```bash
torchrun --standalone --nproc-per-node=2 experiments/megamoe/tests/bf16/bf16_lifecycle.py
```

#### BF16 核内打点

构建时启用 `DEEPEP_MEGAMOE_TIMER=ON`，由 `deep_ep_experimental.megamoe.timer.bf16.capture.TimerCapture` 单独采集前反向。

前向时间线在每个物理核上按实际 Wave 标记 Dispatch、GMM1、SwiGLU、GMM2 和 Combine。例如 `wave_1_AIC_GMM1_0` 是该核在 Wave 1 上的第一个 GMM1 任务；同一核、同一 Wave 内的 GMM1 任务从 0 编号。事件详情含专家号和行段。

反向 Wave 的编号表示 dO Dispatch、dHp 和 SwiGLUGrad 的访问顺序。同一波可包含同一专家内相隔较远的块，事件中的起始行是专家内的规范行号。

`wave_7_AIC_BW_DHP_E2` 表示 Wave 7 计算专家 2 的 dHp。`wave_7_AIC_BW_DW2_EARLY_E1` 表示该核等待 Wave 7 的 dO 时，提前计算了专家 1 的一个 dW2 输出 tile。常规 dW2 显示为 `AIC_BW_DW2_E1`。`save=none` 走专家调度，先重算 GMM1。

反向主图保留 Wave 内的工作片段。对于 dW1、常规 dW2、GMM1 重算等专家阶段，同一物理核连续执行同一专家的工作显示为一块；遇到其他任务或已记录的等待则断开。块长包含相邻 tile 间的调度间隔，`active_us` 是块内各打点片段的时长之和。`*_backward_detail_trace.json` 保留逐 tile 坐标和时间。

每个 AIV 的 dP 归约和回传在反向主图上显示为一段 `AIV_BW_DP_REDUCE_SEND`。在明细图中展开 `AIV_BW_DP_REDUCE`，可查看各批 `AIV_BW_DP_SEND` 的起止时间、专家和目标 rank。

时间线 JSON 的事件列表在 `traceEvents` 字段中。`ts`、`dur` 的单位为微秒；`displayTimeUnit: "ns"` 让 Chrome 按纳秒区分同一核上相邻的短事件。

同一组 EP8 原始文件可以导出两种视图。默认隐藏等待；带等待的前向主图展示各核每个 Wave 的首个阶段等待，详细图保留采集到的全部等待。

```bash
python -m deep_ep_experimental.megamoe.timer.bf16 timer --ranks 0-7 --no-include-wait --output-dir traces/work
python -m deep_ep_experimental.megamoe.timer.bf16 timer --ranks 0-7 --include-wait --output-dir traces/with-wait
```

打开 `all_rank_forward_trace.json` 查看 8 个 rank，或打开 `rank_0_forward_trace.json` 单看一张卡。每个 rank 使用自己的前向起点，跨卡横坐标表示相对进度。
