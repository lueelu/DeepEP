<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# MegaMoE Prefill / Decode Beta

公开入口：`deep_ep_experimental.megamoe`。W4A8 共享专家按实际单卡 B 自动选择小 BS / Prefill 策略；路由专家保留 Prefill 调度。
构建和多机测试入口位于本目录的 `scripts/`；机器地址及环境配置由使用者提供。

## 1. 输入、输出与公开接口

定义：`B` 为每卡 token 数，`H` 为隐藏维度，`I` 为 SwiGLU 后宽度，`K` 为 top-k，
`E` 为全局路由专家数，`L=E/EP` 为每卡路由专家数，`S` 为每卡共享专家数。

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
| `l1_weights / l2_weights` | 两对 `(weight, scale)`，当前测试使用 FP4 E2M1 权重 + E8M0 scale |
| `shared_l1_weights / shared_l2_weights` | 共享专家的两对权重，必须同时传入，数量与 buffer 的 S 一致；默认不使用 |
| `cumulative_local_expert_recv_stats` | 可选 int32 `[L]`，在原值上累加专家接收 token 数 |
| `timer` | 可选 `buffer.allocate_timer()`；仅在单独诊断调用中传入 |

`activation_clamp` 可传非负有限 host 实数，None 使用 FP32 最大有限值。GMM1 的 BF16 输出转 FP32 后，gate 只裁上限，up 裁到 [−C, C]，随后计算 SwiGLU。C 是每次调用按值捕获的参数。其余 recipe、activation、fast_math 参数使用默认值。所有张量在 buffer 所在 NPU 上。
FP8 输入仍需打包到通信区，因此流水中的 `AIV_QUANT` 此时表示打包，不是重新量化。

权重在测试/推理前转换一次，转换过程不计入性能测量：

| 原始 ND 权重 | packed FP4（uint8，两个值/字节） | E8M0 scale |
| --- | --- | --- |
| GMM1 | `[L,2I,H/2]`，N 方向先 gate 后 up | `[L,2I,H/64,2]` |
| GMM2 | `[L,H,I/2]` | `[L,H,I/64,2]` |

调用 `w1, w2 = transform_weights_for_mega_moe(raw_w1, raw_w2)` 得到 Ascend NZ 权重。FP4 routed GMM1 一次准备两套 weight/scale：实际 B≤256 使用 `[gate128, up128]` 融合布局，B>256 使用 `[gate | up]` 配对布局。转换不进入计时区间，但额外保留一份 GMM1 权重和 scale；GMM2 不复制。
请完整保留返回的 `w1`，不要解包后重组、原地修改或自行搬运其张量；权重更新或设备变化时从原始 ND 重新转换。共享专家将首维 L 换成 S，并传入 `shared=True`，FP4 使用 planar；原生 E4M3 的两层保持 ND，第一层同时交织 gate/up 与 E8M0 scale。
不能直接传入已经做过 CUDA 专用重排的权重。

buffer 的构造参数如下，模型参数含义与测试 JSON 相同：

```python
from deep_ep_experimental.megamoe import (
    StoreGroup, SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe,
)

# torchrun 启动后先 torch.npu.set_device(local_rank)。
group = StoreGroup.from_env()  # 主机协调用 TCPStore，算子通信用 SHMEM
buffer = SymmBuffer(
    group, num_experts=384, num_max_tokens_per_rank=4096, num_topk=6,
    hidden=7168, intermediate_hidden=3072,
    num_shared_experts=0, mma_type="fp8xfp4", activation="swiglu",
    ranks_per_node=8, timeout=120,
)
```

准备输入后依次调用 `buffer.validate_inputs(B)`（BF16 时额外传入 `x=x_bf16`）、`buffer.prepare(B)` 和算子。
执行成功后，所有 rank 按相同顺序调用 `buffer.destroy()`、`group.close()`；
异常处理及完整可运行调用示例见 [example.py](python/deep_ep_experimental/megamoe/example.py)。

约束：EP=2～128，E 可整除 EP 且每卡最多 128 个路由专家；S=0～8；B=1～65536；
H=1024～8192 且为64的倍数，I=256～4096 且为128的倍数，1≤K≤min(32,E)。
下面的 16 卡跨机路径另限 EP32、W4A8（A8W4）和实际 B=1～256。
各 rank 同序、同 B 推理。图捕获前准备全部计划，保持输入、输出、权重 owner 和 buffer 存活至最后一次 replay 完成。
`mma_type="fp8xfp8"` 使用 A8W8 路由专家，不支持共享专家/timer；**下文 benchmark 固定测试 W4A8，不提供权重 dtype 开关**。

### 共享计划和权重更新

两机 EP32、每机 16 rank 且最大 BS≤256 时，默认容量为 16 个 (view, BS) 计划。显式 `num_max_configs=1` 沿用单计划工作区。用 `SymmBuffer(group, ..., base=root)` 创建同设备、同组的 view。root 的 E、最大 BS、K、H 需要逐维覆盖每个 view。各 rank 在首次 forward 前按相同顺序调用各 view 的 `prepare(bs)`。不同 view 或图串行使用共享 payload，最后一个 view 的 `destroy()` 关闭通信资源。

`owner = prepare_weights_for_mega_moe(l1, l2, shared=False, input_layout="nd")` 接受连续权重对或两组逐专家的 (weight, scale) 列表。两个投影传 `owner.l1, owner.l2`。routed FP4 也可从 `planar_nz` 生成 fused 布局，或用 `prepared` 和 `planar_l1` 导入两份已有布局。路由 ID 使用 EPLB 映射后的物理专家号。

框架在 capture 外按设备事件先排空旧读者，再更新权重和映射。连续 owner 用 `update`，列表用 `update_experts(indices=slots, l1=l1, l2=l2, replace=True)` 更换存储并保留表地址。索引也可用位置参数或 `experts=` 传入。同一次调用只传一种索引。每个投影可给逐专家的 `[(weight, scale), …]`，或给 `(weights, scales)`，其中两个序列各自可为 list 或 tuple。列表内的权重和 scale 各自可省略 expert 维或保留长度为 1 的首维；按单专家导入的 NZ 权重须占有完整独立分配。连续的线性 FP8 权重与 E8M0 scale 接受格式码 0、2、30；已准备的 packed FP4 权重仍须为 NZ。通过 `expert_views(layout="planar")` / `expert_views(layout="fused")` 可取得整组 W1、W2 及各自 scale，`expert_views(i)` 仍可取单专家视图。列表可用 `refresh_gmm1(indices, source_layout="planar")` 局部刷新，连续 owner 整批刷新。写入一份 GMM1 后调用 `refresh_gmm1(source="planar")` 或 `source="fused"` 同步另一份，全部权重与 scale 已写完时调用 `commit_prepared_update()`。最后一次旧图读取和异步复制完成后可调用 `release_retired()`。

列表模式下，批量导出的 FP4 NZ 权重是原始物理分配。独立的二维或三维权重保持原 rank；二维 NZ 上增加的逻辑单专家维会解析回原二维对象。可直接对这些对象做 packed FP4 format cast，并在物理形状一致时原地 `copy_`。scale 和按索引取得的专家仍按单专家视图导出。替换专家后需重新获取该槽的视图。

### 共享专家策略

当 W4A8 的 S>0 时，由实际 B 选择策略，容量 `num_max_tokens_per_rank` 不参与判断：

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
tiling ABI 为 224 字节。
CATLASS GMM 的布局和同步机制见 [GMM 说明](csrc/kernels/mega_moe/catlass_gmm/README.md)。

所有 rank 必须按相同顺序、相同 B 调用，切换 B 时在计时前集体调用 `prepare(B)`。
已有 `prepare` 一致性检查用于防止各 rank 选择不同握手策略。共享计算五类 timer 保留，
不添加 count 诊断事件；N192/N224 的 tile 元数据按实际 N 分块记录，JSON 格式不变。

验证边界建议覆盖 B=255/256/257/512、S=0/1/2，以及同一 buffer 下反复切换
`256 → 257 → 256` 和 epoch 回绕。CPU 协议测试不能替代 CANN 编译和 NPU 精度/性能验证。

## 路由权重位置与 ProcessGroup

通信实现称为 **MTE** 和 **URMA**，产品识别仍自动选择实现。
`SymmBuffer(..., topk_weights_type=0)` 的参数与 ops-transformer 一致：

| 值 | 路由专家计算顺序 |
| --- | --- |
| 0（默认） | GMM1 → clamp → SwiGLU → BF16/MX 量化 → GMM2 → 乘路由权重 → reduce |
| 1 | GMM1 → clamp → SwiGLU → 乘路由权重 → BF16/MX 量化 → GMM2 → reduce |

共享专家不乘路由权重。两种模式因量化位置不同不保证数值相同。
所有 rank 必须选择相同模式；它影响通信内存布局，buffer 创建后不要修改。
URMA 的提前模式把每条路由的 FP32 权重附在 Scale PUT 中，保持两次 data PUT，增加元数据传输与 Gather；默认模式不传路由权重。
CPU Golden 与 NPU cascade 均按同一模式及 clamp 生成参考结果。benchmark 配置 JSON 的 `model` 支持 `topk_weights_type` 和 `activation_clamp`。

```python
import torch
import torch_npu
import torch.distributed as dist
from deep_ep_experimental.megamoe import SymmBuffer, fp8_fp4_mega_moe

torch.npu.set_device(local_rank)
dist.init_process_group("hccl")
group = dist.group.WORLD
buffer = SymmBuffer(
    group, num_experts=384, num_max_tokens_per_rank=1024,
    num_topk=8, hidden=3584, intermediate_hidden=1280,
    num_shared_experts=0, ranks_per_node=group.size(), topk_weights_type=1,
)
# 填入 buffer.x/x_sf/topk_idx/topk_weights，并提前转换 w1/w2。
buffer.prepare(bs)
fp8_fp4_mega_moe(y, w1, w2, buffer, activation_clamp=7.0)
buffer.destroy()
dist.destroy_process_group()
```

接口语义参考 [ops-transformer MegaMoE](https://gitcode.com/cann/ops-transformer/blob/master/mc2/mega_moe/docs/torchapi_mega_moe.md)。

### 双机 EP32 decode

每机 16 个连续 rank，HNS_1825，两端安装相同的 wheel 和 SHMEM SDK。UBX 在 EP4/8/16、
每 rank 容量不超过 256 时选基础 MTE，更大的容量选 URMA。双机 EP32 decode 采用分层 MTE + RoCE。
目标 BS96 的 BF16 候选调度使用协议 5，构建时设置
`DEEPEP_MEGAMOE_BF16_BS96=ON`。性能对照使用 `DEEPEP_MEGAMOE_TIMER=OFF`。

```bash
bash experiments/megamoe/scripts/build.sh --rdma-backend HNS_1825
```

用同一份脚本对照精简前的 `9080aeb` 和当前 wheel。两端使用相同的协议、构建选项和输入；
每次切换 wheel 后重新启动作业。生成用例使用固定种子，在共享目录准备一次 CPU golden。已有历史固定 bank 时，设备作业可将 `--golden "$CASE/golden"` 换为 `--bank "$CASE/bank"`。
以下命令中的 `NODE_RANK` 在两机分别为 0 和 1，`MASTER_ADDR` 指向第 0 节点。

```bash
script=experiments/megamoe/scripts/check_ep32_replay.py
python "$script" golden --input bf16 --golden "$CASE/golden"
launch=(torchrun --nnodes=2 --nproc-per-node=16 --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR")
# 两机均安装对照 wheel
"${launch[@]}" "$script" eager --golden "$CASE/golden" --out "$CASE/before"
# 两机均切换到当前 wheel
"${launch[@]}" "$script" replay --golden "$CASE/golden" --reference "$CASE/before" --out "$CASE/after"
```

`replay` 比对全 32 rank 的 160 轮输出摘要、实际物理路由计数及 8 轮交替调用。
用 `--input fp8` 重复 golden、eager 和 replay 可复查调用方提供 scale 的路径。
性能作业在 timer-off 的两套 wheel 上按 A/B/B/A 顺序运行 `perf --out 新目录`；
历史输入对照需在四批作业中传入同一 `--bank`。每批预热 20 次、采集 30 次，`fixed_perf.json` 保存每次的 32 个 rank 耗时和后 15 轮汇总。

## 2. 编译安装

W4A8 在 B≤256 时使用较小 wave、rail 路由中转和打包 count。
rail 按每8个 rank 一组，EP≤8 或完整8卡分组启用，其余 EP 回退直发路由。
B>256 保留 Prefill 输入协议，所有路径保留输出跨卡同步。

要求 Linux、Python ≥3.11、Ascend950、CANN 9.2.0 及匹配的 Torch/torch_npu；框架环境由使用者准备。

```bash
git clone https://github.com/lueelu/DeepEP.git
cd DeepEP
source /path/to/cann/set_env.sh
bash experiments/megamoe/scripts/build.sh
python -m pip show ascend-deepep
```

在匹配的 Torch/torch_npu 环境中执行。实验构建入口默认开启 MegaMoE 和 timer，
自动初始化固定版本 SHMEM/CATLASS 子模块、校验或构建 SHMEM，再构建安装 wheel。
`PYTHON=/path/to/python` 可指定解释器；`DEEPEP_MEGAMOE_TIMER=OFF` 关闭设备打点。
已激活 Conda 时，入口会在 Python 启动前优先使用该环境的运行库。
已有 SDK 可用 `SHMEM_ROOT`、`CATLASS_ROOT` 指定；已有子模块须保持干净且版本与[依赖锁文件](../../dependencies.lock.json)一致。
需要重建仓内 SHMEM 时，停止使用该 SDK 的作业后传 `--rebuild-shmem`。
wheel 位于 `dist/build-*/`；运行前按构建输出设置 SHMEM 的 `LD_LIBRARY_PATH`。
基础包的 `pip wheel` 默认关闭 MegaMoE，详见[构建文档](../../docs/build.md)。

## 3. 多机测试

### 配置与启动

所有节点需要相同绝对路径的共享 checkout、匹配的 Conda/CANN 环境、SSH 访问和每机8张空闲 NPU。
在仓库外创建机器配置，通过 `--hosts-file` 显式传入。以下地址仅用于文档示例，需替换为已分配的节点：

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

### 常用参数

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

# BF16 输入，性能 + 自动流水解析。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --input bf16 --bs-values 96

# EP128：将 HOSTS_FILE 改为包含16个节点的配置。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 128
```

### 模型和共享专家配置

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
脚本自动生成对应共享权重、传给公开接口，CPU Golden 也包含共享专家；
**没有 `--shared-experts` 开关**。warmup/iters/seed 同样在 JSON 中修改。

### 测试结果

精度测试以相同输入、权重和路由生成的 CPU Golden 为参考，生成输出和专家计数的校验报告。
CPU Golden 缓存位于 `benchmark-golden/`，相同配置可复用。
性能测试根据关闭 timer 时 profiler 采集的算子 kernel Duration，生成耗时统计报告。
流水分析根据独立采集的设备 timer 数据，生成流水 JSON 和阶段耗时报告，默认解析主节点 rank0～7。

终端输出各 BS 的 `[PERF]` 耗时或 `[ACCURACY SUMMARY]` 精度摘要，结果保存至：
`benchmark-runs/ep<EP>_<时间>_<编号>/`。

| 目录内文件 | 内容 |
| --- | --- |
| `summary.txt` | 各 BS 的性能/精度汇总 |
| `logs/node_<n>_<stage>*.log` | 各节点的构建、Golden、运行、解析日志 |
| `bs_<B>/rank_0/bs_<B>/accuracy.json` | 全部 rank 的精度报告 |
| `bs_<B>/rank_<r>/bs_<B>/profiling.json` | 性能汇总；同目录保留 `kernel_samples.json` 和 `profiler/` |
| `bs_<B>/bs_<B>/timer/` | 主节点8卡原始 `.bin`、`rank_<r>_npu_time.json`、CSV 和 `timer_breakdown.txt/json` |

原始 timer 数据也可离线解析：

```bash
python -m deep_ep_experimental.megamoe.timer /实际结果目录/bs_96/bs_96/timer --ranks 0-7
```

所有 BS 执行成功后，终端输出 `[DONE] all nodes completed`；执行或报告校验失败时返回非零退出码。

## 4. 测试与验证范围

```bash
python -m pip install -r experiments/megamoe/tests/requirements.txt
python -m pytest experiments/megamoe/tests -q
```

保留公开 API、数值参考、Golden 缓存、打包、主机协调与计时解析测试。
CPU 检查不启动设备 kernel，也不作为设备验证结果。Ascend950 设备验证记录按以下维度区分：

- MegaMoE 开启/关闭、timer 开启/关闭的构建安装验证。
- BF16/FP8 输入、不同路由、共享专家、容量边界和多次调用的精度验证。
- 多流生命周期、BS 切换、初始化/销毁和多机长时间运行验证。
- 各声明 EP 规模的设备测试及同输入、同软件版本、同计时范围的性能对比。
