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

### BS96 双机验收

在两台机器的目标 Python 环境设置 `NODE_RANK`（0 或 1）和同一 `MASTER_ADDR`。以下命令按每机 16 卡运行，输出目录以本轮作业新建的目录为准。

```bash
python experiments/megamoe/scripts/build.py --rdma-backend HNS_1825
ranks="$((16 * NODE_RANK))-$((16 * NODE_RANK + 15))"
launch=(torchrun --nnodes=2 --nproc-per-node=16 --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500)
for input in fp8 bf16; do
    python -m deep_ep_experimental.megamoe.golden --config ep32_bs96 --ep-size 32 \
        --ranks "$ranks" --input "$input" --golden-dir benchmark-golden
    "${launch[@]}" -m deep_ep_experimental.megamoe.benchmark --config ep32_bs96 --input "$input" \
        --mode accuracy --check-graph --golden-dir benchmark-golden --output-dir "bs96-$input-accuracy"
    "${launch[@]}" -m deep_ep_experimental.megamoe.benchmark --config ep32_bs96 --input "$input" \
        --mode perf --perf-method legacy-read --output-dir "bs96-$input-perf"
done
```

`accuracy.json` 包含 CPU golden 的误差和物理专家计数。换输入 replay 逐次与 eager 比较。上述性能配置关闭 timer，预热 20 轮、采集 30 轮，按最后 15 轮计算各轮跨 rank 的最小值、最大值及其均值。每个 rank 的 manifest 记录有效 wave 和已装载库的 SHA-256。性能采样检查同一设备流中的 ReduceSum/MegaMoE 对数与顺序。可在 perf 命令加 `--waves all --retest-top 3` 扫描 0–64 并复测前三项，0 使用自动 wave。

#### 独立 eager 与 160 轮变输入重放

单卡运行 `python -m pytest -q experiments/megamoe/tests/test_npu_prepared_weights.py` 检查真实二维、三维及混排 NZ。上面的双机 `--check-graph` 另行检查换表、公开视图写入权重和 scale、刷新 GMM1 后 target/draft 的交替重放，每个 rank 的 BS 目录保存 `eplb_graph.json`。

长跑在两版 wheel 各自的 Python 环境中使用**同一份候选脚本**。下面的 `CASE` 指向能从两节点读取的结果目录，`SCRIPT` 指向候选源码树内脚本的绝对路径。固定输入性能测试还从同一源码树加载 `performance.py`。使用节点本地盘时，在比较前汇集各目录的全部 32 个 `rank_*` 子目录。准备输入和 CPU golden 时每机仅处理本机 16 个 rank。

```bash
SCRIPT="$PWD/experiments/megamoe/scripts/check_ep32_replay.py"
INPUT=bf16  # 分别用 bf16 和 fp8 各跑一遍
CASE="/data/ep32-validation/$INPUT"
python "$SCRIPT" bank --input "$INPUT" --shared-fixture native-fp8 --bank "$CASE/bank" --golden "$CASE/golden" \
    --first-rank "$((16 * NODE_RANK))" --end-rank "$((16 * (NODE_RANK + 1)))"
launch=(torchrun --nnodes=2 --nproc-per-node=16 --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500)
```

两台机器先切换至 `b90b417` 的已安装 wheel 环境，从源码目录外启动独立 eager，等待退出。随后两台机器切换至候选 wheel 环境，另起一组进程运行 replay。每次使用新输出目录；`CANDIDATE_REV` 填入实际构建候选 wheel 的 Git 提交。

```bash
"${launch[@]}" "$SCRIPT" reference --revision b90b417 --bank "$CASE/bank" --out "$CASE/reference"
# 两台机器均切换到候选环境后执行
"${launch[@]}" "$SCRIPT" replay --revision "$CANDIDATE_REV" --bank "$CASE/bank" --out "$CASE/replay"
# 收齐全部 rank 后在候选环境执行一次
python "$SCRIPT" compare --bank "$CASE/bank" --reference "$CASE/reference" --replay "$CASE/replay" --out "$CASE/report"
```

输入目录保存 routed FP4、原生 shared FP8、两者的 E8M0 scale、各轮物理路由及系数。`bf16` 保存激活的原始位模式并传给 BF16 入口，`fp8` 保存预量化值及其 scale。两种输入分别生成 golden 和报告。原生 shared 权重直接生成 E4M3 字节，`--shared-fixture fp4-subset` 可复验旧数据分布。候选预热 20 次，捕获一次，每轮清零计数并读回完整输出，最后用 8 次交替 eager/replay 复查已知输入。比较器核对 160 轮摘要、逐元素输出、各 rank 实际物理路由汇总的计数、首轮 CPU golden，以及 159 对相邻输出。若有跨版本差异，用候选环境另起双机作业，将 `replay` 子命令改为 `candidate-eager` 并写入新目录；再次 `compare` 时增加 `--candidate-eager "$CASE/candidate-eager"` 和新的 `--out`。逐 rank 报告同时保留两版 eager 与候选图的差异。

#### EP32 BS96 BF16 候选构建

构建两端相同的候选 wheel 时设置 `DEEPEP_MEGAMOE_BF16_BS96=ON`，计时用 `DEEPEP_MEGAMOE_TIMER=OFF`。默认构建保留协议 4。候选构建使用协议 5，SHMEM 分配 25 个 QP，Dispatch 使用 1–8，目标形状的 24 个 Combine owner 使用 1–24。仅在 BF16、BS96、H5120、2I4608、K6、每卡 13 个 routed FP4 专家、一个 native FP8 shared 专家、双机 2×16 和 28 AIC/56 AIV 的分层路径上启用新调度。

在两端从源码树外运行同一份新版脚本，共用上面生成的 bank。每次更换 wheel 后重新启动作业。

```bash
# 基线 wheel（协议 4）
"${launch[@]}" "$SCRIPT" candidate-eager --revision "$BASE_REV" --bank "$CASE/bank" --out "$CASE/base-eager"
# 候选 wheel（协议 5）
"${launch[@]}" "$SCRIPT" candidate-eager --revision "$CANDIDATE_REV" --bank "$CASE/bank" --out "$CASE/candidate-eager"
"${launch[@]}" "$SCRIPT" replay --default-wave --revision "$CANDIDATE_REV" --bank "$CASE/bank" --out "$CASE/replay"
python "$SCRIPT" compare --bank "$CASE/bank" --baseline-eager "$CASE/base-eager" \
    --candidate-eager "$CASE/candidate-eager" --replay "$CASE/replay" --out "$CASE/report"
```

`run.json` 记录实际协议、加载库摘要、wave 基数及候选 plan 的 `bf16_bs96_active`。BF16 候选的正确性验证会在同一缓冲区交替调用 BF16 和预量化 FP8，核对调度、输出和专家计数，并记录 `mixed_input_checked`。独立 FP8 bank 可用于核对 FP8 回退的 golden 输出。正式计时按基线、候选、候选、基线依次运行下节的 `candidate-eager --fixed-perf`，四次使用独立输出目录。比较时用 `--baseline-eager` 指向对应基线目录。

#### 固定 BF16 输入的双机 A/B

使用上面 BF16 bank 的第 0 轮。两台机器均用 `npu-smi info` 确认各自的 16 张卡空闲。参考和候选依次使用各自的 timer-off wheel，从源码树外运行同一份脚本。将下列命令中的 `CASE` 指向上面生成的 BF16 目录。

```bash
"${launch[@]}" "$SCRIPT" reference --fixed-perf --revision b90b417 --bank "$CASE/bank" --out "$CASE/ref-fixed"
# 在两台机器均切换到候选 wheel 后执行
"${launch[@]}" "$SCRIPT" candidate-eager --fixed-perf --revision "$CANDIDATE_REV" --bank "$CASE/bank" --out "$CASE/candidate-fixed"
# 收齐 32 个 rank 后，在候选环境执行一次
python "$SCRIPT" compare --fixed-perf --bank "$CASE/bank" --reference "$CASE/ref-fixed" \
    --candidate-eager "$CASE/candidate-fixed" --out "$CASE/fixed-report"
```

权重和 scale 在 CPU 按 gate128/up128 排列，routed W1 另保留 planar 版本。双机在计时前通过公开 prepared 接口导入。预热 20 次并确认实际 wave 为 5，批次前用 20 次 100000×30000 FP32 Exp 对齐。每轮以 int32 ReduceSum 读取 256 MiB 的设备数据，再测 MegaMoE。每个作业输出目录下的 `fixed_perf.json` 保留 32×30 个样本，以及后 15 轮的 rank 0 均值和逐轮最慢 rank 的均值。`fixed-report/summary.json` 汇总完整输出逐位对比、物理专家计数及首轮 CPU golden。

附件对照环境为 CANN 9.2.1、Python 3.12.16、Torch 2.10.0+cpu、torch_npu 2.10.0.post7、HNS_1825，PI_ON_CHIP=0。分阶段测量时依次构建 CANN 兼容基线、提前 drain、提前元数据三个提交，复用同一 bank 并为每次运行指定新目录。阶段 trace 单独使用 timer-on wheel，以同一 owner 的记录检查 Reduce、export、host barrier 与 commit 的顺序。

兼容入口 `mega_moe(x, ids, weights, w1, w2, buffer, ...)` 返回 `(y, per_call_counts)`。
接口签名参照 DeepGEMM `78b6900`。本实现使用 int32 路由 ID、E8M0 输入 scale，
输入 buffer 是普通 NPU 存储，需要打包进入通信区。路由权重在 GMM2 后施加，
与在中间量化前施加权重的实现不保证数值等价；CPU 参考遵循本实现的计算顺序。

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
tiling ABI 为 232 字节。
CATLASS GMM 的布局和同步机制见 [GMM 说明](csrc/kernels/mega_moe/catlass_gmm/README.md)。

所有 rank 必须按相同顺序、相同 B 调用，切换 B 时在计时前集体调用 `prepare(B)`。
已有 `prepare` 一致性检查用于防止各 rank 选择不同握手策略。共享计算五类 timer 保留，
不添加 count 诊断事件；N192/N224 的 tile 元数据按实际 N 分块记录，JSON 格式不变。

验证边界建议覆盖 B=255/256/257/512、S=0/1/2，以及同一 buffer 下反复切换
`256 → 257 → 256` 和 epoch 回绕。CPU 协议测试不能替代 CANN 编译和 NPU 精度/性能验证。

## 路由权重位置与 ProcessGroup

UBX（板卡 ID `0x44`）在创建 `SymmBuffer` 时按进程组大小和 `num_max_tokens_per_rank` 选通信后端。EP4、EP8、EP16 的容量上限在 1–256 时走基础 MTE，超过 256 时走 URMA。双机 EP32 要求 `ranks_per_node=16` 且容量上限在 1–256，走分层 MTE + RoCE。其他 UBX EP32 配置在构造时报告错误。后续 forward 的实际 token 数不会改变后端，共享视图沿用 `base` 的选择。
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

## 2. 编译安装

W4A8 在 B≤256 时使用较小 wave、rail 路由中转和打包 count。
原有路径的 rail 按每8个 rank 一组，EP≤8 或完整8卡分组启用，其余 EP 回退直发路由。
显式指定 `ranks_per_node=16` 时 rail 使用16卡分组；跨机采用下文的 MTE＋RoCE 路径。
B>256 保留 Prefill 输入协议，所有路径保留输出跨卡同步。

要求 Linux、Python ≥3.11、Ascend950 及配套的 CANN、驱动和 Torch/torch_npu。CANN 9.2.0 是资格基线，构建会记录实际 SDK 版本。

```bash
git clone https://github.com/lueelu/DeepEP.git
cd DeepEP
source /path/to/cann/set_env.sh
bash experiments/megamoe/scripts/build.sh
python -m pip show ascend-deepep
```

在匹配的 Torch/torch_npu 环境中执行。实验构建入口默认开启 MegaMoE 和 timer，
先将 `CATLASS_ROOT` 设置为与依赖锁文件匹配的 CATLASS 源码目录。脚本初始化固定版本的 SHMEM 子模块，校验或构建 SDK，再构建并安装 wheel。
`PYTHON=/path/to/python` 可指定解释器；`DEEPEP_MEGAMOE_TIMER=OFF` 关闭设备打点。
已激活 Conda 时，入口会在 Python 启动前优先使用该环境的运行库。
已有 SHMEM SDK 可用 `SHMEM_ROOT` 指定；仓内 SHMEM 子模块须保持干净且版本与[依赖锁文件](../../dependencies.lock.json)一致。
需要重建仓内 SHMEM 时，停止使用该 SDK 的作业后传 `--rebuild-shmem`。
HNS1825 新固件适配固定到 SHMEM !944 合入版本；当前分支重建与历史 commit 兼容迁移见
[SHMEM !944 复测说明](../../docs/shmem-944.md)。
wheel 位于 `dist/build-*/`；运行前按构建输出设置 SHMEM 的 `LD_LIBRARY_PATH`。
基础包的 `pip wheel` 默认关闭 MegaMoE，详见[构建文档](../../docs/build.md)。

两机各 16 卡的 EP32 使用 HNS_1825 SDK。

```bash
# 已有默认 SHMEM 构建时，先停止使用该 SDK 的作业，再显式重建。
bash experiments/megamoe/scripts/build.sh --rdma-backend HNS_1825 --rebuild-shmem

```

首次构建可省略 `--rebuild-shmem`。后续复用仓内 SDK 时继续传入相同的 `--rdma-backend`。
默认值 `NONE` 保留不启用 RoCE 的 SDK 构建；16卡单机及原有8卡分组路径不要求 RoCE。
后端写入 `deepep-sdk.json` 并参与构建缓存和复用检查，MegaMoE 的 Host/Device 编译自动采用匹配配置。
外部 SDK 需由提供方登记真实后端，见[SDK 登记说明](../../docs/build.md#分步骤构建外部-sdk打包或-ci)。
所有运行节点安装相同 wheel、使用与 wheel 匹配的 SHMEM 库，并按部署环境配置 NPU—NIC 绑定与 RoCE 网络。
`DEEPEP_SHMEM_ENDPOINT` 是进程会合地址，不替代网卡 IP/GID 配置。

## 3. 多机测试

DSV41 的 EP16/EP32、BS=6,12,…,144 共 48 个完整 MegaMoE 性能 case，使用独立入口，见
[DSV41 性能测试说明](scripts/dsv41_perf.md)。扩展后的 EP32 范围需要重新编译安装 wheel。

### BS32：保存配置，分别编译和运行

两台机器各16卡的 BS32 精度、性能及流水测试使用 `scripts/run_bs32.sh`，在**已有的同一个 checkout** 中执行。
此入口不重新克隆、不自动拉取代码，也不在运行时编译或安装；依赖留在当前仓库的 `third-party/`。
模型固定为 H=3584、I=1280、E=384、K=8、S=0、EP32，使用 FP8 输入和 FP4 权重、balanced 路由及全部 token 精度检查。

先激活匹配的 Torch/torch_npu Python 环境，在两台机器各配置一次：

```bash
bash experiments/megamoe/scripts/run_bs32.sh configure
# 依次输入本机编号（0/1）、节点0的IP、CANN set_env.sh、HNS_1825。
# 默认保存当前 Python、端口29654/26733、Golden workers=4。

# 两台分别编译安装；只在首次安装或需要更新构建时执行。
bash experiments/megamoe/scripts/run_bs32.sh build

# 两台分别执行，只运行测试。以后重复这条命令即可，不再询问配置。
bash experiments/megamoe/scripts/run_bs32.sh run
```

编译 warning 太多时使用 `bash experiments/megamoe/scripts/run_bs32.sh build --errors-only`。
终端只显示构建阶段提示、错误及其前6行/后8行上下文；完整输出（包括 warning）仍写入
`<case_dir>/build_node_<rank>_<随机字符>/build.log`，成功或失败时都会打印日志路径。
这是终端显示选项，不改变编译器检查或失败退出码，也不会隐藏被提升为 error 的 warning。
默认 `build` 同样保存日志，并显示完整输出。屏蔽模式下没有持续输出时，可用 `tail -f <日志路径>` 查看进度。

配置默认保存到 `.megamoe-bs32/<hostname>.json`，已加入 Git 忽略规则；共享 checkout 的两台主机也各用自己的文件。
Python 使用保存的绝对路径，后续自动加载保存的 CANN 环境；迁移或删除 Python 环境后需显式更新配置。
可以用 `--config /absolute/path/node.json` 指定其他配置文件，之后各阶段都须传同一路径。
配置文件是 JSON 数据，不作为 shell 脚本执行；环境变量不会覆盖已保存的节点/IP设置。

```bash
bash experiments/megamoe/scripts/run_bs32.sh show-config
# 显式修改指定字段，其他设置保留；改节点0的IP时，两台都要修改。
bash experiments/megamoe/scripts/run_bs32.sh configure --master-addr 192.0.2.10
# 输出放到数据盘，避免每次在 /root 新建整个仓库。
bash experiments/megamoe/scripts/run_bs32.sh configure --case-dir /data/megamoe-bs32
# Python 环境变更时：
bash experiments/megamoe/scripts/run_bs32.sh configure --python /path/to/env/bin/python
```

也可直接编辑已保存的 JSON（路径使用绝对路径）。更改 CANN、Python 或 RoCE 后端后应重新执行 `build`；
更换仓内 SHMEM 后端时须停止使用该 SDK 的作业，再执行 `build --rebuild-shmem`。
`run` 自动生成本机16个 rank 的 CPU Golden，后续校验并复用缓存；可提前执行 `golden` 单独准备。
默认 Golden 位于 `benchmark-runs/bs32/golden/`，每次运行的配置快照和日志位于该目录上级的 `node_<rank>_<随机字符>/`。
配置与结果均保留在原目录，`git pull` 不会清除它们。此入口使用仓内 SHMEM 和环境变量 `CATLASS_ROOT` 指定的 CATLASS。
若之前使用的是复制到服务器的旧 `run_bs32.sh`，请改用上述仓库内路径，旧脚本不会随 `git pull` 更新。

精度通过后，直接复用上述机器配置测性能；两台机器都执行同一个命令：

```bash
bash experiments/megamoe/scripts/run_bs32.sh perf
# 旧方法：整批 Exp20 + 连续 MegaMoE（默认）。
bash experiments/megamoe/scripts/run_bs32.sh perf --perf-method legacy
# 新方法：每轮清理 L2 + device sleep + SHMEM 全卡同步 + MegaMoE。
bash experiments/megamoe/scripts/run_bs32.sh perf --perf-method shmem
# 可选：两台使用相同的预热、采样次数。
bash experiments/megamoe/scripts/run_bs32.sh perf --warmup 20 --iters 100
# 可选：在 BS32 后追加 BS1、4、8、16、256。两台必须同时使用此开关。
bash experiments/megamoe/scripts/run_bs32.sh perf --extra-bs --warmup 20 --iters 100
```

`perf` 不生成或读取 Golden，不重新编译。默认预热20轮、采集100轮，并统计后50轮；
模型、BS32、FP8输入、FP4权重、balanced路由均与精度用例一致。
`--extra-bs` 仅用于 `perf`，默认关闭；开启后按 BS32、1、4、8、16、256 的顺序逐项测试。
各项沿用相同模型、seed、路由、`--perf-method`、`--warmup` 和 `--iters`，每项独立启动进程组，
按本项 BS 分配对称窗口，避免 BS256 改变原 BS32 的容量和内存布局。
追加模式下，每项日志和原始结果保存在运行目录的 `bs_<BS>/` 子目录，根目录的
`summary.txt` / `summary.csv` 汇总全部六项的 rank-min-mean 和 rank-max-mean；任一项失败即停止。
两端必须使用相同开关、性能参数和支持 BS256 的新 wheel；首次使用需两端重新执行 `build`
（无需为此增加 `--rebuild-shmem`）。启动前检查已安装 wheel 的 BS 上限，旧 wheel 会提示重建。
BS256 的本地协议及容量回归不替代 NPU 精度和性能验证。
`--perf-method legacy` 保留连续批次方法：整批开始前通过 TCPStore 会合，排入一次 Exp 前置计算，
随后连续下发 MegaMoE。`alignment.repeats` 默认从800回退为20，复用约12 GB显存/卡；
轮间不插入同步或额外计算。`iteration_sync=none`，`alignment_scope=once_before_batch`。

`--perf-method shmem` 参考 [DeepGEMM-Ascend bench_msprof](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/main/deep_gemm/testing/bench.py)：
每轮写零一个大小为设备 L2 两倍的缓冲区，排入20,000,000系统计数周期的设备 sleep，再执行 SHMEM WORLD barrier，
最后下发 MegaMoE。这个模式不分配12 GB Exp缓冲区。单机调用 `aclshmemx_barrier_all_on_stream`；
跨机 RoCE 用单 AIV 辅助 kernel 调用 `aclshmemx_roce_barrier_all`，显式提供256 B UB和event 0，
使用 SDK 的全连接 QP及 WORLD 同步状态。当前锁定 SDK 的通用流上 barrier 通过远端 GM 同步，不能直接替代 RoCE 接口。
`iteration_sync=shmem_world_on_stream`，`alignment_scope=before_each_iteration`。

两个模式都只在批次边界做 Host 同步，不建立 HCCL 进程组。SHMEM barrier 在当前流排队，
MegaMoE 的专属流通过已有事件依赖等待它；下一轮预处理也等待前一轮完成。
只统计 MegaMoE kernel Duration，清 L2、sleep、barrier、Exp均排除在外。
`manifest.json` / `profiling.json` 记录 `perf_method`；每个 BS 的 `measurement.json` 还记录实际接口、L2缓冲字节数和sleep周期，
`shmem` 模式将不使用的 Exp配置记录为 `alignment=null`。两台机器必须选择相同方法，否则在开始采样前报错。
新模式需要两端重新 `build` 安装 wheel（无需为此重建相同版本的 SHMEM SDK）；首次 helper调用在采集外预热。
逐轮 barrier 用于减少快慢卡累积偏移，不保证各卡同一物理时刻启动。两种模式的缓存状态不同，性能结果应分别比较。
结果保存到 `<case_dir>/perf_node_<rank>_<随机字符>/`，终端会打印完整路径和汇总：

| 文件 | 内容 |
| --- | --- |
| `summary.txt` | 全EP每轮最短/最长 rank 的平均 kernel Duration，单位微秒 |
| `case.json`、`launch.json`、`run.json` | 模型与轮次、机器配置、运行模式与选中的 timer ranks |
| `console.log` | torchrun 与 benchmark 完整输出 |
| `rank_<全局rank>/bs_32/profiling.json` | 全32 rank 的汇总及逐轮耗时 |
| `rank_<全局rank>/bs_32/kernel_samples.json`、`profiler/` | 本 rank 的全部原始样本及 profiler 数据 |

现有 benchmark 日志的 `E2E` / `mean_us` 是每轮**最短** rank 的平均值；
`rank-max-mean` / `rank_max_mean_us` 是每轮**最长** rank 的平均值，建议同时记录并查看 rank 分布。
它们是融合 kernel 的设备执行时长（含内部通信和等待），不含 Python 调用、权重准备或退出清理，
也不代表经过跨设备时钟校准的全局起止跨度。`perf` 不传入设备 timer；安装了支持 timer 的 wheel 也不会打点，
wheel 的编译配置可在每个 rank 的 `manifest.json` 中核对。需要恢复不支持 timer 的构建时执行普通 `build`。

#### EP32 的 Dispatch QP8 与 Combine QP16

启动入口默认BS32；`run`、`golden`、`perf`、`trace`均可用 `--bs` 选择每rank实际token数（1～256）。
例如两台机器使用相同参数依次运行：

```bash
bash experiments/megamoe/scripts/run_bs32.sh run --bs 72
bash experiments/megamoe/scripts/run_bs32.sh perf --bs 72
bash experiments/megamoe/scripts/run_bs32.sh perf --bs 128
```

trace需先构建timer版本，再运行 `trace --bs 128 --include-wait`。
`--bs`与原来的 `--extra-bs`批量性能测量互斥。

`run_bs32.sh run/perf/trace` 统一使用 Dispatch QP8 和 Combine QP16，无需附加算法参数：

- SDK 使用 QP0，16 个网关 owner 分别负责 Combine QP1～16。BS96 匹配混合专家配置时由 AIV1 负责，其余目标形状使用 AIV0。
- H3584/K8 默认形状使用 14 个 Dispatch worker、wave 6。BS96/128 的 H5120/K6 混合专家配置使用 28 个 worker、wave 5。
- 按 lane 初始化路由；网关用两个输入缓冲交替加载专家行并完成 FP32 加权归并。
- AIV1 使用跨专家两槽 Combine 行环，连续 Combine 任务之间保留在途搬运。

原有 QP 数量、Dispatch worker 数量和流水实验开关，以及 A/B、pipeline suite 入口已移除。
机器配置、RDMA 后端、采样轮数和 timer 参数仍按下文配置。运行结果标记为
`implementation=dispatch_qp8_combine_qp16`。两端需完整重新构建并重建 buffer；Python/native 接口协议为 17，
旧 wheel 不兼容本次调用接口和通信窗口布局。同版本 SHMEM SDK 可复用。

```bash
# 两台机器分别构建；均完成后，同时执行相同的 run/perf 命令。
bash experiments/megamoe/scripts/run_bs32.sh build --errors-only
bash experiments/megamoe/scripts/run_bs32.sh run
bash experiments/megamoe/scripts/run_bs32.sh perf
```

AIV1→gateway 使用512 B包，每包前508 B保存BF16原始字节，最后4 B保存完整 invocation sequence。
网关检查所需行的每个包标记后才读取数据；没有独立行 ready 区。直接回写同机源卡仍使用普通BF16行。
归并按原 top-k 顺序完成 BF16→FP32、乘权重和累加，每个 token 的远端部分和通过所属 QP 发回源卡。

跨专家行环在槽复用时等待该槽的 MTE3 完成；最后两个槽在尾部排空。
就绪探测失败时保留在途行；实际进入 SwiGLU、共享激活或 Dispatch 前排空，保护复用的 UB 和事件。
因此当前重叠发生在连续 Combine 行之间。SwiGLU 继续保持最高优先级。
计时记录提交、槽复用等待和行生命周期；生命周期包含完成后尚未回收的时间，不是精确链路传输时长。

g00～g07 的 AIV0 必须完成本核全部路由和共享专家 prologue 后才进入归并。
所有 lane（包括空 lane）均在 host barrier 后提交最终 commit，再等待对端 commit；
输出就绪屏障后，owner 执行 quiet，其他 AIV 执行 Unpermute，kernel 完成后才可复用窗口。
性能以 timer 关闭的 `perf` 结果为准，本地 CPU 回归不能替代 CANN 编译、NPU 精度和性能实测。

采集内部流水时，先在两台机器分别构建安装支持 timer 的 wheel，再在两台机器启动 `trace`：

```bash
# 两台均安装完成后再开始 trace；无需重新构建 SHMEM。
bash experiments/megamoe/scripts/run_bs32.sh build --timer --errors-only
bash experiments/megamoe/scripts/run_bs32.sh trace

# 默认采集跨机配对 rank0、rank16；也可扩大覆盖范围，两台参数必须相同。
bash experiments/megamoe/scripts/run_bs32.sh trace --timer-ranks 0-31 --warmup 2 --iters 1
```

`trace` 独立采集设备流水，不执行性能 Profiler 或 Golden 校验。默认预热2轮、运行1轮，
每个 rank 共调用3次 MegaMoE；仅在采集批次的最后一轮、选中的 rank 上设备打点。
可用 `--warmup`、`--iters` 覆盖次数；与 `perf` 一样支持 `--perf-method legacy|shmem`，默认每批 Exp20 后连续下发 MegaMoE，
临时占用约12 GB显存/卡。未选中 timer 的 rank 也必须执行相同轮数。
正式性能测试使用 `perf`，其默认预热20轮、采集100轮不变。
`trace` 不会自动构建；未安装支持 timer 的 wheel 时会在启动 worker 前提示执行 `build --timer`。
更新后需在两台机器重新执行上述 `build --timer` 安装新版 Python 包，才能使用独立的 trace 模式；SHMEM 可复用。
32个 rank 始终全部参加运行；采集后各节点只自动解析自己拥有的选中 rank，默认隐藏 WAIT 事件。
`trace --include-wait` 显示全部 WAIT，`trace --hide-wait` 显式隐藏。该开关只影响流水 JSON 和事件汇总，
原始 BIN/解析 CSV 仍保留 WAIT，其他事件的时间戳不变，不压缩等待时间。

A8W4 的跨机 RDMA 在所属 AIV 的 `/RDMA` 子轨道记录，带有阶段（`dispatch_meta`、
`dispatch_payload`、`combine`）、对端全局 rank、数据/commit 类型及字节数；新增记录还包含QP和payload token区间：

| 事件 | 含义 |
| --- | --- |
| `AIV_RDMA_POST` | 非阻塞 put 调用的提交时间；不代表数据在网络上的完整传输时间 |
| `AIV_RDMA_QUIET_WAIT` | 已有 QP quiet 调用的剩余完成等待；data 的 bytes=0 表示批量字节数未统计，commit 为4字节 |
| `AIV_RDMA_RECV_WAIT` | 已有 commit 等待及后续加载排序；包含等待对端生产、传输和发布 commit 的时间 |

默认只显示 POST；诊断通信等待时使用 `--include-wait`。设备打点不新增通信同步，仍有采样开销，
性能基线继续使用独立的 timer 关闭采样。新打点改变了原始 buffer 布局，两节点都需重新 `build --timer`，
并用同一版本解析器解析新采样；旧 BIN 请用采集时版本解析器。
已有的 `*_parsed.csv` 可直接重新出图（无需重跑设备，也可用于旧布局的已解析 CSV）：

```bash
python -I -m deep_ep_experimental.megamoe.timer.trace_parser \
  --timer-dir /实际结果目录/bs_32/timer --output-dir /实际结果目录/timer_hidden \
  --rank-start 0 --rank-end 0 --time-origin rank
```

上述命令默认隐藏 WAIT；加 `--include-wait` 可恢复显示。节点1的单卡采样对应 rank16，需同时修改两个 rank 参数。

排查 `AIC_WAIT_INPUT_PREP` 结束到首次 GMM 之间的空白时，两节点更新并重新 `build --timer`，
然后运行 `trace --include-wait`。新增的启动阶段位于各核 `/STARTUP` 轨道：

| 事件 | 测量范围 |
| --- | --- |
| `AIC/AIV_MOE_STARTUP` | 路由专家阶段入口到本核首次 Dispatch、GMM1 prologue 或 GMM1 tile 入口；无任务的核在阶段退出时结束 |
| `AIC/AIV_DISPATCH_BUFFER_INIT` | Dispatch 缓冲区视图和 scratch 地址初始化 |
| `AIV_INPUT_PREP_BARRIER_WAIT` | 输入准备后的既有 PIPE_ALL 排空和 AIC/AIV 屏障；AIC 对应原有 `AIC_WAIT_INPUT_PREP` |
| `AIV_COUNT_TABLE_PREPARE` | AIV1 准备 count/prefix 表的完整父区间 |
| `AIV_COUNT_METADATA_WAIT` | 获取所有来源的有效 count：包含 DMA、epoch 校验、转置、重试与退避，并非纯网络延迟 |
| `AIV_GATEWAY_PAYLOAD_CHECK` | count-ready 发布后，消费者批量获取本卡全部16个网关序号槽，并完成 payload DMA 前的同步 |
| `AIV_GATEWAY_PAYLOAD_BROADCAST` | owner 确认本卡入站 payload 后，向同机其他15卡及自身发布序号并完成 DDR 同步；不含此前的入站等待 |
| `AIV_GATEWAY_PAYLOAD_WAIT` | Dispatch 首次观察到当前序号后，对该来源执行 acquire 的区间，记录来源 rank、网关 rank 和完整序号。缓存命中不重复记录 |
| `AIV_COUNT_PREFIX_BUILD` | 计算专家 token 总数和 prefix 表，包含既有向量完成同步 |
| `AIV_COUNT_TABLE_PUBLISH` | count/prefix 写回、既有 MTE3 完成同步和 count-ready 发布 |
| `AIC/AIV_COUNT_READY_WAIT` | AIC/AIV0 等待对应 block 的 AIV1 发布 count-ready |

这些观测不新增 PIPE_ALL、跨核屏障或通信。`AIC/AIV_MOE_STARTUP` 是启动总区间，
不能与其子事件相加；首个 GMM tile 内部的输入等待仍由原 `AIC_GMM1_WAIT` 记录。
owner 的八路入站汇总使用 `AIV_PAYLOAD_ALL_LANES_WAIT`，附带首轮缺失lane位图、轮询轮数及sequence；广播单独使用
`AIV_GATEWAY_PAYLOAD_BROADCAST`。前者是开始检查后的剩余等待，不能当作完整网络传输耗时。
消费者的本地16槽汇合计入 `AIV_GATEWAY_PAYLOAD_CHECK`，不再远端读取网关 commit。
观察 owner 的广播结束是否早于其他核到达输入屏障，才能判断这项工作是否被原有等待覆盖。
如果总启动区间长而以上子阶段都短，剩余部分包含专家扫描、调度及 GMM 调用内部首次 tile 前的准备。

解析同时输出 `startup_breakdown.csv`，按 rank/group/core 列出阶段起点、耗时及网关身份，
始终保留启动 WAIT，不受 JSON 的 `--include-wait` 开关影响。时间基准为每个 rank 的原始首事件；
`scope=envelope` 是父区间，各核和父子区间不可累计当作端到端耗时。
新启动打点再次扩展了 buffer 布局，旧 BIN 需使用采集时的解析器；旧采样无法补出此前未记录的阶段。

头开销细分版本在 `/RELAY` 轨道新增 `AIV_RELAY_STEP`，包含 `packet_wait`、`topk_load`、
`route_build`、`route_post`、`route_complete`、`count_publish` 六类步骤，附带原 source rank 和专家范围。
其中 `route_post` 测量 DMA 提交及相关标量工作，`route_complete` 测量既有完成等待；二者不是链路传输时间。
`count_publish` 是 gateway 向专家卡发布计数，与本卡的 `AIV_COUNT_TABLE_PUBLISH` 含义不同。
JSON 隐藏 WAIT 时也隐藏 `packet_wait` 和 `route_complete`；`startup_breakdown.csv` 始终保留这些步骤。

`AIV_COUNT_METADATA_WAIT` 的 task/args 及 `startup_breakdown.csv` 新增快照次数、重试次数、快照字节数、
DMA/校验转置/退避累计耗时、成功快照耗时，以及最后失败快照的未就绪 source 列表和代表性本卡局部 expert 编号/epoch。
这些字段只覆盖 DMA 快照路径；不具备诊断数据的旧采样或逐槽回退路径留空，不作为0次重试解释。
每次成功获取只写一次诊断汇总。UB 空间足够时交替使用两份现有空闲 Dispatch 缓冲，保留最后失败快照；
成功后用一次 MTE3 批量复制到 timer GM，由主机解析 source/expert/epoch，设备不再逐槽解码。
`diagnostic_us` 报告这次复制及其完成同步的开销；不再包含原来的标量解码开销。无重试时不复制。
快照上限为4096个 int32；超过上限或 UB 不够保留双份快照时，仍保留次数及耗时统计，
`failed_snapshot_available=false`，失败来源字段留空。新 buffer 布局需两节点重新构建并使用同版本解析器。
最后失败快照描述的是一次采样，不能推断真实的网络到达先后顺序。
来源位图覆盖0～63，超出时 `source_mask_overflow=true`，来源列表仍保留完整快照中解出的 source。
上述累计子耗时已经包含在父区间中，不能再与父区间相加。

Relay 的主机容量按完整512 B包计算，包含每包的末尾标记，并按有效 payload 计算批次数。
BS32/K8 的256个 top-k 条目需要3个包，UB 配置为512个槽（有效容量508），
避免原来256槽仅有254个有效位置而产生“254＋2”两批回退；同时满足单批 route 快路径的输出缓冲要求。

跨机 Combine 在 `/COMBINE_PROGRESS`、`/COMBINE_REDUCE` 等轨道记录：

| 事件 | 测量范围 |
| --- | --- |
| `AIV_COMBINE_INIT` | 各 owner 解析自己 token 区间的路由 |
| `AIV_COMBINE_PROGRESS` | ready 检查和归并提交的父区间，包含待发行数、POST 数及扫描统计 |
| `AIV_COMBINE_REDUCE` | 单 token 的加载、FP32 加权归并及部分和写回 |
| `AIV_COMBINE_READY` | 轮询观察到的 pending/ready mask |
| `AIV_COMBINE_SYNC` | Export 与 host barrier 前后的边界 |
| `AIV_HOST_COMBINE_BARRIER_WAIT` | 16卡机内完成屏障 |
| `AIV_COMBINE_ASYNC` | 跨专家行环的提交、槽复用等待及在途生命周期 |

ready 记录不是硬件精确到达时刻；父子区间和重叠区间不能累加作为端到端收益。
解析器保留历史已解析 CSV 的事件解码；旧 BIN 应使用采集时版本的解析器。
`combine_progress_breakdown.csv` 始终保留推进诊断，即使 JSON 隐藏 WAIT。

结果位于 `<case_dir>/trace_node_<rank>_<随机字符>/`，保留配置快照、`console.log` 和各 rank 的资源信息，
不生成 `profiling.json`、`kernel_samples.json` 或 `profiler/`，流水输出包括：

| 文件 | 内容 |
| --- | --- |
| `summary.txt` | 流水采集次数、本机选中 ranks 及输出路径 |
| `parse_timer.log` | 本机流水解析日志 |
| `bs_32/timer/rank_<r>_npu_time.bin` | 原始设备打点，可重复离线解析 |
| `bs_32/timer/rank_<r>_npu_time.json` | 单 rank 的 AIC/AIV 流水，支持 Chrome JSON trace 查看器 |
| `bs_32/timer/all_rank_trace.json` | 本机选中 ranks 的合并流水，不包含另一台机器的原始数据 |
| `bs_32/timer/timer_breakdown.txt/json` | 阶段耗时拆分；本机自动解析后对应本机选中 ranks |
| `bs_32/timer/combine_progress_breakdown.csv` | Combine 初始化、逐次推进、机内屏障和最终 drain 的耗时及待发行数 |

流水按每个 rank 自己的起点归零，不能用不同 rank 的坐标相减计算跨机延迟。
设备打点反映代码阶段及等待，不是逐 WQE 的网卡执行轨迹；正式性能基线使用 `perf` 的结果。
`trace` 默认只采两张卡，以控制原始 timer 文件及解析开销；分析慢 rank 时可通过 `--timer-ranks` 定点采集。
运行、导出或自动解析失败均返回非零退出码，不打印成功的 `[DONE] trace`。

### 通过限长文本复制传输流水

服务器只能复制文本、每次最多65535字节时，使用离线工具 `experiments/megamoe/scripts/trace_text.py`。
只需更新仓库中的 Python 文件；已有采样可直接导出，**无需重编译、重新安装 wheel 或重跑算子**。
工具使用 Python 标准库，不需要 Torch、NPU 或额外压缩软件，也可在 Windows 上恢复。

先在每台服务器分别导出一个已完成的 trace 结果目录：

```bash
# 也可直接传 bs_32/ 或 bs_32/timer/；有多个 BS 时必须指定一个。
# output-dir 必须尚不存在，避免混入上次导出的分片。
python3 experiments/megamoe/scripts/trace_text.py export \
  /实际结果目录/trace_node_0_xxx \
  --output-dir /tmp/trace-share-node0 \
  --label "EP32-BS32-node0"

# 每次只显示一块；完整复制 BEGIN 到 END（含这两行），作为一条消息发送。
python3 experiments/megamoe/scripts/trace_text.py show /tmp/trace-share-node0 --part 1
# 按 export 报告的总块数继续 --part 2、--part 3……
```

默认导出目录中**所有已解析的 rank**，也可用 `--ranks 0,3-5` 缩小范围。每块默认不超过60000字节，
可用 `export --max-bytes 65535` 调整；上限计算包括编号、SHA-256、Base64、头尾标记、最终换行，
并按 CRLF 换行预留空间。`show` 的标准输出只有一块 ASCII 文本，不带日志或颜色。
不要一次打印所有分片，也不要把终端提示符、命令、`...` 或手工删改后的内容放进块内。
每块的完整校验信息都在文本中，**不必另行复制导出目录的 manifest.json**。

传输保留各 rank 的 `rank_<r>_npu_time_parsed.csv` **完整原始字节**，包括全部核、WAIT、
整数 cycle、wave/tile/expert、RDMA 和网关元数据；不裁时间、不抽样、不根据界面的 WAIT 隐藏设置删事件。
整个包先 XZ 无损压缩，再 Base64 编码和分片。导出会检查 CSV 的核覆盖、列数和任务计数，
拒绝明显截断的文件。输出块数取决于实际数据，不保证完整流水只需一块。

标准结果目录下的 `case.json`、`run.json`、所选 rank 的 `manifest.json` 和 `resources.json`
会一起打包。额外的性能结果可显式附带，例如：

```bash
python3 experiments/megamoe/scripts/trace_text.py export /实际结果目录/bs_32/timer \
  --output-dir /tmp/trace-share-with-perf \
  --attach /对应perf结果目录/rank_0/bs_32/profiling.json
```

默认不收集 `launch.json`、环境变量或其他日志。两个节点分别生成独立包，不要把不同节点、不同导出或
不同次运行的分片合并为同一个包；label、rank 列表和运行 manifest 用于识别各自的采样。
只会传输实际采集的卡，导出不能补出未打点的 rank。

接收端把复制的文本存入 UTF-8 文件：可以每块一个文件，也可以在同一个文件中连续粘贴多个完整块。
可以乱序；重复粘贴内容完全相同的块不会影响恢复。检查和恢复示例：

```bash
# 检查已收到多少块、缺哪些编号；不写恢复文件。
python3 experiments/megamoe/scripts/trace_text.py inspect copied-messages.txt

# 校验全部分片、整包和每个原文件后，恢复到一个新目录。
python3 experiments/megamoe/scripts/trace_text.py restore copied-messages.txt \
  --output-dir restored-trace-node0

# 也可传多个文本文件，或用单独的 "-" 从 stdin 读取。
python3 experiments/megamoe/scripts/trace_text.py restore part1.txt part2.txt \
  --output-dir another-restored-trace
```

缺块、冲突重复、Base64 损坏、SHA-256 不符或混包时会明确报错。CRLF 和 Base64 的普通空白/换行变化可接受，
不会尝试猜测修复其他字符。恢复目录必须尚不存在，已有文件不会被覆盖。
恢复后 `timer/` 中包含逐字节一致的 CSV，并自动生成保留 WAIT 的单 rank/合并 JSON、
`startup_breakdown.csv` 和阶段拆分；使用 `restore --no-render` 可只恢复文件。
附带配置位于 `context/`，显式附带文件位于 `attachments/`。
**跨 rank 的时间仍各自归零，不能据此把两张卡的坐标差当作机间延迟。**

已安装新版 Python 包时，也可通过
`python -I -m deep_ep_experimental.megamoe.timer.text_transfer` 使用相同子命令。

### 两机各16卡：MTE＋RoCE Decode

通信模式由已有的 `ranks_per_node` 决定，没有新增公开 `transport` 参数：

| 配置 | 通信路径 |
| --- | --- |
| `ranks_per_node=8` | 保留原有 MTE 路径 |
| `ranks_per_node=16`、EP16、同一主机 | MTE |
| `ranks_per_node=16`、EP32、两台主机 | 机内 MTE，跨机 RoCE |

`ranks_per_node=0` 保留原有默认行为，不自动检测16卡平台。两机路径要求一台机器对应 rank 0～15、
另一台对应 rank 16～31，gateway 配对为 `0↔16`、`1↔17`、…、`15↔31`。
初始化会集体验证 Linux 主机身份、设备唯一性、rank 分组、SDK 后端及库身份；不匹配即报错。
两机路径支持 `mma_type="fp8xfp4"`（W4A8/A8W4），实际每卡 B=1～256；
FP8 和内部量化的 BF16 输入均沿用原接口，容量可大于实际 B。更多16卡主机暂不支持。
A8W8 仅保留单机 MTE 路径。

双机 list 权重（每 rank 至少两个 routed 专家）的 accuracy 模式可加 `--check-graph`。该检查在 capture 后轮换物理专家槽并更新路由，交替 replay 共用 arena 的 target 和半批大小的 draft view，与独立地址表驱动的 eager 算子逐元素比较 BF16 输出和每专家计数。

每卡打包 Top-k ID 和权重后先经配对 gateway 发送元数据。目标形状在机内按 4 mesh × 4 lane 分两级交换，转发前检查整条 stripe 的包尾。每个来源按 512 B 分包，每包包含 127 个路由项和一个完整序号。索引核写入逐专家来源行，builder 用逐来源直方图生成共享前缀和专家行数。配对来源同时生成 Compact Combine 的 token 列表和 top-k mask。

目标形状的 Top-k AllGather 先投递完整来源记录，再按 4 个 mesh 来源检查整条 stripe 的所有 marker。AIV0 在本核 reset 完成后按 host 制表。每条记录占 `ceil(BS*K/127)*512` 字节。制表 UB 在对齐的 prefix 后放 16 条记录和 16 组 256 B 直方图；第二个 host 等前一轮 VF 释放后复用。配对来源的索引写回完成后复用压紧 ID，生成升序 token 列表和 mask。四记录转发与最多八行 trailer 快照分时使用核尾保留区。

Dispatch 的 8 个 QP 按实际 token 区间 `[B*i/8, B*(i+1)/8)` 分工。每包 480 B 激活或 scale 数据和 32 B trailer，最后一个字保存当前计划域的完整序号。对每个专家切片，worker 先搬本机及已确认 gateway 的行，迟到的每个远端来源只记录一个区间。索引区按实际 BS 给 16 个远端来源预留各自对齐的区间，延期来源在同一轮中推进。若 UB 无法同时容纳这些区间，则轮转读取至多 8 行的窗口并保留每行的交付状态。每行的所有包确认后，按原目标行写回。行数据与 metadata 全部写完才发布 tile ready。两个 AIV1 在前台 Dispatch slice 前有机会批量探测未确认的 lane，按各 lane 的实际末包各取 32 B trailer，最多占用 256 B 暂存。配对网关全部非空 lane 就绪后，每个 helper 通知本机一半 rank。HNS 的 route all-gather 路径从一个对齐的 32 B 通知块向八个独立槽发 DMA，整组写入退役后才记录发布完成。双机启动校验 HNS_1825、协议版本 4 及实际 QP/worker 配置。发送 owner 回收自己的 CQ。两台机器均需重建并安装同一套 core、扩展与设备库。

网关将同一 token 的专家行按 top-k 次序用 FP32 加权，16 个 QP owner 将对应 partial 送回源卡。两机 EP32 下，BS96、H5120、I2304、K6、每 rank 13 个 routed FP4 专家、1 个原生 FP8 shared 专家、无 Combine 量化和 top-k 权重预取、16～32 个 AIC 且 AIV 数为其两倍时，由前 16 个 group 的 AIV1 对 compact 序号按 16 交错分工。BF16 和预量化 FP8 激活均适用。其余分层配置由 AIV0 按连续区间分工。空 QP 同样发送 commit。HNS 的目标 BS32/96 在 16～32 对核、BS128 在 28 对核且输入含 FP8 scale 时分别汇总 Dispatch reader 和本地 Combine producer 的完成信号。读者闭合后，owner 在本 QP 全部 partial 已投递时即可提交。28 对核时输出可逐 token 等待，其他核数组合在输出前汇合。输出核按 token 获取所依赖的 QP，合并本地贡献、远端 partial 和 shared 结果后写回 BF16。通用形状使用 host 屏障。每轮 kernel 返回前各 owner 排空发送，两端使用相同的协议和窗口布局。

两机 RoCE runtime 的 `barrier()` 和 `destroy()` 先同步本卡全部 stream，再通过现有
StoreGroup（TCPStore）或 ProcessGroup 交换完成状态；全部 rank 成功后才允许清理 SHMEM。
当前 kernel 在返回前已等待应用 QP1～16 完成，这一顺序确保释放前没有未完成的通信。
这里不调用直接读取对端 GM 的通用 `aclshmem_barrier_all()`，也不要求机内各 rank 建立 RoCE QP。
所有 rank 必须按相同顺序调用，调用期间不得并发提交新的算子；任一 rank 同步失败则保留资源并要求重启作业。
该修复涉及 Python 和原生扩展，两台机器都需重新构建/安装匹配的 wheel；已有 SHMEM SDK 无需为此修改。

在两台已分配设备的机器上分别执行下列命令，只将 `NODE_RANK` 改为0或1。
地址和端口是占位示例，需替换为实际配置；两台机器各启动16个进程：

```bash
export MASTER_ADDR=192.0.2.1
export MASTER_PORT=29654
export DEEPEP_SHMEM_ENDPOINT=tcp://192.0.2.1:26733
export NODE_RANK=0  # 另一台机器使用1。
torchrun --nnodes=2 --nproc-per-node=16 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0 \
  --module deep_ep_experimental.megamoe.example --input bf16
```

示例使用 B=17，并从 `LOCAL_WORLD_SIZE` 传入 `ranks_per_node=16`。它检查输出有限性，
该 example 以及现有 benchmark 均固定使用 W4A8，没有 `--mma-type` 开关。
W4A8 的完整精度比较需要 benchmark 的 CPU Golden。共享存储上先生成所需 case 的 Golden，再在两台机器启动：

```bash
# 仅生成一次；参数须与后续 benchmark 完全一致。
python -m deep_ep_experimental.megamoe.golden --config deepseek_v4_pro \
  --bs-values 1,17,31,32 --ep-size 32 --ranks 0-31 --input bf16 --routing balanced \
  --golden-dir /path/to/shared/golden

# 每台机器分别运行，沿用上面的 MASTER_*、NODE_RANK 和 SHMEM 环境。
torchrun --nnodes=2 --nproc-per-node=16 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0 \
  --module deep_ep_experimental.megamoe.benchmark --config deepseek_v4_pro \
  --bs-values 1,17,31,32 --mode accuracy --input bf16 --routing balanced \
  --golden-dir /path/to/shared/golden --output-dir /path/to/new/accuracy-run
```

性能测试将 `--mode accuracy` 改为 `--mode perf`，并选择新的输出目录；计时范围仍遵循下文说明。
还应覆盖 FP8 输入、random/skewed 路由、S=0/1、重复调用及 B 切换。
本分支的真实 CANN 编译、两机 NPU 精度、RDMA 到达可见性和性能验收仍待完成；CPU 测试不代表设备通过。

A8W8 使用公开 `mega_moe` 接口。下面片段假定32个进程已设置各自 NPU、endpoint 和 `group`，
模型已提供同一 NPU 上的连续输入、路由及原始 FP8 ND 权重：

```python
from deep_ep_experimental.megamoe import SymmBuffer, mega_moe, transform_weights_for_mega_moe

B = 32
# x_bf16: BF16 [B,H]；ids: int32 [B,K]；weights: FP32 [B,K]。
# E 为全局路由专家数，L=E/32。
# raw_w1: (FP8 E4M3 [L,2I,H], E8M0 [L,2I,H/64,2])。
# raw_w2: (FP8 E4M3 [L,H,I], E8M0 [L,H,I/64,2])。
buffer = SymmBuffer(
    group, num_experts=E, num_max_tokens_per_rank=B, num_topk=K,
    hidden=H, intermediate_hidden=I, num_shared_experts=0,
    mma_type="fp8xfp8", ranks_per_node=16,
)
w1, w2 = transform_weights_for_mega_moe(raw_w1, raw_w2)
buffer.topk_idx[:B].copy_(ids)
buffer.topk_weights[:B].copy_(weights)
buffer.validate_inputs(B, x=x_bf16)
buffer.prepare(B)
y, per_call_counts = mega_moe(
    x_bf16, buffer.topk_idx[:B], buffer.topk_weights[:B], w1, w2, buffer,
)
# 所有 rank 成功完成后，同序销毁；异常通知/退出按 example.py 的处理方式。
buffer.destroy()
group.close()
```

A8W8 的完整数值与性能验证需使用匹配 FP8 权重的模型参考，不能将上述 W4A8 Golden 当作 A8W8 结果。

### 原有每机8卡：配置与启动

下面的 `batch_launch.py` 仍按每机8个进程组织作业；两机16卡使用上面的 `torchrun` 命令。
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
| `--perf-method legacy\|shmem` | 默认 legacy：每批 Exp20；shmem：每轮清 L2、device sleep和SHMEM同步，同样作用于timer采集 |
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

### 融合 GMM1 → GMM2 的独立性能测试（单场景与 EP/BS 扫参）

[`scripts/bench_gmm_chain.py`](scripts/bench_gmm_chain.py) 在单张 Ascend950 上测试：
`npu_grouped_matmul_swiglu_quant_v2`（GMM1 + SwiGLU + MX 量化）→ `npu_grouped_matmul`（GMM2）。
GMM2 直接消费融合算子产出的 FP8 激活及 E8M0 scale；权重为 MXFP4、NZ 格式，最终输出 BF16。
接口按官方 [融合算子 MxA8W4 示例](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul_swiglu_quant_v2.md)
及 [GMM mx 伪量化示例](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md) 接入。
需要支持这些接口和数据类型的 CANN、PyTorch/torch_npu 环境；缺失时直接报错，结果中保留实际版本和算子 schema。

默认复现 EP32、每个源 rank BS32、H3584、I1280、E384、TopK8、S0、balanced 路由，seed=20260922。
复用现有 benchmark 的输入、路由、原始权重和 MXFP8 输入量化，按专家分组重建 Dispatch 的逻辑输出。
每卡12个专家，**实际 GMM 总行数 M=256**，各专家行数为 `[22,22,22,22,21,21,21,21,21,21,21,21]`。
GMM1 的 K/N 为3584/2560，GMM2 为1280/3584。`--bs` 是每个源 rank 的 token 数，不能直接当作 GMM 的 M。

在仓库根目录、已激活的 CANN/torch_npu 环境中直接运行，无需重编 DeepEP、初始化 SHMEM 或启动 torchrun：

```bash
# CPU 预览实际路由和矩阵形状，不导入 Torch，不生成权重。
python experiments/megamoe/scripts/bench_gmm_chain.py --dry-run

# 单卡测量，默认20轮预热、100轮正式采样。
python experiments/megamoe/scripts/bench_gmm_chain.py --device 0

# 可选：NPUGraph replay，减少 Python 下发造成的空隙；结果单独标记 graph。
python experiments/megamoe/scripts/bench_gmm_chain.py --device 0 --mode graph

# 复现逻辑 rank30 的输入和权重，仍只需一张物理卡。
python experiments/megamoe/scripts/bench_gmm_chain.py --device 0 --rank 30 --iters 200

# 一次跑完 EP16/32/64 × BS1/8/16/32/256，共15个场景。
python experiments/megamoe/scripts/bench_gmm_chain.py \
  --device 0 --ep-sizes 16,32,64 --bs-values 1,8,16,32,256 \
  --warmup 20 --iters 100

# 同一批场景使用 NPUGraph replay；也可加 --dry-run 只预览形状。
python experiments/megamoe/scripts/bench_gmm_chain.py \
  --device 0 --ep-sizes 16,32,64 --bs-values 1,8,16,32,256 --mode graph
```

`--device` 选择进程可见的 NPU，`--rank` 选择逻辑专家归属，两者互不绑定。
可通过 `--ep-size/--bs/--hidden/--intermediate/--experts/--topk/--routing/--seed` 调整场景。
此脚本只测路由专家，S固定为0。测试期间应让所选设备空闲。

`--ep-sizes` 和 `--bs-values` 执行笛卡尔积，也可以只传其中一项，例如
`--ep-size 16 --bs-values 1,8,16,32,256`。复数参数与对应单数参数互斥；所有场景在启动前验证专家整除和逻辑 rank 范围。
未传复数参数时，保持原来的 EP32/BS32 单场景行为。
默认 balanced 路由下，各逻辑 rank 的专家行数分布相同；默认 `--rank 0` 即复现这种计算形状。
每卡专家数分别为 `384/EP = 24/12/6`，GMM 总行数为 `BS × TopK`：

| 每个源 rank 的 BS | 1 | 8 | 16 | 32 | 256 |
| --- | --- | --- | --- | --- | --- |
| 每卡 GMM 总行数 M（TopK8） | 8 | 64 | 128 | 256 | 2048 |

每个专家有 `floor(M / 本卡专家数)` 或多1行。BS1/EP16 有16个空专家，BS1/EP32 有4个空专家；
group list 保留这些0，不删除专家或扩充 token 数。

批量测试在同一张卡上顺序启动独立子进程，每个场景结束后释放设备、graph和profiler资源。
父目录为 `benchmark-runs/gmm-chain/sweep_<随机编号>/`，每个场景的结果放在
`ep<EP>_bs<BS>/rank_<r>_<随机编号>/`，同级 `console.log` 保留该场景完整日志。
顶层 `summary.txt` 提供15场景对比表；`summary.csv` 提供全部5项计时的 mean/median/min/max，
以及EP、BS、M、专家数、精度状态、采样轮次、结果路径；`summary.json` 另保存逐专家行数、命令和失败原因。
每完成一个场景即更新汇总。单场景失败会保留日志、继续后续场景，最后返回非零退出码；
失败或未完成场景的计时留空，不参与性能比较。每次扫参都会创建新目录，避免读到上次的结果。

单场景每次创建独立的 `benchmark-runs/gmm-chain/rank_<r>_<随机编号>/`，可通过 `--output` 更改父目录。
其中 `results.json` 保存场景、精度报告、输入/权重摘要、软件版本、全部采样及汇总；
`summary.txt` 是可读摘要，`profiler/` 保留原始 trace 和 `kernel_details*.csv`。
按现有性能测试口径取正式采样的后半段（奇数向上取整），每项报告 mean/median/min/max，单位均为微秒：

| 指标 | 范围 |
| --- | --- |
| `fused_gmm1_us` | 融合 GMM1 + SwiGLU + MX 量化的 kernel Duration |
| `gmm2_us` | GMM2 的 kernel Duration |
| `kernel_sum_us` | 同一轮两个 kernel Duration 之和 |
| `chain_span_us` | 从融合 kernel 开始到 GMM2 kernel 结束，包含两者间隙 |
| `inter_kernel_gap_us` | 两个 kernel 之间的空隙，eager 模式下可能受 CPU 下发影响 |

两个算子之间及正式轮次之间不插入 synchronize、Event、CPU 拷贝或额外量化。
统计要求每轮恰好两个匹配的 kernel、处于同一设备同一 stream；发现额外 kernel、重复导出、缺失样本或时序异常会报错，保留原始数据供排查。
时间差使用 CSV 十进制原值计算，避免大绝对时间戳转为 `float` 后丢失亚微秒精度。
采样中的 `start_us_exact` / `duration_us_exact` 字符串保留精确值；`inter_kernel_gap_raw_us` 保留有符号间隙。
重叠容差仍为0.01微秒，超过时报告具体间隙、设备/stream及两条记录的 CSV 行号；容差内的负间隙仅在 `inter_kernel_gap_us` 中记为0。
若所用版本把一个算子展开为多个 kernel，需要根据该版本的 trace 调整归属解析后再统计。

默认在计时前按每个非空专家抽取最多2行，与现有 NumPy `expert_forward` 比较；
使用现有混合误差阈值并报告相对 RMSE。该参考包含 BF16 中间舍入，不是融合算子逐位仿真，
不通过时停止性能采样。`--check-rows-per-expert` 可提高覆盖，设为0显式跳过数值参考（仍检查输出形状、类型和有限值），
跳过结果会标为 `explicitly_skipped`，不能视作精度已验证。Graph 模式还会在计时前检查 replay 与 eager 输出一致。

计时排除 Dispatch、Combine、路由权重乘法、输入量化、权重转换和 CPU 校验。
这里得到的是单卡计算链基线，不是 EP16/32/64 张卡同时运行的统计；与已有 Dispatch/Combine 数据相加可作分段估算，
实际端到端 MoE 耗时仍需用一致的数据、执行模式和计时范围另行测量。

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
