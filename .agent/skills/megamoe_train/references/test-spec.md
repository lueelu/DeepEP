# MegaMoE BF16 训练测试规范

当前 [cases.json](../../../../experiments/megamoe_train/tests/bf16/cases.json) 包含 12 条典型模型尺寸用例：Flash、Pro、GLM 各取
每 rank M=512、2048、4096、8192；另有 24 条路由权重类型/保存模式用例及 6 条 FP32
Gather 边界用例。
清单与设备验收分开记录；设备精度、性能结果对应具体源码提交与构建产物。

## 1. 范围与模型参数

范围为 Ascend950、CANN 9.2、BF16 训练公共接口的前向、四项反向梯度及独立计时。
典型尺寸采用 Flash、Pro、GLM 配置；使用合成输入和缩减专家数，
不加载真实 checkpoint，不复现模型 router、共享专家或量化计算。
典型模型清单的 12 条用例统一使用 BF16 保存；另有 dtype 专项覆盖 `bf16/none/mxfp8/auto`，用于检查类型与保存模式的组合。`mxfp8` 与 `auto` 仍按实验性质对待，清单存在和诊断对照不构成设备验收（见第 5 节）。

| 用例前缀 | hidden H | SwiGLU F / GMM1 N | TopK | 每 rank 专家数 | clipping | 路由权重缩放 | 条数 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `flash_` | 5120 | 2304 / 4608 | 6 | 4 | 10 | 1.5 | 4 |
| `pro_` | 7168 | 3072 / 6144 | 6 | 4 | 10 | 2.5 | 4 |
| `glm_` | 6144 | 2048 / 4096 | 8 | 8 | 0，关闭 | 2.5 | 4 |

| 后缀 | 每 rank M | 选取理由 | Flash/Pro 每专家行数 | GLM 每专家行数 |
| --- | ---: | --- | ---: | ---: |
| `m512` | 512 | 最小典型 token 档位，覆盖多块计算 | 768 | 512 |
| `m2048` | 2048 | 2K token 的完整前后向验证 | 3072 | 2048 |
| `m4096` | 4096 | 4K token，增加权重梯度归约长度 | 6144 | 4096 |
| `m8192` | 8192 | 8K token，覆盖持续多块执行与较大参考数据 | 12288 | 8192 |

前缀与后缀组合即 12 个唯一 ID，例如 `flash_m512`、`pro_m4096`、`glm_m8192`。
这 12 条使用 balanced 路由、random 数据、BF16 路由权重、`save_gmm1_output=bf16` 和 seed=20260825。
均衡路由下每专家行数为 M*TopK/E_local，每 rank 接收 M*TopK 条路由。

M 是进入算子的每 rank token 总数，不能直接等同于模型序列长度 S。
没有序列切分时 M=B*S；仅设置 EP 不会自动将 M 除以卡数。
EP 来自设备列表，本组按 EP2/4/8 检查配置合法性，设备资格需分别验证。
总专家数为 EP*E_local，不代表完整模型固定的 384/384/256 个专家。

## 2. 输入生成与反向含义

单入口是 [test_mega_moe.py](../../../../experiments/megamoe_train/tests/bf16/test_mega_moe.py)。每个 source rank 和输入流使用
独立、固定种子的 Torch CPU Generator；所有参考与融合实现共用同一份 CPU 输入快照。
x/dy 从 [-0.1,0.1] 均匀采样，w1/w2 从 [-1,1] 均匀采样；路由权重由
FP32 softmax(logits) 乘模型缩放系数后按以下选择定型。路由 ID 固定、合法且每 token 不重复。

| 输入选择 | CLI | 用途 |
| --- | --- | --- |
| BF16，默认 | `--topk-weight-dtype bf16` | BF16 路由权重与梯度 |
| FP32，BF16 有效位 | `--topk-weight-dtype fp32 --topk-weight-values bf16-rounded` | 与同 seed 的 BF16 p.float() 逐位相同；其余输入完全一致 |
| FP32，完整有效位 | `--topk-weight-dtype fp32` | 默认 full-precision，保留 softmax 的额外有效位，检测隐式降精 |

`--topk-weight-values full-precision` 只允许 FP32。snapshot、NPU BigOp、契约参考和融合
输入保留所选 p 类型，CPU 真值直接提升到 FP64。dtopk_weights 与 p 类型相同；
y、dx、dw1、dw2 保持 BF16。每次调用所有 EP rank 必须使用相同 p 类型。

本测试将固定随机 dy 作为上游梯度，执行 y.backward(dy)，数学上对应测试损失
sum(y*dy)。随机的是反向输入，dx、dtopk_weights、dw1、dw2 都由实际反向计算得到。
本组不证明真实模型 loss、router 参数梯度或训练收敛。

## 3. 参考与验收

- [cpu_reference.py](../../../../experiments/megamoe_train/tests/bf16/cpu_reference.py)：Torch CPU FP64 前向和 autograd 真值，不插入 BF16 中间舍入。
- [npu_baseline.py](../../../../experiments/megamoe_train/tests/bf16/npu_baseline.py)：`fp32-boundaries-v3-topk-dtype` NPU BigOp 作为 ATK 误差基线，
  使用 FP32 激活、导数、dp 乘积/归约和按 TopK slot 顺序的 FP32 Y/dX 归并；
  Y/dX 最终写 BF16，dp 写输入 p 的 BF16/FP32 类型，矩阵边界仍为 BF16。
  两个参考角色复用公共数值实现；BF16 保存模式下，ATK 和混合容差直接复用同一份小算子结果。
  保存模式仅决定 Gate/Up 的保存、重算或解码，不选择不同的舍入规则。
- [atk_compare.py](../../../../experiments/megamoe_train/tests/bf16/atk_compare.py)：使用 ATK 26.9.8/private 0.1.0 的
  cv_fused_double_benchmark，逐 rank、逐输出执行，不混合不同输出。
- [precision_metrics.py](../../../../experiments/megamoe_train/tests/bf16/precision_metrics.py)：精确 counts 和现有混合容差报告。

每条用例完整检查 y、dx、dtopk_weights、dw1、dw2，不抽样 token、不跳反向。
ATK 最大/平均/RMSE 误差比例门限为 5/1.5/1.5，小值域数量比例为 2。
BF16 辅助混合容差为 atol=rtol=2^-7、匹配比例>=99%，
最大绝对误差<=max(0.1,32*ULP)。当前测试要求混合容差、ATK 和 counts 均通过；
辅助参考不是独立 CPU 真值，混合容差通过不能覆盖 ATK 失败。
FP32 dtopk_weights 沿用已有 FP32 混合门禁：atol=rtol=2^-13、匹配比例>=99%，
最大绝对误差<=max(0.01,32*ULP)。ATK actual 和 benchmark 保持同一输出 dtype，
golden 保持 FP64。
参考实现派生自 `fft__/ascend_deepep_hkq`（`568ac7c`），只取测试侧代码。

## 4. 执行方式

辅助参考的 CPU 回归使用 `python -m unittest discover -s experiments/megamoe_train/tests/bf16 -p test_reference_contract.py -v`。
数值项需要 CPU Torch，检查公共 FP32 H/dH、dp、BF16 输出边界、clipping、按 slot 归并、保存复用，
以及 FP32 p 的有效位、dP dtype、同输入变体、FP64 真值和 MXFP8 诊断重放；不启动 NPU 或 ATK。
CLI 默认值、合法参数及用例覆盖检查不依赖 Torch；没有 Torch 时数值项显示 SKIP，不能计为通过。

`priority`、`model` 选择这 12 条；`dtype` 选择 24 条 H256/F512/M129 小用例，
覆盖 BF16、BF16-rounded FP32、full-precision FP32 三类 p，逐一组合
bf16/none/mxfp8/auto 与 clipping 0/0.1。`gather` 选择六条 FP32 full-precision
用例：M8191/8192/8193、TopK4 或 TopK8，对应 R=M*TopK 为
32764/32768/32772 和 65528/65536/65544，覆盖 32768 元素窗口的边界和第三窗口尾块。
exact 数学边界由布局 CPU 测试覆盖；这些 NPU shape 不声称逐整数扫描窗口边界。
`all` 选择全部 42 条。
`--case-id` 可重复指定，仅选择清单内的 ID。套件参数由 JSON 决定，不被单例 shape 参数覆盖。
`--list-cases` 不导入 Torch/ATK，不启动设备：

```bash
python experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite priority --list-cases
```

安装目标分支构建的完整 wheel，并确认同一 Python 中已安装 ATK 及内部插件。
执行者确认设备获准且空闲，再运行精度；示例中 SHMEM 端口和端口+2 都需空闲：

```bash
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite priority --skip performance --atk-ratios 5 1.5 1.5 \
  --shmem-ip-port tcp://127.0.0.1:29862
```

`--skip precision` 仅运行有限值、计数检查和性能计时，不依赖 ATK；不加 --skip 则先精度后性能。
性能按源仓 `test/bigop_fused` 的口径给出三个同步 Event 区间（前向、反向、前向+反向），`--warmup`/`--loops` 各一组样本，窗口前后各同步一次，不构建也不读取设备 Timer；报告把两侧记作 `megamoe_bf16` 与 `baseline`。`--baseline on`（默认）另测移植的 dropless 小算子链（permute、dispatch、GMM1、带路由权重的 SwiGLU、GMM2、combine、unpermute），与算子共用同一份路由、权重和上游梯度，GEMM 数量对齐（`none` 重算 GMM1，其余复用保存的 Gate/Up），输出各口径 speedup；`--baseline off` 只测 MegaMoE BF16。三个口径同时给 MFU（RFK 模型的 GMM FLOPs ÷ 432 TFLOPS 峰值）与 roofline（源仓 410.4 TFLOPS 分母的 floor/measured/overhead），`floor ÷ measured` 即相对该降额上限的达成率，比同口径 MFU 高约 5%，因此不再单独存百分比字段；区间内含 alltoall/permute/激活，属算子级 GMM 利用率，不是模型级 MFU。baseline 只作信息对照，不参与 PASS/FAIL，无法建立 HCCL 子组或超过 16 GiB 打包路由预算时记 `SKIPPED` 及原因。
性能 baseline 与 ATK 的精度 BigOp 分开：保留移植源链的激活边界，不模拟 MXFP8 保存压缩；
p 的通信和 dP 类型随输入，FP32 p 乘权后的 Hp 转回 BF16 后再进入 GMM。
性能 baseline 的初始测试侧来源为 `ascend_deepep_hkq@f575fc1`。日志级由 `--log` 控制，
默认 `quiet`，排查阻塞时用 `rank0`（保留 rank 0 的阶段日志与 `[PERF] stage` 标记）。
默认跨用例复用 worker/runtime，输入和参考每次重新计算；`--isolate-cases` 使用独立执行器。
CPU FP64 参考和完整权重梯度可能占用较多主机内存，需要先核对实际容器资源。

类型与窗口专项将上例 `--suite priority` 分别替换为 `--suite dtype`、`--suite gather`。
M8192/TopK16 的 131072 路由大用例保留为手动单例，不加入 all 的默认范围；获准设备和资源充足时：

```bash
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --seq 8192 --hidden 256 --ffn 512 --topk 16 --expert-per-rank 8 \
  --topk-weight-dtype fp32 --skip performance --atk-ratios 5 1.5 1.5 \
  --shmem-ip-port tcp://127.0.0.1:29862
```

### 性能指标定义

`floor`、`MFU` 全部由 shape 和常量推导，不是测量值，读报告前先对齐口径：

```text
route_flops = M × topk × F × K            # 每 1 RFK；M 为每 rank token 数，K=hidden，F=ffn
flops        = RFK × route_flops
floor_ms     = flops / (410.4e12) × 1e3   # 410.4 = 432 峰值 × 0.95，源仓常量
MFU          = flops / measured / 432e12  # 裸峰值分母
```

| 口径 | RFK | 构成（每个路由行） |
| --- | --- | --- |
| forward | 6 | GMM1 = 4（`2·K·2F`）+ GMM2 = 2（`2·F·K`） |
| backward | 12 | dHp 2 + dW2 2 + dX 4 + dW1 4；`--save-gmm1-output none` 重算 GMM1 时为 16 |
| complete | 18 / 22 | forward + backward |

保存模式下 `floor` 三口径为 **6 : 12 : 18 = 1 : 2 : 3**；`none` 重算模式为 **6 : 16 : 22**，且与路由分布无关（只与 M/topk/F/K 有关）。三个前提同时成立才成立：恒定 410.4 TF；窗口 100% 为 GEMM（通信、permute、激活、launch 都被忽略，所以它是**下界**）；路由不丢行（行数恰为 `M×topk`，容量不足导致截断时 floor 偏高）。实测低于 floor 说明常量或 RFK 模型有误，不是算子超过硬件。

`--suite` 结束时 rank 0 打印汇总表，列为：`M/H/F/tk/E`（`E = EP × expert_per_rank`，均衡路由下每专家行数 `EP×M×tk/E = M×tk/E_local` 决定 ready 相关分支）、三口径中位时延、`speedup f/b/fb`（baseline ÷ megamoe_bf16）、`MFU% f/b/fb`、`floor f/b/fb`、`fb_overhead`（`complete` 的 measured − floor，即该窗口内的非 GEMM 时间）。三口径独立采样，`fb_overhead` 不等于 `fwd_overhead + bwd_overhead`，不要相加。

每次输出新目录，保留 suite_summary.json、suite.log、逐用例 summary.json 和逐 rank 指标。
ATK 失败保留 actual/benchmark/golden 张量。数值失败使整套失败；运行错误停止后续用例，
不得把 ERROR/NOT_RUN 当作 PASS。报告顶层的 `identity` 自动写入 `revision`、`wheel` 版本和
三个原生库的路径与 SHA256；`revision` 缺省为 `null` 时以 SHA256 作为二进制身份。仍需人工记录
dirty diff、环境、卡号、命令和 seed。

## 5. 未覆盖范围

dtype/gather 清单及 CPU 回归不等于设备验证。BF16/FP32 p 的完整保存模式、
窗口边界与手动大用例需在匹配 wheel 上逐项记录设备结果；BF16 路由权重用例的结果不能代替其他类型与窗口用例的验收。

本清单是算子级回归，不构成发布资格，也不替代其他路由与数据分布、完整专家数、多机与 EP
子组、并发及生命周期、多 microbatch、真实模型 router/共享专家或整网 loss 收敛的专项验收。

H7168 的 `ptrDX` 中间量存在已知精度差异：中间量与最终梯度的验收分离，最终梯度通过不代表该中间量逐元素达标。

精度与性能结果记录在对应源码提交、构建产物和测试配置的运行报告中。
