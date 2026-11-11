---
name: test-megamoe-train-atk
description: 在容器默认 Python 中准备 ATK 与内部 CV 双标杆比较器，运行 DeepEP MegaMoE BF16 训练算子的现有精度用例或套件，并核对报告。用于 ATK 环境安装、用例复测和批量精度验证；不负责内核构建、性能测试或放宽精度门限。
---

# MegaMoE Train ATK 精度测试

使用仓库单入口执行完整前后向，由真实 ATK 私有 `cv_fused_double_benchmark` 比较器给出精度判定。

## 依据与执行范围

开始前读取当前分支的以下文件，命令均从仓库根目录执行：

- [训练模块 README](../../../../experiments/megamoe_train/README.md)：运行环境、测试模式与报告。
- [测试入口说明](../references/test-guide.md)：运行方式、输出结构与已知问题。
- [测试规范](../references/test-spec.md)及[用例清单](../../../../experiments/megamoe_train/tests/bf16/cases.json)：覆盖理由、shape、数据分布和 seed。
- [测试入口](../../../../experiments/megamoe_train/tests/bf16/test_mega_moe.py)及[ATK 适配器](../../../../experiments/megamoe_train/tests/bf16/atk_compare.py)：实际参数、比较器身份和门限。

不依赖 `moe-op-benchmark`，不另建测试框架，不调用 `atk task` 或 ATK 用例生成器。当前 ATK 只比较仓库生成的实际值、标杆和真值，不自动泛化 shape 或输入数值。

只有用户明确指定并允许的主机、容器、物理卡和测试范围才能用于设备运行。复用当前明确授权，不重复询问已有信息；缺少卡号时不自行挑卡。运行前检查指定卡的进程、显存和利用率，忙卡不启动，不换卡、不停止其他任务。仅安装环境、列用例或询问运行方法不意味着获准执行 NPU 测试。

## 1. 对齐 Python、算子和 ATK 环境

默认使用容器默认 Python，除非用户明确选择其他解释器。不自行创建或激活 venv/conda，不将个人容器中的环境脚本当作通用前置步骤。

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip --version
python -m pip show ascend-deepep ATK atk-private-precision
```

加载与已安装 wheel 匹配的 CANN 环境。SHMEM 运行库随完整 wheel 内附，由根加载器校验并加载；测试时不需要选择外部 SDK：

```bash
source /path/to/cann/set_env.sh
```

`ascend-deepep`、ATK 和内部插件必须安装在同一个 Python 中。另一个虚拟环境的 `pip list` 或独立 `atk` 命令可用，不能证明当前 Python 可导入 ATK。

- 缺少训练 wheel 或刚修改 Host/kernel/绑定：按[构建 skill](../build-megamoe-train/SKILL.md)构建安装，再回来测试。仅更新测试文件不需要重编原生算子。
- 缺少 ATK 或内部插件：按[ATK 安装说明](references/atk-install.md)给当前 Python 安装，包含首次安装、复用 wheel 和离线方式。
- 已安装相容版本且加载检查通过：复用，不为了每次测试重装依赖。

按构建 skill 的加载检查，在源码目录外确认根扩展和训练扩展可加载。记录测试源码提交及 dirty 状态、当前 Python、框架版本、实际训练原生库路径和 SHA256；不能只凭包版本号认定已加载本次修改后的内核。

在仓库根目录运行下面的检查。它使用实际适配器校验自动加载状态、私有提供者和比较器身份，不分配 NPU Tensor：

```bash
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path("experiments/megamoe_train/tests/bf16").resolve()))
from atk_compare import require_atk

_, identity = require_atk()
print("Python:", sys.executable)
print(identity)
PY
```

该检查失败时先处理安装或兼容性问题；不通过手动注册、切换开源比较器、删掉 ATK 检查或跳过精度来得到 PASS。

## 2. 选择用例与 EP

物理设备列表通过位置参数传入，EP 等于列表长度；worker 使用映射后的逻辑 local rank。常用 EP2/4/8，实际结论仅覆盖本轮运行规模。多机需要单独确认网络和启动方式，不把单机通过扩大为多机通过。

当前公开清单包含 12 条典型尺寸：Flash、Pro、GLM 各取每 rank M=512、2048、4096、8192，统一 balanced 路由、BF16 路由权重、BF16 保存和固定 seed。`priority`、`model` 选择这 12 条；`dtype` 另有 24 条 BF16/FP32 路由权重、保存模式和 clipping 组合，`gather` 有 6 条 FP32 源窗口边界，`all` 选择全部 42 条。运行前用当前入口核对实际配置：

```bash
: "${MEGAMOE_TEST_DEVICES:?填写本轮用户指定的物理设备列表}"
python experiments/megamoe_train/tests/bf16/test_mega_moe.py "$MEGAMOE_TEST_DEVICES" \
  --suite priority --list-cases
```

`--list-cases` 只读取配置，不导入 Torch/ATK 或执行设备计算。用户已指定 case 或范围时按该范围运行；需要较小的典型档位时可选 `flash_m512`、`pro_m512`、`glm_m512`，不强制重复已完成用例。典型矩阵的完整 CPU FP64 参考可能占用较多主机内存，运行前核对容器内存/CPU 限额和实际可用资源。

## 3. 执行精度测试

先确认指定设备空闲和 bootstrap 端口可用。`MEGAMOE_TEST_ENDPOINT` 填入本轮实际未占用的 `tcp://IP:port`；Gloo 使用该端口加 2，也须可用。不要省略物理设备列表，让入口回退到 0、1。

示例按所需套件修改 `suite`。每轮使用新的报告子目录，这是测试入口的要求；固定父目录 `atk-runs/` 可以复用：

```bash
: "${MEGAMOE_TEST_DEVICES:?填写本轮用户指定的物理设备列表}"
: "${MEGAMOE_TEST_ENDPOINT:?填写本轮未占用的 SHMEM TCP 端点}"
suite=priority
entry=experiments/megamoe_train/tests/bf16/test_mega_moe.py
mkdir -p "$PWD/atk-runs"
run_dir="$PWD/atk-runs/$(date +%Y%m%d-%H%M%S)-$$"

python -u "$entry" "$MEGAMOE_TEST_DEVICES" \
  --suite "$suite" --atk --skip performance \
  --atk-ratios 5 1.5 1.5 \
  --shmem-ip-port "$MEGAMOE_TEST_ENDPOINT" \
  --output "$run_dir"
```

指定用例示例：仅选择 `priority` 中的 `flash_m512`，沿用上面已确认的环境和变量；这是可选示例，不是每次测试的必跑项：

```bash
python -u "$entry" "$MEGAMOE_TEST_DEVICES" \
  --suite priority --case-id flash_m512 --atk --skip performance \
  --atk-ratios 5 1.5 1.5 \
  --shmem-ip-port "$MEGAMOE_TEST_ENDPOINT" \
  --output "${run_dir}-flash_m512"
```

多个 `--case-id` 可重复传入，且必须属于所选套件。suite 的 shape、保存模式、clipping、路由和 seed 来自清单，不再用单 case 参数覆盖它们。

当前精度默认启用 ATK，保留显式 `--atk` 便于审阅命令；`--skip performance` 只测精度。默认不加 skip 会先做精度、再做性能，不属于本 skill 范围；`--skip precision` 不是精度验证。

套件默认复用 worker、进程组和算子实例，逐 case 重新生成输入、参考和梯度。`--suite all` 在一个作业内运行当前 42 条；仅需原 12 条模型用例时选 `priority` 或 `model`。只在需要隔离复现或比较时使用 `--isolate-cases`。套件由一次普通 `python` 启动，不放到 `torchrun` 下重复启动。

执行时保留真实 Python 退出码。套件会实时输出并保存 `suite.log` 和 case 日志，无需另造输出框架；若使用 `tee`，开启 pipefail 并保留 Python 的退出码，不能将 tee 成功当作测试通过。运行期间不更新源码或重新安装 wheel。

设备或集合通信异常时停止后续 case，保留日志，完整重启自己的作业；不只重启部分 rank 或批量清理其他人的进程。数值 FAIL 可以由现有套件继续收集，其余 case 不因此被标为通过。

## 4. 按报告判定与交付

ATK 对每个 rank 的 `y`、`dx`、`dtopk_weights`、`dw1`、`dw2` 分别比较：

- actual：融合 MegaMoE 的结果。
- benchmark：`fp32-boundaries-v3-topk-dtype` NPU BigOp，BF16 GMM 输入输出与 FP32 激活、导数、dp 及有序 Y/dX 归并；p/dp 保持输入 BF16 或 FP32 类型。
- golden：独立 Torch CPU FP64 真值。

显式最大/平均相对误差及 RMSE 比例门限为 5/1.5/1.5，`number_count_radio=2`。ATK 是唯一双标杆；另外仍须通过 `megamoe_contract` 混合容差与精确 counts。BF16 保存时，BigOp 与混合容差共用同一份小算子结果。逐 rank、逐输出判定并使用上述私有双标杆比较器，不为得到 PASS 修改真值、融合结果或门限。参考数值规则升级须记录新版本，保留旧报告。

只修改 BigOp 参考而融合算子与 CPU FP64 真值均未改变时，可复用已有 actual/golden 缓存，仅重算 BigOp 后调用同一 ATK 比较器。核对输入哈希、case、dtype/布局、融合产物及真值来源；新输出目录记录缓存来源和参考版本。缓存只覆盖部分输出时，报告按实际覆盖记录，不记为完整套件通过。

依次检查：

1. 作业退出码和 `suite_summary.json`：是否完整结束，有无 ERROR、NOT_RUN 或未收尾状态。直接 launcher 的 2 表示数值失败，环境/运行异常也会返回非零；清理失败不能算成功。
2. 每个 case 的 `summary.json` 和全部 `rank_<rank>.json`：所有预期 rank 和五项输出都存在，`precision_run=true`，私有提供者正确，ATK、混合容差和 counts 同时通过。
3. 失败项的 rank、输出名、ATK 原始指标、门限和对应配置；不要只看日志最后一条 PASS 或一个汇总比例。

缺少 `atk`、内部插件或原生库属于环境/加载错误，尚不能称为算子精度 FAIL。报告缺失、`NOT_RUN`、作业仍在运行或比较器异常也不能计作 PASS。`--list-cases` 和环境加载成功不构成数值验收。

失败的 ATK actual/benchmark/golden 张量保留在 case 的 `atk_rank_<rank>/` 下，供后续定位。小值域错误数量比例描述错误元素数量，不是梯度数值放大倍数。保留原始失败报告，不覆盖历史记录或按新判据改写旧结论。

交付实际源码/原生库身份、Python 和 ATK 版本、命令、设备及 EP、套件/case 范围、退出码、PASS/FAIL/ERROR/NOT_RUN 数量、各失败输出和报告路径。没有执行的用例明确标为未测；不能将一组通过扩大为其他保存模式、EP 或完整套件通过。

典型模型清单仍仅使用 BF16 保存；dtype/gather 专项的具体配置和验收边界见[测试规范](../references/test-spec.md)，不扩大为其他路由、数据分布或设备规模已通过。测试规范只记录用例与判据，不累积设备数据；清单存在、环境加载或静态检查不证明新二进制已通过。
