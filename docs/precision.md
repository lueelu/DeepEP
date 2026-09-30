<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 精度、Golden 与比较方法

精度检查分为通信搬运、浮点归约和参考计算三个层次。本页说明仓库实际比较方法；CPU 参考或测试适配器通过，不等于真实设备算子通过。

## 各入口的判据

| 入口或输出 | 比较方式 |
| --- | --- |
| HT dispatch 的有效路由、计数、payload/scales 和 weights | 与独立参考逐项或按位比较，保留重复槽位 |
| HT combine 系统测试 | 使用有界整数构造专家输出，精确比较完整归约结果；独立回传的权重按位比较 |
| LL dispatch 系统测试 | 检查实际 LL 路径、有效行、payload/scales、返回字段及行数约定 |
| LL combine 系统测试 | BF16/FP8 场景使用 `rtol=0.01, atol=0.03125`；FP16 场景使用 `rtol=0.003, atol=0.004` |
| CPU 参考与 V1 测试适配器 | 独立 FP64 Golden、固定 dtype 路径模拟与双标杆比较 |

HT 的精确整数用例不代表任意浮点分布均无误差；LL ST 也没有把所有输出交给通用双标杆比较器。禁止把 `combine(dispatch(x)) == x` 作为通用条件：普通布局按目标 rank 去重，expanded 按有效槽位展开，归约重数取决于路由。

HT 模型侧应用 gate，LL combine 内部应用 gate；均只应用一次。具体返回布局和有效区域见 [Elastic 接口](elastic.md)。padding、容量尾部和未定义槽位不进入数值比较。

## 双标杆指标

实现位于 [tests/utils/precision.py](../tests/utils/precision.py)。设实际输出为 a、FP64 Golden 为 g，独立 dtype 路径结果为 b。第二标杆来自 CPU 模拟时，报告明确标记其来源，不称为设备实采结果。

按 dtype 的小值阈值 τ 将元素分为：普通值域 `S={i: abs(g_i)≥τ}`，小值域 `Q={i: abs(g_i)<τ}`。仅在 S 上分别计算 a、b 的三项误差：

```text
MARE = max_i(abs(a_i - g_i) / (abs(g_i) + 1e-7))
MERE = mean_i(abs(a_i - g_i) / (abs(g_i) + 1e-7))
RMSE = sqrt(mean_i((a_i - g_i)^2))
i ∈ S
```

S 为空时三项均为 0。每项误差比为 `metric(a,g) / max(metric(b,g), err)`；L2 对 MARE/MERE/RMSE 的比值上限分别为 2、1.2、1.2，并非原始误差上限。

| dtype | 小值阈值 τ | 小值绝对误差阈值 err |
| --- | ---: | ---: |
| BF16 | `2^-8` | `2^-16` |
| FP8 E4M3FN | `2^-4` | `2^-6` |

Q 中统计 `abs(a_i-g_i)>err` 的元素数 ErrorCount。`ErrorCount(a)/max(ErrorCount(b),1)` 不超过 2 才通过。比较器拒绝 NaN/Inf，不提供非有限值类别或符号匹配；不能将有限输入测试外推到这些情况。

`bootstrap_median_recheck` 对至少 200 个误差比样本执行复检，默认 Bootstrap 2000 次，计算中位数的 95% 置信区间；样本不足或 `CI_lower>1.0` 时失败。它是独立工具，不会自动替代 ST 的失败结论，也不用于覆盖路由、shape、计数或数据破坏错误。

## 参考数据与矩阵

[tests/utils/matrix.py](../tests/utils/matrix.py) 与路由参考包含小规模 EP、不同 hidden、零/不等长 token、重复与无效路由、均匀/正态分布、小值和离群值。它们用于检查参考函数和比较器，不是当前产品的支持矩阵；其中 EP1、H256 或 alignment128 等 case 不能直接用于当前 HT payload。

参考语义固定到 DeepSeek DeepEP `a56d6156febcd9976e55adc85b5155bfac9f28f8`，来源见[第三方说明](../third-party/README.md)。BF16/FP8 模拟明确标出 dtype 舍入和 FP8 每 128 列缩放；实际 LL 的 INT32 打包 scales 另由 LL ST 检查，不与该模拟格式混用。

参考闭环使用 `f_e(x)=(e+1)/8*x+(e-3)/64`。专家乘加、gate 和跨 rank 累加按各适配器约定的 BF16/FP32 路径执行，FP64 Golden 直接从原始输入生成。故障注入检查漏贡献、重复乘权、错误顺序、shape 和非有限值等问题。

## Golden 维护

```bash
python -m tests.golden.generate
python -m pytest tests/test_precision.py -q
```

生成器更新 `tests/golden/deepep_reference_v2.json`，包含参考版本、case 配置、路由与布局摘要、FP64 答案摘要、dtype 模拟摘要和误差指标。输入由确定性公式重建，以规范 JSON 的 SHA-256 锁定；MERE/RMSE 使用 `math.fsum` 保持跨 Python 版本的归约一致性。

修改生成公式、路由、舍入或归并规则时，同时审查生成器与 Golden 差异，不能只替换摘要消除测试失败。设备结果单独保存，不覆盖 CPU Golden。记录 case、rank、dtype、shape、源码、环境与比较方法；所有参与 rank 均通过才记为对应配置通过。
