# 一个run、一个Excel、逐核宽表

各阶段的先后、父子包含关系、同核可加总清单及与kernel的对账公式见
[阶段顺序与核总耗时对账](STAGE_TIMELINE.md)。不要将宽表所有耗时列直接SUM。

`one_click.sh`的timer/reference模式在测试成功后自动离线解析，主文件为
`results/ll-ndr/RUN_NAME-MODE/table-analysis/stage-times.xlsx`。
同一次run的各case/route全部进入这个Excel，以`C001_`、`C002_`区分sheet，报告索引记录对应配置。
每机仍只分析本机完整16rank组，global rank不重编号。不同case不混算均值。

布局沿用参考表的深蓝表头、隔行底色和筛选。Excel逐核表删除实验组、服务器、本地卡、采集轮次列；
reference左侧依次为Rank、核号、收发角色、任务数、核总耗时，这五列及表头冻结。
timer及相关均值/调用次数/明细sheet同步精简，并冻结已有身份字段、任务数和kernel列；
timer未记录任务数时不补造该列。原始Cycle冻结保留的身份字段，不将原始word伪称核总耗时。
run/case和选轮信息仍在采样汇总中，原始JSON及CSV保留完整字段供追溯，多轮长表Excel省略轮次后应回查CSV。
合并stage-times.xlsx继承同样的布局。reference保留实际33阶段及中文标签；
timer使用其实际累计阶段，不用缺失的参考时点拼出29阶段。默认模式为reference；旧timer仅用于历史结果离线解析。
perf模式没有逐核诊断数据，不生成虚构的阶段表。没有OFF对照数据时使用“采样汇总”，不宣称性能对照。

## 已有结果只重导出

更新本分支后，在工程目录运行；路径含空格时加引号。指定新的输出目录，已有目录不会覆盖。
此路径只用Python标准库，不加载CANN、Torch或NPU，不重编SDK/wheel，也不重新采样。
`--analyze-only`须是第一个参数。

```bash
bash tests/ll_rdma/one_click.sh --analyze-only \
  results/ll-ndr/你的运行名-timer \
  --output-dir results/ll-ndr/你的运行名-timer/tables-002
```

reference运行将路径后缀改为`-reference`。也可直接调用
`python tests/ll_rdma/analyze.py RUN_DIR --output-dir NEW_DIR`。
默认`--iteration last`选择末轮；`--iteration N`选从0开始的诊断轮，不改变均值统计范围。
频率来自样本，必要时显式`--frequency-hz`覆盖；无频率时保留ticks，不擅自按1000换算。
汇集同run各机原始JSON后，加`--require-all-ranks`可要求完整全域。

## 表格与明细

| 文件或sheet | 内容 |
|---|---|
| stage-times.xlsx | 全部case的统一多sheet入口 |
| 逐核耗时 / stages.csv | 所选单轮，每个operation/rank/core一行，阶段各占一列 |
| 逐核均值 / stages_mean.csv | 全部capture的有效记录均值，与单轮分开 |
| 阶段调用次数 / stage_calls.csv | 所选轮各阶段calls，和宽表同样的行键 |
| 阶段说明 | 计时口径、频率、有效性及阶段包含关系 |
| 原始Cycle | 所选轮完整原始word；超过15位的整数以文本保存 |
| 阶段统计 / stages_long.csv（timer） | 原长表的mean/min/max/P50/P90/P99及samples |
| 全部轮次明细 / stages_long.csv（reference） | 全部轮次的阶段长表 |
| 通信计数、接收明细、专家统计、专家逐核 | timer中实际存在的相应明细；无专家数据不伪造 |

timer同时包含dispatch/combine时，逐核耗时和均值分别按算子分sheet。

`stages.csv`/`stages_mean.csv` 在阶段列后固定追加 5 个原始计数列：`w37 receive_task_count`、`w41 receive_probe_batch_count`、`w43 send_route_count`、`w48 receive_first_hop_tokens`、`w49 receive_relay_members`。只有置了有效位的计数才有值，未执行或旧 schema 样本留空（均值页这 5 列固定为空）。
CSV位于`case_001/`等目录；中间`core_stage_timings.xlsx`保留，但只需打开根目录的`stage-times.xlsx`。
空白表示无记录，0表示实际记录为零。父子累计阶段不可相加，异步提交不等于完成，不跨rank相减时钟。
原始JSON不修改；多轮原始数据保留目标机。一个run的`return_report.md`不超过60000 UTF-8字节，
包含索引及可容纳的诊断摘要，省略处明确标注；完整表留目标机，回传只需该文件。

旧脚本若依赖长表`stages.csv`，请改读`stages_long.csv`。
独立`ll_stage_timer/analyze.py`和`ll_reference_profile/analyze.py`仍保留原输出根目录，
同时新增根目录`stage-times.xlsx`；one_click统一入口使用`table-analysis/`。

本次只更新离线解析及入口，没有改变kernel、SDK、打点布局、协议或采样顺序。
