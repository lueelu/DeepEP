> 当前为一包深度暂存/put流水，实现身份26、reference schema4；复用原lane，单lane立即提交。S表区分暂存经过、延后间隔、剩余等待及前包提交，不改packet所属核/顺序。详见 [ONE_PACKET_PIPELINE.md](../ll_rdma/ONE_PACKET_PIPELINE.md)；覆盖下方旧立即提交及schema2说明。

> 当前reference schema2新增包诊断，完整边界、覆盖规则及命令见 [PACKET_TRACE.md](../ll_rdma/PACKET_TRACE.md)。旧schema1结果请用旧提交解析，不混装。

# LL dispatch 参考式逐核打点与 Excel

> 当前仅保留 reference dispatch 设备打点，one_click 默认 reference；旧 stage/expert 插桩已删除，旧结果仍可离线解析。详见 [阶段与源码对照](../ll_rdma/STAGE_TIMELINE.md)。

主阶段与子阶段的先后、w0..w57对账公式、阶段和超过核总耗时的原因见
[阶段顺序与核总耗时对账](../ll_rdma/STAGE_TIMELINE.md)。

本入口只测 LL dispatch。阶段与用户提供的逐核 Excel 对齐：29 项耗时，
schema2原始记录每核640个uint64（前64word仍为原主阶段）；实际核数仍由平台与当前设备查询，保持原收发分工。
参考 ZIP 的 HT dispatch 使用每点一次 GetSystemCycle 和一次独占 GM scalar 写；
本实现采用这一记录方式，不移植 HT 算法、固定 64 核或参考实验的业务改动。

## 与旧计时器的区别

使用独立构建 `DEEPEP_LL_REFERENCE_PROFILE=ON`，同时固定
`DEEPEP_LL_STAGE_TIMER=OFF`。两个开关不能同时开启；旧阶段和专家计时在本构建中
完全关闭。普通 OFF 继续可用；旧 timer 仅保留历史结果离线解析，当前记录有独立格式、schema 和能力检查。
SDK、公开 dispatch 输入输出以及原有流水依赖保持；当前通信协议24保持SO整包发布。

- 每个固定边界只记录一次时间戳，不做旧计时器的 first/start/sum/count GM 读改写。
- packet ready 的空扫描不插点；每个有就绪包的批次只读三次时钟，计算轮询和消费累计。
  两个累计量和批次数保留在 Scalar，结束后写出；首次 ready 复用已有 poll-end 时间。
- schema2增加S核第二跨机包细分及提交/quiet边界；R核记录UB包顺序，不逐包读clock。
- R日志只用原lane分配后的剩余UB，不减少lane；本核业务结束后一次DMA导出。不增加业务PIPE_ALL；末尾诊断仍有同步和缓存写回成本。
- 主边界记录仍有指令、GM store、寄存器和最终写回成本。低开销设计不等于零扰动，
  是否比旧计时器更快需要目标机完整 task 对照，不能仅比较不含写回的核内 cycle。

## 两机采样

各机加载已有 CANN 环境，在相同绝对路径的工程根目录运行；使用同一提交和参数。
每机16卡，EP32。匹配 SDK 复用，本次只需构建匹配的完整 wheel。
依赖准备只允许 SHMEM：`git submodule update --init --recursive --depth 1 -- third-party/shmem`。

```bash
ndr_reference() {
    git pull --ff-only || return
    bash tests/ll_rdma/one_click.sh "$1" 192.0.2.10 ndr-reference-001 \
        --mode reference --tokens 1 --dtype fp8 --reference-iters 1
}
# 节点0使用0，节点1使用1。替换示例IP为节点0真实IPv4；重跑换run名。
if ndr_reference 0; then
    printf '%s\n' 'LL dispatch reference capture completed'
else
    printf '%s\n' 'Capture failed; inspect project logs'
fi
```

上述例子为 B1/M256/H3584/K8/E384、预量化 FP8、只测 dispatch；
不是参考文件中的 EP64 实验。去掉 `--tokens 1 --dtype fp8` 将按入口默认 B8、
BF16和预量化FP8分别运行，不把不同dtype/shape合成一个表。
`--reference-iters` 控制独立诊断轮数，默认1；10次预热和100次稳定调用后再采集，
批次开始只做一次all_reduce；预热并同步后固定100次Exp，再连续稳定调用和capture，算子之间不插入采样all_reduce/Exp。当前采样协议为initial-reduce-exp100-batch-v3，
不把参考 ZIP 或历史表格的 50 轮/尾30当作本入口的采样配置。

所有节点都须参与运行；脚本不通过 SSH 启动另一机。每机解析本机16个 global ranks，
第二台保留 rank16..31，不重编号。完整 EP32 表需要汇集同一run的两机原始样本，
再使用 `--require-all-ranks`；缺失远端rank不补零。

## 结果和离线导出

run位于 `results/ll-ndr/ndr-reference-001-reference/`。
one_click自动分析每个case/route，统一生成`table-analysis/stage-times.xlsx`多sheet主文件，
各case的CSV、JSON和中间Excel保留子目录。只回传导出根的`return_report.md`。
下方独立解析命令默认仍使用`reference-analysis/`，也在根目录输出合并的`stage-times.xlsx`。
`stages.csv`为所选轮逐核宽表，原全部轮次长表移至`stages_long.csv`；
`stages_mean.csv`单独保存多轮均值。详见[统一表格导出](../ll_rdma/TABLE_EXPORT.md)。
已有新格式样本重新导出无需重编或重采；旧 schema3 数据不能转换为这些缺失的参考边界。

```bash
python tests/ll_reference_profile/analyze.py \
    --input results/ll-ndr/ndr-reference-001-reference \
    --output-dir results/ll-ndr/ndr-reference-001-reference/export-002 \
    --iteration last
```

Excel 的结构对应参考表：

| 工作表 | 内容 |
|---|---|
| 采样汇总 | 全部诊断轮的逐rank最长单核跨度统计、实际配置、分析范围；无OFF数据不伪造性能对照 |
| 逐核耗时 | 指定单轮的所有rank/核，29项耗时、任务数、ready批次数；默认末轮，非多轮均值 |
| 阶段说明 | 同核word换算、频率、主阶段/细分包含关系和当前NDR边界 |
| 原始Cycle | 选中单轮的前64word主阶段；扩展原始字段见包原始记录页与JSON |

`--iteration N` 使用从0开始的诊断轮编号。所有轮原始JSON保留目标机，汇总仍统计全部轮次。
Excel/原始样本可能超过回传上限，留在目标机查看；只回传生成的紧凑报告，
每份实际UTF-8字节不超过60000，整个run提供一个回传文件，不要求上传所有xlsx和日志。

## 阶段边界和解读

主阶段依次覆盖入口配置、Init、magic、计数Init、收发Init、SendCnt；
发送核执行 Sender，接收核执行 RecvCnt、CalcCumsum、CompleteCumsum、既有计数drain、Receiver；
最后所有核执行 Counts.Finish。细分包含计数ready、cumsum跨核等待、prefix ready、
接收前缀/任务/缓冲准备、每批轮询/消费累计、接收清理与最终drain，以及发送路由准备/提交/收尾。

与参考阶段一一对应，但实际NDR业务依赖按当前源码解释：

- `w39` 紧接 SendWithDedup/SendWithoutDedup 返回；`w40-w39` 包含现有输入复用退休、
  尾 RDMA batch 的 消息完成quiet、lane回收及source metadata完成。不能把这些业务成本排除。
- packed-count ready 后的dense记录加载属于RecvCnt剩余部分；clear-status位于当前
  CalcCumsum中GetCumSum之后，不虚称全部清理都在RecvCnt里。
- `w21-w0` 是核入口至 Counts.Finish 已有最终PIPE_ALL/CTRL恢复，不含末尾诊断发布；
  `w57-w0` 是诊断发布前的跨度。两者都不是profiler完整task或Host API延迟。
- `w33/w34` 是每批区间累计，`w41` 是有ready包的批次数，不是包数或空扫描次数。
- 只有同一核内时间戳作差。默认1000 ticks/µs；无频率时保留ticks。
  原始大整数保留精度，先用Python整数做差，不让Excel大时间戳相减丢精度。
- 未执行阶段留空，已执行但耗时为0保留0。父子和并行阶段不能相加，
  也不能把不同核各阶段的最大值相加当作kernel时延。

源码入口为 `csrc/kernels/dispatch/low_latency.cpp`、`dispatch_server_dedup_kernel.h` 和
`device/dispatch_server_dedup_{counts,send,recv}.h`；记录契约在
`common/ll_reference_profile_layout.h`，纯CPU解析在 `deep_ep/utils/ll_reference_profile_parse.py`。

## 验证范围

CPU检查覆盖记录完整性、阶段换算、空任务/真零、动态核数、大整数精度、构建互斥与Excel导出。
当前365项CPU回归通过、2项无Torch跳过；REF/OFF/旧ON完整wheel在27编译及纯Host导入通过，
全程未使用NPU，详见[验证记录](COMPILE_VALIDATION_20261026.md)。
设备正确性、实际性能和打点扰动必须在目标机测量。
不得把CPU模拟样本或表格预览作为实测结果。
