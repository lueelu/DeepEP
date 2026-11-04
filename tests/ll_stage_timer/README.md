# LL dispatch/combine阶段打点

> 当前仅保留 reference dispatch 设备打点，one_click 默认 reference；旧 stage/expert 插桩已删除，旧结果仍可离线解析。详见 [阶段与源码对照](../ll_rdma/STAGE_TIMELINE.md)。

逐阶段顺序、嵌套关系、同核可加总清单及核总耗时对账见
[阶段顺序与核总耗时对账](../ll_rdma/STAGE_TIMELINE.md)。父子阶段重复计时，不能整行求和。

## 2026-10-22：默认按本机产物分析

两机无需合并文件，各自在本机直接运行：

```bash
python tests/ll_stage_timer/analyze.py results/ll-timer/你的run目录 --trace-iteration 0
# 只看dispatch或combine可增加 --op dispatch / --op combine
```

默认按目录内实际global rank推断完整的16-rank主机组。EP32第一台rank0..15、
第二台rank16..31都可独立分析；不把global rank重新编号，world_size仍为32。
本机缺卡、轮次缺失、重复记录及配置混用仍拒绝。缺失远端rank不当作0值补入。
summary.json记录analysis_scope/analyzed_ranks/missing_remote_ranks，Markdown明确范围；
慢rank、rank偏斜及专家统计均只代表输入的本机rank。
`--latest`支持没有rank0 cases.json、但含本机sample文件的run。
原始样本已存在时，只更新脚本即可，无需重装wheel或重采。

若已合并全部节点，增加`--require-all-ranks`执行全域完整性校验。
本节覆盖后文历史“必须先收齐全部rank”的分析说明；采样/作业自身仍需全部rank共同运行。

维护分支：`kernel/ll-ndr-hns1825`。本功能保持纯AIV、实际核数查询和1:1收发分工，不改变逻辑组配置及EP准入条件。

## EP16通信量与有效带宽（schema 3）

现有 `--ll-stage-timer` 命令现在自动记录五类数据搬运字节和次数，无额外flag。全rank更新并重新安装匹配timer ON wheel、重新采样；此功能不要求重建已通过身份校验的SDK。EP16分析输出 `bandwidth.csv`，`summary.json` 增加逐轮及每rank汇总。带宽为跨卡首发与转发计数字节除以 `max_core_kernel` 时长代理，不是NIC线速；本卡计数不进入分子，读取计数是总计数的子集。旧schema1/2、NO_COMM试验或缺失频率不产生带宽。详细定义、重建命令和结果解读见 [BANDWIDTH.md](BANDWIDTH.md)。

当前NDR去重dispatch已加入源卡/发送核的发送错峰；适用于M>32，含原EP16、M256场景。实现范围、基线提交及目标A/B步骤见[发送错峰](SEND_STAGGER.md)。

## 每次运行自动生成瓶颈总结

采样命令保持不变。分析器从每次raw sample读取真实B/M/H/top-k/专家数/dtype等配置，支持修改hidden等shape后分别分析；不按固定包大小推算通信量。本次只更新离线分析脚本，已有schema1/2/3数据可直接重新分析，不需要为了生成文档重装wheel或重采；旧schema没有字节计数，带宽明确不可用。

```bash
# 分析整个run，自动发现各case和路由；在仓库根目录执行。
python tests/ll_stage_timer/analyze.py "results/$CASE_NAME/qwe" \
  --output-dir "results/$CASE_NAME/analyze-report"
# 也可以沿用单case输入：
# results/$CASE_NAME/qwe/case_00000/normal/ll_stage_timer
```

单case输出 `analyze-report/bottleneck_summary.md`。多个case/路由各自输出该文档，并在输出根生成 `REPORT_INDEX.md` 索引；不同shape不合并统计。使用新的输出目录，已有报告不会覆盖。`--latest`仍支持按run父目录选最新采样。

总结包含配置、逐rank最长单核kernel耗时统计、逐轮rank最大/中位比、慢rank的最长核及同核阶段占比、机内首发/转发/读取计数和有效带宽；采集了专家打点时附专家投递热点。机器可读依据在 `summary.json` 的 `diagnostics` 字段中，已有CSV/trace继续输出。

报告把实测现象和待验证线索分开：阶段占比在同rank/同轮/同核求比再统计，未执行阶段不补零，嵌套阶段不相加。最长核可能随轮次变化，它不等于已证明的因果关键路径；父阶段与子阶段同时出现不能看成百分比饼图。阶段占比30%和rank最大/中位比1.25只是明确标注的启发式提示阈值，不是硬件瓶颈判定标准。5轮数据会提示样本少。

observe MTE下发区间不等于传输完成，drain会改变重叠与等待位置。没有逐peer/逐包数据和物理拓扑，无法仅凭低带宽或高recv_wait证明多对一拥塞。非标准跳过通信数据不生成正常通信瓶颈提示；无频率保留ticks。总结不混用profiler task、Event、API时间，性能收益仍需timer OFF验证。

### recv_consume专项展开

`bottleneck_summary.md`自动展示最慢3个dispatch rank的**全部接收核**：consume累计时间、消费批次数、每次成本、每核跨卡转发字节/次数，并逐核完整列出recv_lane_wait、recv_gm_to_ub、recv_load_wait、recv_metadata、recv_output_wait、recv_output_write、recv_forward及准备/等待/清理阶段；不受原最长核/前8阶段筛选限制。

新增`recv_core_samples.csv`覆盖**全部rank、全部接收核、每轮、每阶段**，包括`present/duration/calls/per_call/consume_fraction`。无记录为不可用，不补0。每核每轮的跨卡转发计数只附在recv_consume行，避免按阶段重复计数；schema1/2无计数，本卡移交不混入跨卡转发。

`summary.json.receiver_summary`提供对应逐核汇总。`per_call.mean`先逐轮除calls再平均；`weighted_per_call`是总时间/总calls，两种权重分别保留。consume.calls是消费批次，forward.calls是转发区间，output_write可能每包多次，均不能当成通用包数。

子阶段占比使用同核同轮累计时间，外层包络必须包含子阶段才给出比例；这不能证明逐次区间可相加。现有首尾包络和累计值不能恢复完整并集，因此不计算“consume减子阶段之和”，也不把差额归为固定开销。比较转发量/批次数及每次成本用于定位，不等于证明链路或流水瓶颈。无需重采，Git更新后对原raw使用新的output-dir重新分析即可。

## 从克隆仓库到执行

以下步骤在**用户目标机**执行。每台参与节点都需准备同一分支、同一提交和匹配wheel；单机EP16，两机EP32，每机16个连续rank。原生LL不支持EP2/4/8。
前置环境：Linux、Python≥3.11、C++17编译器、CMake≥3.20、Make或Ninja，以及目标机器配套CANN、驱动、Torch/torch_npu。CANN 9.3.0按已确认支持准备，不猜测框架版本；requirements-build.txt只安装构建依赖。

### 1. 拉取指定分支

在计划存放工程的父目录执行；两分支建议使用独立目录：

```bash
git clone --branch kernel/ll-ndr-hns1825 --single-branch \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-ndr
cd DeepEP-LL-ndr
git submodule update --init --recursive --depth 1
git rev-parse HEAD
```

多机时各节点使用相同工程路径，并核对输出提交一致；后续全部命令在该仓库根目录执行。

### 2. 准备环境并构建timer wheel

先将下面CANN_ENV替换为本机真实工具链脚本路径；已加载CANN环境时可省略该export。
以下venv复用基础Python环境中的框架包；若导入检查失败，先在此工程内环境安装目标机配套Torch/torch_npu，再继续。

```bash
export CANN_ENV=/实际的CANN安装目录/set_env.sh
source scripts/ll_timer_env.sh
python3 -m venv --system-site-packages .venv-ll-timer
source .venv-ll-timer/bin/activate
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"

unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
python scripts/build_shmem.py
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

环境脚本将日志/缓存/临时文件放在工程内，并添加仓内SHMEM库路径；不自动选卡。SDK匹配时自动校验复用，身份不匹配时按报错处理，停止使用者后才显式--rebuild。每次换提交或修改源码后都需重装wheel；普通增量构建不删除build目录。

### 3. 单机EP16计划检查与执行

新终端先重新激活环境、加载CANN。devices必须按本机已分配设备明确选择；下面列出目标机16卡，runner统一设置四个设备可见性变量，无需重复export。
NDR默认混合后端：机内MTE、跨机RDMA put。EP16选择G=4，EP32选择G=16；单机运行不会产生跨机RDMA流量。

```bash
source .venv-ll-timer/bin/activate
source scripts/ll_timer_env.sh
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
RUN_NAME=timer-001

LL_ARGS=(
  --devices "$DEVICES"
  --num-nodes "$NNODES" --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR" --master-port 29500 --shmem-port 29501
  --num-tokens 32 --num-max-tokens-per-rank 256
  --hidden 7168 --num-topk 6 --num-experts 256 --dtype bf16 --do-cpu-sync 0
  --warmup 10 --num-iters 100
  --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe
  --skip-perf-test --output-dir results/ll-timer --run-name "$RUN_NAME"
)
python tests/ll_stage_timer/run.py "${LL_ARGS[@]}" --plan
python tests/ll_stage_timer/run.py "${LL_ARGS[@]}"
```

--plan只检查CPU计划，不验证设备或互联。实际运行仍执行10轮预热、100轮稳定调用，再采5轮打点；批次初始一次all_reduce、预热后200次Exp，稳定/capture循环内无Exp或采样all_reduce，默认Exp约13.53GiB/卡，保留打点输出也占显存。输出目录不能已存在，重复实验请改RUN_NAME。
需要专家统计时，把数组中的--ll-stage-timer改为--ll-expert-timer，阶段capture会自动开启。只测dispatch加--dispatch-only。

### 4. 双机EP32

每台节点完成步骤1、2，确认匹配的提交/wheel、可达地址与端口。将步骤3的NNODES改为2，两台NODE_RANK分别为0、1，MASTER_ADDR都改为首节点可达IP，RUN_NAME和其他参数完全相同；然后**重新定义LL_ARGS数组**并在两台同时执行实际命令。入口不通过SSH自动启动另一台。
跨机还需要目标机器的HNS_1825设备/provider与网络环境；入口会先执行RDMA preflight，失败不得回退冒充成功。需要NDR自身MTE对照时显式加--backend mte。

### 5. 分析结果

单机完成后直接执行：

```bash
python tests/ll_stage_timer/analyze.py results/ll-timer/timer-001 --trace-iteration 4
```

若改了RUN_NAME，分析路径也相应修改。多机先将同一run各节点的rank数据按原相对目录合并到一个run目录，保留公共manifest；不要覆盖不同rank文件或混入其他实验。缺rank或配置不匹配时分析器会拒绝处理。
默认生成stage-analysis下的summary.json、stages.csv和trace.json；多case分子目录。开启专家统计还会生成experts.csv、expert_cores.csv。trace.json可导入Chrome tracing/Perfetto；跨rank时间戳不对齐。

### 6. 后续Git更新与正式性能

```bash
git status --short
git pull --ff-only
git submodule update --init --recursive --depth 1
source .venv-ll-timer/bin/activate
source scripts/ll_timer_env.sh
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
python scripts/build_shmem.py
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

先确认或妥善保留本地修改，不使用reset/clean覆盖它们。所有节点更新到同一提交；构建依赖变更时重跑requirements-build.txt安装。
正式未插桩基线改为DEEPEP_LL_STAGE_TIMER=OFF重装wheel，并从执行数组中移除timer/专家参数及--skip-perf-test，保留10/100采样（perf默认最后50条）。ON打点耗时不能充当OFF基线。

## 构建与接口

打点默认编译关闭，使用`DEEPEP_LL_STAGE_TIMER=ON`开启。打点最初引入核心协议版本2；混合通信/tiling key使用协议4；当前可选专家诊断缓冲使核心协议升为5。所有rank必须重建并安装同一版本完整wheel，不能混搭旧设备库。RDMA协议见[审计说明](../ll_rdma/AUDIT.md)。

完整部署和运行见上方步骤。NDR固定编入HNS_1825能力，无需额外RDMA编译/运行开关。

已有业务代码可以对单次LL调用使用以下独立诊断上下文，公开dispatch/combine参数不变：

```python
from deep_ep.utils.ll_stage_timer import capture

with capture(buffer, "dispatch", mode="observe") as trace:
    result = buffer.dispatch(...)  # 正常LL参数，实际EP必须合法
trace.save(output_dir / "dispatch-stage.json", metadata={"rank": rank})
```

`output_dir`由业务入口按自身脚本位置解析成工程内绝对路径，避免依赖当前工作目录。

每个上下文只允许同一buffer的一次指定操作，不允许嵌套/复用；timer张量会随LL内部stream保活，离开上下文后复制到CPU并检查完整性。业务示例的自定义metadata不直接用于下述全rank汇总；完整测试入口会写入所需身份及正确性证据。

## 目标机运行

单机与双机完整命令见上方步骤3、4。本地rank数由显式devices推断，LL默认普通页；旧run_ll.py委托同一入口。

这是稳定运行后的独立诊断尾段。跳过perf时，先执行warmup次预热、num-iters次未打点稳定调用，再执行ll-timer-iters次打点；仅批次初始一次all_reduce，预热同步后固定200次Exp，再连续稳定和打点调用；循环内不插入Exp或采样all_reduce。所有打点轮次结束后才复制、校验和保存，不重试失败实验。`--plan`仅打印CPU计划；A5不可执行上面的16卡命令。

去掉`--skip-perf-test`后，每个操作先完成原有profiler采样（初始一次all_reduce、10预热、active profiler.step前固定200次Exp、100次连续采集、统计最后50条），关闭profiler后复用同一Exp缓冲提交200次Exp，再连续追加打点轮次，不再增加all_reduce，再解析性能数据。profiler结束可能包含导出开销，不保证诊断与最后一条性能样本无间隙。timer诊断和profiler样本不混合。旧delay-exp-before/after参数仅接受200/0，次数固定且按批次执行；不要沿用首调用后1000次Exp的旧配置。`--dispatch-only`只诊断/测量dispatch；正常默认同时覆盖dispatch和combine。

M决定去重：`M>32`启用，`M<=32`关闭。若对比两条路径，在满足`B<=M`的前提下分别测量并保留独立输出目录。

## 两种边界口径

- `observe`：不增加流水屏障，记录现有函数/代码段边界。异步MTE的下发和完成可能跨越阶段；`send_issue`表示下发区间，不是网络传输完成时间。
- `drain`：每次Start/End显式加入`PipeBarrier<PIPE_ALL>()`，便于观察阶段执行与排空成本，但会破坏重叠，改变整体性能。必须单独运行，不能与observe或未打点基线混算。

计时本身也有读取计数器、GM写回、分支及寄存器资源成本。ON构建且不传timer时不记录数据，但二进制资源占用仍可能与OFF不同；严格性能基线使用`DEEPEP_LL_STAGE_TIMER=OFF`重新构建的wheel。

## 阶段和源码

| 阶段 | 含义 | 主要源码 |
|---|---|---|
| `kernel` | 本核主体区间；不含timer头初始化和最终完成标志写回，也不等于profiler完整task区间 | `csrc/kernels/{dispatch,combine}/low_latency.cpp` |
| `magic`、`worker_init`、`count_init` | 窗口轮转、worker初始化与计数初始化；dispatch存在嵌套 | `dispatch/low_latency/dispatch_server_dedup_kernel.h`、combine入口 |
| `count_send` | 路由计数及向专家所属rank发布，observe模式不额外等待DMA | `dispatch/low_latency/device/dispatch_server_dedup_counts.h` |
| `count_receive`、`count_prefix`、`count_publish` | 计数ready等待、局部前缀、前缀发布及原有排空 | 同上及dispatch总调度 |
| `sender` | 发送核整个Process父区间 | `dispatch/low_latency/device/dispatch_server_dedup_send.h` |
| `send_prepare`、`send_payload`、`send_metadata` | UB/路由准备，发送主循环，lane收尾及源掩码/权重写回 | 同上 |
| `send_classify` | 去重路由分类累计，不含外部q计算 | 同上`Classify`调用点 |
| `send_pack` | 输入搬入、可选量化、包尾/标志打包及lane等待；不解释为纯计算 | 同上`SendPacket` |
| `send_issue` | 机内为MTE下发；跨机包含暂存和RDMA put/quiet，不能解读为纯网络时间 | 同上 |
| `receiver`、`recv_prepare` | 接收父区间；前缀/任务表和接收UB准备 | `dispatch/low_latency/device/dispatch_server_dedup_recv.h` |
| `recv_wait`、`recv_consume`、`recv_cleanup` | 探测直到出现ready批次、加载/消费及relay、最后lane/标志清理；重复区间累计 | 同上 |
| `count_finish` | 原计数收尾及控制状态恢复 | dispatch总调度 |
| `combine_prepare`、`combine_notify`、`combine_wait_ready` | 本地准备、通知（含等待本卡准备完成）、组内ready等待 | `combine/low_latency/combine_server_dedup_kernel.h` |
| `combine_direct`、`combine_partial` | DIRECT回传及原排空、partial归约回传 | 同上及`combine_server_dedup_partial.h` |
| `combine_aggregate`、`combine_drain`、`combine_final_drain` | M5/M6源token归约、共享输出收尾、最终流水排空 | combine总调度及`combine_server_dedup_source_aggregate.h` |

上表kernel路径相对于`csrc/kernels/`（第一行除外）。公共格式/设备实现位于`dispatch/low_latency/common/ll_stage_timer_{layout,device}.h`；Host校验在`csrc/ops/low_latency.cpp`；Python上下文及解析器在`deep_ep/utils/ll_stage_timer.py`。

## 输出与统计

每个case分别输出：

```text
<run>/case_00000/normal/ll_stage_timer/
  dispatch/rank_000/sample_0000.json
  combine/rank_000/sample_0000.json
  ...
```

每文件包含原始`[实际AIV数,32,8]` int64数组和每核/阶段解析值。第0槽记录magic/schema/操作/核数/核号/模式/完成标志；schema 2/3阶段记录采用4个int64：first、start_or_last_end、累计ticks、count|active(bit62)，从第8个word开始紧凑排列；不是每阶段独占一行。schema 1旧文件仍按8字段解析。分配器创建512B对齐视图，Host拒绝未对齐的直接传入指针；每核独占2048B，避免跨核cache-line写回冲突，当前支持16..72偶数核。每轮新建零缓冲，不覆盖前一轮；未执行阶段保持零值，不冒充零成本样本。

**目标计数器换算已由用户确认：`time_us = ticks / 1000`。** 测试采集默认1GHz，analyze.py默认读取sample中保存的频率，原始ticks仍保留；缺失频率时只统计ticks，可用--frequency-hz显式指定其他已确认环境。

收齐所有rank的同一case目录后，在仓库根目录执行：

```bash
python tests/ll_stage_timer/analyze.py \
  results/ll-stage-001/<run>/case_00000/normal/ll_stage_timer \
  --output-dir results/ll-stage-summary-001
```

输出`summary.json`与`stages_long.csv`，按rank/core/stage给出mean/min/max/P50/P90/P99；JSON另有各rank逐次取core最大阶段累计值的统计。后者不是跨核全局跨度。重复阶段用sum而不是first到last的跨度。嵌套父子阶段、不同核并行阶段不能直接相加；不跨rank相减时间戳。缺rank、缺轮次、损坏/未闭合记录及混配置会报错。
`stages.csv`现为末轮逐核宽表，`stages_mean.csv`为全部轮次均值宽表；同run全部表合入根目录`stage-times.xlsx`。
`--trace-iteration last`为默认，也可指定轮次。入口与sheet说明见[统一表格导出](../ll_rdma/TABLE_EXPORT.md)。

## 验证范围

- CPU：格式/ABI映射、损坏数据、重复与缺rank/轮次、上下文隔离及基线采样时序。
- `bash tests/ll_stage_timer/validate_device.sh`：专用A5入口，强制卡0/1，仅在卡0启动独立timer验证核；验证开/关、两种模式、每核记录及guard，不建立伪造LL域。
- `tests/ll_stage_timer/check_installed.py`：A5卡0验证完整wheel的ABI、查询核数和timer分配，不执行LL通信。
- 完整dispatch/combine的精度、阶段曲线、跨机同步及实际性能必须由用户在合法目标EP上验证。

## Chrome tracing / Perfetto 时间线（NDR，2026-10-16）

参考 Mega_Moe 的 Chrome Trace Event JSON 导出格式，支持将内部阶段导入
`chrome://tracing`（Load）或 Perfetto（Open trace file）查看。无需修改设备kernel、SDK或ABI。

### 新采集自动输出

构建仍使用 `DEEPEP_LL_STAGE_TIMER=ON`，测试保留
`--ll-stage-timer --ll-timer-mode drain`，默认使用用户已确认的1GHz计数器（ticks/1000 us）。
每份 `sample_0000.json` 旁自动生成 `trace_sample_0000.json`，可直接加载。
原始ticks始终保留。Python导出代码位于wheel中，更新后重新安装wheel以启用自动导出；
离线转换不需重编。

### 已有数据离线转换（不需重新采集）

在仓库根目录执行，替换实际run路径；所有相对输入输出均按脚本定位的工程根解析：

```bash
python tests/ll_stage_timer/export_trace.py \
  results/ndr-drain-ep32/<run>/case_00000/normal/ll_stage_timer \
  --frequency-hz 1000000000 \
  --rank 0 --operation dispatch --iteration 0 \
  --output results/trace-dispatch-rank0-sample0.json
```

上例频率可省略，默认1GHz。也可传单个raw sample文件。
去掉rank/operation/iteration筛选可导出多份采集，允许单机的部分rank数据；此转换器
用于浏览，不替代analyze.py对完整rank/轮次的验收。输出文件已存在会报错，避免覆盖。

### 时间线口径

- 每个rank/operation/sample为独立进程轨道，分别以本次最早内部时间戳归零。
  核间保留本设备计数器相对时间；不同rank/不同调用不对齐，不与外部profiler合并。
- 默认每个AIV core只有一条具名轨道，所有阶段使用同一tid，父子区间显示嵌套。重叠或重复包络可能仍有嵌套层级；不改变真实起止点。加`--lane-layout stage`可恢复分阶段轨道排查重叠。
- 单次阶段显示首次到末次真实区间；重复阶段标为`[envelope xN]`，宽度是首尾包络，
  **包括中间空隙，不是累计执行时间**。点击事件查看`total_us/total_ticks`累计耗时、
  `average_us`、`calls`、`gap_ticks`及原始起止ticks。
- schema 1/2/3都没有保存每次重复调用的起止点，无法从现有数据还原Mega_Moe式逐次动态事件；
  不伪造连续片段。父子阶段/并行阶段不可直接相加，drain数据也不是未插桩性能基线。

## 推荐：一个命令完成统计和tracing导出

运行用例时可以提前指定目录名（两台节点参数一致）：

```bash
# 加到原有测试命令末尾；新run名不得与已有目录重名。
--output-dir results/ll-stage-001 --run-name drain-001
```

收齐同一run全部rank后执行：

```bash
python tests/ll_stage_timer/analyze.py results/ll-stage-001/drain-001
```

不指定run-name、仍使用自动随机目录时，无需查目录名：

```bash
python tests/ll_stage_timer/analyze.py results/ll-stage-001 --latest
```

`--latest`根据子目录的cases.json修改时间选最近创建的run，打印实际选择路径；
复制多机数据时保留该manifest时间（或使用固定run-name路径）。不会因最新run缺rank而退回旧run。
多机仍须先合并所有rank的原始数据；analyze会拒绝不完整或混配置样本。

默认在选中目录的stage-analysis下生成summary.json、stages.csv、trace.json。
多个case/路由各自输出到stage-analysis下对应相对目录，不混算配置。
统计覆盖全部轮次；trace.json默认展示第0轮全部rank的dispatch/combine，
可加`--trace-iteration 3`查看第3轮。各rank/操作独立归零，不代表真实跨rank同步。
默认换算ticks/1000 us，原始ticks仍在样本和trace事件参数中保留。

已有报告不覆盖；再次分析使用`--output-dir results/ll-stage-summary-002`。
无需另跑export_trace.py；该入口仅保留给单rank/部分数据浏览及细粒度筛选。

## 稳定尾段采样参数（2026-10-16）

以下两种模式采用相同前处理；参数加到原目标机测试命令中：

```bash
# 无外部profiler：10次预热 + 100次稳定调用 + 5次内部打点
--skip-perf-test --ll-stage-timer --warmup 10 --num-iters 100 \
  --ll-timer-iters 5 --ll-timer-mode observe

# 外部profiler：10次预热 + 100次性能采集 + 5次内部打点
--ll-stage-timer --warmup 10 --num-iters 100 --perf-tail 50 \
  --ll-timer-iters 5 --ll-timer-mode observe
```

需要drain时将observe改成drain。内部打点仍编号0..4，最后一轮分析使用
`analyze.py ... --trace-iteration 4`。perf-tail只控制性能统计，不改变打点数量。

timer缓冲在采样前分配，采集循环只执行前处理、LL调用及设备流依赖；
不逐轮复制到CPU、执行正确性collective或写文件。所有轮次的输出和handle保持存活，
结束后统一完成边界，再逐轮校验、导出。失败不自动重试；analyze会校验新metadata声明的
完整诊断轮数，拒绝中断后只留下部分轮次的结果。

显存注意：非perf打点现在同样使用默认118260×30720 FP32 Exp缓冲（约13.53GiB/卡），
并同时保留N轮输出/handle及timer（每轮每实际AIV 2048B加对齐空间）。N增大会增加峰值显存；
建议先保持默认5轮。A5仅双卡原语或CPU验证，不运行原生LL；小Exp不能代表目标参考负载。

新流程修改了Python capture接口，所有节点须安装匹配版本wheel：更新Git后保持
DEEPEP_LL_STAGE_TIMER=ON，执行既有pip重装命令。
以上稳定尾段采样本身不改变SDK；当前专家诊断新增dispatch参数，须使用ABI5完整wheel，SDK配置不变且通过当前原生源码身份校验时无需重建SHMEM；旧QP扩容SDK必须在停止使用者后显式重建。
默认单AIV轨道适用于自动trace、analyze.py及export_trace.py。更新Git后对已有raw sample重新分析即可，无需重新采集；核间起始偏移保持原样，不做逐核归零。

## dispatch细分阶段（timer schema 2）

阶段细分引入时核心通信ABI为4；当前专家诊断使用核心ABI5；细分阶段最初使用schema2，当前字节计数使用schema3，原始张量形状仍为[实际AIV,32,8]，
每核仍2048B、512B对齐。设备和Host/Python必须全rank重建安装匹配wheel，入口会互检timer schema。
解析器兼容旧schema 1/2文件，但一次统计不混用不同schema；旧文件没有新阶段，必须重新采集才能获得细分。

| 父区间 | 子阶段 | 边界含义 |
|---|---|---|
| recv_consume | recv_lane_wait | IssueLoad中等待MTE3_MTE2，保护lane复用 |
| recv_consume | recv_gm_to_ub | 接收packet的GM→UB搬运下发、相关视图准备和事件发布 |
| recv_consume | recv_load_wait | ConsumeLoaded入口等待MTE2_S，使Scalar能读取包尾 |
| recv_consume | recv_metadata | 包尾解码、type/destination构造、FIRST_HOP转发包尾及marker准备 |
| recv_consume | recv_output_wait | 输出前S_MTE3同步和MTE2_MTE3等待 |
| recv_consume | recv_output_write | token/scale、tokenType、destinationIndex、relayReadIndex等写回；分段累计 |
| recv_consume | recv_forward | FIRST_HOP目标遍历及Packet转发；当前NDR组内relay为机内MTE |
| send_pack | send_input_gm_to_ub | 输入队列分配、可选预量化padding、token/scale搬入下发和EnQue |
| send_pack | send_input_wait | DeQue及非量化路径MTE2_S同步，分段累计 |
| send_pack | send_tail | FillTail填充发送包尾元数据 |
| send_pack | send_lane_wait | 等待MTE3_V，使发送lane可复用 |
| send_issue | send_mte_write | 本机Packet的MTE写入下发 |
| send_issue | send_rdma_stage | 跨机Packet原有排空、UB→注册GM暂存、MTE3_S完成同步 |
| send_issue | send_rdma_body_post | 逐块480B body的RDMA put提交，包含SDK内部开销 |
| send_issue | send_rdma_body_wait | body quiet完成等待 |
| send_issue | send_rdma_flag_post | 逐块32B flag的RDMA put提交 |
| send_issue | send_rdma_flag_wait | flag quiet完成等待，保护暂存复用 |
| send_issue | send_rdma_restore | SDK借用UB前128B的恢复及原有同步/排空 |

发送lane等待和输入搬入原本位于send_pack，不属于send_issue；新增打点保留原算法边界，
不把这些操作移动到send_issue中。send_issue的细分重点是本机MTE或跨机两阶段RDMA发布。

observe不增加流水屏障，搬运区间通常表示下发成本，不能等同于设备传输完成时间。
drain会在每个子阶段Start/End继续加入PIPE_ALL，可能将后续等待成本转移到前一阶段排空，
并破坏预取/发送重叠。因此建议先用observe定位等待，再用drain查看串行化后的成本；
子阶段之和不保证等于父区间（还有边界和插桩开销）。重复区间仍只提供包络与累计值，
不会伪造每次调用的时间点。详细插桩增加开销，正式基线保持timer OFF。

细分timer本身不改变SDK；仅通过当前原生源码身份校验的SDK可复用，旧QP扩容SDK须先停止使用者并显式重建。保持
DEEPEP_LL_STAGE_TIMER=ON，重新完整构建安装wheel，然后使用原测试命令重新采集。
分析仍只需一个analyze.py命令，CSV及trace自动包含新阶段。
## 可选接收侧专家统计（ABI 5，本地修改待设备验收）

在原测试命令上增加`--ll-expert-timer`，它隐含阶段 capture，无需再加 `--ll-stage-timer`；建议先使用observe。
例如既有目标启动命令末尾使用：

```bash
--ll-expert-timer --ll-timer-mode observe --ll-timer-iters 5 \
  --output-dir results/ll-expert --run-name expert-001
```

原stage缓冲保持每核2048B；专家统计使用额外512B对齐缓冲，每次调用占用
`实际AIV数 × (8 + 全局专家数 × 4) × 8 × 8`字节。
仅dispatch分配专家缓冲，combine维持原阶段打点；5轮采样会保留5份。
新增缓冲指针使核心ABI升为5，所有rank必须重建完整匹配wheel；当前阶段schema为3（新增通信计数），专家schema为1。

统计按执行rank/AIV/全局expert ID归属：

- `packets`：本地专家消费的接收packet数（展开投递数，不是全域去重后的唯一输入token数）；仅计数，时间列留空。
- `metadata`：本地专家包尾解码和type/destination构造。
- `output_write`：本地专家token、scale、type和destination写回代码区间。
- `forward`：FIRST_HOP逐目标专家的Packet调用；执行rank可能不同于专家所属rank。

共享首跳发送、FIRST_HOP共享准备及relayReadIndex写回、批量recv_wait不分摊给单个专家，
它们继续由原阶段统计表示。不能把多个专家的值相加当作整算子延时。
observe记录下发/代码边界，drain含额外排空；这些都不等同于专家FFN/GEMM时间或网络端到端时间。

收齐所有rank后一个命令完成统计及trace：

```bash
python tests/ll_stage_timer/analyze.py results/ll-expert/expert-001
```

新增`experts.csv`（按执行rank/专家/阶段，先跨核累加每轮工作量，再做轮次统计）和
`expert_cores.csv`（逐轮逐核原始解析记录）；summary.json含per_expert。
`mean`是执行过该阶段轮次的累计工作时间均值，`mean_span`是同一执行rank内该阶段的首尾包络均值，
两者不同；不做跨rank时间戳对齐。没有执行的阶段不补零耗时样本。
只改变数据导出时，可用已有raw重新分析；新增专家数据必须用新wheel重新采集。

### 默认隐藏六类重复发送包络

send_input_gm_to_ub、send_input_wait、send_tail、send_lane_wait、send_issue、send_mte_write
在calls>1时不再默认绘制首尾包络，累计值仍完整保留在stages.csv。calls=1时保留真实单次区间。
需要恢复包络时，analyze.py或export_trace.py增加`--show-repeated`。

本次远端环境已回收，仅完成本地CPU验证，尚未进行CANN编译和设备运行。
请先重建并用小case验证正确性，再进行目标性能采集；不要将历史A5验证记录视为ABI5的验收。

## 跨kernel分支的统一实验
共同入口、构建和比较步骤见[PARITY.md](PARITY.md)。泛化分支默认G16与NDR的G4可直接作为变量对比，不强制一致；结果记录两边实际G。

NDR dispatch已拆为14个独立特化入口以缩短首次编译，保持全部类型/G和单次启动；内部launch符号v2需整套wheel更新。profiler名称与冷构建实测见[首次编译说明](../../docs/BUILD_COLD.md)。

## 按算子筛选离线分析

```bash
python tests/ll_stage_timer/analyze.py <run目录> --op dispatch --trace-iteration 4
python tests/ll_stage_timer/analyze.py <run目录> --op combine --trace-iteration 4
```

`--operation`是`--op`的同义写法；不传则保留全部算子。筛选覆盖summary、CSV、trace及本分支附加报告，所选算子的全部rank/轮次仍必须完整。默认分别输出stage-analysis-dispatch、stage-analysis-combine；可用--output-dir指定新目录，已有输出不覆盖。仅离线脚本更新，无需重编wheel或重新采样。
