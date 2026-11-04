> 当前默认身份34、直方图身份35：dispatch按GM stage容量批处理，combine回传批处理并将尾批CQ移至M5之后。见[COMBINE_BATCH_RETURN.md](COMBINE_BATCH_RETURN.md)。下方旧身份为历史。

> 当前协议28已移除未确认的EP32/两机特例；旧配对源码已删除，历史可从7293457读取。
> 当前行为、逐项清单及剩余风险见 [MULTI_HOST_LIMIT_AUDIT.md](MULTI_HOST_LIMIT_AUDIT.md)。

> 当前为一包深度暂存/put流水，实现身份26、reference schema4；复用原lane，单lane立即提交。S表区分暂存经过、延后间隔、剩余等待及前包提交，不改packet所属核/顺序。详见 [ONE_PACKET_PIPELINE.md](ONE_PACKET_PIPELINE.md)；覆盖下方旧立即提交及schema2说明。

> 当前reference为schema2，增加S核第二跨机包时序及R核UB包顺序日志；原核总耗时边界不变，写出前总耗时包含日志导出。详见 [PACKET_TRACE.md](PACKET_TRACE.md)。

> 当前协议24：已撤销全部8QP派生调度与缓冲改动，恢复原token分片、核内FIRST_HOP/DIRECT顺序和逐次Vector q；保留路由SO。当前实现以 [PER_CORE_QP_AB.md](PER_CORE_QP_AB.md) 为准，下方实验历史不作为现行行为。

> 2026-10-28更新：本分支当前协议23，每物理核QP，已恢复逐次Vector求q；不启用q预计算。测试脚本normal路由直接调用按用户tmp.py移植的make_random_expert_indices，默认倍率2、seed偏移0的输出与原脚本一致；hotspot保持原方式。下方协议22/保留q预计算描述属于历史对照7fbc309。

> 当前分支为新增每物理核QP基线，协议22，覆盖下方历史8QP说明。见 [PER_CORE_QP_AB.md](PER_CORE_QP_AB.md)。原四个A/B分支不改。

> 本分支属于四版独立性能A/B：当前 kernel/ll-ndr-ab-base。当前版本与对照关系以 [FOUR_WAY_AB.md](FOUR_WAY_AB.md) 为准，下方历史协议描述不作为当前安装身份。

# HNS_1825 混合 MTE / NDR LL（核心协议10）

> 当前协议14：路由SO整帧、每对卡8QP、按实际目标rank选择各自8核并覆盖全部S核。部署与所有权说明见 [ROCE_SO_ROUTE_8QP.md](ROCE_SO_ROUTE_8QP.md)。

> 当前仅保留 reference dispatch 设备打点，one_click 默认 reference；旧 stage/expert 插桩已删除，旧结果仍可离线解析。详见 [阶段与源码对照](../ll_rdma/STAGE_TIMELINE.md)。

阶段耗时为何不能整行相加、timer/reference的先后关系与总耗时对账公式，见
[阶段顺序与核总耗时对账](STAGE_TIMELINE.md)。

2026-10-27目标RoCE的WQE尾片SO语义及“单消息末尾32B FLAG”设计约束见
[ROCE_SO_CONTRACT.md](ROCE_SO_CONTRACT.md)。协议10已把dispatch/combine跨机packet改为
整包单WQE及一次源完成等待，复用既有末尾32B flag；见[实现与测试命令](ROCE_SO_PACKET.md)。
独立路由/计数commit保持原协议依赖，SDK不改。

当前分支用于LL性能穿刺，只构建LL Host/Device，HT notify/dispatch/combine不编译；
误调用HT入口会明确报错。CATLASS已移出子模块，普通LL构建不需要它。

2026-10-26冗余通信/Scalar清理、独立commit清单和OFF验收命令见
[OVERHEAD_OPTIMIZATION.md](OVERHEAD_OPTIMIZATION.md)。公开dispatch/combine输入输出保持；
LL生产轮询不使用GetSystemCycle，锁定SHMEM SDK保持原实现。
27上当前分支的OFF/ON编译修复、纯Host验证和产物位置见
[COMPILE_VALIDATION_20261026.md](COMPILE_VALIDATION_20261026.md)。

非去重dispatch的逐函数执行流程、伪代码、EP16与EP32完整例子见
[NON_DEDUP_DISPATCH_FLOW.md](NON_DEDUP_DISPATCH_FLOW.md)。
去重路径的FIRST_HOP分类、接收转发、sourceMask和权重例子见
[DEDUP_DISPATCH_FLOW.md](DEDUP_DISPATCH_FLOW.md)。

两机各一条命令的构建/测试/本机解析入口见 [ONE_CLICK.md](ONE_CLICK.md)：
`bash tests/ll_rdma/one_click.sh NODE_RANK MASTER_IP RUN_NAME`，默认ON诊断，`--mode perf`为OFF性能。
只测LL dispatch、按参考29阶段记录并导出逐核Excel，使用 `--mode reference`，见
[参考式逐核打点](../ll_reference_profile/README.md)。该构建关闭旧stage/expert计时。
timer/reference结束后统一输出`RUN_DIR/table-analysis/stage-times.xlsx`，全部case合为多sheet；
`stages.csv`改为单轮每核一行、阶段为列。已有样本可通过`one_click.sh --analyze-only`重导出，
无需重编或重采；详见[表格导出](TABLE_EXPORT.md)。

2026-10-30：metadata与每核QP基线统一锁定官方SHMEM master的4c671684（原8e3fc610），必须重建匹配SDK和完整wheel，
具体兼容与部署见 [SHMEM_MASTER_UPDATE.md](SHMEM_MASTER_UPDATE.md)。
每卡3 fullmesh + 4 Clos + 1跨机网口的消息合并、首跳策略及A/B方法见
[EIGHT_PORT_OPTIMIZATION.md](EIGHT_PORT_OPTIMIZATION.md)。

维护分支：`kernel/ll-ndr-hns1825`。原泛化/打点分支 `kernel/ll-configurable-groups` 不包含此改动。

最新EP32初始路由按每卡网口采用16路配对交换，gateway机内重建/发布计数；
协议9删除路由ACK及done汇合，每rank每轮仅body和commit两次put，保留两次quiet。
见[PAIRED_ROUTE_EXCHANGE.md](PAIRED_ROUTE_EXCHANGE.md)。全rank更新完整wheel，SDK不改。

跨机B<32按G16去重见[SMALL_BATCH_DEDUP.md](SMALL_BATCH_DEDUP.md)；
计数帧合并见[PACKED_COUNT_FRAMES.md](PACKED_COUNT_FRAMES.md)。两项为顺序独立提交，
全rank更新完整wheel，SHMEM不变。

后续count_recv分段步长读取优化见[COUNT_RECV_STRIDED.md](COUNT_RECV_STRIDED.md)，
保持协议7与原接收顺序，减少轮询中的映射计算及DMA提交。

2026-10-22：NDR初始化、barrier和销毁改为现有HCCL ProcessGroup的Host协调，
先完成本卡全部stream，再交换所有rank状态；生产路径不再启动设备barrier。
**SHMEM不改、不打补丁，无需重建原SDK，只需全rank更新完整wheel**。
部署及边界见[HOST_BARRIER_FIX.md](HOST_BARRIER_FIX.md)。

dispatch sender跨机packet批处理见[DISPATCH_RDMA_BATCH.md](DISPATCH_RDMA_BATCH.md)：
仅合并完成等待，保留wire格式及原SDK；全rank更新wheel，无需SDK重建。

NDR combine每次提交前增加同一HCCL组的Host完成协调，保护重复调用的双窗口复用；
见[COMBINE_HOST_COORDINATION.md](COMBINE_HOST_COORDINATION.md)。SHMEM保持原样。

**默认混合后端现在采用机内 MTE、跨机 RDMA put，禁止跨机 get。生产默认每机 16 个连续 rank。** Host tiling 选择 EP≤16 的 key/G=4，EP>16 的 key/G=16；两种模板在同一 wheel 中。原生 LL 仍不支持 EP2/4/8；M>32 的去重开关保持不变。详细路径和顺序见 [AUDIT.md](AUDIT.md)。

## Git 获取与构建

```bash
git clone --branch kernel/ll-ndr-hns1825 https://github.com/TangYunfei147/DeepEP-LL.git
cd DeepEP-LL
git submodule update --init --recursive --depth 1
python3 -m venv --system-site-packages .venv-ll-rdma
source .venv-ll-rdma/bin/activate
# 先加载本机 CANN，或设置 CANN_ENV；provider 如有额外路径，设置 DEEPEP_RDMA_PROVIDER_PATH。
source tests/ll_rdma/env.sh
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950 DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
bash tests/ll_rdma/build_smoke.sh
```

SDK 默认构建 HNS_1825，并自动校验复用已有匹配产物，无需 `--if-needed`；身份不一致时先停止使用该 SDK 的任务，再显式 `--rebuild`。仅使用仓内 `third-party/shmem/install/shmem`。全 rank 必须匹配核心协议10；移除旧的 `DEEPEP_LL_RANKS_PER_GROUP` 编译定义，G 不再通过编辑全局宏设置。

环境脚本不默认选卡，runner 要显式 `--devices`，日志/缓存/临时文件保持工程内。后续代码、脚本和指南通过 Git 更新，原始数据与构建产物不提交。

NDR 固定编入 HNS_1825 能力，无需设置 `DEEPEP_LL_RDMA=ON`。共同入口按分支默认选择混合后端，无需设置 `DEEPEP_LL_BACKEND`；`--backend mte` 仅用于显式对照。

## 同机 MTE / 跨机写原语

同机两卡只测 MTE 包、计数、ready 和本地 peer 读取；不要求 RDMA NIC，也不能当作 RDMA 验收：

```bash
python tests/ll_rdma/run_smoke.py --devices 0,1 --endpoint "tcp://$RANK0_IP:29672" --rounds 20
# A5 专用：强制四个可见性变量均为 0,1，最多两 rank。
python tests/ll_rdma/run_smoke.py --validation --endpoint "tcp://$RANK0_IP:29672" --rounds 20
```

跨机各一卡先做环境前检，再在两台目标主机同时执行对应 rank。此模式只测试 RDMA 写入和控制发布，不运行 get：

```bash
python tests/ll_rdma/preflight.py
# 主机 0
python tests/ll_rdma/run_smoke.py --rank 0 --devices 0 --endpoint "tcp://$RANK0_IP:29672" --rounds 20
# 主机 1
python tests/ll_rdma/run_smoke.py --rank 1 --devices 0 --endpoint "tcp://$RANK0_IP:29672" --rounds 20
```

前检可补充 `--npu-bdf "$NPU_BDF" --nic-bdf "$NIC_BDF" --topology "$TOPOLOGY_FILE"`；不要伪造平台拓扑。检查 HNS_1825 驱动/provider、Hcomm 符号、端口、NPU/NIC 亲和性及实际 ROCE IP。Host bootstrap IP/port 不等同于 NIC 绑定配置。

原语轮换 512/1536/15872B、compact/keep-flags marker 和不同数据，覆盖重复缓冲复用。所有 rank 必须 PASS，不能把单台成功视为全域成功。日志在 `.logs/rdma-smoke-*`；local_mte_read 标记是否执行了机内读取验证。`--cores` 仅供原语检查实际可用范围内的偶数 16..72，不改变生产 LL 实际核数查询。失败不自动重试。

## 目标完整 LL

编辑 `target16.json`/`target32.json` 的 master_addr 和空闲端口。每机 16 个连续 rank，EP32 必须两机。生产不能用跨机各 8 卡冒充一个 16-rank 物理机组。

共同入口从 `--devices` 推断本机 rank 数；显式配置 `num_processes` 时保留该值并检查设备数。`--validation --plan` 固定可见设备 0,1，但只生成 EP16 CPU 计划，不启动设备。旧 `run_ll.py` 委托同一入口。LL 测试默认 `use_1g_huge_pages=0`。

```bash
# CPU 计划，不启动设备；A5 仅执行此类原生 LL plan。
python tests/ll_stage_timer/run.py --devices 0,1 --validation \
  --config tests/ll_rdma/target16.json --plan

# 目标单机 EP16：G=4，全部数据走 MTE，无 ROCE 初始化。
python tests/ll_stage_timer/run.py --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --config tests/ll_rdma/target16.json --node-rank 0 --skip-perf-test --route normal

# 目标双机 EP32：G=16；两机分别设置 NODE_RANK=0/1，同时执行。
python tests/ll_stage_timer/run.py --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --config tests/ll_rdma/target32.json --node-rank "$NODE_RANK" --skip-perf-test --route normal
```

normal 全 rank 通过后，将 `--route normal` 改为 `--route hotspot` 再执行。配置覆盖 BF16/FP8、H=256/7168 尾块、B=1/7/65；同一 Buffer 三次改变路由的 dispatch，以及三次匹配 combine。小 B 如需启用去重，可显式 `--num-max-tokens-per-rank 256`，保持 B≤M。EP16 开启去重后有四个逻辑组，跨逻辑组仍为机内 MTE。

通过判据：dispatch 行数、来源、专家计数、scale 和 handle 正确；combine 符合独立 CPU 参考及既有数值容差；无跨机读、无错误回退、CQ 错误、超时或复用异常。可用 `--backend mte` 显式选择 MTE 对照；普通运行不需要后端环境变量。

## 打点、性能和回传

打点额外构建 `DEEPEP_LL_STAGE_TIMER=ON`，运行增加 `--ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe --skip-perf-test`；见 [打点指南](../ll_stage_timer/README.md)。metadata 记录实际选定 G。机内 send_issue 为 MTE 下发区间；跨机 send_issue 包含对称暂存和两次 quiet，不能混称纯网络时间。

正式基线重建 timer OFF，正确性通过后移除 `--skip-perf-test`。保持 10 warmup、100 capture、批次初始一次all_reduce，预热后在active profiler.step前提交一次`torch.npu._sleep`（`--device-busy-cycles`，默认13200000000，即1_650_000_000×8），采样循环内无sleep/all_reduce，每 kernel/task 最后 50 条独立统计。Event/API 与 kernel 时间分开；原语和 stage timer 不替代正式性能。比较时记录实际 G/key、核数、shape、page policy 和负载；G 可由用户作为实验变量，无匹配实测不报告加速。

回传所有主机的 Git SHA、SDK manifest、构建开关/ABI、G/key、实际 AIV、物理 rank 分布、环境前检、原语日志、完整 LL cases/summary/progress/loaded-library hashes、原始 profiler task/trace/样本及实际命令。A5 结果只说明可运行性；跨机 RDMA 写、完整 LL 和性能仍由目标机验收。

公共Runtime/Session保留MTE默认；由已确认LL用途的测试入口选择混合后端，不按缓冲容量猜用途。维护原则见[配置注意事项](../ll_stage_timer/CONFIGURATION_NOTES.md)。

NDR wheel/kernel增量构建、timer对象复用与目标机计时步骤见[增量构建说明](../../docs/BUILD_INCREMENTAL.md)。

发送调度复盘、示例图与重新设计议题见 [EIGHT_QP_SCHEDULING_REVIEW.md](EIGHT_QP_SCHEDULING_REVIEW.md)。
