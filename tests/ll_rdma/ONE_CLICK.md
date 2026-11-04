> 当前默认身份34、直方图身份35：dispatch按GM stage容量批处理，combine回传批处理并将尾批CQ移至M5之后。见[COMBINE_BATCH_RETURN.md](COMBINE_BATCH_RETURN.md)。下方旧身份为历史。

> 当前count源已隔离，SendCnt无quiet；默认身份30、直方图身份31。旧身份及立即quiet描述为历史。见[COUNT_SOURCE_ISOLATION.md](COUNT_SOURCE_ISOLATION.md)。

> 默认身份28；新增`DEEPEP_LL_COUNT_HISTOGRAM=ON`硬件直方图实验身份29，必须完整重编，见[COUNT_HISTOGRAM_AB.md](COUNT_HISTOGRAM_AB.md)。OFF/ON不能混装。
>
> 默认协议28：已移除两机/EP32专限，支持现有EP16/32/64/128/256入口，新增`--num-nodes`。
> 多机count按远端rank分核/分轮直发；首跳按目标主机mask独立选择。变更清单及风险见
> [MULTI_HOST_LIMIT_AUDIT.md](MULTI_HOST_LIMIT_AUDIT.md)。下方协议27及更早说明仅作历史对照。

> 当前为一包深度暂存/put流水，实现身份26、reference schema4；复用原lane，单lane立即提交。S表区分暂存经过、延后间隔、剩余等待及前包提交，不改packet所属核/顺序。详见 [ONE_PACKET_PIPELINE.md](ONE_PACKET_PIPELINE.md)；覆盖下方旧立即提交及schema2说明。

> 当前reference为schema2，增加S核第二跨机包时序及R核UB包顺序日志；原核总耗时边界不变，写出前总耗时包含日志导出。详见 [PACKET_TRACE.md](PACKET_TRACE.md)。

> 当前协议24：已撤销全部8QP派生调度与缓冲改动，恢复原token分片、核内FIRST_HOP/DIRECT顺序和逐次Vector q；保留路由SO。当前实现以 [PER_CORE_QP_AB.md](PER_CORE_QP_AB.md) 为准，下方实验历史不作为现行行为。

> 2026-10-28更新：本分支当前协议23，每物理核QP，已恢复逐次Vector求q；不启用q预计算。测试脚本normal路由直接调用按用户tmp.py移植的make_random_expert_indices，默认倍率2、seed偏移0的输出与原脚本一致；hotspot保持原方式。下方协议22/保留q预计算描述属于历史对照7fbc309。

> 当前分支为新增每物理核QP基线，协议22，覆盖下方历史8QP说明。见 [PER_CORE_QP_AB.md](PER_CORE_QP_AB.md)。原四个A/B分支不改。

> 本分支属于四版独立性能A/B：当前 kernel/ll-ndr-ab-base。当前版本与对照关系以 [FOUR_WAY_AB.md](FOUR_WAY_AB.md) 为准，下方历史协议描述不作为当前安装身份。

# 多机 NDR 一键构建、测试和本机解析

入口：`tests/ll_rdma/one_click.sh`。参考 `kernel/ll-g4-local-q-table@af094ce` 的
环境、完整ON wheel、稳定尾段采样与分析流程，使用本分支的NDR/SDK/计时schema。
每台目标机各执行一条命令；不通过SSH连接其他机器、不终止其他作业。

默认 `--mode reference` 是仅LL dispatch的参考式打点，使用独立reference wheel并关闭旧stage/expert计时；
默认1轮capture，自动生成逐核Excel。阶段定义、BS1命令和回传报告见
[LL dispatch参考式逐核打点](../ll_reference_profile/README.md)。

默认 EP32，每机设备0..15，B8/M256/H3584/K8/E384，BF16和预量化FP8两组，
默认仅dispatch，正确性检查保留；`--mode perf`执行dispatch+combine，FP8配置的combine仍是BF16。
`--num-nodes`默认2，可指定现有域对应的1/2/4/8/16；节点编号必须在0..N-1。
以下0/1示例使用默认两机，多机时每个节点添加相同`--num-nodes "$NODES"`，并使用各自NODE。
实际EP由16×节点数计算；E384可用于EP32/64/128，EP256须显式使用可整除的E，如1024。

默认 `--route normal` 已改为用户指定的严格均衡构造：每rank接收 `B*K` 条路由，每token选K个不同rank，rank内专家轮转；固定 `hot_multiplier_x2=2`。步长自动取 `E/EP`（默认EP32为12，EP64/E384为6）。基础seed为20261014，`--seed`默认0、作为偏移，抽样seed为 `20261014 + seed + B*1000 + 2`。旧 `hotspot` 的专家随机采样保持不变。

结果的 `routing.generator=balanced-rank-distinct-roundrobin-v1` 标识新normal；comparison及reference样本保留seed/步长和实际路由SHA256。多轮正确性后恢复声明seed及匹配handle再测量。旧normal是独立专家随机采样，不能与新normal当作同负载A/B；历史结果不改写。路由在CPU准备阶段构造，不计入kernel采样。
此NDR性能分支仅编译LL，跳过与当前SHMEM内部UDMA接口不匹配的HT路径；
不会初始化CATLASS子模块。HT调用明确拒绝。

当前通信实现为[协议28：源端直发count](DIRECT_RANK_COUNTS.md)，全rank更新完整wheel，不能混装旧协议。

## 使用

所有节点的工程绝对路径必须相同，使用相同分支/提交、run名及参数。
先在各自工程根执行 `git pull --ff-only`，并加载现有CANN环境；
也可设置 `CANN_ENV` 指向实际 `set_env.sh`，脚本会加载它。
本脚本不自动切分支、合并或覆盖用户源码。

首次升级到当前SHMEM主线版本，先停止使用该工程SDK的作业，然后各机分别执行：

```bash
# 节点0：把 192.0.2.10 替换为节点0的真实可达IPv4地址
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-b8-001 --rebuild-shmem

# 节点1：使用同一个节点0地址和run名
bash tests/ll_rdma/one_click.sh 1 192.0.2.10 ndr-b8-001 --rebuild-shmem
```

普通重测换run名并去掉 `--rebuild-shmem`，脚本会校验复用匹配SDK并构建wheel。
已安装与当前源码、SDK、reference模式完全匹配的wheel时，可加 `--skip-build`。
不要同时使用这两个选项。

默认使用三个不同端口：29500（HCCL启动）、29501（SHMEM）、29502（全节点准备确认）。
可用 `--master-port`、`--shmem-port`、`--ready-port` 修改，所有节点配置必须一致。
准备确认只交换版本和参数JSON；不使用Gloo/TCPStore，不参与算子计数、数据通信或采样。
各机完成构建/检查后最多等待其余所有节点1800秒，可用 `--ready-timeout` 调整。
如果一机准备失败，其他节点的等待有超时；测试失败不自动重试。

## 流程与结果

脚本定位自己的工程根，执行：

1. 创建/复用工程 `.venv-ll-rdma`，继承目标已有torch/torch_npu。
2. 获取锁定子模块、校验/显式重建SDK、安装构建依赖和完整wheel。
3. 检查当前源码fingerprint、SDK版本/实际PI模式、安装包NDR协议14、每peer 8个RoCE QP、G16及reference模式。
4. 用既有ST生成CPU计划，执行RDMA环境前检。
5. 所有节点核对工程路径、Git版本、源码、SDK与完整测试参数，一致后进入原HCCL入口。
6. 运行测试；reference模式成功后自动分析各自本机16rank，不要求复制其他节点样本。

2026-10-26：允许各机 `hns_1825_pi_on_chip` 不同，只从跨机比较副本中移除该字段。
各机identity.json仍保留实际PI值，且每台机器的SDK与wheel必须匹配同一个有效0/1值。
不用为了跨机同值重建SDK，也不能手改manifest；实际模式仍须匹配本机固件，
SDK探测fallback不能当作硬件验收。
其他源码/协议/参数差异仍拒绝，错误会列出具体字段及节点0与不匹配节点的值。
`source_sha256`涵盖构建脚本、测试与源码；相同Git提交但已跟踪文件有本地修改仍会不一致。
Git checkout中只纳入已跟踪测试文件，其他分支遗留的未跟踪实验目录保留但不进入指纹；
实际构建目录中的源码/头文件仍完整计入。根目录的 `core` 文件从未参与该指纹。

默认reference模式：reference ON wheel，10次预热、100次稳定调用、1轮独立capture。
每个批次只在开始做一次all_reduce；预热后固定200次Exp，再连续稳定调用和capture。
每次 dispatch/combine 调用前另插入一次标量 all_reduce（异步 + wait）拉齐快慢卡，开销计入每次调用，
因此数值是对齐后的时延而非纯 kernel 时间。默认Exp缓冲约13.53GiB/卡，此模式不运行正式profiler。

结果目录：`results/ll-ndr/<run名>-reference/`。
每机的报告在其下`table-analysis/`，统一打开`stage-times.xlsx`；多个case按sheet前缀区分，
详细索引为`TABLE_INDEX.json`。`stages.csv`改为所选末轮每核一行、阶段为列，
多轮均值另存`stages_mean.csv`，原长表保留`stages_long.csv`。
已有样本只重导出：`bash tests/ll_rdma/one_click.sh --analyze-only RUN_DIR --output-dir NEW_DIR`，
无需CANN或重编/重采。完整说明见[TABLE_EXPORT.md](TABLE_EXPORT.md)。
命令及各步日志在 `.logs/ndr-one-click/<run名>-reference/node-0|1/execute/`。
原始样本、错误日志保留，已有run或日志目录拒绝覆盖，重跑使用新run名。

## OFF性能与参数调整

所有节点同样增加 `--mode perf`，脚本安装OFF wheel，执行10预热/100采集/尾50统计。
rank0输出原有完整性能汇总；各节点保存本地profiler原始数据。
OFF模式不调用阶段解析器，也不接受误装的ON wheel。

```bash
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-perf-001 --mode perf
bash tests/ll_rdma/one_click.sh 1 192.0.2.10 ndr-perf-001 --mode perf
```

常用覆盖，所有节点必须相同：

```bash
# 多B、只测FP8；其余保持默认
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-bs-001 --tokens 1 2 4 8 16 32 --dtype fp8
# 先准备环境，不加入通信；随后相同run名加--skip-build执行
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-ready-001 --prepare-only
# CPU计划，不构建、不下载、不建venv、不启动NPU
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-plan --plan
```

还有 `--capacity`、`--hidden`、`--topk`、`--experts`、`--devices`、`--route hotspot`、
`--reference-iters`；`--help`列出全部选项。每机必须提供16个实际设备，物理机界不改变。
所有日志/缓存/临时/结果保持工程内，相对CANN/provider路径也按工程根解析。

本地验证仅覆盖CPU计划、调用顺序、失败停止、身份拒绝及TCP准备握手。
不代表目标CANN编译、RDMA可达性、NPU正确性或性能已通过。
