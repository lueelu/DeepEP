## 满stage与combine回传（2026-10-29，默认34/直方图35）

- 用户第一项已实施：dispatch GM批容量floor(128KiB/packetBytes)，删除固定8包上限；按peer位图去重避免包序号超过32位，UB lane与GM分配量不改。独立提交4c12865（身份32/33）。
- 第二项仅审计，未修改其他dispatch依赖：数学依赖/方案依赖/编码时序分类见tests/ll_rdma/DISPATCH_DEPENDENCY_AUDIT.md。
- 第三项combine DIRECT/M4 partial改为满stage回传批次，包准备好立即put，每批每peer quiet；M4尾批CQ越过M5在kernel末尾退休。归约/Cast/权重/所有权/wire/SDK不改，不新增UB或全rank屏障。
- 仅减少quiet与串行等待，put次数及字节不变，不宣称已跨token合包。指南tests/ll_rdma/COMBINE_BATCH_RETURN.md；尚无CANN/NPU性能验证。

## Count源隔离（当前默认30/直方图31）

- 用户明确要求按算子数学逻辑判断；共享暂存区造成的实现依赖不能作为count必须等待payload前完成的理由。
- SendCnt无quiet：每bank 1280KiB空闲区按目标rank分配独立GM源，最多40KiB，不再共用payload Stage；UB复制完成后可复用，SO尾FLAG供接收侧判定。
- 未退休CQ仅在本核末尾处理，已有同peer/同QP的payload quiet覆盖count则去重。零payload/空接收核也回收；角色PIPE_ALL后复用有效scratch，不新增UB/lane/QP。末尾回收在w21前计时。
- 其他quiet区分数学依赖与有限源存储复用；详见tests/ll_rdma/COUNT_SOURCE_ISOLATION.md。SDK保持，无NPU验证。

## 跨机count直方图编译对照（2026-10-29）

- 用户要求硬件直方图优化并提供环境变量编译对照。DEEPEP_LL_COUNT_HISTOGRAM默认OFF/身份28，ON/身份29；同wire但全rank同变体完整wheel，不可仅改运行环境。
- 仅替换CountRemoteRank，Vector归一化+MicroAPI Histograms BIN0 FREQUENCY，最终一次显式V_S，每专家一次Scalar填32B记录。subExp/dstExp/work复用，无新增UB/lane；本机计数、首跳、SDK、put/quiet不改。
- Host/Device宏、构建缓存及_build_info、one_click安装身份和全节点握手同步区分。无CANN/NPU实测，不宣称加速。指南tests/ll_rdma/COUNT_HISTOGRAM_AB.md。

## 多机32P特例清理（2026-10-29，默认协议28）

- 用户要求删除未确认的32P多机限制并审计风险。当前工作区/分支保持DeepEP-LL-ndr-overhead/kernel/ll-ndr-ab-per-core-qp；对照7293457/协议27。
- count对所有跨物理机RDMA启用，跳过本机16rank，task=core、core+C分轮；每帧quiet保护同核GM/UB复用。Host容量核算同步；本机count与原payload lane保留区不变。
- 删除EP32首跳门槛及token级单一meshFirstK，每个目标主机mask独立选首跳，防止跨主机误选。q/原权重保持；新域lead选择可能改变浮点累加顺序，不宣称旧路径逐bit一致，必须目标机数值验证。
- 删除已停用的配对route helper/layout及其旧专属模型。one_click增加num_nodes并按实际EP校验；target_ready v2等待全部不同节点同身份后放行。
- 每机16rank、32B wire/top-k32、公开EP16/32/64/128/256、ABI最大256、锁定SDK每peer56QP限制有依据，保持。CPU72核布局测试不等于SDK允许72核跨机。
- 详细删除清单、正确性证明、性能/数值与设备验证边界见tests/ll_rdma/MULTI_HOST_LIMIT_AUDIT.md。SDK及其他独立实验不改；不执行远端或NPU，本地检查后提交推送。

## EP32源端直发count（2026-10-29，协议27历史）

- 用户要求参照main源端计数直达最终卡，跨机按一核一目标卡分工、减少Scalar。当前协议27，分支kernel/ll-ndr-ab-per-core-qp；对照d251739/协议26。
- NDR EP32物理核0..15各统计一个跨机目标rank的全部专家，Vector EQ+GatherMask取数量，每专家只写最终count；同一WQE发送body+32B尾FLAG，每帧put+quiet。删除活跃配对路由交换/代算/远端源count机内分发。本机源count仍按原专家分片直接MTE发布；部分remote owner兼有本机任务。
- 接收混合布局保留expert/source逻辑顺序、全部快照后清共享commit、零计数、双bank及combine_reuse；SDK/payload/q/权重/每核QP/原一包流水保持。跨机临时帧复用workBuf，SDK scratch复用已退休selected输出；Host峰值同步核算work容量，sender长期保留区仍只含原本机count，不因新帧缩减lane。
- EP16/MTE及其他EP域保留原计数通信路径。CANN/设备性能未验证，不把CPU模型通过称为性能收益。全rank完整wheel升级，不混装。测试及复用证明见tests/ll_rdma/DIRECT_RANK_COUNTS.md。

> 当前为一包深度暂存/put流水，实现身份26、reference schema4；复用原lane，单lane立即提交。S表区分暂存经过、延后间隔、剩余等待及前包提交，不改packet所属核/顺序。详见 [ONE_PACKET_PIPELINE.md](tests/ll_rdma/ONE_PACKET_PIPELINE.md)；覆盖下方旧立即提交及schema2说明。

## 包级reference诊断（2026-10-28）

用户要求：主q排除于调度，S核只细分第二跨机packet并记录跨机/本组提交整体区间与原batch quiet；R核记录成功就绪和实际消费的稳定任务身份/顺序，在UB保存、本核w21后一次DMA导出。schema2每核5120B；R日志只取原lane后的剩余UB，最多128条，必须显式标记截断，不减少lane，不逐包加clock或新增业务屏障。当前通信协议24不变、SDK不改；OFF不含包诊断。交错区间不相加，put不是完成，quiet不是接收消费；w57包含导出，ON完整task扰动须另测。详见tests/ll_rdma/PACKET_TRACE.md。

> 当前协议24：已撤销全部8QP派生调度与缓冲改动，恢复原token分片、核内FIRST_HOP/DIRECT顺序和逐次Vector q；保留路由SO。当前实现以 [PER_CORE_QP_AB.md](tests/ll_rdma/PER_CORE_QP_AB.md) 为准，下方实验历史不作为现行行为。

> 2026-10-28更新：本分支当前协议23，每物理核QP，已恢复逐次Vector求q；不启用q预计算。测试脚本normal路由直接调用按用户tmp.py移植的make_random_expert_indices，默认倍率2、seed偏移0的输出与原脚本一致；hotspot保持原方式。下方协议22/保留q预计算描述属于历史对照7fbc309。

> 当前分支为新增每物理核QP基线，协议22，覆盖下方历史8QP说明。见 [PER_CORE_QP_AB.md](tests/ll_rdma/PER_CORE_QP_AB.md)。原四个A/B分支不改。

> 本分支属于四版独立性能A/B：当前 kernel/ll-ndr-ab-base。当前版本与对照关系以 [tests/ll_rdma/FOUR_WAY_AB.md](tests/ll_rdma/FOUR_WAY_AB.md) 为准，下方历史协议描述不作为当前安装身份。

# 目标RoCE通信契约

## NDR采样方式（2026-10-27最新用户要求）

本分支ST采用`initial-reduce-exp100-batch-v3`：每个操作的采样批次只在开始all_reduce一次，
预热同步后、进入active的profiler.step前固定100次Exp，再连续调用算子；循环内没有Exp或
采样all_reduce。独立timer/reference同样为初始all_reduce、预热、100次Exp、连续稳定/capture。
perf后附加打点时另排100次Exp，共用perf初始all_reduce；不在Exp前缀后额外同步。
2026-11-03新增：每次 dispatch/combine 调用前插入一次小张量 all_gather（`aligned()`）拉齐快慢卡，
开销计入每次调用，样本 metadata 记录 `per_iteration_all_gather=true`；与旧样本不可直接同条件比较。
正确性/批次边界和算子内部必要复用协调保留，kernel/SDK不改。新旧采样协议不能混作同条件A/B。
详见tests/ll_probe/PROFILING.md；本条覆盖本分支旧“每轮Exp+all_reduce”采样说明。

## SO发布

2026-10-27用户明确提供：跨机RoCE每个WQE的尾片具有SO特性，保证该SO之前的数据
先到达，且片内保序；单片默认2KB。后续本项目通信设计以此目标硬件契约为依据。

优先设计同一WQE发送一个完整消息：数据/元数据在前，末尾追加一个32B FLAG，接收端
观察尾FLAG确认消息就绪。不把现有body/flag两阶段put+quiet视为唯一正确方案。
2KB是网络分片粒度，不是消息大小上限；不自行扩展到跨QP/独立WQE、MTE或其他平台。

SO不替代发送源缓冲生命周期、接收消费完成、旧FLAG清理或epoch/双bank复用证明。
修改前核对实际SDK WQE拆分/SO覆盖和尾FLAG布局，并同步接收端与协议版本；SDK保持
既有实现，不能为该假设擅改SDK。

协议10已实现跨机payload整包单WQE：复用现有packet最后32B flag，接收端保留完整flag
检查（包含尾flag）；内部wire/机内MTE不改。dispatch删除gate副本/清gate/独立gate put，
每批每peer一次quiet；combine Packet每包一次put+quiet。独立路由/计数commit不因单WQE
SO被删。全rank更新完整wheel；具体实现及验证边界见tests/ll_rdma/ROCE_SO_PACKET.md。

详细记录及现有实现区别见[ROCE_SO_CONTRACT.md](tests/ll_rdma/ROCE_SO_CONTRACT.md)。
