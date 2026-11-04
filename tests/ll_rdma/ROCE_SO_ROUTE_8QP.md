> 当前协议14：在协议13对照上独立恢复路由SO。独立回退与A/B见 [ROUTE_SO_AB.md](ROUTE_SO_AB.md)。以下协议12表示8QP分布方案的历史来源，当前运行身份为14。

# 协议14：独立路由SO与每peer的8核集合

2026-10-28。协议11误将所有peer固定到S0～S7；协议12按用户纠正，改为每个peer独立的8核集合，跨peer覆盖完整S核池（56 AIV时为28个S核）。路由SO保持协议11实现。SDK保持原实现，所有rank必须更新完整wheel，旧协议不能混装。本文覆盖旧指南中“每AIV一个QP”和“路由表body/commit两次put”的当前行为描述。

## 1. 配对路由表：一帧一次WQE

EP32配对peer仍为 `rank ^ 16`，每rank只有core0发布本源路由表。

消息为 `align32(B*K*4) 字节路由数据 + 32B尾FLAG`，一次put对应同一WQE。32B FLAG为四组交替的 `0x52544553, epoch`；接收端检查全部8个word后才加载路由。默认2KB是网络分片粒度，不限制消息总长。

`PostFrame`先将body和FLAG搬到独立、已注册的TX GM槽，保留生产完成依赖再提交整帧。FLAG复用计数阶段现有work UB，不新增常驻UB。删除独立commit put和发布前quiet；`FinishSend`仍做一次quiet，确保源GM和QP0完成后core0才进入payload发送。全流程每rank每轮为 **1 put + 1 quiet**，不是发送后立即允许复用源。

接收RX槽和TX槽按最大body加32B FLAG预留并对齐512B，实际发送长度按本轮B/K确定。epoch仍单调递增，双bank与原combine_reuse协调保持。B/K变小时，新尾部可能落在旧body中；固定标识大于合法expert ID范围，因此不能把恰好等于当前epoch的旧expert ID误当FLAG。旧FLAG不靠简单非零判断，也不需要新增reader ACK。

尾FLAG可以在发送CQ退休前可见；即使随后CQ报错，也不能因此复用源GM。CPU模型验证的是发布与生命周期逻辑，不证明硬件SO属性。SO属性来自用户提供的目标契约；不推导跨WQE或跨QP顺序。

## 2. 每对跨机卡总共8个QP

Session对ROCE配置 `aclshmemx_set_qp_num(..., 8)`。实际AIV数仍为平台/设备查询结果，S/R仍各半；不会为8个QP将kernel整体减到16核。每核128KiB暂存仍按实际AIV数分配，独立于QP数；CommArgs使用原保留地址槽记录暂存结束地址，结构大小不变。

| 算子/阶段 | 跨机提交者 | QP |
|---|---|---|
| EP32配对路由控制 | core0 | QP0，进入本核payload前完成quiet |
| 其他跨机域的计数frame | S0～S7 | 本核对应QP0～7，先完成本核计数发布再发payload |
| dispatch token | 每peer从全部S核中选择8个不同核 | 该peer第i个包使用QPi%8，由映射的S核独占 |
| combine DIRECT回传 | S0～S3 | QP0～3 |
| combine partial回传 | R0～R3，即实际核 `AIV/2 + 0..3` | QP4～7 |

dispatch核集合按peer分散；例如rank0、EP32、S=28时，peer17的QP0～7由S24/S25/S26/S27/S0/S1/S2/S3拥有，peer18由S4～S11拥有。

配对路由peer的QP0保留core0。非配对计数路径的peer只对应一个计数生产者c=peer/(EP/8)，保留该peer的QPc由core c拥有；其余7个QP从排除该核的S核池中按peer轮转选择。这样保持每peer恰好8个不同拥有者，控制与payload可重叠而不会并发操作同一peer/QP。

combine两类回传并行，所以采用4+4互斥分配。所有其他核继续执行各自机内任务、计数/ready、最终聚合等工作。接收转发和combine组内ready仍是机内MTE，不消耗RoCE QP。算子间沿用单stream串行和原完成依赖复用QP，没有 `core_id % 8` 的并发共享，也没有引入QP锁或全核屏障。

计数fallback的生产者分片、接收frame映射和Host UB预算均使用一致的计数核数。EP32配对路径仅发送整张路由表，后续专家计数都是机内发布，因此保留原计数核分片。

## 3. 按实际目标rank轮转dispatch包

每张源卡独立枚举本次调用的原始 `(token, top-k位置)` 顺序。先应用原去重和FIRST_HOP选择，再仅对**实际跨机packet**按目标rank分别编号：

```text
发往rank x的第0、1、2、…、7、8、9个实际包
由发送核      O0 O1 O2 … O7 O0 O1  （O是该peer的核集合，不是固定S0～S7）
使用QP       0  1  2  … 7  0  1
```

这不是 `token_id % 8`，也不是整台机器16张卡共同维护计数。去重后只统计真正发往rank x的首跳或DIRECT包；首跳后由远端机内转发的成员不重复计入。不同QP无需按编号到达；接收仍依靠原目标槽位及完整flag检查。

原q表的低16位保持原专家路由前缀，高位分别暂存实际发送核和该peer的QP编号；既有qSub临时区用于每目标rank的计数，不增加常驻UB、lane或活跃事件数。传到wire的q始终去掉归属位。跨机发送核可能处理原连续token分片之外的token，所以其packet尾部权重读取原始、只读GM权重；本地sourceMask/sourceWeights仍由原分片唯一写出。FIRST_HOP选择、成员顺序、路由槽位、归约顺序和公开输入输出不变。

每个S核均参与本核的归属规划；即使B1的原token分片为空，也可能被某个peer选中，不能在规划前直接退出。只有source任务和remote任务都为0时提前返回。跨机包先按该归属提交，再处理原机内发送及source元数据。批次在原peer数组的高位保存QP编号，put和quiet都显式使用同一peer/QP，不以物理核编号代替QP。原批量GM源、每peer quiet、UB/lane复用和尾批收口依赖保留，不因SO删除生命周期等待。

combine partial的远端任务集中给4个核后，某核任务数可能超过原R核分片的列表容量。采用原UB列表内分页、必要时重新扫描元数据，不扩大UB或改变成员归约顺序。这会带来额外扫描成本；8QP和轮转的整体性能须看完整OFF测试，不能仅凭提交数减少认定加速。

## 4. reference表的解释

仍为dispatch的29项阶段、每核64word；没有新增clock、逐包打点或业务屏障。`w38`之前包含归属计划构造，`w38→w39`包含跨机归属发送及原机内发送。协议12的`w43`记录本核source路由/元数据任务数加实际分配的remote packet任务数，用于识别没有原token分片但确有远端发送工作的核。它是工作量计数，不是去重后全局包数；原始word名称暂保留`send_route_count`，不能与协议11的该计数直接等同。

协议、源码、负载构造和QP数都写入运行身份。默认负载仍为 `balanced-rank-distinct-roundrobin-v1`；采样仍为 `initial-reduce-exp200-batch-v3`。阶段为ON诊断，完整OFF profiler task才用于性能结论。

## 5. 本地检查与目标机命令

CPU检查覆盖实际packet归属函数的EP16/32/64/128/256、B1/2/17/256、K1/8/16/32和热点；验证原去重、FIRST_HOP、q不变，每目标rank按其8核集合轮转且无漏包/重复，并检查8～36个S核的完整核池覆盖、控制QP拥有者及B1高编号发送核。还覆盖真实FillTail的全局权重/q编码、combine的实际选择/分页方法、原UB边界、SO变长帧/延迟消费者/双bank及CQ失败保留源。Host预算覆盖16～72个偶数AIV。检查不代表CANN编译或NPU正确性、性能已通过。

两机相同工程路径、提交和参数，替换节点0的真实IP。已有匹配SDK可继续复用；不要加 `--skip-build`，也无需因本改动强制重建SDK。

```bash
# 两机先分别执行
git pull --ff-only

# 节点0：reference诊断与正确性
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-peer8-reference-001 \
  --tokens 1 8 32 256 --dtype bf16 fp8 --reference-iters 1
# 节点1：相同参数、相同run名
bash tests/ll_rdma/one_click.sh 1 192.0.2.10 ndr-peer8-reference-001 \
  --tokens 1 8 32 256 --dtype bf16 fp8 --reference-iters 1
```

完整dispatch+BF16 combine验证和性能：两机分别将节点编号设为0/1，使用新的run名：

```bash
bash tests/ll_rdma/one_click.sh 0 192.0.2.10 ndr-peer8-perf-001 \
  --mode perf --tokens 1 8 32 256 --dtype bf16 fp8
```

需要热点回归时另换run名并加 `--route hotspot`。reference的完整Excel留目标机，仅回传run级 `table-analysis/return_report.md`（不超过60000字节）。所有原始样本、日志保留。
