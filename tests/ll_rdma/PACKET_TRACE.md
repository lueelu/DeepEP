> 当前为一包深度暂存/put流水，实现身份26、reference schema4；复用原lane，单lane立即提交。S表区分暂存经过、延后间隔、剩余等待及前包提交，不改packet所属核/顺序。详见 [ONE_PACKET_PIPELINE.md](ONE_PACKET_PIPELINE.md)；覆盖下方旧立即提交及schema2说明。

# Reference schema2：S核第二跨机包与R核包顺序

2026-10-28，当前kernel/ll-ndr-ab-per-core-qp。通信协议仍24，诊断schema从1升级为2；全rank更新完整wheel。只在DEEPEP_LL_REFERENCE_PROFILE=ON且当前capture有记录地址时记录；OFF的发送调度、q算法、QP、UB/lane布局和通信协议不改。SDK不改。

## S核定义

第二包是每个S核本次dispatch实际提交的第2个跨机packet，不是第2个token、路由或WQE分片。少于2包时第二包字段为空，不能补0。记录token、header_k、专家及peer。

| 指标 | 边界 |
|---|---|
| 第二包调度扣q | SendPacket入口到Enqueue入口，减包尾成员q区间；主q在进入SendPacket之前，天然排除 |
| 第二包成员q | 仅第二包FillTail里的成员q调用区间累计，不是本次kernel全部q |
| 第二包暂存/提交前等待 | Enqueue入口到put调用前，包括注册GM暂存和原MTE3_S等 |
| 第二包put调用 | put调用前到返回；不代表远端消费完成 |
| 第二包发送提交 | Enqueue入口到put返回；不包含由该包触发的整批quiet |
| 跨机整体提交 | 本核首个跨机包SendPacket入口到末跨机put返回 |
| 跨机源/CQ退休 | 上述首开始到末次原有batch Flush quiet返回；不是接收端消费完成 |
| 本组整体提交 | 本核首个本组包SendPacket入口到末本组提交返回；不额外加MTE3完成等待 |
| 批次quiet累计 | 原有每次Flush的quiet区间之和，附批次数和第二包是否触发满批 |

本组按目标rank/Group等于源rank/Group判定；跨机按原RemoteWrite判定。两类包保持原发送顺序，可能交错，所以整体区间可能包含另一类工作及后续q计算，不能相加，也不能称链路线速。只有第二包调度字段扣除其内部成员q；整体区间不扣除区间内全部q。

主q不计时；成员q前后clock只对第二包启用。为在不额外预扫描路由、不重排发送的条件下得到最后提交时刻，每个实际跨机或本组包提交后读一次clock并覆盖本地末时点；未对其他包保存细分日志。批次quiet沿用原调用位置，不新增quiet。S扩展字段留在本地Scalar状态，Finish时集中写GM。

## R核记录

只记录成功探测发现就绪的packet，不在空扫描里插点或逐次记录。沿用原每成功批次3次clock及两段累计。R核不增加逐包clock，因此不能由包顺序表直接计算单包处理时长。

每条32B记录包含：发现序号、成功就绪批次、处理序号、原始源rank、源token、header_k、原始包角色及转发成员数、稳定槽位。以稳定任务j索引，pending数组换位不改变身份。由槽位解出目标专家和q，并结合本核输出行起点给出输出行。

- 发现序号是成功Probe快照的遍历顺序；同批按快照下标递增记录，不是NIC真实到达顺序。
- 处理序号是ConsumeLoaded读取已加载包之后的记录顺序；原代码同批倒序消费，因此可能与发现顺序不同，不是丢包。
- 在FIRST_HOP改写成Terminal之前保存角色和成员数。
- Terminal头中的header_k是原FIRST_HOP的返回k，不保证是该成员在原top-k中的位置。用源rank/token、目标专家/q识别目标任务。
- 日志仅在原receiver全部业务缓冲及lane分配之后使用剩余UB。最多128条，实际容量min(任务数,128,剩余UB/32)。不降低原lane数，不改变Host业务UB准入。
- 空间不足时记录稳定j前缀，保留全体发现/处理计数及真实序号，标明截断；角色/成员汇总只代表已记录部分。
- 记录过程中只有UB Scalar写，不逐包写GM。每核原业务w21之后，用一次DataCopy导出有效日志，诊断专用S_MTE3/MTE3_S只发生在此阶段。

## 口径及扰动

原29项阶段仍保留，另加w44–w47四个计数延迟探针（仅reference路径）共33项，核总耗时w21-w0不包含本核日志导出。写出前总耗时w57-w0现在包含导出；R导出本身另记w103-w102。新旧写出前耗时不能直接当作相同边界。

“本核业务之后”不是全rank共同结束：其他核可能仍在工作，日志DMA及末尾诊断写回可能与它们竞争资源。S的新增clock、Scalar状态/寄存器，R的UB写和成功批次遍历也可能扰动时序。不增加业务屏障不等于零扰动，必须与同源码OFF完整task对照；CPU测试不证明CANN资源使用或目标性能。

记录每核640个uint64=5120B（80×8，仍512B对齐）：前64word保留旧主阶段，64～95为S元数据，96～127为R覆盖/导出元数据，128～639为最多128条32B日志。R日志由DMA写入，Scalar头部缓存写回不覆盖日志区。旧schema1拒绝混装/混解析；旧数据可用原提交分析，原始文件不覆盖。

## 一键运行

原one_click入口及mode不变。两机同一新提交，完整重编reference wheel，不加--skip-build；使用新的run名：

```bash
ndr_reference() {
    git pull --ff-only || return
    bash tests/ll_rdma/one_click.sh "$1" 90.90.93.60 ndr-ref-packet-schema2 \
        --mode reference \
        --tokens 1 2 4 8 16 24 32 48 64 96 128 192 256 --capacity 256 \
        --hidden 3584 --topk 8 --experts 384 \
        --dtype fp8 --reference-iters 5
}
```

节点0/1分别调用ndr_reference 0 / ndr_reference 1。五次capture仍使用同一份声明seed路由，默认Excel和顺序CSV取末capture；可用--analyze-only RUN_DIR --output-dir NEW_DIR --iteration N重导出其他轮，不需要重跑NPU。

## 输出和检视顺序

原core_stage_timings.xlsx前四页保留；新增：

1. S包时序：第二包细分、整体区间、交错标记、quiet批次数。
2. R包顺序：每核每个成功就绪批次一行，各包按发现顺序组成元组，另列实际处理序号。
3. 包记录覆盖：任务数、记录数、截断、原lane数、导出耗时和已记录角色/成员汇总。
4. 包原始记录：扩展元数据原始ticks，超过Excel精度阈值写文本。

同目录新增sender_packet_timing.csv、receiver_packet_order.csv、receiver_ready_order.csv、packet_trace_coverage.csv、packet_detail_raw.csv。就绪/包顺序CSV按成功批次归并；逐包处理顺序和完整字段另存receiver_packets_detail.csv。完整JSON保留所有capture的原始记录；整轮回传报告仍≤60000字节，只增加覆盖摘要，完整包表留目标机。

先检查覆盖是否完整，再用R表定位晚发现的来源/角色，以及FIRST_HOP成员数。用S第二包判断准备、暂存、put提交中的耗时分布；整体区间看核内发送推进，不跨rank直接相减时钟。不要把没有时间戳的R记录推导成精确到达时间或逐包时延。

本地验证覆盖第二包、满批quiet分离、稳定任务与pending换位、截断、schema身份、整数tick精度和Excel/CSV导出；尚未执行CANN编译或NPU验证。

## R包顺序按扫描批次展示

R包顺序页、receiver_packet_order.csv及receiver_ready_order.csv现在每核每个成功就绪扫描批次一行。先列本次就绪包数，再将包类型、源rank、源token分别组成按发现顺序排列的元组，另列(类型,源rank,源token)组合元组、稳定任务j和处理序号元组；同一位置代表同一包。空扫描从未记录，不补造空扫描行。日志截断时实际就绪包数留空，明确显示已记录包数和记录不完整；即使某批没有保留下来的包，也保留该批次行。逐包完整字段另存receiver_packets_detail.csv。

DEEPEP_LL_NO_COMM目前仍会改变源码路径：定义后绕过DispatchRdmaBatch，进入LlTransport::Packet逐包put+quiet，并非不通信。一键构建未启用此宏，本次保留，不能将它当作无通信基线。
