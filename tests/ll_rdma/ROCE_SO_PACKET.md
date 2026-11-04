# RoCE SO整包发布（核心协议10）

> 2026-10-28更新：当前协议14，路由表同WQE尾FLAG发布，每对跨机卡共8个QP；本文旧QP/独立commit描述作为历史记录。当前实现见 [ROCE_SO_ROUTE_8QP.md](ROCE_SO_ROUTE_8QP.md)。

后续采样更新：当前ST已按用户要求切换为`initial-reduce-sleep-batch-v4`，每批初始一次
all_reduce、active profiler.step前调用一次`torch.npu._sleep`，循环内不再插入采样通信。
旧协议9/10的历史结果若采用逐轮Exp+all_reduce，不能直接与新结果作为同条件A/B；
比较SO kernel时，两版须使用一致的新采样脚本重测。参见[采样协议](../ll_probe/PROFILING.md)。

2026-10-27，基于用户提供的[目标SO契约](ROCE_SO_CONTRACT.md)，优化dispatch和combine
跨机packet发送。对照源码为`dea6124`。SDK固定8e3fc610，不改SDK实现或QP配置。

## 实现

现有wire每512B为480B body+32B flag，整个packet的最后32B已经是flag。
本次直接复用这块尾FLAG，同一WQE发送整个packet。内部flag保持原位，既有Compact/KeepFlags
标记、机内MTE/relay、接收拆包与公开输出均不变；不额外追加重复FLAG，也不扩大包长。

dispatch `DispatchRdmaBatch::Enqueue`：

1. 原UB packet搬到本核已注册的GM暂存槽，保留MTE3_S，保证NIC源数据已完成。
2. 一次`Put`提交完整packet，包括原有尾FLAG。不再复制首gate、Scalar清gate或另发gate。
3. 满批或发送收尾时，每个不同peer执行一次`Quiet`。全部完成后才能复用GM槽。

批量容量为`min(8, floor(128KiB / tokenStride))`。仍保持最多8个在途包、本核独立QP，
不新增UB、lane、event、暂存总容量或全rank屏障。删除额外gate副本后，128KiB极限单包
也直接使用容量1，不需要旧fallback；满批Flush仍在当前包完成暂存之后执行，避免SDK
scratch破坏尚未保存的UB内容。尾批原有PIPE_ALL保护所借UB lane的机内发送读者，继续保留。

combine等同步`LlTransport::Packet`跨机路径同样变为整包一次Put＋一次Quiet。
`preservePacket=true`时仍恢复SDK可能破坏的前128B；单目的combine保留原false选择。
跨机只用put，机内分支仍直接DataCopy，不执行SO发布或CQ等待。

## 为什么接收端仍安全

接收端原有检查覆盖每个有效packet的全部flag，包含最后32B。
目标RoCE的尾片SO保证前面的分片先到达，片内保序保证尾FLAG之前的数据先到达。
因此，当完整flag检查通过时，该消息全部body/metadata已经到达。
首个flag中的1.5标记仍只用于KeepFlags，不把它单独当作就绪条件。
combine也继续检查有效返回packet的各块flag，mask无效槽的零值规则不变。

不要求不同WQE、不同peer或不同QP保序。接收可在发送端Quiet返回前观察到就绪，
但这不能授权发送端覆盖源GM：每批/每包的保留Quiet依然负责源读取与CQ退休。
旧flag清理、同Session同stream顺序、双bank以及combine_reuse协调均保持。
本次不会放宽错误handle、并发stream或任意跨调用顺序。

锁定SDK源码的调用链为`aclshmemx_roce_qp_put_nbi`→`aclshmemi_roce_write`→
`aclshmemi_hns_1825_post_send_read_write`：完整message_len进入一个work request，
构造一次WRITE WQE，SQ head递增一次。≤128KiB packet域没有软件分片循环；
网络默认2KB分片由设备执行。SO语义来自用户给定的目标硬件契约，未由CPU模型证明硬件属性。

## 工作量变化

| 路径 | 原方案 | 协议10 |
|---|---|---|
| dispatch每包 | 整包put＋32B gate put | 整包1次put |
| dispatch每批N包、P个peer | 2N次put、2P次quiet | N次put、P次quiet |
| combine每包L个512B块 | 2L次put、2次quiet | 1次put、1次quiet |

H3584/K8预量化FP8 dispatch仍为4096B包，去掉额外32B gate传输；BF16 combine仍为7680B包，
每包从30次put降到1次。以上是提交工作量，不是实测加速比，SDK自身SQ回压仍可能等待。

旧timer schema3的`send_rdma_body_post`现在包含整个packet；`body_wait`为消息源/CQ完成等待。
`flag_post/flag_wait`在这些packet路径不再有记录，不能把缺失阶段当成单独的性能收益。
reference原有29项边界不变（新增w44–w47四个计数延迟探针后共33项），其“尾batch收尾”包含新的消息quiet。

## 范围和部署

路由表、计数frame及strided Records的body/commit目前是独立区域或头部ready。
它们仍保留原发布顺序；不能依靠单WQE SO直接去掉跨独立WQE的依赖。
本次优化的是所有经过批量sender或Packet的跨机payload，不是控制区格式重构。

核心协议升为10以区分SO发布语义；全部rank拉取并重建完整wheel，拒绝混装协议9。
launch参数及wire布局不变，stage schema3/reference schema1不变。匹配SDK无需重建。

两机加载CANN，在相同路径的工程根执行，节点1将最后的0改成1，替换节点0真实IP：

```bash
ndr_so_perf() {
    git pull --ff-only || return
    bash tests/ll_rdma/one_click.sh "$1" 192.0.2.10 ndr-so-001 \
        --mode perf --tokens 256 --capacity 256 --hidden 3584 \
        --topk 8 --experts 384 --dtype fp8
}
if ndr_so_perf 0; then
    printf '%s\n' 'SO packet test completed'
else
    printf '%s\n' 'Test failed; project logs retained'
fi
```

这是EP32、dispatch+BF16 combine、完整OFF、10预热/100采集/尾50；不加`--skip-build`。
每次换新run名，保留所有rank原始记录。与基线对比时固定SDK、shape、核数、路由和采样。
只需打点时使用已有`--mode reference`独立重建/采样，它仅测dispatch，不代替OFF性能。

## 验证

CPU模型直接编译生产helper，按2KB分片模拟非尾片逆序到达、尾片SO顺序、片内逐word可见；
检查未完整到达时不提前就绪、Compact/KeepFlags、零包、尾批、不同peer/QP、16..72核、
128KiB边界、立即覆盖UB、CQ失败不回收槽，以及到达早于CQ退休。
同步packet重复轮次保留旧payload、仅清flags，覆盖实际窗口复用前提。
不满足SO时先交付全部flags的反例可提前误判，明确该实现不适用于任意乱序传输。
CPU/编译验证不代表目标设备正确性或性能已经通过。

2026-10-27验证记录：整合CPU套件365项通过、2项因无Torch跳过，另13个子测试通过。
该集合包括控制区/双bank延迟消费者模型、14种dispatch选择、combine及参考打点回归。
补充测试还覆盖combine仅观察末flag首word即可就绪时，前面所有body已到达的情况。
协议10的CANN编译尚未完成：经当前授权尝试27隔离目录预检，两次SSH连接均被27关闭，
未开始上传/编译，也未运行NPU。旧协议9的编译记录不能视为本版本的CANN验收。
