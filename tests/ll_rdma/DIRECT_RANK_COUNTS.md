> 当前count源已隔离，SendCnt无quiet；默认身份30、直方图身份31。旧身份及立即quiet描述为历史。见[COUNT_SOURCE_ISOLATION.md](COUNT_SOURCE_ISOLATION.md)。

# 多机源端计数、按跨机目标卡分核直发（协议28）

分支 `kernel/ll-ndr-ab-per-core-qp`。协议27/7293457是原EP32直发实现；协议28移除
两机专限。完整清单与风险见 [MULTI_HOST_LIMIT_AUDIT.md](MULTI_HOST_LIMIT_AUDIT.md)。
EP32配对路径的历史对照为 `d251739`（协议26）。
按用户要求参考 main 的源端 count → 最终专家卡方案，删除 EP32 配对路由表交换、
配对卡重算及其向本机最终目标卡的远端源 count 分发。不是删除所有机内通信。

## 分工与 Scalar 工作

- NDR/RDMA跨物理机均启用，现有多机域EP32/64/128/256。每机16卡，压缩目标空间跳过本机，
  task=coreId、coreId+C等分配给物理核。核数足够时一核一卡，不足时按同一分工分轮直发；
  大B也不受原本地计数参与阈值影响。实际跨机核数仍受锁定SDK每peer最多56QP限制。
- 每个remote owner只读取本源的完整B×K路由，统计该目标卡的全部本地专家。
  `CountRemoteRank` 对每个专家做Vector EQ比较及GatherMask，使用硬件返回的匹配数量，
  Scalar只写一个最终count。不执行按命中路由逐条GetValue/直方图读改写。
  E384/EP32每个remote owner为12个结果写入；这是减少Scalar工作，不是零Scalar。
  Vector仍按专家扫描路由，工作量随B×K及每卡专家数增长，是否更快需实测。
- 本机源到本机专家的count仍按原全局专家分片裁剪到本机范围，直接MTE写最终卡。
  各核先完成自己全部remote count帧，再执行分配到的本机任务；不重新均分本机专家。
  临时帧在每轮quiet后复用，不为更多机器新增活跃UB或lane。
- EP16/MTE保留原计数路径。一包暂存流水、每物理核QP和combine计算代码保持。
  多机首跳改为逐目标主机mask独立选择，不能共享token级firstK；q和原权重保持，
  但lead变化可能改变浮点累加顺序，参见审计中的数值风险，不能宣称逐bit一致。

## 布局、发布与生命周期

目标卡继续按expert-major/source-rank顺序生成prefix。物理count区仍预留E×512B：
同机源保留原分片帧；跨机源的每卡帧位于`source * localExperts * 512`，
body为`localExperts * 32B`，其后紧跟一个32B尾FLAG。每条body记录仍有ready/count。
尾FLAG即使全零count也发布。E384时每帧416B，每源16帧共6656B。

每帧一次RoCE put + 一次quiet，使用既有每核注册GM暂存及独立SDK scratch。
MTE3_S先保护GM暂存完成；目标WQE尾片SO及片内保序保证尾FLAG可见前body到达。
每帧不超过1056B，低于默认2KB分片粒度；依赖的仍是目标SDK单put/WQE契约，SDK不改。
quiet保护GM源/CQ退休，之后才复用work和scratch计算本机count；它不表示远端消费完成。
本机MTE帧仍保留body完成后发布commit，不能借用RoCE SO。

读取计划在源rank的16卡机界处分段，分别计算两种布局的地址，最大9段。
每次轮询先只读commit；到齐后读取body并恢复原逻辑顺序。片段总数汇合证明所有
接收核完成快照后，才由各记录消费者清body，并由帧首记录消费者唯一清commit。
prefix/各核ready矩阵及其等待仍保留，没有新跨rank/全核屏障或ACK。

双bank继续依赖同通信器、同调用次序、单stream：源端开始n+2前必须完成n+1，
其中收到了每个目标rank的n+1 count（含零）；该目标只能在n轮全部读者/清理退休后
开始n+1。因此n+2复用n的bank不会覆盖旧读者。combine沿用combine_reuse，
不能推广至并发stream或错误handle。

跨机帧暂存在既有workBuf，SDK scratch使用已退休的selected-ID输出缓冲。
workBuf容量取原路由scratch与整卡帧（按8条记录对齐）的较大值；quiet后整体可复用。
status保留区仍只容纳原本机分片，不把跨机帧长期保留到payload阶段。
没有新增UB buffer种类或lane/event，Host计数峰值同步核算临时work容量。
CPU逐核验证保留量与原本机分片一致；H3584/K8/E384预量化FP8、B1..256、
16/28/56/72核配置的活跃sender均仍可使用8个lane。设备性能仍须实测。

## 验证与目标测试

本地CPU验证执行生产分工/布局/transport/reader及实际CountRemoteRank函数体，
包含零计数、热点/无效ID、短路由与长路由、16..72核、延迟MTE3、破坏性SDK scratch、
SO尾可见性、共享commit清理、精确prefix及保留区边界。CPU模型不代表CANN编译或设备验证。
协议27历史分工模型覆盖59392组，transport/reader覆盖14832次发布和8256组快照，
实际Vector函数体覆盖768组路由/目标组合。整组回归197项通过、2项因缺少Torch跳过；
work缓冲复用调整后相关17项复验通过（与整组重叠，不相加）。

这是去掉中继阶段的方案，不预先宣称降低延迟：EP32相较d251739每源一次路由表发送，
现在每源16次跨机提交；一般域每源world-16帧，核数不足时分轮，网络字节和quiet成本均需实测。
比较同SDK/核数/shape/路由seed下的完整OFF dispatch，并检查BF16 combine正确性；
ON内部时间不能替代OFF性能。

所有目标机更新同一提交；NODE为0..NODES-1，MASTER沿用节点0地址。使用新run名，
下面仍是默认两机的E384示例；多机添加相同`--num-nodes "$NODES"`且E必须整除实际EP。
全域可用E1024的多机命令见审计指南。
全rank重编完整wheel，不使用旧wheel的`--skip-build`。失败只结束脚本，保留交互shell。

```bash
git fetch origin &&
git switch kernel/ll-ndr-ab-per-core-qp &&
git pull --ff-only

bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" ndr-direct-count-perf-001 \
  --mode perf --tokens 1 2 4 8 16 24 32 48 64 96 128 192 256 \
  --capacity 256 --hidden 3584 --topk 8 --experts 384 --dtype fp8

# 可另开完整reference构建，观察count_send/count_recv/prefix及payload lane。
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" ndr-direct-count-ref-001 \
  --mode reference --tokens 1 8 32 256 --capacity 256 \
  --hidden 3584 --topk 8 --experts 384 --dtype fp8 --reference-iters 5
```

采样协议仍为initial-reduce-exp200-batch-v3，reference schema4不变。
完整原始结果留目标机，按原入口回传单个≤60000字节报告。
