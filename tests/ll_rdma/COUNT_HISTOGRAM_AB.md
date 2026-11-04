> 当前默认身份34、直方图身份35：dispatch按GM stage容量批处理，combine回传批处理并将尾批CQ移至M5之后。见[COMBINE_BATCH_RETURN.md](COMBINE_BATCH_RETURN.md)。下方旧身份为历史。

> 当前count源已隔离，SendCnt无quiet；默认身份30、直方图身份31。旧身份及立即quiet描述为历史。见[COUNT_SOURCE_ISOLATION.md](COUNT_SOURCE_ISOLATION.md)。

# 跨机count硬件直方图编译对照

`DEEPEP_LL_COUNT_HISTOGRAM=OFF`（默认）：保留逐专家Compares/GatherMask/V_S，身份28。
`DEEPEP_LL_COUNT_HISTOGRAM=ON`：使用MicroAPI::Histograms FREQUENCY，身份29。
这是编译选项，不是运行中切换；两机/所有rank必须用相同值并重编完整wheel。
通信wire、每目标一put+quiet、q/权重、首跳策略、SDK及本机计数算法保持。

所有核进入SendCnt，但分配任务不同：EP32核0..15额外各负责一张跨机卡；本机专家仍
按全局专家区间分片后裁剪到本机。Rank0/E384/56核/B<512时核0..27有本机任务，
核28..55没有计数发送任务；核0..15兼做远端整卡统计。不能把调用相同函数当作工作量相同。

## 算法与边界

OFF每个目标专家读一次匹配数量，再用SetValue填入32B记录的count字段。
SetValue不是硬件要求，但这些字段相距32B，不能直接用连续数组替代。
ON仍保留每专家一次最终GetValue/SetValue，先隔离统计收益，不同时引入新散写布局。
核心变化是移除逐专家扫描和V_S等待：一个目标卡只在直方图结束后等待一次。

ON将路由ID用Vector Maxs/Mins/Adds归一到1..L（L=E/EP），区间外映射0或L+1。
先限幅再减first，INT_MIN/MAX也不会发生加减溢出。把归一化int32作为字节输入统计：
每个有效ID只有一个非零字节，其余三个0字节只增加被丢弃的bin0，和字节序无关。
所有有效bin<=32，故只需BIN0的128个uint16结果（256B）。不直接截断全局专家ID，
否则E>256会混桶。已有190KiB计数UB准入保证4*routes<=48640，包含bin0也不溢出uint16。

复用subExpBuf归一化、dstExpBuf存直方图、workBuf存最终帧；三个缓冲原本均存在且
最小256B。V_S后填frame，原S_MTE3/GM暂存完成/put/quiet及下一轮S_V依赖保持。
不新增UB/lane/event存活数量、不新增业务屏障；无CANN/NPU验证，不预先宣称加速。

接口依据：锁定SHMEM示例`examples/shmem_mega_moe/kernel/dispatch/dispatch_common.h`
的GetExpertFreq/HistogramsVf，以及
[Ascend Histograms说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/histogram_compute/Histograms.md)。
实现使用当前目标dav-3510的256B VF寄存器和既有MicroAPI/VF_CALL形式；目标CANN编译兼容性待验证。

## 同源码测试

先在所有节点更新当前分支。NODE为节点号，MASTER为节点0地址；下例EP32/E384对应本次分析。
所有节点完成OFF后再一起切到ON。不要使用旧wheel的--skip-build，不重建SDK，run名必须不同。

```bash
git fetch origin &&
git switch kernel/ll-ndr-ab-per-core-qp &&
git pull --ff-only

# 所有节点先运行OFF，完成后再运行ON。失败保留当前交互shell。
DEEPEP_LL_COUNT_HISTOGRAM=OFF bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" count-hist-off-001 \
  --num-nodes 2 --mode perf --tokens 1 8 32 256 --capacity 256 \
  --hidden 3584 --topk 8 --experts 384 --dtype fp8

DEEPEP_LL_COUNT_HISTOGRAM=ON bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" count-hist-on-001 \
  --num-nodes 2 --mode perf --tokens 1 8 32 256 --capacity 256 \
  --hidden 3584 --topk 8 --experts 384 --dtype fp8
```

逐核诊断可分别使用新run名、改成`--mode reference --reference-iters 5`，仍完整构建相应wheel。
比较count_send和完整OFF dispatch，不能只看某核统计变短；FP8配置的combine仍是BF16。
采样initial-reduce-exp200-batch-v3不变，原始数据留目标机，回传单个<=60000字节报告。

构建配置/缓存key和_build_info记录ll_count_histogram，Host与Device身份28/29互检，
one_click核验请求值/安装身份/元数据并在跨节点握手中比较count_histogram字段。
仅设置运行环境变量不能改变已经编好的kernel。

CPU检查446项通过、2项缺少Torch跳过。直方图测试执行生产归一化与VF循环，覆盖EP32/64/128/256、
随机/热点/全零命中/INT_MIN/MAX、短尾、12160路由及UB哨兵；MicroAPI由CPU语义模型替代。
这些检查不等同于CANN编译、设备正确性或性能通过。
额外构建工具回归49项通过、8项在Windows的Unix编译器查找/符号链接fixture失败；
对应编译器查找函数未修改，不能把这些环境失败计作通过，也未完成Linux/CANN构建验证。
