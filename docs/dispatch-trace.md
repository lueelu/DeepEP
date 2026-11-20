# V2 fresh dispatch 通信 kernel 采集

在 Ascend950 Global Clos 上，给训练进程设置 `DEEPEP_DISPATCH_TRACE_DIR` 即可开启。目录应由当前作业独占并允许各 rank 写入。普通调用方式与上游相同。采集范围为 eager V2 fresh dispatch 的 `v1_dispatch_udma_kernel`，包括该 kernel 内部的元数据和输出处理。notify、layout、expand、cached dispatch、combine 与图重放均在范围之外。

## 运行与校验

按[构建指南](build.md)在含匹配 CANN、Torch、torch_npu 和 SHMEM SDK 的环境安装本分支的 wheel。Host 与 Device 两部分均需重新构建。在单机四卡上可使用现有 benchmark 验证。

```bash
set -e
RUN="$(mktemp -d "$PWD/dispatch-trace-XXXXXX")"
export DEEPEP_DISPATCH_TRACE_DIR="$RUN/trace"
python -u benchmarks/clos/bench_v2_bandwidth.py \
    --ep-size 4 --nnodes 1 --aiv 32 --dtypes bf16 --expand 0 \
    --num-tokens 2048 --hidden 6144 --num-topk 8 --experts-per-rank 32 \
    --warmup 2 --iterations 5 --no-flush-l2-cache --output "$RUN/bench"
python scripts/check_dispatch_trace.py "$RUN/trace" --ranks 4
```

每个文件名包含主机、PID、设备和文件创建时间。校验器检查各 rank 是否有成功完成的 dispatch、launch 与 end 是否配对、时钟换算是否一致，以及是否丢样。基准自己的预热或准备也会调用 dispatch，因此每个 rank 的记录数可以超过 iterations。性能验收需使用相同输入、资源和启动方式做开关交替对照，并同时记录丢样数。

## 时间含义

| 记录 | 来源及含义 |
| --- | --- |
| calibration | 首个启用采集的 UDMA Buffer 在当前 ACL context 创建采集器，短 kernel 读取设备 tick。Host 使用私有流包围这次采样，同步一次，保存主机单调时钟窗口的宽度和墙钟偏移。共享该采集器的 Buffer 共用同一偏移。 |
| launch | Host 在调用 UDMA 启动封装前读取 Unix 墙钟，单位 ns。封装返回后把时间与当前线程的 ACL 状态交给日志线程。 |
| end | 同一个 gid 的完成记录。每个参与 AIV 在本核通信实现的入口和尾部读取 tick。start_tick 取各核最小值，end_tick 取各核最大值，覆盖本 rank 参与核的执行区间。 |
| launch_error / incomplete | 提交路径报告异常，或在采集器关闭时仍缺少有效 tick。这些记录保留 gid 以便定位。 |
| summary | 正常关闭时汇总已完成数、未完成数和队列竞争导致的丢样数。 |

Ascend950 按 1 GHz 换算，令校准核采样为 $C_0$，Host 单调时钟包围采样的中点为 $M_0$，另一次相邻采样所得墙钟和单调时钟为 $W_a,M_a$，所有 Host 时间使用 ns。

$$
O=(W_a-M_a)+M_0-C_0,\qquad W(C)\approx C+O
$$

`start_ns` 和 `end_ns` 为上述参考墙钟。`elapsed_ns = end_tick - start_tick` 独立于墙钟偏移。跨 AIV 取极值的前提是各核 GetSystemCycle 共享计时基准；设备验收需与独立 profiler 对照。校准窗口宽度与设备速率漂移都会影响跨时钟比较，launch 到 start 的差值用于参考，不能作为精确排队时延。

## 上报和资源

每个活跃 ACL context 共享一个后台线程与一个 128 槽的 Host 映射缓冲区，常见的一进程一卡配置对应一个线程。每个槽按 64 个 AIV 预留空间，每个 AIV 独占两条相距 128 字节的记录，整个环约 2 MiB。AIV 使用 MTE3 分别发布起点和终点，复用 dispatch 的 32 字节 UB 暂存区并等待本地写出完成。业务调用路径仅预留槽、读取 Host 时间并发布提交结果。后台线程约每 2 ms 检查一次，只有全部参与核的起止 tick 就绪才回收槽。启动记录可在通信仍执行时落盘。

槽正在使用时，当前调用继续执行，丢样计入 summary。日志写入失败会输出诊断。最后一个共享 Buffer 释放时同步设备并排空日志；被强制终止的进程可能缺少最后一批记录。gid 在进程内分配，文件内的 launch 与 end 按 gid 关联。不同进程可出现相同 gid，本地 instance 编号也不能用于推断跨 rank 的 collective 配对关系。
