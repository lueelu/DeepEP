<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 测试入口

从仓库根目录运行以下命令。CPU 测试不占用 NPU；设备系统测试使用已安装的 wheel，安装步骤见[构建指南](../docs/build.md)。测试通过的范围以实际执行的入口、配置和 rank 结果为准。

## 入口与用途

| 入口 | 检查内容 |
| --- | --- |
| `api/` | 公开导出、签名、参数默认值和未支持操作的错误 |
| `runtime/` | 构建预检、SDK/安装身份、资源和事件生命周期、故障传播 |
| `elastic/test_dispatch.py` | HT Notify 元数据、BF16/FP8 payload、权重、fresh/cached 和路由 |
| `elastic/test_combine.py` | HT dispatch → 专家输出 → BF16 combine、独立权重回传、重复调用及输出生命周期 |
| `elastic/test_ep_low_latency.py` | LL 返回、精度、重复 combine，以及 profiler kernel 耗时 |
| `elastic/test_low_latency_host.py`、`elastic/test_ll_st_contract.py` | LL Host 分支、计数通信、启动参数、测试参考及统计 |
| `integration/` | CPU/NPU 测试适配器、安装检查和结果校验 |
| `../experiments/megamoe/tests/` | MegaMoE 接口、Golden、主机协调、打包与计时解析；设备入口见 [MegaMoE 文档](../experiments/megamoe/README.md) |
| `../experiments/ep_gmm_fused/tests/` | 通信计算融合算子的路由参考、精度、性能与 profiler；命令见[组件测试说明](../experiments/ep_gmm_fused/tests/README.md) |
| `test_reference.py`、`test_precision.py`、`utils/` | 独立路由参考、Golden、dtype 模拟和精度比较器 |
| `legacy/` 及通用 V2 设备用例 | 兼容性测试入口；其参数矩阵不等于当前 Elastic 算子支持范围 |

## CPU 回归

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -m "not framework" -q
```

按变更选择测试文件，例如：

```bash
python -m pytest tests/api tests/elastic/test_low_latency_host.py tests/elastic/test_ll_st_contract.py -q
python -m pytest tests/test_reference.py tests/test_precision.py tests/test_benchmark.py -q
```

普通 pytest 不会自动执行设备 ST。Host 故障注入使用 C++ 编译器；combine 的部分 Host 回归也通过 C++17 编译器或已安装的 ziglang 检查 kernel 判断函数，缺少编译器时相关项跳过。这些检查不验证 AIV 并发、真实 SHMEM 通信或设备性能。

## HT 单机测试

先加载 CANN/SHMEM 环境，在已分配的设备上运行。下面的设备编号、地址和端口为示例，使用前替换为作业资源。输出目录使用新路径。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091

python tests/elastic/test_dispatch.py --ep-size 8 --nproc-per-node 8 \
  --tokens 257 --topk 6 --experts 512 --dtype both --route all --repeat 3 \
  --output /tmp/dispatch-check-001

python tests/elastic/test_combine.py --ep-size 8 --nproc-per-node 8 \
  --tokens 257 --topk 6 --experts 512 --route all --repeat 3 \
  --output /tmp/combine-check-001
```

HT 脚本直接启动本机进程并使用 SHMEM，不创建 Gloo/HCCL 进程组；也支持外部 launcher 提供 rank 环境变量。Notify 验证包含在 fresh dispatch 中。

combine 的专家输出固定 BF16、H=7168；其前置 dispatch 可用 `--dispatch-dtype bf16/fp8` 选择，默认 BF16，不使用 `--dtype` 或 `--hidden`。其 `--route all` 包括 normal、random、duplicates、invalid、empty、local_duplicates、local_experts；dispatch 另含 hotspot。`empty` 表示所有路由无效，所有 rank 仍参加通信。combine 不包含超过固定工作区上限的全局极端热点，容量约束见 [Elastic 接口](../docs/elastic.md)。

combine 默认同时检查独立权重回传；`--without-weights` 关闭该通道，不改变 hidden 的归约语义。

HT 脚本也支持 kernel 测量：dispatch 入口增加 `--profile-op notify` 或 `--profile-op dispatch`，combine 入口增加 `--profile-op combine`。`--profile-warmup` 默认 5 次，`--profile-iters` 默认 10 次。先通过正确性检查，再采集各 rank 的 profiler 记录；计时范围和输出文件见[性能说明](../docs/performance.md#ht-kernel-耗时)。

### Dispatch 性能测试的启动对齐

`test_dispatch.py --profile-op dispatch` 和 `run_peer_table_check.py` 的 cached Dispatch 性能采样默认开启 `--dispatch-barrier`，在每次预热和正式采样的 Dispatch kernel 前插入设备侧 SHMEM barrier。barrier 位于参数检查、输出分配等准备工作之后，与 Dispatch 使用同一通信流，减少各 rank 准备速度不同带来的启动偏斜。测试默认只插入前置 barrier，结束时仍依靠 Dispatch 自身的入向完成协议。

```bash
# 默认开启前置 barrier，测量 cached Dispatch kernel。
python tests/elastic/test_dispatch.py --ep-size 8 --nproc-per-node 8 \
  --tokens 4096 --hidden 7168 --topk 6 --experts 512 --dtype fp8 \
  --route random --profile-op dispatch --profile-warmup 10 --profile-iters 20 \
  --output /tmp/dispatch-perf-aligned-001

# 对照正常调用的启动行为：在同一命令中增加 --no-dispatch-barrier。
```

所有 rank 使用一致选项。`capture.json`、profiler 的 `summary.json`、peer-table 的 rank JSON 和 `kernel-summary.json` 都记录 `dispatch_entry_barrier`，汇总时检查各 rank 的开关是否一致。profiler 只统计 Dispatch kernel；peer-table Event 将 `entry_barrier` 和 `kernel` 分开统计，`device_total` 包含前置 barrier。peer-table 的 API 墙钟时间也包含该 barrier 的等待，比较结果时须保持选项和计时口径一致。

这是性能用例的配置，直接使用 `buffer.dispatch()` 默认仍不插入测试 barrier。正确性阶段、Notify 和 Combine 的测量不受该选项影响。自定义 HT 测试可使用内部诊断开关：

```python
# 在所有 rank 的性能采样阶段设置；实际模型调用无需设置。
buffer._dispatch_diagnostics = {
    "benchmark_entry_barrier": True,
    "kernel_timing": True,  # 收集分阶段 Event 时间；profiler 采样可设为 False。
}
try:
    output = buffer.dispatch(x, handle=handle, do_expand=True)
    output[-1].current_stream_wait()
    # kernel_timing=True 时，在诊断字典中读取 device_ms / device_phase_names。
finally:
    del buffer._dispatch_diagnostics
```

`_dispatch_diagnostics` 是内部测试钩子，直接设置在 buffer 上。`benchmark_barriers=True` 表示同时开启前后置 barrier；`benchmark_entry_barrier` 单独控制前置 barrier。开启 Event 计时时 native 返回前会等待计时流，profiler 用例设置 `kernel_timing=False`，保留批量异步提交方式。该设置用于比较启动对齐后的 kernel 性能，模型调用的时延应另外测量。

## HT 多机测试

四台主机、每台 8 卡运行 EP32。在各节点设置 `NODE_RANK=0..3`，`MASTER_ADDR` 为首台主机可达 IP，使用相同版本的 wheel、输入配置和 SHMEM endpoint：

```bash
export DEEPEP_SHMEM_ENDPOINT="tcp://${MASTER_ADDR}:19091"
python tests/elastic/test_dispatch.py \
  --ep-size 32 --nproc-per-node 8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 19090 \
  --tokens 257 --topk 6 --experts 512 --dtype both --route all --repeat 3 \
  --output "/tmp/dispatch-ep32-node${NODE_RANK}-001"
```

combine 使用 `test_combine.py`，将 `--dtype both` 改为 `--dispatch-dtype bf16` 或 `--dispatch-dtype fp8` 并更换输出目录。共享文件系统中，每台主机的输出路径包含 node rank；不要让多个节点写同一个目录。

HT 的 Notify、Dispatch 和 Combine 用例默认在 EP8/16 使用 rank 去重，其余规模使用 server 去重。`--rank-dedup`（别名 `--no-server-dedup`）和 `--server-dedup` 可显式切换。rank 去重保留同 rank 内归约，各 expert rank 独立回传，不执行 server 内 gather；server 去重在源 rank 命中本卡专家时优先由本卡中转。两种路径均与 CPU golden 比较；`--golden-notify` 可单独用于 CPU 建表对照。rank 去重与 `--with-dispatch` 同用时支持 EP8/16。

EP8 的 rank 去重 Combine 自动使用 MTE 路径，支持 weight 回传，无需额外选项。

对照时保持 EP、token、hidden、top-k、专家数、路由 seed、chunk 和 weights 一致，分别使用 `--server-dedup` 和 `--rank-dedup`。JSON 用 `deduplication=server/rank` 区分结果，并记录实际回传行数。带宽仍为逻辑数据量 `T*K*H*2 / duration / 1e9`（duration 单位为秒）；Notify、golden 建表及打点轮不计入 profiling 的 Combine kernel 时间。

Combine 的 `--local-expert-first` 将 top-k 中属于源 rank 的专家稳定前移，其余专家保持顺序，专家集合不变。该路由对照选项默认关闭，不是本卡优先中转的前提。JSON 中记录 `local_expert_first`，两组测试应使用不同输出目录。

`--without-replacement` 对每个 token 无放回选取专家；`--profile-flush-l2` 在每轮计时前冲刷 L2，清理耗时不计入 kernel 时间。profiling 默认预热 10 次、采样 50 次，逐 rank 保留全部时长。

`--tokens-list`、`--chunk-tokens-list` 在一次初始化中遍历多个 shape；Notify 可用 `--notify-num-cores` 选择 32/64 核，Dispatch 用例的 `--notify-only --notify-num-cores-list 32 64` 只测 Notify。各 rank 必须使用相同配置。

## HT peer-table 检查与计时

[run_peer_table_check.py](elastic/run_peer_table_check.py) 检查 BF16/FP8 dispatch 的路由和搬运结果，并提供独立计时入口。沿用前述 HT 的设备和 endpoint 设置，例如：

```bash
python tests/elastic/run_peer_table_check.py --ep-size 8 --nproc-per-node 8 \
  --tokens 257 --topk 6 --experts 512 --dtype fp8 --weights \
  --kernel-profile --routing-profile --output /tmp/peer-table-check-001
python tests/elastic/dispatch_metrics.py /tmp/peer-table-check-001
```

### Notify 与紧凑表 Event 计时

`tests/elastic/run_peer_table_check.py` 在原有测试参数上增加
`--kernel-profile --routing-profile`，会额外执行 fresh-handle 采样，
按 `--warmup` / `--iterations` 预热与记录，每次重新执行 Notify 和紧凑表构建。
这部分独立于复用 handle 的 dispatch 性能采样，不添加计时前全 rank barrier。
各 rank JSON 的 `routing_device_ms` 保存以下 Event 区间（毫秒）：

- `notify_device_ms`：Notify kernel、后置 barrier 与轮转/转发表准备、expert totals 保存、总时间（包含建表）。
- `prepare_device_ms`：Notify 内部的表清零、建表 kernel、末尾 Event 间隔、准备总时间；这是 Notify 总时间的子区间，不能重复相加。

建表 kernel 包括 peer-table、紧凑 forward plan 和原有 admission 同步；
不包含主机读取 admission 状态。Notify 和 prepare 均不包含 Python 分配与校验。
普通 API 不设置 `_routing_diagnostics`，不创建这些计时 Event。
运行 `dispatch_metrics.py <结果目录>` 后，`kernel-summary.json` 的 `routing`
按每次采样取各 rank 最大值，再计算均值与中位数。不同阶段的 rank 最大值不能直接相加作为端到端耗时。


### Peer-ready 持久 generation 协议

dispatch 不再逐次清零接收控制标志，
移除 payload kernel 的入口 `aclshmemx_barrier_all_vec`；地址发布的本地同步、
MTE 完成后再发布 generation 的顺序，以及出口完成协议保持不变。
64 位 generation 不复用，耗尽后要求重建 Session。

冷启动、布局变化、Notify/LL 工作区复用后的控制区初始化在 Host 侧执行；
缓存调用复用已初始化的控制区，普通 API 不逐次插入测试 barrier。
调用方通过返回 Event 连接后续消费，保持各 rank 的调用顺序和输入输出生命周期。
冷初始化的 Host 同步位于 dispatch Event 区间之外，因此 kernel Event 不能代表完整 API 时延。
性能采样使用的前置 barrier 及其关闭方式见[启动对齐说明](#dispatch-性能测试的启动对齐)。

## LL 正确性与性能

LL 要求 EP16/32/64/128、`0 < B ≤ M ≤ 256`、固定 8 GiB 工作区。默认 `M=B`；BS32/96/256 均可进入 LL，其他 shape 与选项仍须通过检查。先在无设备环境检查矩阵：

```bash
python tests/elastic/test_ep_low_latency.py --plan \
  --num-processes 8 --num-nodes 2 --num-tokens 32 96 256 \
  --hidden 7168 --num-topk 6 --num-experts 256 --dtype bf16
```

LL ST 与上述 HT 脚本的初始化方式不同：它默认创建 HCCL 测试进程组并传给 Buffer，使用 `--do-cpu-sync 0 1` 分别检查容量布局和精确行数布局。下面是两台主机各 8 卡的运行示例，在两台分别设置 `NODE_RANK=0/1`，并配置首台可达 IP：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=19090
export DEEPEP_SHMEM_ENDPOINT="tcp://${MASTER_ADDR}:19091"

python tests/elastic/test_ep_low_latency.py \
  --num-processes 8 --num-nodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --shmem-ip-port "$DEEPEP_SHMEM_ENDPOINT" \
  --num-tokens 32 96 256 --hidden 7168 --num-topk 6 --num-experts 256 \
  --dtype bf16 --do-cpu-sync 0 1 --num-bytes 8589934592 \
  --output-dir /tmp/deepep-ll-runs
```

该入口支持本机 spawn 和 torchrun。测试显式 rank/world_size 方式时，在命令中加入 `--without-buffer-group`：Buffer 使用 `group=None`，LL 在两种 `do_cpu_sync` 设置下都返回容量布局；测试控制进程组仍用于协调测试和收集结果。该选项不表示整个 ST 不使用进程组。`--group-backend gloo` 可将测试进程组后端改为 Gloo，token payload 的 SHMEM/kernel 通信路径不变。各 rank 的可见设备与启动配置须一致。

master 地址、master 端口和 SHMEM endpoint 必须通过 CLI、JSON 或环境变量显式提供，优先使用 CLI/JSON。`--plan` 不要求完整网络配置。使用 [JSON 示例](elastic/config_ll.example.json) 时替换地址和端口占位；`master_port/shmem_port` 是整数。

用 `--dtype fp8 --fp8-scale-dtype int32 float32 --do-cpu-sync 0 1` 覆盖两种 scales 格式及同步选项；JSON matrix 也支持这些组合。FP32 scales 用例按源 rank、token 和分组生成非单位值，检查搬运与行对应关系，并在专家参考计算中反量化。

默认执行正确性和 kernel 测量。`--skip-perf-test` 只检查正确性；`--skip-check` 的结果标记为 `skipped`，不能作为精度通过证据。诊断选项 `--trace-ops` 会影响测量。计时与跨 rank 汇总口径见[性能说明](../docs/performance.md)。

## 日志与结果

HT 阶段日志的 `dispatch.enter/combine.enter` 包含 Host 准备和提交；`return` 只表示接口返回；`readback.done` 表示设备回读结束，随后 `precision.passed` 才表示比较通过。阶段间隔不是 kernel 耗时。

LL 为每次运行建立独立目录，保存配置、case 结果、逐 rank profiler 记录及 `summary.json`；执行性能测量时还写入 `summary.csv/ranks.csv`。查看所有 rank 的退出和销毁日志，不能仅凭摘要文件存在判定整个作业成功。精度判据见[精度说明](../docs/precision.md)。

出现异常时先查原始堆栈及其他 rank 日志。停在 `buffer.destroy.begin` 表示清理尚未完成，不应覆盖原始错误或把重跑记录算作本次成功。

## CPU 与 NPU 参考闭环

这些入口验证测试适配器，分别保留其结果范围：

| 后端 | 执行路径 | 结果含义 |
| --- | --- | --- |
| `functional-stub` | 单进程 CPU，模拟多个 rank 的普通布局 | 参考计算和报告通过，不验证 wheel API 或设备通信 |
| `npu-stub` | NPU Tensor 与 HCCL AllGather/AllReduce | 测试链路通过，不验证 DeepEP 原生通信 |
| `native`（V1 roundtrip） | 安装后的 `Buffer` V1 方法 | 当前缺少所需 V1 通信操作，入口拒绝执行；不能用于验收 V2 |

CPU 闭环无需 Torch、NumPy 或设备：

```bash
python -m tests.integration.run_roundtrip --backend functional-stub
python -m pytest tests/integration -q
```

它覆盖 mixed_k1、cross_rank、same_rank、hotspot、masked_zero、fresh_iterations 六个小场景。专家函数为 `f_e(x)=(e+1)/8*x+(e-3)/64`，按有效槽位累加 `gate*f_e(x)`。普通布局先按目标 rank 去重，再分别保留该 rank 上的专家贡献；gate 应用一次，独立 FP32 值回传另行比较。

功能桩使用专用 handle，不能传给公开 Buffer；它的单轮消费规则不能套用到 native cached/multi-handle 操作。逐 rank 比较采用独立 FP64 Golden 和明确标记的 BF16/FP32 路径模拟，规则见[精度说明](../docs/precision.md)。`suite_wall_seconds` 只是测试耗时。

在有匹配框架、HCCL 和两张已分配 NPU 的环境运行参考闭环：

```bash
python scripts/validate_ep.py --backend npu-stub --devices 0,1 \
  --output /path/to/new-stub-suite
```

脚本将测试复制到源码目录外，使用当前 Python 启动 torchrun，不自动安装依赖或选择设备。可用 `--case` 选择上述场景；每次 torchrun 默认超时 600 秒，进程组超时 120 秒。不要使用 `python -O` 或 `PYTHONOPTIMIZE` 禁用断言。

仅当选定场景的全部 rank、精度和退出检查成功，参考闭环才生成成功摘要。保留日志与 `rank-*.json/summary.json`；缺失、失败或 skip 不记为通过。完整 wheel/sdist 构建及安装验证见[构建指南](../docs/build.md)。
