# Mesh+PCIe、Mesh+Clos 与 Global Clos

一个安装包包含两个独立实现，共用 `dependencies.lock.json` 固定的 CANN SHMEM
`836d3fd6f9c2f3b10a6acac36ccfd531b104d4a7`。Mesh+Clos 保留本仓已有的 HT、LL、
实验组件和接口限制。Direct 实现包含 V1 Buffer、V2 ElasticBuffer、
MTE/UDMA、fresh/cached、FP8/BF16、expand/aggregate 以及资源接口。
各实现继续使用自己的 kernel、工作区、QP 调度和 handle，功能范围不互相推导。
Mesh+PCIe（`mesh_pcie`，当前适配 board `0x6c`）使用 Direct 的 MTE kernel，
Global Clos（`global_clos`）使用 Direct 的 UDMA kernel，不复制公共实现或 SHMEM。

V1 的统一入口为 `deep_ep/buffer.py`，V2 的统一入口为
`deep_ep/buffers/elastic.py`。拓扑入口和公共实现目录如下：

```text
deep_ep/backends/
├── mesh_pcie/       # Mesh+PCIe：同机和远端映射检查，选择 MTE
├── mesh_clos/       # Mesh+Clos：域内 Full Mesh，域间 Clos
├── global_clos/     # Global Clos：选择 UDMA
└── direct/         # Mesh+PCIe 与 Global Clos 共用的接口、资源和调度实现
```

“域”表示物理互联域，不是 EP、TP 或 PP 进程组。`direct` 是内部实现名称，
表示按 EP peer 直接组织通信，不表示物理直连，也不是第四种 `machine_type`。
Mesh+PCIe 当前仍要求已适配的四卡 Mesh 产品、同机 EP 和有效的远端内存映射；
名称调整不扩大硬件支持范围。构造时选择一次后端，
后续 dispatch/combine 直接调用对应实现。Python 公开入口统一为 `deep_ep`；
V1 的 `deep_ep.buffers.legacy` 仅保留兼容导入，不存放另一份算子实现。

旧的 `ascend_deepep` 及其 `buffer`、`elastic`、`config`、`event` 等公开子模块
作为兼容入口转发到同一套实现，构造 Buffer 时仍自动选择机型。兼容映射集中在
`ascend_deepep/__init__.py`，旧子模块动态注册，访问导出对象时才加载对应实现，
不再为每个旧路径保留单独的转发文件。兼容入口不包含
另一份原生库，也不恢复已移除的参数。安装时同一 Python 环境只保留一种 DeepEP
发行包；官方 DeepEP 与本仓共享 `deep_ep` 导入名，应使用独立环境。

公开 Config 默认使用 32 AIV。Clos V1 未传 Config 时使用 `set_num_sms` 设置的
当前核数，初始值为 32；V2 fresh dispatch 的默认核数为 32，cached dispatch 和
combine 未指定核数时沿用 handle 中的核数，包括显式 64 核的 handle。
显式核数继续校验；Mesh+Clos 的调度不受这一默认值调整影响。

## 构建

继续使用本仓的[编译安装流程](build.md)，同一 Python 环境需有匹配的 Torch/torch_npu。
Mesh+Clos 和 Direct 共用一份 SHMEM SDK。
构建产物包括 `_C`、`_clos_C` 和各自的 device library；二者加载同一 SDK 的 `libshmem.so`。
`_clos_C` 和 `libdeepep_clos_kernels.so` 保留二进制名称，实际承载 Direct 的 MTE/UDMA
两条路径；源码位于 `csrc/direct/`，构建定义位于 `cmake/Direct.cmake`。
升级目录布局时应使用干净的构建暂存目录。安装会拒绝残留的旧 backend 实现，
避免将两套 Python 模块同时打进 wheel；旧机型参数仍作为输入别名接受，
识别结果、日志和新配置统一使用上表中的拓扑名称。
构建面向 Ascend950，统一使用 64-bit top-k 索引。

## 运行前检查机型

在已安装本项目的 Python 环境中，检查本机参与任务的物理卡号，例如预期全 Clos 的八卡：

```bash
python -u scripts/check_machine.py --physical-devices 0 1 2 3 4 5 6 7 \
    --expect global_clos --json machine_check.json
```

`--expect` 也可设为 `mesh_clos` 或 `mesh_pcie`，仅校验自动识别结果，不强制改变选择。
成功打印 `MACHINE_CHECK_PASSED` 并返回 0；识别失败、机型不符或各卡结果不一致返回非零。
省略 `--expect` 可先查看识别结果。报告记录识别模块的实际路径、SHA 和逐卡证据；
输出文件须不存在。无需编译或加载 native 扩展；代码尚未安装时可加
`--implementation source` 检查当前 checkout，但这不证明已安装包的版本一致。

脚本与 Buffer 共用拓扑分类逻辑，只读取 rootinfo/topo，必要时执行限时 5 秒的 DCMI
管理查询；不导入 Torch，不设置 NPU，不创建 Buffer/SHMEM 会话或分配设备内存。
卡号必须是物理 chip ID，不是 EP rank 或可见设备的逻辑编号。此检查不验证逻辑卡号映射、
链路健康或通信正确性。实际任务应使用相同代码、拓扑文件及物理设备，并通过
`buffer.machine_type` / `buffer.machine_selection` 确认最终选择。多机需每台分别检查。

## 选择后端

在调用方设置当前 NPU 后构造 Buffer。两个类均接受：

```python
buffer = deep_ep.ElasticBuffer(
    group=ep_group,
    num_max_tokens_per_rank=8192,
    hidden=6144,
    num_topk=8,
    explicitly_destroy=True,
    machine_type="auto",  # auto / mesh_clos / global_clos / mesh_pcie
)
print(buffer.machine_type, buffer.machine_selection)
```

公开 `deep_ep.Buffer` / `deep_ep.ElasticBuffer` 的通信引擎由机型决定：

| `machine_type` | 物理拓扑 | 当前通信实现 |
| --- | --- | --- |
| `auto` | 自动识别 | 应用对应拓扑规则 |
| `mesh_clos` | 域内 Full Mesh，域间 Clos | V2 使用 MTE + UDMA |
| `global_clos` | 全局 Clos 互联 | UDMA payload，MTE 元数据/本地搬运 |
| `mesh_pcie` | 域内 Full Mesh，域间 PCIe | MTE，不创建 UDMA QP/Jetty |

`auto` 读取当前物理 device 对应的 `/etc/hccl_rootinfo.json` 及其中引用的 topo 文件。
`TOPO_FILE_DESC` 按 topo 边声明解释；有 Clos 且无 NPU Mesh 的设备选择 `global_clos`。
Mesh+Clos 还需已核对的 Mesh+Clos board 规则（0x03/0x07）。完整四卡 Mesh 域与 board `0x6c`
共同选择 `mesh_pcie`；主机 NIC 的 `d2h` 声明不算 NPU Clos。
缺省文件缺失时可回退到 Mesh+Clos 及 `0x6c` 的
DCMI board 规则；DCMI 查询超时为 5 秒，不重试。未知 board、缺失端点、文件矛盾或
不完整拓扑报错，要求显式指定，不能仅凭“16 卡”或某一 EID 位推断机型。
`rootinfo_path="/path/to/rootinfo.json"` 可指定文件；显式文件错误不会悄悄回退。
这些信息是拓扑声明，不证明物理连接健康或带宽利用率。

现场没有可靠 rootinfo 时使用 `machine_type="global_clos"` 或 `"mesh_clos"`。显式选择跳过
自动探测，但不修改 SHMEM 自身的拓扑初始化；依赖的驱动/SDK 仍须支持该环境。
传入 EP ProcessGroup/MPI communicator 时，构造阶段先核对全组的后端选择和探测错误，
再进入 SHMEM 初始化。保留 Mesh+Clos 的显式 rank/world_size/MPI 启动方式；没有 EP group 时
不增加框架进程组，调用方须保证全体 rank 选择一致。

后端只在构造时选择一次。全 Clos 按实际 EP group 的 rank 数工作，保留单服务器
16 卡，不加入 `rank // 8` 的分组假设。rank ID 和物理 device ID 不是同一个编号。

显式选择 Clos 的 V1 示例：

```python
config = deep_ep.Config(num_sms=32)
num_bytes = config.get_nvl_buffer_size_hint(6144 * 2, ep_group.size(), 8192)
buffer = deep_ep.Buffer(
    ep_group, num_nvl_bytes=num_bytes, explicitly_destroy=True,
    machine_type="global_clos",
)
# 原 V1 get_dispatch_layout -> dispatch -> combine 调用和返回语义保持不变。
```

V2 的 `get_buffer_size_hint(..., machine_type=...)` 也须选择同一后端；容量查询是
非 collective，不增加跨 rank 同步。直接让构造函数按 shape 计算可避免选错 hint。
Mesh+Clos 保留其 endpoint/MPI bootstrap；Clos 保留默认 UID 与显式 `shmem_ip_port` 方式。

## 全 Clos 的默认调度与手动调优

V1/V2 在 UDMA 下默认 `udma_auto_policy=True`。机型识别先选择通信实现，
Clos 策略再选择发送参数。`clos-v3` 将初始化资源与逐次调用的策略分开：

- Buffer 构造时在 EP group 内交换主机 boot ID。仅在各主机 rank 连续且数量相等时，
  将其识别为物理分组；不会默认每台 8 rank，也不使用 `LOCAL_WORLD_SIZE` 猜测。
  不可识别时采用 rank 顺序；极低核数需要轮转兜底时使用明确的逻辑分组。
- 一次 SHMEM 初始化预留所支持 peer 顺序、P=1/2/4/8 和 W×Q=1/2 的 QP 映射并集。
  每个发送 AIV 仍独占对应 SQ，QP 总数保持 32；预留的 peer/QP 连接数可能比旧策略多。
- 每次调用只查缓存并生成不可变 launch plan，按 EP、实际 payload 行字节数、通信核数、
  dispatch/无权重 combine/带权重 combine 分别选择顺序、P/W/Q、visit 和本卡复制方式。
  不读取设备上的计数，不执行在线摸测，不修改 Buffer 的全局调度状态，也不额外同步。

当前离线模板来自 EP4/8/16/32、AIV8/32、BF16、非展开随机路由测试，按不超过 4 KiB
和较大行分档，覆盖测得的 2 KiB、5632 B、12 KiB 行。它是已有数据的保守拟合，
不是所有 shape 的最优性能保证。模板还有实际分组约束：EP32 的 G=8 测量结果不会
套用到 G=16。AIV20/64、未覆盖的 EP/行大小、FP8 dispatch、展开路径和 V1 保留保守策略。
8 核非展开 BF16 combine 通常先提交远端再复制本卡；32 核无权重 combine 可由空余
AIV 并行复制本卡。带权重 combine 独立选择 UDMA 或 SIMT，不因存在空余核就必选 SIMT。

未命中模板时按核数增加 peers/worker，必要时改为轮转分组，检查发送核数、32-QP 预算
及每 worker 64-peer 容量后才下发。例如 EP128/AIV8、G=16 可用轮转 P=2/W=1/Q=1；
这表示资源上可执行，不代表已验证 EP128 的性能。资源仍放不下时在下发前报错。

策略不增加用户指定的通信核数。V1 通过 `Config(num_sms=...)` 指定；未传 Config
沿用 `set_num_sms`，默认 32。V2 fresh dispatch 的 `num_sms=0` 默认 32；cached dispatch
的 0 沿用 handle。combine 的 `num_sms` 保留 handle 的逻辑通道数；`combine_num_sms`
独立指定通信核数，`reduce_num_sms` 独立指定最终 reduce 核数，二者 0 均继承 handle
核数。前置 aggregate 仍使用 handle 核数，实验性 aggregate pipeline 不允许改变通道数。
fresh dispatch 前的独立 layout 仍按专家数和设备能力选核，不受通信核数参数限制。

`buffer.udma_auto_policy` 是不可变初始资源策略，`.for_num_aiv(n)` 仅是核数兜底，
不代表最终 shape 策略。`.for_launch(n, row_bytes, operation, bf16=True, expanded=False)`
返回实际策略，其中 operation 为 `dispatch`、`combine` 或 `weighted`；V1 查询需加
`v2=False`。带宽报告的 `effective_udma_config.operation_plans` 分别记录三种策略。
手动模式保留原有行为；原生扩展和 Python 必须同步重新编译安装。

自动模式下显式传入的策略参数不改变结果；与有效值不同会在 rank 0 记录 warning。
要调优，请在构造 Buffer 时设 `udma_auto_policy=False`，完整指定所用策略参数。例如 V2：

```python
buffer = deep_ep.ElasticBuffer(
    group=ep_group, num_max_tokens_per_rank=8192, hidden=6144, num_topk=8,
    machine_type="global_clos", explicitly_destroy=True,
    udma_auto_policy=False, udma_schedule="peer",
    udma_dispatch_peer_order="rank", udma_combine_peer_order="rank",
    udma_peers_per_worker=4, udma_workers_per_peer=1,
    udma_qps_per_worker=1, udma_wqes_per_visit=32,
)
```

V1 的 peers/worker 和 WQEs/visit 放在每次调用的 `Config` 中，其余参数在 Buffer 构造时设置。
group 顺序还需 `udma_ranks_per_group`；group_round_robin 还需
`udma_peers_per_group_per_worker`。MTE 不启用 UDMA 自动策略。

UDMA 控制区固定在工作区开头，每个源 rank 保留 512 字节（256 字节对齐），
dispatch/combine 数据区与容量计算同步后移。普通 combine 省去前置 cached-notify，
但保留 kernel Begin 的跨 rank 同步、接收完成等待和发送 Quiet，避免上一轮读写尚未结束就复用数据。
cached dispatch、MTE 和需要独立初始化的实验路径保留对应 notify。
这些改动需要整体重编 Python/native/device 库，不能只替换 Python 文件。

Clos 仅保留两个精度与带宽入口，均使用库的默认自动调度策略，不提供手动调度或参数扫描：

```bash
python benchmarks/clos/bench_v1_bandwidth.py --ep-size 8 --output clos_v1_ep8
python benchmarks/clos/bench_v2_bandwidth.py --ep-size 8 --output clos_v2_ep8
```

每个入口默认使用 32 AIV，覆盖 BF16/FP8 × 展开/不展开共 4 组配置，每组先做精度
检查，再测 fresh dispatch、cached dispatch、无 weight combine 和带 weight combine。
可通过 `--aiv 20` 指定核数；`--dtypes bf16 --expand 0` 可缩小功能组合。
保留 token/hidden/top-k、路由分布、bias、多机和报告重算选项。
V2 通信带宽分母仅为通信 kernel 耗时，notify/reduce 单列；V1 的 fused combine
保留融合耗时，不能将其当作独立通信耗时。

EP32 默认两机、每机 16 rank；每机只有 8 卡时显式设置 `--nnodes 4`。所有节点
使用相同的 `--ep-size`、`--nnodes`、`--master-addr` 和 `--run-id`，分别设置
`--node-rank`。例如两机时在 node0/node1 分别执行以下命令并替换 node-rank：

```bash
python benchmarks/clos/bench_v2_bandwidth.py --ep-size 32 --nnodes 2 --node-rank 0 \
    --master-addr <NODE0_IP> --run-id clos_v2_ep32 --output clos_v2_ep32_node0
```

两个入口共用四个辅助模块：`bandwidth_multinode.py` 负责多机协调与结果收集，
`route_workloads.py` 负责路由生成，`profiling.py` 负责采样执行，
`stage_metrics.py` 负责采样标记、profiler 解析和阶段指标统计。

V1/V2 带宽脚本在每轮预热和 profiling 调用后等待本卡 NPU 执行完成，再提交下一轮，
避免未完成输出和分配持续积压。该 Host 等待位于采样结束标记之外，不计入 kernel
带宽分母；报告通过 `sampling.host_sync_each_iteration=true` 记录此测量方式。
Combine 的预热、profiling（含边界保护轮次）及独立 Event 计时还会在每次调用前，
先等待本卡 NPU 完成，再通过 Buffer 的 Host SHMEM barrier 对齐所有 rank。
V1 使用 `buffer.barrier()`，V2 使用 `buffer.runtime.barrier()`；该对齐位于采样标记、
Event 起点及可选缓存清理之前，不计入耗时，也不改变算子内部同步。
SHMEM Host barrier 内部的设备 barrier kernel 仅在采样区间之外排除，原始 profiler 仍保留它。
它减少 rank 间的启动偏差，但不保证 kernel 同时启动；不同对齐条件下的性能需区分比较。

Direct 原生实现集中在 `csrc/direct/`：`bindings/` 提供 Python 注册，`runtime/`
管理 Buffer 和 SHMEM，`ops/` 处理参数、tensor 分配和 kernel 下发，`common/`
存放 host/device 共用定义。设备实现位于 `kernels/`，按 dispatch、combine、notify、
expand、aggregate 分组。`ops/stream_allocator.h` 由 V1/V2 共用，保持分配流与
执行流的生命周期约束。dispatch/combine 中无 `_udma` 后缀的通信 kernel 使用 MTE，
带 `_udma` 后缀的 kernel 使用 UDMA；`ops/launcher.cpp` 按选定引擎下发。

## Mesh+PCIe：四卡 Mesh 与跨域 PCIe

V1 Buffer 和 V2 ElasticBuffer 均支持 `machine_type="mesh_pcie"`。自动识别或显式
指定后固定使用 MTE，无需传入引擎参数。UDMA token 调度会在初始化前集体报错；
当前适配只支持同机 EP ProcessGroup/MPI communicator。

```python
buffer = deep_ep.ElasticBuffer(
    group=ep_group,
    num_max_tokens_per_rank=8192, hidden=6144, num_topk=8,
    machine_type="mesh_pcie", explicitly_destroy=True,
)
assert buffer.transport == "mte"
```

初始化前比较组内 Linux boot ID；缺失或不同则拒绝，不用 hostname 或 rank 编号猜同机。
SHMEM 初始化后、DeepEP 算子启动前，逐 peer 查询 metadata/workspace 的 Runtime 内存范围。
地址仅非空不足以通过：映射必须覆盖整个区间，查询不支持、失败或范围不足均集体报错并
释放本次 native 资源，不回退为盲读。失败路径不执行 DeepEP 的设备 barrier。
这些检查只在构造阶段运行，fresh/cached 热路径、分核及通信同步不变。

此变更需要按原构建流程重新编译安装 native 扩展；旧库缺少查询方法会拒绝初始化。
映射存在不等于物理链路健康或已验证性能。Mesh+PCIe 域内 EP4、跨域 EP8/EP16、多组 EP4 的
NPU 精度与稳定性尚待实机验证；不能将 Mesh+Clos 环境结果视为 Mesh+PCIe 验证通过。

## 生命周期

同一进程不能并存 Mesh+Clos 和 Direct SHMEM 会话。初始化/销毁使用共享锁，native 端检查
SDK 初始化状态，禁止覆盖另一后端的 QP/heap 配置。必须在整个 EP group 中按相同顺序
完成操作、`destroy()` 后再重建。handle 不能跨 Buffer 或后端复用；公共 EventHandle
可作为 Clos 的 previous_event/epilogue 依赖，等待语义和原有 stream 顺序保留。

业务代码通过 `deep_ep.Buffer` / `deep_ep.ElasticBuffer` 选择后端。
`benchmarks/clos/` 保留现有命令路径，其带宽入口通过 `deep_ep.backends.global_clos`
使用固定的 Global Clos 后端，不经过公开入口的
自动机型选择；`check_machine.py` 的检查结果不会改变这些脚本的后端。
后端专用测试也可直接导入 `deep_ep.backends.direct`。
导入、CPU 契约检查、编译成功都不等于 NPU 精度或性能验证通过。
