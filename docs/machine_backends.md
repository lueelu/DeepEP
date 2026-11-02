# POD、全 Clos 与 PR 机型

一个安装包包含两个独立实现，共用 `dependencies.lock.json` 固定的 CANN SHMEM
`836d3fd6f9c2f3b10a6acac36ccfd531b104d4a7`。POD 保留本仓已有的 HT、LL、
实验组件和接口限制。全 Clos 实现包含 V1 Buffer、V2 ElasticBuffer、
MTE/UDMA、fresh/cached、FP8/BF16、expand/aggregate 以及原有资源接口。
各实现继续使用自己的 kernel、工作区、QP 调度和 handle，功能范围不互相推导。
PR（`CARD_4PMESH`，board `0x6c`）复用迁入实现的 MTE kernel，不另建一份 SHMEM。

V1 的统一入口为 `deep_ep/buffer.py`，V2 的统一入口为
`deep_ep/buffers/elastic.py`；POD 与 Clos 的实现分别位于
`deep_ep/backends/pod/` 和 `deep_ep/backends/clos/`。构造时选择一次后端，
后续 dispatch/combine 直接调用对应实现。Python 公开入口统一为 `deep_ep`；
V1 的 `deep_ep.buffers.legacy` 仅保留兼容导入，不存放另一份算子实现。

旧的 `ascend_deepep` 及其 `buffer`、`elastic`、`config`、`event` 等公开子模块
作为兼容入口转发到同一套实现，构造 Buffer 时仍自动选择机型。兼容入口不包含
另一份原生库，也不恢复已移除的参数。安装时同一 Python 环境只保留一种 DeepEP
发行包；官方 DeepEP 与本仓共享 `deep_ep` 导入名，应使用独立环境。

公开 Config 默认使用 32 AIV。Clos V1 未传 Config 时使用 `set_num_sms` 设置的
当前核数，初始值为 32；V2 fresh dispatch 的默认核数为 32，cached dispatch 和
combine 未指定核数时沿用 handle 中的核数，包括显式 64 核的 handle。
显式核数继续校验；POD 的调度不受这一默认值调整影响。

## 构建

继续使用本仓的[编译安装流程](build.md)，同一 Python 环境需有匹配的 Torch/torch_npu。
POD 和 Clos 共用一份 SHMEM SDK。
构建产物包括 `_C`、`_clos_C` 和各自的 device library；二者加载同一 SDK 的 `libshmem.so`。
本迁移构建面向 Ascend950，统一使用 64-bit top-k 索引。

## 运行前检查机型

在已安装本分支的 Python 环境中，检查本机参与任务的物理卡号，例如预期全 Clos 的八卡：

```bash
python -u scripts/check_machine.py --physical-devices 0 1 2 3 4 5 6 7 \
    --expect clos_only --json machine_check.json
```

`--expect` 也可设为 `pod` 或 `card_4pmesh`，仅校验自动识别结果，不强制改变选择。
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
    machine_type="auto",  # auto / pod / clos_only / card_4pmesh
)
print(buffer.machine_type, buffer.machine_selection)
```

公开 `deep_ep.Buffer` / `deep_ep.ElasticBuffer` 的通信引擎由机型决定，不接受
`transport` 入参：

| `machine_type` | 通信引擎 |
| --- | --- |
| `auto` | 识别机型后应用对应规则 |
| `pod` | 保留 POD 实现原有配置，V2 使用 MTE + UDMA |
| `clos_only` | UDMA payload，MTE 元数据/本地搬运；SHMEM 启用 MTE + UDMA |
| `card_4pmesh` | 仅 MTE，不创建 UDMA QP/Jetty |

旧调用须删除 `transport=...`，包括 `None`、`"auto"` 和与机型匹配的值。
传入 EP group 时，该参数错误在现有机型选择 collective 中统一报告，发生在 SHMEM
初始化前。`ASCEND_DEEPEP_TRANSPORT` 不改变上述公开接口的引擎选择。
迁入实现内部保留引擎参数供后端调用和原有内部测试使用，不是新的机型选择入口。

`auto` 读取当前物理 device 对应的 `/etc/hccl_rootinfo.json` 及其中引用的 topo 文件。
`TOPO_FILE_DESC` 按 topo 边声明解释；有 Clos 且无 NPU Mesh 的设备选择 `clos_only`。
Mesh+Clos 还需已核对的 POD board 规则（0x03/0x07）。完整四卡 Mesh 域与 board `0x6c`
共同选择 `card_4pmesh`；主机 NIC 的 `d2h` 声明不算 NPU Clos。
缺省文件缺失时可回退到 POD 及 `0x6c` 的
DCMI board 规则；DCMI 查询超时为 5 秒，不重试。未知 board、缺失端点、文件矛盾或
不完整拓扑报错，要求显式指定，不能仅凭“16 卡”或某一 EID 位推断机型。
`rootinfo_path="/path/to/rootinfo.json"` 可指定文件；显式文件错误不会悄悄回退。
这些信息是拓扑声明，不证明物理连接健康或带宽利用率。

现场没有可靠 rootinfo 时使用 `machine_type="clos_only"` 或 `"pod"`。显式选择跳过
自动探测，但不修改 SHMEM 自身的拓扑初始化；依赖的驱动/SDK 仍须支持该环境。
传入 EP ProcessGroup/MPI communicator 时，构造阶段先核对全组的后端选择和探测错误，
再进入 SHMEM 初始化。保留 POD 的显式 rank/world_size/MPI 启动方式；没有 EP group 时
不增加框架进程组，调用方须保证全体 rank 选择一致。

后端只在构造时选择一次。全 Clos 按实际 EP group 的 rank 数工作，保留单服务器
16 卡，不加入 `rank // 8` 的分组假设。rank ID 和物理 device ID 不是同一个编号。

显式选择 Clos 的 V1 示例：

```python
config = deep_ep.Config(num_sms=32)
num_bytes = config.get_nvl_buffer_size_hint(6144 * 2, ep_group.size(), 8192)
buffer = deep_ep.Buffer(
    ep_group, num_nvl_bytes=num_bytes, explicitly_destroy=True,
    machine_type="clos_only",
)
# 原 V1 get_dispatch_layout -> dispatch -> combine 调用和返回语义保持不变。
```

V2 的 `get_buffer_size_hint(..., machine_type=...)` 也须选择同一后端；容量查询是
非 collective，不增加跨 rank 同步。直接让构造函数按 shape 计算可避免选错 hint。
POD 保留其 endpoint/MPI bootstrap；Clos 保留默认 UID 与显式 `shmem_ip_port` 方式。

## 全 Clos 的默认调度与手动调优

V1/V2 在 UDMA 下默认 `udma_auto_policy=True`，fresh/cached dispatch 与 combine
采用相同的 peer 调度参数。机型识别与此策略是两层逻辑：先选通信实现，再选 Clos 调度。

| EP | peers/worker | workers/peer | QPs/worker | WQEs/visit | peer 顺序 |
| --- | --- | --- | --- | --- | --- |
| 2–8 | 1 | 2 | 1 | 8 | rank |
| 9–16 | 1 | 1 | 1 | 8 | rank |
| 17–128 | 4 | 1 | 1 | 32 | rank |
| 129–256 | 8 | 1 | 1 | 32 | rank |

后两档是在 32 个 QP 预算内的推广，不代表所有 EP、核数和路由均已有性能测量。
策略不修改用户指定的核数：V1 通过 `Config(num_sms=...)`，V2 在 dispatch/combine
传 `num_sms=...`。上表为初始策略；核数不足时，自动将 peers/worker 按 1、2、4、8
逐档增加，直到 `ceil((EP-1)/peers_per_worker) * workers_per_peer` 不超过核数及 32-QP
预算。workers/peer、QPs/worker 和 WQEs/visit 保持不变；核数仍须为合法正偶数。
例如 8 核下 EP4/8/16/32 的 peers/worker 分别为 1/2/2/4；EP128 的 20 核使用
8 peers/worker、16 个发送 worker。P=8 仍放不下时在下发前报错，例如 EP32 的 2 核。

QP 资源初始化已覆盖上述所有 peer 分组，无需为低核数重建 Buffer。每次调用从不可变的
初始策略与核数推导有效策略，不改变其他 handle；V2 cached dispatch/combine 沿用
handle 核数，因此得到同一策略。`buffer.udma_auto_policy` 返回初始策略，
`.for_num_aiv(n)` 返回指定核数的有效策略；带宽报告记录后者，版本为 `clos-v2`。
手动模式不做自动调整。这里的核数约束作用于通信、notify 及对应 expand/reduce；
fresh dispatch 前的独立 layout 仍按专家数和设备能力选核，不受该参数限制。
实验性的 aggregate pipeline 仍要求 32/64 核。
V1 未传 `Config` 时使用 `set_num_sms` 设置的当前核数，初始值为 32，不按 handle
自动继承核数。V2 fresh dispatch 的 `num_sms=0` 使用 32 核；cached dispatch/combine
的 0 沿用 handle 核数，显式值必须与 handle 一致。`get_theoretical_num_sms()`
仍可供调用方查询，但不参与 fresh dispatch 的默认选核。
带宽脚本默认显式传 32 核，与 V2 fresh dispatch 的库默认值一致。

自动模式下显式传入的策略参数不改变结果；与有效值不同会在 rank 0 记录 warning。
要调优，请在构造 Buffer 时设 `udma_auto_policy=False`，完整指定所用策略参数。例如 V2：

```python
buffer = deep_ep.ElasticBuffer(
    group=ep_group, num_max_tokens_per_rank=8192, hidden=6144, num_topk=8,
    machine_type="clos_only", explicitly_destroy=True,
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

两个入口共用 `bandwidth_multinode.py`、`route_workloads.py` 和 `stage_metrics.py`，
分别负责多机协调与结果收集、路由生成、采样和阶段计时。

V1/V2 带宽脚本在每轮预热和 profiling 调用后等待本卡 NPU 执行完成，再提交下一轮，
避免未完成输出和分配持续积压。该 Host 等待位于采样结束标记之外，不计入 kernel
带宽分母；报告通过 `sampling.host_sync_each_iteration=true` 记录此测量方式。
Combine 的预热、profiling（含边界保护轮次）及独立 Event 计时还会在每次调用前，
先等待本卡 NPU 完成，再通过 Buffer 的 Host SHMEM barrier 对齐所有 rank。
V1 使用 `buffer.barrier()`，V2 使用 `buffer.runtime.barrier()`；该对齐位于采样标记、
Event 起点及可选缓存清理之前，不计入耗时，也不改变算子内部同步。
SHMEM Host barrier 内部的设备 barrier kernel 仅在采样区间之外排除，原始 profiler 仍保留它。
它减少 rank 间的启动偏差，但不保证 kernel 同时启动；不同对齐条件下的性能需区分比较。

CLOS 原生实现集中在 `csrc/clos/`：`bindings/` 提供 Python 注册，`runtime/`
管理 Buffer 和 SHMEM，`ops/` 处理参数、tensor 分配和 kernel 下发，`common/`
存放 host/device 共用定义。设备实现位于 `kernels/`，按 dispatch、combine、notify、
expand、aggregate 分组。`ops/stream_allocator.h` 由 V1/V2 共用，保持分配流与
执行流的生命周期约束。CLOS 仅保留本后端及 CARD MTE 复用的实现。

## PR：四卡 Mesh 与跨域 PCIe

V1 Buffer 和 V2 ElasticBuffer 均支持 `machine_type="card_4pmesh"`。自动识别或显式
指定后固定使用 MTE，无需传入引擎参数。UDMA token 调度会在初始化前集体报错；
当前 PR 适配只支持同机 EP ProcessGroup/MPI communicator。

```python
buffer = deep_ep.ElasticBuffer(
    group=ep_group,
    num_max_tokens_per_rank=8192, hidden=6144, num_topk=8,
    machine_type="card_4pmesh", explicitly_destroy=True,
)
assert buffer.transport == "mte"
```

初始化前比较组内 Linux boot ID；缺失或不同则拒绝，不用 hostname 或 rank 编号猜同机。
SHMEM 初始化后、DeepEP 算子启动前，逐 peer 查询 metadata/workspace 的 Runtime 内存范围。
地址仅非空不足以通过：映射必须覆盖整个区间，查询不支持、失败或范围不足均集体报错并
释放本次 native 资源，不回退为盲读。失败路径不执行 DeepEP 的设备 barrier。
这些检查只在构造阶段运行，fresh/cached 热路径、分核及通信同步不变。

此变更需要按原构建流程重新编译安装 native 扩展；旧库缺少查询方法会拒绝初始化。
映射存在不等于物理链路健康或已验证性能。PR 域内 EP4、跨域 EP8/EP16、多组 EP4 的
NPU 精度与稳定性尚待实机验证；不能将 POD 环境结果视为 PR 验证通过。

## 生命周期

同一进程不能并存 POD 和 Clos SHMEM 会话。初始化/销毁使用共享锁，native 端检查
SDK 初始化状态，禁止覆盖另一后端的 QP/heap 配置。必须在整个 EP group 中按相同顺序
完成操作、`destroy()` 后再重建。handle 不能跨 Buffer 或后端复用；公共 EventHandle
可作为 Clos 的 previous_event/epilogue 依赖，等待语义和原有 stream 顺序保留。

业务代码通过 `deep_ep.Buffer` / `deep_ep.ElasticBuffer` 选择后端。
Clos 带宽入口通过 `deep_ep._clos_adapter` 使用固定的 Clos 后端，不经过公开入口的
自动机型选择；`check_machine.py` 的检查结果不会改变这些脚本的后端。
后端专用测试也可直接导入 `deep_ep.backends.clos`。
导入、CPU 契约检查、编译成功都不等于 NPU 精度或性能验证通过。
