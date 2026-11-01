# POD、全 Clos 与 PR 机型

一个安装包包含两个独立实现，共用 `dependencies.lock.json` 固定的 CANN SHMEM
`836d3fd6f9c2f3b10a6acac36ccfd531b104d4a7`。POD 保留本仓已有的 HT、LL、
实验组件和接口限制。全 Clos 实现从 ascend_deepep 的初始导入 `c0904032`
更新至 `v1-shared-jetty-scheduler` 分支 `6f9d8c9ef9403a302724ce42cf55810843ffebc3`，包含 V1 Buffer、V2 ElasticBuffer、
MTE/UDMA、fresh/cached、FP8/BF16、expand/aggregate 以及原有资源接口。
各实现继续使用自己的 kernel、工作区、QP 调度和 handle，功能范围不互相推导。
PR（`CARD_4PMESH`，board `0x6c`）复用迁入实现的 MTE kernel，不另建一份 SHMEM。

V1 的统一入口为 `deep_ep/buffer.py`，V2 的统一入口为
`deep_ep/buffers/elastic.py`；POD 与 Clos 的实现分别位于
`deep_ep/backends/pod/` 和 `deep_ep/backends/clos/`。构造时选择一次后端，
后续 dispatch/combine 直接调用对应实现。Python 公开入口统一为 `deep_ep`；
V1 的 `deep_ep.buffers.legacy` 仅保留兼容导入，不存放另一份算子实现。

## 构建

继续使用本仓的[编译安装流程](build.md)，同一 Python 环境需有匹配的 Torch/torch_npu。
不额外安装第二份 SHMEM，不运行原 ascend_deepep 的安装脚本。
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
传 `num_sms=...`。核数须为合法偶数，且不少于 `ceil((EP-1)/peers_per_worker) * workers_per_peer`；
不足会报错。例如 EP128 自动策略需要 32 个发送 AIV，不能使用 20 核。
省略核数或传 0 时沿用 32 核缺省语义。

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

带宽入口位于 `benchmarks/clos/bench_v2_auto_bandwidth.py`；支持可选 L2 清理和
`--combine-bias none`，逐项打印通信带宽及独立的 notify/reduce 耗时。
通信带宽分母仅为通信 kernel 耗时，不能直接当作完整 API 吞吐。

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
后端专用测试直接导入 `deep_ep.backends.clos`，不会受开发机的自动分类影响。
导入、CPU 契约检查、编译成功都不等于 NPU 精度或性能验证通过。
