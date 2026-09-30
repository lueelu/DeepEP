<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# API 与运行时

`deep_ep.ElasticBuffer` 提供 V2 dispatch/combine。高吞吐（HT）和低时延（LL）使用相同方法，输入、权重、布局及路径选择见 [Elastic 接口](elastic.md)。本页说明核心 `deep_ep` 的公共对象、初始化、资源生命周期和兼容范围。MegaMoE 的 `SymmBuffer/StoreGroup` 和 EpGmmFused 的 UDMA 域使用独立接口，分别见 [MegaMoE](../experiments/megamoe/README.md)与 [EpGmmFused](../experiments/ep_gmm_fused/README.md)。

## 公共对象

| 对象 | 导入路径 | 用途 |
| --- | --- | --- |
| `ElasticBuffer`、`EPHandle` | `deep_ep`、`deep_ep.buffers`、`deep_ep.buffers.elastic` | V2 通信资源、dispatch/combine 及配对路由信息 |
| `Buffer` | `deep_ep`、`deep_ep.buffers`、`deep_ep.buffers.legacy` | V1 资源构造、通信流、事件与销毁；V1 通信方法不在本版本支持范围内 |
| `Config` | `deep_ep` | 保留配置签名；构造和容量查询报 `NotImplementedError` |
| `EventHandle`、`EventOverlap` | `deep_ep`、`deep_ep.utils`、`deep_ep.utils.event` | 记录 NPU 事件、建立流依赖和保留相关 Tensor |

同一对象的不同导入路径指向同一个类。普通 `import deep_ep` 不加载 Torch、torch_npu 或原生扩展；运行时调用按需加载当前安装目录的扩展并校验 SHMEM 库身份。普通导入成功不表示运行环境、算法或设备验证通过。

## 初始化与资源

调用方先设置当前 NPU，并在所有 rank 配置相同的 `DEEPEP_SHMEM_ENDPOINT=tcp://<IPv4>:<port>`。端口范围为 1024～65535，使用作业独占且未占用的端口。多机作业使用所有节点可达的地址。

### 显式 rank/world_size

当前 HT `test_dispatch.py`、`test_combine.py` 和首页示例使用此方式：`group=None, rank=rank, world_size=ep_size`。Buffer 直接以显式身份初始化 SHMEM，无需创建 Gloo/HCCL 进程组。

`group` 默认值为 `None`，可以省略；此时必须提供 `rank/world_size`，不会自动使用框架的默认进程组。库仅执行本 rank 的配置检查，调用方保证所有 rank 的配置、设备映射和调用顺序一致。LL 也支持此方式，但返回容量缓冲区 `[C,H]`，即使设置 `do_cpu_sync=True`，也不会通过进程组取得精确接收行数。

### 可选的 Gloo/HCCL 进程组

框架已有 EP 进程组、需要借助它进行跨 rank 配置检查，或 LL 需要精确接收行数时，可以传入 `group=ep_group`。该组必须已初始化，后端为 Gloo 或 HCCL；由组内成员顺序确定 EP rank 和 world size，不能同时传入 `rank/world_size`。

| 使用位置 | 进程组的作用 |
| --- | --- |
| Buffer 初始化 | `all_gather` 检查工作区容量、超时、endpoint 和资源状态等配置 |
| HT dispatch/combine | 在启动前检查各 rank 的调用参数及错误状态；不改变 HT 的接收布局 |
| LL dispatch，`do_cpu_sync=None/True` | `all_reduce` 汇总路由计数，返回精确的 `[R,H]` |
| LL dispatch，`do_cpu_sync=False` | 不通过该组汇总精确接收行数，仍返回容量缓冲区 `[C,H]` |

两种身份配置方式下，token payload 均由 SHMEM 和对应 kernel 传输。Gloo/HCCL 在此承担协调和计数，不替代 dispatch/combine 的数据通信后端。LL 系统测试默认使用 HCCL 进程组，测试控制组与 Buffer 是否接收该组的区别见[测试说明](../tests/README.md#ll-正确性与性能)。

库不创建或销毁调用方的进程组，不初始化、重置或释放调用方的 NPU 设备。使用和销毁 Buffer 时保持创建时的设备与 ACL context。

### 容量与运行时

`Buffer/ElasticBuffer` 的 Runtime 接受 EP2/4/8/16/32/64/128/256，以及 2 MiB～32 GiB、按 2 MiB 对齐的对称工作区。各 rank 的容量相同。**这是资源准入范围；算子的 EP、shape 和容量限制另见 Elastic 文档。**

`ElasticBuffer` 使用 MTE+URMA Session。`num_bytes` 可显式指定；省略时由 HT size hint 计算，不能据此满足 LL 的固定 8 GiB 要求。`Buffer` 使用 MTE 配置，要求显式进程组、`num_scaleup_bytes` 和 `explicitly_destroy=True`，拒绝非零 `num_scaleout_bytes`；资源构造不启用 V1 通信算法。

## 生命周期与失败处理

一个进程同时只允许一个活跃 Session，不接管外部已初始化的 SHMEM。所有参与 rank 按相同顺序构造、调用和销毁；通信失败后应结束并重启整个分布式作业。

两个 Buffer 类都要求 `explicitly_destroy=True`。使用结束后，所有 rank 显式调用 `buffer.destroy()`：先等待通信流完成，再执行 SHMEM 集体清理、释放工作区和关闭 Session。如使用进程组，在 Buffer 销毁后再销毁该组。成功销毁后重复调用不再释放资源，其他运行时操作拒绝使用已关闭的对象。

忘记销毁时只发出警告并保留资源，不在 Python GC 或解释器退出时发起 collective。局部参数错误或某个 rank 退出也不代表其他 rank 已安全退出，调用方须管理整个作业的超时与终止。

`EPHandle` 由成功的 dispatch 返回，与创建它的 Buffer 绑定。不要修改内部路由 Tensor，也不要跨实例传入 handle。HT 支持在约束内 cached dispatch 和重复 combine；LL 的可用操作见 [Elastic 接口](elastic.md)。输出 Tensor 拥有独立框架存储，但 Buffer 销毁后 handle 不能继续用于通信。

## 流与事件

`get_comm_stream()` 返回真实 NPU 通信流。`Buffer.capture()` 返回 `EventOverlap`，`ElasticBuffer.capture()` 返回 `EventHandle`，均记录当前流的事件。

`event.current_stream_wait()` 向当前流添加依赖，不等同于 CPU 等待设备完成。读取输出、复用输入或交给另一条流之前，调用方须建立对应依赖。输入的 dtype/contiguous 转换应在 capture 之前完成。

`EventOverlap` 保留事件及相关 Tensor；不能用 `EventOverlap(None)` 表示已经完成。`register_hook_after_wait()` 注册的 hook 在添加等待后执行一次，不是设备执行完成回调。上下文管理器退出也会添加等待。指定 `release_handle=True` 后释放包装所持引用，该包装不能再次等待。

HT 的 `async_with_compute_stream=True` 允许通过返回事件连接后续计算，但计数与容量检查仍可能发生 Host 同步。LL 只接受默认流选项。`ElasticBuffer.barrier()` 只支持显式 `with_cpu_sync=True, sequential=True` 的同步用法。

## 上游兼容与平台命名

接口参考固定到 [DeepSeek DeepEP a56d6156](https://github.com/deepseek-ai/DeepEP/tree/a56d6156febcd9976e55adc85b5155bfac9f28f8)。[机器可读契约](../tests/api/api_contract.json)保存所选对象的参数、默认值、参考源码及哈希；它描述经过平台适配的接口，不代表上游全部 API 或行为均受支持。

`scaleup/scaleout` 表示通信域，`memory_transport` 表示内存传输语义，`compute_units` 表示计算单元，QP 表示后端队列对资源。通信域不等于物理节点或某种传输引擎。迁移关键字时不保留旧平台名称别名，完整签名以机器可读契约为准。

| 上游或旧名称 | 本项目名称 |
| --- | --- |
| `num_sms` | `num_compute_units` |
| `num_max_nvl_chunked_send_tokens/recv_tokens` | `num_max_scaleup_chunked_send_tokens/recv_tokens` |
| `num_max_rdma_chunked_send_tokens/recv_tokens` | `num_max_scaleout_chunked_send_tokens/recv_tokens` |
| `num_nvl_bytes/num_rdma_bytes` | `num_scaleup_bytes/num_scaleout_bytes` |
| `get_nvl_buffer_size_hint/get_rdma_buffer_size_hint` | `get_scaleup_buffer_size_hint/get_scaleout_buffer_size_hint` |
| `get_low_latency_rdma_size_hint` | `get_low_latency_scaleout_size_hint` |
| `num_tokens_per_rdma_rank` | `num_tokens_per_scaleout_rank` |
| `get_theoretical_num_compute_units(rdma_gbs=...)` | `scaleout_gbs=...` |
| `allow_scaleup_for_low_latency_mode/allow_multinode_scaleup` | `allow_memory_transport_for_low_latency_mode/allow_multinode_memory_transport` |

容量和开关名称不承诺同名对象属性，也不代表对应通信路径已启用。V1 的 `num_qps_per_rank` 仅接受默认值 24；Elastic 构造接受 `num_allocated_qps=0/6`，HT 调用接受 `num_qps=0/2`，LL 调用只接受 0。QP 不作为所有 DMA 引擎的统一资源计数。

另有以下接口差异：

- `ElasticBuffer` 支持 `group=None` 配合 keyword-only 的 `rank/world_size`；传入有效进程组时由组取得身份。`get_buffer_size_hint()` 对应使用 `group=None, world_size=ep_size`，或传入进程组而省略 `world_size`。`None` 表示选择显式身份方式，不表示禁止传入进程组。
- `dispatch(do_expand=None)` 用于区分省略与显式传值。fresh 默认 expanded；cached 继承原 handle 的布局，显式冲突会报错。普通去重布局不在支持范围内。
- `EPHandle` 增加 `is_low_latency`，指明实际执行路径，调用方据此遵守相应权重和返回约定。
- 配置、拓扑及调优查询只在实际实现的范围内可用；保留签名的方法可能报 `NotImplementedError`。V1 通信、autograd 和训练反向不在本版本支持范围内。

API 契约和导入测试、资源/事件测试、设备通信测试分别运行，入口见[测试说明](../tests/README.md)。
