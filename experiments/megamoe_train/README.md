# MegaMoE 训练组件（Ascend950，实验性）

`MegaMoeBF16Train` 提供融合 Dispatch、GMM1、SwiGLU、GMM2 和 Combine 的 MoE 前向计算，并支持输入、路由权重和专家权重的自动求导。

## 构建与安装

支持 Linux、Ascend950，当前设备验收范围为单节点。环境与固定依赖要求见
[根 README](../../README.md#环境要求)和[构建指南](../../docs/build.md)。
在目标运行镜像中构建，转移 wheel 时须匹配 Python/Torch ABI 及 glibc 等系统库。

```bash
git submodule update --init third-party/shmem third-party/catlass
source /path/to/cann/set_env.sh
python -m pip install -r requirements-build.txt
python scripts/build_shmem.py
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export CATLASS_ROOT="$PWD/third-party/catlass"
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_BUILD_MEGAMOE_TRAIN=ON
python -m build --wheel --no-isolation
python -m pip install --force-reinstall --no-deps dist/*.whl
```

MegaMoE训练组件默认不构建，需设置 `DEEPEP_BUILD_MEGAMOE_TRAIN=ON`。请安装完整 wheel，
各 EP rank 使用同一版本。wheel 已包含 SHMEM 运行库，运行前加载匹配的 CANN 环境即可。

## 使用示例

在所有 EP rank 的运行环境中配置相同的 SHMEM 端点：

```bash
# 单机
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29862

# 多机：所有节点均使用 EP group 的 master 节点机器的可达 IPv4 地址
export DEEPEP_SHMEM_ENDPOINT=tcp://10.0.0.10:29862
```

多机示例中的 `10.0.0.10` 需替换为 EP group 的 master 节点所在机器的实际地址；
master 节点须能在该地址监听，其他 rank 须能访问。
各节点不要分别填写自己的 IP 或回环地址。端口范围为 1024..65535，须未占用且不与 PyTorch `MASTER_PORT` 冲突。
上述配置用于 SHMEM 初始化。

以下展示算子调用方式，输入参数见下文。

```python
from deep_ep_experimental.megamoe_train import MegaMoeBF16Train

# 构造 MegaMoE Train 算子：初始化 SHMEM 通信、分配每个 rank 的通信窗口，
op = MegaMoeBF16Train(ep_group, local_mem_size=local_mem_size)
# 调用算子
y, counts = op(x, topk_ids, topk_weights, w1, w2, max_recv_tokens,
               swiglu_limit=0.0, save_gmm1_output="bf16")
loss = loss_fn(y, target)
loss.backward()
op.close()  # 所有 rank 完成所需反向后集体关闭，再销毁 EP 进程组
```

## 对外接口与参数

### 构造 MegaMoE 和销毁

通过 `MegaMoeBF16Train(ep_group, local_mem_size, *, timeout=120)` 初始化算子，
`*` 后参数仅接受关键字传参。使用结束后，各 EP rank 调用 `op.close()` 释放通信资源。

| 参数 | 说明 |
| --- | --- |
| `ep_group` | 调用方已有的 ProcessGroup 或 `deep_ep` StoreGroup，不能为 None，由调用方管理；ProcessGroup 支持 Gloo/HCCL 初始化协调，token 数据仍使用 SHMEM MTE 后端 |
| `local_mem_size` | 每 rank 的固定 SHMEM 窗口字节数，至少 2 MiB、按 2 MiB 对齐，各 rank 相同；详见[local_mem_size](#local_mem_size) |
| `timeout` | 初始化和控制操作超时，整数秒，默认 120，范围 1..3600；进程组超时另行设置，不保证覆盖设备内核全部等待 |

所有 EP rank 使用相同的通信窗口容量和超时。

### 算子输入与输出

```python
y, counts = op(x, topk_ids, topk_weights, w1, w2, max_recv_tokens,
               swiglu_limit=0.0, save_gmm1_output="bf16")
```

M 为每 rank 输入 token 数，H 为 hidden，F 为专家中间维，T 为 TopK，E 为每 rank 本地专家数，EP 为组大小。

| 参数 / 输出 | 类型与形状 | 含义 |
| --- | --- | --- |
| `x` | BF16 `[M,H]` | 本 rank 输入；EP 不再切分 M |
| `topk_ids` | int32 `[M,T]` | 专家编号 `ep_rank*E+local_expert`，范围 `[0,EP*E)`，同一 token 内不得重复；使用 EP 组内 rank，非全局 rank |
| `topk_weights` | BF16 或 FP32 `[M,T]` | 路由权重 p，可求导；同次调用各 rank dtype 一致 |
| `w1` | BF16 `[E,2F,H]` | 前 F 行为 Gate、后 F 行为 Up；另支持 `[E,H,2F]` |
| `w2` | BF16 `[E,H,F]` | Down 权重；另支持 `[E,F,H]`，权重梯度保持输入布局 |
| `max_recv_tokens` | int，1..int32 最大值 | 本 rank 所有本地专家合计的接收行容量；详见[max_recv_tokens](#max_recv_tokens) |
| `swiglu_limit` | 可表示为 float32 的有限数值，默认 0.0 | <=0 关闭裁剪；正值 L 对 Gate 做上界 L 裁剪、对 Up 做 `[-L,L]` 裁剪；仅接受关键字传参 |
| `save_gmm1_output` | 字符串，默认 `bf16` | Gate/Up 保存方式，仅接受关键字传参；详见[save_gmm1_output](#save_gmm1_output) |
| `y` | BF16 `[M,H]` | Combine 后恢复本 rank token 顺序的输出 |
| `counts` | int32 `[E]` | 各本地专家接收行数，不可求导 |

输入 Tensor 须连续且位于同一 NPU。支持对 `x`、`topk_weights`、`w1`、`w2` 求梯度，
其中路由权重梯度与输入类型一致，其余梯度为 BF16。不支持二阶求导。

## 关键参数说明

### local_mem_size

`local_mem_size` 指定每个 rank 的固定 SHMEM 通信窗口大小，单位为字节，至少 2 MiB、按 2 MiB 对齐，
各 rank 须相同。它仅包含通信窗口，不含权重、优化器、计算图保存状态和临时工作区。
窗口不足会在调用时触发 Host 布局检查，不会自动扩容；增大窗口须重新构造算子。

可按以下公式估算，单位均为字节：

```text
通信数据 ≈ M * T * (4 * H + b)
计数表   = 4 * EP * align_128(EP * E)
local_mem_size = align_2MiB(通信数据 + 计数表 + 2 MiB)
```

其中 `b` 为路由权重的元素字节数，BF16 取 2、FP32 取 4；`align_n` 表示向上对齐到 n 的倍数。
通信数据主要包括两份 BF16 token 矩阵和路由权重，额外的 2 MiB 用于同步、计数交换和对齐余量。
多层共享或混用 dtype 时，按最大输入规模和 4 字节路由权重估算，各 rank 使用同一值。

例如 `EP=2、M=512、H=5120、T=6、E=4`，BF16 路由权重的通信数据约 60 MiB，
计入计数表和余量后向上对齐，可取 `local_mem_size=64 << 20`（64 MiB）。

### max_recv_tokens

`max_recv_tokens` 指定每张卡所有本地专家合计可接收的 token 数。同一个 token 路由到多个本地专家时，按专家数重复计数。
该值决定接收缓冲区、保存状态和相关工作区的分配大小，**取值越大，显存占用越高**；
它与 `local_mem_size` 分别控制接收侧计算内存和 SHMEM 通信窗口。

应根据路由分布确定各 rank 的最大接收量，并使用相同的 `max_recv_tokens`：

- 完全均衡路由下，各 rank 输入均为 M 个 token 时，每个 rank 接收 `M*T` 行。
  存在路由偏斜时，应按框架能保证的最大接收量增加容量。
- 若不能约束路由分布，在每个 token 的专家 ID 不重复时，最坏路由上界为：

```text
max_recv_tokens = EP * M * min(E, T)
```

这个上界覆盖任意合法路由，但可能显著增加显存占用，不宜直接作为常规配置。
例如 `EP=8、M=4096、E=4、T=6` 时，均衡接收量为 24576 行，最坏上界为 131072 行，
按上界分配的行容量约为均衡值的 5.3 倍。

容量不足会截断路由，导致结果不完整，算子不会自动扩容。因此须同时考虑显存预算和路由负载上限，
不能只用平均接收量代替最大接收量。

### save_gmm1_output

MegaMoE Train 支持多种激活保存方式。`save_gmm1_output` 控制反向计算所需的 GMM1 输出
（Gate/Up）的保存或重算方式，可在显存占用、重算开销和数值精度之间选择，默认使用 `bf16`。
该参数不改变矩阵计算类型或路由权重类型：矩阵计算仍为 BF16，路由权重支持 BF16/FP32。
`mxfp8` 及 `auto` 选择的 MXFP8 路径会引入压缩误差，目前为实验性功能。

| `save_gmm1_output` | 反向使用的 Gate/Up |
| --- | --- |
| `bf16`（默认） | 前向保存的 BF16 值，建议首次接入使用 |
| `none` | 根据保留的 packed X 和 W1 重算；仍保留路由和 packed 输入 |
| `mxfp8` | MXFP8 压缩值及 scale；开启裁剪时额外保存三个 mask 位平面 |
| `auto` | 关闭裁剪时使用 MXFP8，开启时使用 BF16 |

## 生命周期与使用限制

- 当前接口范围：`2<=EP<=1024`（StoreGroup 自身 size 不超过 128），`1<=E<=32`、`EP*E<=2048`，
  `1<=M<=16384`，H/F 为正且为 256 的倍数，`1<=T<=EP*E`；还须通过实际 UB 预算检查。
  接口上限不代表相同规模已完成设备验证。
- 每进程同一时刻仅允许一个存活的 SHMEM Session。多层共享同一 op、EP 组、设备和构造 stream，
  串行传入各层参数；不同时创建其他 DeepEP buffer、推理 Runtime 或 SHMEM Session。
- 前后向使用构造 stream，各 rank 的形状、容量、p dtype、保存模式、裁剪、构建策略和调用顺序一致。
  可排队多个前向并各自保留状态、dtype 和保存模式，对应反向顺序须跨 rank 一致；所需反向完成前不修改权重。
- 某 rank 校验失败或提前退出时，框架须终止整个作业。
- 完成所有所需反向或丢弃不用的图后，各 rank 按相同顺序调用 `close()`，再销毁 EP group。
  `close()` 同步调用 stream 并集体释放根 Session，成功后重复调用直接返回；GC 不执行集合清理。
  `retain_graph=True` 后仍需反向时应保持 op 打开。设备或集合通信故障后重启整个作业。
- 不支持跨流并发和外部窗口注入。实际 ProcessGroup 集成、EP 子组、更大 EP、多节点、TP/SP/PP、
  checkpoint、图编译/捕获及整网收敛均需针对部署环境单独验证。

## 测试

在仓库根目录执行，使用已安装训练组件的 Python 环境并加载匹配的 CANN。
精度测试另需在同一环境安装 `atk` 和 `atk-private-precision`；性能测试不需要这两个包。
将设备列表替换为分配给本作业的物理卡，SHMEM 端口及端口 +2 均须空闲。
入口自行启动测试进程，无需使用 `torchrun`。

```bash
# 精度：检查前向输出、四项梯度和专家计数
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite model --case-id flash_m512 --skip performance \
  --shmem-ip-port tcp://127.0.0.1:29862

# 性能：测量前向、反向及前向加反向耗时
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite model --case-id pro_m2048 --skip precision \
  --shmem-ip-port tcp://127.0.0.1:29862
```

用例见[用例清单](tests/bf16/cases.json)，支持 `--suite model/dtype/gather/all`；
去掉 `--case-id` 可运行所选套件，`--list-cases` 仅列出配置。
精度测试须预留 CPU FP64 参考计算及梯度所需内存；仅运行性能测试不构成精度验收。
可用 `--output <新目录>` 指定报告目录，该目录不能已存在。
