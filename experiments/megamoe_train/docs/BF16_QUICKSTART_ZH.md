# MegaMoE BF16 训练算子：快速开始

## 1. 环境确认

运行前确认以下三项。

1. 已安装启用训练组件的 wheel。构建时需设置 `DEEPEP_BUILD_MEGAMOE_TRAIN=ON`，该选项默认关闭。
   在源码目录之外验证导入，避免导入源码而非已安装包：

   ```bash
   cd /tmp
   python -c "import torch, torch_npu; from deep_ep._native import require_native; require_native('MegaMoE training installation check'); from deep_ep_experimental.megamoe_train import _C; print(_C.__file__)"
   ```

2. 已加载与 wheel 匹配的 CANN 环境。wheel 自带 SHMEM 运行库，运行时不需要 `SHMEM_ROOT` 或外部 SHMEM 动态库路径；使用新进程，避免预加载另一套 SHMEM。
   平台与版本要求见[仓库 README 的环境要求](../../../README.md#环境要求)。
   在目标运行镜像中构建；转移 wheel 时还需匹配 Python/Torch ABI 和 glibc 等系统库。
   根 Runtime、训练 Host 扩展和设备库须来自同一完整构建，各 rank 使用同一份 wheel。
3. 已分配可用的 Ascend950 设备。示例使用 2 张设备，`ASCEND_RT_VISIBLE_DEVICES` 仅包含本作业分配的设备。

## 2. 最小示例

将以下内容保存为 `/tmp/megamoe_train_first.py`：

```python
import os
import torch
import torch.distributed as dist
import torch_npu
from deep_ep_experimental.megamoe_train import MegaMoeBF16Train

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
dist.init_process_group(backend="gloo")          # 仅用于协调；token 数据经 SHMEM 传输
ep_group = dist.group.WORLD
device = f"npu:{local_rank}"

# 每 rank 规模：M 个 token，H 为 hidden，F 为专家中间维（GMM1 输出为 2F），
# T 为 TopK，E 为每 rank 本地专家数，EP 为 EP 组大小
M, H, F, T, E = 512, 5120, 2304, 6, 4
EP = dist.get_world_size(ep_group)

op = MegaMoeBF16Train(ep_group, local_mem_size=64 << 20, timeout=120)
x = torch.randn(M, H, device=device, dtype=torch.bfloat16, requires_grad=True)
# 每行取 T 个互不重复的专家编号
topk_ids = torch.argsort(torch.rand(M, EP * E, device=device), dim=1)[:, :T].to(torch.int32).contiguous()
weights = torch.full((M, T), 1.0 / T, device=device, dtype=torch.bfloat16, requires_grad=True)
w1 = (torch.randn(E, 2 * F, H, device=device, dtype=torch.bfloat16) * 0.01).requires_grad_()
w2 = (torch.randn(E, H, F, device=device, dtype=torch.bfloat16) * 0.01).requires_grad_()

y, counts = op(x, topk_ids, weights, w1, w2, EP * M * min(E, T),   # 末位参数为接收行容量
               swiglu_limit=0.0, save_gmm1_output="bf16")
y.float().square().mean().backward()
print(f"rank={dist.get_rank(ep_group)} y={tuple(y.shape)} counts={counts.cpu().tolist()}")

op.close()                                        # 集体释放，须在所有 backward 之后调用
dist.destroy_process_group()
```

执行：

```bash
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=0,1
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29862
torchrun --standalone --nproc_per_node=2 /tmp/megamoe_train_first.py
```

预期输出两行，形如 `rank=0/1 y=(512, 5120) counts=[...]`。`counts` 为该 rank 各本地专家接收的行数；
容量充足时，均衡路由下单 rank 的总行数为 `M*T`，随机路由下会波动；所有 rank 的总行数之和为 `EP*M*T`。

示例使用 BF16 p；将 `weights` 的构造 dtype 改为 `torch.float32` 即可使用 FP32 p，无需改算子类名或 SO 名称。
`weights.grad` 跟随输入 dtype，其他浮点输入及其梯度仍为 BF16。

## 3. 参数说明

### 3.1 构造 `MegaMoeBF16Train(ep_group, local_mem_size, *, timeout=120, debug_collectives=False)`

| 参数 | 取值 |
| --- | --- |
| `ep_group` | 调用方已有的 EP 进程组，可为 `dist.group.WORLD` 或子组，后端支持 Gloo 与 HCCL。组内 rank 顺序须与专家权重排列一致；不应另建第二个进程组 |
| `local_mem_size` | 每 rank 的通信窗口字节数，不小于 2 MiB、按 2 MiB 对齐，各 rank 相同。取值见 4.3 |
| `timeout` | 初始化与控制操作超时，单位秒，默认 120，取值 1..3600。该超时不覆盖设备内核中的全部等待 |
| `debug_collectives` | bool，默认 False，各 rank 构造时须一致。True 时在每次原生前后向前增加跨 rank dtype/阶段检查 |

默认关闭逐次集体诊断，本地 dtype、流和生命周期检查仍保留；框架须保证同次调用各 rank 的 p dtype 和前后向顺序一致。
调试检查不能识别所有计算图身份或调用顺序错误，其协调与等待会计入调用开销。

### 3.2 前向 `op(x, topk_ids, topk_weights, w1, w2, max_recv_tokens, *, swiglu_limit=..., save_gmm1_output=...)`

| 参数 | 形状 / 取值 | 说明 |
| --- | --- | --- |
| `x` | BF16 `[M,H]` | 本 rank 的 token。`M` 为每 rank 的 token 数，EP 不再切分 |
| `topk_ids` | int32 `[M,T]` | 专家编号取值为 `[0, EP*E)`，实际编号为 `ep_rank*E + 本地专家序号`。同一 token 内不得重复 |
| `topk_weights` | BF16 或 FP32 `[M,T]` | 可直接保留 router 的 FP32 权重；需路由梯度时不得 detach。同次调用各 EP rank dtype 相同 |
| `w1` | BF16 `[E,2F,H]` | 前 F 行为 Gate、后 F 行为 Up；另支持 `[E,H,2F]` 布局 |
| `w2` | BF16 `[E,H,F]` | 另支持 `[E,F,H]` 布局；权重梯度保持输入布局 |
| `max_recv_tokens` | int | 本 rank 所有本地专家合计的接收行容量，取值见 4.2 |
| `swiglu_limit` | float | 不大于 0 时关闭裁剪；为 L 时 Gate 取上界 L，Up 裁剪至 `[-L,L]` |
| `save_gmm1_output` | `bf16` / `none` / `mxfp8` / `auto` | 反向所用 Gate/Up 的保存形式。首次接入使用 `bf16` |

输出 `y` 为 BF16 `[M,H]`，已恢复本 rank 的 token 顺序；`counts` 为 int32 `[E]`，不可求导。
`backward()` 计算 `x.grad`、`topk_weights.grad`、`w1.grad`、`w2.grad`；其中 `topk_weights.grad` 与 p 同类型，其他三项为 BF16。
Gate/Up 保存模式与 p dtype 独立，FP32 p 不先降为 BF16，也不被 MXFP8 保存模式压缩。

Tensor 须为连续布局且位于同一张 NPU。构造前需设置当前 NPU 与 `DEEPEP_SHMEM_ENDPOINT`。
单个进程同一时刻只允许一个存活的算子实例。

## 4. 需自行确定的配置项

### 4.1 SHMEM 端点

构造前设置环境变量：

```bash
export DEEPEP_SHMEM_ENDPOINT=tcp://<IPv4>:<未占用端口>   # 端口范围 1024..65535
```

单节点使用回环地址。多节点时填写 EP 组 rank 0 所在机器的地址，所有 rank 使用同一值，且不得与
`MASTER_ADDR`/`MASTER_PORT` 的端口冲突。

### 4.2 接收行容量 `max_recv_tokens`

```text
max_recv_tokens >= EP * M * min(E, T)
```

该值为最繁忙 rank 的最坏接收量；按平均接收行数取值会截断路由。Host 不据此自动扩容。

### 4.3 窗口容量 `local_mem_size`

```python
def plan_bf16_window_bytes(m, hidden, topk, ep_size, experts_per_rank,
                           topk_weight_bytes=2):
    if topk_weight_bytes not in (2, 4):
        raise ValueError("topk_weight_bytes must be 2 (BF16) or 4 (FP32)")

    def align(v, a):
        return (v + a - 1) // a * a

    control = 72 * 1024 + align(64 * ep_size, 512) + align(72 * 4, 512) \
              + 4 * ep_size * align(ep_size * experts_per_rank, 128)
    payload = 4 * m * topk * hidden + align(topk_weight_bytes * m * topk, 512)
    return align(control + payload, 2 * 1024 * 1024)
```

函数返回字节数，默认 2 字节对应 BF16 p；FP32 p 传 `topk_weight_bytes=4`。
同一实例会切换 dtype 时按 4 字节及最大 shape 规划。它保守计入 compact 计数交换区，不是库导出的 API。

下表按 BF16 p 规划；FP32 p 仍应调用函数按实际字节数计算：

| M | H | F | T | E | EP | `local_mem_size` | `max_recv_tokens` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 5120 | 2304 | 6 | 4 | 8 | 62 MiB | 16384 |
| 4096 | 7168 | 3072 | 6 | 4 | 8 | 674 MiB | 131072 |

窗口按所有层与允许 shape 的最大值规划一次，各 rank 使用相同容量。该容量仅覆盖通信窗口，不含模型、
优化器状态、保存激活与临时工作区。

## 5. 常见错误

| 现象 | 原因与处理 |
| --- | --- |
| `ModuleNotFoundError: deep_ep`，或导入到源码 | 在源码目录之外验证；`PYTHONPATH` 不得包含空项或结尾冒号 |
| `Set DEEPEP_SHMEM_ENDPOINT=...` | 构造前未设置该环境变量，或格式不是 `tcp://IPv4:port` |
| `local_mem_size too small` | 按 4.3 重新计算；修改容量后须重建算子实例 |
| 接收行数不足、结果不完整 | `max_recv_tokens` 偏小，按 4.2 取值 |
| `dtype` / `contiguous` 报错 | IDs 须为 int32，x/专家权重须为 BF16，p 可为 BF16/FP32；全部 Tensor 连续且位于同一设备 |
| 调用阻塞 | 各 rank 的调用顺序或配置不一致，或存在提前退出的 rank；检查端点与端口占用 |
| 重新启动后仍失败 | 上一进程的 SHMEM 会话未释放；重启整个训练任务，不要只重启部分 rank |

## 6. 相关文档

- 字段级契约、保存模式、容量与限制：[用户指南](BF16_USER_GUIDE_ZH.md)
- 数值边界、内存复用与同步协议：[前后向设计](mega_moe_bf16_design.md)
- 精度与性能测试：[测试入口](../tests/bf16/README.md)、[测试规范](../tests/bf16/TEST_SPEC.md)
