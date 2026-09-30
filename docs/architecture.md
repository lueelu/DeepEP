<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 架构与源码组织

DeepEP 的 Python 接口负责参数、路由 handle 与框架流依赖；C++ Host 层负责设备参数和启动；Device kernel 执行 token 通信与归并。模型侧提供路由、专家计算及与通信路径匹配的乘权操作。

## 调用链

```text
deep_ep.ElasticBuffer
    → deep_ep/buffers/elastic.py
    → deep_ep._C（csrc/bindings/module.cpp）
    → csrc/ops/{notify,dispatch,combine,low_latency}.cpp
    → csrc/kernels/ 中的对应 kernel

Python Runtime → C++ Session → SHMEM / CANN
```

`module.cpp` 集中注册运行时和算子入口。`csrc/ops` 校验 Host/Device 参数、组织 tiling 和 kernel 启动，不存在额外的 execution/plan 调用层。Host 扩展使用 C++17；Device 子工程用 Bisheng 编译为 `libdeepep_kernels.so`。构建和安装方式见[构建指南](build.md)。

## 模块职责

| 路径 | 职责 |
| --- | --- |
| `deep_ep/__init__.py` | 公开对象导出 |
| `deep_ep/buffers/elastic.py`、`_elastic_layout.py` | V2 参数校验、HT/LL 选择、容量计算、返回组织 |
| `deep_ep/buffers/legacy.py`、`deep_ep/_config.py` | V1 资源入口及兼容签名 |
| `deep_ep/_native.py` | 原生扩展延迟加载与 SDK 身份校验 |
| `deep_ep/_runtime.py` | 进程级资源 owner、框架通信流和进程组协调 |
| `deep_ep/utils/event.py` | NPU 事件、流等待与 Tensor 引用管理 |
| `csrc/bindings/module.cpp` | Python/C++ 注册入口 |
| `csrc/runtime/session.hpp/.cpp` | SHMEM 初始化、对称工作区、屏障与销毁 |
| `csrc/ops/` | Notify、HT 和 LL 的 Host 封装 |
| `csrc/kernels/` | Device 实现、启动声明和共用布局 |
| `cmake/`、`build_support.py` | 依赖校验、Host/Device 编译和安装 |
| `experiments/megamoe/` | MegaMoE Python API、融合 kernel、构建/多机启动脚本和计时工具 |
| `experiments/ep_gmm_fused/` | 两类通信计算融合算子、UDMA 运行时封装及独立测试 |
| `tests/`、`benchmarks/` | 正确性验证、测试参考及测量工具 |

设备文件导航见 [csrc/kernels/README.md](../csrc/kernels/README.md)。

## 高吞吐路径

fresh dispatch 先运行 Notify，计算接收计数、转发与归并元数据，再分配输出并搬运 payload。cached dispatch 使用同一 Buffer 的合法 handle，复用既有路由。展开布局保留每个有效专家槽位，包括重复路由；hidden、FP8 scales 和权重按各自格式搬运。

专家计算保持 dispatch 给出的行顺序。HT combine 根据 handle 汇总专家输出，可选地通过独立通道回传权重，不自动对 hidden 乘 gate。Session 的固定前半区用于 Notify/dispatch，后半区用于 combine；路由元数据和返回 Tensor 另有框架存储。支持范围和容量约束见 [Elastic 接口](elastic.md)。

## 低时延路径

LL 通过同一组 dispatch/combine 方法进入独立 kernel。fresh dispatch 在实际 shape、选项与容量符合要求时选择 LL；combine 按 handle 的路径执行。LL 使用固定 8 GiB 对称工作区，保存源端权重供 combine 加权。

使用显式 rank/world_size 时返回容量缓冲区。可选地传入已有 Gloo/HCCL EP group 并允许 CPU 同步，可通过该组汇总计数、返回精确接收行数；payload 仍由 SHMEM 和 LL kernel 传输。内部计数、路由表和有效区域由 LL 实现管理，不能按 HT 的字段与布局解释。

## 资源与完成关系

一个活跃 Buffer 对应一个进程级 Runtime 和 SHMEM Session。Runtime 管理通信流、在途引用与错误状态；Session 管理其自身的 SHMEM 和工作区，不拥有调用方的设备或进程组。

所有 rank 以一致顺序执行通信与显式销毁。事件为消费流建立数据依赖；Python 方法返回和向流添加等待都不等于 CPU 已观察到设备完成。详细约束见 [API 与运行时](api.md)。

## MegaMoE 融合流程

MegaMoE 将 token 分发、专家矩阵乘、激活与量化、第二次矩阵乘和加权归并组合为融合前向流程。其模型权重、计算配置和完整 MoE 性能口径与单独 EP 通信不同，说明集中在 [MegaMoE 文档](../experiments/megamoe/README.md)。

MegaMoE 的 `python/deep_ep_experimental/megamoe/api.py` 提供公开接口，经 `csrc/bindings.cpp`、`forward.cpp` 调用 `csrc/kernels/mega_moe/`；`scripts/` 提供构建和多机入口，`tests/` 检查接口、参考和主机协议。其 `SymmBuffer` 与 ElasticBuffer 分开管理，不能套用 HT/LL 的容量和初始化约束。

## 通信计算融合算子

EpGmmFused 在 `deep_ep_experimental.ep_gmm_fused` 下提供 AllToAllV → GMM 和 GMM → AllToAllV。模块使用根 Session 的可选 UDMA 入口管理通信域，通过独立开关构建；输入路由、工作区和进程组生命周期见 [EpGmmFused 文档](../experiments/ep_gmm_fused/README.md)。
