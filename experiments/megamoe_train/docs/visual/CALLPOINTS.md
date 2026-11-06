# 函数、数据与源码调用点

图中节点与下面的编号一一对应；在线链接固定到本次采集的源码提交。

<a id="00-overview"></a>

## MegaMoE BF16 训练做什么

### o_contract · 传入 token、路由与专家参数

调用方已完成路由，提供每个 token 的专家 ID 和路由权重 p。p 支持 BF16 或 FP32，算子边界不重新执行 router。默认数值示例使用 Ascend950、H7168、F3072、BF16 p、BF16 保存和 canonical 专家权重。

- **执行位置**：公开接口
- **读取**：X [M,H] BF16；IDs [M,T] int32；p [M,T] BF16 或 FP32；W1 [E,2F,H]、W2 [E,H,F]

源码：[bf16.py:132](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L132) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### o_experts · 本 rank 持有本地专家参数

每个 EP rank 持有 E 个本地专家；不同源 rank 的 route 汇聚到拥有专家的 rank。canonical W1 的前半、后半分别对应 Gate、Up；W2 把 F 维激活投回 H 维。max_recv_tokens 是预分配接收容量，不是每个专家的实际行数。

- **执行位置**：融合训练算子
- **读取**：weight1、weight2 的 shape；max_recv_tokens
- **写入**：n=2F、transW 布局选择

源码：[training.cpp:67](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L67) · [仓内文件](../../csrc/bf16/training.cpp)

### o_runtime · 一个显式 SHMEM 资源所有者

模块构造时创建训练 Runtime，复用根 DeepEP 的 SHMEM Session 所有权管理。Host 执行器借用窗口地址；窗口不由每次 forward/backward 单独创建和释放。

- **执行位置**：运行资源与构建
- **读取**：已有 EP group；local_mem_size、bootstrap endpoint
- **写入**：对称通信窗口、rank 身份
- **等待**：各 rank 初始化顺序一致

源码：[bf16.py:105](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L105) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### o_forward · 融合前向

专家卡拉取路由 token，计算 Gate/Up、SwiGLU 与 p 加权，再执行 GMM2。专家贡献回传到源 rank，按 token 的 TopK 槽位归并为 Y。具体依赖由 ready 和缓冲交接控制，不是每个阶段都做全局同步。

- **执行位置**：融合训练算子
- **读取**：X、路由、W1、W2；通信 context
- **写入**：Y [M,H]；counts [E]；前向持久状态
- **等待**：当前输入 ready；UB 复用与通信读写边界

```text
R(x) 表示将 x 舍入到 BF16。GEMM 使用 FP32 累加。
G = R(Xpacked · Wgᵀ)
U = R(Xpacked · Wuᵀ)
H = SiLU_FP32(G) × FP32(U)
Hp = R(H × FP32(p))
Yroute = R(Hp · W2ᵀ)
Y = R(Σ_FP32 Yroute)  # 按 TopK 槽求和后写回 BF16
```

源码：[mega_moe_bf16_forward.hpp:1060](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1060) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### o_counts · 计数与整数路由不求导

公开返回值是 (y, counts)。counts 记录本地专家接收行数，用于观测和反向恢复；其本身不参加求导。整数专家 ID 也没有梯度。

- **执行位置**：公开接口
- **读取**：前向 expert counts
- **写入**：不可微 counts 输出

源码：[bf16.py:57](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L57) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### o_backward · 融合反向

反向复用前向路由和保存状态，将原始 dY 分发到专家卡。专家卡完成激活梯度和权重梯度计算；dX 与每个路由权重的梯度回源，dW1、dW2 保留在专家所在 rank。

- **执行位置**：融合训练算子
- **读取**：dY [M,H]；保存的 Gate/Up、Xpacked、p、路由元数据；W1、W2
- **写入**：dX [M,H]、dp [M,T]；dW1、dW2
- **等待**：dO 数据 ready；dO/dX 生命周期与回源同步

源码：[mega_moe_bf16_backward.hpp:1248](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1248) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### o_gradients · autograd 对齐四类梯度

自定义 autograd 将 dX、dTopK、dW1、dW2 返回到对应浮点输入。dp 保留 [M,T]，不对 TopK 求和；本算子的 dX 是专家路径贡献，后续 router 分支和模型中的其他梯度由调用方计算图衔接。

- **执行位置**：公开接口
- **读取**：BackwardResult
- **写入**：x、p、w1、w2 对应的梯度

源码：[bf16.py:79](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L79) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### o_build · 可选训练组件与独立设备库

根构建显式启用训练组件，训练 _C 扩展与 libdeepep_megamoe_train_bf16.so 属于同一次构建。设备代码使用 CATLASS 矩阵计算原语与 SHMEM 通信窗口；这里没有调用 CATCCOS 的 DispatchFFNCombine 算子。

- **执行位置**：运行资源与构建
- **读取**：CANN、CATLASS、SHMEM、Torch/NPU SDK
- **写入**：训练 Python 扩展 _C；BF16 设备动态库

源码：[CMakeLists.txt:70](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/CMakeLists.txt#L70) · [仓内文件](../../CMakeLists.txt)

源码：[mega_moe_bf16_forward.cpp:5](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.cpp#L5) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.cpp)

<a id="01-modules"></a>

## 模块关系与构建边界

### m_python · 公开 Module 与 autograd

MegaMoeBF16Train 继承 nn.Module，管理通信窗口、执行器和参数检查。调用 op(...) 时，PyTorch 先进入这个 Module 的 forward，再由它调用 _MegaMoeBF16TrainFunction.apply(...)。

_MegaMoeBF16TrainFunction 继承 autograd.Function，分别定义前向和反向如何调用自定义 kernel。启用梯度且有浮点输入需要求导时，apply 把这个 Function 记录为计算图节点，并将输出 y 与该节点关联。

训练代码执行 loss.backward() 后，autograd 沿计算图传回 grad_y，并调度 Function.backward。反向接口由 Function 提供，Module 负责公开调用和资源管理。

- **执行位置**：Python 接口
- **读取**：Tensor 输入、保存模式、EP group
- **写入**：autograd 节点、y/counts

```text
op(...) → MegaMoeBF16Train.forward
        → Function.apply → Function.forward
        → op._executor.forward → 前向 kernel

loss.backward() → autograd 引擎传回 grad_y
                → Function.backward
                → op._executor.backward → 反向 kernel
```

源码：[bf16.py:82](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L82) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

源码：[bf16.py:48](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L48) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

源码：[bf16.py:146](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L146) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### m_runtime · 根 Runtime 管理所有权

训练包装类选择根 Runtime 的训练工厂，检查各 rank 的配置和设备身份，并参与进程级 SHMEM 所有者互斥。它与 Host BF16Executor 的计算职责分离。

- **执行位置**：资源与构建
- **读取**：EP rank/world、窗口容量、endpoint
- **写入**：Runtime.native、进程所有者记录
- **等待**：EP 配置交换

源码：[_runtime.py:137](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/deep_ep/_runtime.py#L137) · [仓内文件](../../../../deep_ep/_runtime.py)

### m_binding · 训练扩展注册对象

训练包自己的 _C 注册 BF16Executor 的构造、forward/backward/close，并把 ForwardResult、BackwardResult 字段映射给 Python。它与根 deep_ep._C 是两个用途不同的扩展。

- **执行位置**：C++ Host / 绑定
- **读取**：Host 类型与方法声明
- **写入**：_C.BF16、结果类型绑定

源码：[bindings.cpp:8](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bindings.cpp#L8) · [仓内文件](../../csrc/bindings.cpp)

### m_session · Session 持有 SHMEM 窗口

根原生 Session 创建和持有对称窗口。训练扩展得到的是可借用的基址，关闭窗口和 SHMEM finalization 仍由 Session 完成。模块不再建立第二个独立 SHMEM 生命周期。

- **执行位置**：资源与构建
- **读取**：rank/world、容量、endpoint
- **写入**：对称窗口、peer 地址
- **等待**：Session 集合初始化/关闭

源码：[session.cpp:336](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/csrc/runtime/session.cpp#L336) · [仓内文件](../../../../csrc/runtime/session.cpp)

源码：[module.cpp:64](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/csrc/bindings/module.cpp#L64) · [仓内文件](../../../../csrc/bindings/module.cpp)

### m_host · Host 执行器与启动桥接

执行器保留窗口身份和通信 context，forward/backward 函数做布局校验与分配。launcher 将 Tensor 转为地址，并在当前 NPU stream 调用 C 导出启动符号。

- **执行位置**：C++ Host / 绑定
- **读取**：Tensor、context、借用窗口
- **写入**：输出 Tensor、启动参数

源码：[training.cpp:472](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L472) · [仓内文件](../../csrc/bf16/training.cpp)

源码：[launcher.cpp:63](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/launcher.cpp#L63) · [仓内文件](../../csrc/bf16/launcher.cpp)

### m_tiling · Host tiling 与共享布局

Host 查询 AIC/AIV/UB 资源，解析形状与保存模式，计算通信、持久状态和临时空间容量。共享布局公式被 Host 和 Device 使用，具体设备地址视图在 workspace_view 中建立。

- **执行位置**：C++ Host / 绑定
- **读取**：M/H/F/T、EP、E、max_recv_tokens；设备资源、保存模式
- **写入**：MegaMoeTilingData、各区间大小和偏移

源码：[tiling.cpp:47](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/tiling.cpp#L47) · [仓内文件](../../csrc/bf16/tiling.cpp)

源码：[mega_moe_layout.hpp:274](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_layout.hpp#L274) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_layout.hpp)

### m_device · 前反向设备主控

AscendC MIX 入口创建设备对象，调用 Init 和 Process。主控组织路由、Dispatch、GEMM、激活反向、回源以及真实同步；同一流程在 AIC 与两个配对 AIV 上承担不同角色。

- **执行位置**：AscendC Device
- **读取**：tiling、context、Tensor 地址
- **写入**：Y 或四类梯度；算法状态与ready计数
- **等待**：数据依赖与内存复用边界

源码：[mega_moe_bf16_forward.cpp:20](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.cpp#L20) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.cpp)

源码：[mega_moe_bf16_backward.cpp:18](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.cpp#L18) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.cpp)

### m_catlass · 计算、激活与路由组件

catlass 目录适配 BF16 MMAD，epilogue 负责 SwiGLU/梯度和保存格式，routing 负责路由与回源，schedule 表达 ready 和任务坐标。它们由设备主控直接调用，数据经 GM/UB 和事件交接。

- **执行位置**：AscendC Device
- **读取**：tile、专家参数、路由元数据
- **写入**：计算 tile、回源片段
- **等待**：对应ready和槽归还

源码：[mega_moe_bf16_forward_catlass_impl.hpp:251](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp#L251) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp)

### m_build · 根构建选择训练组件

根 CMake 的训练开关控制根扩展中的训练 Runtime 工厂和训练子目录。子目录分别编译 Device 对象与训练 Host 扩展，链接 CATLASS 头、SHMEM/CANN、Torch/NPU 依赖，并安装同一组产物。构建边不表示运行调用顺序。

- **执行位置**：资源与构建
- **读取**：根 SDK 依赖与构建选项
- **写入**：训练 Host _C 与 BF16 SO

源码：[CMakeLists.txt:43](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/CMakeLists.txt#L43) · [仓内文件](../../../../CMakeLists.txt)

源码：[CMakeLists.txt:51](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/CMakeLists.txt#L51) · [仓内文件](../../CMakeLists.txt)

<a id="02-lifecycle"></a>

## 一次前后向与资源生命周期

### lc_construct · 构造 Module，记录当前 stream

调用方先选好本地 NPU，并传入已有 ProcessGroup 或 StoreGroup。构造函数记录当前 NPU stream，之后每次调用都使用这个 stream 和相同的 rank 调用顺序。

- **执行位置**：Module / autograd
- **读取**：ep_group、local_mem_size、timeout；当前 NPU stream
- **写入**：Module 实例、_stream

源码：[bf16.py:96](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L96) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### lc_owner · 建立共享通信所有者

训练 Runtime 做配置交换与所有权检查，再调用根原生训练工厂创建 Session；Module 初始化的 runtime barrier 确保所有 peer 的窗口初始化完成后再进入设备 generation 协议。

- **执行位置**：Runtime / Session
- **读取**：group身份、endpoint、容量
- **写入**：SHMEM 对称窗口、Runtime.native
- **等待**：各rank配置一致；初始化 barrier

源码：[bf16.py:105](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L105) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

源码：[module.cpp:64](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/csrc/bindings/module.cpp#L64) · [仓内文件](../../../../csrc/bindings/module.cpp)

### lc_context · 执行器借用窗口并建立 context

Python 取出根 Session 的 workspace_address 后构造训练 BF16Executor。执行器核对 rank/world 与已初始化的 SHMEM 身份，从同一窗口建立 peer 地址 context；它借用窗口，不负责单独初始化或释放 SHMEM。

- **执行位置**：Host / Tensor 资源
- **读取**：窗口基址、bytes、rank/world；参考NPU Tensor
- **写入**：BF16Executor、设备 context

源码：[training.cpp:455](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L455) · [仓内文件](../../csrc/bf16/training.cpp)

源码：[launcher.cpp:43](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/launcher.cpp#L43) · [仓内文件](../../csrc/bf16/launcher.cpp)

### lc_forward · 提交一次前向

自定义 Function 调用执行器前向。设备写入 Y、counts、saved_state、dispatch_output；前向函数返回 Tensor 句柄，设备执行与后续依赖沿同一 stream 排序，不把 Host 函数返回等同为设备全局完成。

- **执行位置**：Module / autograd
- **读取**：X、IDs、p、W1、W2；接收容量、保存模式
- **写入**：y、counts、每次调用的状态Tensor
- **等待**：Module仍存活且当前stream一致

源码：[bf16.py:50](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L50) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### lc_saved · 持久状态跟随本次计算图

ctx 保存 p、W1、W2、counts 和两份前向状态。saved_state 保存路由元数据及原始 Gate/Up，dispatch_output 保存 packed X 和 p；保存BF16时反向免于重算GMM1。不同前向图有各自的状态Tensor。

- **执行位置**：Module / autograd
- **读取**：前向结果与参数引用
- **写入**：ctx.saved_tensors
- **等待**：Tensor存活到对应反向

源码：[bf16.py:56](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L56) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### lc_scratch · 临时 workspace 按 stream 保活

前向临时激活和ready区只需要活到本次设备调用结束，Host 对参与异步执行的Tensor记录当前stream，避免分配器过早复用。反向另分配自己的scratch；这与保存到ctx的跨前后向状态是两种生命周期。

- **执行位置**：Host / Tensor 资源
- **读取**：当前stream、参与调用的Tensor
- **写入**：分配器stream使用记录

源码：[training.cpp:288](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L288) · [仓内文件](../../csrc/bf16/training.cpp)

### lc_backward · autograd 提交对应反向

自动求导传入grad_y，Function取出对应ctx的保存Tensor并调用执行器。四项梯度分别对应X、p、W1、W2；counts和整数IDs不求导。本实现使用once_differentiable，图内这里是一次反向边界。

- **执行位置**：Module / autograd
- **读取**：grad_y、ctx.saved_tensors
- **写入**：dx、dtopk_weights、dw1、dw2
- **等待**：同一Module、同一stream

源码：[bf16.py:62](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L62) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### lc_drain · 结束使用后排空创建 stream

调用方完成反向，或丢弃不再使用的计算图后，集合调用close。close先同步创建stream，确保已提交的训练设备访问结束，再释放共享通信资源；已关闭的Module再次close直接返回。

- **执行位置**：Module / autograd
- **读取**：_closed、创建stream
- **写入**：已排空的设备工作
- **等待**：_stream.synchronize()

源码：[bf16.py:150](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L150) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### lc_release · Session 集合释放窗口

根Session的close在锁内检查context，完成SHMEM barrier后释放对称窗口并finalize。资源所有权在成功清理后才释放，随后Python关闭Host执行器。

- **执行位置**：Runtime / Session
- **读取**：当前context、Session窗口
- **写入**：释放的通信窗口、closed状态
- **等待**：aclshmem_barrier_all

源码：[session.cpp:349](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/csrc/runtime/session.cpp#L349) · [仓内文件](../../../../csrc/runtime/session.cpp)

### lc_closed · 执行器清除借用地址和 context

Host执行器把借用窗口地址置零并释放context Tensor。它没有在这里再次free SHMEM；新一轮独立生命周期需要重新构造Module和Runtime。

- **执行位置**：Host / Tensor 资源
- **读取**：address_、context_
- **写入**：address_=0、空context

源码：[training.cpp:466](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L466) · [仓内文件](../../csrc/bf16/training.cpp)

<a id="03-forward-call"></a>

## 一次前向的跨文件调用

### fc_public · 公开 forward 进入 autograd

调用 op(...) 时，nn.Module 的调用机制进入 MegaMoeBF16Train.forward。它校验 max_recv_tokens、clipping limit 与保存模式，再调用 _MegaMoeBF16TrainFunction.apply(self, x, IDs, p, W1, W2, ...)。

启用梯度且有浮点输入需要求导时，apply 建立自定义 autograd 节点并执行 Function.forward。PyTorch 因此知道后续反传应调用哪个 Function.backward。

这里的 save_gmm1_output 默认 bf16；本图选择 canonical 权重、H7168、F3072 和关闭 clipping。

- **执行位置**：Python / autograd
- **读取**：X、IDs、p、W1、W2、max_recv_tokens
- **写入**：自定义autograd调用

```text
op(...) → Module.forward(...) → Function.apply(...) → Function.forward(...)
```

源码：[bf16.py:146](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L146) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### fc_function · Function.forward 调执行器

Function.forward 先调用 op._check_ready，默认只在本地检查 p dtype 是否为 BF16 或 FP32，再调用 op._executor.forward。_executor 是训练包 _C.BF16 对象，负责启动前向 kernel。整数 IDs 用于路由，p 是 BF16 或 FP32 的可微浮点输入。

构造时设置 debug_collectives=True，才在每次原生调用前集体核对当前前向／反向阶段和 p dtype。各 rank 仍须保证 dtype 和调用顺序相容；该检查不核对图身份、全部调用参数或完整操作序列。

ctx 是本次 Function 调用的上下文。前向结束后，它保存本次反向需要的 Tensor 和调用配置；反向时由 autograd 把同一个 ctx 交给 Function.backward。

- **执行位置**：Python / autograd
- **读取**：op、所有前向输入
- **写入**：ForwardResult引用
- **等待**：Module存活、创建stream一致

源码：[bf16.py:50](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L50) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### fc_binding · pybind 转入 BF16Executor

pybind把Python方法映射到C++执行器。执行器检查借用地址仍有效，取得通信context，再调用ops命名空间的MegaMoeBF16Forward。

- **执行位置**：绑定 / Host
- **读取**：Tensor对象、执行器rank/world/窗口
- **写入**：Host前向调用

源码：[bindings.cpp:24](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bindings.cpp#L24) · [仓内文件](../../csrc/bindings.cpp)

源码：[training.cpp:472](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L472) · [仓内文件](../../csrc/bf16/training.cpp)

### fc_host · 检查Tensor与容量边界

Host检查NPU、连续性、dtype、权重布局、token和专家范围，计算M/H/2F/T。接收容量由max_recv_tokens指定，Host据此规划缓冲；它不等于运行时每个专家实际收到的行数。

- **执行位置**：绑定 / Host
- **读取**：输入Tensor与shape；借用SHMEM容量
- **写入**：前向形状、weightLayout

源码：[training.cpp:195](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L195) · [仓内文件](../../csrc/bf16/training.cpp)

### fc_tiling · 生成设备tiling与空间预算

查询平台AIC/AIV/UB资源，解析保存模式、路由容量与前向调度参数。前向GM/UB策略由构建设备SO查询参与Host临时区规划，本图H7168且保存BF16使用UB交接。

- **执行位置**：绑定 / Host
- **读取**：shape、保存模式、平台资源
- **写入**：MegaMoeTilingData、workspace/peer字节数

源码：[tiling.cpp:47](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/tiling.cpp#L47) · [仓内文件](../../csrc/bf16/tiling.cpp)

源码：[launcher.cpp:38](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/launcher.cpp#L38) · [仓内文件](../../csrc/bf16/launcher.cpp)

### fc_allocate · 分配输出、持久状态与临时区

Y/counts为公开输出；saved_state和dispatch_output需要保留至反向；forwardTransientWorkspace只服务当前前向。Host按布局清理控制区，数据输出由kernel完整写入。

- **执行位置**：Tensor 资源
- **读取**：saved-state、dispatch、transient布局
- **写入**：Y [M,H]、counts [E]；saved_state、dispatch_output、forward scratch

源码：[training.cpp:269](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L269) · [仓内文件](../../csrc/bf16/training.cpp)

### fc_launch · 当前NPU stream上的启动桥接

launcher取当前Acl stream，将Tensor data_ptr转为设备地址，把tiling_host和block_dim传入设备库导出函数。它是Host桥接层，尚不是设备主控Process本身。

- **执行位置**：绑定 / Host
- **读取**：context、输入输出Tensor、tiling
- **写入**：C导出启动参数

源码：[launcher.cpp:78](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/launcher.cpp#L78) · [仓内文件](../../csrc/bf16/launcher.cpp)

### fc_export · C导出函数提交MIX kernel

C导出函数把Host tiling重解释为MegaMoeTilingData，以指定block数和stream提交mega_moe_bf16_forward。函数调用返回表示启动流程完成，不在图中额外添加一个全设备完成barrier。

- **执行位置**：绑定 / Host
- **读取**：裸地址、tiling_host、stream
- **写入**：已提交的MIX kernel

源码：[mega_moe_bf16_forward.cpp:34](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.cpp#L34) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.cpp)

### fc_device · 设备入口初始化前向对象

MIX入口初始化设备状态，构造MegaMoeBF16Forward，将持久、dispatch和transient地址映射到设备视图，再启用当前构建选择的Dispatch ready机制。

- **执行位置**：AscendC 设备
- **读取**：context、Tensor地址、tiling
- **写入**：前向主控对象、地址视图

源码：[mega_moe_bf16_forward.cpp:20](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.cpp#L20) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.cpp)

### fc_pipeline · Process执行融合前向

Process先解析保存模式与权重布局，再进入选定GM/UB模板流程。默认路径完成路由整理、计数、Dispatch、Gate/Up、SwiGLU×p、GMM2、回传与Unpermute；真实同步保护输入可见性和缓冲复用。

- **执行位置**：AscendC 设备
- **读取**：X、IDs、p、W1、W2
- **写入**：Y、counts、packed X/p、保存Gate/Up
- **等待**：当前输入ready、配对事件、跨rank边界

源码：[mega_moe_bf16_forward.hpp:1102](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1102) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### fc_return · 保存反向状态，公开返回Y和counts

Host通过recordStream保持异步调用所需的Tensor寿命，Function把反向需要的数据保存到ctx，公开只返回Y和不可微counts。反向将使用这次调用对应的状态，不从当前通信窗口猜测历史路由。

- **执行位置**：Python / autograd
- **读取**：ForwardResult、p、W1、W2
- **写入**：ctx.saved_tensors；(y, counts)

源码：[bf16.py:56](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L56) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

源码：[training.cpp:288](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L288) · [仓内文件](../../csrc/bf16/training.cpp)

<a id="04-backward-call"></a>

## 一次反向的跨文件调用

### bc_autograd · loss.backward() 触发反向

训练代码先用 y 构造 loss，再执行 loss.backward()。autograd 沿计算图算出 dL/dY，并将它作为 grad_y 传给 _MegaMoeBF16TrainFunction.backward(ctx, grad_y)。

Module 的 forward 在调用 Function.apply 时已建立这个节点与输出 y 的关系，因此这里由 autograd 引擎直接调度 Function.backward。

统一测试直接执行 y.backward(dy)，为每个 y 元素指定上游梯度。这等价于构造 loss=(y*dy).sum() 再反向。counts 已标记不可微；本算子为 x、p、W1、W2 返回梯度。

- **执行位置**：Python / autograd
- **读取**：grad_y [M,H]；ctx
- **写入**：反向Host调用

```text
y, counts = op(...)
y.backward(dy)

autograd → Function.backward(ctx, grad_y=dy)
         → op._executor.backward(...)
```

源码：[bf16.py:62](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L62) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

源码：[test_mega_moe.py:1613](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/tests/bf16/test_mega_moe.py#L1613) · [仓内文件](../../tests/bf16/test_mega_moe.py)

### bc_saved · 恢复本次前向的保存Tensor

ctx取回与该前向对应的weights、W1、W2、counts、saved_state和dispatch_output。grad_y被整理为连续Tensor后进入执行器。saved-bf16包含原始Gate/Up，反向因此跳过GMM1重算；none/mxfp8的差异在内核分支中处理。

- **执行位置**：Tensor 资源
- **读取**：p、W1、W2、counts；saved_state、dispatch_output；grad_y
- **写入**：连续grad_y、反向输入引用
- **等待**：Module仍存活且stream一致

源码：[bf16.py:63](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L63) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

### bc_binding · 绑定转入执行器backward

训练扩展把Python方法绑定到C++执行器。执行器带上同一窗口基址、rank/world和context调用ops::MegaMoeBF16Backward，并返回具名BackwardResult。

- **执行位置**：绑定 / Host
- **读取**：反向Tensor、保存状态、执行器资源
- **写入**：ops反向调用

源码：[bindings.cpp:25](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bindings.cpp#L25) · [仓内文件](../../csrc/bindings.cpp)

源码：[training.cpp:482](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L482) · [仓内文件](../../csrc/bf16/training.cpp)

### bc_host · 检查梯度输入与保存状态

Host从grad_y和权重恢复形状，检查dtype、连续性、有效保存状态、dispatch_output容量和路由边界。保存状态来自对应前向，不能被其他调用的scratch或通信窗口内容替代。

- **执行位置**：绑定 / Host
- **读取**：gradY、p、W1/W2、counts；saved_state、dispatch_output
- **写入**：反向shape与布局选择

源码：[training.cpp:291](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L291) · [仓内文件](../../csrc/bf16/training.cpp)

### bc_layout · 规划反向scratch与ready后缀

Host复用共享tiling规则、检查各阶段UB占用，再按保存模式和splitExperts计算scratch。默认FORCE_EXPERT_READY=OFF保留自动选择；统一整专家构建则令splitExperts=0。这里规划容量，并不表示所有输入已完成搬运。

- **执行位置**：绑定 / Host
- **读取**：tiling、savedMode、splitExperts；UB资源
- **写入**：反向workspace各区间大小

源码：[training.cpp:136](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L136) · [仓内文件](../../csrc/bf16/training.cpp)

源码：[training.cpp:423](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L423) · [仓内文件](../../csrc/bf16/training.cpp)

### bc_allocate · 分配四项梯度与scratch

Host分配gradX、gradTopkWeights、gradWeight1、gradWeight2与backwardWorkspace。kernel覆盖所有输出行和非空专家权重梯度，空专家对应权重梯度由设备清零；不把所有输出都预清零。

- **执行位置**：Tensor 资源
- **读取**：backwardLayout、输入Tensor options
- **写入**：dx [M,H]、dp [M,T]；dw1、dw2、backward scratch

源码：[training.cpp:430](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L430) · [仓内文件](../../csrc/bf16/training.cpp)

### bc_launch · Host桥接到反向C启动符号

launcher取得当前Acl stream，把梯度输入、输出、持久状态、scratch、tiling和splitExperts转为C导出启动参数。Host与Device采用同一ready构建策略。

- **执行位置**：绑定 / Host
- **读取**：context、Tensor地址、tiling、splitExperts
- **写入**：C导出调用参数

源码：[launcher.cpp:103](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/launcher.cpp#L103) · [仓内文件](../../csrc/bf16/launcher.cpp)

### bc_export · 导出函数提交反向MIX kernel

设备库的Host导出端解包tiling，在当前stream启动mega_moe_bf16_backward。C入口与前向入口彼此独立，但属于同一训练设备库。

- **执行位置**：绑定 / Host
- **读取**：裸地址、tiling_host、split_experts
- **写入**：反向kernel提交

源码：[mega_moe_bf16_backward.cpp:30](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.cpp#L30) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.cpp)

### bc_device · 设备对象恢复前向状态视图

MIX入口创建反向主控对象。Init把保存状态、dispatch_output、backwardWorkspace及梯度输出地址映射为设备视图，并记录本次splitExperts策略，然后进入Process。

- **执行位置**：AscendC 设备
- **读取**：context、输入输出地址、tiling；保存状态与scratch
- **写入**：反向主控对象与地址视图

源码：[mega_moe_bf16_backward.cpp:18](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.cpp#L18) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.cpp)

### bc_pipeline · Process执行四项梯度流程

主控分发原始dY形成dO，计算dHp和SwiGLU反向得到dZ/Hp/dp partial。保存路径按专家先算dW2，退休dO读者后原地写dX并回传，再计算dW1；AIV归并dX并归约回传dp。函数大阶段标题不等于所有核同时跨越的全局barrier。

- **执行位置**：AscendC 设备
- **读取**：dY、W1/W2、前向持久状态
- **写入**：源rank的dx/dp；专家rank的dw1/dw2
- **等待**：输入ready、dzReady、doReleased、dxReady；跨rank读写与复用边界

源码：[mega_moe_bf16_backward.hpp:1248](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1248) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward.hpp:912](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L912) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### bc_return · 按原输入顺序返回梯度

Host把四个Tensor组织为BackwardResult并记录当前stream，Python按forward输入位置返回梯度。x得到dx，路由权重得到dtopk_weights，两个专家参数得到dw1/dw2；op、IDs、容量、limit和模式没有梯度返回。

返回位置与 Function.apply 的输入位置一一对应。autograd 将 dX、dp、dW1、dW2 继续传向各自上游；对于需要梯度的叶子 Tensor，loss.backward() 会将梯度累积到对应的 .grad。

- **执行位置**：Python / autograd
- **读取**：BackwardResult.dx/dtopk_weights/dw1/dw2
- **写入**：对应输入的autograd梯度

源码：[bf16.py:79](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py#L79) · [仓内文件](../../python/deep_ep_experimental/megamoe_train/bf16.py)

源码：[training.cpp:448](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/bf16/training.cpp#L448) · [仓内文件](../../csrc/bf16/training.cpp)

<a id="05-forward-kernel"></a>

## 前向：Dispatch、GMM1、SwiGLU、GMM2 与回传

### f_select · 选择 H7168 的 UB 前向路径

当前构建将前向 SwiGLU handoff 设为 auto。H7168、保存模式为 bf16 时选择 UB 路径：GMM1 的 Gate/Up 先送入 AIV0 UB，再由 AIV0 保存并计算 SwiGLU。这里固定该路径学习。

- **执行位置**：全核协调 / 所有 AIV
- **读取**：hidden=7168；resolved savedMode=bf16；swiglu_limit 关闭

源码：[mega_moe_bf16_forward.hpp:1091](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1091) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[mega_moe_layout.hpp:152](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_layout.hpp#L152) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_layout.hpp)

### f_routes · 在源 rank 生成排序 route 行

所有 AIV 完成路由初始化：按专家整理源 rank 的 token，结果放在本 rank 的 SHMEM ptrA0。expandedRowIdx 将排序行映射回原 route，expandedRowIdxGather 是逆置换。它们随后把回传的 route 行恢复为原 token/TopK 槽位。整数专家 ID 已经由调用方给出，这里不重新计算 router。

- **执行位置**：全核协调 / 所有 AIV
- **读取**：X [M,H] BF16；topk_ids [M,TopK]
- **写入**：SHMEM ptrA0：排序 X；expandedRowIdx / expandedRowIdxGather；源 rank 专家计数
- **等待**：入口跨 rank 同步，确认本轮通信表可用

源码：[mega_moe_bf16_forward.hpp:1114](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1114) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[moe_init_routing_v3_apt.h:127](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/routing/moe_init_routing_v3/arch35/moe_init_routing_v3_apt.h#L127) · [仓内文件](../../csrc/kernels/bf16/routing/moe_init_routing_v3/arch35/moe_init_routing_v3_apt.h)

### f_route_p · 让 p 与排序 X 一一对应

按 expandedRowIdx 将原 [M,TopK] 概率整理到 SHMEM ptrRoutedTopkWeights。每条排序 route 对应一个原类型 p，BF16/FP32 均保留原始位模式，X 与 p 使用相同排序行号。批量 gather 不适用时走 GatherRoutedTopkWeights 回退，数据顺序与结果契约相同。

- **执行位置**：全核协调 / 所有 AIV
- **读取**：原 topk_weights；排序 route → 原 route 映射
- **写入**：SHMEM ptrRoutedTopkWeights [M×TopK]，与 p 同类型
- **等待**：排序映射已生成；gather UB 释放后再申请 Dispatch UB

源码：[mega_moe_bf16_forward.hpp:1124](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1124) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[mega_moe_bf16_forward.hpp:739](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L739) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### f_counts · 交换计数，确定专家接收地址

计数交换后，专家 rank 知道每个源 rank 给本地各专家多少行。GetCumsumForMMAIV 建立接收行前缀及回传所需的源端专家前缀，并在支持时准备静态生产阶段元数据。每个专家的 M_e 是实际接收行数；max_recv_tokens 是容量 R，两者不能混用。

- **执行位置**：全核协调 / 所有 AIV
- **读取**：各源 rank 的专家计数
- **写入**：本专家接收前缀；ptrSumBeforeRank；prepared phase 元数据；前向持久计数快照
- **等待**：源端排序 X/p 写完成；计数交换及既有核间同步

源码：[mega_moe_bf16_forward.hpp:327](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L327) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[mega_moe_bf16_forward.hpp:1145](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1145) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### f_ready_policy · 逐专家决定 row-ready 或整专家等待

默认 FORCE_EXPERT_READY=OFF 允许自动选择，不能理解为每个专家始终使用 row-ready。shape 需满足本地专家数不超过7及路由量门槛；实际专家至少768行才有资格。prepared 元数据有效时，每个合格专家独立使用自己的 word；否则只有首个非空专家候选，且仍检查实际行数。其余路径使用整专家完成计数。

- **执行位置**：全核协调 / 所有 AIV
- **读取**：M、TopK、E_local；当前专家 M_e；preparedPhaseEnabled_
- **写入**：当前专家 useTileReady

```text
GMM1 compute rows=192；ready rows=384；minimum expert rows=768
```

源码：[mega_moe_bf16_forward.hpp:1168](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1168) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[mega_moe_bf16_forward.hpp:1186](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1186) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[mega_moe_layout.hpp:407](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_layout.hpp#L407) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_layout.hpp)

### f_dispatch_x · 拉取当前 fragment 的 token 行

专家卡 AIV1 按 source/core 分工拉取 X。fragment 是该分工行段与本专家384行 ready block 的交集，可能只有几行，不要求每个源 rank 凑384行。当前普通前向每次 token 拷贝单位为1行，内部用事件0/1保护 ping-pong staging。一次 fragment 先下发其中所有 X 行搬运，再处理对应 p。

- **执行位置**：AIV1：Dispatch / Combine
- **读取**：源 rank SHMEM ptrA0 的源专家行段
- **写入**：本地 ptrA：该专家 packed X [M_e,H]
- **等待**：token staging 槽的上一次 MTE3 已读完

源码：[mega_moe_bf16_forward.hpp:805](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L805) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[mega_moe_layout.hpp:437](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_layout.hpp#L437) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_layout.hpp)

### f_dispatch_p · 拉取同一 fragment 的路由权重

对应 p 从远端排序概率表经 UB 搬入本地 ptrDispatchedTopkWeights，源、目标行段与刚才 X 完全一致。4KiB 是保留的 UB 容量，不是每次实际传输量；fragment 有128行时，BF16 p 为256B，FP32 p 为512B。事件2保护独立 p staging 槽，小数据仍有下发和同步成本。

- **执行位置**：AIV1：Dispatch / Combine
- **读取**：源 rank ptrRoutedTopkWeights
- **写入**：本地 ptrDispatchedTopkWeights
- **等待**：p UB 槽上一次 MTE3 完成；MTE2 读取完成后才下发本地写

源码：[mega_moe_bf16_forward.hpp:781](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L781) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### f_publish_ready · X 与 p 可见后发布 ready

row-ready 分支在 fragment 的 X 与 p 都写完后，建立 MTE3→Scalar 顺序，再给目标 ready block 累加完成行数。同一个384行块可以由多个源 rank 的短 fragment 共同完成。整专家完成计数也继续保留，便于其他专家或回退路径使用；p 尚未完成时不能发布当前协议的 ready。

- **执行位置**：AIV1：Dispatch / Combine
- **读取**：该 fragment 的 X/p 拷贝完成状态
- **写入**：本地 ready block 行计数；整专家 producer 完成计数
- **等待**：X 与 p 对本地 GM 的 MTE3 写入退休

源码：[mega_moe_bf16_forward.hpp:869](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L869) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### f_wait_xp · 按本任务的真实输入等待

AIC 调度到一个 M192×N256 任务时，检查它需要的 X 行是否可用。row-ready 保护的是对应384行块的有效行数，含尾块；整专家回退只在该专家首任务等 producer 计数齐全。任务顺序可以是 prepared phase 或 adaptive N stripe，不能假设总是按连续 M 行访问。

- **执行位置**：AIC：矩阵计算
- **读取**：当前 M tile 对应的 Dispatch ready；当前专家完成计数
- **等待**：该计算任务依赖的 X+p 已完整到达

源码：[mega_moe_bf16_forward_catlass_impl.hpp:323](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp#L323) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp)

### f_gate_up · 同一个 X tile 分别计算 Gate 和 Up

每个逻辑任务以同一 X tile 乘 Gate 与 Up 的相同特征列段，写到两块 UB。weightBlock=0/1 选择权重列 nLoc 和 nLoc+F。AIC 等 AIV0 归还旧 UB 后才可复用，Gate/Up 都完成才 NotifyVector；没有独立 Gate-only 的 SwiGLU 任务。

- **执行位置**：AIC：矩阵计算
- **读取**：Xpacked [M_e,H]；当前专家 W1 [2F,H]
- **写入**：Gate tile BF16 UB；Up tile BF16 UB
- **等待**：X+p ready；上一对 Gate/Up UB 已被 AIV0 消费

```text
G = R(Xpacked · Wgᵀ)
U = R(Xpacked · Wuᵀ)
GEMM 用 FP32 累加，R(x) 表示将结果舍入到 BF16
```

源码：[mega_moe_bf16_forward_catlass_impl.hpp:389](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp#L389) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp)

源码：[mega_moe_bf16_forward_catlass_impl.hpp:381](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp#L381) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp)

### f_save_raw · 保存原始 Gate / Up 给反向

AIV0 WaitForCube 后拿到完整 Gate/Up。saved-bf16 将两块原始 BF16 结果写入 training_saved_state；MTE3 读完 UB 才允许 Vector 覆盖。这些状态对应本次调用，不是跨层共享权重。反向可重建 H 与导数，因此不用重新执行 GMM1。

- **执行位置**：AIV0：保存 / SwiGLU
- **读取**：原始 Gate/Up BF16 UB
- **写入**：持久 savedGate / savedUp [R,F]
- **等待**：AIC 的 Gate 与 Up Fixpipe 均完成；保存读者完成后再覆盖 UB

源码：[block_epilogue_swiglu_forward_bf16.hpp:172](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_forward_bf16.hpp#L172) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_forward_bf16.hpp)

### f_swiglu_p · SwiGLU 后逐行乘路由 p

AIV0 从本地 packed p 表按输入类型批量加载当前 tile 各行的 p，一行一个值，广播到该行全部特征。BF16 p 提升到 FP32，FP32 p 直接参与计算。SiLU(G)、H=SiLU(G)×U 和乘 p 连续使用 FP32 寄存器中间量，最后一次舍入形成 BF16 Hp，交给 GMM2。p 在 GMM2 之前使用，后面的 Combine 不再乘一次。

- **执行位置**：AIV0：保存 / SwiGLU
- **读取**：Gate / Up tile；packed p [tileRows] BF16 或 FP32
- **写入**：Hp tile BF16 UB
- **等待**：p GM→UB 完成；原始 Gate/Up 保存完成

```text
H = SiLU_FP32(G) × FP32(U)
Hp = BF16(H × FP32(p))
```

源码：[block_epilogue_swiglu_forward_bf16.hpp:240](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_forward_bf16.hpp#L240) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_forward_bf16.hpp)

### f_publish_hp · 写 GMM2 输入并发布本专家计数

Hp 按专家 packed 行、F 列写入临时 GM ptrA2。MTE3 写入完成后，给本专家 SwiGLU task counter 加1；随后返回调用者，由 NotifyCube 归还这对 Gate/Up UB。当前专家 GMM2 入口消费的是全专家 Hp 就绪计数。

- **执行位置**：AIV0：保存 / SwiGLU
- **读取**：Hp BF16 UB
- **写入**：ptrA2 [R,F] BF16；SwiGLU→GMM2 task counter；AIV0→AIC 槽释放信号
- **等待**：Hp MTE3 写完成再发布计数

源码：[block_epilogue_swiglu_forward_bf16.hpp:128](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_forward_bf16.hpp#L128) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_forward_bf16.hpp)

### f_release_dispatch_ub · AIV1 释放 Dispatch 的 UB

AIV1 跑完自己的专家 Dispatch 循环，排空管线并销毁 firstPipe，才通知配对 AIC 本核 UB 可用于 GMM2 输出。split-state 路径只等本核 AIV1，不额外等所有其他核的 GMM1。GMM2 仍有专家 Hp ready 的独立门槛。

- **执行位置**：AIV1：Dispatch / Combine
- **读取**：本核 Dispatch staging 使用状态
- **写入**：GMM2_UB_READY_FLAG
- **等待**：本核 Dispatch 的所有 UB 读写退休

源码：[mega_moe_bf16_forward.hpp:1230](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1230) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### f_gmm2_wait · 等当前专家全部 Hp 可见

AIC 先消费本核 UB 释放信号；进入某专家首个 GMM2 任务时，再等该专家所有 GMM1/SwiGLU 任务计数齐全。此处是整专家 Hp 门槛，不是逐 M/N tile 的 Hp ready。不同核的执行进度可以不同，因此早专家的 GMM2 仍可能与别核 SwiGLU 尾部重叠。

- **执行位置**：AIC：矩阵计算
- **读取**：本核 GMM2_UB_READY_FLAG；本专家 swigluFlags
- **等待**：本核 Dispatch UB 已释放；当前专家全部 Hp task 已完成

源码：[mega_moe_bf16_forward.hpp:1256](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1256) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[mega_moe_bf16_forward_catlass_impl.hpp:676](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp#L676) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp)

### f_gmm2 · GMM2 输出到 AIV1 的双 UB 槽

读取当前专家 Hp 与 W2，将逻辑 [M_e,F]×[F,H] 分解为输出 tile；FP32 累加后 Fixpipe 输出 BF16 UB。前两个任务可分别用槽0/1；之后复用某槽前，必须等 AIV1 发送完该槽。pingpongIdx 是本核输出槽，不是路由 rank 或专家编号。

- **执行位置**：AIC：矩阵计算
- **读取**：ptrA2 当前专家 Hp；W2 [H,F]
- **写入**：AIV1 UB 槽0/1：专家输出 tile
- **等待**：当前专家 Hp ready；目标 UB 槽已被 Combine 归还

```text
Yroute = R(Hp · W2ᵀ)
GEMM 用 FP32 累加，R(x) 表示将结果舍入到 BF16
```

源码：[mega_moe_bf16_forward_catlass_impl.hpp:684](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp#L684) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp)

### f_combine · 将专家 tile 拆片回传源 rank

AIV1 WaitForCube 后，按当前专家 packed 接收行与各源 rank 行段求交集，将本 tile 的有效片段写回相应源 rank SHMEM ptrD。一个128行 tile 可以跨多个源 rank。源卡上每条 route 有独立返回位置；此处没有对最终 Y 做原子加，也不重复乘 p。发送完才归还 UB 槽。

- **执行位置**：AIV1：Dispatch / Combine
- **读取**：GMM2 BF16 UB tile；专家内 source 计数与前缀
- **写入**：各源 rank ptrD：排序 route 输出；UB 槽释放通知
- **等待**：GMM2 tile ready；MTE3 完成后才 NotifyCube

源码：[mega_moe_combine_send.hpp:68](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/routing/mega_moe_combine_send.hpp#L68) · [仓内文件](../../csrc/kernels/bf16/routing/mega_moe_combine_send.hpp)

源码：[mega_moe_bf16_forward_catlass_impl.hpp:702](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp#L702) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp)

### f_unpermute · 源 rank 按 token 累加各 route

排空 GMM2/Combine 尾部并建立跨 rank 完成边界后，所有 AIV 负责本 rank 的不同 token。使用 expandedRowIdxGather[token×TopK+slot] 找到 ptrD 返回矩阵的 route 行，将回传的 BF16 route 值提升到 FP32，按 TopK 槽顺序累加，最后一次舍入成 BF16 Y。该索引属于源 rank 的排序 route 布局，不是专家卡跨源拼接后的接收行号。

- **执行位置**：全核协调 / 所有 AIV
- **读取**：本 rank ptrD [M×TopK,H]；expandedRowIdxGather
- **写入**：Y [M,H] BF16
- **等待**：所有本地 Combine 管线退休；跨 rank 确认各 route 输出已送达

```text
acc[0] = FP32(Yroute[slot 0])
acc[j] = acc[j−1] + FP32(Yroute[slot j])，j = 1…TopK−1
Y = R(acc[TopK−1])
R(x) 表示将总和舍入到 BF16
```

源码：[mega_moe_bf16_forward.hpp:1287](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1287) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[mega_moe_unpermute.hpp:93](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/routing/mega_moe_unpermute.hpp#L93) · [仓内文件](../../csrc/kernels/bf16/routing/mega_moe_unpermute.hpp)

源码：[mega_moe_bf16_forward.hpp:1280](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1280) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### f_exit · 所有源 rank 读完 ptrD 才退出

Unpermute 的最后一次读取及输出写入排空后，再进行一次跨 rank 同步。更快的 rank 返回后可能立刻提交下一轮；这个边界确保它不会覆盖仍被慢 rank 消费的 SHMEM 内容。前向返回时持久 saved state 与 dispatch_output 留给匹配的反向，forward transient 的 Hp 不需要保留。

- **执行位置**：全核协调 / 所有 AIV
- **读取**：本轮 ptrD 最后读者退休状态
- **写入**：安全复用通信窗口的退出边界
- **等待**：所有 rank 的 Unpermute 已完成

源码：[mega_moe_bf16_forward.hpp:1290](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1290) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

<a id="06-backward-kernel"></a>

## 反向：从原始 dY 到四项梯度

### b_saved · 读取匹配前向的持久状态

反向读取前向保存的 packed X、packed p、原始 Gate/Up 和路由/计数信息。bf16 模式使用 GM dHp 路径；只有 none 模式在此反向中重新执行 GMM1。这里的 GM 表示激活梯度交接位置，与前向 H7168 选择 UB SwiGLU 是不同策略。

- **执行位置**：所有 AIV / rank 边界
- **读取**：training_saved_state；dispatch_output；grad_y [M,H]；W1/W2
- **写入**：本轮 backward scratch 地址视图
- **等待**：匹配前向完成且保存张量仍存活

源码：[mega_moe_layout.hpp:171](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_layout.hpp#L171) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_layout.hpp)

源码：[mega_moe_bf16_backward.hpp:1322](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1322) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_gather · 按排序 route 复制原始 dY

所有 AIV 按 expandedRowIdx[sortedRoute]/TopK 找到原 token 的 dY，直接复制 BF16 到源 rank SHMEM ptrA0。同一 token 选多个专家时，每条 route 都取得同一份上游 dY。这里不乘 p：前向是先乘 p 再做 GMM2，所以 p 应放在后面的 dH=p×dHp。

- **执行位置**：所有 AIV / rank 边界
- **读取**：grad_y [M,H] BF16；expandedRowIdx
- **写入**：SHMEM ptrA0：排序 raw dY
- **等待**：入口跨 rank 边界；gather 双 staging 槽复用依赖

源码：[mega_moe_bf16_backward.hpp:367](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L367) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward.hpp:409](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L409) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_restore_counts · 恢复前向计数与 source 分段

先核间及跨 rank 同步，保证各源 rank 的排序 dY 可见。AIV0 和 AIV1 随后按源 rank 分工，从持久快照恢复计数并准备符合条件的 producer 元数据。expert0 的 dHpReady cacheline word1 发布元数据完成，dY 搬运进度另由 row-ready 发布。

- **执行位置**：所有 AIV / rank 边界
- **读取**：保存的计数表与源段前缀
- **写入**：本轮计数视图；producer phase 元数据与完成 word
- **等待**：所有源 rank GatherRoutes 完成；元数据读写的核间同步

源码：[mega_moe_bf16_backward.hpp:1268](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1268) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_dispatch_do · 专家卡拉取各源 rank 的 dY

AIV0 按 source/core 分工，将远端 ptrA0 的原始 dY 拉入本地 ptrDO。地址的源偏移使用前向保存的独占专家前缀；目标偏移使用当前专家中各源段的行前缀。AIV1 没有 P0-B 任务，可以先到 P1 等待自己的数据。

- **执行位置**：AIV0：Dispatch / P1 / 回传
- **读取**：各源 SHMEM raw dY；source/expert 前缀
- **写入**：本专家 ptrDO [M_e,H] BF16
- **等待**：本 source 的排序 dY 可见；Dispatch staging 槽可复用

源码：[mega_moe_bf16_backward.hpp:555](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L555) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward.hpp:1290](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1290) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_do_ready · 发布真实 dO 行进度与专家完成

细分策略启用时，每个 source/core 段以最多256行分块搬运，MTE3写入退休后累加该 producer 完成行数。当前专家分工结束还给整专家 bwFlagDispatch 加1。关闭细分或不满足条件时，AIC用整专家计数等待；不能把元数据完成 word 当作 payload ready。

- **执行位置**：AIV0：Dispatch / P1 / 回传
- **读取**：本轮 dO GM 写完成状态
- **写入**：producer 行进度；本专家 Dispatch 完成计数
- **等待**：MTE3→Scalar 发布顺序

源码：[mega_moe_bf16_backward.hpp:592](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L592) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward.hpp:611](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L611) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_dhp · 计算 dHp，写到临时借用的 Hp 平面

AIC 按实际 row-ready 或整专家 ready 等输入，再用 BF16 dO 与 W2 计算 dHp，逻辑形状为 [M_e,H]×[H,F]。累加发生在 FP32，Fixpipe 输出 BF16 GM。这个阶段 ptrHp 里的内容暂时是 dHp，而不是重建的前向 Hp。

- **执行位置**：AIC：dHp、dW2、dX、dW1
- **读取**：ptrDO [M_e,H] BF16；canonical W2 [H,F]
- **写入**：ptrHp 暂存 dHp [M_e,F] BF16
- **等待**：本任务需要的 dO 行已到达；必要的 producer phase 元数据 ready

```text
dHp = R(dO · W2)
GEMM 用 FP32 累加，R(x) 表示将结果舍入到 BF16
```

源码：[mega_moe_bf16_backward_catlass_impl.hpp:304](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L304) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[mega_moe_bf16_backward.hpp:852](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L852) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_dhp_retire · 本专家所有 dHp 写退休

每个 AIC 排空本专家 Fixpipe/GM 写后，给该专家 dHpReadyFlag 加1。两路 AIV 等计数达到 blockNum 才开始消费本专家。这是整专家 dHp 退休边界；默认 saved-bf16 不使用 none 路径的每 tile Fixpipe→AIV 握手。

- **执行位置**：AIC：dHp、dW2、dX、dW1
- **读取**：本核 dHp 写退休状态
- **写入**：dHpReadyFlag，每 AIC 一次 arrival
- **等待**：PipeBarrier 保证 Fixpipe GM 写退休

源码：[mega_moe_bf16_backward_catlass_impl.hpp:326](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L326) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[mega_moe_bf16_backward_catlass_impl.hpp:286](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L286) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

### b_epi0 · AIV0 执行 SwiGLU Grad

反向每个 tile 要重建 Hp、计算 dGate 和 dUp，并归约 dp partial。相同专家行数 Mₑ 下，前向 GMM1 从 H 维算出 Gate 和 Up 共 2F 列，反向 dHp 从 H 维只算出 F 列，GEMM 计算量是前向 GMM1 的一半。dHp 的 AIC 计算通常更短，较难掩盖 SwiGLU Grad。当前专家的 dHp 写完 GM 后，两路 AIV 可以各读各的 tile，并行执行同一个 SwiGLU Grad。

同一核组按本地任务序号 nextTargetAiv & 1U 分配 tile。AIV0 处理 0、2、4…，AIV1 处理 1、3、5…。计数跨专家延续。

AIV0 先做完自己的 Dispatch 循环，再从 ptrHp 把分到的 dHp tile 读入私有 UB，并读取前向 Gate/Up 和 packed p。满足完整 128×256 tile 条件时走融合分支，尾 tile 走分步路径。

本路输出退休后给当前专家的 DZReady 加 1，AIC 等计数达到 2 × blockNum 再算 dW2。

- **执行位置**：AIV0：Dispatch / P1 / 回传
- **读取**：ptrHp 当前仍为 dHp 的 tile；saved Gate/Up；packed p
- **写入**：本 owner 的 UB 工作区
- **等待**：本专家所有 AIC dHp 写完成；AIV0 自身 P0-B 执行流完成

源码：[mega_moe_bf16_backward_catlass_impl.hpp:314](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L314) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[block_epilogue_swiglu_backward_bf16.hpp:168](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp#L168) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp)

### b_epi1 · AIV1 执行 SwiGLU Grad

AIV1 完成 Gather 和计数恢复后先进入 P1。当前专家的 dHpReady 达到 blockNum 后，它可以在 AIV0 搬运后续专家期间先做本专家的 SwiGLU Grad。AIV0 进入 P1 后，两路各从 GM 读取自己的 tile，分担重建 Hp、计算 dZ 和归约 dp partial 的工作。

nextTargetAiv & 1U 按同一核组的本地任务序号轮转，AIV1 处理 1、3、5…，AIV0 处理 0、2、4…。序号跨专家延续，所以新专家的第一个 tile 也可能分给 AIV1。

两路各写各的 dZ、Hp 和 dp partial。本路输出退休后给当前专家的 DZReady 加 1，AIC 等计数达到 2 × blockNum 再算 dW2。

- **执行位置**：AIV1：P1 激活梯度
- **读取**：本 owner 的 BF16 dHp tile；saved Gate/Up 与 packed p
- **写入**：独占 tile 的 Hp、dGate/dUp、dp partial
- **等待**：整专家 dHpReady 计数达到 blockNum

源码：[mega_moe_bf16_backward_catlass_impl.hpp:289](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L289) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[mega_moe_bf16_backward_catlass_impl.hpp:314](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L314) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[block_epilogue_swiglu_backward_bf16.hpp:787](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp#L787) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp)

### b_numeric · 产出 Hp、dZ 与 FP32 dp partial

由保存的 Gate/Up 重建与前向相同的 FP32 H，乘 p 后一次舍入写出 BF16 Hp。dH=FP32(dHp)×FP32(p) 留在寄存器中，参与激活导数计算；最后将 dGate/dUp 分别舍入 BF16，拼成 dZ。

dp 使用 FP32(dHp)×H，每行每 256 列生成一个 FP32 partial，跨 tile 继续用 FP32 归约。最终 dtopk_weights 与 p 类型相同：BF16 舍入，FP32 保留归约结果。

完整 128×256 tile 在一次读取中融合处理。尾块先产出 Hp/partial，再重读 Gate/Up 计算导数，两条路径使用相同的 FP32 H、dH 与 dp 乘积规则。

- **执行位置**：GM 中间量与精度边界
- **读取**：BF16 Gate、Up、dHp；原类型 BF16/FP32 p
- **写入**：ptrHp：BF16 Hp；ptrGradZ [R,2F] BF16；ptrDpProducts [R,ceil(F/256)] FP32
- **等待**：dHp tile 读入完成后才覆盖同地址 Hp

```text
R = BF16舍入；G、U、p、dHp 均按 FP32 参与计算
s = sigmoid(G)，S = SiLU(G)
H = S × U                         # FP32
Hp = R(H × p)
dH = dHp × p                      # FP32
dGate = R((dH × U) × (s × (1 + G × (1 - s))))
dUp = R(dH × S)
dZ = (dGate, dUp)
dp_tile = Σ_FP32(dHp × H)
dp = cast_to_p_dtype(Σ_FP32 dp_tile)
```

源码：[block_epilogue_swiglu_backward_bf16.hpp:751](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp#L751) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp)

源码：[block_epilogue_swiglu_backward_bf16.hpp:740](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp#L740) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp)

源码：[block_epilogue_swiglu_backward_bf16.hpp:805](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp#L805) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp)

源码：[block_epilogue_swiglu_backward_bf16.hpp:446](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp#L446) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp)

### b_dz_ready · 两路 AIV 退休本专家全部 P1 输出

每个核组的 AIV0 和 AIV1 在本专家的 dZ、Hp 和 dp partial 写入退休后，各给本专家的 DZReady 加 1。分到零个 tile 的 AIV 也到达一次。计数达到 2 × blockNum 时，本专家的 P1 输出齐全，AIC 可读取 Hp 计算 dW2。

- **执行位置**：所有 AIV / rank 边界
- **读取**：dZ/Hp/partial MTE3 退休状态
- **写入**：本专家 DZReady arrival
- **等待**：所有当前 lane 的 P1 输出退休

源码：[mega_moe_bf16_backward.hpp:860](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L860) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_wait_dz · 等待 DZReady 后算 dW2

saved-BF16 的 AIC 在 GMM1GradReuseDOPhase 里逐专家读取 DZReady。等所有核组的两路 AIV 都发布完成，即 DZReady >= 2 × blockNum，再用重建的 Hp 和原始 dO 计算 dW2。完成 dW2 后，另有 doReleased 确认 dO 的读者全部退出，才允许写 dX。

- **执行位置**：AIC：dHp、dW2、dX、dW1
- **读取**：DZReady；本专家 dO 与 Hp
- **等待**：该专家两路 AIV 全部输出退休

源码：[mega_moe_bf16_backward.hpp:933](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L933) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_dw2 · 提前计算并保存专家 dW2

canonical W2 的梯度形状是 [H,F]。dO 和 Hp 都是 BF16，AIC 沿该专家实际接收行数 M_e 做 FP32 累加，Fixpipe 写回 GM 时舍入成 BF16。梯度留在专家 rank，框架外层的 DP/FSDP 梯度同步另行执行。

所有 AIC 完成本专家 dW2 读写后，这段 dO 才能复用为 dX。

- **执行位置**：AIC：dHp、dW2、dX、dW1
- **读取**：原始 ptrDO [M_e,H]；重建 ptrHp [M_e,F]
- **写入**：本地 grad_weight2 [E_local,H,F] BF16
- **等待**：当前专家 DZReady >= 2 × blockNum

```text
dW2 = R(dOᵀ · Hp)
GEMM 用 FP32 累加，R(x) 表示将结果舍入到 BF16
```

源码：[mega_moe_bf16_backward.hpp:945](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L945) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward_catlass_impl.hpp:607](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L607) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

### b_do_release · 所有 AIC 共同释放 dO 行段

每个AIC排空本专家dW2的dO读取与梯度写入后，给 doReleased 加1，再等计数达到blockNum。零任务核也参加。此跨核边界保护一块内存从dO生命期切换为dX生命期，别的专家行段仍保持独立。

- **执行位置**：AIC：dHp、dW2、dX、dW1
- **读取**：本专家 dW2 读者退休状态
- **写入**：本专家 DOReleased 计数
- **等待**：全部 AIC 的 dW2 读者完成

源码：[mega_moe_bf16_backward.hpp:950](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L950) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_dx · 计算 dX 并覆盖已释放的 dO

AIC计算 [M_e,2F]×[2F,H]，FP32累加后输出BF16到ptrDX；saved-bf16中ptrDX直接别名ptrDO。完成本专家全部Fixpipe GM写后，各AIC给DXReady加1。AIV0不是拿到一个tile就立即发送，而是等本专家所有AIC输出退休。

- **执行位置**：AIC：dHp、dW2、dX、dW1
- **读取**：ptrGradZ=[dGate,dUp] BF16；canonical W1 [2F,H]
- **写入**：ptrDX=ptrDO [M_e,H] BF16；DXReady arrival
- **等待**：DOReleased 达到 blockNum

```text
dXpacked = R(dZ · W1)
GEMM 用 FP32 累加，R(x) 表示将结果舍入到 BF16
```

源码：[mega_moe_bf16_backward_catlass_impl.hpp:423](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L423) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[mega_moe_bf16_backward_catlass_impl.hpp:475](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L475) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

### b_dx_send · 将 dX route 贡献送回源 rank

AIV0等本专家DXReady达到blockNum，按计算tile从GM加载BF16 dX，经UB调用同一个CombineTokens辅助，拆分到各源rank的排序route返回位置。p已经在P1作用于dH，这里不再乘p。发送过程可以与AIC后续dW1等工作重叠。

- **执行位置**：AIV0：Dispatch / P1 / 回传
- **读取**：本专家 ptrDX；保存的 route/source 前缀
- **写入**：源 rank SHMEM ptrD：BF16 routed dX
- **等待**：本专家全部 AIC dX GM 写退休；UB 上一发送读者退休再加载

源码：[mega_moe_bf16_backward_catlass_impl.hpp:467](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L467) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[mega_moe_bf16_backward_catlass_impl.hpp:436](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L436) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

### b_dw1 · 计算专家 dW1，留在专家 rank

saved-BF16 各专家的 dW2 和 dX 路径走完后，AIC 用 BF16 dZ 和 Xpacked 计算 canonical [2F,H] 的 dW1。沿专家接收行做 FP32 累加，Fixpipe 写回 GM 时舍入到 BF16。dZ 和 Xpacked 保留到此处读完，空专家的权重梯度由 AIV 补零。

- **执行位置**：AIC：dHp、dW2、dX、dW1
- **读取**：ptrGradZ [M_e,2F]；前向 dispatch_output 的 Xpacked [M_e,H]
- **写入**：本地 grad_weight1 [E_local,2F,H] BF16
- **等待**：该专家 P1 dZ 写退休

```text
dW1 = R(dZᵀ · Xpacked)
GEMM 用 FP32 累加，R(x) 表示将结果舍入到 BF16
```

源码：[mega_moe_bf16_backward.hpp:968](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L968) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward.hpp:1419](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1419) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_dx_arrived · 所有 rank 确认 dX 回传结束

两路AIV都进入本地边界并排空各自管线后，再建立跨rank同步。AIV1较早到达不代表同rank的AIV0已经发送完dX。这个顺序让所有源rank读取ptrD时看到完整route贡献，可与AIC的权重梯度计算并行。

- **执行位置**：所有 AIV / rank 边界
- **读取**：各 AIV 的 dX 发送退休状态
- **写入**：P5 可读取 ptrD 的跨 rank 边界
- **等待**：全部本地 AIV 排空并到达；跨 rank 完成

源码：[mega_moe_bf16_backward.hpp:1420](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1420) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_unpermute_dx · 源 rank 用 FP32 累加 route dX

各AIV负责不同token，按expandedRowIdxGather读回TopK条BF16 dX贡献，提升到FP32按槽顺序累加，最后一次舍入BF16写grad_x。前向 Y 的归并也采用同样的 FP32 累加和最终 BF16 舍入。

- **执行位置**：所有 AIV / rank 边界
- **读取**：ptrD 排序 routed dX；原 route→排序行逆置换
- **写入**：grad_x [M,H] BF16
- **等待**：P5 dX 全部到达；结果 UB 的旧 MTE3 读者退出

```text
grad_x = R(Σ_FP32 dXroute)
沿 TopK 槽顺序累加，R(x) 表示将最终结果舍入到 BF16
```

源码：[mega_moe_bf16_backward.hpp:1049](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1049) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward.hpp:1071](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1071) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_reuse_ptrd · 所有 dX 读者结束后才能复用 ptrD

本地Unpermute结束、UB排空和核间同步后，还要跨rank确认所有源卡都完成dX读取。随后P6将ptrD开头复用为一维BF16路由概率梯度表。只有本卡读完不足以允许覆盖，因为其他rank可能仍处于上一阶段。

- **执行位置**：所有 AIV / rank 边界
- **读取**：所有 rank 的 ptrD dX 读者退休状态
- **写入**：ptrD 用途切换边界
- **等待**：本地读者退出；P5 后的跨 rank 同步

源码：[mega_moe_bf16_backward.hpp:1443](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1443) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_reduce_dp · 归约 dp partial，再批量回送

P1 已将每行每 256 列的 FP32 乘积归约成 FP32 partial。P6按source/expert分段把这些partial批量搬入UB，用FP32完成最终F维归约，按输入p类型写出：BF16舍入、FP32保留。一次MTE3发送一批同类型route标量到源rank的ptrD。没有再次乘p；dp是对独立路由权重输入求导。

- **执行位置**：所有 AIV / rank 边界
- **读取**：ptrDpProducts FP32 partials；source/expert 前缀
- **写入**：源 rank ptrD：紧凑 dp 表，与 p 同类型
- **等待**：ptrD 的 dX 生命周期已结束；routeBuf 上次发送完成

```text
dp_route = cast_to_p_dtype(Σ_FP32 dp_partial)
跨 F tile 以 FP32 累加，按 p 类型写出 BF16 或 FP32
```

源码：[mega_moe_bf16_backward.hpp:1129](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1129) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward.hpp:1215](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1215) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### b_dp_output · 恢复 dtopk_weights，清理后退出

先跨rank确认dp发送完整，再按route逆置换，把与p同类型的BF16/FP32标量按原始位恢复为[M,TopK]。这里是重排，不对同token的不同TopK槽求和。grad_x和dtopk_weights属于原token rank，dW1/dW2留在专家rank；整网router、共享专家和DP/FSDP同步由框架图继续处理。最后清理计数表并建立退出边界。

- **执行位置**：所有 AIV / rank 边界
- **读取**：ptrD 紧凑 dp 表；expandedRowIdxGather
- **写入**：grad_topk_weights [M,TopK]，与 p 同类型的 BF16 或 FP32；下一次调用可用的通信表
- **等待**：dp 回传完成的跨 rank 边界；最终表读者退休

源码：[mega_moe_bf16_backward.hpp:1228](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1228) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward.hpp:1454](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1454) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

<a id="07-memory"></a>

## 内存与生命周期：哪些保存，哪些复用

### m_sizes · 用容量 R 计算布局，用 M_e 执行任务

Host按max_recv_tokens=R、hidden=H、中间宽F、保存模式和p类型预留张量空间；p每项BF16为2B、FP32为4B。每个专家实际M_e和行前缀由本轮计数决定。布局按字节对齐，不能用M×TopK替换所有R，也不能把capacity当作专家GEMM的实际M。

- **执行位置**：Host 容量 / 地址契约
- **读取**：R、H、F、TopK、EP、E_local；resolved savedMode
- **写入**：对齐后的各平面偏移和总字节数

源码：[mega_moe_layout.hpp:255](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_layout.hpp#L255) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_layout.hpp)

源码：[mega_moe_layout.hpp:274](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_layout.hpp#L274) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_layout.hpp)

### m_dispatch_persist · 独立 dispatch_output 保留 packed X/p

普通BF16采用split-state：packed X与原类型BF16/FP32 packed p放在独立dispatch_output字节张量中。p类型与Gate/Up保存模式独立，none也保留p。前向Dispatch写入它们，反向分别用X计算dW1、用p计算Hp与dH。其生命周期跨越forward返回，不能当成forward transient随即释放。

- **执行位置**：前向持久 GM 张量
- **读取**：dispatchOutputBase；DispatchOutputLayout offsets
- **写入**：ptrA [R,H] BF16；ptrDispatchedTopkWeights [R]，与 p 同类型
- **等待**：匹配反向读完之前保存该张量

源码：[mega_moe_workspace_view.hpp:178](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp#L178) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp)

### m_saved_state · training_saved_state 保存数值与路由状态

saved-bf16将原始Gate/Up保存在持久GM，保留源计数、前缀、置换和模式元数据。这里没有保存专家权重副本，也不属于balanced planner的执行池。反向读取本次状态即可重建H，无需保留前向Hp或重算GMM1。

- **执行位置**：前向持久 GM 张量
- **读取**：原始 Gate/Up tile；前向 route/count 状态
- **写入**：BF16 savedGate/savedUp [R,F]；当前调用的路由与计数快照
- **等待**：Gate/Up Fixpipe 完成；保存读者结束后才覆盖 UB

源码：[block_epilogue_swiglu_forward_bf16.hpp:167](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_forward_bf16.hpp#L167) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_forward_bf16.hpp)

源码：[mega_moe_bf16_forward.hpp:437](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L437) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### m_forward_transient · 前向 Hp 与 ready flags 属于临时 GM

前向ptrA2是加权后Hp，供GMM2读取；SwiGLU→GMM2和Dispatch→GMM1 flags也在transient区域。反向使用persistent-only视图，不依赖forward scratch跨调用存活。反向自己的Hp由Gate/Up和p重建。

- **执行位置**：调用临时 GM
- **读取**：forward transient base
- **写入**：ptrA2 [R,F] BF16；forward ready counters
- **等待**：本次 forward 对临时区的使用退休后才能释放

源码：[mega_moe_workspace_view.hpp:196](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp#L196) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp)

源码：[mega_moe_layout.hpp:346](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_layout.hpp#L346) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_layout.hpp)

### m_peer_input · SHMEM ptrA0 保存源端排序 payload

同一对称窗口偏移在每rank都可被同组peer寻址。ptrA0前向放排序X，匹配反向重新写入排序raw dY；p有独立ptrRoutedTopkWeights段。各peer先按本轮计数和前缀定位源段，再拉入自己的packed接收矩阵。

- **执行位置**：SHMEM 通信窗口
- **读取**：source X 或 raw dY；SHMEM base 与 layout
- **写入**：ptrA0；ptrRoutedTopkWeights
- **等待**：前一用途的远端读者结束；本轮 gather 完成后再允许 peer 消费

源码：[mega_moe_workspace_view.hpp:298](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp#L298) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp)

源码：[mega_moe_bf16_backward.hpp:379](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L379) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### m_fwd_ub · GMM1 的两块 UB 是 Gate 与 Up

H7168 saved-bf16走UB交接：AIC把同一逻辑tile的Gate和Up写到两块AIV0 UB，再NotifyVector。AIV0保存并激活后NotifyCube归还存储；下一任务AIC的WaitForVector保护覆盖。不要把这两块解释成前后两个任务的ping-pong输出，它们在此处是一对SwiGLU输入。

- **执行位置**：核内 UB 与完成信号
- **读取**：X tile 与 Wg/Wu
- **写入**：AIV0 Gate UB；AIV0 Up UB
- **等待**：旧 Gate/Up 消费者归还存储

源码：[mega_moe_bf16_forward_catlass_impl.hpp:385](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp#L385) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp)

### m_dispatch_buffers · token 与 p 使用不同 staging 事件

AIV1 token staging使用事件0/1交替；p staging用事件2，保留4KiB。p每次搬运有效字节为fragment行数×2，并不总搬4KiB。不同event保护不同buffer依赖，不代表存在三套独立物理DMA引擎。

- **执行位置**：核内 UB 与完成信号
- **读取**：远端排序 X/p
- **写入**：本地 packed X/p
- **等待**：MTE3 旧读者完成后才能 MTE2 重填槽

源码：[mega_moe_bf16_forward.hpp:781](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L781) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

### m_ub_handoff · Dispatch 退出后 GMM2 才占 AIV1 UB

AIV1销毁Dispatch firstPipe后发本核释放信号，AIC才能将GMM2 Fixpipe输出指向这段存储。GMM2随后用两个UB槽按任务轮转，让计算与Combine发送重叠。槽归还由AIV1发送完成后的NotifyCube保证，和GMM1的一对Gate/Up输入是不同用途。

- **执行位置**：核内 UB 与完成信号
- **读取**：本核 Dispatch staging 退休状态；本专家 Hp ready
- **写入**：GMM2 AIV1 ping-pong UB
- **等待**：本核 GMM2_UB_READY_FLAG；某输出槽旧发送读者退休

源码：[mega_moe_bf16_forward.hpp:1230](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_forward.hpp#L1230) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_forward.hpp)

源码：[mega_moe_bf16_forward_catlass_impl.hpp:684](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp#L684) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_forward_catlass_impl.hpp)

### m_backward_planes · 反向按 R 预留 dO、dZ、Hp、partial

默认saved-bf16的主要临时平面是dO[R,H]、dZ[R,2F]、Hp[R,F]和dpPartial[R,ceil(F/256)]。前三者BF16，partial为FP32；none模式另需重算Gate/Up且使用不同交接。大小按R预留，但GEMM只计算各专家实际M_e。

- **执行位置**：调用临时 GM
- **读取**：R、H、2F、savedMode
- **写入**：backward scratch offsets；各专家 ready/retirement flags

源码：[mega_moe_workspace_view.hpp:236](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp#L236) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp)

源码：[mega_moe_workspace_view.hpp:258](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp#L258) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp)

### m_dhp_alias · ptrHp 第一生命期：dHp GM 输出

AIC 暂借 Hp 平面写入 dHp = R(dO · W2)，先用 FP32 累加，再舍入到 BF16。各专家的行段独立，本专家所有 AIC 完成写入并发布 dHpReady 后，AIV 才开始消费。

- **执行位置**：调用临时 GM
- **读取**：dO 与 W2
- **写入**：ptrHp 当前内容=dHp BF16
- **等待**：真实 dO 行就绪；dHpReady 达到 blockNum 才开始消费

源码：[mega_moe_bf16_backward_catlass_impl.hpp:246](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L246) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[mega_moe_bf16_backward_catlass_impl.hpp:326](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L326) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

### m_hp_overwrite · ptrHp 第二生命期：重建的前向 Hp

每个tile只有一个AIV owner；它先把该tile全部dHp搬入私有UB，再用前向Gate/Up与p生成BF16 Hp，覆盖同一GM位置。因为所有AIC dHp写已经退休，且其他AIV不读这个tile，所以不需要额外分配第二个[R,F]平面。后续dW2等待全部P1输出退休。

- **执行位置**：调用临时 GM
- **读取**：ptrHp 当前 dHp tile；saved Gate/Up、packed p
- **写入**：相同 ptrHp 地址上的 BF16 Hp
- **等待**：该 tile dHp GM→UB 完成；本专家 dHp 所有生产者退休

源码：[mega_moe_bf16_backward_catlass_impl.hpp:315](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L315) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[block_epilogue_swiglu_backward_bf16.hpp:805](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp#L805) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp)

### m_dz_partials · dZ 与 dp partial 保持独立平面

dGate和dUp分别以BF16落入dZ行的前F列和后F列，供dX及dW1读取。dp 用 FP32 乘积按 256 列归约为 FP32 标量，每个tile写独占位置，P6才跨tile求最终route梯度。ptrDpProducts这个历史名字当前存的是partial，不是完整[R,F]乘积矩阵。

- **执行位置**：调用临时 GM
- **读取**：dGate/dUp UB；每个 F tile 的 FP32 dp partial，乘积和求和都用 FP32
- **写入**：ptrGradZ [R,2F] BF16；ptrDpProducts [R,ceil(F/256)] FP32
- **等待**：MTE3写退休后发布DZReady

源码：[block_epilogue_swiglu_backward_bf16.hpp:798](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp#L798) · [仓内文件](../../csrc/kernels/bf16/epilogue/block_epilogue_swiglu_backward_bf16.hpp)

### m_do_last_reader · 先完成 dW2 对原始 dO 的读取

P1 的 dHp GEMM 读过 dO 后，原始 dO 还供 dW2 = R(dOᵀ · Hp) 使用。AIC 等 P1 写完 Hp 和 dZ，再算本专家的 dW2，以 FP32 累加并舍入到 BF16。全部 AIC 排空 dO 读取和 dW2 写入并发布 DOReleased 后，才允许 dX 覆盖该专家行段。

- **执行位置**：调用临时 GM
- **读取**：原始 ptrDO；重建 ptrHp
- **写入**：grad_weight2；DOReleased
- **等待**：DZReady 达到2×blockNum；DOReleased 达到blockNum

源码：[mega_moe_bf16_backward.hpp:945](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L945) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### m_dx_alias · ptrDX=ptrDO：切换为 dX 平面

saved模式直接将ptrDX设为ptrDO，少分配一个[R,H] BF16平面。完成DOReleased门槛后，dX GEMM才覆盖它。dX存于GM直到回传读者完成，AIC可继续执行后续dW1；不能把none模式的UB双槽回传协议套到这里。

- **执行位置**：调用临时 GM
- **读取**：ptrGradZ 与 W1
- **写入**：原 dO 地址上的 BF16 dX
- **等待**：全部 AIC 的 dW2 原始 dO 读者退休；AIV0等DXReady后读取GM dX

源码：[mega_moe_workspace_view.hpp:271](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp#L271) · [仓内文件](../../csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp)

源码：[mega_moe_bf16_backward.hpp:958](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L958) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### m_return_dx · ptrD 在 P5 存 routed dX 矩阵

专家rank将各route的BF16 dX回写源rank ptrD，布局是本源的排序route行×H。源rank按逆置换读回，FP32累加TopK后一次舍入BF16得到grad_x。AIC的dW1可与这段发送/本地归并重叠，ptrD的生命周期由AIV发送和读取边界保护。

- **执行位置**：SHMEM 通信窗口
- **读取**：专家 GM dX；原 route 逆置换
- **写入**：源 ptrD [M×TopK,H] BF16；grad_x
- **等待**：整专家 DXReady；本地排空、核间到达、跨rank回传完成

源码：[mega_moe_bf16_backward_catlass_impl.hpp:469](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp#L469) · [仓内文件](../../csrc/kernels/bf16/catlass/mega_moe_bf16_backward_catlass_impl.hpp)

源码：[mega_moe_bf16_backward.hpp:1026](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1026) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### m_ptrd_switch · P5所有读者完成后ptrD才能改用途

AIV排空Unpermute最后读写，销毁UB pipe并本地汇合，再跨rank同步，确认没有任何源rank仍读取routed dX。只有经过此边界，P6才能覆写ptrD开头。发送方结束不等于接收方读完，两个完成条件分别存在。

- **执行位置**：SHMEM 通信窗口
- **读取**：各 rank 的 ptrD dX 读者退休状态
- **写入**：可覆写 ptrD 的跨 rank 边界
- **等待**：所有 rank P5 最后读者退出

源码：[mega_moe_bf16_backward.hpp:1443](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1443) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### m_ptrd_dp · P6复用ptrD开头存紧凑dp表

P6将FP32 partial以FP32归约，按p类型生成BF16或FP32 route标量，回写源rank ptrD连续位置。此时ptrD不再解释为[M×TopK,H]矩阵，而是一维排序route表。dp发送全局完成后，UnpermuteTopkGrad按16/32位原样逆置换复制到[M,TopK]，不把TopK槽相加。

- **执行位置**：SHMEM 通信窗口
- **读取**：FP32 dp partial；原 route 逆置换
- **写入**：ptrD [M×TopK] BF16 或 FP32 dp；grad_topk_weights [M,TopK]
- **等待**：P5读者全退出；P6发送完成跨rank同步

源码：[mega_moe_bf16_backward.hpp:1215](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1215) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

源码：[mega_moe_bf16_backward.hpp:1238](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1238) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)

### m_window_exit · 表清理和退出同步保护下一轮

反向最后的route表读者退出后，清理token计数表，建立退出同步边界，保证下一次前向可以在干净通信表上开始。对称窗口是runtime资源；单次调用保存张量属于autograd状态。前反向完成并排空后再close，避免仍有设备读者时释放通信资源。

- **执行位置**：SHMEM 通信窗口
- **读取**：本轮通信表最后读者状态
- **写入**：清零token表；下一轮窗口复用边界
- **等待**：最终MTE/Scalar访问退休；所有rank退出边界

源码：[mega_moe_bf16_backward.hpp:1460](https://github.com/xixing-321/DeepEP/blob/0b5f385fd404a09446d317235d5e31c01471ed94/experiments/megamoe_train/csrc/kernels/bf16/mega_moe_bf16_backward.hpp#L1460) · [仓内文件](../../csrc/kernels/bf16/mega_moe_bf16_backward.hpp)
