# NDR首次编译：拆分dispatch特化入口

## 原因与改动

原dispatch_server_dedup_kernel把7种类型/量化模式×G4/G16共14套模板展开在一个global入口中。
A3实测时其余6个设备单元约40秒完成，dispatch的单个Bisheng进程持续占用CPU约17分钟。
拆分后每个源文件只实例化一种模式和G，编译器处理较小的函数，并可利用原有并行度4。
没有降低-O2，没有删除FP16/BF16/FP8模式，没有引入设备侧noinline调用或额外kernel启动。

- 公共算法仍在dispatch_server_dedup_kernel.h及原counts/send/recv实现中。
- variants/的14个入口共用dispatch_variant_kernel.h，保持Init/Process、计时、专家统计及对象析构顺序。
- Host从已有world/dtype字段选择一个入口，tiling生成与选择共用字段构造，不增加设备读回或同步。
- 设备入口再次校验上传的tiling与特化一致；不支持的组合明确拒绝。
- 152字节EpServerDedupKernelArgs与核心通信ABI5保持；内部启动符号为dispatch_server_dedup_kernel_do。Host与设备库必须整套更新，混装旧库会缺失符号而失败。
- 后续已回退为main的单入口单TU结构（一个global入口内按tiling选择输入元素类与G，不再有variants/与dispatch_variant_kernel.h）；本文记录的是拆分实验。当前结构以csrc/kernels/dispatch/low_latency.cpp为准。
- 当前NDR阶段schema3和EP16数据计数保留；泛化分支未做此次拆分。

## 同基线冷构建结果

2026-10-16，A3_163_docker，仅编译、未运行设备。原始5e8fbee与同基线加拆分的实验提交790f0d8对比：

| native阶段 | 原单入口 | 拆分14入口 |
|---|---:|---:|
| preflight | 0.177 s | 0.174 s |
| configure | 1.065 s | 1.065 s |
| build | 1052.501 s | 112.614 s |
| install | 0.064 s | 0.064 s |
| total | 1053.808 s | 113.919 s |

native总时间约缩短89.2%，速度约9.25倍。两次均为Release/O2、timer ON、并行4、Ascend950编译目标、空native对象目录；不是缓存命中。
SDK初始化/编译及pip解析、打包时间不计入表格。实验均保留全部14种模式；单BF16/G4编译探针20.44秒仅用于定位，没有作为发布构建裁剪功能。

环境使用Python3.14.6、G++13.4、CMake3.22.1、Bisheng基于clang15.0.5。工具链目录名为cann-9.3.0，但实际compiler/version.info报告9.2.0、timestamp=20260930_000324083；按实际版本记录。
SHMEM锁定4dd80e2129da42bd773982c2bb3758af1ed0bba8。原始数据在A3授权根的原工程/独立候选工程.logs/a3-build及build-timing.json中，不提交日志/二进制。
这证明当前环境的冷构建改善，不代表目标设备运行性能提升，也不能精确分摊并行编译与单函数优化成本的各自贡献。

## 构建与目标机验收

原README的clone、SDK准备和pip安装命令保持不变。使用默认并行4即可，不需要新的编译开关。
每次只编译当前选择的timer版本；plain/timer对象隔离与已有依赖检查继续保留。
首次构建现在包含5个HT对象和16个LL对象（14个dispatch特化、1个Host分发桥、1个combine入口），另外编译Host扩展。

tests/ll_stage_timer/compile_off.sh覆盖全部16个LL单元，固定生产RDMA编译能力，不再构建无RDMA能力的非生产对照。
更新后全rank安装匹配wheel，在目标机器验证BF16、FP16、预量化E4M3的dispatch/combine正确性、G4/16、timer/专家及schema3计数；A3只做编译和CPU测试，不进行NPU验收。

profiler里的dispatch任务名变为dispatch_server_dedup_kernel_{mode}_g{4|16}。
正式ST已识别精确14个名字及旧名字；同次采样混入不同入口/任务名会报错，仍按100条独立记录和最后50条统计。
外部自定义profiler筛选器须同步识别这些名字，不能将不同dtype/G/task合并。

## 验证范围

已完成同基线timer ON完整CANN wheel编译/安装，14个launch导出核验，以及真实G++的14种selector/Host分发、无效输入、timer ON/OFF和ABI测试。
首次候选缺少AICore函数标注导致编译失败，已按既有Host/Device公共函数模式修复，并从空对象目录重测。
最终版恢复与原分支一致的kernel对象析构作用域，并在当前schema3（488b959基线）从空native目录重测：preflight 0.173 s、configure 1.065 s、build 112.007 s、install 0.064 s、total 113.311 s，完整wheel编译安装成功。
另以空目录完成timer OFF设备配置/编译：94.692 s、21个对象；该时间不含Host扩展和wheel打包，不能与表格total直接比较。
本地隔离快照255项CPU回归通过、6项因无C++编译器跳过、10项Linux shell/Torch条件未选；A3上包含实际G++的selector/Host分发、profiler和schema3数据计数测试共56项通过，覆盖本地跳过的6项。
测试期间补齐了原有打包fixture缺失的scripts/shmem_sdk_identity.py，产品SDK逻辑未改。没有目标设备正确性或运行时性能结论。
