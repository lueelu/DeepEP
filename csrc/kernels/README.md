<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 设备算子

- `notify/notify.cpp`：路由元数据计算与交换。
- `dispatch/dispatch.cpp`、`combine/combine.cpp`：高吞吐 dispatch/combine。
- `combine/low_latency.cpp`：低时延 combine 入口；LL dispatch 由 DeepJIT 从 `deep_ep/include` 编译。
- `combine/low_latency/`：低时延 combine 设备实现；共享布局位于 `deep_ep/include/deep_ep/`。
- `launch.hpp`：LL 的导出和启动宏；`common.hpp`：Notify/combine 的 rank 调度映射。

Host 封装位于 `csrc/ops`，本目录的设备入口由 `cmake/device/CMakeLists.txt` 构建到同一个
`libdeepep_kernels.so`。
接口和测试命令见 [Elastic 文档](../../docs/elastic.md)。

UBX 的 RDMA route-plan Combine 单独位于 [`csrc/ubx`](../ubx/)，编译为独立设备库；本目录原有机型实现不参与替换。
