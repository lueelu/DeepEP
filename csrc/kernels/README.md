<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 设备算子

- `notify/notify.cpp`：路由元数据计算与交换。
- `dispatch/dispatch.cpp`、`combine/combine.cpp`：高吞吐 dispatch/combine。
- `dispatch/low_latency.cpp`、`combine/low_latency.cpp`：低时延入口。
- `dispatch/low_latency/`、`combine/low_latency/`：低时延设备实现、Host/Device 共用布局和启动声明。
- `launch.hpp`：LL 的导出和启动宏；`common.hpp`：Notify/combine 的 rank 调度映射。

Host 封装位于 `csrc/ops`，本目录的设备入口由 `cmake/device/CMakeLists.txt` 构建到同一个
`libdeepep_kernels.so`。
接口和测试命令见 [Elastic 文档](../../docs/elastic.md)。
