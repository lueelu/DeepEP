# 此维护分支的生产 LL 路径

本分支没有实验 checkout 的 metadata/native snapshot。生产算法只编辑 `csrc`；不要将其他分支生成目录复制过来作为实现。

| 目的 | 入口/实现 |
|---|---|
| Host API、参数上传、版本互检 | `deep_ep/buffers/elastic.py` → `csrc/ops/low_latency.cpp` |
| Session、对称内存和 transport 建域 | `csrc/runtime/session.cpp` |
| AIV 核数与逻辑组 | `csrc/runtime/ll_core_config.hpp`、`csrc/kernels/dispatch/low_latency/common/ll_core_config.h`、`ep_memory_server_dedup_layout.h`、`common/ll_group_policy.h`（tiling key4/16与物理机界） |
| dispatch 总入口 | `csrc/kernels/dispatch/low_latency.cpp`（单一 kernel，按 tiling 选择输入元素类与 group）→ `dispatch_server_dedup_kernel.h` |
| dispatch 计数、发送、转发、接收 | `csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_{counts,send,recv}.h` |
| combine 总入口/准备/通知/直接回传 | `csrc/kernels/combine/low_latency.cpp`、`low_latency/combine_server_dedup_kernel.h` |
| partial get/归约/回传、最终聚合 | `csrc/kernels/combine/low_latency/combine_server_dedup_{partial,source_aggregate}.h` |
| MTE/RDMA 收口和对称暂存契约 | `csrc/kernels/dispatch/low_latency/device/ll_transport.h`、`common/ll_rdma_layout.h` |
| 完整路径审计、风险与复现 | [RDMA AUDIT](../ll_rdma/AUDIT.md)、[RDMA README](../ll_rdma/README.md) |
| 完整 LL 数值与性能入口 | `tests/elastic/test_ep_low_latency.py`；[统计协议](PROFILING.md) |
| 可选设备打点 | `tests/ll_reference_profile/README.md`，仅 reference dispatch，默认编译 OFF、诊断单独采集 |

UB payload 打包不能直接交给 RDMA；只修改 payload 不足以完成协议。计数、ready、partial 远端读取、flag 发布、缓存可见性及复用必须一起检查。

跨分支共享实验入口：tests/ll_stage_timer/run.py；统计/trace：analyze.py；对比：compare.py；共享文件清单与核验：shared_files.json / verify_shared.py。

首次编译优化将14种dispatch模式分入口编译，通信算法不变；说明见[BUILD_COLD](../../docs/BUILD_COLD.md)。
