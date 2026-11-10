# MegaMoE 合入遗留事项

更新日期：2026-11-10。

本文记录尚未完成的核实与改进项，不代表已经验证通过或已经修改实现。

## 1. UBX MegaMoE 是否可以复用 Elastic UDMA 的资源配置

状态：待核实 SHMEM 队列映射与并发语义，待实机验证。

目标：评估 UBX MegaMoE 能否直接使用 Elastic UDMA 的通信资源配置；如仍需差异配置，以内部资源属性表达，减少公共 runtime 对算子名称及机型的依赖。保持 POD 调用行为和现有用户接口不变。

### 当前实现与已知事实

两者已经复用公共 Runtime/Session 生命周期，但初始化策略仍不同：

| 配置 | Elastic UDMA | UBX MegaMoE |
| --- | --- | --- |
| 通信引擎 | MTE＋UDMA | MTE＋UDMA |
| QP 配置 | 32 个 QP，按调度配置 peer masks，支持 dense 配置 | 每个远端 peer 使用 QP0，配置 QP 数量为 1 |
| 共享 jetty | 开启 | 关闭 |
| 队列使用方式 | 按 Elastic 的 owner/lane 调度使用 QP | 不同 AIV worker 分别负责不同 peer，使用 QP0 提交数据、flag 和完成回收 |

- 仅仅没有使用多个 QP，不构成不能复用的理由。分配多个 QP 后只使用 QP0，可能仅增加资源占用。
- 当前 Elastic 的 `CombineQpMask` 为每个非自身 peer 保留 QP0，`PeerMask` 在此基础上扩展。因此，当前 masks 不会因为缺少 QP0 而直接阻止 UBX 使用。
- UBX 使用 RO 发送数据、SO atomic 发布 flag；两者使用同一 peer 的 QP0。该顺序要求不依赖“总共只分配一个 QP”。
- 目前没有完成“UBX 直接采用 Elastic 全套资源配置”的实机验证，不能认定兼容或不兼容。

### 需要核实的问题

1. **共享 jetty 的实际映射。** 核对当前锁定 SHMEM 版本，在 UBX EP4/EP8/EP16 的实际拓扑下，不同 peer 的逻辑 QP0 是否映射到相同物理 SQ/CQ。现有代码注释指出 Clos 路径可能跨 peer 共享 SQ、MESH 路径独立；应以 SDK 实现和实机映射确认。
2. **并发提交与完成回收。** 如果不同 peer 共享物理队列，确认 SHMEM 是否支持多个 AIV worker 并发提交 WQE、更新队列状态以及执行 `quiet`/消费完成信息。不能仅凭 peer 编号不同认定队列独立，也不能未经核实认定共享必然不安全。
3. **RO→SO 顺序。** 确认复用后数据和对应 flag 仍走同一有序队列，接收端观察到 flag 时数据已可用；同时保留发送源内存生命周期和接收端缓存处理要求。
4. **资源开销与收益。** 比较使用 Elastic 多 QP 配置和现有单 QP 配置时的初始化耗时、通信资源占用及算子性能，判断直接统一是否值得。

### 涉及代码

以下路径相对于仓库根目录：

| 文件 | 作用 |
| --- | --- |
| `deep_ep/_runtime.py` | Python 公共资源生命周期、配置一致性检查及当前 MegaMoE profile 选择 |
| `csrc/runtime/session.hpp`、`csrc/runtime/session.cpp` | SHMEM 初始化、通信引擎/QP/jetty 配置、对称内存及资源释放 |
| `csrc/runtime/udma_peer_masks.hpp` | Elastic peer QP masks 的生成 |
| `csrc/kernels/common.hpp` | Elastic 调度映射与 `CombineQpMask`，包含默认 QP0 保留规则 |
| `csrc/kernels/udma_layout.hpp` | Elastic QP 数量与 owner/lane 到 QP 的映射 |
| `experiments/megamoe/csrc/kernels/mega_moe_urma/common/mega_moe_transport.hpp` | UBX QP0 的 RO 数据发送、SO flag 发布和 `quiet` |
| `dependencies.lock.json` | 确定核实所依据的 SHMEM 版本 |

### 验收与处理方向

- [ ] 给出锁定 SHMEM 版本的队列映射、并发提交及完成回收依据。
- [ ] 在 UBX EP4/EP8/EP16 验证候选配置，覆盖并发 peer 通信、连续多轮调用、空路由边及不同 BS；确认无挂起、数据损坏或跨轮 flag 问题，并通过精度验证。
- [ ] 对比初始化和运行开销，记录是否采用 Elastic 原配置及理由。
- [ ] 回归 POD 路径，确认调用接口、默认配置和执行行为不受影响。

如果确认可直接复用，则合并初始化配置，移除不必要的 UBX 专用分支。如果仅共享 jetty 等属性需要不同，则复用公共初始化流程，仅保留内部资源配置差异，不增加用户参数。后续整理时，将 EP 支持范围、算子容量限制等约束留在 MegaMoE 层；公共 runtime 负责资源配置的有效性、一致性与生命周期。

## 2. 何时使用 UBX，何时使用 URMA 作为名称

状态：待统一命名规范。

当前 backend 使用 `ubx`，实现目录已改为 `kernels/mega_moe_urma`，公共实现放在 `kernels/mega_moe_common`。内部命名空间暂时保留原符号以隔离两个编译目标；部分 kernel、trace 工具及内部符号使用 `urma`。两者含义不同：UBX 是机型，URMA 是通信方案，不应在代码命名中默认等价。

待确定并执行：

- [ ] 明确机型选择、tiling 和适用范围使用 UBX；通信传输及其协议使用 URMA/UDMA 时的边界，并核对 SDK 术语。
- [ ] 排查文件名、命名空间、kernel 符号、日志和性能解析中的混用；公共 MegaMoE 接口保持通用名称。
- [ ] 如需重命名 device 入口，同步更新构建、打包、profiler kernel 名称匹配和 trace 工具，避免仅改显示名称造成采样遗漏。

此项仅记录待办，本次不批量重命名 device ABI 或动态库。

## 3. 每个文件开头的 copyright 如何规范

状态：待明确仓库统一规范并核对文件来源。

- [ ] 区分主仓原有文件、从源分支引入的文件、修改过的第三方文件和本次新写的文件，核实实际版权归属及适用许可证。
- [ ] 明确年份、版权主体、SPDX/许可证说明及修改者署名的统一格式；修改者署名不等同于版权归属。
- [ ] 保留依法和依许可证要求必须保留的原始声明，不将不同来源的文件统一替换成同一个作者。
- [ ] 对复制后改名或合并的文件保留来源与许可信息，避免移动代码时遗漏声明。
