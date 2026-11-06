---
name: test-megamoe-train-perf
description: 在已授权的 Ascend950 机器上，用仓库单入口跑 MegaMoE BF16 性能测试（三口径 Event 计时、baseline 对比、MFU/roofline、跨 case 汇总表），并核对结果与报告。用于性能回归、shape 曲线与回退定位；不负责精度判定、内核构建或修改门限。
---

# MegaMoE Train 性能测试

用仓库单入口 `--skip precision` 走性能段：不给精度结论，也不需要 ATK。

## 依据与执行范围

开始前读取当前分支的以下文件，命令均从仓库根目录执行：

- [测试入口说明](../../../../experiments/megamoe_train/tests/bf16/README.md)：前置条件、三种用法、输出结构与已知问题。
- [测试入口](../../../../experiments/megamoe_train/tests/bf16/test_mega_moe.py)：实际参数、三口径实现、baseline 与指标公式。
- [测试规范](../../../../experiments/megamoe_train/tests/bf16/TEST_SPEC.md)及[用例清单](../../../../experiments/megamoe_train/tests/bf16/cases.json)：12 条典型尺寸、覆盖理由和已知限制。
- [训练模块 README](../../../../experiments/megamoe_train/README.md)：构建、保存模式与数值边界。

只有用户明确指定并允许的主机、容器、物理卡和 shape 范围才能用于设备运行。运行前检查指定卡的
进程、显存和利用率，忙卡不启动，不换卡、不停止其他任务；共享机器上不清理他人进程或容器。
装环境、列用例或询问用法都不等于获准跑性能测试。

本 skill 不判定精度、不改阈值、不构建或重装 wheel、不把 `--skip precision` 的计时当作精度证据。

## 1. 对齐环境

默认使用容器默认 Python，不自行创建或切换 venv/conda：

```bash
source /path/to/cann/set_env.sh
python -c "import torch, torch_npu, deep_ep_experimental.megamoe_train"
```

完整 wheel 自带 SHMEM 运行库，根加载器校验哈希后加载包内文件；性能运行不需要外部 SDK 或其动态库路径。进程已加载另一套 SHMEM 时，重启自己的作业，不替换运行库或绕过校验。

性能段不需要 ATK；只需要正确安装的训练 wheel。先确认指定卡空闲：

```bash
npu-smi info | sed -n '1,40p'
```

记录待测源码 commit 与 dirty 状态、实际原生库路径和 SHA256；使用 Host/设备库匹配的完整
wheel，并保持 CANN、CATLASS 与输入一致。BF16/FP32 p 的新 tiling 布局不能混用旧 Host/设备 SO。

## 2. 选择 case 与 EP

EP 等于位置参数里设备列表的长度；`M` 是每 rank token 数，EP 不再切分。先干跑确认：

```bash
: "${MEGAMOE_TEST_DEVICES:?填写本轮用户指定的物理设备列表}"
python experiments/megamoe_train/tests/bf16/test_mega_moe.py "$MEGAMOE_TEST_DEVICES" \
  --suite priority --list-cases
```

`--list-cases` 只读配置，不导 Torch、不碰设备。四个 M 档（512/2048/4096/8192）横跨内核的
ready 分支门槛（均衡路由下每专家行数 `M×tk/E_local`），只测 4096/8192 时的结论不能外推到 2048/512。
用户未指定范围时按 `priority`（12 条）跑；形状与开关取自清单，suite 内不允许覆盖。

## 3. 执行

每轮使用新的报告目录，端口未被占用（Gloo 用 `port+2`）：

```bash
: "${MEGAMOE_TEST_DEVICES:?填写本轮用户指定的物理设备列表}"
: "${MEGAMOE_TEST_ENDPOINT:?填写本轮未占用的 SHMEM TCP 端点}"
run_dir="$PWD/perf-runs/$(date +%Y%m%d-%H%M%S)-$$"

python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py "$MEGAMOE_TEST_DEVICES" \
  --skip precision --suite priority --log quiet \
  --warmup 2 --loops 10 \
  --shmem-ip-port "$MEGAMOE_TEST_ENDPOINT" \
  --output "$run_dir"
```

- `--suite` 复用 worker、进程组与算子实例，初始化只付一次；12 条比逐条跑省一个数量级的固定开销。
- 单条复测用 `--case-id`；要改 shape/路由/保存模式就去掉 `--suite`，按 README 传单 case 参数。
- 怀疑会卡住时把 `--log` 换成 `rank0`：保留 rank 0 的阶段日志与 `[PERF] stage` 标记。
- 单条只看趋势时 `--baseline off`；需要 speedup 时必须保持默认 `on`。
- 同一 shape 建议 `warmup ≥ 2`、`loops ≥ 10`；样本少时跨进程重复，不要用单次结果下结论。
- 运行期间不更新源码、不重装 wheel；异常后完整重启自己的作业，不只重启部分 rank。

## 4. 判读

三口径：`forward`、`backward`（建图在窗口外）、`complete`；每轮窗口前后各同步一次，不使用设备 Timer。
两侧记作 `megamoe_bf16` 与 `baseline`（小算子链），`speedup = baseline ÷ megamoe_bf16`，>1 表示融合更快。

关键量：`floor = RFK × M × topk × F × K ÷ 410.4e12`（前向 6、反向 12/16、complete 18/22）；
`MFU = flops ÷ measured ÷ 432`；`overhead = measured − floor` 是该窗口的非 GEMM 时间。

- `floor` 与路由分布无关；保存模式三口径为 `6:12:18`，none 重算为 `6:16:22`；它是**下界**（假设窗口 100% 是 GEMM）。
- `MFU` 的分子只有 GMM FLOPs，窗口内还含 alltoall/permute/激活，因此是**算子级 GMM 利用率**，
  不是模型级 MFU；`432` 沿用源仓注释，引用绝对利用率前先与官方规格核对。
- 三个口径独立采样，`complete` 的 overhead 不等于前向与反向相加。
- `baseline` 的 `SKIPPED` 只是没有 HCCL 子组或超出打包路由预算，不影响 PASS/FAIL；`speedup < 1`
  也不判失败，只说明这一 shape 上小算子链更快。

先看 `[PERF]` 块与 `[SUMMARY]` 表，再回 `rank_<r>.json` 复核：`performance.baseline.status`、
`speedup`、`mfu`、`roofline`；跨卡偏差看每个 rank 的那一行，超过 1–2% 先怀疑卡忙或通信争用。

## 5. 记录与交付

保留：源码 commit 与 dirty diff、wheel 和原生库 SHA256、CANN/SHMEM 版本、物理卡号、命令、
`warmup`/`loops`、`suite_summary.json` 路径、`[SUMMARY]` 原文、失败/跳过项。

报告里写清范围：EP、case 列表、保存模式、clip、路由；没有跑的 shape 明确标未测。性能通过
没有自动阈值含义，也不能替代精度验收。历史数字只有在同 commit、同 Host、同 CANN、同 shape
与同 `warmup/loops` 下才可比。

## 6. 已知问题

- `process_group_setup` 花 40–50 s：Gloo 建组；设 `GLOO_SOCKET_IFNAME=<内网网卡>` 可消除。
- `--loops` 较大时性能段卡住且无输出：`--loops ≤ 10` 可稳定运行；用 `--log rank0` 看
  `[PERF] stage` 停在哪一段，把最后一行连同 `npu-smi info` 一起记录。
- `[INPUT]` 的 `received_rows` 与 balanced 预期不符、或 `empty_experts > 0` 时，后面的时延不可信，
  先查路由与容量。
