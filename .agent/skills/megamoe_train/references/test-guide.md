<!--
Copyright (c) 2026, Lu Lu
Modified by guofanfeng 2026
-->

# MegaMoE BF16 测试入口

本目录只有**一个**测试入口 `test_mega_moe.py`，精度与性能共用它。用例清单见
[cases.json](../../../../experiments/megamoe_train/tests/bf16/cases.json)；12 条典型尺寸、24 条类型/保存模式和 6 条 Gather 边界的覆盖理由见
[TEST_SPEC.md](test-spec.md)；模块级构建、接口与保存模式见
[模块 README](../../../../experiments/megamoe_train/README.md)。

## 前置条件

| 项 | 要求 |
| --- | --- |
| 设备 | 单节点 Ascend950；EP = 传入的设备列表长度，物理卡号 |
| 安装 | 已安装带训练扩展的 wheel（`DEEPEP_BUILD_MEGAMOE_TRAIN=ON`），默认 `python` 能导入 `deep_ep_experimental.megamoe_train` |
| 环境 | `source <CANN>/set_env.sh`；SHMEM 使用 wheel 内附的运行库，不预加载其他 SDK |
| 端口 | `--shmem-ip-port tcp://IP:port`；Gloo 另用 `port+2`，两者都不得被占用 |
| 精度 | 需要 ATK 与内部 `atk-private-precision`；`--skip precision` 不需要 |
| 性能 | 不需要 ATK，CPU 只做输入生成与（可选的）参考计算 |

## 两种用法

精度和性能用同一个入口，靠 `--skip` 选择跑哪一个。下面的设备列表 `0,1,2,3,4,5,6,7`、端口
`tcp://127.0.0.1:29862` 和用例 `flash_m512`、`pro_m2048` 都是示例，换成本轮实际分配的物理卡、未占用端口
（Gloo 再用端口 +2）和要测的用例。示例只跑一条；去掉 `--case-id` 会跑 `model` 套件的全部 12 条，
耗时明显更长。

### 精度测试

```bash
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1,2,3,4,5,6,7 \
  --suite model --case-id flash_m512 --skip performance \
  --shmem-ip-port tcp://127.0.0.1:29862 \
  --output "$PWD/runs/$(date +%Y%m%d-%H%M%S)-precision"
```

检查 `y`、`dx`、`dtopk_weights`、`dw1`、`dw2` 和 expert counts：CPU FP64 真值与独立 NPU
参考双路判定。判定器是内部 ATK 组件，需要 `atk` 与 `atk-private-precision` 装在与 wheel
相同的 Python 里；缺了会直接报环境错误，不能算算子精度失败。

不加 `--skip` 就是先精度、全部通过后再计时。

### 性能测试

```bash
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1,2,3,4,5,6,7 \
  --suite model --case-id pro_m2048 --skip precision --log quiet \
  --warmup 2 --loops 10 \
  --shmem-ip-port tcp://127.0.0.1:29862 \
  --output "$PWD/runs/$(date +%Y%m%d-%H%M%S)-perf"
```

只做计时，不加载 ATK、不跑 CPU 参考。三口径 Event 计时、baseline 对比和指标定义见
“性能口径”与“读输出”两节。

`python -u` 关闭 stdout/stderr 缓冲：输出重定向到文件或经 `tee` 时，不加 `-u` 会按块
延迟刷新，看不到实时进度。入口给子进程另外设了 `PYTHONUNBUFFERED`，但仍建议保留 `-u`。

`--output` 必须是**新目录**，入口拒绝已存在的路径。套件用一个协调进程启动，不要套在
`torchrun` 下重复拉起。

## 选择用例

| 方式 | 说明 |
| --- | --- |
| `--suite priority` / `model` | 同一组 12 条模型用例 |
| `--suite dtype` / `gather` / `all` | 分别选择 24 条类型/保存模式、6 条 FP32 Gather 边界、全部 42 条 |
| `--case-id <id>`（可重复） | 在选中套件内取子集，例如 `--case-id pro_m4096 --case-id glm_m8192` |
| `--list-cases` | 只打印配置，不导入 Torch/ATK、不访问设备 |
| 不使用 `--suite` | 单 case 模式，全部参数由命令行给出（见下） |

suite 内不接受 shape、TopK 路由权重类型/数值、路由、数据分布、保存模式、clipping 和 seed 的覆盖。
`dtype` 包含 BF16、BF16-rounded FP32 和 full-precision FP32 三类 TopK 路由权重，分别组合全部四种
保存模式及 clipping 开关；清单存在不等于设备已通过。

自定义 shape 示例：

```bash
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1,2,3,4,5,6,7 \
  --skip precision \
  --seq 4096 --hidden 7168 --ffn 3072 --topk 8 --expert-per-rank 8 \
  --routed-scaling-factor 2.5 --swiglu-limit 10 \
  --routing-mode balanced --save-gmm1-output bf16 \
  --warmup 2 --loops 10 --shmem-ip-port tcp://127.0.0.1:29862 \
  --output "$PWD/runs/$(date +%Y%m%d-%H%M%S)-custom"
```

约束：`hidden % 256 == 0`、`ffn % 256 == 0`（`--ffn` 就是 SwiGLU 输出宽 F，GMM1 输出是 `2F`）、
`1 ≤ seq ≤ 16384`、`1 ≤ expert-per-rank ≤ 32`（EP8 时总专家 E ≤ 256）、
`topk ≤ EP × expert_per_rank`。`seq` 是**每 rank** token 数，EP 不会把它再切分。
单例参数 `--topk-weight-dtype bf16/fp32` 默认为 BF16；选择 FP32 时默认保留完整有效位，
加 `--topk-weight-values bf16-rounded` 则将同一份 BF16 TopK 路由权重无损转换为 FP32（`topk_weights.float()`）。
返回的 `dtopk_weights` 与输入 `topk_weights` 类型一致，GMM 输入/输出及其余最终浮点输出保持 BF16。
旧命令 `--gmm-n N` 仍按 N=2F 解析，
不能与 `--ffn F` 同时指定；清单和报告统一使用 `ffn`。

## 性能口径

三个口径各自独立采样，窗口前后各 `torch.npu.synchronize()` 一次：

| 口径 | 窗口内容 |
| --- | --- |
| `forward` | 仅前向 |
| `backward` | 窗口外先跑一次前向建图（不计时），窗口内只圈 `y.backward(dy)` |
| `complete` | 一次窗口内前向 + 反向 |

`--warmup N --loops M` 对三个口径和 baseline 共用，`--loops` 默认 20。**不构建、不开启、不读取任何设备 Timer**。

`--baseline on`（默认）额外计时移植过来的 dropless 小算子链（permute、alltoall dispatch、
GMM1、带路由权重的 SwiGLU、GMM2、combine、unpermute），与算子共用同一份路由、权重
和上游梯度；`--save-gmm1-output none` 时它同样重算 GMM1，保证两侧 GEMM 数量一致。
baseline 需要一条可复用的 HCCL 子组（协调组是 Gloo），建不出来或打包路由超过 16 GiB
预算时记 `SKIPPED` 并打印原因，**不影响 PASS/FAIL**。`--baseline off` 只测算子。
性能 baseline 保留移植源链的激活数值边界，和 ATK 使用的 `npu_baseline.py` 是两个不同角色。
它不模拟 MXFP8 保存压缩，只对齐 GEMM 数量；TopK 路由权重按输入类型通信。
使用 FP32 TopK 路由权重时，乘权后的 Hp 转回 BF16 供 GMM；返回的 `dtopk_weights` 与输入 `topk_weights` 类型一致。
精度结论仍由 ATK、量化契约和 counts 的原门禁决定。

四个派生量的定义（公式、RFK 构成、三个适用前提）以
[TEST_SPEC 的性能指标定义](test-spec.md#性能指标定义)为准；跑之前只需要记住：

| 量 | 一句话 |
| --- | --- |
| `floor` | 纸面 FLOPs ÷ 410.4 TF 的理论下界；保存模式三口径为 `6:12:18`，none 重算为 `6:16:22` |
| `MFU` | 实测 GMM 吞吐 ÷ 432 TF 裸峰值；窗口内含通信/激活，是算子级而不是模型级 |
| `overhead` | `measured − floor`，该窗口内的非 GEMM 时间 |

`floor` 用的是 410.4 TF（峰值 95%），`MFU` 用 432 TF 裸峰值，所以 `floor ÷ measured` 会比同口径
`MFU` 高约 5%；需要“离工程上限多远”时用这两个字段现算即可（报告不再单独存这个百分比）。

`432` 沿用源仓注释，**尚未与官方 Ascend950 dense-BF16 规格核对**；引用绝对利用率前先确认。
实测低于 `floor` 说明常量或 RFK 模型有误，不是算子超过硬件。

## 读输出

```
[BENCH]/[PERF] rank=0 mode=bf16 warmup=2 loops=10 (6/12/18 RFK, GMM peak 432 TF)
[PERF] forward   megamoe_bf16=  13.361ms baseline=  19.101ms speedup= 1.43x MFU= 85.7%/60.0%
[PERF] backward  megamoe_bf16=  26.459ms baseline=  34.711ms speedup= 1.31x MFU= 86.6%/66.0%
[PERF] complete  megamoe_bf16=  39.731ms baseline=  53.761ms speedup= 1.35x MFU= 86.5%/63.9%
[PERF] roofline floor=12.06/24.11/36.17ms overhead=+1.30/+2.35/+3.56ms @ 410.4 TF (fwd/bwd/fb)
[PERF] rank=1 forward=13.352ms backward=26.516ms complete=39.674ms speedup=1.43/1.31/1.35x
```

- `speedup` = baseline ÷ megamoe_bf16，**>1 表示融合算子更快**；
- `MFU a%/b%` = megamoe_bf16 / baseline 各自占 432 的比例；
- `overhead` = `measured − floor`，即该窗口内的非 GEMM 时间；三个口径独立采样，
  `complete` 的 overhead **不等于** forward + backward 相加；
- 每个 rank 各一行，用于看慢卡；`[PERF] rank=0` 那段是 rank 0 的明细。

`--suite` 结束时 rank 0 再打一张跨 case 汇总表（每行一个 case，取 rank 0 中位数）：
`M/H/F/tk/E`、三口径时延、`speedup f/b/fb`、`MFU% f/b/fb`、`floor f/b/fb`、`fb_overhead`。
其中 `E = EP × expert_per_rank`，均衡路由下每专家行数 `EP×M×tk/E = M×tk/E_local` 决定了 ready 相关分支，是解释
MFU 差异的关键量。

## 输出目录

```
<output>/
├── suite.log                  # 全部输出（persistent 模式）
├── suite_summary.json         # 每个 case 的状态、配置、报告路径
├── suite_runtime.json         # 各 rank 初始/收尾阶段与共享窗口大小
├── <case_id>/
│   ├── summary.json           # 含每个 rank 的完整报告（含 performance 段）
│   ├── rank_<r>.json
│   ├── atk_rank_<r>/          # 仅 ATK 失败时保留的三份张量
│   └── <case_id>.log
```

性能相关的结构化字段在 `rank_<r>.json` 的 `performance`：`forward_ms/backward_ms/complete_ms`
（`samples/median/mean/min/max`）、`mfu`、`baseline`（`status`、`speedup`、`mfu`、三口径
统计）、`roofline`（逐口径 `rfk/flops/floor_ms/measured_ms/overhead_ms/tflops`）。

同一份报告顶层的 `identity` 记录这次跑的是哪个二进制，便于以后对照：

```json
"identity": {
  "revision": null,                          // 构建时的源码 commit；wheel 未记录时为 null
  "wheel": {"name": "ascend-deepep", "version": "0.1.0.dev5"},
  "libraries": {
    "deep_ep._C":            {"path": "/…/deep_ep/_C…so", "sha256": "…"},
    "megamoe_train._C":      {"path": "/…/megamoe_train/_C…so", "sha256": "…"},
    "megamoe_train_kernel":  {"path": "/…/libdeepep_megamoe_train_bf16.so", "sha256": "…"}
  }
}
```

**`libraries` 的 SHA256 才是主键**；`revision` 只是标签，缺失（`null`）不影响追溯。换内核后对比
两组数据时，先比对这两个哈希，再谈时延。

## 判据与边界

- 精度：`y`、`dx`、`dtopk_weights`、`dw1`、`dw2` 与 exact counts；ATK 是唯一双标杆。
  默认不加 `--skip` 时，**先精度后性能**，精度不过则不进入计时。
- 性能：计时后输出/梯度非有限 → 抛错；counts 不符 → `performance.passed=false`；
  baseline 的 `SKIPPED`、`speedup < 1` 都**不**判失败。
- `--skip precision` 的计时不是精度证据；`--list-cases` 与环境加载成功也不构成验收。

## 已知问题

| 现象 | 说明 / 规避 |
| --- | --- |
| `process_group_setup` 较慢 | 核对 Gloo 的网卡选择；多网卡机器可显式设置 `GLOO_SOCKET_IFNAME=<内网网卡>` 后复测 |
| 性能阶段长时间无输出 | 用 `--log rank0` 确认最后的 `[PERF] stage`；记录源码与 SO 版本、shape、EP、保存模式、`warmup/loops`、设备状态和日志，按相同配置复现 |
| `max_recv_tokens` 与预期不符 | 先看 `[INPUT]` 的 `received_rows`/`empty_experts`，路由不符合预期时后面的时间不可信 |
| `local_mem_size` 不足 | 入口按 shape 自动算窗口；需要时显式传 `--local-mem-size`（2 MiB 对齐） |
| 大 shape 主机内存吃紧 | CPU 输入快照与参考计算在 host 侧；`glm` 的 E/rank=8 权重更大 |

`--log` 默认 `quiet`，只保留 case 行、`[PERF]` 块和汇总表；排查卡顿用 `rank0`（rank 0 的
阶段日志 + `[PERF] stage` 标记），复现细节用 `all`。
