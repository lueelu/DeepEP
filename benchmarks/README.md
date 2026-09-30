<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# EP benchmark 工具

通用入口提供配置检查、确定性负载规划和 JSON/CSV 结果协议。当前支持无设备 dry-run；native 测量适配器未接入，非 dry-run 返回非零，不以模拟时间代替通信测量。

HT 和 LL 有独立的设备正确性与 kernel 测量入口，运行方式见[测试说明](../tests/README.md)，统计口径见[性能说明](../docs/performance.md)。

## 无设备运行

从仓库根目录使用 Python 3.11+，无需 Torch、NumPy 或 NPU：

```bash
python -m benchmarks.bench_ep --help
python -m benchmarks.bench_ep --list-scenarios
python -m benchmarks.bench_ep --dry-run --preset smoke
python -m benchmarks.bench_ep --dry-run --preset upstream-bf16
python -m benchmarks.bench_ep --dry-run --preset upstream-fp8
python -m benchmarks.bench_ep --dry-run --operation roundtrip \
  --tokens-per-rank 16,0 --route duplicate --output-dir benchmark-runs/plan-001
```

`--config <file.json>` 与 `--preset` 二选一，显式 CLI 字段覆盖配置。指定输出目录时必须使用新路径。dry-run 输出 `status=planned`，保存配置和计划，不生成时延、带宽或性能 CSV。`upstream-*` 只是固定上游典型规模的规划样本，不是设备支持或性能基线。

## 配置

| 字段 | 含义与取值 |
| --- | --- |
| `api` | `legacy` 或 `elastic` |
| `operation` | `dispatch`、`combine` 或 `roundtrip` |
| `dtype` | dispatch 的 `bf16` 或 `fp8`（E4M3FN）；combine 输入/输出为 BF16 |
| `layout` | `ordinary` 或 `expanded` |
| `dispatch_mode` | `fresh` 或 `cached` |
| `world_size`、`tokens_per_rank` | rank 数及每个 rank 的实际 token 数；允许部分 rank 为 0，全局至少一个 token |
| `hidden`、`topk`、`experts` | H、K、E；E 可被 EP 整除，K≤E |
| `alignment` | 1 或 128 |
| `with_weights` | 独立 FP32 值通道；CLI 使用 `--with-weights/--no-with-weights` |
| `route` | `balanced`、`skew`、`duplicate` 或 `masked` |
| `seed` | 确定性输入和路由的种子 |
| `warmup`、`iterations` | 通用计时协议的预热与采样次数 |

`--tokens-per-rank` 逐一列出全部 rank 的 token 数。legacy 规划仅接受普通 BF16、alignment=1；FP8 的 H 须被 128 整除；elastic combine/roundtrip 的 H 须被 256 整除。**规划器接受的配置不等于当前 native 算子支持的配置。** 设备限制见 [Elastic 接口](../docs/elastic.md)。

JSON 不超过 64 KiB，拒绝重复/未知键、错误类型和非法范围。规划器上限为 EP≤1024、H/E≤65536、K≤256、全局 token≤1048576、路由槽位≤8388608、预热/采样次数≤10000、保留样本≤1000000。这些用于限制规划计算量。

## 确定性负载

对源 rank r、token t、列 c 和路由槽位 s：

```text
x(r,t,c) = ((seed + 13*r + 7*t + c) % 127 - 63) / 16
w(r,t,s) = ((seed + 19*r + 7*t + 3*s) % 127) / 128
e(r,t,s) = (seed + 17*r + 7*t + s) % E
```

hidden 值可被 BF16 精确表示；FP8 先生成 BF16，再按固定参考的每 128 列 scale 编码。`with_weights=false` 不改变 hidden 字节数。

`balanced` 使用轮转 expert 公式，不保证每个配置严格均匀；`skew` 将目标压到 rank 0；`duplicate` 在 K>1 时复制槽位 0 到槽位 1；`masked` 按固定公式将部分槽位置 -1。有效行和逻辑流量由配置与路由计算，不读取内部 handle 来生成预期答案。

## 输出与退出

计划文件包含规范化配置、配置指纹、所需能力、计时定义和逻辑字节数。结果协议见 [results/README.md](results/README.md)，与 LL ST 的输出格式分开。

退出码 0 表示规划成功；当前请求 native 测量返回 2，错误为缺少 native benchmark adapter。参数、配置或输出路径错误也返回非零。结构化失败结果只记录错误类型，详细错误位于 stderr。

`_config.py` 实现配置及字节计算，`_utils.py` 实现通用计时、统计和序列化，`_backend.py` 定义内部适配协议。可运行 `python -m pytest tests/test_benchmark.py -q` 检查这些协议；CPU 合成样本不作为性能数据发布。
