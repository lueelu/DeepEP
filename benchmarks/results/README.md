<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 性能结果记录（schema_version=1）

本页定义通用 benchmark 的结果协议，不适用于 HT/LL ST 的 summary/profiler 格式。[template.csv](template.csv) 仅含表头，单元测试的合成样本不纳入结果仓库。当前通用 CLI 只生成规划结果；`measured` 字段描述已实现的序列化协议，不表示 native 测量入口已接入。

## 输出状态与文件

| 状态 | 输出 | 允许内容 |
| --- | --- | --- |
| `planned` | stdout；指定新目录时写 `config.json`、`plan.json` | 配置、能力、计时定义与逻辑字节数；正确性 `not_run` |
| `measured` | rank 0 stdout、`config.json`、`result.json`、`summary.csv` | 完整原始样本、聚合统计及前后正确性通过 |
| `failed` | stdout 的 schema_version/status/error_type，详细错误在 stderr | 不写成功结果，不以 0 代替时延；参数解析错误直接返回非零 |

只有成功退出原生会话的测量才能发布 `measured`。已有目录不覆盖；写入失败时残留目录不能视为完整记录。未测量时不生成 `statistics`、`rank_samples_us`、`rank_max_samples_us` 或性能 CSV。

## JSON 字段契约

| 字段 | 类型 / 定义 |
| --- | --- |
| `schema_version` | 整数 1；不兼容语义变化需升级版本 |
| `status` | planned / measured / failed |
| `config` | 全量规范化负载配置，字段与约束见[配置说明](../README.md#配置) |
| `config_fingerprint` | 上述配置排序 JSON 的 SHA-256 摘要，64 个十六进制字符（256 bit） |
| `required_capabilities` | 排序的能力字符串数组 |
| `timing` | 时钟、完成边界、包括/排除项、聚合/百分位、缓存策略与操作定义 |
| `traffic` | definition/includes/excludes；每个源 rank 的行数、dispatch/combine 字节数，以及所选操作全 rank effective_bytes |
| `correctness_result` | planned 为 not_run；measured 为 passed_pre_and_post |
| `revision`, `measured_at` | 仅 measured：完整小写 40 位 Git SHA、带时区 ISO 8601 UTC 时间 |
| `metadata` | 仅 measured：backend/device/software/topology/variant；显式公开标识，不自动收集机器信息 |
| `rank_samples_us` | 仅 measured：world_size × iterations 的正有限数，rank-major、逐轮对齐 |
| `rank_max_samples_us` | 仅 measured：每轮跨 rank MAX 的原始聚合数组 |
| `statistics` | 仅 measured：latency_mean/min/max/p50/p95_us、bandwidth_gbps、bandwidth_statistic |

延迟为微秒，带宽为十进制 GB/s。带宽字段 `bandwidth_statistic=effective_bytes_over_mean_rank_max_latency`。逻辑字节数包含本地路由，不代表物理链路流量；FP8 scale、索引、权重及 padding 不计入。

## CSV 与复现

CSV 一次实验一行，保留旧字段并追加 schema、状态、后端、双方向 dtype、布局、缓存模式、alignment、权重开关、路由、mean/min/max、带宽统计口径和完整配置指纹。字段顺序由测试与模板同时锁定。不适用的 nodes/build_info_path/policy_path/num_aiv 等留空；基线提交/延迟/speedup 未测量时留空。

raw_results_path 与 environment_path 都指向随目录归档的 `result.json`；配置位于 `config.json`。协议模板中的 launcher 和输出目录是字段示意，不是可直接执行的命令。发布实测记录时附上实际公开拓扑对应的命令，并保持所有 rank 的配置与源码 SHA 一致。

同配置的 record_id 相同，区分重复实验还需 measured_at、revision、metadata；不要只用 record_id 当全局唯一键。公开前检查所有字段，仅保留可共享信息，不上传完整进程环境、访问凭据或本地机器路径。

[测量口径](../../docs/performance.md) · [配置与工具](../README.md)
