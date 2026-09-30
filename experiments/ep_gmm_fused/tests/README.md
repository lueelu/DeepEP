<!--
Copyright (c) 2026, Lu Lu
Modified by zhu-mingzhe71 2026
-->

# EP_GMM_FUSED 验证工程

从仓库根运行。设备 worker 会排除当前仓库的两个包源码路径，
从当前 Python 环境的同一份 `ascend-deepep` wheel 加载 `deep_ep` 和 `deep_ep_experimental.ep_gmm_fused`。
无需切换工作目录或创建临时软链接。该规则对 accuracy、bench 和 profile 一致生效。
缺少 wheel、原生扩展或已缓存其他来源的包时，在 HCCL 初始化前明确报错，不回退到源码。
报告同时记录 `runtime_package_file`、`package_file`、包/依赖版本、设备、seed 和配置。

```bash
python -m pip install -r requirements-dev.txt -r experiments/ep_gmm_fused/tests/requirements.txt
python -m experiments.ep_gmm_fused.tests.run accuracy --suite smoke --dry-run
python -m experiments.ep_gmm_fused.tests.run accuracy --suite smoke
python -m experiments.ep_gmm_fused.tests.run accuracy --suite model --ep-size 8
python -m experiments.ep_gmm_fused.tests.run accuracy --suite generalization --exclude-category domain_isolation
```

产物默认 `benchmark-runs/ep_gmm_fused/<UTC-run-id>/`：`cases.json` 可完整重放，
`plan.json` 含命令，每 case 保存各 rank 报告和日志；`summary.json/.md` 汇总。
指定请求失败、rank 缺失、空选择、采集缺失都报告失败，不以 skip 充当通过。
`--dry-run` 只生成用例和计划，无需 NPU；设备 smoke/nightly/64 卡/性能/采集需在相应设备执行。

## 精度

本次检视问题的 CPU 回归（不初始化 NPU）：

```bash
python -m pytest tests/runtime/test_ep_gmm_fused_build.py experiments/ep_gmm_fused/tests/test_regressions.py -q
```

UB 的 Python 检查和真实 wheel 切换检查不依赖 Torch；Host C++ 检查需要 C++17 编译器，
内存重叠与 Meta/FakeTensor 检查需要 Linux、CPU 可用的 Torch、C++ 编译器和 Ninja。
缺少依赖时相应测试会 skip，不代表这些检查已通过。

重新编译安装后，运行 `accuracy --suite generalization --category ub_regression`，
覆盖 EP2、自路由、N=2176 的默认配置、29 行合法边界及 30 行拒绝边界；另执行 smoke 精度回归。

生成器和 runner 覆盖真实路由分布检查、同一实例动态调用序列与超时、
零接收/尾块/容量边界，以及合法调度配置扫描。设备执行仍需 Ascend950 验收。
EP4/EP8 模型形状及固定版本的官方 config 来源保存在 [模型配置](MODEL_CASES.json) 中。
两个算子的模型及泛化用例统一选择每卡≤32专家，常规训练规模覆盖单卡展开 M≥262144。

| suite | 数量（默认 seed） | 内容 |
| --- | ---: | --- |
| smoke | 12 | 原小规模 6 个与原 PR 回归 6 个 |
| model | 504 | 12 个模型/EP 组合 × 3 投影 × 7 个 M 档位 × 2 种路由 |
| generalization | 1060 | 505 个基础用例及 555 个定向用例 |
| all | 1576 | 以上三层，无重复 ID |

`nightly` 仅保留旧入口兼容，不属于新的三层统计；包含旧的 >32 专家用例。
模型常规 M 目标为 4096/8192/16384/32768/65536/131072/262144，按 TopK 向上取整；
大负载 524288、压力 1048576 及 R≈2M/4M 热点归入泛化。
生成器为每个模型 token 选择互异专家，遵守配置中的路由组限制，正向展开副本共享原始输入。
manifest 保存模型 revision、目标/实际 M、TopK、路由 seed；可确定性重建 token 到排序行的映射。

模型配置仅保留当前使用的 9 个模型、12 个模型/EP 组合及 36 个投影模板，未启用的 >32 专家部署不入表。
JSON 保留来源 revision、config 链接和原配置校验和；它是生成器的配置注册表，不能直接传给 `--cases`。
所有数据和权重均为 BF16 合成值，无需下载模型权重；数量不代表设备精度、autograd 或完整模型部署已通过。

模型用例假定专家 TP=1、专家按编号连续等分，无 token drop、共享专家或冗余专家，各源 rank 的原始 token 数相同。
令 E 为路由专家总数、e=E/EP、q 为 TopK、T 为每源 rank 的 token 数，展开行数 M=T×q；
实际接收行数 R 由最终路由计数决定，不能用平均 M 代替。`CaseSpec.m` 对正向是源输入行数，
对逆向是返回源 rank 的输出行数，逆向本地 GMM 输入为 R 行。
令 D 为专家输入/输出宽度、F 为专家中间维度，当前保留模型的 D 等于 hidden size：

| 投影 | 算子 | 输入 | 本地权重 | 输出 |
| --- | --- | --- | --- | --- |
| gate/up 合并 | alltoallv_gmm | `[M,D]` | `[e,D,2F]` | `[R,2F]` |
| 单 gate 或 up | alltoallv_gmm | `[M,D]` | `[e,D,F]` | `[R,F]` |
| down | gmm_alltoallv | `[R,F]` | `[e,F,D]` | `[M,D]` |

模型 TopK 路由保证同一 token 的专家互异，并遵守 `n_group/topk_group` 限制；q 个展开副本共享原输入。
M 档位按 `T=ceil(M_target/q)` 向上取整，实际 M=T×q；梯度累积不乘入单次调用 T。
正向实现的每卡专家上限仍为 64，逆向为 32；模型集合统一使用≤32是测试策略，不改变接口限制。
量化/反量化、bias、激活和 TopK 加权归并不在本集合内；gpt-oss 仅覆盖无 bias 的线性投影形状，
`composition` 的两个 GEMM 也不等价于完整 SwiGLU MoE。每卡 BF16 权重字节数为 `2×e×K×N`，
另需按最大 R 为输入、输出、工作区及参考缓存预留资源，不能把算子形状覆盖视为完整模型可部署证明。

泛化用例按以下 `category` 筛选，数量为默认 seed 生成数量，不代表设备通过数量：

| category | 数量 | 覆盖范围 |
| --- | ---: | --- |
| legacy | 505 | 基础边界、路由与输入分布 |
| segment_boundary | 126 | EP2/4/8，双向，专家通信分段边界的 B−1/B/B+1 |
| kn_tail | 54 | K=255/256/257 × N=127/128/129 |
| large_m_boundary | 48 | M=131072/262144/524288/1048576 的 ±1，K=N=32 |
| zero_receive | 12 | 一个空 rank / 仅一个接收 rank |
| capacity | 36 | workspace/UB 的合法边界及首个非法配置 |
| tiling | 144 | 正向普通/导出及逆向的合法配置，固定输入 |
| state_reuse | 24 | 大→小→大、均衡→热点→均衡，checked/queued 重复调用 |
| large_load | 24 | 代表模型的 gate_up/down，M≈524288 |
| stress | 24 | 代表模型的 gate_up/down，M≈1048576 |
| imbalance | 40 | 源 M≈131072/262144，EP4 R≈2M，EP8 R≈2M/4M |
| domain_isolation | 8 | 64 卡 EP4，连续/交错分组，均衡/随机 |
| rank_stagger | 12 | EP2/4/8，固定/轮换延迟 rank |

生成后校验各 rank/专家的实际收发量、零接收 rank、空专家及最大负载占比，
分布不符合目标即失败。检查只覆盖公开输出和路由统计，不代表验证了未导出的内核中间状态。

```bash
# 只生成完整 manifest 和执行计划；不需 NPU
python -m experiments.ep_gmm_fused.tests.run accuracy --suite all --dry-run
# 单个模型、算子、EP 筛选；默认 NPU FP32 单标杆
python -m experiments.ep_gmm_fused.tests.run accuracy --suite model --model Qwen3-30B \
  --ep-size 4 --operator alltoallv_gmm --timeout 1800
# 可重复的任务分片：对同一筛选集合分别运行 shard-index=0..7
python -m experiments.ep_gmm_fused.tests.run accuracy --suite generalization \
  --exclude-category domain_isolation --shard-count 8 --shard-index 0 --timeout 1800
# 分类可重复指定；序列及拒绝契约仅支持 accuracy
python -m experiments.ep_gmm_fused.tests.run accuracy --suite generalization --category state_reuse
```

零接收逆向用例只在零输入 rank 调用拒绝路径，其余 rank 不发起通信；结果明确标记为拒绝契约覆盖。
连续序列在同一 UDMA 域内执行 6 步，复用输出池；queued 模式同一 stream 提交并复制保存每步输出，
之后全量校验，不在步骤之间插入全局 barrier。`progress-rank-*.jsonl` 记录步骤进度。
容量和调度配置当前按 32 个 AIC 推导；native 层仍按实际设备校验，资源不适用会失败，不能计为通过。

`framework/cases.py` 统一用例协议；设备校验使用独立普通 NPU FP32 matmul 生成单一 golden，
CPU 重建通信排列，输入/权重保留实际 BF16 数值，最终 golden 不舍入为 BF16。
`reference/golden.py` 保留独立的 CPU FP32 参考辅助，不作为设备验收后端。
BF16 输出使用 [opbase `81d8b019` 混合容差标准](https://gitcode.com/cann/opbase/blob/81d8b019af1b22a35bcb82a48f28d0f3ab6265ed/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)：
atol=rtol=2^-7，至少 99% 元素通过。
本工程将原文硬上限 `0.1 or 32*ULP` 实现为每元素 `max(0.1,32*ULP_BF16(golden))`，
ULP 按未舍入 golden 的数值区间计算；该合并方式是工程解释，记录在报告中。
统一判据位于 `reference/precision.py`，误差统计使用 FP64 算术，但不做 CPU FP64 GEMM。
导出逐元素、精确标签检查仍作为门禁；任一 rank 失败则整项失败，L2 仅报告。
有限输入用例出现 NaN/Inf 即失败；BF16 次正规数的 ULP 使用最小间距。
参考不调用 DUT 的路由/tiling；确定性重新生成各 rank 输入，因此不假设不均衡输入等长。

nightly 包括 EP1/2/4/8、正向专家数到 64/逆向到 32、上界+1 负例，16/128/256 尾部，
1280/1536 staging 边界与多轮缓冲，K/N 尾部和业务大形状。分布包括 balanced、随机不均衡、
热点专家/rank、local、remote、ring、空专家；随机/热点生成器确保每个目标 rank 至少一行，
避免把当前逆向不支持的零输入误记为正常用例，独立负例覆盖零输入。

输入/权重独立使用 normal/uniform/zero/small/sparse/cancellation/outliers/labels，
pairwise 与固定 seed 随机覆盖 route/device/out 组合。标签和选择矩阵检查精确路由；
composition 使用两组不同权重和 BF16 中间结果。每个正常精度 case 还在相同缓冲和路由上
更换输入/权重再次检查；负例检查专家数、零输入、dtype、route dtype 和错误行数。

`cases.json` 可加入显式 `tiling`；精度和计时分别通过 `accuracy`、`bench` 执行。`load_tiling_config`
拒绝没有 `accuracy_checked=true` 的推荐配置。当前不自动替用户选择新 tiling。
单算子的默认 golden 使用 `reference/streaming.py`：输入分块生成后映射到本地临时文件并复用，
仅计算当前 rank 需要的行；不物化完整参考输出，误差比例按整个输出汇总。
`--reference-cache /local/ssd/scratch` 指定已有的本地临时目录；每个 worker 使用独立目录，正常退出清理。
强制 SIGKILL 后可能遗留 `ep-gmm-fused-reference-*` 临时目录；确认进程退出后可清理。
需要为各源输入和权重预留磁盘空间；mmap 页缓存仍占用主机内存，不代表没有内存开销。
`--blas-threads` 默认每 worker 1 线程，仅限制 CPU 数值库线程，不控制 NPU matmul 并行度。
`--golden-block-rows` 默认 1024，控制参考块大小；参考为全量比较，不抽样输出。

`--golden-backend npu-fp32` 是唯一的设备参考后端，关闭 HF32 后全量比较；
不支持 `torch_npu.npu.matmul.allow_hf32` 开关的框架版本报错，不自动降级。报告使用 `fp32` 字段，
标记 `golden_dtype=float32`、`golden_rounding=none`，并记录标准 revision 和阈值。
composition 的两次 matmul 也使用 NPU FP32，只在两次公开调用之间保留真实 BF16 舍入。
参考沿 M 分块、保留完整 K，不做 split-K；检查全部有效输出元素，按全输出汇总匹配比例。
`reference_seconds` 包括 D2H、golden 计算和比较；`generation_seconds` 记录缓存填充时间，
二者部分重叠，不能直接相加。`dut_call_and_sync_seconds` 记录 DUT 调用与同步，
queued 序列延后检查时不能据此推算提交时延；端到端耗时应测量整个 runner。
这些用例不覆盖标准全文的全部组合与特殊值要求，也不承诺未实测的 NPU 参考耗时。
`--suite smoke` 共 12 个 case：原有 6 个小规模 case 保持不变，加上原 PR 的 6 个业务回归。
这 6 个回归统一为 M=131072、EP8、每卡 32 专家、预分配输出，仅归入 smoke，
不提供独立 business suite，也不加入 nightly。完整 smoke 需要 8 卡；
需充足主机内存并按机器算力设置 `--timeout`，可通过 `--ep-size 1/2` 筛选原有小规模用例。
case ID 根据当前用例的完整配置生成；默认 seed=20260920 时，这 6 个回归的标识如下：

| 来源：原 PR 11602 | K→N | export | case ID |
| --- | --- | --- | --- |
| alltoallv_gmm | 2048→512 | false | `64d79f3ff1f3e5e6` |
| alltoallv_gmm | 2048→512 | true | `e155852c92031034` |
| alltoallv_gmm | 2048→1024 | false | `c58c1210c1097791` |
| alltoallv_gmm | 2048→1024 | true | `158f953cce43d23f` |
| gmm_alltoallv | 512→2048 | false | `84dbe9c348d7b2d6` |
| gmm_alltoallv | 1024→2048 | false | `15563514d1bcadd6` |

## 性能

```bash
python -m experiments.ep_gmm_fused.tests.run bench --suite smoke \
  --warmup 10 --iterations 50 --batches 20

python -m experiments.ep_gmm_fused.tests.run bench --ep-size 2 --operator alltoallv_gmm \
  --warmup 10 --iterations 50 --batches 20
```

`bench` 自动使用普通 `msprof` 的 `--task-time=on` 采集并解析 `op_summary_*.csv`。
每个 rank 独立启动采集进程，准备输入和路由后直接预热、执行测试，不生成 golden、不执行精度比较。
启动前探测 `msprof --help/--version`；缺少工具、所需选项或目标 kernel 记录时明确失败，不回退到 Event 计时。
精度测试单独运行 `accuracy --suite smoke`；性能报告的 `passed` 仅表示性能执行成功，
各 rank 报告记录 `accuracy_checked=false`、`accuracy.status=not_run`，不表示精度通过。
输入准备仍使用可复现的数据生成和临时缓存，位于计时区间之外。
流水采集 `profile` 保留原有精度预检及采集跳过次数。

采样按目标 kernel 的时间顺序，严格要求每 rank 有 `warmup + iterations × batches` 条记录；
剔除最前面的 `warmup` 条后，每次调用保留一个 `Task Duration(us)` 样本。
同一次调用先取各 rank 的最大耗时，再计算 p50/p90/p99；不再统计 batch 平均值，也不混入 Event/Host 耗时。
`composition` 为每次调用的两个目标 kernel 时长之和，不包含两者之间的 host 间隙。
每批开始前仍执行组内 barrier，结束时同步设备；这些操作不会混入目标 kernel 的 CSV 统计。
任务时长只描述采集条件下的设备执行时间，不等于端到端调用耗时或流水完成性验证。

每个 case 保留 `profile-rank-*/PROF_*` 原始采集与导出文件、`profile-command-*.json` 命令、
`capture-*.json` 解析结果和逐次样本；`profiler-node-*.json` 保存工具版本/help。
`summary.json` 使用 `msprof` 字段，MD 显示 msprof p50 和 `Accuracy: not run`。
全部预期 rank 的执行和采集均成功才标记通过，缺少 CSV 或采集次数不符也会失败。
采集及自动导出依据[官方 msprof 说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0011.html)。

可在分别安装两个 EP_GMM_FUSED 构建的环境中，使用同一套用例建立和比较基线：

```bash
python -m experiments.ep_gmm_fused.tests.run bench --cases saved/cases.json \
  --run-id baseline
python -m experiments.ep_gmm_fused.tests.run bench --cases saved/cases.json \
  --baseline benchmark-runs/ep_gmm_fused/baseline/summary.json
```

比较要求 case、rank 数、设备/Torch/torch_npu、msprof 版本、采样配置及统计口径一致。
原来的 NPU Event 基线不能与 msprof 结果直接比较，需重新采集基线。两个环境的导入路径和版本
均保留；设备 runner 始终使用已安装的 `deep_ep_experimental.ep_gmm_fused`。
模型业务中的其他 TP 通信不包含在算子延迟内。

## PipeTimeline / InstrTimeline / op_summary

根据这两个算子已经确认的 Ascend950 能力，直接使用流水采集。分别运行两个采集任务：

```bash
python -m experiments.ep_gmm_fused.tests.run profile --ep-size 2 --operator alltoallv_gmm \
  --metric PipeTimeline --warmup 3 --iterations 2
python -m experiments.ep_gmm_fused.tests.run profile --ep-size 2 --operator gmm_alltoallv \
  --metric InstrTimeline --pipes 'cube|mte2|mte3' --warmup 3 --iterations 2
python -m experiments.ep_gmm_fused.tests.run profile --ep-size 2 --operator alltoallv_gmm \
  --metric op_summary --warmup 3 --iterations 2
```

探测 `msopprof --help/--version`，回退 `msprof op`；按 help 适配 `InstrTimeline` /
`instrTimeLine`。缺少请求能力则显式失败，不因文档的泛化限制关闭该算子的采集。
每 rank 独立 profiler 子进程，跳过精度检查的一次调用和 warmup，使用内核名过滤，
`--kill=off` 正常退出。超时由外层终止整个 torchrun/采集进程组。
不添加 PrintTimeStamp、descid 或内核插桩。源码映射可用 `DEEPEP_LINEINFO=ON` 构建。

每 rank 保存完整命令/help/version、`trace.json`、`visualize_data.bin` 及哈希；解析要求
匹配目标内核和非空事件，报告观察到的 track/pipe，不能据此推断所有核/指令都被完整采样。
二进制用 MindStudio Insight 查看。`op_summary` 独立使用常规 msprof，duration imbalance
只报告，不当作 launch skew，也不否决不均衡路由。

参数参考：[官方 msOpProf 指南](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)。
部署版本的 UDMA replay、进程退出和采集文件结构仍需实机验收。

## 64 卡多个 EP4 域与重放

每节点执行（共享输出文件系统，node-rank 各不相同）：

```bash
python -m experiments.ep_gmm_fused.tests.run accuracy --suite generalization --category domain_isolation \
  --nproc-per-node 8 --nnodes 8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --run-id ep4-64card
```

runner 按相同顺序创建 16 个 EP4 HCCL 组，每进程只初始化所属域；报告按全局 rank 标识。
已存在 run-id 不应复用，否则旧结果可能混入；每次执行使用新 id。

```bash
python -m experiments.ep_gmm_fused.tests.run accuracy --cases previous-run/cases.json --case-id CASE_ID
python -m experiments.ep_gmm_fused.tests.run report --output previous-run
```
