<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# EP benchmark 工具

## Clos 单次初始化摸测

普通 V2 BF16 非展开路径中，EP64 dispatch 的8～64核均保留8个payload发送核；combine的8～30核继承8核计划，32～64核继承32核计划。EP128的8～14核使用rank/P16/W1/Q1/v32，16～64核保持rank/P8/W1/Q1/v16。额外核数不再反复改变发送分工。

所有自动UDMA launch在通信核数大于32时，统一继承相同shape、算子、接口及dtype的完整32核计划，包括P/W/Q、发送顺序、visit及copy/weight偏好；实际启动核数仍使用用户设置值。EP32/G8带权combine的64核不再丢失32核的visit拟合；EP32/G16保持P4/W1/Q1/v32，EP64各hidden对应的既有visit不变。独立reduce核数不参与发送计划选择。

自动策略表直接保存 rank 计划，不再先选 group/RR 再转换。仍保留以下明确的适用条件，不能把“继承32核”理解成所有hidden/API/拓扑都使用同一个计划：

| 条件 | 仍保留的行为 |
| --- | --- |
| EP32/G8带权combine，32核及以上，BF16普通V2路径 | P4/W1/Q1不变；2048<H<4096用v4，4096≤H≤6144用v8，H>6144用v32。EP32/G16不走该拟合，保持v32。 |
| EP64/G16带权combine，32核及以上，BF16普通V2路径 | P4/W1/Q1不变；4096≤H≤6144用v8，其他宽度用v32。本次保留此宽度边界。 |
| EP4/8/16的小hidden | 原profile在BF16 H1024进入，H2048及以下使用small档；更小hidden使用基础计划。不同档可能改变P/W/Q及visit。 |
| EP4/G4带权combine，BF16 H1024～2048 | 8～14核为P1/W1/Q2；16～30核仍使用P1/W2/Q1。两者visit均为32。 |
| 拓扑未识别或不在profile适用G范围、非BF16、V1、按expanded选择的路径 | 不强行套用普通BF16 profile，保留相应基础计划；32核以上仍完整继承该路径自己的32核计划。 |

权重SIMT/UDMA偏好还保留行宽阈值（EP4/8为7168B、EP16/32为8192B、EP64为12288B，且要求有辅助AIV）；EP8/16宽行继承会固定其H6144偏好。默认compact weights可用时实际仍连续UDMA发送，不能仅凭偏好字段判断通信实现。低于8核时仍可能为满足容量回退group_round_robin；EP128在8～14核与16核及以上的P16/v32、P8/v16分档是明确保留的计划。combine默认rows的8192B/32768B边界及不满足合并条件时rows1，属于布局/容量选择，不是发送策略回退。

普通V2 BF16非展开、已识别为连续单机G4的EP4，8～30核dispatch在H7168及更大hidden默认延续H6144的rank/P1/W1/Q2/v32：3个payload发送核，其余核处理metadata和本卡复制。fresh/cached共用该计划；32～64核以及combine计划不变。实际行宽和容量检查不变，宽hidden的设备性能仍需验证。

普通V2 BF16非展开、EP8/16、8～64核的H7168及更大hidden也默认继承H6144的完整发送计划，包括P/W/Q、visit及权重偏好；8～30核继承8核档、32～64核继承32核档，原拓扑适用条件保留。无需设置 `DEEPEP_CLOS_EXTEND_LARGE_ROW_POLICY`。该兼容选项设为`1`时仍可额外延伸EP4的其他路径，设为`0`不会撤销已默认生效的继承。实际tensor行宽、UB和通信容量检查不变。

普通非展开UDMA peer combine默认按实际payload行字节数合并：`<=8192`用rows4，`8192<bytes<32768`用rows2，`>=32768`用rows1。BF16对应H<=4096、4096<H<16384、H>=16384；H7168现在默认rows2，仍可通过 `DEEPEP_CLOS_COMBINE_PAYLOAD_MAX_ROWS` 显式对照。特殊路径继续保留原有rows1安全回退。H7168的rows2已有设备对照收益，更大hidden及新的EP8/16继承计划仍需设备验证。

visit 可扫描 `4/8/16/32/64/128`（`--visits`）；自动策略默认值不变。visit 是一个 peer 每轮的 WQE 预算，64/128 会拆为多个最多32条的提交，不扩大 WQE 暂存区，也不改变 SQ 信用保护。带权重等多字段路径按 WQE 计数，不等于同等数量的 token。新增取值需同步更新 Python 和 Host 库。

普通自动 Buffer 会按本 EP/拓扑的所有可用自动计划预留 QP 并集，涵盖动态2～64核、V1/V2、dtype/展开兼容路径和各行宽档位。每对 peer 保留控制 QP0，并合并双向映射；无需另设 `DEEPEP_CLOS_QP_PLANS`。低于8核的容量回退可能让 EP64/128 的预留集合仍含 RR，但不改变8～64核的 rank launch。手动摸测则按实际候选集合预留。

按 G=min(EP,16) 计算，每 rank 的 peer-QP 位数相较旧的全策略 `TuningPeerMask` 并集：EP4为18～22→16，EP8为66～81→54，EP16为206～265→118，EP32为500～616→121，EP64为1016～1256→313～343，EP128为1851～2460→631～691。此处统计连接映射位，不是物理SQ数量或初始化耗时实测；不同rank可能不同。

`clos/bench_fast_sweep.py` 在候选策略过滤完成后，自动将本次所有算子（包括准备 handle 的 fresh dispatch）的 QP 映射并集传给 SHMEM，每个进程只初始化一个 Buffer。`--plan-only` 和输出 `plan.json` 中的 `qp_plans` 记录建链范围；`BUFFER_READY` 记录初始化耗时。visit 不改变 QP 映射，固定 P/W/Q 的 visit 扫描不会额外预建其他策略的连接。普通模式**不做精度验证**；下面的精简 rank 模式会检查入围配置。

### EP4/8/16/32 精简 rank 摸测

仅重测当前自动 combine（不摸测）时，复用同一入口：

```bash
torchrun --nproc_per_node=16 benchmarks/clos/bench_fast_sweep.py \
  --combine-defaults-only --ep-size 16 --skip-precision --single-pass \
  --no-huge-pages --reduce-cores 48 --warmup 5 --final-samples 30 \
  --output combine_defaults_ep16
```

默认 T8192、H2816/4096/6144、通信8/32核、K8/E256/BF16/expand0，两种权重路径，共12组/EP；也可用重复的 `--case TOKENS,HIDDEN,TOPK,AIV` 缩小范围。每rank仅构造一次 Buffer，所有shape和核数复用；准备 handle 的 dispatch 不计入结果。P/W/Q、visit、payload rows 和 WQE fast 均采用当前自动值，不扫描或覆盖。进程环境须清除其他 `DEEPEP_CLOS_*` 覆盖项；脚本按实际计划并集预留 QP，并保存实际 launch/rows。

所有 CLOS benchmark 在 Buffer 初始化时固定传入 `combine_notify=True`，预热、精度和正式采样使用同一设置，不再读取 `DEEPEP_CLOS_PROFILE_COMBINE_NOTIFY`，也不增加命令行选项。该模式的解析器要求每次调用的顺序为 `采样前barrier → notify → 通信 → reduce`，前置barrier不计入notify。只输出notify/通信的逐rank原始样本、均值/P50/P95和通信带宽，不采集API耗时；reduce仍执行。`--skip-precision` 明确将报告标为 `precision=skipped`，不能据此宣称精度通过。`--single-pass` 不跑反序、不追加采样。带宽分子和逐样本最慢rank口径沿用下面的说明，notify不计入带宽分母。

EP4/8在单机分别用4/8进程，EP32用两机各16进程；各EP独立会话，不能跨EP复用同一个Buffer。先用 `--plan-only` 核对用例。失败时保留构造/设备第一条错误和已完成结果，不自动重启或删掉慢样本。若仍出现初始化同步错误，notify测量尚未开始，不能把该轮当作性能结果。

```bash
torchrun --nproc_per_node=8 benchmarks/clos/bench_fast_sweep.py \
  --small-rank-sweep --ep-size 8 --no-huge-pages --reduce-cores 48 \
  --warmup 3 --samples 8 --final-samples 30 --top 3 --output rank_ep8
```

EP4/16 单机运行时，将进程数和 `--ep-size` 同时改为4/16。EP32 可用两台各16卡的机器，在两台分别执行（`MASTER_ADDR` 为首台可达地址，`NODE_RANK` 分别为0/1）：

```bash
torchrun --nnodes=2 --nproc_per_node=16 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29500 \
  benchmarks/clos/bench_fast_sweep.py \
  --small-rank-sweep --ep-size 32 --no-huge-pages --reduce-cores 48 \
  --warmup 3 --samples 8 --final-samples 30 --top 3 --output rank_ep32
```

两台参数必须一致，并开放 torchrun、控制面（默认29610）及 SHMEM（默认29620）端口。各节点保存自己的逐rank结果，rank0保存总表；回收结果时合并两台的 `rank*/` 目录。每个 EP 独立启动任务，不能用一个32进程任务同时测多个 EP。使用空的新输出目录，清除所有 `DEEPEP_CLOS_*` 覆盖变量，重编安装当前 Host/kernel 和锁定的 SHMEM。模式会核验实际大页关闭、下发的 P/W/Q/visit、dispatch 连续 metadata 与 combine 连续权重/本卡直读；不改变库默认值。`--plan-only` 可在无设备环境查看候选。

每个 EP、每个 rank 只初始化一次 Buffer，涵盖 T4096/8192、H2816/4096/6144、通信8/32核、K8/E256/BF16/expand0，以及 fresh/cached/combine 各自的无权和带权路径。先在 T8192、H2816/6144 下比较三组 P/W/Q 与 visit16/32，再给选中的 P/W/Q 扫 visit4/8/64/128。其余 shape 只验证代表配置与实际 auto；H4096 继承 H6144 候选，检查泛化。

三组分别为单worker、双QP和双worker，P取满足通信核数及32个QP槽位限制的最小合法值。32核不限制为8个发送核。以下括号为实际发送核数，另加当前auto对照；EP4/8候选保持不变。

| EP | 通信核数 | 单worker P/W/Q | 双QP P/W/Q | 双worker P/W/Q |
|---|---|---|---|---|
| 4 | 8/32 | 1/1/1（3） | 1/1/2（3） | 1/2/1（6） |
| 8 | 8 | 1/1/1（7） | 1/1/2（7） | 2/2/1（8） |
| 8 | 32 | 1/1/1（7） | 1/1/2（7） | 1/2/1（14） |
| 16 | 8 | 2/1/1（8） | 2/1/2（8） | 4/2/1（8） |
| 16 | 32 | 1/1/1（15） | 1/1/2（15） | 1/2/1（30） |
| 32 | 8 | 4/1/1（8） | 4/1/2（8） | 8/2/1（8） |
| 32 | 32 | 1/1/1（31） | 2/1/2（16） | 2/2/1（32） |

初筛不逐候选检查精度；入围配置使用独立 CPU 参考进行三次连续复用、零容差检查，再正反顺序各采样30次。全部配置固定 rank，仅变 P/W/Q/visit，保留生产默认优化。rank/payload 摸测不再运行独立完整 API 采样，初筛按通信均值排序，最终选取较慢一轮的通信均值最低者；“最优”仅指本次候选的通信表现，不代表端到端最优。`summary.csv` 给出通信带宽和设备各阶段 mean/P50/P95，不再输出 API 字段；`rank*/` 保留原始样本、profiler 和实际 launch。带宽分子仅为去重后的 BF16 payload（含本卡、不含权重），各次采样取最慢 rank；不将不同阶段的最慢值相加推断 API。每次调用前的全卡 barrier 和调用后的完成等待均保留；combine profiler 原有的额外 notify 诊断保持不变，单独统计，不计入通信 kernel。失败会保留已完成数据并退出，不能把未完成的覆盖记作通过。

其他固定策略脚本可在创建 Buffer 前设置，例如 EP64、rank/P8/W1/Q1：

```bash
export DEEPEP_CLOS_QP_PLANS='rank:8:1:1'
# 同一 Buffer 如还要测试 P4，使用 'rank:8:1:1;rank:4:1:1'
```

格式为 `order:P:W:Q`，多组用分号分隔；order 支持 `rank/group/group_staggered/group_round_robin`，沿用 Buffer 的固定 G。必须包含 dispatch、cached dispatch、combine 和准备阶段实际会用到的 QP。所有 rank 必须配置一致；初始化后不可扩大范围，超出已建链范围的调用会在提交 kernel 前报错。多 Buffer 复用也会校验配置一致。未设置时，自动模式从当前策略选择器生成各合法核数、行宽区间、V1/展开/非BF16回退路径的QP并集；无需为动态通信核数重新初始化。摸测 session 仍预留完整候选范围，手动模式保留原规则，显式设置本变量优先。该选项仅支持全 Clos shared-Jetty UDMA peer 路径，不修改发送策略、QP 编号、visit、权重方式或大页设置。须使用当前 lock 指定的 SHMEM，重编 Host，并在设备上验证初始化和算子精度；连接数量减少不等同于已测得初始化加速。

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

共享 dispatch/combine kernel 统一使用 `direct_dispatch_*` / `direct_combine_*` 名称；profiling 解析器仍可读取归档中的 `v1_dispatch_*` / `v1_combine_*` 名称。公共 V1/V2 API 名称不变。
