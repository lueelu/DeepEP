# DSV41：48 个完整 MegaMoE 性能 case

本入口测试当前 `fp8_fp4_mega_moe` 完整流程：Dispatch、GMM1、SwiGLU、GMM2、Combine，
默认包含每卡 1 个共享专家。输入默认 MXFP8、权重 MXFP4；使用现有 benchmark 的 kernel Duration
计时方法，关闭设备 timer，不使用 NPUGraph。权重生成、转换、输入准备及 warmup 均不计入 kernel 耗时。

| EP | 节点 × 每节点卡数 | 物理路由专家槽 | 每卡专家槽 | 原始 BS | 实际每卡 token 数 | case 数 |
|---:|:---|---:|---:|:---|:---|---:|
| 16 | 1 × 16 | 400 | 25 | 1～24 | 6、12、…、144 | 24 |
| 32 | 2 × 16 | 416 | 13 | 1～24 | 6、12、…、144 | 24 |

两组均为 H=5120、SwiGLU 后 I=2304、top-k=6，GMM1 逻辑投影宽度为 4608。
保留参考脚本的 `384 + EP` 物理专家布局，所有物理槽均参与路由，不另做冗余专家映射。
默认 balanced 路由；`--routing random` 使用每个 token 不重复的合法 top-k。
参考压缩包的 INT8、Dispatch+GMM1、NPUGraph 测量范围不同，不能直接用其耗时计算本入口的加速比。

## 先编译安装

在两台机器的同一版本源码上执行，使用匹配的 Torch、torch_npu、CANN 和 SHMEM 环境：

```bash
source /path/to/cann/set_env.sh
DEEPEP_MEGAMOE_TIMER=OFF bash experiments/megamoe/scripts/build.sh --rdma-backend HNS_1825
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

网卡为 XSCALE 时替换后端；更换已构建的 SHMEM 后端时按 build 入口要求加 `--rebuild-shmem`。
EP16 可复用同一 RoCE wheel，也可单独构建 `--rdma-backend NONE`。
`PYTHON=/path/to/python` 可为 build 和下面的启动脚本指定解释器。

EP32 原有实际 BS 上限为 32，本次将内核的每 token Combine 状态容量及 Host/API 限制扩到 144。
**必须重建并安装本次 wheel**；启动前检查会拒绝能力不足的旧 wheel。
这项扩展已提供 CPU 协议和容量检查，CANN 编译、BS=144 精度及双机运行仍需在目标 NPU 上验证。

## 查看矩阵

```bash
bash experiments/megamoe/scripts/run_dsv41_perf.sh plan
```

只打印 48 个 case 的参数，不导入 Torch、不访问设备、不写文件。
默认每个 case warmup=20、正式迭代=100。汇总沿用现有 benchmark，取正式迭代的后半段（向上取整）。

## EP16：一台机器运行 24 个 case

```bash
bash experiments/megamoe/scripts/run_dsv41_perf.sh run \
  --ep-size 16 --output-dir "$PWD/benchmark-runs/dsv41-ep16-001"
```

## EP32：两台机器同时运行

两台机器使用相同的矩阵、input、routing、shared-experts、warmup、iters、seed、perf-method 和端口。
各自输出目录可以位于本地盘，也支持共享盘上的同一目录。每次复测使用新目录。
默认端口为 master=29654、SHMEM=26733，可通过 `--master-port` 和 `--shmem-port` 修改。

节点 0：

```bash
bash experiments/megamoe/scripts/run_dsv41_perf.sh run \
  --ep-size 32 --node-rank 0 --master-addr <NODE0_IP> \
  --output-dir "$PWD/benchmark-runs/dsv41-ep32-001"
```

节点 1：

```bash
bash experiments/megamoe/scripts/run_dsv41_perf.sh run \
  --ep-size 32 --node-rank 1 --master-addr <NODE0_IP> \
  --output-dir "$PWD/benchmark-runs/dsv41-ep32-001"
```

一个 torchrun 作业顺序运行同一 EP 的全部 24 个 BS，权重只准备一次，各 rank 在 case 边界集体同步。
任一 case 失败即停止该作业，已完成结果保留，未产生报告的 case 标记为 `missing`，不填零耗时。
每台机器保存 `node_N.log`、`status_node_N.json`、`launcher_node_N.json` 和 `summary_node_N/summary.csv`。
原始 profiling 文件位于 `measurements/rank_R/bs_B/`；每个 rank 的汇总包含全 EP 的样本。

先做边界性能试跑可在两边同时添加 `--bs-values 6,144 --warmup 2 --iters 4`。
`--bs-values` 填写乘 6 后的实际值；正式完整矩阵不传此参数。
如需不含共享专家，双方均加 `--shared-experts 0`；结果表会记录该选择。
默认 `--perf-method legacy` 沿用现有整批 Exp 对齐；可显式选择 `shmem`，但比较时必须保持方法一致。

## 合并为 48 行结果

EP16/EP32 两个输出目录可在测试后复制到同一台机器，再运行（汇总无需 NPU）：

```bash
bash experiments/megamoe/scripts/run_dsv41_perf.sh summarize \
  benchmark-runs/dsv41-ep16-001 benchmark-runs/dsv41-ep32-001 \
  --output-dir benchmark-runs/dsv41-summary-001
```

输出 `summary.csv` 和 `summary.json`，包含原始 BS、实际 BS、全局 token 数、专家布局、共享专家数、
输入类型、路由和计时方法。主指标 `max_rank_mean_us` 是各 rank 平均 kernel Duration 的最大值；
`per_round_max_mean_us` 是每轮先取最大值再平均，两者不可混用。
`per_round_min_mean_us` 对应现有 benchmark 打印的 `MegaMoe E2E` 指标。
仅接收全 EP、完整轮数、有限正数样本；缺失或损坏的报告会标记状态并返回非零退出码。
构建信息保存在 `measurements/rank_R/manifest.json`。这些是性能结果，不代表精度测试通过。
