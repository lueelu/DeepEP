# 独立多轮分层 AllToAll（Ascend950）

本样例独立构建 C++/AscendC 可执行程序，不依赖 PyTorch、DeepEP wheel、dispatch/combine。
调用仓内 `csrc/comm` 的公共 block、tile 和 kernel 模板，底层继续使用 Catlass TileSwizzle 与 SHMEM。
当前实现支持一个通信域，或两个等大的通信域，最大双机 32 卡，`QP_NUM=1/2/4`。
默认执行单次功能校验；也可运行独立 AllToAll 带宽测试。

## 数据与多轮划分

每个 rank 的逻辑输入为 `data[world, M, N]`，类型为 MXFP8 E4M3；
`scale[world, M, N/32]`，类型为 E8M0。`N` 必须是正的 32 倍数。
输入的第 `peer` 片发送至 `peer`，写入该 peer 输出的第 `source_rank` 片。
每个 peer 的 data/scale 起始地址分别按 64 字节对齐，padding 不参与发送。
样例在 host 生成有限值 FP8 编码并逐字节校验，不执行量化或矩阵乘法，无需输入文件。

- `WINDOW_M`：kernel 外层按行划分通信批次，最后一轮支持不足整窗的尾块。
- `COMM_BLOCK_M`：编译期参数，swizzle 在一轮内对 MTE 再按行切块。
- 每轮 UDMA/RDMA 发送当前 peer 的整个窗口，block 内按 32 字节边界分给多个 QP。
- MTE block 使用 TileSwizzle 继续切成可放入 UB 的 tile，逐 tile 搬运。
- `GROUP_SIZE`：编译期参数，连续的这几个 rank 必须实际处于同一个 UDMA 域。
  双机 32 卡用 `GROUP_SIZE=16`，rank 0–15 在第一台、16–31 在第二台。
  这是一项部署约定，不是根据 rank 编号自动识别物理拓扑。
- 一条 AIV 独占一个 peer 的全部 QP。增加 QP 不增加该 peer 的 AIV 数量。
  MIX 1:2 下，block 数从平台 AIC 核数读取，physical AIV index `[0, world)` 对应 peer。
  另用 AIV `world` 独立协调接收，因此双机32卡至少需要33个AIV、17个AIC。

例如 `M=257, N=1024, WINDOW_M=64`，共 5 轮：`64+64+64+64+1` 行。
所有轮写入输出矩阵的不同区间，整个输入/输出都保留至 kernel 结束，没有环形缓冲复用。

## 完成通知

每轮分别发送 data、scale，然后通知接收端：

| 通信方式 | 一轮的顺序 | quiet |
| --- | --- | --- |
| 本卡 MTE | data/scale tile 搬运完成 → 本地 ready | MTE 事件确认完成 |
| 同域 UDMA | 提交 data/scale → quiet 所有数据 QP → 发 flag | 每轮；flag 发完再 quiet，防止下一轮 flag 乱序覆盖 |
| 跨域 RDMA | 每条 QP 内 data → scale → flag | 所有窗口提交后，该 AIV 对这个 peer 的每条 QP 各 quiet 一次 |

RDMA 每条 QP 都有独立的 `ready[source_rank, qp]`，不能只检查 QP0。
即使某条 QP 的本轮 data/scale 分片为空，也提交一个 flag。
flag 源值由 host 预先初始化为 `window+1`，每轮一个紧凑 uint32，只读且所有 peer 共享，整个 kernel 期间不修改。
RDMA 优化依赖同一 QP 的 WRITE 保序及远端可见性；不把该保证推广到 UDMA 或不同 QP。
SHMEM 在 SQ/CQ 接近满时仍可能内部 poll 回收队列，这不等于样例每轮调用 quiet。

每卡独立协调 AIV（physical index = world）等待所有 `source × qp` 的 epoch `>= window+1`，然后更新本地 frontier。
样例在 kernel 和最终 barrier 完成后，由 host 逐字节校验完整的 data、scale 和 padding。

每轮 frontier 表示“本卡已收到这一轮所有来源的数据”，不代表远端消费者也已结束。
输入就绪、最终 kernel 完成和校验结束使用独立的 host 顺序 barrier。
循环内部不调用 host，不做逐轮全卡 dissemination barrier。

公共 block 默认等待 RDMA 完成。在 HNS_1825 上，本样例使用 `PerWindowOrderedRdma`：
每条 RDMA QP 依次提交 data、scale 和独立 ready，由 Ready 层的 `FinishPeer` 在全部窗口结束后统一排空。
其他 RDMA 后端使用逐窗完成等待。UDMA 在所有后端都先完成本窗数据，再发布 ready。

## 依赖和构建

两台机器需要匹配的 Ascend950 驱动/固件、CANN（当前仓目标为 9.2）、Bisheng、CMake >= 3.20、Python >= 3.8。
Python 只用于脚本检查，不需要 torch。

准备以下两个依赖：

1. 支持 Ascend950 的 Catlass 源码，可使用本仓 `third-party/catlass` 或已有 Catlass 目录。
   设置 `CATLASS_SOURCE_PATH` 指向包含 `include/catlass` 的源码根目录。
   通信模板已迁入 `csrc/comm`，不再依赖外部 CatCCOS 通信头文件。
2. 已编译安装的 SHMEM SDK，启用 UDMA + RDMA、多链路及目标 backend。
   `SHMEM_ROOT` 必须指向包含 `include/`、`src/`、`lib/libshmem.so` 的 **install/shmem**，
   不能指向 SHMEM 源码根目录或 install 父目录。
   加载 CANN 环境后，在 SHMEM 源码根目录执行：

   ```bash
   bash -e scripts/build.sh \
       -soc_type Ascend950 \
       -enable_rdma \
       -rdma_backend HNS_1825 \
       -multi_link
   ```

   `-soc_type Ascend950` 必须显式指定，否则 `-rdma_backend HNS_1825` 会被拒绝。
   PI_ON_CHIP 必须与 SDK 构建选项及目标固件一致。

公共模板通过 `deepep_comm` CMake INTERFACE target 引入；仓库的 SHMEM 子模块沿用 `release/v1.8.0` 的锁定节点，
此样例独立构建，不修改 SHMEM 源码。
兼容 SHMEM 使用 runtime_config 或旧 `aclshmemx_set_shared_jetty` 设置独占 jetty 的接口版本，
禁止多个 peer AIV 并发修改同一个共享 SQ/CQ。
不创建、不修改 `/etc/hccl_rootinfo.json`；拓扑解析由所选 SHMEM 和机器环境负责。
开始混合通信测试前，应先确认同一 SDK 的 UDMA/RDMA 样例可用。

两台机器分别设置实际路径并编译（路径不要求相同）：

```bash
source /实际路径/cann/set_env.sh
source /实际路径/shmem/install/set_env.sh

export DEEPEP_ROOT=/实际路径/DeepEP
export CATLASS_SOURCE_PATH="$DEEPEP_ROOT/third-party/catlass"
export SHMEM_ROOT=/实际路径/shmem/install/shmem
export GROUP_SIZE=16
export COMM_BLOCK_M=64
export SHMEM_RDMA_BACKEND=HNS_1825
export SHMEM_HNS_1825_PI_ON_CHIP=1  # 若 SDK/固件使用 0，这里必须同样填 0

# 使用本仓 Catlass 且尚未初始化时执行：
git -C "$DEEPEP_ROOT" submodule update --init --recursive third-party/catlass

bash "$DEEPEP_ROOT/experiments/hierarchical_alltoall/scripts/build.sh"
```

若已有 CatCCOS 环境，可直接设置 `CATLASS_SOURCE_PATH="$CATCCOS_SOURCE_PATH/3rdparty/catlass"`。
为兼容原命令，未设置 CATLASS_SOURCE_PATH 时，build.sh 会优先使用已有 CATCCOS_SOURCE_PATH 下的 Catlass，
否则使用本仓 `third-party/catlass`；不再注入 CatCCOS include 路径或覆盖它的头文件。

默认构建目录为 `$DEEPEP_ROOT/build/hierarchical_alltoall`。
编译器优先使用显式设置的 `CXX`；未设置时依次查找 CANN 下的
`tools/bisheng_compiler/bin/bisheng`、`compiler/ccec_compiler/bin/bisheng` 和
`$(uname -m)-linux/bin/bisheng`。可用 `export CXX="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng"` 手动指定。
可设置 `BUILD_DIR` 更换目录，运行时需保持一致。
构建记录 `build_config.json`，包括 SDK 路径、关键头文件/库及可执行文件的 SHA256、
Catlass commit、GROUP_SIZE、backend 和 PI 配置。运行脚本检查 SDK 是否已被替换。
SDK 目录可不提供 `version.info`；此处不从错误的父目录推断版本。

若使用 XSCALE：设置 `SHMEM_RDMA_BACKEND=XSCALE` 并 `unset SHMEM_HNS_1825_PI_ON_CHIP` 后重编译。

## 双机 32 卡运行

两端用一致的 shape、窗口、QP、端口、SDK/backend 配置。
`RANK0_IP` 是第一台机器可被第二台访问的主机 IPv4，两端都填第一台的 IP。
脚本不通过 SSH 自动启动另一台，请分别启动。

先在两端各执行：

```bash
# 保持上面的 CANN、SHMEM 环境已加载；DEEPEP_ROOT 使用本机路径。
export NNODES=2
export RANK0_IP=90.90.93.84  # 替换为实际第一台服务器地址，两端一致
export PORT=8666
export DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export M=257 N=1024 WINDOW_M=64
export QP_NUM=1
export RUN_ID=alltoall-32-qp1
```

第一台：

```bash
NODE_RANK=0 bash "$DEEPEP_ROOT/experiments/hierarchical_alltoall/scripts/run.sh"
```

第二台：

```bash
NODE_RANK=1 bash "$DEEPEP_ROOT/experiments/hierarchical_alltoall/scripts/run.sh"
```

每个 rank 应打印 `PASS data+scale windows=5 qp=1`。
检查**两端总共 32 个 rank**，单端成功不代表完整测试通过。
日志在 `$BUILD_DIR/logs/$RUN_ID-node0` 和 `...-node1`，已有目录拒绝覆盖。
进程名为实际可执行文件 `hierarchical_alltoall`，不改名为 python。
脚本没有外部 timeout/kill；某 rank 出错时，其他 rank 可能仍在等待，需检查两端日志。

分别测试 QP2/QP4：上一轮两端都结束后，两端都修改并重新运行上面的启动命令：

```bash
export QP_NUM=2 PORT=8667 RUN_ID=alltoall-32-qp2
# 两端各运行一次对应 NODE_RANK 的 run.sh，等待两端结束。

export QP_NUM=4 PORT=8668 RUN_ID=alltoall-32-qp4
# 两端再次各运行一次。
```

追加小 scale/空 QP 分片测试：`M=7 N=32 WINDOW_M=3 QP_NUM=4`（3+3+1 行）。
追加长序列测试：`M=1025 N=32 WINDOW_M=1 QP_NUM=4`，覆盖大量 epoch 和 RDMA SQ/CQ 回收。
为每组测试设置新的 RUN_ID。
双机每台不足 16 卡时，需要按每台实际卡数重新编译 GROUP_SIZE；例如每台两卡用 GROUP_SIZE=2。

## 单机运行

`DEVICES` 填同一个 UDMA 域的设备号，卡数最多为编译时的 `GROUP_SIZE`。
若 A5 上 SHMEM 的默认多链路选路把部分 QP 分配到 CLOS，且 CLOS 无法建链，可指定所有 UDMA QP 使用本地直连层：

```bash
NNODES=1 NODE_RANK=0 DEVICES=0,1 QP_NUM=4 UDMA_NET_LAYER=0 \
    bash "$DEEPEP_ROOT/experiments/hierarchical_alltoall/scripts/run.sh"
```

`UDMA_NET_LAYER=0` 选择直连层，`UDMA_NET_LAYER=1` 选择 CLOS 层。设置该选项需要使用启用了
`-multi_link` 的 SHMEM SDK，同一次运行的所有 rank 使用相同配置。省略时由 SHMEM 自动选路。

## 带宽测试

沿用上面的构建命令，默认预热 5 次、计时 50 次。通信大小沿用 bench 的口径：每个 peer 的
MXFP8 **data** 为下表大小，E8M0 scale 随同发送，额外占 data 的 1/32。每个 case 仍校验最后一次输出。

| M × N（每个 peer） | data | scale |
| --- | ---: | ---: |
| 4 × 1024 | 4 KiB | 128 B |
| 64 × 1024 | 64 KiB | 2 KiB |
| 1024 × 1024 | 1 MiB | 32 KiB |
| 2048 × 1024 | 2 MiB | 64 KiB |
| 4096 × 1024 | 4 MiB | 128 KiB |

单机双卡示例（设备须同域且空闲；此例指定 UDMA 直连）：

```bash
NNODES=1 NODE_RANK=0 DEVICES=0,1 UDMA_NET_LAYER=0 QP_NUM=4 \
  RUN_ID=perf-2card-qp4 PORT=8666 bash experiments/hierarchical_alltoall/scripts/perf.sh
```

双机 32 卡沿用上一节的环境变量，在两端用相同的 `RUN_ID`、`PORT`、`QP_NUM`、
`WARMUP`、`ITERATIONS` 和 `WINDOW_M`，分别在各自机器执行：

```bash
# 第一台（沿用 NNODES=2、RANK0_IP、DEVICES 等）
NODE_RANK=0 RUN_ID=perf-32card-qp1 bash experiments/hierarchical_alltoall/scripts/perf.sh

# 第二台
NODE_RANK=1 RUN_ID=perf-32card-qp1 bash experiments/hierarchical_alltoall/scripts/perf.sh
```

默认 `WINDOW_M=256`，可覆盖；连续使用 `PORT` 到 `PORT+4` 五个端口。
单机自动生成 `$BUILD_DIR/logs/$RUN_ID/summary.csv`。双机把两端的 sweep 日志目录收集到同一台机器后汇总：

```bash
python3 experiments/hierarchical_alltoall/scripts/perf_summary.py \
  node0/perf-32card-qp1 node1/perf-32card-qp1 > summary.csv
```

参考 CatCCOS 融合样例将各段分开测量的思路，这里用同一设备 stream 上的 ACL 时间戳事件打点：
数据 kernel 之前、数据 kernel 之后、完成 barrier 之后。分别记录 kernel 和完整完成耗时，不计输入准备、
ready 清零、起跑 barrier 或校验。每次迭代先取所有 rank 的最大耗时，再算均值和完成延迟分位数，
不跨设备相减时间戳。带宽以完整完成耗时为分母：

```text
有效发送 GB/s/rank = (world - 1) × (data_peer_bytes + scale_peer_bytes) / complete_mean_us / 1000
```

这个值只计发给其他 rank 的逻辑字节，包含同域 UDMA 和跨域 RDMA，不计本卡 MTE、接收字节、padding 或 flag，
不能当作某条链路的物理带宽。改变 QP 数时使用新的 RUN_ID。

## 文件

| 文件 | 作用 |
| --- | --- |
| `main.cpp` | 参数、SHMEM/ACL、对称内存、数据生成和校验 |
| `kernel.cpp` / `launch.h` | MIX kernel 与 host 调用接口 |
| `../../csrc/comm/block/` | 公共 HierarchicalCommSwizzle 与 HierarchicalCopy |
| `../../csrc/comm/tile/` | 公共 MTE tile、ready 布局、通知/等待和 quiet 协议 |
| `../../csrc/comm/kernel/` | 公共多轮 AllToAll、参数/矩阵布局及全卡 barrier |
| `scripts/` | 独立构建、SDK 检查和本机 rank 启动 |

公共层接口与复用约束见 [csrc/comm/README.md](../../csrc/comm/README.md)。
迁移代码基于 CatCCOS `2ab9a31`，许可证保存在 `csrc/comm/LICENSE.catccos`。
样例在 `kernel.cpp` 组合数据类型、block 和 scheduler。
