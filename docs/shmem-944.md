# SHMEM !944 与历史版本复测

本页记录历史 EP32 分支（截至 `e2199e6`）的复测方法。运行历史迁移工具时，本地需要保有该分支的 Git 对象。当前分支沿用 `feat/roce-support` 锁定的 SHMEM `b41a8fa`，构建方法见 [构建说明](build.md)。

历史分支固定到 `cann/shmem` master 的 `4c671684e1698aec199619850a3120bfdf85c95e`
（2026-10-31 查询）。此前固定的 `87b63bd3206214450ef09c32010bf9fd1cbad21b` 是
[!944](https://gitcode.com/cann/shmem/pull/944) 合入提交；新版本保留其 HNS1825 PI-on-chip 支持，
并包含 `e658bbc` 的 HNS1825 WQE ULP 修正与 `9c7770c` 的 UDMA 队列状态刷新修正。
六个锁定公共头文件的内容未变化。锁文件与子模块 gitlink 一起更新，不跟随上游 master 自动升级。

新版将 UDMA WQE 拷贝拆成 async/wait 两个 helper，并给 aggregate credit poll 增加 CQ 目标参数。
基础 Dispatch/Combine 调用同步适配，保留每段拷贝完成后才复用 UB／敲 doorbell 的顺序；
FM 有信号批次在发布队列状态前更新 CQ 目标。MegaMoE 路由、计算和通信调度不在本次依赖更新中修改。
必须重新构建 SHMEM 和完整 wheel，不能仅替换 `libshmem.so`。Ascend 设备编译及精度／性能验证仍须上机完成。

## 当前分支：两台机器分别重建

停止使用原 SDK 的作业后，在原 checkout 执行：

```bash
git switch codex/megamoe-hierarchical-roce
git pull --ff-only origin codex/megamoe-hierarchical-roce
git submodule update --init -- third-party/shmem

# HNS1825 新固件使用 1；SDK 与消费它的 Device 内核会使用同一个设置。
export DEEPEP_HNS_1825_PI_ON_CHIP=1
bash experiments/megamoe/scripts/run_bs32.sh build --rebuild-shmem --errors-only
```

两台都构建成功后，分别启动精度测试；精度通过后再测性能：

```bash
bash experiments/megamoe/scripts/run_bs32.sh run
bash experiments/megamoe/scripts/run_bs32.sh perf --perf-method legacy --warmup 20 --iters 100
```

沿用每台机器已有的 `.megamoe-bs32/<hostname>.json` 配置；其 `rdma_backend` 应为 `HNS_1825`。
`--rebuild-shmem` 会清理该子模块的 `build/` 和 `install/` 并重建，避免旧头文件混入新 SDK。
普通后续构建仍复用校验通过的 SDK。构建机需要原有 CANN 9.2.0 与 Torch 环境。

如果 `runtime/kernel.h` 明明存在，编译参数却含有 `libtinfo.so.6: no version information available`
或错误的 `third-party/shmem/x86_64-linux/...` 路径，是 shell 警告污染了 CMake 的架构检测。
准备脚本会在 SHMEM 的 `project()` 后用 Python 检测的主机架构修正 SDK 路径，且保留原生编译模式。
更新脚本后执行上面的 `build --rebuild-shmem --errors-only`，清除上次失败的构建缓存后重试。

如果后续在 `fm_primary.hpp`、`fm_simt.hpp` 或 `wqe_cache.hpp` 报
`aclshmemi_udma_fill_sqe_base_ctx` 需要 5 个参数，则还缺少 Device API 适配。
新 SHMEM 的内部 WQE 构造接口会转换对称堆地址；Dispatch 的 live VA 通过专用适配保留原值，
Combine 则向 SDK 传入本地对称地址，由 SDK 转换到对端 MR。更新当前分支后运行
`build --errors-only` 即可复用已成功构建的新 SHMEM SDK。

## 历史 commit：保留原 SHA，应用依赖适配

Git commit 的内容不可原地修改。直接 checkout 旧 SHA 仍会得到旧的 SHMEM 固定版本。
为保留已有性能记录中的 SHA，使用当前分支提供的脚本创建独立 worktree，并在旧源码上应用兼容修改：

```bash
# 从包含迁移脚本的新版 checkout 执行，两台机器选相同 OLD_REF。
ORIGINAL_ROOT="$PWD"
OLD_REF=55a3d19
python scripts/prepare_shmem_history.py "$OLD_REF" --output "../DeepEP-${OLD_REF}-shmem944"
cd "../DeepEP-${OLD_REF}-shmem944"
git submodule update --init -- third-party/shmem

# 复用该机器原来的 BS32 配置；worktree 不复制未跟踪的主机配置文件。
BS32_CONFIG="$ORIGINAL_ROOT/.megamoe-bs32/$(hostname).json"
export DEEPEP_HNS_1825_PI_ON_CHIP=1
bash experiments/megamoe/scripts/run_bs32.sh build --config "$BS32_CONFIG" --errors-only
```

两台都完成后，在各自历史 worktree 中执行：

```bash
bash experiments/megamoe/scripts/run_bs32.sh run --config "$BS32_CONFIG"
bash experiments/megamoe/scripts/run_bs32.sh perf --config "$BS32_CONFIG" --warmup 20 --iters 100
```

脚本只更新 SHMEM 依赖、Host/Device API 适配、SDK 构建脚本与 doorbell 编译传递；不会复制新版算子、调度、
通信协议或 benchmark 脚本。旧 commit 的计时方法、Exp 次数和参数支持也保留原样，比较数据时需核对一致。
早于 BS32 脚本引入的 commit，依赖迁移仍适用，但须使用该版本原有的构建与测试入口。
历史上已有的编译或算法缺陷不会由依赖迁移自动修复。

worktree 的 HEAD 仍是 `OLD_REF`；兼容修改保留为可查看的工作区差异，SHMEM gitlink 单独暂存以供
`git submodule update` 读取。`shmem-944-migration.json` 记录原提交、新依赖和修改文件校验和。
同一个输出目录不能重复使用。切换版本需重新构建并安装对应 wheel；不要让两个 worktree 的作业并发
覆盖同一个 Python 环境。复测记录应写为“原 SHA + 实际锁定的 SHMEM SHA”，不能标成完全未修改的旧版本。

## Doorbell 模式与其他接口

- 仓内 HNS1825 构建默认 `DEEPEP_HNS_1825_PI_ON_CHIP=1`，适用于新固件。旧固件需显式设置 `0` 并重建。
  这是部署选择，不靠构建机是否能运行 `hinicadm5` 来猜测；不要跨不同固件模式复用同一套产物。
- 模式保存在 SDK 的 `deepep-sdk.json`：`build.hns_1825_pi_on_chip`。Host、Device 与 MegaMoE 的
  直接 Bisheng 编译均从 SDK 读取相同模式；缺失或冲突会停止构建。模式参与构建缓存与 SDK 复用校验。
- 外部 SDK 首次登记时，对 HNS1825 追加 `--rdma-backend HNS_1825 --hns-1825-pi-on-chip 1`，
  数值必须是其实际构建参数。登记不会改变已有二进制；不要修改旧 manifest 绕过重建。
- shared-jetty 已改为 `attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED`。
  新 master 移除了旧 dev 的稀疏 UDMA QP 接口；基础 UDMA 路径须显式设置 `DEEPEP_SHMEM_DENSE_QPS=1`，
  资源需求和性能可能变化，需要单独验证。BS32 MegaMoE RoCE 路径不使用这个接口。

本地检查覆盖依赖锁、doorbell 模式传递与历史迁移，不代表各历史版本已在新驱动上通过设备编译、精度和性能测试。
