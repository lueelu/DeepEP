<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# EpGmmFused Beta

Ascend950 上的两个 BF16 通算融合算子：`alltoallv_gmm`（AllToAllV → GMM）和
`gmm_alltoallv`（GMM → AllToAllV）。源码迁自
[ops-transformer MR 11602](https://gitcode.com/cann/ops-transformer/merge_requests/11602)，
固定提交、原始文件哈希见 [upstream.json](upstream.json)；迁入代码保留原有文件头和 [CatCCOS 许可文本](csrc/catccos_overlay/LICENSE)。

本目录包含内核、Host、Python API 和设备验证工具。根构建按 `DEEPEP_BUILD_EP_GMM_FUSED`
选择完整实验单元。设备算术和 UDMA 完成协议沿用原 PR；Linux/CANN 原生构建、
Ascend950 精度、性能和流水采集仍需实机验收。核心 `deep_ep` 的 V2 expanded dispatch/combine
支持范围见 [Elastic 文档](../../docs/elastic.md)。

```text
ep_gmm_fused/
├── csrc/                    # 两个算子的 Host/Device、工作区、CatCCOS overlay
├── python/deep_ep_experimental/ep_gmm_fused/
├── tests/                   # 精度泛化、性能、采集
└── CMakeLists.txt           # 根 CMake 的可选子单元
```

SHMEM 的初始化、内部对称工作区和销毁复用根 `csrc/runtime/session.hpp/.cpp` 的可选 UDMA 入口。
本模块借用固定 `400 MiB + 4 KiB` 工作区，负责算子布局和实际执行流同步。产品包不依赖测试目录。

## 构建

根构建要求 Linux、Python ≥3.11、Ascend950、CANN 9.2.0、配套 PyTorch/torch_npu 和系统 `libhcomm.so`。
Host 编译器与核心 runtime 的通用要求见[根构建指南](../../docs/build.md)；本模块的依赖准备、安装与验证见本节。
已有主线 SDK 时设置 `SHMEM_ROOT`；没有时由准备脚本构建：

```bash
source /path/to/ascend-toolkit/set_env.sh
export DEEPEP_NPU_ARCH=Ascend950
python -m pip install -r requirements-build.txt -r requirements-dev.txt
bash scripts/prepare_ep_gmm_fused_dependencies.sh
# 将脚本末尾输出的 export 命令复制到当前终端，再执行：
# 编译打包
DEEPEP_BUILD_EP_GMM_FUSED=ON python -m build --wheel --no-isolation
# 单独安装：替换为本次生成的实际文件名
python -m pip install --force-reinstall --no-deps dist/实际文件名.whl
```

更新代码后重新编译，再安装生成的 wheel，无需先卸载。
查询安装用 `python -m pip show ascend-deepep`，卸载用 `python -m pip uninstall ascend-deepep`。
请先激活目标 Conda/venv；使用本次生成的实际 wheel 文件名，避免通配符选中多个旧版本。
常规命令无需 `-I`；`--no-deps` 保留当前框架依赖，构建失败不会影响原有安装。
只改测试或文档无需重新编译；修改 Python 包或 kernel 后需要重新构建并安装。

仅在 `DEEPEP_BUILD_EP_GMM_FUSED=ON` 时，默认 `.egg-info` 位于 `build/metadata/`；
关闭或未设置该开关时，保留主线默认的源码根目录位置。切换到本模块构建前，若已有根目录元数据，可先归档：

```bash
if [ -d ascend_deepep.egg-info ]; then
    mkdir -p build
    mv ascend_deepep.egg-info "$(mktemp -d "$PWD/build/old-metadata-XXXXXX")/"
fi
```

仅新建 Conda 环境不会清理旧源码目录的元数据。排查源码路径干扰时可对比
`python -m pip show ascend-deepep` 与 `python -I -m pip show ascend-deepep`；
后者忽略当前目录、`PYTHONPATH` 和用户级包目录，不作为日常必需参数。

根构建固定 CatCCOS `878c2e0a504be8c49c03b5e1157a7c714a628e29`、其 CATLASS gitlink，
SHMEM 沿用主线 [dependencies.lock.json](../../dependencies.lock.json) 固定的提交。
`SHMEM_SOURCE_PATH` 为同一提交的干净源码，提供 device 私有头；`SHMEM_ROOT` 为已记录身份的外部 SDK。
`CATCCOS_SOURCE_PATH` 可显式指定；提交不匹配、源码脏或子模块不匹配时停止。
准备脚本校验并复用已有 SDK；默认本地 SDK 尚无身份记录时，按锁定提交构建 Ascend950 SDK 并生成 manifest。
构建使用上游 `scripts/build.sh -soc_type Ascend950`，启用 Bash 的 `errexit/pipefail`，
成功后调用 `scripts/record_shmem_sdk.py` 记录身份；已有 manifest 不匹配时停止，不自动改写身份。
可用 `python scripts/record_shmem_sdk.py --sdk "$SHMEM_ROOT" --check` 单独校验。
默认安装目录为 `${SHMEM_SOURCE_PATH}/install/shmem`；显式指定其他 SDK 路径时只校验，不覆盖。
版本、头文件或哈希不匹配会报错；安装包不携带另一套 SHMEM/memfabric 动态库。
准备脚本最后输出 `export` 命令，须复制到当前终端执行；可用 `PYTHON` 指定该脚本的解释器。

仅开启本模块时，`DEEPEP_LINEINFO=ON` 默认选择 RelWithDebInfo 以支持指令 PC/源码映射；
显式设置 `DEEPEP_BUILD_TYPE` 时以该设置为准。关闭本模块时忽略 `DEEPEP_LINEINFO`。
`DEEPEP_BUILD_EP_GMM_FUSED=ON` 将 `deep_ep._C` 和 `deep_ep_experimental.ep_gmm_fused._C` 打入同一 wheel；
默认 OFF 仅包含核心包，沿用主线构建目录。本模块及其 lineinfo 构建默认使用独立暂存目录，避免旧模块混入核心构建。
显式指定的 `--build-lib`、`--build-platlib` 和 `--build-temp` 保持原值；切换构建模式时应为其选择不同目录。
打包时会检查旧版嵌套暂存目录和显式复用目录中的实验产物；若提示 `Stale EP_GMM_FUSED staging`，
将报错指出的暂存根目录移到构建树之外，再重新编译打包。检查不自动删除缓存。
并发由 `CMAKE_BUILD_PARALLEL_LEVEL` 控制，范围 1–64，默认 4；仅开启本模块时兼容 `MAX_JOBS`。
原生实验只支持 wheel 安装，不支持 editable/inplace；关闭本模块时不施加此限制。工具链变更后使用新 build 目录。
加载时使用主线 `deep_ep._native` 核对实际 SHMEM 库身份。两个扩展可在同一进程加载，
但 MTE 和 UDMA 共享单进程 owner，必须先显式销毁前一个通信域，才能创建下一个。
主线 MTE/MTE+UDMA 入口与 EP_GMM_FUSED UDMA 工厂共用 `Session`，分别保留各自的 bootstrap、QP 和工作区配置。

### 构建安装验证

启用本模块后，可按根构建指南运行 `scripts/validate_package.py`，在新 venv 中构建 wheel/sdist，
从 sdist 重建并检查扩展、来源和许可证；拒绝旧版随包 SHMEM 库，并在同一进程检查两个扩展加载。
sdist 重建复用原始绝对 SDK 路径。加载检查不初始化 UDMA、不运行算子，
设备精度、性能和流水采集仍须按[模块测试说明](tests/README.md)在设备执行。
设备测试需另装 `experiments/ep_gmm_fused/tests/requirements.txt` 中的 NumPy；该依赖不加入主线开发环境。

CPU 构建回归位于 `tests/runtime/test_ep_gmm_fused_build.py`，由根目录 pytest/CPU CI 收集；
它覆盖元数据位置及模块启用/关闭时的打包行为，不替代真实 Bisheng 编译和设备验收。

## 初始化和调用

安装包含 EP_GMM_FUSED 的 wheel 后，准备按全局专家排列的 BF16 输入、每卡专家权重和路由计数。
以下片段在 EP4 域内依次调用两个算子，每卡 2 个专家；返回结果保持源 rank 的专家分组顺序，
不包含原始 token 顺序还原或 top-k 加权归并。

```python
from deep_ep_experimental.ep_gmm_fused import udma_group, alltoallv_gmm, gmm_alltoallv

# 用户已绑定 NPU、初始化 torch.distributed，并创建 ep_group。
with udma_group(group=ep_group):
    h = alltoallv_gmm(x, w1, local_counts, received_counts,
                    ep_size=4, expert_num=8)
    y = gmm_alltoallv(h, w2, local_counts, received_counts,
                    ep_size=4, expert_num=8)
```

`group` 仅负责初始化时的控制通信。组内 rank 0 调用官方 C++
`aclshmemx_get_uniqueid`，按真实全局 root 在组内广播字节；HCCL 使用 NPU tensor，
Gloo 使用 CPU tensor。所有成员调用 `aclshmemx_set_attr_uniqueid_args`，选择
`ACLSHMEM_DATA_OP_UDMA` 后通过 UNIQUEID 模式初始化。无需传 `ep_endpoints`，也不依赖
官方 SHMEM Python 包。默认使用已绑定的 NPU；`local_rank=` 可显式覆盖。

不传 `group` 时保留原 PR DEFAULT 路径：

```python
from deep_ep_experimental.ep_gmm_fused import init_udma_group, destroy_udma_group
init_udma_group(rank=ep_rank, world_size=4, local_rank=local_device,
                init_method="tcp://192.0.2.10:29601", heap_size=1 << 30)
# 调用算子...
destroy_udma_group()
```

省略参数会读取 `RANK/WORLD_SIZE/LOCAL_RANK`；地址读取 `UDMA_MASTER_ADDR/UDMA_MASTER_PORT`，
回退到 `MASTER_ADDR` 与 `MASTER_PORT+1`。手动模式多个 EP 域需独立 endpoint。
传 `group` 时禁止同时传 `rank/world_size/init_method`。

一进程一个 UDMA 域，所有 EP 成员按相同顺序初始化、调用和释放；相同配置重复初始化幂等。
组模式会校验全组的堆大小、成员、UID ABI 和初始化状态。嵌套 context 只由创建域的外层释放；
显式初始化的域不会被借用它的 context 销毁。禁止在 context 存活时显式销毁。
不自动接管外部初始化的 SHMEM，也不销毁传入的 ProcessGroup。
初始化、算子调用和销毁须在同一 host 线程按序执行，各 rank 保持相同顺序；SHMEM 清理失败后须重启作业。
普通输入输出使用 NPU 内存；对称内存仅用于算子内部通信工作区。
逆向 `gmm_alltoallv` 的输出行数取决于路由，Meta/FakeTensor 或编译路径须提供形状为
`[sum(local_tokens_per_expert), n]` 的 `out`；普通 eager 调用仍可自动分配输出。

## 64 卡、EP=4、外层 EFTP

创建 16 个互不相交的 EP ProcessGroup，每进程将所属的一个组传入上述接口即可。
全局 rank（0–63）、组内 EP rank（0–3）、可见设备号是三个不同概念。
路由数组按组内 rank 顺序排列；`expert_num` 为该 EP 域内逻辑专家总数。
外部 EFTP 可分片 K/N；所需 TP collective 由上层模型组织，算子内部仍为 TP=1。

连续/交错分组的 64 卡 EP4 域隔离用例及多节点启动参数见[测试说明](tests/README.md)。
跨节点域需要部署的 UDMA 网络可达，资格以实际拓扑测试为准。

## 张量与路由约定

- EP size 1–8；BF16 contiguous NPU 输入；每卡权重 `[local_experts, K, N]`。
- `local_counts` 长度 `expert_num`，按目标 EP rank、目标 local expert 排列。
- `received_counts` 长度相同，按源 EP rank、当前 rank 的 local expert 排列。
- 正向输入 `[M,K]` 按全局专家排列，每 rank 的 M 相同；结果按 local expert、源 rank 排列。
- 逆向输入使用正向结果的排列，结果恢复源 rank 的全局专家顺序；本地输入/输出行数可不同。
- 正向最多 64 个本地专家，逆向最多 32 个；输入 M 必须大于 0。
- 路由可传 host 整数列表、CPU int64、NPU int64。host 路径每次包含 H2D；性能用例默认 NPU。
- 正向可 `received_rows=` 避免 count D2H，可 `output_all_to_allv=True` 返回 `(result, dispatched)`。
- 两算子均支持 `out=`；正向导出还支持 `all_to_allv_out=`。异步结果需按 NPU stream 规则消费。
- 所有 rank 必须使用一致的形状配置、路由对应关系及 tiling；本库不是 autograd 封装。

## API 名称与别名

| 功能 | 主 API | 长名称别名 |
| --- | --- | --- |
| AllToAllV → Grouped Matmul | `alltoallv_gmm` | `all_to_allv_grouped_matmul` |
| Grouped Matmul → AllToAllV | `gmm_alltoallv` | `grouped_matmul_all_to_allv` |

短名称和长名称指向同一函数，参数、返回值和行为完全一致。按需导入一种名称即可：

```python
from deep_ep_experimental.ep_gmm_fused import alltoallv_gmm, gmm_alltoallv
```

偏好长名称时使用：

```python
from deep_ep_experimental.ep_gmm_fused import (
    all_to_allv_grouped_matmul,
    grouped_matmul_all_to_allv,
)
```

完整用例、统计口径、采集和重放见 [测试说明](tests/README.md)。

## 开发与构建接入

本模块沿用仓库根目录的格式和检查配置。C++ 命名空间为 `deepep::ep_gmm_fused`，
头文件保护宏使用 `DEEPEP_EP_GMM_FUSED_*_HPP`；迁入代码保留原始版权与来源记录。

- [组件 CMake](CMakeLists.txt) 定义可选扩展目标，由根构建在开启本模块时接入。
- [依赖校验](../../cmake/EpGmmFusedDependencies.cmake) 解析框架依赖，检查 CatCCOS、CATLASS 和 SHMEM 源码版本及洁净状态。
- [编译辅助](../../cmake/ascend/) 提供本模块的 AscendC 工具链和算子编译规则。
- [依赖准备脚本](../../scripts/prepare_ep_gmm_fused_dependencies.sh) 准备源码和固定版本 SDK，下载目录不纳入本仓提交。

本组件中由 Lu Lu 持有权利的代码采用根目录 [BSD 2-Clause 许可证](../../LICENSE)。迁入代码保留原有版权声明和许可证文本。
