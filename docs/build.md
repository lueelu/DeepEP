<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 构建、打包与安装

wheel 包含 Host 运行时、Notify 及 HT/LL dispatch/combine kernel。输入和执行约束见 [Elastic 接口](elastic.md)，资源与事件见 [API 指南](api.md)。
MegaMoE 和 EpGmmFused 通过独立开关选择构建，见下文实验组件说明。
包名为 `ascend-deepep`，导入名仍为 `deep_ep`；不要与其他提供 `deep_ep` 的包装在同一环境。

## 前置条件

- Linux、Python ≥3.11、C++17 编译器、CMake ≥3.20，以及 Make 或 Ninja；构建工具须在 PATH 中。
- CANN 9.2.0 与配套驱动/框架，目标 Ascend950；构建仅接受该 CANN 版本与目标架构。
- 由本仓子模块构建或使用外部固定提交的 SHMEM SDK（见 `dependencies.lock.json`），安装根包含 `include/`、
  `lib/libshmem.so` 以及 `src/device`、`src/device_simt` 等 Device 头文件。
- 普通（HT）dispatch/combine 依赖 SHMEM 的 shared-jetty 支持，包括 Host 配置接口和 Device 共享队列状态；使用锁文件固定的 `cann/shmem` dev 提交，不直接跟随分支最新版本。
- CANN 安装中须包含 `tools/bisheng_compiler/bin/bisheng`，Device 使用 `-xasc` 编译。
- 普通 CPU 测试仅需 `requirements-dev.txt`；wheel 构建依赖固定在 `pyproject.toml`。
- wheel 构建不自动下载/编译外部 SDK，不将驱动/CANN/SHMEM 打入 wheel。SDK 及其传递依赖必须由部署环境提供。

## 可选实验组件

MegaMoE 的构建和多机测试入口位于 `experiments/megamoe/scripts/`，
命令、依赖和环境配置见 [MegaMoE 使用说明](../experiments/megamoe/README.md#2-编译安装)。
EpGmmFused 通过 `DEEPEP_BUILD_EP_GMM_FUSED=ON` 构建 AllToAllV → GMM 和 GMM → AllToAllV，
需要单独准备 CatCCOS、其 CATLASS 依赖及 SHMEM SDK，见 [EpGmmFused 使用说明](../experiments/ep_gmm_fused/README.md#构建)。
基础包使用下面的通用构建流程，两个实验组件默认均关闭。

## 分步骤构建（外部 SDK、打包或 CI）

先激活已安装匹配 Torch/torch_npu 的 Python 环境。以下命令在仓库根目录执行，使用仓内固定版本依赖：

```bash
source /path/to/ascend-toolkit/set_env.sh
git submodule update --init --recursive
unset SHMEM_ROOT CATLASS_ROOT  # 使用仓内依赖；外部 SDK 用户见下文。
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"

python -m pip install -r requirements-build.txt
```

主构建默认查找 `third-party/shmem/install/shmem`；MegaMoE 启用时默认查找 `third-party/catlass`。
准备脚本默认要求 SHMEM 源码干净且提交与锁文件一致；已有 build/install 时要求显式 `--rebuild`（会再生成这些构建产物）。
更换 SDK 前停止使用它的作业，之后重建 wheel。完整边界及离线依赖见 [第三方依赖](../third-party/README.md)。

已有外部 SDK 时，加载 CANN 环境、设置 `DEEPEP_NPU_ARCH=Ascend950` 并安装 `requirements-build.txt` 后，跳过子模块准备和 SHMEM 构建，改用：

```bash
export SHMEM_ROOT=/path/to/pinned/shmem-sdk
source "$SHMEM_ROOT/set_env.sh"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
# 仅未登记身份的 SDK 运行一次；不能改 manifest 来绕过源码/二进制检查。
python scripts/record_shmem_sdk.py --sdk "$SHMEM_ROOT" \
  --revision "$(python -c 'import json; print(json.load(open("dependencies.lock.json"))["shmem"]["revision"])')"
# 启用 MegaMoE 且不用仓内子模块时，再设置 CATLASS_ROOT=/path/to/pinned/catlass。
```

sdist 不携带子模块，解包后构建需显式设置 SHMEM_ROOT，以及启用 MegaMoE 时所需的 CATLASS_ROOT；EpGmmFused 的外部源码路径按其构建说明设置。
下方自动化验证入口会将原 checkout 的默认依赖路径固定为绝对路径，确保 sdist 重建使用相同 SDK。

版本声明由 SDK 提供方负责，脚本校验固定公共头文件、记录头文件及库的校验和；不把“头文件一致”等同于“二进制来源可信”。
基础构建对 SDK 身份或校验和不匹配发出警告，并记录实际头文件和库的校验和；缺少必需文件仍会失败。
MegaMoE 构建及仓内 SDK 复用保持严格校验，SDK 必须与锁文件匹配，不能修改 manifest 绕过检查。

`setup.py` 元数据查询不需要 SDK/框架。实际构建只经 `build_support.py` 调用根 CMake。
不同工具链、SDK、Python 环境和构建模式使用不同指纹目录；源码哈希仅用于标识产物，
修改 Python、测试或单个 kernel 不切换整个构建目录。`DEEPEP_BUILD_MEGAMOE` 默认 OFF；
ON 的构建与使用方式见 [MegaMoE 文档](../experiments/megamoe/README.md)。
安装产物没有构建机绝对 RPATH，使用部署环境的 SDK 路径；加载时核对实际 SHMEM 二进制身份。

## 编译与安装

Host 编译器优先使用 `CXX` 指定的可执行文件；未设置时依次查找 PATH 中的
`c++`、`g++`、`clang++`、`bisheng++`、`bisheng`，再查找 CANN 的
`tools/bisheng_compiler/bin`。自动探测会跳过无法执行 `--version` 的候选；
显式指定的 `CXX` 无效则报错。保留 `c++ -> ccache` 等符号链接的调用名称，
不要把 `CXX` 设成裸 `ccache`。编译器是否满足 C++17 和链接要求仍由 CMake/实际构建验证。
日志会输出选中的 Host 编译器路径，Device kernel 继续使用 CANN Bisheng。

单机编译并安装仍使用：

```bash
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

这条命令会编译 Host 和 Device、生成 wheel 并安装到当前 Python 环境。
`requirements-build.txt` 只安装构建依赖，不能替代上述命令；修改 kernel 或 Python 包代码后重新执行上述命令。
只改 `tests/` 或文档时，更新共享源码即可，无需重新编译安装。

多机共享目录只需在一台机器生成 wheel，再在各节点的 Python 环境安装同一个文件：

```bash
python -m pip wheel -v --no-build-isolation --no-deps . -w /path/to/shared/wheels
python -m pip install --force-reinstall --no-deps /path/to/shared/wheels/实际文件名.whl
```

`pip wheel` 只编译打包，不安装。第一条仅执行一次，第二条在每个独立 Python 环境执行；
使用本次生成的实际文件名，不要硬编码旧版本，也不要在多台机器同时编译同一共享 checkout。
每台运行节点都需要加载匹配的 CANN/SHMEM 环境并设置运行时共享 endpoint，详见
[单机/多机测试](../tests/README.md)。

若需要同时交付源码包与 wheel，可在安装构建依赖后执行
`python -m build --sdist --wheel --no-isolation`，产物位于 `dist/`；该命令同样不会安装。

以下选项同时作用于 Host 和 Device；默认 Release、4 个编译任务：

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=8
export DEEPEP_BUILD_TYPE=RelWithDebInfo  # Release / RelWithDebInfo / Debug
```

支持 Ninja、Unix Makefiles 等单配置生成器。不同构建模式保留各自的缓存。
并行度改变不会切换缓存；内外层并行任务可能重叠，内存不足时降低并行度。
增量复用要求保留同一 checkout 的 `build/`；从新的源码包目录构建会使用新的缓存。

工程分工：

- `build_support.py`：环境校验、工具链缓存、调用 CMake、写入 `_build_info.json`。
- 根 `CMakeLists.txt`：Host 封装、Python 扩展 `_C.so`、能力声明、链接及安装。
- `cmake/Device.cmake`：每次进入独立 Device 构建，由子工程判断是否需要重新编译。
- `cmake/device/CMakeLists.txt`：Bisheng 源文件列表及编译选项，生成 `libdeepep_kernels.so`。
- `cmake/Dependencies.cmake`：Host/Device 共用的 CANN 和 SHMEM 库定位。

新增算子涉及的源码注册和检查项见[开发指南](development.md)。
编译器依赖信息负责追踪公共头文件；无改动、仅改 Python/测试时，不重新编译原生目标。
修改一个 kernel 时仅编译受影响的对象并重新链接，修改公共头文件时重编依赖它的对象。

wheel 中 `_C.so` 和 `libdeepep_kernels.so` 放在同一目录，通过 `$ORIGIN` 查找。
安装阶段会清除 setuptools 构建目录里遗留的旧 `libdeepep_notify.so`；包校验拒绝混入旧库。
CPU 构建预检和打包回归位于 `tests/runtime/test_build.py`，检查配置、缓存键、构建命令与产物要求；
这些测试不执行真实 Bisheng 编译，也不替代 Device 增量构建和设备验收。

## 自动化构建、隔离安装与验证

在已准备好匹配 torch/torch_npu、pytest 与构建依赖的环境中运行：

```bash
python scripts/validate_package.py --output /path/to/new-validation-directory \
  --system-site-packages
```

该命令创建新 venv，在其中自动安装固定的构建/测试依赖，运行源码测试，构建 wheel/sdist、从 sdist 再构建 wheel、检查内容与元数据，安装后在源码目录外运行测试。
无网络环境使用 `--wheelhouse /path/to/wheels`，只从预下载目录安装；缺包明确失败。
`--system-site-packages` **明确复用已有框架依赖**；不是宣称完全隔离所有 Python 依赖。
省略此选项创建不继承系统包的环境，需要另行准备框架/测试依赖，否则验证失败。
不会修改调用方 Python 环境，也不会默认占用设备。输出目录必须全新；失败日志/产物保留用于定位。

设备资源已由作业调度方分配后，可追加真实运行时验证：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
python scripts/validate_package.py --output /path/to/another-new-directory \
  --system-site-packages --runtime-ranks 2
```

设备编号和端口为示例，必须替换为本作业已预留资源。不要复用已有任务的通信端口。
支持显式选择 EP2/4/8 的资源测试；结果仅覆盖实际执行的规模。
每个 rank 的成功结果、JUnit XML 和 `validation.json` 一并保留；失败不会写成功摘要。

`--runtime-ranks` 只验证资源、事件与生命周期，不是 dispatch/combine 正确性或带宽。安装后分别运行 HT dispatch、HT combine 和 LL 系统测试，命令集中在[测试入口](../tests/README.md)。普通 pytest 和默认构建安装检查不自动执行设备 ST。

`--ep-roundtrip npu-stub` 可在构建安装后追加双卡 NPU Tensor/HCCL 参考闭环；使用同一组两张已分配设备。该入口不是 DeepEP 原生通信验收。`--ep-roundtrip native` 要求 V1 Buffer 通信方法，不能用于当前 V2 算子，详细边界见[参考闭环说明](../tests/README.md#cpu-与-npu-参考闭环)。

## CI 与产物

GitCode 的 `compile` 流水线运行 pre-commit，并按变更范围运行 CPU API、参考、运行时和构建预检。CPU fault-injection 使用 Linux/g++，不等于 CANN ABI 或 NPU 测试。具体作业和触发方式见[开发指南](development.md#提交前检查与-ci)。

在具备 CANN、SHMEM、框架与已分配 NPU 的环境执行本页验证命令。保留 `source-tests.xml`、`installed-tests.xml`、`validation.json` 和逐 rank 输出，wheel/sdist 作为独立产物归档。安装成功、CPU 通过和设备算子通过分别记录，不互相替代。
