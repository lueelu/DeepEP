---
name: build-megamoe-train
description: 在 Ascend950 环境构建和安装含 MegaMoE 的 DeepEP wheel，核对 SHMEM、CATLASS、Timer 配置及 BF16 推理和训练的原生入口。
---

# 构建 MegaMoE Train

量化推理、BF16 推理和训练共用 `ascend-deepep` wheel、`deep_ep_experimental.megamoe` 包和 Host 扩展。

## 依据与范围

开始前读取当前 checkout 的以下文件，按当前分支内容执行：

- [根 README](../../../../README.md)：基础环境要求与安装方式。
- [MegaMoE 模块 README](../../../../experiments/megamoe/README.md#2-编译安装)：共用的构建脚本、配置与安装检查。
- [requirements-build.txt](../../../../requirements-build.txt)：Python 构建工具的固定版本。
- [dependencies.lock.json](../../../../dependencies.lock.json)：CANN、SHMEM 和 CATLASS 的版本约束。

以下命令从仓库根目录运行。

本 skill 的验收止于构建、安装和扩展加载，不运行 NPU 计算、精度或性能用例。远程操作只使用用户指定且已授权的环境，不根据历史连接记录自行选择服务器。

## 选择构建流程

在已加载 CANN、可使用 PyTorch NPU 的 Python 环境中，优先运行：

```bash
bash experiments/megamoe/scripts/build.sh
```

脚本准备依赖，将本次 wheel 写入 `dist/build-*/` 并安装到当前 Python。脚本默认启用 MegaMoE 及设备 Timer 编译；普通调用不采集打点。需要关闭 Timer 编译时，运行 `DEEPEP_MEGAMOE_TIMER=OFF bash experiments/megamoe/scripts/build.sh`。直接用根构建系统打包时，Timer 默认关闭。

第 1 节核对环境，第 4 节验证安装。需手动准备 SDK 或单独打 wheel 时使用第 2、3 节。

## 1. 确认基础环境与 Python

当前 README 要求：

| 项目 | 要求 |
| --- | --- |
| 系统与硬件 | Linux、Ascend950 |
| 昇腾软件 | CANN 9.2.0、配套驱动和固件；CANN 中包含 Ascend C/Bisheng |
| Python | 3.11 或更高 |
| 框架 | 与 CANN、Python 配套的 PyTorch 和 torch_npu |
| 构建工具 | Git、C++17 编译器、CMake 3.20 或更高、Make 或 Ninja |

默认使用容器默认 Python，除非用户明确选择其他解释器。不自行创建或激活 venv/conda。先核对实际解释器与 pip；已处于其他环境时，先对齐用户选择，不把终端提示符当作默认 Python 的证明。

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip --version
cmake --version
```

加载用户实际安装的 CANN 环境，再检查框架。下列 CANN 路径须替换为本机路径，已正确加载时可以复用：

```bash
source /path/to/cann/set_env.sh
python -c "import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__)"
```

README 以这些基础组件已就绪为前提，没有提供从空白系统安装驱动、CANN 和框架的完整流程。缺少时说明缺失项及版本要求，结合用户提供的配套环境或安装材料处理；不要用安装 Python 构建工具代替这一步。

## 2. 安装构建工具并准备依赖

所有 Python 命令使用上一步确认的同一个解释器：

```bash
python -m pip install -r requirements-build.txt
git submodule update --init third-party/shmem third-party/catlass
export CMAKE_BUILD_PARALLEL_LEVEL=2
```

`requirements-build.txt` 包含 `build`、`setuptools`、`wheel` 和 `pybind11`。它不安装完整的 CANN 或 PyTorch 环境。遇到 `No module named build`，先确认 Python 路径，再给该解释器安装这份依赖。

子模块使用仓库记录的提交，不执行跟随远端最新分支的更新。CATLASS 是编译依赖，不需要单独编译成库。

首次构建仓内 SHMEM SDK：

```bash
python scripts/build_shmem.py
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export CATLASS_ROOT="$PWD/third-party/catlass"
```

已有与当前锁文件匹配的 SDK 时，设置其实际 `SHMEM_ROOT` 并复用，跳过 SHMEM 构建；CATLASS 也可通过 `CATLASS_ROOT` 指向固定版本源码。准备脚本拒绝覆盖已有构建时，不把 `--rebuild` 当作默认重试步骤。

`SHMEM_ROOT` 只选择构建 SDK。完整 wheel 将其运行库放入 `deep_ep/.libs`，并记录哈希；安装后根加载器校验并使用包内运行库，不从外部 SDK 回退加载。

离线环境需要完整的仓库、子模块及构建依赖；普通源码 ZIP 不能替代 SHMEM 构建所需的 Git 子模块身份。使用用户提供的离线 wheel 目录安装 Python 构建工具：

```bash
python -m pip install --no-index --find-links=/path/to/offline-wheels -r requirements-build.txt
```

SHMEM 自身的离线源码依赖或已经构建好的 SDK 也须提前准备；仅有 Python wheel 不能完成全部离线构建。

## 3. 手动编译并安装

以下示例构建 W4A8、W8A8 和 BF16，关闭设备 Timer。采集打点时将 `DEEPEP_MEGAMOE_TIMER` 改为 `ON`。

```bash
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=OFF
export DEEPEP_BUILD_EP_GMM_FUSED=OFF

python -m build --wheel --no-isolation
```

编译命令把安装包生成到仓库的 `dist/`，不会安装。`--no-isolation` 使用当前 Python 已安装的构建依赖。保留 `build/` 可复用增量构建；日常不新建虚拟环境或时间戳产物目录。

只在本次构建成功后安装，不在编译失败后继续安装 `dist/` 中的遗留包。确认本次生成的实际 wheel 文件。只有目录中仅有本次目标 wheel 时才使用下面的通配符；有多个 wheel 时改为指定实际文件名，不同时安装不同版本或不同 Python 的包：

```bash
python -m pip install --force-reinstall --no-deps dist/*.whl
```

安装包名为 `ascend-deepep`，模块名为 `deep_ep_experimental.megamoe`。同版本源码重编后也要将新 wheel 安装到实际运行测试的 Python。

ATK 和内部精度插件不参与算子编译，不作为构建依赖安装。需要精度测试时使用 [ATK 测试 skill](../test-megamoe-train-atk/SKILL.md)，在同一个 Python 中准备测试依赖并运行。

## 修改后增量构建

先查看用户改动的文件，按实际变化选择工作：

| 改动 | 需要执行的操作 |
| --- | --- |
| Host、C++ 绑定、kernel 源码或头文件 | 重新构建 wheel 并安装，复用匹配的 SDK |
| 包内 Python 接口 | 重新打包安装，原生目标是否重编由构建依赖判断 |
| 仅测试、文档或 skill | 更新文件即可，不重新编译安装算子 |
| SHMEM 锁定版本或 SDK 内容 | 准备匹配的新 SDK，再构建安装 wheel |
| CATLASS 锁定版本 | 获取对应提交的头文件源码，再构建安装 wheel；不单独编译 CATLASS |
| Python 或框架环境 | 给该环境准备构建依赖，重新构建安装 wheel，核对 SDK 配套关系 |

手动构建时，确认 Python、CANN、`SHMEM_ROOT`、`CATLASS_ROOT` 和第 3 节的开关仍有效，随后执行：

```bash
python -m build --wheel --no-isolation &&
python -m pip install --force-reinstall --no-deps dist/*.whl
```

通配符同样只适用于 `dist/` 中仅有本次目标 wheel 的情况。保持源码目录、`build/` 和 `dist/`，不默认清缓存或从头编译全部依赖。构建系统根据源文件和头文件依赖决定需要重编、重新链接的目标。

换了终端或容器后，先恢复上次实际使用的环境。SDK 可以位于当前 checkout 之外，不要将 `$PWD/third-party/...` 当作已经存在的 SDK。下面的路径替换为用户已有 SDK 和 CATLASS 源码路径，再恢复第 3 节的构建开关：

```bash
source /path/to/cann/set_env.sh
export SHMEM_ROOT=/path/to/existing/shmem-sdk
export CATLASS_ROOT=/path/to/existing/catlass
```

仅修改 MegaMoE 算子代码时，不重新初始化子模块、不重编 SHMEM、不新建 Python 环境。只有依赖缺失或版本发生变化时才处理对应依赖。安装完成后使用新 Python 进程检查扩展；已经运行的 Python 进程仍可能持有旧的原生库。

## 4. 检查安装与原生扩展加载

先查看安装版本和位置：

```bash
python -m pip show ascend-deepep
```

保持同一 Python 和匹配 CANN 环境，从源码目录外启动新进程检查安装产物，避免源码遮蔽已安装的扩展。运行时不需要 `SHMEM_ROOT` 或外部 SHMEM 的 `LD_LIBRARY_PATH`；若进程已经加载另一套 SHMEM，完整重启自己的进程，不替换包内运行库或修改哈希来绕过检查：

```bash
(
  cd /tmp
  python - <<'PY'
import json
from pathlib import Path
import sys

import torch
import torch_npu
import deep_ep
from deep_ep._native import require_native

core_native = require_native("MegaMoE installation check")
from deep_ep_experimental.megamoe import SymmBuffer, bf16_mega_moe, fp8_fp4_mega_moe
from deep_ep_experimental.megamoe import _C as megamoe_native

info = json.loads((Path(deep_ep.__file__).parent / "_build_info.json").read_text())
assert info["megamoe"]["enabled"]
print("Python:", sys.executable)
print("Core extension:", core_native.__file__)
print("MegaMoE extension:", megamoe_native.__file__)
print("BF16 API:", bf16_mega_moe.__name__)
assert megamoe_native.bf16_enabled
assert callable(fp8_fp4_mega_moe) and callable(bf16_mega_moe)
assert callable(megamoe_native.BF16Plan.forward) and callable(megamoe_native.BF16Plan.backward)
print("PASS: installed MegaMoE extensions load")
PY
)
```

若 `PYTHONPATH` 显式包含源码路径，先处理遮蔽问题，保留 CANN 所需路径。只导入扩展，不构造算子或分配 NPU Tensor。安装成功或加载成功均不代表精度、性能通过。

## 交付

向用户给出实际源码提交、Python 路径、关键依赖版本、wheel 路径、安装位置和原生扩展加载结果。构建失败时保留原始错误，区分缺构建工具、SDK/版本不匹配、编译失败和加载失败。构建未执行或加载未验证时明确说明，不报告为成功。
