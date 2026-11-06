# ATK 与内部 CV 比较器安装

仅在当前 Python 缺少依赖、加载失败或用户要求重建环境时读取。安装到用户选定的 Python，默认是容器默认 Python；不新建 venv，不把旧环境的 site-packages 追加到 PYTHONPATH 代替安装。

## 已验证的工具组合与来源

| 项目 | 已验证版本或来源 |
| --- | --- |
| ATK | 26.9.8，baseline，源码提交 `45cc4061bc019e73f63a11e96496e97df0d6810c` |
| 源码 | <https://gitcode.com/Ascend/ATK.git> |
| 内部插件 | `atk-private-precision==0.1.0`，插件 API 1 |
| 比较器 | `cv_fused_double_benchmark`，提供者必须为 private |
| 验证平台 | Linux aarch64、Python 3.12 |

私有包来自用户提供的[内部精度标准包发布页](https://gitcode.com/AscendTest/ATK/releases/v26.9.24-%E5%86%85%E9%83%A8%E7%B2%BE%E5%BA%A6%E6%A0%87%E5%87%86%E5%8C%85)。它是外部安装依赖，仓库不包含该二进制。使用者需要能取得匹配平台的发布包；不能取得时报告缺失，不替换成自写比较器。

已核验的 aarch64 文件为 `atk_private_precision-0.1.0-cp39-abi3-linux_aarch64.whl`，SHA256：

```text
62123de469dd1f68f3fa5f1239f8a2c8710c5d501e3af1a67439f081bf8bcae5
```

使用此文件前确认当前 Python/平台支持它的 wheel tag；其他平台须取得对应包，不能强行安装 aarch64 包。不据此承诺其他 ATK 版本已验证，也不盲目安装 PyPI 上的同名包或升级到最新主分支。

## 1. 确认当前 Python 并记录框架约束

先确认当前 Python 已有与候选算子配套的 Torch、torch_npu，并明确缺少的是 ATK，而非又切换到了其他 Python。ATK、插件和算子最终须在同一个解释器中加载。

安装依赖前记录现有框架和 NumPy 版本。下列目录由用户选择，作为一次性安装材料目录，后续复用；不放入 skill 或算子源码目录：

```bash
ATK_SETUP_DIR=/path/to/atk-setup
mkdir -p "$ATK_SETUP_DIR"
python - "$ATK_SETUP_DIR/framework-constraints.txt" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

constraints = []
for name in ("torch", "torch-npu", "numpy"):
    try:
        constraints.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        if name != "numpy":
            raise
Path(sys.argv[1]).write_text("\n".join(constraints) + "\n", encoding="utf-8")
print("\n".join(constraints))
PY
```

已验证 ATK 源码的依赖包含 `numpy<2.0.0`。若当前环境与依赖约束冲突，明确指出冲突，结合用户选定的配套环境处理，不静默降级 NumPy 或升级 Torch/torch_npu。不要通过默认 `--no-deps` 把缺依赖的安装当作完成。

## 2. 获取 ATK wheel 与内部插件

已有来源和版本可确认的 ATK wheel 时直接复用，跳过源码构建。原来安装在其他环境的 ATK，不必复制整个环境；复用其原始安装包即可。

需要从源码准备 ATK 时，在用户选择的依赖目录克隆固定提交。以下是首次获取示例；已有源码时先核对提交和修改状态，不覆盖用户修改：

```bash
ATK_SOURCE_DIR="$ATK_SETUP_DIR/ATK"
git clone https://gitcode.com/Ascend/ATK.git "$ATK_SOURCE_DIR"
git -C "$ATK_SOURCE_DIR" checkout --detach 45cc4061bc019e73f63a11e96496e97df0d6810c
(
  cd "$ATK_SOURCE_DIR"
  python setup.py bdist_wheel --flavor baseline
)
```

源码打包需要 setuptools/wheel；缺少时可安装本仓根目录 `requirements-build.txt` 中的构建工具。生成 ATK wheel 后记录实际文件名，不将 `dist/` 中多个不相容包一起安装。

从发布页取得内部 wheel，放到用户的安装材料目录。选择上面的已核验文件时检查其 SHA256：

```bash
ATK_PRIVATE_WHEEL="$ATK_SETUP_DIR/atk_private_precision-0.1.0-cp39-abi3-linux_aarch64.whl"
printf '%s  %s\n' \
  62123de469dd1f68f3fa5f1239f8a2c8710c5d501e3af1a67439f081bf8bcae5 \
  "$ATK_PRIVATE_WHEEL" | sha256sum -c -
```

其他平台的文件使用其发布页对应的摘要，不能套用这个 aarch64 文件的摘要。

## 3. 安装到默认 Python

将 `ATK_WHEEL` 替换为本次实际取得的 ATK 安装包文件，使用前面记录的约束，补齐两个包及缺失依赖：

```bash
ATK_WHEEL=/path/to/ATK-26.9.8-py3-none-any.whl
python -m pip install \
  -c "$ATK_SETUP_DIR/framework-constraints.txt" \
  "$ATK_WHEEL" "$ATK_PRIVATE_WHEEL"
```

这里的 pip 属于当前默认 Python，不是另一个虚拟环境的 pip。包版本已相容且加载正常时，不强制重装；需要明确替换已安装包时再使用相应的重装选项。若在线依赖获取失败，保留错误，使用用户指定的可用源或离线包，不修改服务器全局代理或 Shell 配置。

离线安装须提前准备两个主 wheel 及所有缺失依赖的匹配 wheel，只有两个主文件通常不够。可在相同平台和 Python 的联网环境准备材料，再使用：

```bash
ATK_WHEELHOUSE=/path/to/complete-wheelhouse
python -m pip install --no-index --find-links="$ATK_WHEELHOUSE" \
  -c "$ATK_SETUP_DIR/framework-constraints.txt" \
  "$ATK_WHEEL" "$ATK_PRIVATE_WHEEL"
```

## 4. 验收安装

```bash
python -m pip show ATK atk-private-precision
python -c "from importlib import metadata; print({n: metadata.version(n) for n in ('torch', 'torch-npu', 'numpy')})"
```

核对包位置属于用户选定的 Python，框架版本符合安装前约束。随后返回 [测试 skill](../SKILL.md)，在新 Python 进程调用仓库的 `require_atk()`，确认自动加载、比较器名称和 private 提供者。命令行 `atk --version` 或仅有安装元数据不能替代这个检查。

没有完成加载检查时不启动设备测试。环境安装和比较器加载验收不分配 NPU Tensor，也不构成算子精度通过的证据。
