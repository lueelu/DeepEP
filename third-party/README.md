# 第三方依赖与来源

集中管理依赖来源和版本边界。SHMEM、CATLASS 已登记为固定提交的 Git 子模块；不跟随上游分支自动升级。

## 已接入依赖

| 目录 | 用途 | 固定提交 |
| --- | --- | --- |
| `shmem/` | 通信 SDK，提供 shared-jetty 支持 | `7f9d961d378225e69058080532cf4f5be16cc83d` |
| `catlass/` | MegaMoE 编译时使用的模板头文件，不单独编译动态库 | `2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1` |

来源见根 [.gitmodules](../.gitmodules)，版本约束见 [dependencies.lock.json](../dependencies.lock.json)。
SHMEM 固定到 `cann/shmem` 的 dev 提交；升级后需重编 SDK 和 wheel，准备步骤见[构建指南](../docs/build.md)。
两个子仓库的许可证均为 CANN Open Software License Agreement Version 2.0，原文保留在各自的 `LICENSE` 中。
MegaMoE 源文件中的 CANN 版权和许可声明保留，许可全文归档于 [cann.LICENSE](cann.LICENSE)，并随启用 MegaMoE 的 wheel 分发。
第三方许可适用于对应来源内容，不替代项目自身的发行许可；引入依赖时遵循[贡献指南](../CONTRIBUTING.md#许可证)。

基础包按[构建说明](../docs/build.md)准备 SDK 并构建；MegaMoE 的便利入口见[实验使用说明](../experiments/megamoe/README.md)。
SHMEM 已有构建产物时，确认停止使用该 SDK 后可向准备脚本显式传 `--rebuild`。
下列分步骤入口仍保留给 SDK 准备与 CI 使用：

```bash
git submodule update --init --recursive
# 先加载 CANN 9.2.0 环境，再显式准备 SHMEM；不需要另外 clone CATLASS。
python scripts/build_shmem.py
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
```

以上命令从主仓库根目录运行。SHMEM 准备入口检查固定提交、干净源码和 CANN 版本，调用其上游构建脚本，
成功后生成 `third-party/shmem/install/shmem/deepep-sdk.json`。上游首次构建 Ascend950 会下载 nlohmann/json v3.11.3；
离线环境需预先提供 `third-party/shmem/3rdparty/json/single_include/nlohmann/json.hpp` 及对应源码。

已有 build/install 时默认停止，避免覆盖正在被其他作业使用的 SDK。明确需要重建时，先停止使用该 SDK 的作业，
再运行 `python scripts/build_shmem.py --rebuild`；上游会重新生成该子模块的 build/install，随后必须重建依赖它的 wheel。
默认拒绝修改过的源码；开发验证可显式传入 `--allow-local-changes`，脚本会记录修改指纹，并在编译后检查源码未再次变化。该选项不允许更换锁定提交；`--rebuild` 不会强制重置或覆盖源码。

未设置环境变量时，wheel 构建默认使用 `third-party/shmem/install/shmem` 与 `third-party/catlass`。
已有 SDK/源码可继续用 `SHMEM_ROOT`、`CATLASS_ROOT` 显式覆盖。`pip wheel` 本身不下载、不编译外部 SDK。
本入口按锁定版本和上游默认配置构建 SHMEM。

Python sdist/wheel 不携带这两个子模块源码或 SHMEM 动态库；从 sdist 构建时需设置上述两个外部路径（基础 OFF 构建不需要 CATLASS）。
SDK 和 CANN 由部署环境提供，完整编译命令见 [构建说明](../docs/build.md) 与 [MegaMoE README](../experiments/megamoe/README.md)。

## EpGmmFused 依赖

EpGmmFused 使用单独准备的固定版本 CatCCOS、其 CATLASS 子模块及 SHMEM 源码/SDK，不以 MegaMoE 的 CATLASS 路径替代。准备命令、源码路径和版本检查见 [EpGmmFused 构建说明](../experiments/ep_gmm_fused/README.md#构建)；overlay 来源和打包许可见 [NOTICE](../experiments/ep_gmm_fused/NOTICE.md)。

SDK 提供方声明源码版本，`scripts/record_shmem_sdk.py` 校验固定头文件并记录头文件与库的校验和；运行时核对实际加载的 SHMEM 库。头文件哈希相同不能单独证明二进制来源。依赖更新时同步更新 gitlink、锁文件和验证记录。

## API 参考来源

公开 EP 接口参考 DeepSeek DeepEP 提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8`，并采用本项目的平台适配命名。所选源文件、签名和 SHA-256 见 [API 契约](../tests/api/api_contract.json)。原始 MIT 版权与许可全文保留在 [deepseek-deepep.LICENSE](deepseek-deepep.LICENSE)。

接口参考不包含上游 GPU 通信实现，也不表示本项目支持上游全部 API。具体差异和运行约束见 [API 指南](../docs/api.md)。第三方许可适用于对应来源内容，不替代本项目自身的发行许可。

## 测试语义来源

以下固定版本源码用于理解和重新实现测试语义：

- [deep_ep/utils/refs.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/deep_ep/utils/refs.py)：普通布局的 rank 去重、索引本地化和源 token 归并。
- [tests/elastic/test_ep.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/tests/elastic/test_ep.py)：fresh/cached、展开布局、独立权重通道及不等长输入。
- [tests/legacy/test_intranode.py](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/tests/legacy/test_intranode.py)：V1 layout、dispatch/combine 和 handle 复用。

独立参考与测试辅助位于 `tests/utils/` 等目录，包含人工算例、故障注入和边界场景。这些测试不复制上游设备 collective、内部 metadata 或计时实现；参考矩阵也不是当前设备支持表。
