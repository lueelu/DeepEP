<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 开发指南

本文面向代码和文档贡献者。模块职责见[架构说明](architecture.md)，公共调用约束见 [API 指南](api.md)与 [Elastic 接口](elastic.md)。

## 定位与修改实现

从 `deep_ep/buffers/elastic.py` 的公开方法进入，经 `csrc/bindings/module.cpp` 定位 `csrc/ops` Host 封装和 `csrc/kernels` 实现。资源由 Python Runtime 与 `csrc/runtime/session` 管理，事件位于 `deep_ep/utils/event.py`。

新增或修改算子时，同步检查 Python 参数、Host 校验、tiling/工作区、kernel 参数、绑定、构建清单及能力声明。核心 EP 的 Device 源文件登记在 `cmake/device/CMakeLists.txt`，Host 源文件和绑定由根工程编译；不需要为每个算子建立独立 Python 构建器。

单算子的辅助实现放在邻接目录；同族共享代码放在算法族内，只有存在共同调用方和契约时才提取跨算子组件。沿用现有模块和头文件约定，不为目录完整性建立空模块。公共参数使用通信域、数据和资源含义命名，具体硬件机制留在后端实现中。

MegaMoE 的融合计算、量化和完整前向验证集中到其[独立说明](../experiments/megamoe/README.md)，不套用单独 EP 通信的性能口径。EpGmmFused 的可选构建目标和算子验证见[组件说明](../experiments/ep_gmm_fused/README.md#开发与构建接入)。

## 变更验证

接口变化同时更新机器可读契约、使用文档和相应测试。资源/事件变化检查创建、失败、等待和销毁；kernel 变化检查参数、有效区域、容量边界及数值。各入口和命令见[测试说明](../tests/README.md)。

文档或注释修改执行链接、路径、格式和示例参数检查。只有描述或行为涉及设备时才需要相应设备验证；没有运行的项目在变更说明中如实记录。

## 文档与图稿维护

面向使用者写已明确的行为、约束、命令和结果解释；面向开发者写源码职责与验证方法。保持中英文首页的含义、链接及图表对应，避免复制多份详细支持表。

图稿放在 `figures/`，使用自包含 SVG，不依赖外部字体文件或网络资源。中英文架构图分别为 `architecture.svg` 和 `architecture_en.svg`。更新图稿时同步检查引用、文字与实际职责；性能图的数值、单位、测试配置及统计口径与表格一致。

检查 Markdown 本地链接及锚点、代码路径、UTF-8/LF、SVG XML 和 `git diff --check`。删除或合并文档时，也更新源码 docstring、错误信息和配置中的路径引用。

## 提交前检查与 CI

[`.pre-commit-config.yaml`](../.pre-commit-config.yaml) 是本地与 CI 共用的检查入口。开发检查工具版本固定在 `requirements-dev.txt`，hook 固定到提交 SHA。使用 Python 3.11 按[贡献指南](../CONTRIBUTING.md)安装开发依赖并运行 `pre-commit install` 后，每次提交会检查暂存文件。

| 检查 | 范围与约定 |
| --- | --- |
| `trailing-whitespace` | 清理行尾空白，保留 Markdown 的双空格换行 |
| `end-of-file-fixer` | 确保非空文本文件以一个换行符结束 |
| `check-yaml` | 检查 YAML 语法和重复键 |
| `check-json` | 检查 JSON 语法和重复键 |
| `check-added-large-files` | 拦截超过 1 MiB 的普通 Git 文件；Git LFS 文件由工具排除 |
| `check-merge-conflict` | 检查遗留的合并冲突标记 |
| `detect-private-key` | 检查常见私钥文件标记 |
| `ruff-check` | 检查 Python 导入、语句、语法、未使用导入和未定义名称等常规问题，规则见 [`.ruff.toml`](../.ruff.toml) |
| `ruff-format` | 统一 Python 缩进、空格、换行和引号；使用空格缩进、双引号和 LF 换行，目标行宽 120 列 |
| clang-format | 第一方 C/C++/AscendC，包括 `.asc`；采用 [`.clang-format`](../.clang-format) 中的四空格、120 列等规则 |
| codespell | 文档、代码及文本配置；`AGRS` 和 `CANN` 为有效术语，JSON/CSV 数据及 SVG 图稿不做拼写改写 |

第三方源码不运行 Ruff、clang-format 和 codespell；`third-party/README.md` 仍检查拼写。基础检查没有全局排除第三方目录，也不整体跳过 SVG/CSV 文件。私钥标记检查不能替代现有的依赖与安全扫描。

大小检查使用 `--enforce-all`，冲突标记检查使用 `--assume-in-merge`，确保在 CI 的干净 checkout 中也检查已提交文件，不依赖暂存区或正在进行的 merge 状态。

```bash
# 手动检查暂存文件。
pre-commit run --show-diff-on-failure

# 首次接入或修改规则后检查全仓。
pre-commit run --all-files --show-diff-on-failure

# 提交后检查本次 PR 相对目标分支的改动。
git fetch upstream main
PRE_COMMIT_BASE="$(git merge-base upstream/main HEAD)"
pre-commit run --show-diff-on-failure --from-ref "$PRE_COMMIT_BASE" --to-ref HEAD
```

行尾修复、ruff-format 和 clang-format 可能改写文件。检查差异、重新暂存并提交后，重复检查直到退出码为 0 且没有新的改写。首次运行会从 GitCode 镜像获取 hook，并从 Python 包源安装各自的隔离环境；下载失败同样会使检查失败。更新版本时需同步核对版本注释并执行全仓检查。

[GitCode 流水线](../.gitcode/workflows/PR-pipeline_deepep.yml)沿用现有 PR 评论 `compile` 触发方式。`CodeCheck_pre_commit` 检查 PR 的合并结果，使用固定开发依赖运行 `python3.11 -m pre_commit run --all-files --show-diff-on-failure`，不受 `need_build_ut` 条件限制，因此纯文档 PR 也执行。失败直接使作业失败，流水线不会自动提交修改。

CPU 正确性、benchmark dry-run、SCA 和恶意代码扫描保持各自的作业与执行条件。pre-commit 不启动 NPU，也不代替这些验证。
