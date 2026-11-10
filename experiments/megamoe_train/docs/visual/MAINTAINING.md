# 维护源码导览

本目录随 MegaMoE BF16 训练代码一起维护。图中的文字、节点和连线描述源码行为；生成工具负责更新源码片段、位置与离线页面。默认阅读场景为 `save=bf16`、clipping 关闭。

生成产物对应的精确源码提交见 [README 的源码快照](README.md)及 `source-map.json` 的
`sourceCommit`。构建工具从 `SOURCE_REF` 指定的真实 Git 提交读取源码，未指定时解析当前
Git 提交；写入产物的是完整 SHA，不是字面量 `HEAD`，也不代表未提交工作区的内容。

预览尚未提交的源码修改时，设置 `SOURCE_WORKTREE=1` 后构建。此时摘录、行号与
本地跳转读取当前工作区，页面标记为“本地源码”；在线链接仍定位到基准提交中的
对应锚点。源码检查会核对本地文件，行号或内容变化后需要重新构建。

默认构建离线导览，不根据仓库 owner 推导在线站点。生成来源可用以下环境变量明确设置：

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `SOURCE_REF` | 读取源码的真实 Git 提交或引用 | 当前提交 |
| `SOURCE_WORKTREE` | 设为 `1` 时读取当前工作区，基准提交须为 HEAD | 空，读取已提交源码 |
| `SOURCE_REPOSITORY_URL` | 源码链接使用的 HTTPS 仓库地址 | CI 的 `GITHUB_REPOSITORY`，本地为 origin |
| `SOURCE_BRANCH` | 映射文件中记录的来源分支 | CI 分支、当前分支；分离 HEAD 时记录提交 |
| `SITE_URL` | 显式启用在线导览及站点 VSIX 下载入口 | 空，保持离线 |

## 本地修改与检查

在仓库根目录进入本目录，使用 Node.js 22 或更新版本：

```bash
cd experiments/megamoe_train/docs/visual
npm ci
npm run build
npm run check
npm run test:editor
npx playwright install chromium
npm run test:browser
```

PowerShell 中预览未提交的修改：

```powershell
$env:SOURCE_WORKTREE = '1'
npm run build
Remove-Item Env:SOURCE_WORKTREE
npm run check
```

正式生成时使用默认提交模式；本地预览不会提交代码。

阅读时直接打开 [index.html](index.html)，无需启动服务。仓库网页入口是 [README.md](README.md)，保留图文说明和源码链接。

在 VS Code 中推荐使用[源码导览扩展](vscode/README.md)。安装一次后，运行 **MegaMoE: 打开训练源码导览**。点击节点、阶段或等待只查看讲解；点击 **打开源码** 或 **查看对应源码**，再在旁边的编辑器中定位到文件和行号。扩展直接读取当前工作区的页面与映射。

本地构建安装包的命令：

```bash
python tools/package-editor.py
code --install-extension .artifacts/megamoe-train-source-guide-0.2.0.vsix
```

普通浏览器继续直接打开离线 HTML，并提供 **复制定位路径**。使用 VS Code 自带内置浏览器时，也可以复制后按 **Ctrl+P** 粘贴并回车。这里与扩展提供的源码导览是两个不同的打开方式。

修改 `data/*.json` 中对应节点的标题、讲解、输入、输出、等待和连线。`sources` 使用文件路径与源码锚点定位函数或控制点；函数移动或改名时同步修改锚点。节点 ID 保持稳定，便于页面链接和跨图下钻。

三个阅读场景由 `data/learning-case.json` 与 `tools/learning-scenes.mjs` 生成。默认 EP8、每源 M4096、TopK6、每 rank 4 个专家，流水总览固定看 rank0，按专家展示。`assets/full-pipeline.js` 维护前反向总览，`assets/gmm2-pipeline.js` 维护独立双槽练习。输入规则、聚合计数和所选 token 的数据共同驱动路由图、保存桥和流水示意；修改案例后同时检查路由守恒和数值结果。时间轴 tick 表示可调的教学耗时，实测时间及性能记录单独维护。

H、dH 和 dp 乘积保持 FP32；Hp、dGate/dUp 写出时转回 BF16，Y、dX 在 FP32 归约后最终舍入 BF16。dp 的最终类型与输入 p 一致：BF16 舍入、FP32 保留；固定教学数值仍使用 BF16 p。修改数值边界时，同时更新 L5 节点、教学计算和表格；流水详情从 L5 读取同一份公式。

[数值契约](../mega_moe_bf16_design.md#13-数学关系与舍入边界)也用于 NPU BigOp 与契约参考，CPU FP64 真值独立计算。修改测试参考时，核对导览中的公式和源码引用；原始设备报告与临时验证脚本继续存放在仓库外。

浏览器检查覆盖仓内 HTTP 预览、`file://` 离线页面、VS Code 消息桥，以及 `site/` 便携副本的 HTTP 模式。`site/` 是本地打包目录，不表示已发布站点；它使用同一份页面和源码映射。

训练代码更新后：

1. 同时维护训练代码与本目录文档。
2. 核对受影响节点，更新相应的执行流程、张量和同步说明。
3. 运行构建，重新采集源码片段、位置和版本信息。
4. 完成源码引用与浏览器检查，提交修改后的文档、图源和生成文件。

浏览器检查的截图和报告写入 `.artifacts/`；`node_modules/`、`.artifacts/` 和便携副本目录 `site/` 是本地生成目录。离线阅读和文档构建不需要 NPU 或 CANN。

修改编辑器接入时，运行 `npm run test:editor` 检查消息、源码路径和当前文档锚点定位，再执行浏览器检查。`python tools/package-editor.py --site` 同时把 VSIX 放入便携副本的 `downloads/` 目录。扩展代码更新后需重新打包安装；只更新训练代码、图或讲解时无需重装扩展，重新打开导览即可。

## 自动检查与离线附件

工作流位于仓库根目录的 `.github/workflows/megamoe-train-visual.yml`。相关推送及 PR 自动执行依赖安装、离线生成、源码检查、编辑器检查、浏览器检查和扩展打包，也可从 Actions 页手动运行。公共工作流不部署 Pages，也不需要 Pages 写入权限。

检查通过后，Actions 附件 `megamoe-train-visual-offline` 提供页面、源码映射、图和 VS Code 安装包。下载解压后可直接打开 `index.html`，也可运行 `python -m http.server 8000` 后从 `http://localhost:8000` 预览。检查失败时，Actions 保存 `.artifacts/` 中已有的报告和截图。

## 可选的分支站点

需要在线导览的分支可以另行维护 Pages 部署工作流。该工作流显式设置 `SITE_URL`、`SOURCE_REPOSITORY_URL` 和来源分支，复用相同生成、检查和打包命令；将仓库、发布分支和 `github-pages` 环境的部署范围限定到实际维护的站点。首次启用时由仓库维护者在 **Settings → Pages → Source** 选择 **GitHub Actions**。

离线交付不需要携带分支专用的 Pages 工作流，也不依赖在线站点或下载地址。

源码行为变化时，需要一起审阅图的讲解和连线；重新生成会刷新引用内容，执行语义由代码与文档 review 对齐。
