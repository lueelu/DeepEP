# 在 VS Code 中阅读源码导览

安装这个小扩展后，图解里的“打开源码”会直接在旁边的编辑器中打开对应文件，并定位到源码锚点所在行。页面继续使用仓库中的同一份 `index.html` 和 `source-map.json`，无需另起服务。

## 一次性安装

1. 获取本目录扩展打包生成的 `megamoe-train-source-guide-0.2.0.vsix`。
2. 在 VS Code 扩展面板菜单中选择 **Install from VSIX…（从 VSIX 安装）**，选中该文件。
3. 在 VS Code 打开 DeepEP 仓库，运行命令 **MegaMoE: 打开训练源码导览**。

在 `visual` 目录中本地打包安装，无需在线站点：

```bash
python tools/package-editor.py
code --install-extension .artifacts/megamoe-train-source-guide-0.2.0.vsix
```

以后直接打开 `experiments/megamoe_train/docs/visual/index.html`，即可使用源码导览。已经在内置浏览器中打开的旧标签可关闭，再运行上述命令；也可使用 **Reopen Editor With…（重新打开编辑器的方式）→ MegaMoE 训练源码导览**。

使用 Remote SSH 时，将扩展安装到对应远程工作区，源码会在同一远程工作区中打开。扩展使用仓库文件，不需要 NPU、CANN 或 npm 运行时依赖。

## 阅读与更新

- 点击节点、阶段或等待查看讲解；点击 **打开源码** 或 **查看对应源码**，才在旁边打开对应文件和行号。节点有多个来源时，可分别打开。**在线源码**保留提交快照链接。
- 点击页脚 **源码映射**，可直接在编辑器查看映射文件。
- 定位基于当前编辑器中的锚点，未保存的注释导致行号移动时也可定位。函数或锚点改名后，重新生成导览并重新打开页面。
- 多个仓库同时打开时，命令优先当前工作区；发现多份导览时会让你选择。

扩展仅提供导览和源码定位，不修改训练代码。普通浏览器仍可直接阅读离线 `index.html`。
