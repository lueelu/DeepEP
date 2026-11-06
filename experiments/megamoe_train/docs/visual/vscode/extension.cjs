// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

'use strict';

const vscode = require('vscode');
const { randomBytes } = require('node:crypto');

const VIEW_TYPE = 'megamoeTrain.sourceGuide';
const COMMAND = 'megamoeTrain.openSourceGuide';
const INDEX_GLOB = '**/experiments/megamoe_train/docs/visual/index.html';
const INDEX_SUFFIX = '/experiments/megamoe_train/docs/visual/index.html';
const decodeText = bytes => Buffer.from(bytes).toString('utf8');

function repositoryRoot(indexUri) {
  if (!indexUri.path.endsWith(INDEX_SUFFIX)) {
    throw new Error('请打开 experiments/megamoe_train/docs/visual/index.html。');
  }
  return vscode.Uri.joinPath(indexUri, '../../../../..').with({ query: '', fragment: '' });
}

function sourceUri(root, sourcePath) {
  if (typeof sourcePath !== 'string' || !sourcePath || sourcePath.includes('\\') || sourcePath.includes('\0') ||
      sourcePath.startsWith('/') || /^[A-Za-z][A-Za-z\d+.-]*:/.test(sourcePath) ||
      sourcePath.split('/').some(part => !part || part === '.' || part === '..')) {
    throw new Error('源码映射路径无效，请重新生成导览。');
  }
  const uri = vscode.Uri.joinPath(root, ...sourcePath.split('/'));
  if (!uri.path.startsWith(root.path.replace(/\/$/, '') + '/')) {
    throw new Error('源码映射必须位于当前仓库内。');
  }
  return uri;
}

function sourceReferences(manifest, root) {
  if (!Array.isArray(manifest.nodes) || !manifest.files || typeof manifest.files !== 'object') {
    throw new Error('source-map.json 内容不完整，请重新生成导览。');
  }
  const references = new Map();
  for (const node of manifest.nodes) {
    if (typeof node.id !== 'string' || references.has(node.id) || !Array.isArray(node.sources)) {
      throw new Error('源码节点映射无效，请重新生成导览。');
    }
    references.set(node.id, node.sources.map(source => {
      const occurrence = source.occurrence ?? 1;
      if (!Object.hasOwn(manifest.files, source.path) || typeof source.anchor !== 'string' || !source.anchor ||
          !Number.isInteger(occurrence) || occurrence < 1) {
        throw new Error('源码锚点映射无效，请重新生成导览。');
      }
      return { uri: sourceUri(root, source.path), anchor: source.anchor, occurrence };
    }));
  }
  return references;
}

function anchorLine(document, source) {
  const lines = document.getText().split(/\r\n|\r|\n/);
  let occurrence = 0;
  for (let line = 0; line < lines.length; ++line) {
    if (lines[line].includes(source.anchor) && ++occurrence === source.occurrence) return line;
  }
  throw new Error('当前源码中找不到对应锚点，请重新生成导览后打开。');
}

async function showSource(panel, document, line) {
  const line0 = Math.max(0, Math.min(line, document.lineCount - 1));
  const range = new vscode.Range(line0, 0, line0, 0);
  // Anchor navigation to the guide's column. The previously opened source
  // takes focus, so relative Beside on every click can create a third group.
  const column = panel.viewColumn;
  const viewColumn = Number.isInteger(column) && column >= 1 ? Math.min(column + 1, 9) : vscode.ViewColumn.Beside;
  const editor = await vscode.window.showTextDocument(document, {
    viewColumn,
    selection: range,
    preview: false,
  });
  editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
}

function readerHtml(html) {
  const nonce = randomBytes(18).toString('base64');
  const policy = `default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';`;
  if (!/<head\b[^>]*>/i.test(html)) throw new Error('导览页面不完整，请重新生成 index.html。');
  return html
    .replace(/<head\b[^>]*>/i, head => `${head}\n<meta http-equiv="Content-Security-Policy" content="${policy}">`)
    .replace(/<script\b/g, () => `<script nonce="${nonce}"`);
}

async function openGuide() {
  const activeUri = vscode.window.activeTextEditor?.document.uri ||
    vscode.window.tabGroups.activeTabGroup.activeTab?.input?.uri;
  let candidates;
  if (activeUri?.path.endsWith(INDEX_SUFFIX)) {
    candidates = [activeUri];
  } else {
    const folder = activeUri && vscode.workspace.getWorkspaceFolder(activeUri);
    if (folder) {
      candidates = await vscode.workspace.findFiles(new vscode.RelativePattern(folder, INDEX_GLOB), '**/node_modules/**');
    }
    if (!candidates?.length) candidates = await vscode.workspace.findFiles(INDEX_GLOB, '**/node_modules/**');
  }
  if (!candidates.length) throw new Error('请先在 VS Code 打开包含训练源码导览的 DeepEP 仓库。');
  let uri = candidates[0];
  if (candidates.length > 1) {
    const choice = await vscode.window.showQuickPick(candidates.map(candidate => ({
      label: vscode.workspace.asRelativePath(candidate, true),
      description: candidate.scheme === 'file' ? candidate.fsPath : `${candidate.authority}${candidate.path}`,
      uri: candidate,
    })), { placeHolder: '选择要阅读的 MegaMoE 训练源码仓库' });
    if (!choice) return;
    uri = choice.uri;
  }
  await vscode.commands.executeCommand('vscode.openWith', uri, VIEW_TYPE, {
    viewColumn: vscode.ViewColumn.Active,
    preview: false,
  });
}

function activate(context) {
  const reportError = error => vscode.window.showErrorMessage(`MegaMoE：${error.message || error}`);
  const provider = {
    openCustomDocument(uri) {
      repositoryRoot(uri);
      return { uri, dispose() {} };
    },
    async resolveCustomEditor(document, panel, token) {
      const root = repositoryRoot(document.uri);
      const visual = vscode.Uri.joinPath(document.uri, '..');
      const mappingUri = vscode.Uri.joinPath(visual, 'source-map.json');
      const [pageBytes, mappingBytes] = await Promise.all([
        vscode.workspace.fs.readFile(document.uri),
        vscode.workspace.fs.readFile(mappingUri),
      ]);
      if (token.isCancellationRequested) return;
      const references = sourceReferences(JSON.parse(decodeText(mappingBytes)), root);
      panel.webview.options = { enableScripts: true, localResourceRoots: [] };
      let pending = Promise.resolve(), disposed = false;
      const listener = panel.webview.onDidReceiveMessage(message => {
        pending = pending.then(async () => {
          if (disposed) return;
          if (message?.type === 'openMapping') {
            await showSource(panel, await vscode.workspace.openTextDocument(mappingUri), 0);
          } else if (message?.type === 'openSource') {
            if (typeof message.nodeId !== 'string' || !Number.isInteger(message.sourceIndex) || message.sourceIndex < 0) {
              throw new Error('源码节点选择无效，请重新打开导览。');
            }
            const source = references.get(message.nodeId)?.[message.sourceIndex];
            if (!source) throw new Error('找不到对应源码映射，请重新生成导览。');
            const sourceDocument = await vscode.workspace.openTextDocument(source.uri);
            await showSource(panel, sourceDocument, anchorLine(sourceDocument, source));
          }
        }).catch(reportError);
        return pending;
      });
      panel.onDidDispose(() => { disposed = true; listener.dispose(); });
      panel.webview.html = readerHtml(decodeText(pageBytes));
    },
  };
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(VIEW_TYPE, provider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: false,
    }),
    vscode.commands.registerCommand(COMMAND, () => openGuide().catch(reportError)),
  );
}

module.exports = { activate };
