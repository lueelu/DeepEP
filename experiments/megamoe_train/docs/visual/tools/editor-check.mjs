// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

// Exercise the real extension's message bridge against current repository files.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import vm from 'node:vm';
import { setTimeout as delay } from 'node:timers/promises';

const visual = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(visual, '../../../..');
const extensionPath = path.join(visual, 'vscode/extension.cjs');
const indexPath = path.join(visual, 'index.html');
const mappingPath = path.join(visual, 'source-map.json');
const mapping = JSON.parse(await readFile(mappingPath, 'utf8'));
const packageManifest = JSON.parse(await readFile(path.join(visual, 'vscode/package.json'), 'utf8'));
assert.equal(packageManifest.version, '0.2.0');
assert.equal(packageManifest.contributes.customEditors[0].viewType, 'megamoeTrain.sourceGuide');
const [node] = mapping.nodes.filter(entry => entry.sources?.length);
assert(node, 'The current map must contain a source-backed node');
const source = node.sources[0];
const sourcePath = path.resolve(root, source.path);
const sourceText = await readFile(sourcePath, 'utf8');
assert(sourceText.split(/\r\n|\r|\n/)[source.line - 1].includes(source.anchor));

class Uri {
  constructor(url) { this.url = new URL(url); }
  static file(value) { return new Uri(pathToFileURL(value)); }
  static joinPath(base, ...parts) {
    const url = new URL(base.url);
    url.pathname = path.posix.join(base.path, ...parts);
    return new Uri(url);
  }
  get path() { return decodeURIComponent(this.url.pathname); }
  get fsPath() { return fileURLToPath(this.url); }
  get scheme() { return this.url.protocol.slice(0, -1); }
  get authority() { return this.url.host; }
  with(change) {
    const uri = new Uri(this.url);
    if ('query' in change) uri.url.search = change.query;
    if ('fragment' in change) uri.url.hash = change.fragment;
    return uri;
  }
}
class Range {
  constructor(line, character, endLine, endCharacter) {
    this.start = { line, character }; this.end = { line: endLine, character: endCharacter };
  }
}
let provider, receive, command;
let opening = 0, maxConcurrentOpens = 0;
const opened = [], shown = [], revealed = [], errors = [], executed = [];
const documents = new Map();
const host = {
  Uri, Range, ViewColumn: { Beside: -2, Active: -1 }, TextEditorRevealType: { InCenter: 0 },
  workspace: {
    fs: { readFile: uri => readFile(uri.fsPath) },
    async openTextDocument(uri) {
      ++opening; maxConcurrentOpens = Math.max(maxConcurrentOpens, opening);
      try {
        await delay(5);
        opened.push(uri.fsPath);
        if (!documents.has(uri.fsPath)) {
          const document = { uri, text: await readFile(uri.fsPath, 'utf8'), getText() { return this.text; },
            get lineCount() { return this.text.split(/\r\n|\r|\n/).length; } };
          documents.set(uri.fsPath, document);
        }
        return documents.get(uri.fsPath);
      } finally {
        --opening;
      }
    },
  },
  window: {
    activeTextEditor: { document: { uri: Uri.file(indexPath) } },
    registerCustomEditorProvider(type, value) { assert.equal(type, 'megamoeTrain.sourceGuide'); provider = value; return { dispose() {} }; },
    async showTextDocument(document, options) {
      shown.push({ document, options });
      host.window.activeTextEditor = { document, viewColumn: options.viewColumn };
      return { revealRange(range, type) { revealed.push({ range, type }); } };
    },
    showErrorMessage(message) { errors.push(message); },
  },
  commands: {
    registerCommand(id, handler) { assert.equal(id, 'megamoeTrain.openSourceGuide'); command = handler; return { dispose() {} }; },
    async executeCommand(...args) { executed.push(args); },
  },
};
const module = { exports: {} }, require = createRequire(extensionPath);
vm.runInNewContext(await readFile(extensionPath, 'utf8'), {
  module, exports: module.exports, Buffer, require: id => id === 'vscode' ? host : require(id),
}, { filename: extensionPath });
module.exports.activate({ subscriptions: [] });
assert(provider && command, 'activate must register the editor provider and command');
const page = { html: '', options: {}, onDidReceiveMessage(handler) { receive = handler; return { dispose() {} }; } };
const panel = { viewColumn: 1, webview: page, onDidDispose() {} };
await provider.resolveCustomEditor(provider.openCustomDocument(Uri.file(indexPath)), panel, { isCancellationRequested: false });
const nonce = page.html.match(/script-src 'nonce-([^']+)'/)?.[1];
assert(nonce && page.html.includes('http-equiv="Content-Security-Policy"'));
assert(page.html.includes("default-src 'none'"));
const scripts = [...page.html.matchAll(/<script\b[^>]*>/g)];
assert(scripts.length > 0 && scripts.every(match => match[0].includes(`nonce="${nonce}"`)));
assert.equal(page.options.enableScripts, true); assert.equal(page.options.localResourceRoots.length, 0);
await receive({ type: 'openSource', nodeId: node.id, sourceIndex: 0 });
assert.equal(opened.at(-1), sourcePath);
assert.equal(shown.at(-1).options.selection.start.line, source.line - 1);
documents.get(sourcePath).text = '// Unsaved editor-only prefix\n' + sourceText;
await receive({ type: 'openSource', nodeId: node.id, sourceIndex: 0 });
assert.equal(shown.at(-1).options.selection.start.line, source.line);
// Consecutive messages overlap at the caller but are serialized by the real
// bridge, and stay beside the guide after source editors have taken focus.
maxConcurrentOpens = 0;
const beforeQueued = shown.length;
await Promise.all([
  receive({ type: 'openSource', nodeId: node.id, sourceIndex: 0 }),
  receive({ type: 'openSource', nodeId: node.id, sourceIndex: 0 }),
]);
assert.equal(maxConcurrentOpens, 1, 'Native source opens must be serialized');
assert.equal(shown.length, beforeQueued + 2);
assert(shown.slice(beforeQueued).every(item => item.options.viewColumn === 2), 'Rapid clicks reuse the same guide neighbor');
const openCount = opened.length;
await receive({ type: 'openSource', nodeId: '../../outside', sourceIndex: 0 });
await receive({ type: 'openSource', nodeId: node.id, sourceIndex: -1 });
assert.equal(opened.length, openCount); assert.equal(errors.length, 2);
// A rejected message must not poison the queue for the next valid source.
await Promise.all([
  receive({ type: 'openSource', nodeId: node.id, sourceIndex: 999999 }),
  receive({ type: 'openSource', nodeId: node.id, sourceIndex: 0 }),
]);
assert.equal(errors.length, 3);
assert.equal(opened.length, openCount + 1);
assert.equal(shown.at(-1).document.uri.fsPath, sourcePath);
assert.equal(shown.at(-1).options.selection.start.line, source.line);
await receive({ type: 'openMapping', path: '../../outside.json' });
assert.equal(opened.at(-1), mappingPath); assert.equal(shown.at(-1).options.selection.start.line, 0);
for (const [index, item] of shown.entries()) {
  assert.equal(item.options.viewColumn, 2); assert.equal(item.options.preview, false);
  assert.equal(item.options.selection.start.character, 0); assert.equal(item.options.selection.end.character, 0);
  assert.equal(item.options.selection.start.line, item.options.selection.end.line);
  assert.equal(revealed[index].range, item.options.selection); assert.equal(revealed[index].type, host.TextEditorRevealType.InCenter);
}
// Moving the guide also moves its intended neighbor; focus in the old source
// editor must not decide where the next source document is shown.
panel.viewColumn = 2;
await receive({ type: 'openSource', nodeId: node.id, sourceIndex: 0 });
assert.equal(shown.at(-1).options.viewColumn, 3);
host.window.activeTextEditor = { document: { uri: Uri.file(indexPath) } };
await command();
assert.equal(executed[0][0], 'vscode.openWith'); assert.equal(executed[0][1].fsPath, indexPath);
assert.equal(executed[0][2], 'megamoeTrain.sourceGuide'); assert.equal(executed[0][3].preview, false);
console.log('PASS: real editor bridge, nonce CSP, live source/unsaved offsets, serialized jumps, fixed guide neighbor, bad-message recovery and mapping');
