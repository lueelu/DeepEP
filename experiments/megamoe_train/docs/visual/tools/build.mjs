// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

// Generate the offline reader, source map and repository documentation together.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import cp from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { routeGraph, roundedPath } from './router.mjs';
import { buildLearningScenes } from './learning-scenes.mjs';
import { sourceConfig } from './source-config.mjs';

export const visual = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const root = path.resolve(visual, '../../../..');
const git = (...args) => cp.execFileSync('git', ['-C', root, ...args], { encoding: 'utf8', maxBuffer: 12 * 1024 * 1024 }).trimEnd();
const sourceCommit = git('rev-parse', process.env.SOURCE_REF || 'HEAD');
const { repository: repoUrl, sourceBranch, siteUrl } = sourceConfig(process.env, {
  remote: process.env.SOURCE_REPOSITORY_URL || process.env.GITHUB_REPOSITORY ? '' : git('remote', 'get-url', 'origin'),
  branch: git('branch', '--show-current'), commit: sourceCommit,
});
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const urlPath = p => p.split('/').map(encodeURIComponent).join('/');
const colors = {
  python: ['#e9efff', '#3e5baf'], host: ['#e4f3ee', '#27765e'], cube: ['#ffeddd', '#a95c27'],
  vector: ['#e5edff', '#4266ac'], comm: ['#e2f3f2', '#257b80'], memory: ['#f4eafa', '#82539b'], neutral: ['#edf0f4', '#596578'],
};
const readSource = new Map();
const graphFiles = fs.readdirSync(path.join(visual, 'data')).filter(x => /^\d\d-.*\.json$/.test(x)).sort();
if (graphFiles.length !== 8) throw new Error(`Expected eight graph files, found ${graphFiles.length}`);
const graphs = graphFiles.map(file => JSON.parse(fs.readFileSync(path.join(visual, 'data', file), 'utf8')));
const allIds = new Set();
const manifestNodes = [];
for (const graph of graphs) {
  if (!['L1', 'L2', 'L3', 'L4', 'L5'].includes(graph.layer)) throw new Error(`Invalid reading layer: ${graph.id}`);
  const laneIds = new Set(graph.lanes.map(l => l.id));
  const positions = new Set();
  graph.nodes.forEach((node, index) => {
    if (allIds.has(node.id)) throw new Error(`Duplicate node: ${node.id}`);
    allIds.add(node.id);
    if (!laneIds.has(node.lane) || !Number.isInteger(node.step) || node.step < 0) throw new Error(`Invalid position: ${node.id}`);
    const pos = `${node.lane}:${node.step}`;
    if (positions.has(pos)) throw new Error(`Two nodes share lane/step: ${graph.id} ${pos}`);
    positions.add(pos);
    if (!node.sources?.length) throw new Error(`Missing source: ${node.id}`);
    node.ordinal = index + 1;
    node.sources = node.sources.map(source => {
      const relative = source.path.replaceAll('\\', '/');
      if (relative.startsWith('/') || relative.split('/').includes('..')) throw new Error(`Unsafe source path: ${relative}`);
      if (!readSource.has(relative)) {
        const text = cp.execFileSync('git', ['-C', root, 'show', `${sourceCommit}:${relative}`], { encoding: 'utf8' });
        readSource.set(relative, { text, lines: text.split('\n'), sha256: crypto.createHash('sha256').update(text).digest('hex') });
      }
      const file = readSource.get(relative);
      const hits = file.lines.flatMap((line, i) => line.includes(source.anchor) ? [i + 1] : []);
      if (!hits.length || (!source.occurrence && hits.length !== 1)) throw new Error(`Ambiguous/missing anchor ${node.id}: ${source.anchor} (${hits.length})`);
      const line = hits[(source.occurrence || 1) - 1];
      if (!line) throw new Error(`Missing occurrence: ${node.id}`);
      const start = Math.max(1, line - (source.before ?? 2));
      const end = Math.min(file.lines.length, line + (source.after ?? 16));
      return { path: relative, anchor: source.anchor, occurrence: source.occurrence || 1, line, start, end,
        sha256: file.sha256, url: `${repoUrl}/blob/${sourceCommit}/${urlPath(relative)}#L${line}`,
        excerpt: file.lines.slice(start - 1, end).map((text, i) => ({ line: start + i, text })) };
    });
    manifestNodes.push({ id: node.id, graph: graph.id, symbol: node.symbol, sources: node.sources });
  });
  const ids = new Set(graph.nodes.map(n => n.id));
  for (const edge of graph.edges) if (!ids.has(edge.from) || !ids.has(edge.to)) throw new Error(`Dangling edge in ${graph.id}`);
  const laneWidth = 320, gap = 54, margin = 46;
  graph.width = margin * 2 + graph.lanes.length * laneWidth + (graph.lanes.length - 1) * gap;
  graph.height = 164 + (Math.max(...graph.nodes.map(n => n.step)) + 1) * 174;
  graph.lanes.forEach((lane, i) => Object.assign(lane, { x: margin + i * (laneWidth + gap), w: laneWidth }));
  graph.nodes.forEach(node => {
    const lane = graph.lanes.find(l => l.id === node.lane);
    Object.assign(node, { x: lane.x + 12, y: 112 + node.step * 174, w: laneWidth - 24, h: 124, color: lane.color || 'neutral' });
  });
  routeGraph(graph);
}
for (const graph of graphs) for (const node of graph.nodes) if (node.drilldown) {
  const destination = graphs.find(g => g.id === node.drilldown.view);
  if (!destination || (node.drilldown.node && !destination.nodes.some(n => n.id === node.drilldown.node))) throw new Error(`Invalid drilldown: ${node.id}`);
}
const learning = buildLearningScenes();
if (learning.scenes.length !== 3) throw new Error('Expected three learning scenes');
for (const scene of learning.scenes) {
  if (!allIds.has(scene.sourceNode)) throw new Error(`Missing scene source: ${scene.id}`);
  for (const match of scene.html.matchAll(/data-learning-node=["']([^"']+)["']/g)) {
    if (!allIds.has(match[1])) throw new Error(`Unknown learning source ${scene.id}: ${match[1]}`);
  }
}

const measure = (s, size) => [...s].reduce((sum, c) => sum + (c.codePointAt(0) > 255 ? 1 : 0.57) * size, 0);
function wrap(text, width, size, maxLines = 2) {
  const lines = [''];
  for (const c of String(text || '')) {
    if (measure(lines.at(-1) + c, size) > width) lines.push('');
    lines[lines.length - 1] += c;
  }
  if (lines.length > maxLines) throw new Error(`Node label too long (${maxLines} lines): ${text}`);
  return lines;
}
function svgText(text, x, y, width, size, klass, maxLines = 2) {
  return `<text class="${klass}" x="${x}" y="${y}" font-size="${size}">${wrap(text, width, size, maxLines).map((line, i) => `<tspan x="${x}" dy="${i ? size + 5 : 0}">${esc(line)}</tspan>`).join('')}</text>`;
}
function renderSvg(graph) {
  const defs = `<defs><marker id="${graph.id}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#65738a"/></marker></defs>`;
  const lanes = graph.lanes.map(lane => {
    const [fill, ink] = colors[lane.color] || colors.neutral;
    return `<g class="lane"><rect x="${lane.x}" y="38" width="${lane.w}" height="${graph.height - 64}" rx="12" fill="${fill}" opacity=".58"/><rect x="${lane.x}" y="38" width="${lane.w}" height="54" rx="9" fill="${fill}"/><text x="${lane.x + lane.w / 2}" y="71" text-anchor="middle" font-size="18" font-weight="700" fill="${ink}">${esc(lane.label)}</text></g>`;
  }).join('');
  const edges = graph.edges.map(edge => `<g class="edge ${edge.kind}" data-from="${edge.from}" data-to="${edge.to}"><path d="${roundedPath(edge.points)}" fill="none" stroke="${edge.kind === 'sync' ? '#956eb4' : '#65738a'}" stroke-width="1.8" ${['sync', 'reuse', 'build'].includes(edge.kind) ? 'stroke-dasharray="6 5"' : ''} marker-end="url(#${graph.id}-arrow)"/>${edge.labelBox ? `<rect x="${edge.labelBox.x}" y="${edge.labelBox.y}" width="${edge.labelBox.w}" height="${edge.labelBox.h}" rx="4" fill="#ffffff" stroke="#e1e6ed"/><text x="${edge.labelBox.x + edge.labelBox.w / 2}" y="${edge.labelBox.y + 16}" text-anchor="middle" font-size="12" fill="#56637a">${esc(edge.label)}</text>` : ''}</g>`).join('');
  const nodes = graph.nodes.map(node => {
    const [fill, ink] = colors[node.color] || colors.neutral;
    const titleLines = wrap(node.title, node.w - 32, 17);
    const subtitleY = node.y + (titleLines.length > 1 ? 80 : 64);
    return `<g class="node" id="node-${node.id}" data-node="${node.id}" role="button" tabindex="0" aria-label="${esc(node.title)}"><title>${esc(node.symbol || node.title)}</title><rect class="node-box" x="${node.x}" y="${node.y}" width="${node.w}" height="${node.h}" rx="9" fill="#ffffff" stroke="${ink}" stroke-width="1.3"/><rect x="${node.x + 1}" y="${node.y + 1}" width="5" height="${node.h - 2}" rx="3" fill="${ink}"/>${svgText(node.title, node.x + 18, node.y + 28, node.w - 36, 17, 'node-title')}${svgText(node.subtitle || '', node.x + 18, subtitleY, node.w - 36, 13, 'node-subtitle', 2)}<text x="${node.x + node.w - 12}" y="${node.y + node.h - 10}" text-anchor="end" font-size="10" fill="${ink}">${esc(node.id)}</text></g>`;
  }).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${graph.width}" height="${graph.height}" viewBox="0 0 ${graph.width} ${graph.height}" aria-label="${esc(graph.title)}"><style>text{font-family:Inter,"Noto Sans SC","Microsoft YaHei",sans-serif}.node-title{font-weight:650;fill:#20324c}.node-subtitle{fill:#586a80}.node{cursor:pointer}.node:hover .node-box,.node:focus .node-box{stroke-width:3}.node.selected .node-box{stroke-width:3;fill:#f3f7ff}.node.muted{opacity:.23}.edge.dimmed{opacity:.18}.edge.related path{stroke-width:3}.hide-sync .edge.sync,.hide-sync .edge.reuse{display:none}</style><rect width="100%" height="100%" fill="#ffffff"/>${defs}${lanes}${edges}${nodes}</svg>`;
}
function mermaid(graph) {
  const rows = ['flowchart TB'];
  for (const lane of graph.lanes) {
    rows.push(`  subgraph lane_${lane.id}["${lane.label.replaceAll('"', '『')}"]`);
    for (const node of graph.nodes.filter(n => n.lane === lane.id)) rows.push(`    ${node.id}["${node.title.replaceAll('"', '『')}"]`);
    rows.push('  end');
  }
  for (const edge of graph.edges) rows.push(`  ${edge.from} ${['sync', 'reuse', 'build'].includes(edge.kind) ? '-.->' : '-->'} ${edge.to}`);
  return rows.join('\n') + '\n';
}
for (const graph of graphs) {
  graph.svg = renderSvg(graph);
  graph.mermaid = mermaid(graph);
  fs.writeFileSync(path.join(visual, 'diagrams', `${graph.id}.svg`), graph.svg + '\n');
  fs.writeFileSync(path.join(visual, 'diagrams', `${graph.id}.mmd`), graph.mermaid);
}
const manifest = { sourceCommit, sourceRepository: repoUrl, sourceBranch, siteUrl,
  scenario: { platform: 'Ascend950', save: 'bf16', hidden: 7168, ffn: 3072, clipping: false, forceExpertReady: false },
  learningScenes: learning.scenes.map(({ id, title, sourceNode }) => ({ id, title, sourceNode })),
  files: Object.fromEntries([...readSource].map(([file, data]) => [file, { sha256: data.sha256, lines: data.lines.length - 1 }])), nodes: manifestNodes };
fs.writeFileSync(path.join(visual, 'source-map.json'), JSON.stringify(manifest, null, 2) + '\n');
const bundle = { sourceCommit, sourceRepository: repoUrl, scenario: manifest.scenario, graphs, learning };
const payload = JSON.stringify(bundle).replaceAll('<', '\\u003c').replaceAll('\u2028', '\\u2028').replaceAll('\u2029', '\\u2029');
const html = fs.readFileSync(path.join(visual, 'assets', 'page.html'), 'utf8')
  .replace('/*__STYLE__*/', ['reader.css', 'layout.css', 'learning.css', 'gmm2-pipeline.css'].map(file => fs.readFileSync(path.join(visual, 'assets', file), 'utf8')).join('\n'))
  .replace('/*__DATA__*/', payload)
  .replace('/*__APP__*/', () => ['gmm2-pipeline.js', 'full-pipeline.js', 'reader.js'].map(file => fs.readFileSync(path.join(visual, 'assets', file), 'utf8')).join('\n'));
fs.writeFileSync(path.join(visual, 'index.html'), html);
let markdown = `# MegaMoE BF16 训练源码图解\n\n按项目结构、调用链、三核流水和内存复用逐层阅读。默认场景为 Ascend950、save=bf16、H7168/F3072、BF16 路由权重 p、关闭 clipping。\n\n`;
markdown += `源码快照：[${sourceCommit.slice(0, 7)}](${repoUrl}/tree/${sourceCommit})。\n\n[离线交互页面](index.html) · [完整调用点与源码](CALLPOINTS.md) · [维护方法](MAINTAINING.md)\n\n`;
if (siteUrl) markdown += `[在线交互导览](${siteUrl})\n\n`;
markdown += '**VS Code 一键定位**：按[扩展安装说明](vscode/README.md)安装一次，运行 **MegaMoE: 打开训练源码导览**。点击节点、阶段或等待查看讲解，点击 **打开源码** 或 **查看对应源码**，再在旁边的编辑器中定位到文件和行号。\n\n';
if (siteUrl) markdown += `[下载 VS Code 导览扩展](${siteUrl}downloads/megamoe-train-source-guide.vsix)\n\n`;
else markdown += 'VS Code 安装包可按[本地打包方法](vscode/README.md#一次性安装)生成，无需在线站点。\n\n';
markdown += '本地用浏览器打开 index.html，即可点击节点查看读写、等待和带行号源码。仓库网页可以直接阅读下方图文，通过调用点表跳转源码。\n\n';
markdown += '普通浏览器保留 **复制定位路径**：按 **Ctrl+P** 粘贴并回车，打开对应源码和行号。\n\n';
markdown += '数值与验证：[BF16/FP32 舍入规则](../mega_moe_bf16_design.md#13-数学关系与舍入边界) · [测试参考与验收](../../tests/bf16/TEST_SPEC.md)。BigOp 与契约参考共用 FP32 激活、导数及归并规则，CPU FP64 真值独立计算。\n\n';
markdown += '## 三个阅读场景\n\n固定侧栏可以随时回到章节。三个场景共用同一份教学路由，切换后保留所选 token 和前反向阶段；点击数据或阶段可查看真实源码。\n\n';
for (const scene of learning.scenes) {
  markdown += `### ${scene.title}\n\n${scene.summary}\n\n![${scene.title}](diagrams/${scene.id}.png)\n\n`;
}
for (const graph of graphs) {
  markdown += `## ${graph.layer} · ${graph.title}\n\n${graph.summary}\n\n![${graph.title}](diagrams/${graph.id}.png)\n\n[查看矢量图](diagrams/${graph.id}.svg) · [查看图源](diagrams/${graph.id}.mmd) · [本节调用点](CALLPOINTS.md#${graph.id})\n\n`;
}
fs.writeFileSync(path.join(visual, 'README.md'), markdown.trimEnd() + '\n');
let calls = '# 函数、数据与源码调用点\n\n图中节点与下面的编号一一对应；在线链接固定到本次采集的源码提交。\n\n';
for (const graph of graphs) {
  calls += `<a id="${graph.id}"></a>\n\n## ${graph.title}\n\n`;
  for (const node of graph.nodes) {
    calls += `### ${node.id} · ${node.title}\n\n${(node.description || []).join('\n\n')}\n\n`;
    for (const [title, values] of [['执行位置', [graph.lanes.find(l => l.id === node.lane).label]], ['读取', node.reads], ['写入', node.writes], ['等待', node.waits]]) if (values?.length) calls += `- **${title}**：${values.join('；')}\n`;
    if (node.formula) calls += `\n\`\`\`text\n${node.formula}\n\`\`\`\n`;
    calls += '\n';
    for (const source of node.sources) calls += `源码：[${path.basename(source.path)}:${source.line}](${source.url}) · [仓内文件](${path.relative(visual, path.join(root, source.path)).replaceAll('\\', '/')})\n\n`;
  }
}
fs.writeFileSync(path.join(visual, 'CALLPOINTS.md'), calls.trimEnd() + '\n');
const site = path.join(visual, 'site');
fs.mkdirSync(path.join(site, 'diagrams'), { recursive: true });
fs.copyFileSync(path.join(visual, 'index.html'), path.join(site, 'index.html'));
fs.copyFileSync(path.join(visual, 'source-map.json'), path.join(site, 'source-map.json'));
for (const file of fs.readdirSync(path.join(visual, 'diagrams'))) if (/\.(svg|mmd|png)$/.test(file)) fs.copyFileSync(path.join(visual, 'diagrams', file), path.join(site, 'diagrams', file));
fs.writeFileSync(path.join(site, '.nojekyll'), '');
console.log(`Built ${graphs.length} views, ${allIds.size} source-backed nodes from ${sourceCommit.slice(0, 7)} (${repoUrl}).`);
