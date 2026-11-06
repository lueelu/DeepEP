// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import cp from 'node:child_process';
import { fileURLToPath } from 'node:url';
import './source-config-check.mjs';

const visual = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(visual, '../../../..');
const manifest = JSON.parse(fs.readFileSync(path.join(visual, 'source-map.json'), 'utf8'));
const html = fs.readFileSync(path.join(visual, 'index.html'), 'utf8');
const data = JSON.parse(html.match(/<script id="reader-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
assert.equal(data.graphs.length, 8);
assert.equal(data.learning.scenes.length, 3);
assert.equal(manifest.learningScenes.length, 3);
assert.equal(manifest.sourceCommit, data.sourceCommit);
assert.equal(typeof manifest.sourceBranch, 'string');
assert(manifest.sourceBranch.length > 0, 'Source identity must name its branch or detached commit');
const readme = fs.readFileSync(path.join(visual, 'README.md'), 'utf8');
if (manifest.siteUrl) {
  assert(readme.includes(`[在线交互导览](${manifest.siteUrl})`), 'Explicit site navigation');
} else {
  assert(!readme.includes('[在线交互导览]('), 'Offline guide must not advertise an online site');
  assert(!readme.includes('[下载 VS Code 导览扩展]('), 'Offline guide must not depend on a site download');
}
assert.equal(manifest.nodes.length, data.graphs.reduce((sum, g) => sum + g.nodes.length, 0));
assert(!/<script[^>]+src=/.test(html), 'Reader must open offline without external runtime scripts');
assert(!html.includes('/*__'), 'Unreplaced page placeholder');
const nodeIds = new Set();
const sourceCache = new Map();
for (const [file, info] of Object.entries(manifest.files)) {
  const bytes = cp.execFileSync('git', ['-C', root, 'show', `${manifest.sourceCommit}:${file}`]);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), info.sha256, file);
  sourceCache.set(file, bytes.toString('utf8').split('\n'));
  const current = cp.execFileSync('git', ['-C', root, 'show', `HEAD:${file}`]);
  assert.equal(crypto.createHash('sha256').update(current).digest('hex'), info.sha256, `Refresh source content: ${file}`);
}
for (const node of manifest.nodes) {
  assert(!nodeIds.has(node.id), node.id); nodeIds.add(node.id);
  for (const source of node.sources) {
    const lines = sourceCache.get(source.path);
    assert(lines[source.line - 1].includes(source.anchor), `${node.id}: anchor`);
    assert(new URL(source.url).pathname.includes(`/blob/${manifest.sourceCommit}/`), `${node.id}: source URL`);
    for (const row of source.excerpt) assert.equal(row.text, lines[row.line - 1], `${node.id}: excerpt ${row.line}`);
  }
}
for (const scene of data.learning.scenes) {
  assert(nodeIds.has(scene.sourceNode), `${scene.id}: initial source`);
  for (const match of scene.html.matchAll(/data-learning-node=["']([^"']+)["']/g)) assert(nodeIds.has(match[1]), `${scene.id}: ${match[1]}`);
  assert(!/<script\b/.test(scene.html), `${scene.id}: scripts belong to the shared reader`);
}
const overlaps = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
for (const graph of data.graphs) {
  const ids = new Set(graph.nodes.map(n => n.id));
  for (const [i, a] of graph.nodes.entries()) for (const b of graph.nodes.slice(i + 1)) assert(!overlaps(a, b), `${graph.id}: overlapping nodes ${a.id}/${b.id}`);
  for (const edge of graph.edges) {
    assert(ids.has(edge.from) && ids.has(edge.to));
    for (let i = 1; i < edge.points.length; ++i) {
      const a = edge.points[i - 1], b = edge.points[i];
      assert(a.x === b.x || a.y === b.y, `${edge.id}: diagonal connector`);
      for (const n of graph.nodes.filter(n => ![edge.from, edge.to].includes(n.id))) {
        const hit = a.x === b.x
          ? a.x > n.x && a.x < n.x + n.w && Math.min(a.y, b.y) < n.y + n.h && Math.max(a.y, b.y) > n.y
          : a.y > n.y && a.y < n.y + n.h && Math.min(a.x, b.x) < n.x + n.w && Math.max(a.x, b.x) > n.x;
        assert(!hit, `${edge.id}: connector crosses ${n.id}`);
      }
    }
    if (edge.labelBox) assert(!graph.nodes.some(n => overlaps(n, edge.labelBox)), `${edge.id}: label overlaps node`);
  }
}
let links = 0;
for (const file of ['README.md', 'CALLPOINTS.md', 'MAINTAINING.md']) {
  const text = fs.readFileSync(path.join(visual, file), 'utf8');
  assert(!text.includes('\ufffd'), file);
  assert.equal((text.match(/```/g) || []).length % 2, 0, file);
  for (const match of text.matchAll(/\]\(([^)]+)\)/g)) {
    const target = match[1];
    if (/^https?:/.test(target)) continue;
    const [relative] = target.split('#');
    if (!relative) continue;
    // PNGs are rendered by the browser check immediately following this step.
    if (/^diagrams\/[^/]+\.png$/.test(relative)) {
      assert([...data.graphs, ...data.learning.scenes].some(g => relative === `diagrams/${g.id}.png`), relative);
    } else assert(fs.existsSync(path.resolve(visual, decodeURIComponent(relative))), `${file}: ${target}`);
    ++links;
  }
}
fs.mkdirSync(path.join(visual, '.artifacts'), { recursive: true });
const report = { ok: true, sourceCommit: manifest.sourceCommit, graphs: data.graphs.length, learningScenes: data.learning.scenes.length, nodes: nodeIds.size, sourceFiles: sourceCache.size, localLinks: links };
fs.writeFileSync(path.join(visual, '.artifacts', 'source-check.json'), JSON.stringify(report, null, 2) + '\n');
console.log(report);
