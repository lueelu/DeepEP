// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

// Fixed-input teaching scenes. Values are CPU arithmetic, never device timing.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const visual = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const number = value => Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
const vector = values => `[${values.map(number).join(', ')}]`;
const scalar = new Float32Array(1), bits = new Uint32Array(scalar.buffer);
function round(value) {
  scalar[0] = value;
  bits[0] = (bits[0] + 0x7fff + ((bits[0] >>> 16) & 1)) & 0xffff0000;
  return scalar[0];
}
const f32 = Math.fround;
const multiply = (a, b) => f32(a * b);
const add = (a, b) => f32(a + b);
const sum = values => values.reduce(add, 0);
const matvec = (matrix, input) => matrix.map(row => round(sum(row.map((value, index) => multiply(value, input[index])))));
const outer = (a, b) => a.map(value => b.map(other => multiply(value, other)));
const chip = (node, text) => `<button type="button" class="learning-source-chip" data-learning-node="${node}">${escape(text)} ↗</button>`;
let svgSerial = 0;
function group(node, label, body) {
  return `<g class="learning-hotspot" data-learning-node="${node}" tabindex="0" role="button" aria-label="${escape(label)}"><title>${escape(label)}</title>${body}</g>`;
}
function text(x, y, value, cls = '', anchor = 'start') {
  return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escape(value)}</text>`;
}
function box(x, y, w, h, color = 'blue', radius = 12) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" class="learning-fill-${color}"/>`;
}
function arrow(d, color = 'blue', label = '', x = 0, y = 0) {
  return `<path d="${d}" class="learning-arrow learning-stroke-${color}" marker-end="url(#learning-arrow)"/>${label ? text(x, y, label, 'learning-small') : ''}`;
}
function svg(width, height, label, content) {
  const marker = `learning-arrow-${++svgSerial}`;
  return `<div class="learning-diagram"><svg class="learning-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(label)}"><defs><marker id="${marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#74839a"/></marker></defs>${content.replaceAll('url(#learning-arrow)', `url(#${marker})`)}</svg></div>`;
}
function table(head, rows, cls = '') {
  return `<div class="learning-table-wrap ${cls}"><table class="learning-table"><thead><tr>${head.map(value => `<th>${escape(value)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(value => `<td>${value}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
function matrix(values, rowLabels = []) {
  return `<table class="learning-matrix"><tbody>${values.map((row, index) => `<tr>${rowLabels.length ? `<th>${escape(rowLabels[index])}</th>` : ''}${row.map(value => `<td>${number(value)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function routeValues(caseData, route) {
  const { rank, token, expert, p } = route;
  const x = [(1 + (token + rank) % 3) / 2, (1 + (token + 2 * rank) % 4) / 4];
  const dy = caseData.numeric.dy;
  const patterns = caseData.numeric.weightPatterns;
  const weights = patterns[expert % patterns.length];
  const gate = matvec(weights.wg, x), up = matvec(weights.wu, x);
  const sigmoid = gate.map(value => f32(1 / add(1, f32(Math.exp(-value)))));
  const silu = gate.map((value, index) => f32(value / add(1, f32(Math.exp(-value)))));
  const h = silu.map((value, index) => multiply(value, up[index]));
  const hp = h.map(value => round(multiply(value, p)));
  const output = matvec(weights.w2, hp);
  const dhp = matvec(weights.w2[0].map((_, index) => weights.w2.map(row => row[index])), dy);
  const dh = dhp.map(value => multiply(value, p));
  const dup = dh.map((value, index) => round(multiply(value, silu[index])));
  const dgate = dh.map((value, index) => round(multiply(multiply(value, up[index]), multiply(sigmoid[index], add(1, multiply(gate[index], f32(1 - sigmoid[index])))))));
  const dz = [...dgate, ...dup], w1 = [...weights.wg, ...weights.wu];
  const dx = x.map((_, column) => round(sum(dz.map((value, row) => multiply(value, w1[row][column])))));
  const dpProducts = dhp.map((value, index) => multiply(value, h[index]));
  const dp = round(sum(dpProducts));
  return { ...route, x, dy, weights, gate, up, h, hp, output, dhp, dh, dgate, dup, dx, dpProducts, dp,
    dw1: outer(dz, x), dw2: outer(dy, hp) };
}
function deriveCase(caseData) {
  const s = caseData.shape, routes = [], counts = Array.from({ length: s.ep }, () => Array(s.totalExperts).fill(0));
  const probabilities = caseData.routing.probabilities;
  assert(s.topk <= s.totalExperts && s.ep * s.expertsPerRank === s.totalExperts);
  assert.equal(probabilities.length, s.topk);
  assert.equal(sum(probabilities), 1);
  for (let rank = 0; rank < s.ep; rank++) for (let token = 0; token < s.tokensPerRank; token++) {
    for (let slot = 0; slot < s.topk; slot++) {
      const expert = (token + rank + slot) % s.totalExperts;
      routes.push({ rank, token, slot, expert, p: probabilities[slot], originalRoute: token * s.topk + slot, ordinal: counts[rank][expert]++ });
    }
  }
  const expertRows = Array.from({ length: s.totalExperts }, (_, expert) => sum(counts.map(row => row[expert])));
  const rankReceiveRows = Array.from({ length: s.ep }, (_, rank) => sum(expertRows.slice(rank * s.expertsPerRank, (rank + 1) * s.expertsPerRank)));
  assert(counts.every(row => row.every(rows => rows === caseData.routing.expectedRowsPerSourceExpert)));
  assert(expertRows.every(rows => rows === caseData.routing.expectedRowsPerExpert));
  assert(rankReceiveRows.every(rows => rows === caseData.routing.expectedRowsPerRank));
  assert(rankReceiveRows.every(rows => rows <= s.recvCapacity));
  for (const route of routes) {
    route.ownerRank = Math.floor(route.expert / s.expertsPerRank);
    route.localExpert = route.expert % s.expertsPerRank;
    route.sortedRow = sum(counts[route.rank].slice(0, route.expert)) + route.ordinal;
    route.expertRow = sum(counts.slice(0, route.rank).map(row => row[route.expert])) + route.ordinal;
    route.expertPackedRow = sum(expertRows.slice(route.ownerRank * s.expertsPerRank, route.expert)) + route.expertRow;
    route.readyBlock = Math.floor(route.expertRow / 384);
    route.gmm1Tile = Math.floor(route.expertRow / 192);
  }
  const focused = new Map(caseData.focusTokens.map(token => [`${token.rank}/${token.token}`, []]));
  const weightTotals = Array.from({ length: s.totalExperts }, () => ({ dw1: Array.from({ length: 4 }, () => [0, 0]), dw2: [[0, 0], [0, 0]] }));
  for (const route of routes) {
    const numeric = routeValues(caseData, route);
    focused.get(`${route.rank}/${route.token}`)?.push(numeric);
    const totals = weightTotals[route.expert];
    for (const name of ['dw1', 'dw2']) {
      numeric[name].forEach((row, i) => row.forEach((value, j) => { totals[name][i][j] = add(totals[name][i][j], value); }));
    }
  }
  for (const total of weightTotals) for (const name of ['dw1', 'dw2']) total[name] = total[name].map(row => row.map(round));
  const focus = caseData.focusTokens.map(token => {
    const selected = focused.get(`${token.rank}/${token.token}`);
    assert.deepEqual(selected.map(({ expert, slot, p, sortedRow, expertRow }) => ({ expert, slot, p, sortedRow, expertRow })), token.expectedRoutes);
    const y = selected[0].output.map((_, index) => round(sum(selected.map(route => route.output[index]))));
    const dx = selected[0].dx.map((_, index) => round(sum(selected.map(route => route.dx[index]))));
    return { ...token, x: selected[0].x, dy: selected[0].dy, routes: selected, y, dx, dp: selected.map(route => route.dp) };
  });
  return { ...caseData, derived: { counts, expertRows, rankReceiveRows, weightTotals }, focusTokens: focus };
}

function context(caseData, arithmetic = true) {
  const s = caseData.shape;
  return `<div class="learning-context"><strong>${arithmetic ? 'CPU 数值教学' : '流水仿真'}</strong><span>EP${s.ep} · 全局${s.totalExperts}专家 · 每源M${s.tokensPerRank} · TopK${s.topk}</span><span>H${s.hidden} / F${s.ffn} · saved-bf16</span></div>${arithmetic ? '<p class="learning-caption">只展示 H[0:2] 与 F[0:2]，其余坐标全零。R 表示 BF16 最近偶数舍入。</p>' : ''}`;
}
function tokenPanel(token, phase, content, first) {
  return `<div class="learning-token-panel" data-learning-token="${token.id}" data-learning-phase="${phase}"${first && phase === 'forward' ? '' : ' hidden'}>${content}</div>`;
}
function heading(index, title, source, label) {
  return `<div class="learning-section-heading"><span class="learning-section-number">${index}</span><h3>${escape(title)}</h3>${source ? chip(source, label || '对应源码') : ''}</div>`;
}
function routeTable(token) {
  return table(['TopK 槽', '原始 route', '专家 → 所在 rank', '源卡排序行', '专家内接收行', 'p'], token.routes.map(route => [route.slot, `${token.token}×${token.routes.length}+${route.slot}=${route.originalRoute}`, `e${route.expert} → rank${route.ownerRank}`, `<strong>${route.sortedRow}</strong>`, `<strong>${route.expertRow}</strong>`, number(route.p)]));
}
function rankFlow(caseData, token, backward = false) {
  const rows = token.routes.length, rowHeight = 116;
  const center = 55 + ((rows - 1) * rowHeight + 108) / 2;
  const boxTop = center - 62;
  const height = 55 + rows * rowHeight + 36;
  let content = text(35, 30, `${token.label} · ${backward ? `同一份 dY 发给 ${rows} 条 route` : 'X 和 p 使用同一行映射'}`, 'learning-svg-title');
  content += group(backward ? 'b_gather' : 'f_routes', '查看源 rank 路由整理', box(35, boxTop, 220, 125, 'blue') + text(53, boxTop + 28, `源 rank${token.rank}`, 'learning-strong') + text(53, boxTop + 56, `token ${token.token}`, 'learning-medium') + text(53, boxTop + 83, `${backward ? 'dY' : 'X'} = ${vector(backward ? token.dy : token.x)}`, 'learning-code'));
  for (const [index, route] of token.routes.entries()) {
    const y = 55 + index * rowHeight;
    content += arrow(`M255 ${center} H305 V${y + 53} H380`, 'teal', `slot${route.slot} / p=${number(route.p)}`, 273, y + 35);
    content += group(backward ? 'b_dispatch_do' : 'f_dispatch_x', `查看 e${route.expert} 的 Dispatch`, box(385, y, 285, 108, 'teal') + text(402, y + 27, `e${route.expert} 在 rank${route.ownerRank}`, 'learning-strong') + text(402, y + 54, `源排序行 ${route.sortedRow} → 专家行 ${route.expertRow}`, 'learning-small') + text(402, y + 81, backward ? `dO = dY = ${vector(token.dy)}` : `ready块 ${route.readyBlock} / M192块 ${route.gmm1Tile}`, 'learning-code'));
    content += arrow(`M670 ${y + 53} H720 V${center} H775`, 'orange');
  }
  const result = backward ? token.dx : token.y;
  content += group(backward ? 'b_unpermute_dx' : 'f_unpermute', '查看源 rank 的归并', box(780, boxTop, 270, 125, 'orange') + text(798, boxTop + 29, `回到原 rank${token.rank}`, 'learning-strong') + text(798, boxTop + 56, backward ? 'FP32 累加 route dX，再舍入 BF16' : 'FP32 累加 route Y，再舍入 BF16', 'learning-small') + text(798, boxTop + 86, `${backward ? 'dX' : 'Y'} = ${vector(result)}`, 'learning-code'));
  const perSource = caseData.derived.counts[0][0];
  content += text(38, height - 19, backward ? 'p 在后续 FP32 dH = p × dHp 使用；这里发送的 dO 不先乘 p。' : `每个专家 ${caseData.derived.expertRows[0]} 行（${caseData.shape.ep} 个源 rank 各 ${perSource} 行）；${rows} 条 route 各有独立回传位置。`, 'learning-small');
  return svg(1090, height, backward ? 'dY复制、专家梯度与dX回源' : 'token路由与专家输出回源', content);
}
function weightsTiles(token) {
  return `<div class="learning-weights-grid">${token.routes.map(route => `<div class="learning-weight-set"><h4>e${route.expert} 的非零权重子块</h4><div class="learning-weight-triple"><div>${chip('f_gate_up', 'Wg[0:2,0:2]')}${matrix(route.weights.wg)}</div><div>${chip('f_gate_up', 'Wu[0:2,0:2]')}${matrix(route.weights.wu)}</div><div>${chip('f_gmm2', 'W2[0:2,0:2]')}${matrix(route.weights.w2)}</div></div><p class="learning-caption">W1完整形状[2F,H]；Gate位于前F行，Up位于后F行。W2完整形状[H,F]。</p></div>`).join('')}</div>`;
}
function numericForward(token) {
  return table(['route', 'G = R(X · Wgᵀ)', 'U = R(X · Wuᵀ)', 'H = SiLU_FP32(G) × U', 'Hp = R(H × p)', 'Yroute = R(Hp · W2ᵀ)'], token.routes.map(route => [`e${route.expert} / slot${route.slot}`, vector(route.gate), vector(route.up), vector(route.h), `<strong>${vector(route.hp)}</strong>`, vector(route.output)]), 'learning-numeric');
}
function numericBackward(token) {
  const headers = [
    'route',
    'dHp = R(dO · W2)',
    'dH = FP32(dHp) × p',
    'dGate =\nR((dH × U) × SiLU′(G))',
    'dUp =\nR(dH × SiLU(G))',
    'dZ = (dGate, dUp)',
    'dXroute = R(dZ · W1)',
    'dp_route',
  ];
  const rows = token.routes.map(route => [
    `e${route.expert} / slot${route.slot}`,
    vector(route.dhp),
    vector(route.dh),
    vector(route.dgate),
    vector(route.dup),
    vector([...route.dgate, ...route.dup]),
    `<strong>${vector(route.dx)}</strong>`,
    `<strong>${number(route.dp)}</strong>`,
  ]);
  return table(headers, rows, 'learning-numeric learning-backward-numeric');
}
function gradientDestinations(caseData, token) {
  const expertRows = caseData.routing.expectedRowsPerExpert;
  const rows = token.routes.map(route => {
    const total = caseData.derived.weightTotals[route.expert];
    return [`e${route.expert} → 留在rank${route.ownerRank}`, `${number(route.dy[0])} × ${number(route.hp[0])} = ${number(route.dw2[0][0])}`, `${number(route.dgate[0])} × ${number(route.x[0])} = ${number(route.dw1[0][0])}`, `dW2[0,0]=${number(total.dw2[0][0])}<br>dW1[0,0]=${number(total.dw1[0][0])}`];
  });
  return `<div class="learning-gradient-result"><div>${chip('b_unpermute_dx', 'dX [M,H] 回到源rank')}<strong>${vector(token.dx)}</strong><span>按 TopK 槽 FP32 累加，再舍入 BF16</span></div><div>${chip('b_dp_output', 'dp [M,TopK] 回到源rank')}<strong>${vector(token.dp)}</strong><span>按 ${token.routes.length} 个 TopK 槽分别保留</span></div></div>${table(['专家梯度归属', '本route对dW2[0,0]的贡献', '本route对dW1[0,0]的贡献', `该专家全部 ${expertRows} 条 route 的最终梯度`], rows)}<p class="learning-caption">dW2 = R(dOᵀ · Hp)，dW1 = R(dZᵀ · Xpacked)。单 route 的外积是 FP32 贡献，沿 ${expertRows} 行累加后才舍入到 BF16。</p><details class="learning-expand"><summary>展开各 route 的完整非零权重梯度子块</summary><div class="learning-weights-grid">${token.routes.map(route => `<div class="learning-weight-set"><h4>e${route.expert} · 单route贡献（FP32乘积）</h4><div class="learning-weight-pair"><div>${chip('b_dw2', 'dW2贡献 [H0:H1,F0:F1]')}${matrix(route.dw2)}</div><div>${chip('b_dw1', 'dW1贡献：Gate0/1、Up0/1')}${matrix(route.dw1, ['G0', 'G1', 'U0', 'U1'])}</div></div></div>`).join('')}</div></details>`;
}
function renderToken(caseData) {
  const panels = caseData.focusTokens.map((token, index) => {
    const forward = `
      ${heading('1', '先分清三种行号', 'f_routes', '路由索引')}
      <p class="learning-lead">${escape(token.label)} 的 X = <strong>${vector(token.x)}</strong>，选择 ${token.routes.map(route => `e${route.expert}`).join('、')}。原始 route、源卡排序行和专家接收行各有自己的坐标。</p>
      ${routeTable(token)}${rankFlow(caseData, token)}
      ${heading('2', '只展开参与计算的矩阵子块', 'f_gate_up', 'Gate / Up MMAD')}${weightsTiles(token)}
      ${heading('3', '沿实际舍入边界看中间数值', 'f_swiglu_p', 'SwiGLU × p')}${numericForward(token)}
      <div class="learning-equation">按 slot 0–${token.routes.length - 1} 用 FP32 加完再舍入 BF16，Y[token${token.token}] = <strong>${vector(token.y)}</strong></div>
      <p class="learning-caption">H 保持 FP32，乘 p 后才舍入形成 Hp。Gate/Up 保存的是上表原始值，Hp 临时供 GMM2 消费。切到「反向」继续使用同一个 token 和这份前向状态。</p>`;
    const backward = `
      ${heading('1', '最后是求和，所以每条 route 得到同一 dY', 'b_gather', 'GatherRoutes')}${rankFlow(caseData, token, true)}
      ${heading('2', '从 dHp 走到激活梯度、dX 和 dp', 'b_numeric', 'SwiGLU 反向')}
      ${numericBackward(token)}
      <p class="learning-caption">G、U 是 Gate、Up；SiLU′(G) = s × (1 + G × (1 − s))，s = sigmoid(G)。H、dH 与 dp 乘积保持 FP32。关闭 clipping 时按表中公式计算，R 表示舍入到 BF16。dZ 沿特征维拼接，前 F 列为 dGate，后 F 列为 dUp。</p>
      <p class="learning-caption">dp 先对 F 分块归约，再得到每条 route 的标量。本例只有同一块内的 F0、F1 非零。</p>
      <div class="learning-equation">dp(slot0) = R(Σ_FP32 (FP32(dHp) × H)) = <strong>${number(token.routes[0].dp)}</strong>，点积使用保持 FP32 的未乘 p 激活 H。</div>
      ${heading('3', '区分回到 token 和留在专家的梯度', 'b_dw2', 'dW2 / dW1')}${gradientDestinations(caseData, token)}`;
    return tokenPanel(token, 'forward', forward, index === 0) + tokenPanel(token, 'backward', backward, false);
  });
  return `<div class="learning-scene learning-token-scene">${context(caseData)}<p class="learning-caption learning-routing-note">${escape(caseData.routing.rowMeaning)}</p>${panels.join('')}</div>`;
}

function savedBridge() {
  const resources = [
    { title: '原始 Gate / Up', size: 'saved_state · 两个 [R,F] BF16 平面', producer: 'f_save_raw', from: 'GMM1 → AIV0 保存', consumers: [['b_numeric', '重建 H / Hp，计算 dGate / dUp']] },
    { title: 'Xpacked', size: 'dispatch_output · [R,H] BF16', producer: 'f_dispatch_x', from: '专家卡 Dispatch 收到 X', consumers: [['b_dw1', 'dW1 = R(dZᵀ · Xpacked)']] },
    { title: 'packed p', size: 'dispatch_output · 每 route 一个 BF16', producer: 'f_dispatch_p', from: '与 X 对齐的 route 权重', consumers: [['b_numeric', 'Hp = R(H×p)；dH = dHp×p（FP32）']] },
    { title: 'route 正 / 逆置换', size: 'sortedRoute ↔ token×TopK+slot', producer: 'f_routes', from: '前向路由初始化', consumers: [['b_gather', '反向复制对应 token 的原始 dY'], ['b_unpermute_dx', '按原 token 归并 dX'], ['b_dp_output', '按原 TopK 槽恢复 dp']] },
    { title: '专家计数 / source 前缀', size: 'saved_state · 实际行数和源分段', producer: 'f_counts', from: '计数交换与前缀', consumers: [['b_restore_counts', '恢复 Dispatch / 回传行段']] },
  ];
  let content = text(45, 30, '前向生产', 'learning-svg-title') + text(425, 30, '跨越 forward → backward 的持久数据', 'learning-svg-title') + text(810, 30, '反向消费', 'learning-svg-title');
  resources.forEach((item, index) => {
    const y = 57 + index * 106;
    content += `<g data-learning-resource="saved-${index}" data-learning-resource-label="${escape(item.title)}" data-learning-node="${item.producer}" tabindex="0" role="button" aria-label="${escape(item.title)}的生产与消费">`;
    content += group(item.producer, item.from, box(45, y + 8, 270, 65, 'blue') + text(63, y + 35, item.from, 'learning-medium') + text(63, y + 57, '点开定位生产者', 'learning-tiny'));
    content += arrow(`M315 ${y + 41} H395`, 'blue');
    content += box(400, y, 340, 82, 'violet') + text(420, y + 30, item.title, 'learning-strong') + text(420, y + 56, item.size, 'learning-small');
    content += arrow(`M740 ${y + 41} H800`, 'violet');
    item.consumers.forEach(([node, label], position) => {
      const h = item.consumers.length === 1 ? 65 : 25;
      const top = item.consumers.length === 1 ? y + 8 : y + position * 28;
      content += group(node, label, box(805, top, 335, h, 'teal', 7) + text(819, top + (h === 65 ? 39 : 18), label, h === 65 ? 'learning-medium' : 'learning-small'));
    });
    content += '</g>';
  });
  return svg(1170, 595, '前向保存数据到反向消费者的跨阶段桥', content);
}
function lifetimeBands() {
  let content = text(25, 27, '同一地址，按依赖切换用途', 'learning-svg-title') + text(25, 51, '每行只表达该内存的局部生命期；横向长度不代表时间或字节比例。', 'learning-small');
  const band = (row, title, left, boundary, right, sources) => {
    const y = 90 + row * 117;
    content += text(25, y + 34, title, 'learning-code');
    content += group(sources[0], left[0], box(220, y, 330, 70, 'blue') + text(239, y + 29, left[0], 'learning-strong') + text(239, y + 54, left[1], 'learning-small'));
    content += group(sources[1], boundary, `<path d="M575 ${y - 6} V${y + 77}" class="learning-boundary"/>` + text(578, y - 13, boundary, 'learning-tiny', 'middle'));
    content += arrow(`M550 ${y + 35} H625`, 'violet');
    content += group(sources[2], right[0], box(630, y, 430, 70, 'teal') + text(648, y + 29, right[0], 'learning-strong') + text(648, y + 54, right[1], 'learning-small'));
  };
  band(0, 'ptrHp [R,F]', ['暂存 dHp = R(dO×W2)', '整专家 dHpReady 后，owner 先读入 tile UB'], '本 tile dHp 已读入 UB', ['改存重建 Hp = R(H×p)', '各 tile 独占；P2 用于 dW2'], ['b_dhp', 'm_hp_overwrite', 'b_numeric']);
  band(1, 'ptrDO = ptrDX', ['保留原始 dO', 'dHp GEMM 与 dW2 都需要读取'], 'DOReleased = blockNum', ['覆盖为 dXroute', 'AIC 写 GM → DXReady → AIV0 回传'], ['b_dw2', 'b_do_release', 'b_dx']);
  band(2, 'SHMEM ptrD', ['接收 routed dX 矩阵', '源 rank 读完 TopK 行并得到 grad_x'], '所有 rank 的 dX 读者退出', ['复用为连续 dp 标量表', 'P6 回传后按逆置换恢复 [M,TopK]'], ['b_dx_send', 'b_reuse_ptrd', 'b_reduce_dp']);
  return svg(1100, 460, 'dHp到Hp、dO到dX和ptrD的三次复用', content);
}
function renderSaved(caseData) {
  const snapshots = caseData.focusTokens.map((token, index) => {
    const route = token.routes[0];
    return `<div class="learning-snapshot" data-learning-token="${token.id}"${index ? ' hidden' : ''}><strong>${escape(token.label)} → e${route.expert}</strong><span>保存 G=${vector(route.gate)}、U=${vector(route.up)}、p=${number(route.p)}</span><span>反向重建 H=${vector(route.h)}，再得到 Hp=${vector(route.hp)}</span></div>`;
  }).join('');
  return `<div class="learning-scene learning-saved-scene">${context(caseData)}${heading('1', '前向保存的张量与反向用途', 'lc_saved', 'ctx 保存 Tensor')}${snapshots}${savedBridge()}<p class="learning-caption">W1/W2 由 ctx 保持参数引用；上图是算子生成的持久状态。前向临时 Hp 不跨越反向，反向用 Gate/Up 和 p 重建。none 模式不保存 Gate/Up，而用 Xpacked、W1 重算 GMM1；本图固定 bf16。</p>${heading('2', '写入新含义之前，旧读者必须全部结束', 'm_do_last_reader', '内存复用协议')}${lifetimeBands()}<div class="learning-memory-footer"><span><strong>packed X/p</strong>、原始 Gate/Up、路由元数据：活到匹配反向。</span><span><strong>前向 Hp / ready</strong>：当前前向临时区。</span><span><strong>反向 dO/dX、Hp、dZ、dp partial</strong>：当前反向 scratch。</span></div></div>`;
}
function pipelineSwimlanes() {
  let content = text(25, 28, '同一核组的 GMM2 连续任务：tile0 / tile1 / tile2', 'learning-svg-title') + text(25, 53, 'AIC 与两个配对 AIV；专家计数覆盖所有相关核。横向只是依赖顺序，不是实测时间轴。', 'learning-small');
  for (const [index, title] of ['AIC · MMAD', 'AIV0 · 激活', 'AIV1 · 传输'].entries()) {
    const y = 90 + index * 130;
    content += `<rect x="130" y="${y - 13}" width="1005" height="92" rx="9" class="learning-lane"/>` + text(23, y + 28, title, 'learning-medium');
  }
  const work = (node, x, y, w, title, detail, color) => group(node, title, box(x, y, w, 62, color) + text(x + 13, y + 25, title, 'learning-medium') + text(x + 13, y + 47, detail, 'learning-tiny'));
  content += work('f_gmm2', 315, 90, 135, '计算 tile0', '输出到槽0', 'orange');
  content += work('f_gmm2', 480, 90, 135, '计算 tile1', '输出到槽1', 'orange');
  content += work('f_gmm2', 645, 90, 125, '等槽0归还', 'tile2不能覆盖', 'neutral');
  content += work('f_gmm2', 800, 90, 135, '计算 tile2', '再次使用槽0', 'orange');
  content += work('f_gmm2_wait', 145, 220, 160, '当前专家 Hp 全齐', 'SwiGLU task计数', 'blue');
  content += work('f_swiglu_p', 360, 220, 445, '其他专家的 SwiGLU 尾部', '条件允许时，可与早专家GMM2交错；不是全卡统一阶段', 'violet');
  content += work('f_release_dispatch_ub', 145, 350, 160, 'Dispatch 结束', '释放本核AIV1 UB', 'blue');
  content += work('f_combine', 480, 350, 255, '发送 tile0 / 槽0', 'MTE3完成 → NotifyCube(槽0)', 'teal');
  content += work('f_combine', 765, 350, 175, '发送 tile1 / 槽1', '完成后归还槽1', 'teal');
  content += work('f_combine', 970, 350, 160, '发送 tile2 / 槽0', '复用后的新结果', 'teal');
  content += arrow('M305 251 H310 V121 H315', 'blue');
  content += arrow('M305 381 H310 V121', 'blue');
  content += arrow('M450 121 H465 V381 H480', 'orange', 'tile ready', 371, 181);
  content += arrow('M615 121 H630 V305 H750 V381 H765', 'orange', 'tile ready', 651, 324);
  content += arrow('M735 350 V180 H707 V152', 'teal', '槽0发送已完成', 757, 188);
  content += arrow('M935 121 H951 V381 H970', 'orange');
  content += text(146, 450, '槽0：tile0 → 发送完 → tile2；槽1：tile1 → 发送完 → 下一次使用。GMM1 的 Gate/Up 两片 UB 是另一种用途。', 'learning-small');
  return svg(1160, 483, 'GMM2双槽与AIC AIV0 AIV1依赖泳道', content);
}
function backwardRetirement() {
  const entries = [
    ['b_dz_ready', 'P1 两路 AIV 完成', 'dZ / Hp 已写完', 'blue'],
    ['b_dw2', 'AIC 先算 dW2', '读取原 dO 与 Hp', 'orange'],
    ['b_do_release', '全部 AIC 退休', 'DOReleased齐全', 'violet'],
    ['b_dx', 'AIC 写 dX', '覆盖原 dO 行段', 'orange'],
    ['b_dx_send', 'AIV0 回传', '整专家DXReady后读', 'teal'],
  ];
  let content = text(25, 29, '反向 saved-bf16：为什么 dW2 必须先于 dO 覆盖', 'learning-svg-title');
  entries.forEach(([node, title, detail, color], index) => {
    const x = 25 + index * 219;
    content += group(node, title, box(x, 65, 190, 78, color) + text(x + 13, 96, title, 'learning-medium') + text(x + 13, 123, detail, 'learning-small'));
    if (index) content += arrow(`M${x - 29} 104 H${x - 5}`, 'violet');
  });
  content += text(28, 180, '仅本核 dW2 算完不够：别的 AIC 可能仍在读取同一专家 dO。各专家使用独立行段与计数。', 'learning-small');
  return svg(1120, 211, '反向dO复用的真实退休链', content);
}
function gmm2Simulation() {
  return `<section class="gmm2-simulation" data-gmm2-pipeline aria-labelledby="gmm2-title">
    <h3 id="gmm2-title">GMM2 → Combine 双槽流水仿真</h3>
    <p class="learning-caption">一个核组连续处理同一专家的 6 个 tile。初始时，本核 Dispatch UB 已释放，专家 Hp 已写齐。耗时以教学 tick 表示。</p>
    <div class="gmm2-presets" role="group" aria-label="耗时预设">
      <button type="button" data-gmm2-preset="3,5">搬运较慢 · 3 / 5</button>
      <button type="button" data-gmm2-preset="5,3">计算较慢 · 5 / 3</button>
    </div>
    <div class="gmm2-controls">
      <div><label for="gmm2-compute">GMM2 每 tile 计算耗时</label><output id="gmm2-compute-value" for="gmm2-compute"></output><input id="gmm2-compute" type="range" min="1" max="20" step="1"></div>
      <div><label for="gmm2-combine">Combine 每 tile 搬运耗时</label><output id="gmm2-combine-value" for="gmm2-combine"></output><input id="gmm2-combine" type="range" min="1" max="20" step="1"></div>
    </div>
    <div class="gmm2-summary" aria-live="polite"></div>
    <div class="gmm2-chart-scroll" role="region" aria-label="GMM2、Combine 与两个 UB 槽的时间轴" tabindex="0"></div>
    <div class="gmm2-legend" aria-label="时间轴图例"><span><i class="compute"></i>计算</span><span><i class="send"></i>搬运</span><span><i class="wait"></i>等待</span><span><i class="hold"></i>槽占用</span></div>
    <div class="gmm2-selection" aria-live="polite"></div>
  </section>`;
}
function fullPipeline(caseData) {
  return `<section class="gmm2-simulation full-pipeline" data-full-pipeline aria-labelledby="full-pipeline-title">
    <h3 id="full-pipeline-title">普通 BF16 完整流水</h3>
    <p class="learning-caption">EP${caseData.shape.ep} · 固定 rank0 · 每 rank ${caseData.shape.expertsPerRank} 个专家。显示一个代表核组，其他核组和 rank 参与就绪判断。阶段窗口包含内部等待，默认教学 tick 参考打点图的阶段比例。</p>
    <div class="gmm2-presets" role="group" aria-label="前后向总览">
      <button type="button" data-overview-direction="forward">前向总览</button>
      <button type="button" data-overview-direction="backward">反向总览</button>
    </div>
    <div class="pipeline-overview-controls">
      <label>调节阶段 <select data-overview-stage></select></label>
      <label class="pipeline-duration">工作耗时 <input type="range" data-overview-ticks><output data-overview-tick-value></output></label>
      <button type="button" data-overview-reset>恢复默认</button>
    </div>
    <div class="gmm2-summary" data-overview-end aria-live="polite"></div>
    <div class="gmm2-chart-scroll" data-overview-chart role="region" aria-label="AIC、AIV0、AIV1 完整时间轴" tabindex="0"></div>
    <div class="gmm2-legend" data-overview-legend aria-label="阶段图例"></div>
    <p class="pipeline-overlap-note" data-overview-overlap></p>
    <p class="learning-caption" data-overview-note></p>
    <label class="pipeline-wait-picker">定位等待 <select data-overview-waits></select></label>
    <div class="gmm2-selection" data-overview-detail aria-live="polite"></div>
  </section>`;
}
function renderPipeline(caseData) {
  return `<div class="learning-scene learning-pipeline-scene">
    ${context(caseData, false)}${fullPipeline(caseData)}
    <details class="pipeline-expand" data-isolated-slots>
      <summary>打开独立双槽练习 · GMM2 → Combine</summary>
      ${gmm2Simulation()}
      ${heading('1', '把双槽协议真正展开三轮', 'f_gmm2', 'GMM2 与 Combine')}
      ${pipelineSwimlanes()}
      <p class="learning-caption">先等本核 Dispatch 释放 UB、当前专家写完 Hp。tile0 发送期间可以计算 tile1，tile2 要等槽0归还。</p>
    </details>
    ${heading('2', '反向 dW2、dX 与回传的退休顺序', 'b_dw2', 'P2 的实际顺序')}${backwardRetirement()}
    ${heading('3', '三个复用位置，各自等待不同读者', 'm_dx_alias', '同地址不同生命期')}${lifetimeBands()}
  </div>`;
}

export function buildLearningScenes() {
  svgSerial = 0;
  const caseData = deriveCase(JSON.parse(fs.readFileSync(path.join(visual, 'data/learning-case.json'), 'utf8')));
  return { case: caseData, scenes: [
    { id: 'learn-token', title: '追踪一个 token 的前反向', summary: '同一份固定输入：看 route 行号、关键数值、矩阵 tile 和四项梯度最终归属。', sourceNode: 'f_routes', phaseSources: { forward: 'f_routes', backward: 'b_gather' }, html: renderToken(caseData) },
    { id: 'learn-saved', title: '前向保存的数据，反向怎样使用', summary: '连接生产者与消费者，再沿同一地址看三次生命周期切换。', sourceNode: 'b_saved', html: renderSaved(caseData) },
    { id: 'learn-pipeline', title: '前后向流水与缓冲复用', summary: '在 rank0 的前后向专家时间轴上调节耗时、点等待查看源码，再用独立练习展开 GMM2 双槽。', sourceNode: 'f_routes', html: renderPipeline(caseData) },
  ] };
}
