// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const visual = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const artifacts = path.join(visual, '.artifacts');
fs.mkdirSync(artifacts, { recursive: true });
const edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const executablePath = process.env.BROWSER_PATH || (process.platform === 'win32' && fs.existsSync(edge) ? edge : undefined);
const browser = await chromium.launch({ headless: true, executablePath });
const errors = [];
const reports = [];
const server = http.createServer((req, res) => {
  const relative = new URL(req.url, 'http://localhost').pathname;
  if (relative === '/favicon.ico') { res.writeHead(204).end(); return; }
  const file = path.resolve(visual, '.' + (relative === '/' ? '/index.html' : decodeURIComponent(relative)));
  if (!file.startsWith(visual + path.sep)) { res.writeHead(403).end(); return; }
  try {
    const types = { '.svg': 'image/svg+xml', '.png': 'image/png', '.json': 'application/json', '.mmd': 'text/plain; charset=utf-8' };
    res.setHeader('Content-Type', types[path.extname(file)] || 'text/html; charset=utf-8');
    res.end(fs.readFileSync(file));
  }
  catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
async function chooseView(page, id) {
  const button = page.locator(`[data-view="${id}"]`);
  if (!await button.isVisible()) await page.locator('#nav-toggle').click();
  await button.click();
}
try {
  const context = await browser.newContext({ viewport: { width: 1600, height: 1060 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(() => window.__reader?.data.graphs.length === 8);
  const ids = await page.evaluate(() => window.__reader.data.graphs.map(g => g.id));
  const syntaxPage = await context.newPage();
  await syntaxPage.addScriptTag({ path: path.join(visual, 'node_modules', 'mermaid', 'dist', 'mermaid.min.js') });
  const diagrams = await page.evaluate(() => window.__reader.data.graphs.map(g => ({ id: g.id, source: g.mermaid })));
  for (const diagram of diagrams) await syntaxPage.evaluate(async diagram => {
    window.mermaid.initialize({ startOnLoad: false });
    await window.mermaid.parse(diagram.source);
  }, diagram);
  await syntaxPage.close();
  assert.equal(await page.locator('[data-view]').count(), 11);
  assert.equal(await page.locator('.source-location').first().inputValue(), await page.evaluate(() => {
    const source = window.__reader.selected().sources[0]; return `${source.path}:${source.line}`;
  }));
  for (const id of ids) {
    await chooseView(page, id);
    assert.equal(await page.evaluate(() => window.__reader.graph().id), id);
    const first = await page.locator('#canvas [data-node]').first().getAttribute('data-node');
    await page.locator(`#canvas [data-node="${first}"]`).click();
    assert(await page.locator('.code-line.anchor-line').count() > 0, `${id}: source excerpt`);
    assert((await page.locator('.source-links a').first().getAttribute('href')).includes('/blob/'));
    const nodes = await page.locator('#canvas [data-node]').count();
    await page.locator('#next').click();
    assert.equal(await page.evaluate(() => window.__reader.selected().ordinal), 2);
    await page.locator('#previous').click();
    assert.equal(await page.evaluate(() => window.__reader.selected().ordinal), 1);
    await page.locator('#query').fill('不存在的函数xyz');
    assert.equal(await page.locator('#canvas .node.muted').count(), nodes);
    await page.locator('#query').fill('');
    await page.locator('#show-sync').uncheck();
    assert(await page.locator('#viewport').evaluate(el => el.classList.contains('hide-sync')));
    await page.locator('#show-sync').check();
    const before = await page.locator('#zoom-label').innerText();
    await page.locator('#zoom-in').click();
    assert.notEqual(await page.locator('#zoom-label').innerText(), before);
    await page.locator('#fit').click();
    await page.locator('#expand').click();
    assert.equal(await page.locator('#expand').getAttribute('aria-pressed'), 'true');
    await page.locator('#expand').click();
    const clipped = await page.locator('#canvas svg').evaluate(svg => [...svg.querySelectorAll('text')].filter(el => {
      const box = el.getBBox(), view = svg.viewBox.baseVal;
      return box.x < view.x - 2 || box.y < view.y - 2 || box.x + box.width > view.width + 2 || box.y + box.height > view.height + 2;
    }).map(el => el.textContent));
    assert.deepEqual(clipped, [], `${id}: clipped text`);
    const overflowing = await page.locator('#canvas .node').evaluateAll(nodes => nodes.flatMap(node => {
      const rect = node.querySelector('.node-box');
      const x = +rect.getAttribute('x'), y = +rect.getAttribute('y');
      const width = +rect.getAttribute('width'), height = +rect.getAttribute('height');
      return [...node.querySelectorAll('text')].filter(text => {
        const box = text.getBBox();
        return box.x < x || box.y < y || box.x + box.width > x + width || box.y + box.height > y + height;
      }).map(text => `${node.dataset.node}: ${text.textContent}`);
    }));
    assert.deepEqual(overflowing, [], `${id}: text exceeds its node`);
    await page.screenshot({ path: path.join(artifacts, `${id}.png`) });
    reports.push({ id, nodes, source: true, navigation: true, clipped });
    // Static PNGs use the same generated SVG as the interactive reader.
    const svgPage = await context.newPage();
    await svgPage.goto(pathToFileURL(path.join(visual, 'diagrams', `${id}.svg`)).href);
    const image = path.join(visual, 'diagrams', `${id}.png`);
    await svgPage.locator('svg').screenshot({ path: image, scale: 'css' });
    fs.copyFileSync(image, path.join(visual, 'site', 'diagrams', `${id}.png`));
    await svgPage.close();
  }
  const scenes = await page.evaluate(() => window.__reader.data.learning.scenes.map(scene => scene.id));
  const focus = await page.evaluate(() => window.__reader.data.learning.case.focusTokens.map(token => token.id));
  const sceneReports = [];
  for (const id of scenes) {
    const isPipeline = id === 'learn-pipeline';
    await chooseView(page, id);
    assert.equal(await page.evaluate(() => window.__reader.view()), id);
    assert(await page.locator('#canvas').evaluate(canvas => canvas.clientWidth <= canvas.parentElement.clientWidth + 2), `${id}: content fits its panel`);
    if (!isPipeline) await page.locator('#case-token').selectOption(focus[0]);
    if (id === 'learn-token') await page.locator('#case-phase').selectOption('forward');
    assert(await page.locator('#canvas [data-learning-node]:visible').count() > 0, `${id}: clickable sources`);
    const source = page.locator('#canvas [data-learning-node]:visible').first();
    const nodeId = await source.getAttribute('data-learning-node');
    await source.click();
    assert.equal(await page.evaluate(() => window.__reader.selected().id), nodeId);
    assert.equal(await page.evaluate(() => window.__reader.view()), id, 'Source inspection keeps the scene');
    const output = path.join(visual, 'diagrams', `${id}.png`);
    await page.locator('#viewport').screenshot({ path: output, scale: 'css' });
    fs.copyFileSync(output, path.join(visual, 'site', 'diagrams', `${id}.png`));
    if (!isPipeline) await page.locator('#case-token').selectOption(focus[1]);
    if (id === 'learn-token') await page.locator('#case-phase').selectOption('backward');
    if (id === 'learn-saved') {
      await page.locator('#canvas [data-learning-resource]').first().click();
      assert.equal(await page.locator('#canvas .learning-resource-active').count(), 1, 'Producer and consumers share the selected tensor');
    }
    const before = await page.evaluate(() => window.__reader.learningState());
    await chooseView(page, '00-overview');
    await chooseView(page, id);
    assert.deepEqual(await page.evaluate(() => window.__reader.learningState()), before, `${id}: preserved context`);
    const invalidVisible = await page.locator('#canvas [data-learning-token]:visible').evaluateAll((elements, token) => elements.filter(el => el.dataset.learningToken !== token).map(el => el.dataset.learningToken), focus[1]);
    assert.deepEqual(invalidVisible, [], `${id}: visible token slices`);
    await page.screenshot({ path: path.join(artifacts, `${id}.png`) });
    sceneReports.push({ id, source: true, tokenAndPhase: true, context: true });
  }
  const overview = page.locator('[data-full-pipeline]');
  const overviewChart = overview.locator('[data-overview-chart]');
  const overviewFormula = overview.locator('[data-overview-detail] .formula');
  const l5Formulas = await page.evaluate(() => Object.fromEntries(window.__reader.data.graphs
    .flatMap(graph => graph.nodes.filter(node => node.formula).map(node => [node.id, node.formula]))));
  assert.match(await page.locator('.learning-context').innerText(), /EP8 · 全局32专家 · 每源M4096 · TopK6/);
  assert.match(await overview.innerText(), /固定 rank0 · 每 rank 4 个专家/);
  assert.equal(await page.locator('#case-token').isVisible(), false);
  const readSpans = (stage, chart = overviewChart) => chart.locator(`.gmm2-segment:not(.wait)[data-stage="${stage}"]`)
    .evaluateAll(bars => bars.map(bar => ({ label: bar.textContent, start: +bar.dataset.start, end: +bar.dataset.end })));
  const overlaps = (a, b) => Math.max(a.start, b.start) < Math.min(a.end, b.end);
  const expertLabels = ['e0', 'e1', 'e2', 'e3'];
  for (const stage of ['dispatch', 'gmm1', 'swiglu', 'gmm2', 'combine']) {
    assert.deepEqual((await readSpans(stage)).map(bar => bar.label), expertLabels, `Forward shows one ${stage} window per expert`);
  }
  const gmm1 = await readSpans('gmm1'), swiglu = await readSpans('swiglu');
  assert(gmm1.every((span, expert) => overlaps(span, swiglu[expert])), 'Each GMM1 expert window intersects its SwiGLU window');
  const gmm2 = await readSpans('gmm2');
  assert(gmm2.every((span, expert) => expert === 0 || span.start === gmm2[expert - 1].end), 'Default GMM2 feeds Combine without stalling on the slots');
  assert.match(await overview.locator('[data-overview-note]').innerText(), /GMM1 工作 64 tick，GMM2 工作 32 tick，SwiGLU 工作 8 tick/);
  await overview.locator('[data-overview-stage]').selectOption('gmm1');
  assert.equal(await overview.locator('[data-overview-tick-value]').innerText(), '64 tick / 专家');
  const expertBoundary = spans => spans[1].start - spans[0].end;
  assert.equal(expertBoundary(gmm1), 1, 'Only the last tile epilogue delays the next expert');
  await overview.locator('[data-overview-stage]').selectOption('swiglu');
  await overview.locator('[data-overview-ticks]').press('End');
  assert.equal(expertBoundary(await readSpans('gmm1')), 20, 'Slowing SwiGLU exposes the real UB reuse wait');
  const ubWait = overviewChart.locator('[data-wait-for="gate-up-ub"]').first();
  assert.equal(await ubWait.evaluate(button => button.classList.contains('pipeline-short-wait-button')), false, 'A longer wait keeps its duration bar');
  await overview.locator('[data-overview-waits]').selectOption(await ubWait.getAttribute('data-sim-event'));
  assert.match(await page.locator('#inspector details[open] .anchor-line').innerText(), /WaitForVector\(\)/);
  await overview.locator('[data-overview-reset]').click();
  const shortUbWait = overviewChart.locator('.pipeline-short-wait').filter({ has: page.locator('[data-wait-for="gate-up-ub"]') }).first();
  const waitLine = shortUbWait.locator('.pipeline-short-wait-line');
  const waitControl = shortUbWait.getByRole('button');
  assert.equal(await waitLine.evaluate(line => line.getBoundingClientRect().width), 1);
  assert((await waitControl.boundingBox()).width >= 24);
  await waitControl.click();
  assert.match(await page.locator('#inspector details[open] .anchor-line').innerText(), /WaitForVector\(\)/);
  assert.equal(await waitControl.getAttribute('aria-pressed'), 'true');
  assert.equal(await overviewChart.locator('.gmm2-track').first().locator('.pipeline-overlap-band').count(), 4);
  await overviewChart.locator('[data-wait-for="gate-up-ready"]').first().click();
  assert.match(await overview.locator('[data-overview-detail]').innerText(), /WaitForCube\(\).*NotifyVector\(\)/s);
  assert.equal(await page.evaluate(() => window.__reader.selected().id), 'f_gate_up');
  const yWrite = overviewChart.locator('[data-stage="y"]').first();
  const firstWrite = Number(await yWrite.getAttribute('data-start'));
  await overview.locator('[data-overview-stage]').selectOption('peer');
  await overview.locator('[data-overview-ticks]').press('End');
  assert(Number(await yWrite.getAttribute('data-start')) > firstWrite, 'Y waits for a slower peer');
  await overview.locator('[data-overview-reset]').click();
  assert.equal(Number(await yWrite.getAttribute('data-start')), firstWrite);
  await yWrite.click();
  assert.equal(await overviewFormula.textContent(), l5Formulas.f_unpermute);
  assert.equal(await overviewChart.locator('[data-wait-for="slot"]').count(), 0);
  const combineTail = overviewChart.locator('[data-wait-for="combine-tail"]');
  await overview.locator('[data-overview-waits]').selectOption(await combineTail.getAttribute('data-sim-event'));
  assert.match(await page.locator('#inspector details[open] .anchor-line').innerText(), /EndGMM2Sync\(\)/);
  const yArrived = overviewChart.locator('[data-wait-for="y-arrived"]').last();
  await yArrived.click();
  assert.equal(await overview.locator('[data-overview-waits]').inputValue(), await yArrived.getAttribute('data-sim-event'));
  await waitControl.click();
  await overview.screenshot({ path: path.join(artifacts, 'full-forward.png') });

  await overview.locator('[data-overview-direction="backward"]').click();
  for (const stage of ['gather', 'dispatch', 'dhp', 'epi', 'dw2', 'dx', 'returnDx', 'dw1', 'unpermuteDx', 'dp', 'outputDp']) {
    assert(await overviewChart.locator(`[data-stage="${stage}"]`).count(), `Backward shows ${stage}`);
  }
  for (const stage of ['dw2', 'dx', 'returnDx', 'dw1']) {
    assert.deepEqual((await readSpans(stage)).map(bar => bar.label), expertLabels, `Backward shows four ${stage} windows`);
  }
  const dw1 = await readSpans('dw1');
  const dw2 = await readSpans('dw2'), dx = await readSpans('dx');
  const dhp = await readSpans('dhp');
  const aiv0 = overviewChart.locator('.gmm2-lane').filter({ has: page.getByText('AIV0', { exact: true }) });
  const aiv1 = overviewChart.locator('.gmm2-lane').filter({ has: page.getByText('AIV1', { exact: true }) });
  assert.equal((await readSpans('gather', aiv1)).length, 1);
  assert.equal((await readSpans('restore', aiv1)).length, 1);
  assert.equal((await readSpans('dispatch', aiv1)).length, 0, 'Only AIV0 pulls payload in P0-B');
  const aiv1Epi = await readSpans('epi', aiv1);
  assert(aiv1Epi.every((span, expert) => span.start >= dhp[expert].end), 'SwiGLU Grad waits for the matching expert dHp');
  assert((await readSpans('dispatch', aiv0)).some(span => overlaps(span, aiv1Epi[0])), 'AIV1 can work on e0 while AIV0 dispatches later experts');
  for (const [lane, node] of [[aiv0, 'b_epi0'], [aiv1, 'b_epi1']]) {
    await lane.locator('[data-stage="epi"]').first().click();
    assert.equal(await page.evaluate(() => window.__reader.selected().id), node);
    const explanation = await overview.locator('[data-overview-detail]').innerText();
    assert.match(explanation, /AIV0 还在 Dispatch 时，AIV1 可先做已就绪的专家/);
    assert.match(explanation, /nextTargetAiv & 1U.*序号跨专家延续/);
    assert.match(explanation, /DZReady >= 2 × blockNum.*算 dW2/);
    assert.equal(await overviewFormula.textContent(), l5Formulas.b_numeric);
  }
  await overview.screenshot({ path: path.join(artifacts, 'full-backward-swiglu.png') });
  for (const [stage, node] of [['dw1', 'b_dw1'], ['dw2', 'b_dw2'], ['dx', 'b_dx'], ['unpermuteDx', 'b_unpermute_dx'], ['dp', 'b_reduce_dp']]) {
    await overviewChart.locator(`.gmm2-segment:not(.wait)[data-stage="${stage}"]`).first().click();
    assert.equal(await overviewFormula.textContent(), l5Formulas[node]);
  }
  const width = ({ start, end }) => end - start;
  assert.equal(width(dx[0]), 2 * width(dhp[0]));
  assert.equal(width(dw1[0]), 2 * width(dw2[0]));
  assert(dx.every((span, expert) => span.start - dw2[expert].end <= 2), 'Default dO retirement is brief between dW2 and dX');
  const dxMerge = (await readSpans('unpermuteDx'))[0];
  assert(dw1.some(span => overlaps(span, dxMerge)), 'dX Unpermute overlaps dW1');
  assert.match(await overview.locator('[data-overview-overlap]').innerText(), /dX 归并与 dW1 · 重叠/);
  const dxReadyWait = overviewChart.locator('[data-wait-for="dx-ready"]').first();
  assert.equal(await dxReadyWait.innerText(), '等 dX');
  await dxReadyWait.click();
  assert.equal(await overviewFormula.count(), 0);
  assert.match(await page.locator('#inspector details[open] .anchor-line').innerText(), /ReadGmByPassDCache\(dxReady\)/);
  const dpWait = overviewChart.locator('[data-wait-for="dp-arrived"]').first();
  const ptrdWait = overviewChart.locator('[data-wait-for="ptrd-reuse"]').first();
  await dpWait.click();
  assert.equal(await page.evaluate(() => window.__reader.selected().id), 'b_dp_output');
  await ptrdWait.click();
  assert.match(await overview.locator('[data-overview-detail]').innerText(), /P5 → P6 CrossRankSyncInWorldSize/);
  assert.equal(await page.evaluate(() => window.__reader.selected().id), 'b_reuse_ptrd');
  await overview.screenshot({ path: path.join(artifacts, 'full-backward.png') });
  await overview.locator('[data-overview-direction="forward"]').click();

  const openExercise = async target => {
    const exercise = target.locator('[data-isolated-slots]');
    if (await exercise.getAttribute('open') === null) await exercise.locator('summary').click();
  };
  const sim = page.locator('[data-gmm2-pipeline]');
  await openExercise(page);
  const threeRounds = page.locator('[data-isolated-slots]').getByRole('heading', { name: '把双槽协议真正展开三轮' });
  assert(await threeRounds.isVisible());
  assert.equal(await sim.locator('.compute[data-stage="gmm2"]').count(), 6);
  const firstSlotWait = sim.locator('[data-wait-for="slot"]').first();
  assert.equal(await firstSlotWait.getAttribute('data-start'), '6');
  assert.equal(await firstSlotWait.getAttribute('data-end'), '8');
  await firstSlotWait.click();
  assert.match(await sim.locator('.gmm2-selection').innerText(), /WaitForVector\(0 \+ FLAG_ID_MAX\).*NotifyCube\(0\)/s);
  assert.equal(await page.evaluate(() => window.__reader.selected().id), 'f_gmm2');
  await sim.locator('#gmm2-compute').press('ArrowRight');
  assert.equal(await sim.locator('#gmm2-compute-value').innerText(), '4 tick');
  assert.equal(await firstSlotWait.getAttribute('data-start'), '8');
  assert.equal(await firstSlotWait.getAttribute('data-end'), '9');
  await sim.locator('[data-gmm2-preset="5,3"]').click();
  assert.equal(await sim.locator('[data-wait-for="slot"]').count(), 0);
  await sim.locator('[data-wait-for="output"][data-start="8"]').click();
  assert.match(await sim.locator('.gmm2-selection').innerText(), /WaitForCube\(1\).*NotifyVector\(1 \+ FLAG_ID_MAX\)/s);
  assert.match(await page.locator('#inspector details[open] .anchor-line').innerText(), /WaitForCube\(pingpongIdx\)/);
  await sim.locator('#gmm2-combine').press('ArrowRight');
  assert.equal(await sim.locator('#gmm2-combine-value').innerText(), '4 tick');
  await chooseView(page, '00-overview');
  await chooseView(page, 'learn-pipeline');
  await openExercise(page);
  assert.equal(await sim.locator('#gmm2-combine').inputValue(), '4', 'Simulation inputs survive changing scenes');
  const gateEvents = await page.evaluate(() => simulateGmm2Combine({ computeTicks: 3, combineTicks: 5, dispatchReadyAt: 2, hpReadyAt: 7 })
    .events.filter(event => event.lane === 'aic').slice(0, 3).map(({ start, end, source }) => [start, end, source.node, source.index]));
  assert.deepEqual(gateEvents, [[0, 2, 'f_gmm2_wait', 0], [2, 7, 'f_gmm2_wait', 1], [7, 10, 'f_gmm2', 0]]);
  await sim.locator('[data-gmm2-preset="3,5"]').click();
  await firstSlotWait.click();
  await sim.screenshot({ path: path.join(artifacts, 'gmm2-pipeline.png') });
  await page.locator('[data-isolated-slots]').screenshot({ path: path.join(artifacts, 'gmm2-three-rounds.png') });
  const sceneHash = await page.evaluate(() => location.hash);
  await page.reload();
  assert.equal(await page.evaluate(() => location.hash), sceneHash, 'Scene deep link reload');
  await chooseView(page, '06-backward-kernel');
  const download = page.waitForEvent('download');
  await page.locator('#mmd-download').click();
  assert.equal((await download).suggestedFilename(), '06-backward-kernel.mmd');
  await page.screenshot({ path: path.join(artifacts, 'desktop.png') });
  // Verify that navigation down to another view actually selects that view.
  const targets = await page.evaluate(() => window.__reader.data.graphs.flatMap(graph => graph.nodes
    .filter(node => node.drilldown).map(node => ({ sourceView: graph.id, sourceNode: node.id, ...node.drilldown }))));
  for (const target of targets) {
    await page.evaluate(target => { window.__reader.showView(target.sourceView, target.sourceNode); }, target);
    await page.locator('[data-drill-view]').click();
    assert.equal(await page.evaluate(() => window.__reader.graph().id), target.view);
    if (target.node) assert.equal(await page.evaluate(() => window.__reader.selected().id), target.node);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await chooseView(page, 'learn-token');
  assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), 'Mobile page overflow');
  await page.screenshot({ path: path.join(artifacts, 'mobile.png'), fullPage: true });
  await chooseView(page, 'learn-pipeline');
  assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), 'Simulation scrolls within its mobile panel');
  await overview.locator('[data-overview-direction="backward"]').click();
  assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), 'Backward overview scrolls within its mobile panel');
  await overviewChart.locator('.pipeline-short-wait-button[data-wait-for="ptrd-reuse"]').first().click();
  assert.equal(await page.evaluate(() => window.__reader.selected().id), 'b_reuse_ptrd');
  await openExercise(page);
  await sim.locator('[data-gmm2-preset="5,3"]').click();
  await sim.locator('[data-wait-for="output"]').first().click();
  assert.match(await sim.locator('.gmm2-selection').innerText(), /WaitForCube\(0\)/);
  await chooseView(page, '00-overview');
  // A fresh context with all network requests blocked must still read file://.
  const offline = await browser.newContext({ viewport: { width: 1440, height: 960 }, offline: true });
  const local = await offline.newPage();
  local.on('pageerror', error => errors.push(error.message));
  await local.goto(pathToFileURL(path.join(visual, 'index.html')).href);
  await local.waitForFunction(() => window.__reader?.data.graphs.length === 8);
  const localSources = await local.evaluate(() => [...new Set(window.__reader.data.graphs
    .flatMap(graph => graph.nodes.flatMap(node => node.sources.map(source => source.path))))]
    .map(file => new URL(`../../../../${file}`, location.href).href));
  for (const source of localSources) assert(fs.existsSync(fileURLToPath(source)), `Local source file: ${source}`);
  await chooseView(local, 'learn-pipeline');
  await local.locator('[data-overview-direction="backward"]').click();
  await local.locator('[data-overview-chart] [data-wait-for="ptrd-reuse"]').first().click();
  assert.equal(await local.evaluate(() => window.__reader.selected().id), 'b_reuse_ptrd');
  await openExercise(local);
  await local.locator('[data-gmm2-preset="5,3"]').click();
  await local.locator('[data-gmm2-pipeline] [data-wait-for="output"]').first().click();
  assert.match(await local.locator('#inspector details[open] .anchor-line').innerText(), /WaitForCube\(pingpongIdx\)/);
  await chooseView(local, '06-backward-kernel');
  assert.equal(await local.locator('.source-links a[href^="file:"]').count(), 0, 'No misleading raw file link');
  const source = await local.evaluate(() => window.__reader.selected().sources[0]);
  const sourcePath = path.resolve(visual, '../../../..', source.path);
  const location = await local.locator('.source-location').first().inputValue();
  assert.equal(location.toLowerCase(), `${sourcePath.replaceAll('\\', '/')}:${source.line}`.toLowerCase());
  const editor = new URL(await local.locator('.source-links a[href^="vscode:"]').first().getAttribute('href'));
  assert.equal(editor.host, 'file');
  assert.equal(editor.hash + editor.search, '', 'Editor path must not become a query or fragment');
  assert.equal(decodeURIComponent(editor.pathname).replace(/^\/([A-Za-z]:\/)/, '$1').toLowerCase(), `${location}:1`.toLowerCase());
  await local.locator('[data-copy-location]').first().click();
  const copyStatus = await local.locator('.copy-status').first().innerText();
  assert(copyStatus.startsWith('已复制') || copyStatus.startsWith('路径已选中'), 'Copy or explicit keyboard fallback');
  const hash = await local.evaluate(() => location.hash);
  await local.reload();
  assert.equal(await local.evaluate(() => location.hash), hash);
  await local.screenshot({ path: path.join(artifacts, 'offline.png') });
  await offline.close();
  // Serve the exact Pages output, including its relative source-map/image paths.
  const published = await context.newPage();
  published.on('pageerror', error => errors.push(error.message));
  const siteResponse = await published.goto(`http://127.0.0.1:${server.address().port}/site/index.html`);
  assert.equal(siteResponse.status(), 200);
  await published.waitForFunction(() => window.__reader?.data.graphs.length === 8);
  const siteCommit = await published.evaluate(() => window.__reader.data.sourceCommit);
  assert.equal(siteCommit, await page.evaluate(() => window.__reader.data.sourceCommit));
  const siteMapResponse = await published.request.get(new URL('source-map.json', published.url()).href);
  assert.equal(siteMapResponse.status(), 200);
  assert.equal((await siteMapResponse.json()).sourceCommit, siteCommit);
  await chooseView(published, 'learn-pipeline');
  await published.locator('[data-overview-direction="backward"]').click();
  await published.locator('[data-overview-chart] [data-stage="epi"]').first().click();
  assert.equal(await published.locator('[data-overview-detail] .formula').textContent(), l5Formulas.b_numeric);
  assert.match(await published.locator('[data-overview-detail]').innerText(), /dH = dHp × p.*FP32/s);
  await published.getByRole('button', { name: /查看对应源码/ }).first().click();
  const websiteSource = await published.evaluate(() => window.__reader.selected().sources[0]);
  assert(new URL(websiteSource.url).pathname.includes(`/blob/${siteCommit}/`));
  assert.equal(await published.locator('.source-links a[href^="file:"],.source-links a[href^="vscode:"]').count(), 0);
  const siteImage = await published.request.get(new URL('diagrams/learn-pipeline.png', published.url()).href);
  assert.equal(siteImage.status(), 200);
  assert.equal(siteImage.headers()['content-type'], 'image/png');
  assert.deepEqual([...(await siteImage.body()).subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  await published.screenshot({ path: path.join(artifacts, 'site-mode.png') });
  await published.close();
  // The optional editor host uses native messages instead of file/protocol navigation.
  const hosted = await browser.newContext({ viewport: { width: 1440, height: 960 } });
  await hosted.addInitScript(() => {
    window.__editorMessages = []; window.__editorAcquisitions = 0;
    window.acquireVsCodeApi = () => {
      ++window.__editorAcquisitions;
      return { postMessage: message => window.__editorMessages.push(message) };
    };
  });
  const native = await hosted.newPage();
  native.on('pageerror', error => errors.push(error.message));
  await native.goto(`http://127.0.0.1:${server.address().port}`);
  assert.equal(await native.evaluate(() => window.__editorAcquisitions), 1);
  assert.equal(await native.locator('[data-copy-location],.source-location').count(), 0);
  await chooseView(native, 'learn-token');
  const teachingTarget = native.locator('#canvas [data-learning-node]:visible').first();
  const teachingId = await teachingTarget.getAttribute('data-learning-node');
  await teachingTarget.click();
  assert.equal(await native.evaluate(() => window.__reader.selected().id), teachingId);
  await chooseView(native, 'learn-pipeline');
  await native.locator('[data-overview-direction="backward"]').click();
  await native.locator('[data-overview-chart] [data-wait-for="ptrd-reuse"]').first().click();
  assert.equal(await native.evaluate(() => window.__reader.selected().id), 'b_reuse_ptrd');
  await openExercise(native);
  const nativeSim = native.locator('[data-gmm2-pipeline]');
  await nativeSim.locator('[data-wait-for="slot"]').first().click();
  assert.equal(await native.evaluate(() => window.__reader.selected().id), 'f_gmm2');
  await nativeSim.locator('[data-wait-for="output"]').first().click();
  assert.equal(await native.evaluate(() => window.__reader.selected().id), 'f_combine');
  assert.deepEqual(await native.evaluate(() => window.__editorMessages), []);
  await nativeSim.getByRole('button', { name: '查看对应源码' }).click();
  assert.deepEqual(await native.evaluate(() => window.__editorMessages.at(-1)), { type: 'openSource', nodeId: 'f_combine', sourceIndex: 1 });
  await chooseView(native, '00-overview');
  const graphNode = await native.locator('#canvas [data-node]').nth(1).getAttribute('data-node');
  await native.locator(`#canvas [data-node="${graphNode}"]`).click();
  assert.equal(await native.evaluate(() => window.__reader.selected().id), graphNode);
  assert.equal(await native.evaluate(() => window.__editorMessages.length), 1);
  await native.locator('[data-open-source]').first().click();
  assert.deepEqual(await native.evaluate(() => window.__editorMessages.at(-1)), { type: 'openSource', nodeId: graphNode, sourceIndex: 0 });
  await native.locator('#source-map').click();
  assert.deepEqual(await native.evaluate(() => window.__editorMessages.at(-1)), { type: 'openMapping' });
  await native.screenshot({ path: path.join(artifacts, 'editor-bridge.png') });
  await hosted.close();
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(artifacts, 'browser-check.json'), JSON.stringify({ ok: true, errors, views: reports, scenes: sceneReports, fullForward: true, fullBackward: true, gmm2Simulation: true, offline: true, site: true, mobile: true, editorBridge: true, drilldown: targets.length, downloads: true }, null, 2) + '\n');
  console.log(`Browser PASS: ${reports.length} source views + ${sceneReports.length} learning scenes; full forward/backward, GMM2 zoom, offline, site, mobile and native bridge.`);
} catch (error) {
  fs.writeFileSync(path.join(artifacts, 'browser-error.txt'), String(error.stack));
  throw error;
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
