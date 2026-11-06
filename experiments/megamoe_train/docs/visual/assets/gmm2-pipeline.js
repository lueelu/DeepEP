// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

// Each lane advances independently. A gate records when its producer retires.
const pipelineSource = (node, index = 0) => ({ node, index });
function createPipelineSchedule() {
  const events = [];
  const cursors = new Map();
  const time = lane => cursors.get(lane) || 0;
  const add = (lane, start, end, event) => {
    if (start < end) events.push({ ...event, lane, start, end });
  };
  const run = (lane, ticks, event) => {
    const start = time(lane), end = start + ticks;
    add(lane, start, end, event);
    cursors.set(lane, end);
    return end;
  };
  const wait = (lane, readyAt, event) => run(lane, Math.max(0, readyAt - time(lane)), { ...event, kind: 'wait' });
  return { events, time, add, run, wait };
}

function waitForDispatchUb(schedule, lane, readyAt) {
  schedule.wait(lane, readyAt, {
    label: '等 Dispatch UB', waitFor: 'dispatch-ub', signal: 'WaitForVector(GMM2_UB_READY_FLAG + FLAG_ID_MAX)',
    detail: '本核 AIV1 排空 Dispatch 管线并释放 UB 后，调用 NotifyCube(GMM2_UB_READY_FLAG)。',
    source: pipelineSource('f_gmm2_wait'),
  });
}

function scheduleGmm2Tiles(schedule, lanes, tiles, computeTicks, combineTicks) {
  const { aic, aiv1, slots } = lanes;
  const returnedAt = [0, 0];
  for (const [tile, task] of tiles.entries()) {
    const slot = tile % 2;
    const tag = task.expert === undefined ? '' : `e${task.expert} `;
    if (task.hpReadyAt !== undefined) schedule.wait(aic, task.hpReadyAt, {
      stage: 'gmm2', label: `${tag}等 Hp`, waitFor: 'hp', signal: 'swigluFlags == target',
      detail: '当前专家所有 SwiGLU 任务完成 Hp 写入后，计数达到 GMM2 读取的目标值。',
      source: pipelineSource('f_gmm2_wait', 1),
    });
    schedule.wait(aic, returnedAt[slot], {
      stage: 'gmm2', label: `等槽${slot}`, tile, slot, waitFor: 'slot',
      signal: `WaitForVector(${slot} + FLAG_ID_MAX)`,
      detail: `tile${tile} 要复用槽${slot}。AIV1 发完 tile${tile - 2}、等待 MTE3 退休，再用 NotifyCube(${slot}) 归还槽。`,
      source: pipelineSource('f_gmm2'),
    });
    const computeStart = schedule.time(aic);
    const outputReadyAt = schedule.run(aic, computeTicks, {
      stage: 'gmm2', kind: 'compute', label: `${tag}算 tile${tile}`, tile, slot, expert: task.expert,
      signal: `NotifyVector(${slot} + FLAG_ID_MAX)`,
      detail: `AIC 计算 tile${tile}，结果写入槽${slot}，随后通知 AIV1。`,
      source: pipelineSource('f_gmm2'),
    });
    schedule.wait(aiv1, outputReadyAt, {
      stage: 'combine', label: `等 tile${tile}`, tile, slot, waitFor: 'output',
      signal: `WaitForCube(${slot})`,
      detail: `AIV1 等 tile${tile} 写入槽${slot}。AIC 计算完成后用 NotifyVector(${slot} + FLAG_ID_MAX) 发布结果。`,
      source: pipelineSource('f_combine', 1),
    });
    const sentAt = schedule.run(aiv1, combineTicks, {
      stage: 'combine', kind: 'send', label: `${tag}发 tile${tile}`, tile, slot, expert: task.expert,
      signal: `MTE3_V → NotifyCube(${slot})`,
      detail: `AIV1 从槽${slot} 读取 tile${tile} 并回传各源 rank。MTE3 完成后归还槽。`,
      source: pipelineSource('f_combine', 1),
    });
    schedule.add(slots[slot], computeStart, sentAt, {
      stage: 'combine', kind: 'hold', label: `${tag}tile${tile}`, tile, slot,
      signal: `NotifyCube(${slot})`,
      detail: `从 AIC 开始计算到 AIV1 发送退休，槽${slot} 始终由 tile${tile} 占用。`,
      source: pipelineSource('f_combine', 1),
    });
    returnedAt[slot] = sentAt;
  }
}

// The isolated exercise starts after the two entry gates by default.
function simulateGmm2Combine({ computeTicks, combineTicks, dispatchReadyAt = 0, hpReadyAt = 0, tileCount = 6 }) {
  const schedule = createPipelineSchedule();
  waitForDispatchUb(schedule, 'aic', dispatchReadyAt);
  schedule.run('aiv1', dispatchReadyAt, { kind: 'send', label: 'Dispatch', signal: 'GMM2_UB_READY_FLAG',
    detail: '本核 AIV1 完成 Dispatch 后释放 UB。', source: pipelineSource('f_release_dispatch_ub') });
  const tiles = Array.from({ length: tileCount }, (_, tile) => tile === 0 ? { hpReadyAt } : {});
  scheduleGmm2Tiles(schedule, { aic: 'aic', aiv1: 'aiv1', slots: ['slot0', 'slot1'] }, tiles, computeTicks, combineTicks);
  const { events } = schedule;
  const waitTicks = reason => events.filter(event => event.waitFor === reason)
    .reduce((total, event) => total + event.end - event.start, 0);
  return {
    lanes: [ ['aic', 'AIC'], ['aiv1', 'AIV1'], ['slot0', 'UB 槽 0'], ['slot1', 'UB 槽 1'] ],
    events, finish: schedule.time('aiv1'), slotWait: waitTicks('slot'), outputWait: waitTicks('output'),
  };
}

function renderShortWait(anchor, control, attributes) {
  return `<div class="pipeline-short-wait">
    <span class="pipeline-short-wait-line" style="left:${anchor}%" aria-hidden="true"></span>
    <svg class="pipeline-short-wait-leader" viewBox="0 0 100 13" preserveAspectRatio="none" aria-hidden="true">
      <path d="M${control} 13 L${anchor} 7 V2" vector-effect="non-scaling-stroke"/>
    </svg>
    <button type="button" class="pipeline-short-wait-button" style="left:${control}%" ${attributes}>等</button>
  </div>`;
}

function renderPipelineTimeline({ lanes, events, finish, origin = 0, pixelsPerTick = 14, overlaps = [], shortWaitTicks = 0 }) {
  const span = finish - origin;
  const percent = ticks => 100 * (ticks - origin) / span;
  const ticks = Array.from({ length: 6 }, (_, index) => origin + Math.round(span * index / 5));
  const axis = ticks.map(tick => `<span style="left:${percent(tick)}%">${tick}</span>`).join('');
  const guides = ticks.slice(1, -1).map(tick => `<i style="left:${percent(tick)}%"></i>`).join('');
  const overlapBands = overlaps.map(({ start, end }) =>
    `<i class="pipeline-overlap-band" style="left:${percent(start)}%;width:${100 * (end - start) / span}%"></i>`).join('');
  const isShortWait = event => event.kind === 'wait' && event.end - event.start <= shortWaitTicks;
  const rows = lanes.map(([id, label]) => {
    const shortCount = events.filter(event => event.lane === id && isShortWait(event)).length;
    // Give the controls their own rail so nearby waits keep separate hit targets.
    // The gap leaves room for a 24px button even at the minimum chart width.
    const gap = 6, inset = gap / 2;
    let previousControl = inset - gap, remaining = shortCount;
    const bars = events.map((event, index) => {
      if (event.lane !== id) return '';
      const title = `${label} · ${event.label} · ${event.start}–${event.end} tick`;
      const attributes = `data-sim-event="${index}" data-stage="${event.stage || ''}" data-wait-for="${event.waitFor || ''}" data-start="${event.start}" data-end="${event.end}" aria-pressed="false" aria-label="${title}" title="${title}"`;
      if (isShortWait(event)) {
        const anchor = percent((event.start + event.end) / 2);
        remaining -= 1;
        const lastAllowed = 100 - inset - remaining * gap;
        const control = Math.max(previousControl + gap, Math.min(anchor, lastAllowed));
        previousControl = control;
        return renderShortWait(anchor, control, attributes);
      }
      return `<button type="button" class="gmm2-segment ${event.kind}" ${attributes} style="left:${percent(event.start)}%;width:${100 * (event.end - event.start) / span}%">${event.shortLabel ?? event.label}</button>`;
    }).join('');
    return `<div class="gmm2-lane"><strong>${label}</strong><div class="gmm2-track${shortCount ? ' with-short-waits' : ''}">${guides}${overlapBands}${bars}</div></div>`;
  }).join('');
  return `<div class="gmm2-chart" style="min-width:${Math.max(560, span * pixelsPerTick + 82)}px"><div class="gmm2-lane gmm2-axis"><strong>tick</strong><div>${axis}</div></div>${rows}</div>`;
}

function showPipelineEvent(chart, detail, index, events, inspectSource, formula) {
  const segment = events[index];
  for (const bar of chart.querySelectorAll('[data-sim-event]')) bar.setAttribute('aria-pressed', String(Number(bar.dataset.simEvent) === index));
  detail.innerHTML = `<div><strong>${segment.label}</strong><span>${segment.start}–${segment.end} tick</span></div><p>${segment.kind === 'wait' ? '等待信号' : '交接信号'} <code>${segment.signal}</code></p><p>${segment.detail}</p><button type="button" class="learning-source-chip">查看对应源码 ↗</button>`;
  inspectSource(segment.source);
  const openSource = detail.querySelector('button');
  openSource.onclick = () => inspectSource(segment.source, true);
  if (formula) {
    const block = document.createElement('pre');
    block.className = 'formula';
    block.textContent = formula;
    openSource.before(block);
  }
}

function mountGmm2Pipeline(panel, inputs, inspectSource) {
  const compute = panel.querySelector('#gmm2-compute');
  const combine = panel.querySelector('#gmm2-combine');
  const chart = panel.querySelector('.gmm2-chart-scroll');
  const detail = panel.querySelector('.gmm2-selection');
  const hint = '点一段等待，查看它等的信号和解除条件。';
  let simulation;

  function redraw() {
    compute.value = inputs.computeTicks;
    combine.value = inputs.combineTicks;
    panel.querySelector('#gmm2-compute-value').value = `${compute.value} tick`;
    panel.querySelector('#gmm2-combine-value').value = `${combine.value} tick`;
    simulation = simulateGmm2Combine(inputs);
    chart.innerHTML = renderPipelineTimeline(simulation);
    panel.querySelector('.gmm2-summary').innerHTML = `<span>最后一块发完 <strong>${simulation.finish}</strong> tick</span><span>AIC 等槽 <strong>${simulation.slotWait}</strong> tick</span><span>AIV1 等输出 <strong>${simulation.outputWait}</strong> tick</span>`;
    detail.textContent = hint;
    for (const preset of panel.querySelectorAll('[data-gmm2-preset]')) {
      preset.setAttribute('aria-pressed', String(preset.dataset.gmm2Preset === `${compute.value},${combine.value}`));
    }
  }

  compute.oninput = combine.oninput = () => {
    inputs.computeTicks = compute.valueAsNumber;
    inputs.combineTicks = combine.valueAsNumber;
    redraw();
  };
  for (const preset of panel.querySelectorAll('[data-gmm2-preset]')) {
    preset.onclick = () => {
      [inputs.computeTicks, inputs.combineTicks] = preset.dataset.gmm2Preset.split(',').map(Number);
      redraw();
    };
  }
  chart.onclick = event => {
    const button = event.target.closest('[data-sim-event]');
    if (!button) return;
    showPipelineEvent(chart, detail, Number(button.dataset.simEvent), simulation.events, inspectSource);
  };
  redraw();
}
