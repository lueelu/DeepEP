// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

// Two representative core groups per rank feed the local and cross-rank gates.
const pipelineParts = [0, 1];
// Several representative GMM1 tasks can read the same ready input partition.
const gmm1TasksPerPart = 4;
const gmm1TasksPerExpert = pipelineParts.length * gmm1TasksPerPart;
const gmm2TasksPerExpert = 8;
// Defaults follow the phase proportions in the forward/backward timer slides.
// They are teaching ticks for the EP8 case, independent of the slide's timebase.
const forwardTiming = {
  route: { label: '路由与权重整理', ticks: 5, kind: 'prepare', node: 'f_routes', signal: 'sorted routes / packed p', detail: '排列 X 与路由权重 p，准备专家计数和源端逆置换索引。' },
  counts: { label: '交换计数与保存', ticks: 4, kind: 'prepare', node: 'f_counts', signal: 'CrossRankSyncCntPerExpert → SyncAll', detail: '交换专家计数并生成接收偏移。AIV0 保存稳定的计数，之后进入 Dispatch 与 GMM1。' },
  dispatch: { label: 'Dispatch X + p', ticks: 8, tasks: pipelineParts.length, kind: 'send', node: 'f_dispatch_x', signal: 'MTE3_S → row ready', detail: 'AIV1 搬运本段 X 和 p，写入退休后再发布输入就绪。' },
  gmm1: { label: 'GMM1 Gate / Up', ticks: 8, tasks: gmm1TasksPerExpert, kind: 'cube1', node: 'f_gate_up', signal: 'NotifyVector()', detail: 'AIC 在同一个任务内计算 Gate 和 Up 两片，交给配对 AIV0。' },
  swiglu: { label: 'SwiGLU 与保存', ticks: 1, tasks: gmm1TasksPerExpert, kind: 'vector', node: 'f_swiglu_p', signal: 'Hp 写退休 → groupFlagList++', detail: 'AIV0 保存原始 Gate/Up，计算含 p 的 Hp，完成写入后发布计数并归还 UB。窗口中的空档主要在等 AIC 的下一块 Gate/Up。' },
  gmm2: { label: 'GMM2', ticks: 4, tasks: gmm2TasksPerExpert, signal: 'NotifyVector(pingpongIdx + FLAG_ID_MAX)', detail: 'AIC 将输出交给 AIV1。继续复用同一槽时，要等上一份输出发完。' },
  combine: { label: 'Combine', ticks: 4, tasks: gmm2TasksPerExpert, signal: 'MTE3_V → NotifyCube(pingpongIdx)', detail: 'AIV1 把专家输出发回各源 rank，发送退休后归还槽。' },
  y: { label: 'Unpermute 写 Y', ticks: 5, kind: 'finish', node: 'f_unpermute', index: 1, signal: 'UnpermuteV2 → Y', detail: '所有 BF16 route 回传到达后，源 rank 按 TopK 槽用 FP32 累加，最后一次舍入写 BF16 Y。' },
  peer: { label: 'rank1 回传额外耗时', ticks: 0, tasks: gmm2TasksPerExpert },
};

function createOverview(timing, inputs, shape) {
  const ticks = Object.fromEntries(Object.entries(timing).map(([stage, spec]) => [stage, inputs[stage] ?? spec.ticks]));
  const schedule = createPipelineSchedule();
  const experts = Array.from({ length: shape.expertsPerRank }, (_, expert) => expert);
  const groups = Array.from({ length: shape.ep * 2 }, (_, index) => {
    const prefix = `r${Math.floor(index / 2)}g${index % 2}`;
    return { rank: Math.floor(index / 2), group: index % 2, aic: `${prefix}-aic`, aiv0: `${prefix}-aiv0`,
      aiv1: `${prefix}-aiv1`, slots: [`${prefix}-ub0`, `${prefix}-ub1`] };
  });
  const localGroups = group => groups.filter(peer => peer.rank === group.rank);
  const latest = (selected, finish) => Math.max(...selected.map(finish));
  const stage = (lane, key, label, extra = 0, expert) => {
    const spec = timing[key];
    return schedule.run(lane, ticks[key] + extra, { stage: key, label, kind: spec.kind, expert,
      signal: spec.signal, detail: spec.detail, source: pipelineSource(spec.node, spec.index) });
  };
  const wait = (lane, at, reason, label, signal, detail, node, index = 0) => schedule.wait(lane, at,
    { waitFor: reason, label, signal, detail, source: pipelineSource(node, index) });
  return { ticks, schedule, groups, experts, localGroups, latest, stage, wait };
}

function simulateFullForward(inputs, shape) {
  const { schedule, groups, experts, localGroups, latest, stage, wait, ticks } = createOverview(forwardTiming, inputs, shape);
  for (const group of groups) {
    stage(group.aiv0, 'route', '路由 / p');
    stage(group.aiv1, 'route', '路由 / p');
    stage(group.aiv0, 'counts', '计数 / 保存');
    stage(group.aiv1, 'counts', '计数');
    group.prepared = Math.max(schedule.time(group.aiv0), schedule.time(group.aiv1));
  }

  for (const group of groups) {
    const preparedAt = latest(localGroups(group), peer => peer.prepared);
    wait(group.aiv1, preparedAt, 'forward-prep', '等计数', 'SyncAll<false>()',
      '同 rank 的计数和保存完成后，释放前置阶段使用的 UB。', 'f_counts', 1);
    wait(group.aic, preparedAt, 'forward-prep', '等准备', 'SyncAll<false>()',
      '等待同 rank 的专家计数、偏移和保存状态完成。', 'f_counts', 1);
    group.xReady = experts.map(expert => pipelineParts.map(part =>
      stage(group.aiv1, 'dispatch', `e${expert} X+p ${part + 1}`, group.group, expert)));
    group.dispatchUbReady = schedule.time(group.aiv1);
  }

  for (const group of groups) {
    let gateUpFreeAt = 0;
    group.hpReady = experts.map(expert => {
      for (const part of pipelineParts) {
        const xReadyAt = latest(localGroups(group), peer => peer.xReady[expert][part]);
        wait(group.aic, xReadyAt, 'x-ready', `e${expert} 等 X+p`, 'Ready384 计数达到目标行数',
          `e${expert} 的代表任务等待输入块 ${part + 1}。对应各搬运段的 X 与 p 写入齐全后，AIC 才能读取。`, 'f_wait_xp');
        for (let task = 0; task < gmm1TasksPerPart; task++) {
          wait(group.aic, gateUpFreeAt, 'gate-up-ub', '等上一块 Gate/Up UB', 'WaitForVector()',
            '只等配对 AIV0 完成上一块的 epilogue 并 NotifyCube()，跨专家时也沿用同一个 UB 交接。', 'f_gate_up', 1);
          const gateUpReadyAt = stage(group.aic, 'gmm1', `e${expert} GMM1`, 0, expert);
          wait(group.aiv0, gateUpReadyAt, 'gate-up-ready', `e${expert} 等 Gate/Up`, 'WaitForCube()',
            'AIC 写完同一任务的 Gate 和 Up，再用 NotifyVector() 交给 AIV0。', 'f_gate_up');
          gateUpFreeAt = stage(group.aiv0, 'swiglu', `e${expert} SwiGLU`, 0, expert);
        }
      }
      return gateUpFreeAt;
    });
  }

  for (const group of groups) {
    // This AIC has already traversed all of its GMM1 tasks.
    waitForDispatchUb(schedule, group.aic, group.dispatchUbReady);
    const tiles = experts.flatMap(expert => Array.from({ length: gmm2TasksPerExpert }, (_, task) => ({
      expert, hpReadyAt: task === 0 ? latest(localGroups(group), peer => peer.hpReady[expert]) : undefined,
    })));
    // Slot ownership carries across expert boundaries.
    scheduleGmm2Tiles(schedule, group, tiles, ticks.gmm2, ticks.combine + (group.rank === 1 ? ticks.peer : 0));
    group.combineDone = schedule.time(group.aiv1);
    wait(group.aic, group.combineDone, 'combine-tail', '等 Combine 尾槽', 'EndGMM2Sync()',
      '本核结束 GMM2 计算后，等 AIV1 回传尾块并归还 UB 槽，随后进入收尾同步。', 'f_unpermute', 2);
  }

  const routesArrived = latest(groups, group => group.combineDone);
  for (const group of groups) {
    for (const lane of [group.aiv0, group.aiv1]) {
      wait(lane, routesArrived, 'y-arrived', '等 Y 回传', 'SyncAll + CrossRankSyncInWorldSize()',
        '本地发送尾部全部退休，并收到对端完成信号后，源 rank 才读 ptrD 的 route 输出。', 'f_unpermute');
      stage(lane, 'y', '归并写 Y', group.group);
    }
    group.outputAt = schedule.time(group.aiv1);
  }
  const finish = latest(groups, group => group.outputAt);
  for (const group of groups) {
    for (const lane of [group.aiv0, group.aiv1]) wait(lane, finish, 'forward-exit', '等 Y 读写退休', 'CrossRankSyncInWorldSize()',
      '所有 rank 结束 Unpermute，当前调用的 SHMEM 区域才可交给下一轮。', 'f_exit');
  }
  return { groups, events: schedule.events, finish };
}

const backwardTiming = {
  gather: { label: 'Gather 原始 dY', ticks: 5, kind: 'prepare', node: 'b_gather', signal: 'ptrA0 写退休', detail: 'AIV0 和 AIV1 都按前向路由收集原始 dY 到 ptrA0。路由权重 p 留给 SwiGLU 反向。' },
  restore: { label: '恢复计数与行段', ticks: 4, kind: 'prepare', node: 'b_restore_counts', signal: '计数和 producer 元数据 ready', detail: '两路 AIV 按源 rank 分工，从前向保存的计数恢复接收位置，并为符合条件的专家准备行段元数据。' },
  dispatch: { label: 'Dispatch dY', ticks: 3, tasks: pipelineParts.length, kind: 'send', node: 'b_dispatch_do', signal: 'MTE3_S → producer row ready', detail: 'AIV0 从对端 ptrA0 拉取 dY，写入本专家 dO，并发布完成的行段。AIV1 同期可处理 dHp 已就绪的专家。' },
  dhp: { label: 'AIC 计算 dHp', ticks: 6, tasks: pipelineParts.length, kind: 'cube1', node: 'b_dhp', signal: 'Fixpipe GM 写退休 → dHpReady', detail: 'AIC 读取 dO 与 W2 计算 dHp。每个专家的全部 AIC 写完后，AIV 才能读这个专家的 dHp。' },
  epi: {
    label: 'SwiGLU 反向', ticks: 5, tasks: 1, kind: 'vector', node: 'b_epi0',
    signal: 'dHpReady >= blockNum → SwiGLU Grad → DZReady++',
    detail: [
      '<strong>为什么用两路 AIV</strong>　反向每块要重建 Hp，还要算 dGate/dUp、归约 dp partial。同样的专家行数下，前向 GMM1 产出 Gate 和 Up 共 2F 列，反向 dHp 只有 F 列，GEMM 计算量约为一半。AIC 的 dHp 阶段通常更短，留给 SwiGLU Grad 的重叠窗口也更短。dHp 写在 GM 后，两路 AIV 可以并行读各自的 tile。AIV0 还在 Dispatch 时，AIV1 可先做已就绪的专家。',
      '<strong>tile 怎么分</strong>　同一核组按本地任务序号 <code>nextTargetAiv &amp; 1U</code> 轮转，AIV0 做 0、2、4…，AIV1 做 1、3、5…。序号跨专家延续，两路执行同一个 SwiGLU Grad。',
      '<strong>dW2 等什么</strong>　每个 AIV 在本专家的 dZ、Hp、dp partial 写完并退休后给 <code>DZReady</code> 加 1。AIC 等 <code>DZReady &gt;= 2 × blockNum</code>，再用原始 dO 和 Hp 算 dW2。',
    ].join('<br>'),
  },
  dw2: { label: 'AIC 计算 dW2', ticks: 12, tasks: 1, kind: 'weight2', node: 'b_dw2', signal: 'PIPE_ALL → doReleased++', detail: '用原始 dO 和重建 Hp 算 dW2。排空本核的 dO 读取后增加 doReleased。' },
  dx: { label: 'AIC 计算 dX', ticks: 24, tasks: 1, kind: 'compute', node: 'b_dx', signal: 'Fixpipe GM 写退休 → DXReady', detail: '用 dZ 和 W1 计算 dX，写入原 dO 行段。各 AIC 完成 GM 写后发布到达计数。' },
  returnDx: { label: 'Combine 回传 dX', ticks: 4, tasks: 1, kind: 'return', node: 'b_dx_send', signal: 'dX 回传 MTE3 退休', detail: 'AIV0 等整专家 DXReady 后才读取 GM 中的 dX，并发回源 rank。' },
  dw1: { label: 'AIC 计算 dW1', ticks: 24, tasks: 1, kind: 'weight1', node: 'b_dw1', signal: 'dW1 写入本地专家梯度', detail: '本核走完逐专家 dW2/dX 后，用 dZ 与 packed X 计算 dW1。dW1、dW2 留在专家 rank。' },
  unpermuteDx: { label: 'Unpermute 归并 dX', ticks: 4, kind: 'finish', node: 'b_unpermute_dx', signal: 'dX 读完 → SyncAll', detail: '从 ptrD 取回各路 BF16 dX，按 TopK 槽 FP32 累加，最后一次舍入写 grad_x。' },
  dp: { label: '归约并回传 dp', ticks: 2, kind: 'return', node: 'b_reduce_dp', signal: 'dp 回传 MTE3 退休', detail: '用 FP32 跨 tile 累加 dp partial，按输入 p 类型写出：BF16 舍入、FP32 保留，再复用 ptrD 按原始 route 回传。' },
  outputDp: { label: '还原概率梯度', ticks: 1, kind: 'finish', node: 'b_dp_output', signal: 'UnpermuteTopkGrad', detail: 'dp 按源 rank 的 token 与 TopK 槽还原为 dtopk_weights。' },
  cleanup: { label: '清理计数', ticks: 1, kind: 'prepare', node: 'b_dp_output', index: 1, signal: '清理 → 退出同步', detail: '最后一轮读写退休后重置计数，建立当前调用与下一轮的边界。' },
  peer: { label: 'rank1 dX 回传额外耗时', ticks: 0, tasks: 1 },
};

function simulateFullBackward(inputs, shape) {
  const { schedule, groups, experts, localGroups, latest, stage, wait, ticks } = createOverview(backwardTiming, inputs, shape);
  for (const group of groups) {
    stage(group.aiv0, 'gather', 'Gather dY');
    group.gathered = stage(group.aiv1, 'gather', 'Gather dY');
  }
  const gatheredAt = latest(groups, group => group.gathered);
  for (const group of groups) {
    for (const lane of [group.aiv0, group.aiv1]) {
      wait(lane, gatheredAt, 'dy-arrived', '等 dY 整理', 'CrossRankSyncInWorldSize()',
        '所有 rank 先把原始 dY 整理到 ptrA0，随后才可从对端拉取。', 'b_gather');
      stage(lane, 'restore', '恢复计数');
    }
    group.restored = schedule.time(group.aiv0);
  }

  for (const group of groups) {
    const restoredAt = latest(localGroups(group), peer => peer.restored);
    wait(group.aic, restoredAt, 'producer-metadata', '等行段信息', 'dHpReady cacheline word1',
      '计数和 producer 元数据准备齐全。读取 dO 还需等对应行段搬运完成。', 'b_restore_counts');
    group.doReady = experts.map(expert => pipelineParts.map(part =>
      stage(group.aiv0, 'dispatch', `e${expert} dY ${part + 1}`, group.group, expert)));
  }

  for (const group of groups) {
    group.dhpReady = experts.map(expert => {
      for (const part of pipelineParts) {
        const readyAt = latest(localGroups(group), peer => peer.doReady[expert][part]);
        wait(group.aic, readyAt, 'do-ready', `e${expert} 等 dO`, 'producer row progress',
          `e${expert} 的输入段 ${part + 1} 写入 GM 后，AIC 才能读取该段 dO。`, 'b_dhp');
        stage(group.aic, 'dhp', `e${expert} dHp`, 0, expert);
      }
      return schedule.time(group.aic);
    });
  }

  for (const group of groups) {
    group.dzReady = experts.map(expert => {
      const dhpAt = latest(localGroups(group), peer => peer.dhpReady[expert]);
      for (const [lane, node] of [[group.aiv0, 'b_epi0'], [group.aiv1, 'b_epi1']]) {
        wait(lane, dhpAt, 'dhp-ready', `e${expert} 等 dHp`, 'dHpReady >= blockNum',
          `e${expert} 的全部 AIC 都完成 dHp 写入后，本路 AIV 才读取自己分领的 GM tile。`, 'b_dhp_retire', 1);
        const spec = backwardTiming.epi;
        schedule.run(lane, ticks.epi + group.group, { stage: 'epi', kind: spec.kind, expert, label: `e${expert} SwiGLU Grad`,
          signal: spec.signal, detail: spec.detail, source: pipelineSource(node) });
      }
      return Math.max(schedule.time(group.aiv0), schedule.time(group.aiv1));
    });
  }

  for (const expert of experts) {
    for (const group of groups) {
      const dzAt = latest(localGroups(group), peer => peer.dzReady[expert]);
      wait(group.aic, dzAt, 'dz-ready', `e${expert} 等 dZ/Hp`, 'DZReady >= 2 × blockNum',
        '两路 AIV 的 dZ、Hp 和 dp partial 写完并退休后，AIC 才进入该专家的 dW2。', 'b_wait_dz');
      group.doReleased = stage(group.aic, 'dw2', `e${expert} dW2`, group.group, expert);
    }
    for (const group of groups) {
      const releasedAt = latest(localGroups(group), peer => peer.doReleased);
      wait(group.aic, releasedAt, 'do-released', `e${expert} 等 dO 退休`, 'doReleased >= blockNum',
        '所有 AIC 已完成该专家的 dW2 和 dO 读取，才允许把 dO 行段覆盖成 dX。', 'b_do_release');
      group.dxReady = stage(group.aic, 'dx', `e${expert} dX`, group.group, expert);
    }
    for (const group of groups) {
      const dxAt = latest(localGroups(group), peer => peer.dxReady);
      wait(group.aiv0, dxAt, 'dx-ready', `e${expert} 等 dX`, 'DXReady >= blockNum',
        `AIC 先完成 e${expert} 的 dW2，等 dO 读完才能写 dX。全部 AIC 发布 DXReady 后，AIV0 开始回传 e${expert}。`, 'b_dx_send', 1);
      stage(group.aiv0, 'returnDx', `e${expert} 回传 dX`, group.rank === 1 ? ticks.peer : 0, expert);
    }
  }

  for (const group of groups) {
    for (const expert of experts) stage(group.aic, 'dw1', `e${expert} dW1`, group.group, expert);
    group.dxSent = Math.max(schedule.time(group.aiv0), schedule.time(group.aiv1));
  }
  const dxArrivedAt = latest(groups, group => group.dxSent);
  for (const group of groups) {
    const localDxAt = latest(localGroups(group), peer => peer.dxSent);
    for (const lane of [group.aiv0, group.aiv1]) {
      wait(lane, localDxAt, 'dx-local', '等本卡回传', 'PIPE_ALL → SyncAll<true>()',
        '本 rank 两路 AIV 的发送管线都退休，才可对其他 rank 发布完成。', 'b_dx_arrived');
      wait(lane, dxArrivedAt, 'dx-peer', '等对端 dX', 'CrossRankSyncInWorldSize()',
        '对端 dX 全部写入 ptrD 后，源端才可以归并。', 'b_dx_arrived');
      stage(lane, 'unpermuteDx', '归并写 dX', group.group);
    }
    group.dxRead = schedule.time(group.aiv0);
  }
  const allDxReadAt = latest(groups, group => group.dxRead);
  for (const group of groups) {
    for (const lane of [group.aiv0, group.aiv1]) {
      wait(lane, allDxReadAt, 'ptrd-reuse', '等 ptrD 读完', 'P5 → P6 CrossRankSyncInWorldSize()',
        '确认所有 rank 已读完 dX，再把同一块 ptrD 改作 dp 返回区。', 'b_reuse_ptrd');
      stage(lane, 'dp', '归约 / 回传 dp', group.group);
    }
    group.dpSent = schedule.time(group.aiv0);
  }
  const dpArrivedAt = latest(groups, group => group.dpSent);
  for (const group of groups) {
    for (const lane of [group.aiv0, group.aiv1]) {
      wait(lane, dpArrivedAt, 'dp-arrived', '等 dp 回传', 'CrossRankSyncInWorldSize()',
        '跨 rank 的 dp 写入全齐后，源端才能读取 ptrD 的 route 标量。', 'b_dp_output', 1);
      stage(lane, 'outputDp', '还原 dp');
      stage(lane, 'cleanup', '清理');
    }
    group.outputAt = schedule.time(group.aiv0);
  }
  const clearedAt = latest(groups, group => group.outputAt);
  for (const group of groups) {
    for (const lane of [group.aiv0, group.aiv1]) wait(lane, clearedAt, 'backward-exit', '等计数清理', 'CrossRankSyncInWorldSize()',
      '所有 rank 清理完成后，下一轮才能使用这份共享窗口。', 'b_dp_output', 1);
  }
  const finish = latest(groups, group => Math.max(schedule.time(group.aic), schedule.time(group.aiv0), schedule.time(group.aiv1)));
  return { groups, events: schedule.events, finish };
}

function expertOverviewEvents(events, lanes, timing) {
  const ids = new Set(lanes.map(([id]) => id));
  const compact = { route: '路由', counts: '计数', y: 'Y', gather: 'dY', restore: '计数', unpermuteDx: 'dX', dp: 'dp', outputDp: '重排', cleanup: '清' };
  const waitLabels = { 'dhp-ready': '等 dHp', 'dx-ready': '等 dX', 'dx-local': '等本卡回传', 'dx-peer': '等跨卡回传' };
  const blocks = [], waits = [], windows = new Map();
  for (const event of events) {
    if (!ids.has(event.lane)) continue;
    if (event.kind === 'wait') {
      if (event.waitFor === 'output' && event.tile === 0) {
        waits.push({ ...event, label: '等首份 GMM2 输出', shortLabel: '等输出',
          detail: 'AIV1 完成 Dispatch 后，等 AIC 发布首份 GMM2 输出，随后开始 Combine。' });
      } else if (event.waitFor !== 'slot' && event.waitFor !== 'output') {
        waits.push({ ...event, shortLabel: waitLabels[event.waitFor] ?? '等' });
      }
      continue;
    }
    if (event.expert === undefined) {
      blocks.push({ ...event, shortLabel: compact[event.stage] });
      continue;
    }
    const key = `${event.lane}/${event.stage}/${event.expert}`;
    const spec = timing[event.stage];
    const block = windows.get(key) ?? { ...event, kind: event.stage === 'combine' ? 'return' : event.kind,
      label: `e${event.expert} · ${spec.label}`, shortLabel: `e${event.expert}`,
      signal: spec.signal, detail: spec.detail, work: 0 };
    block.end = event.end;
    block.work += event.end - event.start;
    windows.set(key, block);
  }
  for (const block of windows.values()) {
    const span = block.end - block.start;
    blocks.push({ ...block, detail: `${block.detail} 阶段窗口 ${span} tick，工作 ${block.work} tick，内部等待 ${span - block.work} tick。` });
  }
  // A wait inside an expert window is included in its duration.
  return blocks.concat(waits.filter(wait => !blocks.some(block =>
    block.lane === wait.lane && block.start <= wait.start && wait.end <= block.end)))
    .sort((a, b) => a.start - b.start);
}

function mountFullPipeline(panel, state, shape, inspectSource, formulas) {
  const chart = panel.querySelector('[data-overview-chart]');
  const detail = panel.querySelector('[data-overview-detail]');
  const stagePicker = panel.querySelector('[data-overview-stage]');
  const duration = panel.querySelector('[data-overview-ticks]');
  const tickValue = panel.querySelector('[data-overview-tick-value]');
  const waitPicker = panel.querySelector('[data-overview-waits]');
  let timing, events;

  function editStage() {
    const key = stagePicker.value;
    const { ticks, tasks } = timing[key];
    const step = tasks ?? 1;
    duration.min = key === 'peer' ? 0 : step;
    duration.max = Math.max(20, ticks * 2) * step;
    duration.step = step;
    duration.value = (state.inputs[state.direction][key] ?? ticks) * step;
    tickValue.value = `${duration.value} tick${tasks ? ' / 专家' : ''}`;
  }

  function redraw() {
    const forward = state.direction === 'forward';
    const inputs = state.inputs[state.direction];
    const selectedStage = stagePicker.value;
    timing = forward ? forwardTiming : backwardTiming;
    stagePicker.innerHTML = Object.entries(timing).map(([key, spec]) => `<option value="${key}">${spec.label}</option>`).join('');
    stagePicker.value = Object.hasOwn(timing, selectedStage) ? selectedStage : 'dispatch';
    editStage();

    const model = forward ? simulateFullForward(inputs, shape) : simulateFullBackward(inputs, shape);
    const group = model.groups[0];
    const lanes = [[group.aic, 'AIC'], [group.aiv0, 'AIV0'], [group.aiv1, 'AIV1']];
    events = expertOverviewEvents(model.events, lanes, timing);
    const aicWork = events.filter(event => event.lane === group.aic && event.stage === (forward ? 'gmm1' : 'dw1'));
    const aivWork = events.filter(event => event.lane === group.aiv0 && event.stage === (forward ? 'swiglu' : 'unpermuteDx'));
    const overlaps = aicWork.flatMap(cube => aivWork
      .filter(vector => !forward || vector.expert === cube.expert)
      .map(vector => ({ expert: cube.expert, start: Math.max(cube.start, vector.start), end: Math.min(cube.end, vector.end) }))
      .filter(({ start, end }) => start < end));
    chart.innerHTML = renderPipelineTimeline({ lanes, events, finish: model.finish, pixelsPerTick: 0, overlaps, shortWaitTicks: 2 });
    const waitOptions = events.map((event, index) => event.kind === 'wait'
      ? `<option value="${index}">${lanes.find(([id]) => id === event.lane)[1]} · ${event.label} · ${event.start}–${event.end}</option>` : '');
    waitPicker.innerHTML = '<option value="">选择一段等待</option>' + waitOptions.join('');
    detail.textContent = '点阶段、虚线等待条或箭头下的「等」，查看交接条件和对应源码。';
    panel.querySelector('[data-overview-end]').innerHTML = `<span>rank0 代表组${forward ? '写完 Y' : '清理完成'} <strong>${group.outputAt}</strong> tick</span><span>整个调用完成 <strong>${model.finish}</strong> tick</span>`;
    const ranges = overlaps.map(({ expert, start, end }) => `e${expert} ${start}–${end}`).join(' · ');
    panel.querySelector('[data-overview-overlap]').textContent = `${forward ? 'GMM1 与 SwiGLU 阶段窗口' : 'dX 归并与 dW1'} · ${ranges ? `重叠 ${ranges} tick` : '当前耗时下无重叠'}`;
    panel.querySelector('[data-overview-note]').textContent = forward
      ? `每专家 GMM1 工作 ${aicWork[0].work} tick，GMM2 工作 ${events.find(event => event.kind === 'compute' && event.stage === 'gmm2').work} tick，SwiGLU 工作 ${aivWork[0].work} tick。浅紫窗口包含等下一块 Gate/Up 的时间。跨专家时，AIC 只等上一块的 UB 归还。`
      : '两路 AIV 都参与 Gather 和计数恢复。AIV0 拉取后续专家的 dY 时，AIV1 已可处理 dHp 就绪的专家。SwiGLU Grad 前等 dHp，回传前等 dX。';
    panel.querySelector('[data-overview-legend]').innerHTML = (forward
      ? [['prepare', '准备'], ['send', 'Dispatch'], ['cube1', 'GMM1'], ['vector', 'SwiGLU'], ['compute', 'GMM2'], ['return', 'Combine'], ['finish', '写 Y']]
      : [['prepare', 'Gather / 计数'], ['send', 'Dispatch'], ['cube1', 'dHp'], ['vector', 'SwiGLU Grad'], ['weight2', 'dW2'], ['compute', 'dX'], ['weight1', 'dW1'], ['return', '回传'], ['finish', '归并']])
      .concat([['wait', '等待'], ['short-wait', '短等待 ≤ 2 tick']]).map(([kind, label]) => `<span><i class="${kind}"></i>${label}</span>`).join('');
    for (const button of panel.querySelectorAll('[data-overview-direction]')) {
      button.setAttribute('aria-pressed', String(button.dataset.overviewDirection === state.direction));
    }
  }

  function selectEvent(index) {
    const event = events[index];
    waitPicker.value = event.kind === 'wait' ? String(index) : '';
    if (event.kind !== 'wait' && Object.hasOwn(timing, event.stage)) {
      stagePicker.value = event.stage;
      editStage();
    }
    const formulaNode = event.stage === 'epi' ? 'b_numeric' : event.source.node;
    const formula = event.kind === 'wait' ? undefined : formulas.get(formulaNode);
    showPipelineEvent(chart, detail, index, events, inspectSource, formula);
  }
  chart.onclick = event => {
    const button = event.target.closest('[data-sim-event]');
    if (button) selectEvent(Number(button.dataset.simEvent));
  };
  waitPicker.onchange = () => {
    if (waitPicker.value !== '') selectEvent(Number(waitPicker.value));
  };
  stagePicker.onchange = editStage;
  duration.oninput = () => {
    const key = stagePicker.value;
    state.inputs[state.direction][key] = duration.valueAsNumber / (timing[key].tasks ?? 1);
    redraw();
  };
  for (const button of panel.querySelectorAll('[data-overview-direction]')) {
    button.onclick = () => { state.direction = button.dataset.overviewDirection; redraw(); };
  }
  panel.querySelector('[data-overview-reset]').onclick = () => { state.inputs[state.direction] = {}; redraw(); };
  redraw();
}
