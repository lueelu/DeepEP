"""Optional, separate LL device-stage diagnostics. No Torch import for parsing.

Raw values are GetSystemCycle ticks. Supply that counter's frequency explicitly
to convert to microseconds; do not substitute the AIV compute clock.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading

MAGIC = 0x4C4C5431
SCHEMA = 3
DEFAULT_FREQUENCY_HZ = 1_000_000_000  # Target-confirmed: time_us = ticks / 1000.
SLOTS, FIELDS = 32, 8
STAGES = ('header', 'kernel', 'magic', 'worker_init', 'count_init', 'count_send', 'sender',
          'send_prepare', 'send_payload', 'send_metadata', 'count_receive', 'count_prefix',
          'count_publish', 'receiver', 'recv_prepare', 'recv_wait', 'recv_consume', 'recv_cleanup',
          'count_finish', 'combine_prepare', 'combine_notify', 'combine_wait_ready',
          'combine_direct', 'combine_partial', 'combine_aggregate', 'combine_drain', 'combine_final_drain',
          'send_classify', 'send_pack', 'send_issue')
LEGACY_STAGES = STAGES
STAGES += ('recv_lane_wait', 'recv_gm_to_ub', 'recv_load_wait', 'recv_metadata',
           'recv_output_wait', 'recv_output_write', 'recv_forward',
           'send_input_gm_to_ub', 'send_input_wait', 'send_tail', 'send_lane_wait',
           'send_mte_write', 'send_rdma_stage', 'send_rdma_body_post', 'send_rdma_body_wait',
           'send_rdma_flag_post', 'send_rdma_flag_wait', 'send_rdma_restore')
TRAFFIC_CLASSES = ('local', 'intrahost_first', 'intrahost_forward',
                   'interhost_first', 'interhost_forward')
TRAFFIC_OFFSET, TRAFFIC_FLAGS_OFFSET = 208, 228
ACTIVE = 1 << 62
NOISY_SEND_ENVELOPES = frozenset(('send_input_gm_to_ub', 'send_input_wait', 'send_tail',
                                 'send_lane_wait', 'send_issue', 'send_mte_write'))
_state = threading.local()


def parse_records(raw, operation, frequency_hz=None):
    if operation not in ('dispatch', 'combine'):
        raise ValueError('Expected dispatch or combine')
    if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError('Counter frequency must be finite and positive')
    cores = len(raw)
    if not 16 <= cores <= 72 or cores % 2:
        raise ValueError('Expected actual even AIV count in [16,72]')
    op = 1 if operation == 'dispatch' else 2
    records, traffic, modes, schemas = [], [], set(), set()
    for core, slots in enumerate(raw):
        if len(slots) != SLOTS or any(len(row) != FIELDS for row in slots):
            raise ValueError('Timer shape mismatch')
        header = slots[0]
        schema = header[1]
        if schema not in (1, 2, 3):
            raise ValueError('Unsupported timer schema')
        names = LEGACY_STAGES if schema == 1 else STAGES
        schemas.add(schema)
        if any(type(value) is not int or value < 0 for value in header):
            raise ValueError("Invalid timer header word")
        if header[:5] != [MAGIC, schema, op, cores, core] or header[6:] != [1, len(names)]:
            raise ValueError(f'Incomplete or mismatched timer header on core {core}')
        if header[5] not in (1, 2):
            raise ValueError('Unknown boundary mode')
        modes.add(header[5])
        if schema in (2, 3):
            words = [value for row in slots for value in row]
            if any(type(value) is not int or not 0 <= value < (1 << 63) for value in words):
                raise ValueError('Invalid raw timer word')
            stage_end = 8 + (len(names)-1)*4
            if schema == 3:
                if any(words[stage_end:TRAFFIC_OFFSET]) or any(words[TRAFFIC_FLAGS_OFFSET+1:]):
                    raise ValueError('Nonzero reserved traffic words')
                flags = words[TRAFFIC_FLAGS_OFFSET]
                if flags & ~1:
                    raise ValueError('Unknown traffic flags')
                for index, category in enumerate(TRAFFIC_CLASSES):
                    offset = TRAFFIC_OFFSET + index*4
                    byte_count, operations, read_bytes, read_operations = words[offset:offset+4]
                    if read_bytes > byte_count or read_operations > operations:
                        raise ValueError('Traffic reads must be a subset of total traffic')
                    traffic.append(dict(core=core, category=category, bytes=byte_count,
                                        operations=operations, read_bytes=read_bytes,
                                        read_operations=read_operations, flags=flags))
            normalized = [header]
            for offset in range(8, stage_end if schema == 3 else SLOTS * FIELDS, 4):
                first, last, total, count = words[offset:offset+4]
                normalized.append([first, last, total, count & ~ACTIVE, first, int(bool(count & ACTIVE)), 0, 0])
            slots = normalized
        if slots[1][3] != 1:
            raise ValueError('Exactly one completed kernel interval is required')
        for stage, row in enumerate(slots[1:], 1):
            if any(type(value) is not int or value < 0 for value in row):
                raise ValueError('Invalid raw timer word')
            if stage >= len(names):
                if any(row):
                    raise ValueError('Unknown stage data')
                continue
            first, last, total, count, active_start, active, reserved0, reserved1 = row
            if active or reserved0 or reserved1:
                raise ValueError(f'Unclosed or invalid stage {STAGES[stage]} on core {core}')
            if not count:
                if any(row):
                    raise ValueError('Unfinished stage')
                continue
            if last < first or active_start < first or active_start > last or total > last-first:
                raise ValueError('Invalid timestamp order or accumulated interval')
            record = dict(core=core, role='send' if core < cores//2 else 'receive',
                          stage=names[stage], first_tick=first, last_tick=last,
                          total_ticks=total, calls=count, average_ticks=total/count)
            if frequency_hz is not None:
                record['total_us'] = total*1e6/frequency_hz
                record['average_us'] = total*1e6/frequency_hz/count
            records.append(record)
    if len(modes) != 1:
        raise ValueError('Mixed boundary modes')
    if len(schemas) != 1:
        raise ValueError('Mixed timer schemas')
    return dict(schema=schema, operation=operation, active_aiv=cores,
                boundary_mode='observe' if modes == {1} else 'drain', frequency_hz=frequency_hz,
                clock='AscendC::GetSystemCycle; no cross-rank clock alignment assumed', records=records,
                traffic_available=schema == 3, traffic=traffic if schema == 3 else None,
                interpretation='Nested and overlapping stages must not be summed; repeated-stage totals exclude gaps.')


def chrome_trace(saved_records, frequency_hz=None, lane_layout='core', show_repeated=False):
    """Export rank/sample-local timelines; schema 1 cannot reconstruct repeated calls."""
    if lane_layout not in ('core', 'stage'):
        raise ValueError('Expected core or stage lane layout')
    events, identities = [], set()
    for pid, saved in enumerate(saved_records, 1):
        frequency = frequency_hz if frequency_hz is not None else saved.get('frequency_hz')
        if frequency is None:
            raise ValueError('Chrome trace requires explicit GetSystemCycle frequency_hz')
        report = parse_records(saved['raw'], saved['operation'], frequency)
        meta = saved.get('metadata', {})
        rank, iteration = meta.get('rank', 'unknown'), meta.get('iteration', pid - 1)
        identity = (meta.get('run_id'), meta.get('case_id'), report['operation'], rank, iteration)
        if identity in identities:
            raise ValueError('Duplicate rank/operation/sample trace')
        identities.add(identity)
        origin = min(row['first_tick'] for row in report['records'])
        name = f"{report['operation']} rank {rank} sample {iteration} ({report['boundary_mode']}; independent origin)"
        events.append(dict(ph='M', name='process_name', pid=pid, tid=0, args={'name': name}))
        lanes = set()
        # Parents first at equal starts, so viewers can nest complete events.
        records = sorted(report['records'], key=lambda row: (row['core'], row['first_tick'], -row['last_tick']))
        for row in records:
            if not show_repeated and row['calls'] > 1 and row['stage'] in NOISY_SEND_ENVELOPES:
                continue
            tid = row['core'] if lane_layout == 'core' else row['core'] * 64 + STAGES.index(row['stage'])
            lane = f"AIV {row['core']:02d} {row['role']}"
            if lane_layout == 'stage':
                lane += f" / {row['stage']}"
            if tid not in lanes:
                events.append(dict(ph='M', name='thread_name', pid=pid, tid=tid, args={'name': lane}))
                events.append(dict(ph='M', name='thread_sort_index', pid=pid, tid=tid, args={'sort_index': tid}))
                lanes.add(tid)
            repeated = row['calls'] > 1
            span = row['last_tick'] - row['first_tick']
            args = dict(row, rank=rank, iteration=iteration, operation=report['operation'],
                        sampling=meta.get('sampling'),
                        boundary_mode=report['boundary_mode'], frequency_hz=frequency,
                        origin_tick=origin, span_ticks=span, gap_ticks=span-row['total_ticks'],
                        interval_kind='repeated envelope' if repeated else 'single interval',
                        clock_alignment='Independent origin per rank/operation/sample; not profiler aligned')
            events.append(dict(ph='X', cat='LL.stage.envelope' if repeated else 'LL.stage',
                               name=row['stage'] + (f" [envelope x{row['calls']}]" if repeated else ''),
                               pid=pid, tid=tid, ts=(row['first_tick']-origin)*1e6/frequency,
                               dur=span*1e6/frequency, args=args))
    if not identities:
        raise ValueError('No LL stage records')
    # Chrome Trace Event ts/dur are microseconds. displayTimeUnit is a viewer
    # preference with a different enum; Chrome tracing rejects 'us'. Omit it.
    return dict(traceEvents=events,
                metadata={'format': 'LL stage Chrome trace (schemas 1/2/3)', 'lane_layout': lane_layout,
                          'hidden_repeated_send_envelopes': not show_repeated,
                          'interpretation': 'Repeated events show envelopes including gaps; args.total_us is accumulated active time. Independent rank/sample origins; no cross-rank or profiler clock alignment.'})


class StageTrace:
    def __init__(self, buffer, operation, mode='observe', frequency_hz=DEFAULT_FREQUENCY_HZ,
                 expert_config=None):
        if operation not in ('dispatch', 'combine') or mode not in ('observe', 'drain'):
            raise ValueError('Invalid LL stage diagnostic configuration')
        if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz <= 0):
            raise ValueError('Counter frequency must be finite and positive')
        from .._native import require_native
        import torch
        native = require_native('LL stage timer')
        if not getattr(native, 'low_latency_stage_timer_enabled', False):
            raise RuntimeError('Rebuild the full wheel with DEEPEP_LL_STAGE_TIMER=ON')
        schema = native.low_latency_stage_timer_layout()
        if schema != dict(schema=SCHEMA, slots=SLOTS, fields=FIELDS, stages=list(STAGES)):
            raise RuntimeError('LL timer Python/native schema mismatch')
        cfg = native.low_latency_core_config()
        if not cfg['runtime_device_queried']:
            raise RuntimeError('Bind the actual NPU before allocating a timer')
        words = cfg['active_aiv'] * SLOTS * FIELDS
        storage = torch.zeros(words + 63, dtype=torch.int64, device=f"npu:{cfg['device']}")
        # Align the view to 512B even if an allocator returns only 8B alignment.
        offset = (-storage.data_ptr()) % 512 // 8
        self.tensor = storage.narrow(0, offset, words).view(cfg['active_aiv'], SLOTS, FIELDS)
        self.expert_tensor, self.expert_raw, self.expert_config = None, None, expert_config
        if expert_config is not None:
            from . import ll_expert_timer as expert
            rank, world, count = (expert_config[key] for key in ('rank','world','experts'))
            if (operation != 'dispatch' or any(type(v) is not int for v in (rank,world,count)) or
                    not (0 <= rank < world <= count <= 1024 and world >= 16 and world%16==0 and count%world==0)):
                raise ValueError('Expert capture requires a valid dispatch expert configuration')
            if native.low_latency_core_protocol not in (5, 6, 7, 8, 9, 10) or not hasattr(native, 'low_latency_expert_timer_layout'):
                raise RuntimeError('Expert diagnostics require the matching ABI 5 wheel on every rank')
            if native.low_latency_expert_timer_layout() != dict(schema=expert.SCHEMA, header_rows=expert.HEADER_ROWS,
                                                               fields=expert.FIELDS, stages=list(expert.STAGES)):
                raise RuntimeError('Expert timer Python/native schema mismatch')
            rows=expert.HEADER_ROWS+count*len(expert.STAGES)
            expert_words=cfg['active_aiv']*rows*expert.FIELDS
            backing=torch.zeros(expert_words+63,dtype=torch.int64,device=f"npu:{cfg['device']}")
            offset=(-backing.data_ptr())%512//8
            self.expert_tensor=backing.narrow(0,offset,expert_words).view(cfg['active_aiv'],rows,expert.FIELDS)
        self.buffer, self.operation, self.mode = buffer, operation, mode
        self.frequency_hz, self.launches, self.raw, self.report = frequency_hz, 0, None, None

    def complete(self):
        if self.launches != 1:
            raise RuntimeError('A diagnostic context must contain exactly one LL operation')
        # Public LL calls establish a current-stream dependency on completion.
        self.raw = self.tensor.cpu().tolist()
        self.report = parse_records(self.raw, self.operation, self.frequency_hz)
        if self.expert_tensor is not None:
            from .ll_expert_timer import parse_records as parse_experts
            self.expert_raw=self.expert_tensor.cpu().tolist()
            report=parse_experts(self.expert_raw, **self.expert_config, frequency_hz=self.frequency_hz)
            if report['boundary_mode'] != self.mode:
                raise RuntimeError('Expert/stage timer mode mismatch')

    def save(self, path, metadata=None):
        if self.report is None:
            raise RuntimeError('Trace is not complete')
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('Resolve trace output from your project root and pass an absolute path')
        path.parent.mkdir(parents=True, exist_ok=True)
        saved=dict(**self.report, metadata=metadata or {}, raw=self.raw)
        if getattr(self, 'expert_raw', None) is not None:
            saved['expert']=dict(raw=self.expert_raw, **self.expert_config)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(saved, stream, indent=2)
        if self.frequency_hz is not None:
            # Prefix avoids the analyzer's sample_*.json raw-record glob.
            trace_path = path.with_name('trace_' + path.name)
            saved = dict(raw=self.raw, operation=self.operation,
                         frequency_hz=self.frequency_hz, metadata=metadata or {})
            with trace_path.open('x', encoding='utf-8') as stream:
                json.dump(chrome_trace([saved]), stream)


@contextmanager
def capture(buffer, operation, mode='observe', frequency_hz=DEFAULT_FREQUENCY_HZ,
            *, prepared=None, defer_complete=False):
    """Capture exactly one payload LL call; use independently of profiler loops."""
    import sys
    reference = sys.modules.get('deep_ep.utils.ll_reference_profile')
    active = getattr(_state, 'active', {})
    if id(buffer) in active or id(buffer) in getattr(getattr(reference, '_state', None), 'active', {}):
        raise RuntimeError('Nested timer contexts for the same buffer are unsupported')
    trace = prepared if prepared is not None else StageTrace(buffer, operation, mode, frequency_hz)
    if prepared is not None and (trace.buffer is not buffer or trace.operation != operation or
                                 trace.mode != mode or trace.frequency_hz != frequency_hz or trace.launches):
        raise ValueError('Prepared trace must be unused and match this capture')
    _state.active = active
    active[id(buffer)] = trace
    try:
        yield trace
        if trace.launches != 1:
            raise RuntimeError('A diagnostic context must contain exactly one LL operation')
        if not defer_complete:
            trace.complete()
    finally:
        active.pop(id(buffer), None)


def attach(buffer, tensors, operation):
    """Internal LL-only hook; tensor is retained/record_stream'ed by its caller."""
    trace = getattr(_state, 'active', {}).get(id(buffer))
    if trace is None:
        return
    if trace.operation != operation or trace.launches:
        raise RuntimeError('Wrong operation or multiple launches in timer context')
    if trace.tensor.device != tensors['input'].device:
        raise ValueError('Timer and LL input must be on the same NPU')
    trace.launches += 1
    tensors['ll_stage_timer'] = trace.tensor
    if getattr(trace, 'expert_tensor', None) is not None:
        tensors['ll_expert_timer'] = trace.expert_tensor
    # The native binding derives mode from this CPU scalar, not an NPU read.
    tensors['ll_stage_timer_mode'] = 1 if trace.mode == 'observe' else 2
