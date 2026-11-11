"""Separate LL dispatch reference-cycle capture; no device imports for parsing."""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading

from .ll_reference_profile_parse import (
    DEFAULT_FREQUENCY_HZ, FIELDS, FORMAT, SCHEMA, SLOTS, WORDS, parse_records,
)

_state = threading.local()


class ReferenceTrace:
    def __init__(self, buffer, operation='dispatch', mode='reference', frequency_hz=DEFAULT_FREQUENCY_HZ):
        if operation != 'dispatch' or mode != 'reference':
            raise ValueError('LL reference profiling supports dispatch with reference boundaries only')
        if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz <= 0):
            raise ValueError('Counter frequency must be finite and positive')
        from .._native import require_native
        import torch
        native = require_native('LL dispatch reference profile')
        if (getattr(native, 'low_latency_reference_profile_enabled', None) is not True or
                getattr(native, 'low_latency_stage_timer_enabled', None) is not False):
            raise RuntimeError('Rebuild the complete wheel with DEEPEP_LL_REFERENCE_PROFILE=ON and DEEPEP_LL_STAGE_TIMER=OFF')
        expected = dict(schema=SCHEMA, slots=SLOTS, fields=FIELDS, words=list(WORDS))
        layout = native.low_latency_reference_profile_layout()
        if (layout != expected or any(type(layout.get(key)) is not int for key in ('schema', 'slots', 'fields'))):
            raise RuntimeError('LL reference profile Python/native layout mismatch')
        cfg = native.low_latency_core_config()
        cores = cfg['active_aiv']
        if (cfg['runtime_device_queried'] is not True or type(cores) is not int or
                not 16 <= cores <= 72 or cores % 2):
            raise RuntimeError('Bind the actual NPU before allocating a reference profile')
        words = cores * SLOTS * FIELDS
        backing = torch.zeros(words + 63, dtype=torch.int64, device=f"npu:{cfg['device']}")
        offset = (-backing.data_ptr()) % 512 // 8
        self.tensor = backing.narrow(0, offset, words).view(cores, SLOTS, FIELDS)
        self.buffer, self.operation, self.mode = buffer, operation, mode
        self.frequency_hz = frequency_hz
        self.launches, self.raw, self.report = 0, None, None

    def complete(self):
        if self.launches != 1:
            raise RuntimeError('A reference profile must contain exactly one LL dispatch')
        self.raw = self.tensor.cpu().tolist()
        self.report = parse_records(self.raw, self.operation, self.frequency_hz)

    def save(self, path, metadata=None):
        if self.report is None:
            raise RuntimeError('Reference profile is not complete')
        path = Path(path)
        if not path.is_absolute():
            raise ValueError('Resolve profile output from the project root and pass an absolute path')
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = dict(self.report)
        saved.update(format=FORMAT, raw=self.raw, operation=self.operation,
                     frequency_hz=self.frequency_hz, metadata=metadata or {})
        with path.open('x', encoding='utf-8') as stream:
            json.dump(saved, stream, indent=2)


@contextmanager
def capture(buffer, operation='dispatch', mode='reference', frequency_hz=DEFAULT_FREQUENCY_HZ,
            *, prepared=None, defer_complete=False):
    from . import ll_stage_timer
    active = getattr(_state, 'active', {})
    if id(buffer) in active or id(buffer) in getattr(ll_stage_timer._state, 'active', {}):
        raise RuntimeError('Nested reference/stage contexts for the same buffer are unsupported')
    trace = prepared if prepared is not None else ReferenceTrace(buffer, operation, mode, frequency_hz)
    if prepared is not None and (not isinstance(trace, ReferenceTrace) or trace.buffer is not buffer or
                                 trace.operation != operation or trace.mode != mode or
                                 trace.frequency_hz != frequency_hz or trace.launches):
        raise ValueError('Prepared reference profile must be unused and match this capture')
    _state.active = active
    active[id(buffer)] = trace
    try:
        yield trace
        if trace.launches != 1:
            raise RuntimeError('A reference profile must contain exactly one LL dispatch')
        if not defer_complete:
            trace.complete()
    finally:
        active.pop(id(buffer), None)


def attach(buffer, tensors, operation):
    """Private LL hook; the caller retains and records the tensor on its stream."""
    trace = getattr(_state, 'active', {}).get(id(buffer))
    if trace is None:
        return
    if operation != 'dispatch' or trace.operation != operation or trace.launches:
        raise RuntimeError('Reference profiling accepts exactly one LL dispatch, never combine')
    if any(tensors.get(key) is not None for key in ('ll_stage_timer', 'll_expert_timer')):
        raise RuntimeError('Reference profiling cannot run with legacy stage/expert tracing')
    if trace.tensor.device != tensors['input'].device:
        raise ValueError('Reference profile and LL input must be on the same NPU')
    trace.launches += 1
    tensors['ll_reference_profile'] = trace.tensor
