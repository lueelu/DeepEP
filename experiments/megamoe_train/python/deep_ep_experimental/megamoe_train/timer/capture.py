# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""A capture belongs to one autograd graph, independently of later forwards."""

import hashlib
import json
from pathlib import Path


class TimerCapture:
    def __init__(self, op, *, enabled=True):
        self.op = op
        self.enabled = enabled
        self.forward = None
        self.backward = None
        self._claimed = False
        self._backward_started = False

    def __enter__(self):
        self.op._check_ready()
        if self.enabled:
            from .. import _C

            if not _C.timer_enabled:
                raise RuntimeError("Training Timer requires DEEPEP_MEGAMOE_TIMER=ON")
            if getattr(self.op, "_timer_capture", None) is not None:
                raise RuntimeError("A training Timer capture is already active")
            self.op._timer_capture = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if getattr(self.op, "_timer_capture", None) is self:
            self.op._timer_capture = None

    def claim(self):
        if self._claimed:
            raise RuntimeError("Use one forward per training Timer capture")
        self._claimed = True

    def begin_backward(self):
        if self._backward_started:
            raise RuntimeError("Use one backward per training Timer capture")
        self._backward_started = True

    def export(self, directory, *, parse=True):
        """Synchronize/copy after measurement; preserve raw buffers for offline use."""
        if not self.enabled:
            return None
        if self.forward is None:
            raise RuntimeError("The capture has no completed forward")
        from .. import _C

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rank = self.op._runtime.rank
        manifest = dict(rank=rank, world_size=self.op._runtime.world_size, phases=[], libraries={})
        for name, tensor in (("forward", self.forward), ("backward", self.backward)):
            if tensor is not None:
                values = tensor.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
                values.tofile(directory / f"rank_{rank}_{name}_time.bin")
                manifest["phases"].append(name)
        native = Path(_C.__file__)
        for name, path in (("host", native), ("device", native.with_name("libdeepep_megamoe_train_bf16.so"))):
            manifest["libraries"][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (directory / f"rank_{rank}_capture.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if parse:
            from .trace_parser import export_rank

            export_rank(directory, rank)
        return manifest
