# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""DeepEP resource/event interfaces; imports do not require Torch or an NPU SDK.

Installed native wheels provide communication resources and EP operations.
Framework events require Torch/torch_npu only. Communication runtime calls without
the extension fail explicitly. See docs/api.md.
"""

from ._config import Config
from .buffer import Buffer
from .buffers.elastic import ElasticBuffer, EPHandle
from .utils import EventHandle, EventOverlap

__all__ = ["Buffer", "ElasticBuffer", "EPHandle", "Config", "EventHandle", "EventOverlap"]


def __getattr__(name):
    # Keep package import metadata-only; Clos helpers load Torch on demand.
    if name in {"topk_idx_t", "ExpandHandle", "Handle", "Slot", "SlotState"}:
        from .backends import clos

        return getattr(clos, name)
    raise AttributeError(name)
