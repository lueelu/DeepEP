# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""DeepEP resource/event interfaces; imports do not require Torch or an NPU SDK.

Installed native wheels provide resources, events, and ElasticBuffer EP operations.
Without the extension, runtime calls fail explicitly. See docs/api.md.
"""

from ._config import Config
from .buffers import Buffer, ElasticBuffer, EPHandle
from .utils import EventHandle, EventOverlap

__all__ = ["Buffer", "ElasticBuffer", "EPHandle", "Config", "EventHandle", "EventOverlap"]
