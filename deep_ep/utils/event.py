# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""Real framework events; empty or fabricated handles never report completion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from .._native import npu_api, require_native

if TYPE_CHECKING:
    import torch


class EventHandle:
    def __init__(self) -> None:
        require_native("EventHandle.__init__")
        npu = npu_api()
        self._device = npu.current_device()
        self._stream = npu.current_stream(self._device)
        self._event = npu.Event()
        self._event.record(self._stream)

    def current_stream_wait(self) -> None:
        require_native("EventHandle.current_stream_wait")
        npu = npu_api()
        if npu.current_device() != self._device:
            raise RuntimeError("Cannot wait for an event from another NPU device")
        self._event.wait(npu.current_stream(self._device))


class EventOverlap:
    def __init__(
        self,
        event: Optional[EventHandle] = None,
        extra_tensors: Optional[Tuple[torch.Tensor]] = None,
    ) -> None:
        require_native("EventOverlap.__init__")
        if not isinstance(event, EventHandle) or not hasattr(event, "_event"):
            raise ValueError("EventOverlap requires a recorded EventHandle")
        if extra_tensors is not None and not isinstance(extra_tensors, tuple):
            raise TypeError("extra_tensors must be a tuple or None")
        if extra_tensors:
            import torch

            if any(
                not isinstance(t, torch.Tensor) or t.device.type != "npu" or t.device.index != event._device
                for t in extra_tensors
            ):
                raise ValueError("extra_tensors must be NPU tensors on the event device")
            # The wrapper may be abandoned without a wait. Protect producer
            # stream use as well as the eventual consumer stream use.
            for tensor in extra_tensors:
                tensor.record_stream(event._stream)
        self.event = event
        self.extra_tensors = extra_tensors
        self._hook = None
        self._release = False

    def current_stream_wait(self, release_handle: bool = False) -> None:
        require_native("EventOverlap.current_stream_wait")
        if type(release_handle) is not bool:
            raise TypeError("release_handle must be bool")
        if self.event is None:
            raise RuntimeError("EventOverlap was released")
        self.event.current_stream_wait()
        # A stream wait is not host completion. Register consumer-stream use
        # with the allocator before dropping retained tensor references.
        if self.extra_tensors:
            stream = npu_api().current_stream(self.event._device)
            for tensor in self.extra_tensors:
                tensor.record_stream(stream)
        hook, self._hook = self._hook, None
        if hook is not None:
            hook()
        if release_handle:
            self.event = None
            self.extra_tensors = None

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        require_native("EventOverlap.register_hook_after_wait")
        if self.event is None:
            raise RuntimeError("EventOverlap was released")
        if not callable(hook_after_wait):
            raise TypeError("hook_after_wait must be callable")
        if self._hook is not None:
            raise RuntimeError("A hook is already registered")
        self._hook = hook_after_wait

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        require_native("EventOverlap.__call__")
        if type(release_handle) is not bool:
            raise TypeError("release_handle must be bool")
        self._release = release_handle
        return self

    def __enter__(self) -> Any:
        require_native("EventOverlap.__enter__")
        if self.event is None:
            raise RuntimeError("EventOverlap was released")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        require_native("EventOverlap.__exit__")
        try:
            self.current_stream_wait(release_handle=self._release)
        finally:
            self._release = False
