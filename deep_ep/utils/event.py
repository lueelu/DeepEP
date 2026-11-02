# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""Real framework events; empty or fabricated handles never report completion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from .._native import npu_api

if TYPE_CHECKING:
    import torch


class EventHandle:
    def __init__(self, stream: Any = None) -> None:
        npu = npu_api()
        self._device = npu.current_device()
        self._stream = npu.current_stream(self._device) if stream is None else stream
        stream_device = getattr(self._stream, "device", self._device)
        stream_device = getattr(stream_device, "index", stream_device)
        if stream_device != self._device:
            raise RuntimeError("Cannot record an event on another NPU device")
        self._event = npu.Event()
        self._event.record(self._stream)

    @property
    def event(self) -> Any:
        """Underlying framework event, retained for the CLOS import path."""
        return self._event

    def current_stream_wait(self) -> None:
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
        self._retained_objects = None
        self._empty_result = False

    @classmethod
    def _from_resources(cls, event: EventHandle, resources: Tuple[Any, ...]) -> "EventOverlap":
        """Retain backend objects separately from stream-tracked NPU tensors."""
        import torch

        tensors = tuple(t for t in resources if isinstance(t, torch.Tensor) and t.device.type == "npu")
        overlap = cls(event, tensors)
        overlap._retained_objects = resources
        return overlap

    @classmethod
    def _empty(cls) -> "EventOverlap":
        """Synchronous API result after its compute-stream dependency is queued.

        This is not a completion event and cannot be explicitly waited on.
        The public constructor still requires a recorded EventHandle.
        """
        overlap = cls.__new__(cls)
        overlap.event = None
        overlap.extra_tensors = None
        overlap._hook = None
        overlap._release = False
        overlap._retained_objects = None
        overlap._empty_result = True
        return overlap

    @property
    def hook_after_wait(self) -> Optional[Callable]:
        return self._hook

    @hook_after_wait.setter
    def hook_after_wait(self, hook: Optional[Callable]) -> None:
        if hook is not None and not callable(hook):
            raise TypeError("hook_after_wait must be callable or None")
        self._hook = hook

    def current_stream_wait(self, release_handle: bool = False) -> None:
        if type(release_handle) is not bool:
            raise TypeError("release_handle must be bool")
        if getattr(self, "event", None) is None:
            raise RuntimeError("EventOverlap has no event or was released")
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
            self._retained_objects = None

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        if getattr(self, "event", None) is None:
            raise RuntimeError("EventOverlap has no event or was released")
        if not callable(hook_after_wait):
            raise TypeError("hook_after_wait must be callable")
        if self._hook is not None:
            raise RuntimeError("A hook is already registered")
        self._hook = hook_after_wait

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        if type(release_handle) is not bool:
            raise TypeError("release_handle must be bool")
        self._release = release_handle
        return self

    def __enter__(self) -> Any:
        if getattr(self, "event", None) is None and not getattr(self, "_empty_result", False):
            raise RuntimeError("EventOverlap has no event or was released")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if not self._empty_result:
                self.current_stream_wait(release_handle=self._release)
        finally:
            self._release = False
