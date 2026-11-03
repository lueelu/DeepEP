# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""NPU events and upstream-compatible overlap wrappers for both backends."""

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
    """Upstream-compatible event wrapper shared by Mesh+Clos and CLOS.

    An empty wrapper is valid, but has no event to explicitly wait for.
    NPU tensor stream tracking is internal and does not narrow the public
    constructor's duck-typed event and extra_tensors arguments.
    """

    def __init__(
        self,
        event: Optional[EventHandle] = None,
        extra_tensors: Optional[Tuple[torch.Tensor]] = None,
    ) -> None:
        self.event = event
        self.extra_tensors = extra_tensors
        self._release_handle_by_call = False
        self.hook_after_wait: Optional[Callable] = None
        self._retained_objects = None
        self._stream_tensors = ()
        if isinstance(event, EventHandle) and extra_tensors is not None:
            self._track_tensors(extra_tensors)

    def _track_tensors(self, resources) -> None:
        import torch

        values = resources if isinstance(resources, (tuple, list)) else (resources,)
        self._stream_tensors = tuple(
            t
            for t in values
            if isinstance(t, torch.Tensor) and t.device.type == "npu" and t.device.index == self.event._device
        )
        # Protect producer-stream use even if the wrapper is abandoned.
        for tensor in self._stream_tensors:
            tensor.record_stream(self.event._stream)

    @classmethod
    def _from_resources(cls, event: EventHandle, resources: Tuple[Any, ...]) -> "EventOverlap":
        """Own backend resources without changing public extra_tensors semantics."""
        overlap = cls(event)
        overlap._retained_objects = resources
        overlap._track_tensors(resources)
        return overlap

    def current_stream_wait(self, release_handle: bool = False) -> Any:
        assert self.event is not None
        self.event.current_stream_wait()
        # Enqueuing a wait is not host completion. Register consumer use before
        # dropping the internal owners, including CLOS handles and epilogues.
        if self._stream_tensors:
            stream = npu_api().current_stream(self.event._device)
            for tensor in self._stream_tensors:
                tensor.record_stream(stream)
        result = None
        if self.hook_after_wait is not None:
            result = self.hook_after_wait()
            self.hook_after_wait = None
        if release_handle:
            self.event = None
            self._retained_objects = None
            self._stream_tensors = ()
        # Upstream keeps extra_tensors until the wrapper itself is released.
        return result

    def wait(self) -> Any:
        """Wait on the current stream and return the registered epilogue result."""
        return self.current_stream_wait()

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        assert self.hook_after_wait is None, "A hook is already registered on this `EventOverlap`"
        self.hook_after_wait = hook_after_wait

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.event is not None:
            self.current_stream_wait(release_handle=self._release_handle_by_call)
        self._release_handle_by_call = False
