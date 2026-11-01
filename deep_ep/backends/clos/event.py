"""NPU event wrappers used to overlap communication and computation."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple


def _get_npu_api() -> Any:
    """Load torch-npu lazily so importing API metadata does not initialize NPU."""

    try:
        import torch_npu  # noqa: F401
        import torch
    except ImportError as exc:
        raise ImportError("torch_npu must be installed before recording an NPU event") from exc
    return torch.npu


class EventHandle:
    """A recorded NPU event with DeepEP-compatible wait semantics.

    Constructing a handle records a new event on the current NPU stream unless
    an explicit stream is supplied. Device synchronization is not performed.
    """

    def __init__(self, stream: Any = None) -> None:
        npu = _get_npu_api()
        self._event = npu.Event()
        self._event.record(npu.current_stream() if stream is None else stream)

    @property
    def event(self) -> Any:
        """Return the underlying ``torch.npu.Event``."""

        return self._event

    def current_stream_wait(self) -> None:
        """Make future work on the current NPU stream wait for this event."""

        npu = _get_npu_api()
        self._event.wait(npu.current_stream())


class EventOverlap:
    """Python owner for an asynchronous communication event."""

    def __init__(
        self,
        event: Optional[EventHandle] = None,
        extra_tensors: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        if extra_tensors is not None and not isinstance(extra_tensors, tuple):
            raise TypeError("extra_tensors must be a tuple or None")
        self.event = event
        # Retaining outputs until the event has been consumed prevents storage
        # reuse while an asynchronous communication operation is still active.
        self.extra_tensors = extra_tensors
        self.hook_after_wait: Optional[Callable[[], Any]] = None
        self._release_handle_by_call = False

    def current_stream_wait(self, release_handle: bool = False) -> None:
        """Wait on the current stream and optionally release retained objects."""

        if not isinstance(release_handle, bool):
            raise TypeError("release_handle must be a bool")
        if self.event is None:
            raise RuntimeError("EventOverlap has no event to wait for")

        self.event.current_stream_wait()
        if self.hook_after_wait is not None:
            hook = self.hook_after_wait
            self.hook_after_wait = None
            hook()

        if release_handle:
            self.event = None
            self.extra_tensors = None

    def register_hook_after_wait(self, hook_after_wait: Callable[[], Any]) -> None:
        """Register one hook to run after the next successful stream wait."""

        if not callable(hook_after_wait):
            raise TypeError("hook_after_wait must be callable")
        if self.hook_after_wait is not None:
            raise RuntimeError("a hook is already registered on this EventOverlap")
        self.hook_after_wait = hook_after_wait

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        """Configure handle release for the next context-manager exit."""

        if not isinstance(release_handle, bool):
            raise TypeError("release_handle must be a bool")
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self) -> "EventOverlap":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if self.event is not None:
                self.current_stream_wait(release_handle=self._release_handle_by_call)
        finally:
            self._release_handle_by_call = False
