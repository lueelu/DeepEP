"""Named communication-slot and dispatch-handle lifecycle."""

from __future__ import annotations

from enum import Enum
from threading import RLock
from typing import Any, Optional
from weakref import ReferenceType, ref


class SlotState(str, Enum):
    FREE = "free"
    ACQUIRED = "acquired"
    DISPATCH_IN_FLIGHT = "dispatch_in_flight"
    DISPATCH_READY = "dispatch_ready"
    COMBINE_IN_FLIGHT = "combine_in_flight"
    RELEASABLE = "releasable"


_ALLOWED_TRANSITIONS = {
    SlotState.FREE: frozenset((SlotState.ACQUIRED,)),
    SlotState.ACQUIRED: frozenset((SlotState.FREE, SlotState.DISPATCH_IN_FLIGHT, SlotState.DISPATCH_READY)),
    SlotState.DISPATCH_IN_FLIGHT: frozenset((SlotState.DISPATCH_READY,)),
    SlotState.DISPATCH_READY: frozenset((SlotState.COMBINE_IN_FLIGHT, SlotState.RELEASABLE)),
    SlotState.COMBINE_IN_FLIGHT: frozenset((SlotState.RELEASABLE,)),
    SlotState.RELEASABLE: frozenset((SlotState.FREE,)),
}


class Slot:
    """One generation of the native Buffer's named SHMEM slot set."""

    def __init__(
        self,
        slot_id: int = 0,
        *,
        owner: Any = None,
        native_resources: Any = None,
    ) -> None:
        if not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0:
            raise ValueError("slot_id must be a non-negative integer")
        self._slot_id = slot_id
        self._owner_ref: Optional[ReferenceType[Any]] = ref(owner) if owner is not None else None
        self._native_resources = native_resources
        self._state = SlotState.FREE
        self._generation = 0
        self._held_resources: tuple[Any, ...] = ()
        self._lock = RLock()

    @property
    def slot_id(self) -> int:
        return self._slot_id

    @property
    def state(self) -> SlotState:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def owner(self) -> Any:
        return self._owner_ref() if self._owner_ref is not None else None

    @property
    def comm_meta_ptr(self) -> int:
        return int(self._require_native().get_comm_meta_ptr())

    @property
    def held_resources(self) -> tuple[Any, ...]:
        with self._lock:
            return self._held_resources

    def acquire(self) -> int:
        with self._lock:
            if self._state is not SlotState.FREE:
                raise RuntimeError(f"slot {self._slot_id} is {self._state.value}, expected free")
            self._generation += 1
            self._state = SlotState.ACQUIRED
            return self._generation

    def transition(self, target: SlotState) -> None:
        if not isinstance(target, SlotState):
            raise TypeError("target must be a SlotState")
        with self._lock:
            if target not in _ALLOWED_TRANSITIONS[self._state]:
                raise RuntimeError(f"invalid slot {self._slot_id} transition: {self._state.value} -> {target.value}")
            self._state = target
            if target is SlotState.FREE:
                self._held_resources = ()

    def hold(self, *resources: Any) -> None:
        with self._lock:
            if self._state is SlotState.FREE:
                raise RuntimeError("a FREE slot cannot hold in-flight resources")
            self._held_resources += tuple(resources)

    def release(self) -> None:
        with self._lock:
            if self._state not in (SlotState.ACQUIRED, SlotState.RELEASABLE):
                raise RuntimeError(f"slot {self._slot_id} cannot be released from {self._state.value}")
            self._state = SlotState.FREE
            self._held_resources = ()

    def validate_generation(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                raise RuntimeError(
                    f"stale handle for slot {self._slot_id}: generation {generation}, "
                    f"current generation {self._generation}"
                )
            if self._state is SlotState.FREE:
                raise RuntimeError(f"slot {self._slot_id} generation {generation} has been released")

    def _require_native(self) -> Any:
        if self._native_resources is None:
            raise RuntimeError("Slot is not bound to native Buffer resources")
        owner = self.owner
        if owner is not None:
            owner._ensure_alive()
        return self._native_resources

    def __repr__(self) -> str:
        return f"Slot(slot_id={self.slot_id}, state={self.state.value!r}, generation={self.generation})"


class Handle:
    """Dispatch ``put_offset`` plus Buffer/Slot generation ownership."""

    def __init__(
        self,
        buffer: Any,
        slot: Slot,
        put_offset: Any,
        *,
        generation: Optional[int] = None,
    ) -> None:
        if buffer is None:
            raise ValueError("buffer must not be None")
        if not isinstance(slot, Slot):
            raise TypeError("slot must be a Slot")
        if slot.owner is not buffer:
            raise ValueError("slot does not belong to the supplied Buffer")
        if put_offset is None:
            raise ValueError("put_offset must not be None")
        self._buffer = buffer
        self._slot = slot
        self._generation = slot.generation if generation is None else generation
        self._put_offset = put_offset
        self.validate(buffer)

    @property
    def buffer(self) -> Any:
        return self._buffer

    @property
    def slot(self) -> Slot:
        return self._slot

    @property
    def slot_id(self) -> int:
        return self._slot.slot_id

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def put_offset(self) -> Any:
        self.validate()
        return self._put_offset

    def validate(self, buffer: Any = None) -> None:
        if buffer is not None and buffer is not self._buffer:
            raise ValueError("Handle belongs to a different Buffer")
        if self._slot.owner is not self._buffer:
            raise RuntimeError("Handle Slot ownership no longer matches its Buffer")
        self._slot.validate_generation(self._generation)

    def __repr__(self) -> str:
        return f"Handle(slot_id={self.slot_id}, generation={self.generation})"
