"""Legacy import aliases; all implementations and native libraries live in deep_ep."""

from deep_ep import Buffer, Config, ElasticBuffer, EPHandle, EventHandle, EventOverlap

__all__ = [
    "Config",
    "Buffer",
    "ElasticBuffer",
    "EPHandle",
    "topk_idx_t",
    "EventHandle",
    "EventOverlap",
    "ExpandHandle",
    "Handle",
    "Slot",
    "SlotState",
    "BufferRuntime",
    "ShmemInstance",
]


def __getattr__(name):
    if name in {"BufferRuntime", "ShmemInstance"}:
        from deep_ep.backends.clos._native import get_native_class

        return get_native_class(name)
    if name in __all__:
        import deep_ep

        return getattr(deep_ep, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
