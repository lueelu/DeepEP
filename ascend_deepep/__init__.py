"""Legacy import aliases; all implementations and native libraries live in deep_ep."""

from importlib import import_module as _import_module
from importlib.machinery import ModuleSpec as _ModuleSpec
import sys as _sys
from types import ModuleType as _ModuleType

from deep_ep import Buffer, Config, ElasticBuffer, EPHandle, EventHandle, EventOverlap

# Register the old paths without separate forwarding files or eager backend imports.
_SUBMODULES = {
    "buffer": ("deep_ep.buffer", ("Buffer",)),
    "elastic": ("deep_ep.buffers.elastic", ("ElasticBuffer", "EPHandle")),
    "config": ("deep_ep", ("Config",)),
    "event": ("deep_ep.utils.event", ("EventHandle", "EventOverlap")),
    "expand": ("deep_ep.backends.clos.expand", ("ExpandHandle",)),
    "resources": ("deep_ep.backends.clos.resources", ("Handle", "Slot", "SlotState")),
    "dtypes": ("deep_ep.backends.clos.dtypes", ("topk_idx_t",)),
    "utils": ("deep_ep.backends.clos.utils", ("align_up",)),
}


def _register_submodule(name, target, exports):
    fullname = f"{__name__}.{name}"
    module = _ModuleType(fullname, f"Legacy imports forwarded to {target}.")
    module.__all__ = list(exports)
    module.__file__ = __file__
    module.__package__ = fullname if name == "utils" else __name__
    module.__spec__ = _ModuleSpec(fullname, loader=None, origin=__file__, is_package=name == "utils")
    if name == "utils":
        module.__path__ = module.__spec__.submodule_search_locations

    def get_attribute(attribute):
        if attribute in exports:
            return getattr(_import_module(target), attribute)
        raise AttributeError(f"module {fullname!r} has no attribute {attribute!r}")

    module.__getattr__ = get_attribute
    module.__dir__ = lambda: sorted(set(module.__dict__) | set(exports))
    # Preserve existing module references when the compatibility root is reloaded.
    globals()[name] = _sys.modules.setdefault(fullname, module)


for _name, (_target, _exports) in _SUBMODULES.items():
    _register_submodule(_name, _target, _exports)
del _name, _target, _exports

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
