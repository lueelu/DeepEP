# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Private native-adapter contract. PR3 does not implement a communication backend."""

from importlib import import_module
from typing import Protocol


class BenchmarkRuntime(Protocol):
    """Private protocol v1; see docs/performance.md for operation semantics.

    A context-managed factory owns only resources it creates. All collective
    operations and teardown must have bounded timeouts. No adapter is supplied
    by this protocol declaration.
    """

    rank: int
    world_size: int
    metadata: dict

    def agree_config(self, fingerprint: str) -> None: ...
    def prepare(self) -> None: ...
    def verify(self) -> None: ...
    def all_ranks_ok(self, local_ok: bool) -> bool: ...
    def barrier(self) -> None: ...
    def synchronize(self) -> None: ...
    def invoke(self) -> object: ...
    def wait(self, output: object) -> None: ...
    def release(self, output: object) -> None: ...
    def gather_samples(self, samples: list) -> list: ...


class BackendUnavailable(RuntimeError):
    pass


def resolve_backend(config):
    """Only a fixed first-party module may supply the adapter; no plugin from CLI.

    describe_benchmark(config_dict) is a CPU-only, side-effect-free native
    capability/shape check. It must reject unsupported configurations without
    importing torch_npu, initializing NPU/HCCL/SHMEM or allocating buffers.
    create_benchmark(config_dict) runs only after this gate.
    """
    native = import_module("deep_ep._native")
    describe = getattr(native, "describe_benchmark", None)
    factory = getattr(native, "create_benchmark", None)
    if not callable(describe) or not callable(factory):
        raise BackendUnavailable("native benchmark adapter is not implemented; use --dry-run to inspect the plan")
    descriptor = describe(config.to_dict())
    if (
        not isinstance(descriptor, dict)
        or type(descriptor.get("protocol_version")) is not int
        or descriptor["protocol_version"] != 1
    ):
        raise BackendUnavailable("native benchmark adapter protocol must be version 1")
    supported = descriptor.get("capabilities")
    if not isinstance(supported, (list, tuple)) or any(not isinstance(c, str) for c in supported):
        raise BackendUnavailable("native benchmark capabilities must be a string array")
    missing = config.required_capabilities() - set(supported)
    if missing:
        raise BackendUnavailable("unsupported benchmark capabilities: " + ", ".join(sorted(missing)))
    if descriptor.get("shape_supported") is not True:
        raise BackendUnavailable("native backend does not support the requested shape")
    return factory
