"""Keep API tests with fake native runtimes independent of NPU availability."""

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def cpu_api_streams(request, monkeypatch):
    if request.node.path.name not in {
        "test_public_api.py",
        "test_v2_elastic_api.py",
        "test_shmem_bootstrap.py",
        "test_udma_auto_policy.py",
    }:
        return
    import torch

    npu = getattr(torch, "npu", None)
    if npu is None:
        return

    def forbidden(*args, **kwargs):
        pytest.fail("CPU API tests must not initialize the NPU runtime")

    # The native calls and payloads are already CPU doubles, but a capture or
    # allocation-stream query can otherwise initialize a real torch_npu context.
    # Tests of particular stream/event behavior can still override these doubles.
    monkeypatch.setattr(npu, "is_current_stream_capturing", lambda: False, raising=False)
    monkeypatch.setattr(npu, "current_stream", lambda *args, **kw: SimpleNamespace(stream_id=0), raising=False)
    if hasattr(npu, "_lazy_init"):
        monkeypatch.setattr(npu, "_lazy_init", forbidden)
    import torch_npu

    if hasattr(torch_npu, "_C") and hasattr(torch_npu._C, "_npu_init"):
        monkeypatch.setattr(torch_npu._C, "_npu_init", forbidden)
