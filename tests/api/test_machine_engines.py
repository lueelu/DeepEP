"""Public machine policy reaches the native engine without an override path."""

import inspect
from types import SimpleNamespace

import pytest

from deep_ep import _machine


@pytest.fixture
def public_runtime(monkeypatch):
    pytest.importorskip("torch")
    import deep_ep
    from deep_ep.backends import mesh_pcie as _card_adapter
    from deep_ep import buffer as legacy
    from deep_ep.buffers import elastic
    from deep_ep.backends.direct import buffer as source_v1, elastic as source_v2

    calls = []
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)

    def gather(_, record):
        return [record, record]

    monkeypatch.setattr(_machine, "gather_records", gather)
    monkeypatch.setattr(_card_adapter, "gather_records", gather)
    monkeypatch.setattr(_card_adapter, "host_identity", lambda: "one-host")
    monkeypatch.setattr(elastic, "require_native", lambda _: None)
    monkeypatch.setattr(legacy, "require_native", lambda _: None)

    class Native:
        def __init__(self, *args):
            assert len(args) == 18
            assert type(args[16]) is int and args[16] in (1, 2)
            assert type(args[17]) is bool and args[17] == (args[3] == "udma")
            calls.append(("native", args[3]))

        def mte_mapping_errors(self):
            calls.append(("mapping",))
            return []

        def num_device_aiv(self):
            return 64

    def bootstrap(*args, **kwargs):
        calls.append(("bootstrap", args[5]))
        if args[5] == "udma":
            assert kwargs["udma_auto_options"][0] == "clos-v4"
        return [], "test"

    monkeypatch.setattr(source_v1, "_is_capturing", lambda: False)
    for module in (source_v1, source_v2):
        monkeypatch.setattr(module, "get_native_class", lambda _: Native)
        monkeypatch.setattr(module, "_prepare_shmem_bootstrap", bootstrap)
        monkeypatch.setattr(module, "_create_communication_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(source_v2, "_validate_group_consensus", lambda *args: None)
    monkeypatch.setattr(source_v2.ElasticBuffer, "barrier", lambda *args, **kw: calls.append(("barrier",)))
    return deep_ep, group, calls


@pytest.mark.parametrize("api", ["Buffer", "ElasticBuffer"])
@pytest.mark.parametrize("machine,engine", [("global_clos", "udma"), ("mesh_pcie", "mte")])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("environment", ["mte", "udma", "invalid"])
@pytest.mark.parametrize(
    "legacy", [{}, *({"transport": value} for value in (None, "auto", "mte", "udma", "roce", "invalid"))]
)
def test_machine_fixes_engine_before_bootstrap(
    public_runtime, monkeypatch, api, machine, engine, automatic, environment, legacy
):
    package, group, calls = public_runtime
    detections = []

    def detect(_):
        detections.append(True)
        return {"machine_type": machine, "source": "test"}

    monkeypatch.setattr(_machine, "detect", detect)
    monkeypatch.setenv("ASCEND_DEEPEP_TRANSPORT", environment)
    cls = getattr(package, api)
    assert "transport" not in inspect.signature(cls).parameters
    instance = cls(group, machine_type="auto" if automatic else machine, num_bytes=128 * 1024 * 1024, **legacy)
    try:
        assert detections == ([True] if automatic else [])
        assert instance.machine_type == machine and instance.transport == engine
        if machine == "global_clos":
            policy = instance._udma_auto_policy
            assert (policy.peers_per_worker, policy.workers_per_peer, policy.wqes_per_visit) == (1, 2, 8)
            assert policy.config(20).num_aiv == 20
        else:
            assert instance._udma_auto_policy is None
        assert calls == (
            [("bootstrap", engine), ("native", engine)]
            + ([("mapping",)] if machine == "mesh_pcie" else [])
            + ([("barrier",)] if api == "ElasticBuffer" else [])
        )
    finally:
        # CPU doubles own no native resources.
        instance._destroyed = True


@pytest.mark.parametrize("api", ["Buffer", "ElasticBuffer"])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("value", [None, "auto", "mte", "udma", "roce", "invalid"])
def test_legacy_transport_does_not_reach_pod_constructor(public_runtime, monkeypatch, api, automatic, value):
    package, group, calls = public_runtime
    monkeypatch.setattr(_machine, "detect", lambda _: {"machine_type": "mesh_clos", "source": "test"})
    cls = getattr(package, api)

    def initialize(self, *args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(cls.__bases__[0], "__init__", initialize)
    instance = cls(group, machine_type="auto" if automatic else "mesh_clos", transport=value, explicitly_destroy=True)
    assert instance.machine_type == "mesh_clos"
    assert calls == [((group,), {"explicitly_destroy": True})]
