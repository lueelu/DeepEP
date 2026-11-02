"""Migrated public helpers and Clos ABI contracts; CPU/framework only."""

import inspect
import pytest

torch = pytest.importorskip("torch")
import deep_ep  # noqa: E402
from deep_ep.backends.clos import Config as OriginalConfig  # noqa: E402
from deep_ep.backends.clos.elastic import EPHandle as ClosHandle  # noqa: E402
from deep_ep._clos_adapter import Buffer, ElasticBuffer  # noqa: E402


def test_platform_aliases_preserve_source_size_hints_including_sixteen_ranks():
    canonical = deep_ep.Config(num_compute_units=32, udma_peers_per_worker=2)
    old = deep_ep.Config(num_sms=32, udma_peers_per_worker=2)
    for world in (2, 8, 16, 64):
        expected = OriginalConfig(32, udma_peers_per_worker=2).get_nvl_buffer_size_hint(12288, world, 129)
        assert canonical.get_scaleup_buffer_size_hint(12288, world, 129) == expected
        assert old.get_nvl_buffer_size_hint(12288, world, 129) == expected
    with pytest.raises(TypeError, match="conflicting"):
        deep_ep.Config(num_compute_units=20, num_sms=32)


def test_selected_clos_api_publishes_upstream_names():
    assert "num_nvl_bytes" in inspect.signature(Buffer).parameters
    assert "num_gpu_timeout_secs" in inspect.signature(ElasticBuffer).parameters
    assert "num_sms" in inspect.signature(ElasticBuffer.dispatch).parameters
    assert "num_sms" in inspect.signature(ElasticBuffer.combine).parameters


def test_public_handle_recognizes_both_backend_types_but_pod_checks_remain_specific():
    h = object.__new__(ClosHandle)
    h.num_sms = 32
    assert isinstance(h, deep_ep.EPHandle)
    assert h.num_compute_units == 32
    assert not issubclass(type(h), deep_ep.EPHandle)
    assert isinstance(object.__new__(deep_ep.EPHandle), deep_ep.EPHandle)


def test_cpu_helpers_do_not_initialize_npu():
    before = torch.npu.is_initialized() if hasattr(torch, "npu") else False
    deep_ep.Config(num_compute_units=64).get_scaleup_buffer_size_hint(256, 16, 17)
    assert (torch.npu.is_initialized() if hasattr(torch, "npu") else False) == before


def test_clos_accepts_public_event_dependencies_without_recording_or_waiting():
    from deep_ep.backends.clos.elastic import _validate_event_dependency, _validate_epilogue_event

    event = object.__new__(deep_ep.EventHandle)
    _validate_event_dependency(event, True)
    _validate_epilogue_event(event)
    with pytest.raises(ValueError):
        _validate_event_dependency(event, False)
    with pytest.raises(TypeError):
        _validate_epilogue_event(object())


def test_clos_event_imports_are_the_public_classes():
    from deep_ep.backends import clos
    from deep_ep.backends.clos import event

    assert clos.EventHandle is event.EventHandle is deep_ep.EventHandle
    assert clos.EventOverlap is event.EventOverlap is deep_ep.EventOverlap
