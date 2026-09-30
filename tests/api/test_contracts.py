# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Compare declarations with the offline, platform-adapted API contract."""

import copy
import importlib
import inspect
import json
from pathlib import Path

import pytest

import deep_ep


BASELINE = json.loads(Path(__file__).with_name("api_contract.json").read_text(encoding="utf-8"))
METHODS = [
    (name, method, spec)
    for name, object_spec in BASELINE["objects"].items()
    for method, spec in object_spec["methods"].items()
]


def test_public_exports_and_type_identity():
    assert deep_ep.__all__ == BASELINE["exports"]
    for name, spec in BASELINE["objects"].items():
        assert getattr(deep_ep, name) is getattr(importlib.import_module(spec["module"]), name)
    from deep_ep.buffers import Buffer, ElasticBuffer, EPHandle
    from deep_ep.utils import EventHandle, EventOverlap

    assert (Buffer, ElasticBuffer, EPHandle) == (
        deep_ep.Buffer,
        deep_ep.ElasticBuffer,
        deep_ep.EPHandle,
    )
    assert (EventHandle, EventOverlap) == (deep_ep.EventHandle, deep_ep.EventOverlap)


@pytest.mark.parametrize("name,spec", BASELINE["objects"].items())
def test_public_methods_have_no_undeclared_aliases(name, spec):
    public = {
        key
        for key, value in vars(getattr(deep_ep, name)).items()
        if not key.startswith("_") and (callable(value) or isinstance(value, staticmethod))
    }
    assert public == {key for key in spec["methods"] if not key.startswith("_")}


@pytest.mark.parametrize("name,method,spec", METHODS, ids=[f"{n}.{m}" for n, m, _ in METHODS])
def test_signature_matches_project_contract(name, method, spec):
    expected = copy.deepcopy(spec["parameters"])
    for deviation in BASELINE["deviations"]:
        if deviation["symbol"] == f"{name}.{method}":
            parameter = next(p for p in expected if p["name"] == deviation["parameter"])
            assert parameter["default"] == deviation["upstream_default"]
            parameter["default"] = deviation["project_default"]
    cls = getattr(deep_ep, name)
    actual = []
    for parameter in inspect.signature(getattr(cls, method)).parameters.values():
        value = {"name": parameter.name, "kind": parameter.kind.name}
        if parameter.default is not inspect.Parameter.empty:
            value["default"] = parameter.default
        actual.append(value)
    assert actual == expected
    assert isinstance(inspect.getattr_static(cls, method), staticmethod) == (spec["kind"] == "staticmethod")


def test_snapshot_pins_sources_and_does_not_claim_implementation():
    assert BASELINE["schema_version"] == 3
    assert "not an exact upstream signature snapshot" in BASELINE["contract"]
    assert set(BASELINE["platform_naming"]) == {
        "scaleup",
        "scaleout",
        "memory_transport",
        "qps",
        "compute_units",
        "device",
        "aligned_col_major_sf",
        "compatibility",
    }
    assert BASELINE["upstream"]["revision"] == "a56d6156febcd9976e55adc85b5155bfac9f28f8"
    assert all(len(digest) == 64 for digest in BASELINE["upstream"]["sources"].values())
    assert BASELINE["implementation_status"] == "declarations_only"
    assert set(BASELINE["exports"]) == set(BASELINE["objects"])


def test_no_variadic_escape_hatch_for_unknown_arguments():
    with pytest.raises(TypeError, match="unexpected keyword"):
        deep_ep.ElasticBuffer.dispatch(object(), object(), made_up_option=True)
    with pytest.raises(TypeError, match="missing"):
        deep_ep.ElasticBuffer.combine(object(), object())


@pytest.mark.parametrize(
    "operation,old,new",
    [
        (deep_ep.Config, "num_max_rdma_chunked_send_tokens", "num_max_scaleout_chunked_send_tokens"),
        (deep_ep.Config, "num_max_rdma_chunked_recv_tokens", "num_max_scaleout_chunked_recv_tokens"),
        (deep_ep.Buffer, "num_rdma_bytes", "num_scaleout_bytes"),
        (deep_ep.Buffer, "allow_scaleup_for_low_latency_mode", "allow_memory_transport_for_low_latency_mode"),
        (deep_ep.Buffer, "allow_multinode_scaleup", "allow_multinode_memory_transport"),
        (deep_ep.Buffer.dispatch, "num_tokens_per_rdma_rank", "num_tokens_per_scaleout_rank"),
        (deep_ep.ElasticBuffer.get_theoretical_num_compute_units, "rdma_gbs", "scaleout_gbs"),
    ],
)
def test_transport_keyword_migration_has_no_implicit_alias(operation, old, new):
    signature = inspect.signature(operation)
    value = object()
    assert signature.bind_partial(**{new: value}).arguments[new] is value
    with pytest.raises(TypeError, match="unexpected keyword"):
        signature.bind_partial(**{old: value})


@pytest.mark.parametrize(
    "cls,old,new",
    [
        (deep_ep.Config, "get_rdma_buffer_size_hint", "get_scaleout_buffer_size_hint"),
        (deep_ep.Buffer, "get_low_latency_rdma_size_hint", "get_low_latency_scaleout_size_hint"),
    ],
)
def test_transport_size_hint_migration_has_no_implicit_alias(cls, old, new):
    assert callable(getattr(cls, new))
    assert not hasattr(cls, old)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mode", [None, False, True])
def test_v2_caller_shapes_bind_without_executing(cached, mode):
    arguments = dict(
        x=object(),
        num_experts=64,
        allocate_on_comm_stream=True,
        previous_event=object(),
        topk_weights=object(),
    )
    if cached:
        arguments["handle"] = object()
    else:
        arguments["topk_idx"] = object()
    if mode is not None:
        arguments["do_expand"] = mode
    bound = inspect.signature(deep_ep.ElasticBuffer.dispatch).bind(object(), **arguments)
    bound.apply_defaults()
    assert bound.arguments["do_expand"] is mode
    assert bound.arguments["allocate_on_comm_stream"] is True


def test_v2_output_and_event_annotations_remain_documented():
    dispatch = str(inspect.signature(deep_ep.ElasticBuffer.dispatch).return_annotation)
    combine = str(inspect.signature(deep_ep.ElasticBuffer.combine).return_annotation)
    assert "EPHandle" in dispatch and "EventOverlap" in dispatch
    assert "EventOverlap" in combine
