"""Public spelling must not change values, defaults, events or backend policy."""

import inspect
from types import SimpleNamespace

import pytest

import deep_ep
from deep_ep import _compat, _machine
from deep_ep.backends.pod import elastic, buffer as legacy


@pytest.fixture(autouse=True)
def public_native_gate(monkeypatch):
    from deep_ep import buffer
    from deep_ep.buffers import elastic as public_elastic

    monkeypatch.setattr(buffer, "require_native", lambda _: None)
    monkeypatch.setattr(public_elastic, "require_native", lambda _: None)


def test_upstream_signature_preserves_defaults_positions_and_events(monkeypatch):
    def implementation(x, num_compute_units=32, *, previous_event=None, async_finish=False):
        return x, num_compute_units, previous_event, async_finish

    call = _compat.public_keywords(
        implementation, {"num_sms": "num_compute_units"}, aliases={"num_aiv": "num_compute_units"}
    )
    signature = inspect.signature(call)
    assert list(signature.parameters) == ["x", "num_sms", "previous_event", "async_finish"]
    assert signature.parameters["num_sms"].default == 32
    assert signature.parameters["previous_event"].kind == inspect.Parameter.KEYWORD_ONLY
    monkeypatch.setattr(_compat.inspect, "signature", lambda _: pytest.fail("hot-path reflection"))
    x, event = object(), object()
    assert call(x) == (x, 32, None, False)
    for keyword in ("num_sms", "num_compute_units", "num_aiv"):
        assert call(x, **{keyword: 64}, previous_event=event, async_finish=True) == (x, 64, event, True)
    assert call(x, 64, num_sms=64, num_aiv=64)[1] == 64
    assert call(x, num_sms=64, num_compute_units=64, num_aiv=64)[1] == 64
    for kwargs in ({"num_sms": 32}, {"num_aiv": 32}, {"num_compute_units": 32, "num_sms": 64}):
        with pytest.raises(TypeError, match="conflicting|multiple values"):
            call(x, 64, **kwargs)
    with pytest.raises(TypeError, match="unexpected"):
        call(x, unknown=1)


def test_invalid_alias_target_is_rejected_at_definition():
    with pytest.raises(ValueError, match="unknown implementation parameter"):
        _compat.public_keywords(lambda compute_read_gbs=200: None, {"sm_read_gbs": "compute_unit_read_gbs"})


@pytest.mark.parametrize("api", ["Buffer", "ElasticBuffer"])
@pytest.mark.parametrize("style", ["upstream", "platform", "both", "positional"])
def test_pod_constructors_forward_identical_values(monkeypatch, api, style):
    calls = []
    monkeypatch.setattr(_machine, "select", lambda *a, **kw: {"machine_type": "pod"})
    module = legacy if api == "Buffer" else elastic
    monkeypatch.setattr(module, "require_native", lambda _: None)
    monkeypatch.setattr(module, "warn_unclosed", lambda _: None)

    def runtime(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(module, "Runtime", runtime)
    group, capacity = object(), 128 << 20
    args, options = [group], {"machine_type": "pod", "explicitly_destroy": True}
    if api == "Buffer":
        if style in ("upstream", "both"):
            options["num_nvl_bytes"] = capacity
        if style in ("platform", "both"):
            options["num_scaleup_bytes"] = capacity
        if style == "positional":
            args.append(capacity)
    else:
        options["num_bytes"] = capacity
        if style in ("upstream", "both"):
            options["num_gpu_timeout_secs"] = 100
        if style in ("platform", "both"):
            options["num_device_timeout_secs"] = 100
        if style == "positional":
            args.extend([capacity, 0, 0, 0, 0, False, False, True, True, True, 3, 0, 300, 100])
            options.pop("num_bytes")
    instance = getattr(deep_ep, api)(*args, **options)
    assert instance.machine_type == "pod" and len(calls) == 1
    assert calls[0][0][:2] == (group, capacity)
    if api == "ElasticBuffer":
        assert calls[0][0][2] == 300
        assert calls[0][1]["udma"] is True


@pytest.mark.parametrize(
    "api,kwargs",
    [
        ("Buffer", {"num_nvl_bytes": 128, "num_scaleup_bytes": 256}),
        ("Buffer", {"allow_mnnvl": False, "allow_multinode_memory_transport": True}),
        ("ElasticBuffer", {"num_gpu_timeout_secs": 100, "num_device_timeout_secs": 200}),
    ],
)
def test_constructor_conflicts_fail_before_native_initialization(monkeypatch, api, kwargs):
    monkeypatch.setattr(_machine, "select", lambda *a, **kw: {"machine_type": "pod"})
    module = legacy if api == "Buffer" else elastic
    monkeypatch.setattr(module, "require_native", lambda _: None)
    monkeypatch.setattr(module, "warn_unclosed", lambda _: None)
    monkeypatch.setattr(module, "Runtime", lambda *a, **kw: pytest.fail("conflict reached native runtime"))
    with pytest.raises(TypeError, match="conflicting"):
        getattr(deep_ep, api)(None, machine_type="pod", **kwargs)


@pytest.mark.parametrize("method", ["get_theoretical_num_sms", "get_theoretical_num_compute_units"])
@pytest.mark.parametrize(
    "kwargs",
    [{"sm_read_gbs": 200}, {"compute_read_gbs": 200}, {"compute_unit_read_gbs": 200}],
)
def test_pod_estimator_still_rejects_unsupported_operation(method, kwargs):
    with pytest.raises(NotImplementedError, match="get_theoretical_num_compute_units"):
        getattr(deep_ep.ElasticBuffer, method)(object(), 256, 8, **kwargs)


def test_handle_core_alias_does_not_rescale_or_diverge():
    handle = object.__new__(deep_ep.EPHandle)
    handle.num_compute_units = 32
    assert handle.num_sms == 32
    handle.num_sms = 64
    assert handle.num_compute_units == 64


def test_clos_aliases_preserve_aiv_count_and_hint():
    pytest.importorskip("torch")
    upstream = deep_ep.Config(num_sms=32)
    platform = deep_ep.Config(num_compute_units=32)
    assert upstream.num_aiv == platform.num_aiv == upstream.num_sms == 32
    assert upstream.get_nvl_buffer_size_hint(12288, 16, 129) == platform.get_scaleup_buffer_size_hint(12288, 16, 129)
    with pytest.raises(TypeError, match="conflicting|multiple values"):
        deep_ep.Config(32, num_compute_units=64)
