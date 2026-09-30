# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Explicit device opt-in and pre-device capability gates."""

import os
import pytest


DEVICE_CAPABILITIES = frozenset(
    {
        "legacy",
        "elastic-ordinary",
        "elastic-expanded",
        "elastic-fp8",
        "elastic-combine",
        "elastic-cached",
        "elastic-reinitialize",
        "elastic-single-rank",
    }
)


def pytest_addoption(parser):
    parser.addoption(
        "--require-framework",
        action="store_true",
        help="Fail rather than skip when Torch/torch_npu are unavailable.",
    )
    parser.addoption(
        "--run-device",
        action="store_true",
        help="Require selected device tests to execute under a torchrun EP launch.",
    )
    parser.addoption(
        "--device-capabilities",
        help=(
            "Comma-separated backend capabilities, or 'all'. Required with "
            "--run-device; unsupported cases are skipped before NPU setup."
        ),
    )


def _parse_capabilities(config):
    raw = config.getoption("--device-capabilities") or os.environ.get("DEEPEP_DEVICE_CAPABILITIES", "")
    if len(raw) > 512:
        raise pytest.UsageError("device capability declaration is too long")
    values = {value.strip() for value in raw.split(",") if value.strip()}
    if values == {"all"}:
        return DEVICE_CAPABILITIES
    unknown = values - DEVICE_CAPABILITIES
    if unknown:
        raise pytest.UsageError("unknown device capabilities: " + ", ".join(sorted(unknown)))
    if not values:
        raise pytest.UsageError("--run-device requires --device-capabilities (or DEEPEP_DEVICE_CAPABILITIES)")
    return frozenset(values)


def _requirements(item):
    if "/legacy/" in item.nodeid.replace("\\", "/"):
        return {"legacy"}
    params = getattr(getattr(item, "callspec", None), "params", {})
    if "spec" in params:
        spec = params["spec"]
        expanded = spec["expanded"]
        dtype = spec["dtype"]
        operation = spec["operation"]
    else:
        expanded = params.get("expanded")
        dtype = params.get("dtype", "bfloat16")
        operation = params.get("operation", "dispatch")
    required = {"elastic-expanded" if expanded else "elastic-ordinary"}
    if dtype == "float8_e4m3fn":
        required.add("elastic-fp8")
    if operation in {"combine", "roundtrip"}:
        required.add("elastic-combine")
    if "cached" in item.name:
        required.add("elastic-cached")
    if params.get("spec", {}).get("world_size") == 1:
        required.add("elastic-single-rank")
    return required


def _case_world_size(item):
    params = getattr(getattr(item, "callspec", None), "params", {})
    if "spec" in params:
        return params["spec"]["world_size"]
    return 2


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-device"):
        reason = "NPU communication not run: requires implemented backend and --run-device"
        for item in items:
            if item.get_closest_marker("device"):
                item.add_marker(pytest.mark.skip(reason=reason))
        return

    device_items = [item for item in items if item.get_closest_marker("device")]
    if not device_items:
        config._device_supported_count = 0
        return
    # Preserve the primary capability-declaration error if parsing aborts
    # collection; collection_finish must not replace it with an empty-selection
    # message.
    config._device_supported_count = -1
    capabilities = _parse_capabilities(config)
    try:
        launch_world_size = int(os.environ.get("WORLD_SIZE", ""))
    except ValueError:
        launch_world_size = None
    supported = []
    for item in device_items:
        expected_world_size = _case_world_size(item)
        if launch_world_size and expected_world_size != launch_world_size:
            item.add_marker(
                pytest.mark.skip(
                    reason=(f"world-size gate: case requires EP{expected_world_size}, launch is EP{launch_world_size}")
                )
            )
            continue
        missing = _requirements(item) - capabilities
        if missing:
            item.add_marker(
                pytest.mark.skip(reason=("backend capability gate: unsupported " + ", ".join(sorted(missing))))
            )
        else:
            supported.append(item)
    config._device_supported_count = len(supported)
    if len(supported) > 1 and "elastic-reinitialize" not in capabilities:
        raise pytest.UsageError(
            "backend cannot safely reinitialize communication buffers; select "
            "exactly one supported device case per torchrun launch"
        )


def pytest_collection_finish(session):
    if session.config.getoption("--run-device") and not getattr(session.config, "_device_supported_count", 0):
        raise pytest.UsageError("--run-device requires at least one selected, supported device test")


@pytest.fixture(scope="session")
def device_session():
    from tests.utils.envs import npu_session

    with npu_session() as session:
        yield session
