# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Check the installed packet protocol before allocating GM or starting SHMEM."""

from pathlib import Path


def validate_layouts(python_layout, native_layout, location):
    expected = {"aggregate_protocol": 1, "aggregate_packet_bytes": 512}
    for name, layout in (("Python", python_layout), ("native", native_layout)):
        if any(layout.get(key) != value for key, value in expected.items()):
            actual = {key: layout.get(key) for key in expected}
            raise RuntimeError(
                f"Installed {name} Dispatch packet protocol is stale or incompatible: "
                f"{location}; expected={expected}, "
                f"actual={actual}. "
                "Updating the checkout and running tests does not update the installed wheel. "
                "Rebuild and reinstall the wheel in the test Python environment on every node."
            )
    differences = {
        key: (python_layout.get(key), value) for key, value in native_layout.items() if python_layout.get(key) != value
    }
    if differences:
        raise RuntimeError(
            f"Installed Python/native Dispatch layouts differ: {location}; {differences}. "
            "Rebuild and reinstall the wheel on every node."
        )


def installed_dispatch_info():
    import deep_ep
    from deep_ep.buffers._elastic_layout import dispatch_layout
    from deep_ep._native import require_native

    location = str(Path(deep_ep.__file__).resolve())
    # Pure host layout queries; no device selection, allocation or SHMEM init.
    # Check Python first so an old wheel gets an actionable error even if its
    # native extension cannot load in the current SDK environment.
    python_layout = dispatch_layout(128, 32, 6, 256, 256, True, weights=True)
    validate_layouts(python_layout, python_layout, location)
    native = require_native("Dispatch packet protocol preflight")
    native_layout = native.dispatch_layout(0, 128, 32, 6, 256, 256, True, True)
    validate_layouts(python_layout, native_layout, location)
    return dict(
        path=location,
        native_path=str(native.__file__),
        aggregate_protocol=python_layout["aggregate_protocol"],
        aggregate_packet_bytes=python_layout["aggregate_packet_bytes"],
    )
