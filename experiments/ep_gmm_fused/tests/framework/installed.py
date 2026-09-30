# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""Select both DUT packages from one installed wheel inside a device worker."""

import importlib
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import sys


def load_installed_ep_gmm_fused():
    # -m adds the checkout to sys.path; PYTHONPATH can add either source root too.
    # experiments is already imported and retains its package search path.
    root = Path(__file__).resolve().parents[4]
    sources = {root, root / "experiments/ep_gmm_fused/python"}
    sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() not in sources]
    install_hint = (
        "Build with DEEPEP_BUILD_EP_GMM_FUSED=ON and install the wheel using "
        f"{sys.executable} -m pip install --force-reinstall <wheel>."
    )
    try:
        installed = distribution("ascend-deepep")
    except PackageNotFoundError as error:
        raise RuntimeError(f"No installed ascend-deepep wheel in this Python environment. {install_hint}") from error
    if installed.read_text("WHEEL") is None:
        raise RuntimeError(f"Device tests require an installed wheel, not source/egg metadata. {install_hint}")

    directories = {
        name: Path(installed.locate_file(name.replace(".", "/"))).resolve()
        for name in ("deep_ep", "deep_ep_experimental", "deep_ep_experimental.ep_gmm_fused")
    }
    for name, directory in directories.items():
        if not (directory / "__init__.py").is_file():
            raise RuntimeError(f"Installed wheel is missing {name} at {directory}. {install_hint}")
        if name != "deep_ep_experimental" and not any(
            (directory / ("_C" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES
        ):
            raise RuntimeError(f"Installed wheel is missing the compiled {name}._C extension. {install_hint}")

    # Never replace already imported modules or unload a native runtime.
    for name, module in tuple(sys.modules.items()):
        directory = directories.get(name.split(".")[0])
        if directory is not None:
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(directory):
                raise RuntimeError(
                    f"{name} was already imported from {origin}; expected {directory}. Restart the worker."
                )

    sys.path.insert(0, str(Path(installed.locate_file("")).resolve()))
    for name, directory in directories.items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != directory / "__init__.py":
            raise RuntimeError(f"{name} loaded from {module.__file__}; expected the installed wheel at {directory}.")
    return module
