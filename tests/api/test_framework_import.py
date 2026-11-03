# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Compatibility with an installed framework; no NPU allocation or collective."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.framework
def test_framework_import_and_uninitialized_calls_do_not_initialize_devices(request):
    missing = [name for name in ("torch", "torch_npu") if importlib.util.find_spec(name) is None]
    if missing:
        reason = "Framework compatibility requires: " + ", ".join(missing)
        if request.config.getoption("--require-framework"):
            pytest.fail(reason)
        pytest.skip(reason)
    root = Path(__file__).resolve().parents[2]
    code = """
import json
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import pytest
import torch, torch_npu
assert not torch.npu.is_initialized(), 'Framework import already initialized an NPU'
import deep_ep
import sys
assert 'deep_ep._C' not in sys.modules, 'DeepEP import eagerly loaded the native extension'
directory = Path(deep_ep.__file__).resolve().parent
native_present = any((directory / ('_C' + suffix)).is_file() for suffix in EXTENSION_SUFFIXES)
# Without a constructed Buffer, this calls the inherited Mesh+Clos entry points;
# it does not select a Direct backend or exercise communication.
checks = (
    (deep_ep.Buffer.dispatch, NotImplementedError, 'Buffer.dispatch is not implemented in DeepEP.'),
    (deep_ep.ElasticBuffer.dispatch,
     RuntimeError if native_present else NotImplementedError,
     'Buffer runtime was not initialized' if native_present
     else 'ElasticBuffer.dispatch is not implemented in DeepEP.'),
)
for operation, error_type, message in checks:
    with pytest.raises(error_type) as caught:
        operation(None, None)
    assert type(caught.value) is error_type
    assert str(caught.value).startswith(message), str(caught.value)
    assert not torch.npu.is_initialized(), 'An uninitialized call initialized an NPU'
assert deep_ep.Config().num_compute_units == 32
assert not torch.npu.is_initialized(), 'DeepEP declarations initialized an NPU'
print(json.dumps({'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                  'native_present': native_present, 'npu_initialized': torch.npu.is_initialized(),
                  'deep_ep': deep_ep.__file__}))
"""
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONNOUSERSITE="1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
