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
def test_torch_npu_import_does_not_enable_communication_or_initialize_devices(request):
    missing = [name for name in ("torch", "torch_npu") if importlib.util.find_spec(name) is None]
    if missing:
        reason = "Framework compatibility requires: " + ", ".join(missing)
        if request.config.getoption("--require-framework"):
            pytest.fail(reason)
        pytest.skip(reason)
    root = Path(__file__).resolve().parents[2]
    code = """
import json
import torch, torch_npu
assert not torch.npu.is_initialized(), 'Framework import already initialized an NPU'
import deep_ep
for construct in (lambda: deep_ep.Buffer.dispatch(None, None),
                  lambda: deep_ep.ElasticBuffer.dispatch(None, None),
                  deep_ep.Config):
    try:
        construct()
    except NotImplementedError:
        continue
    raise AssertionError('Unimplemented communication/configuration unexpectedly succeeded')
assert not torch.npu.is_initialized(), 'DeepEP declarations initialized an NPU'
print(json.dumps({'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                  'npu_initialized': torch.npu.is_initialized(), 'deep_ep': deep_ep.__file__}))
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
