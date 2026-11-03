"""Compatibility import for :mod:`deep_ep.backends.mesh_clos._elastic_layout`."""

import importlib
import sys

# Alias the module itself so legacy imports share classes and native state.
sys.modules[__name__] = importlib.import_module("deep_ep.backends.mesh_clos._elastic_layout")
