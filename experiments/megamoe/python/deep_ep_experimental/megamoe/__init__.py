# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Ascend MegaMoE Beta. No framework import or device initialization on import."""

from .api import SymmBuffer, fp8_fp4_mega_moe, mega_moe, transform_weights_for_mega_moe
from deep_ep._store import StoreGroup

__all__ = ["SymmBuffer", "StoreGroup", "mega_moe", "fp8_fp4_mega_moe", "transform_weights_for_mega_moe"]
