# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026
# Modified by xixing 2026

"""Ascend950 MegaMoE training. Native loading happens on construction."""

__all__ = ["SymmBuffer", "bf16_mega_moe_train", "MegaMoeBF16Train"]


def __getattr__(name):
    if name in ("SymmBuffer", "bf16_mega_moe_train"):
        from . import api

        return getattr(api, name)
    if name == "MegaMoeBF16Train":
        from .bf16 import MegaMoeBF16Train

        return MegaMoeBF16Train
    raise AttributeError(name)
