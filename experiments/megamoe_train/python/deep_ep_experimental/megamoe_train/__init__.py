# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026
# Modified by xixing 2026

"""Ascend950 MegaMoE training. Native loading happens on construction."""

__all__ = ["MegaMoeBF16Train"]


def __getattr__(name):
    if name == "MegaMoeBF16Train":
        from .bf16 import MegaMoeBF16Train

        return MegaMoeBF16Train
    raise AttributeError(name)
