"""Dtypes shared by the Python API and the current native kernels."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ._build_config import NUM_TOPK_IDX_BITS as _BUILD_NUM_TOPK_IDX_BITS


_KERNEL_BUILD_CONFIG = Path(__file__).resolve().parent / "lib" / "topk_idx_bits.txt"
if _KERNEL_BUILD_CONFIG.exists():
    _NUM_TOPK_IDX_BITS = int(_KERNEL_BUILD_CONFIG.read_text(encoding="ascii").strip())
else:
    _NUM_TOPK_IDX_BITS = int(os.environ.get("EP_NUM_TOPK_IDX_BITS", str(_BUILD_NUM_TOPK_IDX_BITS)))
if _NUM_TOPK_IDX_BITS not in (32, 64):
    raise RuntimeError("EP_NUM_TOPK_IDX_BITS must be either 32 or 64")

topk_idx_t = torch.int32 if _NUM_TOPK_IDX_BITS == 32 else torch.int64
topk_idx_bytes = _NUM_TOPK_IDX_BITS // 8


__all__ = ["topk_idx_t"]
