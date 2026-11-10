# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DeepEP-compatible tensor and arithmetic helpers for Ascend."""

from __future__ import annotations

from typing import Tuple

import torch


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return DeepEP's normalized tensor difference metric."""

    x_value, y_value = x.double() + 1, y.double() + 1
    denominator = (x_value * x_value + y_value * y_value).sum()
    similarity = 2 * (x_value * y_value).sum() / denominator
    return (1 - similarity).item()


def safe_div(a, b) -> float:
    """Divide while retaining DeepEP's zero-over-zero convention."""

    try:
        return a / b
    except ZeroDivisionError:
        if a == 0:
            return 0
        raise


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def per_token_cast_to_fp8(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize each group of 128 values to E4M3 with FP32 scales."""

    if x.dim() != 2:
        raise AssertionError("x must be a 2D tensor")
    num_tokens, hidden = x.shape
    aligned_hidden = align(hidden, 128)
    x_padded = torch.nn.functional.pad(x, (0, aligned_hidden - hidden), mode="constant", value=0)
    x_view = x_padded.view(num_tokens, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(num_tokens, -1).clamp(1e-4)
    x_fp8 = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return (
        x_fp8.view(num_tokens, aligned_hidden)[:, :hidden].contiguous(),
        (x_amax / 448.0).view(num_tokens, -1),
    )


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor) -> torch.Tensor:
    """Restore a tensor produced by :func:`per_token_cast_to_fp8`."""

    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)
    if x_fp8.dim() != 2:
        raise AssertionError("x_fp8 must be a 2D tensor")
    num_tokens, hidden = x_fp8.shape
    aligned_hidden = align(hidden, 128)
    x_padded = torch.nn.functional.pad(x_fp8, (0, aligned_hidden - hidden), mode="constant", value=0)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_padded.to(torch.float32).view(num_tokens, -1, 128)
    scales = x_scales.view(num_tokens, -1, 1)
    return (x_fp32 * scales).view(num_tokens, aligned_hidden).to(torch.bfloat16)[:, :hidden].contiguous()


def inplace_unique(x: torch.Tensor, num_slots: int) -> None:
    """Replace each row with its descending unique nonnegative slot IDs."""

    if x.dim() != 2:
        raise AssertionError("x must be a 2D tensor")
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    sorted_count, sorted_idx = torch.sort(bin_count[:, :num_slots], dim=-1, descending=True)
    sorted_idx.masked_fill_(sorted_count == 0, -1)
    sorted_idx = torch.sort(sorted_idx, descending=True, dim=-1).values
    x.fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_idx[:, :valid_len]


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    num_tokens, num_experts = scores.shape
    grouped_scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1)
    return (grouped_scores * mask.expand_as(grouped_scores)).view(num_tokens, num_experts)


def hash_tensor(tensor: torch.Tensor) -> int:
    return tensor.view(torch.int).sum().item()


def hash_tensors(*tensors) -> int:
    value = 0
    for tensor in tensors:
        if isinstance(tensor, (tuple, list)):
            value ^= hash_tensors(*tensor)
        elif tensor is not None and isinstance(tensor, torch.Tensor):
            value ^= hash_tensor(tensor)
    return value


def count_bytes(*tensors) -> int:
    total = 0
    for tensor in tensors:
        if isinstance(tensor, (tuple, list)):
            total += count_bytes(*tensor)
        elif tensor is not None:
            total += tensor.numel() * tensor.element_size()
    return total


__all__ = [
    "align",
    "calc_diff",
    "ceil_div",
    "count_bytes",
    "create_grouped_scores",
    "hash_tensor",
    "hash_tensors",
    "inplace_unique",
    "per_token_cast_back",
    "per_token_cast_to_fp8",
    "safe_div",
]
