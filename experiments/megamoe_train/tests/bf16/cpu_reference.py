# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""Torch CPU FP64 truth at the public x/ids/weights/w1/w2 boundary.

Each expert chunk builds a CPU forward graph and uses autograd for all four
input gradients. No BF16 intermediate rounding, saved-state compression,
router recomputation, or fused-operator calls enter this mathematical truth.
"""

import math

import torch
import torch.nn.functional as F


@torch.enable_grad()
def run_reference(inputs, expert_start, chunk_rows=256, swiglu_limit=0.0):
    """Return owned expert contributions; the runner reduces token results.

    Inputs are CPU Tensor snapshots with canonical w1=[E,2F,H], w2=[E,H,F].
    Returned floating-point tensors are detached CPU FP64 values. Chunking
    bounds the live autograd graph rather than retaining every routed token.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    x, ids, p, w1, w2, dy = (inputs[key].detach() for key in ("x", "ids", "p", "w1", "w2", "dy"))
    if any(value.device.type != "cpu" for value in (x, ids, p, w1, w2, dy)):
        raise ValueError("CPU truth requires CPU Tensor snapshots")
    if x.ndim != 2 or ids.ndim != 2 or ids.shape != p.shape or ids.shape[0] != len(x) or dy.shape != x.shape:
        raise ValueError("Invalid token, route, probability, or upstream-gradient shape")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[1] != 2 * w2.shape[2] or w2.shape[:2] != (len(w1), x.shape[1]):
        raise ValueError("Expected canonical w1=[E,2F,H], w2=[E,H,F]")
    if w1.shape[2] != x.shape[1] or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Invalid weight hidden dimension or expert-ID dtype")
    sorted_ids = ids.sort(dim=1).values
    if bool((ids < 0).any()) or bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
        raise ValueError("Expert IDs must be nonnegative and distinct per token")
    if any(not bool(torch.isfinite(value).all()) for value in (x, p, w1, w2, dy)):
        raise ValueError("Non-finite reference input")
    limit = torch.tensor(swiglu_limit, dtype=torch.float32, device="cpu").item()
    if not math.isfinite(limit):
        raise ValueError("swiglu_limit must be representable as finite FP32")
    x, p, dy = x.double(), p.double(), dy.double()
    result = {
        "y": torch.zeros_like(x),
        "dx": torch.zeros_like(x),
        "dtopk_weights": torch.zeros_like(p),
        "dw1": torch.zeros(w1.shape, dtype=torch.float64, device="cpu"),
        "dw2": torch.zeros(w2.shape, dtype=torch.float64, device="cpu"),
        "counts": torch.zeros(len(w1), dtype=torch.int64, device="cpu"),
    }
    for expert in range(len(w1)):
        rows, slots = (ids == expert_start + expert).nonzero(as_tuple=True)
        result["counts"][expert] = rows.numel()
        if not rows.numel():
            continue
        a = w1[expert].double().requires_grad_(True)
        b = w2[expert].double().requires_grad_(True)
        for start in range(0, rows.numel(), chunk_rows):
            token, route = rows[start : start + chunk_rows], slots[start : start + chunk_rows]
            chunk_x = x[token].detach().requires_grad_(True)
            probability = p[token, route, None].detach().requires_grad_(True)
            gate, up = (chunk_x @ a.T).chunk(2, dim=-1)
            if limit > 0:
                gate, up = gate.clamp_max(limit), up.clamp(-limit, limit)
            output = ((F.silu(gate) * up) * probability) @ b.T
            dx, dp, dw1, dw2 = torch.autograd.grad(output, (chunk_x, probability, a, b), grad_outputs=dy[token])
            with torch.no_grad():
                result["y"].index_add_(0, token, output.detach())
                result["dx"].index_add_(0, token, dx)
                result["dtopk_weights"][token, route] += dp[:, 0]
                result["dw1"][expert].add_(dw1)
                result["dw2"][expert].add_(dw2)
    return result
