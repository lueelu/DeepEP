# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""NPU-only adapters. Framework collectives are explicit, never a fallback."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sysconfig


REQUIRED_OPERATIONS = frozenset({"Buffer.get_dispatch_layout", "Buffer.dispatch", "Buffer.combine"})


def require_tensor(torch, value, *, device, dtype, shape=None):
    if not isinstance(value, torch.Tensor) or value.device != device or value.dtype != dtype:
        raise AssertionError(f"Expected {dtype} Tensor on {device}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise AssertionError(f"Tensor shape {tuple(value.shape)} != {tuple(shape)}")


def wait_completion(event):
    if getattr(event, "event", None) is None or not callable(getattr(event, "current_stream_wait", None)):
        raise AssertionError("Asynchronous API must return a recorded completion event")
    event.current_stream_wait()


def installed_native():
    import deep_ep
    from deep_ep._native import require_native

    roots = [Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")]
    package = Path(deep_ep.__file__).resolve().parent
    if not any(package.is_relative_to(root) for root in roots):
        raise RuntimeError("Native roundtrip requires an installed wheel, not a source/editable import")
    metadata = package / "_build_info.json"
    if metadata.stat().st_size > 1 << 20:
        raise RuntimeError("Oversized build identity")
    data = metadata.read_bytes()
    info = json.loads(data)
    operations = info.get("communication_operations")
    if not isinstance(operations, list) or any(not isinstance(value, str) for value in operations):
        raise RuntimeError("Invalid communication_operations build metadata")
    missing = REQUIRED_OPERATIONS - set(operations)
    if missing:
        raise NotImplementedError("Installed wheel lacks required EP operations: " + ", ".join(sorted(missing)))
    native = require_native("V1 roundtrip")
    if Path(native.__file__).resolve().parent != package:
        raise RuntimeError("Native extension and package identity differ")
    return deep_ep, hashlib.sha256(data).hexdigest()


@dataclass
class NpuReceived:
    x: object
    indices: object
    weights: object
    counts: object
    handle: object
    event: object


class _Completion:
    def __init__(self, torch):
        self._torch = torch
        self.event = torch.npu.Event()
        self.event.record(torch.npu.current_stream())

    def current_stream_wait(self):
        self.event.wait(self._torch.npu.current_stream())


class NpuStub:
    """NPU Tensor routing with HCCL AllGather/AllReduce, NOT DeepEP kernels.

    Equal token counts, two ranks, ordinary BF16 layout, one active handle.
    Inputs/outputs, selection, expert payloads and reductions stay on NPUs.
    CPU copies are used only for public counts and correctness inspection.
    """

    def __init__(self, torch, group, rank):
        self.torch, self.group, self.rank = torch, group, rank
        self.active = None

    def dispatch(self, x, indices, weights, experts):
        torch = self.torch
        if torch.distributed.get_world_size(self.group) != 2:
            raise ValueError("NPU stub requires exactly two ranks")
        if x.device.type != "npu" or x.dtype != torch.bfloat16:
            raise ValueError("NPU stub requires NPU BF16 input")
        require_tensor(torch, indices, device=x.device, dtype=torch.int64)
        require_tensor(torch, weights, device=x.device, dtype=torch.float32, shape=indices.shape)

        def gather(tensor):
            output = [torch.empty_like(tensor) for _ in range(2)]
            torch.distributed.all_gather(output, tensor.contiguous(), group=self.group)
            return torch.cat(output, dim=0)

        all_x, all_indices, all_weights = gather(x), gather(indices), gather(weights)
        local_experts = experts // 2
        owned = (all_indices >= self.rank * local_experts) & (all_indices < (self.rank + 1) * local_experts)
        local_indices = torch.where(owned, all_indices % local_experts, -1)
        positions = owned.any(dim=1).nonzero().flatten()
        received_x = all_x.index_select(0, positions)
        received_indices = local_indices.index_select(0, positions)
        received_weights = all_weights.index_select(0, positions)
        counts = [int((received_indices == expert).sum().item()) for expert in range(local_experts)]
        self.active = (positions, received_indices, x.shape[0], x.shape[1])
        return NpuReceived(received_x, received_indices, received_weights, counts, self.active, _Completion(torch))

    def combine(self, x, handle, values):
        if handle is None or handle is not self.active:
            raise ValueError("Foreign, stale or consumed NPU stub handle")
        torch = self.torch
        positions, indices, tokens, hidden = handle
        require_tensor(torch, x, device=indices.device, dtype=torch.bfloat16, shape=(len(positions), hidden))
        require_tensor(torch, values, device=x.device, dtype=torch.float32, shape=indices.shape)
        payload = torch.zeros((2 * tokens, hidden), dtype=torch.float32, device=x.device)
        returned = torch.zeros((2 * tokens, indices.shape[1]), dtype=torch.float32, device=x.device)
        payload.index_copy_(0, positions, x.float())
        returned.index_copy_(0, positions, torch.where(indices >= 0, values, 0.0))
        torch.distributed.all_reduce(payload, group=self.group)
        torch.distributed.all_reduce(returned, group=self.group)
        self.active = None
        start = self.rank * tokens
        return (
            payload[start : start + tokens].to(torch.bfloat16),
            returned[start : start + tokens].clone(),
            _Completion(torch),
        )

    def destroy(self):
        self.active = None


class NativeV1:
    """Only installed public Buffer methods; no private kernel imports."""

    def __init__(self, module, group, case):
        configs = [module.Buffer.get_dispatch_config(2), module.Buffer.get_combine_config(2)]
        self.buffer = module.Buffer(
            group,
            num_scaleup_bytes=max(config.get_scaleup_buffer_size_hint(case.hidden * 2, 2) for config in configs),
            num_scaleout_bytes=max(config.get_scaleout_buffer_size_hint(case.hidden * 2, 2) for config in configs),
            explicitly_destroy=True,
        )

    def dispatch(self, x, indices, weights, experts):
        per_rank, per_scaleout, per_expert, mask, event = self.buffer.get_dispatch_layout(
            indices, experts, async_finish=True
        )
        wait_completion(event)
        result = self.buffer.dispatch(
            x,
            topk_idx=indices,
            topk_weights=weights,
            num_tokens_per_rank=per_rank,
            num_tokens_per_scaleout_rank=per_scaleout,
            num_tokens_per_expert=per_expert,
            is_token_in_rank=mask,
            async_finish=True,
        )
        return NpuReceived(*result)

    def combine(self, x, handle, values):
        return self.buffer.combine(x, handle, topk_weights=values, async_finish=True)

    def destroy(self):
        self.buffer.destroy()
