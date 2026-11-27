# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Thin inference-only API; all algorithm selection remains in the native layer."""

from __future__ import annotations

from functools import lru_cache
import importlib
import threading
from typing import TYPE_CHECKING

from deep_ep._runtime import integer, warn_unclosed
from ._runtime import _MegaRuntime
from deep_ep._native import npu_api, require_native
from deep_ep._store import StoreGroup
from .validation import collective_check
from .numerics import normalize_clamp

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributed import ProcessGroup

    WeightPair = tuple[Tensor, Tensor]


@lru_cache(maxsize=1)
def _extension():
    core = require_native("MegaMoE")
    if not hasattr(core, "create_megamoe_runtime"):
        raise RuntimeError("Install a wheel built with DEEPEP_BUILD_MEGAMOE=ON")
    try:
        extension = importlib.import_module(__package__ + "._C")
    except ImportError as error:
        raise ImportError("Cannot load MegaMoE Beta: check matching Torch, torch_npu, CANN and SHMEM") from error
    if getattr(extension, "api_protocol", None) != 11:
        raise RuntimeError("MegaMoE Python/native versions differ; rebuild and reinstall the complete Beta wheel")
    # Device code is compiled on demand from the sources bundled in the core
    # wheel, so bind the runtime compiler to that installation exactly once.
    import os
    from pathlib import Path

    import deep_ep

    extension.init_megamoe_jit(
        str(Path(deep_ep.__file__).resolve().parent), os.environ.get("ASCEND_HOME_PATH", "")
    )
    return extension


def _pair(value, name):
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be (quantized_data, E8M0_scales)")
    return value


def _check_fp4_weight_format(weight, name):
    import torch_npu

    actual = torch_npu.get_npu_format(weight)
    # 请求 NZ(29) 不等于返回码必为29：torch_npu 将 FP4 C0=32 映射成 uint8 C0=16(50)。
    # 不能放行所有 NZ 子格式，uint8 C0=32(51) 并非本路径的打包布局。
    if actual not in (29, 50):
        raise ValueError(
            f"{name}: packed FP4 weights require NZ format 29 or 50 (NZ_C0_16), "
            f"got format={actual}, dtype={weight.dtype}, shape={tuple(weight.shape)}; "
            "call transform_weights_for_mega_moe on raw ND weights once"
        )


_DEFAULT_TIMEOUT = 120


class SymmBuffer:
    """Collective resource owner. All ranks construct/call/destroy in the same order.

    num_experts is global; intermediate_hidden is the post-SwiGLU width.
    This Beta supports nonempty, same-BS calls and one in-flight operation per buffer.
    Use base to share one symmetric allocation and local Plan workspace across sequential configs.
    """

    def __init__(
        self,
        group: ProcessGroup | StoreGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        num_shared_experts: int = 0,
        mma_type: str = "fp8xfp4",
        activation: str = "swiglu",
        base: SymmBuffer | None = None,
        *,
        ranks_per_node: int = 0,
        topk_weights_type: int = 0,
    ):
        import torch
        import torch.distributed as dist

        if not isinstance(group, (StoreGroup, dist.ProcessGroup)):
            raise TypeError("Pass a StoreGroup or an initialized ProcessGroup explicitly")
        size = group.size() if isinstance(group, StoreGroup) else dist.get_world_size(group)
        from .backend import select_backend

        self._backend_name = select_backend(group)
        urma = self._backend_name == "urma"
        error = None
        try:
            integer("num_experts", num_experts, 1, 16384)
            integer("num_max_tokens_per_rank", num_max_tokens_per_rank, 1, 16384 if urma else 65536)
            integer("num_topk", num_topk, 1, min(8 if urma else 32, num_experts))
            integer("hidden", hidden, 256 if urma else 1024, 8192)
            integer("intermediate_hidden", intermediate_hidden, 256, 16384 if urma else 4096)
            integer("num_shared_experts", num_shared_experts, 0, 1 if urma else 8)
            integer("world_size", size, 2, 128)
            integer("ranks_per_node", ranks_per_node, 0, size)
            integer("topk_weights_type", topk_weights_type, 0, 1)
            h_align, i_align = (256, 256) if urma else (64, 128)
            if hidden % h_align or intermediate_hidden % i_align:
                raise ValueError(f"hidden must be divisible by {h_align}; intermediate_hidden by {i_align}")
            if urma and (size not in (4, 8, 16) or mma_type != "fp8xfp4"):
                raise ValueError("URMA supports EP4/EP8/EP16 and routed FP8xFP4")
            if activation != "swiglu" or mma_type not in {"fp8xfp4", "fp8xfp8"}:
                raise ValueError("Only SwiGLU with FP8xFP4/FP8xFP8 is implemented; no BF16xBF16")
            if mma_type != "fp8xfp4" and num_shared_experts:
                raise ValueError("Shared experts currently require FP8xFP4 weights")
            if num_experts % size or num_experts // size > 128:
                raise ValueError("Experts must divide EP, with at most 128 local experts")
            if ranks_per_node and size % ranks_per_node:
                raise ValueError("ranks_per_node must divide EP (contiguous node-major ranks)")
            if base is not None:
                if not isinstance(base, SymmBuffer) or base._destroyed or base._runtime is None:
                    raise ValueError("Cannot reuse an invalid symmetric buffer")
                base._runtime.ensure_alive()
                if base.group is not group or base._backend_name != self._backend_name:
                    raise ValueError("base buffer must share the same group and backend")
        except (TypeError, ValueError, RuntimeError) as failure:
            error = str(failure)
        # 在进入 SHMEM 之前共同校验布局，不能各 rank 各自选择通信偏移。
        config = (
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            num_shared_experts,
            mma_type,
            activation,
            ranks_per_node,
            topk_weights_type,
            base is not None,
        )
        collective_check(group, "configuration", error, config)
        ext = _extension()
        capacity = int(
            ext.buffer_size(
                size, num_experts // size, num_max_tokens_per_rank, num_topk, hidden, backend=self._backend_name, topk_weights_type=topk_weights_type
            )
        )
        if base is not None and capacity > base.num_bytes:
            raise ValueError(f"Requires {capacity} bytes, but the base buffer has {base.num_bytes}")
        self._lock = threading.Lock()
        self._device = torch.device(f"npu:{npu_api().current_device()}")
        self._plan = self._plan_bs = None
        self._generation = 0
        self._failure = None
        self._destroyed = False
        self._base = base
        self._owns_runtime = base is None
        self._world_size = size
        self.group, self.num_experts = group, num_experts
        self.num_max_tokens_per_rank, self.num_topk = num_max_tokens_per_rank, num_topk
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts, self.mma_type = num_shared_experts, mma_type
        self.activation = activation
        self.ranks_per_node = ranks_per_node
        self.topk_weights_type = topk_weights_type
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        error = None
        try:
            self._allocate_inputs()
        except Exception as failure:
            error = str(failure)
        # 输入分配失败先通知所有 rank，再进入 SHMEM，避免其他 rank 已在建链等待。
        collective_check(group, "input allocation", error)
        if base is None:
            self._runtime = _MegaRuntime(group, capacity, _DEFAULT_TIMEOUT, backend=self._backend_name)
        else:
            self._runtime = base._runtime

    def _allocate_inputs(self):
        import torch

        self.x = torch.empty(
            (self.num_max_tokens_per_rank, self.hidden), dtype=torch.float8_e4m3fn, device=self._device
        )
        self.x_sf = torch.empty(
            (self.num_max_tokens_per_rank, self.hidden // 64), dtype=torch.int16, device=self._device
        )
        self.topk_idx = torch.empty(
            (self.num_max_tokens_per_rank, self.num_topk), dtype=torch.int64, device=self._device
        )
        self.topk_weights = torch.empty_like(self.topk_idx, dtype=torch.float32)

    def _ensure_alive(self):
        if self._destroyed or self._runtime is None:
            raise RuntimeError("Buffer runtime has been destroyed")
        self._runtime.ensure_alive()

    def _shared_workspace_from_base(self):
        if self._base is None or self._backend_name != "mte":
            return None
        base_plan = self._base._plan
        if base_plan is None:
            return None
        workspace = getattr(base_plan, "workspace", None)
        if workspace is None:
            return None
        ext = _extension()
        ws_size = getattr(ext, "workspace_size", None)
        if ws_size is not None:
            required = int(
                ws_size(
                    self.hidden,
                    self.intermediate_hidden,
                    self.num_topk,
                    self.num_experts // self._world_size,
                    self._world_size,
                    self.num_max_tokens_per_rank,
                    self.mma_type == "fp8xfp4",
                    self.num_shared_experts,
                    self.ranks_per_node,
                    backend=self._backend_name,
                    topk_weights_type=self.topk_weights_type,
                )
            )
            if workspace.nbytes < required:
                return None
        return workspace

    def _prepare(self, bs):
        integer("BS", bs, 1, self.num_max_tokens_per_rank)
        if self._plan is not None:
            return
        runtime = self._runtime
        address = require_native("MegaMoE").megamoe_workspace_address(runtime.native)
        ext = _extension()
        plan_kwargs = dict(backend=self._backend_name, topk_weights_type=self.topk_weights_type)
        shared_workspace = self._shared_workspace_from_base()
        if shared_workspace is not None:
            plan_kwargs["shared_workspace"] = shared_workspace
        self._plan = ext.Plan(
            self.x,
            self.intermediate_hidden,
            self.num_topk,
            self.num_experts // self._world_size,
            self._world_size,
            self.num_max_tokens_per_rank,
            self.mma_type == "fp8xfp4",
            self.num_shared_experts,
            self.ranks_per_node,
            ext.ffts_config(),
            address,
            runtime.native.capacity,
            **plan_kwargs,
        )
        # The plan owns capacity for num_max_tokens_per_rank; each forward
        # supplies its active BS without retiring the device workspace.
        self._plan_bs = self.num_max_tokens_per_rank

    @property
    def num_bytes(self) -> int:
        self._ensure_alive()
        return self._runtime.native.capacity

    @property
    def timer_enabled(self) -> bool:
        return bool(_extension().timer_enabled and self.mma_type == "fp8xfp4")

    def allocate_timer(self) -> Tensor:
        """Fresh zeroed device trace; pass only to a separate diagnostic invocation."""
        self._ensure_alive()
        if not self.timer_enabled:
            raise RuntimeError("W4A8 timer requires DEEPEP_MEGAMOE_TIMER=ON")
        return _extension().allocate_timer(self.x, backend=self._backend_name)

    def destroy(self):
        with self._lock:
            if self._owns_runtime and self._runtime is not None:
                self._runtime.close()
            self._destroyed = True
            self._runtime = None
            self._base = None
            self._plan = self._plan_bs = None
            self.x = self.x_sf = self.topk_idx = self.topk_weights = None

    def __del__(self):
        if getattr(self, "_owns_runtime", False):
            warn_unclosed(self)


def _interleave_gate_up_128(tensor):
    """Permute the N axis of raw [E, 2I, ...] storage without numeric conversion."""
    experts, columns, *tail = tensor.shape
    return (
        tensor.reshape((experts, 2, columns // 256, 128, *tail))
        .transpose(1, 2)
        .contiguous()
        .reshape((experts, columns, *tail))
    )


class _RoutedWeightPair(tuple):
    """Tuple-compatible immutable prepared weights; both layouts are built off the hot path."""

    def __new__(cls, fused, planar):
        value = super().__new__(cls, fused)
        value._planar = planar
        value._versions = tuple(t._version for t in (*fused, *planar))
        return value

    def select(self, fused):
        if tuple(t._version for t in (*self, *self._planar)) != self._versions:
            raise ValueError("Prepared GMM1 weights/scales were modified in-place; transform raw ND weights again")
        return tuple(self) if fused else self._planar


def transform_weights_for_mega_moe(
    l1_weights: WeightPair, l2_weights: WeightPair, activation: str = "swiglu"
) -> tuple[WeightPair, WeightPair]:
    """Convert raw Ascend packed weights to NZ once, without quantizing again.

    Raw inputs use [gate | up]. 3D inputs [E, N, K/2] are routed weights (FP4 GMM1
    prepares both planar Prefill and [gate128, up128] Decode layouts; keep the
    returned l1 pair intact and immutable). 2D inputs [N, K] / [N, K/2] are shared
    expert weights (aligned with DeepGEMM-Ascend) and retain their planar layout.
    Scales accept packed int16 [..., N, K/64], E8M0 [..., N, K/64, 2], or packed
    int32 [..., N, K/128].
    """
    import torch
    import torch_npu

    if activation != "swiglu":
        raise NotImplementedError("Only SwiGLU is supported")
    pairs = (("l1_weights", _pair(l1_weights, "l1_weights")), ("l2_weights", _pair(l2_weights, "l2_weights")))
    if pairs[0][1][0].ndim != pairs[1][1][0].ndim:
        raise ValueError("l1_weights and l2_weights must have the same rank (3D routed or 2D shared)")
    transformed = []
    for name, (weight, scale) in pairs:
        if weight.device.type != "npu" or scale.device != weight.device:
            raise ValueError("Move weights and scales to the same NPU before preparing")
        if weight.ndim not in (2, 3):
            raise ValueError(f"{name}: weights must be 3D [E,N,K]/[E,N,K/2] (routed) or 2D [N,K]/[N,K/2] (shared)")
        is_shared = weight.ndim == 2
        if weight.dtype in {torch.int8, torch.uint8} and torch_npu.get_npu_format(weight) in (29, 50):
            raise ValueError(f"{name}: prepare from raw ND weights, not already converted NZ weights")
        if weight.dtype == torch.int8:
            weight = weight.view(torch.uint8)  # 重解释 FP4 的位模式，不能做数值类型转换。
        if weight.dtype not in {torch.uint8, torch.float8_e4m3fn}:
            raise TypeError("Weights must be packed FP4 int8/uint8 or FP8 E4M3")
        k = weight.shape[-1] * (2 if weight.dtype == torch.uint8 else 1)
        prefix = weight.shape[:-1]
        expected = (*prefix, k // 64, 2)
        if k % 64:
            raise ValueError(f"{name}: K must be divisible by 64")
        if scale.dtype == torch.int16 and tuple(scale.shape) == (*prefix, k // 64):
            scale = scale.contiguous().view(torch.uint8).view(torch.float8_e8m0fnu).reshape(expected)
        elif scale.dtype == torch.int32 and tuple(scale.shape) == (*prefix, k // 128) and k % 128 == 0:
            # 仅接受逻辑行优先、尚未做 CUDA UTCCP 重排的 packed E8M0。
            scale = scale.contiguous().view(torch.uint8).view(torch.float8_e8m0fnu).reshape(expected)
        if scale.dtype != torch.float8_e8m0fnu or tuple(scale.shape) != expected:
            raise ValueError(
                f"{name}: expected int16 scales {(*prefix, k // 64)}, E8M0 {expected}, or packed int32 {(*prefix, k // 128)}"
            )
        scale = scale.contiguous()
        if weight.dtype == torch.uint8:
            if name == "l1_weights" and not is_shared:
                if weight.shape[1] % 256:
                    raise ValueError("Routed FP4 GMM1 requires gate/up widths divisible by 128")
                # 两种布局一次准备；独立存储并启用版本计数，禁止原地修改后两份权重失配。
                with torch.inference_mode(False):
                    planar_weight = torch_npu.npu_format_cast(
                        weight, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
                    )
                    planar_scale = scale.view(torch.uint8).clone().view(torch.float8_e8m0fnu)
                    weight = _interleave_gate_up_128(weight)
                    scale = _interleave_gate_up_128(scale.view(torch.uint8)).view(torch.float8_e8m0fnu)
                    weight = torch_npu.npu_format_cast(
                        weight, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
                    )
                    _check_fp4_weight_format(planar_weight, name)
                    _check_fp4_weight_format(weight, name)
                    transformed.append(_RoutedWeightPair((weight, scale), (planar_weight, planar_scale)))
                continue
            # uint8 只是两个 FP4 的存储载体；FP8 布局参数不能代替输入的 FP4 类型声明。
            weight = torch_npu.npu_format_cast(
                weight,
                29,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
            _check_fp4_weight_format(weight, name)
        transformed.append((weight, scale))
    return tuple(transformed)


def mega_moe(
    x,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    out=None,
    timer=None,
    activation_clamp=None,
):
    """Explicit-input convenience API; prefer fp8_fp4_mega_moe for DeepGEMM-style use.

    x: BF16 [B,H], or (FP8 E4M3 [B,H], E8M0 [B,H/32]).
    FP8 is copied into the original communication record, never dequantized/requantized.
    Routed work uses the prefill scheduler; W4 shared strategy switches at B=256.
    No backward or graph guarantee yet.
    """
    return _run(
        x,
        topk_idx,
        topk_weights,
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        out=out,
        timer=timer,
        activation_clamp=activation_clamp,
    )


def _run(
    x,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    out=None,
    timer=None,
    cumulative_stats=None,
    activation_clamp=None,
):
    import torch

    if not isinstance(sym_buffer, SymmBuffer):
        raise TypeError("sym_buffer must be this installation's MegaMoE SymmBuffer")
    data, scales = _pair(x, "x") if isinstance(x, tuple) else (x, None)
    w1, s1 = _pair(l1_weights, "l1_weights")
    w2, s2 = _pair(l2_weights, "l2_weights")
    shared = (None, None, None, None)
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError("Supply both shared weight pairs, or neither")
    if shared_l1_weights is not None:
        sw1, ss1 = _pair(shared_l1_weights, "shared_l1_weights")
        sw2, ss2 = _pair(shared_l2_weights, "shared_l2_weights")
        if sw1.ndim == 2:
            if sym_buffer.num_shared_experts != 1:
                raise ValueError("2D shared weights require num_shared_experts=1")
            sw1 = sw1.reshape((1, *sw1.shape))
            ss1 = ss1.reshape((1, *ss1.shape))
            sw2 = sw2.reshape((1, *sw2.shape))
            ss2 = ss2.reshape((1, *ss2.shape))
        shared = (sw1, sw2, ss1, ss2)
    clamp = normalize_clamp(activation_clamp)
    b = sym_buffer
    urma = b._backend_name == "urma"
    if urma and scales is None:
        raise TypeError("URMA input requires (FP8 E4M3, E8M0 scales)")
    if scales is not None and scales.dtype == torch.int16 and tuple(scales.shape) == (data.shape[0], b.hidden // 64):
        scales = scales.contiguous().view(torch.uint8).view(torch.float8_e8m0fnu).reshape((data.shape[0], b.hidden // 32))
    if data.ndim != 2 or not 0 < data.shape[0] <= b.num_max_tokens_per_rank or data.shape[1] != b.hidden:
        raise ValueError("x must be nonempty [BS,H] within the buffer capacity")
    if out is not None and tuple(out.shape) != tuple(data.shape):
        raise ValueError("out shape must match x within the buffer capacity")
    if data.device != b._device:
        raise ValueError("x must be on the buffer's creating device")
    expected = (b.num_experts // b.group.size(), b.intermediate_hidden * 2)
    if tuple(w1.shape[:2]) != expected or tuple(topk_idx.shape) != (data.shape[0], b.num_topk):
        raise ValueError("Weight/routing shapes do not match the buffer configuration")
    if (shared[0].shape[0] if shared[0] is not None else 0) != b.num_shared_experts:
        raise ValueError("Shared expert count does not match the buffer configuration")
    expected_dtype = torch.uint8 if b.mma_type == "fp8xfp4" else torch.float8_e4m3fn
    if w1.dtype != expected_dtype:
        raise TypeError("Weight dtype differs from buffer mma_type")
    prepared_inputs = ()
    if w1.dtype == torch.uint8:
        fused = urma or data.shape[0] <= _extension().small_batch_max_tokens
        if isinstance(l1_weights, _RoutedWeightPair):
            prepared_inputs = (*l1_weights, *l1_weights._planar)
            w1, s1 = l1_weights.select(fused)
        elif urma or not fused:
            raise ValueError(
                "Prefill requires both prepared GMM1 layouts; call transform_weights_for_mega_moe "
                "on raw ND weights and keep its returned l1 pair intact"
            )
    if any(t is not None and t.requires_grad for t in (data, scales, topk_weights, w1, s1, w2, s2, out, *shared)):
        raise NotImplementedError("MegaMoE Beta is inference-only")
    if out is not None and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (data, scales, topk_idx, topk_weights, w1, w2, s1, s2, timer, *shared, *prepared_inputs)
        if tensor is not None
    ):
        raise ValueError("out must not overlap any input or timer storage")
    if topk_idx.dtype not in (torch.int64, torch.int32):
        raise TypeError("topk_idx must be int64 or int32; convert explicitly before timing")
    if w1.dtype == torch.uint8:
        for name, weight in (
            ("l1_weights", w1),
            ("l2_weights", w2),
            ("shared_l1_weights", shared[0]),
            ("shared_l2_weights", shared[1]),
        ):
            if weight is not None:
                if urma and name.startswith("shared_"):
                    import torch_npu

                    if (
                        weight.dtype != torch.float8_e4m3fn
                        or not weight.is_contiguous()
                        or torch_npu.get_npu_format(weight) != 2
                    ):
                        raise ValueError("URMA shared weights require contiguous ND FP8 E4M3")
                else:
                    _check_fp4_weight_format(weight, name)
    if cumulative_stats is not None:
        stats = cumulative_stats
        if (
            stats.dtype != torch.int32
            or tuple(stats.shape) != (b.num_experts // b.group.size(),)
            or stats.device != b._device
            or not stats.is_contiguous()
        ):
            raise ValueError("Cumulative stats must be contiguous local-expert int32 on the buffer device")
        if any(
            stats.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (data, scales, topk_idx, topk_weights, w1, w2, s1, s2, out, timer, *shared, *prepared_inputs)
            if t is not None
        ):
            raise ValueError("Cumulative stats must not alias inputs, output or timer")
    with b._lock:
        b._ensure_alive()
        runtime = b._runtime
        if b._failure is not None:
            raise RuntimeError(f"MegaMoE buffer failed; restart the distributed job: {b._failure}")
        caller = npu_api().current_stream(runtime.device)
        runtime.stream.wait_stream(caller)
        # 同一 owner 的调用串行进入专属流，禁止不同流覆盖仍在使用的对称空间。
        with npu_api().stream(runtime.stream):
            try:
                b._prepare(data.shape[0])
                b._generation += 1
                result = b._plan.forward(
                    data, scales, topk_idx, topk_weights, w1, w2, s1, s2, timer, *shared, out, b._generation, clamp
                )
            except Exception as error:
                if urma:
                    b._failure = str(error)
                    if isinstance(b.group, StoreGroup):
                        b.group.abort(b._failure)
                raise
            if cumulative_stats is not None:
                cumulative_stats.add_(result[1])
                cumulative_stats.record_stream(runtime.stream)
        caller.wait_stream(runtime.stream)
        for tensor in result:
            tensor.record_stream(caller)
        return result


def fp8_fp4_mega_moe(
    y: Tensor,
    l1_weights: WeightPair,
    l2_weights: WeightPair,
    sym_buffer: SymmBuffer,
    shared_l1_weights: WeightPair | None = None,
    shared_l2_weights: WeightPair | None = None,
    cumulative_local_expert_recv_stats: Tensor | None = None,
    recipe: tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: float | None = None,
    fast_math: bool = True,
    *,
    x: Tensor | None = None,
    timer: Tensor | None = None,
) -> None:
    """Write BF16 y in place, return None; accumulate stats if supplied.

    Positional parameters follow DeepGEMM. Ascend-only keyword extensions:
    x is optional BF16 [BS,H] (quantized internally, NOT BF16 GEMM);
    timer is a fresh buffer.allocate_timer() for a separate diagnostic call.
    Routed weight placement follows sym_buffer.topk_weights_type; see README.
    """
    if recipe != (1, 1, 32) or activation != "swiglu" or fast_math is not True:
        raise NotImplementedError("Beta supports recipe=(1,1,32), SwiGLU, default math only")
    if not isinstance(sym_buffer, SymmBuffer):
        raise TypeError("sym_buffer must be this installation's MegaMoE SymmBuffer")
    import torch

    sym_buffer._ensure_alive()
    if y.ndim != 2 or y.dtype != torch.bfloat16:
        raise ValueError("y must be BF16 [BS,H]")
    bs = y.shape[0]
    if x is not None and (x.dtype != torch.bfloat16 or tuple(x.shape) != (bs, sym_buffer.hidden)):
        raise ValueError("x override must be BF16 [BS,H]; omit x for buffered FP8 input")
    _run(
        (sym_buffer.x[:bs], sym_buffer.x_sf[:bs]) if x is None else x,
        sym_buffer.topk_idx[:bs],
        sym_buffer.topk_weights[:bs],
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        out=y,
        timer=timer,
        cumulative_stats=cumulative_local_expert_recv_stats,
        activation_clamp=activation_clamp,
    )
