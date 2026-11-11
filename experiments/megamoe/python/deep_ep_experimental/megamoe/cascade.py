# Copyright (c) 2026, Lu Lu
# Modified by SimpleBrightMan 2026

"""NPU operator cascade for accuracy, using the same raw FP4 data as MegaMoE.

GMM outputs and activated inputs round to BF16. Routing, EP reduction and
shared-expert addition use FP32 before one final BF16 conversion, as in the CPU
Golden contract. This module does not import Torch or initialize HCCL on import.
"""

import math
import struct


def _positive_fp32(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite FP32 number")
    try:
        value = struct.unpack("f", struct.pack("f", float(value)))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise ValueError(f"{name} must be a positive finite FP32 number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite FP32 number")
    return value


def normalize_activation(activation="swiglu", activation_params=None):
    """Return (name, beta, optional linear_beta) in the native FP32 representation."""
    if activation not in ("swiglu", "situglu"):
        raise ValueError("activation must be 'swiglu' or 'situglu'")
    if activation_params is None:
        activation_params = {}
    if not isinstance(activation_params, dict):
        raise TypeError("activation_params must be a dictionary or None")
    allowed = {"beta", "linear_beta"} if activation == "situglu" else set()
    unknown = set(activation_params) - allowed
    if unknown:
        raise ValueError(f"Unsupported activation_params for {activation}: {list(unknown)!r}")
    if activation == "swiglu":
        return activation, 1.0, None
    beta = _positive_fp32("beta", activation_params.get("beta", 1.0))
    linear_beta = (
        _positive_fp32("linear_beta", activation_params["linear_beta"]) if "linear_beta" in activation_params else None
    )
    return activation, beta, linear_beta


def _runtime():
    import torch
    import torch_npu
    import torch.distributed as dist

    return torch, torch_npu, dist


def _check_process_group(dist, host_group):
    if not dist.is_initialized():
        raise RuntimeError("Initialize the cascade HCCL process group before constructing CascadeReference")
    if (
        str(dist.get_backend()).lower() != "hccl"
        or dist.get_rank() != host_group.rank()
        or dist.get_world_size() != host_group.size()
    ):
        raise RuntimeError("Cascade requires an HCCL process group with the StoreGroup rank and world size")


def init_process_group(host_group):
    """Reuse a matching HCCL group, or initialize one on the existing host Store.

    Return True only when this call owns the new default process group. The
    caller must destroy an owned group after the cascade's last collective and
    before closing host_group; an existing group remains owned by its caller.
    """
    _, _, dist = _runtime()
    if dist.is_initialized():
        _check_process_group(dist, host_group)
        return False
    if host_group._store is None:
        raise RuntimeError("Cannot initialize cascade HCCL from a closed StoreGroup")
    store = dist.PrefixStore(host_group._prefix + "cascade_hccl", host_group._store)
    dist.init_process_group(
        "hccl", store=store, rank=host_group.rank(), world_size=host_group.size(), timeout=host_group._timeout
    )
    _check_process_group(dist, host_group)
    return True


class CascadeReference:
    """Independent MXFP8/FP4 FFN cascade; prepare planar weights once.

    raw and shared_raw are the two (packed_uint8, E8M0_uint8) weight pairs from
    reference.make_weights, before any gate/up interleaving or NZ conversion.
    All ranks must call forward in the same order with the same batch size.
    """

    def __init__(self, model, raw, shared_raw=None, *, host_group):
        self.torch, self.npu, self.dist = _runtime()
        _check_process_group(self.dist, host_group)
        self.rank, self.world = host_group.rank(), host_group.size()
        self.hidden = int(model["hidden"])
        self.intermediate = int(model["intermediate_hidden"])
        self.experts = int(model["num_experts"])
        self.topk = int(model["num_topk"])
        self.shared_experts = int(model["num_shared_experts"])
        self.shared_intermediate = int(model.get("shared_intermediate_hidden", self.intermediate))
        self.shared_output_rounding = model.get("shared_output_rounding", "single_cast")
        if self.experts <= 0 or self.experts % self.world:
            raise ValueError("Cascade requires global experts divisible by EP world size")
        if self.hidden <= 0 or self.intermediate <= 0 or self.hidden % 64 or self.intermediate % 64:
            raise ValueError("Cascade requires positive hidden and intermediate widths divisible by 64")
        if not 0 < self.topk <= self.experts or self.shared_experts < 0:
            raise ValueError("Invalid cascade top-k or shared-expert count")
        self.local_experts = self.experts // self.world
        self.first_expert = self.rank * self.local_experts
        self.activation, self.beta, self.linear_beta = normalize_activation(
            model.get("activation", "swiglu"), model.get("activation_params")
        )
        self._situ_glu = None
        if self.activation == "situglu":
            # aclnnSituGlu multiplies by FP32 reciprocals internally; reject
            # overflow here instead of constructing a NaN reference at zero.
            for name, value in (("beta", self.beta), ("linear_beta", self.linear_beta)):
                if value is not None and 1.0 / value > self.torch.finfo(self.torch.float32).max:
                    raise ValueError(f"Official aclnnSituGlu requires a finite FP32 reciprocal for {name}")
            try:
                from cann_ops_nn import situ_glu
            except ImportError as error:
                raise RuntimeError(
                    "SiTUGLU cascade requires the CANN cann_ops_nn.situ_glu operator; "
                    "source the matching CANN set_env.sh before running accuracy tests"
                ) from error
            self._situ_glu = situ_glu
        self.fp8 = self._tensor_dtype("float8_e4m3fn")
        self.e8m0 = self._tensor_dtype("float8_e8m0fnu")
        self.e8m0_operator = self._operator_dtype("float8_e8m0fnu")
        self.fp4 = self._operator_dtype("float4_e2m1fn_x2")
        self.weights = self._prepare_weights(raw, self.local_experts, "routed")
        if self.shared_experts:
            self.shared_weights = self._prepare_weights(shared_raw, self.shared_experts, "shared")
        else:
            if shared_raw is not None:
                raise ValueError("shared_raw must be None when num_shared_experts is zero")
            self.shared_weights = None

    def _tensor_dtype(self, name):
        # torch_npu may export integer operator enums (e.g. E8M0 = 293).
        # Tensor.view(int) selects the shape overload, so only actual dtypes
        # are valid for byte reinterpretation and tensor dtype comparisons.
        for namespace in (self.torch, self.npu):
            dtype = getattr(namespace, name, None)
            if isinstance(dtype, self.torch.dtype):
                return dtype
        raise RuntimeError(f"Cascade requires a genuine torch.dtype for {name}; use a matching Torch/NPU runtime")

    def _operator_dtype(self, name):
        dtype = getattr(self.npu, name, getattr(self.torch, name, None))
        if dtype is None:
            raise RuntimeError(f"Cascade requires torch_npu support for {name}")
        return dtype

    def _raw_bytes(self, value, name, *, scale=False):
        torch = self.torch
        value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        allowed = (torch.uint8, torch.int8, self.e8m0) if scale else (torch.uint8, torch.int8)
        if value.dtype not in allowed:
            raise TypeError(f"{name} requires raw packed uint8/int8 codes")
        if value.device.type == "npu" and self.npu.get_npu_format(value) in (29, 50):
            raise ValueError(f"{name} requires raw ND data, before NZ conversion")
        return value.contiguous().view(torch.uint8).to(device="npu")

    def _prepare_weights(self, raw, experts, name):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise TypeError(f"{name} weights require raw GMM1 and GMM2 pairs")
        prepared = []
        intermediate = self.shared_intermediate if name == "shared" else self.intermediate
        for layer, (n, k) in enumerate(((2 * intermediate, self.hidden), (self.hidden, intermediate))):
            pair = raw[layer]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(f"{name} GMM{layer + 1} requires (packed_data, E8M0_scales)")
            data = self._raw_bytes(pair[0], f"{name} GMM{layer + 1} data")
            scale = self._raw_bytes(pair[1], f"{name} GMM{layer + 1} scales", scale=True)
            if tuple(data.shape) != (experts, n, k // 2) or tuple(scale.shape) != (experts, n, k // 64, 2):
                raise ValueError(f"{name} GMM{layer + 1} raw packed weight/scale shapes differ from the model")
            # The raw GMM1 rows are [gate | up], including at decode batch sizes.
            data = self.npu.npu_format_cast(data, 29, customize_dtype=self.fp8, input_dtype=self.fp4)
            prepared.append((data.transpose(1, 2), scale.view(self.e8m0).transpose(1, 2)))
        return tuple(prepared)

    def _normalize_scale(self, scale, rows, k):
        if scale.dtype not in (self.e8m0, self.torch.uint8) or scale.numel() != rows * (k // 32):
            raise ValueError("Cascade quantization requires one E8M0 code per 32 input values")
        # npu_dynamic_mx_quant can return uint8 storage containing E8M0 codes.
        # Reinterpret those bytes rather than numerically converting the scales.
        return scale.contiguous().view(self.e8m0).reshape(rows, k // 64, 2)

    def _quantize(self, x):
        data, scale = self.npu.npu_dynamic_mx_quant(x, axis=-1, dst_type=self.fp8, block_size=32)
        if data.dtype != self.fp8 or tuple(data.shape) != tuple(x.shape):
            raise RuntimeError("npu_dynamic_mx_quant returned an unexpected MXFP8 representation")
        return data, self._normalize_scale(scale, x.shape[0], x.shape[1])

    def _all_gather(self, value, *, codes=False):
        if self.world == 1:
            return value.contiguous()
        # HCCL transports bytes, without interpreting FP8/E8M0 as numeric values.
        source = value.contiguous().view(self.torch.uint8) if codes else value.contiguous()
        gathered = source.new_empty((source.shape[0] * self.world, *source.shape[1:]))
        self.dist.all_gather_into_tensor(gathered, source)
        return gathered.view(value.dtype) if codes else gathered

    def _repeat_codes(self, value, *repeats):
        # NPU Repeat does not support E8M0. Duplicate its byte storage without
        # converting FP8 data or E8M0 scales to a different numeric format.
        return value.view(self.torch.uint8).repeat(*repeats).view(value.dtype)

    def _route(self, value, ids):
        return self.npu.npu_moe_init_routing_v2(
            value,
            ids,
            scale=None,
            active_num=int(ids.numel()),
            expert_num=self.experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[self.first_expert, self.first_expert + self.local_experts],
        )

    def _dispatch(self, data, scale, ids):
        torch = self.torch
        # Routing only copies these BF16 bit patterns; FP8 codes are never cast.
        expanded, row_idx, counts, _ = self._route(data.contiguous().view(torch.bfloat16), ids)
        # Every byte code is an exact BF16 integer. This also avoids casting the
        # E8M0 numeric scale itself on runtimes without an E8M0 conversion kernel.
        scale_codes = scale.contiguous().view(torch.uint8).reshape(scale.shape[0], -1).to(torch.bfloat16)
        expanded_scale, _, _, _ = self._route(scale_codes, ids)
        expanded_scale = expanded_scale.to(torch.uint8).view(self.e8m0).reshape(-1, self.hidden // 64, 2)
        counts = counts.to(torch.int64)
        if tuple(counts.shape) != (self.local_experts,):
            raise RuntimeError("npu_moe_init_routing_v2 did not return counts for the active expert range")
        # Keep the full routing allocation: Finalize requires expandedX and its
        # row map to have matching lengths. Ascend950 GMM permits the group count
        # sum to be smaller than M; only the active prefix participates in GMM.
        return expanded.view(self.fp8), expanded_scale, row_idx, counts

    def _gmm(self, data, scale, counts, weight):
        return self.npu.npu_grouped_matmul(
            x=[data],
            weight=[weight[0]],
            antiquant_scale=[weight[1]],
            group_list=counts,
            group_type=0,
            group_list_type=1,
            per_token_scale=[scale],
            split_item=2,
            output_dtype=self.torch.bfloat16,
            per_token_scale_dtype=self.e8m0_operator,
            weight_dtype=self.fp4,
        )[0]

    def _activation_quantize(self, first):
        torch = self.torch
        if self.activation == "swiglu":
            gate, up = first.to(torch.float32).chunk(2, dim=-1)
            activated = gate * torch.sigmoid(gate) * up
        else:
            # Official aclnnSituGlu takes planar [gate | up], computes in FP32
            # and rounds back to the BF16 GMM1 dtype before MX quantization.
            activated = self._situ_glu(
                first,
                dim=-1,
                beta=self.beta,
                linear_beta=0.0 if self.linear_beta is None else self.linear_beta,
                activate_left=True,
            )
        return self._quantize(activated.to(torch.bfloat16))

    def _ffn(self, data, scale, counts, weights):
        first = self._gmm(data, scale, counts, weights[0])
        activated, activated_scale = self._activation_quantize(first)
        return self._gmm(activated, activated_scale, counts, weights[1])

    def forward(self, x, ids, weights, *, input_kind, quantized=None):
        """Return (BF16 local output, int64 local expert receive counts).

        BF16 mode consumes the actual NPU x passed to MegaMoE. FP8 mode consumes
        quantized=(buffer.x[:bs], buffer.x_sf[:bs]) unchanged and requires x=None.
        ids/weights are the actual buffer tensors, including negative FP32 weights.
        """
        torch = self.torch
        if input_kind == "bf16":
            if x is None or x.dtype != torch.bfloat16 or x.device.type != "npu" or quantized is not None:
                raise ValueError("BF16 cascade input requires NPU BF16 x and quantized=None")
            if x.ndim != 2 or x.shape[1] != self.hidden:
                raise ValueError("Cascade x must have shape [BS, hidden]")
            data, scale = self._quantize(x)
        elif input_kind == "fp8":
            if x is not None or not isinstance(quantized, (tuple, list)) or len(quantized) != 2:
                raise ValueError("FP8 cascade input requires x=None and the actual (FP8, E8M0) buffer views")
            data, scale = quantized
            if data.ndim != 2 or data.shape[1] != self.hidden or data.dtype != self.fp8:
                raise ValueError("Cascade FP8 data must have shape [BS, hidden] and E4M3 dtype")
            if data.device.type != "npu" or scale.device != data.device:
                raise ValueError("Cascade FP8 data and scales must reside on the same NPU")
            scale = self._normalize_scale(scale, data.shape[0], self.hidden)
        else:
            raise ValueError("Cascade input_kind must be bf16 or fp8")
        bs = data.shape[0]
        if bs <= 0 or tuple(ids.shape) != (bs, self.topk) or tuple(weights.shape) != (bs, self.topk):
            raise ValueError("Cascade ids/weights must have shape [positive BS, top-k]")
        if ids.dtype != torch.int32 or weights.dtype != torch.float32:
            raise TypeError("Cascade requires int32 expert IDs and FP32 route weights")
        ids = ids.to(device=data.device).contiguous()
        weights = weights.to(device=data.device).contiguous()
        with torch.no_grad():
            global_data = self._all_gather(data, codes=True)
            global_scale = self._all_gather(scale, codes=True)
            global_ids = self._all_gather(ids)
            global_weights = self._all_gather(weights)
            expanded, expanded_scale, row_idx, counts = self._dispatch(global_data, global_scale, global_ids)
            if int(counts.sum().item()):
                second = self._ffn(expanded, expanded_scale, counts, self.weights)
                routed = self.npu.npu_moe_finalize_routing(
                    second.to(torch.float32),
                    skip1=None,
                    skip2=None,
                    bias=None,
                    scales=global_weights,
                    expanded_src_to_dst_row=row_idx,
                    export_for_source_row=global_ids,
                    drop_pad_mode=2,
                ).reshape(bs * self.world, self.hidden)
            else:
                routed = torch.zeros((bs * self.world, self.hidden), device=data.device, dtype=torch.float32)
            if routed.dtype != torch.float32:
                raise RuntimeError("Cascade finalize routing must preserve FP32 accumulation")
            if self.world > 1:
                output = torch.empty((bs, self.hidden), device=data.device, dtype=torch.float32)
                self.dist.reduce_scatter_tensor(output, routed.contiguous())
            else:
                output = routed
            if self.shared_weights is not None:
                if self.shared_output_rounding == "route_bf16_then_add":
                    output = output.to(torch.bfloat16).to(torch.float32)
                shared_data = self._repeat_codes(data, self.shared_experts, 1)
                shared_scale = self._repeat_codes(scale, self.shared_experts, 1, 1)
                shared_counts = torch.full((self.shared_experts,), bs, device=data.device, dtype=torch.int64)
                shared = self._ffn(shared_data, shared_scale, shared_counts, self.shared_weights)
                shared = shared.to(torch.float32).reshape(self.shared_experts, bs, self.hidden)
                output = output + shared.sum(dim=0)
            return output.to(torch.bfloat16), counts
