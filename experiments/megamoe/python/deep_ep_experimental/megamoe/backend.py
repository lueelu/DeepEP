"""Internal MegaMoE product selection; public signatures remain in api.py."""

import threading

from deep_ep._store import StoreGroup
from deep_ep._runtime import integer
from deep_ep._native import npu_api
from .validation import collective_check


def cv_rows(world, bs):
    return 128 if world == 16 and bs in (1024, 2048) else 256


def select_backend(group):
    # Product identity is distinct from fabric topology. Unknown products keep
    # the POD entry; its existing shape/platform/peer checks still apply.
    try:
        from deep_ep._machine import board_id, physical_device

        product = board_id(physical_device())
    except Exception:
        product = None
    selected = "ubx" if product == 0x44 else "pod"
    collective_check(group, "MegaMoE product selection", None, selected)
    return selected


class _ProcessGroupStore(StoreGroup):
    """Adapt existing ProcessGroup control collectives, never tensor contents."""

    def __init__(self, group):
        self.group = group
        self._failure = None

    def rank(self):
        import torch.distributed as dist

        return dist.get_rank(self.group)

    def size(self):
        import torch.distributed as dist

        return dist.get_world_size(self.group)

    def all_gather(self, value, stage):
        import torch.distributed as dist

        if self._failure:
            raise RuntimeError(self._failure)
        records = [None] * self.size()
        dist.all_gather_object(records, (stage, value), group=self.group)
        if any(record[0] != stage for record in records):
            raise RuntimeError("MegaMoE control collective order mismatch")
        return [record[1] for record in records]

    def barrier(self, stage):
        self.all_gather(None, stage)

    def abort(self, error):
        # No hidden device collective after a rank failure; launcher/PG timeout
        # terminates peers, as for other ProcessGroup failures.
        self._failure = str(error)


class UbxBuffer:
    def __init__(
        self,
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts,
        mma_type,
        activation,
        ranks_per_node,
        timeout,
    ):
        import torch
        import torch.distributed as dist
        from .urma import UrmaBuffer

        if not isinstance(group, (StoreGroup, dist.ProcessGroup)):
            raise TypeError("Pass a StoreGroup or an initialized ProcessGroup")
        control = group if isinstance(group, StoreGroup) else _ProcessGroupStore(group)
        error = None
        try:
            if mma_type != "fp8xfp4" or activation != "swiglu":
                raise ValueError("UBX supports routed FP8xFP4 with SwiGLU")
            integer("ranks_per_node", ranks_per_node, 0, control.size())
            if ranks_per_node and control.size() % ranks_per_node:
                raise ValueError("ranks_per_node must divide EP")
        except (TypeError, ValueError) as exc:
            error = str(exc)
        collective_check(control, "UBX model", error, (mma_type, activation, ranks_per_node))
        self.control = control
        self._engine = UrmaBuffer(
            control,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_experts=num_experts,
            num_topk=num_topk,
            num_shared_experts=num_shared_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            cv_rows=128,
            timeout=timeout,
        )
        self._runtime = self._engine._runtime
        self._device = self._engine._device
        self._lock = threading.RLock()
        self._key = None
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        error = None
        try:
            self.x = torch.empty((num_max_tokens_per_rank, hidden), device=self._device, dtype=torch.float8_e4m3fn)
            self.x_sf = torch.empty(
                (num_max_tokens_per_rank, hidden // 32), device=self._device, dtype=torch.float8_e8m0fnu
            )
            self.topk_idx = torch.empty((num_max_tokens_per_rank, num_topk), device=self._device, dtype=torch.int32)
            self.topk_weights = torch.empty(
                (num_max_tokens_per_rank, num_topk), device=self._device, dtype=torch.float32
            )
        except Exception as exc:
            error = str(exc)
        collective_check(control, "UBX input allocation", error)

    def prepare(self, bs):
        integer("BS", bs, 1, self._engine.num_max_tokens_per_rank)
        self._runtime.ensure_alive()
        # Shared shape arrives with weights, so native shared scratch is bound
        # on first forward. No route values are read by prepare.

    @property
    def workspace_bytes(self):
        return self._engine._plan.workspace_bytes if self._engine._plan else 0

    @property
    def timer_enabled(self):
        return self._engine.timer_enabled

    def allocate_timer(self):
        return self._engine.allocate_trace(capacity=65536)

    def run(
        self,
        x,
        topk_idx,
        topk_weights,
        l1_weights,
        l2_weights,
        *,
        shared_l1_weights=None,
        shared_l2_weights=None,
        out=None,
        timer=None,
        cumulative_stats=None,
    ):
        import torch

        if not isinstance(x, tuple) or len(x) != 2:
            raise TypeError("UBX input requires (FP8 E4M3, E8M0 scales)")
        bs = x[0].shape[0]
        self.prepare(bs)
        engine = self._engine
        if topk_idx.dtype != torch.int32:
            raise TypeError("topk_idx must be int32, matching the public MegaMoE API")
        shared_i = engine.intermediate_hidden
        if engine.num_shared_experts:
            if shared_l1_weights is None or shared_l2_weights is None:
                raise ValueError("Shared experts require both shared weight pairs")
            sw1, ss1 = shared_l1_weights
            sw2, ss2 = shared_l2_weights
            if sw1.ndim != 3 or sw1.shape[0] != 1 or sw1.shape[1] % 2:
                raise ValueError("Shared GMM1 must be [1,2*shared_I,H]")
            shared_i = sw1.shape[1] // 2
            integer("shared weight intermediate", shared_i, 256, 16384)
            if shared_i % 256 or tuple(sw2.shape) != (1, engine.hidden, shared_i):
                raise ValueError("Shared GMM2 shape or intermediate alignment mismatch")
        operands = [*x, topk_idx, topk_weights, *l1_weights, *l2_weights]
        if hasattr(l1_weights, "_planar"):
            operands.extend(l1_weights._planar)
        if (shared_l1_weights is None) != (shared_l2_weights is None):
            raise ValueError("Supply both shared weight pairs, or neither")
        if shared_l1_weights is not None:
            operands.extend((*shared_l1_weights, *shared_l2_weights))
        if timer is not None:
            operands.append(timer)
        if out is not None and any(
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in operands if t is not None
        ):
            raise ValueError("out must not overlap inputs")
        if cumulative_stats is not None:
            if (
                cumulative_stats.dtype != torch.int32
                or cumulative_stats.device != self._device
                or tuple(cumulative_stats.shape) != (engine.num_experts // self.control.size(),)
                or not cumulative_stats.is_contiguous()
            ):
                raise ValueError("Cumulative stats must be contiguous NPU int32 [localExperts]")
            if any(
                cumulative_stats.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                for t in (*operands, out, timer)
                if t is not None
            ):
                raise ValueError("Cumulative stats must not alias inputs or output")
        with self._lock:
            key = (cv_rows(self.control.size(), bs), shared_i)
            if key != self._key:
                # CV change is a collective resource transition, not a hidden
                # readback of route counts. Old source buffers must be retired.
                self._runtime.stream.synchronize()
                collective_check(self.control, "UBX resource transition", None, key)
                engine._plan = None
                engine.cv_rows, engine.shared_intermediate_hidden = key
                self._key = key
            # Public timer remains the fine trace tensor. The matching aggregate
            # diagnostic is internal and is never allocated for formal calls.
            diagnostic_timer = engine.allocate_timer() if timer is not None else None
            self._diagnostic_timer = diagnostic_timer
            output = engine.forward(
                x,
                topk_idx,
                topk_weights,
                l1_weights,
                l2_weights,
                shared_l1_weights=shared_l1_weights,
                shared_l2_weights=shared_l2_weights,
                out=out,
                timer=diagnostic_timer,
                trace=timer,
            )
            counts = engine._plan.expert_counts
            caller = npu_api().current_stream(self._runtime.device)
            counts.record_stream(caller)
            if cumulative_stats is not None:
                cumulative_stats.add_(counts)
                cumulative_stats.record_stream(caller)
            return output, counts

    def destroy(self):
        self._engine.destroy()
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
