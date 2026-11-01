"""Public V2 ElasticBuffer entry point and machine-aware workspace sizing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .._machine import MachineSelector
from .._native import require_native
from ..backends.pod import elastic as _pod
from ..backends.pod.elastic import EPHandle

if TYPE_CHECKING:
    import torch.distributed as dist


class ElasticBuffer(_pod.ElasticBuffer, metaclass=MachineSelector):
    """Select POD, Clos or CARD_4PMESH while preserving the V2 interface."""

    @staticmethod
    def get_buffer_size_hint(
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
        *,
        world_size: Optional[int] = None,
        machine_type: str = "auto",
        rootinfo_path: Optional[str] = None,
    ) -> int:
        """Use the selected backend's layout without constructing a Buffer."""
        require_native("ElasticBuffer.get_buffer_size_hint")
        from .._machine import select

        if select(machine_type, rootinfo_path=rootinfo_path)["machine_type"] in ("clos_only", "card_4pmesh"):
            from .._clos_adapter import ElasticBuffer as ClosElasticBuffer

            if world_size is not None:
                raise ValueError("Clos size hint requires its EP group, not an explicit world_size")
            return ClosElasticBuffer.get_buffer_size_hint(
                group,
                num_max_tokens_per_rank,
                hidden,
                num_topk,
                use_fp8_dispatch,
                allow_hybrid_mode,
                allow_multiple_reduction,
            )
        return _pod.ElasticBuffer.get_buffer_size_hint(
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            use_fp8_dispatch,
            allow_hybrid_mode,
            allow_multiple_reduction,
            world_size=world_size,
        )


def __getattr__(name):
    # Preserve old imports of POD helpers used by the existing test scripts.
    return getattr(_pod, name)


__all__ = ["ElasticBuffer", "EPHandle"]
