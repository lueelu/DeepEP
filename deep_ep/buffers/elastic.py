"""Public V2 ElasticBuffer entry point and machine-aware workspace sizing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .._machine import MachineSelector
from .._native import require_native
from ..backends.mesh_clos import elastic as _mesh_clos
from ..backends.mesh_clos.elastic import EPHandle

if TYPE_CHECKING:
    import torch.distributed as dist


class ElasticBuffer(_mesh_clos.ElasticBuffer, metaclass=MachineSelector):
    """Select Mesh+Clos, Global Clos or Mesh+PCIe while preserving the V2 interface."""

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

        if select(machine_type, rootinfo_path=rootinfo_path)["machine_type"] in ("global_clos", "mesh_pcie"):
            from ..backends.direct.api import ElasticBuffer as DirectElasticBuffer

            if world_size is not None:
                raise ValueError("Direct size hint requires its EP group, not an explicit world_size")
            return DirectElasticBuffer.get_buffer_size_hint(
                group,
                num_max_tokens_per_rank,
                hidden,
                num_topk,
                use_fp8_dispatch,
                allow_hybrid_mode,
                allow_multiple_reduction,
            )
        return _mesh_clos.ElasticBuffer.get_buffer_size_hint(
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
    # Preserve old imports of Mesh+Clos helpers used by the existing test scripts.
    return getattr(_mesh_clos, name)


__all__ = ["ElasticBuffer", "EPHandle"]
