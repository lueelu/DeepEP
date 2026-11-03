"""Metadata returned by the optional expert-layout expansion operator."""

from __future__ import annotations

from typing import Any, List


class ExpandHandle:
    """Reusable mapping between rank-deduplicated and expert-major rows."""

    __slots__ = (
        "expert_offsets",
        "actual_expert_counts",
        "recv_to_expanded_idx",
        "num_tokens_per_expert_list",
        "num_recv_tokens",
        "num_expanded_tokens",
        "num_topk",
        "expert_alignment",
    )

    def __init__(
        self,
        expert_offsets: Any,
        actual_expert_counts: Any,
        recv_to_expanded_idx: Any,
        num_tokens_per_expert_list: List[int],
        num_recv_tokens: int,
        num_expanded_tokens: int,
        num_topk: int,
        expert_alignment: int,
    ) -> None:
        self.expert_offsets = expert_offsets
        self.actual_expert_counts = actual_expert_counts
        self.recv_to_expanded_idx = recv_to_expanded_idx
        self.num_tokens_per_expert_list = list(num_tokens_per_expert_list)
        self.num_recv_tokens = int(num_recv_tokens)
        self.num_expanded_tokens = int(num_expanded_tokens)
        self.num_topk = int(num_topk)
        self.expert_alignment = int(expert_alignment)
