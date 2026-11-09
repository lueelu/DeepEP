"""Device-generated route tensor views; no public Host routing backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceRoutes:
    order: object  # protocol4: None; debug_snapshot reconstructs compact order after execution
    reduce_index: object
    expert_counts: object
    expert_offsets: object
