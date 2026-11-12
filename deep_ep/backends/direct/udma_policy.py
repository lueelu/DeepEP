"""Full-CLOS host scheduling. Core counts remain operation inputs."""

import logging
from dataclasses import dataclass, replace
from functools import lru_cache

from .config import Config, _MAX_NUM_AIV


@lru_cache(maxsize=128)
def _warn_ignored(name, supplied, effective):
    logging.getLogger(__name__).warning(
        "UDMA auto policy: ignoring explicit %s=%s; constructor fallback %s=%s. "
        "The effective launch plan also depends on shape and communication cores. "
        "Set udma_auto_policy=False to enable manual tuning.",
        name,
        supplied,
        name,
        effective,
    )


def _check_override(name, supplied, effective, rank):
    if rank == 0 and supplied is not None and supplied != effective:
        _warn_ignored(name, repr(supplied), repr(effective))


def validate_manual_options(
    schedule,
    dispatch_order,
    combine_order,
    ranks_per_group,
    peers_per_group_per_worker,
    workers,
    qps,
    peers=None,
    wqes=None,
    *,
    operation_config=False,
):
    """Require all knobs used by this schedule; V1 P/WQE live in Config."""
    required = {"udma_schedule": schedule}
    if schedule in (None, "peer"):
        required.update(
            udma_dispatch_peer_order=dispatch_order,
            udma_combine_peer_order=combine_order,
            udma_workers_per_peer=workers,
            udma_qps_per_worker=qps,
        )
        if any(order is not None and order != "rank" for order in (dispatch_order, combine_order)):
            required["udma_ranks_per_group"] = ranks_per_group
        if "group_round_robin" in (dispatch_order, combine_order):
            required["udma_peers_per_group_per_worker"] = peers_per_group_per_worker
        if not operation_config and (dispatch_order, combine_order) != ("group_round_robin", "group_round_robin"):
            required["udma_peers_per_worker"] = peers
    if not operation_config:
        required["udma_wqes_per_visit"] = wqes
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("manual UDMA policy requires explicit parameters: " + ", ".join(missing))


def validate_manual_config(config, schedule, dispatch_order, combine_order):
    required = ["udma_wqes_per_visit"]
    if schedule == "peer" and (dispatch_order, combine_order) != ("group_round_robin", "group_round_robin"):
        required.append("udma_peers_per_worker")
    flags = {"udma_wqes_per_visit": "_udma_wqes_explicit", "udma_peers_per_worker": "_udma_peers_explicit"}
    missing = [name for name in required if config is None or not getattr(config, flags[name], False)]
    if missing:
        raise ValueError("manual UDMA policy requires explicit Config parameters: " + ", ".join(missing))


@dataclass(frozen=True)
class UdmaPolicy:
    world_size: int
    peers_per_worker: int
    workers_per_peer: int
    qps_per_worker: int
    wqes_per_visit: int
    version: str = "clos-v4"
    ranks_per_group: int = 0
    topology_known: bool = False
    peer_order: int = 0
    self_copy: int = 0  # 0: before send; 1: spare AIVs; 2: after submit.
    weight_mode: int = 0  # 0: legacy; 1: UDMA; 2: SIMT when spare AIVs exist.

    @property
    def send_workers(self):
        if self.peer_order == 3:
            return ((self.ranks_per_group - 1) // self.peers_per_worker + 1) * self.workers_per_peer
        return ((self.world_size - 2) // self.peers_per_worker + 1) * self.workers_per_peer

    def signature(self):
        return (
            self.version,
            self.world_size,
            self.peers_per_worker,
            self.workers_per_peer,
            self.qps_per_worker,
            self.wqes_per_visit,
            self.ranks_per_group,
            self.topology_known,
        )

    def validate_num_aiv(self, num_aiv):
        """Check the supplied count without changing it or the selected schedule."""
        if isinstance(num_aiv, bool) or not isinstance(num_aiv, int):
            raise TypeError("num_aiv/num_sms must be an integer")
        if num_aiv < 0 or num_aiv % 2 or num_aiv > _MAX_NUM_AIV:
            raise ValueError(f"num_aiv/num_sms must be zero or an even count <= {_MAX_NUM_AIV}")
        effective = 32 if num_aiv == 0 else num_aiv
        if self.send_workers > effective or self.send_workers * self.qps_per_worker > 32:
            raise ValueError(
                f"UDMA policy requires {self.send_workers} sending AIVs and "
                f"{self.send_workers * self.qps_per_worker} QP slots; got {effective} AIVs "
                "with a 32-QP budget. Increase the operation core count or use manual tuning."
            )

    @lru_cache(maxsize=512, typed=True)
    def for_num_aiv(self, num_aiv):
        """Fit ownership without mutating resources or another handle's plan.

        Grow P first, then use single-worker round robin if rank ownership
        cannot fit. Every candidate must fit the reserved mapping union.
        """
        if isinstance(num_aiv, bool) or not isinstance(num_aiv, int):
            raise TypeError("num_aiv/num_sms must be an integer")
        if num_aiv < 0 or num_aiv % 2 or num_aiv > _MAX_NUM_AIV:
            raise ValueError(f"num_aiv/num_sms must be zero or an even count <= {_MAX_NUM_AIV}")
        budget = min(32 // self.qps_per_worker, 32 if num_aiv == 0 else num_aiv)
        policy = self
        while policy.send_workers > budget and policy.peers_per_worker < 8:
            policy = replace(policy, peers_per_worker=policy.peers_per_worker * 2)
        if policy.send_workers > budget:
            # Rank ownership cannot fit EP128/AIV8. Logical groups are a
            # correctness fallback, never evidence of physical server locality.
            group = self.ranks_per_group or _logical_group(self.world_size)
            for p in (1, 2, 4):
                candidate = replace(
                    self, ranks_per_group=group, peer_order=3, peers_per_worker=p, workers_per_peer=1, qps_per_worker=1
                )
                if candidate.send_workers <= min(32, num_aiv or 32) and candidate.valid_group_capacity():
                    policy = candidate
                    break
        policy.validate_num_aiv(num_aiv)
        return policy

    def valid_group_capacity(self):
        if self.peer_order != 3:
            return True
        owners = self.send_workers // self.workers_per_peer
        return ((self.ranks_per_group + owners - 1) // owners) * (self.world_size // self.ranks_per_group) <= 64

    def native_plan(self):
        """Versioned, per-launch value. No mutation of Buffer or SHMEM state."""
        return [
            1,
            self.peer_order,
            self.ranks_per_group,
            self.peers_per_worker,
            self.workers_per_peer,
            self.qps_per_worker,
            self.wqes_per_visit,
            self.self_copy,
            self.weight_mode,
        ]

    def bind_group(self, group):
        """Collect host identities once at construction, in EP-rank order."""
        from pathlib import Path
        import uuid
        from ..._machine import gather_records

        # Boot identity is shared across containers on the same Linux host.
        # Never guess G from the process count or a user-overridden hostname.
        try:
            host = uuid.UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
            identity = str(host) if host.int else None
        except (OSError, ValueError):
            identity = None
        hosts = gather_records(group, identity)
        if len(hosts) != self.world_size:
            raise ValueError("topology records do not match the EP group")
        size = _contiguous_group_size(hosts)
        return replace(self, ranks_per_group=size or _logical_group(self.world_size), topology_known=bool(size))

    @lru_cache(maxsize=1024, typed=True)
    def for_launch(self, num_aiv, row_bytes, operation, *, bf16=False, expanded=False, v2=True):
        """Select using rank-invariant metadata only; never read tensor values.

        Profiles are in-sample BF16 fits, not performance guarantees. Unmeasured
        dtypes/expanded/V1 paths retain the baseline algorithm. Token/receive
        counts deliberately do not participate, so skew cannot split protocols.
        """
        if operation not in ("dispatch", "combine", "weighted"):
            raise ValueError("unknown UDMA operation")
        if isinstance(row_bytes, bool) or not isinstance(row_bytes, int) or row_bytes <= 0:
            raise ValueError("payload row bytes must be a positive integer")
        base = self.for_num_aiv(num_aiv)
        cores = num_aiv or 32
        plan = replace(base, wqes_per_visit=32, self_copy=0, weight_mode=0)
        if v2 and not expanded and bf16:
            bucket = "small" if row_bytes <= 4096 else "large"
            entry = _PROFILES.get((self.world_size, cores, bucket, operation))
            if (
                entry is not None
                and 2048 <= row_bytes <= 12288
                and self.topology_known
                and self.ranks_per_group in entry[0]
            ):
                _, order, p, w, q, visit, copy = entry
                plan = replace(
                    self,
                    peer_order=order,
                    peers_per_worker=p,
                    workers_per_peer=w,
                    qps_per_worker=q,
                    wqes_per_visit=visit,
                    self_copy=copy,
                )
            elif operation != "dispatch":
                plan = replace(plan, self_copy=2)
            if operation == "combine":
                # Count payload AIVs, not QPs. Keep at least one spare copy
                # AIV per sender; otherwise overlap self-copy after submission.
                # Use this launch's communication budget, independent of reduce.
                spare_aivs = cores - plan.send_workers
                plan = replace(plan, self_copy=1 if spare_aivs >= plan.send_workers else 2)
            if operation == "weighted":
                # Low-core weighted cases prefer payload QPs even with spare
                # cores. A spare core alone is not a reason to enable SIMT.
                plan = replace(plan, weight_mode=2 if cores >= 32 and plan.send_workers < cores else 1)
        plan.validate_num_aiv(num_aiv)
        if not plan.valid_group_capacity():
            raise ValueError("automatic UDMA plan exceeds the per-worker peer table")
        return plan

    def config(self, num_aiv=0):
        policy = self.for_num_aiv(num_aiv)
        return Config(num_aiv, udma_peers_per_worker=policy.peers_per_worker, udma_wqes_per_visit=policy.wqes_per_visit)

    def resolve_config(self, config, default_num_aiv, *, rank=0):
        from ..._config import Config as PublicConfig

        if config is None:
            return self.config(default_num_aiv)
        if isinstance(config, PublicConfig):
            config = config._impl
        if not isinstance(config, Config):
            raise TypeError("config must be a Config")
        policy = self.for_num_aiv(config.num_aiv)
        for name, flag, effective in (
            ("udma_peers_per_worker", "_udma_peers_explicit", policy.peers_per_worker),
            ("udma_wqes_per_visit", "_udma_wqes_explicit", policy.wqes_per_visit),
        ):
            if getattr(config, flag, True):
                _check_override(name, getattr(config, name), effective, rank)
        return Config(
            config.num_aiv,
            config.num_max_nvl_chunked_send_tokens,
            config.num_max_nvl_chunked_recv_tokens,
            config.num_max_rdma_chunked_send_tokens,
            config.num_max_rdma_chunked_recv_tokens,
            udma_peers_per_worker=policy.peers_per_worker,
            udma_wqes_per_visit=policy.wqes_per_visit,
        )


def resolve_udma_policy(
    enabled,
    world_size,
    transport,
    schedule,
    dispatch_order,
    combine_order,
    ranks_per_group,
    peers_per_group_per_worker,
    workers=None,
    qps=None,
    peers=None,
    wqes=None,
    *,
    rank=0,
    operation_config=False,
):
    """Auto is authoritative. Manual mode requires its relevant inputs."""
    if not isinstance(enabled, bool):
        raise TypeError("udma_auto_policy must be a bool")
    # Native transport='auto' currently selects MTE, not UDMA.
    if transport != "udma":
        return None
    if not enabled:
        validate_manual_options(
            schedule,
            dispatch_order,
            combine_order,
            ranks_per_group,
            peers_per_group_per_worker,
            workers,
            qps,
            peers,
            wqes,
            operation_config=operation_config,
        )
        return None
    if isinstance(world_size, bool) or not isinstance(world_size, int) or not 2 <= world_size <= 1024:
        raise ValueError("automatic UDMA policy requires EP between 2 and 1024")
    w, q = (2 if world_size <= 8 else 1), 1
    p, visit = (1, 8) if world_size <= 16 else (4, 32)
    while ((world_size - 2) // p + 1) * w * q > 32 and p < 8:
        p *= 2
    if ((world_size - 2) // p + 1) * w * q > 32:
        raise ValueError("automatic UDMA policy cannot fit the 32-QP budget with P<=8")
    for name, supplied, effective in (
        ("udma_schedule", schedule, "peer"),
        ("udma_dispatch_peer_order", dispatch_order, "rank"),
        ("udma_combine_peer_order", combine_order, "rank"),
        ("udma_ranks_per_group", ranks_per_group, None),
        ("udma_peers_per_group_per_worker", peers_per_group_per_worker, None),
        ("udma_workers_per_peer", workers, w),
        ("udma_qps_per_worker", qps, q),
        ("udma_peers_per_worker", peers, p),
        ("udma_wqes_per_visit", wqes, visit),
    ):
        _check_override(name, supplied, effective, rank)
    return UdmaPolicy(world_size, p, w, q, visit)


def _logical_group(world):
    return next((g for g in (16, 8, 4, 2) if world % g == 0), world)


def _contiguous_group_size(hosts):
    if not hosts or any(not isinstance(host, str) or not host for host in hosts):
        return 0
    blocks = []
    for host in hosts:
        if not blocks or blocks[-1][0] != host:
            blocks.append([host, 1])
        else:
            blocks[-1][1] += 1
    sizes = {count for _, count in blocks}
    if len(sizes) != 1 or len({host for host, _ in blocks}) != len(blocks):
        return 0
    return blocks[0][1]


# In-sample conservative fits: BF16, non-expanded, EP4/8/16/32,
# 2 KiB / 5.5 KiB / 12 KiB rows, random routing. Group sizes are
# eligibility constraints, NOT an assumed physical rank-to-server mapping.
# key: EP, communication AIVs, row-byte bucket, operation.
# value: measured G, order (rank/group/staggered/RR), P, W, Q, visit, self-copy.
_PROFILES = {
    (4, 8, "large", "dispatch"): ((4,), 2, 1, 1, 2, 32, 0),
    (4, 8, "large", "combine"): ((4,), 3, 1, 2, 1, 32, 2),
    (4, 8, "large", "weighted"): ((4,), 3, 1, 2, 1, 32, 2),
    (4, 8, "small", "dispatch"): ((4,), 3, 1, 2, 1, 32, 0),
    (4, 8, "small", "combine"): ((4,), 3, 1, 2, 1, 32, 2),
    (4, 8, "small", "weighted"): ((4,), 1, 1, 1, 2, 32, 2),
    (4, 32, "small", "dispatch"): ((4,), 0, 1, 2, 1, 32, 0),
    (4, 32, "small", "combine"): ((4,), 3, 1, 2, 1, 32, 1),
    (4, 32, "small", "weighted"): ((4,), 3, 1, 2, 1, 32, 2),
    (8, 8, "large", "dispatch"): ((8,), 0, 2, 1, 2, 32, 0),
    (8, 8, "large", "combine"): ((8,), 1, 1, 1, 2, 32, 2),
    (8, 8, "large", "weighted"): ((8,), 3, 1, 1, 1, 32, 2),
    (8, 8, "small", "dispatch"): ((8,), 3, 1, 1, 2, 32, 0),
    (8, 8, "small", "combine"): ((8,), 2, 1, 1, 2, 32, 2),
    (8, 8, "small", "weighted"): ((8,), 3, 1, 1, 2, 32, 2),
    (8, 32, "large", "dispatch"): ((8,), 2, 1, 1, 1, 16, 0),
    (8, 32, "large", "combine"): ((8,), 3, 1, 1, 1, 16, 1),
    (8, 32, "large", "weighted"): ((8,), 2, 2, 2, 1, 8, 0),
    (8, 32, "small", "dispatch"): ((8,), 3, 1, 2, 1, 16, 0),
    (8, 32, "small", "combine"): ((8,), 2, 1, 2, 1, 8, 1),
    (8, 32, "small", "weighted"): ((8,), 3, 1, 1, 1, 16, 2),
    (16, 8, "large", "dispatch"): ((8, 16), 2, 2, 1, 1, 16, 0),
    (16, 8, "large", "combine"): ((8, 16), 0, 2, 1, 1, 16, 2),
    (16, 8, "large", "weighted"): ((8, 16), 2, 2, 1, 1, 32, 2),
    (16, 8, "small", "dispatch"): ((8, 16), 1, 2, 1, 2, 32, 0),
    (16, 8, "small", "combine"): ((8, 16), 0, 2, 1, 2, 32, 2),
    (16, 8, "small", "weighted"): ((8, 16), 2, 4, 2, 1, 32, 2),
    (16, 32, "large", "dispatch"): ((8, 16), 2, 4, 2, 1, 16, 0),
    (16, 32, "large", "combine"): ((8, 16), 2, 2, 1, 1, 8, 1),
    (16, 32, "large", "weighted"): ((8, 16), 2, 2, 1, 1, 4, 0),
    (16, 32, "small", "dispatch"): ((8, 16), 0, 2, 2, 1, 8, 0),
    (16, 32, "small", "combine"): ((8, 16), 0, 2, 1, 1, 32, 1),
    (16, 32, "small", "weighted"): ((8, 16), 1, 2, 2, 1, 4, 0),
    (32, 32, "large", "dispatch"): ((8,), 1, 4, 1, 1, 32, 0),
    (32, 32, "large", "combine"): ((8,), 3, 1, 1, 1, 8, 1),
    (32, 32, "large", "weighted"): ((8,), 2, 4, 1, 1, 4, 0),
}
