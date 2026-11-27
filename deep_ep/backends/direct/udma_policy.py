"""Full-CLOS host scheduling. Core counts remain operation inputs."""

import logging
import os
from dataclasses import dataclass, replace
from functools import lru_cache

from .config import Config, _MAX_NUM_AIV


def _simt_weight_row_fits(world_size, row_bytes):
    # Prefer SIMT through H4096 BF16 for EP16/32, independent of core budget
    # and topology. EP16 has measured gains; EP32 shares this configured limit.
    # The caller still requires a spare AIV. H6144 keeps its UDMA preference.
    if world_size in (16, 32):
        return row_bytes <= 8192
    # Other small EPs retain their row limit; EP64 measured SIMT wins
    # through 12288 B. Extend those anchors by EP bands, not payload profiles.
    # EP128 reports use SIMT when auxiliary AIVs exist; larger EPs follow that
    # preference without an unmeasured upper cutoff. These are heuristic fits.
    if world_size < 64:
        return row_bytes <= 7168
    if world_size < 128:
        return row_bytes <= 12288
    return True


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
    self_copy: int = 2  # 1: spare AIVs; 2: after submit; legacy 0 aliases 2.
    weight_mode: int = 0  # 0: legacy; 1: UDMA; 2: SIMT when spare AIVs exist.
    extend_large_rows: bool = False  # Opt-in H6144 sender-plan extension; captured at Buffer construction.

    def __post_init__(self):
        # Historical profiles/manual callers may still supply 0. Never report
        # or submit a copy-before-send plan, including the fixed EP128 profile.
        if self.self_copy == 0:
            object.__setattr__(self, "self_copy", 2)

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
            self.extend_large_rows,
            self.reservation_spec(),
        )

    @lru_cache(maxsize=128)
    def reservation_spec(self):
        """Union all automatic ownership mappings for this Buffer's topology.

        Core counts remain dynamic. Baseline plans cover V1, non-BF16 and
        expanded paths; BF16 V2 plans cover every row-size policy interval.
        Visit, copy/weight mode and payload coalescing do not change QP masks.
        """
        mappings = set()
        # Representatives of the selector's <=4096 bucket, 2048..12288
        # profile range and 8192..12288 weighted range, including fallbacks.
        row_sizes = (1, 2048, 4096, 4097, 8192, 12288, 12289)
        for cores in range(2, _MAX_NUM_AIV + 1, 2):
            try:
                baseline = self.for_num_aiv(cores)
            except ValueError:
                continue  # This topology cannot launch at this core budget.
            plans = [baseline]
            for size in row_sizes:
                for operation in ("dispatch", "combine", "weighted"):
                    plans.append(self.for_launch(cores, size, operation, bf16=True))
            for plan in plans:
                mappings.add((plan.peer_order, plan.peers_per_worker, plan.workers_per_peer, plan.qps_per_worker))
        names = ("rank", "group", "group_staggered", "group_round_robin")
        return ";".join(f"{names[order]}:{p}:{w}:{q}" for order, p, w, q in sorted(mappings))

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
        if self.world_size == 128 and 8 <= num_aiv < 16 and self.workers_per_peer == self.qps_per_worker == 1:
            policy = replace(self, peer_order=0, peers_per_worker=16, wqes_per_visit=32)
        while policy.send_workers > budget and policy.peers_per_worker < 8:
            policy = replace(policy, peers_per_worker=policy.peers_per_worker * 2)
        if policy.send_workers > budget:
            # Other constrained budgets retain the existing fallback. Logical groups are a
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
        if (
            self.extend_large_rows
            and self.world_size in (4, 8, 16)
            and v2
            and not expanded
            and bf16
            and 8 <= cores <= 64
            and row_bytes > 12288
        ):
            # Only inherit scheduling. Actual tensor sizes, transfer bounds and
            # payload coalescing remain based on the real row width in the Host/kernel.
            return self.for_launch(num_aiv, 12288, operation, bf16=True)
        inherit_sender_tier = (
            v2
            and not expanded
            and bf16
            and 8 <= cores <= 64
            and (
                (operation == "dispatch" and self.world_size in (4, 8, 16, 32, 64))
                or (operation in ("combine", "weighted") and self.world_size in (8, 16, 64))
            )
        )
        if inherit_sender_tier:
            tier = 8 if cores < 32 else 32
            if cores != tier:
                # Select the complete sender plan at the measured core budget.
                # Extra cores may assist local work, but cannot change ownership,
                # peer order, visit, or the weight transport preference.
                plan = self.for_launch(tier, row_bytes, operation, bf16=bf16, expanded=expanded, v2=v2)
                if operation == "combine":
                    plan = replace(plan, self_copy=1 if cores - plan.send_workers >= plan.send_workers else 2)
                plan.validate_num_aiv(num_aiv)
                return plan
        plan = replace(base, wqes_per_visit=32, self_copy=0, weight_mode=0)
        if self.world_size == 128 and cores >= 16:
            # One fixed EP128 schedule for every row size and operation. Match
            # the measured rank/P8/W1/Q1/visit16 candidate and weight path.
            # Legacy self-copy 0 is normalized to after-submit. Budgets 8..14
            # retain rank/P16/visit32 selected by for_num_aiv().
            plan = replace(
                base,
                peer_order=0,
                peers_per_worker=8,
                workers_per_peer=1,
                qps_per_worker=1,
                wqes_per_visit=16,
                self_copy=0,
                weight_mode=0,
            )
            plan.validate_num_aiv(num_aiv)
            return plan
        if v2 and not expanded and bf16:
            bucket = "small" if row_bytes <= 4096 else "large"
            # Extra auxiliary AIVs must not discard the EP32 dispatch profile.
            # EP32 unweighted combine has no fitted override: every core budget
            # retains rank/P4/W1/Q1/visit32, with spare cores used independently.
            # Select a measured tier without truncating the actual launch budget.
            # Paths not covered by sender-tier inheritance above retain their
            # existing intermediate and large-budget selection rules.
            profile_cores = 8 if 8 <= cores < 16 else 32 if 32 <= cores < 64 else None
            if self.world_size == 32 and operation == "dispatch" and cores >= 32:
                profile_cores = 32
            entry = _PROFILES.get((self.world_size, profile_cores, bucket, operation))
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
            if operation in ("dispatch", "weighted") and cores >= 16 and 0 < cores - plan.send_workers <= 2:
                # Avoid assigning all metadata/weights to just one or two AIVs.
                # First remove duplicate payload workers, then grow peer ownership
                # only as needed. No fixed sender/helper ratio or low-core change.
                candidate = replace(plan, workers_per_peer=1)
                max_peers = 4 if candidate.peer_order == 3 else 8
                while 0 < cores - candidate.send_workers <= 2 and candidate.peers_per_worker < max_peers:
                    candidate = replace(candidate, peers_per_worker=candidate.peers_per_worker * 2)
                if cores - candidate.send_workers > 2 and candidate.valid_group_capacity():
                    plan = candidate
            if operation == "combine":
                # Count payload AIVs, not QPs. Keep at least one spare copy
                # AIV per sender; otherwise overlap self-copy after submission.
                # Use this launch's communication budget, independent of reduce.
                spare_aivs = cores - plan.send_workers
                plan = replace(plan, self_copy=1 if spare_aivs >= plan.send_workers else 2)
            if operation == "weighted":
                # Keep the complete measured plan together: the weight-path
                # comparison used RR/1/1/1/visit8 and after-submit self-copy.
                # Retain its sender budget and visit; rank normalization below
                # converts grouped ownership. Weight preference is selected below.
                # Keep the EP64 plan with larger core budgets too: additional
                # auxiliary AIVs must not silently switch weights back to UDMA.
                # Interpolate between measured row widths; retain both endpoint
                # plans and the existing topology/core-budget requirements.
                measured_weight_fit = (
                    self.topology_known
                    and (self.world_size, self.ranks_per_group) in ((32, 8), (64, 16))
                    and (32 <= cores < 64 or (self.world_size == 64 and cores >= 64))
                    and 8192 <= row_bytes <= 12288
                )
                if measured_weight_fit:
                    plan = replace(
                        plan,
                        peer_order=3,
                        peers_per_worker=1,
                        workers_per_peer=1,
                        qps_per_worker=1,
                        wqes_per_visit=8,
                        self_copy=2,
                    )
                # SIMT needs an AIV outside the payload sending plan.
                use_simt_weights = _simt_weight_row_fits(self.world_size, row_bytes) and plan.send_workers < cores
                plan = replace(plan, weight_mode=2 if use_simt_weights else 1)
            if (self.world_size == 64 and cores >= 8) or (
                self.world_size in (4, 8, 16, 32) and (8 <= cores < 16 or 32 <= cores < 64)
            ):
                # Normalize order after selecting P/W/Q, visit and copy/weight
                # modes. EP64 also normalizes its large-core weighted plan.
                # RR -> rank can free a self-only owner slot; do not implicitly
                # switch UDMA weights to SIMT when that exposes another helper.
                # RR counts peers within a group. Convert that ownership width
                # to global rank peers so EP32/G8 P1 retains eight senders with
                # rank P4, instead of expanding to 31 senders with rank P1.
                rank_peers = plan.peers_per_worker
                if plan.peer_order == 3:
                    rank_peers *= self.world_size // plan.ranks_per_group
                plan = replace(plan, peer_order=0, peers_per_worker=rank_peers)
            if self.world_size == 64 and cores == 32 and operation == "dispatch":
                # Keep eight payload owners and leave 24 AIVs for local work.
                # Channel count and visit remain launch inputs, unchanged here.
                plan = replace(plan, peer_order=0, peers_per_worker=8, workers_per_peer=1, qps_per_worker=1)
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
    if world_size == 128:
        p, visit = 8, 16
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
    extend = os.environ.get("DEEPEP_CLOS_EXTEND_LARGE_ROW_POLICY", "0")
    if extend not in ("0", "1"):
        raise ValueError("DEEPEP_CLOS_EXTEND_LARGE_ROW_POLICY must be 0 or 1")
    return UdmaPolicy(world_size, p, w, q, visit, extend_large_rows=extend == "1")


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
# The large-row EP8/16 rank candidates share parameters across the measured
# H2816/4096/6144 interval; no token-count or exact-hidden exceptions.
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
    (8, 8, "large", "dispatch"): ((8,), 0, 2, 2, 1, 32, 0),
    (8, 8, "large", "combine"): ((8,), 0, 1, 1, 1, 32, 2),
    (8, 8, "large", "weighted"): ((8,), 3, 1, 1, 1, 32, 2),
    (8, 8, "small", "dispatch"): ((8,), 3, 1, 1, 2, 32, 0),
    (8, 8, "small", "combine"): ((8,), 2, 1, 1, 2, 32, 2),
    (8, 8, "small", "weighted"): ((8,), 3, 1, 1, 2, 32, 2),
    (8, 32, "large", "dispatch"): ((8,), 2, 1, 1, 1, 16, 0),
    (8, 32, "large", "combine"): ((8,), 0, 1, 1, 1, 32, 1),
    (8, 32, "large", "weighted"): ((8,), 0, 1, 1, 1, 32, 0),
    (8, 32, "small", "dispatch"): ((8,), 3, 1, 2, 1, 16, 0),
    (8, 32, "small", "combine"): ((8,), 2, 1, 2, 1, 8, 1),
    (8, 32, "small", "weighted"): ((8,), 3, 1, 1, 1, 16, 2),
    (16, 8, "large", "dispatch"): ((8, 16), 2, 2, 1, 1, 16, 0),
    (16, 8, "large", "combine"): ((8, 16), 0, 2, 1, 1, 8, 2),
    (16, 8, "large", "weighted"): ((8, 16), 0, 2, 1, 1, 8, 2),
    (16, 8, "small", "dispatch"): ((8, 16), 1, 2, 1, 2, 32, 0),
    (16, 8, "small", "combine"): ((8, 16), 0, 2, 1, 2, 32, 2),
    (16, 8, "small", "weighted"): ((8, 16), 2, 4, 2, 1, 32, 2),
    (16, 32, "large", "dispatch"): ((8, 16), 2, 4, 2, 1, 16, 0),
    (16, 32, "large", "combine"): ((8, 16), 2, 2, 1, 1, 8, 1),
    (16, 32, "large", "weighted"): ((8, 16), 0, 2, 1, 1, 8, 0),
    (16, 32, "small", "dispatch"): ((8, 16), 0, 2, 2, 1, 8, 0),
    (16, 32, "small", "combine"): ((8, 16), 0, 2, 1, 1, 32, 1),
    (16, 32, "small", "weighted"): ((8, 16), 1, 2, 2, 1, 4, 0),
    (32, 32, "large", "dispatch"): ((8,), 1, 4, 1, 1, 32, 0),
    (32, 32, "large", "weighted"): ((8,), 2, 4, 1, 1, 4, 0),
}
