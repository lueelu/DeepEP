# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Host mirror of kernels/mega_moe/common/ascend_timer_v2.hpp."""

STARTUP_FIXED_NAMES = [
    "AIC_MOE_STARTUP",
    "AIV_MOE_STARTUP",
    "AIC_DISPATCH_BUFFER_INIT",
    "AIV_DISPATCH_BUFFER_INIT",
    "AIV_INPUT_PREP_BARRIER_WAIT",
    "AIV_COUNT_TABLE_PREPARE",
    "AIV_COUNT_METADATA_WAIT",
    "AIV_GATEWAY_PAYLOAD_CHECK",
    "AIV_COUNT_PREFIX_BUILD",
    "AIV_COUNT_TABLE_PUBLISH",
    "AIC_COUNT_READY_WAIT",
    "AIV_COUNT_READY_WAIT",
]

COMBINE_PROGRESS_FIXED_NAMES = [
    "AIV_COMBINE_INIT",
    "AIV_HOST_COMBINE_BARRIER_WAIT",
]

FIXED_NAMES = [
    "KERNEL_TIMING",
    "AIV_INPUT_BUFFER_INIT",
    "AIV_QUANT",
    "AIV_ROUTE_MASK",
    "AIV_RESET",
    "AIV_INPUT_CROSS_RANK_SYNC",
    "AIV_EXPORT_SYNC",
    "AIV_FINAL_CROSS_RANK_WAIT",
    "AIV_UNPERMUTE",
    "AIC_WAIT_INPUT_PREP",
    *STARTUP_FIXED_NAMES,
    *COMBINE_PROGRESS_FIXED_NAMES,
    "AIV_GATEWAY_PAYLOAD_BROADCAST",
]

# Append the new fixed slot without renumbering existing fixed events.
STARTUP_FIXED_NAMES.append("AIV_GATEWAY_PAYLOAD_BROADCAST")

DYNAMIC_NAMES = [
    "AIV_DISPATCH",
    "AIV_SWIGLU_WAIT",
    "AIV_SWIGLU",
    "AIV_COMBINE_WAIT",
    "AIV_COMBINE",
    "AIV_GMM1_PROLOGUE",
    "AIV_GMM2_PROLOGUE",
    "AIC_GMM1_WAIT",
    "AIC_GMM1",
    "AIC_GMM2_WAIT",
    "AIC_GMM2",
    "AIC_SHARED_GMM1",
    "AIV_SHARED_GMM1_PROLOGUE",
    "AIC_SHARED_GMM2",
    "AIV_SHARED_GMM2_PROLOGUE",
    "AIV_SHARED_SWIGLU",
    "AIV_RDMA_POST",
    "AIV_RDMA_QUIET_WAIT",
    "AIV_RDMA_RECV_WAIT",
    "AIV_GATEWAY_PAYLOAD_WAIT",
    "AIV_COMBINE_PROGRESS",
    "AIV_COMBINE_DRAIN",
    "AIV_RELAY_STEP",
    "AIV_COMBINE_RUN",
    "AIV_COMBINE_REDUCE",
    "AIV_COMBINE_ASYNC",
    "AIV_COMBINE_READY",
    "AIV_COMBINE_SYNC",
    "AIV_PAYLOAD_ALL_LANES_WAIT",
]

RDMA_PHASES = ("dispatch_meta", "dispatch_payload", "combine")
RELAY_STEPS = (
    "packet_wait",
    "topk_load",
    "route_build",
    "route_post",
    "route_complete",
    "count_publish",
    "meta_pack",
    "clos_copy",
    "mesh_copy",
    "source_wait",
    "local_route",
    "table_wait",
    "host_histogram",
    "compact_plan",
    "dispatch_route_read",
    "dispatch_flag_check",
    "dispatch_retry",
    "dispatch_backoff",
    "dispatch_payload_copy",
    "dispatch_drain",
    "dispatch_tile_publish",
)
COMBINE_RUN_STOPS = ("tensor_end", "route_gap", "producer_not_ready", "scan_budget")
COUNT_DIAGNOSTIC_NAMES = [
    "COUNT_PROFILE_VALID",
    "COUNT_SNAPSHOT_ATTEMPTS",
    "COUNT_SNAPSHOT_FAILURES",
    "COUNT_DMA_CYCLES",
    "COUNT_CHECK_CYCLES",
    "COUNT_BACKOFF_CYCLES",
    "COUNT_SUCCESS_DMA_CYCLES",
    "COUNT_SUCCESS_CHECK_CYCLES",
    "COUNT_LAST_STALE_WORDS",
    "COUNT_LAST_STALE_SOURCE_MASK",
    "COUNT_LAST_STALE_SOURCE_PLUS_ONE",
    "COUNT_LAST_STALE_EXPERT_PLUS_ONE",
    "COUNT_LAST_OBSERVED_EPOCH",
    "COUNT_EXPECTED_EPOCH",
    "COUNT_DIAGNOSTIC_CYCLES",
    "COUNT_SNAPSHOT_BYTES",
    "COUNT_SOURCE_MASK_OVERFLOW",
    "COUNT_SNAPSHOT_SOURCES",
    "COUNT_SNAPSHOT_EXPERTS",
    "COUNT_CAPTURED_SNAPSHOT_WORDS",
]

# W4 动态槽保存 tile/slice；其他路径保持原有区间协议。
MAX_DYNAMIC_ITER = 4096
N_CORE_COUNT = 96
CYCLE_TO_US = 1000.0

COUNTERS_PER_CORE = ((len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER) * 2 + 15) // 16 * 16
N_TIMING_COUNTER = COUNTERS_PER_CORE * N_CORE_COUNT
DYNAMIC_ITER_PER_CORE = (len(DYNAMIC_NAMES) + 15) // 16 * 16
TASK_METADATA_OFFSET = N_TIMING_COUNTER + DYNAMIC_ITER_PER_CORE * N_CORE_COUNT
TASK_METADATA_SIZE = N_CORE_COUNT * len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER * 2
NO_WAVE = -1
WAVE_ID_OFFSET = TASK_METADATA_OFFSET + TASK_METADATA_SIZE
WAVE_ID_PER_CORE = (len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER + 15) // 16 * 16
COUNT_DIAGNOSTIC_OFFSET = WAVE_ID_OFFSET + WAVE_ID_PER_CORE * N_CORE_COUNT
COUNT_SNAPSHOT_MAX_WORDS = 4096
COUNT_SNAPSHOT_OFFSET = (COUNT_DIAGNOSTIC_OFFSET + N_CORE_COUNT * len(COUNT_DIAGNOSTIC_NAMES) + 3) // 4 * 4
COUNT_SNAPSHOT_SLOTS_PER_CORE = COUNT_SNAPSHOT_MAX_WORDS // 2
TOTAL_BUFFER_SIZE = COUNT_SNAPSHOT_OFFSET + N_CORE_COUNT * COUNT_SNAPSHOT_SLOTS_PER_CORE


def failed_count_snapshot(raw, core_id, values):
    words = values["COUNT_CAPTURED_SNAPSHOT_WORDS"]
    result = dict(
        failed_snapshot_available=bool(words),
        last_failed_source_mask=None,
        last_failed_sources=None,
        last_failed_source_rank=None,
        last_failed_local_expert_id=None,
        last_failed_observed_epoch=None,
        source_mask_overflow=None,
    )
    if not words:
        return result
    sources, experts = values["COUNT_SNAPSHOT_SOURCES"], values["COUNT_SNAPSHOT_EXPERTS"]
    stride = (experts + 7) // 8 * 8
    if sources < 1 or experts < 1 or words != sources * stride or words > COUNT_SNAPSHOT_MAX_WORDS:
        raise ValueError(f"invalid failed count snapshot geometry: core={core_id}")
    offset = COUNT_SNAPSHOT_OFFSET + core_id * COUNT_SNAPSHOT_SLOTS_PER_CORE
    failed_sources, stale_words = [], 0
    for source in range(sources):
        for expert in range(experts):
            index = source * stride + expert
            word = (int(raw[offset + index // 2]) >> (32 * (index % 2))) & 0xFFFFFFFF
            epoch = word >> 24
            if epoch == values["COUNT_EXPECTED_EPOCH"]:
                continue
            stale_words += 1
            if not failed_sources or failed_sources[-1] != source:
                failed_sources.append(source)
            if result["last_failed_source_rank"] is None:
                result.update(
                    last_failed_source_rank=source, last_failed_local_expert_id=expert, last_failed_observed_epoch=epoch
                )
    if stale_words != values["COUNT_LAST_STALE_WORDS"] or not stale_words:
        raise ValueError(f"invalid failed count snapshot contents: core={core_id}")
    result.update(
        last_failed_sources=failed_sources,
        last_failed_source_mask=sum(1 << source for source in failed_sources if source < 64),
        source_mask_overflow=any(source >= 64 for source in failed_sources),
    )
    return result


def count_diagnostics(raw, core_id):
    offset = COUNT_DIAGNOSTIC_OFFSET + core_id * len(COUNT_DIAGNOSTIC_NAMES)
    values = dict(zip(COUNT_DIAGNOSTIC_NAMES, raw[offset : offset + len(COUNT_DIAGNOSTIC_NAMES)], strict=True))
    if not values["COUNT_PROFILE_VALID"]:
        return {}
    attempts, failures = values["COUNT_SNAPSHOT_ATTEMPTS"], values["COUNT_SNAPSHOT_FAILURES"]
    if attempts < 1 or failures != attempts - 1:
        raise ValueError(f"invalid count snapshot diagnostics: core={core_id}, attempts/failures")
    source_mask = int(values["COUNT_LAST_STALE_SOURCE_MASK"]) & ((1 << 64) - 1)
    result = dict(
        snapshot_attempts=attempts,
        snapshot_retries=failures,
        snapshot_bytes=values["COUNT_SNAPSHOT_BYTES"],
        last_failed_stale_words=values["COUNT_LAST_STALE_WORDS"],
        last_failed_source_mask=source_mask,
        last_failed_sources=[source for source in range(64) if source_mask & (1 << source)],
        last_failed_source_rank=values["COUNT_LAST_STALE_SOURCE_PLUS_ONE"] - 1 if failures else None,
        last_failed_local_expert_id=values["COUNT_LAST_STALE_EXPERT_PLUS_ONE"] - 1 if failures else None,
        last_failed_observed_epoch=values["COUNT_LAST_OBSERVED_EPOCH"] if failures else None,
        expected_epoch=values["COUNT_EXPECTED_EPOCH"],
        source_mask_overflow=bool(values["COUNT_SOURCE_MASK_OVERFLOW"]),
    )
    if values["COUNT_SNAPSHOT_SOURCES"]:
        result["failed_snapshot_available"] = False
        if failures:
            result.update(failed_count_snapshot(raw, core_id, values))
    for field, label in (
        ("COUNT_DMA_CYCLES", "snapshot_dma"),
        ("COUNT_CHECK_CYCLES", "snapshot_check"),
        ("COUNT_BACKOFF_CYCLES", "snapshot_backoff"),
        ("COUNT_SUCCESS_DMA_CYCLES", "successful_snapshot_dma"),
        ("COUNT_SUCCESS_CHECK_CYCLES", "successful_snapshot_check"),
        ("COUNT_DIAGNOSTIC_CYCLES", "diagnostic"),
    ):
        cycles = int(values[field])
        if cycles < 0:
            raise ValueError(f"negative count snapshot cycles: core={core_id}, field={field}")
        result[label + "_cycles"] = cycles
        result[label + "_us"] = cycles / CYCLE_TO_US
    return result


def task_metadata(raw, core_id, logical_index, combine_qps=16):
    """坐标直接来自设备，不根据 shape 或耗时猜测任务身份。"""
    dynamic_index = logical_index - len(FIXED_NAMES)
    if dynamic_index < 0:
        return count_diagnostics(raw, core_id) if FIXED_NAMES[logical_index] == "AIV_COUNT_METADATA_WAIT" else {}
    offset = TASK_METADATA_OFFSET + (core_id * len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER + dynamic_index) * 2
    first, second = raw[offset : offset + 2]
    family = DYNAMIC_NAMES[dynamic_index // MAX_DYNAMIC_ITER]
    mask = (1 << 32) - 1
    if family == "AIV_COMBINE_ASYNC":
        expert, row = ((first >> 32) & mask) - 1, first & mask
        phase, slot = second & mask, ((second >> 32) & mask) - 1
        slots = 4 if combine_qps == 24 else 2
        if expert < 0 or phase not in (1, 2, 3) or not -1 <= slot < slots or (slot == -1 and phase != 1):
            raise ValueError(f"invalid async Combine metadata: core={core_id}, slot={dynamic_index}")
        result = dict(
            granularity="combine_async",
            expert_id=expert,
            row_start=row,
            row_count=1,
            scope={1: "row_lifetime", 2: "issue", 3: "reuse_wait"}[phase],
        )
        if slot >= 0:
            result.update(buffer_slot=slot, observation="software_retirement" if phase == 1 else "phase_interval")
        return result
    if family == "AIV_COMBINE_READY":
        qp, token = (first >> 32) & mask, first & mask
        pending, observed = (second >> 32) & mask, second & mask
        if not (1 <= qp <= combine_qps and token < 256) or observed & ~pending:
            raise ValueError(f"invalid Combine ready snapshot: core={core_id}, slot={dynamic_index}")
        return dict(
            granularity="combine_ready",
            qp_index=qp,
            lane_count=combine_qps,
            token_index=token,
            pending_topk_mask=pending,
            observed_ready_mask=observed,
            token_ready=bool(pending) and pending == observed,
            observation="poll_snapshot",
        )
    if family == "AIV_COMBINE_SYNC":
        qp, phase = (first >> 32) & mask, first & mask
        phases = ("before_export_sync", "after_export_sync", "before_host_barrier", "after_host_barrier")
        if not (1 <= qp <= combine_qps and 1 <= phase <= len(phases) and 0 <= second <= 8192):
            raise ValueError(f"invalid Combine sync marker: core={core_id}, slot={dynamic_index}")
        return dict(
            granularity="combine_sync", qp_index=qp, lane_count=combine_qps, sync_phase=phases[phase - 1], pending_rows=second
        )
    if family == "AIV_COMBINE_REDUCE":
        qp, token = (first >> 32) & mask, first & mask
        topk_mask, hidden = (second >> 32) & mask, second & mask
        if not (1 <= qp <= combine_qps and token < 256 and topk_mask and 1024 <= hidden <= 8192 and hidden % 64 == 0):
            raise ValueError(f"invalid combine reduction metadata: core={core_id}, slot={dynamic_index}")
        return dict(
            granularity="combine_reduce",
            qp_index=qp,
            lane_count=combine_qps,
            token_index=token,
            topk_mask=topk_mask,
            expert_rows=topk_mask.bit_count(),
            hidden=hidden,
        )
    if family == "AIV_RELAY_STEP":
        step, source = ((first >> 32) & mask), (first & mask) - 1
        if step >= len(RELAY_STEPS) or source < 0:
            raise ValueError(f"invalid relay metadata: core={core_id}, slot={dynamic_index}")
        name = RELAY_STEPS[step]
        a, b = (second >> 32) & mask, second & mask
        if step < 6:
            return dict(granularity="relay_step", relay_step=name, source_rank=source, first_expert=a, expert_count=b)
        if name in ("clos_copy", "mesh_copy", "host_histogram", "table_wait"):
            unit = {"clos_copy": "bytes", "mesh_copy": "bytes", "table_wait": "packets", "host_histogram": "items"}[
                name
            ]
            return dict(granularity="relay_step", relay_step=name, source_host=source, **{unit: b})
        if name == "source_wait":
            return dict(granularity="relay_step", relay_step=name, source_rank=source, bytes=b)
        if name in ("dispatch_payload_copy", "dispatch_route_read"):
            return dict(granularity="relay_step", relay_step=name, source_rank=source, row_start=a, row_count=b)
        if name == "dispatch_flag_check" and b & (1 << 31):
            return dict(
                granularity="relay_step",
                relay_step=name,
                consumer_rank=source,
                source_ranks=[rank for rank in range(32) if a & (1 << rank)],
                row_count=b & 0x7FFFFFFF,
            )
        if name.startswith("dispatch_"):
            return dict(granularity="relay_step", relay_step=name, consumer_rank=source, row_start=a, row_count=b)
        return dict(granularity="relay_step", relay_step=name, source_rank=source, items=b)
    if family == "AIV_COMBINE_RUN":
        start, rows = (first >> 32) & mask, first & mask
        reason = second & 0xFF
        if reason >= len(COMBINE_RUN_STOPS) or not 0 < rows <= 1024 or start + rows > 1024 or second & ~0x703:
            raise ValueError(f"invalid combine run metadata: core={core_id}, slot={dynamic_index}")
        return dict(
            granularity="combine_run",
            run_start=start,
            run_rows=rows,
            decision="defer" if second & 0x100 else "post",
            stop_reason=COMBINE_RUN_STOPS[reason],
            next_pending=bool(second & 0x200),
            already_deferred=bool(second & 0x400),
        )
    if family in ("AIV_COMBINE_PROGRESS", "AIV_COMBINE_DRAIN"):
        # Raw storage is signed int64; these two metadata words are uint64.
        first = int(first) & ((1 << 64) - 1)
        result = dict(
            granularity="combine_progress" if family == "AIV_COMBINE_PROGRESS" else "combine_drain",
            pending_before=first & 0xFFFF,
            pending_after=(first >> 16) & 0xFFFF,
        )
        before, after = result["pending_before"], result["pending_after"]
        if not 0 <= after <= before <= 1024:
            raise ValueError(f"invalid combine metadata: core={core_id}, slot={dynamic_index}, pending counts")
        if family == "AIV_COMBINE_PROGRESS":
            scan_cycles = int(second) & ((1 << 64) - 1)
            posts, scanned = (first >> 32) & 0xFFFF, (first >> 48) & 0xFFFF
            sent = before - after
            if not (posts <= sent <= scanned <= 1024 and (posts == 0) == (sent == 0)):
                raise ValueError(f"invalid combine metadata: core={core_id}, slot={dynamic_index}, progress counts")
            result.update(
                posts=posts,
                scanned_rows=scanned,
                scan_control_cycles=scan_cycles,
                scan_control_us=scan_cycles / CYCLE_TO_US,
            )
        elif (first >> 32) != 0 or int(second) != 0:
            raise ValueError(f"invalid combine metadata: core={core_id}, slot={dynamic_index}, unused drain fields")
        # Drain is an enclosing interval, not a second scan measurement. Its
        # unused metadata fields must not become measured zeroes in reports.
        return result
    if family == "AIV_GATEWAY_PAYLOAD_WAIT":
        source = ((first >> 32) & mask) - 1
        gateway = (first & mask) - 1
        if not (0 <= source < 32 and gateway == (source + 16) % 32 and 0 < second <= mask):
            raise ValueError(f"invalid gateway wait metadata: core={core_id}, slot={dynamic_index}")
        return dict(granularity="startup_dependency", source_rank=source, gateway_rank=gateway, sequence=second)
    if family == "AIV_PAYLOAD_ALL_LANES_WAIT":
        missing, polls = first >> 32, first & mask
        if missing > 255 or polls == 0 or not 0 < second <= mask:
            raise ValueError("invalid payload lane wait metadata")
        return dict(
            granularity="startup_dependency",
            initial_missing_mask=missing,
            initial_missing_lanes=[i for i in range(8) if missing & (1 << i)],
            poll_rounds=polls,
            sequence=second,
            lane_count=8,
        )
    if family.startswith("AIV_RDMA_"):
        peer = ((first >> 32) & mask) - 1
        phase = (first & 255) - 1
        row_count = (first & mask) >> 8
        kind = (second >> 32) & 255
        qp = (second >> 40) & 255
        row_start = (second >> 48) & 65535
        lanes = combine_qps if phase == 2 else 8
        if not (0 <= peer < 32 and 0 <= phase < len(RDMA_PHASES) and kind in (0, 1) and qp <= lanes):
            raise ValueError(f"invalid RDMA metadata: core={core_id}, slot={dynamic_index}")
        result = dict(
            granularity="rdma",
            peer_rank=peer,
            phase=RDMA_PHASES[phase],
            kind="commit" if kind else "data",
            bytes=second & mask,
        )
        if qp:
            result["qp"] = qp
            if phase in (1, 2):
                result["lane_count"] = lanes
                result["lane"] = qp - 1
        if row_count:
            result.update(row_start=row_start, row_count=row_count, row_end=row_start + row_count)
        return result
    if first == second == 0:
        return {}
    expert = ((first >> 32) & mask) - 1
    row_start = first & mask
    n_tile = ((second >> 32) & mask) - 1
    row_count = second & mask
    if expert < 0 or row_count == 0:
        raise ValueError(f"invalid task metadata: core={core_id}, slot={dynamic_index}")
    result = dict(
        expert_id=expert,
        row_start=row_start,
        row_count=row_count,
        m_tile=row_start // 256,
        granularity="tile" if n_tile >= 0 else "slice",
    )
    if "_SHARED_" in family:
        result["expert_kind"] = "shared"
    if n_tile >= 0:
        result.update(n_tile=n_tile, tile_id=dynamic_index % MAX_DYNAMIC_ITER)
    return result
