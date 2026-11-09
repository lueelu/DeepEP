"""CPU-only V1/V2 bandwidth accounting; decimal GB, never wire bytes."""

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

L2_CACHE_FLUSH_BYTES = 256_000_000
L2_FLUSH_RANGE = "deepep_benchmark_l2_flush"
PROFILE_SAMPLE_RANGE = "deepep_benchmark_sample/"
PROFILE_GUARD_ITERS = 2


class ProfileSamples:
    """Device event boundaries, linked to explicit sample IDs by profiler flows.

    Keep events alive until export finishes. Host annotation timestamps must
    never be used as device execution boundaries (their clocks can differ).
    """

    def __init__(self, torch):
        self.torch = torch
        self.events = []

    def __call__(self, sample, invoke):
        def mark(edge):
            event = self.torch.npu.Event()
            self.events.append(event)
            with self.torch.autograd.profiler.record_function(f"{PROFILE_SAMPLE_RANGE}{sample}/{edge}"):
                event.record()

        mark("start")
        invoke()  # includes the public API's completion wait on the caller stream
        mark("end")


class L2CacheFlush:
    def __init__(self, torch, buffer, enabled):
        self.torch = torch
        self.buffer = buffer
        self.enabled = enabled
        if enabled:
            self.stream = torch.npu.Stream()
            with torch.npu.stream(self.stream):
                self.cache = torch.empty(L2_CACHE_FLUSH_BYTES // 4, dtype=torch.int32, device="npu")

    def __call__(self):
        if not self.enabled:
            return
        # Finish the preceding invocation before eviction. The host wait also
        # separates calls; the parser identifies zero_ using profiler flows,
        # not by treating this Host annotation as a device execution interval.
        self.stream.wait_stream(self.torch.npu.current_stream())
        self.stream.wait_stream(self.buffer.get_comm_stream())
        self.stream.synchronize()
        with self.torch.autograd.profiler.record_function(L2_FLUSH_RANGE):
            with self.torch.npu.stream(self.stream):
                self.cache.zero_()
            self.stream.synchronize()

    def wrap(self, fn):
        def invoke():
            self()
            return fn()

        return invoke


STAGES = {
    "dispatch_layout_kernel": "layout",
    "notify_dispatch_kernel": "notify",
    "cached_notify_dispatch_kernel": "notify",
    "cached_notify_combine_kernel": "notify",
    "v1_dispatch_udma_kernel": "comm",
    "v1_combine_udma_comm_kernel": "comm",
    "v1_combine_udma_early_kernel": "early_submit",
    "v1_combine_udma_remaining_kernel": "comm",
    "v1_combine_udma_reduce_kernel": "reduce",
    "v1_combine_udma_kernel": "fused_combine",
    "expand_kernel": "expand",
    "aggregate_kernel": "aggregate",
}
LIMITS = [
    "New captures use two guard iterations on each side and fixed sample IDs linked to caller-stream "
    "device events. Guards and event markers are excluded from kernel timing. Missing measured stages "
    "remain errors. Legacy captures retain strict whole-window validation.",
    "official_logical_bytes follows the DeepEP-Ascend test numerator: all rank-deduplicated receive "
    "rows including self; dispatch/cached use x+scales+full top-k indices/weights, combine uses BF16 x only. "
    "This is logical accounting, not actual wire bytes. Matching the numerator does not match timing boundaries.",
    "Remote application bytes exclude self, count rank-deduplicated payload and actual field writes. "
    "They are not NIC/wire bytes: packet headers, ACK/retries and SHMEM collective control traffic are unmeasured.",
    "comm_us is the complete communication kernel, including local work, synchronization, "
    "metadata and (dispatch) GM output copy; it is not an isolated NIC transfer timer.",
    "device_sum_us sums all API device kernels. device_span_us includes gaps between those kernels; "
    "caller_us is measured separately without profiler and includes host submission/event waits.",
    "combine_stage_us = aggregate_us + comm_us; excludes notify and final reduce. "
    "It is reported separately from comm_us and full API time.",
    "For early single-source combine, comm_us is early_submit_us + late_comm_us. "
    "The NIC can progress during aggregate; comm_us alone is not the full transfer interval. "
    "Use combine_stage_us for aggregate plus communication, or combine_stage_span_us including kernel gaps.",
    "Algorithm bytes = T*K*H*element_bytes, including invalid top-k slots. "
    "valid_algorithm_bytes excludes invalid slots. Combine payload is always BF16.",
    "L2 eviction mode is recorded in flush_l2_cache/l2_flush_bytes. When enabled, eviction precedes "
    "each full API call and is excluded from all API timing fields; internal stages are not flushed. "
    "This is cache eviction by memory writes, not a hardware invalidation guarantee. "
    "Repeated fixed routing/input, caller allocation, independent GM outputs. "
    "A bandwidth comparison also requires matching routing, caching, hardware and library versions.",
]
V1_LIMITS = [s for s in LIMITS if not s.startswith(("combine_stage_us", "For early single-source"))] + [
    "V1 uses public Buffer APIs: fresh dispatch includes get_dispatch_layout; expanded "
    "dispatch adds Buffer.expand; expanded combine adds unweighted Buffer.aggregate. "
    "Top-k weights are returned metadata (expanded_values), not payload multipliers.",
    "Public V1 combine fuses transfer and final rank reduction. fused_combine_us measures "
    "that entire kernel; comm_us/reduce_us/combine_stage_us and their bandwidths are null, "
    "not zero. logical_fused_combine_GBps uses the official logical numerator but INCLUDES "
    "reduce, so it is not the official communication bandwidth. combine_with_reduce_us "
    "adds aggregate to that fused kernel; notify is still separate.",
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def file_identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return dict(path=str(path.resolve()), size=path.stat().st_size, sha256=h.hexdigest())


def route_counts(routes_by_rank, experts_per_rank):
    """Independent directed matrices: [source][destination], before device use."""
    world = len(routes_by_rank)
    rows, slots, valid = [], [], []
    for routes in routes_by_rank:
        counts, selections, valid_slots = [0] * world, [0] * world, 0
        for token in routes:
            peers = set()
            for expert in token:
                if expert == -1:
                    continue
                if not 0 <= expert < world * experts_per_rank:
                    raise ValueError(f"invalid expert {expert}")
                peer = expert // experts_per_rank
                selections[peer] += 1
                valid_slots += 1
                peers.add(peer)
            for peer in peers:
                counts[peer] += 1
        rows.append(counts)
        slots.append(selections)
        valid.append(valid_slots)
    return dict(rows=rows, slots=slots, valid_slots=valid)


def byte_ledger(counts, rank, *, tokens, hidden, topk, dtype, idx_bytes, op):
    """Matches peer sender WQE lengths and sparse SIMT stores, not padded HBM strides."""
    if dtype not in ("bf16", "fp8") or idx_bytes not in (4, 8):
        raise ValueError("unsupported byte-accounting contract")
    if op not in ("dispatch", "cached-dispatch", "combine_unweighted", "combine_weighted"):
        raise ValueError(op)
    world = len(counts["rows"])
    combine = op.startswith("combine")
    tx = [counts["rows"][p][rank] for p in range(world)] if combine else counts["rows"][rank]
    rx = counts["rows"][rank] if combine else [counts["rows"][p][rank] for p in range(world)]
    remote_rows = sum(tx) - tx[rank]
    element_bytes = 2 if combine or dtype == "bf16" else 1
    payload = remote_rows * hidden * element_bytes
    scales = remote_rows * ((hidden + 127) // 128) * 4 if not combine and dtype == "fp8" else 0
    fresh = op == "dispatch"
    remote_slots = sum(counts["slots"][rank]) - counts["slots"][rank][rank]
    # Fresh metadata: one src_idx per row, index/weight only for each valid route.
    src = remote_rows * 4 if fresh else 0
    idx = remote_slots * idx_bytes if fresh else 0
    weights = remote_rows * topk * 4 if op == "combine_weighted" else remote_slots * 4 if fresh else 0
    # Peer scheduling publishes one uint64 SO marker per non-self peer, even if empty.
    markers = (world - 1) * 8
    # DeepEP-Ascend tests/ep/test_ep.py uses handle num_recv_tokens INCLUDING self.
    # Cached dispatch deliberately keeps the same logical numerator as fresh,
    # although this benchmark's cached path only transmits x/scales.
    logical_rows = sum(counts["rows"][p][rank] for p in range(world))
    logical_row_size = (
        hidden * 2
        if combine
        else hidden * element_bytes + (((hidden + 127) // 128) * 4 if dtype == "fp8" else 0) + topk * (idx_bytes + 4)
    )
    return dict(
        remote_tx_rows=remote_rows,
        remote_rx_rows=sum(rx) - rx[rank],
        self_rows=tx[rank],
        remote_x_bytes=payload,
        remote_scale_bytes=scales,
        remote_src_idx_bytes=src,
        remote_topk_idx_bytes=idx,
        remote_weight_bytes=weights,
        remote_so_bytes=markers,
        remote_payload_bytes=payload + scales,
        remote_application_bytes=payload + scales + src + idx + weights + markers,
        official_logical_rows=logical_rows,
        official_logical_bytes_per_row=logical_row_size,
        official_logical_bytes=logical_rows * logical_row_size,
        algorithm_bytes=tokens * topk * hidden * element_bytes,
        valid_algorithm_bytes=counts["valid_slots"][rank] * hidden * element_bytes,
        wire_bytes=None,
    )


def select_profile_samples(rows, trace, iterations, device, *, allow_shmem_barrier=False):
    """Select fixed IDs 0..N-1 using caller-stream device events, not row count.

    Guards may be clipped by capture boundaries. A missing marker/kernel in a
    measured sample is still an error; never independently shift rank samples.
    """
    events = json.loads(Path(trace).read_text(encoding="utf-8"), parse_float=Decimal)
    if isinstance(events, dict):
        events = events.get("traceEvents", [])
    annotations, starts, ends, tasks = {}, [], {}, {}
    for event in events:
        name, phase = event.get("name", ""), event.get("ph")
        if phase == "X" and name.startswith(PROFILE_SAMPLE_RANGE):
            try:
                sample, edge = name[len(PROFILE_SAMPLE_RANGE) :].split("/")
                sample = int(sample)
            except ValueError as exc:
                raise ValueError("invalid profile sample marker") from exc
            if edge not in ("start", "end"):
                raise ValueError("invalid profile sample edge")
            if not -PROFILE_GUARD_ITERS <= sample < iterations + PROFILE_GUARD_ITERS:
                raise ValueError("profile sample ID out of range")
            if 0 <= sample < iterations:
                key = (sample, edge)
                if key in annotations:
                    raise ValueError("duplicate profile sample marker")
                annotations[key] = event
        if phase == "X":
            tasks.setdefault((event.get("pid"), event.get("tid"), Decimal(str(event["ts"]))), []).append(event)
        if name == "torch_to_npu":
            key = (event.get("cat"), str(event.get("id")))
            if phase == "s":
                starts.append((key, event))
            elif phase == "f":
                ends.setdefault(key, []).append(event)
    expected = {(i, edge) for i in range(iterations) for edge in ("start", "end")}
    if set(annotations) != expected:
        raise ValueError("missing profile sample markers; refusing to shift samples")
    points = {}
    for key, annotation in annotations.items():
        begin = Decimal(str(annotation["ts"]))
        duration = Decimal(str(annotation["dur"]))
        if not begin.is_finite() or not duration.is_finite() or duration <= 0:
            raise ValueError("invalid sample annotation time")
        links = [
            flow
            for flow, source in starts
            if source.get("pid") == annotation.get("pid")
            and source.get("tid") == annotation.get("tid")
            and begin <= Decimal(str(source["ts"])) < begin + duration
        ]
        if len(links) != 1 or len(ends.get(links[0], [])) != 1:
            raise ValueError(f"missing/ambiguous profile sample event flow: {key}")
        target = ends[links[0]][0]
        timestamp = Decimal(str(target["ts"]))
        linked = tasks.get((target.get("pid"), target.get("tid"), timestamp), [])
        records = [e for e in linked if e.get("name") == "EVENT RECORD"]
        if len(records) != 1 or not timestamp.is_finite():
            raise ValueError(f"profile sample marker is not a device EVENT RECORD: {key}")
        points[key] = (timestamp, target.get("pid"), target.get("tid"))
    intervals, host_ranges = [], []
    for i in range(iterations):
        a, b = points[i, "start"], points[i, "end"]
        if a[1:] != b[1:] or (i and a[1:] != points[0, "start"][1:]) or a[0] >= b[0]:
            raise ValueError("invalid profile sample device interval/stream")
        if intervals and intervals[-1][1] > a[0]:
            raise ValueError("overlapping profile samples")
        intervals.append((a[0], b[0]))
        first, last = annotations[i, "start"], annotations[i, "end"]
        if (first.get("pid"), first.get("tid")) != (last.get("pid"), last.get("tid")):
            raise ValueError("profile sample Host thread changed")
        host_ranges.append(
            (
                Decimal(str(first["ts"])),
                Decimal(str(last["ts"])) + Decimal(str(last["dur"])),
                first.get("pid"),
                first.get("tid"),
            )
        )
    selected = []
    for row in rows:
        begin, duration = Decimal(row["Start Time(us)"].strip()), Decimal(row["Duration(us)"])
        if not begin.is_finite() or not duration.is_finite() or duration <= 0:
            raise ValueError("invalid kernel timestamp/duration")
        overlap = [i for i, (a, b) in enumerate(intervals) if begin < b and begin + duration > a]
        if not overlap:
            if begin < intervals[0][0] or begin >= intervals[-1][1]:
                continue  # explicitly outside the fixed measured interval
            # CANN SHMEM's blocking Host barrier launches this device kernel.
            # Only benchmark alignment in a gap is excluded; an in-sample
            # barrier is retained, and unknown/out-of-device tasks still fail.
            if (
                allow_shmem_barrier
                and row["Name"] == "_Z24barrier_on_stream_kernelim"
                and str(row["Device_id"]).strip() == str(device)
                and str(row["Step Id"]).strip() == "1"
            ):
                continue
            raise ValueError("unassigned kernel between measured samples")
        if len(overlap) != 1:
            raise ValueError("kernel spans profile samples")
        a, b = intervals[overlap[0]]
        if begin < a or begin + duration > b or str(row["Device_id"]).strip() != str(device):
            raise ValueError("kernel crosses sample boundary or has wrong device")
        selected.append(row)
    return selected, intervals, host_ranges, len(rows) - len(selected)


def split_cache_flush(rows, trace, iterations, device, *, sample_host_ranges=None):
    """Follow torch_to_npu flows from annotated Host calls to device kernels.

    Host annotations are not device execution intervals. Match flow endpoints
    to trace tasks, then use device-side name/start time to identify CSV rows.
    An API's own zero_ is retained even when it has the same kernel name.
    """
    events = json.loads(Path(trace).read_text(encoding="utf-8"), parse_float=Decimal)
    if isinstance(events, dict):
        events = events.get("traceEvents", [])
    annotations = []
    tasks, flow_starts, flow_ends = {}, [], {}
    for event in events:
        if event.get("name") == L2_FLUSH_RANGE and event.get("ph") == "X":
            start, duration = Decimal(str(event["ts"])), Decimal(str(event["dur"]))
            if not start.is_finite() or not duration.is_finite() or duration <= 0:
                raise ValueError("invalid L2 flush annotation timestamp")
            entry = (start, start + duration, event.get("pid"), event.get("tid"))
            if sample_host_ranges is None or any(
                pid == entry[2] and tid == entry[3] and a <= start and start + duration <= b
                for a, b, pid, tid in sample_host_ranges
            ):
                annotations.append(entry)
        if event.get("ph") == "X":
            key = (event.get("pid"), event.get("tid"), Decimal(str(event["ts"])))
            tasks.setdefault(key, []).append(event)
        if event.get("name") == "torch_to_npu":
            key = (event.get("cat"), str(event.get("id")))
            if event.get("ph") == "s":
                flow_starts.append((key, event))
            elif event.get("ph") == "f":
                flow_ends.setdefault(key, []).append(event)
    annotations.sort(key=lambda a: a[0])
    if len(annotations) != iterations or any(a[1] > b[0] for a, b in zip(annotations, annotations[1:])):
        raise ValueError("expected one nonoverlapping L2 flush annotation per iteration; check profiler trace export")

    row_indices = {}
    for index, row in enumerate(rows):
        start, duration = Decimal(row["Start Time(us)"].strip()), Decimal(row["Duration(us)"])
        if not start.is_finite() or not duration.is_finite() or duration <= 0:
            raise ValueError("invalid kernel timestamp/duration")
        row_indices.setdefault((row["Name"], start), []).append(index)

    ownership = {}
    for key, source in flow_starts:
        source_time = Decimal(str(source["ts"]))
        owners = [
            i
            for i, (a, b, pid, tid) in enumerate(annotations)
            if source.get("pid") == pid and source.get("tid") == tid and a <= source_time < b
        ]
        if not owners:
            continue
        destinations = flow_ends.get(key, [])
        if len(owners) != 1 or len(destinations) != 1:
            raise ValueError(f"missing/ambiguous L2 torch_to_npu flow: {key}")
        destination = destinations[0]
        device_start = Decimal(str(destination["ts"]))
        linked = tasks.get((destination.get("pid"), destination.get("tid"), device_start), [])
        if not linked:
            raise ValueError(f"L2 flow endpoint has no device task: {key}")
        for task in linked:
            matches = row_indices.get((task["name"], device_start), [])
            if not matches:
                if str(task.get("args", {}).get("Task Type", "")).startswith("KERNEL"):
                    raise ValueError(f"L2 device kernel missing from CSV: {task['name']} at {device_start}")
                continue  # e.g. EVENT RECORD; not a kernel_details entry
            if len(matches) != 1:
                raise ValueError("ambiguous L2 kernel CSV identity")
            index = matches[0]
            if rows[index]["Name"] in STAGES:
                raise ValueError("L2 flush flow points to an API stage")
            if index in ownership and ownership[index] != owners[0]:
                raise ValueError("device kernel belongs to multiple L2 flush calls")
            ownership[index] = owners[0]

    kept, excluded = [], [[] for _ in annotations]
    for index, row in enumerate(rows):
        if index not in ownership:
            kept.append(row)
            continue
        if str(row["Device_id"]).strip() != str(device) or str(row["Step Id"]).strip() != "1":
            raise ValueError("unexpected device or step in L2 flush")
        excluded[ownership[index]].append(
            dict(name=row["Name"], start_us=row["Start Time(us)"].strip(), duration_us=row["Duration(us)"])
        )
    if any(not group for group in excluded):
        raise ValueError("L2 flush has no linked device kernels; check torch_to_npu flows in trace export")
    # Sample boundaries must also use device timestamps, never Host ranges.
    intervals = [
        (
            min(Decimal(r["start_us"]) for r in group),
            max(Decimal(r["start_us"]) + Decimal(r["duration_us"]) for r in group),
        )
        for group in excluded
    ]
    return kept, intervals, excluded


def parse_profile(
    root,
    op,
    expanded,
    iterations,
    device,
    aiv,
    *,
    api="v2",
    flush_l2_cache=False,
    sample_markers=False,
    require_combine_notify=False,
):
    """Fail closed on missing, duplicated or unrecognizable profiler samples."""
    if api not in ("v1", "v2") or iterations <= 0:
        raise ValueError("invalid API or iteration count")
    root = Path(root)
    csvs = list(root.rglob("kernel_details.csv"))
    traces = list(root.rglob("trace_view.json"))
    infos = list(root.rglob("profiler_info*.json"))
    if len(csvs) != 1 or len(traces) != 1 or len(infos) != 1:
        raise ValueError(f"{root}: expected one kernel CSV, trace and profiler_info")
    artifacts = [file_identity(p) for p in csvs + traces + infos]
    if any(p["size"] == 0 for p in artifacts):
        raise ValueError("empty profiler artifact")
    with csvs[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: Decimal(r["Start Time(us)"].strip()))
    sample_intervals, sample_host_ranges, discarded = [], None, 0
    if sample_markers:
        rows, sample_intervals, sample_host_ranges, discarded = select_profile_samples(
            rows, traces[0], iterations, device, allow_shmem_barrier=op.startswith("combine")
        )
    flush_intervals, flush_kernels = [], []
    if flush_l2_cache:
        rows, flush_intervals, flush_kernels = split_cache_flush(
            rows, traces[0], iterations, device, sample_host_ranges=sample_host_ranges
        )
    combine = op.startswith("combine")
    fused = api == "v1" and combine
    counts = Counter(r["Name"] for r in rows)
    early = bool(counts["v1_combine_udma_early_kernel"] or counts["v1_combine_udma_remaining_kernel"])
    if early and not (combine and expanded and api == "v2"):
        raise ValueError("early single-source stages require expanded combine")
    main = (
        ("v1_combine_udma_remaining_kernel" if early else "v1_combine_udma_comm_kernel")
        if combine
        else "v1_dispatch_udma_kernel"
    )
    if fused:
        main = "v1_combine_udma_kernel"
    required = [main]
    # New V2 runs require a separate all-AIV entry rendezvous. Archived reports
    # without it remain readable; partial/mixed sequences always fail below.
    combine_notify = (
        ["cached_notify_combine_kernel"]
        if early or counts["cached_notify_combine_kernel"] or (combine and require_combine_notify)
        else []
    )
    required += (
        combine_notify + ([] if fused else ["v1_combine_udma_reduce_kernel"])
        if combine
        else ["dispatch_layout_kernel", "notify_dispatch_kernel"]
        if op == "dispatch"
        else ["cached_notify_dispatch_kernel"]
    )
    if expanded:
        required.append("aggregate_kernel" if combine else "expand_kernel")
    if early:
        required.append("v1_combine_udma_early_kernel")
        if counts["v1_combine_udma_comm_kernel"]:
            raise ValueError("mixed serial and early communication stages")
    if any(counts[n] != iterations for n in required):
        raise ValueError(f"{root}: unexpected kernel counts {counts}; required {required}")
    if any(name in STAGES and name not in required for name in counts):
        raise ValueError(f"unexpected known stage for {api}/{op}: {counts}")
    forbidden = (
        ["expand_kernel", "dispatch_layout_kernel", "notify_dispatch_kernel"]
        if combine
        else ["aggregate_kernel", "v1_combine_udma_reduce_kernel"]
    )
    forbidden += (
        [
            "v1_combine_udma_comm_kernel",
            "v1_combine_udma_reduce_kernel",
            "v1_combine_udma_early_kernel",
            "v1_combine_udma_remaining_kernel",
        ]
        if fused
        else ["v1_combine_udma_kernel"]
    )
    if not expanded:
        forbidden += ["aggregate_kernel", "expand_kernel"]
    if op == "cached-dispatch":
        forbidden += ["dispatch_layout_kernel", "notify_dispatch_kernel"]
    if any(counts[n] for n in forbidden):
        raise ValueError(f"unexpected stages: {counts}")
    if not rows or len(rows) % iterations:
        raise ValueError("incomplete profiler iteration")
    width = len(rows) // iterations
    sequence = [r["Name"] for r in rows[:width]]
    if any([r["Name"] for r in rows[i * width : (i + 1) * width]] != sequence for i in range(iterations)):
        raise ValueError("kernel sequence changed between iterations; cannot assign samples")
    if combine and not fused and sequence.index(main) >= sequence.index("v1_combine_udma_reduce_kernel"):
        raise ValueError("reduce precedes communication")
    if expanded and (
        (combine and sequence.index("aggregate_kernel") >= sequence.index(main))
        or (not combine and sequence.index(main) >= sequence.index("expand_kernel"))
    ):
        raise ValueError("local stage order mismatch")
    if early and not (
        sequence.index("cached_notify_combine_kernel")
        < sequence.index("v1_combine_udma_early_kernel")
        < sequence.index("aggregate_kernel")
    ):
        raise ValueError("early send must follow initialization and precede aggregate")
    notify = (
        "cached_notify_combine_kernel"
        if combine
        else ("notify_dispatch_kernel" if op == "dispatch" else "cached_notify_dispatch_kernel")
    )
    if notify in sequence and sequence.index(notify) >= sequence.index(main):
        raise ValueError("communication precedes notify")
    if op == "dispatch" and sequence.index("dispatch_layout_kernel") >= sequence.index(notify):
        raise ValueError("notify precedes layout")
    samples = []
    kernels = []
    for i in range(iterations):
        chunk = rows[i * width : (i + 1) * width]
        if sample_markers:
            a, b = sample_intervals[i]
            if any(
                Decimal(r["Start Time(us)"].strip()) < a
                or Decimal(r["Start Time(us)"].strip()) + Decimal(r["Duration(us)"]) > b
                for r in chunk
            ):
                raise ValueError("kernel sequence does not match explicit sample ID")
        if flush_l2_cache:
            begin = Decimal(chunk[0]["Start Time(us)"].strip())
            end = max(Decimal(r["Start Time(us)"].strip()) + Decimal(r["Duration(us)"]) for r in chunk)
            if begin < flush_intervals[i][1] or (i + 1 < iterations and end > flush_intervals[i + 1][0]):
                raise ValueError("API sample is not between consecutive L2 flushes")
        times = {
            name + "_us": Decimal(0)
            for name in (
                "layout",
                "notify",
                "early_submit",
                "comm",
                "expand",
                "aggregate",
                "reduce",
                "fused_combine",
                "other",
            )
        }
        for ordinal, row in enumerate(chunk):
            if str(row["Device_id"]).strip() != str(device) or str(row["Step Id"]).strip() != "1":
                raise ValueError("unexpected device or profiler step")
            name = row["Name"]
            duration = Decimal(row["Duration(us)"])
            if not duration.is_finite() or duration <= 0:
                raise ValueError("nonpositive/nonfinite kernel duration")
            if not Decimal(row["Start Time(us)"].strip()).is_finite():
                raise ValueError("nonfinite kernel timestamp")
            if (
                STAGES.get(name) in ("early_submit", "comm", "expand", "aggregate", "reduce", "fused_combine")
                or name == "cached_notify_combine_kernel"
            ) and int(row["Block Num"]) != aiv:
                raise ValueError("unexpected launch AIV count")
            times[STAGES.get(name, "other") + "_us"] += duration
            kernels.append(
                dict(
                    sample=i,
                    ordinal=ordinal,
                    name=name,
                    stage=STAGES.get(name, "other"),
                    start_us=row["Start Time(us)"].strip(),
                    duration_us=str(duration),
                    device=device,
                    block_num=row["Block Num"],
                )
            )
        times["device_sum_us"] = sum(times.values())
        times["late_comm_us"] = times["comm_us"]
        times["comm_us"] += times["early_submit_us"]
        starts = [Decimal(r["Start Time(us)"].strip()) for r in chunk]
        ends = [s + Decimal(r["Duration(us)"]) for s, r in zip(starts, chunk)]
        times["device_span_us"] = max(ends) - min(starts)
        times["combine_stage_us"] = times["aggregate_us"] + times["comm_us"] if combine else None
        stage_rows = [r for r in chunk if STAGES.get(r["Name"]) in ("early_submit", "aggregate", "comm")]
        times["combine_stage_span_us"] = (
            (
                max(Decimal(r["Start Time(us)"]) + Decimal(r["Duration(us)"]) for r in stage_rows)
                - min(Decimal(r["Start Time(us)"]) for r in stage_rows)
            )
            if combine and not fused
            else None
        )
        times["combine_with_reduce_us"] = times["aggregate_us"] + times["fused_combine_us"] if fused else None
        if fused:
            for key in ("comm_us", "late_comm_us", "reduce_us", "combine_stage_us", "combine_stage_span_us"):
                times[key] = None
        samples.append(dict(sample=i, **{k: None if v is None else str(v) for k, v in times.items()}))
    result = dict(
        samples=samples,
        artifacts=artifacts,
        kernel_sequence=sequence,
        kernels=kernels,
        l2_flush_kernels=flush_kernels,
        unseparated_stages=["comm", "reduce"] if fused else [],
        absent_stages=[
            k
            for k in ("layout", "expand", "aggregate", "reduce")
            if not (fused and k == "reduce") and not any(STAGES.get(n) == k for n in sequence)
        ],
    )
    if sample_markers:
        result["sampling"] = dict(
            mode="device_event_markers_v1",
            sample_ids=list(range(iterations)),
            guard_iterations_per_side=PROFILE_GUARD_ITERS,
            excluded_boundary_kernels=discarded,
            device_intervals_us=[[str(a), str(b)] for a, b in sample_intervals],
        )
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("refusing to publish an empty result table")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    """Rank-local and job-wide effective throughput, never average(rank GB/s)."""
    output = []
    for op in records[0]["operators"]:
        local = []
        for record in records:
            data = record["operators"][op]
            samples = data["profile"]["samples"]
            n = len(samples)
            row = dict(rank=record["rank"], op=op, **data["bytes"])
            for key in samples[0]:
                if key != "sample":
                    row[key] = None if samples[0][key] is None else float(sum(Decimal(s[key]) for s in samples) / n)
            row["caller_us"] = sum(data["caller_us"]) / len(data["caller_us"])
            row["remote_payload_GBps"] = bandwidth(row["remote_payload_bytes"], row["comm_us"])
            row["remote_application_GBps"] = bandwidth(row["remote_application_bytes"], row["comm_us"])
            row["official_logical_GBps"] = bandwidth(row["official_logical_bytes"], row["comm_us"])
            row["logical_fused_combine_GBps"] = bandwidth(row["official_logical_bytes"], row["fused_combine_us"])
            row["logical_combine_with_reduce_GBps"] = bandwidth(
                row["official_logical_bytes"], row["combine_with_reduce_us"]
            )
            row["remote_payload_fused_combine_GBps"] = bandwidth(row["remote_payload_bytes"], row["fused_combine_us"])
            row["official_logical_combine_stage_GBps"] = (
                None
                if row["combine_stage_us"] is None
                else row["official_logical_bytes"] / row["combine_stage_us"] / 1000
            )
            row["algorithm_device_sum_GBps"] = row["algorithm_bytes"] / row["device_sum_us"] / 1000
            row["algorithm_device_span_GBps"] = row["algorithm_bytes"] / row["device_span_us"] / 1000
            row["algorithm_caller_GBps"] = row["algorithm_bytes"] / row["caller_us"] / 1000
            output.append(row)
            local.append(row)
        # Lockstep sample number, max-rank elapsed per operation. No cross-host clock subtraction.
        job = dict(rank="all", op=op)
        job["official_logical_rows"] = sum(r["official_logical_rows"] for r in local)
        for key in local[0]:
            if key.endswith("_bytes") and key != "wire_bytes":
                job[key] = sum(r[key] for r in local)
        for key in (
            "layout_us",
            "notify_us",
            "early_submit_us",
            "late_comm_us",
            "comm_us",
            "expand_us",
            "aggregate_us",
            "reduce_us",
            "fused_combine_us",
            "combine_with_reduce_us",
            "other_us",
            "device_sum_us",
            "device_span_us",
            "combine_stage_us",
            "combine_stage_span_us",
        ):
            if local[0][key] is None:
                job[key] = None
                continue
            groups = [r["operators"][op]["profile"]["samples"] for r in records]
            job[key] = float(
                sum(max(Decimal(g[i][key]) for g in groups) for i in range(len(groups[0]))) / len(groups[0])
            )
        groups = [r["operators"][op]["caller_us"] for r in records]
        job["caller_us"] = sum(max(g[i] for g in groups) for i in range(len(groups[0]))) / len(groups[0])
        for numerator, denominator, label in [
            ("official_logical_bytes", "comm_us", "official_logical_GBps"),
            ("remote_payload_bytes", "comm_us", "remote_payload_GBps"),
            ("remote_application_bytes", "comm_us", "remote_application_GBps"),
            ("official_logical_bytes", "fused_combine_us", "logical_fused_combine_GBps"),
            ("official_logical_bytes", "combine_with_reduce_us", "logical_combine_with_reduce_GBps"),
            ("remote_payload_bytes", "fused_combine_us", "remote_payload_fused_combine_GBps"),
            ("algorithm_bytes", "device_sum_us", "algorithm_device_sum_GBps"),
            ("algorithm_bytes", "device_span_us", "algorithm_device_span_GBps"),
            ("algorithm_bytes", "caller_us", "algorithm_caller_GBps"),
        ]:
            job[label] = bandwidth(job[numerator], job[denominator])
        job["official_logical_combine_stage_GBps"] = (
            None if job["combine_stage_us"] is None else job["official_logical_bytes"] / job["combine_stage_us"] / 1000
        )
        output.append(job)
    return output


def bandwidth(byte_count, microseconds):
    """An absent/fused-away timer has no independently measured bandwidth."""
    return None if microseconds is None or microseconds == 0 else byte_count / microseconds / 1000
