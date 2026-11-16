# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Aligned eager batches and device-kernel measurements."""

from datetime import timedelta
from decimal import Decimal, InvalidOperation
import csv
import re
import math
from pathlib import Path
import statistics


# 用于对齐各 rank 测试排队的默认前置计算，不在算子/计时区间内。
DEFAULT_ALIGNMENT = dict(rows=100000, cols=30000, repeats=20)


def alignment_config(config):
    value = dict(config.get("alignment", DEFAULT_ALIGNMENT))
    if set(value) != set(DEFAULT_ALIGNMENT) or any(type(n) is not int or n < 1 for n in value.values()):
        raise ValueError("alignment requires positive integer rows, cols, repeats")
    return value


def flush_l2_before_call():
    """Evict device L2 before one performance-sampled MegaMoE call."""
    from deep_ep.utils.testing import flush_l2_cache

    flush_l2_cache(True)


def enqueue_batch(call, torch, prefix, alignment, iterations, timer=None):
    """No host barriers, synchronize, or D2H between collective launches."""
    for _ in range(alignment["repeats"]):
        torch.exp(prefix, out=prefix)
    for iteration in range(iterations):
        flush_l2_before_call()
        # 所有 rank 排入同样次数；只在最后一轮、选中的 rank 上打点。
        call(timer if iteration + 1 == iterations else None)


def read_kernel_samples(directory, iterations):
    """Read only this rank/case's fresh profiler export; never subtract device clocks."""
    samples = []
    paths = sorted(Path(directory).rglob("kernel_details*.csv"))
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            for index, row in enumerate(reader):
                row = {key.strip(): value for key, value in row.items() if key is not None}
                name = " ".join(row.get(key, "") for key in ("Name", "Type", "Kernel Name"))
                if "megamoe" not in name.replace("_", "").lower():
                    continue
                try:
                    start, duration = (float(row[key]) for key in ("Start Time(us)", "Duration(us)"))
                except (KeyError, ValueError) as error:
                    raise ValueError(f"Invalid MegaMoE profiler row {path}:{index + 2}: {row}") from error
                if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Nonfinite/nonpositive MegaMoE timing: {path}:{index + 2}")
                samples.append(
                    dict(
                        start_us=start,
                        duration_us=duration,
                        name=name,
                        stream_id=row.get("Stream ID"),
                        device_id=row.get("Device_id"),
                        file=str(path),
                        row=index + 2,
                    )
                )
    samples.sort(key=lambda sample: sample["start_us"])
    keys = [(s["device_id"], s["stream_id"], s["start_us"]) for s in samples]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate MegaMoE kernel samples under {directory}; refuse misaligned rounds")
    if len(samples) != iterations:
        raise ValueError(
            f"{directory}: expected {iterations} MegaMoE kernels, found {len(samples)} "
            f"in {len(paths)} kernel_details CSV files; inspect the raw profiler export"
        )
    return samples


def summarize_rounds(per_rank, ranks_per_node=None):
    """Aggregate only the last ceil(formal rounds / 2) samples."""
    if not per_rank or not per_rank[0] or any(len(row) != len(per_rank[0]) for row in per_rank):
        raise ValueError("Missing ranks or mismatched profiling rounds")
    if any(not 0 < value < float("inf") for row in per_rank for value in row):
        raise ValueError("Non-finite/nonpositive timing sample")
    # Validate the complete capture before selection; malformed early samples
    # must not silently pass. Raw kernel_samples.json keeps every formal round.
    recorded_rounds = len(per_rank[0])
    first_selected_iteration = recorded_rounds // 2
    per_rank = [row[first_selected_iteration:] for row in per_rank]
    minimum = [min(values) for values in zip(*per_rank)]
    maximum = [max(values) for values in zip(*per_rank)]
    report = dict(
        metric="megamoe_kernel_duration_us",
        aggregation="mean_of_per_round_min_rank",
        iteration_selection="tail_half_ceil",
        recorded_rounds=recorded_rounds,
        first_selected_iteration=first_selected_iteration,
        rounds=len(minimum),
        mean_us=statistics.mean(minimum),
        median_us=statistics.median(minimum),
        min_us=min(minimum),
        max_us=max(minimum),
        rank_max_mean_us=statistics.mean(maximum),
        rank_min_rounds_us=minimum,
        rank_max_rounds_us=maximum,
        per_rank_rounds_us=per_rank,
        timer_pass=False,
        scope="all_ep_ranks",
    )
    if ranks_per_node is not None:
        if ranks_per_node < 1 or len(per_rank) % ranks_per_node:
            raise ValueError("ranks_per_node must divide the sampled world size")
        report["per_node"] = [
            dict(
                node=node,
                rank_start=offset,
                mean_us=statistics.mean(min(values) for values in zip(*per_rank[offset : offset + ranks_per_node])),
            )
            for node, offset in enumerate(range(0, len(per_rank), ranks_per_node))
        ]
    return report


def collect_profile(call, group, torch, config, prefix, directory, bs):
    import torch_npu

    profiler = torch_npu.profiler
    experimental = profiler._ExperimentalConfig(
        export_type=profiler.ExportType.Text,
        aic_metrics=profiler.AiCMetrics.PipeUtilization,
        profiler_level=profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental,
    ) as prof:
        # profiler 本身可能初始化很慢，必须在进入后对齐；不能逐轮打断设备队列。
        torch.npu.synchronize()
        group.barrier(f"profiler ready BS={bs}")
        enqueue_batch(call, torch, prefix, config["alignment"], config["iters"])
        torch.npu.synchronize()
        print(f"[PROFILE] BS={bs} batch complete; exporting {directory}", flush=True)
        prof.step()
    return read_kernel_samples(directory, config["iters"])


class UrmaLaunchGuard:
    """Benchmark-only launch alignment plus reusable FP32 exp buffers.

    The default backend owns a dedicated HCCL domain. The ``store`` backend
    keeps the same exp sequence but uses the host Store barrier instead, for
    machines where the HCCL device transport cannot initialize (for example a
    down NPU network port). StoreGroup remains responsible for control/reporting.
    Its unique session namespaces the HCCL rendezvous without creating a second
    TCP listener or reusing another job's communication domain. This standalone
    benchmark must not already have a torch.distributed default process group.
    """

    def __init__(self, group, shape=(100000, 20000), repeat=50, timeout=300, *, backend="hccl"):
        import torch
        import torch.distributed as dist

        self.torch, self.dist = torch, dist
        self.group = group
        self.backend = backend
        self.shape, self.repeat = tuple(shape), repeat
        self.bytes = math.prod(self.shape) * 4 * 2
        self.input = self.output = self.store = self.process_group = None
        self.alignment = None
        error = None
        try:
            if backend not in ("hccl", "store"):
                raise ValueError("launch guard backend must be 'hccl' or 'store'")
            if backend == "hccl" and dist.is_initialized():
                raise RuntimeError("URMA standalone benchmark requires no pre-existing ProcessGroup")
            if not self.shape or any(type(n) is not int or n < 1 for n in self.shape):
                raise ValueError("exp shape dimensions must be positive integers")
            if type(repeat) is not int or repeat < 0 or timeout <= 0:
                raise ValueError("exp repeat must be nonnegative and HCCL timeout positive")
            if repeat:
                self.input = torch.zeros(self.shape, dtype=torch.float32, device="npu")
                self.output = torch.empty_like(self.input)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        # Propagate allocation/preflight failures before peers enter HCCL init.
        checks = group.all_gather(error, "URMA launch guard allocation")
        if any(checks):
            self.input = self.output = None
            raise RuntimeError(
                f"URMA launch guard needs {self.bytes / 2**30:.2f} GiB/rank for two FP32 buffers; "
                "adjust the alignment rows/cols/repeats in the model config. "
                + "; ".join(f"rank={rank}: {item}" for rank, item in enumerate(checks) if item)
            )
        if backend == "hccl":
            self.store = dist.PrefixStore(group._prefix + "urma-benchmark-hccl/", group._store)
            dist.init_process_group(
                backend="hccl",
                store=self.store,
                rank=group.rank(),
                world_size=group.size(),
                timeout=timedelta(seconds=timeout),
            )
            self.process_group = dist.group.WORLD

    def begin(self):
        """Enqueue the guard on the caller stream; never synchronize after exp.

        The HCCL barrier is a device collective; the Store fallback is a host
        barrier for machines where HCCL transport cannot initialize. The public
        forward subsequently inserts owner_stream.wait_stream(caller),
        so the dedicated native stream cannot overtake this exp sequence.
        """
        if self.process_group is not None:
            self.dist.barrier(group=self.process_group)
        else:
            self.group.barrier("URMA guard start")
        for _ in range(self.repeat):
            self.torch.exp(self.input, out=self.output)

    def prepare_per_call_alignment(self):
        if self.process_group is None:
            raise RuntimeError("per-call device all-reduce requires the HCCL guard; no host fallback")
        # SUM of zero stays finite across every invocation; allocate outside profiling.
        self.alignment = self.torch.zeros((1,), dtype=self.torch.float32, device="npu")

    def before_forward(self):
        if self.alignment is None:
            raise RuntimeError("prepare_per_call_alignment must precede profiling")
        self.dist.all_reduce(self.alignment, group=self.process_group)

    def end(self):
        # The public forward orders caller after owner. The barrier therefore sees
        # the completed block; neither this barrier nor synchronization is timed.
        if self.process_group is not None:
            self.dist.barrier(group=self.process_group)
        else:
            self.group.barrier("URMA guard end")
        self.torch.npu.synchronize()

    def close(self):
        if self.process_group is not None:
            self.dist.destroy_process_group(self.process_group)
            self.process_group = None
        self.input = self.output = self.alignment = None

    def describe(self):
        if self.backend == "hccl":
            method = "HCCL barrier + finite-input exp before continuous block; HCCL barrier + synchronize after"
            limitations = "reduces host launch skew; does not equalize physical device speeds; device all-reduce precedes each invocation when enabled"
        else:
            method = (
                "host Store barrier + finite-input exp before continuous block; host Store barrier + synchronize after"
            )
            limitations = (
                "host-side barrier used where HCCL transport cannot initialize; start skew is larger than the "
                "HCCL device barrier, so label these numbers separately from HCCL-guard runs"
            )
        return dict(
            method=method,
            per_call_device_all_reduce=self.alignment is not None,
            backend=self.backend,
            exp_shape=list(self.shape),
            exp_repeat=self.repeat,
            allocated_bytes_per_rank=self.bytes if self.repeat else 0,
            scope="one profiler block: warmup + formal iterations + optional fine",
            stream_order="caller barrier/exp -> owner.wait_stream(caller) -> native -> caller.wait_stream(owner)",
            limitations=limitations,
        )


# CANN may export the raw C++ symbol rather than the demangled kernel name.
# Match the exact device entry/signature so the host launcher (_do) and other
# kernels remain excluded; do not loosen this to a substring search.
_URMA_KERNEL = re.compile(
    r"(?:^|::|\s)mega_moe_urma_(?:shared_)?kernel(?=\s*(?:$|\(|<))"
    r"|^_Z20mega_moe_urma_kernelN11MegaMoeUrma9ArgumentsE$"
    r"|^_Z27mega_moe_urma_shared_kernelN11MegaMoeUrma9ArgumentsE$"
)


def read_profiled_kernels(directory, expected_samples):
    """Read exact URMA device tasks, retaining precise timestamps for ordering.

    No timestamp subtraction across ranks: timestamps only establish each
    rank's launch order; device task duration supplies the measured time.
    """
    samples = []
    paths = sorted(Path(directory).rglob("kernel_details*.csv"))
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for index, raw in enumerate(csv.DictReader(stream), start=2):
                row = {key.strip(): (value or "").strip() for key, value in raw.items() if key is not None}
                name = next(
                    (row[key] for key in ("Name", "Kernel Name", "Op Name") if _URMA_KERNEL.search(row.get(key, ""))),
                    None,
                )
                if name is None:
                    continue
                try:
                    start = Decimal(row["Start Time(us)"])
                    duration = Decimal(row["Duration(us)"])
                except (KeyError, InvalidOperation) as error:
                    raise ValueError(f"Invalid URMA profiler row {path}:{index}: {row}") from error
                if not start.is_finite() or not duration.is_finite() or start < 0 or duration <= 0:
                    raise ValueError(f"Nonfinite/negative/nonpositive URMA timing: {path}:{index}")
                if not math.isfinite(float(start)) or not math.isfinite(float(duration)) or float(duration) <= 0:
                    raise ValueError(f"URMA timing outside supported numeric range: {path}:{index}")
                samples.append(
                    dict(
                        start_us=float(start),
                        start_us_decimal=str(start),
                        duration_us=float(duration),
                        name=name,
                        stream_id=row.get("Stream ID") or row.get("Stream Id"),
                        device_id=row.get("Device_id") or row.get("Device ID"),
                        file=str(path),
                        row=index,
                    )
                )
    samples.sort(key=lambda sample: Decimal(sample["start_us_decimal"]))
    keys = [(s["device_id"], s["stream_id"], Decimal(s["start_us_decimal"])) for s in samples]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate URMA kernel samples under {directory}; refuse misaligned rounds")
    if len({s["stream_id"] for s in samples}) > 1 or len({s["device_id"] for s in samples}) > 1:
        raise ValueError(f"Multiple URMA device/stream IDs under {directory}; expected one rank-local owner stream")
    if len(samples) != expected_samples:
        raise ValueError(
            f"{directory}: expected {expected_samples} mega_moe_urma_kernel tasks, found {len(samples)} "
            f"in {len(paths)} kernel_details CSV files; inspect the raw profiler export"
        )
    return samples


def measure_profiled_forward(forward, *, warmup, iters, guard, directory, fine_forward=None):
    """Profile one guarded continuous block, with no Events or per-round sync.

    The optional fine call runs once after formal iterations in the same
    profiler context and is excluded from formal statistics.
    """
    import torch_npu

    if type(warmup) is not int or warmup < 0 or type(iters) is not int or iters < 1:
        raise ValueError("warmup must be a nonnegative integer and iters a positive integer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=profiler._ExperimentalConfig(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level0,
            aic_metrics=profiler.AiCMetrics.AiCoreNone,
            data_simplification=False,
        ),
    ) as capture:
        # Initialize profiler before the existing guard; never drain between
        # exp and the continuous launches or insert a profiler step per round.
        guard.begin()
        for _ in range(warmup + iters):
            guard.before_forward()
            flush_l2_before_call()
            forward()
        if fine_forward is not None:
            guard.before_forward()
            flush_l2_before_call()
            fine_forward()
        guard.end()
        capture.step()
    modes = ["fine"] if fine_forward is not None else []
    samples = read_profiled_kernels(directory, warmup + iters + len(modes))
    for iteration, sample in enumerate(samples):
        sample["iteration"] = iteration
        sample["phase"] = (
            "warmup"
            if iteration < warmup
            else "formal"
            if iteration < warmup + iters
            else modes[iteration - warmup - iters]
        )
    diagnostics = dict(zip(modes, samples[warmup + iters :]))
    return dict(
        samples=samples,
        durations_us=[sample["duration_us"] for sample in samples[warmup : warmup + iters]],
        diagnostics=diagnostics,
        diagnostic_sample=samples[-1] if modes else None,
        profile_dir=str(directory),
    )


def summarize_profile_rounds(per_rank, ranks_per_node=16):
    """Average cross-rank minima/maxima per iteration, using every formal round."""
    if not per_rank or not per_rank[0] or any(len(row) != len(per_rank[0]) for row in per_rank):
        raise ValueError("Missing ranks or mismatched profiling rounds")
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for row in per_rank for value in row):
        raise ValueError("Non-finite/nonpositive timing sample")
    if type(ranks_per_node) is not int or ranks_per_node < 1 or len(per_rank) % ranks_per_node:
        raise ValueError("ranks_per_node must divide the sampled world size")
    rows = [[float(value) for value in row] for row in per_rank]
    minimum = [min(values) for values in zip(*rows)]
    maximum = [max(values) for values in zip(*rows)]
    report = dict(
        metric="mega_moe_urma_kernel_device_duration_us",
        aggregation="mean_of_per_round_cross_rank_extrema",
        iteration_selection="all_formal_rounds",
        recorded_rounds=len(minimum),
        first_selected_iteration=0,
        rounds=len(minimum),
        rank_max_mean_us=statistics.mean(maximum),
        rank_min_mean_us=statistics.mean(minimum),
        rank_min_rounds_us=minimum,
        rank_max_rounds_us=maximum,
        raw_per_rank_rounds_us=rows,
        per_rank_rounds_us=rows,
        timer_pass=False,
        scope="all_ep_ranks",
    )
    report["per_node"] = []
    for node, offset in enumerate(range(0, len(rows), ranks_per_node)):
        node_rounds = list(zip(*rows[offset : offset + ranks_per_node]))
        report["per_node"].append(
            dict(
                node=node,
                rank_start=offset,
                rank_max_mean_us=statistics.mean(max(values) for values in node_rounds),
                rank_min_mean_us=statistics.mean(min(values) for values in node_rounds),
            )
        )
    return report
