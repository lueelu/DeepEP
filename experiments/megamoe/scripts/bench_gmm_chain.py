# Copyright (c) 2026, Lu Lu

"""Profile torch_npu's fused MXFP8/FP4 GMM1 + SwiGLU + quant -> GMM2 on one NPU.

Run directly from the checkout; no DeepEP extension, SHMEM or process group is
needed. --rank selects the logical expert owner in the existing EP benchmark,
not a distributed process. All data preparation and validation precede profiling.
"""

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/megamoe/python"))

from deep_ep_experimental.megamoe.benchmark import inputs  # noqa: E402
from deep_ep_experimental.megamoe.golden import compare  # noqa: E402
from deep_ep_experimental.megamoe.reference import (  # noqa: E402
    dequantize_input,
    error_metrics,
    expert_forward,
    make_weights,
    quantize_input,
    unpack_weight,
)


def positive_values(value):
    try:
        values = [int(part.strip()) for part in value.split(",")]
        if not values or min(values) < 1 or len(set(values)) != len(values):
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use distinct positive integers separated by commas") from error
    return values


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Physical NPU index in this process")
    parser.add_argument(
        "--rank", type=int, default=0, help="Logical expert-owner rank whose inputs/weights to reproduce"
    )
    ep = parser.add_mutually_exclusive_group()
    ep.add_argument("--ep-size", type=int)
    ep.add_argument("--ep-sizes", type=positive_values, help="Sweep logical EP sizes, e.g. 16,32,64")
    bs = parser.add_mutually_exclusive_group()
    bs.add_argument("--bs", type=int, help="Original tokens per source rank, NOT the GMM row count")
    bs.add_argument("--bs-values", type=positive_values, help="Sweep source BS values, e.g. 1,8,16,32,256")
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument(
        "--experts", type=int, default=384, help="Global routed experts; no shared experts in this test"
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--routing", choices=("balanced", "random", "skewed"), default="balanced")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--mode", choices=("eager", "graph"), default="eager", help="Graph mode explicitly uses NPUGraph replay"
    )
    parser.add_argument(
        "--check-rows-per-expert",
        type=int,
        default=2,
        help="CPU oracle samples per expert; 0 explicitly skips numerical validation",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-runs/gmm-chain", help="Parent of a fresh run directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print real routing counts/shapes without importing Torch or allocating weights",
    )
    args = parser.parse_args(argv)
    args.ep_size = 32 if args.ep_size is None else args.ep_size
    args.bs = 32 if args.bs is None else args.bs
    ep_sizes = args.ep_sizes or [args.ep_size]
    if args.experts < 1 or any(ep < 1 or args.experts % ep for ep in ep_sizes):
        parser.error("experts must be divisible by a positive ep-size")
    if any(not 0 <= args.rank < ep or not 1 <= args.experts // ep <= 128 for ep in ep_sizes):
        parser.error("rank must be in [0,ep-size), with 1..128 local experts")
    if args.bs < 1 or not 1 <= args.topk <= args.experts:
        parser.error("bs must be positive and topk must be in [1,experts]")
    if args.hidden < 64 or args.hidden % 64 or args.intermediate < 128 or args.intermediate % 128:
        parser.error("hidden must be a positive multiple of 64; intermediate a positive multiple of 128")
    if args.device < 0 or args.seed < 0 or args.warmup < 1 or args.iters < 1 or args.check_rows_per_expert < 0:
        parser.error("device/seed/check-rows-per-expert must be nonnegative; warmup/iters positive")
    return args


def sweep_cases(args):
    """Keep single-case defaults; plural arguments select the Cartesian product."""
    return [
        argparse.Namespace(**(vars(args) | dict(ep_size=ep, bs=bs, ep_sizes=None, bs_values=None)))
        for ep in args.ep_sizes or [args.ep_size]
        for bs in args.bs_values or [args.bs]
    ]


def case_config(args):
    return dict(
        schema_version=1,
        model=dict(
            hidden=args.hidden,
            intermediate_hidden=args.intermediate,
            num_experts=args.experts,
            num_topk=args.topk,
            num_shared_experts=0,
        ),
        bs_values=[args.bs],
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
    )


def routed_input(config, world, rank, bs, routing):
    """Reproduce Dispatch's logical output using the benchmark's source inputs.

    Stable expert grouping retains source-rank/token/top-k order within an expert.
    No route weights are applied: this benchmark ends before weighted Combine.
    """
    import numpy as np

    local = config["model"]["num_experts"] // world
    first = rank * local
    values, experts, coordinates = [], [], []
    for source in range(world):
        x, ids, _ = inputs(config, source, bs, routing, world=world)
        token, slot = np.nonzero((ids >= first) & (ids < first + local))
        values.append(x[token])
        experts.append(ids[token, slot] - first)
        coordinates.append(np.column_stack((np.full(len(token), source), token, slot)))
    expert = np.concatenate(experts)
    order = np.argsort(expert, kind="stable")
    return (
        np.ascontiguousarray(np.concatenate(values)[order]),
        np.bincount(expert, minlength=local).astype(np.int64),
        np.ascontiguousarray(np.concatenate(coordinates)[order], dtype=np.int64),
    )


def describe(args, counts):
    rows, local = int(counts.sum()), len(counts)
    return dict(
        scope="single_npu_logical_expert_owner",
        logical_rank=args.rank,
        ep_size=args.ep_size,
        source_bs=args.bs,
        local_experts=local,
        routed_rows=rows,
        expert_counts=counts.tolist(),
        group_list_type=1,
        group_list=counts.tolist(),
        gmm1=dict(m=rows, k=args.hidden, n=2 * args.intermediate),
        activation_shape=[rows, args.intermediate],
        gmm2=dict(m=rows, k=args.intermediate, n=args.hidden),
        input="MXFP8 E4M3",
        weights="MXFP4 E2M1 FRACTAL_NZ",
        scales="E8M0 block32",
        output="BF16",
        mode=args.mode,
        excluded=["Dispatch", "Combine", "route weights", "input quantization", "weight preparation", "CPU validation"],
    )


def prepare_chain(torch, torch_npu, args, routed, counts, raw_weights):
    """Match the documented MxA8W4 NZ API, using planar gate/up weights.

    See Ascend/op-plugin docs/zh/custom_APIs/torch_npu/
    torch_npu-npu_grouped_matmul_swiglu_quant_v2.md (MxA8W4 example) and
    torch_npu-npu_grouped_matmul.md (mx pseudo-quantization example).
    """
    device = f"npu:{args.device}"
    codes, scales = quantize_input(routed)
    x = torch.from_numpy(codes).view(torch.float8_e4m3fn).to(device)
    x_scale = torch.from_numpy(scales.reshape(len(routed), args.hidden // 64, 2)).to(device)
    group_list = torch.from_numpy(counts).to(device)
    prepared = []
    for packed, scale in raw_weights:
        # Conversion is outside the timed callable. Preserve the transpose of
        # both weight and its paired scale; do not contiguous() these views.
        nz = torch_npu.npu_format_cast(
            torch.from_numpy(packed).to(device),
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        prepared.append((nz.transpose(-1, -2), torch.from_numpy(scale).to(device).transpose(-3, -2)))
    (w1, s1), (w2, s2) = prepared

    def call():
        activation, activation_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x,
            [w1],
            [s1],
            x_scale,
            group_list,
            dequant_mode=2,
            dequant_dtype=torch.float32,
            quant_mode=2,
            quant_dtype=torch.float8_e4m3fn,
            group_list_type=1,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu,
        )
        # A8W4 uses antiquant_scale, unlike the FP8-weight all-quantized path.
        return torch_npu.npu_grouped_matmul(
            [activation],
            [w2],
            antiquant_scale=[s2],
            per_token_scale=[activation_scale],
            group_list=group_list,
            group_list_type=1,
            group_type=0,
            split_item=2,
            output_dtype=torch.bfloat16,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
        )[0]

    return call, (codes, scales)


def validate_output(actual, quantized_input, counts, raw_weights, samples):
    """Check sampled expert outputs before routing weights/reduction, outside timing."""
    import numpy as np

    if not np.isfinite(actual).all():
        raise ValueError("GMM chain output contains non-finite values")
    if samples == 0:
        return dict(passed=None, policy="explicitly_skipped", checked_rows=0)
    expected, selected, begin = [], [], 0
    for expert, count in enumerate(counts):
        count = int(count)
        if count:
            rows = begin + np.unique(np.linspace(0, count - 1, min(samples, count)).astype(np.int64))
            x = dequantize_input(quantized_input[0][rows], quantized_input[1][rows])
            w1, w2 = (unpack_weight((data[expert], scale[expert])) for data, scale in raw_weights)
            expected.append(expert_forward(x, w1, w2))
            selected.extend(rows.tolist())
        begin += count
    expected = np.concatenate(expected)
    report = compare(actual[selected], expected)
    report.update(error_metrics(actual[selected], expected))
    return dict(
        report,
        checked_rows=len(selected),
        rows=selected,
        policy="existing_megamoe_numpy_mixed_error_before_route_weight",
        reference="BF16 GMM outputs/activation and block32 MXFP8; not a bitwise Cube emulator",
    )


def kernel_kind(name):
    normalized = "".join(char for char in name.lower() if char.isalnum())
    if "groupedmatmul" not in normalized and "groupmatmul" not in normalized:
        return None
    return "fused_gmm1" if "swiglu" in normalized else "gmm2"


def read_pairs(directory, iterations):
    """Require two ordered kernels per call, on one device/stream; fail closed."""
    kernels = []
    for path in sorted(Path(directory).rglob("kernel_details*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for line, row in enumerate(csv.DictReader(stream), start=2):
                row = {key.strip(): value for key, value in row.items() if key is not None}
                name = " ".join(row.get(key, "") for key in ("Name", "Type", "Kernel Name"))
                kind = kernel_kind(name)
                if kind is None:
                    raise ValueError(f"Unexpected kernel in two-op capture: {path}:{line}: {name}")
                try:
                    # Epoch timestamps in microseconds can have a float ULP of
                    # 0.25 us, exceeding our 0.01 us overlap tolerance. Keep the
                    # CSV decimals intact until all differences are calculated.
                    start, duration = (Decimal(row[key]) for key in ("Start Time(us)", "Duration(us)"))
                except (KeyError, TypeError, InvalidOperation) as error:
                    raise ValueError(f"Invalid kernel timing at {path}:{line}") from error
                if (
                    not start.is_finite()
                    or not duration.is_finite()
                    or not math.isfinite(float(start))
                    or not math.isfinite(float(duration))
                    or duration <= 0
                ):
                    raise ValueError(f"Non-finite/nonpositive kernel timing at {path}:{line}")
                device = row.get("Device_id", row.get("Device ID"))
                stream_id = row.get("Stream ID")
                if not device or not stream_id:
                    raise ValueError(f"Missing device/stream identity at {path}:{line}")
                kernels.append(
                    dict(
                        kind=kind,
                        name=name,
                        start_us=start,
                        duration_us=duration,
                        device=device,
                        stream=stream_id,
                        file=str(path),
                        row=line,
                    )
                )
    kernels.sort(key=lambda item: item["start_us"])
    if len(kernels) != iterations * 2:
        raise ValueError(f"Expected {iterations} fused GMM1/GMM2 pairs; found {len(kernels)} kernels under {directory}")
    if len({(k["device"], k["stream"]) for k in kernels}) != 1:
        raise ValueError("Capture spans devices/streams; refuse to subtract unrelated clocks or pair concurrent calls")
    if len({k["start_us"] for k in kernels}) != len(kernels):
        raise ValueError("Duplicate kernel timestamps/exports")
    for previous, current in zip(kernels, kernels[1:]):
        gap = current["start_us"] - previous["start_us"] - previous["duration_us"]
        if gap < Decimal("-0.01"):
            raise ValueError(
                f"Overlapping kernels on the captured stream: gap={gap} us (tolerance=0.01 us), "
                f"device={current['device']}, stream={current['stream']}; "
                f"{previous['file']}:{previous['row']} {previous['kind']} "
                f"start={previous['start_us']} duration={previous['duration_us']} -> "
                f"{current['file']}:{current['row']} {current['kind']} "
                f"start={current['start_us']} duration={current['duration_us']}"
            )
    pairs = []
    for index in range(iterations):
        first, second = kernels[2 * index : 2 * index + 2]
        if first["kind"] != "fused_gmm1" or second["kind"] != "gmm2":
            raise ValueError(f"Unexpected kernel ordering in round {index}: {first['name']} -> {second['name']}")
        gap = second["start_us"] - first["start_us"] - first["duration_us"]
        pairs.append(
            dict(
                iteration=index,
                fused_gmm1_us=float(first["duration_us"]),
                gmm2_us=float(second["duration_us"]),
                kernel_sum_us=float(first["duration_us"] + second["duration_us"]),
                chain_span_us=float(second["start_us"] - first["start_us"] + second["duration_us"]),
                inter_kernel_gap_us=max(0.0, float(gap)),
                inter_kernel_gap_raw_us=float(gap),
                kernels=[first, second],
            )
        )
    # Preserve numeric fields for existing consumers and exact strings for
    # auditing. Never use the rounded absolute start_us for timing arithmetic.
    for kernel in kernels:
        for key in ("start_us", "duration_us"):
            kernel[f"{key}_exact"] = str(kernel[key])
            kernel[key] = float(kernel[key])
    return pairs


METRICS = ("fused_gmm1_us", "gmm2_us", "kernel_sum_us", "chain_span_us", "inter_kernel_gap_us")


def summarize(pairs):
    if not pairs:
        raise ValueError("No formal timing samples")
    selected = pairs[len(pairs) // 2 :]
    report = dict(
        scope="single_npu_logical_expert_owner",
        iteration_selection="tail_half_ceil",
        recorded_rounds=len(pairs),
        first_selected_iteration=len(pairs) // 2,
        rounds=len(selected),
    )
    for metric in METRICS:
        values = [pair[metric] for pair in selected]
        report[metric] = dict(
            mean=statistics.mean(values), median=statistics.median(values), min=min(values), max=max(values)
        )
    return report


def collect_profile(call, torch, torch_npu, iterations, directory):
    profiler = torch_npu.profiler
    experimental = profiler._ExperimentalConfig(
        export_type=profiler.ExportType.Text,
        profiler_level=profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental,
    ) as prof:
        torch.npu.synchronize()
        for _ in range(iterations):
            call()
        # No event, synchronization, CPU copy or route processing between GMM1
        # and GMM2, or between formal rounds. Both kernels stay on one stream.
        torch.npu.synchronize()
        prof.step()
    return read_pairs(directory, iterations)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(args, config, routed, counts, coordinates, directory, report):
    import numpy as np
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    required = (
        "npu_grouped_matmul_swiglu_quant_v2",
        "npu_grouped_matmul",
        "npu_format_cast",
        "float4_e2m1fn_x2",
        "float8_e8m0fnu",
    )
    missing = [name for name in required if not hasattr(torch_npu, name)]
    if missing:
        raise RuntimeError(f"This torch_npu installation lacks the required MxA8W4 API: {missing}")
    report["environment"] = dict(
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        device_name=torch.npu.get_device_name(args.device),
        device=args.device,
        operator_schemas={name: str(getattr(torch.ops.npu, name).default._schema) for name in required[:2]},
    )
    write_json(directory / "results.json", report)
    raw = make_weights(len(counts), args.hidden, args.intermediate, args.seed + args.rank)
    report["input_sha256"] = hashlib.sha256(routed.tobytes() + coordinates.tobytes()).hexdigest()
    report["weights_sha256"] = [hashlib.sha256(data.tobytes() + scale.tobytes()).hexdigest() for data, scale in raw]
    with torch.inference_mode():
        print("[PREPARE] Quantizing input and converting packed FP4 weights to NZ", flush=True)
        call, quantized = prepare_chain(torch, torch_npu, args, routed, counts, raw)
        output = call()
        torch.npu.synchronize()
        if tuple(output.shape) != (len(routed), args.hidden) or output.dtype != torch.bfloat16:
            raise ValueError(
                f"Expected BF16 output {(len(routed), args.hidden)}, got {output.dtype} {tuple(output.shape)}"
            )
        print(f"[CHECK] CPU reference: up to {args.check_rows_per_expert} rows per expert", flush=True)
        report["accuracy"] = validate_output(
            output.float().cpu().numpy(), quantized, counts, raw, args.check_rows_per_expert
        )
        write_json(directory / "results.json", report)
        if report["accuracy"]["passed"] is False:
            raise RuntimeError(
                "Chain failed the existing MegaMoE numerical criterion; inspect results.json before comparing performance"
            )
        del raw, quantized
        print(f"[WARMUP] {args.warmup} rounds; mode={args.mode}", flush=True)
        for _ in range(args.warmup):
            output = call()
        torch.npu.synchronize()
        # Keep the eager closure (and thus all captured input/weight tensors)
        # alive for the entire replay/profiling lifetime.
        measured_call = call
        if args.mode == "graph":
            # Warm up on a side stream before capture, as required by NPUGraph.
            warm_stream = torch.npu.Stream()
            warm_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(warm_stream):
                for _ in range(3):
                    output = call()
            torch.npu.current_stream().wait_stream(warm_stream)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured_output = call()
            measured_call = graph.replay
            for _ in range(args.warmup):
                measured_call()
            torch.npu.synchronize()
            # Retain the captured output and verify replay before measuring.
            if not np.array_equal(captured_output.float().cpu().numpy(), output.float().cpu().numpy()):
                raise RuntimeError("NPUGraph replay differs from the validated eager output")
        print(f"[PROFILE] {args.iters} rounds; exporting raw kernel timings", flush=True)
        report["samples"] = collect_profile(measured_call, torch, torch_npu, args.iters, directory / "profiler")
    report["summary"] = summarize(report["samples"])
    report["status"] = "complete"
    write_json(directory / "results.json", report)
    summary = report["summary"]
    lines = [
        f"[GMM CHAIN] EP{args.ep_size} BS{args.bs} logical_rank={args.rank} device={args.device} mode={args.mode}",
        f"Routed M={len(routed)}, local experts={len(counts)}, counts={counts.tolist()}",
        f"Accuracy: {report['accuracy']['policy']}; passed={report['accuracy']['passed']}",
        f"[PERF SUMMARY] tail {summary['rounds']}/{summary['recorded_rounds']} rounds; microseconds",
        *[
            f"{metric}: mean={summary[metric]['mean']:.3f}, median={summary[metric]['median']:.3f}, "
            f"min={summary[metric]['min']:.3f}, max={summary[metric]['max']:.3f}"
            for metric in METRICS
        ],
        "kernel_sum_us adds the two kernel durations; chain_span_us also includes their intervening gap.",
        "Single NPU compute only: no Dispatch, Combine, route weights, input quantization or weight preparation.",
        "Logical EP shape reproduction, not a concurrent all-rank maximum or a measured end-to-end MoE time.",
        f"Full results and raw profiler: {directory}",
    ]
    summary_text = "\n".join(lines) + "\n"
    (directory / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text, end="", flush=True)


def case_command(args, output):
    command = [sys.executable, "-I", str(Path(__file__).resolve())]
    for name in (
        "device",
        "rank",
        "ep_size",
        "bs",
        "hidden",
        "intermediate",
        "experts",
        "topk",
        "seed",
        "routing",
        "warmup",
        "iters",
        "mode",
        "check_rows_per_expert",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    return command + ["--output", str(output)]


def execute_case(command, log_path):
    """Use a fresh process per shape to retire profiler/graph state and NPU memory."""
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                return process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise


def write_sweep_summary(directory, entries):
    columns = [
        "ep_size",
        "source_bs",
        "logical_rank",
        "device",
        "mode",
        "status",
        "local_experts",
        "routed_rows",
        "accuracy_passed",
        "accuracy_policy",
        "recorded_rounds",
        "rounds",
    ]
    timings = [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "median", "min", "max")]
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns + timings + ["results_file", "log_file", "error"])
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: entry.get(key, "") for key in writer.fieldnames})
    write_json(directory / "summary.json", dict(scope="single_npu_logical_expert_owner", cases=entries))
    lines = [
        "[GMM CHAIN SWEEP] Single NPU; logical EP shapes; mean kernel times in us (tail half).",
        "EP   BS    M     Experts Status      GMM1+SwiGLU   GMM2       Sum        Span       Gap",
    ]
    for entry in entries:
        values = [f"{entry[f'{metric}_mean']:.3f}" if f"{metric}_mean" in entry else "-" for metric in METRICS]
        lines.append(
            f"{entry['ep_size']:<4} {entry['source_bs']:<5} {entry.get('routed_rows', '-'):<5} "
            f"{entry.get('local_experts', '-'):<7} {entry['status']:<11} "
            + " ".join(f"{value:<10}" for value in values)
        )
    lines.extend(
        [
            "GMM1 includes SwiGLU and MX quantization; Span includes the gap between the two kernels.",
            "No Dispatch/Combine; EP does not request that many physical devices.",
            f"Full per-case status, validation, logs and timings: {directory / 'summary.json'}",
        ]
    )
    (directory / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args):
    cases = sweep_cases(args)
    if args.dry_run:
        plans = []
        for case in cases:
            config = case_config(case)
            _, counts, _ = routed_input(config, case.ep_size, case.rank, case.bs, case.routing)
            plans.append(dict(config=config, plan=describe(case, counts)))
        print(json.dumps(dict(case_count=len(plans), cases=plans), indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="sweep_", dir=args.output)).resolve()
    entries = [
        dict(
            ep_size=case.ep_size,
            source_bs=case.bs,
            logical_rank=case.rank,
            device=case.device,
            mode=case.mode,
            local_experts=case.experts // case.ep_size,
            status="pending",
            log_file=str(directory / f"ep{case.ep_size}_bs{case.bs}" / "console.log"),
        )
        for case in cases
    ]
    write_json(
        directory / "manifest.json",
        dict(
            arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            case_count=len(cases),
            scope="single_npu_logical_expert_owner",
        ),
    )
    write_sweep_summary(directory, entries)
    print(f"[SWEEP] {len(cases)} cases; output={directory}", flush=True)
    for index, (case, entry) in enumerate(zip(cases, entries), start=1):
        case_dir = Path(entry["log_file"]).parent
        case_dir.mkdir()
        entry.update(status="running", command=case_command(case, case_dir))
        write_sweep_summary(directory, entries)
        print(f"[CASE {index}/{len(cases)}] EP{case.ep_size} BS{case.bs}; log={entry['log_file']}", flush=True)
        try:
            code = execute_case(entry["command"], Path(entry["log_file"]))
            entry["returncode"] = code
            reports = list(case_dir.glob("rank_*/results.json"))
            if len(reports) != 1:
                raise ValueError(f"Expected one case report, found {len(reports)}; process exit={code}")
            entry["results_file"] = str(reports[0])
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            if code != 0 or report.get("status") != "complete":
                raise ValueError(f"Case failed (exit={code}); inspect {reports[0]} and console.log")
            plan, summary = report["plan"], report["summary"]
            if (plan["ep_size"], plan["source_bs"], plan["logical_rank"], plan["mode"]) != (
                case.ep_size,
                case.bs,
                case.rank,
                case.mode,
            ) or summary["recorded_rounds"] != case.iters:
                raise ValueError("Case report does not match the requested scenario/iterations")
            accuracy = report["accuracy"]["passed"]
            if accuracy is False:
                raise ValueError("Case numerical validation failed")
            timings = {
                f"{metric}_{stat}": summary[metric][stat]
                for metric in METRICS
                for stat in ("mean", "median", "min", "max")
            }
            if any(not math.isfinite(value) or value < 0 for value in timings.values()):
                raise ValueError("Case report contains invalid timings")
            entry.update(
                status="complete",
                routed_rows=plan["routed_rows"],
                expert_counts=plan["expert_counts"],
                accuracy_passed=accuracy,
                accuracy_policy=report["accuracy"]["policy"],
                recorded_rounds=summary["recorded_rounds"],
                rounds=summary["rounds"],
                **timings,
            )
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(f"[FAILED] EP{case.ep_size} BS{case.bs}: {error}", flush=True)
        except KeyboardInterrupt:
            entry.update(status="interrupted", error="User interrupted the sweep")
            raise
        finally:
            write_sweep_summary(directory, entries)
    print((directory / "summary.txt").read_text(encoding="utf-8"), end="", flush=True)
    failures = sum(entry["status"] != "complete" for entry in entries)
    print(
        f"[{'FAILED' if failures else 'DONE'}] {len(cases) - failures}/{len(cases)} cases completed: {directory}",
        flush=True,
    )
    return 1 if failures else 0


def main(argv=None):
    args = arguments(argv)
    if args.ep_sizes is not None or args.bs_values is not None:
        return run_sweep(args)
    config = case_config(args)
    routed, counts, coordinates = routed_input(config, args.ep_size, args.rank, args.bs, args.routing)
    plan = describe(args, counts)
    if args.dry_run:
        print(json.dumps(dict(config=config, plan=plan), indent=2))
        return
    if len(routed) == 0:
        raise ValueError("Selected logical rank has no routed rows; choose a rank with work")
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"rank_{args.rank}_", dir=args.output)).resolve()
    report = dict(
        status="started",
        config=config,
        plan=plan,
        routing=args.routing,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    write_json(directory / "results.json", report)
    print(
        f"[GMM CHAIN] logical rank {args.rank}: M={len(routed)}, experts={len(counts)}, counts={counts.tolist()}",
        flush=True,
    )
    print(f"[OUTPUT] {directory}", flush=True)
    try:
        run(args, config, routed, counts, coordinates, directory, report)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        write_json(directory / "results.json", report)
        raise


if __name__ == "__main__":
    sys.exit(main())
