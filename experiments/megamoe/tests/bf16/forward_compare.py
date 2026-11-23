"""Same-input BF16 forward and backward comparison of MegaMoE, BigOp and DeepEP MC."""

import csv
from contextlib import ExitStack
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from perf_compare import baseline_plan, bench_scope, device_group, exchanged_rows, fused_forward, reorder_chunks


BACKENDS = ("megamoe", "bigop", "mc")


def kernel_rounds(directory, iterations, paired_steps=False):
    """Union of device task intervals per profiler step; includes compute and communication."""
    from deep_ep._timer import merge_intervals

    steps = {}
    devices = set()
    for path in Path(directory).rglob("kernel_details*.csv"):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                step = int(row["Step Id"])
                start, duration = Decimal(row["Start Time(us)"].strip()), Decimal(row["Duration(us)"].strip())
                if not start.is_finite() or not duration.is_finite() or duration < 0:
                    raise ValueError(f"Invalid kernel timing in {path}: {start}, {duration}")
                if duration:
                    steps.setdefault(step, []).append((start, start + duration))
                    devices.add(row["Device_id"])
    expected = iterations * (2 if paired_steps else 1)
    if len(devices) != 1 or len(steps) != expected or max(steps) - min(steps) + 1 != expected:
        raise ValueError(
            f"Expected {expected} consecutive device steps on one rank under {directory}, got {sorted(steps)}"
        )
    selected = sorted(steps)[1::2] if paired_steps else sorted(steps)
    return [float(sum(end - start for start, end in merge_intervals(steps[i]))) / 1000 for i in selected]


def profile_scope(body, prepare, directory, loops):
    import torch
    import torch.distributed as dist
    from torch_npu import profiler

    config = profiler._ExperimentalConfig(
        export_type=profiler.ExportType.Text, profiler_level=profiler.ProfilerLevel.Level1
    )
    dist.barrier()
    torch.npu.synchronize()
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=loops * (2 if prepare else 1), repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=config,
    ) as prof:
        for _ in range(loops):
            state = prepare() if prepare else None
            if prepare:
                torch.npu.synchronize()
                prof.step()  # Close the graph preparation step before backward.
            body(state)
            del state
            torch.npu.synchronize()
            prof.step()
    return kernel_rounds(directory, loops, paired_steps=prepare is not None)


def reference_calls(backend, tensors, args, rank, world, resources):
    import torch
    import torch.distributed as dist

    from npu_baseline import _grouped_matmul, _grouped_wgrad, _swiglu_backward_fp32, _swiglu_fp32

    x, ids, p, w1, w2 = tensors
    group, reason = device_group()
    if group is None:
        raise RuntimeError(reason)
    plan, reason = baseline_plan(ids, w1, w2, group, args, rank, world, args.save_gmm1_output)
    if plan is None:
        raise RuntimeError(reason)
    inverse = torch.argsort(plan.order)

    def dispatch(values):
        rows = exchanged_rows(group, values, plan.output_splits, plan.input_splits)
        return reorder_chunks(rows, plan.received_sizes, plan.sort_index, plan.local_experts)

    def return_rows(rows):
        packed = reorder_chunks(rows, plan.return_sizes, plan.restore_index, plan.local_experts)
        return exchanged_rows(group, packed, plan.input_splits, plan.output_splits)

    def combine(routes):
        routes = routes.index_select(0, inverse).reshape(plan.tokens, plan.topk, plan.hidden)
        value = torch.zeros_like(x, dtype=torch.float32)
        for slot in range(plan.topk):
            value.add_(routes[:, slot])
        return value.to(x.dtype)

    if backend == "mc":
        from deep_ep_experimental.ep_gmm_fused import alltoallv_gmm, gmm_alltoallv, udma_group

        resources.enter_context(udma_group(group=dist.group.WORLD, local_rank=rank))
        opts = dict(ep_size=world, expert_num=world * plan.local_experts)
        local_counts = torch.bincount(ids.flatten().long(), minlength=opts["expert_num"]).to(torch.int64)
        received_counts = torch.tensor(plan.received_sizes, dtype=torch.int64, device=x.device)
        received = sum(plan.received_sizes)

        def dispatch_gmm(packed, weight, output, exported):
            result = alltoallv_gmm(
                packed,
                weight,
                local_counts,
                received_counts,
                received_rows=received,
                out=output,
                output_all_to_allv=exported is not None,
                all_to_allv_out=exported,
                **opts,
            )
            return result if exported is not None else (result, None)

        def gmm_return(values, weight, output):
            return gmm_alltoallv(values, weight, local_counts, received_counts, out=output, **opts)

        gmm1_out = x.new_empty((received, 2 * args.ffn))
        routes_out = x.new_empty((ids.numel(), args.hidden))
        if args.compare_backward:
            saved_x = x.new_empty((received, args.hidden))
            dhp_out = x.new_empty((received, args.ffn))
            do_out = x.new_empty((received, args.hidden))
        else:
            saved_x = None
    else:

        def dispatch_gmm(packed, weight, output, exported):
            expert_rows = dispatch(packed)
            return _grouped_matmul(expert_rows, weight, plan.counts), expert_rows

        def gmm_return(values, weight, output):
            return return_rows(_grouped_matmul(values, weight, plan.counts))

        gmm1_out = routes_out = saved_x = dhp_out = do_out = None

    def forward():
        gate_up, expert_x = dispatch_gmm(x.index_select(0, plan.route_row), plan.weight1, gmm1_out, saved_x)
        weights = dispatch(p.reshape(-1).index_select(0, plan.order))
        hp = (_swiglu_fp32(gate_up, args.swiglu_limit) * weights.float().unsqueeze(1)).to(x.dtype)
        y = combine(gmm_return(hp, plan.weight2, routes_out))
        return y, (gate_up, expert_x, weights) if args.compare_backward else None

    def backward(state, dy):
        _, (gate_up, expert_x, weights) = state
        # The input weights already have the contiguous [expert,N,K] orientation.
        dhp, expert_do = dispatch_gmm(dy.index_select(0, plan.route_row), w2, dhp_out, do_out)
        h = _swiglu_fp32(gate_up, args.swiglu_limit)
        hp = (h * weights.float().unsqueeze(1)).to(x.dtype)
        dw2 = _grouped_wgrad(hp, expert_do, plan.counts).reshape_as(plan.weight2).transpose(1, 2).contiguous()
        dp = (dhp.float() * h).sum(-1).to(p.dtype)
        dz = _swiglu_backward_fp32(dhp.float() * weights.float().unsqueeze(1), gate_up, args.swiglu_limit)
        dw1 = _grouped_wgrad(expert_x, dz, plan.counts).reshape_as(plan.weight1).transpose(1, 2).contiguous()
        dx = combine(gmm_return(dz, w1, routes_out))
        dp = return_rows(dp).index_select(0, inverse).reshape_as(p)
        return dx, dp, dw1, dw2

    return forward, backward


def check_training_outputs(rank, backend, reference_dir, outputs):
    import torch
    from precision_metrics import OUTPUTS, mixed_tolerance

    checks = {}
    for name, tensor in zip(OUTPUTS, outputs, strict=True):
        path = Path(reference_dir) / f"{rank}_{name}.pt"
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{backend} rank {rank}: non-finite {name}")
        if backend == "bigop":
            torch.save(value, path)
            continue
        truth = torch.load(path, mmap=True, map_location="cpu", weights_only=True)
        if value.shape != truth.shape or value.dtype != truth.dtype:
            raise ValueError(f"{backend} rank {rank}: {name} shape or dtype differs from BigOp")
        exact = torch.equal(value.view(torch.uint8), truth.view(torch.uint8))
        checks[name] = {"bitwise": exact, "passed": True}
        if not exact:
            checks[name].update(mixed_tolerance(value, truth, value.dtype))
    return checks


def worker(rank, args, backend, directory, reference_dir):
    import torch
    import torch.distributed as dist

    from test_mega_moe import make_buffer, make_inputs, prepare_worker
    from precision_metrics import write_report

    world = len(args.devices)
    buffer_type = prepare_worker(rank, args, rank, world, {})
    os.environ["DEEPEP_SHMEM_ENDPOINT"] = args.shmem_ip_port
    try:
        inputs = make_inputs(args, rank, world, global_tokens=False)
        names = ("x", "ids", "p", "w1", "w2") + (("dy",) if args.compare_backward else ())
        digest = hashlib.sha256()
        for name in names:
            digest.update(memoryview(inputs[name].contiguous().view(torch.uint8).numpy()).cast("B"))
        tensors = [inputs[name].to(f"npu:{rank}").contiguous() for name in names[:5]]
        dy = inputs["dy"].to(f"npu:{rank}") if args.compare_backward else None
        rows = int(inputs["all_counts"].reshape(world, -1).sum(1).max())
        capacity = math.ceil(rows / 128) * 128
        del inputs
        with ExitStack() as resources:
            if backend == "megamoe":
                op = make_buffer(buffer_type, dist.group.WORLD, args, world)
                resources.callback(op.destroy)
                # Exercise the training forward with its requested save policy.
                for t in (tensors[0], *tensors[2:]):
                    t.requires_grad_(True)

                def forward():
                    y, _ = fused_forward(
                        op,
                        *tensors,
                        capacity,
                        save_gmm1_output=args.save_gmm1_output,
                        swiglu_limit=args.swiglu_limit,
                        api=args.api,
                        collect_counts=False,
                    )
                    return y, None

                def backward(state, upstream):
                    return torch.autograd.grad(state[0], (tensors[0], *tensors[2:]), upstream)
            else:
                forward, backward = reference_calls(backend, tensors, args, rank, world, resources)
            state = forward()
            y = state[0]
            if y.shape != tensors[0].shape or y.dtype != torch.bfloat16 or not torch.isfinite(y).all().item():
                raise ValueError(f"{backend}: invalid BF16 forward output on rank {rank}")
            report = {"rank": rank, "input_sha256": digest.hexdigest()}
            if args.compare_backward:
                report["checks_vs_bigop"] = check_training_outputs(
                    rank, backend, reference_dir, (y, *backward(state, dy))
                )
                if not all(check["passed"] for check in report["checks_vs_bigop"].values()):
                    write_report(directory / f"rank_{rank}.json", report)
                    raise RuntimeError(f"{backend} rank {rank}: output mismatch; see {directory}")
            del state, y
            scopes = [("forward", lambda _: forward(), None)]
            if args.compare_backward:
                scopes.append(("backward", lambda prepared: backward(prepared, dy), forward))
            for phase, body, prepare in scopes:
                dist.barrier()
                profile_dir = f"rank_{rank}_{phase}_profile" if args.compare_backward else f"rank_{rank}_profile"
                times = {
                    "event_ms": bench_scope(body, prepare, args),
                    "kernel_ms": profile_scope(body, prepare, directory / profile_dir, args.loops),
                }
                if args.compare_backward:
                    report[phase] = times
                else:
                    report.update(times)
            write_report(directory / f"rank_{rank}.json", report)
    finally:
        dist.destroy_process_group()


def compare_forward(args):
    import torch.multiprocessing as mp
    from deep_ep_experimental.megamoe.performance import summarize_rounds
    from precision_metrics import write_report

    scope = "training" if args.compare_backward else "forward"
    output = args.output or Path("output") / f"{scope}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    digests = None
    timings = {}
    order = ("bigop", "megamoe", "mc") if args.compare_backward else BACKENDS
    with TemporaryDirectory(dir=output, prefix="reference_") as reference_dir:
        for backend in order:
            directory = output / backend
            directory.mkdir()
            mp.spawn(worker, args=(args, backend, directory, reference_dir), nprocs=len(args.devices), join=True)
            ranks = [json.loads((directory / f"rank_{i}.json").read_text()) for i in range(len(args.devices))]
            fingerprints = [r["input_sha256"] for r in ranks]
            if digests is not None and fingerprints != digests:
                raise ValueError(f"{backend}: inputs differ from {order[0]}")
            digests = fingerprints
            timings[backend] = {}
            for phase in ("forward", "backward") if args.compare_backward else ("forward",):
                values = [r[phase] for r in ranks] if args.compare_backward else ranks
                target = timings[backend].setdefault(phase, {}) if args.compare_backward else timings[backend]
                for metric in ("event_ms", "kernel_ms"):
                    us = [[v * 1000 for v in r[metric]] for r in values]
                    target[metric] = summarize_rounds(us)["rank_max_mean_us"] / 1000
            print(f"[{scope.upper()}] {backend} {timings[backend]}", flush=True)
    write_report(
        output / "summary.json",
        {
            "scope": scope,
            "aggregation": "tail-half mean of per-round slowest rank",
            "kernel_metric": "union of device computation and communication task intervals per local profiler step",
            "config": {k: v for k, v in vars(args).items() if k != "output"},
            "rank_input_sha256": digests,
            "timings": timings,
        },
    )
    return 0
