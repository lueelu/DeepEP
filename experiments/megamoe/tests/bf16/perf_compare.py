# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""NPU Event measurements for the BF16 training forward, backward and full call."""

import json
import time
import statistics


PEAK_TFLOPS = 432.0  # source comment: Ascend950 BF16 cube peak
ROOFLINE_TFLOPS = round(PEAK_TFLOPS * 0.95, 1)  # 410.4: the source's derated roofline
FORWARD_RFK = 6.0  # GMM1 (4) + GMM2 (2) per routed row
SAVED_MODE_RFK = {"none": 16.0, "bf16": 12.0, "mxfp8": 12.0}
BASELINE_BUDGET_BYTES = 16 << 30  # packed-route working set the baseline may allocate


def event_stats(samples):
    """Keep paired rounds and use an inclusive, linearly interpolated P95."""
    return {
        "samples": list(samples),
        "median": statistics.median(samples),
        "p95": statistics.quantiles(samples, n=100, method="inclusive")[94] if len(samples) > 1 else samples[0],
        "mean": statistics.mean(samples),
        "min": min(samples),
        "max": max(samples),
    }


def bench_scope(body, prepare, args, host_samples=None):
    """warmup+loops Event windows, synchronized before and after every window.

    The source synchronizes each iteration and rebuilds a forward graph for
    every backward sample, so queued SHMEM calls never share one workspace.
    """
    import torch

    samples = []
    for iteration in range(args.warmup + args.loops):
        torch.npu.synchronize()
        state = prepare() if prepare is not None else None
        if prepare is not None:
            torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        if host_samples is not None:
            started = time.perf_counter_ns()
        body(state)
        if host_samples is not None and iteration >= args.warmup:
            host_samples.append((started, time.perf_counter_ns()))
        end.record()
        torch.npu.synchronize()
        if iteration >= args.warmup:
            samples.append(float(start.elapsed_time(end)))
    return samples


def exchanged_rows(group, value, receive_splits, send_splits):
    """Blocking all_to_all_single with Megatron's split orientation.

    The source keeps an asynchronous handle only to overlap a shared expert this
    entry does not have; the blocking form moves the same bytes.
    """
    import torch.distributed as dist

    if dist.get_world_size(group) == 1:
        return value
    value = value.contiguous()
    output = value.new_empty([sum(receive_splits), *value.shape[1:]])
    dist.all_to_all_single(output, value, output_split_sizes=receive_splits, input_split_sizes=send_splits, group=group)
    return output


def reorder_chunks(value, sizes, order, local_experts=None):
    """Megatron sort_chunks_by_idxs; zero-row chunks keep empty experts valid.

    The source skips permute2/unpermute1 entirely with one local expert, where
    the reorder is an identity copy; keep that so neither side pays for it.
    """
    import torch

    if local_experts == 1:
        return value
    chunks = torch.split(value, sizes, dim=0)
    return torch.cat([chunks[index] for index in order], dim=0)


def clip_gate_up(gate, up, limit):
    """Source clip semantics: Gate one-sided, Up two-sided, disabled for limit<=0."""
    import torch

    if not 0 < limit < torch.finfo(torch.bfloat16).max:
        return gate, up
    return gate.clamp_max(limit), up.clamp(-limit, limit)


class BaselinePlan:
    """Row order, exchange splits, expert counts and weight views of one case."""

    __slots__ = (
        "group",
        "tokens",
        "topk",
        "hidden",
        "local_experts",
        "swiglu_limit",
        "order",
        "route_row",
        "counts",
        "input_splits",
        "output_splits",
        "received_sizes",
        "return_sizes",
        "sort_index",
        "restore_index",
        "weight1",
        "weight2",
        "recompute",
    )


def baseline_plan(ids, w1, w2, group, args, rank, world, mode):
    """Derive the baseline chain metadata from the shared routing ids.

    Returns ``(plan, None)`` or ``(None, reason)``, so a shape the baseline
    cannot afford prints why it is absent instead of failing the case. Every
    value here is built once, outside all Event windows.
    """
    import torch
    import torch.distributed as dist

    local_experts = args.expert_per_rank
    tokens, topk = (int(size) for size in ids.shape)
    routes = tokens * topk
    working_set = routes * (args.hidden + args.ffn) * 2 * 6
    if working_set > BASELINE_BUDGET_BYTES:
        return None, (
            f"packed routes need ~{working_set / 2**30:.1f} GiB, above the "
            f"{BASELINE_BUDGET_BYTES >> 30} GiB baseline budget"
        )
    flat = ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    experts = world * local_experts
    gathered = [None] * world
    dist.all_gather_object(gathered, torch.bincount(flat, minlength=experts).tolist())
    # Rows rank `source` produces for this rank, split by local expert.
    sent_to_us = torch.tensor(gathered, dtype=torch.int64)[:, rank * local_experts : (rank + 1) * local_experts]
    order = torch.argsort(flat, stable=True)
    plan = BaselinePlan()
    plan.group = group
    plan.tokens, plan.topk, plan.hidden = tokens, topk, args.hidden
    plan.local_experts, plan.swiglu_limit, plan.recompute = local_experts, args.swiglu_limit, mode == "none"
    plan.order = order.to(ids.device)
    plan.route_row = (order // topk).to(ids.device)
    plan.counts = sent_to_us.sum(dim=0).to(ids.device)
    plan.input_splits = torch.bincount(flat // local_experts, minlength=world).tolist()
    plan.output_splits = sent_to_us.sum(dim=1).tolist()
    plan.received_sizes = sent_to_us.reshape(-1).tolist()
    plan.return_sizes = sent_to_us.T.reshape(-1).tolist()
    plan.sort_index = torch.arange(experts).reshape(world, local_experts).T.reshape(-1).tolist()
    plan.restore_index = torch.arange(experts).reshape(local_experts, world).T.reshape(-1).tolist()
    # Megatron stores [H, 2F*E] and [F*E, H]; the operator takes per-expert
    # tensors, so read both once here rather than copying inside a window.
    plan.weight1 = w1.detach().transpose(1, 2).contiguous()
    plan.weight2 = w2.detach().transpose(1, 2).contiguous()
    if sum(plan.received_sizes) != sum(plan.output_splits) or int(plan.counts.sum()) != sum(plan.received_sizes):
        return None, "inconsistent dispatch splits"
    return plan, None


_BASELINE_CHAIN = None
_HCCL = {}


def device_group():
    """One reusable HCCL subgroup for the baseline; the default group is Gloo.

    MegaMoE BF16 communicates inside its own runtime, so the baseline
    chain needs a Torch-level device collective. Creation is cached: a suite
    must not accumulate one communicator per case.
    """
    import torch.distributed as dist

    if "group" not in _HCCL:
        try:
            _HCCL["group"] = dist.new_group(ranks=list(range(dist.get_world_size())), backend="hccl")
            _HCCL["reason"] = None
        except Exception as error:  # capability probe; the reason is reported, not hidden
            _HCCL["group"] = None
            _HCCL["reason"] = f"no HCCL subgroup for all_to_all ({type(error).__name__}: {error})"
    return _HCCL["group"], _HCCL["reason"]


def baseline_chain():
    """Build the baseline autograd Function once; keeps module import Torch-free."""
    global _BASELINE_CHAIN
    if _BASELINE_CHAIN is not None:
        return _BASELINE_CHAIN

    import torch

    class BaselineChain(torch.autograd.Function):
        """The source's dropless small-operator chain with its hand-written backward.

        permute1 -> dispatch -> permute2 -> GMM1 -> SwiGLU*probs -> GMM2 ->
        unpermute1 -> combine -> unpermute2. ``recompute`` mirrors the source's
        coupled regime: MegaMoE BF16 saves Gate/Up for bf16/mxfp8 and
        recomputes GMM1 for none, so the baseline performs the same GEMM work
        (12 versus 16 RFK).
        """

        @staticmethod
        def forward(ctx, x, topk_weights, w1, w2, plan):
            from npu_baseline import _grouped_matmul

            counts = plan.counts
            global_x = reorder_chunks(
                exchanged_rows(plan.group, x.index_select(0, plan.route_row), plan.output_splits, plan.input_splits),
                plan.received_sizes,
                plan.sort_index,
                plan.local_experts,
            )
            global_p = reorder_chunks(
                exchanged_rows(
                    plan.group,
                    topk_weights.reshape(-1).index_select(0, plan.order),
                    plan.output_splits,
                    plan.input_splits,
                ),
                plan.received_sizes,
                plan.sort_index,
                plan.local_experts,
            )
            gate_up = _grouped_matmul(global_x, plan.weight1, counts)
            gate, up = clip_gate_up(*torch.chunk(gate_up, 2, dim=-1), plan.swiglu_limit)
            activation = torch.nn.functional.silu(gate) * up
            if global_p.dtype == torch.float32:
                # FP32 p must not promote the following BF16 GMM operands.
                intermediate = (activation.float() * global_p.unsqueeze(-1)).to(activation.dtype)
            else:
                intermediate = activation * global_p.unsqueeze(-1)
            packed_y = _grouped_matmul(intermediate, plan.weight2, counts)
            combined = exchanged_rows(
                plan.group,
                reorder_chunks(packed_y, plan.return_sizes, plan.restore_index, plan.local_experts),
                plan.input_splits,
                plan.output_splits,
            )
            output = torch.zeros((plan.tokens, plan.hidden), dtype=combined.dtype, device=combined.device)
            output.scatter_add_(0, plan.route_row.unsqueeze(1).expand(-1, plan.hidden), combined)
            ctx.plan = plan
            saved = [global_x, global_p] if plan.recompute else [global_x, global_p, gate_up]
            ctx.save_for_backward(*saved)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            from npu_baseline import _grouped_matmul, _grouped_wgrad

            plan = ctx.plan
            counts = plan.counts
            global_x, global_p = ctx.saved_tensors[0], ctx.saved_tensors[1]
            weight1_transposed = plan.weight1.transpose(-1, -2).contiguous()
            weight2_transposed = plan.weight2.transpose(-1, -2).contiguous()
            grad_packed_y = reorder_chunks(
                exchanged_rows(
                    plan.group,
                    grad_output.index_select(0, plan.route_row),
                    plan.output_splits,
                    plan.input_splits,
                ),
                plan.received_sizes,
                plan.sort_index,
                plan.local_experts,
            )
            gate_up = _grouped_matmul(global_x, plan.weight1, counts) if plan.recompute else ctx.saved_tensors[2]
            gate, up = clip_gate_up(*torch.chunk(gate_up, 2, dim=-1), plan.swiglu_limit)
            activation = torch.nn.functional.silu(gate) * up
            if global_p.dtype == torch.float32:
                intermediate = (activation.float() * global_p.unsqueeze(-1)).to(activation.dtype)
            else:
                intermediate = activation * global_p.unsqueeze(-1)
            grad_intermediate = _grouped_matmul(grad_packed_y, weight2_transposed, counts)
            grad_w2 = _grouped_wgrad(intermediate, grad_packed_y, counts)
            grad_w2 = grad_w2.reshape(plan.weight2.shape).transpose(1, 2).contiguous()
            grad_activation = grad_intermediate * global_p.unsqueeze(-1)
            if global_p.dtype == torch.float32:
                grad_p = (grad_intermediate.float() * activation.float()).sum(dim=-1)
            else:
                grad_p = (grad_intermediate * activation).sum(dim=-1)
            # The source differentiates the same PyTorch graph it ran in forward
            # instead of substituting a different activation backward.
            with torch.enable_grad():
                replay = gate_up.detach().requires_grad_(True)
                replay_gate, replay_up = clip_gate_up(*torch.chunk(replay, 2, dim=-1), plan.swiglu_limit)
                (grad_gate_up,) = torch.autograd.grad(
                    torch.nn.functional.silu(replay_gate) * replay_up, replay, grad_activation
                )
            grad_x = _grouped_matmul(grad_gate_up, weight1_transposed, counts)
            grad_w1 = _grouped_wgrad(global_x, grad_gate_up, counts)
            grad_w1 = grad_w1.reshape(plan.weight1.shape).transpose(1, 2).contiguous()
            grad_x = reorder_chunks(grad_x, plan.return_sizes, plan.restore_index, plan.local_experts)
            grad_p = reorder_chunks(grad_p, plan.return_sizes, plan.restore_index, plan.local_experts)
            grad_x = exchanged_rows(plan.group, grad_x, plan.input_splits, plan.output_splits)
            grad_p = exchanged_rows(plan.group, grad_p, plan.input_splits, plan.output_splits)
            grad_input = torch.zeros((plan.tokens, plan.hidden), dtype=grad_x.dtype, device=grad_x.device)
            grad_input.scatter_add_(0, plan.route_row.unsqueeze(1).expand(-1, plan.hidden), grad_x)
            grad_weights = torch.zeros(plan.tokens * plan.topk, dtype=grad_p.dtype, device=grad_p.device)
            grad_weights.index_copy_(0, plan.order, grad_p)
            return grad_input, grad_weights.reshape(plan.tokens, plan.topk), grad_w1, grad_w2, None

    _BASELINE_CHAIN = BaselineChain
    return _BASELINE_CHAIN


def fused_forward(op, x, ids, weights, w1, w2, capacity, *, save_gmm1_output, swiglu_limit, api, collect_counts=True):
    """Test adapter for the unified in-place interface."""
    import torch
    from deep_ep_experimental.megamoe import bf16_mega_moe

    op.x, op.topk_idx, op.topk_weights = x, ids, weights
    op.gmm1_save_type = save_gmm1_output
    op.max_recv_tokens = capacity
    y = torch.empty_like(x, requires_grad=False)
    counts = torch.zeros(w1.shape[0], dtype=torch.int32, device=x.device) if collect_counts else None
    bf16_mega_moe(y, w1, w2, op, cumulative_local_expert_recv_stats=counts, activation_clamp=swiglu_limit)
    return y, counts


def profile_training_forward(forward, args, path):
    import torch
    import torch.distributed as dist
    import torch_npu.profiler as profiler

    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]) as prof:
        dist.barrier()
        for _ in range(args.warmup + args.loops):
            forward()
            torch.npu.synchronize()
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_bytes())
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    kernels = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and "mega_moe_bf16_forward" in e.get("name", "")
            and e.get("args", {}).get("Task Type", "").startswith("KERNEL")
        ),
        key=lambda e: float(e["ts"]),
    )
    if len(kernels) != args.warmup + args.loops:
        raise ValueError(f"{path}: expected {args.warmup + args.loops} forward kernels, found {len(kernels)}")
    return event_stats([float(e["dur"]) / 1000 for e in kernels[args.warmup :]])


def measure(op, tensors, capacity, args, mode, output):
    """Three Event scopes for the MegaMoE BF16 operator, optional small-operator baseline.

    Ported protocol: identical warmup/loop counts for both sides, one window per
    sample, a rebuilt forward graph for every backward sample, and the coupled
    backward regime. The reference is informational; it never decides the case.
    """
    import torch
    import torch.distributed as dist

    x, ids, p, w1, w2, dy = tensors
    leaves = (x, p, w1, w2)
    rank = dist.get_rank()

    def stage(name):
        """Progress markers: a silent hang in this phase is otherwise invisible."""
        if rank == 0 and args.log != "quiet":
            print(f"[PERF] stage {name}", flush=True)

    def reset():
        for leaf in leaves:
            leaf.grad = None

    def megamoe_forward(collect_counts=False):
        reset()
        y, counts = fused_forward(
            op,
            x,
            ids,
            p,
            w1,
            w2,
            capacity,
            save_gmm1_output=args.save_gmm1_output,
            swiglu_limit=args.swiglu_limit,
            api=args.api,
            collect_counts=collect_counts,
        )
        return y, counts

    def megamoe_complete(collect_counts=False):
        y, counts = megamoe_forward(collect_counts)
        y.backward(dy)
        return y, counts

    stage(f"scopes start warmup={args.warmup} loops={args.loops}")
    megamoe = {}
    host_forward = []
    for name, body, prepare in (
        ("forward", lambda state: megamoe_forward(), None),
        ("backward", lambda state: state[0].backward(dy), megamoe_forward),
        ("complete", lambda state: megamoe_complete(), None),
    ):
        megamoe[f"{name}_ms"] = event_stats(
            bench_scope(body, prepare, args, host_forward if name == "forward" else None)
        )
        stage(f"{name} scope done")
    y, counts = megamoe_complete(collect_counts=True)
    if not all(bool(torch.isfinite(value).all().item()) for value in (y, x.grad, p.grad, w1.grad, w2.grad)):
        raise RuntimeError("Non-finite output or gradient after timing")
    stage("megamoe_bf16 valid")
    if args.profile_forward_kernel:
        megamoe["forward_kernel_ms"] = profile_training_forward(
            megamoe_forward, args, output / f"rank_{rank}_forward_profiler.json"
        )
    report = {
        "operator": "megamoe_bf16",
        "forward_host_monotonic_ns": host_forward,
        "warmup": args.warmup,
        "loops": args.loops,
        "scope": (
            "Synchronized NPU Event latency of forward, autograd backward with the graph built outside the "
            "window, and forward+backward; reference, reporting and loss construction excluded; no device Timer"
        ),
        "baseline": {"status": "OFF", "reason": "--baseline off"},
        **megamoe,
    }
    route_flops = args.seq * args.topk * args.ffn * args.hidden  # FLOPs per 1 RFK
    backward_rfk = SAVED_MODE_RFK.get(mode, 12.0)
    scoped_rfk = {
        "forward": FORWARD_RFK,
        "backward": backward_rfk,
        "complete": FORWARD_RFK + backward_rfk,
    }

    def utilization(scopes_ms):
        """GMM-only achieved TFLOPS and its share of the BF16 cube peak.

        Non-GEMM work (all-to-all, permute, activation) is inside the window, so
        this is the operator's GMM utilization: achieved / peak, not a claim
        about the end-to-end model.
        """
        result = {}
        for name, rfk_value in scoped_rfk.items():
            latency = scopes_ms[f"{name}_ms"]["median"]
            flops = rfk_value * route_flops
            throughput = flops / (latency * 1e-3) / 1e12 if latency > 0 else float("nan")
            result[name] = {
                "rfk": rfk_value,
                "flops": flops,
                "tflops": throughput,
                "mfu": 100.0 * throughput / PEAK_TFLOPS,
            }
        return result

    megamoe_utilization = utilization(megamoe)
    if args.baseline == "on":
        stage("baseline plan start")
        group, reason = device_group()
        plan = None
        if group is not None:
            plan, reason = baseline_plan(ids, w1, w2, group, args, rank, dist.get_world_size(), mode)
        if plan is None:
            report["baseline"] = {"status": "SKIPPED", "reason": reason}
            stage(f"baseline skipped: {reason}")
        else:
            stage("baseline plan ready")
            chain = baseline_chain()

            def reference_forward():
                reset()
                return chain.apply(x, p, w1, w2, plan)

            def reference_complete():
                out = reference_forward()
                out.backward(dy)
                return out

            reference = {}
            for name, body, prepare in (
                ("forward", lambda state: reference_forward(), None),
                ("backward", lambda state: state.backward(dy), reference_forward),
                ("complete", lambda state: reference_complete(), None),
            ):
                reference[f"{name}_ms"] = event_stats(bench_scope(body, prepare, args))
                stage(f"baseline {name} scope done")
            out = reference_complete()
            if all(bool(torch.isfinite(value).all().item()) for value in (out, x.grad, p.grad, w1.grad, w2.grad)):
                speedup = {
                    scope: reference[f"{scope}_ms"]["median"] / max(megamoe[f"{scope}_ms"]["median"], 1e-9)
                    for scope in ("forward", "backward", "complete")
                }
                reference_utilization = utilization(reference)
                report["baseline"] = {
                    "status": "RUN",
                    "recompute_gmm1": plan.recompute,
                    "speedup": speedup,
                    "mfu": {name: value["mfu"] for name, value in reference_utilization.items()},
                    **reference,
                }
            else:
                report["baseline"] = {
                    "status": "SKIPPED",
                    "reason": "baseline produced non-finite values; its samples are not comparable",
                }
    report["mfu"] = {
        "peak_tflops": PEAK_TFLOPS,
        "peak_note": (
            "432 TFLOPS is the BF16 cube peak inherited from the source harness comment; confirm it against the "
            "official Ascend950 dense-BF16 specification before quoting an absolute utilization"
        ),
        "scope": (
            "GMM-only FLOPs (RFK model) divided by the windowed latency, as a share of the BF16 cube peak; "
            "all-to-all, permute and activation time stays inside the window, so this is not a model-level MFU"
        ),
        **megamoe_utilization,
    }
    # The source printed a backward-only roofline. Keep its derated denominator,
    # but report every scope so the three numbers stay comparable.
    roofline = {}
    for name, value in megamoe_utilization.items():
        measured_ms = megamoe[f"{name}_ms"]["median"]
        floor_ms = value["flops"] / (ROOFLINE_TFLOPS * 1e12) * 1e3
        roofline[name] = {
            "rfk": value["rfk"],
            "flops": value["flops"],
            "measured_ms": measured_ms,
            "floor_ms": floor_ms,
            "overhead_ms": measured_ms - floor_ms,
            "tflops": value["tflops"],
        }
    report["roofline"] = {"mode": mode, "roofline_tflops": ROOFLINE_TFLOPS, **roofline}
    return report, counts


def summarize_event_rounds(reports):
    """Mean of the slowest EP rank per selected round, using the W4A8 tail-half policy."""
    from deep_ep_experimental.megamoe.performance import summarize_rounds

    runs = [r["performance"] for r in reports]
    groups = {"megamoe": runs}
    if all(run["baseline"].get("status") == "RUN" for run in runs):
        groups["bigop"] = [run["baseline"] for run in runs]
    summary = {}
    for name, group in groups.items():
        summary[name] = {}
        for scope in ("forward", "backward", "complete"):
            samples_us = [[ms * 1000 for ms in run[f"{scope}_ms"]["samples"]] for run in group]
            rounds = summarize_rounds(samples_us)
            summary[name][scope] = rounds["rank_max_mean_us"] / 1000
    return summary


def performance_summary(reports):
    """One terminal block: a line per scope with both sides, speedup and MFU.

    Workers only collect data; rank 0 presents it here, after the JSON exists.
    """
    first = reports[0]
    perf = first["performance"]
    baseline = perf.get("baseline", {})
    speeds = baseline.get("speedup") or {}
    base_mfu = baseline.get("mfu") or {}
    roofline = perf.get("roofline", {})
    scopes = ("forward", "backward", "complete")
    rfk = "/".join(f"{roofline[scope]['rfk']:.0f}" for scope in scopes if scope in roofline)
    lines = [
        f"[PERF] rank=0 mode={roofline.get('mode', '?')} warmup={perf['warmup']} loops={perf['loops']} "
        f"({rfk} RFK, GMM peak {PEAK_TFLOPS:.0f} TF)"
    ]
    for scope in scopes:
        row = f"[PERF] {scope:<9} megamoe_bf16={perf[f'{scope}_ms']['median']:8.3f}ms"
        if scope in speeds:
            row += f" baseline={baseline[f'{scope}_ms']['median']:8.3f}ms speedup={speeds[scope]:5.2f}x"
        else:
            row += " baseline=        --"
        mfu = perf.get("mfu", {}).get(scope, {}).get("mfu")
        if mfu is not None:
            row += f" MFU={mfu:5.1f}%" + (f"/{base_mfu[scope]:.1f}%" if scope in base_mfu else "")
        lines.append(row)
    if roofline:
        floor = "/".join(f"{roofline[scope]['floor_ms']:.2f}" for scope in scopes)
        overhead = "/".join(f"{roofline[scope]['overhead_ms']:+.2f}" for scope in scopes)
        lines.append(
            f"[PERF] roofline floor={floor}ms overhead={overhead}ms @ {roofline['roofline_tflops']} TF (fwd/bwd/fb)"
        )
    if baseline.get("status") != "RUN":
        lines.append(f"[PERF] baseline {str(baseline.get('status', '?')).lower()}: {baseline.get('reason', '')}")
    for report in reports[1:]:
        other = report["performance"]
        other_speed = other.get("baseline", {}).get("speedup") or {}
        row = (
            f"[PERF] rank={report['rank']} "
            f"forward={other['forward_ms']['median']:.3f}ms "
            f"backward={other['backward_ms']['median']:.3f}ms "
            f"complete={other['complete_ms']['median']:.3f}ms"
        )
        if other_speed:
            row += f" speedup={other_speed['forward']:.2f}/{other_speed['backward']:.2f}/{other_speed['complete']:.2f}x"
        lines.append(row)
    return lines


def group_performance(reports):
    """Combine corresponding rounds before computing percentiles."""
    per_rank = [record["performance"] for record in reports]
    summary = {}
    for scope in ("forward", "backward", "complete"):
        key = f"{scope}_ms"
        if key not in per_rank[0]:
            continue
        rounds = zip(*(perf[key]["samples"] for perf in per_rank), strict=True)
        summary[key] = event_stats([max(values) for values in rounds])
        if all(perf.get("baseline", {}).get("status") == "RUN" for perf in per_rank):
            rounds = zip(*(perf["baseline"][key]["samples"] for perf in per_rank), strict=True)
            baseline = event_stats([max(values) for values in rounds])
            summary.setdefault("baseline", {})[key] = baseline
            summary["baseline"].setdefault("speedup", {})[scope] = baseline["median"] / summary[key]["median"]
        if scope in per_rank[0].get("mfu", {}):
            flops = per_rank[0]["mfu"][scope]["flops"]
            summary.setdefault("mfu", {})[scope] = 100 * flops / (PEAK_TFLOPS * 1e9 * summary[key]["median"])
    if "batch_ms_per_call" in per_rank[0]:
        summary["batch_ms_per_call"] = max(p["batch_ms_per_call"] for p in per_rank)
        summary["kernel_rank_min_mean_ms"] = statistics.mean(
            min(values) for values in zip(*(p["forward_ms"]["samples"] for p in per_rank), strict=True)
        )
    if "forward_kernel_ms" in per_rank[0]:
        rounds = zip(*(p["forward_kernel_ms"]["samples"] for p in per_rank), strict=True)
        summary["forward_kernel_ms"] = event_stats([max(values) for values in rounds])
    return summary
