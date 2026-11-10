# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Installed-package accuracy/performance runner. Launch with torchrun, not st/run.sh."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import random
import threading
from importlib.resources import files
import json
import os
from pathlib import Path

from .timer.__main__ import parse_ranks
from .performance import alignment_config, collect_profile, enqueue_batch, summarize_rounds


def load_config(name, bs_values=None):
    path = Path(name)
    if not path.is_file():
        path = files(__package__).joinpath("configs", name + ".json")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "model", "bs_values", "warmup", "iters", "seed"}
    if not required <= set(config) or set(config) - required - {"alignment"} or config["schema_version"] != 1:
        raise ValueError("Invalid config schema; see the packaged smoke.json")
    if set(config["model"]) - {"shared_intermediate_hidden", "shared_output_rounding"} != {
        "hidden",
        "intermediate_hidden",
        "num_experts",
        "num_topk",
        "num_shared_experts",
    }:
        raise ValueError("Model must specify H, intermediate width, global experts, top-k and shared experts")
    if bs_values is not None:
        config["bs_values"] = [int(part) for part in bs_values.split(",")]
    batches = config["bs_values"]
    if (
        not batches
        or len(set(batches)) != len(batches)
        or any(type(b) is not int or not 0 < b <= 65536 for b in batches)
    ):
        raise ValueError("BS values must be distinct integers in [1,65536]")
    for name, minimum in (("warmup", 0), ("iters", 1), ("seed", 0)):
        if type(config[name]) is not int or config[name] < minimum:
            raise ValueError(f"Invalid {name}")
    config["alignment"] = alignment_config(config)
    return config


ROUTING_SEED = 20261014
_cache_lock = threading.Lock()


@lru_cache(maxsize=1)
def _global_routes(bs, ep, experts, topk, seed):
    import numpy as np

    local_experts = experts // ep
    tokens = ep * bs
    # total/ep = BS*TopK exactly; this mode never creates a hot rank.
    remaining = [bs * topk] * ep
    occurrences = [0] * ep
    cap = (topk + ep - 1) // ep
    rng = random.Random(seed + bs * 1000 + 2)
    routes = np.empty((tokens, topk), dtype=np.int32)
    for token in range(tokens):
        left = tokens - token
        # Anything that cannot fit in subsequent tokens must go in this one.
        counts = [max(0, n - (left - 1) * cap) for n in remaining]
        chosen = [rank for rank, count in enumerate(counts) for _ in range(count)]
        if len(chosen) > topk or any(n > cap for n in counts):
            raise RuntimeError("Infeasible random-balanced rank quotas")
        while len(chosen) < topk:
            candidates = [r for r in range(ep) if counts[r] < cap and remaining[r] > counts[r]]
            if not candidates:
                raise RuntimeError("Random-balanced routing exhausted its candidates")
            rank = rng.choices(candidates, weights=[remaining[r] - counts[r] for r in candidates])[0]
            chosen.append(rank)
            counts[rank] += 1
        rng.shuffle(chosen)
        for slot, rank in enumerate(chosen):
            # Advance on EACH occurrence, including repeated destinations in EP4.
            routes[token, slot] = rank * local_experts + occurrences[rank] % local_experts
            remaining[rank] -= 1
            occurrences[rank] += 1
    if any(remaining):
        raise RuntimeError("Random-balanced routing left unassigned quotas")
    routes.setflags(write=False)
    return routes


def random_balanced_indices(bs, ep, experts, topk, rank, *, seed=ROUTING_SEED):
    """Return this source rank's deterministic slice of the global token stream.

    The routing seed intentionally matches the supplied script, independently
    of the model/input seed. Cache one global table per process, not per source
    rank; CPU Golden threads share it. Callers receive a writable private copy.
    """
    if any(type(n) is not int for n in (bs, ep, experts, topk, rank, seed)):
        raise ValueError("Routing dimensions and seed must be integers")
    if bs <= 0 or ep <= 0 or experts <= 0 or experts % ep or not 0 < topk <= experts or not 0 <= rank < ep:
        raise ValueError("Invalid random-balanced BS/EP/experts/TopK/rank")
    with _cache_lock:
        routes = _global_routes(bs, ep, experts, topk, seed)
        return routes[rank * bs : (rank + 1) * bs].copy()


def inputs(config, rank, bs, routing="balanced", *, world):
    import numpy as np
    from .reference import bf16

    model = config["model"]
    if world <= 0 or model["num_experts"] <= 0 or model["num_experts"] % world:
        raise ValueError("Global experts must divide positive EP size")
    if not 0 <= rank < world or bs <= 0 or not 0 < model["num_topk"] <= model["num_experts"]:
        raise ValueError("Invalid rank, BS or top-k for input generation")
    rng = np.random.default_rng(config["seed"] + rank * 1009 + bs)
    x = bf16(rng.normal(0, 0.2, (bs, model["hidden"])).astype(np.float32))
    if routing == "balanced":
        # Enumerate destinations rank-first, then advance the local expert. Each source
        # takes its own slice of the global [token, top-k] route stream.
        local_experts = model["num_experts"] // world
        route_start = rank * bs * model["num_topk"]
        routes = np.arange(route_start, route_start + bs * model["num_topk"], dtype=np.int64)
        ids = ((routes % world) * local_experts + (routes // world) % local_experts).reshape(bs, model["num_topk"])
    elif routing == "random-balanced":
        ids = random_balanced_indices(bs, world, model["num_experts"], model["num_topk"], rank)
    elif routing == "skewed":
        ids = np.broadcast_to(np.arange(model["num_topk"]), (bs, model["num_topk"]))
    elif routing == "random":
        ids = np.array([rng.choice(model["num_experts"], model["num_topk"], replace=False) for _ in range(bs)])
    else:
        raise ValueError(f"Unknown routing pattern: {routing}")
    weights = rng.random(ids.shape, dtype=np.float32)
    if routing == "random":
        weights = 2 * weights - 1  # 精度覆盖负权重；不能隐含依赖概率非负。
    weights /= np.abs(weights).sum(axis=1, keepdims=True)
    return x, ids.astype(np.int32), weights


def accuracy(
    config,
    rank,
    world,
    bs,
    output,
    counts,
    check_tokens,
    routing="balanced",
    cumulative_stats=None,
    *,
    group,
    golden_dir,
    input_kind,
):
    import numpy as np
    from .golden import cache_path, compare, contract, load
    from .reference import error_metrics

    spec = contract(config, world, bs, input_kind, routing, check_tokens)
    expected, expected_local, selected = load(golden_dir, spec, rank)
    full_output = output.float().cpu().numpy()
    actual = full_output[selected]
    report = compare(actual, expected)
    report.update(error_metrics(actual, expected))
    report["finite"] = bool(np.isfinite(full_output).all())
    report.update(
        rank=rank,
        checked_token_count=len(selected),
        total_tokens=bs,
        check_tokens=check_tokens,
        golden_path=str(cache_path(golden_dir, spec, rank)),
        counts_equal=bool(np.array_equal(counts.cpu().numpy(), expected_local)),
        policy="cpu_mixed_error",
        reference="cached_numpy_quantization_aware_not_bitwise",
    )
    report["passed"] &= report["counts_equal"] and report["finite"]
    if cumulative_stats is not None:
        report["cumulative_counts_equal"] = bool(np.array_equal(cumulative_stats.cpu().numpy(), 7 + 2 * expected_local))
        report["passed"] &= report["cumulative_counts_equal"]
    # 仅交换小报告，不把 [BS,H] 矩阵或全量 token 索引放进 Store。
    return group.all_gather(report, f"accuracy reports BS={bs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="smoke", help="Packaged name or JSON path")
    parser.add_argument("--bs-values", help="Only model config override, e.g. 96,4096")
    parser.add_argument("--mode", choices=("accuracy", "perf"), default="accuracy")
    parser.add_argument("--input", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument(
        "--routing",
        choices=("balanced", "random", "skewed", "random-balanced"),
        default="balanced",
        help="Reproducible accuracy stress pattern; record the same pattern for performance comparisons",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory; must not already exist")
    parser.add_argument("--timer", action="store_true", help="One separate instrumented call per case")
    parser.add_argument("--timer-ranks", type=parse_ranks, help="Default: global ranks 0..min(7,EP-1)")
    parser.add_argument("--check-tokens", type=int, default=0, help="Accuracy samples per rank; default 0 checks all")
    parser.add_argument(
        "--golden-dir", type=Path, default=Path("benchmark-golden"), help="Precomputed CPU Golden cache"
    )
    parser.add_argument("--relative-rmse", type=float, default=0.02, help="Timer on/off output tolerance only")
    parser.add_argument("--dual-reference", action="store_true", help="Compare CPU Golden and NPU cascade")
    args = parser.parse_args()
    config = load_config(args.config, args.bs_values)
    if args.check_tokens < 0 or not 0 < args.relative_rmse < 1:
        parser.error("Invalid accuracy sample count or tolerance")
    import torch
    import torch_npu  # noqa: F401 - Registers NPU dtype/device/stream support.
    from .api import StoreGroup

    torch.set_num_threads(1)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    group = StoreGroup.from_env()
    try:
        run(args, config, group)
    except BaseException as error:
        group.abort(f"benchmark: {type(error).__name__}: {error}")
        # 异常 rank 不再进入 SHMEM 全体清理屏障；保留根因，由启动器终止本次作业。
        raise


def run(args, config, group):
    from .backend import select_backend

    backend = select_backend(group)
    ubx = backend == "ubx"
    import torch
    from .api import SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe
    from .reference import make_weights, quantize_input
    from .validation import collective_check

    rank, world = group.rank(), group.size()
    collective_check(
        group,
        "benchmark configuration",
        None,
        (
            config,
            args.mode,
            args.input,
            args.routing,
            args.timer,
            args.timer_ranks,
            args.check_tokens,
            args.relative_rmse,
            bool(getattr(args, "dual_reference", False)),
        ),
    )
    if ubx and args.input != "fp8":
        raise ValueError("UBX currently supports --input fp8; BF16 input remains a POD capability")
    timer_ranks = args.timer_ranks if args.timer_ranks is not None else list(range(min(world, 8)))
    if any(value >= world for value in timer_ranks):
        raise ValueError("timer rank exceeds EP world size")
    if config["model"]["num_experts"] % world:
        raise ValueError("Global experts must divide EP")
    if args.mode == "accuracy":
        from .golden import contract, load

        error = None
        try:
            for bs in config["bs_values"]:
                load(args.golden_dir, contract(config, world, bs, args.input, args.routing, args.check_tokens), rank)
        except Exception as failure:
            error = f"rank={rank} Golden validation: {failure}"
        collective_check(group, "accuracy Golden cache", error)
    # 每台机器可使用独立输出盘；rank 子目录防止共享盘写冲突，不收集机器凭据。
    output_error = None
    try:
        rank_dir = args.output_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        output_error = str(error)
    errors = group.all_gather(output_error, "output directory")
    if any(errors):
        raise RuntimeError(f"Output creation failed; choose a new output directory: {errors}")
    manifest = dict(
        config=config,
        input=args.input,
        mode=args.mode,
        routing=args.routing,
        ep_size=world,
        rank=rank,
        local_rank=int(os.environ["LOCAL_RANK"]),
        timer_ranks=timer_ranks,
        host_coordination="tcpstore",
        backend=backend,
        device_communication="shmem_mte_udma" if ubx else "shmem_mte",
        api="fp8_fp4_mega_moe",
        numerical_contract="ascend_post_gmm2_route_weight_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        build=json.loads(files("deep_ep").joinpath("_build_info.json").read_text()),
    )
    if ubx:
        manifest.update(
            implementation="consumer",
            precision="routed_w4_shared_w8",
            dual_reference=bool(getattr(args, "dual_reference", False)),
            wait_events=False,
            device_routing_in_kernel=True,
        )
    (rank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model = config["model"]
    if (
        not ubx
        and model.get("shared_intermediate_hidden", model["intermediate_hidden"]) != model["intermediate_hidden"]
    ):
        raise ValueError("POD shared width must match routed width")
    if getattr(args, "dual_reference", False) and args.mode != "accuracy":
        raise ValueError("--dual-reference requires --mode accuracy")
    buffer = SymmBuffer(
        group,
        num_max_tokens_per_rank=max(config["bs_values"]),
        ranks_per_node=int(os.environ["LOCAL_WORLD_SIZE"]),
        **{k: v for k, v in model.items() if k not in ("shared_intermediate_hidden", "shared_output_rounding")},
    )

    def device_pair(pair):
        data, scale = pair
        return (torch.from_numpy(data).npu(), torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu))

    if args.timer and not buffer.timer_enabled:
        raise RuntimeError("Timer requires a W4A8 wheel built with DEEPEP_MEGAMOE_TIMER=ON")
    raw = make_weights(
        model["num_experts"] // world, model["hidden"], model["intermediate_hidden"], config["seed"] + rank
    )
    prepared = transform_weights_for_mega_moe(*map(device_pair, raw))
    shared_raw, shared_kwargs = None, {}
    if model["num_shared_experts"]:
        shared_raw = make_weights(
            model["num_shared_experts"],
            model["hidden"],
            model.get("shared_intermediate_hidden", model["intermediate_hidden"]),
            config["seed"] + 100000,
        )

        def shared_pair(pair):
            if not ubx:
                return device_pair(pair)
            from .reference import expand_fp4_to_fp8

            data, scale = device_pair(expand_fp4_to_fp8(pair))
            return data.view(torch.float8_e4m3fn), scale

        shared = transform_weights_for_mega_moe(*map(shared_pair, shared_raw), shared=True)
        shared_kwargs = dict(shared_l1_weights=shared[0], shared_l2_weights=shared[1])
    cascade = None
    cascade_owned = False
    if getattr(args, "dual_reference", False):
        from .cascade import CascadeReference, init_process_group

        cascade_owned = init_process_group(group)
        cascade = CascadeReference(model, raw, shared_raw, host_group=group)
    guard = None
    if ubx and args.mode == "perf":
        from .performance import UrmaLaunchGuard

        alignment = config["alignment"]
        guard = UrmaLaunchGuard(group, (alignment["rows"], alignment["cols"]), alignment["repeats"])
        guard.prepare_per_call_alignment()
        manifest["alignment"] = guard.describe()
        (rank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for bs in config["bs_values"]:
        case = rank_dir / f"bs_{bs}"
        case.mkdir()
        x, ids, weights = inputs(config, rank, bs, args.routing, world=world)
        if args.input == "bf16":
            activation = torch.from_numpy(x).to(device="npu", dtype=torch.bfloat16)
        else:
            data, scale = quantize_input(x)
            buffer.x[:bs].copy_(torch.from_numpy(data).npu().view(torch.float8_e4m3fn))
            buffer.x_sf[:bs].copy_(torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu))
            activation = None
        buffer.topk_idx[:bs].copy_(torch.from_numpy(ids))
        buffer.topk_weights[:bs].copy_(torch.from_numpy(weights))
        if not ubx:
            buffer.validate_inputs(bs, x=activation)
        buffer.prepare(bs)
        result = torch.empty((bs, model["hidden"]), device="npu", dtype=torch.bfloat16)

        def call(timer=None, stats=None):
            fp8_fp4_mega_moe(
                result,
                *prepared,
                buffer,
                cumulative_local_expert_recv_stats=stats,
                x=activation,
                timer=timer,
                **shared_kwargs,
            )

        if ubx:
            # Shared dimensions are inferred by native forward. Allocate its
            # workspace before profiling, using the same public call as POD.
            call()
            torch.npu.synchronize()
        resources = dict(
            workspace_bytes=buffer.workspace_bytes,
            symmetric_buffer_bytes=buffer.symmetric_buffer_bytes,
            prepared_bs=bs,
            max_tokens=buffer.num_max_tokens_per_rank,
        )
        (case / "resources.json").write_text(json.dumps(resources, indent=2), encoding="utf-8")
        timer, error = None, None
        if args.timer:
            try:
                timer = buffer.allocate_timer() if rank in timer_ranks else None
            except Exception as failure:
                error = str(failure)
            collective_check(group, f"timer allocation BS={bs}", error)
        prefix = None
        if not ubx and (args.mode == "perf" or args.timer):
            error = None
            try:
                alignment = config["alignment"]
                prefix = torch.zeros((alignment["rows"], alignment["cols"]), dtype=torch.float32, device="npu")
                torch.exp(prefix, out=prefix)  # 预热 Exp，不能把懒初始化带入采集批次。
            except Exception as failure:
                error = f"rank={rank} BS={bs} alignment={config['alignment']}: {failure}"
            collective_check(group, f"alignment allocation BS={bs}", error)
            if rank == 0:
                print(
                    f"[PROFILE] BS={bs} alignment={alignment} "
                    f"temporary_bytes={alignment['rows'] * alignment['cols'] * 4} "
                    f"batch_iterations={config['iters']}",
                    flush=True,
                )
        if not (ubx and args.mode == "perf"):
            for _ in range(config["warmup"]):
                call()
            torch.npu.synchronize()
            group.barrier(f"warmup BS={bs}")
        if args.mode == "accuracy":
            stats = torch.full((model["num_experts"] // world,), 7, dtype=torch.int32, device="npu")
            call(stats=stats)
            counts = stats - 7
            call(stats=stats)  # 非零初值和重复累加，验证公开接口而非内部 forward。
            torch.npu.synchronize()
            reports = accuracy(
                config,
                rank,
                world,
                bs,
                result,
                counts,
                args.check_tokens,
                args.routing,
                stats,
                group=group,
                golden_dir=args.golden_dir,
                input_kind=args.input,
            )
            if cascade is not None:
                from .golden import contract, load
                from .accuracy import compare_outputs

                expected, expected_counts, selected = load(
                    args.golden_dir, contract(config, world, bs, args.input, args.routing, args.check_tokens), rank
                )
                other, other_counts = cascade.forward(
                    activation,
                    buffer.topk_idx[:bs],
                    buffer.topk_weights[:bs],
                    input_kind=args.input,
                    quantized=(buffer.x[:bs], buffer.x_sf[:bs]) if activation is None else None,
                )
                dual = compare_outputs(
                    result.float().cpu().numpy(),
                    other.float().cpu().numpy(),
                    expected,
                    counts.cpu().numpy(),
                    other_counts.cpu().numpy(),
                    expected_counts,
                    selected,
                )
                dual["passed"] &= bool((stats == 7 + 2 * counts).all().item())
                reports = group.all_gather(dual, f"dual reference BS={bs}")
            (case / "accuracy.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
            if not all(item["passed"] for item in reports):
                raise AssertionError(f"BS={bs}: accuracy FAILED; see {case / 'accuracy.json'}")
            if rank == 0:
                print(
                    f"[ACCURACY] BS={bs} all {world} ranks passed ({args.input}, sampled={args.check_tokens})",
                    flush=True,
                )
        elif ubx:
            from .performance import measure_profiled_forward, summarize_profile_rounds

            capture = measure_profiled_forward(
                call,
                warmup=config["warmup"],
                iters=config["iters"],
                guard=guard,
                directory=case / "profiler",
                fine_forward=(lambda: call(timer)) if args.timer else None,
            )
            (case / "kernel_samples.json").write_text(json.dumps(capture["samples"], indent=2), encoding="utf-8")
            per_rank = group.all_gather(capture["durations_us"], f"profiling BS={bs}")
            report = summarize_profile_rounds(per_rank, int(os.environ.get("LOCAL_WORLD_SIZE", world)))
            report.update(includes_device_routing=True)
            (case / "performance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            samples = collect_profile(call, group, torch, config, prefix, case / "profiler", bs)
            (case / "kernel_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
            timings = [sample["duration_us"] for sample in samples]
            per_rank = group.all_gather(timings, f"profiling BS={bs}")
            report = summarize_rounds(per_rank, int(os.environ["LOCAL_WORLD_SIZE"]))
            report["alignment"] = config["alignment"]
            (case / "profiling.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if rank == 0:
                print(
                    f"[PERF] BS={bs} MegaMoe E2E={report['mean_us']:.3f}us "
                    f"rank-max-mean={report['rank_max_mean_us']:.3f}us "
                    f"rounds={report['rounds']}/{report['recorded_rounds']} "
                    f"(last ceil(iters/2); kernel Duration; rank-min mean; timer off)",
                    flush=True,
                )
        if args.timer:
            uninstrumented = result.float().cpu().numpy() if args.mode == "accuracy" else None
            if ubx:
                if args.mode == "accuracy":
                    call(timer)
            else:
                torch.npu.synchronize()
                group.barrier(f"timer before warmup BS={bs}")
                for _ in range(config["warmup"]):
                    call()
                torch.npu.synchronize()
                group.barrier(f"timer batch BS={bs}")
                enqueue_batch(call, torch, prefix, config["alignment"], config["iters"], timer)
            torch.npu.synchronize()
            if uninstrumented is not None:
                from .reference import error_metrics

                metrics = error_metrics(result.float().cpu().numpy(), uninstrumented)
                comparisons = group.all_gather(dict(rank=rank, **metrics), f"timer accuracy BS={bs}")
                (case / "timer_accuracy.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
                if any(not item["finite"] or item["relative_rmse"] > args.relative_rmse for item in comparisons):
                    raise AssertionError(f"BS={bs}: timer-on/off output mismatch")
            if ubx:
                from .urma_trace import export_diagnostics

                export_diagnostics(group, timer, case, timer_ranks=timer_ranks)
            else:
                _export_pod_timer(args.output_dir, bs, rank, group, timer)
            del timer
        del prefix
        group.barrier(f"case complete BS={bs}")
    group.barrier("before SHMEM teardown")
    buffer.destroy()
    if guard is not None:
        guard.close()
    if cascade_owned:
        import torch.distributed as dist

        dist.destroy_process_group()
    group.close()


def _export_pod_timer(output_dir, bs, rank, group, timer):
    timer_dir = output_dir / f"bs_{bs}" / "timer"
    breakdown = None
    if timer is not None:
        from .timer.breakdown import compact_events, rank_breakdown

        timer_dir.mkdir(parents=True, exist_ok=True)
        raw_timer = timer.cpu().numpy().astype("<i8", copy=False).reshape(-1)
        raw_timer.tofile(timer_dir / f"rank_{rank}_npu_time.bin")
        breakdown = rank_breakdown(compact_events(raw_timer, rank), rank)
        del raw_timer
        print(f"[TIMER] rank={rank} BS={bs} raw saved; offline parse required", flush=True)
    records = group.all_gather(breakdown, f"timer breakdown BS={bs}")
    if rank == 0:
        from .timer.breakdown import write_breakdown

        timer_dir.mkdir(parents=True, exist_ok=True)
        write_breakdown([r for r in records if r is not None], timer_dir, f"BS={bs}")


if __name__ == "__main__":
    main()
