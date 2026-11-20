# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Benchmarks for the platform MTE and URMA plan ABI."""

from datetime import datetime, timezone
from importlib.resources import files
import json
import os

from .benchmark import _export_mte_timer, accuracy, inputs
from .performance import collect_profile, enqueue_batch, flush_l2_before_call, summarize_rounds


def run(args, config, group, backend):
    urma = backend == "urma"
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
    if urma and args.input != "fp8":
        raise ValueError("URMA currently supports --input fp8; BF16 input remains a MTE capability")
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
        device_communication="shmem_mte_udma" if urma else "shmem_mte",
        api="fp8_fp4_mega_moe",
        numerical_contract=("ascend_pre_quant_route_weight_v1" if config["model"].get("topk_weights_type", 0) else "ascend_post_gmm2_route_weight_v1"),
        created_utc=datetime.now(timezone.utc).isoformat(),
        build=json.loads(files("deep_ep").joinpath("_build_info.json").read_text()),
    )
    if urma:
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
        not urma
        and model.get("shared_intermediate_hidden", model["intermediate_hidden"]) != model["intermediate_hidden"]
    ):
        raise ValueError("MTE shared width must match routed width")
    if getattr(args, "dual_reference", False) and args.mode != "accuracy":
        raise ValueError("--dual-reference requires --mode accuracy")
    buffer = SymmBuffer(
        group,
        num_max_tokens_per_rank=max(config["bs_values"]),
        ranks_per_node=int(os.environ["LOCAL_WORLD_SIZE"]),
        **{k: v for k, v in model.items() if k not in ("shared_intermediate_hidden", "shared_output_rounding", "activation_clamp")},
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
            if not urma:
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
    if urma and args.mode == "perf":
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
        if not urma:
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
                activation_clamp=config.get("activation_clamp", model.get("activation_clamp")),
                **shared_kwargs,
            )

        def profiled_mte_call(timer):
            flush_l2_before_call()
            call(timer)

        if urma:
            # Shared dimensions are inferred by native forward. Allocate its
            # workspace before profiling, using the same public call as MTE.
            if args.mode == "perf":
                flush_l2_before_call()
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
        if not urma and (args.mode == "perf" or args.timer):
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
        if not (urma and args.mode == "perf"):
            for _ in range(config["warmup"]):
                if args.mode == "perf":
                    flush_l2_before_call()
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
        elif urma:
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
            samples = collect_profile(profiled_mte_call, group, torch, config, prefix, case / "profiler", bs)
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
            if urma:
                if args.mode == "accuracy":
                    call(timer)
            else:
                torch.npu.synchronize()
                group.barrier(f"timer before warmup BS={bs}")
                for _ in range(config["warmup"]):
                    call()
                torch.npu.synchronize()
                group.barrier(f"timer batch BS={bs}")
                enqueue_batch(profiled_mte_call, torch, prefix, config["alignment"], config["iters"], timer)
            torch.npu.synchronize()
            if uninstrumented is not None:
                from .reference import error_metrics

                metrics = error_metrics(result.float().cpu().numpy(), uninstrumented)
                comparisons = group.all_gather(dict(rank=rank, **metrics), f"timer accuracy BS={bs}")
                (case / "timer_accuracy.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
                if any(not item["finite"] or item["relative_rmse"] > args.relative_rmse for item in comparisons):
                    raise AssertionError(f"BS={bs}: timer-on/off output mismatch")
            if urma:
                from .urma_trace import export_diagnostics

                export_diagnostics(group, timer, case, timer_ranks=timer_ranks)
            else:
                _export_mte_timer(args.output_dir, bs, rank, group, timer)
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
