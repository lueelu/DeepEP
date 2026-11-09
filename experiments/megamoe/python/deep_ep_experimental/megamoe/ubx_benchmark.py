"""UBX driver for the existing benchmark CLI and public MegaMoE API."""

import json
import os


def run(args, config, group):
    import numpy as np
    import torch
    from .api import SymmBuffer, mega_moe, transform_weights_for_mega_moe
    from .backend import cv_rows
    from .benchmark import inputs
    from .reference import make_weights, quantize_input, expand_fp4_to_fp8
    from .golden import contract, load, compare
    from .validation import collective_check
    from .urma_timing import UrmaLaunchGuard, measure_profiled_forward, summarize_profile_rounds

    if args.input != "fp8":
        raise ValueError("UBX currently supports --input fp8; BF16 input remains a POD capability")
    model = config["model"]
    rank, world = group.rank(), group.size()
    collective_check(group, "UBX benchmark", None, (config, args.mode, args.routing, args.dual_reference, args.timer))
    directory = args.output_dir / f"rank_{rank}"
    error = None
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        error = str(exc)
    collective_check(group, "UBX output directory", error)
    owner = SymmBuffer(
        group,
        **{k: v for k, v in model.items() if k not in ("shared_intermediate_hidden", "shared_output_rounding")},
        num_max_tokens_per_rank=max(config["bs_values"]),
    )
    raw = make_weights(
        model["num_experts"] // world, model["hidden"], model["intermediate_hidden"], config["seed"] + rank
    )

    def device_pair(pair):
        data, scales = pair
        return torch.from_numpy(data).npu(), torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu)

    w1, w2 = transform_weights_for_mega_moe(*map(device_pair, raw))
    shared_raw, shared = None, {}
    if model["num_shared_experts"]:
        shared_raw = make_weights(
            1,
            model["hidden"],
            model.get("shared_intermediate_hidden", model["intermediate_hidden"]),
            config["seed"] + 100000,
        )

        def shared_pair(pair):
            data, scale = device_pair(expand_fp4_to_fp8(pair))
            return data.view(torch.float8_e4m3fn), scale

        sw1, sw2 = transform_weights_for_mega_moe(*map(shared_pair, shared_raw), shared=True)
        shared = dict(shared_l1_weights=sw1, shared_l2_weights=sw2)
    guard, cascade, cascade_owned = None, None, False
    try:
        if args.dual_reference:
            if args.mode != "accuracy":
                raise ValueError("--dual-reference requires --mode accuracy")
            from .cascade import CascadeReference, init_process_group

            cascade_owned = init_process_group(group)
            cascade = CascadeReference(model, raw, shared_raw, host_group=group)
        if args.mode == "perf":
            alignment = config["alignment"]
            guard = UrmaLaunchGuard(group, (alignment["rows"], alignment["cols"]), alignment["repeats"])
            guard.prepare_per_call_alignment()
        manifest = dict(
            backend="ubx",
            implementation="consumer",
            config=config,
            precision="routed_w4_shared_w8",
            dual_reference=args.dual_reference,
            wait_events=False,
            device_routing_in_kernel=True,
            alignment=guard.describe() if guard else None,
        )
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        timer_ranks = args.timer_ranks if args.timer_ranks is not None else list(range(min(8, world)))
        if any(r < 0 or r >= world for r in timer_ranks):
            raise ValueError("timer rank exceeds EP world size")
        for bs in config["bs_values"]:
            case = directory / f"bs_{bs}"
            case.mkdir()
            host_x, ids, weights = inputs(config, rank, bs, args.routing, world=world)
            codes, scales = quantize_input(host_x)
            x = (
                torch.from_numpy(codes).npu().view(torch.float8_e4m3fn),
                torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu),
            )
            ids = torch.from_numpy(ids).npu().to(torch.int32)
            weights = torch.from_numpy(weights).npu()
            output = torch.empty((bs, model["hidden"]), dtype=torch.bfloat16, device="npu")

            def call(timer=None):
                return mega_moe(x, ids, weights, w1, w2, owner, out=output, timer=timer, **shared)

            # Resolve shape/CV/shared resources before measured invocations.
            result, counts = call()
            torch.npu.synchronize()
            trace = owner.allocate_timer() if args.timer and rank in timer_ranks else None
            baseline = None
            if args.mode == "accuracy":
                error, report = None, None
                try:
                    expected, expected_counts, selected = load(
                        args.golden_dir, contract(config, world, bs, "fp8", args.routing, args.check_tokens), rank
                    )
                    actual = result.float().cpu().numpy()
                    baseline = actual.copy()
                    actual_counts = counts.cpu().numpy()
                    if cascade is not None:
                        from .urma_accuracy import compare_outputs

                        other, other_counts = cascade.forward(None, ids, weights, input_kind="fp8", quantized=x)
                        report = compare_outputs(
                            actual,
                            other.float().cpu().numpy(),
                            expected,
                            actual_counts,
                            other_counts.cpu().numpy(),
                            expected_counts,
                            selected,
                        )
                    else:
                        report = compare(actual[selected], expected)
                        report["finite"] = bool(np.isfinite(actual).all())
                        report["counts_equal"] = bool(np.array_equal(actual_counts, expected_counts))
                        report["passed"] &= report["finite"] and report["counts_equal"]
                except Exception as exc:
                    error = str(exc)
                collective_check(group, "UBX accuracy execution", error)
                reports = group.all_gather(report, f"UBX accuracy BS={bs}")
                (case / "accuracy.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
                if not all(r["passed"] for r in reports):
                    raise AssertionError(f"UBX accuracy failed at BS={bs}")
                if args.timer:
                    call(trace)
                    torch.npu.synchronize()
            else:
                samples = measure_profiled_forward(
                    call,
                    warmup=config["warmup"],
                    iters=config["iters"],
                    guard=guard,
                    directory=case / "profiler",
                    fine_forward=(lambda: call(trace)) if args.timer else None,
                )
                rows = group.all_gather(samples["durations_us"], f"UBX samples BS={bs}")
                report = summarize_profile_rounds(rows, int(os.environ.get("LOCAL_WORLD_SIZE", world)))
                report.update(cv_rows=cv_rows(world, bs), includes_device_routing=True)
                (case / "performance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if args.timer:
                torch.npu.synchronize()
                if baseline is not None:
                    from .reference import error_metrics

                    error = error_metrics(result.float().cpu().numpy(), baseline)
                    collective_check(
                        group,
                        "UBX timer precision",
                        None if error["finite"] and error["relative_rmse"] <= args.relative_rmse else str(error),
                    )
                # Export only selected ranks, with one collective for every rank.
                failure = None
                if trace is not None:
                    try:
                        from .urma_trace import export_trace

                        raw_trace = trace.cpu().numpy()
                        raw_trace.astype("<i8", copy=False).tofile(case / "urma_trace.bin")
                        export_trace(
                            raw_trace, case / "urma_trace.json", rank=rank, clock_mhz=1000.0, include_waits=False
                        )
                    except Exception as exc:
                        failure = str(exc)
                collective_check(group, "UBX trace export", failure)
    finally:
        owner.destroy()
        if guard is not None:
            guard.close()
        if cascade_owned:
            import torch.distributed as dist

            dist.destroy_process_group()
