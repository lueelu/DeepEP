# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""BF16 device accuracy against the small-operator and CPU FP64 references."""


def combine_baseline(partial, args, rank, world, device):
    """Move route values through Gloo, then apply the selected NPU combine."""
    import torch
    import torch.distributed as dist

    from npu_baseline import unpermute_routes

    m, h = args.seq, args.hidden
    if args.inference:
        weights = partial["weights"][rank * m : (rank + 1) * m].to(device=device, dtype=torch.float32)
        for destination in range(world):
            buffer = torch.zeros((m, args.topk, h), dtype=torch.float32)
            for route in partial["routes"]:
                mask = route["rows"] // m == destination
                buffer[route["rows"][mask] % m, route["slots"][mask]] = route["y"][mask].float()
            dist.reduce(buffer.view(torch.int32), dst=destination, op=dist.ReduceOp.BOR)
            if destination == rank:
                routes = buffer.to(device)
                y = routes[:, 0] * weights[:, 0:1]
                for slot in range(1, args.topk):
                    y += routes[:, slot] * weights[:, slot : slot + 1]
        return {"y": y.bfloat16().cpu()}
    for destination in range(world):
        # Every row/slot has one owning expert. FP32 carries BF16 y/dx and
        # either BF16 or FP32 dp exactly; no route contributions are combined.
        buffer = torch.zeros((m, args.topk, 2 * h + 1), dtype=torch.float32, device="cpu")
        for route in partial["routes"]:
            mask = route["rows"] // m == destination
            rows, slots = route["rows"][mask] % m, route["slots"][mask]
            buffer[rows, slots, :h] = route["y"][mask].float()
            buffer[rows, slots, h : 2 * h] = route["dx"][mask].float()
            buffer[rows, slots, -1] = route["dp"][mask].float()
        # Each slot has a single writer. Transport its bits, including negative zero.
        dist.reduce(buffer.view(torch.int32), dst=destination, op=dist.ReduceOp.BOR)
        if rank == destination:
            ids = partial["ids"][rank * m : (rank + 1) * m]
            result = unpermute_routes(
                buffer, ids, device, reference=partial["reference"], topk_weight_dtype=partial["topk_weight_dtype"]
            )
    result.update(dw1=partial["dw1"], dw2=partial["dw2"])
    return result


def local_truth(inputs, args, rank):
    import torch.distributed as dist

    from cpu_reference import run_reference

    result = run_reference(
        inputs, rank * args.expert_per_rank, args.chunk_rows, args.swiglu_limit, forward_only=args.inference
    )
    sl = slice(rank * args.seq, (rank + 1) * args.seq)
    for name in ("y",) if args.inference else ("y", "dx", "dtopk_weights"):
        dist.all_reduce(result[name])
        result[name] = result[name][sl].clone()
    return result


def prepare_references(inputs, args, rank, world, device, mode, trace, phase):
    from npu_baseline import run_baseline

    with phase("cpu_fp64_reference"):
        truth = local_truth(inputs, args, rank)
    with phase("bigop_reference"):
        raw_baseline = combine_baseline(
            run_baseline(
                inputs,
                device,
                rank * args.expert_per_rank,
                "bf16",
                args.swiglu_limit,
                reference="bigop",
                inference=args.inference,
                trace=trace if mode == "bf16" else None,
            ),
            args,
            rank,
            world,
            device,
        )
    if args.inference or mode == "bf16":
        # Both checks use the same uncompressed small-op reference.
        mixed_baseline = raw_baseline
    else:
        with phase("megamoe_contract_reference"):
            mixed_baseline = combine_baseline(
                run_baseline(inputs, device, rank * args.expert_per_rank, mode, args.swiglu_limit, trace=trace),
                args,
                rank,
                world,
                device,
            )
    return truth, raw_baseline, mixed_baseline


def bitwise_bigop(actual, baseline):
    """Compare native bytes per element. Count an element once even if several bytes differ."""
    from precision_metrics import OUTPUTS, compare_bits

    checks = {}
    for name in OUTPUTS:
        a, b = actual[name], baseline[name]
        result = compare_bits(a, b)
        if "numel" in result:
            checks[name] = dict(
                passed=result["passed"],
                numel=result["numel"],
                mismatches=result["different_elements"],
                first_flat=result["first_flat_index"],
            )
        else:
            checks[name] = {"passed": False, "error": f"{a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"}
    return {"passed": all(check["passed"] for check in checks.values()), "outputs": checks}


def check_accuracy(values, mixed_baseline, raw_baseline, truth, expected, args, phase, output, rank):
    import torch
    from precision_metrics import evaluate, compare_bits

    with phase("precision_metrics"):
        actual = {name: value.detach().cpu().clone() for name, value in values.items()}
        result = evaluate(actual, mixed_baseline, expected, outputs=tuple(mixed_baseline))
        if args.bitwise:
            exact = {name: compare_bits(actual[name], ref) for name, ref in mixed_baseline.items()}
            exact["counts"] = compare_bits(actual["counts"], expected.to(torch.int32))
            result["bitwise"] = exact
            result["passed"] &= all(check["passed"] for check in exact.values())
        if args.bitwise_bigop:
            result["bitwise_bigop"] = bitwise_bigop(actual, raw_baseline)
            result["passed"] &= result["bitwise_bigop"]["passed"]
    if args.atk:
        with phase("atk_compare"):
            from atk_compare import DEFAULT_RATIOS, compare_with_atk

            result["atk"] = compare_with_atk(
                actual,
                raw_baseline,
                truth,
                output / f"atk_rank_{rank}",
                args.atk_ratios or DEFAULT_RATIOS,
                outputs=tuple(raw_baseline),
            )
        result["passed"] &= result["atk"]["passed"]
    return actual, result
