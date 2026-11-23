"""Compare eager and captured MegaMoE on two 16-rank hosts, using the same physical routes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

WORLD, STEPS = 32, 160
TAIL = (159, 0, 80, 1, 158, 2, 79, 3)


def digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def input_at(folder, config, kind, rank, step):
    from deep_ep_experimental.megamoe.benchmark import inputs
    from deep_ep_experimental.megamoe.reference import quantize_input

    if folder:
        fields = ("bf16", "ids", "coeff") if kind == "bf16" else ("x", "scale", "ids", "coeff")
        with np.load(folder / f"input_{step}.npz") as data:
            return tuple(data[k] for k in fields)
    sample = dict(config, seed=config["seed"] + 65537 * step)
    x, ids, coeff = inputs(sample, rank, config["bs_values"][0], "balanced", world=WORLD)
    ids = (ids + step) % config["model"]["num_experts"]
    if kind == "bf16":
        return (x.view(np.uint32) >> 16).astype(np.uint16), ids, coeff
    return *quantize_input(x), ids, coeff


def weights(folder, config, rank, role, torch):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe
    from deep_ep_experimental.megamoe.reference import make_weights

    model = config["model"]
    shared = role == "shared"
    n = model["num_shared_experts"] if shared else model["num_experts"] // WORLD
    if folder:
        pairs = []
        for j in range(2):
            with np.load(folder / f"{role}{j}.npz") as data:
                pairs.append((data["w"], data["sf"]))
    else:
        pairs = make_weights(n, model["hidden"], model["intermediate_hidden"],
                             config["seed"] + (100000 if shared else rank), native_fp8=shared)
    hashes, groups = [], []
    for w, sf in pairs:
        hashes.append(digest((w, sf)))
        dtype = torch.float8_e4m3fn if shared else torch.uint8
        groups.append(([torch.from_numpy(wi).npu().view(dtype) for wi in w],
                       [torch.from_numpy(si).npu().view(torch.float8_e8m0fnu) for si in sf]))
    owner = prepare_weights_for_mega_moe(*groups, shared=shared)
    return owner, hashes


def profile(torch, group, forward, dest, bs):
    from deep_ep_experimental.megamoe.performance import DEFAULT_ALIGNMENT, collect_profile, summarize_rounds

    cfg = dict(iters=30, alignment=DEFAULT_ALIGNMENT)
    prefix = torch.zeros((DEFAULT_ALIGNMENT["rows"], DEFAULT_ALIGNMENT["cols"]), device="npu", dtype=torch.float32)
    torch.exp(prefix, out=prefix)
    flush = torch.zeros(256 * 1024**2 // 4, device="npu", dtype=torch.int32)
    total = torch.empty((), device="npu", dtype=torch.int32)
    torch.sum(flush, dim=(0,), dtype=torch.int32, out=total)
    samples = collect_profile(lambda _: forward(None), group, torch, cfg, prefix, dest / "profiler", bs,
                              method="legacy-read", flush_buffer=flush, sum_out=total)
    rows = group.all_gather([s["duration_us"] for s in samples], "fixed device times")
    if group.rank() == 0:
        report = summarize_rounds(rows, 16)
        (dest.parent / "fixed_perf.json").write_text(json.dumps(dict(
            rank0_mean_us=report["rank0_mean_us"], rank_max_mean_us=report["rank_max_mean_us"], all_32x30_us=rows)))


def run(args, config, group, torch):
    from deep_ep_experimental.megamoe import SymmBuffer, fp8_fp4_mega_moe, golden
    from deep_ep_experimental.megamoe.api import _extension
    from deep_ep_experimental.megamoe.reference import error_metrics

    bs = config["bs_values"][0]
    rank = group.rank()
    if group.size() != WORLD or int(os.environ["LOCAL_WORLD_SIZE"]) != 16:
        raise ValueError("Requires two hosts with 16 ranks each")
    folder = args.bank / f"rank_{rank}" if args.bank else None
    meta = json.loads((folder / "bank.json").read_text()) if folder else None
    kind = args.input
    if meta and (meta["input"] != kind or meta["rank"] != rank or meta["config"] != config):
        raise ValueError("Bank identity or model mismatch")
    dest = args.out / f"rank_{rank}"
    dest.mkdir(parents=True, exist_ok=False)
    routed, rh = weights(folder, config, rank, "routed", torch)
    shared, sh = weights(folder, config, rank, "shared", torch)
    weight_hashes = dict(routed=rh, shared=sh)
    if meta and weight_hashes != meta["weights"]:
        raise ValueError("Weight bank checksum mismatch")
    opts = dict(ranks_per_node=16, num_max_tokens_per_rank=bs, num_max_configs=16, **config["model"])
    root = SymmBuffer(group, **opts)
    buf = SymmBuffer(group, base=root, **opts)
    root.prepare(bs)
    buf.prepare(bs)
    y = torch.empty_like(buf.x, dtype=torch.bfloat16)
    x = torch.empty_like(y) if kind == "bf16" else None
    experts = config["model"]["num_experts"]
    expected_active = kind == "bf16" and bs == 96 and (experts == 416 or not args.baseline_b78fd2c)
    stats = torch.zeros(experts // WORLD, dtype=torch.int32, device="npu")

    def install(target, step):
        sample = input_at(folder, config, kind, rank, step)
        *features, ids, coeff = map(torch.from_numpy, sample)
        if x is None:
            target.x.copy_(features[0].view(torch.float8_e4m3fn))
            target.x_sf.copy_(features[1].view(torch.float8_e8m0fnu))
        else:
            x.copy_(features[0].view(torch.bfloat16))
        target.topk_idx.copy_(ids)
        target.topk_weights.copy_(coeff)
        checksum = digest(sample)
        if meta and checksum != meta["hashes"][step]:
            raise ValueError(f"Input checksum mismatch on rank {rank}, step {step}")
        return checksum, np.bincount(sample[-2].ravel(), minlength=experts)

    def forward(counts=stats, target=buf, activation=x):
        fp8_fp4_mega_moe(y, routed.l1, routed.l2, target, x=activation, activation_clamp=10,
                         shared_l1_weights=shared.l1, shared_l2_weights=shared.l2, cumulative_local_expert_recv_stats=counts)

    stream = buf._runtime.stream
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        install(buf, 0)
        for _ in range(20):
            forward(None)
        stream.synchronize()
        ext = _extension()
        plan = dict(protocol=ext.combine_reduce_version, dispatch_qps=ext.dispatch_qps, combine_qps=ext.combine_qps,
                    wave=buf._plan.effective_wave, bf16_bs96_active=buf._plan.bf16_bs96_active, extension=ext.__file__)
        (dest / "plan.json").write_text(json.dumps(plan))
        if (plan["protocol"], plan["dispatch_qps"], plan["combine_qps"], plan["wave"], plan["bf16_bs96_active"]) != (
            5, 8, 24, 5, expected_active
        ):
            raise ValueError(f"Unexpected schedule: {plan}")
        if args.mode == "perf":
            if buf.timer_enabled or kind != "bf16":
                raise ValueError("Performance comparison requires a BF16 bank and timer-off wheel")
            profile(torch, group, forward, dest, bs)
        else:
            if folder:
                with np.load(folder / "golden.npz") as data:
                    gy, gc = data["output"], data["counts"]
                if digest((gy, gc)) != meta["golden_sha256"]:
                    raise ValueError("Invalid bank golden")
            else:
                spec = golden.contract(config, WORLD, bs, kind, "balanced", 0)
                gy, gc, _ = golden.load(args.golden, spec, rank)
            install(root, 0)
            stats.zero_()
            forward(target=root)
            root_result = digest((y.float().cpu().numpy(), stats.cpu().numpy()))
            mixed = None
            if x is not None:
                from deep_ep_experimental.megamoe.reference import quantize_input
                codes, sf = quantize_input(x.float().cpu().numpy())
                root.x.copy_(torch.from_numpy(codes).view(torch.float8_e4m3fn))
                root.x_sf.copy_(torch.from_numpy(sf).view(torch.float8_e8m0fnu))
                stats.zero_()
                forward(target=root, activation=None)
                mixed = digest((y.float().cpu().numpy(), stats.cpu().numpy()))
                if root._plan.bf16_bs96_active:
                    raise ValueError("Prequantized input selected the BF16 path")
                stats.zero_()
                forward(target=root)
                if root._plan.bf16_bs96_active != expected_active or digest((y.float().cpu().numpy(), stats.cpu().numpy())) != root_result:
                    raise ValueError("Same-plan BF16 re-entry changed the schedule or result")
            replay = args.mode == "replay"
            if replay:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    forward()
            results = []
            steps = range(1 if args.mode == "fixed" else STEPS)
            for i, step in enumerate([*steps, *(TAIL if replay else ())]):
                checksum, local = install(buf, step)
                hist = np.sum(group.all_gather(local.tolist(), f"physical routes {i}"), axis=0)
                stats.zero_()
                y.fill_(float("nan"))
                if replay and (i < STEPS or (i - STEPS) % 2):
                    graph.replay()
                else:
                    forward()
                out, counts = y.float().cpu().numpy(), stats.cpu().numpy()
                if args.mode == "fixed":
                    # Every finite BF16 bit pattern has a distinct exact FP32 representation, including signed zero.
                    np.save(dest / "fixed.npy", out)
                    if args.reference:
                        expected = np.load(args.reference / dest.name / "fixed.npy")
                        metrics = error_metrics(out, expected)
                        mismatches = int(np.count_nonzero(out.view(np.uint32) != expected.view(np.uint32)))
                        (dest / "fixed_metrics.json").write_text(json.dumps(dict(metrics, mismatches=mismatches, values=out.size)))
                        if mismatches:
                            raise ValueError(f"Fixed BF16 output differs on rank {rank}: {mismatches}/{out.size}")
                if not np.isfinite(out).all() or not np.array_equal(counts, np.split(hist, WORLD)[rank]):
                    raise ValueError(f"Invalid result at rank {rank}, step {step}")
                outcome = digest((out, counts))
                if i == 0 and (root_result != outcome or not golden.compare(out, gy)["passed"] or not np.array_equal(counts, gc)):
                    raise ValueError(f"First output disagrees with shared view or CPU golden on rank {rank}")
                results.append([checksum, outcome])
            record = dict(weights=weight_hashes, mixed=mixed, rounds=results[:STEPS])
            if any(a[0] == b[0] or a[1] == b[1] for a, b in zip(results[:STEPS-1], results[1:STEPS])):
                raise ValueError("Adjacent rounds reused input or output")
            if args.reference:
                oracle = json.loads((args.reference / f"rank_{rank}/result.json").read_text())
                if record != oracle or (replay and [results[j] for j in TAIL] != results[STEPS:]):
                    raise ValueError(f"Bitwise baseline mismatch on rank {rank}")
            (dest / "result.json").write_text(json.dumps(record))
    buf.destroy()
    root.destroy()


def main():
    from deep_ep_experimental.megamoe import StoreGroup, golden
    from deep_ep_experimental.megamoe.benchmark import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("golden", "fixed", "eager", "replay", "perf"))
    parser.add_argument("--experts", type=int, choices=(384, 416), default=416)
    parser.add_argument("--bs", type=int, choices=(96, 128), default=96)
    parser.add_argument("--baseline-b78fd2c", action="store_true", help="Expect the E384 fallback on the original wheel")
    parser.add_argument("--input", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--bank", type=Path, help="Existing fixed input bank, overrides generated inputs")
    for key in ("golden", "reference", "out"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    config = dict(load_config("ep32_bs96", str(args.bs)), shared_fixture="native-fp8")
    config["model"]["num_experts"] = args.experts
    if args.mode == "golden":
        if not args.golden:
            parser.error("golden requires --golden")
        golden.prepare(config, WORLD, args.bs, list(range(WORLD)), args.golden, input_kind=args.input)
        return
    if not args.out or (args.mode != "perf" and not (args.bank or args.golden)) or (args.mode == "replay" and not args.reference):
        parser.error("Specify --out, --bank or --golden for accuracy, and --reference for replay")
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    group = StoreGroup.from_env()
    try:
        run(args, config, group, torch)
    except BaseException as exc:
        group.abort(str(exc))
        raise


if __name__ == "__main__":
    main()
