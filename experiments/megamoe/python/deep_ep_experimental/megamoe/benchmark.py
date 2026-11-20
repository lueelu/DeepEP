# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Installed-package accuracy/performance/trace runner. Launch with torchrun, not st/run.sh."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import lru_cache
import random
import threading
from importlib.resources import files
import json
import os
from pathlib import Path

from .timer.__main__ import parse_ranks
from .performance import PERF_METHODS, SLEEP_CYCLES, alignment_config, collect_profile, enqueue_batch, summarize_rounds


def loaded_libraries(buffer):
    from . import _platform_api, api

    module = _platform_api if isinstance(buffer, _platform_api.SymmBuffer) else api
    return _library_hashes(Path(module._extension().__file__), Path("/proc/self/maps"))


def _library_hashes(extension, maps):
    import hashlib

    paths = {extension.resolve()}
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[-1]).is_absolute():
            path = Path(fields[-1])
            if (
                path.is_file()
                and ".so" in path.name
                and any(key in path.name.lower() for key in ("shmem", "megamoe", "mega_moe", "deep_ep"))
            ):
                paths.add(path)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def load_config(name, bs_values=None):
    path = Path(name)
    if not path.is_file():
        path = files(__package__).joinpath("configs", name + ".json")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "model", "bs_values", "warmup", "iters", "seed"}
    if (
        not required <= set(config)
        or set(config) - required - {"alignment", "activation_clamp", "shared_dtype", "weight_mode"}
        or config["schema_version"] != 1
    ):
        raise ValueError("Invalid config schema; see the packaged smoke.json")
    if set(config["model"]) - {"shared_intermediate_hidden", "shared_output_rounding", "activation_clamp", "topk_weights_type"} != {
        "hidden",
        "intermediate_hidden",
        "num_experts",
        "num_topk",
        "num_shared_experts",
    }:
        raise ValueError("Model must specify H, intermediate width, global experts, top-k and shared experts")
    from .numerics import normalize_clamp
    normalize_clamp(config["model"].get("activation_clamp"))
    route_mode = config["model"].get("topk_weights_type", 0)
    if type(route_mode) is not int or route_mode not in (0, 1):
        raise ValueError("topk_weights_type must be 0 or 1")
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
    if config.get("shared_dtype", "fp4") not in ("fp4", "fp8") or config.get("weight_mode", "tensor") not in (
        "tensor",
        "list",
    ):
        raise ValueError("Unknown shared_dtype or weight_mode")
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


def _check_eplb_replays(
    group, config, rank, world, target, draft, prepared, activation, shared_kwargs, target_capture, raw_groups
):
    """Replace physical expert bindings after capture and alternate two plans on one arena."""
    import torch
    from .api import fp8_fp4_mega_moe
    from .weights import prepare_weights_for_mega_moe
    from .validation import collective_check

    target_graph, target_y, target_stats = target_capture
    bs = target_y.shape[0]
    draft_bs = max(1, bs // 2)
    draft_y = torch.empty_like(target_y[:draft_bs])
    draft_stats = torch.zeros_like(target_stats)
    draft_activation = None if activation is None else activation[:draft_bs].clone()
    if activation is None:
        draft.x[:draft_bs].copy_(target.x[:draft_bs])
        draft.x_sf[:draft_bs].copy_(target.x_sf[:draft_bs])

    owner = prepared[0].owner
    slots = list(range(owner.count))
    original = {layout: owner.expert_views(layout=layout) for layout in ("planar", "fused")}
    tables = [owner.select(f)[2] for f in (False, True)]
    table_values = [table.clone() for table in tables]

    def oracle_for(order):
        def projection(pair):
            return [(pair[0][i], pair[1][i]) for i in order]

        return prepare_weights_for_mega_moe(
            projection(original["fused"][0]),
            projection(original["planar"][1]),
            input_layout="prepared",
            planar_l1=projection(original["planar"][0]),
        )

    oracle = oracle_for(slots[1:] + slots[:1])

    def launch(view, act, y, stats, handles):
        fp8_fp4_mega_moe(
            y,
            *handles,
            view,
            x=act,
            cumulative_local_expert_recv_stats=stats,
            activation_clamp=config.get("activation_clamp", config["model"].get("activation_clamp")),
            **shared_kwargs,
        )

    def set_routes(view, size, pattern, remap):
        _, ids, weights = inputs(config, rank, size, pattern, world=world)
        if remap:
            ids = ids // owner.count * owner.count + (ids % owner.count - 1) % owner.count
        view.topk_idx[:size].copy_(torch.from_numpy(ids))
        view.topk_weights[:size].copy_(torch.from_numpy(weights))

    set_routes(draft, draft_bs, "skewed", False)
    launch(draft, draft_activation, draft_y, draft_stats, prepared)
    torch.npu.synchronize()
    draft_graph = torch.npu.NPUGraph()
    with torch.npu.graph(draft_graph):
        launch(draft, draft_activation, draft_y, draft_stats, prepared)
    torch.npu.synchronize()  # Existing graph readers have finished before publication.
    # Replay observes tables rebuilt from raw ND slices. The eager oracle still reads
    # the original NZ allocations in the same physical slot order.
    owner.update_experts(slots, *(group[1:] + group[:1] for group in raw_groups), replace=True)
    set_routes(target, bs, "balanced", True)
    set_routes(draft, draft_bs, "skewed", True)
    cases = (
        (target, activation, target_graph, target_y, target_stats),
        (draft, draft_activation, draft_graph, draft_y, draft_stats),
    )

    def check_replays(oracle, phase):
        expected = []
        for view, act, _, y, stats in cases:
            expected_y = torch.empty_like(y)
            expected_stats = torch.zeros_like(stats)
            launch(view, act, expected_y, expected_stats, (oracle.l1, oracle.l2))
            expected.append((expected_y, expected_stats))
        # No eager launches between replays: each graph must re-acquire the shared arena itself.
        for (view, _, graph, y, stats), (expected_y, expected_stats) in list(zip(cases, expected)) * 2:
            stats.zero_()
            y.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            ok = torch.equal(y, expected_y) and torch.equal(stats, expected_stats)
            collective_check(
                group,
                f"EPLB {phase} graph replay view={view._view_id} BS={y.shape[0]}",
                None if ok else "remapped output/count mismatch",
            )

    current = owner.expert_views(layout="planar")
    for fused, table, old in zip((False, True), tables, table_values):
        first, second = owner.expert_views(layout="fused" if fused else "planar")
        expected_rows = [[t.data_ptr() for t in group] for group in (first[0], second[0], first[1], second[1])]
        assert owner.select(fused)[2].data_ptr() == table.data_ptr()
        assert table.tolist() == expected_rows and (table != old).all().item()
    check_replays(oracle, "replace")

    # Old allocations are independent eager inputs. Restore their values in the new
    # allocations through exactly the objects available to framework callers.
    published = [table.clone() for table in tables]
    for dst_pair, src_pair in zip(current, original["planar"]):
        for destinations, sources in zip(dst_pair, src_pair):
            for dst, src in zip(destinations, sources):
                dst.copy_(src)
    owner.refresh_gmm1(slots, source_layout="planar")
    for fused, table, before in zip((False, True), tables, published):
        assert owner.select(fused)[2].data_ptr() == table.data_ptr() and torch.equal(table, before)
    baseline = oracle_for(slots)
    check_replays(baseline, "external write")
    first, second = baseline.expert_views(layout="fused")
    planar, _ = baseline.expert_views(layout="planar")
    owner.update_experts(slots, first, second, planar_l1=planar, input_layout="prepared", replace=True)
    torch.npu.synchronize()
    owner.release_retired()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="smoke", help="Packaged name or JSON path")
    parser.add_argument("--bs-values", help="Only model config override, e.g. 96,4096")
    parser.add_argument("--mode", choices=("accuracy", "perf", "trace"), default="accuracy")
    parser.add_argument(
        "--perf-method",
        choices=PERF_METHODS,
        default="legacy",
        help="legacy uses an Exp prefix; legacy-read adds a read-only L2 sum; legacy-zero clears L2; shmem uses a per-round device barrier",
    )
    parser.add_argument("--waves", help="Comma-separated overrides or all for 0..64 (perf mode)")
    parser.add_argument("--retest-top", type=int, default=0, help="Repeat the fastest N wave settings")
    parser.add_argument(
        "--wave", type=int, choices=range(65), default=0, metavar="0..64", help="0 selects the native default"
    )
    parser.add_argument("--check-graph", action="store_true", help="Compare changed-input graph replays against eager")
    parser.add_argument("--input", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument(
        "--routing",
        choices=("balanced", "random", "skewed", "random-balanced"),
        default="balanced",
        help="Reproducible accuracy stress pattern; record the same pattern for performance comparisons",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory; must not already exist")
    parser.add_argument("--timer", action="store_true", help="Collect device timers; always enabled in trace mode")
    parser.add_argument("--timer-ranks", type=parse_ranks, help="Default: global ranks 0..min(7,EP-1)")
    parser.add_argument("--check-tokens", type=int, default=0, help="Accuracy samples per rank; default 0 checks all")
    parser.add_argument(
        "--golden-dir", type=Path, default=Path("benchmark-golden"), help="Precomputed CPU Golden cache"
    )
    parser.add_argument("--relative-rmse", type=float, default=0.02, help="Timer on/off output tolerance only")
    parser.add_argument("--dual-reference", action="store_true", help="Compare CPU Golden and NPU cascade")
    args = parser.parse_args()
    args.timer = args.timer or args.mode == "trace"
    if args.perf_method != "legacy" and args.mode == "accuracy" and not args.timer:
        parser.error("--perf-method shmem requires perf, trace, or --timer")
    if args.waves:
        try:
            args.waves = list(range(65)) if args.waves == "all" else [int(n) for n in args.waves.split(",")]
        except ValueError:
            parser.error("--waves accepts integers or all")
        if args.mode != "perf" or args.timer or args.wave or any(not 0 <= n <= 64 for n in args.waves):
            parser.error("wave sweeps require values 0..64 and timer-off perf")
    if not 0 <= args.retest_top <= len(set(args.waves or [])) or (args.check_graph and args.mode != "accuracy"):
        parser.error("invalid retest count or graph mode")
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

    capacity = max(config["bs_values"])
    ranks_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    backend = select_backend(group, capacity, ranks_per_node)
    urma = backend == "urma"
    if urma or config["model"].get("topk_weights_type", 0):
        from . import _platform_benchmark

        return _platform_benchmark.run(args, config, group, backend)
    import torch
    from .api import SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe
    from .reference import make_weights, quantize_input
    from .validation import collective_check

    rank, world = group.rank(), group.size()
    hierarchical = not urma and ranks_per_node == 16 and world == 32
    collective_check(
        group,
        "benchmark configuration",
        None,
        (
            config,
            args.mode,
            args.perf_method,
            args.input,
            args.routing,
            args.timer,
            args.timer_ranks,
            args.check_tokens,
            args.relative_rmse,
            args.wave,
            args.waves,
            args.retest_top,
            args.check_graph,
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
        perf_method=args.perf_method,
        iteration_sync="shmem_world_on_stream" if args.perf_method == "shmem" else "none",
        alignment_scope=("before_each_iteration" if args.perf_method == "shmem" else "once_before_batch")
        if args.mode == "perf" or args.timer
        else "none",
        device_communication=(
            "shmem_urma" if urma else "shmem_mte_roce" if hierarchical else "shmem_mte"
        ),
        api="fp8_fp4_mega_moe",
        numerical_contract=(
            "ascend_post_gmm2_host_reduce_fp32_v1" if hierarchical else
            "ascend_pre_quant_route_weight_v1" if config["model"].get("topk_weights_type", 0) else "ascend_post_gmm2_route_weight_v1"
        ),
        implementation=backend,
        created_utc=datetime.now(timezone.utc).isoformat(),
        build=json.loads(files("deep_ep").joinpath("_build_info.json").read_text()),
    )
    model = config["model"]
    check_eplb = (
        args.check_graph and hierarchical and config.get("weight_mode") == "list" and model["num_experts"] > world
    )
    buffer_options = {k: v for k, v in model.items() if k not in ("shared_intermediate_hidden", "shared_output_rounding", "activation_clamp")}
    buffer = SymmBuffer(
        group,
        num_max_tokens_per_rank=capacity,
        ranks_per_node=ranks_per_node,
        num_max_configs=max(16, len(config["bs_values"]) * (2 if check_eplb else 1)) if hierarchical else 1,
        **buffer_options,
    )

    draft = None
    if check_eplb:
        draft = SymmBuffer(
            group,
            num_max_tokens_per_rank=max(1, max(config["bs_values"]) // 2),
            ranks_per_node=buffer.ranks_per_node,
            base=buffer,
            **buffer_options,
        )
        for size in sorted({max(1, bs // 2) for bs in config["bs_values"]}):
            draft.prepare(size)
            draft.set_wave(args.wave)
    if hierarchical:
        compiled = buffer._roce_config
        manifest["compiled_roce_config"] = compiled
        manifest["implementation"] = f"dispatch_qp{compiled['dispatch_qps']}_combine_qp{compiled['combine_qps']}"
    (rank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if buffer._domains.configs > 1:
        for bs in config["bs_values"]:
            buffer.prepare(bs)

    def device_pair(pair, fp8=False):
        data, scale = pair
        if fp8:
            from .reference import fp4_to_fp8

            data = torch.from_numpy(fp4_to_fp8(data)).npu().view(torch.float8_e4m3fn)
        else:
            data = torch.from_numpy(data).npu()
        return (data, torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu))

    if args.timer and not buffer.timer_enabled:
        raise RuntimeError("Timer requires a W4A8 wheel built with DEEPEP_MEGAMOE_TIMER=ON")
    raw = make_weights(
        model["num_experts"] // world, model["hidden"], model["intermediate_hidden"], config["seed"] + rank
    )

    def prepare(raw, shared=False):
        from .weights import prepare_weights_for_mega_moe

        pairs = tuple(device_pair(pair, shared and config.get("shared_dtype") == "fp8") for pair in raw)
        if config.get("weight_mode") == "list":
            groups = [
                [
                    (
                        (w[i : i + 1] if args.check_graph and i % 2 == 0 else w[i]).clone(),
                        (s[i : i + 1] if args.check_graph and i % 2 else s[i]).clone(),
                    )
                    for i in range(w.shape[0])
                ]
                for w, s in pairs
            ]
            owner = prepare_weights_for_mega_moe(*groups, shared=shared)
            return owner.l1, owner.l2
        return transform_weights_for_mega_moe(*pairs, shared=shared)

    prepared = prepare(raw)
    shared_raw, shared_kwargs = None, {}
    if model["num_shared_experts"]:
        shared_raw = make_weights(
            model["num_shared_experts"],
            model["hidden"],
            model.get("shared_intermediate_hidden", model["intermediate_hidden"]),
            config["seed"] + 100000,
        )
        shared = prepare(shared_raw, shared=True)
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
        buffer.set_wave(args.wave)
        (case / "resources.json").write_text(
            json.dumps(
                dict(
                    workspace_bytes=buffer.workspace_bytes,
                    symmetric_buffer_bytes=buffer.symmetric_buffer_bytes,
                    prepared_bs=bs,
                    max_tokens=buffer.num_max_tokens_per_rank,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        result = torch.empty((bs, model["hidden"]), device="npu", dtype=torch.bfloat16)

        def call(timer=None, stats=None, output=None):
            fp8_fp4_mega_moe(
                result if output is None else output,
                *prepared,
                buffer,
                cumulative_local_expert_recv_stats=stats,
                x=activation,
                timer=timer,
                activation_clamp=config.get("activation_clamp", model.get("activation_clamp")),
                **shared_kwargs,
            )

        prefix, flush_buffer = None, None
        batch_options = dict(method=args.perf_method)
        measurement = dict(
            perf_method=args.perf_method,
            iteration_sync="none",
            alignment_scope="once_before_batch",
            alignment=config["alignment"],
            l2_flush_bytes=0,
            sleep_cycles=0,
        )
        if args.mode == "perf" or args.timer:
            error = None
            try:
                if args.perf_method.startswith("legacy"):
                    alignment = config["alignment"]
                    prefix = torch.zeros((alignment["rows"], alignment["cols"]), dtype=torch.float32, device="npu")
                    torch.exp(prefix, out=prefix)  # 预热 Exp，不能把懒初始化带入采集批次。
                    if args.perf_method in ("legacy-read", "legacy-zero"):
                        l2 = int(torch.npu.get_device_properties(torch.npu.current_device()).L2_cache_size)
                        if l2 <= 0:
                            raise ValueError("Missing L2_cache_size")
                        flush_buffer = torch.zeros((2 * l2 + 3) // 4, dtype=torch.int32, device="npu")
                        batch_options.update(
                            flush_buffer=flush_buffer, sum_out=torch.empty((), dtype=torch.int32, device="npu")
                        )
                        measurement["l2_flush_bytes"] = flush_buffer.numel() * 4
                else:
                    properties = torch.npu.get_device_properties(torch.npu.current_device())
                    l2_bytes = int(properties.L2_cache_size)
                    if l2_bytes <= 0:
                        raise ValueError("SHMEM profiling requires a positive device L2_cache_size")
                    flush_buffer = torch.empty(2 * l2_bytes, dtype=torch.uint8, device="npu")
                    flush_buffer.zero_()
                    buffer.benchmark_sleep(0)  # Warm up the helper before capture.
                    batch_options.update(
                        barrier=buffer.benchmark_barrier, sleep=buffer.benchmark_sleep, flush_buffer=flush_buffer
                    )
                    measurement.update(
                        iteration_sync="shmem_world_on_stream",
                        alignment_scope="before_each_iteration",
                        alignment=None,
                        l2_flush_bytes=2 * l2_bytes,
                        sleep_cycles=SLEEP_CYCLES,
                        shmem_api="aclshmemx_roce_barrier_all"
                        if buffer._hierarchical_roce
                        else "aclshmemx_barrier_all_on_stream",
                    )
            except Exception as failure:
                error = f"rank={rank} BS={bs} perf_method={args.perf_method}: {failure}"
            collective_check(group, f"alignment allocation BS={bs}", error)
            if args.perf_method == "shmem":
                torch.npu.synchronize()
                group.barrier(f"SHMEM profiling helpers ready BS={bs}")
                buffer.benchmark_barrier()  # Resolve first-use launch costs outside capture.
                torch.npu.synchronize()
            (case / "measurement.json").write_text(json.dumps(measurement, indent=2), encoding="utf-8")
            if rank == 0:
                stage = "TRACE" if args.mode == "trace" else "PROFILE"
                print(
                    f"[{stage}] BS={bs} method={args.perf_method} measurement={measurement} "
                    f"batch_iterations={config['iters']}",
                    flush=True,
                )
        if args.mode != "trace":
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
            if args.check_graph:
                static_stats = torch.zeros_like(stats)
                expected_y = torch.empty_like(result)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    call(stats=static_stats)
                for replay in range(8):
                    variant = dict(config, seed=config["seed"] + 1 + replay)
                    vx, vi, vw = inputs(variant, rank, bs, "random" if replay % 2 == 0 else "skewed", world=world)
                    if activation is None:
                        vd, vs = quantize_input(vx)
                        buffer.x[:bs].copy_(torch.from_numpy(vd).npu().view(torch.float8_e4m3fn))
                        buffer.x_sf[:bs].copy_(torch.from_numpy(vs).npu().view(torch.float8_e8m0fnu))
                    else:
                        activation.copy_(torch.from_numpy(vx))
                    buffer.topk_idx[:bs].copy_(torch.from_numpy(vi))
                    buffer.topk_weights[:bs].copy_(torch.from_numpy(vw))
                    eager_stats = torch.zeros_like(stats)
                    call(stats=eager_stats, output=expected_y)
                    result.fill_(float("nan"))
                    static_stats.zero_()
                    graph.replay()
                    torch.npu.synchronize()
                    ok = torch.equal(expected_y, result) and torch.equal(eager_stats, static_stats)
                    collective_check(group, f"changed input replay {replay}", None if ok else "output/count mismatch")
                if draft is not None:
                    _check_eplb_replays(
                        group,
                        config,
                        rank,
                        world,
                        buffer,
                        draft,
                        prepared,
                        activation,
                        shared_kwargs,
                        (graph, result, static_stats),
                        [
                            [(w[i] if i % 2 else w[i : i + 1], s[i]) for i in range(w.shape[0])]
                            for w, s in (device_pair(pair) for pair in raw)
                        ],
                    )
                    (case / "eplb_graph.json").write_text(
                        json.dumps(
                            dict(
                                passed=True, phases=["replace", "external write"], replay_bs=[bs, max(1, bs // 2)] * 2
                            ),
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                # Restore the original case for the independent CPU golden.
                if activation is None:
                    buffer.x[:bs].copy_(torch.from_numpy(data).npu().view(torch.float8_e4m3fn))
                    buffer.x_sf[:bs].copy_(torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu))
                else:
                    activation.copy_(torch.from_numpy(x))
                buffer.topk_idx[:bs].copy_(torch.from_numpy(ids))
                buffer.topk_weights[:bs].copy_(torch.from_numpy(weights))
                call()
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
        elif args.mode == "perf":

            def profile_wave(wave, target, warmup):
                target.mkdir(exist_ok=True)
                buffer.set_wave(wave)
                for _ in range(warmup):
                    call()
                torch.npu.synchronize()
                with torch.npu.stream(buffer._runtime.stream) if args.perf_method == "legacy-read" else nullcontext():
                    samples = collect_profile(
                        call, group, torch, config, prefix, target / "profiler", bs, **batch_options
                    )
                (target / "kernel_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
                timings = [sample["duration_us"] for sample in samples]
                per_rank = group.all_gather(timings, f"profiling BS={bs} wave={wave}")
                actual = buffer._plan.effective_wave
                collective_check(group, "effective wave", None, actual)
                report = summarize_rounds(per_rank, int(os.environ["LOCAL_WORLD_SIZE"]))
                report.update(
                    measurement, effective_wave=actual, requested_wave=wave, implementation=manifest["implementation"]
                )
                (target / "profiling.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                if rank == 0:
                    print(
                        f"[PERF] BS={bs} wave={actual} rank-min={report['mean_us']:.3f}us rank-max={report['rank_max_mean_us']:.3f}us",
                        flush=True,
                    )
                return report

            if args.waves:
                by_wave = {}
                for trial, wave in enumerate(args.waves):
                    directory = f"wave_{wave}" if wave not in by_wave else f"wave_{wave}_trial_{trial}"
                    report = profile_wave(wave, case / directory, config["warmup"])
                    by_wave.setdefault(wave, []).append(report["mean_us"])
                ordered = sorted(by_wave, key=lambda w: sum(by_wave[w]) / len(by_wave[w]))
                for wave in ordered[: args.retest_top]:
                    profile_wave(wave, case / f"retest_{wave}", config["warmup"])
            else:
                profile_wave(args.wave, case, 0)
        if args.timer:
            uninstrumented = result.float().cpu().numpy() if args.mode == "accuracy" else None
            timer, error = None, None
            try:
                timer = buffer.allocate_timer() if rank in timer_ranks else None
            except Exception as failure:
                error = str(failure)
            collective_check(group, f"timer allocation BS={bs}", error)
            torch.npu.synchronize()
            group.barrier(f"timer before warmup BS={bs}")
            for _ in range(config["warmup"]):
                call()
            torch.npu.synchronize()
            group.barrier(f"timer batch BS={bs}")
            with torch.npu.stream(buffer._runtime.stream) if args.perf_method == "legacy-read" else nullcontext():
                enqueue_batch(call, torch, prefix, config["alignment"], config["iters"], timer, **batch_options)
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
        effective = buffer._plan.effective_wave
        collective_check(group, f"final effective wave BS={bs}", None, effective)
        manifest.setdefault("effective_waves", {})[str(bs)] = effective
        manifest["loaded_libraries"] = loaded_libraries(buffer)
        (rank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        batch_options.clear()
        prefix = flush_buffer = None
        group.barrier(f"case complete BS={bs}")
    group.barrier("before SHMEM teardown")
    if draft is not None:
        draft.destroy()
    buffer.destroy()
    group.close()


def _export_mte_timer(output_dir, bs, rank, group, timer):
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
