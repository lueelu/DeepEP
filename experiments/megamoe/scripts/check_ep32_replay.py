"""Run this same file with each installed wheel. Only 'bank' needs the candidate generators."""

import argparse
import hashlib
from importlib.resources import files
import json
import os
import platform
from pathlib import Path

import numpy as np


WORLD, BS, STEPS, WAVE = 32, 96, 160, 5
MODEL = dict(hidden=5120, intermediate_hidden=2304, num_experts=416, num_topk=6, num_shared_experts=1)
LOCAL = MODEL["num_experts"] // WORLD
TAIL = (159, 0, 80, 1, 158, 2, 79, 3)
INPUTS = ("x", "scale", "ids", "coeff")


def digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def put_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_round(config, rank, step, *, world=WORLD, bs=BS, input_kind="fp8"):
    from deep_ep_experimental.megamoe.benchmark import inputs
    from deep_ep_experimental.megamoe.reference import quantize_input

    seed = config["seed"] + 65537 * step
    x, ids, coeff = inputs(dict(config, seed=seed), rank, bs, "balanced", world=world)
    ids = (ids + step) % config["model"]["num_experts"]
    if input_kind == "bf16":
        return (x.view(np.uint32) >> 16).astype(np.uint16), ids, coeff
    codes, scale = quantize_input(x)
    return codes, scale, ids, coeff


def bank(args):
    from deep_ep_experimental.megamoe import golden
    from deep_ep_experimental.megamoe.benchmark import load_config
    from deep_ep_experimental.megamoe.reference import make_weights, fp4_to_fp8

    config = dict(load_config("ep32_bs96"), shared_fixture=args.shared_fixture)
    if config["model"] != MODEL or config["activation_clamp"] != 10 or config["shared_dtype"] != "fp8":
        raise ValueError("Unexpected EP32 fixture configuration")
    ranks = list(range(args.first_rank, args.end_rank))
    spec = golden.prepare(config, WORLD, BS, ranks, args.golden, input_kind=args.input, workers=args.workers)
    for rank in ranks:
        dest = args.bank / f"rank_{rank}"
        dest.mkdir(parents=True, exist_ok=False)
        output, counts, _ = golden.load(args.golden, spec, rank)
        np.savez(dest / "golden.npz", output=output, counts=counts)
        weights = {"routed": [], "shared": []}
        for role, n, seed in (("routed", LOCAL, config["seed"] + rank), ("shared", 1, config["seed"] + 100000)):
            native = role == "shared" and args.shared_fixture == "native-fp8"
            for j, (w, sf) in enumerate(
                make_weights(n, MODEL["hidden"], MODEL["intermediate_hidden"], seed, native_fp8=native)
            ):
                if role == "shared" and not native:
                    w = fp4_to_fp8(w)
                weights[role].append(digest((w, sf)))
                np.savez(dest / f"{role}{j}.npz", w=w, sf=sf)
        hashes = []
        for step in range(STEPS):
            values = make_round(config, rank, step, world=WORLD, bs=BS, input_kind=args.input)
            hashes.append(digest(values))
            np.savez(dest / f"input_{step}.npz", **dict(zip(input_fields(args.input), values, strict=True)))
        if len(set(hashes)) != STEPS:
            raise ValueError(f"Duplicate inputs on rank {rank}")
        put_json(
            dest / "bank.json",
            dict(
                rank=rank,
                config=config,
                input=args.input,
                golden_sha256=digest((output, counts)),
                hashes=hashes,
                weights=weights,
            ),
        )


def input_fields(kind):
    if kind == "fp8":
        return INPUTS
    if kind == "bf16":
        return ("bf16", "ids", "coeff")
    raise ValueError(f"Unknown input dtype {kind}")


def interleave_rows(array):
    gate, up = np.split(array, 2)
    shape = (-1, 128, *array.shape[1:])
    return np.stack((gate.reshape(shape), up.reshape(shape)), axis=1).reshape(array.shape)


def load_weights(folder, role, torch, prepare, torch_npu):
    groups, hashes = [], []
    planar = ([], [])
    shared = role == "shared"

    def upload(weight, scale):
        w = torch.from_numpy(np.ascontiguousarray(weight)).npu()
        if shared:
            w = w.view(torch.float8_e4m3fn)
        else:
            w = torch_npu.npu_format_cast(
                w, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
            )
        return w, torch.from_numpy(np.ascontiguousarray(scale)).npu().view(torch.float8_e8m0fnu)

    for j in range(2):
        with np.load(folder / f"{role}{j}.npz") as data:
            w, sf = data["w"], data["sf"]
        hashes.append(digest((w, sf)))
        weights, scales = [], []
        for wi, si in zip(w, sf, strict=True):
            if j == 0:
                if not shared:
                    pw, ps = upload(wi, si)
                    planar[0].append(pw)
                    planar[1].append(ps)
                wi, si = interleave_rows(wi), interleave_rows(si)
            wi, si = upload(wi, si)
            weights.append(wi)
            scales.append(si)
        groups.append((weights, scales))
    owner = prepare(*groups, shared=shared, input_layout="prepared", planar_l1=None if shared else planar)
    return (owner.l1, owner.l2), hashes


def route_histogram(ids, experts):
    if (
        ids.dtype != np.int32
        or np.any(ids < 0)
        or np.any(ids >= experts)
        or np.any(np.diff(np.sort(ids), axis=-1) == 0)
    ):
        raise ValueError("Expected distinct physical expert IDs per token in range")
    return np.bincount(ids.ravel(), minlength=experts)


def load_inputs(folder, rank, group, *, fixed=False):
    meta = json.loads((folder / "bank.json").read_text())
    if meta["rank"] != rank or len(meta["hashes"]) != STEPS or len(set(meta["hashes"])) != STEPS:
        raise ValueError("Invalid input bank identity or duplicate inputs")
    inputs, counts = [], []
    for step, expected in enumerate(meta["hashes"][:1] if fixed else meta["hashes"]):
        with np.load(folder / f"input_{step}.npz") as data:
            arrays = tuple(data[key] for key in input_fields(meta.get("input", "fp8")))
        if digest(arrays) != expected:
            raise ValueError(f"Input checksum mismatch at rank={rank} step={step}")
        inputs.append(arrays)
        hist = route_histogram(arrays[-2], MODEL["num_experts"])
        gathered = group.all_gather(hist.tolist(), f"physical routes {step}")
        counts.append(np.sum(gathered, axis=0)[rank * LOCAL : (rank + 1) * LOCAL])
    return inputs, meta, np.stack(counts)


def set_wave(buf, reference):
    if reference:
        target = buf if hasattr(buf, "set_wave_groups_for_benchmark") else buf._plan
        target.set_wave_groups_for_benchmark(WAVE)
    else:
        buf.set_wave(WAVE)


def effective_wave(buf, reference):
    if reference:
        target = buf if hasattr(buf, "wave_groups") else buf._plan
        value = target.wave_groups
        return value() if callable(value) else value
    return buf._plan.effective_wave


def compiled_configuration(core, extension, reference):
    fields = ("combine_reduce_version", "combine_qps", "dispatch_qps", "dispatch_workers")
    compiled = {key: getattr(extension, key, None) for key in fields}
    protocol = core.megamoe_combine_reduce_version
    compiled["core_combine_reduce_version"] = protocol
    if (reference and protocol != 2) or (not reference and (
        protocol not in (4, 5) or compiled["combine_reduce_version"] != protocol
    )):
        raise ValueError(f"Unexpected compiled EP32 configuration {compiled}")
    return compiled


def run(args, group, torch, torch_npu):
    from deep_ep import _C as core
    from deep_ep_experimental.megamoe import SymmBuffer, fp8_fp4_mega_moe, prepare_weights_for_mega_moe
    from deep_ep_experimental.megamoe.api import _extension
    _C = _extension()

    rank = group.rank()
    if group.size() != WORLD or int(os.environ["LOCAL_WORLD_SIZE"]) != 16:
        raise ValueError("Requires two hosts with 16 ranks each")
    folder, dest = (p / f"rank_{rank}" for p in (args.bank, args.out))
    dest.mkdir(parents=True, exist_ok=False)
    inputs, bank_meta, expected_counts = load_inputs(folder, rank, group, fixed=args.fixed_perf)
    routed, rh = load_weights(folder, "routed", torch, prepare_weights_for_mega_moe, torch_npu)
    shared, sh = load_weights(folder, "shared", torch, prepare_weights_for_mega_moe, torch_npu)
    if bank_meta["weights"] != {"routed": rh, "shared": sh}:
        raise ValueError(f"Weight checksum mismatch on rank {rank}")
    if any(peer != sh for peer in group.all_gather(sh, "shared weights")):
        raise ValueError("Shared FP8 bits/scales differ between ranks")
    if args.fixed_perf and (
        bank_meta.get("input") != "bf16"
        or bank_meta["config"].get("shared_fixture") != "native-fp8"
        or _C.timer_enabled
    ):
        raise ValueError("Fixed profiling requires a BF16 native-shared bank and timer-off wheel")
    reference = args.mode == "reference"
    buf = SymmBuffer(group, ranks_per_node=16, num_max_tokens_per_rank=BS, num_max_configs=16, **MODEL)
    buf.prepare(BS)
    if not args.default_wave:
        set_wave(buf, reference)
    compiled = compiled_configuration(core, _C, reference)
    y = torch.empty((BS, MODEL["hidden"]), device="npu", dtype=torch.bfloat16)
    stats = torch.zeros(LOCAL, device="npu", dtype=torch.int32)

    kind = bank_meta.get("input", "fp8")
    activation = torch.empty_like(y) if kind == "bf16" else None

    def install(step):
        *features, ids, coeff = (torch.from_numpy(a) for a in inputs[step])
        if activation is None:
            x, sf = features
            buf.x[:BS].copy_(x.view(torch.float8_e4m3fn))
            buf.x_sf[:BS].copy_(sf.view(torch.float8_e8m0fnu))
        else:
            activation.copy_(features[0].view(torch.bfloat16))
        buf.topk_idx[:BS].copy_(ids)
        buf.topk_weights[:BS].copy_(coeff)

    def forward(counts=stats, x=activation):
        fp8_fp4_mega_moe(
            y,
            *routed,
            buf,
            x=x,
            shared_l1_weights=shared[0],
            shared_l2_weights=shared[1],
            activation_clamp=10,
            cumulative_local_expert_recv_stats=counts,
        )

    stream = buf._runtime.stream
    torch.npu.synchronize()
    replay = args.mode == "replay"
    with torch.npu.stream(stream):
        install(0)
        for _ in range(20):
            forward(None if args.fixed_perf else stats)
        stream.synchronize()
        wave = effective_wave(buf, reference)
        bf16_bs96_active = getattr(getattr(buf, "_plan", None), "bf16_bs96_active", None)
        if compiled["core_combine_reduce_version"] == 5 and bf16_bs96_active != (kind == "bf16"):
            raise ValueError(f"Unexpected BS96 schedule for {kind} input: {bf16_bs96_active}")
        if wave != WAVE:
            raise ValueError(f"Unexpected effective wave {wave}")
        mixed_input_checked = bf16_bs96_active is True and not args.fixed_perf
        if mixed_input_checked:
            from deep_ep_experimental.megamoe.reference import quantize_input

            bits = inputs[0][0].astype(np.uint32) << 16
            codes, scales = quantize_input(bits.view(np.float32))
            seen = {}
            valid = True
            for use_bf16 in (True, False, True, False, True):
                if not use_bf16:
                    buf.x[:BS].copy_(torch.from_numpy(codes).view(torch.float8_e4m3fn))
                    buf.x_sf[:BS].copy_(torch.from_numpy(scales).view(torch.float8_e8m0fnu))
                y.fill_(float("nan"))
                stats.zero_()
                forward(x=activation if use_bf16 else None)
                output = y.float().cpu().numpy()
                valid &= buf._plan.bf16_bs96_active == use_bf16
                valid &= np.isfinite(output).all() and np.array_equal(stats.cpu().numpy(), expected_counts[0])
                if use_bf16 in seen:
                    valid &= np.array_equal(output.view(np.uint32), seen[use_bf16])
                else:
                    seen[use_bf16] = output.view(np.uint32).copy()
            if not all(group.all_gather(bool(valid), "mixed BF16/FP8 schedule and outputs")):
                raise ValueError("Mixed BF16/FP8 calls changed schedule, output, or counts on the same buffer")
        if replay:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                forward()
        iterations = (0,) if args.fixed_perf else [*range(STEPS), *(TAIL if replay else ())]
        for i, step in enumerate(iterations):
            install(step)
            stats.zero_()
            y.fill_(float("nan"))
            if replay and (i < STEPS or (i - STEPS) % 2 == 1):
                graph.replay()
            else:
                forward()
            out = y.float().cpu().numpy()
            counts = stats.cpu().numpy()
            np.savez(dest / f"result_{i}.npz", output=out, counts=counts)
        if args.fixed_perf:
            profile_fixed(torch, group, forward, dest)
    np.save(dest / "physical_counts.npy", expected_counts)
    libs = {Path(_C.__file__).resolve(), Path(core.__file__).resolve()}
    for line in Path("/proc/self/maps").read_text().splitlines():
        entry = line.split(maxsplit=5)
        if len(entry) == 6 and entry[-1].startswith("/"):
            lib = Path(entry[-1])
            if (
                lib.is_file()
                and ".so" in lib.name
                and any(k in lib.name.lower() for k in ("megamoe", "mega_moe", "shmem", "deep_ep"))
            ):
                libs.add(lib)
    build = json.loads(files("deep_ep").joinpath("_build_info.json").read_text())
    cann = (Path(os.environ["ASCEND_HOME_PATH"]) / "compiler/version.info").read_text().strip()
    props = torch.npu.get_device_properties(torch.npu.current_device())
    cores = {key: int(getattr(props, key)) for key in ("cube_core_num", "vector_core_num") if hasattr(props, key)}
    shmem = Path(core.shmem_library_path()).resolve()
    libs.add(shmem)
    libraries = {}
    for path in sorted(libs):
        with path.open("rb") as stream:
            libraries[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    expected_shmem = build["shmem"]["sha256"]["lib/libshmem.so"]
    if libraries[str(shmem)] != expected_shmem:
        raise ValueError(f"Loaded SHMEM differs from wheel build on rank {rank}: {shmem}")
    identity = [
        args.mode,
        args.revision,
        compiled,
        bf16_bs96_active,
        wave,
        sorted(libraries.values()),
        platform.python_version(),
        bool(_C.timer_enabled),
        str(torch.__version__),
        str(torch_npu.__version__),
        cann,
        cores,
    ]
    if any(peer != identity for peer in group.all_gather(identity, "wheel and runtime identity")):
        raise ValueError("Ranks loaded different MegaMoE builds or execution configurations")
    put_json(
        dest / "run.json",
        dict(
            rank=rank,
            mode=args.mode,
            revision=args.revision,
            model=MODEL,
            wave=wave,
            bs=BS,
            clamp=10,
            input=kind,
            fixed_perf=args.fixed_perf,
            default_wave=args.default_wave,
            timer_enabled=bool(_C.timer_enabled),
            compiled=compiled,
            bf16_bs96_active=bf16_bs96_active,
            mixed_input_checked=mixed_input_checked,
            hashes=bank_meta["hashes"],
            routed=rh,
            shared=sh,
            build=build,
            torch=str(torch.__version__),
            torch_npu=str(torch_npu.__version__),
            device=torch.npu.get_device_name(),
            device_properties=str(props),
            cores=cores,
            cann=cann,
            libraries=libraries,
            shmem_library=str(shmem),
            python=platform.python_version(),
        ),
    )
    if replay:
        del graph
    group.barrier("before replay validation teardown")
    buf.destroy()
    group.close()


def profile_fixed(torch, group, forward, dest):
    # Run the same profiler/parser against both wheels, including older installations.
    import runpy

    helpers = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "python/deep_ep_experimental/megamoe/performance.py")
    )
    alignment = helpers["DEFAULT_ALIGNMENT"]
    prefix = torch.zeros((alignment["rows"], alignment["cols"]), device="npu", dtype=torch.float32)
    torch.exp(prefix, out=prefix)
    flush = torch.zeros((256 * 1024**2) // 4, device="npu", dtype=torch.int32)
    total = torch.empty((), device="npu", dtype=torch.int32)
    torch.sum(flush, dim=(0,), dtype=torch.int32, out=total)
    samples = helpers["collect_profile"](
        lambda _: forward(None),
        group,
        torch,
        dict(iters=30, alignment=alignment),
        prefix,
        dest / "profiler",
        BS,
        method="legacy-read",
        flush_buffer=flush,
        sum_out=total,
    )
    put_json(dest / "kernel_samples.json", samples)
    all_ranks = group.all_gather([s["duration_us"] for s in samples], "fixed BF16 kernel duration")
    if group.rank() == 0:
        report = helpers["summarize_rounds"](all_ranks, 16)
        put_json(
            dest.parent / "fixed_perf.json",
            dict(
                rank0_mean_us=report["rank0_mean_us"],
                rank_max_mean_us=report["rank_max_mean_us"],
                all_32x30_us=all_ranks,
                tail_rounds=15,
                flush_bytes=flush.numel() * 4,
                alignment=alignment,
                warmups=20,
            ),
        )


def differences(actual, expected, shape):
    if actual.shape != shape or expected.shape != shape:
        return dict(shape=[list(actual.shape), list(expected.shape)])
    finite = np.isfinite(actual) & np.isfinite(expected)
    different = ~finite | (actual != expected)
    if actual.dtype == expected.dtype == np.float32:
        different |= actual.view(np.uint32) != expected.view(np.uint32)
    if np.any(different):
        return dict(
            mismatched=int(different.sum()),
            nonfinite=int((~finite).sum()),
            max_abs=float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else None,
            relative_rmse=float(np.linalg.norm(actual[finite].astype(np.float64) - expected[finite]) /
                                max(np.linalg.norm(expected[finite].astype(np.float64)), 1e-30)) if finite.any() else None,
        )
    return {}


def compare_rank(args, rank):
    from deep_ep_experimental.megamoe.golden import compare as cpu_compare

    bank_dir = args.bank / f"rank_{rank}"
    baseline = getattr(args, "baseline_eager", None)
    oracle = "baseline-eager" if baseline else "reference"
    runs = {
        name: directory / f"rank_{rank}"
        for name, directory in (
            (oracle, baseline or args.reference),
            ("replay", args.replay),
            ("candidate-eager", args.candidate_eager),
        )
        if directory
    }
    meta = {key: json.loads((p / "run.json").read_text()) for key, p in runs.items()}
    bank_meta = json.loads((bank_dir / "bank.json").read_text())
    hashes = bank_meta["hashes"]
    fixed = args.fixed_perf
    shape = (BS, MODEL["hidden"])
    failures = []
    first = meta[oracle]
    for key, record in meta.items():
        if baseline:
            protocol, lanes = (4, 16) if key == oracle else (5, 24)
            expected = (protocol, protocol, lanes, 8, 14)
            fields = ("core_combine_reduce_version", "combine_reduce_version", "combine_qps",
                      "dispatch_qps", "dispatch_workers")
            if tuple(record.get("compiled", {}).get(f) for f in fields) != expected:
                failures.append(dict(reason="compiled protocol or QP mismatch", run=key))
            if key != oracle and record.get("bf16_bs96_active") != (record["input"] == "bf16"):
                failures.append(dict(reason="effective BF16 schedule mismatch", run=key))
            if key == "replay" and "candidate-eager" in meta:
                eager = meta["candidate-eager"]
                if any(record.get(f) != eager.get(f) for f in ("revision", "libraries", "compiled")):
                    failures.append(dict(reason="candidate eager and replay builds differ", run=key))
        if (
            record["mode"] != ("candidate-eager" if key == "baseline-eager" else key)
            or record["rank"] != rank
            or record.get("fixed_perf", False) != fixed
            or record["input"] != bank_meta.get("input", "fp8")
            or record["hashes"] != hashes
            or record["model"] != MODEL
            or record["bs"] != BS
            or record["wave"] != WAVE
            or record["clamp"] != 10
            or ("weights" in bank_meta and any(record[k] != bank_meta["weights"][k] for k in ("shared", "routed")))
            or (
                fixed
                and (
                    record["input"] != "bf16"
                    or record.get("timer_enabled") is not False
                    or bank_meta["config"].get("shared_fixture") != "native-fp8"
                )
            )
            or len(hashes) != STEPS
            or len(set(hashes)) != STEPS
        ):
            failures.append(dict(reason="run or input bank identity mismatch", run=key))
        for field in (
            "routed",
            "shared",
            "wave",
            "bs",
            "model",
            "clamp",
            "input",
            "device",
            "cores",
        ):
            if record[field] != first[field]:
                failures.append(dict(reason=f"different {field}", run=key))
    counts = {key: np.load(p / "physical_counts.npy") for key, p in runs.items()}
    for key, value in counts.items():
        if value.shape != (1 if fixed else STEPS, LOCAL) or not np.array_equal(value, counts[oracle]):
            raise ValueError(f"Invalid physical counts rank={rank} run={key}")
    with np.load(bank_dir / "golden.npz") as gold:
        gold_y, gold_counts = gold["output"], gold["counts"]
    if "golden_sha256" in bank_meta and digest((gold_y, gold_counts)) != bank_meta["golden_sha256"]:
        raise ValueError("First-round Golden checksum mismatch")
    previous = None
    stale_detected = 0
    adjacent_changes = []
    for index, step in enumerate((0,) if fixed else [*range(STEPS), *TAIL]):
        results = {}
        for key, folder in runs.items():
            iteration = index if key == "replay" else step
            with np.load(folder / f"result_{iteration}.npz") as result:
                y, stats = result["output"], result["counts"]
            results[key] = y
            physical = counts[key][step]
            if stats.shape != physical.shape:
                failures.append(dict(step=step, iteration=index, run=key, count_shape=list(stats.shape)))
            elif np.any(stats != physical):
                bad = np.flatnonzero(stats != physical)
                failures.append(
                    dict(
                        step=step,
                        iteration=index,
                        run=key,
                        physical_slots=(rank * LOCAL + bad).tolist(),
                        actual=stats[bad].tolist(),
                        expected=physical[bad].tolist(),
                    )
                )
            if index == 0:
                cpu = cpu_compare(y, gold_y)
                if not cpu["passed"] or not np.array_equal(stats, gold_counts):
                    failures.append(
                        dict(step=0, run=key, cpu=cpu, golden_counts=gold_counts.tolist(), counts=stats.tolist())
                    )
        ref = results[oracle]
        if index < STEPS:
            if previous is not None and previous.shape == shape and ref.shape == shape:
                changed = int(np.count_nonzero(previous != ref))
                adjacent_changes.append(changed)
                stale_detected += int(changed > 0)
            previous = ref
        comparisons = [] if fixed else [("replay", oracle)]
        if "candidate-eager" in runs:
            comparisons.append(("candidate-eager", oracle))
            if not fixed:
                comparisons.append(("replay", "candidate-eager"))
        for left, right in comparisons:
            mismatch = differences(results[left], results[right], shape)
            if mismatch:
                failures.append(dict(step=step, iteration=index, comparison=[left, right], **mismatch))
    if not fixed and stale_detected != STEPS - 1:
        failures.append(dict(reason="stale-output negative control", detected=stale_detected, expected=STEPS - 1))
    return dict(
        rank=rank,
        passed=not failures,
        rounds=1 if fixed else STEPS,
        alternate=0 if fixed else len(TAIL),
        stale_detected=stale_detected,
        adjacent_changed_elements=adjacent_changes,
        failures=failures,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("bank", "reference", "replay", "candidate-eager", "compare"):
        cmd = commands.add_parser(mode)
        cmd.add_argument("--bank", type=Path, required=True)
        if mode == "bank":
            cmd.add_argument("--first-rank", type=int, default=0)
            cmd.add_argument("--end-rank", type=int, default=WORLD)
            cmd.add_argument("--golden", type=Path, required=True)
            cmd.add_argument("--workers", type=int, default=2)
            cmd.add_argument("--input", choices=("fp8", "bf16"), default="fp8")
            cmd.add_argument("--shared-fixture", choices=("native-fp8", "fp4-subset"), default="native-fp8")
        else:
            cmd.add_argument(
                "--fixed-perf", action="store_true", help="Fixed BF16 input accuracy and timer-off performance"
            )
            cmd.add_argument("--out", type=Path, required=True)
            if mode == "compare":
                oracle = cmd.add_mutually_exclusive_group(required=True)
                oracle.add_argument("--reference", type=Path)
                oracle.add_argument("--baseline-eager", type=Path, help="Protocol 4 candidate-eager output from the base wheel")
                cmd.add_argument("--replay", type=Path)
                cmd.add_argument("--candidate-eager", type=Path)
            else:
                cmd.add_argument("--default-wave", action="store_true", help="Keep the public API wave selection")
                cmd.add_argument("--revision", required=True, help="Git revision used to build the installed wheel")
    args = parser.parse_args()
    if args.mode == "replay" and args.fixed_perf:
        parser.error("Use reference or candidate-eager for fixed profiling")
    if args.mode == "compare" and args.baseline_eager and not args.candidate_eager:
        parser.error("--baseline-eager requires --candidate-eager for the candidate wheel")
    if args.mode == "compare" and (
        (args.fixed_perf and (not args.candidate_eager or args.replay)) or (not args.fixed_perf and not args.replay)
    ):
        parser.error("Compare needs --replay for graph runs, or --fixed-perf with --candidate-eager for fixed runs")
    if args.mode == "bank":
        if not 0 <= args.first_rank < args.end_rank <= WORLD:
            parser.error("invalid half-open rank range")
        bank(args)
    elif args.mode == "compare":
        args.out.mkdir(parents=True, exist_ok=False)
        first = None
        for rank in range(WORLD):
            try:
                report = compare_rank(args, rank)
            except (OSError, KeyError, ValueError) as exc:
                report = dict(rank=rank, passed=False, failures=[dict(reason=str(exc))])
            put_json(args.out / f"rank_{rank}.json", report)
            if not report["passed"] and first is None:
                first = dict(rank=rank, **report["failures"][0])
        put_json(args.out / "summary.json", dict(ranks=WORLD, passed=first is None, first_failure=first))
        if first is not None:
            raise SystemExit(f"EP32 replay failed: {first}")
        print(
            f"{WORLD} ranks passed fixed BF16 comparison"
            if args.fixed_perf
            else f"{WORLD} ranks passed {STEPS} rounds and {len(TAIL)} alternating calls"
        )
    else:
        import torch
        import torch_npu
        from deep_ep_experimental.megamoe import StoreGroup

        torch.set_num_threads(1)
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        group = StoreGroup.from_env()
        try:
            run(args, group, torch, torch_npu)
        except BaseException as error:
            group.abort(f"EP32 replay: {error}")
            raise


if __name__ == "__main__":
    main()
