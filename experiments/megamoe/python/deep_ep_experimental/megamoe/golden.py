# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Prepare reusable CPU Golden before torchrun; no NPU or distributed collectives."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

DEFAULT_WORKERS = min(16, os.cpu_count() or 1)


def token_indices(bs, check_tokens):
    import numpy as np

    if bs <= 0 or check_tokens < 0:
        raise ValueError("BS must be positive and check_tokens nonnegative")
    return np.arange(bs) if check_tokens == 0 else np.unique(np.linspace(0, bs - 1, min(bs, check_tokens)).astype(int))


def contract(config, world, bs, input_kind, routing, check_tokens):
    from . import reference
    from .benchmark import inputs

    if world <= 0 or config["model"]["num_experts"] % world:
        raise ValueError("Global experts must divide positive EP size")
    if input_kind not in ("bf16", "fp8") or routing not in ("balanced", "random", "skewed"):
        raise ValueError("Invalid Golden input/routing")
    token_indices(bs, check_tokens)
    # 权重/输入或参考算法变化自动失效；编译参数、性能轮数不影响 Golden。
    source = inspect.getsource(reference) + inspect.getsource(inputs) + Path(__file__).read_text(encoding="utf-8")
    return dict(
        version=1,
        reference_sha256=hashlib.sha256(source.encode()).hexdigest(),
        model=config["model"],
        seed=config["seed"],
        world=world,
        bs=bs,
        input=input_kind,
        routing=routing,
        check_tokens=check_tokens,
        numerical_contract="ascend_post_gmm2_route_weight_v1",
    )


def cache_path(directory, spec, rank):
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return Path(directory) / key / f"rank_{rank}.npz"


def digest(*arrays):
    import numpy as np

    result = hashlib.sha256()
    for array in arrays:
        result.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return result.hexdigest()


def load(directory, spec, rank):
    import numpy as np

    path = cache_path(directory, spec, rank)
    if not path.is_file():
        raise FileNotFoundError(
            f"Golden missing: rank={rank} BS={spec['bs']} {path}; "
            "run python -m deep_ep_experimental.megamoe.golden before torchrun "
            "(experiments/megamoe/scripts/batch_launch.py does this automatically)"
        )
    try:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            output, counts, selected = (saved[name] for name in ("output", "counts", "selected"))
        expected_shape = (len(token_indices(spec["bs"], spec["check_tokens"])), spec["model"]["hidden"])
        if (
            metadata != dict(contract=spec, rank=rank, sha256=digest(output, counts, selected))
            or output.shape != expected_shape
            or output.dtype != np.float32
            or not np.isfinite(output).all()
            or counts.shape != (spec["model"]["num_experts"] // spec["world"],)
            or counts.dtype != np.int64
            or np.any(counts < 0)
            or selected.dtype != np.int64
            or not np.array_equal(selected, token_indices(spec["bs"], spec["check_tokens"]))
        ):
            raise ValueError("metadata, shape, dtype or checksum mismatch")
    except Exception as error:
        raise ValueError(f"Invalid Golden {path}: {error}; remove this cache file and prepare again") from error
    return output, counts, selected


def save(directory, spec, rank, output, counts, selected):
    import numpy as np

    path = cache_path(directory, spec, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(contract=spec, rank=rank, sha256=digest(output, counts, selected))
    # 完整写入后原子发布；中断不能留下看似有效的半份 Golden。
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(
                stream, output=output, counts=counts, selected=selected, metadata=json.dumps(metadata, sort_keys=True)
            )
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(config, spec, ranks, *, workers=DEFAULT_WORKERS, chunk_tokens=256):
    """Compute a node's source ranks together; bound dense weights by worker count."""
    import numpy as np
    from .benchmark import inputs
    from .reference import bf16, dequantize_input, expert_forward, make_weights, quantize_input, unpack_weight

    if workers < 1 or chunk_tokens < 1 or not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("Need positive workers/chunk_tokens and distinct source ranks")
    if any(not 0 <= rank < spec["world"] for rank in ranks):
        raise ValueError("Golden source rank exceeds EP size")
    model, bs, world = spec["model"], spec["bs"], spec["world"]
    local = model["num_experts"] // world
    selected = token_indices(bs, spec["check_tokens"])
    counts = np.zeros(model["num_experts"], dtype=np.int64)
    sources = {}
    for source in range(world):
        x, ids, weights = inputs(config, source, bs, spec["routing"], world=world)
        counts += np.bincount(ids.ravel(), minlength=len(counts))
        if source in ranks:
            sources[source] = (dequantize_input(*quantize_input(x[selected])), ids[selected], weights[selected])
        if source == 0 or (source + 1) % 8 == 0:
            print(f"[GOLDEN] BS={bs} input/counts source={source + 1}/{world}", flush=True)
    x = np.concatenate([sources[rank][0] for rank in ranks])
    ids = np.concatenate([sources[rank][1] for rank in ranks])
    weights = np.concatenate([sources[rank][2] for rank in ranks])
    del sources
    output = np.zeros_like(x)

    def jobs():
        for owner in range(world):
            needed = [e for e in range(local) if np.any(ids == owner * local + e)]
            if not needed:
                continue
            # 必须保留原始 RNG 的整组生成顺序；单专家重新播种会改变待测权重。
            raw = make_weights(local, model["hidden"], model["intermediate_hidden"], config["seed"] + owner)
            for expert in needed:
                token, slot = np.where(ids == owner * local + expert)
                pair = tuple((data[expert], scale[expert]) for data, scale in raw)
                yield token, weights[token, slot], pair
        shared = model["num_shared_experts"]
        if shared:
            raw = make_weights(shared, model["hidden"], model["intermediate_hidden"], config["seed"] + 100000)
            for expert in range(shared):
                yield np.arange(len(x)), None, tuple((data[expert], scale[expert]) for data, scale in raw)

    def forward(job):
        token, route_weight, pair = job
        w1, w2 = map(unpack_weight, pair)
        result = np.empty((len(token), model["hidden"]), dtype=np.float32)
        for start in range(0, len(token), chunk_tokens):
            stop = start + chunk_tokens
            result[start:stop] = expert_forward(x[token[start:stop]], w1, w2)
        if route_weight is not None:
            result *= route_weight[:, None]
        return token, result

    # 有界提交、按专家顺序累加：避免一次展开全部专家，避免线程完成次序改变归约。
    pending, iterator, completed = deque(), iter(jobs()), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            while len(pending) < workers:
                job = next(iterator, None)
                if job is None:
                    break
                pending.append(pool.submit(forward, job))
            if not pending:
                break
            token, result = pending.popleft().result()
            np.add.at(output, token, result)
            completed += 1
            if completed == 1 or completed % 16 == 0:
                print(f"[GOLDEN] BS={bs} ranks={ranks} experts_done={completed}", flush=True)
    output = bf16(output)
    return {
        rank: (
            output[index * len(selected) : (index + 1) * len(selected)],
            counts[rank * local : (rank + 1) * local],
            selected,
        )
        for index, rank in enumerate(ranks)
    }


def prepare(
    config,
    world,
    bs,
    ranks,
    directory,
    *,
    input_kind="bf16",
    routing="balanced",
    check_tokens=0,
    workers=DEFAULT_WORKERS,
    chunk_tokens=256,
):
    spec = contract(config, world, bs, input_kind, routing, check_tokens)
    missing = []
    for rank in ranks:
        if cache_path(directory, spec, rank).exists():
            load(directory, spec, rank)  # 缓存校验失败直接报错，不静默使用或覆盖。
            print(f"[GOLDEN] reuse BS={bs} rank={rank}", flush=True)
        else:
            missing.append(rank)
    if missing:
        for rank, values in generate(config, spec, missing, workers=workers, chunk_tokens=chunk_tokens).items():
            save(directory, spec, rank, *values)
            print(f"[GOLDEN] saved BS={bs} rank={rank} {cache_path(directory, spec, rank)}", flush=True)
    return spec


def compare(actual, expected, *, diff_thd=0.01, pct_thd=0.05, max_diff=0.1):
    """Compare CPU results using absolute and relative error thresholds."""
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    result = dict(
        passed=False,
        elements=int(expected.size),
        bad_elements=0,
        max_abs_error=None,
        max_mixed_error=None,
        reason="",
        diff_thd=diff_thd,
        pct_thd=pct_thd,
        max_diff=max_diff,
    )
    if actual.shape != expected.shape:
        return dict(result, reason=f"shape {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return dict(result, reason="non-finite actual or Golden")
    actual, expected = actual.astype(np.float32), expected.astype(np.float32)
    absolute = np.abs(actual - expected)
    denominator = np.maximum(np.maximum(np.abs(actual), np.abs(expected)), (1.0 / (1 << 14)) / diff_thd) + 1e-9
    mixed = np.where(absolute < diff_thd, absolute, absolute / denominator)
    bad, largest = int(np.count_nonzero(mixed > diff_thd)), float(mixed.max(initial=0))
    return dict(
        result,
        passed=bad <= pct_thd * expected.size and largest <= max_diff,
        bad_elements=bad,
        max_abs_error=float(absolute.max(initial=0)),
        max_mixed_error=largest,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="smoke")
    parser.add_argument("--bs-values")
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--ranks", required=True, help="Source ranks on this host, e.g. 0-7")
    parser.add_argument("--golden-dir", type=Path, default=Path("benchmark-golden"))
    parser.add_argument("--input", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--routing", choices=("balanced", "random", "skewed"), default="balanced")
    parser.add_argument("--check-tokens", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_tokens < 1:
        parser.error("workers and chunk-tokens must be positive")
    # 在 NumPy/BLAS 导入前设置，每台机器仅一个生成进程，防止八个 rank 争用 CPU。
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    threads = max(1, cpus // args.workers)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(threads)
    from .benchmark import load_config
    from .timer.__main__ import parse_ranks

    config, ranks = load_config(args.config, args.bs_values), parse_ranks(args.ranks)
    if args.ep_size <= 0 or any(rank >= args.ep_size for rank in ranks):
        parser.error("ranks must be within EP size")
    print(f"[GOLDEN] workers={args.workers} BLAS_threads={threads} chunk_tokens={args.chunk_tokens}", flush=True)
    for bs in config["bs_values"]:
        prepare(
            config,
            args.ep_size,
            bs,
            ranks,
            args.golden_dir,
            input_kind=args.input,
            routing=args.routing,
            check_tokens=args.check_tokens,
            workers=args.workers,
            chunk_tokens=args.chunk_tokens,
        )


if __name__ == "__main__":
    main()
