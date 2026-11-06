# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026

"""No-Timer paired full operator Events. All validation is outside Events."""

from pathlib import Path
from datetime import timedelta
import argparse
import ctypes
import hashlib
import json
import os
import statistics

PLANS = {
    "k3": dict(m=8192, k=3584, f=3072, ep=4, e=7, topk=16),
    "common": dict(m=4096, k=7168, f=4096, ep=8, e=4, topk=16),
}


def main(a):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401 - Register NPU for both independently imported packages.

    if a.backend == "source":
        import ascend_megamoe as package_module
        from ascend_megamoe import MegaMoeBF16Train, ShmemInstance
    else:
        import deep_ep_experimental.megamoe_train as package_module
        from deep_ep_experimental.megamoe_train import MegaMoeBF16Train
    p = PLANS[a.shape]
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == p["ep"]
    torch.npu.set_device(local)
    dist.init_process_group("hccl", timeout=timedelta(seconds=240))
    torch.npu.reset_peak_memory_stats()
    torch.manual_seed(20261027 + rank)
    dev = f"npu:{local}"
    dtype = torch.bfloat16
    x = (torch.randn(p["m"], p["k"], device=dev, dtype=dtype) * 0.1).requires_grad_()
    routes = np.arange(p["m"] * p["topk"], dtype=np.int64).reshape(p["m"], p["topk"])

    def ids_for(source):
        return ((routes % p["ep"]) * p["e"] + (routes // p["ep"] + source) % p["e"]).astype(np.int32)

    ids_np = ids_for(rank)
    assert np.all(np.diff(np.sort(ids_np, axis=1), axis=1) != 0)
    ids = torch.from_numpy(ids_np).to(dev)
    weights = torch.softmax(torch.randn(p["m"], p["topk"], device=dev), -1).to(dtype).requires_grad_()
    w1 = (torch.randn(p["e"], 2 * p["f"], p["k"], device=dev, dtype=dtype) * 0.01).requires_grad_()
    w2 = (torch.randn(p["e"], p["k"], p["f"], device=dev, dtype=dtype) * 0.01).requires_grad_()
    dy = torch.randn_like(x) * 0.1
    leaves = (x, weights, w1, w2)
    expected = sum(np.bincount(ids_for(source).ravel(), minlength=p["ep"] * p["e"]) for source in range(p["ep"]))[
        rank * p["e"] : (rank + 1) * p["e"]
    ].tolist()
    assert sum(expected) == p["m"] * p["topk"]
    shmem = None
    if a.backend == "source":
        shmem = ShmemInstance(rank, p["ep"], os.environ["DEEPEP_SHMEM_ENDPOINT"])
        shmem.set_data_op_engine_type(0x08)
        shmem.init(3 * 1024**3)
        op = MegaMoeBF16Train(shmem, 3 * 1024**3)
    else:
        op = MegaMoeBF16Train(dist.group.WORLD, 3 * 1024**3)
    package = Path(package_module.__file__).resolve().parent
    kernel_path = package / (
        "lib/libascend_megamoe_bf16.so" if a.backend == "source" else "libdeepep_megamoe_train_bf16.so"
    )
    host_path = next(package.glob("_C*.so"))
    libraries = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [kernel_path, host_path]}
    if a.kernel_sha:
        assert libraries[kernel_path.name] == a.kernel_sha
    if a.host_sha:
        assert libraries[host_path.name] == a.host_sha
    lib = ctypes.CDLL(str(kernel_path))
    # Public package intentionally exports no Timer APIs.
    for name in ["mega_moe_bf16_timer_enabled", "mega_moe_bf16_backward_timer_enabled"]:
        assert not hasattr(lib, name)
    decoupled = None
    query_name = ("" if a.backend == "source" else "deepep_train_") + "mega_moe_bf16_swiglu_decoupled_for_saved_mode"
    if hasattr(lib, query_name):
        query = getattr(lib, query_name)
        query.argtypes = [ctypes.c_uint64, ctypes.c_uint32]
        query.restype = ctypes.c_int
        decoupled = query(p["k"], dict(none=1, bf16=2, mxfp8=3)[a.mode])

    def forward():
        return op(x, ids, weights, w1, w2, p["m"] * p["topk"], save_gmm1_output=a.mode, swiglu_limit=0.0)

    def digest(t):
        cpu = t.detach().cpu().contiguous()
        return dict(
            shape=list(cpu.shape),
            dtype=str(cpu.dtype),
            sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        )

    inputs = {
        name: digest(value) for name, value in zip(["X", "ids", "p", "W1", "W2", "dY"], [x, ids, weights, w1, w2, dy])
    }
    y, counts = forward()
    grads = torch.autograd.grad(y, leaves, dy)
    torch.npu.synchronize()
    assert counts.cpu().reshape(-1).tolist() == expected
    assert all(bool(torch.isfinite(t).all().item()) for t in (y, *grads))
    outputs = {name: digest(t) for name, t in zip(["Y", "dX", "dp", "dW1", "dW2"], (y, *grads))}
    # Full output hashes and samples allow explicit attribution of dependency
    # changes; precision must be checked before interpreting performance.
    dw1, dw2 = grads[2], grads[3]
    samples = dict(
        Y=y[:32].detach().float().cpu().numpy(),
        dX=grads[0][:32].detach().float().cpu().numpy(),
        dp=grads[1][:32].detach().float().cpu().numpy(),
        dW1=dw1[:, :: max(1, dw1.shape[1] // 32), :: max(1, dw1.shape[2] // 32)].detach().float().cpu().numpy(),
        dW2=dw2[:, :: max(1, dw2.shape[1] // 32), :: max(1, dw2.shape[2] // 32)].detach().float().cpu().numpy(),
    )
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / f"rank_{rank}_samples.npz", **samples)
    del samples, y, counts, grads, dw1, dw2
    phases = {}
    for phase in ["forward", "backward"]:
        times = []
        for iteration in range(a.warmup + a.loops):
            if phase == "backward":
                y, counts = forward()
                torch.npu.synchronize()
            dist.barrier()
            torch.npu.synchronize()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            if phase == "forward":
                y, counts = forward()
            else:
                grads = torch.autograd.grad(y, leaves, dy)
            end.record()
            torch.npu.synchronize()
            if iteration >= a.warmup:
                times.append(float(start.elapsed_time(end)))
            if iteration == a.warmup + a.loops - 1:
                assert counts.cpu().reshape(-1).tolist() == expected
                assert all(bool(torch.isfinite(t).all().item()) for t in ((y,) if phase == "forward" else (y, *grads)))
            if phase == "backward":
                del grads
            del y, counts
        phases[phase] = dict(
            samples_ms=times,
            p50_ms=statistics.median(times),
            p95_ms=float(np.percentile(times, 95)),
            mean_ms=statistics.mean(times),
            min_ms=min(times),
            max_ms=max(times),
        )
        print(
            f"[PERF] rank={rank} {a.shape} {a.mode} {a.variant} {phase} p50={statistics.median(times):.6f}", flush=True
        )
    dist.barrier()
    torch.npu.synchronize()
    memory = dict(
        torch_peak_allocated_bytes=torch.npu.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.npu.max_memory_reserved(),
        symmetric_window_bytes=3 * 1024**3,
        scope="Torch allocator peaks exclude the separately allocated SHMEM window and driver memory",
    )
    op.close()
    if shmem is not None:
        shmem.destroy()
    dist.destroy_process_group()
    record = dict(
        backend=a.backend,
        rank=rank,
        plan=p,
        shape=a.shape,
        mode=a.mode,
        variant=a.variant,
        phases=phases,
        counts=expected,
        input_hashes=inputs,
        output_hashes=outputs,
        libraries=libraries,
        decoupled=decoupled,
        warmup=a.warmup,
        loops=a.loops,
        memory=memory,
        scope="Complete operator forward and autograd.grad backward Events; router, forward preparation for backward, loss generation and optimizer excluded; no Timer",
    )
    (a.out / f"rank_{rank}.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=PLANS, required=True)
    parser.add_argument("--mode", choices=["none", "bf16", "mxfp8"], required=True)
    parser.add_argument("--variant", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--backend", choices=["source", "deepep"], default="deepep")
    parser.add_argument("--kernel-sha")
    parser.add_argument("--host-sha")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20)
    main(parser.parse_args())
