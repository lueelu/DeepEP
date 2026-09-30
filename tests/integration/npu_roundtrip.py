# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""torchrun worker for real NPU Tensors; never imported by device-free tests."""

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path

from tests.utils.checks import assert_close, assert_dispatch
from tests.utils.envs import read_launch
from tests.utils.precision import _float32 as fp32, compare_dual, quantize_bfloat16 as bf16
from .npu_backend import NativeV1, NpuStub, installed_native, require_tensor, wait_completion
from .workflow import next_iteration, smoke_cases, source_answers


def expert_step(torch, received, *, rank, experts):
    x, indices, weights = received.x, received.indices, received.weights
    device = x.device
    require_tensor(torch, x, device=device, dtype=torch.bfloat16)
    require_tensor(torch, indices, device=device, dtype=torch.int64)
    require_tensor(torch, weights, device=device, dtype=torch.float32, shape=indices.shape)
    total = torch.zeros(x.shape, dtype=torch.float32, device=device)
    for slot in range(indices.shape[1]):
        local = indices[:, slot]
        expert = local + rank * (experts // 2)
        # Separate eager ops preserve the FP32 -> BF16 rounding boundaries.
        a = (expert.float() + 1.0) / 8.0
        b = (expert.float() - 3.0) / 64.0
        transformed = (x.float() * a.unsqueeze(1) + b.unsqueeze(1)).to(torch.bfloat16)
        contribution = transformed.float() * weights[:, slot].unsqueeze(1)
        total = total + torch.where((local >= 0).unsqueeze(1), contribution, 0.0)
    values = torch.where(indices >= 0, 0.25 + weights * 0.5, 777.0)
    return total.to(torch.bfloat16), values


def run_case(torch, adapter, case, rank, device):
    tokens = len(case.x[rank])
    x = torch.tensor(case.x[rank], dtype=torch.bfloat16, device=device).reshape(tokens, case.hidden)
    indices = torch.tensor(case.routes[rank], dtype=torch.int64, device=device).reshape(tokens, case.topk)
    weights = torch.tensor(case.weights[rank], dtype=torch.float32, device=device).reshape(tokens, case.topk)
    received = adapter.dispatch(x, indices, weights, case.experts)
    wait_completion(received.event)
    rows = received.x.shape[0]
    require_tensor(torch, received.x, device=device, dtype=torch.bfloat16, shape=(rows, case.hidden))
    require_tensor(torch, received.indices, device=device, dtype=torch.int64, shape=(rows, case.topk))
    require_tensor(torch, received.weights, device=device, dtype=torch.float32, shape=(rows, case.topk))
    transported = replace(
        case,
        x=tuple(tuple(tuple(bf16(v) for v in row) for row in source) for source in case.x),
        weights=tuple(tuple(tuple(fp32(v) for v in row) for row in source) for source in case.weights),
    )
    assert_dispatch(
        transported,
        rank,
        False,
        received.x.cpu().tolist(),
        received.indices.cpu().tolist(),
        received.weights.cpu().tolist(),
        received.counts,
    )
    payload, values = expert_step(torch, received, rank=rank, experts=case.experts)
    require_tensor(torch, payload, device=device, dtype=torch.bfloat16, shape=(rows, case.hidden))
    output, returned, event = adapter.combine(payload, received.handle, values)
    wait_completion(event)
    require_tensor(torch, output, device=device, dtype=torch.bfloat16, shape=(tokens, case.hidden))
    require_tensor(torch, returned, device=device, dtype=torch.float32, shape=(tokens, case.topk))
    golden, benchmark, expected_values = source_answers(case)
    assert_close(returned.cpu().tolist(), expected_values[rank], label="NPU combine value channel")
    report = compare_dual(output.cpu().tolist(), benchmark[rank], golden[rank], dtype="bfloat16", level="L2")
    if not report.passed:
        raise AssertionError(f"{case.name} rank {rank}: {report.to_dict()}")
    return {
        "name": case.name,
        "npu_tensors_checked": True,
        "dispatch_checked": True,
        "precision": report.to_dict(),
        "passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("npu-stub", "native"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not __debug__:
        parser.error("NPU validation requires Python assertions; do not use -O")
    cases = smoke_cases()
    if args.case != "all":
        cases = tuple(case for case in cases if case.name == args.case)
    if not cases or (args.backend == "native" and len(cases) != 1):
        parser.error("Select one native case per process; use the outer launcher for the full suite")
    launch = read_launch(os.environ)
    if launch.world_size != 2 or os.environ.get("LOCAL_WORLD_SIZE") != "2":
        parser.error("NPU roundtrip requires two ranks on one host")
    # Missing native algorithms fail before setting a device or allocating SHMEM.
    module, build_identity = installed_native() if args.backend == "native" else (None, None)
    import torch
    import torch_npu

    dist = torch.distributed
    if dist.is_initialized():
        raise RuntimeError("NPU roundtrip requires a fresh process group")
    if not torch.npu.is_available() or launch.local_rank >= torch.npu.device_count():
        raise RuntimeError("Requested NPU is unavailable")
    torch.npu.set_device(launch.local_rank)
    device = torch.device("npu", launch.local_rank)
    adapter = None
    records = []
    try:
        dist.init_process_group("hccl" if args.backend == "npu-stub" else "gloo", timeout=timedelta(seconds=120))
        group = dist.group.WORLD
        adapter = (
            NpuStub(torch, group, launch.rank) if args.backend == "npu-stub" else NativeV1(module, group, cases[0])
        )
        for case in cases:
            record = run_case(torch, adapter, case, launch.rank, device)
            if case.name == "fresh_iterations":
                record["second_iteration"] = run_case(torch, adapter, next_iteration(case), launch.rank, device)
            records.append(record)
        torch.npu.synchronize()
    finally:
        try:
            if adapter is not None:
                adapter.destroy()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    report = {
        "schema_version": 1,
        "backend": args.backend,
        "rank": launch.rank,
        "world_size": 2,
        "data_transport": "hccl-reference-collectives" if args.backend == "npu-stub" else "installed-deepep",
        "npu_tensor_roundtrip_tested": True,
        "native_ep_verified": args.backend == "native",
        "wheel_api_verified": args.backend == "native",
        "build_identity_sha256": build_identity,
        "device_reference_captured": False,
        "second_reference": "cpu-bf16-fp32-path-simulation",
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__),
        "results": records,
        "passed": True,
    }
    # Only publish success after stream completion and owned resource teardown.
    with (args.result_dir / f"rank-{launch.rank}.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
