# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""torchrun worker. The installed DUT is imported without adding its source path."""

import argparse
from dataclasses import replace
from datetime import timedelta
import importlib.metadata
import json
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .cases import CaseSpec
from .installed import load_installed_ep_gmm_fused
from ..reference.golden import route_matrix
from ..reference.routing import route_statistics
from ..reference.streaming import ReferenceStore, check_streaming


def environment(torch, torch_npu, package):
    import numpy as np
    import deep_ep

    package_dir = Path(package.__file__).resolve().parent
    native = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package_dir.glob("_C*.so")}
    stack = {}
    ascend = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest"))
    for label, paths in {
        "cann": [ascend / "version.cfg", ascend / "version.info", ascend / "compiler/version.info"],
        "driver": [Path("/usr/local/Ascend/driver/version.info")],
    }.items():
        stack[label] = next((p.read_text(errors="replace") for p in paths if p.is_file()), None)
    build_info = package_dir / "_build_info.json"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "blas_thread_env": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "torch_npu": torch_npu.__version__,
        "package_file": str(Path(package.__file__).resolve()),
        "runtime_package_file": str(Path(deep_ep.__file__).resolve()),
        "package_version": importlib.metadata.version("ascend-deepep"),
        "catccos_commit": package.CATCCOS_COMMIT,
        "shmem_commit": package.SHMEM_COMMIT,
        "device": torch_npu.npu.get_device_name(),
        "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "stack_versions": stack,
        "extension_sha256": native,
        "build_info": json.loads(build_info.read_text()) if build_info.exists() else None,
        "global_rank": int(os.environ["RANK"]),
        "local_rank": int(os.environ["LOCAL_RANK"]),
    }


def prepare(case, rank, package, torch, device, *, cache_dir=None, output_pool=None):
    table = route_matrix(case)
    store = ReferenceStore(case, table, cache_dir)
    x_cpu, weight_cpu = store.x(rank), store.w(rank)
    x = torch.from_numpy(x_cpu).to(device=device, dtype=torch.bfloat16)
    w = torch.from_numpy(weight_cpu).to(device=device, dtype=torch.bfloat16)
    if case.expected_error == "zero_input":
        x = x[:0]
    elif case.expected_error == "dtype":
        x = x.float()
    e = case.local_experts
    local = table[rank].tolist()
    glob = table[:, rank * e : (rank + 1) * e].reshape(-1).tolist()
    if case.expected_error == "route_sum":
        # Host-known invalid sums are rejected before any peer can launch a kernel.
        if case.operator == "alltoallv_gmm":
            local[0] += 1
        else:
            glob[0] += 1
    if case.route_device != "host" and case.expected_error != "route_sum":
        target = "cpu" if case.route_device == "cpu" else device
        local = torch.tensor(local, dtype=torch.int64, device=target)
        glob = torch.tensor(glob, dtype=torch.int64, device=target)
    if case.expected_error == "route_dtype":
        local = torch.tensor(table[rank].tolist(), dtype=torch.int32)
    common = dict(ep_size=case.ep_size, expert_num=case.ep_size * e, tiling=case.tiling)
    received = int(table[:, rank * e : (rank + 1) * e].sum())

    def output(name, rows, columns):
        if output_pool is not None:
            return output_pool[name][:rows, :columns]
        return torch.empty((rows, columns), device=device, dtype=x.dtype)

    if case.operator == "gmm_alltoallv":
        options = {"out": output("out", case.m, case.n)} if case.preallocate else {}

        def call():
            return package.gmm_alltoallv(x, w, local, glob, **common, **options)
    else:
        options = {"output_all_to_allv": case.export}
        if case.explicit_received:
            options["received_rows"] = received
        if case.preallocate:
            options["out"] = output("out", received, case.n)
            if case.export:
                options["all_to_allv_out"] = output("export", received, case.k)
        if case.operator == "alltoallv_gmm":

            def call():
                return package.alltoallv_gmm(x, w, local, glob, **common, **options)
        else:
            second_w = store.w(rank, second=True)
            w2 = torch.from_numpy(second_w).to(device=device, dtype=torch.bfloat16)

            def call():
                middle = package.alltoallv_gmm(x, w, local, glob, **common, received_rows=received)
                return package.gmm_alltoallv(middle, w2, local, glob, **common)

    def refresh(updated):
        nonlocal store
        if store:
            store.__exit__(None, None, None)
            store = ReferenceStore(updated, table, cache_dir)
            call.reference_store = store
        new_x, new_w = store.x(rank), store.w(rank)
        x.copy_(torch.from_numpy(new_x).to(device=device, dtype=x.dtype))
        w.copy_(torch.from_numpy(new_w).to(device=device, dtype=w.dtype))
        if case.operator == "composition":
            second = store.w(rank, second=True)
            w2.copy_(torch.from_numpy(second).to(device=device, dtype=w2.dtype))

    call.refresh = refresh
    call.reference_store = store
    call.close = lambda: store.__exit__(None, None, None) if store else None
    return call, table


def check(case, rank, call, table, torch, npu, *, backend="npu-fp32", block_rows=1024, result=None):
    import numpy as np

    if case.expected_error:
        try:
            call()
        except (ValueError, RuntimeError, TypeError) as error:
            patterns = {
                "experts": ("expert",),
                "zero_input": ("positive", "shape[0]"),
                "capacity": ("workspace", "ub capacity"),
                "dtype": ("dtype",),
                "route_dtype": ("dtype",),
                "route_sum": ("sum(",),
            }
            if not any(text in str(error).lower() for text in patterns[case.expected_error]):
                raise
            return {"passed": True, "expected_rejection": str(error)}
        raise AssertionError(f"invalid {case.expected_error} input was accepted")
    dut_start = time.perf_counter()
    result = call() if result is None else result
    npu.synchronize()
    dut_seconds = time.perf_counter() - dut_start
    actual, exported = result if isinstance(result, tuple) else (result, None)
    expected_rows = (
        int(table[:, rank * case.local_experts : (rank + 1) * case.local_experts].sum())
        if case.operator == "alltoallv_gmm"
        else case.m
    )
    output_columns = case.k if case.operator == "composition" else case.n
    if tuple(actual.shape) != (expected_rows, output_columns):
        raise AssertionError("DUT output shape mismatch")
    if exported is not None and tuple(exported.shape) != (expected_rows, case.k):
        raise AssertionError("DUT exported output shape mismatch")
    if case.export and exported is None:
        raise AssertionError("requested dispatch export is missing")
    weight_key, device_weight = None, None

    def matmul(x, w):
        nonlocal weight_key, device_weight
        key = (str(getattr(w, "filename", "")), w.__array_interface__["data"][0], w.shape)
        if key != weight_key:
            device_weight = torch.from_numpy(np.array(w)).to(device=actual.device, dtype=torch.float32)
            weight_key = key
        operand = torch.from_numpy(np.array(x)).to(device=actual.device, dtype=torch.float32)
        return (operand @ device_weight).cpu().numpy()

    streamed = check_streaming(
        case,
        rank,
        table,
        call.reference_store,
        lambda start, count: actual[start : start + count].cpu().float().numpy(),
        (lambda start, count: exported[start : start + count].cpu().float().numpy()) if exported is not None else None,
        block_rows=block_rows,
        device_matmul=matmul,
    )
    streamed["dut_call_and_sync_seconds"] = dut_seconds
    streamed["backend"] = backend
    return streamed


def run(args):
    package = load_installed_ep_gmm_fused()
    import torch
    import torch.distributed as dist
    import torch_npu
    from .scenarios import domain_groups, run_sequence

    case = CaseSpec(**json.loads(Path(args.case).read_text()))
    validate_accuracy = args.mode != "bench"
    scenario = case.scenario or {}
    if args.mode != "accuracy" and (
        case.expected_error
        or scenario.get("category") == "state_reuse"
        or scenario.get("contract") == "reject_zero_input"
    ):
        raise ValueError("negative cases and accuracy-only scenarios cannot be timed or profiled")
    local_rank, rank, size = (int(os.environ[key]) for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    torch_npu.npu.set_device(local_rank)
    if size % case.ep_size:
        raise ValueError("WORLD_SIZE must be divisible by case EP size")
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout))
    # All world members create subgroups in the same order, including nonmembers.
    ep_group = None
    groups = domain_groups(size, case.ep_size, (case.scenario or {}).get("layout", "contiguous"))
    for domain, members in enumerate(groups):
        group = dist.new_group(members, backend="hccl")
        if rank in members:
            ep_group = group
            domain_id = domain
    ep_rank = dist.get_rank(ep_group)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "case": case.record(),
        "environment": environment(torch, torch_npu, package),
        "global_rank": rank,
        "ep_rank": ep_rank,
        "domain": domain_id,
        "mode": args.mode,
        "accuracy_checked": False,
    }
    if (case.scenario or {}).get("required_world_size", size) != size:
        raise ValueError("domain-isolation case requires exactly 64 ranks")
    if (case.scenario or {}).get("category") == "domain_isolation":
        case = replace(case, seed=case.seed + domain_id * 10000019)
        report["execution_case"] = case.record()  # Distinct data catches cross-domain leakage.
    if validate_accuracy and args.golden_backend == "npu-fp32":
        # Fail explicitly on runtimes without an exposed FP32 precision control.
        if not hasattr(torch_npu.npu, "matmul") or not hasattr(torch_npu.npu.matmul, "allow_hf32"):
            raise RuntimeError("npu-fp32 golden requires torch_npu.npu.matmul.allow_hf32")
        torch_npu.npu.matmul.allow_hf32 = False
    call = None
    try:
        init = (
            {"group": ep_group}
            if args.bootstrap == "group"
            else {"rank": ep_rank, "world_size": case.ep_size, "local_rank": local_rank}
        )
        if args.bootstrap == "default" and size != case.ep_size:
            raise ValueError("multi-domain tests require group bootstrap")
        with package.udma_group(**init):
            scenario = case.scenario or {}
            if scenario.get("category") == "state_reuse":
                if args.mode != "accuracy":
                    raise ValueError("state-reuse sequences are accuracy-only")
                report["accuracy"] = run_sequence(
                    case, ep_rank, package, torch, torch_npu.npu, f"npu:{local_rank}", args, output, rank
                )
            else:
                call, table = prepare(
                    case, ep_rank, package, torch, f"npu:{local_rank}", cache_dir=args.reference_cache
                )
                report["routing"] = {
                    "matrix": table.tolist(),
                    "sha256": hashlib.sha256(table.tobytes()).hexdigest(),
                    **route_statistics(case, table),
                }
                zero_contract = scenario.get("contract") == "reject_zero_input"
                if not validate_accuracy:
                    report["accuracy"] = {"status": "not_run", "reason": "benchmark mode"}
                elif zero_contract:
                    # Only zero-input ranks invoke the rejecting API. Other peers
                    # must not launch a communication kernel waiting for them.
                    if ep_rank in report["routing"]["zero_receive_ranks"]:
                        report["accuracy"] = check(
                            replace(case, expected_error="zero_input"), ep_rank, call, table, torch, torch_npu.npu
                        )
                    else:
                        report["accuracy"] = dict(passed=True, role="nonlaunching_contract_peer")
                    report["accuracy"]["coverage"] = "zero-input rejection; no distributed kernel execution"
                else:
                    if scenario.get("category") == "rank_stagger" and ep_rank == 0:
                        time.sleep(scenario["delay_seconds"])
                    report["accuracy"] = check(
                        case,
                        ep_rank,
                        call,
                        table,
                        torch,
                        torch_npu.npu,
                        backend=args.golden_backend,
                        block_rows=args.golden_block_rows,
                    )
            if (
                args.mode == "accuracy"
                and not case.expected_error
                and scenario.get("category") != "state_reuse"
                and not scenario.get("contract") == "reject_zero_input"
            ):
                # Reuse the same out buffers and route tables with changed input data.
                changed = replace(case, seed=case.seed + 1000003)
                call.refresh(changed)
                if scenario.get("category") == "rank_stagger":
                    delayed = 0 if scenario["delay_pattern"] == "first" else 1
                    if ep_rank == delayed:
                        time.sleep(scenario["delay_seconds"])
                report["changed_data"] = check(
                    changed,
                    ep_rank,
                    call,
                    table,
                    torch,
                    torch_npu.npu,
                    backend=args.golden_backend,
                    block_rows=args.golden_block_rows,
                )
                report["accuracy"]["passed"] &= report["changed_data"]["passed"]
            if validate_accuracy:
                passed = torch.tensor(
                    [int(report["accuracy"]["passed"])], dtype=torch.int32, device=f"npu:{local_rank}"
                )
                dist.all_reduce(passed, op=dist.ReduceOp.MIN)
                if not passed.item():
                    raise AssertionError("accuracy failed on at least one rank")
                report["accuracy_checked"] = True
            if args.mode in ("bench", "capture"):
                for _ in range(args.warmup):
                    call()
                torch_npu.npu.synchronize()
                batches = args.batches if args.mode == "bench" else 1
                if args.mode == "capture":
                    report["samples_us"], report["host_completion_us"] = [], []
                for _ in range(batches):
                    dist.barrier(group=ep_group)
                    if args.mode == "bench":
                        # msprof records individual target tasks. Do not add Event
                        # timing or a golden call to this execution sequence.
                        for _ in range(args.iterations):
                            call()
                        torch_npu.npu.synchronize()
                        continue
                    begin, end = torch_npu.npu.Event(enable_timing=True), torch_npu.npu.Event(enable_timing=True)
                    host_start = time.perf_counter_ns()
                    begin.record()
                    for _ in range(args.iterations):
                        call()
                    end.record()
                    end.synchronize()
                    report["samples_us"].append(begin.elapsed_time(end) * 1000 / args.iterations)
                    report["host_completion_us"].append((time.perf_counter_ns() - host_start) / 1000 / args.iterations)
                report["timing"] = {
                    "warmup": args.warmup,
                    "iterations_per_batch": args.iterations,
                    "batches": batches,
                }
                if args.mode == "bench":
                    report["timing"].update(backend="msprof_op_summary", sampling="individual target kernel durations")
                else:
                    report["timing"].update(
                        backend="npu_event",
                        completion="NPU event after PR UDMA completion protocol",
                        host_scope="enqueue plus batch completion, excluding pre-batch barrier",
                    )
            report["passed"] = True
    except Exception as error:
        report.update(passed=False, error=repr(error))
        raise
    finally:
        if call is not None:
            call.close()
        (output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        # torchrun handles process failure/timeout; do not insert another collective here.
        if report.get("passed"):
            dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("accuracy", "bench", "capture"), required=True)
    parser.add_argument("--bootstrap", choices=("default", "group"), default="group")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--profiler-config")
    parser.add_argument("--golden-backend", choices=("npu-fp32",), default="npu-fp32")
    parser.add_argument("--golden-block-rows", type=int, default=1024)
    parser.add_argument("--reference-cache", help="local scratch directory for temporary reference inputs")
    args = parser.parse_args()
    if args.profiler_config:
        from .cases import CaseSpec
        from ..profiling.timeline import Profiler, command, discover, inspect_capture

        config = json.loads(Path(args.profiler_config).read_text())
        profiler = Profiler(**config["profiler"]) if config["profiler"] else discover(config["metric"])
        if args.mode == "bench" and config["metric"] != "op_summary":
            raise ValueError("bench requires msprof op_summary collection")
        case = CaseSpec(**json.loads(Path(args.case).read_text()))
        rank_dir = Path(args.output) / f"profile-rank-{os.environ['RANK']}"
        arguments = sys.argv[1:]
        index = arguments.index("--profiler-config")
        del arguments[index : index + 2]
        cmd = command(
            profiler,
            config["metric"],
            case,
            rank_dir,
            [sys.executable, "-m", __spec__.name, *arguments],
            args.warmup,
            args.iterations,
            config["pipes"],
        )
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / f"profile-command-{os.environ['RANK']}.json").write_text(json.dumps(cmd, indent=2))
        subprocess.run(cmd, check=True)
        if args.mode == "bench":
            from ..profiling.msprof_parser import parse_benchmark

            capture = parse_benchmark(
                rank_dir, case.operator, warmup=args.warmup, iterations=args.iterations, batches=args.batches
            )
        else:
            capture = inspect_capture(
                rank_dir, case.operator, metric=config["metric"], warmup=args.warmup, iterations=args.iterations
            )
        capture.update(global_rank=int(os.environ["RANK"]), profiler=profiler.record())
        (Path(args.output) / f"capture-{os.environ['RANK']}.json").write_text(json.dumps(capture, indent=2))
        if not capture["passed"]:
            raise RuntimeError(f"incomplete profiler capture; see {args.output}/capture-{os.environ['RANK']}.json")
    else:
        run(args)


if __name__ == "__main__":
    main()
