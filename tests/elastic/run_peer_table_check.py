# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Installed-wheel BF16/FP8 peer-table correctness and public API timing."""

import argparse
import json
import hashlib
import os
from pathlib import Path
import runpy
import time
import traceback
from datetime import timedelta

helper = runpy.run_path(str(Path(__file__).with_name("test_dispatch.py")))
metrics = runpy.run_path(str(Path(__file__).with_name("dispatch_metrics.py")))
private_routes = runpy.run_path(str(Path(__file__).with_name("private_routes.py")))
PayloadCheck = runpy.run_path(str(Path(__file__).with_name("dispatch_validation.py")))["PayloadCheck"]
installed_dispatch_info = runpy.run_path(str(Path(__file__).with_name("dispatch_protocol.py")))[
    "installed_dispatch_info"
]


def bootstrap_gate(rank, world, host, port):
    """Untimed CPU rendezvous: no rank enters SHMEM before rank 0 is ready.

    Retain the returned store until runtime teardown so its server remains alive.
    This port is separate from SHMEM's bootstrap endpoint (port + 1).
    """
    import torch.distributed as dist

    store = dist.TCPStore(host, port, world, rank == 0, timedelta(seconds=120))
    store.set(f"ready/{rank}", "1")
    store.wait([f"ready/{peer}" for peer in range(world)])
    return store


def worker(local_rank, args):
    import numpy as np

    rank, world = helper["_worker_identity"](local_rank, args)
    log = helper["_progress_logger"]("peer-table", rank)
    generation = None
    try:
        deep_ep = helper["_import_installed_deep_ep"]()
        import torch
        import torch_npu  # noqa: F401
        from deep_ep.buffers._elastic_layout import dispatch_layout

        log("installed.aggregate_protocol " + json.dumps(installed_dispatch_info()))
        torch.set_num_threads(1)
        torch.npu.set_device(local_rank)
        peer_table = args.backend == "peer-table"
        fp8 = args.dtype == "fp8"
        info = json.loads(Path(deep_ep.__file__).with_name("_build_info.json").read_text())
        assert "elastic-dispatch-expanded-v1" in info["capabilities"]
        # Independent expert-major reference. Row ordering: expert, source,
        # token, top-k slot; no device Notify metadata enters this oracle.
        if args.route_profile == "private-random":
            routes = private_routes["private_random_routes"](world, args.tokens, args.topk, args.experts, args.seed)
        else:
            routes = (
                np.arange(world)[:, None, None] * 17
                + np.arange(args.tokens)[None, :, None] * 7
                + np.arange(args.topk)[None, None, :] * 131
            ) % args.experts
        route_sha256 = hashlib.sha256(routes.astype("<i4").tobytes()).hexdigest()
        log(f"routing={args.route_profile} seed={args.seed} experts={args.experts} sha256={route_sha256}")
        local_experts = args.experts // world
        counts = np.bincount(routes.reshape(-1), minlength=args.experts)
        rows = int(counts.reshape(world, local_experts).sum(axis=1).max())
        capacity = (
            2
            * dispatch_layout(world, args.tokens, args.topk, args.experts, max(1, rows), fp8, weights=args.weights)[
                "num_bytes"
            ]
        )
        flat = routes.reshape(-1)
        selected = np.flatnonzero((flat // local_experts) == rank)
        selected = selected[np.argsort(flat[selected], kind="stable")]
        sources = torch.from_numpy((selected // (args.tokens * args.topk)).astype(np.int64))
        tokens = torch.from_numpy((selected // args.topk % args.tokens).astype(np.int64))
        _init_store = None
        if args.bootstrap_gate:
            log(f"bootstrap.gate.begin pid={os.getpid()}")
            _init_store = bootstrap_gate(rank, world, args.master_addr, args.master_port)
            log("bootstrap.gate.done")
        log(f"init.begin pid={os.getpid()} capacity={capacity} installed={deep_ep.__file__}")
        buffer = deep_ep.ElasticBuffer(
            rank=rank,
            world_size=world,
            num_bytes=capacity,
            num_max_tokens_per_rank=args.tokens,
            hidden=7168,
            num_topk=args.topk,
            explicitly_destroy=True,
        )
        log("init.done")
        log(f"build source_sha256={info.get('source_sha256')} shmem={info.get('shmem', {}).get('revision')}")
        handle = None
        timings = []
        clean_device_ms, kernel_profiles = [], []
        # Keep generation 0's framework allocations live. A stale peer address
        # must not be hidden by allocator reuse or by later successful launches.
        retained_outputs, retained_values = (), ()

        def check_retained_outputs():
            for tensor, value in zip(retained_outputs, retained_values):
                assert torch.equal(tensor.cpu().view(torch.uint8), value), "previous Dispatch output overwritten"

        for generation in range(3):
            hidden, scales = helper["payload"](rank, args.tokens, generation, fp8)
            x = (hidden.npu(), scales.npu()) if fp8 else hidden.npu()
            input_weights = (
                (
                    (
                        rank * 17
                        + torch.arange(args.tokens)[:, None] * 7
                        + torch.arange(args.topk)[None, :] * 3
                        + generation * 11
                    )
                    % 127
                )
                .float()
                .div(128)
                .npu()
                if args.weights
                else None
            )
            expected_weights = (
                ((sources * 17 + tokens * 7 + torch.from_numpy(selected % args.topk) * 3 + generation * 11) % 127)
                .float()
                .div(128)
                if args.weights
                else None
            )
            indices = torch.from_numpy(routes[rank].copy()).npu() if handle is None else None
            torch.npu.synchronize()
            started = time.perf_counter()
            output, _, weights, handle, event = buffer.dispatch(
                x,
                topk_idx=indices,
                num_experts=args.experts if handle is None else None,
                handle=handle,
                do_expand=True,
                topk_weights=input_weights,
            )
            event.current_stream_wait()
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            log(f"dispatch.done generation={generation} public_api_ms={elapsed_ms:.3f}")
            assert hasattr(handle, "_peer_plan") == peer_table, "Unexpected dispatch backend"
            if not args.weights:
                assert weights is None
            if args.tokens <= 64 and generation == 0:
                golden = helper["build_notify_golden"](routes.tolist(), args.experts, 256)
                helper["check_handle"](handle, golden, rank, routes.tolist(), args.experts)
            actual = (output[0] if fp8 else output).cpu().view(torch.uint8)
            actual_scales = output[1].cpu() if fp8 else None
            assert actual.shape == (len(selected), 7168 * (1 if fp8 else 2))
            validation = PayloadCheck(
                rank, world, sources.numpy(), tokens.numpy(), selected % args.topk, flat[selected]
            )
            if args.weights:
                validation.add("weight", 0, weights.cpu().numpy(), expected_weights.numpy())
            for first in range(0, len(selected), 128):
                src = sources[first : first + 128, None]
                tok = tokens[first : first + 128, None]
                expected = (
                    ((src * 13 + tok * 7 + torch.arange(7168)[None, :] * 3 + generation * 11) % 127 - 63)
                    .float()
                    .div(16)
                    .to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
                )
                expected_scales = ((src + tok + torch.arange(56)[None, :] + generation) % 7 + 1).float().div(8)
                validation.add(
                    "hidden_bytes", first, actual[first : first + 128].numpy(), expected.view(torch.uint8).numpy()
                )
                if fp8:
                    validation.add("scale", first, actual_scales[first : first + 128].numpy(), expected_scales.numpy())
            fields = validation.summary()
            if generation == 0:
                packet_layout = dispatch_layout(
                    world, args.tokens, args.topk, args.experts, handle.address_stride, fp8, weights=args.weights
                )
                log(
                    "validation.aggregate_protocol "
                    + json.dumps(
                        {
                            key: packet_layout.get(key)
                            for key in (
                                "aggregate_protocol",
                                "aggregate_packet_bytes",
                                "scale_batch_rows",
                                "inbox_stride",
                                "weight_inbox_stride",
                            )
                        }
                    )
                )
            input_hidden_ok = torch.equal((x[0] if fp8 else x).cpu().view(torch.uint8), hidden.view(torch.uint8))
            input_scale_ok = not fp8 or torch.equal(x[1].cpu(), scales)
            passed = all(item["bad_rows"] == 0 for item in fields.values()) and input_hidden_ok and input_scale_ok

            def tensor_info(tensor):
                return (
                    None
                    if tensor is None
                    else dict(
                        address=hex(tensor.data_ptr()),
                        shape=list(tensor.shape),
                        stride=list(tensor.stride()),
                        dtype=str(tensor.dtype),
                    )
                )

            report = dict(
                rank=rank,
                world=world,
                generation=generation,
                passed=passed,
                tokens=args.tokens,
                topk=args.topk,
                dtype=args.dtype,
                weights=args.weights,
                route_sha256=route_sha256,
                source_sha256=info.get("source_sha256"),
                shmem_revision=info.get("shmem", {}).get("revision"),
                fields=fields,
                input_hidden_unchanged=input_hidden_ok,
                input_scale_unchanged=input_scale_ok,
                address_stride=int(handle.address_stride),
                output_rows=len(selected),
                layout=dispatch_layout(
                    world, args.tokens, args.topk, args.experts, handle.address_stride, fp8, weights=args.weights
                ),
                buffers=dict(
                    input_hidden=tensor_info(x[0] if fp8 else x),
                    input_scale=tensor_info(x[1] if fp8 else None),
                    output_hidden=tensor_info(output[0] if fp8 else output),
                    output_scale=tensor_info(output[1] if fp8 else None),
                    output_weight=tensor_info(weights),
                ),
            )
            args.output.mkdir(parents=True, exist_ok=True)
            report_path = args.output / f"validation-rank-{rank}-generation-{generation}.json"
            temporary = report_path.with_suffix(f".tmp-{os.getpid()}")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(report_path)
            log(
                f"validation generation={generation} "
                + " ".join(
                    f"{name}={'PASS' if not item['bad_rows'] else 'FAIL'} bad_rows={item['bad_rows']}/{item['checked_rows']}"
                    for name, item in fields.items()
                )
                + f" report={report_path}"
            )
            if not passed:
                log(f"validation.inputs hidden_unchanged={input_hidden_ok} scale_unchanged={input_scale_ok}")
                log("validation.buffers " + json.dumps(report["buffers"]))
                for name, item in fields.items():
                    if item["bad_rows"]:
                        log(
                            f"validation.{name} same64_bad_rows={item['same_64_bad_rows']} cross64_bad_rows={item['cross_64_bad_rows']}"
                        )
                        if name == "scale":
                            log(
                                "validation.scale.sources "
                                + json.dumps([source for source in item["by_source"] if source["bad_rows"]])
                            )
                        for sample in item["samples"]:
                            log(f"validation.{name}.sample " + json.dumps(sample))
                raise AssertionError(f"payload validation failed rank={rank} generation={generation}; {report_path}")
            check_retained_outputs()
            current_outputs = tuple(output) if fp8 else (output,)
            if weights is not None:
                current_outputs += (weights,)
            if generation == 0:
                retained_outputs = current_outputs
                retained_values = tuple(tensor.cpu().view(torch.uint8) for tensor in retained_outputs)
            else:
                for tensor, previous in zip(current_outputs, retained_outputs):
                    assert not tensor.numel() or tensor.data_ptr() != previous.data_ptr(), (
                        "output storage reused while retained"
                    )
            log(f"correctness.passed generation={generation} rows={len(selected)}")
        for iteration in range(args.warmup + args.iterations):
            started = time.perf_counter()
            output, _, weights, handle, event = buffer.dispatch(
                x, handle=handle, do_expand=True, topk_weights=input_weights
            )
            event.current_stream_wait()
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            if iteration >= args.warmup:
                timings.append(elapsed_ms)
        if args.kernel_profile or args.event_only:
            # Clean Event measurements and instrumented diagnostics are separate
            # from the original Python API wall-clock samples above.
            buffer._dispatch_diagnostics = {"rotation_wqes": args.rotation_wqes}
            for iteration in range(args.warmup + args.iterations):
                output, _, weights, handle, event = buffer.dispatch(
                    x, handle=handle, do_expand=True, topk_weights=input_weights
                )
                event.current_stream_wait()
                torch.npu.synchronize()
                if iteration >= args.warmup:
                    clean_device_ms.append(buffer._dispatch_diagnostics["device_ms"])
            assert torch.equal((output[0] if fp8 else output).cpu().view(torch.uint8), actual), (
                "clean timing hidden mismatch"
            )
            if fp8:
                assert torch.equal(output[1].cpu(), actual_scales), "clean timing scale mismatch"
            if args.weights:
                assert torch.equal(weights.cpu(), expected_weights), "clean timing weight mismatch"
            log("clean.kernel_event.done")
            if not args.kernel_profile:
                del buffer._dispatch_diagnostics
        if args.kernel_profile:
            profile = torch.zeros((64, 32), dtype=torch.int64, device=(x[0] if fp8 else x).device)
            buffer._dispatch_diagnostics["profile"] = profile
            weight_profile = (
                torch.zeros((32, 16), dtype=torch.int64, device=(x[0] if fp8 else x).device) if args.weights else None
            )
            if args.weights:
                buffer._dispatch_diagnostics["weight_profile"] = weight_profile
            detail = torch.zeros((64, 32), dtype=torch.int64, device=profile.device) if args.submit_detail else None
            if detail is not None:
                buffer._dispatch_diagnostics["detail_profile"] = detail
            # Warm up the instrumented specialization separately from clean Event runs.
            for _ in range(args.profile_warmup):
                if detail is not None:
                    buffer._dispatch_diagnostics["detail_mode"] = 1
                output, _, weights, handle, event = buffer.dispatch(
                    x, handle=handle, do_expand=True, topk_weights=input_weights
                )
                event.current_stream_wait()
                torch.npu.synchronize()
            samples = args.profile_samples * (2 if args.submit_detail else 1)
            for sample in range(samples):
                mode = 1 if sample < args.profile_samples else 2
                if detail is not None:
                    detail.zero_()
                    buffer._dispatch_diagnostics["detail_mode"] = mode
                profile.zero_()
                if weight_profile is not None:
                    weight_profile.zero_()
                torch.npu.synchronize()
                output, _, weights, handle, event = buffer.dispatch(
                    x, handle=handle, do_expand=True, topk_weights=input_weights
                )
                event.current_stream_wait()
                torch.npu.synchronize()
                cores = metrics["decode_profile"](
                    profile.cpu().tolist(), rank, fp8=fp8, weights=args.weights, world=world, qp1_workers=8
                )
                assert torch.equal((output[0] if fp8 else output).cpu().view(torch.uint8), actual), (
                    "profile hidden mismatch"
                )
                if fp8:
                    assert torch.equal(output[1].cpu(), actual_scales), "profile scale mismatch"
                weight_cores = []
                if args.weights:
                    assert torch.equal(weights.cpu(), expected_weights), "profile weight mismatch"
                    weight_cores = metrics["decode_weight_profile"](weight_profile.cpu().tolist(), rank)
                    assert sum(c["received_entries"] for c in weight_cores) == len(selected)
                detail_cores = (
                    metrics["decode_detail"](detail.cpu().tolist(), rank, args.rotation_wqes, mode)
                    if detail is not None
                    else []
                )
                kernel_profiles.append(
                    {
                        "sample": sample,
                        "detail_mode": mode if detail is not None else 0,
                        "detail_cores": detail_cores,
                        "cores": cores,
                        "weight_cores": weight_cores,
                        "instrumented_device_ms": buffer._dispatch_diagnostics["device_ms"],
                    }
                )
            del buffer._dispatch_diagnostics
            log(f"kernel.profile.done samples={args.profile_samples}")
        routing_samples = []
        if args.routing_profile:
            # Fresh handles force both Notify and peer-table preparation on every sample.
            indices = torch.from_numpy(routes[rank].copy()).npu()
            for iteration in range(args.warmup + args.iterations):
                buffer._routing_diagnostics = {}
                output, _, weights, fresh_handle, event = buffer.dispatch(
                    x, topk_idx=indices, num_experts=args.experts, do_expand=True, topk_weights=input_weights
                )
                event.current_stream_wait()
                torch.npu.synchronize()
                if iteration >= args.warmup:
                    routing_samples.append(dict(buffer._routing_diagnostics))
                del fresh_handle
            del buffer._routing_diagnostics
            assert torch.equal((output[0] if fp8 else output).cpu().view(torch.uint8), actual), (
                "routing timing hidden mismatch"
            )
            if fp8:
                assert torch.equal(output[1].cpu(), actual_scales), "routing timing scale mismatch"
            if args.weights:
                assert torch.equal(weights.cpu(), expected_weights), "routing timing weight mismatch"
            log(f"routing.events.done samples={len(routing_samples)}")
        log("destroy.begin")
        buffer.destroy()
        log("destroy.done")
        check_retained_outputs()
        assert torch.equal((output[0] if fp8 else output).cpu().view(torch.uint8), actual), (
            "output invalid after destroy"
        )
        if fp8:
            assert torch.equal(output[1].cpu(), actual_scales), "output scales invalid after destroy"
        if args.weights:
            assert torch.equal(weights.cpu(), expected_weights), "output weights invalid after destroy"
        log("correctness.output_lifetime.passed")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "rank": rank,
                    "world": world,
                    "tokens": args.tokens,
                    "rotation_wqes": args.rotation_wqes,
                    "topk": args.topk,
                    "experts": args.experts,
                    "dtype": args.dtype,
                    "weights": args.weights,
                    "qp1_workers": 8,
                    "fm_transport": "urma",
                    "hidden": 7168,
                    "scales": fp8,
                    "route_profile": args.route_profile,
                    "route_seed": args.seed,
                    "route_sha256": route_sha256,
                    "backend": args.backend,
                    "event_only": args.event_only,
                    "checked_generations": 3,
                    "checked_rows": len(selected),
                    "checked_retained_outputs": True,
                    "source_sha256": info["source_sha256"],
                    "cached_public_api_wall_ms": timings,
                    "timing_boundary": "Python dispatch through NPU synchronize, includes API validation and host synchronization; not kernel-only",
                    "payload_copy_mode": "none; direct framework GM input/output",
                    "copy_timing_note": "input_copy/output_copy are legacy empty event intervals, not payload copies",
                    "routing_device_ms": routing_samples,
                    "routing_time_columns": {
                        "notify_device_ms": ["kernel", "post_barrier", "count_copy", "total"],
                        "prepare_device_ms": ["clear", "kernel", "event_gap", "total"],
                    },
                    "clean_device_ms": clean_device_ms,
                    "kernel_profiles": kernel_profiles,
                    "device_time_columns": ["input_copy", "kernel", "output_copy", "device_total"],
                    "profile_ticks_per_us": 1000,
                },
                indent=2,
            )
            + "\n"
        )
    except BaseException:
        log(f"FAILED rank={rank} generation={generation}; traceback follows")
        traceback.print_exc()
        # Preserve only this worker's SHMEM logs, before scheduler cleanup.
        for path in Path("/root/shmem/log").glob(f"aclshmem_{os.getpid()}_*.log"):
            try:
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / f"rank-{rank}-{path.name}").write_bytes(path.read_bytes())
            except OSError:
                pass
        # Avoid collective destructor hangs after a peer fails. The bounded
        # launcher waits for all nodes; the rack scheduler owns final cleanup.
        os._exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    helper["_add_launch_args"](parser)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--backend", choices=("peer-table",), default="peer-table")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--routing-profile",
        action="store_true",
        help="Measure Notify and peer-table preparation with fresh handles; requires --kernel-profile",
    )
    parser.add_argument("--kernel-profile", action="store_true")
    parser.add_argument(
        "--event-only", action="store_true", help="Clean kernel Events only; no instrumented Dispatch passes"
    )
    parser.add_argument("--rotation-wqes", type=int, choices=(4, 16), default=4)
    parser.add_argument("--submit-detail", action="store_true")
    parser.add_argument("--bootstrap-gate", action="store_true")
    parser.add_argument("--weights", action="store_true", help="Enable FP32 route weights and bitwise validation")
    parser.add_argument("--without-weights", dest="weights", action="store_false")
    parser.set_defaults(weights=False)
    parser.add_argument("--route-profile", choices=("private-random", "pattern"), default="private-random")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile-samples", type=int, default=2)
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 10240 or not 1 <= args.topk <= 16 or args.ep_size**2 * args.tokens * args.topk >= 2**31:
        parser.error("Require T1..10240, K1..16 and EP*EP*T*K < 2**31")
    if args.event_only and args.kernel_profile:
        parser.error("Choose event-only or kernel-profile, not both")
    if not __debug__ or args.iterations < 1 or args.warmup < 0:
        parser.error("Assertions and positive iterations required")
    if args.kernel_profile and (args.backend != "peer-table" or args.profile_samples < 1 or args.profile_warmup < 0):
        parser.error("Kernel profile requires peer-table and positive profile samples")
    if args.routing_profile and not args.kernel_profile:
        parser.error("Routing profile requires kernel-profile for the combined result summary")
    if args.submit_detail and not args.kernel_profile:
        parser.error("Submit detail requires kernel-profile")
    helper["_launch"](args, worker)


if __name__ == "__main__":
    main()
