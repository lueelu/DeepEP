# Copyright (c) 2026, Lu Lu

"""Forward-list contract checks and installed-wheel, bitwise NPU dispatch validation.

CPU: python -m pytest tests/elastic/test_dispatch.py -q
NPU: python tests/elastic/test_dispatch.py --ep-size 2 --nproc-per-node 2 \
     --output /tmp/dispatch-check-001
"""

import argparse
from contextlib import contextmanager, nullcontext
import csv
import math
import statistics
import json
import os
from pathlib import Path
import importlib
from importlib.machinery import PathFinder
import sys
import sysconfig
import time
import traceback
import runpy
import random


_ROUTES = ("normal", "random", "duplicates", "invalid", "empty", "hotspot")
_profile = runpy.run_path(str(Path(__file__).with_name("test_notify.py")))


def add_dispatch_barrier_arg(parser):
    parser.add_argument(
        "--dispatch-barrier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align HT Dispatch performance samples with an entry device barrier (default: enabled)",
    )


@contextmanager
def dispatch_benchmark(buffer, *, enabled=True, kernel_timing=False):
    """Apply benchmark options temporarily, including when sampling fails."""
    missing = object()
    previous = getattr(buffer, "_dispatch_diagnostics", missing)
    diagnostics = dict(previous) if isinstance(previous, dict) else {}
    diagnostics.update(
        benchmark_barriers=False,
        benchmark_entry_barrier=enabled,
        kernel_timing=kernel_timing,
    )
    buffer._dispatch_diagnostics = diagnostics
    try:
        yield diagnostics
    finally:
        if previous is missing:
            del buffer._dispatch_diagnostics
        else:
            buffer._dispatch_diagnostics = previous


def _progress_logger(operation, rank):
    started = previous = time.perf_counter()

    def log(message):
        nonlocal previous
        now = time.perf_counter()
        print(
            f"[{operation}][rank={rank}] {time.strftime('%H:%M:%S')} "
            f"elapsed={now - started:.3f}s delta={now - previous:.3f}s {message}",
            flush=True,
        )
        previous = now

    return log


def _log_failure(log):
    # Cleanup may wait for device work or peers; expose the original error first.
    log("FAILED: traceback before buffer.destroy")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()


def group_peer(rank, world, group, index):
    if world == 128:
        toggle, subround = divmod(group, 4)
        row, side = rank % 64 // 32, rank % 8 // 4
        peer_side = side ^ (subround % 2)
        peer_row = (row + subround // 2) % 2
        frame = rank // 64 ^ side ^ toggle
        return frame * 64 + (peer_row * 4 + index // 4) * 8 + peer_side * 4 + index % 4
    width, half = min(world, 16), world // 2
    half_width, groups = width // 2, world // width
    own = rank % half // half_width
    target = (own + group) % groups
    return index // half_width * half + target * half_width + index % half_width


def group_steps(rank, world):
    steps = [0] * world
    for group in range(world // min(world, 16)):
        for index in range(min(world, 16)):
            steps[group_peer(rank, world, group, index)] = (group * min(world, 16) + index) // min(world, 64)
    return steps


def build_notify_golden(routes, experts, chunk_tokens, *, rank_dedup=None):
    world = len(routes)
    if rank_dedup is None:
        rank_dedup = world in (8, 16)
    if rank_dedup:
        return _profile["build_notify_golden"](routes, experts, chunk_tokens, rank_dedup=True)
    if world not in (2, 4, 8, 16, 32, 64, 128, 256) or experts <= 0 or experts % world:
        raise ValueError("unsupported Notify shape")
    tokens = len(routes[0])
    topk = len(routes[0][0]) if tokens else 0
    if (
        chunk_tokens <= 0
        or not 1 <= topk <= 16
        or any(len(rows) != tokens or any(len(row) != topk for row in rows) for rows in routes)
    ):
        raise ValueError("routes must have a common [tokens, topk] shape")
    local_experts = experts // world
    splits = min(4, tokens, chunk_tokens)
    first = min(tokens, chunk_tokens)
    chunks = (tokens + chunk_tokens - 1) // chunk_tokens + splits - 1

    def token_chunk(token):
        return token * splits // first if token < first else token // chunk_tokens + splits - 1

    def chunk_end(chunk):
        return (
            (chunk + 1) * first // splits + bool((chunk + 1) * first % splits)
            if chunk < splits
            else min((chunk + 2 - splits) * chunk_tokens, tokens)
        )

    counts = [0] * experts
    for rows in routes:
        for row in rows:
            for expert in row:
                if 0 <= expert < experts:
                    counts[expert] += 1
    starts = [0] * experts
    rank_rows = [0] * world
    for expert, count in enumerate(counts):
        peer = expert // local_experts
        starts[expert] = rank_rows[peer]
        rank_rows[peer] += count
    stride = max(1, max(rank_rows, default=0))
    invalid = -(1 << 63)
    plans = []
    for source in range(world):
        plans.append(
            {
                "server_mask": [0] * tokens,
                "dst": [invalid] * (tokens * topk),
                "weight_return_meta": [-1] * rank_rows[source],
                "forward_list": [[] for _ in range(world)],
                "backward_list": [[] for _ in range(min(world, 8))],
                "chunk_ranges": [[(0, 0) for _ in range(chunks)] for _ in range(world)],
                "chunk_masks": [0] * (world // min(world, 64) * chunks),
            }
        )

    expert_cursors = [0] * experts
    gathers = [0] * world
    for source, rows in enumerate(routes):
        for token, route in enumerate(rows):
            by_server = {}
            for slot, expert in enumerate(route):
                if not 0 <= expert < experts:
                    continue
                peer = expert // local_experts
                output_row = starts[expert] + expert_cursors[expert]
                expert_cursors[expert] += 1
                plans[peer]["weight_return_meta"][output_row] = (source * tokens + token) * topk + slot
                server = peer // 8
                plans[source]["server_mask"][token] |= 1 << server
                by_server.setdefault(server, []).append((slot, peer, output_row))
            for records in by_server.values():
                primary_slot, gateway, primary_row = next((r for r in records if r[1] == source), records[0])
                for slot, peer, output_row in records:
                    encoded = peer * stride + output_row
                    value = encoded if slot == primary_slot else ~encoded
                    plans[source]["dst"][token * topk + slot] = value
                forward = plans[gateway]["forward_list"][source]
                if len(records) == 1:
                    forward.append((source, token, primary_row, -1, -1, -1))
                    continue
                for slot, peer, output_row in records:
                    if slot == primary_slot:
                        continue
                    reduce_row = -1
                    if peer != gateway:
                        reduce_row = gathers[gateway]
                        gathers[gateway] += 1
                        plans[peer]["backward_list"][gateway % 8].append((source, output_row, reduce_row, token))
                    forward.append((source, token, primary_row, peer, output_row, reduce_row))

    for gateway, plan in enumerate(plans):
        for source, block in enumerate(plan["forward_list"]):
            begin = 0
            while begin < len(block):
                end = begin + 1
                while end < len(block) and block[end][1] == block[begin][1]:
                    end += 1
                for row in range(begin, end):
                    block[row] = (end, *block[row][1:])
                begin = end
            cursor = 0
            for chunk in range(chunks):
                begin = cursor
                token_end = chunk_end(chunk)
                while cursor < len(block) and block[cursor][1] < token_end:
                    peer = block[cursor][3]
                    if peer >= 0 and peer != gateway:
                        plan["chunk_masks"][group_steps(gateway, world)[source] * chunks + chunk] |= 1 << (peer % 8)
                    cursor += 1
                plan["chunk_ranges"][source][chunk] = (begin, cursor)
        for lane, block in enumerate(plan["backward_list"]):
            gateway_peer = gateway // 8 * 8 + lane
            gateway_steps = group_steps(gateway_peer, world)
            ordered = list(enumerate(block))
            ordered.sort(
                key=lambda item: (
                    token_chunk(item[1][3]),
                    gateway_steps[item[1][0]],
                    item[1][0],
                    item[0],
                )
            )
            block[:] = [item[1] for item in ordered]
            for index, (source, row, reduce_row, token) in enumerate(block):
                key = gateway_steps[source] * chunks + token_chunk(token) + 1
                next_key = 0
                if index + 1 == len(block):
                    next_key = key
                else:
                    next_source, _, _, next_token = block[index + 1]
                    candidate = gateway_steps[next_source] * chunks + token_chunk(next_token) + 1
                    if candidate != key:
                        next_key = key
                block[index] = (source, row, reduce_row | (next_key << 32))
        plan["forward_counts"] = [len(block) for block in plan["forward_list"]]
        plan["backward_counts"] = [len(block) for block in plan["backward_list"]]
    for plan in plans:
        plan["rank_rows"] = rank_rows
        plan["gather_rows"] = gathers
    return {"plans": plans, "rank_rows": rank_rows, "gather_rows": plans[0]["gather_rows"], "address_stride": stride}


def make_routes(world, tokens, topk, experts, generation, mode, *, seed=1, without_replacement=False):
    """Generate deterministic routes, including duplicate expert slots.

    Seed/source/token determine routing; generation only changes normal fixtures.
    Keep random routes fixed across payload generations for cached dispatch.
    """
    if mode == "random" and without_replacement and not 1 <= topk <= experts:
        raise ValueError("Random sampling without replacement requires 1 <= topk <= experts")
    result = []
    for source in range(world):
        rows = []
        for token in range(tokens):
            row = [(source * 17 + token * 7 + slot * 131 + generation) % experts for slot in range(topk)]
            if mode == "random":
                state = (seed ^ (source * 0x9E3779B9) ^ (token * 0x85EBCA6B)) & 0xFFFFFFFF
                if without_replacement:
                    row = random.Random(state).sample(range(experts), topk)
                else:
                    row = []
                    for _ in range(topk):
                        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                        row.append((state >> 8) % experts)
            if mode == "duplicates" and topk > 1:
                row[1] = row[0]
            if mode == "invalid":
                row = [
                    (experts + 17 if (source + token + slot) % 2 else -1) if (source + token + slot) % 3 == 0 else e
                    for slot, e in enumerate(row)
                ]
            if mode == "empty":
                row = [-1] * topk
            if mode == "hotspot":
                row = [experts - 1] * topk
            if mode == "sq_pressure":
                row = [0 if source in (8, 20, 48, 52, 80, 84, 112, 116) else -1] * topk
            if mode in ("local_duplicates", "local_experts"):
                local = experts // world
                peer = (source + token) % world
                row = [peer * local + (slot % local if mode == "local_experts" else 0) for slot in range(topk)]
            rows.append(row)
        result.append(rows)
    return result


def receive_capacity(routes, experts):
    """Exact maximum expanded rows, without constructing all Notify metadata."""
    world = len(routes)
    if experts < world or experts % world:
        raise ValueError("experts must be divisible by EP and at least EP")
    local = experts // world
    counts = [0] * world
    for source in routes:
        for token in source:
            for expert in token:
                if 0 <= expert < experts:
                    counts[expert // local] += 1
    return max(1, max(counts))


def test_random_routes_match_reference_and_preserve_cached_routing():
    import runpy

    # Compare with the CPU reference's independent random route implementation.
    reference = runpy.run_path(str(Path(__file__).with_name("combine_reference.py")))
    for world in (2, 8, 256):
        for seed in (0, 1, 2026, -1, 1 << 40):
            case = reference["CombineCase"](world, 7, 6, world * 2, "random", seed)
            actual = make_routes(world, 7, 6, world * 2, 0, "random", seed=seed)
            expected = [
                [reference["route_experts"](source, token, case) for token in range(7)] for source in range(world)
            ]
            assert actual == expected
            assert actual == make_routes(world, 7, 6, world * 2, 1, "random", seed=seed)
            assert all(0 <= expert < world * 2 for rows in actual for row in rows for expert in row)
    routes = make_routes(8, 32, 6, 16, 0, "random", seed=1)
    assert routes != make_routes(8, 32, 6, 16, 0, "random", seed=2026)
    assert routes[0] != routes[1]
    assert any(len(set(row)) < len(row) for rows in routes for row in rows)
    counts = [len(expert_rows(routes, 16, rank)) for rank in range(8)]
    assert len(set(counts)) > 1  # No balancing pass or per-token deduplication.
    assert receive_capacity(routes, 16) == max(counts)
    assert receive_capacity([[[-1, 2]], [[-1, 2]]], 2) == 1
    assert make_routes(8, 7, 6, 16, 0, "normal", seed=1) == make_routes(8, 7, 6, 16, 0, "normal", seed=2026)


def test_random_dispatch_cli(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_dispatch.py",
            "--ep-size",
            "256",
            "--nproc-per-node",
            "8",
            "--node-rank",
            "31",
            "--tokens",
            "4096",
            "--topk",
            "6",
            "--experts",
            "512",
            "--route",
            "random",
            "--seed",
            "2026",
            "--output",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(sys.modules[__name__], "_launch", lambda args, worker: calls.append(args))
    main()
    (args,) = calls
    assert (args.ep_size, args.node_rank, args.route, args.seed) == (256, 31, "random", 2026)


def check_handle(handle, golden, rank, routes, experts):
    import torch

    plan = golden["plans"][rank]
    if handle.metadata_only or handle.metadata_version != 7:
        raise AssertionError("Expected expanded dispatch handle metadata v6")
    for name in (
        "server_mask",
        "dst",
        "forward_counts",
        "backward_counts",
        "chunk_ranges",
        "chunk_masks",
        "rank_rows",
        "gather_rows",
    ):
        actual = getattr(handle, name)
        if actual.device.type != "npu":
            raise AssertionError(f"{name} is not an NPU tensor")
        expected = torch.tensor(plan[name], dtype=actual.dtype).reshape(actual.shape)
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError(f"rank {rank}: {name} mismatch")
    for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
        actual = getattr(handle, name).cpu()
        for row, count in enumerate(plan[counts]):
            expected = torch.tensor(plan[name][row], dtype=actual.dtype).reshape(count, actual.shape[-1])
            if not torch.equal(actual[row, :count], expected):
                raise AssertionError(f"rank {rank}: {name}[{row}] mismatch")
    weight_meta = handle.weight_return_meta
    weight_meta_cpu = weight_meta.cpu()
    expected_weight_meta = torch.tensor(
        [(s * handle.num_tokens + t) * handle.num_topk + k for s, t, k in expert_rows(routes, experts, rank)],
        dtype=torch.int32,
    )
    if (
        weight_meta.dtype != torch.int32
        or weight_meta.device.type != "npu"
        or not weight_meta.is_contiguous()
        or not torch.equal(weight_meta_cpu, expected_weight_meta)
    ):
        raise AssertionError(f"rank {rank}: weight_return_meta mismatch")
    totals = [0] * experts
    for source in routes:
        for row in source:
            for expert in row:
                if 0 <= expert < experts:
                    totals[expert] += 1
    local = experts // len(routes)
    expected_counts = totals[rank * local : (rank + 1) * local]
    if handle.num_recv_tokens_per_expert_list != expected_counts:
        raise AssertionError("Expert counts mismatch")
    for name, expected in (
        ("expert_totals", torch.tensor(totals, dtype=torch.int32)),
        ("num_unaligned_recv_tokens_per_expert", torch.tensor(expected_counts, dtype=torch.int32)),
        (
            "psum_num_recv_tokens_per_expert",
            torch.tensor(expected_counts, dtype=torch.int32).cumsum(0, dtype=torch.int32),
        ),
    ):
        if not torch.equal(getattr(handle, name).cpu(), expected):
            raise AssertionError(f"rank {rank}: {name} mismatch")
    if handle.num_recv_tokens is not None or handle.dst_buffer_slot_idx is not None:
        raise AssertionError("Expanded dispatch must leave ordinary-layout fields unset")
    if handle.address_stride != golden["address_stride"]:
        raise AssertionError("Destination stride mismatch")
    if handle.num_expanded_tokens != golden["rank_rows"][rank]:
        raise AssertionError("Expanded token count mismatch")


def _import_installed_deep_ep():
    # Launching from the checkout or inheriting PYTHONPATH must not shadow the wheel.
    locations = list(dict.fromkeys(sysconfig.get_path(name) for name in ("purelib", "platlib")))
    spec = PathFinder.find_spec("deep_ep", locations)
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"deep_ep is not installed in {sys.executable}; "
            "run python -m pip install --no-build-isolation --no-deps . first"
        )
    expected = Path(spec.origin).resolve()
    sys.path.insert(0, str(Path(spec.origin).parent.parent))
    package = importlib.import_module("deep_ep")
    actual = Path(package.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"deep_ep was already loaded from {actual}; expected installed package {expected}")
    return package


def _add_launch_args(parser):
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8899)
    parser.add_argument("--shmem-ip-port", help="Defaults to DEEPEP_SHMEM_ENDPOINT or master-port + 1")


def _launch(args, worker):
    # torchrun may still supply process identities, but no process group is created.
    launched = [name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    if any(launched) and not all(launched):
        raise ValueError("RANK, WORLD_SIZE and LOCAL_RANK must be supplied together")
    if all(launched):
        args.ep_size = int(os.environ["WORLD_SIZE"])
    elif (
        args.nproc_per_node < 1
        or args.ep_size % args.nproc_per_node
        or not 0 <= args.node_rank < args.ep_size // args.nproc_per_node
    ):
        raise ValueError("EP must be divisible by nproc-per-node; node-rank must be within the job")
    if args.ep_size not in (2, 4, 8, 16, 32, 64, 128, 256):
        raise ValueError("Dispatch/combine EP must be a power of two from 2 to 256")
    endpoint = args.shmem_ip_port or os.environ.get("DEEPEP_SHMEM_ENDPOINT")
    if not endpoint:
        master = os.environ.get("MASTER_ADDR", args.master_addr) if all(launched) else args.master_addr
        port = int(os.environ.get("MASTER_PORT", args.master_port)) if all(launched) else args.master_port
        endpoint = f"tcp://{master}:{port + 1}"
    os.environ["DEEPEP_SHMEM_ENDPOINT"] = endpoint
    if all(launched):
        worker(int(os.environ["LOCAL_RANK"]), args)
    else:
        import torch.multiprocessing as mp

        mp.spawn(worker, args=(args,), nprocs=args.nproc_per_node, join=True)


def _worker_identity(local_rank, args):
    rank = int(os.environ.get("RANK", args.node_rank * args.nproc_per_node + local_rank))
    if not 0 <= rank < args.ep_size or local_rank < 0:
        raise ValueError("Invalid rank or local rank")
    return rank, args.ep_size


def test_ep256_launcher(monkeypatch):
    from types import ModuleType

    parser = argparse.ArgumentParser()
    _add_launch_args(parser)
    args = parser.parse_args(
        [
            "--ep-size",
            "256",
            "--nproc-per-node",
            "8",
            "--node-rank",
            "31",
            "--shmem-ip-port",
            "tcp://192.0.2.1:19091",
        ]
    )
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    torch = ModuleType("torch")
    mp = ModuleType("torch.multiprocessing")
    mp.spawn = lambda worker, args, nprocs, join: calls.append((args, nprocs, join))
    torch.multiprocessing = mp
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", mp)
    # Let monkeypatch restore the endpoint written by the launcher.
    monkeypatch.setenv("DEEPEP_SHMEM_ENDPOINT", "tcp://192.0.2.2:1")
    _launch(args, lambda *_: None)
    assert calls == [((args,), 8, True)]
    assert os.environ["DEEPEP_SHMEM_ENDPOINT"] == "tcp://192.0.2.1:19091"
    assert [_worker_identity(local, args)[0] for local in range(8)] == list(range(248, 256))


def expert_rows(routes, experts, rank):
    """Independent destination order: local expert, source rank, token, top-k slot."""
    local = experts // len(routes)
    begin = rank * local
    buckets = [[] for _ in range(local)]
    for source, tokens in enumerate(routes):
        for token, row in enumerate(tokens):
            for slot, selected in enumerate(row):
                if begin <= selected < begin + local:
                    buckets[selected - begin].append((source, token, slot))
    return [entry for bucket in buckets for entry in bucket]


def test_forward_list_covers_unique_expert_rows():
    # Exercise both primary core maps, cross-server forwarding, same-rank
    # duplicates, singleton records, empty ranks and non-power-of-two strides.
    for world in (2, 8, 16, 32, 64, 128, 256):
        for mode in _ROUTES:
            tokens, topk, experts = 3, 6, world * 2
            routes = make_routes(world, tokens, topk, experts, 0, mode)
            golden = build_notify_golden(routes, experts, 256)
            stride = golden["address_stride"]
            output = [[None] * n for n in golden["rank_rows"]]

            def write(peer, row, value):
                assert output[peer][row] is None, "Two tasks write the same output row"
                output[peer][row] = value

            for source, plan in enumerate(golden["plans"]):
                for route, encoded in enumerate(plan["dst"]):
                    if encoded >= 0:
                        peer, row = divmod(encoded, stride)
                        write(peer, row, (source, route // topk))
            primary = [list(rows) for rows in output]
            workers = 16 if world == 128 else 24
            for gateway, plan in enumerate(golden["plans"]):
                for source, records in enumerate(plan["forward_list"]):
                    covered = []
                    for worker in range(workers):
                        begin, end = len(records) * worker // workers, len(records) * (worker + 1) // workers
                        covered.extend(range(begin, end))
                        for record in records[begin:end]:
                            _, token, primary_row, peer, row, _ = record
                            if token < 0:
                                token = (-token - 1) // max((world + 7) // 8, topk)
                            assert primary[gateway][primary_row] == (source, token)
                            if peer >= 0:
                                assert gateway // 8 == peer // 8
                                write(peer, row, primary[gateway][primary_row])
                    assert covered == list(range(len(records)))
            for rank in range(world):
                assert output[rank] == [(source, token) for source, token, _ in expert_rows(routes, experts, rank)]


def _host_elastic_buffer(monkeypatch):
    """Exercise Python orchestration with real CPU tensors and a fake native backend."""
    from collections import Counter
    from contextlib import nullcontext
    from threading import RLock
    from types import SimpleNamespace
    import pytest
    from deep_ep.buffers import elastic
    from deep_ep.utils import event

    torch = pytest.importorskip("torch")
    calls = Counter()
    device = SimpleNamespace(type="npu", index=0)
    empty = torch.empty

    class Tensor(torch.Tensor):
        @property
        def device(self):
            return device

        def record_stream(self, stream):
            calls["record_stream"] += 1

        def cpu(self):
            calls["cpu"] += 1
            return super().cpu()

    def allocate(*args, **kwargs):
        if kwargs.get("device") is device:
            kwargs["device"] = "cpu"
            return torch.Tensor._make_subclass(Tensor, empty(*args, **kwargs), False)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    stream = SimpleNamespace(npu_stream=1, synchronize=lambda: calls.update(["sync"]))
    npu = SimpleNamespace(
        stream=lambda _: nullcontext(),
        current_device=lambda: 0,
        current_stream=lambda _: stream,
        Event=lambda: SimpleNamespace(record=lambda _: None, wait=lambda _: calls.update(["event_wait"])),
    )

    def notify_layout(rank, world, tokens, topk, experts, chunk, **kwargs):
        shapes = {name: ((16,), "int32") for name in elastic._PERSISTENT}
        shapes.update(
            dst=((tokens, topk), "int64"),
            rank_rows=((world,), "int32"),
            gather_rows=((world,), "int32"),
            expert_totals=((experts,), "int32"),
            weight_return_meta=((world * tokens * topk,), "int32"),
            _peer_plan=((16,), "uint8"),
        )
        return {
            "workspace_bytes": elastic.workspace_layout(world, tokens, topk, experts, chunk)["workspace_bytes"],
            "tensors": shapes,
        }

    def notify(session, indices, outputs, *args, num_cores=64, rank_dedup=False):
        assert num_cores in (32, 64)
        calls["rank_notify" if rank_dedup else "server_notify"] += 1
        calls["notify"] += 1
        for tensor in outputs.values():
            assert not torch.is_inference(tensor)
            tensor.zero_()
        outputs["rank_rows"].fill_(2)
        outputs["expert_totals"].fill_(2)
        outputs["gather_rows"].fill_(1)
        return None, None

    def layout(kind, args):
        calls[kind + "_layout"] += 1
        return getattr(elastic, kind + "_layout")(*args[1:])

    native = SimpleNamespace(
        notify_check_mapping=lambda _: None,
        notify_layout=notify_layout,
        notify=notify,
        dispatch_layout=lambda *args: layout("dispatch", args),
        combine_layout=lambda *args: layout("combine", args),
        dispatch=lambda session, tensors, *args: calls.update(
            ["dispatch", "rank_dispatch" if tensors["dedup_group_size"] == 1 else "server_dispatch"]
        ),
        combine=lambda session, tensors, *args: calls.update(
            ["combine", "combine_ep8_pipeline" if tensors.get("ep8_rank_pipeline", False) else "combine_default"]
        ),
        combine_ep8_mte_version=lambda: 2,
    )
    for module in (elastic, event):
        monkeypatch.setattr(module, "require_native", lambda _: native)
        monkeypatch.setattr(module, "npu_api", lambda: npu)
    buffer = elastic.ElasticBuffer.__new__(elastic.ElasticBuffer)
    buffer.hidden = 0
    buffer.num_max_tokens_per_rank, buffer.num_topk = 16, 1
    buffer._handle_token = object()
    buffer._runtime = SimpleNamespace(
        ensure_alive=lambda: None,
        notify_failed=False,
        notify_lock=RLock(),
        group=None,
        debug_collectives=False,
        device=0,
        rank=0,
        world_size=2,
        stream=stream,
        native=SimpleNamespace(peer_table=True),
        capacity=1 << 32,
        combine_only=False,
        closed=True,
    )
    return buffer, lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device), calls


def test_inference_dispatch_combine_and_cached_host_work(monkeypatch):
    import pytest
    from deep_ep.buffers.elastic import _PERSISTENT

    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    monkeypatch.setattr(dist, "all_gather", lambda *a, **kw: pytest.fail("Unexpected HT preflight collective"))
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    buffer._runtime.group = object()
    with torch.inference_mode():
        x, indices = tensor((2, 7168), torch.bfloat16), tensor((2, 1), torch.int64)
        recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=2)
        assert torch.is_inference(x) and torch.is_inference(indices)
        assert handle.topk_idx.data_ptr() != indices.data_ptr()
        for name in ("topk_idx", "dispatch_dst", *_PERSISTENT, "_peer_plan"):
            value = getattr(handle, name)
            assert not torch.is_inference(value)
            assert handle._routing_versions[name] == (value.data_ptr(), value._version)
        assert calls["cpu"] == 1
        assert calls["sync"] == 0
        assert handle.address_stride == 2 and handle._gather_stride == 1
        calls.clear()
        for cpu_sync in (None, False, True):
            buffer.combine(recv, handle, async_with_compute_stream=True)
            buffer.dispatch(x, handle=handle, do_cpu_sync=cpu_sync, async_with_compute_stream=True)
        assert calls["combine"] == calls["dispatch"] == 3
        assert calls["cpu"] == calls["sync"] == calls["notify"] == 0
        assert calls["combine_layout"] == 1 and calls["dispatch_layout"] == 0
        assert calls["event_wait"] and calls["record_stream"]
        handle.forward_counts.add_(1)
        with pytest.raises(ValueError, match="Cached routing was modified"):
            buffer.combine(recv, handle)
        with pytest.raises(ValueError, match="Cached routing was modified"):
            buffer.dispatch(x, handle=handle)
        assert calls["combine"] == calls["dispatch"] == 3


def test_dispatch_benchmark_flags_and_restore(monkeypatch):
    import pytest
    from deep_ep.buffers import elastic

    torch = pytest.importorskip("torch")
    buffer, tensor, _ = _host_elastic_buffer(monkeypatch)
    native = elastic.require_native("benchmark regression")
    forwarded = []
    monkeypatch.setattr(native, "dispatch", lambda session, tensors, *args: forwarded.append(dict(tensors)))
    x, indices = tensor((2, 7168), torch.bfloat16), tensor((2, 1), torch.int64)
    _, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=2)
    assert not forwarded[-1].get("benchmark_entry_barrier", False)
    assert not forwarded[-1].get("benchmark_barriers", False)
    parser = argparse.ArgumentParser()
    add_dispatch_barrier_arg(parser)
    for argv, expected in (([], True), (["--no-dispatch-barrier"], False)):
        enabled = parser.parse_args(argv).dispatch_barrier
        assert enabled is expected
        with dispatch_benchmark(buffer, enabled=enabled):
            buffer.dispatch(x, handle=handle)
            assert forwarded[-1]["benchmark_entry_barrier"] is enabled
            assert forwarded[-1]["benchmark_barriers"] is False
            assert forwarded[-1]["kernel_timing"] is False
    buffer.dispatch(x, handle=handle)
    assert "benchmark_entry_barrier" not in forwarded[-1]
    assert "kernel_timing" not in forwarded[-1]
    previous = {"benchmark_barriers": True}
    buffer._dispatch_diagnostics = previous
    with pytest.raises(RuntimeError, match="sample failed"):
        with dispatch_benchmark(buffer, kernel_timing=True):
            buffer.dispatch(x, handle=handle)
            assert forwarded[-1]["kernel_timing"] is True
            raise RuntimeError("sample failed")
    assert buffer._dispatch_diagnostics is previous
    assert previous["benchmark_barriers"] is True


def test_dispatch_cpu_sync_validation(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    x, indices = tensor((2, 128), torch.bfloat16), tensor((2, 1), torch.int64)
    with pytest.raises(NotImplementedError, match="Fresh notify"):
        buffer.dispatch(x, topk_idx=indices, num_experts=2, do_cpu_sync=False)
    assert calls["notify"] == calls["dispatch"] == 0
    _, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=2)
    for invalid in (0, 1, "false"):
        for route in ({"handle": handle}, {"topk_idx": indices, "num_experts": 2}):
            with pytest.raises(TypeError, match="do_cpu_sync"):
                buffer.dispatch(x, do_cpu_sync=invalid, **route)
    assert calls["notify"] == calls["dispatch"] == 1


def test_inference_routing_copy_policy(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    ordinary_indices = tensor((2, 1), torch.int64)
    with torch.inference_mode():
        x, indices = tensor((2, 7168), torch.bfloat16), tensor((2, 1), torch.int64)
        with pytest.raises(ValueError, match="do_handle_copy=True"):
            buffer.dispatch(x, topk_idx=indices, num_experts=2, do_handle_copy=False)
        assert calls["notify"] == calls["dispatch"] == 0
        assert not buffer._runtime.notify_failed
        recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=ordinary_indices, num_experts=2, do_handle_copy=False)
        assert handle.topk_idx is ordinary_indices
    buffer.combine(recv, handle)
    with torch.inference_mode():
        handle.forward_list = tensor(handle.forward_list.shape, handle.forward_list.dtype)
        with pytest.raises(ValueError, match="Cached routing was modified"):
            buffer.combine(recv, handle)
    assert calls["combine"] == 1


def test_default_dedup_dispatch_combine_and_cached_handle(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    for world in (2, 4, 8, 16, 32, 64, 128, 256):
        for override in (None, False):
            with monkeypatch.context() as patch:
                buffer, tensor, calls = _host_elastic_buffer(patch)
                buffer._runtime.world_size = world
                if override is not None:
                    buffer._notify_rank_dedup = override
                x, indices = tensor((2, 128), torch.bfloat16), tensor((2, 1), torch.int64)
                indices.zero_()
                recv, _, _, handle, _ = buffer.dispatch(
                    x, topk_idx=indices, num_experts=world, async_with_compute_stream=True
                )
                expected = world in (8, 16) if override is None else override
                assert handle._rank_return_plan == expected
                assert calls["rank_notify"] == int(expected)
                assert calls["server_notify"] == int(not expected)
                assert calls["rank_dispatch"] == int(expected)
                assert calls["server_dispatch"] == int(not expected)
                buffer.combine(recv, handle)
                assert calls["combine_ep8_pipeline"] == int(world == 8 and expected)
                assert calls["combine_default"] == int(world != 8 or not expected)
                _, _, _, cached, _ = buffer.dispatch(x, handle=handle)
                assert cached is handle and calls["notify"] == 1
                assert calls["combine"] == 1 and calls["dispatch"] == 2
                assert calls["rank_dispatch"] == 2 * int(expected)
                assert calls["server_dispatch"] == 2 * int(not expected)


def test_ep8_combine_weights_use_mte(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    buffer._runtime.world_size = 8
    x, indices = tensor((2, 128), torch.bfloat16), tensor((2, 1), torch.int64)
    indices.zero_()
    recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=8)
    buffer.combine(recv, handle)
    out, weights, _ = buffer.combine(recv, handle, topk_weights=tensor((2,), torch.float32))
    assert out.shape == (2, 128) and weights.shape == (2, 1)
    assert calls["combine_ep8_pipeline"] == 2 and calls["combine_default"] == 0
    assert calls["notify"] == 1 and calls["combine"] == 2


def test_ep8_combine_weights_require_native_support(monkeypatch):
    import pytest
    from deep_ep.buffers import elastic

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    buffer._runtime.world_size = 8
    native = elastic.require_native("test")
    native.combine_ep8_mte_version = lambda: 1
    x, indices = tensor((2, 128), torch.bfloat16), tensor((2, 1), torch.int64)
    indices.zero_()
    recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=8)
    buffer.combine(recv, handle)
    with pytest.raises(RuntimeError, match="EP8 return pipeline v2"):
        buffer.combine(recv, handle, topk_weights=tensor((2,), torch.float32))
    assert calls["combine"] == 1 and not buffer._runtime.notify_failed
    native.combine_ep8_mte_version = lambda: 2
    buffer.combine(recv, handle, topk_weights=tensor((2,), torch.float32))
    assert calls["combine_ep8_pipeline"] == 2


def test_cached_layout_variants_and_capacity_checks(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    x, indices = tensor((2, 7168), torch.bfloat16), tensor((2, 1), torch.int64)
    recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=2)
    weights = tensor((2, 1), torch.float32)
    buffer.dispatch(x, handle=handle, topk_weights=weights)
    buffer.dispatch(x, handle=handle, topk_weights=weights)
    fp8 = (tensor((2, 7168), torch.float8_e4m3fn), tensor((2, 56), torch.float32))
    buffer.dispatch(fp8, handle=handle, topk_weights=weights)
    buffer.dispatch(x, handle=handle)
    assert calls["dispatch_layout"] == 3
    buffer.combine(recv, handle)
    buffer._runtime.capacity = 0
    with pytest.raises(ValueError, match="Combine capacity"):
        buffer.combine(recv, handle)
    with pytest.raises(ValueError, match="SHMEM"):
        buffer.dispatch(x, handle=handle)
    assert calls["combine"] == 1 and calls["dispatch"] == 5
    assert not buffer._runtime.notify_failed


def test_hidden_shapes_and_cached_contract(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    indices = tensor((2, 1), torch.int64)
    for width in (16, 48, 128, 144, 1024, 4096, 6144, 7040, 7152, 7168):
        x = tensor((2, width), torch.bfloat16)
        recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=2)
        assert recv.shape == (2, width) and handle.hidden == width
        out, _, _ = buffer.combine(recv, handle)
        assert out.shape == (2, width)
        buffer.dispatch(x, handle=handle)
        if width % 128 == 0:
            fp8 = (tensor((2, width), torch.float8_e4m3fn), tensor((2, width // 128), torch.float32))
            recv_fp8, _, _, _, _ = buffer.dispatch(fp8, handle=handle)
            assert recv_fp8[0].shape == (2, width) and recv_fp8[1].shape == (2, width // 128)
        with pytest.raises(ValueError, match="original hidden"):
            buffer.dispatch(tensor((2, 32 if width == 16 else 16), torch.bfloat16), handle=handle)
        with pytest.raises(ValueError, match="matching the handle"):
            buffer.combine(tensor((2, width + 16), torch.bfloat16), handle)
    launched = calls["dispatch"]
    for width in (0, 15, 17, 7169, 7296):
        with pytest.raises(ValueError, match="hidden"):
            buffer.dispatch(tensor((2, width), torch.bfloat16), topk_idx=indices, num_experts=2)
    with pytest.raises(ValueError, match="scales"):
        buffer.dispatch(
            (tensor((2, 128), torch.float8_e4m3fn), tensor((2, 56), torch.float32)), topk_idx=indices, num_experts=2
        )
    buffer.hidden = 4096
    with pytest.raises(ValueError, match="buffer configuration"):
        buffer.dispatch(tensor((2, 7168), torch.bfloat16), topk_idx=indices, num_experts=2)
    assert calls["dispatch"] == launched


def test_scale_packet_scatter_hidden_sizes(tmp_path):
    import shutil
    import subprocess
    import pytest
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    header = (Path(__file__).resolve().parents[2] / "csrc/kernels/dispatch/scale_pipeline.hpp").read_text(
        encoding="utf-8"
    )
    expand = cpp_functions(header, "Expand")
    source = tmp_path / "scale.cpp"
    source.write_text(
        r"""

#include <cstdint>
#include <cassert>
#include <vector>
#define __simt_vf__
#define __launch_bounds__(n)
#define __ubuf__
#define __gm__
struct {
    unsigned x;
} threadIdx;
void asc_threadfence() {}
"""
        + expand
        + r"""

int main()
{
    for (unsigned words = 1; words <= 56; ++words) {
        std::vector<uint32_t> packed(3 * 128, 0xdeadbeef), errors(128, 0);
        auto* slots = reinterpret_cast<int32_t*>(packed.data());
        const int32_t targets[] = {2, 0, 1, -1, 3, -1};
        for (unsigned row = 0; row < 6; ++row) {
            slots[row / 2 * 128 + 112 + row % 2] = targets[row];
            for (unsigned col = 0; col < words; ++col)
                packed[row / 2 * 128 + row % 2 * 56 + col] = (row + 1) * 1000 + col;
        }
        std::vector<uint32_t> output(4 * words + 16, 0xabcddcba), expected = output;
        for (unsigned row = 0; row < 6; ++row)
            if (targets[row] >= 0)
                for (unsigned col = 0; col < words; ++col)
                    expected[8 + targets[row] * words + col] = (row + 1) * 1000 + col;
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            Expand(packed.data(), slots, output.data() + 8, 6, words, 56, 4, errors.data());
        assert(output == expected);
        output[8] ^= 1;
        assert(output != expected);
        for (auto error : errors) assert(error == 0);
        slots[112] = 4;
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            Expand(packed.data(), slots, output.data() + 8, 6, words, 56, 4, errors.data());
        unsigned bad = 0;
        for (auto error : errors) bad += error;
        assert(bad == 1);
    }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("scale.exe" if sys.platform == "win32" else "scale")
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=30)


def test_collective_check_copies_one_summary(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from deep_ep.buffers.elastic import _collective_check

    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    seen = []
    fail_rank = None

    def gather(peers, local, group):
        assert not torch.is_inference(local)
        assert local.numel() == 18
        assert len({peer.untyped_storage().data_ptr() for peer in peers}) == 1
        # Backend workers may write after leaving the caller's inference context.
        with torch.inference_mode(False):
            for rank, peer in enumerate(peers):
                peer.copy_(local)
                if rank == fail_rank:
                    peer[0] = 1
        seen.append(len(peers))

    monkeypatch.setattr(dist, "get_backend", lambda _: "gloo")
    monkeypatch.setattr(dist, "all_gather", gather)
    runtime = SimpleNamespace(group=object(), world_size=4, debug_collectives=True)
    with torch.inference_mode():
        _collective_check(runtime, None, [2, 1, 2], "Dispatch")
        fail_rank = 2
        with pytest.raises(ValueError, match=r"Dispatch preflight failed on ranks \[2\]"):
            _collective_check(runtime, None, [2, 1, 2], "Dispatch")
        with pytest.raises(ValueError, match="Combine preflight failed"):
            _collective_check(runtime, ValueError("invalid handle"), [], "Combine")
        with pytest.raises(ValueError, match="signature exceeds"):
            _collective_check(runtime, None, [0] * 15, "Combine")
    assert seen == [4] * 4


def _gloo_preflight_worker(rank, store_path):
    from datetime import timedelta
    import pytest
    import torch
    import torch.distributed as dist
    from deep_ep.buffers.elastic import _collective_check

    dist.init_process_group(
        "gloo", init_method=Path(store_path).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            buffer, tensor, calls = _host_elastic_buffer(patch)
            runtime = buffer._runtime
            runtime.group, runtime.rank = dist.group.WORLD, rank
            gather = dist.all_gather

            def counted_gather(*args, **kwargs):
                calls["all_gather"] += 1
                return gather(*args, **kwargs)

            patch.setattr(dist, "all_gather", counted_gather)
            for debug in (False, True):
                runtime.debug_collectives = debug
                calls.clear()
                x, indices = tensor((2, 128), torch.bfloat16), tensor((2, 1), torch.int64)
                weights = tensor((2, 1), torch.float32)
                recv, _, recv_weights, handle, _ = buffer.dispatch(
                    x, topk_idx=indices, topk_weights=weights, num_experts=2
                )
                assert calls["all_gather"] == (4 if debug else 0)
                assert calls["sync"] == (2 if debug else 0) and calls["cpu"] == 1
                calls.clear()
                buffer.combine(recv, handle, async_with_compute_stream=True)
                buffer.dispatch(x, handle=handle, do_cpu_sync=False, async_with_compute_stream=True)
                buffer.combine(recv, handle, topk_weights=recv_weights, async_with_compute_stream=True)
                assert calls["all_gather"] == (4 if debug else 0)
                assert calls["sync"] == calls["cpu"] == calls["notify"] == 0

                bad_x = tensor((2, 128), torch.float32) if rank == 1 else recv
                if debug or rank == 1:
                    with pytest.raises(ValueError, match="Combine preflight failed on ranks" if debug else "BF16"):
                        buffer.combine(bad_x, handle)
                assert calls["combine"] == 2

            for operation, phase, signature, message in (
                ("Combine", 0, [0] * (14 if rank else 9), "parameters"),
                ("Combine" if rank else "Dispatch", 0, [], "call order"),
                ("Dispatch", rank, [], "call order"),
            ):
                with pytest.raises(ValueError, match=message):
                    _collective_check(runtime, None, signature, operation, phase)

            with pytest.raises(ValueError, match=r"Dispatch preflight failed on ranks \[1\]"):
                buffer.dispatch(x, handle=handle, do_cpu_sync="invalid" if rank else False)
            original_notify = buffer._notify
            patch.setattr(buffer, "_notify", lambda *a, **kw: pytest.fail("Launched Notify after preflight failure"))
            with pytest.raises(ValueError, match=r"Dispatch preflight failed on ranks \[1\]"):
                buffer.dispatch(x, topk_idx=indices, num_experts=2, do_cpu_sync=False if rank else None)
            patch.setattr(buffer, "_notify", original_notify)
    finally:
        dist.destroy_process_group()


def test_ht_preflight_with_real_process_group(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Requires the CPU Gloo backend")
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_gloo_preflight_worker, args=(rank, str(tmp_path / "gloo"))) for rank in range(2)]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 90
        for worker in workers:
            worker.join(max(0, deadline - time.monotonic()))
        assert [worker.exitcode for worker in workers] == [0, 0]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(5)


def test_dispatch_control_reuse_host_waits(tmp_path):
    import shutil
    import subprocess
    import pytest

    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    root = Path(__file__).resolve().parents[2]
    wrapper = (root / "csrc/ops/dispatch.cpp").read_text(encoding="utf-8")
    body = wrapper[
        wrapper.index("        DevicePhaseTiming timing(measure);") : wrapper.index(
            "    });", wrapper.index("        DevicePhaseTiming timing(measure);")
        )
    ]
    session = (root / "csrc/runtime/session.hpp").read_text(encoding="utf-8")
    controls = session[session.index("    using DispatchControlLayout") : session.index("    bool udma()")]
    source = tmp_path / "dispatch_controls.cpp"
    source.write_text(
        r"""

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernels/dispatch/tiling.hpp"
using ascend_deepep::DispatchTiling;
std::string calls;
bool fail_clear = false;
struct Session {
"""
        + controls
        + r"""

DispatchControlLayout dispatch_control_layout_{};
bool dispatch_controls_valid_ = false;
uint64_t capacity() const
{
    return 1ULL << 32;
}
void barrier()
{
    calls += 'H';
}
}
;
int aclrtSynchronizeStream(void*)
{
    calls += 'S';
    return 0;
}
int aclrtMemsetAsync(void*, size_t bytes, int value, size_t count, void*)
{
    assert(value == 0 && bytes == count);
    calls += 'Z';
    return fail_clear ? -1 : 0;
}
void notify_dispatch_barrier_kernel_do(void*, unsigned cores)
{
    assert(cores == 64);
    calls += 'B';
}
void dispatch_kernel_do(void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*)
{
    calls += 'K';
}
struct DevicePhaseTiming {
    explicit DevicePhaseTiming(bool enabled)
    {
        assert(!enabled);
    }
    void mark(unsigned, void*) {}
    std::vector<float> finish(void*)
    {
        calls += 'F';
        return {};
    }
};
std::vector<float> run(Session& session, const DispatchTiling& t, bool with_weights,
                       bool benchmark_barriers = false, bool entry_only = false)
{
    const bool benchmark_entry_barrier = benchmark_barriers || entry_only;
    const bool measure = false;
    void* stream = nullptr;
    std::vector<uint8_t> storage(t.workspace_bytes);
    auto* workspace = storage.data();
    uint8_t *destinations = nullptr, *forward = nullptr, *counts = nullptr, *scales = nullptr;
    uint8_t* weights = with_weights ? workspace : nullptr;
    auto config = t;
"""
        + body
        + r"""

}
int main()
{
    Session session;
    auto check = [&](const DispatchTiling& t, bool weights, bool cold, bool benchmark = false, bool entry = false) {
        calls.clear();
        run(session, t, weights, benchmark, entry);
        const auto clears = 3U + t.world * (t.fp8 + unsigned(weights));
        const std::string launch = benchmark ? "BKBF" : (entry ? "BKF" : "KF");
        assert(calls == (cold ? "SH" + std::string(clears, 'Z') + "SH" + launch : launch));
    };
    for (bool fp8 : {false, true})
        for (bool weights : {false, true}) {
            for (unsigned rows : {16U, 256U}) {
                auto t = ascend_deepep::MakeDispatchTiling(0, 16, 16, 6, 384, rows, fp8, weights, 128);
                check(t, weights, true);
                for (int i = 0; i < 4; ++i) check(t, weights, false);
                check(t, weights, false, true);
                check(t, weights, false, false, true);
                check(t, weights, false);
                session.invalidate_dispatch_controls();  // Notify/LL overwrote the workspace.
                check(t, weights, true);
                check(t, weights, false);
                session.invalidate_dispatch_controls();
                fail_clear = true;
                try {
                    run(session, t, weights);
                    assert(false);
                } catch (const std::runtime_error&) {
                }
                assert(!session.dispatch_controls_valid_);
                fail_clear = false;
                check(t, weights, true);
            }
        }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("dispatch_controls.exe" if sys.platform == "win32" else "dispatch_controls")
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(root / "csrc"), str(source), "-o", str(binary)], check=True, timeout=60
    )
    subprocess.run([str(binary)], check=True, timeout=30)


def test_dispatch_destination_initialization():
    import ast
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[2] / "deep_ep/buffers/elastic.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Execute the production initialization block without requiring an NPU.
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "dispatch_dst" for target in statement.targets)
            for statement in node.body
        )
    )
    code = compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec")

    class ScalarTensor(int):
        def to(self, dtype):
            return self

    torch = SimpleNamespace(
        int32="int32", int64="int64", where=lambda condition, yes, no: ScalarTensor(yes if condition else no)
    )
    for raw, expected in ((0, 0), (2, 2), (3, 4), (5, 6), (-1, -1), (-4, -5), (-6, -7), (-(1 << 63), -(1 << 63))):
        handle = SimpleNamespace(dispatch_dst=None, dst=ScalarTensor(raw))
        scope = dict(handle=handle, rows=3, layout={"address_stride": 4}, torch=torch)
        exec(code, scope)
        assert isinstance(handle.dispatch_dst, ScalarTensor)
        assert handle.dispatch_dst == expected
        cached = handle.dispatch_dst
        handle.dst = object()  # Cached calls must not read/re-encode the Notify tensor.
        exec(code, scope)
        assert handle.dispatch_dst is cached


def test_dispatch_regions_and_encoding_bounds():
    import pytest
    from deep_ep.buffers._elastic_layout import dispatch_layout, workspace_layout

    for world in (2, 8, 16, 64, 128, 256):
        for fp8 in (False, True):
            for rows in (1, 3, world * 33 * 6):
                layout = dispatch_layout(world, 33, 6, 1024, rows, fp8)
                assert layout["sync"] == workspace_layout(world, 33, 6)["workspace_bytes"]
                assert rows <= layout["address_stride"] < 2 * rows
                offsets = [
                    layout[name]
                    for name in ("sync", "book", "input", "output", "weights", "scales", "inbox", "peer_book", "done")
                ]
                assert offsets == sorted(offsets) and all(n % 512 == 0 for n in offsets)
                row_bytes = 7168 if fp8 else 14336
                assert layout["weights"] - layout["output"] >= rows * row_bytes
                assert layout["peer_book"] == layout["inbox"] + world * layout["inbox_stride"]
                assert layout["done"] == layout["peer_book"] + world * 512
                assert layout["workspace_bytes"] == layout["done"] + world * 3 * 512
                assert layout["num_bytes"] >= layout["workspace_bytes"]
                assert layout["num_bytes"] % (2 << 20) == 0
    for arguments in ((1024, 1, 1, 1024, 1), (2, 1, 1, 1024, 3), (2, 1, 1, 1024, 0)):
        with pytest.raises(ValueError):
            dispatch_layout(*arguments)


def assert_bytes(actual, expected, name, progress=None):
    import torch

    if progress:
        progress(f"{name}: readback.begin (waiting for device result)")
    actual = actual.cpu().contiguous()
    if progress:
        progress(f"{name}: readback.done; precision.begin")
    expected = expected.cpu().contiguous()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape/dtype {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
    a, b = actual.view(torch.uint8).flatten(), expected.view(torch.uint8).flatten()
    bad = torch.nonzero(a != b).flatten()
    if bad.numel():
        first = bad[:8]
        raise AssertionError(
            f"{name}: {bad.numel()} mismatched bytes; offsets={first.tolist()}, "
            f"actual={a[first].tolist()}, expected={b[first].tolist()}"
        )
    if progress:
        progress(f"{name}: precision.passed")


def payload(source, tokens, generation, fp8, width=7168):
    import torch

    # Bounded independently of EP, token count and hidden size. Neither BF16
    # nor FP8 construction can overflow; no arithmetic runs in dispatch.
    rows = torch.arange(tokens, dtype=torch.int32)[:, None]
    columns = torch.arange(width, dtype=torch.int32)[None, :]
    hidden = ((source * 13 + rows * 7 + columns * 3 + generation * 11) % 127 - 63).float() / 16
    hidden = hidden.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
    scales = ((source + rows + torch.arange(width // 128)[None, :] + generation) % 7 + 1).float() / 8
    return hidden, scales


def weights_for(source, tokens, topk, generation):
    import torch

    return ((torch.arange(tokens * topk).reshape(tokens, topk) + source * 3 + generation) % 31 + 1).float() / 32


def expected_payload(routes, experts, rank, generation, fp8, *, entries=None, width=7168):
    import torch

    if entries is None:
        entries = expert_rows(routes, experts, rank)
    topk = len(routes[0][0])
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    hidden = torch.empty((len(entries), width), dtype=dtype)
    scales = torch.empty((len(entries), width // 128), dtype=torch.float32)
    weights = torch.empty((len(entries),), dtype=torch.float32)
    if not entries:
        return hidden, scales, weights
    columns = torch.arange(width, dtype=torch.int32)[None, :]
    # Quantize the 127 distinct rows once; gather bytes to preserve FP8 rounding.
    phases = torch.arange(127, dtype=torch.int32)[:, None]
    lookup = (((phases + columns * 3) % 127 - 63).float() / 16).to(dtype).view(torch.uint8)
    scale_columns = torch.arange(width // 128, dtype=torch.int32)[None, :]
    for begin in range(0, len(entries), 256):
        end = min(begin + 256, len(entries))
        coordinates = torch.tensor(entries[begin:end], dtype=torch.int32)
        source, token, slot = (coordinates[:, i : i + 1] for i in range(3))
        phase = ((source * 13 + token * 7 + generation * 11) % 127).flatten().long()
        hidden[begin:end].view(torch.uint8).copy_(lookup.index_select(0, phase))
        scales[begin:end] = ((source + token + scale_columns + generation) % 7 + 1).float() / 8
        weights[begin:end] = ((token * topk + slot + source * 3 + generation) % 31 + 1).flatten().float() / 32
    return hidden, scales, weights


_PROFILE_KERNELS = {"notify": "notify_full_kernel", "dispatch": "dispatch_kernel", "combine": "combine_kernel"}


add_profile_args = _profile["add_profile_args"]
validate_profile_args = _profile["validate_profile_args"]


def _statistics(kernel, durations):
    return dict(
        kernel=kernel,
        unit="us",
        samples=len(durations),
        durations_us=durations,
        avg_us=statistics.mean(durations),
        median_us=statistics.median(durations),
        min_us=min(durations),
        max_us=max(durations),
    )


def read_kernel_summary(directory, operator, expected_samples):
    """Select one CSV source only; retain all samples and reject incomplete captures."""
    kernel = _PROFILE_KERNELS[operator]
    wanted = {kernel}
    if operator == "combine":
        wanted.add("combine_prepare_kernel")
    for pattern in ("kernel_details*.csv", "op_summary*.csv", "task_time*.csv"):
        tables = []
        for path in sorted(Path(directory).rglob(pattern)):
            values = {name: [] for name in wanted}
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                headers = {key.strip(): key for key in reader.fieldnames or ()}
                name_key = next((headers[k] for k in ("Name", "Op Name", "Kernel Name") if k in headers), None)
                units = [
                    (f"{prefix}({unit})", scale)
                    for unit, scale in (("us", 1.0), ("ns", 0.001), ("ms", 1000.0))
                    for prefix in ("Duration", "Task Duration")
                ]
                duration = next(((headers[k], scale) for k, scale in units if k in headers), None)
                if name_key is None or duration is None:
                    continue
                duration_key, scale = duration
                for row in reader:
                    name = (row.get(name_key) or "").strip()
                    if name not in wanted:
                        continue
                    try:
                        value = float(row[duration_key]) * scale
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"Invalid duration for {name} in {path}") from exc
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"Invalid duration for {name} in {path}: {value}")
                    values[name].append(value)
            if values[kernel]:
                tables.append((path, values))
        if not tables:
            continue
        if len(tables) != 1:
            raise ValueError(f"Ambiguous {kernel} capture: {len(tables)} {pattern} tables under {directory}")
        path, values = tables[0]
        for name in wanted:
            if len(values[name]) != expected_samples:
                raise ValueError(f"{name}: expected {expected_samples} samples, found {len(values[name])} in {path}")
        summary = _statistics(kernel, values[kernel])
        summary.update(
            expected_samples=expected_samples, source_csv=str(path.resolve()), metric="device_kernel_duration"
        )
        if operator == "combine":
            summary["prepare"] = _statistics("combine_prepare_kernel", values["combine_prepare_kernel"])
        return summary
    raise ValueError(f"No {kernel} duration records found in {directory}")


def capture_calls(profiler, invoke, synchronize, warmup, iterations, log):
    """One warmup step and one active step, with exactly the requested call counts."""
    last = None
    if warmup:
        log(f"profile warmup.begin calls={warmup}")
        for _ in range(warmup):
            last = invoke()
        synchronize()
        profiler.step()
    log(f"profile capture.begin calls={iterations}")
    for _ in range(iterations):
        last = invoke()
    synchronize()
    profiler.step()
    return last


def profile_operator(args, rank, scenario, invoke, validate, build_info, log, *, stream=None):
    """Capture rank-local kernel durations after warmup."""
    import torch
    import torch_npu

    invoke = _profile["with_l2_flush"](args, invoke, stream)

    directory = args.output / "profile" / args.profile_op / scenario / f"rank_{rank}" / str(time.time_ns())
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        operator=args.profile_op,
        dispatch_entry_barrier=args.profile_op == "dispatch" and args.dispatch_barrier,
        rank=rank,
        world_size=args.ep_size,
        scenario=scenario,
        seed=args.seed,
        without_replacement=getattr(args, "without_replacement", False),
        tokens=args.tokens,
        topk=args.topk,
        experts=args.experts,
        hidden=getattr(args, "hidden", 7168),
        warmup=args.profile_warmup,
        iterations=args.profile_iters,
        flush_l2=getattr(args, "profile_flush_l2", False),
        flush_l2_bytes=_profile["_PROFILE_L2_BYTES"] if getattr(args, "profile_flush_l2", False) else 0,
        torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__),
        ascend_home_path=os.environ.get("ASCEND_HOME_PATH"),
        build_info=build_info,
    )
    toolkit = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    metadata["cann_version_files"] = {
        str(path): path.read_text(encoding="utf-8", errors="replace")
        for path in (toolkit / "version.cfg", toolkit / "version.info")
        if path.is_file()
    }
    (directory / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log(f"profile op={args.profile_op} scenario={scenario} flush_l2={int(metadata['flush_l2'])} output={directory}")
    torch.npu.synchronize()
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=int(args.profile_warmup > 0), active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=profiler._ExperimentalConfig(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level0,
            aic_metrics=profiler.AiCMetrics.AiCoreNone,
            data_simplification=False,
        ),
    ) as capture:
        last = capture_calls(capture, invoke, torch.npu.synchronize, args.profile_warmup, args.profile_iters, log)
    torch.npu.synchronize()
    validate(last)
    summary = read_kernel_summary(directory, args.profile_op, args.profile_iters)
    summary.update(metadata, precision_checked=True)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(
        f"profile kernel={summary['kernel']} samples={summary['samples']} avg_us={summary['avg_us']:.3f} "
        f"median_us={summary['median_us']:.3f} min_us={summary['min_us']:.3f} max_us={summary['max_us']:.3f}"
    )
    if "prepare" in summary:
        log(f"profile combine_prepare_kernel avg_us={summary['prepare']['avg_us']:.3f} excluded_from_combine=1")
    return summary


def notify_profile_call(buffer, handle, experts):
    """Prepare the existing native Notify entry independently of payload dispatch."""
    import torch
    from deep_ep._native import require_native
    from deep_ep._runtime import runtime_of

    runtime = runtime_of(buffer)
    native = require_native("notify profiling")
    specification = native.notify_layout(runtime.rank, runtime.world_size, handle.num_tokens, handle.num_topk, experts)
    if specification["workspace_bytes"] > runtime.capacity // 2:
        raise ValueError("Notify profile exceeds the reserved SHMEM half")
    native.notify_check_mapping(runtime.native)
    indices = handle.topk_idx
    with torch.npu.stream(runtime.stream):
        outputs = {
            name: torch.empty(shape, dtype=getattr(torch, dtype), device=indices.device)
            for name, (shape, dtype) in specification["tensors"].items()
        }
    torch.npu.synchronize()

    def invoke():
        with torch.npu.stream(runtime.stream):
            native.notify(
                runtime.native,
                indices,
                outputs,
                handle.num_tokens,
                handle.num_topk,
                experts,
                runtime.stream.npu_stream,
                rank_dedup=handle._rank_return_plan,
            )
        return outputs

    def validate(result):
        # The baseline handle has already passed the independent CPU golden.
        for name in (
            "server_mask",
            "dst",
            "forward_counts",
            "backward_counts",
            "chunk_ranges",
            "chunk_masks",
            "rank_rows",
            "gather_rows",
            "expert_totals",
        ):
            if not torch.equal(result[name].cpu(), getattr(handle, name).cpu()):
                raise AssertionError(f"profile notify: {name} differs from validated metadata")
        for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
            for lane, count in enumerate(getattr(handle, counts).cpu().tolist()):
                if not torch.equal(result[name][lane, :count].cpu(), getattr(handle, name)[lane, :count].cpu()):
                    raise AssertionError(f"profile notify: {name}[{lane}] differs from validated metadata")
        if not torch.equal(
            result["weight_return_meta"][: handle.num_expanded_tokens].cpu(), handle.weight_return_meta.cpu()
        ):
            raise AssertionError("profile notify: weight_return_meta differs from validated metadata")

    return invoke, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _add_launch_args(parser)
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp8", "both"), default="both")
    parser.add_argument("--route", choices=(*_ROUTES, "all"), default="all")
    _profile["add_random_route_args"](parser)
    _profile["add_dedup_args"](parser)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    add_profile_args(parser, ("notify", "dispatch"))
    add_dispatch_barrier_arg(parser)
    args = parser.parse_args()
    _profile["resolve_dedup_args"](args)
    if args.rank_dedup and args.ep_size not in (8, 16):
        parser.error("Rank-deduplicated Dispatch supports EP8/16")
    _profile["validate_random_route_args"](parser, args)
    validate_profile_args(parser, args)
    if not 0 < args.hidden <= 7168 or args.hidden % (16 if args.dtype == "bf16" else 128):
        parser.error("hidden must be <=7168 and divisible by 16 (BF16) or 128 (FP8)")
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE") or args.repeat < 2:
        parser.error("Enable assertions and use --repeat >=2")
    _launch(args, worker)


def worker(local_rank, args):
    rank, world = _worker_identity(local_rank, args)
    log = _progress_logger("dispatch", rank)
    log(f"worker.begin route={args.route} seed={args.seed}; import installed package")
    deep_ep = _import_installed_deep_ep()
    import torch
    import torch_npu  # noqa: F401
    from deep_ep.buffers._elastic_layout import dispatch_layout

    info = json.loads(Path(deep_ep.__file__).with_name("_build_info.json").read_text())
    if "elastic-dispatch-expanded-v1" not in info.get("capabilities", []):
        raise RuntimeError("Installed wheel does not contain payload dispatch")
    if "elastic-notify-metadata-v7" not in info.get("capabilities", []):
        raise RuntimeError("Rebuild/install the wheel with Notify metadata v7")
    torch.npu.set_device(local_rank)
    modes = _ROUTES if args.route == "all" else (args.route,)
    dtypes = (False, True) if args.dtype == "both" else (args.dtype == "fp8",)
    if local_rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    # Small correctness runs reserve the exact all-to-one bound. The public
    # hint budgets four times average rows and is intentionally smaller at EP>4.
    log("capacity.begin")
    # Random routing is reproducible, so reserve its actual maximum rather than
    # the all-to-one bound (which overflows address/workspace limits at EP256).
    random_routes = (
        make_routes(
            world,
            args.tokens,
            args.topk,
            args.experts,
            0,
            "random",
            seed=args.seed,
            without_replacement=getattr(args, "without_replacement", False),
        )
        if args.route == "random"
        else None
    )
    rows = (
        receive_capacity(random_routes, args.experts) if random_routes is not None else world * args.tokens * args.topk
    )
    capacity = 2 * max(
        dispatch_layout(world, args.tokens, args.topk, args.experts, rows, fp8, weights=True, hidden=args.hidden)[
            "num_bytes"
        ]
        for fp8 in dtypes
    )
    log(f"buffer.init.begin bytes={capacity}")
    buffer = deep_ep.ElasticBuffer(
        rank=rank,
        world_size=world,
        num_bytes=capacity,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.topk,
        explicitly_destroy=True,
    )
    log("buffer.init.done")
    buffer._notify_rank_dedup = args.rank_dedup
    checks, previous_output = 0, None
    try:
        for mode in modes:
            log(f"route={mode} routes_and_notify_golden.begin (CPU)")
            routes = (
                random_routes
                if random_routes is not None
                else make_routes(
                    world,
                    args.tokens,
                    args.topk,
                    args.experts,
                    0,
                    mode,
                    seed=args.seed,
                    without_replacement=getattr(args, "without_replacement", False),
                )
            )
            notify_golden = build_notify_golden(routes, args.experts, 256, rank_dedup=args.rank_dedup)
            entries = expert_rows(routes, args.experts, rank)
            log(f"route={mode} routes_and_notify_golden.done")
            for fp8 in dtypes:
                context = f"dtype={'fp8' if fp8 else 'bf16'} route={mode}"
                for with_weights in (False, True):
                    handle = None
                    for generation in range(args.repeat):
                        use_weights = with_weights if generation % 2 == 0 else not with_weights
                        iteration = (
                            f"{context} weights_case={with_weights} weights={use_weights} "
                            f"iter={generation + 1}/{args.repeat}"
                        )
                        log(f"{iteration} inputs.begin")
                        local, scales = payload(rank, args.tokens, generation, fp8, args.hidden)
                        local_weights = weights_for(rank, args.tokens, args.topk, generation)
                        producer = torch.npu.Stream()
                        with torch.npu.stream(producer):
                            x = (local.npu(), scales.npu()) if fp8 else local.npu()
                            w = local_weights.npu() if use_weights else None
                            indices = (
                                torch.tensor(routes[rank], dtype=torch.int64, device="npu") if handle is None else None
                            )
                            before = buffer.capture()
                        old_handle = handle
                        log(f"{iteration} dispatch.enter mode={'fresh' if handle is None else 'cached'}")
                        output, index, weights, handle, event = buffer.dispatch(
                            x,
                            topk_idx=indices,
                            topk_weights=w,
                            num_experts=args.experts if handle is None else None,
                            handle=handle,
                            previous_event=before,
                            async_with_compute_stream=True,
                            do_expand=True,
                        )
                        log(f"{iteration} dispatch.return (async; device completion not yet confirmed)")
                        event.current_stream_wait()
                        if index is not None or handle.metadata_only:
                            raise AssertionError("Expected expanded payload output")
                        if old_handle is None:
                            log(f"{iteration} handle_check.begin (includes device readback)")
                            check_handle(handle, notify_golden, rank, routes, args.experts)
                            log(f"{iteration} handle_check.done")
                        if old_handle is not None and old_handle is not handle:
                            raise AssertionError("Cached dispatch replaced its routing handle")
                        log(f"{iteration} payload_golden.begin (CPU)")
                        expected, expected_scales, expected_weights = expected_payload(
                            routes, args.experts, rank, generation, fp8, entries=entries, width=args.hidden
                        )
                        log(f"{iteration} payload_golden.done")
                        data = output[0] if fp8 else output
                        assert_bytes(data, expected, f"{iteration} dispatch hidden", progress=log)
                        log(f"{iteration} remaining_checks.begin")
                        if fp8:
                            assert_bytes(output[1], expected_scales, "scales")
                        if use_weights:
                            assert_bytes(weights, expected_weights, "weights")
                        elif weights is not None:
                            raise AssertionError("Weights returned when none were supplied")
                        if args.profile_op and generation == 0 and (args.profile_op != "notify" or not with_weights):
                            if args.profile_op == "notify":
                                invoke, validate = notify_profile_call(buffer, handle, args.experts)
                            else:

                                def invoke():
                                    result = buffer.dispatch(
                                        x,
                                        topk_weights=w,
                                        handle=handle,
                                        do_expand=True,
                                        async_with_compute_stream=True,
                                    )
                                    result[-1].current_stream_wait()
                                    return result

                                def validate(result):
                                    received, _, returned_weights, returned_handle, _ = result
                                    if returned_handle is not handle:
                                        raise AssertionError("Profile dispatch replaced the routing handle")
                                    assert_bytes(received[0] if fp8 else received, expected, "profile dispatch hidden")
                                    if fp8:
                                        assert_bytes(received[1], expected_scales, "profile dispatch scales")
                                    if use_weights:
                                        assert_bytes(returned_weights, expected_weights, "profile dispatch weights")
                                    elif returned_weights is not None:
                                        raise AssertionError("Profile dispatch unexpectedly returned weights")

                            alignment = (
                                dispatch_benchmark(buffer, enabled=args.dispatch_barrier)
                                if args.profile_op == "dispatch"
                                else nullcontext()
                            )
                            with alignment:
                                profile_operator(
                                    args,
                                    rank,
                                    f"{mode}-{'fp8' if fp8 else 'bf16'}-weights{int(use_weights)}",
                                    invoke,
                                    validate,
                                    info,
                                    log,
                                    stream=buffer._runtime.stream,
                                )
                        if previous_output is not None:
                            for prior, expected_prior in previous_output:
                                assert_bytes(prior, expected_prior, "retained output across cached/fresh calls")
                        previous_output = [(data, expected)]
                        if fp8:
                            previous_output.append((output[1], expected_scales))
                        if use_weights:
                            previous_output.append((weights, expected_weights))
                        if indices is not None:
                            indices.fill_(-1)
                            assert_bytes(handle.topk_idx, torch.tensor(routes[rank], dtype=torch.int64), "routing copy")
                        if expected.numel():
                            corrupt = expected.view(torch.uint8).clone()
                            corrupt.flatten()[0] ^= 1
                            try:
                                assert_bytes(data, corrupt.view(expected.dtype), "deliberate corruption")
                            except AssertionError:
                                pass
                            else:
                                raise AssertionError("Precision comparator failed to detect corruption")
                        checks += 1
                        log(f"{iteration} iteration.passed")
                    print(
                        f"rank={rank} dtype={'fp8' if fp8 else 'bf16'} route={mode} weights={with_weights} passed",
                        flush=True,
                    )
    except BaseException:
        _log_failure(log)
        raise
    finally:
        log("buffer.destroy.begin (may wait for device work/peers)")
        buffer.destroy()
        log("buffer.destroy.done")
    # Returned tensors must also remain valid after workspace destruction.
    if previous_output is not None:
        for prior, expected_prior in previous_output:
            assert_bytes(prior, expected_prior, "retained output after destroy")
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                world_size=world,
                checks=checks,
                passed=True,
                source_sha256=info["source_sha256"],
                tokens=args.tokens,
                topk=args.topk,
                experts=args.experts,
                hidden=args.hidden,
                dtype=args.dtype,
                route=args.route,
                seed=args.seed,
                without_replacement=getattr(args, "without_replacement", False),
            ),
            indent=2,
        )
        + "\n"
    )


def test_direct_and_external_launcher(monkeypatch):
    from types import SimpleNamespace
    import pytest

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "DEEPEP_SHMEM_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    parser = argparse.ArgumentParser()
    _add_launch_args(parser)
    args = parser.parse_args(
        ["--ep-size", "64", "--nproc-per-node", "8", "--node-rank", "5", "--master-addr", "141.61.54.91"]
    )
    calls = []

    def spawn(worker, args, nprocs, join):
        assert nprocs == 8 and join
        for local_rank in range(nprocs):
            worker(local_rank, *args)

    mp = SimpleNamespace(spawn=spawn)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(multiprocessing=mp))
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", mp)

    def worker(local_rank, args):
        calls.append(_worker_identity(local_rank, args))

    _launch(args, worker)
    assert calls == [(rank, 64) for rank in range(40, 48)]
    assert os.environ["DEEPEP_SHMEM_ENDPOINT"] == "tcp://141.61.54.91:8900"
    calls.clear()
    monkeypatch.setenv("RANK", "23")
    monkeypatch.setenv("WORLD_SIZE", "32")
    monkeypatch.setenv("LOCAL_RANK", "7")
    _launch(args, worker)
    assert calls == [(23, 32)]
    assert os.environ["DEEPEP_SHMEM_ENDPOINT"] == "tcp://141.61.54.91:8900"
    monkeypatch.delenv("WORLD_SIZE")
    with pytest.raises(ValueError, match="supplied together"):
        _launch(args, worker)


def test_hand_computed_routes():
    golden = build_notify_golden([[[0, 2]], [[1, 2]]], 4, 256)
    assert golden["rank_rows"] == [2, 2]
    assert golden["plans"][0]["dst"] == [0, ~2]
    assert golden["plans"][1]["dst"] == [~1, 3]
    duplicate = build_notify_golden([[[0, 0]], [[-1, -1]]], 4, 256)
    assert duplicate["rank_rows"] == [2, 0]
    assert duplicate["plans"][0]["dst"] == [0, ~1]


def test_group_chunk_order_and_token_ends():
    # Source 1 prefers itself; other sources use gateway 0. Each token has
    # two contributions on each of ranks 0 and 1. Two tokens per raw chunk.
    routes = [[[0, 2, 0, 2] for _ in range(6)] for _ in range(32)]
    golden = build_notify_golden(routes, 64, 2)
    for source in range(32):
        gateway = 1 if source == 1 else 0
        forward = golden["plans"][gateway]["forward_list"][source]
        assert [row[0] for row in forward] == [end for end in (3, 6, 9, 12, 15, 18) for _ in range(3)]
    # One 32-rank group. First raw chunk [0,2) splits into [0,1), [1,2).
    expected = []
    for token_ids in ((0,), (1,), (2, 3), (4, 5)):
        for source in range(32):
            if source == 1:
                continue
            expected.extend((source, token, slot) for token in token_ids for slot in (1, 3))
    backward = golden["plans"][1]["backward_list"][0]
    entries = expert_rows(routes, 64, 1)
    assert [entries[row] for _, row, _ in backward] == expected
    tails = {61: 1, 123: 2, 247: 3, 371: 4}
    assert [packed >> 32 for _, _, packed in backward] == [tails.get(i, 0) for i in range(len(backward))]


def test_installed_import_ignores_checkout_and_pythonpath(tmp_path):
    import subprocess

    installed = tmp_path / "site-packages"
    checkout = tmp_path / "checkout"
    for root, marker in ((installed, "installed"), (checkout, "source")):
        (root / "deep_ep").mkdir(parents=True)
        (root / "deep_ep/__init__.py").write_text(f"marker = {marker!r}\n", encoding="utf-8")
    # A separate process keeps pytest's already-imported source package intact.
    script = """
import runpy, sys, sysconfig
namespace = runpy.run_path(sys.argv[1])
sysconfig.get_path = lambda name: sys.argv[2]
mode = sys.argv[3]
if mode == "cached":
    import deep_ep
    assert deep_ep.marker == "source"
try:
    package = namespace["_import_installed_deep_ep"]()
except RuntimeError as error:
    assert mode != "installed", str(error)
    assert ("already loaded" if mode == "cached" else "not installed") in str(error)
else:
    assert mode == "installed"
    assert package.marker == "installed"
"""
    for mode in ("installed", "cached", "missing"):
        location = installed if mode != "missing" else tmp_path / "empty-site-packages"
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve()), str(location), mode],
            cwd=checkout,
            env={**os.environ, "PYTHONPATH": str(checkout)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_dispatch_barrier_timing_and_profile_versions():
    from collections import Counter
    import runpy

    import pytest

    metrics = runpy.run_path(str(Path(__file__).with_name("dispatch_metrics.py")))
    records = []
    for core in range(64):
        words = [0] * 32
        words[:8] = range(100, 108)
        words[26:31] = [0x44504550, 0, core, 1, 8]
        if core == 0:
            words[8:14] = [104, 105, 105, 1, 1, 1]
        records.append(words)
    current = metrics["decode_profile"](records, 0, fp8=False, world=2)
    assert all("store_drain_ms" in r["phases_ms"] and "finish_wait_ms" not in r["phases_ms"] for r in current)
    for words in records:
        words[30] = 7
    legacy = metrics["decode_profile"](records, 0, fp8=False, world=2)
    assert current[0]["phases_ms"]["store_drain_ms"] == legacy[0]["phases_ms"]["finish_wait_ms"]

    for words in records:
        words[30] = 16
    for world, expected in (
        (2, {"fm": 1, "local_forward": 8, "idle": 23}),
        (8, {"fm": 7, "local_forward": 25}),
        (16, {"fm": 7, "relay_helper": 1, "clos_qp0": 8, "clos_qp1": 8, "local_forward": 8}),
    ):
        for core, words in enumerate(records):
            words[8:14] = [104] * 6
            words[16:24] = [104] * 8
            if core < min(world, 8) - 1:
                words[8:14] = [104, 105, 105, 1, 1, core + 1]
        decoded = metrics["decode_profile"](records, 0, fp8=False, world=world)
        assert Counter(row["role"] for row in decoded[:32]) == expected
        assert "forward_scan_ms" in decoded[0]["phases_ms"]

    ranks = [
        dict(
            rank=rank,
            world=2,
            passed=True,
            tokens=4096,
            topk=6,
            experts=384,
            dtype="bf16",
            weights=False,
            backend="udma",
            source_sha256="test",
            dispatch_entry_barrier=True,
            clean_device_ms=[[0.1, 1.0, 0.2, 1.3]],
            kernel_profiles=[],
            event_only=True,
        )
        for rank in range(2)
    ]
    old_report = metrics["summarize"](ranks)
    assert old_report["input_copy"]["mean_ms"] == 0.1
    columns = ["entry_barrier", "kernel", "exit_barrier", "device_total"]
    for rank in ranks:
        rank["device_time_columns"] = columns
    report = metrics["summarize"](ranks)
    assert "input_copy" not in report and "output_copy" not in report
    assert report["entry_barrier"]["mean_ms"] == 0.1
    assert report["exit_barrier"]["mean_ms"] == 0.2
    assert report["algorithm_bandwidth"] == old_report["algorithm_bandwidth"]
    assert report["dispatch_entry_barrier"] is True
    ranks[1]["dispatch_entry_barrier"] = False
    with pytest.raises(ValueError, match="Inconsistent Dispatch entry-barrier settings"):
        metrics["summarize"](ranks)
    ranks[1]["dispatch_entry_barrier"] = True
    ranks[1].pop("device_time_columns")
    with pytest.raises(ValueError, match="Mixed device timing columns"):
        metrics["summarize"](ranks)


if __name__ == "__main__":
    main()


def test_rank_notify_dispatch_device_plans(tmp_path):
    import shutil
    import subprocess

    import pytest
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Requires a host C++17 compiler")
    root = Path(__file__).resolve().parents[2]
    notify = (root / "csrc/kernels/notify/notify.cpp").read_text(encoding="utf-8")
    forward = (root / "csrc/kernels/dispatch/fm_forward.hpp").read_text(encoding="utf-8")
    peer = (root / "csrc/kernels/dispatch/peer_plan.hpp").read_text(encoding="utf-8")
    parse = cpp_functions(notify, "ParseLocalTopkTokensVf", "CountLocalTopkTilesVf", "ParseLocalTopkVf")
    collect = cpp_functions(notify, "NotifyBuildForwardRowsVf")
    carry = cpp_functions(forward, "AppendForwardPairsVf", "CarryForwardPairsVf")
    finalize = cpp_functions(notify, "NotifyFinalizeOutputsVf")
    code = (
        r"""

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <climits>
#include <vector>
#define __simt_vf__
#define __launch_bounds__(...)
#define __gm__
#define __ubuf__
struct {
    uint32_t x;
} threadIdx;
template <class T>
T asc_ldcg(const T* p)
{
    return *p;
}
template <class T, class V>
void asc_stcg(T* p, V v)
{
    *p = T(v);
}
void asc_threadfence() {}
// Simulate each shuffle operand at the preceding prefix-scan step from
// an immutable copy; previous lanes may already have written their output.
int32_t host_prefix_block[512];
uint32_t host_prefix_count;
int32_t asc_shfl_up(int32_t, uint32_t delta, int32_t width)
{
    assert(width == 32);
    const auto lane = threadIdx.x % 32, bucket = threadIdx.x / 32;
    if (lane < delta) return 0;
    const int first = std::max(0, int(lane) - int(delta) * 2 + 1), last = int(lane - delta);
    int32_t sum = 0;
    for (int j = first; j <= last; ++j)
        for (uint32_t i = 0; i < 4; ++i)
            if (uint32_t(j) * 4 + i < host_prefix_count) sum += host_prefix_block[(j * 4 + i) * 4 + bucket];
    return sum;
}
namespace ascend_deepep {
using topk_idx_t = int64_t;
}
constexpr uint32_t kSimtCores = 32, kSimtThreads = 128, kTileTokens = 16, kMetaFields = 6;
constexpr uint32_t kNotifySimtThreads = 128, kForwardTile = 32;
constexpr uint32_t kWireRankMask = 511, kWireTopkMask = 15;
constexpr uint32_t kWirePrimaryTopkShift = 9, kWireSecondaryTopkShift = 13;
"""
        + finalize
        + parse
        + collect
        + carry
        + cpp_functions(forward, "DecodeForwardRowsVf")
        + cpp_functions(
            peer,
            "NotifyDispatchRowsVf",
            "NotifyDispatchEmitCachedVf",
            "NotifyDispatchPrefixBlockVf",
            "NotifyDispatchDirectoryBlockVf",
            "NotifyDispatchInitVf",
        )
        + r"""

template <uint32_t Cores>
void build_plan(std::vector<int64_t>& dst, uint32_t stride, uint32_t rank, uint32_t w, uint32_t t, uint32_t k,
                uint32_t group, std::vector<int32_t>& directory, std::vector<int32_t>& rows)
{
    const uint32_t tiles = (t + 15) / 16, hs = (w * 2 + 15) / 16 * 16;
    std::vector<int32_t> routes(t * k * 6), hist(tiles * hs), segments((tiles + 15) / 16 * hs), gateways(w), bases(w),
        scratch(w * tiles * 4);
    for (uint32_t token = 0; token < t; ++token)
        for (uint32_t slot = 0; slot < k; ++slot) {
            const uint32_t r = token * k + slot;
            const int64_t d = dst[r];
            if (d == INT64_MIN) {
                routes[r * 6] = -1;
                continue;
            }
            const uint64_t encoded = d < 0 ? uint64_t(~d) : uint64_t(d);
            routes[r * 6] = encoded / stride;
            routes[r * 6 + 1] = encoded / stride;
            routes[r * 6 + 4] = encoded % stride;
            routes[r * 6 + 5] = d < 0 ? 9 : 7;
            if (d >= 0) ++hist[token / 16 * hs + w + encoded / stride];
        }
    for (uint32_t peer = 0; peer < w; ++peer) {
        int32_t total = 0, inside = 0;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            if (tile % 16 == 0) {
                segments[tile / 16 * hs + w + peer] = total;
                inside = 0;
            }
            auto& x = hist[tile * hs + w + peer];
            const auto n = x;
            x = inside;
            inside += n;
            total += n;
        }
        gateways[peer] = total;
    }
    for (uint32_t core = 0; core < Cores; ++core)
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            NotifyDispatchRowsVf<false, Cores>(routes.data(), hist.data(), segments.data(), gateways.data(),
                                               bases.data(), directory.data(), rows.data(), scratch.data(), rank, w, t,
                                               k, w, group, core);
    // Cached production path uses exactly the 32-token, core-strided UB
    // layout already loaded by Notify ForwardCount, including partial tails.
    std::vector<std::vector<int32_t>> cached(Cores);
    for (uint32_t core = 0; core < Cores; ++core) {
        for (uint32_t tile = core; tile < (t + 31) / 32; tile += Cores) {
            const uint32_t begin = tile * 32, end = std::min(t, begin + 32);
            const auto pos = tile / Cores * 32 * k * 6;
            cached[core].resize(pos + 32 * k * 6, -123);
            std::copy(routes.begin() + begin * k * 6, routes.begin() + end * k * 6, cached[core].begin() + pos);
        }
    }
    const auto uncached_counts = scratch;
    std::fill(scratch.begin(), scratch.end(), -99);
    for (uint32_t core = 0; core < Cores; ++core)
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            NotifyDispatchRowsVf<false, Cores, true>(routes.data(), hist.data(), segments.data(), gateways.data(),
                                                     bases.data(), directory.data(), rows.data(), scratch.data(), rank,
                                                     w, t, k, w, group, core, cached[core].data());
    assert(scratch == uncached_counts);
    std::vector<std::vector<int32_t>> saved(Cores);
    for (uint32_t core = 0; core < Cores; ++core) {
        saved[core].resize(cached[core].size() / 3);
        for (uint32_t local_route = 0; local_route < cached[core].size() / 6; ++local_route) {
            const uint32_t local_token = local_route / k,
                           token = (core + local_token / 32 * Cores) * 32 + local_token % 32;
            if (token >= t) continue;
            const auto d = dst[token * k + local_route % k];
            if (d < 0) continue;
            bool relay = false;
            const auto peer = uint64_t(d) / stride;
            for (uint32_t j = 0; j < k; ++j) {
                const auto other = dst[token * k + j];
                if (other < 0 && other != INT64_MIN && uint64_t(~other) / stride / group == peer / group) relay = true;
            }
            saved[core][local_route * 2] = relay ? -1 : 0;
            saved[core][local_route * 2 + 1] = relay ? 0 : 1;
        }
    }
    std::fill(scratch.begin(), scratch.end(), -99);
    for (uint32_t core = 0; core < Cores; ++core)
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            NotifyDispatchRowsVf<false, Cores, true, true>(
                routes.data(), hist.data(), segments.data(), gateways.data(), bases.data(), directory.data(),
                rows.data(), scratch.data(), rank, w, t, k, w, group, core, cached[core].data(), saved[core].data());
    assert(scratch == uncached_counts);
    for (uint32_t peer = 0; peer < w; ++peer) {
        int32_t block[512]{}, carry[4]{};
        for (uint32_t at = 0; at < tiles; at += 128) {
            const auto n = std::min(128U, tiles - at);
            std::copy_n(scratch.data() + (peer * tiles + at) * 4, n * 4, block);
            std::copy_n(block, 512, host_prefix_block);
            host_prefix_count = n;
            for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
                NotifyDispatchPrefixBlockVf(block, carry, n, at == 0, at + n == tiles, directory.data(), peer, w);
            std::copy_n(block, n * 4, scratch.data() + (peer * tiles + at) * 4);
        }
    }
    for (uint32_t core = 0; core < Cores; ++core) {
        uint32_t base = 0;
        for (uint32_t begin = 0; begin < w * 4; begin += 128) {
            const auto n = std::min(128U, w * 4 - begin);
            int32_t block[256], sum = 0;
            std::copy_n(directory.data() + begin * 2, n * 2, block);
            for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
                NotifyDispatchDirectoryBlockVf<Cores>(block, &sum, directory.data(), begin, n, base, core);
            base = sum;
        }
        assert(base <= t * k);
    }
    for (uint32_t core = 0; core < Cores; ++core)
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            NotifyDispatchRowsVf<true, Cores>(routes.data(), hist.data(), segments.data(), gateways.data(),
                                              bases.data(), directory.data(), rows.data(), scratch.data(), rank, w, t,
                                              k, w, group, core);
    const auto uncached_rows = rows;
    std::fill(rows.begin(), rows.end(), -1);
    for (uint32_t core = 0; core < Cores; ++core)
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            NotifyDispatchRowsVf<true, Cores, true>(routes.data(), hist.data(), segments.data(), gateways.data(),
                                                    bases.data(), directory.data(), rows.data(), scratch.data(), rank,
                                                    w, t, k, w, group, core, cached[core].data());
    assert(rows == uncached_rows);
    std::fill(rows.begin(), rows.end(), -1);
    for (uint32_t core = 0; core < Cores; ++core)
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            NotifyDispatchEmitCachedVf<Cores>(cached[core].data(), saved[core].data(), bases.data(), directory.data(),
                                              rows.data(), scratch.data(), rank, w, t, k, w, core);
    assert(rows == uncached_rows);
    // Independent reference: assign QP on the whole positive peer stream,
    // then classify phases, preserving token/slot ordering inside buckets.
    std::vector<std::vector<int32_t>> expected(w * 4);
    std::vector<uint32_t> seen(w);
    for (uint32_t token = 0; token < t; ++token)
        for (uint32_t slot = 0; slot < k; ++slot) {
            const auto d = dst[token * k + slot];
            if (d < 0) continue;
            const uint32_t peer = uint64_t(d) / stride;
            if (peer == rank) continue;
            const auto ordinal = seen[peer]++, total = uint32_t(gateways[peer]), full = total / 64 * 64;
            const uint32_t qp =
                peer / 8 != rank / 8 &&
                (ordinal < full ? ordinal % 64 >= 48 : ordinal - full >= total - full - (total - full) / 4);
            bool relay = false;
            for (uint32_t j = 0; j < k; ++j) {
                const auto other = dst[token * k + j];
                if (other < 0 && other != INT64_MIN && uint64_t(~other) / stride / group == peer / group) relay = true;
            }
            auto& bucket = expected[((relay ? 0 : w) + peer) * 2 + qp];
            bucket.push_back(token);
            bucket.push_back(uint64_t(d) % stride);
        }
    uint32_t cursor = 0;
    for (uint32_t e = 0; e < w * 4; ++e) {
        assert(directory[e * 2] == int32_t(cursor) && directory[e * 2 + 1] == int32_t(expected[e].size() / 2));
        assert(std::equal(expected[e].begin(), expected[e].end(), rows.begin() + cursor * 2));
        cursor += expected[e].size() / 2;
    }
}
void build_plan_runtime(std::vector<int64_t>& dst, uint32_t stride, uint32_t rank, uint32_t w, uint32_t t, uint32_t k,
                        uint32_t group, std::vector<int32_t>& directory, std::vector<int32_t>& rows, uint32_t cores)
{
    if (cores == 32)
        build_plan<32>(dst, stride, rank, w, t, k, group, directory, rows);
    else
        build_plan<64>(dst, stride, rank, w, t, k, group, directory, rows);
}
template <bool RankDedup>
void parse_check(uint32_t cores, uint32_t rank)
{
    const uint32_t w = 32, t = 257, k = 6, e = 512, hs = (e + w + 15) / 16 * 16;
    std::vector<int64_t> input(t * k), mask(t, -1), cached_mask(t, -1);
    for (uint32_t token = 0; token < t; ++token) {
        // Different experts on one rank, two ranks in a server, invalid routes,
        // repeated experts, cross-server hits, and a final partial token tile.
        const int64_t values[] = {16, 17, 32, 16, -1, 256};
        for (uint32_t j = 0; j < k; ++j) input[token * k + j] = values[j];
    }
    std::vector<int32_t> meta(t * k * 6, -99), cached(meta), hist(((t + 15) / 16) * hs, 0);
    std::vector<int32_t> routes(((t + 15) / 16 + cores - 1) / cores * 16 * k * 2, -99);
    for (uint32_t c = 0; c < cores; ++c)
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x) {
            if (cores == 32) {
                ParseLocalTopkVf<false, 32, RankDedup>(input.data(), mask.data(), meta.data(), hist.data(), rank, w, t,
                                                       k, e, 256, c, nullptr);
                ParseLocalTopkTokensVf<32, RankDedup>(input.data(), cached_mask.data(), cached.data(), rank, w, t, k, e,
                                                      256, c, routes.data());
            } else {
                ParseLocalTopkVf<false, 64, RankDedup>(input.data(), mask.data(), meta.data(), hist.data(), rank, w, t,
                                                       k, e, 256, c, nullptr);
                ParseLocalTopkTokensVf<64, RankDedup>(input.data(), cached_mask.data(), cached.data(), rank, w, t, k, e,
                                                      256, c, routes.data());
            }
        }
    for (uint32_t token = 0; token < t; ++token) {
        assert(mask[token] == (RankDedup ? 7 : 5));  // compact rank-slot bits vs server bits
        assert(mask[token] == cached_mask[token]);
        for (uint32_t j = 0; j < k; ++j)
            for (uint32_t f : {0U, 1U, 2U, 3U, 5U})
                assert(meta[(token * k + j) * 6 + f] == cached[(token * k + j) * 6 + f]);
        assert(meta[(token * k + 2) * 6 + 1] == (RankDedup || rank == 2 ? 2 : 1));
        if (!RankDedup && rank == 2) {
            // The source rank may occur after the server's first top-k hit.
            assert(meta[token * k * 6 + 1] == 2 && meta[token * k * 6 + 2] == 2);
        }
    }
}
void forwarding_check(bool direct_local, uint32_t cores)
{
    const uint32_t w = 8, k = 2, t = 1, stride = 32, rank = 2;
    std::vector<int64_t> dst = {10, direct_local ? int64_t(stride + 11) : ~int64_t(stride + 11)};
    std::vector<int32_t> meta(w * 8, 0), rows(t * k * 2, -1);
    build_plan_runtime(dst, stride, rank, w, t, k, 8, meta, rows, cores);
    assert(meta[1] == (direct_local ? 0 : 1));  // FM peer 0 critical range / early SO
    assert(meta[w * 4 + 1] == (direct_local ? 1 : 0));
}
void return_check()
{
    const uint32_t w = 8, t = 1, k = 6, cap = 6, words = 16 + cap * 6;
    std::vector<int32_t> packets(w * words, 0), offsets(w, 0), prefix(w, 0), forward(w * cap * 6, -1), counts(w, 0),
        ids(cap, -1);
    auto* packet = packets.data();
    packet[0] = 1;
    auto* r = packet + 16;
    r[0] = (2 << 9) | (3 << 13);
    r[1] = 0;
    r[2] = 0;
    r[3] = 2;
    r[4] = 1;
    r[5] = 1;
    for (uint32_t core = 0; core < 32; ++core)
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            NotifyBuildForwardRowsVf<32, true>(packets.data(), offsets.data(), prefix.data(), forward.data(),
                                               counts.data(), ids.data(), 2, w, cap, words, core, t, k);
    assert(forward[1] == -2 && ids[0] == 2 && ids[1] == 3);
    assert(packets[17] == 0);  // original token remains available for chunk boundaries
}
void large_destination_check(uint32_t cores)
{
    const uint32_t w = 512, k = 6, stride = 1U << 27;
    std::vector<int64_t> dst = {int64_t(511) * stride + 123, ~(int64_t(511) * stride + 124),
                                int64_t(511) * stride + 125, ~(int64_t(511) * stride + 126),
                                int64_t(256) * stride + 127, INT64_MIN};
    std::vector<int32_t> meta(w * 8, 0), rows(k * 2, -1);
    build_plan_runtime(dst, stride, 0, w, 1, k, 8, meta, rows, cores);
    assert(meta[511 * 4 + 1] == 2 && meta[(w + 256) * 4 + 1] == 1);
    const uint32_t critical = uint32_t(meta[511 * 4]), direct = uint32_t(meta[(w + 256) * 4]);
    assert(rows[critical * 2] == 0 && rows[critical * 2 + 1] == 123 && rows[critical * 2 + 3] == 125);
    assert(rows[direct * 2] == 0 && rows[direct * 2 + 1] == 127);
}
void decode_check()
{
    std::vector<int32_t> input(128 * 6, -1), targets(256);
    std::vector<uint32_t> pairs(256);
    for (uint32_t i = 0; i < 127; ++i) {
        input[i * 6 + 2] = i;
        input[i * 6 + 3] = i % 8;
        input[i * 6 + 4] = i + 300;
    }
    for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
        DecodeForwardRowsVf(input.data(), targets.data(), pairs.data(), 127);
    for (uint32_t i = 0; i < 128; ++i) {
        assert(targets[i * 2] == (i < 127 ? int32_t(i % 8) : -1));
        assert(targets[i * 2 + 1] == targets[i * 2]);
        assert(pairs[i * 2] == (i < 127 ? i : 0));
        assert(pairs[i * 2 + 1] == (i < 127 ? i + 300 : 0));
    }
}
void carry_check()
{
    // Empty/sparse tiles, full tiles, source boundaries and a non-full tail.
    std::vector<uint32_t> selected(256), accumulated(320, 0xdeadbeefU), expected, actual;
    uint32_t pending = 0, sequence = 0;
    for (uint32_t count : {0U, 1U, 31U, 32U, 33U, 127U, 128U, 0U, 7U, 128U, 1U}) {
        for (uint32_t i = 0; i < count * 2; ++i) {
            selected[i] = sequence++;
            expected.push_back(selected[i]);
        }
        for (threadIdx.x = 127; threadIdx.x < 128; --threadIdx.x)
            AppendForwardPairsVf(accumulated.data(), selected.data(), pending, count);
        const uint32_t ready = (pending + count) / 32 * 32;
        actual.insert(actual.end(), accumulated.begin(), accumulated.begin() + ready * 2);
        pending += count;
        pending -= ready;
        if (ready && pending)
            for (threadIdx.x = 127; threadIdx.x < 128; --threadIdx.x)
                CarryForwardPairsVf(accumulated.data(), ready, pending);
        assert(pending < 32);
        assert(std::equal(actual.begin(), actual.end(), expected.begin()));
        assert(std::equal(accumulated.begin(), accumulated.begin() + pending * 2, expected.begin() + actual.size()));
    }
    actual.insert(actual.end(), accumulated.begin(), accumulated.begin() + pending * 2);
    assert(actual == expected);
}
void finalize_check(uint32_t cores)
{
    std::vector<int32_t> totals(512), output(512, -1);
    for (uint32_t i = 0; i < 512; ++i) totals[i] = i * 3;
    std::vector<uint32_t> plan(4096, 0xdeadbeefU);
    for (uint32_t c = 0; c < cores; ++c)
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x) {
            if (cores == 32)
                NotifyFinalizeOutputsVf<32>(totals.data(), output.data(), 512, plan.data(), 96, 1024 * 4, 2048, c);
            else
                NotifyFinalizeOutputsVf<64>(totals.data(), output.data(), 512, plan.data(), 96, 1024 * 4, 2048, c);
        }
    assert(output == totals);
    for (uint32_t i = 0; i < plan.size(); ++i)
        assert(plan[i] == ((i < 96 || (i >= 1024 && i < 3072)) ? 0U : 0xdeadbeefU));
}
int main()
{
    for (auto cores : {32U, 64U})
        for (auto rank : {0U, 1U, 2U, 16U}) {
            parse_check<true>(cores, rank);
            parse_check<false>(cores, rank);
        }
    for (auto cores : {32U, 64U}) {
        finalize_check(cores);
        decode_check();
        carry_check();
        forwarding_check(false, cores);
        forwarding_check(true, cores);
        large_destination_check(cores);
    }
    return_check();
    for (auto cores : {32U, 64U})
        for (auto w : {8U, 16U, 32U, 128U, 256U})
            for (auto group : {1U, 8U})
                for (auto t : {1U, 15U, 16U, 17U, 63U, 64U, 65U, 257U, 4096U, 16384U}) {
                    const uint32_t k = 6, stride = 200000;
                    std::vector<int64_t> dst(t * k);
                    for (uint32_t token = 0; token < t; ++token) {
                        const uint32_t peer = token % 5 ? w - 1 : token % w;
                        dst[token * k] = int64_t(peer) * stride + token * 2;
                        dst[token * k + 1] = token % 3 ? ~(int64_t(peer) * stride + token * 2 + 1) : INT64_MIN;
                        dst[token * k + 2] = int64_t((peer + 1) % w) * stride + token;
                        dst[token * k + 3] = INT64_MIN;
                        dst[token * k + 4] = dst[token * k + 2];
                        dst[token * k + 5] = INT64_MIN;
                    }
                    std::vector<int32_t> directory(w * 8), rows(t * k * 2, -1);
                    build_plan_runtime(dst, stride, 0, w, t, k, group, directory, rows, cores);
                }
}
"""
    )
    source = tmp_path / "rank-server.cpp"
    source.write_text(code, encoding="utf-8")
    binary = tmp_path / "rank-server"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=60)


def test_scale_worker_shards_and_native_inbox(tmp_path):
    """All workers have peers; shards cover each route once with separate flags."""
    import shutil
    import subprocess
    import pytest
    from deep_ep.buffers._elastic_layout import dispatch_layout

    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    source = tmp_path / "scale_shards.cpp"
    source.write_text(
        r"""

#include <iostream>
#include "kernels/dispatch/tiling.hpp"
int main()
{
    for (unsigned world : {2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U, 512U})
        for (unsigned tokens : {1U, 257U, 4096U, 16384U}) {
            auto t = ascend_deepep::MakeDispatchTiling(0, world, tokens, 6, 1024, 128, true);
            std::cout << world << ' ' << tokens << ' ' << ascend_deepep::scale_pipeline::Shards(world) << ' '
                      << t.inbox_stride << '\n';
        }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("scale_shards.exe" if sys.platform == "win32" else "scale_shards")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I" + str(Path(__file__).resolve().parents[2] / "csrc"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=60,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
    for line in result.stdout.splitlines():
        world, tokens, shards, stride = map(int, line.split())
        rows = min(128, world * tokens * 6)
        layout = dispatch_layout(world, tokens, 6, 1024, rows, True)
        assert stride == layout["inbox_stride"]
        assert stride % (512 * shards) == 0
        routes = tokens * 6
        span = (routes + shards - 1) // shards
        region = stride // shards
        assert region >= 512 + ((span + 1) // 2) * 512
        coverage = {peer: [] for peer in range(world)}
        for worker in range(32):
            owner = worker % world if shards > 1 else worker
            shard = worker // world if shards > 1 else 0
            peers = list(range(owner, world, 32))
            assert peers, (world, worker)  # No structurally idle scale AIV.
            begin, end = min(shard * span, routes), min((shard + 1) * span, routes)
            if tokens >= 257:
                assert begin < end
            assert (shard + 1) * region <= stride
            for peer in peers:
                coverage[peer].append((begin, end, shard * region))
        for intervals in coverage.values():
            assert len(intervals) == shards
            intervals.sort()
            cursor = 0
            regions = set()
            for begin, end, offset in intervals:
                assert begin == cursor
                cursor = end
                assert offset not in regions
                regions.add(offset)
            assert cursor == routes
