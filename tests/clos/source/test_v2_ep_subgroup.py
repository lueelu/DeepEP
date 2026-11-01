"""Single-node V2 subgroup UID and BF16 communication regression (torchrun).

Run all subgroup layouts, including two destroy/recreate cycles per layout::

    torchrun --standalone --nnodes=1 --nproc-per-node=4 test/test_v2_ep_subgroup.py
    torchrun --standalone --nnodes=1 --nproc-per-node=8 test/test_v2_ep_subgroup.py

Use --layout ep2 or --layout interleaved to select one layout, and
--transport mte or --transport udma to select the SHMEM transport.
UID bootstrap is the default. --explicit-port BASE uses a separate localhost
port (BASE + group ID) for each subgroup and skips UID checks.
--allow-duplicate-experts permits repeated expert IDs within a token's top-k.
"""

import argparse
import hashlib
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from deep_ep.backends.clos import ElasticBuffer, topk_idx_t
from deep_ep.backends.clos._native import load_native_module


def _layouts(world_size):
    if world_size not in (4, 8):
        raise ValueError("this test requires 4 or 8 ranks on a single node")
    layouts = {
        "ep2": [list(range(start, start + 2)) for start in range(0, world_size, 2)],
        "ep4": [list(range(start, start + 4)) for start in range(0, world_size, 4)],
        "interleaved": [list(range(0, world_size, 2)), list(range(1, world_size, 2))],
    }
    if world_size == 8:
        layouts["ep8"] = [list(range(world_size))]
    return layouts


def _gather(value, group=None):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _check_uids(identities, memberships, seen_uids):
    # WORLD gathers UIDs in global-rank order; each EP group shares one UID.
    current_uids = set()
    for group_id, ranks in enumerate(memberships):
        group_uids = {identities[rank] for rank in ranks}
        if len(group_uids) != 1 or b"" in group_uids:
            raise AssertionError(f"group {group_id} has empty or inconsistent UIDs")
        uid = next(iter(group_uids))
        if uid in current_uids:
            raise AssertionError(f"group {group_id} shares a UID with another group")
        if uid in seen_uids:
            raise AssertionError(f"group {group_id} reused a finalized UID")
        current_uids.add(uid)
    return current_uids


def _check(name, actual, expected):
    actual = actual.cpu()
    if actual.dtype != expected.dtype or not torch.equal(actual, expected):
        raise AssertionError(
            f"{name}: actual {actual.shape}/{actual.dtype}, expected {expected.shape}/{expected.dtype}; values differ"
        )


def _precision(buffer, group, global_rank, cycle, args):
    rank, size = group.rank(), group.size()
    experts_per_rank, num_topk = 16, 8
    num_experts = size * experts_per_rank

    # Expert IDs use EP-local ranks. Global-rank payloads expose cross-group traffic.
    generator = torch.Generator().manual_seed(7301 + global_rank + cycle * 101)
    if args.allow_duplicate_experts:
        route = torch.randint(
            num_experts,
            (args.num_tokens, num_topk),
            generator=generator,
            dtype=topk_idx_t,
        )
    else:
        route = torch.stack(
            [torch.randperm(num_experts, generator=generator)[:num_topk] for _ in range(args.num_tokens)]
        ).to(topk_idx_t)
    route[::7, 0] = -1
    route[0] = -1
    weights = torch.rand((args.num_tokens, num_topk), generator=generator)
    x = (
        (torch.arange(args.num_tokens)[:, None] % 16).float()
        + (torch.arange(args.hidden)[None, :] % 8).float() / 8
        + global_rank * 16
        + cycle * 0.5
    ).to(torch.bfloat16)

    # Build the CPU reference only from this subgroup's source tensors.
    rows, indices, route_weights, source_indices = [], [], [], []
    for source_x, source_route, source_weights in _gather((x, route, weights), group):
        local = (source_route >= rank * experts_per_rank) & (source_route < (rank + 1) * experts_per_rank)
        # Non-expanded dispatch sends one row per destination rank, even for duplicates.
        take = local.any(dim=1)
        rows.append(source_x[take])
        indices.append(torch.where(local[take], source_route[take] - rank * experts_per_rank, -1))
        route_weights.append(torch.where(local[take], source_weights[take], 0))
        source_indices.append(torch.arange(args.num_tokens, dtype=torch.int32)[take])
    expected_x = torch.cat(rows)
    expected_idx = torch.cat(indices)
    expected_weights = torch.cat(route_weights)
    expected_src = torch.cat(source_indices)
    destinations = torch.stack(
        [
            ((route >= peer * experts_per_rank) & (route < (peer + 1) * experts_per_rank)).any(dim=1)
            for peer in range(size)
        ]
    ).sum(dim=0)
    combined_weights_ref = torch.where(route >= 0, weights, 0)

    # Fresh V2 dispatch/combine: check data, routing metadata, and returned weights.
    recv_x, recv_idx, recv_weights, handle, event = buffer.dispatch(
        x.npu(),
        topk_idx=route.npu(),
        topk_weights=weights.npu(),
        num_experts=num_experts,
        num_sms=args.num_aiv,
    )
    assert event.event is None
    assert handle.num_recv_tokens == expected_x.size(0)
    assert handle.num_max_tokens_per_rank == args.num_tokens
    _check("dispatch payload", recv_x, expected_x)
    _check("dispatch indices", recv_idx, expected_idx)
    _check("dispatch weights", recv_weights, expected_weights)
    _check("dispatch source indices", handle._dispatch_layout[3], expected_src)

    combined_x, combined_weights, event = buffer.combine(
        recv_x,
        handle,
        topk_weights=recv_weights,
        num_sms=args.num_aiv,
    )
    assert event.event is None
    _check("combine payload", combined_x, (x.float() * destinations[:, None]).to(x.dtype))
    _check("combine weights", combined_weights, combined_weights_ref)

    # Reuse the V2 handle with changed payloads and weights in the same subgroup.
    x2 = (x.float() + 0.5).to(x.dtype)
    recv2, cached_idx, weights2, cached_handle, event = buffer.dispatch(
        x2.npu(),
        handle=handle,
        topk_weights=(weights * 2).npu(),
        num_sms=args.num_aiv,
    )
    assert event.event is None
    assert cached_handle is handle and cached_idx is None
    _check("cached dispatch payload", recv2, (expected_x.float() + 0.5).to(x.dtype))
    _check("cached dispatch weights", weights2, expected_weights * 2)
    combined2, combined_weights2, event = buffer.combine(
        recv2,
        handle,
        topk_weights=weights2,
        num_sms=args.num_aiv,
    )
    assert event.event is None
    _check("cached combine payload", combined2, (x2.float() * destinations[:, None]).to(x.dtype))
    _check("cached combine weights", combined_weights2, combined_weights_ref * 2)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("ep2", "ep4", "interleaved", "ep8", "all"), default="all")
    parser.add_argument("--transport", choices=("auto", "mte", "udma"), default="auto")
    parser.add_argument("--backend", choices=("gloo", "hccl"), default="gloo")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-aiv", type=int, default=8)
    parser.add_argument("--explicit-port", type=int, help="use IP/port bootstrap with BASE + group ID")
    parser.add_argument("--allow-duplicate-experts", action="store_true")
    args = parser.parse_args()
    if args.cycles < 2:
        parser.error("cycles must be >= 2 to check UID recreation")
    if args.num_tokens < 2 or args.hidden < 1:
        parser.error("num-tokens must be >= 2 and hidden must be positive")
    if args.num_aiv < 2 or args.num_aiv % 2:
        parser.error("num-aiv must be a positive even integer >= 2")
    if args.explicit_port is None and "SHMEM_UID_SESSION_ID" in os.environ:
        parser.error("unset SHMEM_UID_SESSION_ID to test independent subgroup UIDs")
    return args


def main():
    args = _parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    layouts = _layouts(world_size)
    if int(os.environ["LOCAL_WORLD_SIZE"]) != world_size:
        raise ValueError("run this test on a single node")
    if args.layout != "all" and args.layout not in layouts:
        raise ValueError(f"layout {args.layout} requires 8 ranks")
    selected_layouts = list(layouts) if args.layout == "all" else [args.layout]
    max_groups = max(len(layouts[layout]) for layout in selected_layouts)
    if args.explicit_port is not None and not 1 <= args.explicit_port <= 65536 - max_groups:
        raise ValueError("explicit-port + group ID must be between 1 and 65535")
    os.environ["ASCEND_DEEPEP_TRANSPORT"] = args.transport
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    dist.init_process_group(args.backend, timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    native = load_native_module()
    seen_uids = set()
    buffer = None
    try:
        for layout in selected_layouts:
            memberships = layouts[layout]
            # Every WORLD rank creates groups in the same order, including nonmembers.
            groups = [dist.new_group(ranks=ranks, backend=args.backend) for ranks in memberships]
            group_id = next(i for i, ranks in enumerate(memberships) if rank in ranks)
            group = groups[group_id]
            endpoint = None if args.explicit_port is None else f"tcp://127.0.0.1:{args.explicit_port + group_id}"
            for cycle in range(args.cycles):
                dist.barrier()
                buffer = ElasticBuffer(
                    group,
                    num_max_tokens_per_rank=args.num_tokens,
                    hidden=args.hidden,
                    num_topk=8,
                    explicitly_destroy=True,
                    shmem_ip_port=endpoint,
                )
                assert buffer.rank == memberships[group_id].index(rank)
                assert buffer.num_ranks == len(memberships[group_id])

                identities = None
                if endpoint is None:
                    # A probe key still returns the active native UID, as in the V1 test.
                    _, uid = native.get_shmem_bootstrap("probe")
                    identities = _gather(bytes(uid))
                    seen_uids.update(_check_uids(identities, memberships, seen_uids))
                _precision(buffer, group, rank, cycle, args)

                # All groups finish checking before collective teardown and the next cycle.
                dist.barrier()
                buffer.destroy()
                buffer = None
                state, released_uid = native.get_shmem_bootstrap("probe")
                if state != 0 or released_uid:
                    raise AssertionError("destroy left an active SHMEM session")
                dist.barrier()
                if rank == 0:
                    print(
                        json.dumps(
                            dict(
                                api="V2",
                                layout=layout,
                                cycle=cycle,
                                transport=args.transport,
                                bootstrap="uid" if endpoint is None else "ip_port",
                                explicit_port=args.explicit_port,
                                allow_duplicate_experts=args.allow_duplicate_experts,
                                groups=memberships,
                                uid_sha256=[hashlib.sha256(identities[ranks[0]]).hexdigest() for ranks in memberships]
                                if identities is not None
                                else None,
                                status="PASS",
                            )
                        ),
                        flush=True,
                    )
            dist.destroy_process_group(group)
            dist.barrier()
    finally:
        # A failing rank must not enter Buffer's collective destroy on its own.
        buffer = None
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
