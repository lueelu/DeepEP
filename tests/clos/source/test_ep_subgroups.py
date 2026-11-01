"""Eight-device subgroup bootstrap and communication regression (torchrun)."""

import argparse
import hashlib
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from deep_ep.backends.clos import Buffer, Config, topk_idx_t
from deep_ep.backends.clos._native import load_native_module


def gather(value, group=None):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def check(name, actual, expected):
    actual = actual.cpu()
    if not torch.equal(actual, expected):
        raise AssertionError(f"{name}: {actual.shape} != golden {expected.shape}; values differ")


def precision(buffer, group, global_rank, tokens, hidden, allow_duplicate_experts=False):
    rank, size = group.rank(), group.size()
    experts, k = size * 16, 8
    gen = torch.Generator().manual_seed(7301 + global_rank)
    if allow_duplicate_experts:
        route = torch.randint(experts, (tokens, k), generator=gen, dtype=topk_idx_t)
    else:
        route = torch.stack([torch.randperm(experts, generator=gen)[:k] for _ in range(tokens)]).to(topk_idx_t)
    route[::7, 0] = -1
    route[0] = -1
    weights = torch.rand((tokens, k), generator=gen)
    x = (
        (torch.arange(tokens)[:, None] % 16).float()
        + (torch.arange(hidden)[None, :] % 8).float() / 8
        + global_rank * 16
    ).to(torch.bfloat16)
    peers = gather((x, route, weights), group)
    rows, idx, wt, src = [], [], [], []
    for sx, si, sw in peers:
        valid = (si >= rank * 16) & (si < (rank + 1) * 16)
        take = valid.any(dim=1)
        rows.append(sx[take])
        idx.append(torch.where(valid[take], si[take] - rank * 16, -1))
        wt.append(torch.where(valid[take], sw[take], 0))
        src.append(torch.arange(tokens, dtype=torch.int32)[take])
    expected_x, expected_idx, expected_wt = map(torch.cat, (rows, idx, wt))
    x_npu, route_npu, weights_npu = [v.npu() for v in (x, route, weights)]
    counts, _, expert_counts, mask, event = buffer.get_dispatch_layout(route_npu, experts, async_finish=True)
    event.current_stream_wait()
    torch.npu.synchronize()
    print(f"rank={global_rank} layout_complete", flush=True)
    config = Config(32, 6, 256, 6, 128)
    recv, recv_idx, recv_wt, _, handle, event = buffer.dispatch(
        x_npu,
        topk_idx=route_npu,
        topk_weights=weights_npu,
        num_tokens_per_rank=counts,
        num_tokens_per_expert=expert_counts,
        is_token_in_rank=mask,
        config=config,
        async_finish=True,
    )
    event.current_stream_wait()
    print(f"rank={global_rank} dispatch_submitted", flush=True)
    check("dispatch payload", recv, expected_x)
    check("dispatch indices", recv_idx, expected_idx)
    check("dispatch weights", recv_wt, expected_wt)
    check("dispatch source indices", handle[3], torch.cat(src))
    destinations = torch.stack([((route >= r * 16) & (route < (r + 1) * 16)).any(dim=1) for r in range(size)]).sum(
        dim=0
    )
    print(f"rank={global_rank} dispatch_checked", flush=True)
    combined, _, event = buffer.combine(recv, handle, config=config, async_finish=True)
    event.current_stream_wait()
    check("combine", combined, (x.float() * destinations[:, None]).to(x.dtype))

    x2 = (x.float() + 0.5).to(x.dtype)
    recv2, _, _, _, _, event = buffer.dispatch(x2.npu(), handle=handle, config=config, async_finish=True)
    event.current_stream_wait()
    check("cached dispatch", recv2, (expected_x.float() + 0.5).to(x.dtype))
    combined2, _, event = buffer.combine(recv2, handle, config=config, async_finish=True)
    event.current_stream_wait()
    check("cached combine", combined2, (x2.float() * destinations[:, None]).to(x.dtype))
    torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("mte", "udma"), required=True)
    parser.add_argument("--layout", choices=("ep2", "ep4", "interleaved", "ep8", "all"), required=True)
    parser.add_argument("--backend", choices=("gloo", "hccl"), default="gloo")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--explicit-port", type=int)
    parser.add_argument("--allow-duplicate-experts", action="store_true")
    args = parser.parse_args()
    # V1 Buffer ignores inherited endpoint variables; --explicit-port is the
    # only way this test opts into IP/port bootstrap.
    assert "SHMEM_UID_SESSION_ID" not in os.environ
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    dist.init_process_group(args.backend, timeout=timedelta(minutes=5))
    assert dist.get_world_size() == 8
    layouts = {
        "ep2": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "ep4": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "interleaved": [[0, 2, 4, 6], [1, 3, 5, 7]],
        "ep8": [list(range(8))],
    }
    selected_layouts = list(layouts) if args.layout == "all" else [args.layout]
    for selected_layout in selected_layouts:
        args.layout = selected_layout
        groups = [dist.new_group(ranks=ranks, backend=args.backend) for ranks in layouts[args.layout]]
        group_id = next(i for i, ranks in enumerate(layouts[args.layout]) if rank in ranks)
        group = groups[group_id]
        previous_uid = None
        for cycle in range(args.cycles):
            dist.barrier()
            torch.npu.synchronize()
            endpoint = None if args.explicit_port is None else f"tcp://127.0.0.1:{args.explicit_port + group_id}"
            buffer = Buffer(
                group,
                num_bytes=64 * 1024 * 1024,
                transport=args.transport,
                explicitly_destroy=True,
                shmem_ip_port=endpoint,
            )
            # Native query returns the current UID even when the probe key differs.
            _, uid = load_native_module().get_shmem_bootstrap("probe")
            identities = gather((group_id, bytes(uid)))
            if endpoint is None:
                for gid, peer_uid in identities:
                    assert bool(peer_uid) and (peer_uid == bytes(uid)) == (gid == group_id)
                assert uid != previous_uid, "recreated Buffer reused a finalized UID"
            previous_uid = uid
            print(f"rank={rank} bootstrap_complete layout={args.layout} cycle={cycle}", flush=True)
            precision(buffer, group, rank, args.tokens, args.hidden, args.allow_duplicate_experts)
            buffer.destroy()
            assert load_native_module().get_shmem_bootstrap("probe")[0] == 0
            print(
                json.dumps(
                    dict(
                        rank=rank,
                        group=group_id,
                        cycle=cycle,
                        transport=args.transport,
                        layout=args.layout,
                        uid_sha256=hashlib.sha256(bytes(uid)).hexdigest(),
                        status="PASS",
                    )
                ),
                flush=True,
            )
    dist.barrier()
    torch.npu.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
