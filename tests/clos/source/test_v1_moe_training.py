"""V1 framework-style autograd regression; launch with torchrun.

Only layout/dispatch/combine are used from DeepEP. Expert permutation and
weighted unpermutation use PyTorch. The independent CPU reference receives
inputs from deterministic generators, never from the tested communication.
"""

import argparse
import json
import os
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from deep_ep.backends.clos import Buffer, Config, topk_idx_t


class FusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, route, weights, buffer, experts, config, asynchronous, comm, fixed):
        counts, _, expert_counts, mask, event = buffer.get_dispatch_layout(
            route, experts, async_finish=asynchronous, allocate_on_comm_stream=comm
        )
        recv, indices, probs, _, handle, done = buffer.dispatch(
            x,
            topk_idx=route,
            topk_weights=weights,
            num_tokens_per_rank=counts,
            num_tokens_per_expert=expert_counts,
            is_token_in_rank=mask,
            expert_alignment=1,
            config=config,
            previous_event=event,
            async_finish=asynchronous,
            allocate_on_comm_stream=comm,
            num_worst_tokens=buffer.group_size * buffer.num_max_tokens_per_rank if fixed else 0,
        )
        if asynchronous:
            done.current_stream_wait()
        ctx.buffer, ctx.handle, ctx.config = buffer, handle, config
        ctx.asynchronous, ctx.comm = asynchronous, comm
        ctx.mark_non_differentiable(indices)
        return recv, indices, probs, handle

    @staticmethod
    def backward(ctx, grad_x, _grad_indices, grad_probs, _grad_handle):
        dx, dw, event = ctx.buffer.combine(
            grad_x.contiguous(),
            ctx.handle,
            topk_weights=grad_probs.float().contiguous(),
            config=ctx.config,
            async_finish=ctx.asynchronous,
            allocate_on_comm_stream=ctx.comm,
        )
        if ctx.asynchronous:
            event.current_stream_wait()
        return dx, None, dw, None, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, buffer, handle, config, asynchronous, comm):
        out, _, event = buffer.combine(
            x,
            handle,
            config=config,
            async_finish=asynchronous,
            allocate_on_comm_stream=comm,
        )
        if asynchronous:
            event.current_stream_wait()
        ctx.buffer, ctx.handle, ctx.config = buffer, handle, config
        ctx.asynchronous, ctx.comm = asynchronous, comm
        return out

    @staticmethod
    def backward(ctx, grad):
        dx, _, _, _, _, event = ctx.buffer.dispatch(
            grad.contiguous(),
            handle=ctx.handle,
            config=ctx.config,
            async_finish=ctx.asynchronous,
            allocate_on_comm_stream=ctx.comm,
        )
        if ctx.asynchronous:
            event.current_stream_wait()
        return dx, None, None, None, None, None


def token_counts(size, step, args):
    if not args.uneven:
        return [args.num_tokens] * size
    shift = step if args.vary_tokens else 0
    return [
        0 if (rank + shift) % 8 == 0 else 1 if (rank + shift) % 8 == 1 else max(1, args.num_tokens - (rank + shift) % 5)
        for rank in range(size)
    ]


def inputs(rank, step, microbatch, args):
    gen = torch.Generator().manual_seed(args.seed + rank * 97 + step * 10007 + microbatch * 1009)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokens = token_counts(rank + 1, step, args)[rank]
    x = (torch.randn(tokens, args.hidden, generator=gen) * 0.1).to(dtype)
    route = (
        torch.stack([torch.randperm(args.num_experts, generator=gen)[: args.num_topk] for _ in range(tokens)]).to(
            topk_idx_t
        )
        if tokens
        else torch.empty((0, args.num_topk), dtype=topk_idx_t)
    )
    # Keep same-rank duplicates by expert, plus absent routes and empty tokens.
    route[::7, 0] = -1
    if tokens:
        route[0] = -1
    if args.empty_route_cycle and step % 4 == 2:
        route.fill_(-1)
    weights = torch.rand(tokens, args.num_topk, generator=gen)
    weights /= weights.sum(dim=-1, keepdim=True)
    dy = (torch.randn(tokens, args.hidden, generator=gen) * 0.1).to(dtype)
    return x, route, weights, dy


def parameters(rank, size, args, device):
    nlocal = args.num_experts // size
    out = []
    for rows, cols, salt in (
        (args.hidden, args.ffn_hidden, 0),
        (args.hidden, args.ffn_hidden, 1),
        (args.ffn_hidden, args.hidden, 2),
    ):
        values = []
        for expert in range(rank * nlocal, (rank + 1) * nlocal):
            gen = torch.Generator().manual_seed(args.seed + expert * 13 + salt)
            values.append(torch.randn(rows, cols, generator=gen) * 0.04)
        out.append(torch.stack(values).to(device).requires_grad_(True))
    return out


def experts(recv, local_indices, probs, params):
    # Accumulate expert contributions in FP32, then cast once per destination
    # rank, matching the payload passed into V1 combine.
    x = recv.float()
    result = x * 0.0 + probs.sum() * 0.0
    for expert in range(params[0].size(0)):
        positions = torch.nonzero(local_indices == expert, as_tuple=False)
        if positions.size(0) == 0:
            # Give unused parameters explicit zero gradients on both paths.
            result = result + sum(p[expert].sum() * 0.0 for p in params)
            continue
        rows, slots = positions[:, 0], positions[:, 1]
        selected = x.index_select(0, rows)
        y = (F.silu(selected @ params[0][expert]) * (selected @ params[1][expert])) @ params[2][expert]
        y = y.to(recv.dtype).float() * probs[rows, slots].unsqueeze(1)
        result = result.index_add(0, rows, y)
    return result.to(recv.dtype)


def fixed_experts(recv, local_indices, probs, params):
    # Graph integration reference implementation: fixed rows per expert avoid
    # nonzero()/CPU counts. This is deliberately not an optimized grouped GEMM.
    # The CPU oracle keeps the independent sparse permutation above.
    x = recv.float()
    result = x * 0.0 + probs.sum() * 0.0
    for expert in range(params[0].size(0)):
        y = (F.silu(x @ params[0][expert]) * (x @ params[1][expert])) @ params[2][expert]
        # Preserve a distinct low-precision cast per route slot, including
        # repeated expert IDs. Summing weights before this cast changes the
        # backward rounding relative to the independent sparse formulation.
        y = y[:, None, :].expand(-1, probs.size(1), -1).to(recv.dtype).float()
        weight = probs * (local_indices == expert)
        result = result + (y * weight.unsqueeze(-1)).sum(dim=1)
    return result.to(recv.dtype)


def reference(rank, size, step, microbatch, args, params):
    all_inputs = [inputs(source, step, microbatch, args) for source in range(size)]
    xs, routes, weights, dys = (torch.cat(v) for v in zip(*all_inputs))
    nlocal = args.num_experts // size
    valid = (routes >= rank * nlocal) & (routes < (rank + 1) * nlocal)
    selected = valid.any(-1).nonzero().flatten()
    received = xs.index_select(0, selected).detach().requires_grad_(True)
    local_indices = torch.where(valid, routes - rank * nlocal, -1).index_select(0, selected)
    probs = torch.where(valid, weights, 0).index_select(0, selected).detach().requires_grad_(True)
    partial = experts(received, local_indices, probs, params)
    partial.backward(dys.index_select(0, selected))
    out = torch.zeros(xs.size(0), args.hidden)
    dx = torch.zeros_like(out)
    dw = torch.zeros(xs.size(0), args.num_topk)
    out.index_copy_(0, selected, partial.detach().float())
    dx.index_copy_(0, selected, received.grad.float())
    dw.index_copy_(0, selected, probs.grad)
    # CPU/Gloo only: combine independently generated rank-local references.
    for tensor in (out, dx, dw):
        dist.all_reduce(tensor)
    counts = token_counts(size, step, args)
    lo, hi = sum(counts[:rank]), sum(counts[: rank + 1])
    return out[lo:hi].to(xs.dtype), dx[lo:hi].to(xs.dtype), dw[lo:hi]


def check(name, actual, expected, args):
    actual = actual.detach().cpu().float()
    expected = expected.detach().float()
    try:
        torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
        relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-7)
        if relative_l2.item() > 0.03 and (actual - expected).norm().item() > 1e-7:
            raise AssertionError(f"relative L2 error {relative_l2.item()}")
    except AssertionError as error:
        maximum = (actual - expected).abs().max().item() if actual.numel() else 0
        raise AssertionError(f"{name} mismatch: max_abs={maximum}") from error


def check_optimizer_update(index, actual, actual_before, expected, expected_before, args):
    actual_delta = actual.detach().cpu().float() - actual_before
    expected_delta = expected.detach().float() - expected_before
    # Parameter values are much larger than a single SGD update. Comparing
    # only the parameters can accept an entirely skipped optimizer step.
    # Allow FP32 subtraction/rounding noise, then compare the updates directly.
    scale = max(actual_before.abs().max().item(), expected_before.abs().max().item(), 1e-7)
    rounding = 2 * torch.finfo(torch.float32).eps * scale
    try:
        torch.testing.assert_close(actual_delta, expected_delta, atol=rounding, rtol=args.rtol)
    except AssertionError as error:
        raise AssertionError(
            f"optimizer update {index} mismatch: "
            f"max_abs={(actual_delta - expected_delta).abs().max().item()}, "
            f"rounding_atol={rounding}"
        ) from error
    return dict(
        parameter=index,
        actual_l2=actual_delta.norm().item(),
        expected_l2=expected_delta.norm().item(),
        rounding_atol=rounding,
        max_abs_error=(actual_delta - expected_delta).abs().max().item(),
    )


def cpu_reference_check(args):
    # Check the new graph-friendly expert expression against the pre-existing
    # sparse expert formula, including all gradients and empty inputs/routes.
    torch.set_num_threads(1)
    for tokens in (0, 1, 17):
        for empty in (False, True):
            gen = torch.Generator().manual_seed(args.seed + tokens)
            x = torch.randn(tokens, 8, generator=gen).bfloat16()
            route = torch.randint(0, 4, (tokens, 2), generator=gen)
            if empty:
                route.fill_(-1)
            probs = torch.rand(tokens, 2, generator=gen)
            base = [torch.randn(shape, generator=gen) * 0.04 for shape in ((4, 8, 16), (4, 8, 16), (4, 16, 8))]
            results = []
            for impl in (experts, fixed_experts):
                xx = x.clone().requires_grad_(True)
                pp = probs.clone().requires_grad_(True)
                weights = [p.clone().requires_grad_(True) for p in base]
                y = impl(xx, route, pp, weights)
                y.float().sum().backward()
                results.append([y, xx.grad, pp.grad, *(p.grad for p in weights)])
            for actual, expected in zip(*results):
                check("CPU sparse/dense expert parity", actual, expected, args)
    before = torch.tensor([0.04, -0.03])
    after = before - torch.tensor([1e-5, -5e-6])
    check_optimizer_update(0, after, before, after, before, args)
    try:
        check_optimizer_update(0, before, before, after, before, args)
    except AssertionError:
        pass
    else:
        raise AssertionError("optimizer update guard accepted a skipped SGD step")
    print("CPU_TRAINING_REFERENCE_PASS", flush=True)


def run(args):
    import torch_npu  # noqa: F401

    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    rank, size = dist.get_rank(), dist.get_world_size()
    assert args.num_experts % size == 0
    config = Config(args.num_aiv, 6, 256, 6, 128)
    nbytes = max(16 * 1024**2, config.get_nvl_buffer_size_hint(args.hidden * 2, size, num_tokens=args.num_tokens))
    buffer = Buffer(
        dist.group.WORLD,
        num_nvl_bytes=nbytes,
        transport=args.transport,
        explicitly_destroy=True,
        num_max_tokens_per_rank=args.num_tokens if args.uneven or args.graph else None,
    )
    actual_params = parameters(rank, size, args, "npu")
    expected_params = parameters(rank, size, args, "cpu")
    actual_optim = torch.optim.SGD(actual_params, lr=0.03)
    expected_optim = torch.optim.SGD(expected_params, lr=0.03)
    saved = []

    def layer(x, w, route):
        recv, indices, probs, handle = FusedDispatch.apply(
            x,
            route,
            w,
            buffer,
            args.num_experts,
            config,
            not args.synchronous,
            args.allocate_on_comm_stream,
            args.graph,
        )
        if args.checkpoint == "none" and not args.graph:
            saved.append((recv, recv.detach().clone(), probs, probs.detach().clone()))
        local = (fixed_experts if args.graph else experts)(recv, indices, probs, actual_params)
        return FusedCombine.apply(local, buffer, handle, config, not args.synchronous, args.allocate_on_comm_stream)

    def forward_batch(x, route, w):
        if args.checkpoint == "none":
            return layer(x, w, route)

        def call(x_arg, w_arg, route_arg=route):
            return layer(x_arg, w_arg, route_arg)

        with set_checkpoint_early_stop(False):
            return checkpoint(call, x, w, use_reentrant=args.checkpoint == "reentrant")

    def make_batch(step):
        batch = []
        for mb in range(args.microbatches):
            x, route, w, dy = (v.npu() for v in inputs(rank, step, mb, args))
            x.requires_grad_(True)
            w.requires_grad_(True)
            batch.append((mb, x, route, w, dy))
        return batch

    def train_iteration(batch):
        actual_optim.zero_grad(set_to_none=False)
        pending = []
        for mb, x, route, w, dy in batch:
            if x.grad is not None:
                x.grad.zero_()
            if w.grad is not None:
                w.grad.zero_()
            output = forward_batch(x, route, w)
            pending.append((mb, x, w, output, dy))
        for mb, x, w, output, dy in reversed(pending):
            (output.float() * dy.float()).sum().backward()
        actual_optim.step()
        return pending

    graph = None
    caller = torch.npu.Stream()
    caller.wait_stream(torch.npu.current_stream())
    if args.graph:
        batch = make_batch(0)
        caller.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(caller):
            for _ in range(3):
                pending = train_iteration(batch)
        caller.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=caller):
            pending = train_iteration(batch)
        # Warmup performs real optimizer steps. Restore the independently
        # generated initial weights before the first measured graph replay.
        with torch.no_grad():
            for actual, expected in zip(actual_params, expected_params):
                actual.copy_(expected)
        caller.wait_stream(torch.npu.current_stream())
        pointers = [v.data_ptr() for _, x, w, output, _ in pending for v in (x, w, output)]

    torch.npu.reset_peak_memory_stats()
    for step in range(args.steps):
        actual_before = [p.detach().cpu().float().clone() for p in actual_params]
        expected_before = [p.detach().float().clone() for p in expected_params]
        expected_optim.zero_grad(set_to_none=True)
        if graph is None:
            pending = train_iteration(make_batch(step))
        else:
            with torch.npu.stream(caller), torch.no_grad():
                for mb, x, route, w, dy in batch:
                    for target, source in zip((x, route, w, dy), inputs(rank, step, mb, args)):
                        target.copy_(source)
                graph.replay()
            caller.synchronize()
            assert pointers == [v.data_ptr() for _, x, w, output, _ in pending for v in (x, w, output)]
        if args.inject_optimizer_error and step == 0:
            # Simulate a dropped optimizer update while preserving this
            # iteration's forward outputs and gradients.
            with torch.no_grad():
                for actual, before in zip(actual_params, actual_before):
                    actual.copy_(before)
        errors = []

        def checked(*values):
            try:
                check(*values, args)
            except Exception:
                errors.append(traceback.format_exc())

        for mb, x, w, output, dy in reversed(pending):
            expected_out, expected_dx, expected_dw = reference(rank, size, step, mb, args, expected_params)
            if args.inject_error and step == 0 and mb == args.microbatches - 1:
                if expected_dx.numel():
                    expected_dx[0, 0] += 1
                else:
                    expected_params[0].grad[0, 0, 0] += 1
            checked("output", output, expected_out)
            checked("loss", (output.float() * dy.float()).sum(), (expected_out.float() * dy.cpu().float()).sum())
            checked("input gradient", x.grad, expected_dx)
            checked("router gradient", w.grad, expected_dw)
        for i, (actual, expected) in enumerate(zip(actual_params, expected_params)):
            checked(f"expert gradient {i}", actual.grad, expected.grad)
        expected_optim.step()
        update_checks = []
        for i, (actual, expected) in enumerate(zip(actual_params, expected_params)):
            checked(f"optimizer parameter {i}", actual, expected)
            try:
                update_checks.append(
                    check_optimizer_update(i, actual, actual_before[i], expected, expected_before[i], args)
                )
            except Exception:
                errors.append(traceback.format_exc())
        for recv, recv_snapshot, probs, probs_snapshot in saved:
            try:
                torch.testing.assert_close(recv, recv_snapshot, atol=0, rtol=0)
                torch.testing.assert_close(probs, probs_snapshot, atol=0, rtol=0)
            except Exception:
                errors.append(traceback.format_exc())
        saved.clear()
        all_errors = [None] * size
        dist.all_gather_object(all_errors, errors)
        if any(all_errors):
            raise AssertionError("coordinated training check failed: " + str(all_errors))
        print(
            json.dumps(
                dict(
                    rank=rank,
                    step=step,
                    status="TRAINING_STEP_PASS",
                    checkpoint=args.checkpoint,
                    microbatches=args.microbatches,
                    graph=args.graph,
                    local_tokens=token_counts(size, step, args)[rank],
                    optimizer_updates=update_checks,
                )
            ),
            flush=True,
        )
    torch.npu.synchronize()
    allocated = torch.npu.max_memory_allocated()
    reserved = torch.npu.max_memory_reserved()
    if graph is not None:
        graph.reset()
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()
    print(
        json.dumps(
            dict(
                rank=rank,
                status="TRAINING_PASS",
                steps=args.steps,
                torch_peak_allocated=allocated,
                torch_peak_reserved=reserved,
                shmem_requested=nbytes,
                args=vars(args),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("mte", "udma"), required=True)
    parser.add_argument("--num-tokens", type=int, default=129)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ffn-hidden", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--num-aiv", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--checkpoint", choices=("none", "reentrant", "nonreentrant"), default="none")
    parser.add_argument("--allocate-on-comm-stream", action="store_true")
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument("--inject-error", action="store_true")
    parser.add_argument("--inject-optimizer-error", action="store_true")
    parser.add_argument("--uneven", action="store_true")
    parser.add_argument("--vary-tokens", action="store_true")
    parser.add_argument("--empty-route-cycle", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--cpu-reference-check", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=0.03)
    args = parser.parse_args()
    if args.graph and args.vary_tokens:
        parser.error("graph requires fixed physical shapes; change route masks instead")
    if args.vary_tokens and not args.uneven:
        parser.error("vary-tokens requires uneven inputs")
    assert min(args.num_tokens, args.hidden, args.ffn_hidden, args.steps, args.microbatches) > 0
    assert 0 < args.num_topk <= args.num_experts
    if args.cpu_reference_check:
        cpu_reference_check(args)
    else:
        try:
            run(args)
        except BaseException:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
