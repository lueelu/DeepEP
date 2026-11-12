# Copyright (c) 2026, Lu Lu
"""Fixed-address LL graph wrapper for DeepJIT and SuperKernel capture.

The JIT path compiles the Dispatch/Combine entries outside graph capture; this
module only replays the already prepared launches. A SuperKernel only exists
inside a compiled graph, so ``capture`` marks the enclosed calls with a
super_kernel_scope and asks the NPU graph backend to optimise them afterwards.

Every step must be a prepared, allocation-free launch: device addresses and
tiling are fixed before capture and only tensor contents change between
iterations. The jit graph plan binds the AOT device library, whose ordinary and
SK entries share one body, because DeepJIT builds one kernel entry per ELF.
"""

import contextlib

import torch

__all__ = ["LLGraph", "super_kernel_scope"]


@contextlib.contextmanager
def super_kernel_scope(name):
    """Mark the enclosed dispatches as one SuperKernel fusion scope."""
    npu = getattr(torch, "npu", None)
    begin = getattr(npu, "super_kernel_scope_begin", None) if npu is not None else None
    end = getattr(npu, "super_kernel_scope_end", None) if npu is not None else None
    if begin is None or end is None:
        raise NotImplementedError(
            "SuperKernel scopes need torch_npu with super_kernel_scope_begin/end; "
            "direct-launch fusion is supported by the npugraph_ex backend only"
        )
    begin(name)
    try:
        yield
    finally:
        end(name)


class LLGraph:
    """Capture and replay a fixed-address Dispatch -> compute -> Combine chain.

    ``steps`` are callables with no arguments; each submits one prepared
    launch on the communication stream. Construction and compilation must
    happen before ``capture``. All ranks call ``capture`` and ``replay`` in the
    same order and keep the graph and every plan alive.
    """

    def __init__(self, steps, *, scope="ll_ep"):
        if not steps:
            raise ValueError("LLGraph needs at least one prepared launch")
        self.steps = tuple(steps)
        self.scope = scope
        self.graph = None

    def capture(self):
        graph = torch.npu.CUDAGraph()
        with torch.npu.graph(graph):
            with super_kernel_scope(self.scope):
                for step in self.steps:
                    step()
        optimize = getattr(torch.npu, "super_kernel_optimize", None)
        if optimize is None:
            raise NotImplementedError(
                "This torch_npu build cannot optimise a captured graph for SuperKernel"
            )
        optimize(graph)
        self.graph = graph
        return self

    def replay(self):
        if self.graph is None:
            raise RuntimeError("Capture the LL graph before replay")
        self.graph.replay()

    def reset(self):
        """Release the captured graph after the last device use has retired."""
        graph, self.graph = self.graph, None
        if graph is not None:
            graph.reset()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.reset()
