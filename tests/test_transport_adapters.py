# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Device-free keyword plumbing checks, not native communication evidence."""

from types import SimpleNamespace

import pytest

import deep_ep
from tests.utils import device


@pytest.mark.parametrize("rank", [0, 1])
def test_generalized_legacy_adapter_uses_domain_names(monkeypatch, rank):
    calls = []

    class Config:
        def __init__(self, multiplier):
            self.multiplier = multiplier

        def get_scaleup_buffer_size_hint(self, hidden_bytes, ranks):
            return hidden_bytes * ranks * self.multiplier

        def get_scaleout_buffer_size_hint(self, hidden_bytes, ranks):
            return hidden_bytes * self.multiplier

    class Buffer:
        get_dispatch_config = staticmethod(lambda ranks: Config(1))
        get_combine_config = staticmethod(lambda ranks: Config(2))

        def __init__(self, group, *, num_scaleup_bytes, num_scaleout_bytes, explicitly_destroy):
            calls.append(("create", group, num_scaleup_bytes, num_scaleout_bytes, explicitly_destroy))

        def get_dispatch_layout(self, idx, experts):
            assert experts == 4
            return "per_rank", "per_scaleout", "per_expert", "mask", None

        def dispatch(
            self,
            x,
            *,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_scaleout_rank,
            num_tokens_per_expert,
            is_token_in_rank,
        ):
            calls.append(("dispatch", num_tokens_per_rank, num_tokens_per_scaleout_rank))
            assert num_tokens_per_expert == "per_expert" and is_token_in_rank == "mask"
            return x, topk_idx, topk_weights, [1, 0], "handle", None

        def destroy(self):
            calls.append("destroy")

    monkeypatch.setattr(deep_ep, "Buffer", Buffer)
    monkeypatch.setattr(device, "tensor", lambda torch, values, shape, dtype, device: SimpleNamespace(dtype=dtype))
    case = SimpleNamespace(
        world_size=2,
        hidden=8,
        topk=1,
        experts=4,
        dtype="bfloat16",
        x=(((1.0,) * 8,), ((2.0,) * 8,)),
        routes=(((0,),), ((2,),)),
        weights=(((1.0,),), ((1.0,),)),
    )
    torch = SimpleNamespace(bfloat16="bf16", int64="int64", float32="fp32")
    session = (torch, "group", SimpleNamespace(rank=rank, local_rank=rank))
    with device.buffer_for(session, case, "legacy") as buffer:
        received = device.launch_dispatch(session, buffer, case, "legacy", False)
        assert received[-1] == "handle"
    assert calls == [("create", "group", 64, 32, True), ("dispatch", "per_rank", "per_scaleout"), "destroy"]
