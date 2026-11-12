# Copyright (c) 2026, Lu Lu
"""CPU oracle/fault checks for the device FFN E2E test; no NPU qualification."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


spec = importlib.util.spec_from_file_location("ubx_ffn_st", Path(__file__).with_name("test_ffn.py"))
ffn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ffn)


def test_help_and_three_round_default_without_npu():
    output = subprocess.run([sys.executable, ffn.__file__, "--help"], capture_output=True, text=True, check=True)
    assert "--ffn-hidden" in output.stdout and "--repeat" in output.stdout
    args = ffn.parser().parse_args(["--output", "unused"])
    assert args.repeat == 3 and args.ffn_hidden == 256


def test_expert_ranges_keep_empty_experts_and_ignore_capacity_tail():
    assert ffn.expert_ranges([1, 3, 3, 3, 4, 5], 2, [3, 0, 2], 10) == [(0, 3), (3, 3), (3, 5)]


@pytest.mark.parametrize(
    "prefix,counts,capacity",
    [
        ([1], [1], 8),
        ([-1, 2], [2], 8),
        ([3, 2], [2], 8),
        ([1, 2], [3], 8),
        ([1, 2], [2], 1),
    ],
)
def test_bad_dispatch_prefix_fails(prefix, counts, capacity):
    with pytest.raises(AssertionError):
        ffn.expert_ranges(prefix, 2, counts, capacity)


@pytest.mark.parametrize("name", ["bf16", "fp16"])
def test_cpu_dense_ffn_roundtrip_matches_independent_source_oracle(name):
    torch = pytest.importorskip("torch")
    dtype = torch.bfloat16 if name == "bf16" else torch.float16
    world, tokens, hidden, intermediate, experts = 2, 3, 16, 8, 8
    routes = [[[0, 4], [1, 3], [0, 5]], [[2, 4], [0, 3], [1, 5]]]
    inputs = [ffn.input_tensor(r, tokens, hidden, dtype, 7, 0) for r in range(world)]
    weights = [ffn.routing_weights(r, tokens, 2, 7, 0) for r in range(world)]
    actual = [torch.zeros(tokens, hidden) for _ in range(world)]
    for expert in range(experts):
        # Pack in reverse source/token order, independently of the oracle.
        entries = [
            (r, t, k)
            for r in reversed(range(world))
            for t in reversed(range(tokens))
            for k, e in enumerate(routes[r][t])
            if e == expert
        ]
        if not entries:
            continue
        received = torch.stack([inputs[r][t] for r, t, _ in entries])
        values = ffn.expert_forward(received, ffn.expert_parameters(expert, hidden, intermediate, dtype, 7))
        for row, (source, token, slot) in enumerate(entries):
            actual[source][token] += values[row].float() * weights[source][token, slot]
    for rank in range(world):
        expected = ffn.reference_combine(inputs[rank], routes[rank], weights[rank], intermediate, 7)
        assert ffn.check_close(actual[rank].to(dtype), expected, name)["max_scaled"] <= 1


@pytest.mark.parametrize("fault", ["missing-expert", "wrong-expert", "double-weight", "nan"])
def test_oracle_rejects_wrong_route_weight_and_nonfinite_output(fault):
    torch = pytest.importorskip("torch")
    x = ffn.input_tensor(1, 4, 32, torch.bfloat16, 5, 0)
    weights = ffn.routing_weights(1, 4, 2, 5, 0)
    routes = [[1, 3], [0, 2], [2, 3], [1, 2]]
    expected = ffn.reference_combine(x, routes, weights, 16, 5)
    if fault == "wrong-expert":
        actual = ffn.reference_combine(x, [[e + 1 for e in row] for row in routes], weights, 16, 5)
    elif fault == "double-weight":
        actual = ffn.reference_combine(x, routes, weights.square(), 16, 5)
    elif fault == "missing-expert":
        broken_weights = weights.clone()
        broken_weights[:, 0] = 0
        actual = ffn.reference_combine(x, routes, broken_weights, 16, 5)
    else:
        actual = expected.clone()
        actual[0, 0] = float("nan")
    with pytest.raises(AssertionError):
        ffn.check_close(actual, expected, "bf16")


def test_inputs_and_routing_weights_change_between_bank_reuses():
    torch = pytest.importorskip("torch")
    x = ffn.input_tensor(17, 4, 32, torch.bfloat16, 5, 0)
    changed = ffn.input_tensor(17, 4, 32, torch.bfloat16, 5, 1)
    assert torch.equal(x[:, :4], changed[:, :4]) and not torch.equal(x[:, 4:], changed[:, 4:])
    weights = ffn.routing_weights(17, 4, 8, 5, 0)
    assert not torch.equal(weights, ffn.routing_weights(17, 4, 8, 5, 1))
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))
