# Copyright (c) 2026, Lu Lu
# Modified by Joyce_An 2026

"""Hand-calculated anchors and all-rank mathematical invariants."""

from dataclasses import replace

import pytest

from tests.utils.cases import Case, ROUTES, make_case
from tests.utils.checks import assert_close, assert_dispatch
from tests.utils.reference import (
    Payload,
    combine,
    dispatch,
    expected_expert_combine,
    expert_payload,
)


@pytest.fixture
def tiny():
    return Case(
        "hand",
        4,
        1,
        2,
        (((2,), (3,)), ((5,),)),
        (((0, 1), (2, -1)), ((1, 1),)),
        (((0.25, 0.5), (0.75, 9.0)), ((0.125, 0.625),)),
    )


def raw_dispatch(layout, rank):
    rows = layout.ranks[rank]
    x = [list(row.x) for row in rows]
    indices = None if layout.expanded else [list(row.indices) for row in rows]
    has_weights = any(row.weights is not None for rows in layout.ranks for row in rows)
    weights = None if not has_weights else [row.weights[0] if layout.expanded else list(row.weights) for row in rows]
    return x, indices, weights, list(layout.counts[rank])


def test_hand_calculated_dispatch(tiny):
    ordinary = dispatch(tiny)
    assert ordinary.counts == ((1, 3), (1, 0))
    assert [(row.source, row.token, row.indices, row.x, row.weights) for row in ordinary.ranks[0]] == [
        (0, 0, (0, 1), (2,), (0.25, 0.5)),
        (1, 0, (1, 1), (5,), (0.125, 0.625)),
    ]
    assert ordinary.ranks[1][0].indices == (0, -1)
    assert ordinary.ranks[1][0].weights == (0.75, 9.0)
    expanded = dispatch(tiny, expanded=True)
    assert [(row.expert, row.source, row.token, row.slots, row.x, row.weights) for row in expanded.ranks[0]] == [
        (0, 0, 0, (0,), (2,), (0.25,)),
        (1, 0, 0, (1,), (2,), (0.5,)),
        (1, 1, 0, (0,), (5,), (0.125,)),
        (1, 1, 0, (1,), (5,), (0.625,)),
    ]


@pytest.mark.parametrize("expanded", [False, True])
def test_hand_calculated_combine_keeps_value_channel_separate(tiny, expanded):
    layout = dispatch(tiny, expanded=expanded)
    # Explicit expert sums: 2*(1+2)=6; 3*3=9; 5*(2+2)=20.
    expected = ((((6,), (9,)), ((20,),)), (((0.25, 0.0), (-0.25, 0.0)), ((0.375, -0.125),)))
    assert expected_expert_combine(tiny) == expected
    assert combine(tiny, layout, expert_payload(tiny, layout)) == expected


@pytest.mark.parametrize("name", ROUTES)
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("with_weights", [False, True])
def test_named_scenarios(name, expanded, with_weights):
    case = make_case(name, with_weights=with_weights)
    layout = dispatch(case, expanded=expanded)
    for rank in range(case.world_size):
        raw = raw_dispatch(layout, rank)
        # An all-masked case still has an empty weight tensor when requested.
        if with_weights and raw[2] is None:
            raw = raw[:2] + ([], raw[3])
        assert_dispatch(case, rank, expanded, *raw)
    standalone = combine(case, layout, expert_payload(case, layout))
    roundtrip = combine(case, layout, expert_payload(case, layout, use_received_x=True))
    assert_close(standalone, expected_expert_combine(case))
    assert_close(roundtrip, expected_expert_combine(case))


def test_identity_roundtrip_is_not_the_input(tiny):
    # One row per destination rank versus one row per valid slot.
    ordinary, expanded = dispatch(tiny), dispatch(tiny, expanded=True)

    def identity(layout):
        payloads = tuple(tuple(Payload(row.x, row.weights) for row in rows) for rows in layout.ranks)
        return combine(tiny, layout, payloads)[0]

    assert identity(ordinary) == (((2.0,), (3.0,)), ((5.0,),))
    assert identity(expanded) == (((4.0,), (3.0,)), ((10.0,),))


@pytest.mark.parametrize(
    "change",
    [
        {"experts": 3},
        {"hidden": 0},
        {"routes": ()},
        {"weights": ()},
        {"routes": (((4, 0),), ((0, 1),))},
        {"routes": (((-2, 0), (0, 1)), ((0, 1),))},
        {"routes": (((0,), (0, 1)), ((0, 1),))},
    ],
)
def test_invalid_reference_inputs_fail(tiny, change):
    with pytest.raises(ValueError):
        dispatch(replace(tiny, **change))


@pytest.mark.parametrize("fault", ["wrong_rank", "missing", "duplicate", "weight", "index", "count"])
@pytest.mark.parametrize("expanded", [False, True])
def test_dispatch_checker_rejects_injected_faults(tiny, fault, expanded):
    layout = dispatch(tiny, expanded=expanded)
    x, indices, weights, counts = raw_dispatch(layout, 0)
    if fault == "wrong_rank":
        x[0] = [3]  # A valid token that belongs exclusively to rank 1.
    elif fault == "missing":
        x.pop()
    elif fault == "duplicate":
        x[0] = x[-1]
    elif fault == "weight":
        if expanded:
            weights[0] += 0.125
        else:
            weights[0][0] += 0.125
    elif fault == "index":
        if expanded:
            indices = []  # Wrong layout channel.
        else:
            indices[0][0] = -1
    else:
        counts[0] += 1
    with pytest.raises(AssertionError):
        assert_dispatch(tiny, 0, expanded, x, indices, weights, counts)


@pytest.mark.parametrize("fault", ["missing_contribution", "gate_twice", "wrong_value", "nan", "shape"])
def test_combine_checker_rejects_injected_faults(tiny, fault):
    layout = dispatch(tiny, expanded=True)
    payloads = [list(rank) for rank in expert_payload(tiny, layout)]
    row = payloads[0][0]
    if fault == "missing_contribution":
        payloads[0][0] = replace(row, x=(0.0,))
    elif fault == "gate_twice":
        payloads[0][0] = replace(row, x=(row.x[0] * 0.25,))
    elif fault == "wrong_value":
        payloads[0][0] = replace(row, weights=(7.0,))
    elif fault == "nan":
        payloads[0][0] = replace(row, x=(float("nan"),))
    result = combine(tiny, layout, payloads)
    if fault == "shape":
        result = (result[0][:-1], result[1])
    with pytest.raises(AssertionError):
        assert_close(result, expected_expert_combine(tiny))


def test_masked_dispatch_weights_are_not_part_of_the_contract(tiny):
    layout = dispatch(tiny)
    raw = raw_dispatch(layout, 1)
    raw[2][0][1] = 123.0
    assert_dispatch(tiny, 1, False, *raw)


def test_scheduling_order_may_change_but_multiplicity_must_not(tiny):
    layout = dispatch(tiny)
    x, indices, weights, counts = raw_dispatch(layout, 0)
    assert_dispatch(tiny, 0, False, x[::-1], indices[::-1], weights[::-1], counts)


def test_missing_payload_fails_instead_of_truncating(tiny):
    layout = dispatch(tiny)
    payloads = expert_payload(tiny, layout)
    with pytest.raises(ValueError, match="row count"):
        combine(tiny, layout, (payloads[0][:-1], payloads[1]))
