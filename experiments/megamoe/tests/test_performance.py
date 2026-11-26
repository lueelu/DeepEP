# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

import pytest

from deep_ep_experimental.megamoe.performance import summarize_rounds


def test_round_policy_node_scope_and_no_device_timestamp_subtraction():
    per_rank = [[1000, 900, 3, 9], [1100, 800, 8, 4], [1200, 700, 20, 2], [1300, 600, 12, 15]]
    report = summarize_rounds(per_rank, 2)
    assert report["mean_us"] == 2.5  # min per round, NOT min of rank means
    assert report["rank_max_mean_us"] == 17.5
    assert [node["mean_us"] for node in report["per_node"]] == [3.5, 7]
    assert report["metric"] == "megamoe_kernel_duration_us"
    assert report["per_rank_rounds_us"] == [[3, 9], [8, 4], [20, 2], [12, 15]]
    assert report["recorded_rounds"] == 4 and report["rounds"] == 2
    assert per_rank[0] == [1000, 900, 3, 9]  # Selection does not mutate the raw input.
    with pytest.raises(ValueError, match="divide"):
        summarize_rounds(per_rank, 3)


@pytest.mark.parametrize(
    "iterations,first,selected,mean",
    [
        (1, 0, 1, 1),
        (2, 1, 1, 2),
        (3, 1, 2, 2.5),
        (5, 2, 3, 4),
        (20, 10, 10, 15.5),
        (40, 20, 20, 30.5),
    ],
)
def test_tail_half_selection_even_odd_and_single_round(iterations, first, selected, mean):
    report = summarize_rounds([list(range(1, iterations + 1)), list(range(2, 2 * iterations + 1, 2))])
    assert report["iteration_selection"] == "tail_half_ceil"
    assert report["recorded_rounds"] == iterations
    assert report["first_selected_iteration"] == first
    assert report["rounds"] == selected
    assert report["mean_us"] == mean
    assert report["rank_max_mean_us"] == 2 * mean
    assert report["rank_min_rounds_us"][0] == first + 1
    assert report["rank_min_rounds_us"][-1] == iterations
