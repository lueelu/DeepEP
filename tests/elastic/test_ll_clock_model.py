# Copyright (c) 2026, Lu Lu

"""CPU checks for the shared affine clock model.

The model converts between a rank's device counter and rank 0's counter in both
directions. These tests need no NPU: they build synthetic four-timestamp rows
whose true rate and offset are known and verify that the fitted model recovers
them, that deadlines are never early, and that large anchors stay exact.
"""

from fractions import Fraction

import pytest

from deep_ep.utils import ll_clock_model

SPAN = 5 * ll_clock_model.CYCLES_PER_SECOND


def sample_rows(local, reference, sequences=8, round_trip=18_000, path=1_000):
    """Build rows whose local/reference midpoints are exactly ``local``/``reference``."""
    half = round_trip // 2 + path
    rows = []
    for sequence in range(sequences):
        t0 = local - half
        t3 = local + half
        t1 = reference - path
        t2 = reference + path
        offset_sum = (t1 - t0) + (t2 - t3)
        offset = (abs(offset_sum) // 2) * (-1 if offset_sum < 0 else 1)
        packed = sequence & 0xFFFFFFFF
        rows.append([t0, t1, t2, t3, round_trip, offset, packed, 0])
    return rows


def window(rank, index, local, reference):
    return ll_clock_model.window_from_rows(sample_rows(local, reference), rank, index)


def drifting_windows(rank, ppm, l0, r0, count=3, span=SPAN):
    """Exact integer windows on R = R0 + (1 + ppm/1e6) * (L - L0)."""
    windows = []
    for index in range(count):
        local = l0 + index * span
        drift = (index * span) * ppm // 1_000_000
        windows.append(window(rank, index, local, r0 + index * span + drift))
    return windows


def test_sample_validation_rejects_torn_rows():
    rows = sample_rows(10 * SPAN, 10 * SPAN)
    assert ll_clock_model.parse_sample(rows[0], 0) is not None
    assert ll_clock_model.parse_sample(rows[0], 1) is None          # sequence mismatch
    torn = list(rows[0])
    torn[4] += 1                                                    # reported RTT mismatch
    assert ll_clock_model.parse_sample(torn, 0) is None
    failed = list(rows[0])
    failed[6] |= 1 << 32                                            # non-zero status
    assert ll_clock_model.parse_sample(failed, 0) is None


def test_zero_drift_keeps_the_offset():
    windows = drifting_windows(3, 0, 10 * SPAN, 10 * SPAN + 4_242)
    model = ll_clock_model.fit_model(windows, 3)
    assert model["alpha_num"] == model["alpha_den"] == 1
    assert model["rate_correction"] == 0.0
    assert model["offset_cycles"] == 4_242
    target = 20 * SPAN
    assert ll_clock_model.local_deadline(model, target) == target - 4_242
    assert ll_clock_model.forward_reference(model, target - 4_242) == target


@pytest.mark.parametrize("ppm", [2, -2, 40, -40])
def test_drift_is_recovered_and_deadlines_stay_late(ppm):
    windows = drifting_windows(5, ppm, 100 * SPAN, 100 * SPAN + 900)
    model = ll_clock_model.fit_model(windows, 5)
    assert model["rate_ppm"] == pytest.approx(ppm, abs=1e-9)
    assert model["residual_cycles"] == 0
    assert model["residual_validated"] is True
    targets = [100 * SPAN + step * 1_000_000 for step in range(1, 200)]
    deadlines = [ll_clock_model.local_deadline(model, target) for target in targets]
    assert deadlines == sorted(deadlines)
    for target, deadline in zip(targets, deadlines):
        assert ll_clock_model.forward_reference(model, deadline) >= target


def test_two_windows_report_an_unvalidated_rate_but_still_fit():
    windows = drifting_windows(2, 4, 40 * SPAN, 40 * SPAN, count=2)
    model = ll_clock_model.fit_model(windows, 2)
    assert model["rate_resolved"] is True
    assert model["holdout_window"] is None
    assert model["residual_cycles"] is None
    assert model["rate_ppm"] == pytest.approx(4, abs=1e-9)


def test_single_window_cannot_resolve_a_rate():
    model = ll_clock_model.fit_model(drifting_windows(7, 4, SPAN, SPAN, count=1), 7)
    assert model["rate_resolved"] is False
    assert model["rate_ppm"] == 0.0
    assert model["sigma_alpha_ppm"] is None


def test_rank_zero_is_the_identity():
    model = ll_clock_model.fit_model(drifting_windows(0, 9, 7 * SPAN, 7 * SPAN, count=3), 0)
    assert model["offset_cycles"] == 0 and model["alpha"] == 1.0
    assert ll_clock_model.local_deadline(model, 12 * SPAN) == 12 * SPAN
    assert ll_clock_model.forward_reference(model, 12 * SPAN) == 12 * SPAN


def test_large_anchors_stay_integer_exact():
    base = (1 << 62) // 3
    windows = drifting_windows(11, 3, base, base + 1_000_000)
    model = ll_clock_model.fit_model(windows, 11)
    target = base + 3 * SPAN
    deadline = ll_clock_model.local_deadline(model, target)
    assert isinstance(deadline, int)
    assert ll_clock_model.forward_reference(model, deadline) >= target
    broken = dict(model, alpha_num=-1, alpha_den=1)
    with pytest.raises(ValueError):
        ll_clock_model.local_deadline(broken, target)


def test_reanchor_keeps_the_rate_and_moves_the_intercept():
    windows = drifting_windows(4, 6, 20 * SPAN, 20 * SPAN, count=3)
    model = ll_clock_model.fit_model(windows, 4)
    later_local = 30 * SPAN
    later_reference = ll_clock_model.forward_reference(model, later_local)
    anchored = ll_clock_model.reanchor(model, window(4, 3, later_local, later_reference))
    assert anchored["alpha_num"] == model["alpha_num"]
    assert anchored["l0"] == later_local and anchored["r0"] == later_reference
    target = later_local + SPAN
    assert ll_clock_model.forward_reference(
        anchored, ll_clock_model.local_deadline(anchored, target)) >= target


def test_error_budget_grows_with_projection_distance():
    windows = drifting_windows(6, 5, 10 * SPAN, 10 * SPAN)
    model = ll_clock_model.fit_model(windows, 6)
    near = ll_clock_model.error_budget(model, model["l0"])
    far = ll_clock_model.error_budget(model, model["l0"] + 60 * SPAN)
    assert far["total_cycles"] >= near["total_cycles"]
    assert far["rate_cycles"] == pytest.approx(
        abs(far["rate_cycles"]), abs=1e-6)
    assert far["total_ns"] == far["total_cycles"]


def test_window_requires_enough_valid_samples():
    rows = sample_rows(SPAN, SPAN)
    rows[0][6] |= 1 << 32
    rows[1][6] |= 1 << 32
    rows[2][6] |= 1 << 32
    with pytest.raises(RuntimeError):
        ll_clock_model.window_from_rows(rows, 1, 0)


def test_alpha_beyond_the_admitted_rate_is_clamped():
    windows = drifting_windows(8, 0, 10 * SPAN, 10 * SPAN)
    windows[1]["r"] += 10 ** 12
    model = ll_clock_model.fit_model(windows, 8)
    assert model["rate_clamped"] is True
    assert model["alpha"] == 1.0
    assert model["rate_resolved"] is False
    assert Fraction(model["alpha_num"], model["alpha_den"]) == 1
