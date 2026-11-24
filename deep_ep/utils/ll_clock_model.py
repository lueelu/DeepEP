# Copyright (c) 2026, Lu Lu

"""Shared affine device-clock model for the low-latency clock-alignment protocol.

The device counter runs at a rank-specific rate. Relative to rank 0's counter,
rank ``r`` is

    R_r(L) = R0_r + alpha_r * (L - L0_r)

where ``L`` is rank ``r``'s local counter, ``R`` is the same instant on rank
0's counter, and ``alpha_r`` is the ratio of the reference rate to the local
rate. ``(L0_r, R0_r)`` is one measured anchor. A single offset cannot represent
this over a graph that spans seconds: applying ``offset`` again every iteration
re-introduces a growing error, which is why the previous ``rate_correction=0``
estimator drifted by microseconds inside a 100-call graph.

Both directions come from the same model:

    R(L) = R0 + alpha * (L - L0)                 forward, for reporting
    L(T) = L0 + ceil((T - R0) / alpha)           inverse, for deadlines

The inverse rounds up, so a deadline never releases before the estimated model;
it cannot remove the model error itself. Callers must publish the error budget
from :func:`error_budget` and label the model domain rather than claiming an
absolute physical alignment.

All absolute counters stay integers and every conversion uses exact rational
arithmetic, so large anchors are never floated. Only uncertainties are floats.
"""

import math
import time
from fractions import Fraction

CYCLES_PER_SECOND = 1_000_000_000
SAMPLE_WORDS = 8
MODEL_SCHEMA = "deepep_clock_model.v2"

# A relative frequency difference beyond this is a broken exchange, not drift.
MAX_RATE_PPM = 500.0
# Window history kept for reporting; re-anchoring does not grow without bound.
MAX_MODEL_WINDOWS = 8
# Deadline inversion rounds up by less than one cycle; forward rounds by half.
ROUNDING_CYCLES = 1


def parse_sample(row, sequence):
    """Validate one four-timestamp sample; ``None`` means it is unusable."""
    if len(row) != SAMPLE_WORDS:
        return None
    t0, t1, t2, t3, reported_rtt, reported_offset, packed, _ = map(int, row)
    actual_sequence = packed & 0xFFFFFFFF
    status = (packed >> 32) & 0xFFFFFFFF
    if status != 0 or actual_sequence != sequence:
        return None
    if min(t0, t1, t2, t3) <= 0 or t3 < t0 or t2 < t1 or t3 - t0 < t2 - t1:
        return None
    round_trip = (t3 - t0) - (t2 - t1)
    offset_sum = (t1 - t0) + (t2 - t3)
    offset = (abs(offset_sum) // 2) * (-1 if offset_sum < 0 else 1)
    # The device reports both derived values; a mismatch means the sample was
    # torn or reordered, so it must not enter the fit.
    if round_trip != reported_rtt or offset != reported_offset:
        return None
    return dict(
        sequence=int(sequence),
        round_trip_cycles=round_trip,
        local_cycle=(t0 + t3) // 2,
        reference_cycle=(t1 + t2) // 2,
        error_cycles=(round_trip + 1) // 2,
    )


def window_from_rows(rows, rank, window=0):
    """Reduce one calibration window to its lowest-RTT anchor.

    Every sample in a window is taken within a few milliseconds of the others,
    so a window cannot resolve frequency drift by itself; it contributes one
    (local, reference, error) point to the fit.
    """
    candidates = []
    for sequence, row in enumerate(rows):
        sample = parse_sample(row, sequence)
        if sample is not None:
            candidates.append(sample)
    minimum = max(4, math.ceil(len(rows) * 0.75))
    if len(candidates) < minimum:
        raise RuntimeError(
            f"Clock calibration rank {rank} window {window} has {len(candidates)}/{len(rows)} "
            f"valid samples; require {minimum}"
        )
    anchor = min(candidates, key=lambda sample: (sample["round_trip_cycles"], sample["sequence"]))
    return dict(
        window=int(window),
        rank=int(rank),
        l=anchor["local_cycle"],
        r=anchor["reference_cycle"],
        error_cycles=anchor["error_cycles"],
        round_trip_cycles=anchor["round_trip_cycles"],
        sequence=anchor["sequence"],
        valid_samples=len(candidates),
        rejected_samples=len(rows) - len(candidates),
    )


def _normalized_windows(windows):
    result = []
    for window in windows:
        result.append(dict(
            window=int(window["window"]),
            l=int(window["l"]),
            r=int(window["r"]),
            error_cycles=int(window["error_cycles"]),
            round_trip_cycles=int(window["round_trip_cycles"]),
        ))
    return result


def _slope_uncertainty(fit_windows, anchor, alpha, sxx):
    """Rate uncertainty in reference cycles per local cycle."""
    deltas = [(int(w["l"]) - int(anchor["l"]), int(w["r"]) - int(anchor["r"])) for w in fit_windows]
    if len(fit_windows) >= 3:
        sse = 0.0
        for dl, dr in deltas:
            sse += float(Fraction(dr) - alpha * dl) ** 2
        if sse <= 0.0 or sxx <= 0:
            return 0.0
        return math.sqrt(sse / (len(fit_windows) - 2) / sxx)
    far = max(fit_windows, key=lambda w: abs(int(w["l"]) - int(anchor["l"])))
    span = abs(int(far["l"]) - int(anchor["l"]))
    if span == 0:
        return None
    return (int(anchor["error_cycles"]) + int(far["error_cycles"])) / span


def fit_model(windows, rank):
    """Fit one rank's affine model from ordered calibration windows.

    With three or more windows the last one is held out of the fit and used as
    a future-time residual check, so the reported model is validated on data it
    never saw. Two windows are enough for a slope but not for validation.
    """
    windows = list(windows)
    if not windows:
        raise ValueError("Clock model needs at least one calibration window")
    anchor = min(windows, key=lambda window: (int(window["error_cycles"]), int(window["window"])))
    l0, r0 = int(anchor["l"]), int(anchor["r"])
    anchor_error = int(anchor["error_cycles"])
    now = time.time()

    if int(rank) == 0:
        # Rank 0 is the reference; its model is the identity by definition.
        return dict(
            schema=MODEL_SCHEMA,
            rank=0,
            reference_rank=0,
            state="ready",
            alpha_num=1,
            alpha_den=1,
            alpha=1.0,
            rate_correction=0.0,
            rate_ppm=0.0,
            l0=l0,
            r0=l0,
            anchor_cycle=l0,
            offset_cycles=0,
            anchor_error_cycles=0,
            min_round_trip_cycles=int(anchor["round_trip_cycles"]),
            error_bound_cycles=0,
            sigma_alpha_ppm=0.0,
            rate_resolved=True,
            rate_clamped=False,
            residual_cycles=None,
            residual_validated=True,
            fit_windows=len(windows),
            holdout_window=None,
            valid_samples=sum(int(w["valid_samples"]) for w in windows),
            rejected_samples=sum(int(w["rejected_samples"]) for w in windows),
            sample_sequence=int(anchor["sequence"]),
            windows=_normalized_windows(windows),
            calibrated_wall=now,
            alpha_calibrated_wall=now,
            reanchors=0,
        )

    fit_windows = windows[:-1] if len(windows) >= 3 else windows
    holdout = windows[-1] if len(windows) >= 3 else None
    sxx = 0
    sxy = 0
    for window in fit_windows:
        dl = int(window["l"]) - l0
        dr = int(window["r"]) - r0
        sxx += dl * dl
        sxy += dl * dr

    clamped = False
    sigma = None
    if sxx == 0:
        # One effective anchor: only the offset is measurable, so the rate stays
        # unresolved instead of being reported as a zero-drift model.
        alpha = Fraction(1)
    else:
        alpha = Fraction(sxy, sxx)
        if alpha <= 0:
            alpha = Fraction(1)
            clamped = True
        elif abs(float(alpha) - 1.0) * 1e6 > MAX_RATE_PPM:
            alpha = Fraction(1)
            clamped = True
        sigma = _slope_uncertainty(fit_windows, anchor, alpha, sxx)
    rate_resolved = sigma is not None and not clamped

    residual = None
    validated = True
    if holdout is not None:
        predicted = Fraction(r0) + alpha * (int(holdout["l"]) - l0)
        residual = _round_half_up(predicted) - int(holdout["r"])
        validated = abs(residual) <= int(holdout["error_cycles"]) + ROUNDING_CYCLES

    rate_correction = float(alpha) - 1.0
    return dict(
        schema=MODEL_SCHEMA,
        rank=int(rank),
        reference_rank=0,
        state="ready",
        alpha_num=alpha.numerator,
        alpha_den=alpha.denominator,
        alpha=float(alpha),
        rate_correction=rate_correction,
        rate_ppm=rate_correction * 1e6,
        l0=l0,
        r0=r0,
        anchor_cycle=l0,
        offset_cycles=r0 - l0,
        anchor_error_cycles=anchor_error,
        min_round_trip_cycles=int(anchor["round_trip_cycles"]),
        error_bound_cycles=anchor_error,
        sigma_alpha_ppm=None if sigma is None else float(sigma) * 1e6,
        rate_resolved=rate_resolved,
        rate_clamped=clamped,
        residual_cycles=residual,
        residual_validated=validated,
        fit_windows=len(fit_windows),
        holdout_window=None if holdout is None else int(holdout["window"]),
        valid_samples=sum(int(w["valid_samples"]) for w in windows),
        rejected_samples=sum(int(w["rejected_samples"]) for w in windows),
        sample_sequence=int(anchor["sequence"]),
        windows=_normalized_windows(windows),
        calibrated_wall=now,
        alpha_calibrated_wall=now,
        reanchors=0,
    )


def reanchor(model, window):
    """Move the anchor of an existing model to a new window.

    The fitted rate stays; only the intercept moves, so a replay does not need
    a new multi-window calibration while the model is still younger than its
    admitted age.
    """
    updated = dict(model)
    updated.update(
        l0=int(window["l"]),
        r0=int(window["l"]) if int(model["rank"]) == 0 else int(window["r"]),
        anchor_cycle=int(window["l"]),
        offset_cycles=0 if int(model["rank"]) == 0 else int(window["r"]) - int(window["l"]),
        anchor_error_cycles=int(window["error_cycles"]),
        min_round_trip_cycles=int(window["round_trip_cycles"]),
        error_bound_cycles=int(window["error_cycles"]),
        sample_sequence=int(window["sequence"]),
        reanchor_wall=time.time(),
    )
    windows = list(model.get("windows", []))
    windows.append(dict(
        window=int(window["window"]),
        l=int(window["l"]),
        r=int(window["r"]),
        error_cycles=int(window["error_cycles"]),
        round_trip_cycles=int(window["round_trip_cycles"]),
    ))
    updated["windows"] = windows[-MAX_MODEL_WINDOWS:]
    updated["reanchors"] = int(model.get("reanchors", 0)) + 1
    return updated


def _alpha(model):
    return Fraction(int(model["alpha_num"]), int(model["alpha_den"]))


def _round_half_up(value):
    return math.floor(value + Fraction(1, 2))


def forward_reference(model, local_cycle):
    """Map one local device counter to the reference (rank 0) domain."""
    value = Fraction(int(model["r0"])) + _alpha(model) * (int(local_cycle) - int(model["l0"]))
    return _round_half_up(value)


def local_deadline(model, target_cycle):
    """Invert the model and round up, so a deadline is never early."""
    alpha = _alpha(model)
    if alpha <= 0:
        raise ValueError("Clock model rate must be positive")
    deadline = int(model["l0"]) + math.ceil(Fraction(int(target_cycle) - int(model["r0"])) / alpha)
    if not 0 <= deadline < 1 << 64:
        raise ValueError("Clock alignment deadline is outside uint64")
    return deadline


def reference_now(model):
    """Reference-domain time of the most recent anchor, for choosing a base."""
    windows = model.get("windows") or []
    local = int(windows[-1]["l"]) if windows else int(model["l0"])
    return forward_reference(model, local)


def model_age_seconds(model, now=None):
    """Age of the fitted rate, which is the part a re-anchor cannot refresh."""
    calibrated = model.get("alpha_calibrated_wall")
    if calibrated is None:
        return float("inf")
    return max(0.0, (time.time() if now is None else float(now)) - float(calibrated))


def error_budget(model, local_cycle):
    """Propagated error of one conversion at ``local_cycle``.

    E(L) = E_anchor + |L - L0| * sigma_alpha + E_residual + E_round.
    The result is a model-domain bound; it does not include the asymmetric
    transport path, so it must not be advertised as physical first-instruction
    alignment across hosts.
    """
    span = abs(int(local_cycle) - int(model["l0"]))
    sigma = model.get("sigma_alpha_ppm")
    rate_cycles = 0.0 if sigma is None else span * float(sigma) * 1e-6
    residual = abs(int(model["residual_cycles"])) if model.get("residual_cycles") is not None else 0
    anchor = int(model["anchor_error_cycles"])
    total = math.ceil(anchor + rate_cycles + residual + ROUNDING_CYCLES)
    return dict(
        anchor_cycles=anchor,
        rate_cycles=rate_cycles,
        residual_cycles=residual,
        rounding_cycles=ROUNDING_CYCLES,
        total_cycles=total,
        total_ns=total * 1_000_000_000 // CYCLES_PER_SECOND,
        rate_resolved=bool(model.get("rate_resolved")),
    )
