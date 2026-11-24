# Copyright (c) 2026, Lu Lu

"""Shared affine device-clock model for the low-latency clock-alignment protocol.

The device counter runs at a rank-specific rate. Relative to rank 0's counter,
rank ``r`` is

    R_r(L) = R0_r + alpha_r * (L - L0_r)

where ``L`` is rank ``r``'s local counter, ``R`` is the same instant on rank
0's counter, and ``alpha_r`` is the ratio of the reference rate to the local
rate. ``(L0_r, R0_r)`` is one measured anchor. A single offset cannot represent
this over a graph that spans seconds: applying ``offset`` again every iteration
re-introduces a growing error, which is why a ``rate_correction=0`` estimator
drifts by microseconds inside a 100-call graph.

Both directions come from the same model:

    R(L) = R0 + alpha * (L - L0)                 forward, for reporting
    L(T) = L0 + ceil((T - R0) / alpha)           inverse, for deadlines

The inverse rounds up, so a deadline never releases before the estimated model;
it cannot remove the model error itself. Callers must publish the error budget
from :func:`error_budget` and label the model domain rather than claiming an
absolute physical alignment.

Every absolute counter stays an integer and every conversion uses exact
rational arithmetic, so large anchors are never floated. Only display values
(ppm) are floats.

Calibration is split into training windows and one held-out window. The
holdout never contributes to the anchor, the slope, the intercept or the slope
bound; it is only used to check the fitted model at a future time.
"""

import math
import time
from fractions import Fraction

CYCLES_PER_SECOND = 1_000_000_000
SAMPLE_WORDS = 8
MODEL_SCHEMA = "deepep_clock_model.v3"

# A relative frequency difference beyond this is a broken exchange, not drift.
MAX_RATE_PPM = 500.0
# Window history kept for reporting; re-anchoring does not grow without bound.
MAX_MODEL_WINDOWS = 8
# Deadline inversion rounds up by less than one cycle; forward rounds by half.
ROUNDING_CYCLES = 1
UNCERTAINTY_METHOD = "rtt_bound_v1"

REQUIRED_MODEL_FIELDS = (
    "schema", "rank", "reference_rank", "state",
    "l0", "r0", "alpha_num", "alpha_den",
    "anchor_error_cycles", "alpha_error_num", "alpha_error_den", "uncertainty_method",
    "rate_resolved", "rate_clamped", "residual_cycles", "residual_validated",
    "holdout_window", "alpha_calibrated_wall",
)


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


def rate_bound(fit_windows, anchor, sxx):
    """Conservative RTT bound on ``alpha``.

    Each window's reference coordinate is only known to within its own RTT
    error, and the anchor within ``e_a``. The slope is a linear combination of
    those differences, so

        B_alpha = sum_i |d_i| * (e_i + e_a) / Sxx,   d_i = L_i - L_a.

    This is a worst-case bound, not a statistical deviation: perfect synthetic
    data still carries it, because the measurement error is what limits the
    projection. ``None`` means the windows cannot resolve a slope at all.
    """
    if sxx <= 0:
        return None
    anchor_error = int(anchor["error_cycles"])
    numerator = 0
    for window in fit_windows:
        span = abs(int(window["l"]) - int(anchor["l"]))
        numerator += span * (int(window["error_cycles"]) + anchor_error)
    return Fraction(numerator, sxx)


def fit_model(windows, rank):
    """Fit one rank's affine model from ordered calibration windows.

    Three or more windows reserve the last one as an independent holdout: the
    anchor, slope, intercept and slope bound come from the training windows
    only, and the holdout is used once to check a future-time prediction.
    """
    windows = list(windows)
    if not windows:
        raise ValueError("Clock model needs at least one calibration window")
    holdout = windows[-1] if len(windows) >= 3 else None
    fit_windows = windows[:-1] if holdout is not None else windows
    anchor = min(fit_windows, key=lambda window: (int(window["error_cycles"]), int(window["window"])))
    l0, r0 = int(anchor["l"]), int(anchor["r"])
    anchor_error = int(anchor["error_cycles"])
    now = time.time()

    if int(rank) == 0:
        # Rank 0 is the reference; its model is the identity by definition, so
        # there is nothing to validate against another clock.
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
            alpha_error_num=0,
            alpha_error_den=1,
            alpha_error_bound_ppm=0.0,
            sigma_alpha_ppm=0.0,
            uncertainty_method=UNCERTAINTY_METHOD,
            rate_resolved=True,
            rate_clamped=False,
            residual_cycles=None,
            residual_validated=True,
            holdout_window=None,
            holdout_limit_cycles=None,
            fit_windows=len(fit_windows),
            valid_samples=sum(int(w["valid_samples"]) for w in windows),
            rejected_samples=sum(int(w["rejected_samples"]) for w in windows),
            sample_sequence=int(anchor["sequence"]),
            windows=_normalized_windows(windows),
            calibrated_wall=now,
            alpha_calibrated_wall=now,
            reanchors=0,
        )

    sxx = 0
    sxy = 0
    for window in fit_windows:
        dl = int(window["l"]) - l0
        dr = int(window["r"]) - r0
        sxx += dl * dl
        sxy += dl * dr

    clamped = False
    if sxx == 0:
        # One effective anchor: only the offset is measurable, so the rate stays
        # unresolved instead of being reported as a zero-drift model.
        alpha = Fraction(1)
        bound = None
    else:
        alpha = Fraction(sxy, sxx)
        if alpha <= 0 or abs(float(alpha) - 1.0) * 1e6 > MAX_RATE_PPM:
            alpha = Fraction(1)
            clamped = True
        bound = rate_bound(fit_windows, anchor, sxx)
    if clamped:
        # A clamped rate cannot carry a meaningful projection bound.
        bound = None
    rate_resolved = bound is not None and not clamped

    residual = None
    limit = None
    validated = False
    if holdout is not None and bound is not None:
        predicted = Fraction(r0) + alpha * (int(holdout["l"]) - l0)
        residual = _round_half_up(predicted) - int(holdout["r"])
        # The threshold excludes the residual being tested: anchor error, the
        # slope bound projected over the holdout span, the holdout's own
        # measurement error and the rounding term.
        limit = (Fraction(anchor_error)
                 + abs(int(holdout["l"]) - l0) * bound
                 + int(holdout["error_cycles"])
                 + ROUNDING_CYCLES)
        validated = Fraction(abs(residual)) <= limit
    elif holdout is not None:
        residual = None

    rate_correction = float(alpha) - 1.0
    bound_ppm = None if bound is None else float(bound) * 1e6
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
        alpha_error_num=None if bound is None else bound.numerator,
        alpha_error_den=None if bound is None else bound.denominator,
        alpha_error_bound_ppm=bound_ppm,
        # Compatibility field: the conservative RTT bound, not a statistical
        # standard deviation.
        sigma_alpha_ppm=bound_ppm,
        uncertainty_method=UNCERTAINTY_METHOD,
        rate_resolved=rate_resolved,
        rate_clamped=clamped,
        residual_cycles=residual,
        residual_validated=bool(validated),
        holdout_window=None if holdout is None else int(holdout["window"]),
        holdout_limit_cycles=None if limit is None else math.ceil(limit),
        fit_windows=len(fit_windows),
        valid_samples=sum(int(w["valid_samples"]) for w in windows),
        rejected_samples=sum(int(w["rejected_samples"]) for w in windows),
        sample_sequence=int(anchor["sequence"]),
        windows=_normalized_windows(windows),
        calibrated_wall=now,
        alpha_calibrated_wall=now,
        reanchors=0,
    )


def validate_model(model, expected_rank=None, require_holdout=True):
    """Reject a model that must not be used for a formal deadline table.

    This is shared by the native ST and the AOT/JIT benchmark: one set of
    admission rules, no per-caller copies.
    """
    if not isinstance(model, dict):
        raise ValueError(f"Clock model must be a dict, got {type(model).__name__}")
    missing = [name for name in REQUIRED_MODEL_FIELDS if name not in model]
    if missing:
        rank = model.get("rank")
        raise ValueError(f"Clock model rank {rank} is missing {missing}")
    rank = int(model["rank"])
    if expected_rank is not None and rank != int(expected_rank):
        raise ValueError(f"Clock model rank {rank} was expected to be {int(expected_rank)}")
    if model["schema"] != MODEL_SCHEMA:
        raise ValueError(f"Clock model rank {rank} has schema {model['schema']}, need {MODEL_SCHEMA}")
    if int(model["reference_rank"]) != 0:
        raise ValueError(f"Clock model rank {rank} does not reference rank 0")
    if model["state"] != "ready":
        raise ValueError(f"Clock model rank {rank} is in state {model['state']}")
    if model.get("uncertainty_method") != UNCERTAINTY_METHOD:
        raise ValueError(
            f"Clock model rank {rank} uses {model.get('uncertainty_method')}, need {UNCERTAINTY_METHOD}")
    numerator, denominator = int(model["alpha_num"]), int(model["alpha_den"])
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"Clock model rank {rank} has an invalid rate {numerator}/{denominator}")
    error_numerator = int(model["alpha_error_num"]) if model["alpha_error_num"] is not None else None
    error_denominator = int(model["alpha_error_den"]) if model["alpha_error_den"] is not None else None
    if model["rate_clamped"]:
        raise ValueError(f"Clock model rank {rank} was clamped; recalibrate")
    if rank == 0:
        if numerator != denominator:
            raise ValueError("The reference rank model must be the identity")
        if int(model["anchor_error_cycles"]) != 0 or error_numerator != 0:
            raise ValueError("The reference rank model must not carry an error bound")
        return
    if not model["rate_resolved"]:
        raise ValueError(f"Clock model rank {rank} did not resolve a rate; recalibrate with a wider span")
    if error_numerator is None or error_denominator is None or error_denominator <= 0:
        raise ValueError(f"Clock model rank {rank} has an invalid rate bound")
    if error_numerator < 0:
        raise ValueError(f"Clock model rank {rank} has a negative rate bound")
    ppm = model.get("alpha_error_bound_ppm")
    if ppm is None or not math.isfinite(float(ppm)):
        raise ValueError(f"Clock model rank {rank} has a non-finite rate bound")
    if require_holdout:
        if model["holdout_window"] is None:
            raise ValueError(f"Clock model rank {rank} has no held-out window; recalibrate")
        if not model["residual_validated"]:
            raise ValueError(f"Clock model rank {rank} failed its held-out residual check")


def validate_models(models, world, require_holdout=True):
    """Validate a gathered model list: coverage, order, then each model."""
    world = int(world)
    if not isinstance(models, list) or len(models) != world:
        raise ValueError(f"Clock models must cover ranks 0..{world - 1}")
    ranks = [model.get("rank") if isinstance(model, dict) else None for model in models]
    if ranks != list(range(world)):
        raise ValueError(f"Clock models do not follow EP rank order: {ranks}")
    for rank, model in enumerate(models):
        validate_model(model, expected_rank=rank, require_holdout=require_holdout)


def reanchor_residual(model, window):
    """Prediction error of an old model at a new anchor window."""
    predicted = Fraction(int(model["r0"])) + _alpha(model) * (int(window["l"]) - int(model["l0"]))
    return _round_half_up(predicted) - int(window["r"])


def reanchor_limit(model, window):
    """Admitted deviation of a new anchor from the old model's prediction."""
    return error_budget(model, int(window["l"]))["total_cycles"] + int(window["error_cycles"])


def reanchor(model, window):
    """Move the anchor of an existing model to a new window.

    The fitted rate, its bound and the rate's calibration time stay; only the
    intercept and the anchor error move, so a replay does not need a new
    multi-window calibration while the model is younger than its admitted age.
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

    E(L) = E_anchor + |L - L_a| * B_alpha + E_residual + E_round, evaluated with
    exact rationals and rounded up once. A model that cannot supply a rate
    bound (unresolved, clamped, or built by an older schema) raises instead of
    reporting a budget, so no formal path can silently publish a finite number.

    The result is a model-domain bound; it does not include the asymmetric
    transport path, so it must not be advertised as physical first-instruction
    alignment across hosts.
    """
    if model.get("uncertainty_method") != UNCERTAINTY_METHOD:
        raise ValueError(
            f"Clock model rank {model.get('rank')} has no {UNCERTAINTY_METHOD} bound")
    if model.get("rate_clamped"):
        raise ValueError(f"Clock model rank {model.get('rank')} was clamped; recalibrate")
    if not model.get("rate_resolved"):
        raise ValueError(f"Clock model rank {model.get('rank')} did not resolve a rate")
    try:
        bound = Fraction(int(model["alpha_error_num"]), int(model["alpha_error_den"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Clock model rank {model.get('rank')} has no usable rate bound") from exc
    if bound < 0 or bound.denominator <= 0:
        raise ValueError(f"Clock model rank {model.get('rank')} has an invalid rate bound")
    ppm = model.get("alpha_error_bound_ppm")
    if ppm is not None and not math.isfinite(float(ppm)):
        raise ValueError(f"Clock model rank {model.get('rank')} has a non-finite rate bound")
    span = abs(int(local_cycle) - int(model["l0"]))
    rate_cycles = span * bound
    residual = abs(int(model["residual_cycles"])) if model.get("residual_cycles") is not None else 0
    anchor = int(model["anchor_error_cycles"])
    total = math.ceil(Fraction(anchor) + rate_cycles + residual + ROUNDING_CYCLES)
    return dict(
        anchor_cycles=anchor,
        rate_cycles=math.ceil(rate_cycles),
        residual_cycles=residual,
        rounding_cycles=ROUNDING_CYCLES,
        total_cycles=total,
        total_ns=total * 1_000_000_000 // CYCLES_PER_SECOND,
        rate_bound_ppm=float(bound) * 1e6,
        rate_resolved=True,
        uncertainty_method=model.get("uncertainty_method"),
    )
