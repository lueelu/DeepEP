# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""Pinned BF16 mixed-tolerance policy; FP64 is used only for error statistics."""

import numpy as np

RTOL = ATOL = 2**-7
REQUIRED_RATIO = 0.99
ABS_FLOOR = 0.1
REVISION = "81d8b019af1b22a35bcb82a48f28d0f3ab6265ed"


def policy():
    return dict(
        standard="opbase_mixed_tolerance_bf16",
        opbase_revision=REVISION,
        rtol=RTOL,
        atol=ATOL,
        required_matched_ratio=REQUIRED_RATIO,
        hard_limit_rule="per_element_max(0.1,32*ULP_BF16(golden))",
        hard_limit_interpretation="project interpretation of the standard's 'or'",
        golden_rounding="none",
        nonfinite_policy="reject; finite-input suite",
    )


class Comparison:
    def __init__(self):
        self.elements = self.matches = self.violations = 0
        self.finite, self.max_abs = True, 0.0
        self.error_squared = self.reference_squared = 0.0
        self.worst = []

    def update(self, actual, expected, row):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError("reference block shape mismatch")
        error = np.abs(a - b)
        self.elements += error.size
        self.matches += int(np.sum(error <= ATOL + RTOL * np.abs(b)))
        exponent = np.maximum(np.floor(np.log2(np.maximum(np.abs(b), 2.0**-126))), -126)
        self.violations += int(np.sum(~(error <= np.maximum(ABS_FLOOR, 32 * np.exp2(exponent - 7)))))
        finite = np.isfinite(a).all() and np.isfinite(b).all()
        self.finite &= bool(finite)
        if finite:
            self.max_abs = max(self.max_abs, float(error.max(initial=0)))
            self.error_squared += float(np.sum(error * error))
            self.reference_squared += float(np.sum(b * b))
        # Bounded failure diagnostics; no full-output sort.
        if error.size and (not finite or np.any(error > ATOL + RTOL * np.abs(b))):
            scores = np.nan_to_num(error.reshape(-1), nan=np.inf)
            indices = np.argpartition(scores, -min(20, scores.size))[-20:]
            for i in indices:
                self.worst.append((float(scores[i]), row + int(i) // a.shape[1], int(i) % a.shape[1]))
            self.worst = sorted(self.worst, reverse=True)[:20]

    def report(self):
        ratio = self.matches / self.elements if self.elements else 1.0
        return dict(
            passed=self.finite and ratio >= REQUIRED_RATIO and not self.violations,
            finite=self.finite,
            matched_ratio=ratio,
            hard_limit_violations=self.violations,
            max_abs=self.max_abs if self.finite else None,
            relative_l2=(self.error_squared**0.5 / max(self.reference_squared**0.5, 1e-300)) if self.finite else None,
            checked_elements=self.elements,
            **policy(),
            top_errors=[
                dict(abs_error=err if np.isfinite(err) else None, index=[row, col]) for err, row, col in self.worst
            ],
        )
