"""A posteriori absolute-tail contracts for cylindrical harmonics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy import special


def _series_tail_bound(x: float, first_order: int) -> float:
    """Bound ``sum_{m>=first_order} |J_m(x)|`` by absolute series terms.

    For integer ``m >= 0``, the Bessel series gives

    ``|J_m(x)| <= (x/2)^m / m! * exp(x^2 / (4(m+1)))``.

    The successive upper-bound ratio is at most ``x/(2(m+1))``.  The
    requested ``first_order`` is chosen in the decreasing tail.
    """

    x = abs(float(x))
    first_order = int(first_order)
    if first_order < 0:
        raise ValueError("first_order must be non-negative")
    if x == 0.0:
        return 1.0 if first_order == 0 else 0.0
    ratio = x / (2.0 * (first_order + 1.0))
    if ratio >= 1.0:
        return float("inf")
    log_first = (
        first_order * math.log(x / 2.0)
        - float(special.gammaln(first_order + 1.0))
        + x * x / (4.0 * (first_order + 1.0))
    )
    if log_first < math.log(np.finfo(np.float64).tiny):
        first = 0.0
    elif log_first > math.log(np.finfo(np.float64).max):
        return float("inf")
    else:
        first = math.exp(log_first)
    return first / (1.0 - ratio)


def bessel_two_sided_tail_bound(
    x: float,
    cutoff: int,
    *,
    evaluated_margin: int = 32,
) -> float:
    """Numerically certify ``sum_|m|>H |J_m(x)|`` for ``H=cutoff``.

    Orders through a conservative finite margin are evaluated explicitly.
    The remaining infinite tail is bounded from the absolute Bessel series.
    The result is safe at zeros of individual Bessel functions because no
    adjacent-order ratio is used in the selection decision.
    """

    x = abs(float(x))
    cutoff = int(cutoff)
    evaluated_margin = int(evaluated_margin)
    if cutoff < 0 or evaluated_margin < 8:
        raise ValueError("cutoff must be non-negative and evaluated_margin at least 8")
    if x == 0.0:
        return 0.0
    stop = max(
        cutoff + evaluated_margin,
        int(math.ceil(x + 12.0 * np.cbrt(x + 1.0) + 64.0)),
        int(math.floor(x / 2.0)) + evaluated_margin,
    )
    orders = np.arange(cutoff + 1, stop + 1, dtype=np.int64)
    explicit = float(np.sum(np.abs(special.jv(orders, x)), dtype=np.float64))
    remainder = _series_tail_bound(x, stop + 1)
    safety = 1.0 + 64.0 * np.finfo(np.float64).eps
    return 2.0 * safety * (explicit + remainder)


@dataclass(frozen=True)
class ToleranceDrivenHarmonicPlan:
    """Per-row support selected from an absolute kernel-tail contract."""

    q_radius: np.ndarray
    coefficient_envelope: np.ndarray
    requested_tolerance: np.ndarray
    required_cutoff: np.ndarray
    compute_cutoff: np.ndarray
    certified_tail_bound: np.ndarray
    nested_cutoff: np.ndarray
    nested_tail_bound: np.ndarray
    ratio_diagnostic: np.ndarray
    nyquist_limited: np.ndarray
    passed: np.ndarray
    n_phi: int
    symmetric_nyquist: int
    nested_padding: int

    def __post_init__(self) -> None:
        for value in (
            self.q_radius,
            self.coefficient_envelope,
            self.requested_tolerance,
            self.required_cutoff,
            self.compute_cutoff,
            self.certified_tail_bound,
            self.nested_cutoff,
            self.nested_tail_bound,
            self.ratio_diagnostic,
            self.nyquist_limited,
            self.passed,
        ):
            value.setflags(write=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "q_radius": self.q_radius.tolist(),
            "coefficient_envelope": self.coefficient_envelope.tolist(),
            "requested_tolerance": self.requested_tolerance.tolist(),
            "required_cutoff": self.required_cutoff.tolist(),
            "compute_cutoff": self.compute_cutoff.tolist(),
            "certified_tail_bound": self.certified_tail_bound.tolist(),
            "nested_cutoff": self.nested_cutoff.tolist(),
            "nested_tail_bound": self.nested_tail_bound.tolist(),
            "ratio_diagnostic": self.ratio_diagnostic.tolist(),
            "nyquist_limited": self.nyquist_limited.tolist(),
            "passed": self.passed.tolist(),
            "n_phi": self.n_phi,
            "symmetric_nyquist": self.symmetric_nyquist,
            "nested_padding": self.nested_padding,
            "selection_uses_ratio": False,
        }


def build_tolerance_driven_harmonic_plan(
    q_radius: Any,
    *,
    tolerance: Any,
    n_phi: int,
    coefficient_envelope: Any = 1.0,
    nested_padding: int = 4,
) -> ToleranceDrivenHarmonicPlan:
    """Choose the smallest non-decreasing support satisfying an absolute tail.

    The certified quantity is

    ``coefficient_envelope * sum_|m|>H |J_m(qR)| <= tolerance``.

    This is an operator-norm style bound for convolution with an input whose
    relevant norm is bounded by ``coefficient_envelope``.  A nested larger
    cutoff is returned for an independent difference diagnostic.
    """

    x = np.asarray(q_radius, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("q_radius must be a non-empty finite non-negative vector")
    tol = np.asarray(tolerance, dtype=np.float64)
    if tol.ndim == 0:
        tol = np.full(x.shape, float(tol))
    envelope = np.asarray(coefficient_envelope, dtype=np.float64)
    if envelope.ndim == 0:
        envelope = np.full(x.shape, float(envelope))
    if tol.shape != x.shape or not np.all(np.isfinite(tol)) or np.any(tol <= 0):
        raise ValueError("tolerance must be positive and scalar or match q_radius")
    if envelope.shape != x.shape or not np.all(np.isfinite(envelope)) or np.any(envelope <= 0):
        raise ValueError("coefficient_envelope must be positive and scalar or match q_radius")
    n_phi = int(n_phi)
    nested_padding = int(nested_padding)
    if n_phi < 3 or nested_padding < 1:
        raise ValueError("n_phi must be at least 3 and nested_padding positive")
    nyquist = (n_phi - 1) // 2

    required = np.empty(x.size, dtype=np.int64)
    required_bound = np.empty(x.size, dtype=np.float64)
    for index, value in enumerate(x):
        if value == 0.0:
            required[index] = 0
            required_bound[index] = 0.0
            continue
        maximum_search = int(math.ceil(value + 14.0 * np.cbrt(value + 1.0) + 96.0))
        selected = maximum_search
        selected_bound = float("inf")
        for cutoff in range(maximum_search + 1):
            bound = envelope[index] * bessel_two_sided_tail_bound(value, cutoff)
            if bound <= tol[index]:
                selected = cutoff
                selected_bound = bound
                break
        required[index] = selected
        required_bound[index] = selected_bound

    order = np.argsort(x, kind="stable")
    monotone = required.copy()
    monotone[order] = np.maximum.accumulate(required[order])
    compute = np.minimum(monotone, nyquist).astype(np.int64)
    bounds = np.asarray(
        [envelope[i] * bessel_two_sided_tail_bound(x[i], int(compute[i])) for i in range(x.size)]
    )
    nested = np.minimum(compute + nested_padding, nyquist).astype(np.int64)
    nested_bounds = np.asarray(
        [envelope[i] * bessel_two_sided_tail_bound(x[i], int(nested[i])) for i in range(x.size)]
    )
    ratio = np.full(x.shape, np.nan, dtype=np.float64)
    for index, (value, cutoff) in enumerate(zip(x, compute)):
        denominator = float(special.jv(int(cutoff), value))
        numerator = float(special.jv(int(cutoff) + 1, value))
        scale = max(abs(numerator), 1.0)
        if abs(denominator) > 32.0 * np.finfo(np.float64).eps * scale:
            ratio[index] = abs(numerator / denominator)
    limited = monotone > nyquist
    passed = bounds <= tol
    return ToleranceDrivenHarmonicPlan(
        q_radius=np.array(x, copy=True),
        coefficient_envelope=np.array(envelope, copy=True),
        requested_tolerance=np.array(tol, copy=True),
        required_cutoff=np.array(monotone, copy=True),
        compute_cutoff=np.array(compute, copy=True),
        certified_tail_bound=np.array(bounds, copy=True),
        nested_cutoff=np.array(nested, copy=True),
        nested_tail_bound=np.array(nested_bounds, copy=True),
        ratio_diagnostic=np.array(ratio, copy=True),
        nyquist_limited=np.array(limited, copy=True),
        passed=np.array(passed, copy=True),
        n_phi=n_phi,
        symmetric_nyquist=nyquist,
        nested_padding=nested_padding,
    )


def nested_relative_difference(lower: Any, upper: Any) -> float:
    """Relative L2 difference used beside the analytic tail certificate."""

    low = np.asarray(lower)
    high = np.asarray(upper)
    if low.shape != high.shape or low.size == 0:
        raise ValueError("lower and upper must be non-empty arrays with equal shape")
    denominator = max(float(np.linalg.norm(high.ravel())), np.finfo(np.float64).tiny)
    return float(np.linalg.norm((low - high).ravel()) / denominator)

