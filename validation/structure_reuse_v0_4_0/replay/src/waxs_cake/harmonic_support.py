"""Diagnostics for choosing and auditing angular-harmonic support.

The helpers in this module deliberately separate three ideas:

* a geometry-based cutoff, such as a Bessel-tail estimate;
* source angular content measured on a uniform azimuth grid; and
* an a posteriori output-tail or nested-application diagnostic.

None of the diagnostics is advertised as a general analytic error bound.
They provide a reproducible numerical contract for freezing a cutoff before
headline timing or reconstruction runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += int(ndim)
    if axis < 0 or axis >= int(ndim):
        raise np.AxisError(axis, ndim=ndim)
    return axis


def signed_angular_modes(n_phi: int) -> np.ndarray:
    """Return integer FFT modes in NumPy storage order."""

    n_phi = int(n_phi)
    if n_phi < 2:
        raise ValueError("n_phi must be at least 2")
    return np.rint(np.fft.fftfreq(n_phi) * n_phi).astype(np.int64)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return a finite relative L2 error, including the zero-reference case."""

    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    denominator = float(np.linalg.norm(reference.ravel()))
    numerator = float(np.linalg.norm((candidate - reference).ravel()))
    return numerator if denominator == 0.0 else numerator / denominator


def harmonic_tail_fraction(
    fourier_values: np.ndarray,
    max_h: int,
    *,
    axis: int = -1,
) -> float:
    """Return the relative L2 energy beyond the symmetric cutoff ``max_h``.

    ``fourier_values`` may contain arbitrary leading dimensions. Parseval
    scaling cancels in the returned ratio.
    """

    values = np.asarray(fourier_values)
    axis = _normalize_axis(axis, values.ndim)
    n_phi = values.shape[axis]
    max_h = int(max_h)
    if max_h < 0 or max_h > n_phi // 2:
        raise ValueError("max_h must be in [0, n_phi // 2]")
    modes = signed_angular_modes(n_phi)
    omitted = np.abs(modes) > max_h
    total = float(np.sum(np.abs(values) ** 2))
    if total == 0.0:
        return 0.0
    tail = float(np.sum(np.abs(np.compress(omitted, values, axis=axis)) ** 2))
    return float(np.sqrt(tail / total))


def minimum_symmetric_cutoff(
    fourier_values: np.ndarray,
    relative_l2_tolerance: float,
    *,
    axis: int = -1,
) -> int:
    """Return the smallest global ``H`` whose measured Fourier tail passes."""

    tolerance = float(relative_l2_tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("relative_l2_tolerance must lie in (0, 1)")
    values = np.asarray(fourier_values)
    axis = _normalize_axis(axis, values.ndim)
    for max_h in range(values.shape[axis] // 2 + 1):
        if harmonic_tail_fraction(values, max_h, axis=axis) <= tolerance:
            return max_h
    return values.shape[axis] // 2


@dataclass(frozen=True)
class HarmonicSupportDecision:
    """Serializable source-aware global-support decision."""

    n_phi: int
    geometry_cutoff: int
    source_required_cutoff: int
    resolved_cutoff: int
    source_tail_tolerance: float
    source_tail_at_geometry_cutoff: float
    source_tail_at_resolved_cutoff: float
    nyquist_limited: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class CoupledHarmonicSupportDecision:
    """Serializable source–kernel energy-proxy decision."""

    n_phi: int
    required_cutoff: int
    relative_tail_tolerance: float
    tail_at_required_cutoff: float
    nyquist_limited: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def _radial_mode_energy(
    values: np.ndarray,
    *,
    radial_axis: int,
    mode_axis: int,
) -> np.ndarray:
    array = np.asarray(values)
    radial_axis = _normalize_axis(radial_axis, array.ndim)
    mode_axis = _normalize_axis(mode_axis, array.ndim)
    if radial_axis == mode_axis:
        raise ValueError("radial_axis and mode_axis must differ")
    moved = np.moveaxis(array, (radial_axis, mode_axis), (-2, -1))
    reduction_axes = tuple(range(moved.ndim - 2))
    return np.sum(np.abs(moved) ** 2, axis=reduction_axes)


def resolve_coupled_harmonic_cutoff(
    source_fourier: np.ndarray,
    kernel_harmonics: np.ndarray,
    *,
    source_radial_axis: int,
    source_mode_axis: int,
    kernel_radial_axis: int,
    kernel_mode_axis: int,
    relative_tail_tolerance: float,
) -> CoupledHarmonicSupportDecision:
    """Choose ``H`` from a source-energy times kernel-energy proxy.

    The proxy preserves radial overlap and sums all other dimensions:

    ``E_h = sum_r E_source[r,h] * E_kernel[r,h]``.

    It is tighter than source-only support when high source modes coincide
    with negligible Bessel modes. It remains an a priori heuristic rather
    than a rigorous error bound, so a nested or high-support output check is
    still required.
    """

    tolerance = float(relative_tail_tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("relative_tail_tolerance must lie in (0, 1)")
    source_energy = _radial_mode_energy(
        source_fourier,
        radial_axis=source_radial_axis,
        mode_axis=source_mode_axis,
    )
    kernel_energy = _radial_mode_energy(
        kernel_harmonics,
        radial_axis=kernel_radial_axis,
        mode_axis=kernel_mode_axis,
    )
    if source_energy.shape != kernel_energy.shape:
        raise ValueError(
            "source and kernel radial-mode energy arrays must have equal shape"
        )
    n_phi = source_energy.shape[1]
    modes = signed_angular_modes(n_phi)
    coupled = np.sum(source_energy * kernel_energy, axis=0)
    total = float(np.sum(coupled))
    if total == 0.0:
        return CoupledHarmonicSupportDecision(
            n_phi=n_phi,
            required_cutoff=0,
            relative_tail_tolerance=tolerance,
            tail_at_required_cutoff=0.0,
            nyquist_limited=False,
        )
    required = n_phi // 2
    tail_at_required = 0.0
    for max_h in range(n_phi // 2 + 1):
        tail = float(
            np.sqrt(np.sum(coupled[np.abs(modes) > max_h]) / total)
        )
        if tail <= tolerance:
            required = max_h
            tail_at_required = tail
            break
    return CoupledHarmonicSupportDecision(
        n_phi=n_phi,
        required_cutoff=required,
        relative_tail_tolerance=tolerance,
        tail_at_required_cutoff=tail_at_required,
        nyquist_limited=bool(required >= n_phi // 2),
    )


def resolve_source_aware_cutoff(
    source_values: np.ndarray,
    geometry_cutoff: int,
    *,
    relative_source_tail_tolerance: float,
    axis: int = -1,
    values_are_fourier: bool = False,
) -> HarmonicSupportDecision:
    """Combine geometry support with measured source angular support.

    This is intentionally conservative: it selects one contiguous global
    support ``|h| <= H``. Sparse-mode implementations may retain only the
    required source modes instead. The source criterion is a preflight
    diagnostic, not a bound on the final contracted output.
    """

    source = np.asarray(source_values)
    axis = _normalize_axis(axis, source.ndim)
    n_phi = source.shape[axis]
    geometry_cutoff = int(geometry_cutoff)
    if geometry_cutoff < 0:
        raise ValueError("geometry_cutoff must be non-negative")
    geometry_cutoff = min(geometry_cutoff, n_phi // 2)
    fourier = source if values_are_fourier else np.fft.fft(source, axis=axis)
    source_required = minimum_symmetric_cutoff(
        fourier,
        relative_source_tail_tolerance,
        axis=axis,
    )
    resolved = max(geometry_cutoff, source_required)
    return HarmonicSupportDecision(
        n_phi=n_phi,
        geometry_cutoff=geometry_cutoff,
        source_required_cutoff=source_required,
        resolved_cutoff=resolved,
        source_tail_tolerance=float(relative_source_tail_tolerance),
        source_tail_at_geometry_cutoff=harmonic_tail_fraction(
            fourier,
            geometry_cutoff,
            axis=axis,
        ),
        source_tail_at_resolved_cutoff=harmonic_tail_fraction(
            fourier,
            resolved,
            axis=axis,
        ),
        nyquist_limited=bool(resolved >= n_phi // 2),
    )


def nested_cutoff_diagnostic(
    low_support: np.ndarray,
    next_support: np.ndarray,
    high_support_reference: np.ndarray,
) -> dict[str, float]:
    """Compare one cutoff, its next increment and a frozen high-support run."""

    return {
        "low_vs_next_relative_l2": relative_l2(low_support, next_support),
        "low_vs_reference_relative_l2": relative_l2(
            low_support,
            high_support_reference,
        ),
        "next_vs_reference_relative_l2": relative_l2(
            next_support,
            high_support_reference,
        ),
    }
