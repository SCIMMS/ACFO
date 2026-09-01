"""Frozen qR/Bessel harmonic plans for adjoint-paired ACFO retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .axisymmetric_operator import PreparedAxisymmetricOperator
from .solvers import estimate_bessel_cutoff

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class QrHarmonicCutoffPlan:
    """Immutable per-orbit qR cutoff and its actual FFT storage mask."""

    q_radius: "NDArray[np.float64]"
    kernel_cutoff: "NDArray[np.int64]"
    padded_cutoff: "NDArray[np.int64]"
    compute_cutoff: "NDArray[np.int64]"
    angular_modes: "NDArray[np.int64]"
    mode_mask: "NDArray[np.bool_]"
    nyquist_limited: "NDArray[np.bool_]"
    radius: float
    tolerance: float
    mode_padding: int
    cutoff_bin_size: int
    symmetric_nyquist: int

    def __post_init__(self) -> None:
        arrays = (
            self.q_radius,
            self.kernel_cutoff,
            self.padded_cutoff,
            self.compute_cutoff,
            self.angular_modes,
            self.mode_mask,
            self.nyquist_limited,
        )
        for array in arrays:
            array.setflags(write=False)

    @property
    def n_orbit(self) -> int:
        return int(self.q_radius.size)

    @property
    def n_phi(self) -> int:
        return int(self.angular_modes.size)

    def to_dict(self) -> dict[str, object]:
        return {
            "radius": self.radius,
            "tolerance": self.tolerance,
            "mode_padding": self.mode_padding,
            "cutoff_bin_size": self.cutoff_bin_size,
            "symmetric_nyquist": self.symmetric_nyquist,
            "q_radius": self.q_radius.tolist(),
            "kernel_cutoff": self.kernel_cutoff.tolist(),
            "padded_cutoff": self.padded_cutoff.tolist(),
            "compute_cutoff": self.compute_cutoff.tolist(),
            "nyquist_limited": self.nyquist_limited.tolist(),
            "non_decreasing_with_q_radius": True,
        }


def build_qr_harmonic_cutoff_plan(
    q_perp: "ArrayLike",
    *,
    radius: float,
    tolerance: float,
    n_phi: int,
    mode_padding: int = 0,
    cutoff_bin_size: int = 1,
) -> QrHarmonicCutoffPlan:
    """Build a label-free, non-decreasing qR/Bessel cutoff plan.

    ``mode_padding`` is deliberately separate from the tolerance-based kernel
    cutoff.  The returned mask excludes the unpaired even-grid Nyquist mode,
    so a symmetric band ``-H,...,+H`` is always represented exactly.
    """

    q_perp_array = np.asarray(q_perp, dtype=np.float64)
    radius = float(radius)
    tolerance = float(tolerance)
    n_phi = int(n_phi)
    mode_padding = int(mode_padding)
    cutoff_bin_size = int(cutoff_bin_size)
    if q_perp_array.ndim != 1 or q_perp_array.size == 0:
        raise ValueError("q_perp must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(q_perp_array)):
        raise ValueError("q_perp must contain only finite values")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    if n_phi < 3:
        raise ValueError("n_phi must be at least three")
    if mode_padding < 0:
        raise ValueError("mode_padding must be non-negative")
    if cutoff_bin_size <= 0:
        raise ValueError("cutoff_bin_size must be positive")

    q_radius = np.abs(q_perp_array) * radius
    raw = np.asarray(
        [estimate_bessel_cutoff(value, tol=tolerance) for value in q_radius],
        dtype=np.int64,
    )
    order = np.argsort(q_radius, kind="stable")
    kernel_cutoff = raw.copy()
    kernel_cutoff[order] = np.maximum.accumulate(raw[order])
    padded = kernel_cutoff + mode_padding
    rounded = (
        (padded + cutoff_bin_size - 1) // cutoff_bin_size
    ) * cutoff_bin_size
    symmetric_nyquist = (n_phi - 1) // 2
    compute = np.minimum(rounded, symmetric_nyquist).astype(np.int64)
    if np.any(np.diff(compute[order]) < 0):
        raise RuntimeError("qR cutoff plan is not non-decreasing")
    angular_modes = np.rint(np.fft.fftfreq(n_phi) * n_phi).astype(np.int64)
    mode_mask = np.abs(angular_modes)[None, :] <= compute[:, None]
    if n_phi % 2 == 0 and np.any(mode_mask[:, n_phi // 2]):
        raise RuntimeError("the unpaired even-grid Nyquist mode must be excluded")
    return QrHarmonicCutoffPlan(
        q_radius=np.array(q_radius, dtype=np.float64, copy=True),
        kernel_cutoff=np.array(kernel_cutoff, dtype=np.int64, copy=True),
        padded_cutoff=np.array(padded, dtype=np.int64, copy=True),
        compute_cutoff=np.array(compute, dtype=np.int64, copy=True),
        angular_modes=np.array(angular_modes, dtype=np.int64, copy=True),
        mode_mask=np.array(mode_mask, dtype=bool, copy=True),
        nyquist_limited=np.array(padded > symmetric_nyquist, dtype=bool, copy=True),
        radius=radius,
        tolerance=tolerance,
        mode_padding=mode_padding,
        cutoff_bin_size=cutoff_bin_size,
        symmetric_nyquist=symmetric_nyquist,
    )


class PreparedCutoffAxisymmetricOperator:
    """Forward-adjoint pair with one immutable q-dependent harmonic mask."""

    def __init__(
        self,
        base: PreparedAxisymmetricOperator,
        plan: QrHarmonicCutoffPlan,
        *,
        angles: "ArrayLike | None" = None,
        phase_offsets: "ArrayLike | None" = None,
    ) -> None:
        if plan.n_orbit != base.manifold.n_u or plan.n_phi != base.phi.size:
            raise ValueError("cutoff plan shape does not match the prepared operator")
        if not np.array_equal(plan.angular_modes, base.angular_modes):
            raise ValueError("cutoff plan and prepared operator use different mode storage")
        self.base = base
        self.plan = plan
        self.object_shape = base.object_shape
        if angles is None:
            self.angles = None
            self.data_shape = base.data_shape
        else:
            angle_array = np.asarray(angles, dtype=np.float64)
            if (
                angle_array.ndim != 1
                or angle_array.size == 0
                or not np.all(np.isfinite(angle_array))
            ):
                raise ValueError("angles must be a non-empty finite vector")
            self.angles = np.array(angle_array, dtype=np.float64, copy=True)
            self.angles.setflags(write=False)
            self.data_shape = (base.manifold.n_u, self.angles.size)
        if phase_offsets is None:
            self.phase_offsets = None
        else:
            offsets = np.asarray(phase_offsets, dtype=np.float64)
            if offsets.shape != (base.manifold.n_u,) or not np.all(
                np.isfinite(offsets)
            ):
                raise ValueError(
                    f"phase_offsets must be finite with shape ({base.manifold.n_u},)"
                )
            self.phase_offsets = np.array(offsets, dtype=np.float64, copy=True)
            self.phase_offsets.setflags(write=False)

    def forward(self, object_values: "ArrayLike") -> np.ndarray:
        if self.angles is None:
            return self.base.forward_mode_mask(
                object_values,
                self.plan.mode_mask,
                phase_offsets=self.phase_offsets,
            )
        return self.base.forward_at_angles_mode_mask(
            object_values,
            self.angles,
            self.plan.mode_mask,
            phase_offsets=self.phase_offsets,
        )

    def adjoint_euclidean(self, data_values: "ArrayLike") -> np.ndarray:
        if self.angles is None:
            return self.base.adjoint_mode_mask_euclidean(
                data_values,
                self.plan.mode_mask,
                phase_offsets=self.phase_offsets,
            )
        return self.base.adjoint_at_angles_mode_mask_euclidean(
            data_values,
            self.angles,
            self.plan.mode_mask,
            phase_offsets=self.phase_offsets,
        )
