"""Finite rigid translations as cylindrical-harmonic convolutions."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import special

from .tolerance_cutoff import build_tolerance_driven_harmonic_plan


class PreparedRigidModalTranslation:
    r"""Translate Fourier data on complete SO(2) target orbits.

    For transverse displacement ``d=(dx,dy)``, angular Fourier coefficients
    obey

    ``A'_m = sum_l (-i)^l J_l(q_perp |d|) exp(-i l phi_d) A_(m-l)``.

    Coefficients use NumPy FFT ordering along the final axis.  The prepared
    Bessel kernel is zero outside a tolerance-driven support, and all three
    execution paths use the same frozen kernel.
    """

    def __init__(
        self,
        q_perp: Any,
        n_phi: int,
        displacement_xy: tuple[float, float],
        *,
        tolerance: float = 1e-12,
        coefficient_envelope: Any = 1.0,
        nested_padding: int = 4,
    ) -> None:
        q = np.asarray(q_perp, dtype=np.float64)
        if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)) or np.any(q < 0):
            raise ValueError("q_perp must be a non-empty finite non-negative vector")
        n_phi = int(n_phi)
        if n_phi < 3:
            raise ValueError("n_phi must be at least 3")
        dx, dy = (float(displacement_xy[0]), float(displacement_xy[1]))
        if not np.isfinite(dx) or not np.isfinite(dy):
            raise ValueError("displacement must be finite")
        distance = float(np.hypot(dx, dy))
        direction = float(np.arctan2(dy, dx)) if distance else 0.0
        qd = q * distance
        plan = build_tolerance_driven_harmonic_plan(
            qd,
            tolerance=tolerance,
            n_phi=n_phi,
            coefficient_envelope=coefficient_envelope,
            nested_padding=nested_padding,
        )
        if not bool(np.all(plan.passed)):
            raise ValueError("angular Nyquist support cannot satisfy the translation tolerance")

        modes = np.rint(np.fft.fftfreq(n_phi) * n_phi).astype(np.int64)
        kernel = np.empty((q.size, n_phi), dtype=np.complex128)
        for row, value in enumerate(qd):
            active = np.abs(modes) <= plan.compute_cutoff[row]
            kernel[row] = 0.0
            selected = modes[active]
            kernel[row, active] = (
                np.power(-1j, selected)
                * special.jv(selected, value)
                * np.exp(-1j * selected * direction)
            )
        angles = 2.0 * np.pi * np.arange(n_phi, dtype=np.float64) / n_phi
        exact_phase = np.exp(-1j * qd[:, None] * np.cos(angles[None, :] - direction))
        nested_kernel = np.empty_like(kernel)
        full_coefficients = np.fft.fft(exact_phase, axis=-1) / n_phi
        for row in range(q.size):
            active = np.abs(modes) <= plan.nested_cutoff[row]
            nested_kernel[row] = 0.0
            nested_kernel[row, active] = full_coefficients[row, active]

        self.q_perp = np.array(q, copy=True)
        self.n_phi = n_phi
        self.displacement_xy = (dx, dy)
        self.distance = distance
        self.direction = direction
        self.modes = modes
        self.plan = plan
        self.kernel = kernel
        self.nested_kernel = nested_kernel
        self.exact_phase = exact_phase

    @property
    def data_shape(self) -> tuple[int, int]:
        return (self.q_perp.size, self.n_phi)

    def _coefficients(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.complex128)
        if array.ndim < 2 or tuple(array.shape[-2:]) != self.data_shape:
            raise ValueError(f"coefficients must end with shape {self.data_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("coefficients must be finite")
        return array

    def apply(
        self,
        coefficients: Any,
        *,
        method: Literal["direct_phase", "banded", "fft"] = "banded",
        nested: bool = False,
    ) -> np.ndarray:
        """Apply the prepared translation using one of three equivalent paths."""

        values = self._coefficients(coefficients)
        if method == "direct_phase":
            samples = np.fft.ifft(values, axis=-1)
            return np.fft.fft(samples * self.exact_phase, axis=-1)
        kernel = self.nested_kernel if nested else self.kernel
        if method == "fft":
            return np.fft.ifft(
                np.fft.fft(values, axis=-1) * np.fft.fft(kernel, axis=-1),
                axis=-1,
            )
        if method != "banded":
            raise ValueError("method must be direct_phase, banded, or fft")
        output = np.zeros_like(values)
        target = np.arange(self.n_phi, dtype=np.int64)
        leading = (1,) * (values.ndim - 2)
        for row in range(self.q_perp.size):
            active_modes = self.modes[np.abs(self.modes) <= self.plan.compute_cutoff[row]]
            row_values = values[..., row, :]
            row_output = output[..., row, :]
            for mode in active_modes:
                source = np.mod(target - mode, self.n_phi)
                coefficient = self.kernel[row, mode % self.n_phi]
                row_output += coefficient * row_values[..., source]
        return output

    def adjoint(
        self,
        coefficients: Any,
        *,
        method: Literal["direct_phase", "banded", "fft"] = "banded",
    ) -> np.ndarray:
        """Apply ``T(d)^*=T(-d)`` with the same support contract."""

        inverse = PreparedRigidModalTranslation(
            self.q_perp,
            self.n_phi,
            (-self.displacement_xy[0], -self.displacement_xy[1]),
            tolerance=float(np.min(self.plan.requested_tolerance)),
            coefficient_envelope=self.plan.coefficient_envelope,
            nested_padding=self.plan.nested_padding,
        )
        return inverse.apply(coefficients, method=method)

    def relative_nested_difference(self, coefficients: Any, *, method: str = "fft") -> float:
        low = self.apply(coefficients, method=method, nested=False)
        high = self.apply(coefficients, method=method, nested=True)
        denominator = max(float(np.linalg.norm(high.ravel())), np.finfo(np.float64).tiny)
        return float(np.linalg.norm((low - high).ravel()) / denominator)

