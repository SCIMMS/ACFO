"""Exact-coordinate cylindrical-harmonic reference evaluators.

This module deliberately preserves per-source ``(R, beta, z)`` coordinates.
It is a correctness/reference path for testing source-discretization schemes,
not yet a production replacement for the binned sparse projection kernels.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import fft as scipy_fft
from scipy import special


def exact_coordinate_harmonic_amplitude(
    coords: np.ndarray,
    q_perp: np.ndarray,
    q_z: np.ndarray,
    phi: np.ndarray,
    *,
    atom_coefficients: np.ndarray | None = None,
    atom_weights: np.ndarray | None = None,
    harmonic_margin: int = 32,
    atom_chunk_size: int = 256,
    bessel_backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a nonuniform source with an exact Jacobi--Anger expansion.

    Parameters
    ----------
    coords
        Cartesian source coordinates with shape ``(n_source, 3)``.
    q_perp, q_z
        Axisymmetric meridional target coordinates with shape ``(n_q,)``.
    phi
        Arbitrary target azimuths. They need not coincide with a source grid.
    atom_coefficients
        Optional per-source, per-q coefficients with shape ``(n_source, n_q)``.
    atom_weights
        Optional per-source complex weights, applied in addition to
        ``atom_coefficients``.

    Returns
    -------
    amplitude, cutoffs
        Complex amplitude with shape ``(n_q, n_phi)`` and the q-local maximum
        absolute harmonic used for each row.

    Notes
    -----
    The identity used is

    ``exp(i x cos(phi-beta)) = sum_h i**h J_h(x) exp(i h phi) exp(-i h beta)``.

    Runtime scales with source count times the retained harmonic count. The
    routine is intended as an exact-coordinate oracle and small-case bridge.
    """

    coords = np.asarray(coords, dtype=np.float64)
    q_perp = np.asarray(q_perp, dtype=np.float64)
    q_z = np.asarray(q_z, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_source, 3)")
    if q_perp.ndim != 1 or q_z.shape != q_perp.shape:
        raise ValueError("q_perp and q_z must be one-dimensional with equal shape")
    if phi.ndim != 1:
        raise ValueError("phi must be one-dimensional")
    if harmonic_margin < 0:
        raise ValueError("harmonic_margin must be non-negative")
    if bessel_backend not in {"auto", "scipy", "cpp_miller", "cpp_fused"}:
        raise ValueError(
            "bessel_backend must be 'auto', 'scipy', 'cpp_miller', or 'cpp_fused'"
        )
    atom_chunk_size = int(atom_chunk_size)
    if atom_chunk_size <= 0:
        raise ValueError("atom_chunk_size must be positive")

    n_source = coords.shape[0]
    n_q = q_perp.size
    if atom_coefficients is None:
        coefficients = np.ones((n_source, n_q), dtype=np.complex128)
    else:
        coefficients = np.asarray(atom_coefficients, dtype=np.complex128)
        if coefficients.shape != (n_source, n_q):
            raise ValueError("atom_coefficients must have shape (n_source, n_q)")
    if atom_weights is not None:
        weights = np.asarray(atom_weights, dtype=np.complex128)
        if weights.shape != (n_source,):
            raise ValueError("atom_weights must have shape (n_source,)")
        coefficients = coefficients * weights[:, None]

    radius = np.hypot(coords[:, 0], coords[:, 1])
    beta = np.mod(np.arctan2(coords[:, 1], coords[:, 0]), 2.0 * np.pi)
    z = coords[:, 2]
    r_max = float(radius.max(initial=0.0))
    cutoffs = np.ceil(np.abs(q_perp) * r_max).astype(np.int64) + int(
        harmonic_margin
    )
    out = np.empty((n_q, phi.size), dtype=np.complex128)

    miller_function = None
    fused_function = None
    if bessel_backend in {"auto", "cpp_miller", "cpp_fused"}:
        try:
            from . import _cpp_solvers
        except ImportError:
            if bessel_backend in {"cpp_miller", "cpp_fused"}:
                raise
        else:
            miller_function = _cpp_solvers.analytic_kernel_hat_modes_miller
            if bessel_backend == "cpp_fused":
                fused_function = _cpp_solvers.exact_beta_harmonic_coefficients_miller

    fused_coefficients = None
    if fused_function is not None:
        fused_coefficients = np.asarray(
            fused_function(
                np.ascontiguousarray(radius),
                np.ascontiguousarray(beta),
                np.ascontiguousarray(z),
                np.ascontiguousarray(coefficients),
                np.ascontiguousarray(q_perp),
                np.ascontiguousarray(q_z),
                np.ascontiguousarray(cutoffs),
                32,
            ),
            dtype=np.complex128,
        )

    for iq in range(n_q):
        cutoff = int(cutoffs[iq])
        modes = np.arange(cutoff + 1, dtype=np.int64)
        if fused_coefficients is not None:
            positive_sum = fused_coefficients[iq, 0, : modes.size]
            negative_sum = fused_coefficients[iq, 1, : modes.size]
        else:
            positive_sum = np.zeros(modes.size, dtype=np.complex128)
            negative_sum = np.zeros(modes.size, dtype=np.complex128)
            for start in range(0, n_source, atom_chunk_size):
                stop = min(start + atom_chunk_size, n_source)
                beta_phase = np.exp(-1j * beta[start:stop, None] * modes[None, :])
                axial = coefficients[start:stop, iq] * np.exp(
                    1j * q_z[iq] * z[start:stop]
                )
                if miller_function is None:
                    x = q_perp[iq] * radius[start:stop, None]
                    kernel = special.jv(modes[None, :], x) * np.power(1j, modes)
                else:
                    n_phi_miller = max(2 * cutoff + 2, 4)
                    kernel = np.asarray(
                        miller_function(
                            np.ascontiguousarray(q_perp[iq : iq + 1]),
                            np.ascontiguousarray(radius[start:stop]),
                            n_phi_miller,
                            cutoff,
                            32,
                        )[0],
                        dtype=np.complex128,
                    ) / n_phi_miller
                positive_sum += np.sum(
                    axial[:, None] * kernel * beta_phase,
                    axis=0,
                )
                negative_sum += np.sum(
                    axial[:, None] * kernel * np.conjugate(beta_phase),
                    axis=0,
                )
        # i^{-m} J_{-m} = i^m J_m, so the positive/negative kernel is equal.
        positive_coeff = positive_sum
        negative_coeff = negative_sum
        positive_basis = np.exp(1j * modes[:, None] * phi[None, :])
        out[iq] = positive_coeff @ positive_basis
        if cutoff:
            out[iq] += negative_coeff[1:] @ np.conjugate(positive_basis[1:])
    return out, cutoffs


def exact_coordinate_harmonic_amplitude_factorized(
    coords: np.ndarray,
    q_perp: np.ndarray,
    q_z: np.ndarray,
    phi: np.ndarray,
    *,
    element_indices: np.ndarray,
    form_factors: np.ndarray,
    atom_weights: np.ndarray | None = None,
    harmonic_margin: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused exact-beta evaluator with element/form-factor factorization.

    Unlike :func:`exact_coordinate_harmonic_amplitude`, this path does not
    materialize an ``(n_source, n_q)`` coefficient matrix.  It is intended for
    many-q workloads where each source coefficient factors into an atom weight
    and an element-specific form factor.
    """

    coords = np.asarray(coords, dtype=np.float64)
    q_perp = np.asarray(q_perp, dtype=np.float64)
    q_z = np.asarray(q_z, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    element_indices = np.asarray(element_indices, dtype=np.int64)
    form_factors = np.asarray(form_factors, dtype=np.complex128)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_source, 3)")
    if q_perp.ndim != 1 or q_z.shape != q_perp.shape:
        raise ValueError("q_perp and q_z must be one-dimensional with equal shape")
    if phi.ndim != 1:
        raise ValueError("phi must be one-dimensional")
    if element_indices.shape != (coords.shape[0],):
        raise ValueError("element_indices must have shape (n_source,)")
    if form_factors.ndim != 2 or form_factors.shape[1] != q_perp.size:
        raise ValueError("form_factors must have shape (n_elements, n_q)")
    if harmonic_margin < 0:
        raise ValueError("harmonic_margin must be non-negative")
    if atom_weights is None:
        weights = np.ones(coords.shape[0], dtype=np.complex128)
    else:
        weights = np.asarray(atom_weights, dtype=np.complex128)
        if weights.shape != (coords.shape[0],):
            raise ValueError("atom_weights must have shape (n_source,)")

    try:
        from . import _cpp_solvers
    except ImportError as exc:
        raise ImportError("factorized exact-beta evaluation requires _cpp_solvers") from exc
    if not hasattr(
        _cpp_solvers, "exact_beta_harmonic_coefficients_factorized_miller"
    ):
        raise RuntimeError(
            "_cpp_solvers must be rebuilt with the factorized exact-beta kernel"
        )

    radius = np.hypot(coords[:, 0], coords[:, 1])
    beta = np.mod(np.arctan2(coords[:, 1], coords[:, 0]), 2.0 * np.pi)
    z = coords[:, 2]
    r_max = float(radius.max(initial=0.0))
    cutoffs = np.ceil(np.abs(q_perp) * r_max).astype(np.int64) + int(
        harmonic_margin
    )
    coefficients = np.asarray(
        _cpp_solvers.exact_beta_harmonic_coefficients_factorized_miller(
            np.ascontiguousarray(radius),
            np.ascontiguousarray(beta),
            np.ascontiguousarray(z),
            np.ascontiguousarray(element_indices),
            np.ascontiguousarray(weights),
            np.ascontiguousarray(form_factors),
            np.ascontiguousarray(q_perp),
            np.ascontiguousarray(q_z),
            np.ascontiguousarray(cutoffs),
            32,
        ),
        dtype=np.complex128,
    )

    out = np.empty((q_perp.size, phi.size), dtype=np.complex128)
    for iq, cutoff_value in enumerate(cutoffs):
        cutoff = int(cutoff_value)
        modes = np.arange(cutoff + 1, dtype=np.int64)
        basis = np.exp(1j * modes[:, None] * phi[None, :])
        out[iq] = coefficients[iq, 0, : cutoff + 1] @ basis
        if cutoff:
            out[iq] += coefficients[iq, 1, 1 : cutoff + 1] @ np.conjugate(
                basis[1:]
            )
    return out, cutoffs


class PreparedExactCoordinateHarmonicPlan:
    """Prepared factorized exact-coordinate evaluator with stage timings.

    Construction performs geometry-only work once. Each :meth:`execute` call
    recomputes source-dependent harmonic coefficients and then synthesizes the
    requested azimuth samples. The existing unprepared function remains the
    conservative reference path used by prior benchmark receipts.
    """

    def __init__(
        self,
        coords: np.ndarray,
        q_perp: np.ndarray,
        q_z: np.ndarray,
        phi: np.ndarray,
        *,
        element_indices: np.ndarray,
        form_factors: np.ndarray,
        harmonic_margin: int = 32,
        prepare_direct_basis: bool = True,
        coefficient_backend: str = "baseline",
    ) -> None:
        setup_start = time.perf_counter()
        coords = np.asarray(coords, dtype=np.float64)
        q_perp = np.asarray(q_perp, dtype=np.float64)
        q_z = np.asarray(q_z, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        element_indices = np.asarray(element_indices, dtype=np.int64)
        form_factors = np.asarray(form_factors, dtype=np.complex128)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (n_source, 3)")
        if q_perp.ndim != 1 or q_z.shape != q_perp.shape:
            raise ValueError("q_perp and q_z must be one-dimensional with equal shape")
        if phi.ndim != 1:
            raise ValueError("phi must be one-dimensional")
        if element_indices.shape != (coords.shape[0],):
            raise ValueError("element_indices must have shape (n_source,)")
        if form_factors.ndim != 2 or form_factors.shape[1] != q_perp.size:
            raise ValueError("form_factors must have shape (n_elements, n_q)")
        if np.any(element_indices < 0) or np.any(
            element_indices >= form_factors.shape[0]
        ):
            raise ValueError("element_indices contains an out-of-range value")
        if harmonic_margin < 0:
            raise ValueError("harmonic_margin must be non-negative")
        if coefficient_backend not in {"baseline", "fused_phase", "cached_phase"}:
            raise ValueError(
                "coefficient_backend must be 'baseline', 'fused_phase', or 'cached_phase'"
            )

        try:
            from . import _cpp_solvers
        except ImportError as exc:
            raise ImportError("prepared exact-beta evaluation requires _cpp_solvers") from exc
        if not hasattr(
            _cpp_solvers, "exact_beta_harmonic_coefficients_factorized_miller"
        ):
            raise RuntimeError(
                "_cpp_solvers must be rebuilt with the factorized exact-beta kernel"
            )

        coefficient_name = {
            "baseline": "exact_beta_harmonic_coefficients_factorized_miller",
            "fused_phase": (
                "exact_beta_harmonic_coefficients_factorized_miller_fused_phase"
            ),
            "cached_phase": (
                "exact_beta_harmonic_coefficients_factorized_miller_cached_phase"
            ),
        }[coefficient_backend]
        if not hasattr(_cpp_solvers, coefficient_name):
            raise RuntimeError(
                f"_cpp_solvers must be rebuilt with {coefficient_name}"
            )
        self._coefficient_function = getattr(_cpp_solvers, coefficient_name)
        self.coefficient_backend = coefficient_backend
        self.radius = np.ascontiguousarray(np.hypot(coords[:, 0], coords[:, 1]))
        self.beta = np.ascontiguousarray(
            np.mod(np.arctan2(coords[:, 1], coords[:, 0]), 2.0 * np.pi)
        )
        self.z = np.ascontiguousarray(coords[:, 2])
        self.q_perp = np.ascontiguousarray(q_perp)
        self.q_z = np.ascontiguousarray(q_z)
        self.phi = np.ascontiguousarray(phi)
        self.element_indices = np.ascontiguousarray(element_indices)
        self.form_factors = np.ascontiguousarray(form_factors)
        self.cutoffs = np.ascontiguousarray(
            np.ceil(np.abs(q_perp) * float(self.radius.max(initial=0.0))).astype(
                np.int64
            )
            + int(harmonic_margin)
        )
        self.max_cutoff = int(self.cutoffs.max(initial=0))
        self.modes = np.arange(self.max_cutoff + 1, dtype=np.int64)
        self._cached_angular_phase = None
        if coefficient_backend == "cached_phase":
            self._cached_angular_phase = np.ascontiguousarray(
                np.exp(
                    1j
                    * (0.5 * np.pi - self.beta[:, None])
                    * self.modes[None, :]
                )
            )
        self._basis = (
            np.exp(1j * self.modes[:, None] * self.phi[None, :])
            if prepare_direct_basis
            else None
        )
        self._uniform_phi = False
        if self.phi.size >= 2:
            expected_step = 2.0 * np.pi / self.phi.size
            self._uniform_phi = bool(
                np.allclose(
                    np.diff(self.phi),
                    expected_step,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        self._fft_supported = bool(
            self._uniform_phi and self.max_cutoff < self.phi.size // 2
        )
        self._fft_positive_phase = np.exp(1j * self.modes * self.phi[0])
        self._default_weights = np.ones(coords.shape[0], dtype=np.complex128)
        self.setup_seconds = time.perf_counter() - setup_start
        self.last_profile: dict[str, float | str] | None = None

    @property
    def fft_supported(self) -> bool:
        return self._fft_supported

    def _synthesize_direct(self, coefficients: np.ndarray) -> np.ndarray:
        if self._basis is None:
            raise ValueError(
                "direct synthesis requires prepare_direct_basis=True at plan construction"
            )
        out = coefficients[:, 0, :] @ self._basis
        if self.max_cutoff:
            out += coefficients[:, 1, 1:] @ np.conjugate(self._basis[1:])
        return out

    def _synthesize_fft(self, coefficients: np.ndarray) -> np.ndarray:
        if not self._fft_supported:
            raise ValueError(
                "FFT synthesis requires a uniform full-circle phi grid and cutoff < Nyquist"
            )
        n_phi = self.phi.size
        spectrum = np.zeros((self.q_perp.size, n_phi), dtype=np.complex128)
        spectrum[:, : self.max_cutoff + 1] = (
            coefficients[:, 0, :] * self._fft_positive_phase[None, :]
        )
        if self.max_cutoff:
            negative_modes = self.modes[1:]
            spectrum[:, n_phi - negative_modes] = (
                coefficients[:, 1, 1:]
                * np.conjugate(self._fft_positive_phase[None, 1:])
            )
        return scipy_fft.ifft(spectrum, axis=1, workers=1) * n_phi

    def execute(
        self,
        *,
        atom_weights: np.ndarray | None = None,
        synthesis_backend: str = "direct",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute the plan and record coefficient/synthesis wall times."""

        if synthesis_backend not in {"direct", "fft", "auto"}:
            raise ValueError("synthesis_backend must be 'direct', 'fft', or 'auto'")
        if synthesis_backend == "auto":
            synthesis_backend = "fft" if self._fft_supported else "direct"
        if atom_weights is None:
            weights = self._default_weights
        else:
            weights = np.asarray(atom_weights, dtype=np.complex128)
            if weights.shape != self._default_weights.shape:
                raise ValueError("atom_weights must have shape (n_source,)")
            weights = np.ascontiguousarray(weights)

        total_start = time.perf_counter()
        coefficient_start = time.perf_counter()
        coefficient_args = [
                self.radius,
                self.beta,
                self.z,
        ]
        if self._cached_angular_phase is not None:
            coefficient_args.append(self._cached_angular_phase)
        coefficient_args.extend(
            [
                self.element_indices,
                weights,
                self.form_factors,
                self.q_perp,
                self.q_z,
                self.cutoffs,
                32,
            ]
        )
        coefficients = np.asarray(
            self._coefficient_function(*coefficient_args),
            dtype=np.complex128,
        )
        coefficient_seconds = time.perf_counter() - coefficient_start
        synthesis_start = time.perf_counter()
        if synthesis_backend == "fft":
            out = self._synthesize_fft(coefficients)
        else:
            out = self._synthesize_direct(coefficients)
        synthesis_seconds = time.perf_counter() - synthesis_start
        total_seconds = time.perf_counter() - total_start
        self.last_profile = {
            "backend": synthesis_backend,
            "coefficient_backend": self.coefficient_backend,
            "coefficient_cache_mib": (
                0.0
                if self._cached_angular_phase is None
                else self._cached_angular_phase.nbytes / (1024.0**2)
            ),
            "coefficient_contraction_seconds": coefficient_seconds,
            "azimuth_synthesis_seconds": synthesis_seconds,
            "total_seconds": total_seconds,
        }
        return out, self.cutoffs.copy()
