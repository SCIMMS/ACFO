"""Tabulated central-field orbitals and transition-density Fourier helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_legendre, j0

from .spherical_molecular import (
    spherical_harmonic_indices,
    spherical_harmonic_matrix,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _unit_sphere_quadrature(
    n_theta: int,
    n_phi: int,
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    cosine, polar_weights = np.polynomial.legendre.leggauss(int(n_theta))
    phi = 2.0 * np.pi * np.arange(int(n_phi), dtype=np.float64) / int(n_phi)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    directions = np.stack(
        np.broadcast_arrays(
            sine[:, None] * np.cos(phi)[None, :],
            sine[:, None] * np.sin(phi)[None, :],
            cosine[:, None],
        ),
        axis=-1,
    ).reshape(-1, 3)
    weights = np.broadcast_to(
        polar_weights[:, None] * (2.0 * np.pi / int(n_phi)),
        (int(n_theta), int(n_phi)),
    ).reshape(-1)
    return directions, weights


@dataclass(frozen=True)
class TabulatedRadialOrbital:
    r"""Numerical central-field orbital ``R_l(r) Y_l0(Omega)``."""

    radial_grid: "NDArray[np.float64]"
    radial_values: "NDArray[np.float64]"
    angular_degree: int
    energy: float
    label: str

    @classmethod
    def soft_coulomb(
        cls,
        angular_degree: int,
        *,
        effective_charge: float = 2.0,
        softening: float = 0.15,
        r_max: float = 24.0,
        n_grid: int = 2048,
        radial_state_index: int = 0,
        label: str | None = None,
    ) -> "TabulatedRadialOrbital":
        r"""Solve a finite-difference radial soft-Coulomb Hamiltonian.

        Atomic units are used for

        ``[-1/2 d2/dr2 + l(l+1)/(2r2) - Z/sqrt(r2+a2)] u = E u``.

        The returned radial function is ``R(r)=u(r)/r`` and is normalized by
        ``integral r2 |R(r)|2 dr = 1``.  This is a controlled numerical model,
        not an element-specific self-consistent atomic orbital.
        """

        angular_degree = int(angular_degree)
        radial_state_index = int(radial_state_index)
        n_grid = int(n_grid)
        effective_charge = float(effective_charge)
        softening = float(softening)
        r_max = float(r_max)
        if angular_degree < 0 or radial_state_index < 0:
            raise ValueError("degrees and state indices must be non-negative")
        if n_grid < 64:
            raise ValueError("n_grid must be at least 64")
        if not np.isfinite(effective_charge) or effective_charge <= 0.0:
            raise ValueError("effective_charge must be finite and positive")
        if not np.isfinite(softening) or softening <= 0.0:
            raise ValueError("softening must be finite and positive")
        if not np.isfinite(r_max) or r_max <= 0.0:
            raise ValueError("r_max must be finite and positive")

        spacing = r_max / (n_grid + 1)
        interior_r = spacing * np.arange(1, n_grid + 1, dtype=np.float64)
        diagonal = (
            1.0 / spacing**2
            + angular_degree * (angular_degree + 1.0) / (2.0 * interior_r**2)
            - effective_charge / np.sqrt(interior_r**2 + softening**2)
        )
        off_diagonal = np.full(n_grid - 1, -0.5 / spacing**2, dtype=np.float64)
        eigenvalues, eigenvectors = eigh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(radial_state_index, radial_state_index),
            check_finite=False,
        )
        reduced = np.asarray(eigenvectors[:, 0], dtype=np.float64)
        largest = int(np.argmax(np.abs(reduced)))
        if reduced[largest] < 0.0:
            reduced *= -1.0
        radial_grid = np.concatenate(([0.0], interior_r, [r_max]))
        radial_values = np.zeros(n_grid + 2, dtype=np.float64)
        radial_values[1:-1] = reduced / interior_r
        if angular_degree == 0:
            radial_values[0] = radial_values[1]
        norm = float(
            np.sqrt(simpson(radial_grid**2 * radial_values**2, x=radial_grid))
        )
        radial_values /= norm
        if label is None:
            label = f"soft-Coulomb l={angular_degree} n_r={radial_state_index}"
        for array in (radial_grid, radial_values):
            array.setflags(write=False)
        return cls(
            radial_grid=radial_grid,
            radial_values=radial_values,
            angular_degree=angular_degree,
            energy=float(eigenvalues[0]),
            label=str(label),
        )

    def sample(self, radial_nodes: "ArrayLike") -> "NDArray[np.float64]":
        """Cubic-spline sample the tabulated radial function."""

        nodes = np.asarray(radial_nodes, dtype=np.float64)
        if not np.all(np.isfinite(nodes)) or np.any(nodes < 0.0):
            raise ValueError("radial_nodes must be finite and non-negative")
        tolerance = 32.0 * np.finfo(np.float64).eps * self.radial_grid[-1]
        if np.any(nodes > self.radial_grid[-1] + tolerance):
            raise ValueError("radial_nodes extend beyond the orbital table")
        spline = CubicSpline(
            self.radial_grid,
            self.radial_values,
            bc_type="natural",
            extrapolate=False,
        )
        return np.asarray(spline(np.minimum(nodes, self.radial_grid[-1])), dtype=np.float64)

    @property
    def norm(self) -> float:
        """Return the tabulated radial normalization."""

        return float(
            simpson(
                self.radial_grid**2 * self.radial_values**2,
                x=self.radial_grid,
            )
        )


def axisymmetric_transition_angular_coefficients(
    initial_degree: int,
    final_degree: int,
) -> "NDArray[np.complex128]":
    r"""Expand ``Y_lf,0(Omega) Y_li,0(Omega)`` in spherical harmonics."""

    initial_degree = int(initial_degree)
    final_degree = int(final_degree)
    if initial_degree < 0 or final_degree < 0:
        raise ValueError("orbital degrees must be non-negative")
    total_degree = initial_degree + final_degree
    directions, weights = _unit_sphere_quadrature(
        total_degree + 1,
        2 * total_degree + 1,
    )
    harmonics = spherical_harmonic_matrix(directions, total_degree)
    ell, order = spherical_harmonic_indices(total_degree)
    initial_index = int(
        np.flatnonzero((ell == initial_degree) & (order == 0))[0]
    )
    final_index = int(np.flatnonzero((ell == final_degree) & (order == 0))[0])
    product = harmonics[:, initial_index] * harmonics[:, final_index]
    coefficients = np.sum(
        weights[:, None] * np.conj(harmonics) * product[:, None],
        axis=0,
    )
    scale = max(float(np.max(np.abs(coefficients))), 1.0)
    coefficients[np.abs(coefficients) < 5e-14 * scale] = 0.0
    return np.asarray(coefficients, dtype=np.complex128)


def tabulated_transition_amplitude_coefficients(
    initial: TabulatedRadialOrbital,
    final: TabulatedRadialOrbital,
    *,
    transform,
) -> "NDArray[np.complex128]":
    """Return Fourier spherical coefficients for two tabulated orbitals."""

    total_degree = initial.angular_degree + final.angular_degree
    ell, _ = spherical_harmonic_indices(total_degree)
    angular = axisymmetric_transition_angular_coefficients(
        initial.angular_degree,
        final.angular_degree,
    )
    radial_profile = initial.sample(transform.radial_nodes) * final.sample(
        transform.radial_nodes
    )
    result = np.zeros(
        (transform.q_values.size, angular.size),
        dtype=np.complex128,
    )
    for degree in np.unique(ell[angular != 0.0]):
        selected = ell == degree
        radial = transform.transform_degree(radial_profile, int(degree))
        result[:, selected] = (
            4.0
            * np.pi
            * (1j ** int(degree))
            * radial[:, None]
            * angular[selected][None, :]
        )
    return result


def direct_tabulated_transition_fourier(
    initial: TabulatedRadialOrbital,
    final: TabulatedRadialOrbital,
    q_vectors: "ArrayLike",
    *,
    radial_order: int = 256,
    polar_order: int = 160,
    chunk_size: int = 8,
) -> "NDArray[np.complex128]":
    r"""Independent direct cylindrical-Bessel quadrature.

    The azimuthal integral of the axisymmetric ``m=0`` transition is evaluated
    analytically as ``2 pi J_0(q_perp r sqrt(1-mu^2))``.  The remaining radial
    and polar integrals use Gauss-Legendre quadrature and do not reuse the
    spherical-Bessel or spherical-harmonic transform implementation.
    """

    vectors = np.asarray(q_vectors, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3 or not np.all(np.isfinite(vectors)):
        raise ValueError("q_vectors must be finite and end in dimension three")
    radial_order = int(radial_order)
    polar_order = int(polar_order)
    chunk_size = int(chunk_size)
    if radial_order <= 0 or polar_order <= 0 or chunk_size <= 0:
        raise ValueError("quadrature orders and chunk_size must be positive")
    r_max = min(float(initial.radial_grid[-1]), float(final.radial_grid[-1]))
    canonical_r, canonical_weight = np.polynomial.legendre.leggauss(radial_order)
    radial = 0.5 * r_max * (canonical_r + 1.0)
    radial_weight = 0.5 * r_max * canonical_weight
    cosine, polar_weight = np.polynomial.legendre.leggauss(polar_order)
    angular_profile = (
        np.sqrt((2.0 * initial.angular_degree + 1.0) / (4.0 * np.pi))
        * eval_legendre(initial.angular_degree, cosine)
        * np.sqrt((2.0 * final.angular_degree + 1.0) / (4.0 * np.pi))
        * eval_legendre(final.angular_degree, cosine)
    )
    radial_profile = initial.sample(radial) * final.sample(radial)
    weighted_density = (
        2.0
        * np.pi
        * radial_weight[:, None]
        * radial[:, None] ** 2
        * radial_profile[:, None]
        * polar_weight[None, :]
        * angular_profile[None, :]
    )
    radial_polar = radial[:, None] * cosine[None, :]
    radial_transverse = radial[:, None] * np.sqrt(1.0 - cosine[None, :] ** 2)
    flat_vectors = vectors.reshape(-1, 3)
    result = np.empty(flat_vectors.shape[0], dtype=np.complex128)
    for start in range(0, flat_vectors.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_vectors.shape[0])
        selected = flat_vectors[start:stop]
        q_parallel = selected[:, 2]
        q_transverse = np.linalg.norm(selected[:, :2], axis=-1)
        phase = np.exp(1j * q_parallel[:, None, None] * radial_polar[None, :, :])
        azimuth = j0(q_transverse[:, None, None] * radial_transverse[None, :, :])
        result[start:stop] = np.sum(
            weighted_density[None, :, :] * phase * azimuth,
            axis=(-2, -1),
        )
    return result.reshape(vectors.shape[:-1])
