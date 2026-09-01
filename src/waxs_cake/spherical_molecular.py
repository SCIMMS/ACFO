"""Spherical-harmonic contractions for molecular scattering on ring stacks.

This module isolates the molecular-frame calculation used by aligned-ensemble
diffraction models. It provides two mathematically equivalent routes to the
intensity harmonics:

``pair_intensity_coefficients``
    Expand every atom-pair phase directly. Its molecular setup cost is
    quadratic in the number of atoms.

``amplitude_coefficients`` + ``PreparedSphericalGauntProduct``
    Expand the atomic density/amplitude first and multiply the two truncated
    spherical series on an exact-enough product quadrature. Its atom-dependent
    setup cost is linear in the number of atoms.

The second route is a pseudospectral evaluation of the Gaunt contraction. It
does not yet include an SO(3) orientation-distribution/Wigner-D layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import sph_harm_y, spherical_jn

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def spherical_harmonic_indices(
    l_max: int,
) -> tuple["NDArray[np.int64]", "NDArray[np.int64]"]:
    """Return degree-major ``(ell, m)`` indices through ``l_max``."""

    l_max = int(l_max)
    if l_max < 0:
        raise ValueError("l_max must be non-negative")
    degrees = np.arange(l_max + 1, dtype=np.int64)
    ell = np.repeat(degrees, 2 * degrees + 1)
    order = np.concatenate(
        [np.arange(-degree, degree + 1, dtype=np.int64) for degree in degrees]
    )
    return ell, order


def _directions_and_angles(
    directions: "ArrayLike",
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"]:
    vectors = np.asarray(directions, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError("directions must end in dimension three")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("directions must be finite")
    norms = np.linalg.norm(vectors, axis=-1)
    if np.any(norms == 0.0):
        raise ValueError("directions must be nonzero")
    unit = vectors / norms[..., None]
    theta = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(unit[..., 1], unit[..., 0]), 2.0 * np.pi)
    return unit, theta, phi


def spherical_harmonic_matrix(
    directions: "ArrayLike",
    l_max: int,
) -> "NDArray[np.complex128]":
    """Evaluate complex orthonormal spherical harmonics at unit directions."""

    _, theta, phi = _directions_and_angles(directions)
    ell, order = spherical_harmonic_indices(l_max)
    return np.asarray(
        sph_harm_y(
            ell,
            order,
            theta[..., None],
            phi[..., None],
        ),
        dtype=np.complex128,
    )


def _positions(positions: "ArrayLike") -> "NDArray[np.float64]":
    result = np.asarray(positions, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] == 0:
        raise ValueError("positions must have shape (n_atom, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError("positions must be finite")
    return result


def _q_values(q_values: "ArrayLike") -> "NDArray[np.float64]":
    result = np.asarray(q_values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("q_values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("q_values must be finite and non-negative")
    return result


def _weights(
    weights: "ArrayLike",
    *,
    n_q: int,
    n_atom: int,
) -> "NDArray[np.complex128]":
    result = np.asarray(weights, dtype=np.complex128)
    if result.shape == (n_atom,):
        result = np.broadcast_to(result[None, :], (n_q, n_atom))
    if result.shape != (n_q, n_atom):
        raise ValueError("weights must have shape (n_atom,) or (n_q, n_atom)")
    if not np.all(np.isfinite(result)):
        raise ValueError("weights must be finite")
    return result


def _position_harmonics(
    positions: "NDArray[np.float64]",
    l_max: int,
) -> tuple["NDArray[np.float64]", "NDArray[np.complex128]"]:
    radii = np.linalg.norm(positions, axis=1)
    directions = np.zeros_like(positions)
    nonzero = radii > 0.0
    directions[nonzero] = positions[nonzero] / radii[nonzero, None]
    directions[~nonzero, 2] = 1.0
    return radii, spherical_harmonic_matrix(directions, l_max)


def amplitude_coefficients(
    positions: "ArrayLike",
    weights: "ArrayLike",
    q_values: "ArrayLike",
    l_max: int,
) -> "NDArray[np.complex128]":
    r"""Return coefficients of ``F(q,qhat) = sum_lm A_lm(q) Y_lm(qhat)``."""

    positions_array = _positions(positions)
    q_array = _q_values(q_values)
    weight_array = _weights(
        weights,
        n_q=q_array.size,
        n_atom=positions_array.shape[0],
    )
    ell, _ = spherical_harmonic_indices(l_max)
    radii, atom_harmonics = _position_harmonics(positions_array, l_max)
    result = np.empty((q_array.size, ell.size), dtype=np.complex128)
    for degree in range(int(l_max) + 1):
        selected = ell == degree
        radial = spherical_jn(degree, q_array[:, None] * radii[None, :])
        result[:, selected] = (
            4.0
            * np.pi
            * (1j**degree)
            * (weight_array * radial)
            @ np.conj(atom_harmonics[:, selected])
        )
    return result


def pair_intensity_coefficients(
    positions: "ArrayLike",
    weights: "ArrayLike",
    q_values: "ArrayLike",
    l_max: int,
) -> "NDArray[np.complex128]":
    r"""Return intensity harmonics by an explicit ordered atom-pair expansion."""

    positions_array = _positions(positions)
    q_array = _q_values(q_values)
    n_atom = positions_array.shape[0]
    weight_array = _weights(weights, n_q=q_array.size, n_atom=n_atom)
    displacement = (
        positions_array[:, None, :] - positions_array[None, :, :]
    ).reshape(-1, 3)
    pair_weight = (
        weight_array[:, :, None] * np.conj(weight_array[:, None, :])
    ).reshape(q_array.size, -1)
    ell, _ = spherical_harmonic_indices(l_max)
    radii, pair_harmonics = _position_harmonics(displacement, l_max)
    result = np.empty((q_array.size, ell.size), dtype=np.complex128)
    for degree in range(int(l_max) + 1):
        selected = ell == degree
        radial = spherical_jn(degree, q_array[:, None] * radii[None, :])
        result[:, selected] = (
            4.0
            * np.pi
            * (1j**degree)
            * (pair_weight * radial)
            @ np.conj(pair_harmonics[:, selected])
        )
    return result


def synthesize_spherical_series(
    coefficients: "ArrayLike",
    directions: "ArrayLike",
) -> "NDArray[np.complex128]":
    """Synthesize degree-major spherical coefficients on directions."""

    values = np.asarray(coefficients, dtype=np.complex128)
    if values.ndim < 1:
        raise ValueError("coefficients must have at least one dimension")
    n_coeff = values.shape[-1]
    l_max = int(round(np.sqrt(n_coeff) - 1.0))
    if (l_max + 1) ** 2 != n_coeff:
        raise ValueError("the coefficient count must be a perfect square")
    harmonics = spherical_harmonic_matrix(directions, l_max)
    return np.einsum("...c,...c->...", values, harmonics)


def direct_molecular_amplitude(
    positions: "ArrayLike",
    weights: "ArrayLike",
    q_vectors: "ArrayLike",
) -> "NDArray[np.complex128]":
    """Evaluate the direct atomic amplitude on ``(n_q, ..., 3)`` vectors."""

    positions_array = _positions(positions)
    vectors = np.asarray(q_vectors, dtype=np.float64)
    if vectors.ndim < 2 or vectors.shape[-1] != 3:
        raise ValueError("q_vectors must have shape (n_q, ..., 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("q_vectors must be finite")
    weight_array = _weights(
        weights,
        n_q=vectors.shape[0],
        n_atom=positions_array.shape[0],
    )
    phase = np.einsum("q...d,ad->q...a", vectors, positions_array)
    weight_shape = (vectors.shape[0],) + (1,) * (vectors.ndim - 2) + (
        positions_array.shape[0],
    )
    return np.sum(weight_array.reshape(weight_shape) * np.exp(1j * phase), axis=-1)


@dataclass(frozen=True)
class PreparedSphericalGauntProduct:
    """Prepared pseudospectral product for two truncated spherical series."""

    amplitude_l_max: int
    intensity_l_max: int
    quadrature_directions: "NDArray[np.float64]"
    quadrature_weights: "NDArray[np.float64]"
    amplitude_synthesis: "NDArray[np.complex128]"
    intensity_projection: "NDArray[np.complex128]"

    @classmethod
    def build(
        cls,
        amplitude_l_max: int,
        *,
        intensity_l_max: int | None = None,
    ) -> "PreparedSphericalGauntProduct":
        """Prepare a product quadrature exact for the retained bandlimits."""

        amplitude_l_max = int(amplitude_l_max)
        if amplitude_l_max < 0:
            raise ValueError("amplitude_l_max must be non-negative")
        if intensity_l_max is None:
            intensity_l_max = 2 * amplitude_l_max
        intensity_l_max = int(intensity_l_max)
        if intensity_l_max < 0 or intensity_l_max > 2 * amplitude_l_max:
            raise ValueError(
                "intensity_l_max must lie between zero and twice amplitude_l_max"
            )
        full_product_l_max = 2 * amplitude_l_max
        n_theta = full_product_l_max + 1
        n_phi = 2 * full_product_l_max + 1
        cosine, polar_weights = np.polynomial.legendre.leggauss(n_theta)
        theta = np.arccos(cosine)
        phi = 2.0 * np.pi * np.arange(n_phi, dtype=np.float64) / n_phi
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        sine_grid = np.sin(theta_grid)
        directions = np.stack(
            (
                sine_grid * np.cos(phi_grid),
                sine_grid * np.sin(phi_grid),
                np.cos(theta_grid),
            ),
            axis=-1,
        ).reshape(-1, 3)
        weights = np.broadcast_to(
            polar_weights[:, None] * (2.0 * np.pi / n_phi),
            (n_theta, n_phi),
        ).reshape(-1)
        amplitude_synthesis = spherical_harmonic_matrix(
            directions,
            amplitude_l_max,
        )
        intensity_harmonics = spherical_harmonic_matrix(
            directions,
            intensity_l_max,
        )
        intensity_projection = weights[:, None] * np.conj(intensity_harmonics)
        for array in (
            directions,
            weights,
            amplitude_synthesis,
            intensity_projection,
        ):
            array.setflags(write=False)
        return cls(
            amplitude_l_max=amplitude_l_max,
            intensity_l_max=intensity_l_max,
            quadrature_directions=directions,
            quadrature_weights=weights,
            amplitude_synthesis=amplitude_synthesis,
            intensity_projection=intensity_projection,
        )

    @property
    def prepared_bytes(self) -> int:
        """Return bytes held by the prepared quadrature matrices."""

        return int(
            self.quadrature_directions.nbytes
            + self.quadrature_weights.nbytes
            + self.amplitude_synthesis.nbytes
            + self.intensity_projection.nbytes
        )

    def intensity_coefficients(
        self,
        amplitude_coefficients_array: "ArrayLike",
    ) -> "NDArray[np.complex128]":
        """Contract ``F_lm`` with its conjugate into ``I_LM`` coefficients."""

        coefficients = np.asarray(
            amplitude_coefficients_array,
            dtype=np.complex128,
        )
        expected = (self.amplitude_l_max + 1) ** 2
        if coefficients.ndim < 1 or coefficients.shape[-1] != expected:
            raise ValueError(
                f"amplitude coefficients must end in length {expected}"
            )
        field = coefficients @ self.amplitude_synthesis.T
        intensity = np.abs(field) ** 2
        return np.asarray(intensity @ self.intensity_projection, dtype=np.complex128)
