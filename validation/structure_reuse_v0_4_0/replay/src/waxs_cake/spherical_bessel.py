"""Prepared spherical-Bessel transforms and Gaussian transition densities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import fht, fhtoffset
from scipy.integrate import simpson
from scipy.special import spherical_jn

from .atomic_orbitals import (
    gaussian_orbital_degree,
    gaussian_orbital_solid_harmonic,
)
from .spherical_molecular import (
    spherical_harmonic_indices,
    spherical_harmonic_matrix,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _finite_vector(values: "ArrayLike", *, name: str) -> "NDArray[np.float64]":
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PreparedSphericalBesselTransform:
    r"""Prepared transforms ``int r^2 f(r) j_l(qr) dr`` on a radial quadrature."""

    radial_nodes: "NDArray[np.float64]"
    radial_weights: "NDArray[np.float64]"
    q_values: "NDArray[np.float64]"
    degrees: "NDArray[np.int64]"
    kernels: "NDArray[np.float64]"

    @classmethod
    def gauss_legendre(
        cls,
        q_values: "ArrayLike",
        *,
        r_max: float,
        n_radial: int,
        degrees: "ArrayLike",
    ) -> "PreparedSphericalBesselTransform":
        """Prepare Gauss-Legendre radial quadrature and all requested kernels."""

        q_array = _finite_vector(q_values, name="q_values").copy()
        if np.any(q_array < 0.0):
            raise ValueError("q_values must be non-negative")
        r_max = float(r_max)
        n_radial = int(n_radial)
        if not np.isfinite(r_max) or r_max <= 0.0:
            raise ValueError("r_max must be finite and positive")
        if n_radial <= 0:
            raise ValueError("n_radial must be positive")
        degree_values = np.asarray(degrees, dtype=np.int64)
        if degree_values.ndim != 1 or degree_values.size == 0:
            raise ValueError("degrees must be a nonempty one-dimensional array")
        degree_values = np.unique(degree_values)
        if np.any(degree_values < 0):
            raise ValueError("degrees must be non-negative")

        canonical_nodes, canonical_weights = np.polynomial.legendre.leggauss(n_radial)
        radial_nodes = 0.5 * r_max * (canonical_nodes + 1.0)
        radial_weights = 0.5 * r_max * canonical_weights
        arguments = q_array[None, :, None] * radial_nodes[None, None, :]
        kernels = spherical_jn(
            degree_values[:, None, None],
            arguments,
        )
        kernels = np.asarray(
            kernels * (radial_weights * radial_nodes**2)[None, None, :],
            dtype=np.float64,
        )
        for array in (
            radial_nodes,
            radial_weights,
            q_array,
            degree_values,
            kernels,
        ):
            array.setflags(write=False)
        return cls(
            radial_nodes=radial_nodes,
            radial_weights=radial_weights,
            q_values=q_array,
            degrees=degree_values,
            kernels=kernels,
        )

    @property
    def prepared_bytes(self) -> int:
        """Return the storage owned by the prepared quadrature and kernels."""

        return int(
            self.radial_nodes.nbytes
            + self.radial_weights.nbytes
            + self.q_values.nbytes
            + self.degrees.nbytes
            + self.kernels.nbytes
        )

    def degree_index(self, degree: int) -> int:
        """Return the prepared kernel index for one spherical-Bessel degree."""

        matches = np.flatnonzero(self.degrees == int(degree))
        if matches.size != 1:
            raise ValueError(f"degree {degree} is not present in this transform")
        return int(matches[0])

    def transform_degree(
        self,
        radial_profiles: "ArrayLike",
        degree: int,
    ) -> "NDArray[np.complex128]":
        """Transform one or more profiles whose final axis is the radial grid."""

        profiles = np.asarray(radial_profiles)
        if profiles.ndim < 1 or profiles.shape[-1] != self.radial_nodes.size:
            raise ValueError(
                "radial_profiles must end in the prepared radial-node count"
            )
        if not np.all(np.isfinite(profiles)):
            raise ValueError("radial_profiles must be finite")
        kernel = self.kernels[self.degree_index(degree)]
        result = profiles @ kernel.T
        return np.asarray(result, dtype=np.complex128)

    def transform(
        self,
        radial_profiles: "ArrayLike",
        profile_degrees: "ArrayLike",
    ) -> "NDArray[np.complex128]":
        """Transform a batch with one requested degree per radial profile."""

        profiles = np.asarray(radial_profiles)
        if profiles.ndim != 2 or profiles.shape[1] != self.radial_nodes.size:
            raise ValueError("radial_profiles must have shape (n_profile, n_radial)")
        degrees = np.asarray(profile_degrees, dtype=np.int64)
        if degrees.shape != (profiles.shape[0],):
            raise ValueError("profile_degrees must have one entry per profile")
        result = np.empty((profiles.shape[0], self.q_values.size), dtype=np.complex128)
        for degree in np.unique(degrees):
            selected = degrees == degree
            result[selected] = self.transform_degree(profiles[selected], int(degree))
        return result


@dataclass(frozen=True)
class PreparedFFTLogSphericalBesselTransform:
    r"""Prepared FFTLog evaluation of ``int r^2 f(r) j_l(qr) dr``.

    Profiles are sampled on ``radial_nodes``, a uniform logarithmic grid.  Each
    degree uses its own low-ringing FFTLog offset and a prepared six-point
    interpolation from the resulting logarithmic q grid to ``q_values``.
    """

    radial_nodes: "NDArray[np.float64]"
    q_values: "NDArray[np.float64]"
    degrees: "NDArray[np.int64]"
    dln: float
    bias: float
    offsets: "NDArray[np.float64]"
    internal_q_values: "NDArray[np.float64]"
    positive_q_indices: "NDArray[np.int64]"
    interpolation_indices: "NDArray[np.int64]"
    interpolation_weights: "NDArray[np.float64]"

    @classmethod
    def logarithmic(
        cls,
        q_values: "ArrayLike",
        *,
        r_min: float,
        r_max: float,
        n_radial: int,
        degrees: "ArrayLike",
        bias: float = 0.0,
    ) -> "PreparedFFTLogSphericalBesselTransform":
        """Prepare FFTLog grids and cubic interpolation for requested q values."""

        q_array = _finite_vector(q_values, name="q_values").copy()
        if np.any(q_array < 0.0):
            raise ValueError("q_values must be non-negative")
        r_min = float(r_min)
        r_max = float(r_max)
        n_radial = int(n_radial)
        bias = float(bias)
        if not np.isfinite(r_min) or r_min <= 0.0:
            raise ValueError("r_min must be finite and positive")
        if not np.isfinite(r_max) or r_max <= r_min:
            raise ValueError("r_max must be finite and greater than r_min")
        if n_radial < 8:
            raise ValueError("n_radial must be at least 8")
        if not np.isfinite(bias):
            raise ValueError("bias must be finite")
        degree_values = np.asarray(degrees, dtype=np.int64)
        if degree_values.ndim != 1 or degree_values.size == 0:
            raise ValueError("degrees must be a nonempty one-dimensional array")
        degree_values = np.unique(degree_values)
        if np.any(degree_values < 0):
            raise ValueError("degrees must be non-negative")

        dln = float(np.log(r_max / r_min) / (n_radial - 1))
        centre_index = 0.5 * (n_radial - 1)
        radial_centre = float(np.sqrt(r_min * r_max))
        log_radial = np.log(radial_centre) + (
            np.arange(n_radial, dtype=np.float64) - centre_index
        ) * dln
        radial_nodes = np.exp(log_radial)

        positive_q_indices = np.flatnonzero(q_array > 0.0).astype(np.int64)
        if positive_q_indices.size:
            positive_q = q_array[positive_q_indices]
            q_centre = float(np.sqrt(np.min(positive_q) * np.max(positive_q)))
        else:
            positive_q = np.empty(0, dtype=np.float64)
            q_centre = 1.0 / radial_centre
        initial_offset = float(np.log(q_centre * radial_centre))

        offsets = np.empty(degree_values.size, dtype=np.float64)
        internal_q_values = np.empty(
            (degree_values.size, n_radial), dtype=np.float64
        )
        interpolation_indices = np.empty(
            (degree_values.size, positive_q.size, 6), dtype=np.int64
        )
        interpolation_weights = np.empty(
            (degree_values.size, positive_q.size, 6), dtype=np.float64
        )
        relative_nodes = np.arange(n_radial, dtype=np.float64) - centre_index
        for degree_index, degree in enumerate(degree_values):
            offset = float(
                fhtoffset(
                    dln,
                    mu=float(degree) + 0.5,
                    initial=initial_offset,
                    bias=bias,
                )
            )
            offsets[degree_index] = offset
            q_internal = np.exp(offset - np.log(radial_centre) + relative_nodes * dln)
            internal_q_values[degree_index] = q_internal
            if positive_q.size == 0:
                continue
            coordinate = (np.log(positive_q) - np.log(q_internal[0])) / dln
            if np.any(coordinate < 2.0) or np.any(coordinate > n_radial - 3.0):
                raise ValueError(
                    "positive q_values must lie at least two FFTLog nodes inside "
                    "the output interval; widen the logarithmic radial interval"
                )
            centre = np.floor(coordinate).astype(np.int64)
            centre = np.clip(centre, 2, n_radial - 4)
            fraction = coordinate - centre
            stencil = np.arange(-2, 4, dtype=np.int64)
            interpolation_indices[degree_index] = centre[:, None] + stencil
            for output_index, node in enumerate(stencil):
                weight = np.ones(positive_q.size, dtype=np.float64)
                for other_node in stencil:
                    if other_node != node:
                        weight *= (fraction - other_node) / (node - other_node)
                interpolation_weights[degree_index, :, output_index] = weight

        for array in (
            radial_nodes,
            q_array,
            degree_values,
            offsets,
            internal_q_values,
            positive_q_indices,
            interpolation_indices,
            interpolation_weights,
        ):
            array.setflags(write=False)
        return cls(
            radial_nodes=radial_nodes,
            q_values=q_array,
            degrees=degree_values,
            dln=dln,
            bias=bias,
            offsets=offsets,
            internal_q_values=internal_q_values,
            positive_q_indices=positive_q_indices,
            interpolation_indices=interpolation_indices,
            interpolation_weights=interpolation_weights,
        )

    @property
    def prepared_bytes(self) -> int:
        """Return storage owned by the prepared FFTLog and interpolation grids."""

        return int(
            self.radial_nodes.nbytes
            + self.q_values.nbytes
            + self.degrees.nbytes
            + self.offsets.nbytes
            + self.internal_q_values.nbytes
            + self.positive_q_indices.nbytes
            + self.interpolation_indices.nbytes
            + self.interpolation_weights.nbytes
        )

    def degree_index(self, degree: int) -> int:
        """Return the prepared FFTLog index for one spherical-Bessel degree."""

        matches = np.flatnonzero(self.degrees == int(degree))
        if matches.size != 1:
            raise ValueError(f"degree {degree} is not present in this transform")
        return int(matches[0])

    def transform_degree(
        self,
        radial_profiles: "ArrayLike",
        degree: int,
    ) -> "NDArray[np.complex128]":
        """Transform profiles and interpolate them to the prepared q values."""

        profiles = np.asarray(radial_profiles)
        if profiles.ndim < 1 or profiles.shape[-1] != self.radial_nodes.size:
            raise ValueError(
                "radial_profiles must end in the prepared radial-node count"
            )
        if not np.all(np.isfinite(profiles)):
            raise ValueError("radial_profiles must be finite")
        degree = int(degree)
        degree_index = self.degree_index(degree)
        weighted_profiles = profiles * self.radial_nodes**1.5
        transformed_real = fht(
            np.asarray(weighted_profiles.real, dtype=np.float64),
            self.dln,
            mu=degree + 0.5,
            offset=float(self.offsets[degree_index]),
            bias=self.bias,
        )
        transformed = np.asarray(transformed_real, dtype=np.complex128)
        if np.iscomplexobj(weighted_profiles):
            transformed.imag = fht(
                np.asarray(weighted_profiles.imag, dtype=np.float64),
                self.dln,
                mu=degree + 0.5,
                offset=float(self.offsets[degree_index]),
                bias=self.bias,
            )
        transformed *= (
            np.sqrt(np.pi / 2.0)
            * self.internal_q_values[degree_index] ** -1.5
        )

        result = np.zeros(
            profiles.shape[:-1] + (self.q_values.size,), dtype=np.complex128
        )
        if self.positive_q_indices.size:
            indices = self.interpolation_indices[degree_index]
            weights = self.interpolation_weights[degree_index]
            interpolated = np.sum(transformed[..., indices] * weights, axis=-1)
            result[..., self.positive_q_indices] = interpolated
        zero_q_indices = np.flatnonzero(self.q_values == 0.0)
        if degree == 0 and zero_q_indices.size:
            zero_value = simpson(
                profiles * self.radial_nodes**3,
                dx=self.dln,
                axis=-1,
            )
            result[..., zero_q_indices] = zero_value[..., None]
        return result

    def transform(
        self,
        radial_profiles: "ArrayLike",
        profile_degrees: "ArrayLike",
    ) -> "NDArray[np.complex128]":
        """Transform a batch with one requested degree per radial profile."""

        profiles = np.asarray(radial_profiles)
        if profiles.ndim != 2 or profiles.shape[1] != self.radial_nodes.size:
            raise ValueError("radial_profiles must have shape (n_profile, n_radial)")
        degrees = np.asarray(profile_degrees, dtype=np.int64)
        if degrees.shape != (profiles.shape[0],):
            raise ValueError("profile_degrees must have one entry per profile")
        result = np.empty((profiles.shape[0], self.q_values.size), dtype=np.complex128)
        for degree in np.unique(degrees):
            selected = degrees == degree
            result[selected] = self.transform_degree(profiles[selected], int(degree))
        return result


def _spherical_product_quadrature(
    l_max: int,
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    n_theta = l_max + 1
    n_phi = 2 * l_max + 1
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
    return directions, weights


def gaussian_transition_angular_coefficients(
    initial_orbital: str,
    final_orbital: str,
) -> "NDArray[np.complex128]":
    r"""Expand ``H_final(Omega) H_initial(Omega)`` in ``Y_LM(Omega)``."""

    total_degree = gaussian_orbital_degree(initial_orbital) + gaussian_orbital_degree(
        final_orbital
    )
    directions, weights = _spherical_product_quadrature(total_degree)
    initial = gaussian_orbital_solid_harmonic(initial_orbital, *directions.T)
    final = gaussian_orbital_solid_harmonic(final_orbital, *directions.T)
    harmonics = spherical_harmonic_matrix(directions, total_degree)
    coefficients = np.sum(
        weights[:, None] * np.conj(harmonics) * (initial * final)[:, None],
        axis=0,
    )
    scale = max(float(np.max(np.abs(coefficients))), 1.0)
    coefficients[np.abs(coefficients) < 5e-14 * scale] = 0.0
    return np.asarray(coefficients, dtype=np.complex128)


def gaussian_transition_amplitude_coefficients(
    initial_orbital: str,
    final_orbital: str,
    *,
    alpha_initial: float,
    alpha_final: float,
    transform: PreparedSphericalBesselTransform,
) -> "NDArray[np.complex128]":
    """Return spherical coefficients of a Gaussian-orbital transition density."""

    alpha_initial = float(alpha_initial)
    alpha_final = float(alpha_final)
    if (
        not np.isfinite(alpha_initial)
        or not np.isfinite(alpha_final)
        or alpha_initial <= 0.0
        or alpha_final <= 0.0
    ):
        raise ValueError("Gaussian exponents must be finite and positive")
    total_degree = gaussian_orbital_degree(initial_orbital) + gaussian_orbital_degree(
        final_orbital
    )
    ell, _ = spherical_harmonic_indices(total_degree)
    angular = gaussian_transition_angular_coefficients(
        initial_orbital,
        final_orbital,
    )
    radial_profile = transform.radial_nodes**total_degree * np.exp(
        -(alpha_initial + alpha_final) * transform.radial_nodes**2
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


def direct_gaussian_transition_fourier(
    initial_orbital: str,
    final_orbital: str,
    q_vectors: "ArrayLike",
    *,
    alpha_initial: float,
    alpha_final: float,
    quadrature_order: int = 24,
    chunk_size: int = 128,
) -> "NDArray[np.complex128]":
    """Independent tensor Gauss-Hermite Fourier integral for validation."""

    alpha_initial = float(alpha_initial)
    alpha_final = float(alpha_final)
    alpha_total = alpha_initial + alpha_final
    if not np.isfinite(alpha_total) or alpha_initial <= 0.0 or alpha_final <= 0.0:
        raise ValueError("Gaussian exponents must be finite and positive")
    quadrature_order = int(quadrature_order)
    chunk_size = int(chunk_size)
    if quadrature_order <= 0 or chunk_size <= 0:
        raise ValueError("quadrature_order and chunk_size must be positive")
    vectors = np.asarray(q_vectors, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3 or not np.all(np.isfinite(vectors)):
        raise ValueError("q_vectors must be finite and end in dimension three")

    canonical_nodes, canonical_weights = np.polynomial.hermite.hermgauss(
        quadrature_order
    )
    nodes = canonical_nodes / np.sqrt(alpha_total)
    xx, yy, zz = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    weights = (
        canonical_weights[:, None, None]
        * canonical_weights[None, :, None]
        * canonical_weights[None, None, :]
        / alpha_total**1.5
    ).reshape(-1)
    transition_polynomial = (
        gaussian_orbital_solid_harmonic(initial_orbital, *points.T)
        * gaussian_orbital_solid_harmonic(final_orbital, *points.T)
    )
    weighted_transition = weights * transition_polynomial
    flat_vectors = vectors.reshape(-1, 3)
    result = np.empty(flat_vectors.shape[0], dtype=np.complex128)
    for start in range(0, flat_vectors.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_vectors.shape[0])
        phase = flat_vectors[start:stop] @ points.T
        result[start:stop] = np.exp(1j * phase) @ weighted_transition
    return result.reshape(vectors.shape[:-1])
