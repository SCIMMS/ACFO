"""Restriction-extension pair for axisymmetric homogeneous PDE modes.

This module deliberately uses the physical Fourier convention

``restriction: exp(-i Gamma.x)`` and ``extension: exp(+i Gamma.x)``.

It is separate from the established positive-sign ACFO point-sampling path.
The coefficient space stores azimuthal Fourier coefficients ``a_h(s_j)`` and
uses the declared manifold quadrature in its inner product.  The field space
is a finite set of Cartesian samples with explicit spatial quadrature weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import jv

from .axisymmetric_manifold import AxisymmetricManifold
from .axisymmetric_operator import normalized_adjoint_error

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class AxisymmetricPDEPair:
    """Prepared low-angular-bandwidth PDE restriction-extension pair.

    For signed modes ``h = -max_h, ..., max_h``, the extension is

    ``2*pi*sum_j,h w_j i**h a[j,h] J_h(q_perp[j]*R)``
    ``* exp(i*h*beta) * exp(i*q_z[j]*z)``.

    The restriction is its exact adjoint under

    ``<a,b>_Gamma = 2*pi*sum_j,h w_j conj(a[j,h])*b[j,h]``

    and

    ``<u,v>_Omega = sum_n spatial_weight[n]*conj(u[n])*v[n]``.

    Cartesian points are processed in blocks; the dense point-by-mode matrix
    is never materialized.
    """

    def __init__(
        self,
        manifold: AxisymmetricManifold,
        cartesian_coords: "ArrayLike",
        *,
        manifold_weights: "ArrayLike",
        spatial_weights: "ArrayLike",
        max_h: int,
        point_block_size: int = 8192,
    ) -> None:
        if not isinstance(manifold, AxisymmetricManifold):
            raise TypeError("manifold must be an AxisymmetricManifold")
        coords = np.array(cartesian_coords, dtype=np.float64, copy=True)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
            raise ValueError("cartesian_coords must have shape (n_points, 3)")
        if not np.all(np.isfinite(coords)):
            raise ValueError("cartesian_coords must contain only finite values")

        gamma_weights = np.array(manifold_weights, dtype=np.float64, copy=True)
        if gamma_weights.shape != (manifold.n_u,):
            raise ValueError(f"manifold_weights must have shape ({manifold.n_u},)")
        if not np.all(np.isfinite(gamma_weights)) or np.any(gamma_weights < 0.0):
            raise ValueError("manifold_weights must be finite and non-negative")
        if not np.any(gamma_weights > 0.0):
            raise ValueError("manifold_weights must contain a positive value")

        omega_weights = np.array(spatial_weights, dtype=np.float64, copy=True)
        if omega_weights.shape != (coords.shape[0],):
            raise ValueError(
                f"spatial_weights must have shape ({coords.shape[0]},)"
            )
        if not np.all(np.isfinite(omega_weights)) or np.any(omega_weights < 0.0):
            raise ValueError("spatial_weights must be finite and non-negative")
        if not np.any(omega_weights > 0.0):
            raise ValueError("spatial_weights must contain a positive value")

        max_h = int(max_h)
        if max_h < 0:
            raise ValueError("max_h must be non-negative")
        point_block_size = int(point_block_size)
        if point_block_size <= 0:
            raise ValueError("point_block_size must be positive")

        radius = np.hypot(coords[:, 0], coords[:, 1])
        beta = np.arctan2(coords[:, 1], coords[:, 0])
        beta[radius == 0.0] = 0.0

        self.manifold = manifold
        self.max_h = max_h
        self.point_block_size = point_block_size
        self.modes = np.arange(-max_h, max_h + 1, dtype=np.int64)
        self.cartesian_coords = coords
        self.manifold_weights = gamma_weights
        self.spatial_weights = omega_weights
        self.radius = radius
        self.beta = beta
        self.z = coords[:, 2].copy()

        for array in (
            self.modes,
            self.cartesian_coords,
            self.manifold_weights,
            self.spatial_weights,
            self.radius,
            self.beta,
            self.z,
        ):
            array.setflags(write=False)

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        return (self.manifold.n_u, self.modes.size)

    @property
    def field_shape(self) -> tuple[int]:
        return (self.cartesian_coords.shape[0],)

    def _coefficients(self, values: "ArrayLike") -> "NDArray[np.complex128]":
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != self.coefficient_shape:
            raise ValueError(f"coefficients must have shape {self.coefficient_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("coefficients must contain only finite values")
        return array

    def _field(self, values: "ArrayLike") -> "NDArray[np.complex128]":
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != self.field_shape:
            raise ValueError(f"field must have shape {self.field_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("field must contain only finite values")
        return array

    def _blocks(self):
        for start in range(0, self.field_shape[0], self.point_block_size):
            yield slice(start, min(start + self.point_block_size, self.field_shape[0]))

    def extension(self, coefficients: "ArrayLike") -> "NDArray[np.complex128]":
        """Synthesize Cartesian field samples with the positive phase sign."""

        coeff = self._coefficients(coefficients)
        output = np.zeros(self.field_shape, dtype=np.complex128)
        weighted_coeff = self.manifold_weights[:, None] * coeff
        q_perp = self.manifold.q_perp[:, None]
        q_z = self.manifold.q_z[:, None]

        for block in self._blocks():
            radius = self.radius[block]
            beta = self.beta[block]
            z = self.z[block]
            argument = q_perp * radius[None, :]
            axial_phase = np.exp(1j * q_z * z[None, :])
            result = np.zeros(radius.size, dtype=np.complex128)
            for mode_index, mode in enumerate(self.modes):
                if not np.any(weighted_coeff[:, mode_index]):
                    continue
                meridional = np.sum(
                    weighted_coeff[:, mode_index, None]
                    * jv(int(mode), argument)
                    * axial_phase,
                    axis=0,
                )
                result += (
                    (1j ** int(mode))
                    * np.exp(1j * int(mode) * beta)
                    * meridional
                )
            output[block] = 2.0 * np.pi * result
        return output

    def restriction(self, field: "ArrayLike") -> "NDArray[np.complex128]":
        """Restrict a Cartesian field with the negative phase sign."""

        values = self._field(field)
        output = np.zeros(self.coefficient_shape, dtype=np.complex128)
        q_perp = self.manifold.q_perp[:, None]
        q_z = self.manifold.q_z[:, None]

        for block in self._blocks():
            radius = self.radius[block]
            beta = self.beta[block]
            z = self.z[block]
            weighted_field = self.spatial_weights[block] * values[block]
            argument = q_perp * radius[None, :]
            axial_phase = np.exp(-1j * q_z * z[None, :])
            for mode_index, mode in enumerate(self.modes):
                angular_field = (
                    ((-1j) ** int(mode))
                    * np.exp(-1j * int(mode) * beta)
                    * weighted_field
                )
                output[:, mode_index] += np.sum(
                    jv(int(mode), argument)
                    * axial_phase
                    * angular_field[None, :],
                    axis=1,
                )
        return output

    def manifold_inner_product(
        self,
        left: "ArrayLike",
        right: "ArrayLike",
    ) -> complex:
        """Evaluate the declared harmonic coefficient-space inner product."""

        left_array = self._coefficients(left)
        right_array = self._coefficients(right)
        return complex(
            2.0
            * np.pi
            * np.sum(
                self.manifold_weights[:, None]
                * np.conj(left_array)
                * right_array
            )
        )

    def spatial_inner_product(
        self,
        left: "ArrayLike",
        right: "ArrayLike",
    ) -> complex:
        """Evaluate the declared finite-domain Cartesian inner product."""

        left_array = self._field(left)
        right_array = self._field(right)
        return complex(
            np.sum(self.spatial_weights * np.conj(left_array) * right_array)
        )

    def adjoint_test(
        self,
        coefficients: "ArrayLike",
        field: "ArrayLike",
    ) -> float:
        """Return the normalized weighted extension-restriction mismatch."""

        coeff = self._coefficients(coefficients)
        values = self._field(field)
        left = self.spatial_inner_product(self.extension(coeff), values)
        right = self.manifold_inner_product(coeff, self.restriction(values))
        return normalized_adjoint_error(left, right)
