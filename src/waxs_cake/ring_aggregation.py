"""Exact coherent aggregation of axisymmetric Fourier rings.

The base ACFO operator produces samples with shape ``(ring, phi)``.  This
module composes that operator with a linear map along the ring axis,

``B = C A``,

without changing the azimuthal Fourier convention.  Rows of ``C`` may encode
quadrature weights, spline test functions, detector binning, or any other
*linear coherent-amplitude* projection.  Intensity aggregation is nonlinear
and deliberately outside this contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .axisymmetric_operator import (
    PreparedAxisymmetricOperator,
    normalized_adjoint_error,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .solvers import FormFactors, PreparedCakePlan


def _contraction_matrix(
    values: "ArrayLike",
    n_rings: int,
    complex_dtype: np.dtype,
) -> "NDArray[np.complexfloating]":
    matrix = np.array(values, dtype=complex_dtype, copy=True)
    if matrix.ndim != 2 or matrix.shape[1] != n_rings or matrix.shape[0] == 0:
        raise ValueError(
            f"contraction must have shape (n_outputs, {n_rings}) with "
            "n_outputs > 0"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("contraction must contain only finite values")
    matrix.setflags(write=False)
    return matrix


def trapezoid_weights(coordinate: "ArrayLike") -> "NDArray[np.float64]":
    """Return nonuniform one-dimensional trapezoidal quadrature weights."""

    values = np.asarray(coordinate, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("coordinate must be a finite vector with at least two entries")
    intervals = np.diff(values)
    if np.any(intervals <= 0.0):
        raise ValueError("coordinate must be strictly increasing")
    weights = np.empty_like(values)
    weights[0] = 0.5 * intervals[0]
    weights[-1] = 0.5 * intervals[-1]
    if values.size > 2:
        weights[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    return weights


def surface_of_revolution_jacobian(
    coordinate: "ArrayLike",
    q_perp: "ArrayLike",
    q_z: "ArrayLike",
) -> "NDArray[np.float64]":
    """Return ``rho * hypot(d rho/du, d z/du)`` for a revolved meridian."""

    u, rho, z = _regular_meridian(coordinate, q_perp, q_z)
    edge_order = 2 if u.size >= 3 else 1
    rho_derivative = np.gradient(rho, u, edge_order=edge_order)
    z_derivative = np.gradient(z, u, edge_order=edge_order)
    return rho * np.hypot(rho_derivative, z_derivative)


def surface_of_revolution_jacobian_action(
    coordinate: "ArrayLike",
    q_perp: "ArrayLike",
    q_z: "ArrayLike",
    delta_q_perp: "ArrayLike",
    delta_q_z: "ArrayLike",
) -> "NDArray[np.float64]":
    """Differentiate the sampled surface-of-revolution Jacobian.

    Derivatives of the meridian use the same linear finite-difference stencil
    as :func:`surface_of_revolution_jacobian`, making this the exact directional
    derivative of that discrete Jacobian except at a zero-speed tangent.
    """

    u, rho, z = _regular_meridian(coordinate, q_perp, q_z)
    delta_rho = np.asarray(delta_q_perp, dtype=np.float64)
    delta_z = np.asarray(delta_q_z, dtype=np.float64)
    if delta_rho.shape != u.shape or delta_z.shape != u.shape:
        raise ValueError(f"geometry perturbations must have shape {u.shape}")
    if not np.all(np.isfinite(delta_rho)) or not np.all(np.isfinite(delta_z)):
        raise ValueError("geometry perturbations must contain only finite values")
    edge_order = 2 if u.size >= 3 else 1
    rho_derivative = np.gradient(rho, u, edge_order=edge_order)
    z_derivative = np.gradient(z, u, edge_order=edge_order)
    delta_rho_derivative = np.gradient(delta_rho, u, edge_order=edge_order)
    delta_z_derivative = np.gradient(delta_z, u, edge_order=edge_order)
    speed = np.hypot(rho_derivative, z_derivative)
    if np.any(speed <= np.finfo(np.float64).eps):
        raise ValueError("surface Jacobian derivative requires a nonzero meridian tangent")
    delta_speed = (
        rho_derivative * delta_rho_derivative
        + z_derivative * delta_z_derivative
    ) / speed
    return delta_rho * speed + rho * delta_speed


def _regular_meridian(
    coordinate: "ArrayLike",
    q_perp: "ArrayLike",
    q_z: "ArrayLike",
) -> tuple[
    "NDArray[np.float64]",
    "NDArray[np.float64]",
    "NDArray[np.float64]",
]:
    u = np.asarray(coordinate, dtype=np.float64)
    rho = np.asarray(q_perp, dtype=np.float64)
    z = np.asarray(q_z, dtype=np.float64)
    if u.ndim != 1 or u.size < 2 or rho.shape != u.shape or z.shape != u.shape:
        raise ValueError("coordinate, q_perp, and q_z must be matching vectors")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(rho)) or not np.all(
        np.isfinite(z)
    ):
        raise ValueError("meridian values must be finite")
    if np.any(np.diff(u) <= 0.0):
        raise ValueError("coordinate must be strictly increasing")
    if np.any(rho < 0.0):
        raise ValueError("q_perp must be non-negative")
    return u, rho, z


class PreparedRingAggregation:
    """Compose a prepared ring operator with an exact ring-axis contraction.

    The contraction is applied before the final azimuthal inverse FFT.  Since
    it acts only on the ring axis, this is algebraically identical to applying
    it to the sampled rings after the inverse FFT.
    """

    def __init__(
        self,
        operator: PreparedAxisymmetricOperator,
        contraction: "ArrayLike",
    ) -> None:
        if not isinstance(operator, PreparedAxisymmetricOperator):
            raise TypeError("operator must be a PreparedAxisymmetricOperator")
        self.operator = operator
        self.complex_dtype = operator.complex_dtype
        self.contraction = _contraction_matrix(
            contraction,
            operator.manifold.n_u,
            self.complex_dtype,
        )

    @property
    def object_shape(self) -> tuple[int, int, int, int]:
        return self.operator.object_shape

    @property
    def data_shape(self) -> tuple[int, int]:
        return (self.contraction.shape[0], self.operator.phi.size)

    @property
    def phi(self) -> "NDArray[np.float64]":
        return self.operator.phi

    def _ring_data_array(
        self,
        values: "ArrayLike",
        *,
        field_name: str,
    ) -> "NDArray[np.complexfloating]":
        array = np.asarray(values, dtype=self.complex_dtype)
        if array.shape != self.operator.data_shape:
            raise ValueError(
                f"{field_name} must have shape {self.operator.data_shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{field_name} must contain only finite values")
        return array

    def _data_array(
        self,
        values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        array = np.asarray(values, dtype=self.complex_dtype)
        if array.shape != self.data_shape:
            raise ValueError(f"data values must have shape {self.data_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("data values must contain only finite values")
        return array

    def aggregate_data(
        self,
        ring_data: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply ``C`` to already evaluated spatial-domain rings."""

        data = self._ring_data_array(ring_data, field_name="ring data")
        return (self.contraction @ data).astype(self.complex_dtype, copy=False)

    def aggregate_fourier(
        self,
        ring_fourier: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply ``C`` to ring azimuthal Fourier coefficients."""

        coefficients = self._ring_data_array(
            ring_fourier,
            field_name="ring Fourier coefficients",
        )
        return (self.contraction @ coefficients).astype(
            self.complex_dtype,
            copy=False,
        )

    def expand_data(
        self,
        data_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply ``C^H`` and return data on the base ring grid."""

        data = self._data_array(data_values)
        return (self.contraction.conj().T @ data).astype(
            self.complex_dtype,
            copy=False,
        )

    def prepare_object(
        self,
        object_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Reuse the base operator's azimuthal object FFT."""

        return self.operator.prepare_object(object_values)

    def apply_prepared_object_fourier(
        self,
        object_fourier: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply geometry and ``C`` while retaining Fourier coefficients."""

        ring_fourier = self.operator.apply_prepared_object_fourier(object_fourier)
        return self.aggregate_fourier(ring_fourier)

    def forward_fourier(
        self,
        object_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Return aggregated azimuthal Fourier coefficients."""

        return self.apply_prepared_object_fourier(self.prepare_object(object_values))

    def apply_prepared_object(
        self,
        object_fourier: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply the composed operator to a reusable object FFT."""

        return np.fft.ifft(
            self.apply_prepared_object_fourier(object_fourier),
            axis=-1,
        ).astype(self.complex_dtype, copy=False)

    def forward(
        self,
        object_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply the exact coherent multi-ring contraction ``C A``."""

        return self.apply_prepared_object(self.prepare_object(object_values))

    def geometry_jacobian_action(
        self,
        object_values: "ArrayLike",
        delta_q_perp: "ArrayLike",
        delta_q_z: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Differentiate ``C A`` for fixed ``C`` and a geometry perturbation.

        If the contraction itself depends on the geometry (for example through
        a surface-Jacobian quadrature weight), its separate ``delta_C @ A``
        contribution must be added by the caller.
        """

        ring_action = self.operator.geometry_jacobian_action(
            object_values,
            delta_q_perp,
            delta_q_z,
        )
        return self.aggregate_data(ring_action)

    def adjoint_euclidean(
        self,
        data_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply the exact Euclidean adjoint ``A^H C^H``."""

        return self.operator.adjoint_euclidean(self.expand_data(data_values))

    def adjoint_test(
        self,
        object_values: "ArrayLike",
        data_values: "ArrayLike",
    ) -> float:
        """Return the normalized dot-product mismatch of ``C A`` and its adjoint."""

        data = self._data_array(data_values)
        object_array = np.asarray(object_values, dtype=self.complex_dtype)
        if object_array.shape != self.object_shape:
            raise ValueError(f"object values must have shape {self.object_shape}")
        if not np.all(np.isfinite(object_array)):
            raise ValueError("object values must contain only finite values")
        left = complex(np.vdot(self.forward(object_array), data))
        right = complex(np.vdot(object_array, self.adjoint_euclidean(data)))
        return normalized_adjoint_error(left, right)

    def fuse(self) -> "PreparedFusedRingAggregation":
        """Precontract the ring axis into a reusable exact projected kernel."""

        return PreparedFusedRingAggregation(self)


class PreparedFusedRingAggregation:
    """Exact ``C A`` with the ring contraction precomputed into the kernel.

    This path avoids materializing or evaluating every output ring when the
    geometry and contraction are reused for multiple objects.  Its prepared
    kernel has shape ``(n_outputs, element, r, z, harmonic)``; consequently it
    is most attractive when ``n_outputs << n_rings`` and that storage fits.
    Geometry or geometry-dependent quadrature changes require rebuilding it.
    """

    def __init__(self, aggregation: PreparedRingAggregation) -> None:
        if not isinstance(aggregation, PreparedRingAggregation):
            raise TypeError("aggregation must be a PreparedRingAggregation")
        self.aggregation = aggregation
        self.operator = aggregation.operator
        self.complex_dtype = aggregation.complex_dtype
        projected_kernel = np.einsum(
            "kj,ej,jz,jrh->kerzh",
            aggregation.contraction,
            self.operator.form_factors,
            self.operator.z_phase,
            self.operator.kernel_fft,
            optimize=True,
        ).astype(self.complex_dtype, copy=False)
        projected_kernel.setflags(write=False)
        self.projected_kernel = projected_kernel

    @property
    def object_shape(self) -> tuple[int, int, int, int]:
        return self.operator.object_shape

    @property
    def data_shape(self) -> tuple[int, int]:
        return self.aggregation.data_shape

    @property
    def prepared_nbytes(self) -> int:
        return int(self.projected_kernel.nbytes)

    def _data_array(
        self,
        values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        array = np.asarray(values, dtype=self.complex_dtype)
        if array.shape != self.data_shape:
            raise ValueError(f"data values must have shape {self.data_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("data values must contain only finite values")
        return array

    def prepare_object(
        self,
        object_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        return self.operator.prepare_object(object_values)

    def apply_prepared_object_fourier(
        self,
        object_fourier: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        histogram_fft = np.asarray(object_fourier, dtype=self.complex_dtype)
        if histogram_fft.shape != self.object_shape:
            raise ValueError(f"object Fourier values must have shape {self.object_shape}")
        if not np.all(np.isfinite(histogram_fft)):
            raise ValueError("object Fourier values must contain only finite values")
        data_fft = np.einsum(
            "kerzh,erzh->kh",
            self.projected_kernel,
            histogram_fft,
            optimize=True,
        )
        return data_fft.astype(self.complex_dtype, copy=False)

    def forward_fourier(
        self,
        object_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        return self.apply_prepared_object_fourier(self.prepare_object(object_values))

    def forward(
        self,
        object_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        return np.fft.ifft(
            self.forward_fourier(object_values),
            axis=-1,
        ).astype(self.complex_dtype, copy=False)

    def adjoint_euclidean(
        self,
        data_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        data = self._data_array(data_values)
        data_fft = np.fft.fft(data, axis=-1)
        histogram_fft = np.einsum(
            "kerzh,kh->erzh",
            np.conj(self.projected_kernel),
            data_fft,
            optimize=True,
        )
        return np.fft.ifft(histogram_fft, axis=-1).astype(
            self.complex_dtype,
            copy=False,
        )

    def adjoint_test(
        self,
        object_values: "ArrayLike",
        data_values: "ArrayLike",
    ) -> float:
        object_array = np.asarray(object_values, dtype=self.complex_dtype)
        if object_array.shape != self.object_shape:
            raise ValueError(f"object values must have shape {self.object_shape}")
        if not np.all(np.isfinite(object_array)):
            raise ValueError("object values must contain only finite values")
        data = self._data_array(data_values)
        left = complex(np.vdot(self.forward(object_array), data))
        right = complex(np.vdot(object_array, self.adjoint_euclidean(data)))
        return normalized_adjoint_error(left, right)


def streaming_circular_ahat_projection(
    plan: "PreparedCakePlan",
    contraction: "ArrayLike",
    *,
    form_factors: "FormFactors" = None,
    q_block_size: int | None = None,
) -> "NDArray[np.complexfloating]":
    """Project ``PreparedCakePlan.circular_ahat`` without storing all rings.

    This is an exact memory-streaming path.  It reduces the materialized output
    from ``(n_q, n_phi)`` to ``(n_outputs, n_phi)`` but does not by itself avoid
    evaluating each requested ring kernel.
    """

    block_size = plan.q_block_size if q_block_size is None else int(q_block_size)
    if block_size <= 0:
        raise ValueError("q_block_size must be positive")
    matrix = _contraction_matrix(
        contraction,
        int(plan.q.size),
        np.dtype(plan.complex_dtype),
    )
    output = np.zeros(
        (matrix.shape[0], plan.binned.n_phi),
        dtype=plan.complex_dtype,
    )
    for start in range(0, plan.q.size, block_size):
        stop = min(start + block_size, plan.q.size)
        indices = np.arange(start, stop, dtype=np.int64)
        block = plan.circular_ahat(
            q_indices=indices,
            form_factors=form_factors,
            q_block_size=block_size,
        )
        output += matrix[:, start:stop] @ block
    return output
