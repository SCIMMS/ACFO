"""Prepared forward-adjoint pair for axisymmetric Fourier sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .axisymmetric_manifold import AxisymmetricManifold
from .solvers import FormFactors, normalize_form_factors

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .histogram import BinnedStructure


def normalized_adjoint_error(left: complex, right: complex) -> float:
    """Return the symmetric relative mismatch between two complex dot products."""

    denominator = abs(left) + abs(right)
    if denominator == 0.0:
        return 0.0 if left == right else float("inf")
    return float(abs(left - right) / denominator)


class PreparedAxisymmetricOperator:
    """Reusable complex forward-adjoint pair on one cylindrical object grid.

    Histogram entries are treated as discrete source coefficients, so the
    object space uses the Euclidean inner product. The forward always returns
    point samples. ``adjoint_euclidean`` uses the Euclidean data inner product;
    ``adjoint_weighted`` applies explicit radial data weights first.
    """

    def __init__(
        self,
        template: "BinnedStructure",
        manifold: AxisymmetricManifold,
        *,
        form_factors: FormFactors = None,
        complex_dtype: np.dtype | str = np.complex128,
    ) -> None:
        self.manifold = manifold
        self.complex_dtype = np.dtype(complex_dtype)
        if self.complex_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
            raise ValueError("complex_dtype must be complex64 or complex128")

        self.elements = tuple(template.elements)
        self.r_centers = np.array(template.r_centers, dtype=np.float64, copy=True)
        self.z_centers = np.array(template.z_centers, dtype=np.float64, copy=True)
        self.phi = np.array(template.beta_centers, dtype=np.float64, copy=True)
        if not self.elements:
            raise ValueError("template must contain at least one element")
        if self.r_centers.ndim != 1 or self.z_centers.ndim != 1 or self.phi.ndim != 1:
            raise ValueError("template coordinate arrays must be one-dimensional")
        if not self.r_centers.size or not self.z_centers.size or not self.phi.size:
            raise ValueError("template coordinate arrays must not be empty")

        self.object_shape = (
            len(self.elements),
            self.r_centers.size,
            self.z_centers.size,
            self.phi.size,
        )
        if np.asarray(template.hist).shape != self.object_shape:
            raise ValueError("template histogram shape does not match its coordinates")

        self.geometry_derivatives_supported = form_factors is None
        self.form_factors = normalize_form_factors(
            self.elements,
            manifold.q_norm,
            form_factors,
        ).astype(self.complex_dtype, copy=False)
        self.z_phase = np.exp(1j * manifold.q_z[:, None] * self.z_centers[None, :]).astype(
            self.complex_dtype,
            copy=False,
        )
        kernel_angles = np.arange(self.phi.size, dtype=np.float64) * (
            2.0 * np.pi / self.phi.size
        )
        kernel = np.exp(
            1j
            * manifold.q_perp[:, None, None]
            * self.r_centers[None, :, None]
            * np.cos(kernel_angles)[None, None, :]
        ).astype(self.complex_dtype, copy=False)
        self.kernel_fft = np.fft.fft(kernel, axis=-1).astype(self.complex_dtype, copy=False)

        for array in (
            self.r_centers,
            self.z_centers,
            self.phi,
            self.form_factors,
            self.z_phase,
            self.kernel_fft,
        ):
            array.setflags(write=False)

    @property
    def data_shape(self) -> tuple[int, int]:
        return (self.manifold.n_u, self.phi.size)

    @property
    def angular_modes(self) -> "NDArray[np.int64]":
        """Return signed FFT mode numbers in storage order."""

        return np.rint(np.fft.fftfreq(self.phi.size) * self.phi.size).astype(np.int64)

    def _object_array(self, values: "ArrayLike") -> "NDArray[np.complexfloating]":
        array = np.asarray(values, dtype=self.complex_dtype)
        if array.shape != self.object_shape:
            raise ValueError(f"object values must have shape {self.object_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("object values must contain only finite values")
        return array

    def _data_array(self, values: "ArrayLike") -> "NDArray[np.complexfloating]":
        array = np.asarray(values, dtype=self.complex_dtype)
        if array.shape != self.data_shape:
            raise ValueError(f"data values must have shape {self.data_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("data values must contain only finite values")
        return array

    def _phase_offsets(self, values: "ArrayLike | None") -> "NDArray[np.float64] | None":
        if values is None:
            return None
        offsets = np.asarray(values, dtype=np.float64)
        expected = (self.manifold.n_u,)
        if offsets.shape != expected or not np.all(np.isfinite(offsets)):
            raise ValueError(f"phase_offsets must be finite with shape {expected}")
        return offsets

    def _angles(self, values: "ArrayLike") -> "NDArray[np.float64]":
        angles = np.asarray(values, dtype=np.float64)
        if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
            raise ValueError("angles must be a non-empty finite one-dimensional array")
        return angles

    def _data_at_angles_array(
        self,
        values: "ArrayLike",
        n_angles: int,
    ) -> "NDArray[np.complexfloating]":
        array = np.asarray(values, dtype=self.complex_dtype)
        expected = (self.manifold.n_u, n_angles)
        if array.shape != expected:
            raise ValueError(f"data values must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("data values must contain only finite values")
        return array

    def _mode_mask(self, values: "ArrayLike") -> "NDArray[np.bool_]":
        mask = np.asarray(values)
        if mask.shape != self.data_shape or mask.dtype != np.bool_:
            raise ValueError(
                f"mode_mask must be a Boolean array with shape {self.data_shape}"
            )
        return mask

    def _offset_phase(
        self,
        phase_offsets: "ArrayLike | None",
        *,
        sign: float,
        subtract_grid_origin: bool = False,
    ) -> "NDArray[np.complexfloating] | None":
        offsets = self._phase_offsets(phase_offsets)
        if offsets is None and not subtract_grid_origin:
            return None
        if offsets is None:
            offsets = np.zeros(self.manifold.n_u, dtype=np.float64)
        if subtract_grid_origin:
            offsets = offsets - self.phi[0]
        phase = np.exp(
            sign * 1j * offsets[:, None] * self.angular_modes[None, :]
        )
        return phase.astype(self.complex_dtype, copy=False)

    def _radial_weights(self, values: "ArrayLike | None") -> "NDArray[np.float64]":
        if values is None:
            weights = self.manifold.resolved_data_weights
        else:
            weights = np.asarray(values, dtype=np.float64)
        if weights.shape != (self.manifold.n_u,):
            raise ValueError(f"data_weights must have shape ({self.manifold.n_u},)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("data_weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("data_weights must contain at least one positive value")
        return weights

    def forward_fourier(
        self,
        object_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Return azimuthal Fourier coefficients before the final inverse FFT."""

        return self.apply_prepared_object_fourier(self.prepare_object(object_values))

    def prepare_object(
        self,
        object_values: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Transform cylindrical object coefficients along the azimuth axis."""

        histogram = self._object_array(object_values)
        return np.fft.fft(histogram, axis=-1).astype(self.complex_dtype, copy=False)

    def apply_prepared_object_fourier(
        self,
        object_fourier: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Contract a pretransformed object and return data Fourier coefficients."""

        histogram_fft = self._object_array(object_fourier)
        data_fft = np.einsum(
            "ej,jz,jrh,erzh->jh",
            self.form_factors,
            self.z_phase,
            self.kernel_fft,
            histogram_fft,
            optimize=True,
        )
        return data_fft.astype(self.complex_dtype, copy=False)

    def apply_prepared_object(
        self,
        object_fourier: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply geometry with an optional starting azimuth per target orbit."""

        data_fourier = self.apply_prepared_object_fourier(object_fourier)
        phase = self._offset_phase(phase_offsets, sign=1.0)
        if phase is not None:
            data_fourier = data_fourier * phase

        return np.fft.ifft(
            data_fourier,
            axis=-1,
        ).astype(self.complex_dtype, copy=False)

    def forward(
        self,
        object_values: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply the unweighted point-evaluation forward operator."""

        return self.apply_prepared_object(
            self.prepare_object(object_values),
            phase_offsets=phase_offsets,
        )

    def apply_prepared_object_mode_mask(
        self,
        object_fourier: "ArrayLike",
        mode_mask: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply a fixed per-orbit Fourier-mode mask before the final IFFT."""

        mask = self._mode_mask(mode_mask)
        data_fourier = self.apply_prepared_object_fourier(object_fourier)
        data_fourier = np.where(mask, data_fourier, 0.0)
        phase = self._offset_phase(phase_offsets, sign=1.0)
        if phase is not None:
            data_fourier *= phase
        return np.fft.ifft(data_fourier, axis=-1).astype(
            self.complex_dtype, copy=False
        )

    def forward_mode_mask(
        self,
        object_values: "ArrayLike",
        mode_mask: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply the forward operator with one frozen per-orbit mode mask."""

        return self.apply_prepared_object_mode_mask(
            self.prepare_object(object_values),
            mode_mask,
            phase_offsets=phase_offsets,
        )

    def apply_prepared_object_at_angles(
        self,
        object_fourier: "ArrayLike",
        angles: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Evaluate a prepared object at arbitrary physical azimuths.

        ``angles`` are expressed in the local Cartesian frame used by the
        cylindrical template.  ``phase_offsets`` supplies one additional
        starting azimuth for every meridional target orbit.  Unlike the FFT
        path, the angles need not be uniform or span a complete revolution.
        """

        angle_array = self._angles(angles)
        data_fourier = self.apply_prepared_object_fourier(object_fourier)
        offset_phase = self._offset_phase(
            phase_offsets,
            sign=1.0,
            subtract_grid_origin=True,
        )
        assert offset_phase is not None
        modes = self.angular_modes.astype(np.float64, copy=False)
        angular_basis = np.exp(1j * angle_array[:, None] * modes[None, :])
        angular_basis /= self.phi.size
        result = (data_fourier * offset_phase) @ angular_basis.T
        return result.astype(self.complex_dtype, copy=False)

    def forward_at_angles(
        self,
        object_values: "ArrayLike",
        angles: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply the forward operator at arbitrary physical azimuths."""

        return self.apply_prepared_object_at_angles(
            self.prepare_object(object_values),
            angles,
            phase_offsets=phase_offsets,
        )

    def apply_prepared_object_at_angles_mode_mask(
        self,
        object_fourier: "ArrayLike",
        angles: "ArrayLike",
        mode_mask: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Evaluate a prepared object at arbitrary angles with a frozen mask."""

        angle_array = self._angles(angles)
        mask = self._mode_mask(mode_mask)
        data_fourier = np.where(
            mask,
            self.apply_prepared_object_fourier(object_fourier),
            0.0,
        )
        offset_phase = self._offset_phase(
            phase_offsets,
            sign=1.0,
            subtract_grid_origin=True,
        )
        assert offset_phase is not None
        modes = self.angular_modes.astype(np.float64, copy=False)
        angular_basis = np.exp(1j * angle_array[:, None] * modes[None, :])
        angular_basis /= self.phi.size
        result = (data_fourier * offset_phase) @ angular_basis.T
        return result.astype(self.complex_dtype, copy=False)

    def forward_at_angles_mode_mask(
        self,
        object_values: "ArrayLike",
        angles: "ArrayLike",
        mode_mask: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply the fixed-mask forward operator at arbitrary angles."""

        return self.apply_prepared_object_at_angles_mode_mask(
            self.prepare_object(object_values),
            angles,
            mode_mask,
            phase_offsets=phase_offsets,
        )

    def geometry_derivatives(
        self,
        object_values: "ArrayLike",
    ) -> tuple["NDArray[np.complexfloating]", "NDArray[np.complexfloating]"]:
        """Return exact derivatives with respect to sampled ``Q_perp`` and ``Q_z``.

        The two returned arrays both have ``data_shape``. They differentiate
        each radial row independently while keeping its azimuth samples fixed.
        Geometry derivatives currently require constant unit form factors;
        differentiating a user-supplied q-dependent form-factor model would
        require its radial derivative as an additional contract.
        """

        if not self.geometry_derivatives_supported:
            raise ValueError(
                "geometry derivatives require form_factors=None because "
                "q-dependent form-factor derivatives are not available"
            )
        histogram = self._object_array(object_values)
        cos_beta = np.cos(self.phi)[None, None, None, :]
        sin_beta = np.sin(self.phi)[None, None, None, :]
        radius = self.r_centers[None, :, None, None]
        height = self.z_centers[None, None, :, None]
        amplitude_x = self.forward(histogram * radius * cos_beta)
        amplitude_y = self.forward(histogram * radius * sin_beta)
        amplitude_z = self.forward(histogram * height)
        cos_phi = np.cos(self.phi)[None, :]
        sin_phi = np.sin(self.phi)[None, :]
        derivative_q_perp = 1j * (cos_phi * amplitude_x + sin_phi * amplitude_y)
        derivative_q_z = 1j * amplitude_z
        return (
            derivative_q_perp.astype(self.complex_dtype, copy=False),
            derivative_q_z.astype(self.complex_dtype, copy=False),
        )

    def geometry_jacobian_action(
        self,
        object_values: "ArrayLike",
        delta_q_perp: "ArrayLike",
        delta_q_z: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        """Apply the exact geometry Jacobian to one meridional perturbation."""

        delta_perp = np.asarray(delta_q_perp, dtype=np.float64)
        delta_z = np.asarray(delta_q_z, dtype=np.float64)
        expected_shape = (self.manifold.n_u,)
        if delta_perp.shape != expected_shape or delta_z.shape != expected_shape:
            raise ValueError(f"geometry perturbations must have shape {expected_shape}")
        if not np.all(np.isfinite(delta_perp)) or not np.all(np.isfinite(delta_z)):
            raise ValueError("geometry perturbations must contain only finite values")
        derivative_perp, derivative_z = self.geometry_derivatives(object_values)
        return derivative_perp * delta_perp[:, None] + derivative_z * delta_z[:, None]

    def geometry_loss_gradient(
        self,
        object_values: "ArrayLike",
        residual: "ArrayLike",
    ) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
        """Differentiate ``0.5 * ||residual||_2**2`` over sampled geometry."""

        residual_array = self._data_array(residual)
        derivative_perp, derivative_z = self.geometry_derivatives(object_values)
        gradient_perp = np.real(
            np.sum(np.conj(residual_array) * derivative_perp, axis=1)
        )
        gradient_z = np.real(np.sum(np.conj(residual_array) * derivative_z, axis=1))
        return gradient_perp.astype(np.float64), gradient_z.astype(np.float64)

    def forward_harmonic_cutoff(
        self,
        object_values: "ArrayLike",
        max_h: int,
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply the forward operator after retaining modes ``|h| <= max_h``."""

        max_h = int(max_h)
        if max_h < 0 or max_h > self.phi.size // 2:
            raise ValueError("max_h must be in [0, n_phi // 2]")
        data_fft = self.forward_fourier(object_values).copy()
        data_fft[..., np.abs(self.angular_modes) > max_h] = 0.0
        phase = self._offset_phase(phase_offsets, sign=1.0)
        if phase is not None:
            data_fft *= phase
        return np.fft.ifft(data_fft, axis=-1).astype(self.complex_dtype, copy=False)

    def _adjoint_core(
        self,
        data_values: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        data = self._data_array(data_values)
        data_fft = np.fft.fft(data, axis=-1)
        phase = self._offset_phase(phase_offsets, sign=-1.0)
        if phase is not None:
            data_fft *= phase
        return self._adjoint_from_data_fourier(data_fft)

    def _adjoint_from_data_fourier(
        self,
        data_fourier: "ArrayLike",
    ) -> "NDArray[np.complexfloating]":
        data_fft = np.asarray(data_fourier, dtype=self.complex_dtype)
        if data_fft.shape != self.data_shape or not np.all(np.isfinite(data_fft)):
            raise ValueError(f"data Fourier values must be finite with shape {self.data_shape}")
        histogram_fft = np.einsum(
            "ej,jz,jrh,jh->erzh",
            np.conj(self.form_factors),
            np.conj(self.z_phase),
            np.conj(self.kernel_fft),
            data_fft,
            optimize=True,
        )
        return np.fft.ifft(histogram_fft, axis=-1).astype(self.complex_dtype, copy=False)

    def adjoint_euclidean(
        self,
        data_values: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply the adjoint for Euclidean object and data inner products."""

        return self._adjoint_core(data_values, phase_offsets=phase_offsets)

    def adjoint_mode_mask_euclidean(
        self,
        data_values: "ArrayLike",
        mode_mask: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply the Euclidean adjoint of ``forward_mode_mask``."""

        data = self._data_array(data_values)
        mask = self._mode_mask(mode_mask)
        data_fourier = np.fft.fft(data, axis=-1)
        phase = self._offset_phase(phase_offsets, sign=-1.0)
        if phase is not None:
            data_fourier *= phase
        data_fourier = np.where(mask, data_fourier, 0.0)
        return self._adjoint_from_data_fourier(data_fourier)

    def adjoint_at_angles_euclidean(
        self,
        data_values: "ArrayLike",
        angles: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply the Euclidean adjoint of ``forward_at_angles``."""

        angle_array = self._angles(angles)
        data = self._data_at_angles_array(data_values, angle_array.size)
        modes = self.angular_modes.astype(np.float64, copy=False)
        angular_adjoint = np.exp(-1j * angle_array[:, None] * modes[None, :])
        data_fourier = data @ angular_adjoint
        offset_phase = self._offset_phase(
            phase_offsets,
            sign=-1.0,
            subtract_grid_origin=True,
        )
        assert offset_phase is not None
        data_fourier *= offset_phase
        return self._adjoint_from_data_fourier(data_fourier)

    def adjoint_at_angles_mode_mask_euclidean(
        self,
        data_values: "ArrayLike",
        angles: "ArrayLike",
        mode_mask: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply the Euclidean adjoint of the fixed-mask arbitrary-angle path."""

        angle_array = self._angles(angles)
        data = self._data_at_angles_array(data_values, angle_array.size)
        mask = self._mode_mask(mode_mask)
        modes = self.angular_modes.astype(np.float64, copy=False)
        angular_adjoint = np.exp(-1j * angle_array[:, None] * modes[None, :])
        data_fourier = data @ angular_adjoint
        offset_phase = self._offset_phase(
            phase_offsets,
            sign=-1.0,
            subtract_grid_origin=True,
        )
        assert offset_phase is not None
        data_fourier *= offset_phase
        data_fourier = np.where(mask, data_fourier, 0.0)
        return self._adjoint_from_data_fourier(data_fourier)

    def adjoint_weighted(
        self,
        data_values: "ArrayLike",
        *,
        data_weights: "ArrayLike | None" = None,
        phase_offsets: "ArrayLike | None" = None,
    ) -> "NDArray[np.complexfloating]":
        """Apply ``A^H W`` for explicit radial data-space weights ``W``."""

        data = self._data_array(data_values)
        weights = self._radial_weights(data_weights)
        return self._adjoint_core(
            data * weights[:, None],
            phase_offsets=phase_offsets,
        )

    def data_inner_product(
        self,
        left: "ArrayLike",
        right: "ArrayLike",
        *,
        weighted: bool = False,
        data_weights: "ArrayLike | None" = None,
    ) -> complex:
        """Evaluate the declared discrete data-space inner product."""

        left_array = self._data_array(left)
        right_array = self._data_array(right)
        if weighted:
            weights = self._radial_weights(data_weights)
            right_array = right_array * weights[:, None]
        return complex(np.vdot(left_array, right_array))

    def adjoint_test(
        self,
        object_values: "ArrayLike",
        data_values: "ArrayLike",
        *,
        weighted: bool = False,
        data_weights: "ArrayLike | None" = None,
        phase_offsets: "ArrayLike | None" = None,
    ) -> float:
        """Return the normalized forward-adjoint dot-product mismatch."""

        object_array = self._object_array(object_values)
        data_array = self._data_array(data_values)
        forward = self.forward(object_array, phase_offsets=phase_offsets)
        if weighted:
            adjoint = self.adjoint_weighted(
                data_array,
                data_weights=data_weights,
                phase_offsets=phase_offsets,
            )
        else:
            adjoint = self.adjoint_euclidean(
                data_array,
                phase_offsets=phase_offsets,
            )
        left = self.data_inner_product(
            forward,
            data_array,
            weighted=weighted,
            data_weights=data_weights,
        )
        right = complex(np.vdot(object_array, adjoint))
        return normalized_adjoint_error(left, right)

    def adjoint_test_at_angles(
        self,
        object_values: "ArrayLike",
        data_values: "ArrayLike",
        angles: "ArrayLike",
        *,
        phase_offsets: "ArrayLike | None" = None,
    ) -> float:
        """Return the Euclidean dot-product mismatch for arbitrary angles."""

        object_array = self._object_array(object_values)
        angle_array = self._angles(angles)
        data_array = self._data_at_angles_array(data_values, angle_array.size)
        forward = self.forward_at_angles(
            object_array,
            angle_array,
            phase_offsets=phase_offsets,
        )
        adjoint = self.adjoint_at_angles_euclidean(
            data_array,
            angle_array,
            phase_offsets=phase_offsets,
        )
        left = complex(np.vdot(forward, data_array))
        right = complex(np.vdot(object_array, adjoint))
        return normalized_adjoint_error(left, right)
