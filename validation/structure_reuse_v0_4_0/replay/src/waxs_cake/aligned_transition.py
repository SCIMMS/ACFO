"""Prepared polarization and aligned-ensemble contractions for transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import eval_legendre

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _vectors(values: "ArrayLike", *, name: str) -> "NDArray[np.float64]":
    result = np.asarray(values, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and end in dimension three")
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"{name} must be nonzero")
    return result / norms


def _unit_vector(value: "ArrayLike", *, name: str) -> "NDArray[np.float64]":
    result = _vectors(value, name=name)
    if result.shape != (3,):
        raise ValueError(f"{name} must be a three-vector")
    return result


def _orientation_quadrature(
    alignment_axis: "ArrayLike",
    concentration: float,
    n_theta: int,
    n_phi: int,
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    axis = _unit_vector(alignment_axis, name="alignment_axis")
    concentration = float(concentration)
    n_theta = int(n_theta)
    n_phi = int(n_phi)
    if not np.isfinite(concentration) or concentration < 0.0:
        raise ValueError("concentration must be finite and non-negative")
    if n_theta < 2 or n_phi < 3:
        raise ValueError("orientation quadrature is too small")
    cosine, polar_weight = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0 * np.pi * np.arange(n_phi, dtype=np.float64) / n_phi
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    orientations = np.stack(
        np.broadcast_arrays(
            sine[:, None] * np.cos(phi)[None, :],
            sine[:, None] * np.sin(phi)[None, :],
            cosine[:, None],
        ),
        axis=-1,
    ).reshape(-1, 3)
    base_weight = np.broadcast_to(
        polar_weight[:, None] * (2.0 * np.pi / n_phi),
        (n_theta, n_phi),
    ).reshape(-1)
    exponent = concentration * (orientations @ axis) ** 2
    exponent -= np.max(exponent)
    weights = base_weight * np.exp(exponent)
    weights /= np.sum(weights)
    return orientations, weights


def watson_alignment_moments(
    concentration: float,
    *,
    degrees: "ArrayLike" = (2, 4),
    n_theta: int = 96,
    n_phi: int = 192,
) -> "NDArray[np.float64]":
    """Return ``<P_l(n.a)>`` for an apolar Watson distribution."""

    orientations, weights = _orientation_quadrature(
        (0.0, 0.0, 1.0), concentration, n_theta, n_phi
    )
    degree_values = np.asarray(degrees, dtype=np.int64)
    if degree_values.ndim != 1 or np.any(degree_values < 0):
        raise ValueError("degrees must be a one-dimensional non-negative array")
    cosine = orientations[:, 2]
    return np.asarray(
        [np.sum(weights * eval_legendre(int(degree), cosine)) for degree in degree_values],
        dtype=np.float64,
    )


def _polarization_weight(
    orientations: "NDArray[np.float64]",
    outgoing_directions: "NDArray[np.float64]",
    incident_polarization: "NDArray[np.float64]",
    tensor_perpendicular: complex,
    tensor_parallel: complex,
) -> "NDArray[np.float64]":
    projection = orientations @ incident_polarization
    response = (
        complex(tensor_perpendicular) * incident_polarization[None, :]
        + (complex(tensor_parallel) - complex(tensor_perpendicular))
        * orientations
        * projection[:, None]
    )
    response_norm = np.sum(np.abs(response) ** 2, axis=-1)
    longitudinal = outgoing_directions @ response.T
    result = response_norm[None, :] - np.abs(longitudinal) ** 2
    return np.maximum(result.real, 0.0)


@dataclass(frozen=True)
class PreparedAlignedTransitionContraction:
    r"""Prepared ``L,L'`` kernel for polarization-weighted ensemble intensity."""

    degrees: "NDArray[np.int64]"
    detector_shape: tuple[int, ...]
    pair_left: "NDArray[np.int64]"
    pair_right: "NDArray[np.int64]"
    pair_factor: "NDArray[np.float64]"
    kernel_pairs: "NDArray[np.float64]"
    alignment_moments: "NDArray[np.float64]"
    orientation_count: int

    @classmethod
    def build(
        cls,
        degrees: "ArrayLike",
        q_directions: "ArrayLike",
        outgoing_directions: "ArrayLike",
        *,
        incident_polarization: "ArrayLike",
        alignment_axis: "ArrayLike",
        concentration: float,
        tensor_perpendicular: complex = 1.0,
        tensor_parallel: complex = 1.0,
        n_theta: int = 24,
        n_phi: int = 48,
        detector_chunk_size: int = 256,
    ) -> "PreparedAlignedTransitionContraction":
        """Prepare the orientation average for a fixed detector geometry."""

        degree_values = np.asarray(degrees, dtype=np.int64)
        if degree_values.ndim != 1 or degree_values.size == 0:
            raise ValueError("degrees must be a nonempty one-dimensional array")
        degree_values = np.unique(degree_values)
        if np.any(degree_values < 0):
            raise ValueError("degrees must be non-negative")
        q_unit = _vectors(q_directions, name="q_directions")
        outgoing_unit = _vectors(outgoing_directions, name="outgoing_directions")
        if q_unit.shape != outgoing_unit.shape:
            raise ValueError("q and outgoing directions must have matching shape")
        incident = _unit_vector(
            incident_polarization,
            name="incident_polarization",
        )
        orientations, orientation_weights = _orientation_quadrature(
            alignment_axis,
            concentration,
            n_theta,
            n_phi,
        )
        flat_q = q_unit.reshape(-1, 3)
        flat_outgoing = outgoing_unit.reshape(-1, 3)
        n_detector = flat_q.shape[0]
        n_degree = degree_values.size
        kernel = np.empty((n_detector, n_degree, n_degree), dtype=np.float64)
        detector_chunk_size = int(detector_chunk_size)
        if detector_chunk_size <= 0:
            raise ValueError("detector_chunk_size must be positive")
        normalization = np.sqrt((2.0 * degree_values + 1.0) / (4.0 * np.pi))
        for start in range(0, n_detector, detector_chunk_size):
            stop = min(start + detector_chunk_size, n_detector)
            cosine = flat_q[start:stop] @ orientations.T
            basis = np.stack(
                [
                    normalization[index] * eval_legendre(int(degree), cosine)
                    for index, degree in enumerate(degree_values)
                ],
                axis=-1,
            )
            polarization = _polarization_weight(
                orientations,
                flat_outgoing[start:stop],
                incident,
                tensor_perpendicular,
                tensor_parallel,
            )
            combined_weight = polarization * orientation_weights[None, :]
            kernel[start:stop] = np.einsum(
                "po,pol,pom->plm",
                combined_weight,
                basis,
                basis,
                optimize=True,
            )
        alignment_moments = watson_alignment_moments(
            concentration,
            degrees=(2, 4),
            n_theta=max(64, 2 * int(n_theta)),
            n_phi=max(128, 2 * int(n_phi)),
        )
        kernel = kernel.reshape(q_unit.shape[:-1] + (n_degree, n_degree))
        pair_left, pair_right = np.triu_indices(n_degree)
        pair_factor = np.where(pair_left == pair_right, 1.0, 2.0)
        kernel_pairs = np.asarray(
            kernel[..., pair_left, pair_right],
            dtype=np.float64,
        )
        for array in (
            degree_values,
            pair_left,
            pair_right,
            pair_factor,
            kernel_pairs,
            alignment_moments,
        ):
            array.setflags(write=False)
        return cls(
            degrees=degree_values,
            detector_shape=q_unit.shape[:-1],
            pair_left=pair_left,
            pair_right=pair_right,
            pair_factor=pair_factor,
            kernel_pairs=kernel_pairs,
            alignment_moments=alignment_moments,
            orientation_count=orientations.shape[0],
        )

    @property
    def prepared_bytes(self) -> int:
        """Return storage owned by the small-degree detector kernel."""

        return int(
            self.degrees.nbytes
            + self.pair_left.nbytes
            + self.pair_right.nbytes
            + self.pair_factor.nbytes
            + self.kernel_pairs.nbytes
            + self.alignment_moments.nbytes
        )

    def intensity(self, degree_amplitudes: "ArrayLike") -> "NDArray[np.float64]":
        """Contract complex degree amplitudes into ensemble-averaged intensity."""

        amplitudes = np.asarray(degree_amplitudes, dtype=np.complex128)
        if amplitudes.ndim < 1 or amplitudes.shape[-1] != self.degrees.size:
            raise ValueError("degree_amplitudes must end in the prepared degree count")
        pair_products = (
            amplitudes[..., self.pair_left]
            * np.conj(amplitudes[..., self.pair_right])
        ).real
        pair_products *= self.pair_factor
        return np.asarray(
            np.sum(pair_products * self.kernel_pairs, axis=-1),
            dtype=np.float64,
        )

    def intensity_radial(
        self,
        radial_degree_amplitudes: "ArrayLike",
    ) -> "NDArray[np.float64]":
        """Evaluate amplitudes shared over every detector angle at fixed q.

        This avoids repeating the molecular ``l,l'`` products around each
        detector ring and is the optimized path for radial orbital amplitudes.
        The last axes of the input must be ``(radial q, molecular l)``.
        """

        amplitudes = np.asarray(radial_degree_amplitudes, dtype=np.complex128)
        expected = (self.detector_shape[0], self.degrees.size)
        if amplitudes.ndim < 2 or amplitudes.shape[-2:] != expected:
            raise ValueError(
                "radial_degree_amplitudes must end in "
                f"(n_q, n_degree)={expected}"
            )
        pair_products = (
            amplitudes[..., self.pair_left]
            * np.conj(amplitudes[..., self.pair_right])
        ).real
        pair_products *= self.pair_factor
        flat_kernel = self.kernel_pairs.reshape(
            self.detector_shape[0],
            -1,
            self.pair_left.size,
        )
        flat_result = np.einsum(
            "...qp,qdp->...qd",
            pair_products,
            flat_kernel,
            optimize=True,
        )
        return np.asarray(
            flat_result.reshape(amplitudes.shape[:-2] + self.detector_shape),
            dtype=np.float64,
        )

    def prepare_weighted_least_squares(
        self,
        data: "ArrayLike",
        *,
        weights: "ArrayLike" = 1.0,
    ) -> "PreparedPairIntensityLeastSquares":
        """Contract a fixed detector dataset into pair-feature statistics."""

        return PreparedPairIntensityLeastSquares.build(
            self.degrees,
            self.detector_shape,
            self.pair_left,
            self.pair_right,
            self.pair_factor,
            self.kernel_pairs.reshape(
                self.detector_shape[0],
                -1,
                self.pair_left.size,
            ),
            data,
            weights=weights,
        )


@dataclass(frozen=True)
class PreparedAxisymmetricLegendreADM:
    r"""Prepared scalar-scattering ADM baseline for an apolar Watson ensemble.

    The molecular-frame amplitude is assumed to have the axisymmetric form

    ``sum_l amplitude_l(q) sqrt((2l+1)/(4 pi)) P_l(q_hat.n)``.

    Products of the molecular Legendre functions are reduced once to even
    laboratory-frame degrees ``J``.  The Watson orientation average then uses
    the addition-theorem identity

    ``<P_J(q_hat.n)> = <P_J(a.n)> P_J(q_hat.a)``.

    This is the optimized ADM/Legendre comparator for non-resonant scalar
    scattering.  Unlike :class:`PreparedAlignedTransitionContraction`, it does
    not perform an orientation quadrature for every detector pixel during
    setup and stores no per-pixel ``l,l'`` matrix.
    """

    degrees: "NDArray[np.int64]"
    adm_degrees: "NDArray[np.int64]"
    detector_shape: tuple[int, ...]
    detector_pair_kernel: "NDArray[np.float64]"
    pair_left: "NDArray[np.int64]"
    pair_right: "NDArray[np.int64]"
    pair_factor: "NDArray[np.float64]"
    pair_coupling: "NDArray[np.float64]"
    alignment_moments: "NDArray[np.float64]"

    @classmethod
    def build(
        cls,
        degrees: "ArrayLike",
        q_directions: "ArrayLike",
        outgoing_directions: "ArrayLike",
        *,
        incident_polarization: "ArrayLike",
        alignment_axis: "ArrayLike",
        concentration: float,
        n_alignment: int = 64,
    ) -> "PreparedAxisymmetricLegendreADM":
        """Prepare an analytic axisymmetric ADM/Legendre detector operator."""

        degree_values = np.asarray(degrees, dtype=np.int64)
        if degree_values.ndim != 1 or degree_values.size == 0:
            raise ValueError("degrees must be a nonempty one-dimensional array")
        degree_values = np.unique(degree_values)
        if np.any(degree_values < 0):
            raise ValueError("degrees must be non-negative")
        q_unit = _vectors(q_directions, name="q_directions")
        outgoing_unit = _vectors(outgoing_directions, name="outgoing_directions")
        if q_unit.shape != outgoing_unit.shape:
            raise ValueError("q and outgoing directions must have matching shape")
        if q_unit.ndim < 2:
            raise ValueError("q directions must contain a leading radial axis")
        incident = _unit_vector(
            incident_polarization,
            name="incident_polarization",
        )
        axis = _unit_vector(alignment_axis, name="alignment_axis")
        concentration = float(concentration)
        n_alignment = int(n_alignment)
        if not np.isfinite(concentration) or concentration < 0.0:
            raise ValueError("concentration must be finite and non-negative")
        if n_alignment < 2:
            raise ValueError("n_alignment must be at least two")

        maximum_degree = int(np.max(degree_values))
        adm_degrees = np.arange(0, 2 * maximum_degree + 1, 2, dtype=np.int64)

        # A one-dimensional Watson integral is sufficient because the
        # distribution is axisymmetric and apolar.  Subtracting the largest
        # exponent keeps the normalization stable at strong alignment.
        alignment_cosine, alignment_weight = np.polynomial.legendre.leggauss(
            n_alignment
        )
        exponent = concentration * alignment_cosine**2
        exponent -= np.max(exponent)
        watson_weight = alignment_weight * np.exp(exponent)
        watson_weight /= np.sum(watson_weight)
        alignment_moments = np.asarray(
            [
                np.sum(
                    watson_weight
                    * eval_legendre(int(degree), alignment_cosine)
                )
                for degree in adm_degrees
            ],
            dtype=np.float64,
        )

        # The triple-Legendre products are polynomials, so this modest Gauss
        # order evaluates the product coefficients to roundoff for all
        # prepared degrees.
        product_order = max(16, 2 * maximum_degree + 2)
        product_cosine, product_weight = np.polynomial.legendre.leggauss(
            product_order
        )
        molecular_basis = np.stack(
            [
                eval_legendre(int(degree), product_cosine)
                for degree in degree_values
            ],
            axis=-1,
        )
        adm_basis = np.stack(
            [
                eval_legendre(int(degree), product_cosine)
                for degree in adm_degrees
            ],
            axis=-1,
        )
        normalization = np.sqrt((2.0 * degree_values + 1.0) / (4.0 * np.pi))
        pair_left, pair_right = np.triu_indices(degree_values.size)
        pair_factor = np.where(pair_left == pair_right, 1.0, 2.0)
        pair_coupling = np.empty(
            (pair_left.size, adm_degrees.size),
            dtype=np.float64,
        )
        for pair_index, (left, right) in enumerate(
            zip(pair_left, pair_right, strict=True)
        ):
            product = molecular_basis[:, left] * molecular_basis[:, right]
            pair_coupling[pair_index] = (
                normalization[left]
                * normalization[right]
                * (2.0 * adm_degrees + 1.0)
                * 0.5
                * np.sum(
                    product_weight[:, None] * product[:, None] * adm_basis,
                    axis=0,
                )
            )

        cosine = q_unit @ axis
        detector_adm = np.stack(
            [eval_legendre(int(degree), cosine) for degree in adm_degrees],
            axis=-1,
        )
        polarization = 1.0 - np.abs(outgoing_unit @ incident) ** 2
        angular_kernel = (
            polarization[..., None]
            * detector_adm
            * alignment_moments
        ).reshape(q_unit.shape[0], -1, adm_degrees.size)
        detector_pair_kernel = np.einsum(
            "pj,qdj->qdp",
            pair_coupling,
            angular_kernel,
            optimize=True,
        )

        for array in (
            degree_values,
            adm_degrees,
            detector_pair_kernel,
            pair_left,
            pair_right,
            pair_factor,
            pair_coupling,
            alignment_moments,
        ):
            array.setflags(write=False)
        return cls(
            degrees=degree_values,
            adm_degrees=adm_degrees,
            detector_shape=q_unit.shape[:-1],
            detector_pair_kernel=detector_pair_kernel,
            pair_left=pair_left,
            pair_right=pair_right,
            pair_factor=pair_factor,
            pair_coupling=pair_coupling,
            alignment_moments=alignment_moments,
        )

    @property
    def prepared_bytes(self) -> int:
        """Return storage owned by the ADM coupling and detector tables."""

        return int(
            self.degrees.nbytes
            + self.adm_degrees.nbytes
            + self.detector_pair_kernel.nbytes
            + self.pair_left.nbytes
            + self.pair_right.nbytes
            + self.pair_factor.nbytes
            + self.pair_coupling.nbytes
            + self.alignment_moments.nbytes
        )

    def intensity(
        self,
        radial_degree_amplitudes: "ArrayLike",
    ) -> "NDArray[np.float64]":
        """Evaluate states whose last axes are ``(radial q, molecular l)``."""

        amplitudes = np.asarray(radial_degree_amplitudes, dtype=np.complex128)
        expected = (self.detector_shape[0], self.degrees.size)
        if amplitudes.ndim < 2 or amplitudes.shape[-2:] != expected:
            raise ValueError(
                "radial_degree_amplitudes must end in "
                f"(n_q, n_degree)={expected}"
            )
        pair_products = (
            amplitudes[..., self.pair_left]
            * np.conj(amplitudes[..., self.pair_right])
        ).real
        pair_products *= self.pair_factor
        flat_result = np.einsum(
            "...qp,qdp->...qd",
            pair_products,
            self.detector_pair_kernel,
            optimize=True,
        )
        return np.asarray(
            flat_result.reshape(amplitudes.shape[:-2] + self.detector_shape),
            dtype=np.float64,
        )

    def prepare_weighted_least_squares(
        self,
        data: "ArrayLike",
        *,
        weights: "ArrayLike" = 1.0,
    ) -> "PreparedPairIntensityLeastSquares":
        """Contract a fixed detector dataset into pair-feature statistics."""

        return PreparedPairIntensityLeastSquares.build(
            self.degrees,
            self.detector_shape,
            self.pair_left,
            self.pair_right,
            self.pair_factor,
            self.detector_pair_kernel,
            data,
            weights=weights,
        )


@dataclass(frozen=True)
class PreparedPairIntensityLeastSquares:
    r"""Sufficient statistics for a weighted detector least-squares loss.

    For a detector model ``I_q = D_q x_q`` in the small real pair-feature
    vector ``x_q``, this object stores

    ``G_q = D_q.T W_q D_q`` and ``h_q = D_q.T W_q data_q``.

    It evaluates the exact loss without materializing the detector image:

    ``sum_q x_q.T G_q x_q - 2 h_q.T x_q + data.T W data``.
    """

    degrees: "NDArray[np.int64]"
    detector_shape: tuple[int, ...]
    pair_left: "NDArray[np.int64]"
    pair_right: "NDArray[np.int64]"
    pair_factor: "NDArray[np.float64]"
    gram: "NDArray[np.float64]"
    data_projection: "NDArray[np.float64]"
    data_norm: float

    @classmethod
    def build(
        cls,
        degrees: "ArrayLike",
        detector_shape: tuple[int, ...],
        pair_left: "ArrayLike",
        pair_right: "ArrayLike",
        pair_factor: "ArrayLike",
        detector_pair_kernel: "ArrayLike",
        data: "ArrayLike",
        *,
        weights: "ArrayLike" = 1.0,
    ) -> "PreparedPairIntensityLeastSquares":
        """Prepare exact diagonal-weighted least-squares statistics."""

        degree_values = np.asarray(degrees, dtype=np.int64)
        left = np.asarray(pair_left, dtype=np.int64)
        right = np.asarray(pair_right, dtype=np.int64)
        factor = np.asarray(pair_factor, dtype=np.float64)
        kernel = np.asarray(detector_pair_kernel, dtype=np.float64)
        detector_shape = tuple(int(value) for value in detector_shape)
        if len(detector_shape) == 0:
            raise ValueError("detector_shape must contain a radial axis")
        n_q = detector_shape[0]
        detector_count_per_q = int(np.prod(detector_shape[1:], dtype=np.int64))
        expected_kernel_shape = (n_q, detector_count_per_q, left.size)
        if kernel.shape != expected_kernel_shape:
            raise ValueError(
                "detector_pair_kernel must have shape "
                f"{expected_kernel_shape}"
            )
        if not (
            left.shape == right.shape == factor.shape
            and left.ndim == 1
            and np.all(left >= 0)
            and np.all(right < degree_values.size)
        ):
            raise ValueError("pair arrays are inconsistent with degrees")
        observed = np.asarray(data, dtype=np.float64)
        if observed.shape != detector_shape or not np.all(np.isfinite(observed)):
            raise ValueError("data must be finite and match detector_shape")
        weight_values = np.asarray(weights, dtype=np.float64)
        try:
            weight_values = np.broadcast_to(weight_values, detector_shape)
        except ValueError as exc:
            raise ValueError("weights must broadcast to detector_shape") from exc
        if not np.all(np.isfinite(weight_values)) or np.any(weight_values < 0.0):
            raise ValueError("weights must be finite and non-negative")
        flat_data = observed.reshape(n_q, detector_count_per_q)
        flat_weight = weight_values.reshape(n_q, detector_count_per_q)
        gram = np.einsum(
            "qdp,qd,qdr->qpr",
            kernel,
            flat_weight,
            kernel,
            optimize=True,
        )
        data_projection = np.einsum(
            "qdp,qd,qd->qp",
            kernel,
            flat_weight,
            flat_data,
            optimize=True,
        )
        data_norm = float(np.sum(flat_weight * flat_data**2))
        for array in (
            degree_values,
            left,
            right,
            factor,
            gram,
            data_projection,
        ):
            array.setflags(write=False)
        return cls(
            degrees=degree_values,
            detector_shape=detector_shape,
            pair_left=left,
            pair_right=right,
            pair_factor=factor,
            gram=gram,
            data_projection=data_projection,
            data_norm=data_norm,
        )

    @property
    def prepared_bytes(self) -> int:
        """Return storage used by the sufficient statistics."""

        return int(
            self.degrees.nbytes
            + self.pair_left.nbytes
            + self.pair_right.nbytes
            + self.pair_factor.nbytes
            + self.gram.nbytes
            + self.data_projection.nbytes
        )

    def pair_features(
        self,
        radial_degree_amplitudes: "ArrayLike",
    ) -> "NDArray[np.float64]":
        """Return real Hermitian pair features for each radial q."""

        amplitudes = np.asarray(radial_degree_amplitudes, dtype=np.complex128)
        expected = (self.detector_shape[0], self.degrees.size)
        if amplitudes.ndim < 2 or amplitudes.shape[-2:] != expected:
            raise ValueError(
                "radial_degree_amplitudes must end in "
                f"(n_q, n_degree)={expected}"
            )
        features = (
            amplitudes[..., self.pair_left]
            * np.conj(amplitudes[..., self.pair_right])
        ).real
        features *= self.pair_factor
        return np.asarray(features, dtype=np.float64)

    def loss(
        self,
        radial_degree_amplitudes: "ArrayLike",
    ) -> "NDArray[np.float64]":
        """Evaluate the exact detector weighted least-squares loss."""

        features = self.pair_features(radial_degree_amplitudes)
        quadratic = np.einsum(
            "...qp,qpr,...qr->...",
            features,
            self.gram,
            features,
            optimize=True,
        )
        linear = np.einsum(
            "...qp,qp->...",
            features,
            self.data_projection,
            optimize=True,
        )
        return np.asarray(
            quadratic - 2.0 * linear + self.data_norm,
            dtype=np.float64,
        )


def direct_aligned_transition_intensity(
    degree_amplitudes: "ArrayLike",
    degrees: "ArrayLike",
    q_directions: "ArrayLike",
    outgoing_directions: "ArrayLike",
    *,
    incident_polarization: "ArrayLike",
    alignment_axis: "ArrayLike",
    concentration: float,
    tensor_perpendicular: complex = 1.0,
    tensor_parallel: complex = 1.0,
    n_theta: int = 64,
    n_phi: int = 128,
    detector_chunk_size: int = 128,
) -> "NDArray[np.float64]":
    """Direct high-order orientation average used as an independent oracle."""

    degree_values = np.asarray(degrees, dtype=np.int64)
    amplitudes = np.asarray(degree_amplitudes, dtype=np.complex128)
    q_unit = _vectors(q_directions, name="q_directions")
    outgoing_unit = _vectors(outgoing_directions, name="outgoing_directions")
    if q_unit.shape != outgoing_unit.shape:
        raise ValueError("q and outgoing directions must have matching shape")
    if amplitudes.shape != q_unit.shape[:-1] + (degree_values.size,):
        raise ValueError("degree_amplitudes must match detector shape and degree count")
    incident = _unit_vector(incident_polarization, name="incident_polarization")
    orientations, weights = _orientation_quadrature(
        alignment_axis,
        concentration,
        n_theta,
        n_phi,
    )
    flat_q = q_unit.reshape(-1, 3)
    flat_outgoing = outgoing_unit.reshape(-1, 3)
    flat_amplitudes = amplitudes.reshape(-1, degree_values.size)
    normalization = np.sqrt((2.0 * degree_values + 1.0) / (4.0 * np.pi))
    result = np.empty(flat_q.shape[0], dtype=np.float64)
    for start in range(0, flat_q.shape[0], int(detector_chunk_size)):
        stop = min(start + int(detector_chunk_size), flat_q.shape[0])
        cosine = flat_q[start:stop] @ orientations.T
        basis = np.stack(
            [
                normalization[index] * eval_legendre(int(degree), cosine)
                for index, degree in enumerate(degree_values)
            ],
            axis=-1,
        )
        molecular_amplitude = np.einsum(
            "pl,pol->po",
            flat_amplitudes[start:stop],
            basis,
        )
        polarization = _polarization_weight(
            orientations,
            flat_outgoing[start:stop],
            incident,
            tensor_perpendicular,
            tensor_parallel,
        )
        result[start:stop] = np.sum(
            weights[None, :] * polarization * np.abs(molecular_amplitude) ** 2,
            axis=-1,
        )
    return result.reshape(q_unit.shape[:-1])
