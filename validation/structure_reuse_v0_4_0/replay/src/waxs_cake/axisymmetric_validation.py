"""Independent small-problem references for axisymmetric manifold validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .axisymmetric_manifold import AxisymmetricManifold
from .solvers import FormFactors, normalize_form_factors

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .histogram import BinnedStructure


@dataclass(frozen=True)
class ComplexErrorMetrics:
    """Complex-amplitude error metrics with low-amplitude phase masking."""

    relative_l2: float
    relative_linf: float
    phase_rms_rad: float
    max_absolute_error: float
    phase_mask_count: int
    phase_threshold: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class AnisotropicGaussianMixture:
    """Continuous Gaussian mixture with an analytic positive-sign Fourier transform."""

    coefficients: "ArrayLike"
    means: "ArrayLike"
    covariances: "ArrayLike"

    def __post_init__(self) -> None:
        coefficients = np.array(self.coefficients, dtype=np.complex128, copy=True)
        means = np.array(self.means, dtype=np.float64, copy=True)
        covariances = np.array(self.covariances, dtype=np.float64, copy=True)
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("coefficients must be a non-empty one-dimensional array")
        if means.shape != (coefficients.size, 3):
            raise ValueError("means must have shape (n_components, 3)")
        if covariances.shape != (coefficients.size, 3, 3):
            raise ValueError("covariances must have shape (n_components, 3, 3)")
        if not (
            np.all(np.isfinite(coefficients))
            and np.all(np.isfinite(means))
            and np.all(np.isfinite(covariances))
        ):
            raise ValueError("Gaussian parameters must contain only finite values")
        if not np.allclose(
            covariances,
            np.swapaxes(covariances, 1, 2),
            rtol=0.0,
            atol=1e-13,
        ):
            raise ValueError("covariances must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariances)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("covariances must be positive definite")

        coefficients.setflags(write=False)
        means.setflags(write=False)
        covariances.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "covariances", covariances)

    @property
    def n_components(self) -> int:
        return int(self.coefficients.size)

    def density(self, coords: "ArrayLike") -> "NDArray[np.complex128]":
        """Evaluate the continuous mixture density at Cartesian coordinates."""

        coords_array = np.asarray(coords, dtype=np.float64)
        if coords_array.ndim != 2 or coords_array.shape[1] != 3:
            raise ValueError("coords must have shape (n_points, 3)")
        if not np.all(np.isfinite(coords_array)):
            raise ValueError("coords must contain only finite values")

        result = np.zeros(coords_array.shape[0], dtype=np.complex128)
        for coefficient, mean, covariance in zip(
            self.coefficients,
            self.means,
            self.covariances,
            strict=True,
        ):
            delta = coords_array - mean
            precision = np.linalg.inv(covariance)
            exponent = -0.5 * np.einsum(
                "ni,ij,nj->n",
                delta,
                precision,
                delta,
                optimize=True,
            )
            result += coefficient * np.exp(exponent)
        return result

    def fourier_nodes(self, q_nodes: "ArrayLike") -> "NDArray[np.complex128]":
        """Evaluate the analytic transform ``integral rho(r) exp(i q.r) dr``."""

        nodes = np.asarray(q_nodes, dtype=np.float64)
        if nodes.ndim < 2 or nodes.shape[-1] != 3:
            raise ValueError("q_nodes must end with a Cartesian axis of length 3")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("q_nodes must contain only finite values")

        flat_nodes = nodes.reshape(-1, 3)
        result = np.zeros(flat_nodes.shape[0], dtype=np.complex128)
        normalization = (2.0 * np.pi) ** 1.5
        for coefficient, mean, covariance in zip(
            self.coefficients,
            self.means,
            self.covariances,
            strict=True,
        ):
            envelope = np.exp(
                -0.5
                * np.einsum(
                    "ni,ij,nj->n",
                    flat_nodes,
                    covariance,
                    flat_nodes,
                    optimize=True,
                )
            )
            shift_phase = np.exp(1j * (flat_nodes @ mean))
            result += (
                coefficient
                * normalization
                * np.sqrt(np.linalg.det(covariance))
                * envelope
                * shift_phase
            )
        return result.reshape(nodes.shape[:-1])

    def fourier_manifold(
        self,
        manifold: AxisymmetricManifold,
        phi: "ArrayLike",
    ) -> "NDArray[np.complex128]":
        """Evaluate the analytic transform on an axisymmetric target grid."""

        return self.fourier_nodes(manifold.target_nodes(phi))


def sample_gaussian_mixture_midpoint_grid(
    mixture: AnisotropicGaussianMixture,
    *,
    half_width: float,
    n_per_axis: int,
) -> tuple["NDArray[np.float64]", "NDArray[np.complex128]", float]:
    """Sample a mixture on a cubic midpoint grid and include voxel volume."""

    half_width = float(half_width)
    n_per_axis = int(n_per_axis)
    if not np.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("half_width must be finite and positive")
    if n_per_axis <= 0:
        raise ValueError("n_per_axis must be positive")

    spacing = 2.0 * half_width / n_per_axis
    axis = -half_width + (np.arange(n_per_axis, dtype=np.float64) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    voxel_volume = spacing**3
    weights = mixture.density(coords) * voxel_volume
    return coords, weights, voxel_volume


class PreparedFinufftAxisymmetricReference:
    """Optional FINUFFT type-3 plan on one axisymmetric target grid."""

    def __init__(
        self,
        coords: "ArrayLike",
        manifold: AxisymmetricManifold,
        phi: "ArrayLike",
        *,
        eps: float = 1e-9,
        nthreads: int = 1,
    ) -> None:
        try:
            import finufft
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("finufft is not installed") from exc

        coords_array = np.asarray(coords, dtype=np.float64)
        phi_array = np.asarray(phi, dtype=np.float64)
        if coords_array.ndim != 2 or coords_array.shape[1] != 3 or not coords_array.shape[0]:
            raise ValueError("coords must have shape (n_sources, 3) with at least one source")
        if not np.all(np.isfinite(coords_array)):
            raise ValueError("coords must contain only finite values")
        if phi_array.ndim != 1 or not phi_array.size or not np.all(np.isfinite(phi_array)):
            raise ValueError("phi must be a non-empty finite one-dimensional array")
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        nthreads = int(nthreads)
        if nthreads <= 0:
            raise ValueError("nthreads must be positive")

        self.coords = np.ascontiguousarray(coords_array)
        self.source_x = np.ascontiguousarray(coords_array[:, 0])
        self.source_y = np.ascontiguousarray(coords_array[:, 1])
        self.source_z = np.ascontiguousarray(coords_array[:, 2])
        self.manifold = manifold
        self.phi = np.ascontiguousarray(phi_array)
        self.eps = float(eps)
        self.nthreads = nthreads
        targets = manifold.target_nodes(phi_array).reshape(-1, 3)
        self.plan = finufft.Plan(
            3,
            3,
            n_trans=1,
            eps=self.eps,
            isign=1,
            dtype="complex128",
            nthreads=self.nthreads,
        )
        self.plan.setpts(
            self.source_x,
            self.source_y,
            self.source_z,
            np.ascontiguousarray(targets[:, 0]),
            np.ascontiguousarray(targets[:, 1]),
            np.ascontiguousarray(targets[:, 2]),
        )
        self._targets = np.ascontiguousarray(targets)
        self._target_x = np.ascontiguousarray(targets[:, 0])
        self._target_y = np.ascontiguousarray(targets[:, 1])
        self._target_z = np.ascontiguousarray(targets[:, 2])
        self._adjoint_plan = None

    @property
    def source_count(self) -> int:
        return int(self.coords.shape[0])

    @property
    def data_shape(self) -> tuple[int, int]:
        return (self.manifold.n_u, self.phi.size)

    def execute(self, source_weights: "ArrayLike") -> "NDArray[np.complex128]":
        weights = np.asarray(source_weights, dtype=np.complex128)
        if weights.shape != (self.source_count,):
            raise ValueError("source_weights must have one entry per source")
        if not np.all(np.isfinite(weights)):
            raise ValueError("source_weights must contain only finite values")
        values = self.plan.execute(np.ascontiguousarray(weights))
        return np.asarray(values, dtype=np.complex128).reshape(self.data_shape)

    def adjoint(self, data_values: "ArrayLike") -> "NDArray[np.complex128]":
        """Apply the Euclidean adjoint using an independently prepared type-3 plan."""

        data = np.asarray(data_values, dtype=np.complex128)
        if data.shape != self.data_shape or not np.all(np.isfinite(data)):
            raise ValueError(f"data_values must be finite with shape {self.data_shape}")
        if self._adjoint_plan is None:
            import finufft

            self._adjoint_plan = finufft.Plan(
                3,
                3,
                n_trans=1,
                eps=self.eps,
                isign=-1,
                dtype="complex128",
                nthreads=self.nthreads,
            )
            self._adjoint_plan.setpts(
                self._target_x,
                self._target_y,
                self._target_z,
                self.source_x,
                self.source_y,
                self.source_z,
            )
        values = self._adjoint_plan.execute(np.ascontiguousarray(data.ravel()))
        return np.asarray(values, dtype=np.complex128)


def complex_error_metrics(
    actual: "ArrayLike",
    reference: "ArrayLike",
    *,
    phase_threshold_fraction: float = 1e-3,
) -> ComplexErrorMetrics:
    """Measure complex error without assigning phase to near-zero samples."""

    actual_array = np.asarray(actual, dtype=np.complex128)
    reference_array = np.asarray(reference, dtype=np.complex128)
    if actual_array.shape != reference_array.shape:
        raise ValueError("actual and reference must have matching shapes")
    if actual_array.size == 0:
        raise ValueError("actual and reference must not be empty")
    if not np.all(np.isfinite(actual_array)) or not np.all(np.isfinite(reference_array)):
        raise ValueError("actual and reference must contain only finite values")
    if not np.isfinite(phase_threshold_fraction) or phase_threshold_fraction < 0.0:
        raise ValueError("phase_threshold_fraction must be finite and non-negative")

    difference = actual_array - reference_array
    reference_l2 = float(np.linalg.norm(reference_array.ravel()))
    reference_linf = float(np.max(np.abs(reference_array)))
    if reference_l2 == 0.0 or reference_linf == 0.0:
        raise ValueError("reference must have nonzero norm")

    phase_threshold = phase_threshold_fraction * reference_linf
    phase_mask = np.abs(reference_array) > phase_threshold
    if not np.any(phase_mask):
        raise ValueError("phase threshold removed every reference sample")
    phase_delta = np.angle(actual_array[phase_mask] * np.conj(reference_array[phase_mask]))

    return ComplexErrorMetrics(
        relative_l2=float(np.linalg.norm(difference.ravel()) / reference_l2),
        relative_linf=float(np.max(np.abs(difference)) / reference_linf),
        phase_rms_rad=float(np.sqrt(np.mean(phase_delta * phase_delta))),
        max_absolute_error=float(np.max(np.abs(difference))),
        phase_mask_count=int(np.count_nonzero(phase_mask)),
        phase_threshold=float(phase_threshold),
    )


def direct_axisymmetric_amplitude(
    coords: "ArrayLike",
    manifold: AxisymmetricManifold,
    phi: "ArrayLike",
    *,
    elements: Sequence[str] | None = None,
    form_factors: FormFactors = None,
    source_weights: "ArrayLike | None" = None,
) -> "NDArray[np.complex128]":
    """Evaluate an independent Cartesian exponent sum.

    This reference uses no cylindrical harmonic, Bessel function, angular FFT,
    or prepared kernel. It is intended for small correctness problems.
    """

    coords_array = np.asarray(coords, dtype=np.float64)
    phi_array = np.asarray(phi, dtype=np.float64)
    if coords_array.ndim != 2 or coords_array.shape[1] != 3:
        raise ValueError("coords must have shape (n_sources, 3)")
    if coords_array.shape[0] == 0:
        raise ValueError("coords must contain at least one source")
    if not np.all(np.isfinite(coords_array)):
        raise ValueError("coords must contain only finite values")
    if phi_array.ndim != 1 or phi_array.size == 0:
        raise ValueError("phi must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(phi_array)):
        raise ValueError("phi must contain only finite values")

    n_sources = coords_array.shape[0]
    if elements is None:
        source_elements = np.full(n_sources, "X", dtype=object)
    else:
        source_elements = np.asarray(list(elements), dtype=object)
        if source_elements.shape != (n_sources,):
            raise ValueError("elements must have one entry per source")

    if source_weights is None:
        weights = np.ones(n_sources, dtype=np.complex128)
    else:
        weights = np.asarray(source_weights, dtype=np.complex128)
        if weights.shape != (n_sources,):
            raise ValueError("source_weights must have one entry per source")
        if not np.all(np.isfinite(weights)):
            raise ValueError("source_weights must contain only finite values")

    ordered_elements = tuple(dict.fromkeys(str(element) for element in source_elements))
    element_to_index = {element: index for index, element in enumerate(ordered_elements)}
    source_element_indices = np.array(
        [element_to_index[str(element)] for element in source_elements],
        dtype=np.intp,
    )
    form_factor_values = normalize_form_factors(
        ordered_elements,
        manifold.q_norm,
        form_factors,
    )

    cos_phi = np.cos(phi_array)
    sin_phi = np.sin(phi_array)
    transverse_projection = (
        coords_array[:, 0, None] * cos_phi[None, :]
        + coords_array[:, 1, None] * sin_phi[None, :]
    )
    z = coords_array[:, 2, None]
    amplitude = np.empty((manifold.n_u, phi_array.size), dtype=np.complex128)
    for index in range(manifold.n_u):
        coefficients = weights * form_factor_values[source_element_indices, index]
        phase = manifold.q_perp[index] * transverse_projection + manifold.q_z[index] * z
        amplitude[index] = np.sum(coefficients[:, None] * np.exp(1j * phase), axis=0)
    return amplitude


def direct_axisymmetric_adjoint(
    coords: "ArrayLike",
    manifold: AxisymmetricManifold,
    phi: "ArrayLike",
    data_values: "ArrayLike",
    *,
    elements: Sequence[str] | None = None,
    form_factors: FormFactors = None,
    data_weights: "ArrayLike | None" = None,
) -> "NDArray[np.complex128]":
    """Apply an independent Cartesian conjugate exponent sum."""

    coords_array = np.asarray(coords, dtype=np.float64)
    phi_array = np.asarray(phi, dtype=np.float64)
    data = np.asarray(data_values, dtype=np.complex128)
    if coords_array.ndim != 2 or coords_array.shape[1] != 3:
        raise ValueError("coords must have shape (n_sources, 3)")
    if not np.all(np.isfinite(coords_array)):
        raise ValueError("coords must contain only finite values")
    if phi_array.ndim != 1 or not phi_array.size or not np.all(np.isfinite(phi_array)):
        raise ValueError("phi must be a non-empty finite one-dimensional array")
    if data.shape != (manifold.n_u, phi_array.size):
        raise ValueError("data_values shape must match the manifold and phi grid")
    if not np.all(np.isfinite(data)):
        raise ValueError("data_values must contain only finite values")

    n_sources = coords_array.shape[0]
    if elements is None:
        source_elements = np.full(n_sources, "X", dtype=object)
    else:
        source_elements = np.asarray(list(elements), dtype=object)
        if source_elements.shape != (n_sources,):
            raise ValueError("elements must have one entry per source")
    ordered_elements = tuple(dict.fromkeys(str(element) for element in source_elements))
    element_to_index = {element: index for index, element in enumerate(ordered_elements)}
    source_element_indices = np.array(
        [element_to_index[str(element)] for element in source_elements],
        dtype=np.intp,
    )
    form_factor_values = normalize_form_factors(
        ordered_elements,
        manifold.q_norm,
        form_factors,
    )

    if data_weights is None:
        weighted_data = data
    else:
        weights = np.asarray(data_weights, dtype=np.float64)
        if weights.shape != (manifold.n_u,):
            raise ValueError("data_weights must have one entry per u sample")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("data_weights must be finite and non-negative")
        weighted_data = data * weights[:, None]

    cos_phi = np.cos(phi_array)
    sin_phi = np.sin(phi_array)
    transverse_projection = (
        coords_array[:, 0, None] * cos_phi[None, :]
        + coords_array[:, 1, None] * sin_phi[None, :]
    )
    z = coords_array[:, 2, None]
    output = np.zeros(n_sources, dtype=np.complex128)
    for index in range(manifold.n_u):
        phase = manifold.q_perp[index] * transverse_projection + manifold.q_z[index] * z
        angular_sum = np.sum(np.exp(-1j * phase) * weighted_data[index][None, :], axis=1)
        output += np.conj(form_factor_values[source_element_indices, index]) * angular_sum
    return output


def binned_structure_grid(
    binned: "BinnedStructure",
) -> tuple["NDArray[np.float64]", tuple[str, ...]]:
    """Expand every cylindrical histogram cell into a Cartesian source grid."""

    shape = np.asarray(binned.hist).shape
    expected = (
        len(binned.elements),
        len(binned.r_centers),
        len(binned.z_centers),
        len(binned.beta_centers),
    )
    if shape != expected:
        raise ValueError("binned histogram shape does not match its coordinates")
    element_index, r_index, z_index, beta_index = np.indices(expected).reshape(4, -1)
    radius = np.asarray(binned.r_centers, dtype=np.float64)[r_index]
    beta = np.asarray(binned.beta_centers, dtype=np.float64)[beta_index]
    coords = np.column_stack(
        (
            radius * np.cos(beta),
            radius * np.sin(beta),
            np.asarray(binned.z_centers, dtype=np.float64)[z_index],
        )
    )
    elements = tuple(binned.elements[index] for index in element_index)
    return coords, elements


def binned_structure_sources(
    binned: "BinnedStructure",
) -> tuple["NDArray[np.float64]", tuple[str, ...], "NDArray[np.complex128]"]:
    """Expand nonzero histogram bins into Cartesian point sources.

    The expansion defines the exact discrete object represented by the prepared
    circular solver. Comparing this object avoids mixing histogram placement
    error into the manifold-operator error.
    """

    histogram = np.asarray(binned.hist)
    if histogram.ndim != 4:
        raise ValueError("binned.hist must have shape (element, r, z, beta)")
    nonzero = np.nonzero(histogram)
    if nonzero[0].size == 0:
        raise ValueError("binned.hist must contain at least one nonzero source")

    element_index, r_index, z_index, beta_index = nonzero
    radius = np.asarray(binned.r_centers, dtype=np.float64)[r_index]
    beta = np.asarray(binned.beta_centers, dtype=np.float64)[beta_index]
    coords = np.column_stack(
        (
            radius * np.cos(beta),
            radius * np.sin(beta),
            np.asarray(binned.z_centers, dtype=np.float64)[z_index],
        )
    )
    elements = tuple(binned.elements[index] for index in element_index)
    weights = np.asarray(histogram[nonzero], dtype=np.complex128)
    return coords, elements, weights
