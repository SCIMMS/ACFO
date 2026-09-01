"""Measure-correct cylindrical reconstruction states for ODT prototypes.

The prepared ACFO ODT operator uses *integrated* cylindrical coefficients

    c_i = V_i f_i,  V_i = rho_i * dr * dbeta * dz,

where ``f`` is the sampled physical scattering potential.  Optimizing ``c``
with an ordinary Euclidean norm therefore does not optimize the physical
field in the cylindrical volume measure.  This module exposes the symmetric
state

    u_i = sqrt(V_i) f_i,

whose Euclidean inner product is the midpoint approximation of the physical
``L2(rho dr dbeta dz)`` inner product.

The implementation is intentionally NumPy-only and independent of the
production ACFO plans.  ``MeasureCorrectOperator`` can wrap any coefficient
operator whose adjoint is the Euclidean adjoint with respect to ``c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


Array = np.ndarray


def _require_shape(value: Array, shape: tuple[int, int, int], name: str) -> Array:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


@dataclass(frozen=True)
class CylindricalGrid:
    """Uniform cell-centred ``(rho, z, beta)`` grid with no node at the axis."""

    r_axis: Array
    z_axis: Array
    beta_axis: Array
    dr: float
    dz: float
    dbeta: float
    volume_weights: Array

    @classmethod
    def uniform_half_cell(
        cls,
        *,
        n_r: int,
        n_z: int,
        n_beta: int,
        r_max: float,
        z_max: float,
    ) -> "CylindricalGrid":
        """Construct the same half-cell layout used by the ACFO ODT scripts."""

        if n_r <= 0 or n_z <= 0 or n_beta <= 0:
            raise ValueError("n_r, n_z, and n_beta must be positive")
        if r_max <= 0.0 or z_max <= 0.0:
            raise ValueError("r_max and z_max must be positive")

        dr = float(r_max) / int(n_r)
        dz = 2.0 * float(z_max) / int(n_z)
        dbeta = 2.0 * np.pi / int(n_beta)
        r_axis = (np.arange(n_r, dtype=np.float64) + 0.5) * dr
        z_axis = -float(z_max) + (np.arange(n_z, dtype=np.float64) + 0.5) * dz
        beta_axis = np.arange(n_beta, dtype=np.float64) * dbeta
        volume = np.broadcast_to(
            r_axis[:, None, None] * dr * dbeta * dz,
            (n_r, n_z, n_beta),
        ).copy()
        return cls(
            r_axis=r_axis,
            z_axis=z_axis,
            beta_axis=beta_axis,
            dr=dr,
            dz=dz,
            dbeta=dbeta,
            volume_weights=np.ascontiguousarray(volume),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            int(self.r_axis.size),
            int(self.z_axis.size),
            int(self.beta_axis.size),
        )

    @property
    def sqrt_volume_weights(self) -> Array:
        return np.sqrt(self.volume_weights)

    @property
    def r_max(self) -> float:
        return float(self.r_axis[-1] + 0.5 * self.dr)

    @property
    def z_max(self) -> float:
        return float(self.z_axis[-1] + 0.5 * self.dz)


@dataclass(frozen=True)
class CylindricalStateMetric:
    """Conversions among physical, integrated, and symmetric state variables."""

    grid: CylindricalGrid

    def f_to_c(self, f: Array) -> Array:
        """Map physical samples ``f`` to integrated ACFO coefficients ``c``."""

        return _require_shape(f, self.grid.shape, "f") * self.grid.volume_weights

    def c_to_f(self, c: Array) -> Array:
        return _require_shape(c, self.grid.shape, "c") / self.grid.volume_weights

    def f_to_u(self, f: Array) -> Array:
        """Map physical samples to the Euclidean metric state ``u=sqrt(V)f``."""

        return _require_shape(f, self.grid.shape, "f") * self.grid.sqrt_volume_weights

    def u_to_f(self, u: Array) -> Array:
        return _require_shape(u, self.grid.shape, "u") / self.grid.sqrt_volume_weights

    def c_to_u(self, c: Array) -> Array:
        return _require_shape(c, self.grid.shape, "c") / self.grid.sqrt_volume_weights

    def u_to_c(self, u: Array) -> Array:
        return _require_shape(u, self.grid.shape, "u") * self.grid.sqrt_volume_weights

    def physical_inner(self, f: Array, g: Array) -> complex:
        """Discrete ``L2(rho dr dbeta dz)`` inner product ``<f,g>_V``."""

        f_checked = _require_shape(f, self.grid.shape, "f")
        g_checked = _require_shape(g, self.grid.shape, "g")
        return complex(np.vdot(f_checked, self.grid.volume_weights * g_checked))


@dataclass(frozen=True)
class MeasureCorrectOperator:
    """State-scaled wrapper around an ACFO-style coefficient operator.

    ``forward_coefficient`` computes ``A_c c`` and ``adjoint_coefficient``
    computes its ordinary Euclidean adjoint ``A_c**H y``.  The methods below
    implement

    ``A_f = A_c V`` and ``A_u = A_c sqrt(V)``

    with their exact adjoints in the stated domain metric.
    """

    metric: CylindricalStateMetric
    forward_coefficient: Callable[[Array], Array]
    adjoint_coefficient: Callable[[Array], Array]

    def forward_c(self, c: Array) -> Array:
        return np.asarray(self.forward_coefficient(_require_shape(c, self.metric.grid.shape, "c")))

    def adjoint_c(self, data: Array) -> Array:
        return _require_shape(
            np.asarray(self.adjoint_coefficient(data)),
            self.metric.grid.shape,
            "coefficient adjoint",
        )

    def forward_f(self, f: Array) -> Array:
        return self.forward_c(self.metric.f_to_c(f))

    def adjoint_f_euclidean(self, data: Array) -> Array:
        """Euclidean adjoint of ``forward_f``: ``V A_c**H data``."""

        return self.metric.grid.volume_weights * self.adjoint_c(data)

    def adjoint_f_physical(self, data: Array) -> Array:
        """Adjoint of ``forward_f`` in ``<.,.>_V``: ``A_c**H data``."""

        return self.adjoint_c(data)

    def forward_u(self, u: Array) -> Array:
        return self.forward_c(self.metric.u_to_c(u))

    def adjoint_u(self, data: Array) -> Array:
        """Euclidean adjoint of ``forward_u``: ``sqrt(V) A_c**H data``."""

        return self.metric.grid.sqrt_volume_weights * self.adjoint_c(data)


@dataclass(frozen=True)
class CylindricalTVGradient:
    """Collocated cylindrical gradient for TV in the symmetric ``u`` state.

    The physical field is ``f = u / sqrt(V)``.  This class applies

    ``Gamma u = sqrt(V) * (d_r f, d_z f, d_s f)``,

    where ``d_s = d_beta / r`` is the azimuthal arc-length derivative.  The
    radial forward stencil crosses the unresolved axis with the
    scalar-coordinate identity ``f(-r, beta) = f(r, beta + pi)``.  Radial and
    axial outer rows have zero forward difference, and azimuth is periodic.

    ``adjoint_euclidean`` is the exact conjugate transpose of the implemented
    stencil, not a separately discretized continuum divergence.  Consequently
    the pair can be composed safely with matrix-free primal-dual solvers.
    """

    grid: CylindricalGrid

    def __post_init__(self) -> None:
        n_r, _, n_beta = self.grid.shape
        if n_r < 2:
            raise ValueError("cylindrical TV requires at least two radial cells")
        if n_beta < 4 or n_beta % 2:
            raise ValueError(
                "cylindrical TV requires an even azimuthal grid with n_beta >= 4"
            )
        for name, value in (
            ("dr", self.grid.dr),
            ("dz", self.grid.dz),
            ("dbeta", self.grid.dbeta),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"grid {name} must be finite and positive")
        if not np.all(np.isfinite(self.grid.volume_weights)) or np.any(
            self.grid.volume_weights <= 0.0
        ):
            raise ValueError("grid volume weights must be finite and positive")

    @property
    def object_shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    @property
    def dual_shape(self) -> tuple[int, int, int, int]:
        return (3, *self.grid.shape)

    @property
    def dual_weights(self) -> Array:
        """Weights ``sqrt(V)`` in ``TV(f)=sum sqrt(V)||Gamma u||``."""

        return self.grid.sqrt_volume_weights

    def _gradient_f(self, f: Array) -> Array:
        field = _require_shape(f, self.grid.shape, "f")
        dtype = np.result_type(field.dtype, np.complex128)
        gradient = np.zeros(self.dual_shape, dtype=dtype)
        radial, axial, angular = gradient

        # At the first positive-radius row, use the diametrically opposite
        # sample as the negative-radius scalar continuation.
        axis_opposite = np.roll(field[0], self.grid.shape[2] // 2, axis=1)
        radial[0] = (field[1] - axis_opposite) / (
            self.grid.r_axis[1] + self.grid.r_axis[0]
        )
        if self.grid.shape[0] > 2:
            denominator = (
                self.grid.r_axis[2:] - self.grid.r_axis[1:-1]
            )[:, None, None]
            radial[1:-1] = (field[2:] - field[1:-1]) / denominator

        if self.grid.shape[1] > 1:
            denominator = (self.grid.z_axis[1:] - self.grid.z_axis[:-1])[
                None, :, None
            ]
            axial[:, :-1] = (field[:, 1:] - field[:, :-1]) / denominator

        angular[:] = (
            np.roll(field, -1, axis=2) - field
        ) / (self.grid.r_axis[:, None, None] * self.grid.dbeta)
        return gradient

    def _gradient_f_adjoint(self, values: Array) -> Array:
        dual = np.asarray(values)
        if dual.shape != self.dual_shape:
            raise ValueError(f"gradient dual must have shape {self.dual_shape}")
        if not np.all(np.isfinite(dual)):
            raise ValueError("gradient dual must contain only finite values")
        dtype = np.result_type(dual.dtype, np.complex128)
        output = np.zeros(self.grid.shape, dtype=dtype)
        radial, axial, angular = dual

        first_denominator = self.grid.r_axis[1] + self.grid.r_axis[0]
        output[1] += radial[0] / first_denominator
        output[0] -= np.roll(
            radial[0], self.grid.shape[2] // 2, axis=1
        ) / first_denominator
        if self.grid.shape[0] > 2:
            denominator = (
                self.grid.r_axis[2:] - self.grid.r_axis[1:-1]
            )[:, None, None]
            weighted = radial[1:-1] / denominator
            output[2:] += weighted
            output[1:-1] -= weighted

        if self.grid.shape[1] > 1:
            denominator = (self.grid.z_axis[1:] - self.grid.z_axis[:-1])[
                None, :, None
            ]
            weighted = axial[:, :-1] / denominator
            output[:, 1:] += weighted
            output[:, :-1] -= weighted

        scaled_angular = angular / (
            self.grid.r_axis[:, None, None] * self.grid.dbeta
        )
        output += np.roll(scaled_angular, 1, axis=2) - scaled_angular
        return output

    def forward(self, u: Array) -> Array:
        """Apply ``Gamma = sqrt(V) D V**(-1/2)``."""

        symmetric = _require_shape(u, self.grid.shape, "u")
        sqrt_volume = self.grid.sqrt_volume_weights
        return sqrt_volume[None, ...] * self._gradient_f(
            symmetric / sqrt_volume
        )

    def adjoint_euclidean(self, values: Array) -> Array:
        """Apply the exact Euclidean adjoint ``Gamma**H``."""

        dual = np.asarray(values)
        if dual.shape != self.dual_shape:
            raise ValueError(f"gradient dual must have shape {self.dual_shape}")
        sqrt_volume = self.grid.sqrt_volume_weights
        return self._gradient_f_adjoint(sqrt_volume[None, ...] * dual) / sqrt_volume

    def value_u(self, u: Array) -> float:
        """Return midpoint isotropic TV of the physical field ``f``."""

        gradient = self.forward(u)
        point_norm = np.sqrt(np.sum(np.abs(gradient) ** 2, axis=0))
        return float(np.sum(self.dual_weights * point_norm))

    def value_f(self, f: Array) -> float:
        """Return midpoint isotropic TV directly from physical samples."""

        return self.value_u(
            _require_shape(f, self.grid.shape, "f")
            * self.grid.sqrt_volume_weights
        )


@dataclass(frozen=True)
class CylindricalH1Seminorm:
    """Metric-aware quadratic H1 seminorm of the physical field ``f``.

    The discretized continuum energy is

    ``integral (|d_r f|^2 + |d_z f|^2 + |d_beta f|^2/r^2) r dr dbeta dz``.

    Radial and axial derivatives use conservative edge energies.  Angular
    derivatives are spectral and periodic.  At the unresolved core
    ``0 <= r <= dr/2``, each nonzero mode is continued as ``r**|m|``.  Its
    exact minimum Dirichlet energy contributes ``|m|`` times the boundary
    modal L2 energy.  This avoids division by zero, gives a single-valued axis,
    and leaves the constant mode in the nullspace.  Natural (zero-flux)
    conditions are used at the outer radial and axial boundaries.
    """

    grid: CylindricalGrid

    def _angular_eigenvalues(self) -> Array:
        n_r, _, n_beta = self.grid.shape
        modes = np.fft.fftfreq(n_beta, d=1.0 / n_beta)
        mode_abs = np.abs(modes)
        mode_sq = modes * modes
        eigenvalues = np.empty((n_r, n_beta), dtype=np.float64)
        # Core 0..dr/2: exact harmonic extension.  Outer half dr/2..dr:
        # piecewise-constant angular energy, whose radial integral is log(2).
        eigenvalues[0] = mode_abs + mode_sq * np.log(2.0)
        if n_r > 1:
            index = np.arange(1, n_r, dtype=np.float64)
            annular_log = np.log((index + 1.0) / index)
            eigenvalues[1:] = annular_log[:, None] * mode_sq[None, :]
        return eigenvalues

    def energy_f(self, f: Array) -> float:
        """Return the physical cylindrical H1 seminorm squared."""

        f_checked = _require_shape(f, self.grid.shape, "f")
        radial_difference = f_checked[1:, :, :] - f_checked[:-1, :, :]
        radial_faces = np.arange(1, self.grid.shape[0], dtype=np.float64) * self.grid.dr
        radial_weights = radial_faces * self.grid.dbeta * self.grid.dz / self.grid.dr
        radial_energy = np.sum(
            radial_weights[:, None, None] * np.abs(radial_difference) ** 2
        )

        axial_difference = f_checked[:, 1:, :] - f_checked[:, :-1, :]
        axial_weights = (
            self.grid.r_axis * self.grid.dr * self.grid.dbeta / self.grid.dz
        )
        axial_energy = np.sum(
            axial_weights[:, None, None] * np.abs(axial_difference) ** 2
        )

        beta_spectrum = np.fft.fft(f_checked, axis=2, norm="ortho")
        angular_energy = self.grid.dz * self.grid.dbeta * np.sum(
            self._angular_eigenvalues()[:, None, :] * np.abs(beta_spectrum) ** 2
        )
        return float(np.real(radial_energy + axial_energy + angular_energy))

    def normal_f(self, f: Array) -> Array:
        """Euclidean-coordinate normal ``L_f f`` satisfying ``f^H L_f f=E``."""

        f_checked = _require_shape(f, self.grid.shape, "f")
        dtype = np.result_type(f_checked.dtype, np.complex128)
        out = np.zeros(self.grid.shape, dtype=dtype)

        radial_difference = f_checked[1:, :, :] - f_checked[:-1, :, :]
        radial_faces = np.arange(1, self.grid.shape[0], dtype=np.float64) * self.grid.dr
        radial_weights = radial_faces * self.grid.dbeta * self.grid.dz / self.grid.dr
        weighted_radial = radial_weights[:, None, None] * radial_difference
        out[:-1, :, :] -= weighted_radial
        out[1:, :, :] += weighted_radial

        axial_difference = f_checked[:, 1:, :] - f_checked[:, :-1, :]
        axial_weights = (
            self.grid.r_axis * self.grid.dr * self.grid.dbeta / self.grid.dz
        )
        weighted_axial = axial_weights[:, None, None] * axial_difference
        out[:, :-1, :] -= weighted_axial
        out[:, 1:, :] += weighted_axial

        beta_spectrum = np.fft.fft(f_checked, axis=2, norm="ortho")
        weighted_spectrum = (
            self.grid.dz
            * self.grid.dbeta
            * self._angular_eigenvalues()[:, None, :]
            * beta_spectrum
        )
        out += np.fft.ifft(weighted_spectrum, axis=2, norm="ortho")
        return out

    def normal_u(self, u: Array) -> Array:
        """Euclidean normal in the recommended state ``u=sqrt(V)f``."""

        sqrt_volume = self.grid.sqrt_volume_weights
        f = _require_shape(u, self.grid.shape, "u") / sqrt_volume
        return self.normal_f(f) / sqrt_volume

    def normal_c(self, c: Array) -> Array:
        """Euclidean normal when legacy integrated coefficients are optimized."""

        volume = self.grid.volume_weights
        f = _require_shape(c, self.grid.shape, "c") / volume
        return self.normal_f(f) / volume


def naive_unweighted_coefficient_gradient_normal(c: Array) -> Array:
    """Legacy-style unweighted difference normal applied directly to ``c``."""

    coefficient = np.asarray(c)
    if coefficient.ndim != 3:
        raise ValueError("c must be a three-dimensional (rho, z, beta) array")
    out = np.zeros_like(coefficient, dtype=np.result_type(coefficient, np.complex128))
    radial = coefficient[1:, :, :] - coefficient[:-1, :, :]
    out[:-1, :, :] -= radial
    out[1:, :, :] += radial
    axial = coefficient[:, 1:, :] - coefficient[:, :-1, :]
    out[:, :-1, :] -= axial
    out[:, 1:, :] += axial
    out += (
        2.0 * coefficient
        - np.roll(coefficient, 1, axis=2)
        - np.roll(coefficient, -1, axis=2)
    )
    return out


def naive_unweighted_coefficient_gradient_energy(c: Array) -> float:
    """Energy of the legacy-style unweighted differences applied directly to c."""

    coefficient = np.asarray(c)
    if coefficient.ndim != 3:
        raise ValueError("c must be a three-dimensional (rho, z, beta) array")
    radial = coefficient[1:, :, :] - coefficient[:-1, :, :]
    axial = coefficient[:, 1:, :] - coefficient[:, :-1, :]
    angular = np.roll(coefficient, -1, axis=2) - coefficient
    return float(
        np.real(
            np.sum(np.abs(radial) ** 2)
            + np.sum(np.abs(axial) ** 2)
            + np.sum(np.abs(angular) ** 2)
        )
    )


def neumann_manufactured_h1_case(
    grid: CylindricalGrid,
    *,
    mode_amplitudes: tuple[tuple[int, float], ...] = (
        (0, 0.6),
        (1, 0.4),
        (4, 0.2),
    ),
) -> tuple[Array, float]:
    """Return a smooth Neumann-compatible field and its exact H1 energy.

    Every component has axial factor ``cos(pi*z/Z)``, so its derivative
    vanishes at ``z=+-Z``.  For ``m>0`` the radial factor is

    ``r**m * (1 - m/(m+2) * (r/R)**2)``,

    which is regular at the axis and has zero derivative at ``r=R``.  The
    ``m=0`` component uses ``(1-(r/R)**2)**2``.  Orthogonality in beta makes
    the exact total energy the sum of closed-form one-dimensional integrals.
    """

    if not mode_amplitudes:
        raise ValueError("mode_amplitudes must not be empty")
    if any(mode < 0 for mode, _ in mode_amplitudes):
        raise ValueError("manufactured angular modes must be nonnegative")
    if len({mode for mode, _ in mode_amplitudes}) != len(mode_amplitudes):
        raise ValueError("manufactured angular modes must be unique")
    max_mode = max(mode for mode, _ in mode_amplitudes)
    if 2 * max_mode >= grid.shape[2]:
        raise ValueError("n_beta must exceed twice the largest manufactured mode")

    rr, zz, bb = np.meshgrid(
        grid.r_axis, grid.z_axis, grid.beta_axis, indexing="ij"
    )
    radius = grid.r_max
    half_height = grid.z_max
    axial = np.cos(np.pi * zz / half_height)
    field = np.zeros(grid.shape, dtype=np.float64)
    exact_energy = 0.0
    axial_l2 = half_height
    axial_derivative_l2 = np.pi**2 / half_height

    for mode, amplitude in mode_amplitudes:
        amplitude_sq = float(amplitude) ** 2
        if mode == 0:
            radial = (1.0 - (rr / radius) ** 2) ** 2
            radial_derivative_energy = 2.0 / 3.0
            radial_mass = radius**2 / 10.0
            beta_factor = 2.0 * np.pi
            angular_energy = 0.0
        else:
            mode_float = float(mode)
            coefficient = mode_float / (mode_float + 2.0)
            radial = rr**mode * (
                1.0 - coefficient * (rr / radius) ** 2
            )
            radial_derivative_energy = mode_float**2 * radius ** (2 * mode) * (
                1.0 / (2.0 * mode_float)
                - 2.0 / (2.0 * mode_float + 2.0)
                + 1.0 / (2.0 * mode_float + 4.0)
            )
            angular_energy = mode_float**2 * radius ** (2 * mode) * (
                1.0 / (2.0 * mode_float)
                - 2.0 * coefficient / (2.0 * mode_float + 2.0)
                + coefficient**2 / (2.0 * mode_float + 4.0)
            )
            radial_mass = radius ** (2 * mode + 2) * (
                1.0 / (2.0 * mode_float + 2.0)
                - 2.0 * coefficient / (2.0 * mode_float + 4.0)
                + coefficient**2 / (2.0 * mode_float + 6.0)
            )
            beta_factor = np.pi
        field += float(amplitude) * radial * np.cos(mode * bb) * axial
        exact_energy += amplitude_sq * beta_factor * (
            axial_l2 * (radial_derivative_energy + angular_energy)
            + axial_derivative_l2 * radial_mass
        )
    return np.ascontiguousarray(field), float(exact_energy)


@dataclass(frozen=True)
class DenseReconstructionSmokeResult:
    """Solutions and JSON-ready diagnostics from the small dense oracle."""

    f_solution: Array
    u_solution: Array
    legacy_c_solution: Array
    diagnostics: dict[str, Any]


def _dense_matrix_from_apply(
    apply: Callable[[Array], Array], shape: tuple[int, int, int]
) -> Array:
    columns: list[Array] = []
    for column in range(int(np.prod(shape))):
        basis = np.zeros(shape, dtype=np.complex128)
        basis.ravel()[column] = 1.0
        columns.append(np.asarray(apply(basis), dtype=np.complex128).ravel())
    return np.column_stack(columns)


def _relative_hermitian_error(matrix: Array) -> float:
    scale = max(float(np.linalg.norm(matrix)), 1e-300)
    return float(np.linalg.norm(matrix - matrix.conj().T) / scale)


def _relative_normal_residual(matrix: Array, solution: Array, rhs: Array) -> float:
    return float(
        np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1e-300)
    )


def _physical_relative_l2(
    metric: CylindricalStateMetric, candidate: Array, reference: Array
) -> float:
    delta = np.asarray(candidate) - np.asarray(reference)
    numerator = max(float(np.real(metric.physical_inner(delta, delta))), 0.0)
    denominator = max(
        float(np.real(metric.physical_inner(reference, reference))), 1e-300
    )
    return float(np.sqrt(numerator / denominator))


def dense_reconstruction_equivalence_smoke(
    *,
    operator: MeasureCorrectOperator,
    regularizer: CylindricalH1Seminorm,
    data: Array,
    lambda_h1: float,
    lambda_legacy: float | None = None,
    truth_f: Array | None = None,
) -> DenseReconstructionSmokeResult:
    """Solve one physical objective independently in ``f`` and ``u`` coordinates.

    This is a deliberately small dense oracle, not a production reconstruction
    routine.  It constructs the matrix of a real ACFO-style operator from basis
    actions and solves

    ``0.5 ||A_c V f-y||^2 + 0.5 lambda_h1 f**H L_f f``

    both in physical samples ``f`` and in ``u=sqrt(V)f``.  A third solve uses
    unweighted differences on integrated coefficients ``c`` as a descriptive
    negative control.  No gate requires that the negative control be worse.
    """

    if lambda_h1 < 0.0:
        raise ValueError("lambda_h1 must be nonnegative")
    if lambda_legacy is None:
        lambda_legacy = lambda_h1
    if lambda_legacy < 0.0:
        raise ValueError("lambda_legacy must be nonnegative")
    grid = operator.metric.grid
    if regularizer.grid.shape != grid.shape or not np.allclose(
        regularizer.grid.volume_weights, grid.volume_weights, rtol=0.0, atol=0.0
    ):
        raise ValueError("operator and regularizer must use the same cylindrical grid")

    shape = grid.shape
    bins = int(np.prod(shape))
    data_vector = np.asarray(data, dtype=np.complex128).ravel()
    coefficient_matrix = _dense_matrix_from_apply(operator.forward_c, shape)
    if coefficient_matrix.shape[0] != data_vector.size:
        raise ValueError("data size does not match the wrapped coefficient operator")
    h1_matrix = _dense_matrix_from_apply(regularizer.normal_f, shape)
    naive_matrix = _dense_matrix_from_apply(
        naive_unweighted_coefficient_gradient_normal, shape
    )

    volume = grid.volume_weights.ravel()
    sqrt_volume = np.sqrt(volume)
    a_f = coefficient_matrix * volume[None, :]
    a_u = coefficient_matrix * sqrt_volume[None, :]
    h1_u = h1_matrix / sqrt_volume[:, None] / sqrt_volume[None, :]

    normal_f = a_f.conj().T @ a_f + float(lambda_h1) * h1_matrix
    normal_u = a_u.conj().T @ a_u + float(lambda_h1) * h1_u
    rhs_f = a_f.conj().T @ data_vector
    rhs_u = a_u.conj().T @ data_vector
    f_vector = np.linalg.solve(normal_f, rhs_f)
    u_vector = np.linalg.solve(normal_u, rhs_u)
    f_from_u = u_vector / sqrt_volume

    normal_legacy = (
        coefficient_matrix.conj().T @ coefficient_matrix
        + float(lambda_legacy) * naive_matrix
    )
    rhs_legacy = coefficient_matrix.conj().T @ data_vector
    legacy_c_vector = np.linalg.solve(normal_legacy, rhs_legacy)
    legacy_f_vector = legacy_c_vector / volume

    def physical_objective(f_value: Array) -> float:
        residual = a_f @ f_value - data_vector
        return float(
            0.5 * np.vdot(residual, residual).real
            + 0.5
            * float(lambda_h1)
            * np.vdot(f_value, h1_matrix @ f_value).real
        )

    f_objective = physical_objective(f_vector)
    u_objective = physical_objective(f_from_u)
    legacy_physical_objective = physical_objective(legacy_f_vector)
    legacy_residual = coefficient_matrix @ legacy_c_vector - data_vector
    legacy_objective = float(
        0.5 * np.vdot(legacy_residual, legacy_residual).real
        + 0.5
        * float(lambda_legacy)
        * np.vdot(legacy_c_vector, naive_matrix @ legacy_c_vector).real
    )

    h1_hermitian = _relative_hermitian_error(h1_matrix)
    h1_u_hermitian = _relative_hermitian_error(h1_u)
    h1_eigenvalues = np.linalg.eigvalsh(0.5 * (h1_matrix + h1_matrix.conj().T))
    h1_u_eigenvalues = np.linalg.eigvalsh(0.5 * (h1_u + h1_u.conj().T))
    h1_scale = max(float(np.max(np.abs(h1_eigenvalues))), 1e-300)
    h1_u_scale = max(float(np.max(np.abs(h1_u_eigenvalues))), 1e-300)
    solution_relative_l2 = _physical_relative_l2(
        operator.metric, f_from_u.reshape(shape), f_vector.reshape(shape)
    )
    objective_relative_difference = float(
        abs(f_objective - u_objective)
        / max(abs(f_objective), abs(u_objective), 1e-300)
    )
    f_normal_residual = _relative_normal_residual(
        normal_f, f_vector, rhs_f
    )
    u_normal_residual = _relative_normal_residual(
        normal_u, u_vector, rhs_u
    )
    gates = {
        "h1_hermitian_pass": h1_hermitian <= 1e-12,
        "h1_u_hermitian_pass": h1_u_hermitian <= 1e-12,
        "h1_psd_pass": float(np.min(h1_eigenvalues)) >= -1e-12 * h1_scale,
        "h1_u_psd_pass": float(np.min(h1_u_eigenvalues)) >= -1e-12 * h1_u_scale,
        "solution_equivalence_pass": solution_relative_l2 <= 1e-9,
        "objective_equivalence_pass": objective_relative_difference <= 1e-10,
        "f_normal_residual_pass": f_normal_residual <= 1e-9,
        "u_normal_residual_pass": u_normal_residual <= 1e-9,
    }
    gates["all_required_pass"] = bool(all(gates.values()))

    diagnostics: dict[str, Any] = {
        "dimensions": {
            "state_bins": bins,
            "data_samples": int(data_vector.size),
        },
        "parameters": {
            "lambda_h1": float(lambda_h1),
            "lambda_legacy": float(lambda_legacy),
        },
        "regularizer": {
            "h1_f_relative_hermitian_error": h1_hermitian,
            "h1_u_relative_hermitian_error": h1_u_hermitian,
            "h1_f_min_eigenvalue": float(np.min(h1_eigenvalues)),
            "h1_f_max_eigenvalue": float(np.max(h1_eigenvalues)),
            "h1_u_min_eigenvalue": float(np.min(h1_u_eigenvalues)),
            "h1_u_max_eigenvalue": float(np.max(h1_u_eigenvalues)),
        },
        "equivalent_physical_objective": {
            "f_solution_objective": f_objective,
            "u_solution_objective": u_objective,
            "objective_relative_difference": objective_relative_difference,
            "f_vs_u_physical_relative_l2": solution_relative_l2,
            "f_normal_relative_residual": f_normal_residual,
            "u_normal_relative_residual": u_normal_residual,
            "f_normal_condition_number": float(np.linalg.cond(normal_f)),
            "u_normal_condition_number": float(np.linalg.cond(normal_u)),
        },
        "legacy_c_naive_difference_negative_control": {
            "legacy_objective": legacy_objective,
            "physical_objective_evaluated_at_legacy_solution": legacy_physical_objective,
            "physical_objective_ratio_vs_f_solution": float(
                legacy_physical_objective / max(f_objective, 1e-300)
            ),
            "legacy_normal_relative_residual": _relative_normal_residual(
                normal_legacy, legacy_c_vector, rhs_legacy
            ),
            "legacy_normal_condition_number": float(np.linalg.cond(normal_legacy)),
            "is_required_gate": False,
        },
        "gates": gates,
    }
    if truth_f is not None:
        truth_checked = _require_shape(truth_f, shape, "truth_f")
        diagnostics["truth_comparison"] = {
            "f_solution_physical_relative_l2": _physical_relative_l2(
                operator.metric, f_vector.reshape(shape), truth_checked
            ),
            "u_solution_physical_relative_l2": _physical_relative_l2(
                operator.metric, f_from_u.reshape(shape), truth_checked
            ),
            "legacy_solution_physical_relative_l2": _physical_relative_l2(
                operator.metric, legacy_f_vector.reshape(shape), truth_checked
            ),
        }

    return DenseReconstructionSmokeResult(
        f_solution=np.ascontiguousarray(f_vector.reshape(shape)),
        u_solution=np.ascontiguousarray(u_vector.reshape(shape)),
        legacy_c_solution=np.ascontiguousarray(legacy_c_vector.reshape(shape)),
        diagnostics=diagnostics,
    )
