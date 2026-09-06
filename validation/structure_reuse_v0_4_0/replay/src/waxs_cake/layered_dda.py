"""Prepared discrete-dipole solves above an isotropic planar substrate.

The implementation combines the analytic homogeneous-space electric dyadic
with :class:`~waxs_cake.layered_dyadic_green.PreparedLayeredDyadicGreenOperator`.
Repeated pair geometries are compressed before reflected Green blocks and
their height derivatives are prepared.  The resulting DDA linear system
supports forward/adjoint solves and implicit derivatives for a common particle
height above the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .layered_dyadic_green import PreparedLayeredDyadicGreenOperator


Array = np.ndarray


def free_space_electric_dyadic(displacement: Any, wavenumber: complex) -> Array:
    r"""Return ``(I + grad grad/k^2) exp(ikr)/(4 pi r)`` off the origin."""

    vectors = np.asarray(displacement, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError("displacement must end in a Cartesian axis of length three")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("displacement must contain only finite values")
    k = complex(wavenumber)
    if not np.isfinite(k.real) or not np.isfinite(k.imag) or abs(k) == 0.0:
        raise ValueError("wavenumber must be finite and nonzero")
    distance = np.linalg.norm(vectors, axis=-1)
    if np.any(distance <= 0.0):
        raise ValueError("free-space dyadic is singular at zero displacement")
    unit = vectors / distance[..., None]
    kr = k * distance
    scalar = np.exp(1j * kr) / (4.0 * np.pi * distance)
    transverse = 1.0 + 1j / kr - 1.0 / kr**2
    longitudinal = -1.0 - 3j / kr + 3.0 / kr**2
    identity = np.eye(3, dtype=np.complex128)
    return scalar[..., None, None] * (
        transverse[..., None, None] * identity
        + longitudinal[..., None, None] * unit[..., :, None] * unit[..., None, :]
    )


def clausius_mossotti_polarizability(
    voxel_volume: float,
    particle_permittivity: complex,
    *,
    medium_permittivity: complex = 1.0,
    medium_wavenumber: float | None = None,
    radiative_correction: bool = True,
) -> complex:
    """Return an isotropic voxel polarizability in relative-volume units."""

    volume = float(voxel_volume)
    particle = complex(particle_permittivity)
    medium = complex(medium_permittivity)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("voxel_volume must be finite and positive")
    for name, value in (("particle_permittivity", particle), ("medium_permittivity", medium)):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"{name} must be finite")
    denominator = particle + 2.0 * medium
    if denominator == 0.0:
        raise ValueError("Clausius-Mossotti denominator is zero")
    alpha = 3.0 * volume * (particle - medium) / denominator
    if radiative_correction:
        if medium_wavenumber is None:
            raise ValueError("medium_wavenumber is required for radiative correction")
        k = float(medium_wavenumber)
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("medium_wavenumber must be finite and positive")
        alpha = alpha / (1.0 - 1j * k**3 * alpha / (6.0 * np.pi))
    return complex(alpha)


def clausius_mossotti_polarizabilities(
    voxel_volumes: Any,
    particle_permittivity: complex,
    *,
    medium_permittivity: complex = 1.0,
    medium_wavenumber: float | None = None,
    radiative_correction: bool = True,
) -> Array:
    """Vectorized Clausius--Mossotti polarizability for unequal voxels."""

    volumes = np.asarray(voxel_volumes, dtype=np.float64)
    if (
        volumes.ndim != 1
        or volumes.size == 0
        or not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0.0)
    ):
        raise ValueError("voxel_volumes must be a non-empty positive finite vector")
    particle = complex(particle_permittivity)
    medium = complex(medium_permittivity)
    for name, value in (("particle_permittivity", particle), ("medium_permittivity", medium)):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"{name} must be finite")
    denominator = particle + 2.0 * medium
    if denominator == 0.0:
        raise ValueError("Clausius-Mossotti denominator is zero")
    alpha = 3.0 * volumes.astype(np.complex128) * (
        particle - medium
    ) / denominator
    if radiative_correction:
        if medium_wavenumber is None:
            raise ValueError("medium_wavenumber is required for radiative correction")
        k = float(medium_wavenumber)
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("medium_wavenumber must be finite and positive")
        alpha = alpha / (1.0 - 1j * k**3 * alpha / (6.0 * np.pi))
    return np.ascontiguousarray(alpha)


def elliptic_cylinder_dipoles(
    *,
    semiaxis_x: float,
    semiaxis_y: float,
    height: float,
    spacing: float,
    bottom_height: float,
    rotation: float = 0.0,
) -> Array:
    """Voxel-centre discretization of a rotated elliptic nanocylinder."""

    a = float(semiaxis_x)
    b = float(semiaxis_y)
    cylinder_height = float(height)
    spacing = float(spacing)
    bottom_height = float(bottom_height)
    rotation = float(rotation)
    values = (a, b, cylinder_height, spacing, bottom_height, rotation)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("all geometry parameters must be finite")
    if a <= 0.0 or b <= 0.0 or cylinder_height <= 0.0 or spacing <= 0.0:
        raise ValueError("semiaxes, height and spacing must be positive")
    if bottom_height <= 0.0:
        raise ValueError("bottom_height must be positive")

    x_axis = _cell_centres(-a, a, spacing)
    y_axis = _cell_centres(-b, b, spacing)
    z_axis = _cell_centres(bottom_height, bottom_height + cylinder_height, spacing)
    xx, yy, zz = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    inside = (xx / a) ** 2 + (yy / b) ** 2 <= 1.0
    local_x = xx[inside]
    local_y = yy[inside]
    cosine = np.cos(rotation)
    sine = np.sin(rotation)
    positions = np.column_stack(
        (
            cosine * local_x - sine * local_y,
            sine * local_x + cosine * local_y,
            zz[inside],
        )
    )
    if positions.size == 0:
        raise ValueError("spacing is too coarse to place a dipole in the cylinder")
    return np.ascontiguousarray(positions, dtype=np.float64)


def elliptic_cylinder_fractional_voxels(
    *,
    semiaxis_x: float,
    semiaxis_y: float,
    height: float,
    spacing: float,
    bottom_height: float,
    rotation: float = 0.0,
    boundary_quadrature_order: int = 16,
    use_occupied_centroid: bool = True,
    return_lattice_indices: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, Array]:
    """Return occupied-cell centroids and volumes for an elliptic cylinder.

    Tensor Gauss--Legendre sampling resolves the ellipse fraction within each
    Cartesian boundary cell.  The returned local volume weights are globally
    normalized to the analytic cylinder volume, while their relative values
    and occupied centroids retain the boundary information.
    """

    a = float(semiaxis_x)
    b = float(semiaxis_y)
    cylinder_height = float(height)
    spacing = float(spacing)
    bottom_height = float(bottom_height)
    rotation = float(rotation)
    order = int(boundary_quadrature_order)
    values = (a, b, cylinder_height, spacing, bottom_height, rotation)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("all geometry parameters must be finite")
    if a <= 0.0 or b <= 0.0 or cylinder_height <= 0.0 or spacing <= 0.0:
        raise ValueError("semiaxes, height and spacing must be positive")
    if bottom_height <= 0.0:
        raise ValueError("bottom_height must be positive")
    if order < 2:
        raise ValueError("boundary_quadrature_order must be at least two")

    nx = max(1, int(np.ceil(2.0 * a / spacing)))
    ny = max(1, int(np.ceil(2.0 * b / spacing)))
    nz = max(1, int(np.ceil(cylinder_height / spacing)))
    dx = 2.0 * a / nx
    dy = 2.0 * b / ny
    dz = cylinder_height / nz
    x_centres = -a + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y_centres = -b + (np.arange(ny, dtype=np.float64) + 0.5) * dy
    z_centres = bottom_height + (np.arange(nz, dtype=np.float64) + 0.5) * dz
    nodes, weights = np.polynomial.legendre.leggauss(order)
    tensor_weights = weights[:, None] * weights[None, :]
    occupied_xy: list[tuple[float, float, float, int, int]] = []
    for x_index, x_centre in enumerate(x_centres):
        sample_x = x_centre + 0.5 * dx * nodes[:, None]
        for y_index, y_centre in enumerate(y_centres):
            sample_y = y_centre + 0.5 * dy * nodes[None, :]
            mask = (sample_x / a) ** 2 + (sample_y / b) ** 2 <= 1.0
            occupied_weight = tensor_weights * mask
            weight_sum = float(np.sum(occupied_weight))
            if weight_sum == 0.0:
                continue
            area = 0.25 * dx * dy * weight_sum
            centroid_x = float(np.sum(occupied_weight * sample_x) / weight_sum)
            centroid_y = float(np.sum(occupied_weight * sample_y) / weight_sum)
            if use_occupied_centroid:
                position_x, position_y = centroid_x, centroid_y
            else:
                position_x, position_y = float(x_centre), float(y_centre)
            occupied_xy.append(
                (position_x, position_y, area, x_index, y_index)
            )
    if not occupied_xy:
        raise ValueError("spacing is too coarse to resolve the cylinder")

    cosine = np.cos(rotation)
    sine = np.sin(rotation)
    positions: list[tuple[float, float, float]] = []
    volumes: list[float] = []
    lattice_indices: list[tuple[int, int, int]] = []
    for centroid_x, centroid_y, area, x_index, y_index in occupied_xy:
        lab_x = cosine * centroid_x - sine * centroid_y
        lab_y = sine * centroid_x + cosine * centroid_y
        for z_index, z_centre in enumerate(z_centres):
            positions.append((lab_x, lab_y, float(z_centre)))
            volumes.append(area * dz)
            lattice_indices.append((x_index, y_index, z_index))
    volume_array = np.asarray(volumes, dtype=np.float64)
    analytic_volume = np.pi * a * b * cylinder_height
    volume_array *= analytic_volume / float(np.sum(volume_array))
    position_array = np.ascontiguousarray(positions, dtype=np.float64)
    if return_lattice_indices:
        return (
            position_array,
            np.ascontiguousarray(volume_array),
            np.ascontiguousarray(lattice_indices, dtype=np.int32),
        )
    return position_array, np.ascontiguousarray(volume_array)


@dataclass(frozen=True)
class DDASolveResult:
    dipoles: Array
    converged: bool
    iterations: int
    relative_residual: float
    method: str
    preconditioner: str


@dataclass(frozen=True)
class PreparedLayeredDDAOperator:
    """Dense-kernel, matrix-free DDA system with compressed Green preparation."""

    positions: Array
    polarizability: Array
    free_space_kernel: Array
    reflected_kernel: Array
    reflected_height_derivative: Array
    interaction_scale: complex
    unique_pair_count: int
    pair_count: int
    green_contracts: tuple[dict[str, int | float | bool | str], ...]
    reflected_self_voxel_side: float
    reflected_near_voxel_side: float
    reflected_near_image_radius_factor: float

    @classmethod
    def build(
        cls,
        *,
        positions: Any,
        polarizability: Any,
        radial_wavenumbers: Any,
        quadrature_weights: Any,
        phi: Any,
        upper_wavenumber: float,
        lower_wavenumber: float,
        damping: float,
        voxel_side: float = 0.0,
        voxel_rotation: float = 0.0,
        reflected_self_voxel_side: float = 0.0,
        reflected_self_radial_wavenumbers: Any | None = None,
        reflected_self_quadrature_weights: Any | None = None,
        reflected_self_phi: Any | None = None,
        reflected_near_voxel_side: float = 0.0,
        reflected_near_image_radius_factor: float = 3.0,
        reflected_near_radial_wavenumbers: Any | None = None,
        reflected_near_quadrature_weights: Any | None = None,
        reflected_near_phi: Any | None = None,
        interaction_scale: complex | None = None,
        green_chunk_size: int = 256,
        geometry_decimals: int = 14,
        kernel_tolerance: float = 1.0e-12,
        tensor_tail_tolerance: float = 1.0e-13,
        mode_padding: int = 8,
        miller_margin: int = 32,
    ) -> "PreparedLayeredDDAOperator":
        points = np.asarray(positions, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or points.shape[0] == 0
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("positions must be a non-empty finite (n,3) array")
        if np.any(points[:, 2] <= 0.0):
            raise ValueError("all dipoles must lie above the interface")
        alpha = _polarizability_blocks(polarizability, points.shape[0])
        green_chunk_size = int(green_chunk_size)
        geometry_decimals = int(geometry_decimals)
        if green_chunk_size <= 0:
            raise ValueError("green_chunk_size must be positive")
        if geometry_decimals < 0 or geometry_decimals > 15:
            raise ValueError("geometry_decimals must lie in [0,15]")
        voxel_side = float(voxel_side)
        reflected_self_voxel_side = float(reflected_self_voxel_side)
        reflected_near_voxel_side = float(reflected_near_voxel_side)
        reflected_near_image_radius_factor = float(
            reflected_near_image_radius_factor
        )
        if not np.isfinite(voxel_side) or voxel_side < 0.0:
            raise ValueError("voxel_side must be finite and non-negative")
        if (
            not np.isfinite(reflected_self_voxel_side)
            or reflected_self_voxel_side < 0.0
        ):
            raise ValueError(
                "reflected_self_voxel_side must be finite and non-negative"
            )
        if voxel_side > 0.0 and reflected_self_voxel_side > 0.0:
            raise ValueError(
                "choose all-pair voxel integration or reflected-self integration, not both"
            )
        if (
            not np.isfinite(reflected_near_voxel_side)
            or reflected_near_voxel_side < 0.0
        ):
            raise ValueError(
                "reflected_near_voxel_side must be finite and non-negative"
            )
        if (
            not np.isfinite(reflected_near_image_radius_factor)
            or reflected_near_image_radius_factor <= 0.0
        ):
            raise ValueError(
                "reflected_near_image_radius_factor must be finite and positive"
            )
        if sum(
            value > 0.0
            for value in (
                voxel_side,
                reflected_self_voxel_side,
                reflected_near_voxel_side,
            )
        ) > 1:
            raise ValueError(
                "all-pair, reflected-self and reflected-near voxel integration are mutually exclusive"
            )
        scale = (
            complex(float(upper_wavenumber) ** 2)
            if interaction_scale is None
            else complex(interaction_scale)
        )
        if not np.isfinite(scale.real) or not np.isfinite(scale.imag):
            raise ValueError("interaction_scale must be finite")

        target = points[:, None, :]
        source = points[None, :, :]
        displacement = target - source
        pair_geometry = np.empty((points.shape[0], points.shape[0], 3), dtype=np.float64)
        pair_geometry[..., :2] = displacement[..., :2]
        pair_geometry[..., 2] = target[..., 2] + source[..., 2]
        rounded = np.round(pair_geometry.reshape(-1, 3), decimals=geometry_decimals)
        unique_geometry, inverse = np.unique(rounded, axis=0, return_inverse=True)
        unique_kernel = np.empty((3, 3, unique_geometry.shape[0]), dtype=np.complex128)
        unique_height_derivative = np.empty_like(unique_kernel)
        contracts: list[dict[str, int | float | bool | str]] = []
        for start in range(0, unique_geometry.shape[0], green_chunk_size):
            stop = min(start + green_chunk_size, unique_geometry.shape[0])
            geometry = unique_geometry[start:stop]
            plan = PreparedLayeredDyadicGreenOperator.build(
                radial_wavenumbers=radial_wavenumbers,
                quadrature_weights=quadrature_weights,
                phi=phi,
                x=geometry[:, 0],
                y=geometry[:, 1],
                height=geometry[:, 2],
                upper_wavenumber=upper_wavenumber,
                lower_wavenumber=lower_wavenumber,
                damping=damping,
                voxel_side=voxel_side,
                voxel_rotation=voxel_rotation,
                kernel_tolerance=kernel_tolerance,
                tensor_tail_tolerance=tensor_tail_tolerance,
                mode_padding=mode_padding,
                miller_margin=miller_margin,
            )
            unique_kernel[:, :, start:stop] = plan.kernel()
            unique_height_derivative[:, :, start:stop] = plan.derivative_kernels()[2]
            contracts.append(plan.contract.to_dict())
        if reflected_self_voxel_side > 0.0:
            self_mask = (unique_geometry[:, 0] == 0.0) & (
                unique_geometry[:, 1] == 0.0
            )
            if not np.any(self_mask):
                raise RuntimeError("compressed pair geometry contains no self offsets")
            self_radial = (
                radial_wavenumbers
                if reflected_self_radial_wavenumbers is None
                else reflected_self_radial_wavenumbers
            )
            self_weights = (
                quadrature_weights
                if reflected_self_quadrature_weights is None
                else reflected_self_quadrature_weights
            )
            if (reflected_self_radial_wavenumbers is None) != (
                reflected_self_quadrature_weights is None
            ):
                raise ValueError(
                    "reflected self radial nodes and weights must be supplied together"
                )
            self_phi = phi if reflected_self_phi is None else reflected_self_phi
            self_geometry = unique_geometry[self_mask]
            self_plan = PreparedLayeredDyadicGreenOperator.build(
                radial_wavenumbers=self_radial,
                quadrature_weights=self_weights,
                phi=self_phi,
                x=self_geometry[:, 0],
                y=self_geometry[:, 1],
                height=self_geometry[:, 2],
                upper_wavenumber=upper_wavenumber,
                lower_wavenumber=lower_wavenumber,
                damping=damping,
                voxel_side=reflected_self_voxel_side,
                voxel_rotation=voxel_rotation,
                h_cutoff=0,
                kernel_tolerance=kernel_tolerance,
                tensor_tail_tolerance=tensor_tail_tolerance,
                mode_padding=mode_padding,
                miller_margin=miller_margin,
            )
            unique_kernel[:, :, self_mask] = self_plan.kernel()
            unique_height_derivative[:, :, self_mask] = (
                self_plan.derivative_kernels()[2]
            )
            self_contract = self_plan.contract.to_dict()
            self_contract["application"] = "reflected_self_uniform_voxel"
            contracts.append(self_contract)
        if reflected_near_voxel_side > 0.0:
            image_distance = np.linalg.norm(unique_geometry, axis=1)
            near_mask = image_distance <= (
                reflected_near_image_radius_factor * reflected_near_voxel_side
            )
            if not np.any(near_mask):
                raise RuntimeError(
                    "no compressed pair geometry satisfies the reflected near criterion"
                )
            if (reflected_near_radial_wavenumbers is None) != (
                reflected_near_quadrature_weights is None
            ):
                raise ValueError(
                    "reflected near radial nodes and weights must be supplied together"
                )
            near_radial = (
                radial_wavenumbers
                if reflected_near_radial_wavenumbers is None
                else reflected_near_radial_wavenumbers
            )
            near_weights = (
                quadrature_weights
                if reflected_near_quadrature_weights is None
                else reflected_near_quadrature_weights
            )
            near_phi = phi if reflected_near_phi is None else reflected_near_phi
            near_geometry = unique_geometry[near_mask]
            near_plan = PreparedLayeredDyadicGreenOperator.build(
                radial_wavenumbers=near_radial,
                quadrature_weights=near_weights,
                phi=near_phi,
                x=near_geometry[:, 0],
                y=near_geometry[:, 1],
                height=near_geometry[:, 2],
                upper_wavenumber=upper_wavenumber,
                lower_wavenumber=lower_wavenumber,
                damping=damping,
                voxel_side=reflected_near_voxel_side,
                voxel_rotation=voxel_rotation,
                kernel_tolerance=kernel_tolerance,
                tensor_tail_tolerance=tensor_tail_tolerance,
                mode_padding=mode_padding,
                miller_margin=miller_margin,
            )
            unique_kernel[:, :, near_mask] = near_plan.kernel()
            unique_height_derivative[:, :, near_mask] = (
                near_plan.derivative_kernels()[2]
            )
            near_contract = near_plan.contract.to_dict()
            near_contract["application"] = "reflected_near_uniform_voxel"
            near_contract["image_radius_factor"] = (
                reflected_near_image_radius_factor
            )
            near_contract["corrected_unique_geometry_count"] = int(
                np.sum(near_mask)
            )
            contracts.append(near_contract)
        n = points.shape[0]
        reflected = unique_kernel[:, :, inverse].reshape(3, 3, n, n)
        reflected = np.transpose(reflected, (2, 0, 3, 1))
        reflected_height = unique_height_derivative[:, :, inverse].reshape(3, 3, n, n)
        reflected_height = np.transpose(reflected_height, (2, 0, 3, 1))

        free = np.zeros((n, 3, n, 3), dtype=np.complex128)
        off_diagonal = ~np.eye(n, dtype=bool)
        free_values = free_space_electric_dyadic(
            displacement[off_diagonal], complex(float(upper_wavenumber), 0.0)
        )
        target_indices, source_indices = np.nonzero(off_diagonal)
        free[target_indices, :, source_indices, :] = free_values
        return cls(
            positions=_readonly(points),
            polarizability=_readonly(alpha),
            free_space_kernel=_readonly(free),
            reflected_kernel=_readonly(reflected),
            reflected_height_derivative=_readonly(reflected_height),
            interaction_scale=scale,
            unique_pair_count=int(unique_geometry.shape[0]),
            pair_count=int(n * n),
            green_contracts=tuple(contracts),
            reflected_self_voxel_side=reflected_self_voxel_side,
            reflected_near_voxel_side=reflected_near_voxel_side,
            reflected_near_image_radius_factor=(
                reflected_near_image_radius_factor
            ),
        )

    @property
    def dipole_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def field_shape(self) -> tuple[int, int]:
        return (self.dipole_count, 3)

    @property
    def compression_ratio(self) -> float:
        return float(self.pair_count / self.unique_pair_count)

    @property
    def kernel_bytes(self) -> int:
        return int(
            self.free_space_kernel.nbytes
            + self.reflected_kernel.nbytes
            + self.reflected_height_derivative.nbytes
        )

    def total_green_kernel(self) -> Array:
        return self.free_space_kernel + self.reflected_kernel

    def interaction_field(self, dipoles: Any) -> Array:
        values = self._dipole_field(dipoles, name="dipoles")
        return self.interaction_scale * np.einsum(
            "iajb,jb->ia", self.total_green_kernel(), values, optimize=True
        )

    def interaction(self, dipoles: Any) -> Array:
        field = self.interaction_field(dipoles)
        return np.einsum("iab,ib->ia", self.polarizability, field, optimize=True)

    def matvec(self, dipoles: Any) -> Array:
        values = self._dipole_field(dipoles, name="dipoles")
        return values - self.interaction(values)

    def adjoint_matvec(self, cotangent: Any) -> Array:
        values = self._dipole_field(cotangent, name="cotangent")
        alpha_adjoint = np.einsum(
            "iba,ib->ia", np.conjugate(self.polarizability), values, optimize=True
        )
        interaction_adjoint = np.conjugate(self.interaction_scale) * np.einsum(
            "iajb,ia->jb",
            np.conjugate(self.total_green_kernel()),
            alpha_adjoint,
            optimize=True,
        )
        return values - interaction_adjoint

    def right_hand_side(self, incident_field: Any) -> Array:
        incident = self._dipole_field(incident_field, name="incident_field")
        return np.einsum(
            "iab,ib->ia", self.polarizability, incident, optimize=True
        )

    def dense_matrix(self) -> Array:
        n = self.dipole_count
        interaction = self.interaction_scale * np.einsum(
            "iac,icjb->iajb",
            self.polarizability,
            self.total_green_kernel(),
            optimize=True,
        )
        identity = np.eye(3 * n, dtype=np.complex128).reshape(n, 3, n, 3)
        return (identity - interaction).reshape(3 * n, 3 * n)

    def block_jacobi_inverse(self, *, adjoint: bool = False) -> Array:
        """Return inverse 3-by-3 self blocks of the DDA system matrix."""

        indices = np.arange(self.dipole_count)
        diagonal_green = self.total_green_kernel()[indices, :, indices, :]
        diagonal_interaction = self.interaction_scale * np.einsum(
            "iac,icb->iab",
            self.polarizability,
            diagonal_green,
            optimize=True,
        )
        blocks = np.eye(3, dtype=np.complex128)[None, :, :] - diagonal_interaction
        if adjoint:
            blocks = np.conjugate(np.swapaxes(blocks, 1, 2))
        return np.linalg.inv(blocks)

    def solve(
        self,
        incident_field: Any,
        *,
        method: Literal["dense", "gmres"] = "gmres",
        relative_tolerance: float = 1.0e-10,
        maximum_iterations: int | None = None,
        restart: int | None = None,
        preconditioner: Literal["none", "block_jacobi"] = "block_jacobi",
    ) -> DDASolveResult:
        return self._solve_right_hand_side(
            self.right_hand_side(incident_field),
            adjoint=False,
            method=method,
            relative_tolerance=relative_tolerance,
            maximum_iterations=maximum_iterations,
            restart=restart,
            preconditioner=preconditioner,
        )

    def solve_adjoint(
        self,
        cotangent: Any,
        *,
        method: Literal["dense", "gmres"] = "gmres",
        relative_tolerance: float = 1.0e-10,
        maximum_iterations: int | None = None,
        restart: int | None = None,
        preconditioner: Literal["none", "block_jacobi"] = "block_jacobi",
    ) -> DDASolveResult:
        right = self._dipole_field(cotangent, name="cotangent")
        return self._solve_right_hand_side(
            right,
            adjoint=True,
            method=method,
            relative_tolerance=relative_tolerance,
            maximum_iterations=maximum_iterations,
            restart=restart,
            preconditioner=preconditioner,
        )

    def common_height_system_jvp(
        self,
        dipoles: Any,
        *,
        incident_rhs_height_derivative: Any | None = None,
    ) -> Array:
        """Right-hand side for the implicit common-height solution derivative."""

        values = self._dipole_field(dipoles, name="dipoles")
        derivative_field = 2.0 * self.interaction_scale * np.einsum(
            "iajb,jb->ia",
            self.reflected_height_derivative,
            values,
            optimize=True,
        )
        right = np.einsum(
            "iab,ib->ia", self.polarizability, derivative_field, optimize=True
        )
        if incident_rhs_height_derivative is not None:
            right = right + self._dipole_field(
                incident_rhs_height_derivative,
                name="incident_rhs_height_derivative",
            )
        return right

    def rotation_green_derivative(self) -> Array:
        """Derivative of the total dyadic kernel under a global z rotation."""

        omega = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.complex128,
        )
        green = self.total_green_kernel()
        left = np.einsum("ac,icjb->iajb", omega, green, optimize=True)
        right = np.einsum("iajc,cb->iajb", green, omega, optimize=True)
        return left - right

    def rotation_polarizability_derivative(self) -> Array:
        """Derivative of lab-frame polarizability blocks under a z rotation."""

        omega = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.complex128,
        )
        return np.einsum(
            "ac,icb->iab", omega, self.polarizability, optimize=True
        ) - np.einsum(
            "iac,cb->iab", self.polarizability, omega, optimize=True
        )

    def rotation_system_jvp(
        self,
        dipoles: Any,
        *,
        incident_rhs_rotation_derivative: Any | None = None,
    ) -> Array:
        """Right-hand side for the implicit global-orientation derivative."""

        values = self._dipole_field(dipoles, name="dipoles")
        green = self.total_green_kernel()
        base_field = self.interaction_scale * np.einsum(
            "iajb,jb->ia", green, values, optimize=True
        )
        derivative_field = self.interaction_scale * np.einsum(
            "iajb,jb->ia", self.rotation_green_derivative(), values, optimize=True
        )
        right = np.einsum(
            "iab,ib->ia",
            self.rotation_polarizability_derivative(),
            base_field,
            optimize=True,
        ) + np.einsum(
            "iab,ib->ia", self.polarizability, derivative_field, optimize=True
        )
        if incident_rhs_rotation_derivative is not None:
            right = right + self._dipole_field(
                incident_rhs_rotation_derivative,
                name="incident_rhs_rotation_derivative",
            )
        return right

    def solution_height_jvp(
        self,
        dipoles: Any,
        *,
        incident_rhs_height_derivative: Any | None = None,
        method: Literal["dense", "gmres"] = "gmres",
        relative_tolerance: float = 1.0e-10,
    ) -> DDASolveResult:
        right = self.common_height_system_jvp(
            dipoles,
            incident_rhs_height_derivative=incident_rhs_height_derivative,
        )
        return self._solve_right_hand_side(
            right,
            adjoint=False,
            method=method,
            relative_tolerance=relative_tolerance,
            maximum_iterations=None,
            restart=None,
            preconditioner="block_jacobi",
        )

    def solution_height_vjp(
        self,
        dipoles: Any,
        cotangent: Any,
        *,
        incident_rhs_height_derivative: Any | None = None,
        method: Literal["dense", "gmres"] = "gmres",
        relative_tolerance: float = 1.0e-10,
    ) -> tuple[float, DDASolveResult]:
        adjoint = self.solve_adjoint(
            cotangent,
            method=method,
            relative_tolerance=relative_tolerance,
        )
        right = self.common_height_system_jvp(
            dipoles,
            incident_rhs_height_derivative=incident_rhs_height_derivative,
        )
        gradient = float(np.vdot(adjoint.dipoles, right).real)
        return gradient, adjoint

    def solution_rotation_jvp(
        self,
        dipoles: Any,
        *,
        incident_rhs_rotation_derivative: Any | None = None,
        method: Literal["dense", "gmres"] = "gmres",
        relative_tolerance: float = 1.0e-10,
    ) -> DDASolveResult:
        right = self.rotation_system_jvp(
            dipoles,
            incident_rhs_rotation_derivative=incident_rhs_rotation_derivative,
        )
        return self._solve_right_hand_side(
            right,
            adjoint=False,
            method=method,
            relative_tolerance=relative_tolerance,
            maximum_iterations=None,
            restart=None,
            preconditioner="block_jacobi",
        )

    def solution_rotation_vjp(
        self,
        dipoles: Any,
        cotangent: Any,
        *,
        incident_rhs_rotation_derivative: Any | None = None,
        method: Literal["dense", "gmres"] = "gmres",
        relative_tolerance: float = 1.0e-10,
    ) -> tuple[float, DDASolveResult]:
        adjoint = self.solve_adjoint(
            cotangent,
            method=method,
            relative_tolerance=relative_tolerance,
        )
        right = self.rotation_system_jvp(
            dipoles,
            incident_rhs_rotation_derivative=incident_rhs_rotation_derivative,
        )
        gradient = float(np.vdot(adjoint.dipoles, right).real)
        return gradient, adjoint

    def _solve_right_hand_side(
        self,
        right_hand_side: Array,
        *,
        adjoint: bool,
        method: Literal["dense", "gmres"],
        relative_tolerance: float,
        maximum_iterations: int | None,
        restart: int | None,
        preconditioner: Literal["none", "block_jacobi"],
    ) -> DDASolveResult:
        right = self._dipole_field(right_hand_side, name="right_hand_side")
        tolerance = float(relative_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("relative_tolerance must be finite and positive")
        flat_right = right.ravel()
        denominator = max(float(np.linalg.norm(flat_right)), 1.0e-300)
        if method == "dense":
            matrix = self.dense_matrix()
            if adjoint:
                matrix = np.conjugate(matrix.T)
            solution = np.linalg.solve(matrix, flat_right)
            iterations = 1
            info = 0
            used_preconditioner = "none"
        elif method == "gmres":
            size = flat_right.size
            action = self.adjoint_matvec if adjoint else self.matvec
            operator = LinearOperator(
                (size, size),
                matvec=lambda vector: action(vector.reshape(self.field_shape)).ravel(),
                dtype=np.complex128,
            )
            if preconditioner == "none":
                preconditioner_operator = None
            elif preconditioner == "block_jacobi":
                inverse_blocks = self.block_jacobi_inverse(adjoint=adjoint)
                preconditioner_operator = LinearOperator(
                    (size, size),
                    matvec=lambda vector: np.einsum(
                        "iab,ib->ia",
                        inverse_blocks,
                        vector.reshape(self.field_shape),
                        optimize=True,
                    ).ravel(),
                    dtype=np.complex128,
                )
            else:
                raise ValueError("preconditioner must be 'none' or 'block_jacobi'")
            effective_restart = min(size, 200) if restart is None else int(restart)
            if effective_restart <= 0:
                raise ValueError("restart must be positive or None")
            counter = [0]

            def callback(_: Any) -> None:
                counter[0] += 1

            solution, info = gmres(
                operator,
                flat_right,
                rtol=tolerance,
                atol=0.0,
                restart=effective_restart,
                maxiter=maximum_iterations,
                M=preconditioner_operator,
                callback=callback,
                callback_type="pr_norm",
            )
            iterations = counter[0]
            used_preconditioner = preconditioner
        else:
            raise ValueError("method must be 'dense' or 'gmres'")
        dipoles = solution.reshape(self.field_shape)
        residual = (
            self.adjoint_matvec(dipoles) if adjoint else self.matvec(dipoles)
        ) - right
        relative_residual = float(np.linalg.norm(residual.ravel()) / denominator)
        return DDASolveResult(
            dipoles=np.ascontiguousarray(dipoles),
            converged=bool(info == 0 and relative_residual <= 10.0 * tolerance),
            iterations=int(iterations),
            relative_residual=relative_residual,
            method=method,
            preconditioner=used_preconditioner,
        )

    def _dipole_field(self, values: Any, *, name: str) -> Array:
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != self.field_shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {self.field_shape}")
        return array


def plane_wave_field(
    positions: Any,
    wavevector: Any,
    polarization: Any,
    *,
    amplitude: complex = 1.0,
) -> Array:
    """Sample a complex plane wave at dipole positions."""

    points = np.asarray(positions, dtype=np.float64)
    wave = np.asarray(wavevector, dtype=np.complex128)
    vector = np.asarray(polarization, dtype=np.complex128)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("positions must be a finite (n,3) array")
    if wave.shape != (3,) or vector.shape != (3,):
        raise ValueError("wavevector and polarization must have shape (3,)")
    if not np.all(np.isfinite(wave)) or not np.all(np.isfinite(vector)):
        raise ValueError("wavevector and polarization must be finite")
    value = complex(amplitude)
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError("amplitude must be finite")
    phase = np.exp(1j * (points @ wave))
    return value * phase[:, None] * vector[None, :]


def _polarizability_blocks(values: Any, count: int) -> Array:
    array = np.asarray(values, dtype=np.complex128)
    identity = np.eye(3, dtype=np.complex128)
    if array.ndim == 0:
        blocks = np.broadcast_to(array * identity, (count, 3, 3)).copy()
    elif array.shape == (count,):
        blocks = array[:, None, None] * identity[None, :, :]
    elif array.shape == (3, 3):
        blocks = np.broadcast_to(array, (count, 3, 3)).copy()
    elif array.shape == (count, 3, 3):
        blocks = np.array(array, copy=True)
    else:
        raise ValueError(
            "polarizability must be scalar, (n,), (3,3), or (n,3,3)"
        )
    if not np.all(np.isfinite(blocks)):
        raise ValueError("polarizability must contain only finite values")
    for block in blocks:
        singular_values = np.linalg.svd(block, compute_uv=False)
        if singular_values[-1] <= 1.0e-14 * singular_values[0]:
            raise ValueError("polarizability blocks must be nonsingular")
    return np.ascontiguousarray(blocks)


def _cell_centres(lower: float, upper: float, spacing: float) -> Array:
    count = max(1, int(np.ceil((upper - lower) / spacing)))
    actual = (upper - lower) / count
    return lower + (np.arange(count, dtype=np.float64) + 0.5) * actual


def _readonly(values: Array) -> Array:
    array = np.ascontiguousarray(values)
    array.setflags(write=False)
    return array
