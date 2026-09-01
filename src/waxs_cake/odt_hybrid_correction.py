"""Hybrid Cartesian-volume and on-shell boundary-defect correction tools.

The volume path treats a cell-centred Cartesian array as a union of constant
voxels.  Its transform is a point-centre DFT multiplied by the exact voxel
shape factor.  A prepared surface-normal ACFO plan can then add the difference
between a smooth physical boundary and the staircased voxel boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .surface_normal import (
    SurfaceNormalCylindricalPlan,
    SurfaceNormalMoments,
    SurfaceNormalSources,
    coerce_mesh,
    median_line_sources,
    mesh_face_normals,
    mesh_signed_volume,
    q_grid_from_cylindrical,
)


ArrayLike = Any
Bounds3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def _unit_direction(value: ArrayLike, name: str) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return direction / norm


def _validate_bounds(bounds: Bounds3D) -> tuple[np.ndarray, np.ndarray]:
    if len(bounds) != 3:
        raise ValueError("bounds must contain x, y, and z intervals")
    lower = np.asarray([interval[0] for interval in bounds], dtype=np.float64)
    upper = np.asarray([interval[1] for interval in bounds], dtype=np.float64)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("bounds must be finite")
    if np.any(upper <= lower):
        raise ValueError("every bound must be increasing")
    return lower, upper


def _q_array(q_xyz: ArrayLike) -> np.ndarray:
    q = np.asarray(q_xyz, dtype=np.float64)
    if q.ndim < 2 or q.shape[-1] != 3:
        raise ValueError("q_xyz must have shape (..., 3)")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_xyz must contain only finite values")
    return q


@dataclass(frozen=True)
class UniformSphere:
    """Piecewise-constant spherical inclusion."""

    center: tuple[float, float, float]
    radius: float
    contrast: float = 1.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite three-vector")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        if not np.isfinite(self.contrast):
            raise ValueError("contrast must be finite")

    @property
    def volume(self) -> float:
        return 4.0 * np.pi * float(self.radius) ** 3 / 3.0


@dataclass(frozen=True)
class UniformEllipsoid:
    """Uniform ellipsoid with a proper local-to-laboratory rotation."""

    center: tuple[float, float, float]
    semiaxes: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    contrast: float = 1.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        semiaxes = np.asarray(self.semiaxes, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite three-vector")
        if semiaxes.shape != (3,) or not np.all(np.isfinite(semiaxes)) or np.any(semiaxes <= 0.0):
            raise ValueError("semiaxes must be three positive finite values")
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation must be a finite 3 x 3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=1e-12, atol=1e-12):
            raise ValueError("rotation must be orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("rotation must be proper (determinant +1)")
        if not np.isfinite(self.contrast):
            raise ValueError("contrast must be finite")

    @property
    def volume(self) -> float:
        return 4.0 * np.pi * float(np.prod(self.semiaxes)) / 3.0


def _normalized_sphere_form(qr: np.ndarray) -> np.ndarray:
    form = np.ones_like(qr)
    active = np.abs(qr) > 1e-6
    if np.any(active):
        x = qr[active]
        form[active] = 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3)
    if np.any(~active):
        x = qr[~active]
        x2 = x * x
        form[~active] = 1.0 - x2 / 10.0 + x2 * x2 / 280.0
    return form


def sphere_fourier(
    spheres: UniformSphere | Iterable[UniformSphere],
    q_xyz: ArrayLike,
    *,
    phase_sign: int = 1,
) -> np.ndarray:
    """Analytic Fourier transform of noninteracting uniform spheres."""

    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be -1 or +1")
    sphere_list = (spheres,) if isinstance(spheres, UniformSphere) else tuple(spheres)
    if not sphere_list:
        raise ValueError("at least one sphere is required")
    q = _q_array(q_xyz)
    flat_q = q.reshape(-1, 3)
    q_norm = np.linalg.norm(flat_q, axis=1)
    out = np.zeros(flat_q.shape[0], dtype=np.complex128)
    for sphere in sphere_list:
        qr = q_norm * float(sphere.radius)
        form = _normalized_sphere_form(qr)
        phase = np.exp(
            1j
            * float(phase_sign)
            * (flat_q @ np.asarray(sphere.center, dtype=np.float64))
        )
        out += float(sphere.contrast) * sphere.volume * form * phase
    return out.reshape(q.shape[:-1])


def ellipsoid_fourier(
    ellipsoid: UniformEllipsoid,
    q_xyz: ArrayLike,
    *,
    phase_sign: int = 1,
) -> np.ndarray:
    """Analytic transform of a rotated uniform ellipsoid."""

    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be -1 or +1")
    q = _q_array(q_xyz)
    flat_q = q.reshape(-1, 3)
    rotation = np.asarray(ellipsoid.rotation, dtype=np.float64)
    semiaxes = np.asarray(ellipsoid.semiaxes, dtype=np.float64)
    qr = np.linalg.norm((flat_q @ rotation) * semiaxes[None, :], axis=1)
    phase = np.exp(
        1j
        * float(phase_sign)
        * (flat_q @ np.asarray(ellipsoid.center, dtype=np.float64))
    )
    out = float(ellipsoid.contrast) * ellipsoid.volume * _normalized_sphere_form(qr) * phase
    return out.reshape(q.shape[:-1])


def sample_spheres_on_voxels(
    spheres: UniformSphere | Iterable[UniformSphere],
    shape: tuple[int, int, int],
    bounds: Bounds3D,
) -> np.ndarray:
    """Sample sphere indicators at Cartesian voxel centres."""

    sphere_list = (spheres,) if isinstance(spheres, UniformSphere) else tuple(spheres)
    if not sphere_list:
        raise ValueError("at least one sphere is required")
    grid_shape = np.asarray(shape, dtype=np.int64)
    if grid_shape.shape != (3,) or np.any(grid_shape <= 0):
        raise ValueError("shape must contain three positive integers")
    lower, upper = _validate_bounds(bounds)
    spacing = (upper - lower) / grid_shape
    axes = tuple(
        lower[axis] + (np.arange(grid_shape[axis]) + 0.5) * spacing[axis]
        for axis in range(3)
    )
    x, y, z = np.meshgrid(*axes, indexing="ij")
    values = np.zeros(tuple(int(v) for v in grid_shape), dtype=np.float64)
    for sphere in sphere_list:
        center = np.asarray(sphere.center, dtype=np.float64)
        inside = (
            (x - center[0]) ** 2
            + (y - center[1]) ** 2
            + (z - center[2]) ** 2
            <= float(sphere.radius) ** 2
        )
        values[inside] += float(sphere.contrast)
    return np.ascontiguousarray(values)


def sample_ellipsoid_on_voxels(
    ellipsoid: UniformEllipsoid,
    shape: tuple[int, int, int],
    bounds: Bounds3D,
) -> np.ndarray:
    """Sample a rotated ellipsoid indicator at Cartesian voxel centres."""

    grid_shape = np.asarray(shape, dtype=np.int64)
    if grid_shape.shape != (3,) or np.any(grid_shape <= 0):
        raise ValueError("shape must contain three positive integers")
    lower, upper = _validate_bounds(bounds)
    spacing = (upper - lower) / grid_shape
    axes = tuple(
        lower[axis] + (np.arange(grid_shape[axis]) + 0.5) * spacing[axis]
        for axis in range(3)
    )
    x, y, z = np.meshgrid(*axes, indexing="ij")
    points = np.stack([x, y, z], axis=-1)
    local = (points - np.asarray(ellipsoid.center)) @ np.asarray(ellipsoid.rotation)
    inside = np.sum((local / np.asarray(ellipsoid.semiaxes)) ** 2, axis=-1) <= 1.0
    return np.ascontiguousarray(inside.astype(np.float64) * float(ellipsoid.contrast))


def sphere_mesh(
    sphere: UniformSphere,
    *,
    n_lat: int = 24,
    n_lon: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an outward-oriented latitude-longitude triangular sphere mesh."""

    n_lat = int(n_lat)
    n_lon = int(n_lon)
    if n_lat < 3 or n_lon < 4:
        raise ValueError("n_lat >= 3 and n_lon >= 4 are required")
    radius = float(sphere.radius)
    center = np.asarray(sphere.center, dtype=np.float64)
    vertices: list[list[float]] = [[0.0, 0.0, radius]]
    for i in range(1, n_lat):
        theta = np.pi * i / float(n_lat)
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / float(n_lon)
            vertices.append(
                [
                    radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta),
                ]
            )
    bottom = len(vertices)
    vertices.append([0.0, 0.0, -radius])

    def ring_index(ring: int, column: int) -> int:
        return 1 + ring * n_lon + (column % n_lon)

    faces: list[list[int]] = []
    for j in range(n_lon):
        faces.append([0, ring_index(0, j), ring_index(0, j + 1)])
    for ring in range(n_lat - 2):
        for j in range(n_lon):
            a = ring_index(ring, j)
            b = ring_index(ring, j + 1)
            c = ring_index(ring + 1, j)
            d = ring_index(ring + 1, j + 1)
            faces.extend(([a, c, b], [b, c, d]))
    last_ring = n_lat - 2
    for j in range(n_lon):
        faces.append([ring_index(last_ring, j), bottom, ring_index(last_ring, j + 1)])

    vertices_array = np.asarray(vertices, dtype=np.float64) + center[None, :]
    faces_array = np.asarray(faces, dtype=np.int64)
    normals = mesh_face_normals(vertices_array, faces_array)
    centroids = vertices_array[faces_array].mean(axis=1)
    inward = np.einsum("ij,ij->i", normals, centroids - center[None, :]) < 0.0
    faces_array[inward] = faces_array[inward][:, [0, 2, 1]]
    if mesh_signed_volume(vertices_array, faces_array) < 0.0:
        faces_array = faces_array[:, [0, 2, 1]].copy()
    return vertices_array, faces_array


def ellipsoid_mesh(
    ellipsoid: UniformEllipsoid,
    *,
    n_lat: int = 24,
    n_lon: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an outward-oriented triangular mesh of a rotated ellipsoid."""

    unit_vertices, faces = sphere_mesh(
        UniformSphere((0.0, 0.0, 0.0), 1.0),
        n_lat=n_lat,
        n_lon=n_lon,
    )
    vertices = (
        (unit_vertices * np.asarray(ellipsoid.semiaxes)[None, :])
        @ np.asarray(ellipsoid.rotation).T
        + np.asarray(ellipsoid.center)[None, :]
    )
    if mesh_signed_volume(vertices, faces) < 0.0:
        faces = faces[:, [0, 2, 1]].copy()
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def voxel_union_surface_mesh(
    occupied: ArrayLike,
    bounds: Bounds3D,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate exposed faces of a binary Cartesian voxel union."""

    mask = np.asarray(occupied, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("occupied must be a nonempty three-dimensional mask")
    lower, upper = _validate_bounds(bounds)
    shape = np.asarray(mask.shape, dtype=np.int64)
    spacing = (upper - lower) / shape

    # Quad corner orders are outward when viewed from outside the occupied cell.
    face_specs = (
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    for index in np.argwhere(mask):
        for direction, corners in face_specs:
            neighbor = index + np.asarray(direction, dtype=np.int64)
            exposed = np.any(neighbor < 0) or np.any(neighbor >= shape)
            if not exposed:
                exposed = not bool(mask[tuple(neighbor)])
            if not exposed:
                continue
            base = len(vertices)
            cell_origin = lower + index * spacing
            vertices.extend(
                cell_origin + np.asarray(corner, dtype=np.float64) * spacing
                for corner in corners
            )
            faces.extend(([base, base + 1, base + 2], [base, base + 2, base + 3]))
    mesh = coerce_mesh((np.asarray(vertices), np.asarray(faces, dtype=np.int64)))
    return mesh.vertices, mesh.faces


def _voxel_geometry(values: np.ndarray, bounds: Bounds3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = _validate_bounds(bounds)
    shape = np.asarray(values.shape, dtype=np.int64)
    if values.ndim != 3 or np.any(shape <= 0):
        raise ValueError("values must be a nonempty three-dimensional array")
    spacing = (upper - lower) / shape
    first_center = lower + 0.5 * spacing
    return lower, spacing, first_center


def voxel_shape_factor(q_xyz: ArrayLike, spacing: ArrayLike) -> np.ndarray:
    """Exact normalized Fourier transform of one centred rectangular voxel."""

    q = _q_array(q_xyz)
    h = np.asarray(spacing, dtype=np.float64)
    if h.shape != (3,) or np.any(h <= 0.0):
        raise ValueError("spacing must be a positive three-vector")
    return np.prod(np.sinc(q * h / (2.0 * np.pi)), axis=-1)


def direct_voxel_fourier(
    values: ArrayLike,
    bounds: Bounds3D,
    q_xyz: ArrayLike,
    *,
    phase_sign: int = 1,
    continuous_voxels: bool = True,
    q_block_size: int = 256,
) -> np.ndarray:
    """Direct transform of cell-centred voxel values on arbitrary q nodes."""

    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be -1 or +1")
    array = np.asarray(values, dtype=np.complex128)
    _, spacing, first_center = _voxel_geometry(array, bounds)
    axes = tuple(first_center[axis] + np.arange(array.shape[axis]) * spacing[axis] for axis in range(3))
    x, y, z = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    weights = array.ravel() * float(np.prod(spacing))
    q = _q_array(q_xyz)
    flat_q = q.reshape(-1, 3)
    out = np.empty(flat_q.shape[0], dtype=np.complex128)
    block_size = int(q_block_size)
    if block_size <= 0:
        raise ValueError("q_block_size must be positive")
    for start in range(0, flat_q.shape[0], block_size):
        local = slice(start, min(start + block_size, flat_q.shape[0]))
        out[local] = np.exp(
            1j * float(phase_sign) * (flat_q[local] @ points.T)
        ) @ weights
    if continuous_voxels:
        out *= voxel_shape_factor(flat_q.reshape(q.shape), spacing).ravel()
    return out.reshape(q.shape[:-1])


def interpolated_voxel_fft(
    values: ArrayLike,
    bounds: Bounds3D,
    q_xyz: ArrayLike,
    *,
    phase_sign: int = 1,
    pad_factor: int = 2,
    method: str = "linear",
    continuous_voxels: bool = True,
) -> np.ndarray:
    """Zero-padded 3-D FFT followed by Cartesian reciprocal interpolation."""

    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be -1 or +1")
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'")
    pad_factor = int(pad_factor)
    if pad_factor < 1:
        raise ValueError("pad_factor must be at least one")
    array = np.asarray(values, dtype=np.complex128)
    _, spacing, first_center = _voxel_geometry(array, bounds)
    padded_shape = tuple(pad_factor * size for size in array.shape)
    padded = np.zeros(padded_shape, dtype=np.complex128)
    padded[tuple(slice(0, size) for size in array.shape)] = array * float(np.prod(spacing))
    if phase_sign == 1:
        transform = np.fft.ifftn(padded) * float(np.prod(padded_shape))
    else:
        transform = np.fft.fftn(padded)
    q_axes = tuple(2.0 * np.pi * np.fft.fftfreq(padded_shape[axis], d=spacing[axis]) for axis in range(3))
    q_mesh = np.meshgrid(*q_axes, indexing="ij", sparse=True)
    phase = np.exp(
        1j
        * float(phase_sign)
        * sum(q_mesh[axis] * first_center[axis] for axis in range(3))
    )
    transform *= phase
    shifted_axes = tuple(np.fft.fftshift(axis) for axis in q_axes)
    shifted = np.fft.fftshift(transform)

    q = _q_array(q_xyz)
    flat_q = q.reshape(-1, 3)
    real_interp = RegularGridInterpolator(
        shifted_axes,
        shifted.real,
        method=method,
        bounds_error=True,
    )
    imag_interp = RegularGridInterpolator(
        shifted_axes,
        shifted.imag,
        method=method,
        bounds_error=True,
    )
    out = real_interp(flat_q) + 1j * imag_interp(flat_q)
    if continuous_voxels:
        out *= voxel_shape_factor(q, spacing).ravel()
    return out.reshape(q.shape[:-1])


def phase_modulated_surface_sources(
    sources: SurfaceNormalSources,
    q_shift: ArrayLike,
    *,
    phase_sign: int = 1,
) -> SurfaceNormalSources:
    """Move a fixed reciprocal-space shift into the boundary source weights.

    For ``q_total = q_base + q_shift``, the identity
    ``exp(i q_total.r) = exp(i q_base.r) exp(i q_shift.r)`` preserves the
    complete-ring geometry of ``q_base``.  Low-q moments are intentionally
    dropped because the modulated source no longer has the original moments.
    """

    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be -1 or +1")
    shift = np.asarray(q_shift, dtype=np.float64)
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError("q_shift must be a finite three-vector")
    phase = np.exp(
        1j
        * float(phase_sign)
        * (np.asarray(sources.points, dtype=np.float64) @ shift)
    )
    return SurfaceNormalSources(
        points=np.array(sources.points, copy=True),
        vector_weights=np.ascontiguousarray(sources.vector_weights * phase[:, None]),
        moments=None,
        metadata={
            **dict(sources.metadata),
            "source_type": "phase-modulated-boundary-source",
            "q_shift": shift.tolist(),
            "phase_sign": int(phase_sign),
        },
    )


class ShiftedEwaldSurfacePlan:
    """Surface-normal ACFO evaluator for an off-axis ODT Ewald cap.

    The detector cap remains a stack of complete rings about the laboratory
    ``+z`` axis.  Changing the incident direction translates that cap by
    ``k (zhat - s_in)`` in reciprocal space.  This fixed translation is moved
    into the surface weights, while the final divergence-theorem contraction
    uses the physical shifted scattering vector.

    One prepared plan is currently required per incident direction.  The
    implementation is therefore a correctness and regime-boundary primitive,
    not yet a fused multi-illumination contraction.
    """

    def __init__(
        self,
        sources: SurfaceNormalSources,
        q: ArrayLike,
        wavelength: float,
        incident_direction: ArrayLike,
        **plan_kwargs: Any,
    ) -> None:
        wavelength = float(wavelength)
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("wavelength must be positive and finite")
        incident = _unit_direction(incident_direction, "incident_direction")
        reference = np.asarray([0.0, 0.0, 1.0])
        wave_number = 2.0 * np.pi / wavelength
        self.q_shift = wave_number * (reference - incident)
        self.incident_direction = incident
        self.reference_direction = reference
        modulated = phase_modulated_surface_sources(sources, self.q_shift)
        # The base plan only evaluates vector surface fields, so the missing
        # low-q moments on the modulated source are harmless.  The physical
        # q_total=0 case is rejected explicitly in amplitude().
        self.base_plan = SurfaceNormalCylindricalPlan.from_sources(
            modulated,
            q=q,
            wavelength=wavelength,
            **plan_kwargs,
        )

    @property
    def phi(self) -> np.ndarray:
        return self.base_plan.phi

    @property
    def q_xyz(self) -> np.ndarray:
        base = q_grid_from_cylindrical(
            self.base_plan.q_perp,
            self.base_plan.q_z,
            self.phi,
        ).reshape(self.base_plan.q.size, self.phi.size, 3)
        return base + self.q_shift[None, None, :]

    def field_components(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return phase-shifted vector boundary transforms on base rings."""

        return self.base_plan.field_components(**kwargs)

    def amplitude(self, **kwargs: Any) -> np.ndarray:
        """Return the boundary amplitude at the physical shifted-cap nodes."""

        components = self.field_components(**kwargs)
        q_xyz = self.q_xyz
        q_norm2 = np.sum(q_xyz * q_xyz, axis=-1)
        if np.any(q_norm2 <= 0.0):
            raise ValueError(
                "shifted cap contains q_total=0; use an explicit low-q moment path"
            )
        numer = sum(q_xyz[..., axis] * components[axis] for axis in range(3))
        return np.ascontiguousarray(numer / (1j * q_norm2))

    def circular_fft(self, **kwargs: Any) -> np.ndarray:
        return self.amplitude(method="circular", **kwargs)


def fixed_boundary_coefficients_forward(
    coefficients: ArrayLike,
    boundary_templates: ArrayLike,
) -> np.ndarray:
    """Apply a linear combination of precomputed fixed-boundary amplitudes."""

    templates = np.asarray(boundary_templates, dtype=np.complex128)
    coeff = np.asarray(coefficients, dtype=np.complex128)
    if templates.ndim < 2:
        raise ValueError("boundary_templates must have shape (channels, ...)")
    if coeff.shape != (templates.shape[0],):
        raise ValueError("coefficients must match the boundary template channels")
    return np.ascontiguousarray(np.tensordot(coeff, templates, axes=(0, 0)))


def fixed_boundary_coefficients_adjoint(
    boundary_templates: ArrayLike,
    residual: ArrayLike,
    *,
    data_weights: ArrayLike | None = None,
) -> np.ndarray:
    """Exact adjoint for fixed-shape contrast/template coefficients.

    This is deliberately narrower than a voxel or shape adjoint.  It supports
    inverse problems where geometry is fixed and only a small set of complex
    contrast, material, or illumination coefficients changes.
    """

    templates = np.asarray(boundary_templates, dtype=np.complex128)
    values = np.asarray(residual, dtype=np.complex128)
    if templates.ndim < 2 or values.shape != templates.shape[1:]:
        raise ValueError("residual must match the non-channel template shape")
    if data_weights is not None:
        weights = np.asarray(data_weights, dtype=np.float64)
        try:
            values = values * np.broadcast_to(weights, values.shape)
        except ValueError as exc:
            raise ValueError("data_weights must broadcast to the residual shape") from exc
    flattened = values.reshape(-1)
    return np.ascontiguousarray(np.conj(templates.reshape(templates.shape[0], -1)) @ flattened)


def combine_surface_sources(sources: Iterable[SurfaceNormalSources]) -> SurfaceNormalSources:
    """Concatenate vector sources and sum compatible low-q moments."""

    source_list = tuple(sources)
    if not source_list:
        raise ValueError("at least one source set is required")
    points = np.ascontiguousarray(np.concatenate([item.points for item in source_list], axis=0))
    weights = np.ascontiguousarray(
        np.concatenate([item.vector_weights for item in source_list], axis=0)
    )
    moments = None
    if all(item.moments is not None for item in source_list):
        moments = SurfaceNormalMoments(
            volume=float(sum(item.moments.volume for item in source_list if item.moments is not None)),
            first_moment=np.sum(
                [item.moments.first_moment for item in source_list if item.moments is not None],
                axis=0,
            ),
            second_moment=np.sum(
                [item.moments.second_moment for item in source_list if item.moments is not None],
                axis=0,
            ),
            delta_rho=1.0,
        )
    return SurfaceNormalSources(
        points=points,
        vector_weights=weights,
        moments=moments,
        metadata={
            "source_type": "combined-boundary-defect",
            "parts": len(source_list),
            "num_sources": int(points.shape[0]),
        },
    )


def boundary_defect_sources(
    smooth_meshes: Iterable[tuple[Any, float]],
    voxel_mesh: Any,
    *,
    voxel_contrast: float = 1.0,
    quadrature_order: int = 4,
) -> SurfaceNormalSources:
    """Build ``smooth boundary - voxel boundary`` vector source quadrature."""

    parts = [
        median_line_sources(mesh, K=quadrature_order, delta_rho=float(contrast))
        for mesh, contrast in smooth_meshes
    ]
    if not parts:
        raise ValueError("at least one smooth mesh is required")
    parts.append(
        median_line_sources(
            voxel_mesh,
            K=quadrature_order,
            delta_rho=-float(voxel_contrast),
        )
    )
    return combine_surface_sources(parts)
