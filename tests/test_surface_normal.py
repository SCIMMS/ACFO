from __future__ import annotations

import numpy as np

from waxs_cake import (
    SurfaceNormalCylindricalPlan,
    amplitude_from_sources,
    exact_triangle_amplitude,
    median_line_sources,
    mesh_face_normals,
    mesh_signed_volume,
    q_grid_from_cylindrical,
)
from waxs_cake.geometry import ewald_ring
from waxs_cake.metrics import relative_l2


def _orient_faces_away_from(vertices: np.ndarray, faces: np.ndarray, center: np.ndarray) -> np.ndarray:
    faces = faces.copy()
    normals = mesh_face_normals(vertices, faces)
    centroids = vertices[faces].mean(axis=1)
    inward = np.einsum("ij,ij->i", normals, centroids - center[None, :]) < 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    if mesh_signed_volume(vertices, faces) < 0.0:
        faces = faces[:, [0, 2, 1]].copy()
    return faces


def _sphere_mesh(
    radius: float = 1.0,
    n_lat: int = 20,
    n_lon: int = 40,
    rough: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = [[0.0, 0.0, float(radius)]]
    for i in range(1, n_lat):
        theta = np.pi * i / float(n_lat)
        st = np.sin(theta)
        ct = np.cos(theta)
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / float(n_lon)
            scale = 1.0
            if rough:
                scale += 0.08 * np.sin(3.0 * phi) * np.sin(2.0 * theta)
                scale += 0.04 * np.cos(5.0 * phi + 0.3) * np.sin(theta) ** 2
            vertices.append(
                [
                    float(radius) * scale * st * np.cos(phi),
                    float(radius) * scale * st * np.sin(phi),
                    float(radius) * scale * ct,
                ]
            )
    bottom = len(vertices)
    vertices.append([0.0, 0.0, -float(radius)])

    def ring_index(ring: int, col: int) -> int:
        return 1 + ring * n_lon + (col % n_lon)

    faces: list[list[int]] = []
    for j in range(n_lon):
        faces.append([0, ring_index(0, j), ring_index(0, j + 1)])
    for ring in range(n_lat - 2):
        for j in range(n_lon):
            a = ring_index(ring, j)
            b = ring_index(ring, j + 1)
            c = ring_index(ring + 1, j)
            d = ring_index(ring + 1, j + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])
    last_ring = n_lat - 2
    for j in range(n_lon):
        faces.append([ring_index(last_ring, j), bottom, ring_index(last_ring, j + 1)])

    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = _orient_faces_away_from(vertices_arr, np.asarray(faces, dtype=np.int64), np.zeros(3))
    return vertices_arr, faces_arr


def _ellipsoid_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _sphere_mesh(n_lat=18, n_lon=36)
    return vertices * np.asarray([1.0, 0.65, 0.42]), faces


def _cone_mesh(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    vertices = [[0.0, 0.0, 0.65], [0.0, 0.0, -0.45]]
    for j in range(n):
        phi = 2.0 * np.pi * j / n
        vertices.append([0.58 * np.cos(phi), 0.58 * np.sin(phi), -0.45])
    faces = []
    for j in range(n):
        a = 2 + j
        b = 2 + ((j + 1) % n)
        faces.append([0, a, b])
        faces.append([1, b, a])
    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = _orient_faces_away_from(vertices_arr, np.asarray(faces, dtype=np.int64), np.array([0.0, 0.0, -0.1]))
    return vertices_arr, faces_arr


def _double_cone_mesh(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    vertices = [[0.0, 0.0, 0.65], [0.0, 0.0, -0.65]]
    for j in range(n):
        phi = 2.0 * np.pi * j / n
        vertices.append([0.55 * np.cos(phi), 0.55 * np.sin(phi), 0.0])
    faces = []
    for j in range(n):
        a = 2 + j
        b = 2 + ((j + 1) % n)
        faces.append([0, a, b])
        faces.append([1, b, a])
    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = _orient_faces_away_from(vertices_arr, np.asarray(faces, dtype=np.int64), np.zeros(3))
    return vertices_arr, faces_arr


def _shell_mesh() -> tuple[np.ndarray, np.ndarray]:
    outer_v, outer_f = _sphere_mesh(radius=1.0, n_lat=16, n_lon=32)
    inner_v, inner_f = _sphere_mesh(radius=0.55, n_lat=12, n_lon=24)
    inner_f = inner_f[:, [0, 2, 1]]
    vertices = np.vstack([outer_v, inner_v])
    faces = np.vstack([outer_f, inner_f + outer_v.shape[0]])
    return vertices, faces


def _sphere_form_factor(q: np.ndarray, radius: float = 1.0) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    qr = q * radius
    volume = 4.0 * np.pi * radius**3 / 3.0
    out = np.empty_like(q)
    small = np.abs(qr) < 1e-6
    out[small] = volume
    z = qr[~small]
    out[~small] = volume * 3.0 * (np.sin(z) - z * np.cos(z)) / (z**3)
    return out


def _q_line(q_values: np.ndarray, direction: tuple[float, float, float]) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float64)
    vec /= np.linalg.norm(vec)
    return q_values[:, None] * vec[None, :]


def test_exact_triangle_sphere_and_low_q_volume() -> None:
    vertices, faces = _sphere_mesh(radius=1.0, n_lat=24, n_lon=48)
    q_values = np.linspace(0.0, 3.5, 28)
    q_xyz = _q_line(q_values, (1.0, 0.31, -0.17))

    got = exact_triangle_amplitude((vertices, faces), q_xyz, q_switch=1e-9, q_block_size=12)
    expected = _sphere_form_factor(q_values)

    assert abs(got[0].real - mesh_signed_volume(vertices, faces)) < 1e-12
    assert np.linalg.norm(got.imag) / np.linalg.norm(got) < 1e-12
    assert relative_l2(got.real, expected) < 8e-3


def test_median_line_sources_match_exact_reference_for_morphologies() -> None:
    cases = [
        _sphere_mesh(n_lat=18, n_lon=36),
        _ellipsoid_mesh(),
        _cone_mesh(),
        _double_cone_mesh(),
        _shell_mesh(),
        _sphere_mesh(n_lat=18, n_lon=36, rough=True),
    ]
    q_values = np.linspace(0.25, 2.25, 18)
    q_xyz = _q_line(q_values, (0.42, -0.31, 0.85))

    for mesh in cases:
        exact = exact_triangle_amplitude(mesh, q_xyz, q_switch=1e-9, q_block_size=12)
        sources = median_line_sources(mesh, K=4)
        got = amplitude_from_sources(sources, q_xyz, backend="direct", q_switch=1e-9)
        assert relative_l2(got, exact) < 6e-3


def test_surface_normal_cylindrical_plan_matches_direct_ewald_and_flat() -> None:
    mesh = _sphere_mesh(n_lat=14, n_lon=28)
    sources = median_line_sources(mesh, K=3)
    q = np.linspace(0.25, 2.2, 7)
    wavelength = 0.1

    ewald_plan = SurfaceNormalCylindricalPlan.ewald_ring(
        sources,
        q,
        wavelength,
        n_phi=128,
        n_r=56,
        n_z=56,
        circular_backend="numpy",
    )
    q_perp, q_z = ewald_ring(q, wavelength)
    q_xyz = q_grid_from_cylindrical(q_perp, q_z, ewald_plan.phi)
    direct = amplitude_from_sources(sources, q_xyz, backend="direct").reshape(q.size, ewald_plan.phi.size)
    got = ewald_plan.circular_fft()
    assert relative_l2(got, direct) < 8e-4

    flat_q = np.linspace(0.0, 2.2, 8)
    flat_plan = SurfaceNormalCylindricalPlan.flat_detector(
        sources,
        flat_q,
        n_phi=128,
        n_r=56,
        n_z=56,
        circular_backend="numpy",
    )
    flat_xyz = q_grid_from_cylindrical(flat_q, np.zeros_like(flat_q), flat_plan.phi)
    flat_direct = amplitude_from_sources(sources, flat_xyz, backend="direct").reshape(
        flat_q.size,
        flat_plan.phi.size,
    )
    flat_got = flat_plan.circular_fft()
    assert np.allclose(flat_got[0], sources.moments.volume)
    assert relative_l2(flat_got[1:], flat_direct[1:]) < 8e-4


def test_surface_normal_r_dependent_path_matches_full_circular_with_large_margin() -> None:
    mesh = _ellipsoid_mesh()
    sources = median_line_sources(mesh, K=3)
    q = np.linspace(0.35, 3.0, 9)
    plan = SurfaceNormalCylindricalPlan.flat_detector(
        sources,
        q,
        n_phi=192,
        n_r=64,
        n_z=64,
        circular_backend="numpy",
    )

    expected = plan.circular_fft()
    got = plan.circular_fft_r_dependent_bandlimit(
        margin=10_000,
        cutoff_bin_size=8,
        analytic_kernel=True,
    )

    assert relative_l2(got, expected) < 1e-10
