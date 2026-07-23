"""Surface-normal boundary-integral SAXS from closed triangular meshes.

The fast path converts a sharp-interface mesh into q-independent vector
surface sources, evaluates the three vector Fourier components, then combines
them as ``A(q) = q . F(q) / (i |q|^2)``.  Exact triangle integration is kept as
the reference path because its face weights depend on q and do not directly fit
the prepared circular-harmonic or NUFFT factorization.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .geometry import ewald_ring
from .histogram import BinnedStructure, make_cylindrical_histogram
from .solvers import PreparedCakePlan


ArrayLike = Any
SurfaceBackend = Literal["direct", "finufft", "cufinufft"]


@dataclass(frozen=True)
class MeshData:
    """Validated triangular mesh arrays."""

    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    signed_volume: float
    faces_flipped: bool = False


@dataclass(frozen=True)
class SurfaceNormalMoments:
    """Low-q volume moments for a piecewise-constant sharp-interface object."""

    volume: float
    first_moment: np.ndarray
    second_moment: np.ndarray
    delta_rho: float = 1.0


@dataclass(frozen=True)
class SurfaceNormalSources:
    """Point/vector source representation used by fast transforms."""

    points: np.ndarray
    vector_weights: np.ndarray
    metadata: dict[str, Any]
    moments: SurfaceNormalMoments | None = None


def _as_q_array(q_xyz: ArrayLike) -> tuple[np.ndarray, bool]:
    q = np.asarray(q_xyz, dtype=np.float64)
    scalar_input = False
    if q.ndim == 1:
        if q.shape[0] != 3:
            raise ValueError("q_xyz must have shape (3,) or (M, 3)")
        q = q.reshape(1, 3)
        scalar_input = True
    elif q.ndim == 2:
        if q.shape[1] != 3:
            raise ValueError("q_xyz must have shape (3,) or (M, 3)")
    else:
        raise ValueError("q_xyz must have shape (3,) or (M, 3)")
    return q, scalar_input


def _validate_phase_sign(phase_sign: int) -> None:
    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be +1 or -1")


def mesh_face_normals(vertices: ArrayLike, faces: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """Return unit face normals using the current triangle winding."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if np.any(norms[:, 0] < eps):
        raise ValueError("mesh contains at least one degenerate triangle")
    return raw / norms


def mesh_signed_volume(vertices: ArrayLike, faces: ArrayLike) -> float:
    """Signed volume of a closed triangular mesh."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def _extract_mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if isinstance(mesh, MeshData):
        return mesh.vertices, mesh.faces, mesh.face_normals
    if isinstance(mesh, dict):
        vertices = mesh.get("vertices")
        faces = mesh.get("faces")
        face_normals = mesh.get("face_normals")
    elif isinstance(mesh, (tuple, list)) and len(mesh) >= 2:
        vertices = mesh[0]
        faces = mesh[1]
        face_normals = mesh[2] if len(mesh) >= 3 else None
    else:
        vertices = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        face_normals = getattr(mesh, "face_normals", None)

    if vertices is None or faces is None:
        raise TypeError(
            "mesh must be MeshData, a dict with vertices/faces, a "
            "(vertices, faces) tuple, or an object with vertices and faces"
        )
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        None if face_normals is None else np.asarray(face_normals, dtype=np.float64),
    )


def coerce_mesh(mesh: Any, *, orient_positive: bool = True) -> MeshData:
    """Convert a mesh-like object to validated ``MeshData``."""

    vertices, faces, face_normals = _extract_mesh_arrays(mesh)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh faces must have shape (F, 3)")
    if faces.shape[0] == 0:
        raise ValueError("mesh has no faces")

    volume = mesh_signed_volume(vertices, faces)
    flipped = False
    if orient_positive and volume < 0.0:
        faces = faces[:, [0, 2, 1]].copy()
        volume = -volume
        face_normals = None
        flipped = True

    if face_normals is None:
        face_normals = mesh_face_normals(vertices, faces)
    else:
        if face_normals.shape != (faces.shape[0], 3):
            raise ValueError("face_normals must have shape (F, 3)")
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        if np.any(norms[:, 0] <= 0.0):
            raise ValueError("face_normals contains zero-length normals")
        face_normals = face_normals / norms

    return MeshData(
        vertices=vertices,
        faces=faces.astype(np.int64, copy=False),
        face_normals=face_normals.astype(np.float64, copy=False),
        signed_volume=float(volume),
        faces_flipped=flipped,
    )


def mesh_volume_moments(
    mesh: Any,
    *,
    delta_rho: float = 1.0,
    orient_positive: bool = True,
) -> SurfaceNormalMoments:
    """Return volume, first moment, and second raw moment from mesh tetrahedra."""

    mesh_data = coerce_mesh(mesh, orient_positive=orient_positive)
    tri = mesh_data.vertices[mesh_data.faces]
    signed_vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0

    volume = float(signed_vol.sum())
    vertex_sum = tri.sum(axis=1)
    first = np.einsum("f,fi->i", signed_vol / 4.0, vertex_sum)

    second = np.zeros((3, 3), dtype=np.float64)
    for vol, verts, summed in zip(signed_vol, tri, vertex_sum):
        second += (vol / 20.0) * (summed[:, None] * summed[None, :] + verts.T @ verts)

    density = float(delta_rho)
    return SurfaceNormalMoments(
        volume=density * volume,
        first_moment=density * first,
        second_moment=density * second,
        delta_rho=density,
    )


def moment_expansion_amplitude(
    moments: SurfaceNormalMoments,
    q_xyz: ArrayLike,
    *,
    phase_sign: int = 1,
    order: int = 2,
) -> np.ndarray:
    """Evaluate the low-q expansion through the requested moment order."""

    _validate_phase_sign(phase_sign)
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1, or 2")
    q, scalar_input = _as_q_array(q_xyz)
    out = np.full(q.shape[0], complex(moments.volume), dtype=np.complex128)
    if order >= 1:
        out += 1j * float(phase_sign) * (q @ moments.first_moment)
    if order >= 2:
        out -= 0.5 * np.einsum("mi,ij,mj->m", q, moments.second_moment, q)
    return out[0] if scalar_input else out


def _sinc_unscaled(x: np.ndarray) -> np.ndarray:
    return np.sinc(x / np.pi)


def _g(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(0.5j * x) * _sinc_unscaled(0.5 * x)


def _h_equal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.complex128)
    small = np.abs(x) < 1e-4
    if np.any(~small):
        z = x[~small]
        iz = 1j * z
        ez = np.exp(1j * z)
        out[~small] = ez / iz + (1.0 - ez) / (iz * iz)
    if np.any(small):
        z = x[small].astype(np.complex128)
        term = np.full(z.shape, 0.5 + 0.0j, dtype=np.complex128)
        power = np.ones(z.shape, dtype=np.complex128)
        factorial = 1.0
        for n in range(1, 12):
            power = power * (1j * z)
            factorial *= float(n)
            term += power / (factorial * float(n + 2))
        out[small] = term
    return out


def triangle_h(a: ArrayLike, b: ArrayLike, *, near_tol: float = 1e-8) -> np.ndarray:
    """Stable exact ``H(a,b)`` over a reference triangle."""

    a_arr, b_arr = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    out = np.empty(a_arr.shape, dtype=np.complex128)
    scale = 1.0 + np.maximum(np.abs(a_arr), np.abs(b_arr))
    near = np.abs(a_arr - b_arr) <= float(near_tol) * scale
    if np.any(~near):
        aa = a_arr[~near]
        bb = b_arr[~near]
        out[~near] = 1j * (_g(bb) - _g(aa)) / (aa - bb)
    if np.any(near):
        out[near] = _h_equal(0.5 * (a_arr[near] + b_arr[near]))
    return out


def _triangle_arrays(mesh: MeshData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri = mesh.vertices[mesh.faces]
    r0 = tri[:, 0]
    u = tri[:, 1] - tri[:, 0]
    v = tri[:, 2] - tri[:, 0]
    cross = np.cross(u, v)
    double_area = np.linalg.norm(cross, axis=1)
    return r0, u, v, double_area, mesh.face_normals


def exact_triangle_amplitude(
    mesh: Any,
    q_xyz: ArrayLike,
    *,
    delta_rho: float = 1.0,
    q_switch: float = 1e-10,
    q_block_size: int = 64,
    phase_sign: int = 1,
    orient_positive: bool = True,
    moment_order: int = 2,
) -> np.ndarray:
    """Direct exact triangle boundary-integral amplitude."""

    _validate_phase_sign(phase_sign)
    q, scalar_input = _as_q_array(q_xyz)
    mesh_data = coerce_mesh(mesh, orient_positive=orient_positive)
    moments = mesh_volume_moments(mesh_data, delta_rho=delta_rho, orient_positive=False)
    r0, u, v, double_area, normals = _triangle_arrays(mesh_data)

    q_norm2 = np.einsum("ij,ij->i", q, q)
    out = np.empty(q.shape[0], dtype=np.complex128)
    low = q_norm2 < float(q_switch) ** 2
    if np.any(low):
        out[low] = moment_expansion_amplitude(
            moments,
            q[low],
            phase_sign=phase_sign,
            order=moment_order,
        )

    active_idx = np.flatnonzero(~low)
    if active_idx.size:
        block_size = max(1, int(q_block_size))
        for start in range(0, active_idx.size, block_size):
            idx = active_idx[start : start + block_size]
            qb = q[idx]
            q_eval = float(phase_sign) * qb
            a = q_eval @ u.T
            b = q_eval @ v.T
            h = triangle_h(a, b)
            phase = np.exp(1j * (q_eval @ r0.T))
            q_dot_n = qb @ normals.T
            surface_sum = np.sum(q_dot_n * double_area[None, :] * phase * h, axis=1)
            out[idx] = float(delta_rho) * surface_sum / (
                1j * float(phase_sign) * q_norm2[idx]
            )

    return out[0] if scalar_input else out


def _reorder_triangle_for_shortest_opposite_edge(tri: np.ndarray) -> tuple[np.ndarray, int]:
    e01 = np.sum((tri[0] - tri[1]) ** 2)
    e02 = np.sum((tri[0] - tri[2]) ** 2)
    e12 = np.sum((tri[1] - tri[2]) ** 2)
    opposite = int(np.argmin([e12, e02, e01]))
    if opposite == 0:
        return tri[[0, 1, 2]], opposite
    if opposite == 1:
        return tri[[1, 0, 2]], opposite
    return tri[[2, 0, 1]], opposite


def median_line_sources(
    mesh: Any,
    *,
    K: int = 4,
    delta_rho: float = 1.0,
    reorder_shortest_edge: bool = True,
    orient_positive: bool = True,
) -> SurfaceNormalSources:
    """Build median-line quadrature sources for fast vector transforms."""

    if K < 1:
        raise ValueError("K must be at least 1")
    mesh_data = coerce_mesh(mesh, orient_positive=orient_positive)
    tri = mesh_data.vertices[mesh_data.faces]
    if reorder_shortest_edge:
        reordered = np.empty_like(tri)
        opposite_vertices = np.empty(tri.shape[0], dtype=np.int64)
        for i in range(tri.shape[0]):
            reordered[i], opposite_vertices[i] = _reorder_triangle_for_shortest_opposite_edge(tri[i])
        tri = reordered
    else:
        opposite_vertices = np.zeros(tri.shape[0], dtype=np.int64)

    r0 = tri[:, 0]
    u = tri[:, 1] - tri[:, 0]
    v = tri[:, 2] - tri[:, 0]
    median = 0.5 * (u + v)
    double_area = np.linalg.norm(np.cross(u, v), axis=1)

    nodes, weights = np.polynomial.legendre.leggauss(int(K))
    tau = 0.5 * (nodes + 1.0)
    omega = 0.5 * weights

    points = []
    vector_weights = []
    for t, w in zip(tau, omega):
        points.append(r0 + float(t) * median)
        vector_weights.append(
            float(delta_rho)
            * double_area[:, None]
            * float(w)
            * float(t)
            * mesh_data.face_normals
        )

    points_arr = np.concatenate(points, axis=0)
    weights_arr = np.concatenate(vector_weights, axis=0)
    moments = mesh_volume_moments(mesh_data, delta_rho=delta_rho, orient_positive=False)
    return SurfaceNormalSources(
        points=points_arr,
        vector_weights=weights_arr,
        moments=moments,
        metadata={
            "source_type": "median_line",
            "K": int(K),
            "delta_rho": float(delta_rho),
            "num_faces": int(mesh_data.faces.shape[0]),
            "num_sources": int(points_arr.shape[0]),
            "reorder_shortest_edge": bool(reorder_shortest_edge),
            "faces_flipped": bool(mesh_data.faces_flipped),
            "signed_volume": float(mesh_data.signed_volume),
            "opposite_vertex_counts": np.bincount(opposite_vertices, minlength=3)
            .astype(int)
            .tolist(),
        },
    )


def centroid_sources(
    mesh: Any,
    *,
    delta_rho: float = 1.0,
    orient_positive: bool = True,
) -> SurfaceNormalSources:
    """Build one vector source at each triangle centroid."""

    mesh_data = coerce_mesh(mesh, orient_positive=orient_positive)
    tri = mesh_data.vertices[mesh_data.faces]
    points = tri.mean(axis=1)
    double_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    weights = float(delta_rho) * (0.5 * double_area)[:, None] * mesh_data.face_normals
    return SurfaceNormalSources(
        points=points,
        vector_weights=weights,
        moments=mesh_volume_moments(mesh_data, delta_rho=delta_rho, orient_positive=False),
        metadata={
            "source_type": "centroid",
            "delta_rho": float(delta_rho),
            "num_faces": int(mesh_data.faces.shape[0]),
            "num_sources": int(points.shape[0]),
            "faces_flipped": bool(mesh_data.faces_flipped),
            "signed_volume": float(mesh_data.signed_volume),
        },
    )


def _direct_vector_transform(
    points: np.ndarray,
    vector_weights: np.ndarray,
    q: np.ndarray,
    phase_sign: int,
) -> np.ndarray:
    phase = np.exp(1j * (float(phase_sign) * q @ points.T))
    return phase @ vector_weights


def _scaled_type3_points(points: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs_point = float(np.max(np.abs(points))) if points.size else 0.0
    source_limit = 0.95 * np.pi
    scale = max(1.0, max_abs_point / source_limit)
    return np.asarray(points / scale, dtype=np.float64), np.asarray(q * scale, dtype=np.float64)


def _finufft_vector_transform(
    points: np.ndarray,
    vector_weights: np.ndarray,
    q: np.ndarray,
    phase_sign: int,
    eps: float,
) -> np.ndarray:
    try:
        import finufft  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("backend='finufft' requires the optional finufft package") from exc

    points_scaled, q_scaled = _scaled_type3_points(points, q)
    coeff = np.ascontiguousarray(vector_weights.T, dtype=np.complex128)
    values = finufft.nufft3d3(
        np.ascontiguousarray(points_scaled[:, 0], dtype=np.float64),
        np.ascontiguousarray(points_scaled[:, 1], dtype=np.float64),
        np.ascontiguousarray(points_scaled[:, 2], dtype=np.float64),
        coeff,
        np.ascontiguousarray(float(phase_sign) * q_scaled[:, 0], dtype=np.float64),
        np.ascontiguousarray(float(phase_sign) * q_scaled[:, 1], dtype=np.float64),
        np.ascontiguousarray(float(phase_sign) * q_scaled[:, 2], dtype=np.float64),
        isign=1,
        eps=float(eps),
    )
    return np.asarray(values, dtype=np.complex128).T


def _prepare_cufinufft_runtime() -> None:
    os.environ.setdefault("CUPY_CACHE_DIR", str(Path.cwd() / ".cache" / "cupy"))
    try:
        Path(os.environ["CUPY_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    candidates: list[Path] = []
    for entry in [Path(p) for p in sys.path if p]:
        if entry.name == "site-packages":
            candidates.extend([entry / "cufinufft", entry / "cufinufft.libs"])
    try:
        venv = Path(sys.executable).resolve().parents[1]
        candidates.extend(
            [
                venv / "Lib" / "site-packages" / "cufinufft",
                venv / "Lib" / "site-packages" / "cufinufft.libs",
                venv / "Lib" / "site-packages" / "torch" / "lib",
            ]
        )
    except Exception:
        pass

    if hasattr(os, "add_dll_directory"):
        for path in candidates:
            try:
                if path.exists():
                    os.add_dll_directory(str(path))
            except Exception:
                pass


def _cufinufft_vector_transform(
    points: np.ndarray,
    vector_weights: np.ndarray,
    q: np.ndarray,
    phase_sign: int,
    eps: float,
) -> np.ndarray:
    try:
        _prepare_cufinufft_runtime()
        import cupy as cp  # type: ignore
        import cufinufft  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("backend='cufinufft' requires cupy and cufinufft") from exc

    points_scaled, q_scaled = _scaled_type3_points(points, q)
    x = cp.asarray(np.ascontiguousarray(points_scaled[:, 0], dtype=np.float64))
    y = cp.asarray(np.ascontiguousarray(points_scaled[:, 1], dtype=np.float64))
    z = cp.asarray(np.ascontiguousarray(points_scaled[:, 2], dtype=np.float64))
    s = cp.asarray(np.ascontiguousarray(float(phase_sign) * q_scaled[:, 0], dtype=np.float64))
    t = cp.asarray(np.ascontiguousarray(float(phase_sign) * q_scaled[:, 1], dtype=np.float64))
    u = cp.asarray(np.ascontiguousarray(float(phase_sign) * q_scaled[:, 2], dtype=np.float64))
    coeff = cp.asarray(np.ascontiguousarray(vector_weights.T, dtype=np.complex128))
    out = cp.empty((3, q.shape[0]), dtype=cp.complex128)
    plan = cufinufft.Plan(3, 3, n_trans=3, eps=float(eps), isign=1, dtype="complex128")
    plan.setpts(x, y, z, s, t, u)
    values = plan.execute(coeff, out=out)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(values).T


def surface_normal_amplitude_from_sources(
    points: ArrayLike,
    vector_weights: ArrayLike,
    q_xyz: ArrayLike,
    *,
    backend: SurfaceBackend = "direct",
    q_switch: float = 1e-10,
    low_q_moments: SurfaceNormalMoments | None = None,
    low_q_amplitude: complex | None = None,
    phase_sign: int = 1,
    eps: float = 1e-6,
    q_block_size: int = 8192,
    moment_order: int = 2,
) -> np.ndarray:
    """Evaluate ``A(q)`` from vector surface sources on arbitrary q targets."""

    _validate_phase_sign(phase_sign)
    points_arr = np.asarray(points, dtype=np.float64)
    weights_arr = np.asarray(vector_weights, dtype=np.complex128)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if weights_arr.shape != points_arr.shape:
        raise ValueError("vector_weights must have shape (N, 3)")

    q, scalar_input = _as_q_array(q_xyz)
    q_norm2 = np.einsum("ij,ij->i", q, q)
    out = np.empty(q.shape[0], dtype=np.complex128)
    low = q_norm2 < float(q_switch) ** 2
    if np.any(low):
        if low_q_moments is not None:
            out[low] = moment_expansion_amplitude(
                low_q_moments,
                q[low],
                phase_sign=phase_sign,
                order=moment_order,
            )
        elif low_q_amplitude is not None:
            out[low] = complex(low_q_amplitude)
        else:
            raise ValueError("low_q_moments or low_q_amplitude is required for low-q targets")

    active_idx = np.flatnonzero(~low)
    if active_idx.size:
        block_size = max(1, int(q_block_size))
        backend_norm = str(backend).lower().strip()
        for start in range(0, active_idx.size, block_size):
            idx = active_idx[start : start + block_size]
            qb = q[idx]
            if backend_norm == "direct":
                f_q = _direct_vector_transform(points_arr, weights_arr, qb, phase_sign)
            elif backend_norm == "finufft":
                f_q = _finufft_vector_transform(points_arr, weights_arr, qb, phase_sign, eps)
            elif backend_norm == "cufinufft":
                f_q = _cufinufft_vector_transform(points_arr, weights_arr, qb, phase_sign, eps)
            else:
                raise ValueError("backend must be 'direct', 'finufft', or 'cufinufft'")
            numer = np.einsum("ij,ij->i", qb, f_q)
            out[idx] = numer / (1j * float(phase_sign) * q_norm2[idx])

    return out[0] if scalar_input else out


def amplitude_from_sources(
    sources: SurfaceNormalSources,
    q_xyz: ArrayLike,
    *,
    backend: SurfaceBackend = "direct",
    phase_sign: int = 1,
    q_switch: float = 1e-10,
    eps: float = 1e-6,
    q_block_size: int = 8192,
    moment_order: int = 2,
) -> np.ndarray:
    """Convenience wrapper for ``SurfaceNormalSources`` objects."""

    low_q = None if sources.moments is not None else sources.metadata.get("signed_volume")
    if low_q is not None:
        low_q = complex(float(sources.metadata.get("delta_rho", 1.0)) * float(low_q))
    return surface_normal_amplitude_from_sources(
        sources.points,
        sources.vector_weights,
        q_xyz,
        backend=backend,
        q_switch=q_switch,
        low_q_moments=sources.moments,
        low_q_amplitude=low_q,
        phase_sign=phase_sign,
        eps=eps,
        q_block_size=q_block_size,
        moment_order=moment_order,
    )


def q_grid_from_cylindrical(q_perp: ArrayLike, q_z: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """Return flattened ``(qx,qy,qz)`` targets for a cylindrical detector grid."""

    q_perp_arr = np.asarray(q_perp, dtype=np.float64)
    q_z_arr = np.asarray(q_z, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    if q_perp_arr.shape != q_z_arr.shape:
        raise ValueError("q_perp and q_z must have the same shape")
    qx = q_perp_arr[:, None] * np.cos(phi_arr)[None, :]
    qy = q_perp_arr[:, None] * np.sin(phi_arr)[None, :]
    qz = np.broadcast_to(q_z_arr[:, None], qx.shape)
    return np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)


def _resolve_q_geometry(
    q: ArrayLike | None,
    wavelength: float | None,
    q_perp: ArrayLike | None,
    q_z: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if (q_perp is None) != (q_z is None):
        raise ValueError("q_perp and q_z must be provided together")
    if q_perp is None:
        if q is None or wavelength is None:
            raise ValueError("q and wavelength are required when q_perp/q_z are not provided")
        q_arr = np.asarray(q, dtype=np.float64)
        q_perp_arr, q_z_arr = ewald_ring(q_arr, float(wavelength))
        return q_arr, q_perp_arr, q_z_arr, float(wavelength)

    q_perp_arr = np.asarray(q_perp, dtype=np.float64)
    q_z_arr = np.asarray(q_z, dtype=np.float64)
    if q_perp_arr.shape != q_z_arr.shape:
        raise ValueError("q_perp and q_z must have matching shapes")
    q_arr = np.sqrt(q_perp_arr * q_perp_arr + q_z_arr * q_z_arr) if q is None else np.asarray(q, dtype=np.float64)
    if q_arr.shape != q_perp_arr.shape:
        raise ValueError("q must have the same shape as q_perp/q_z")
    return q_arr, q_perp_arr, q_z_arr, 1.0 if wavelength is None else float(wavelength)


class SurfaceNormalCylindricalPlan:
    """Prepared vector-source surface-normal solver on a cylindrical q grid."""

    def __init__(
        self,
        points: ArrayLike,
        vector_weights: ArrayLike,
        *,
        q: ArrayLike | None = None,
        wavelength: float | None = None,
        q_perp: ArrayLike | None = None,
        q_z: ArrayLike | None = None,
        n_phi: int = 180,
        phi: ArrayLike | None = None,
        q_switch: float = 1e-10,
        low_q_moments: SurfaceNormalMoments | None = None,
        phase_sign: int = 1,
        n_r: int = 32,
        n_z: int = 32,
        r_max: float | None = None,
        z_range: tuple[float, float] | None = None,
        hist_backend: str = "numpy",
        hist_dtype: np.dtype | str = np.complex128,
        angle_lut_size: int = 0,
        angle_lut_mode: str = "nearest",
        circular_backend: str = "auto",
        complex_dtype: np.dtype | str | None = None,
        q_block_size: int = 128,
    ) -> None:
        _validate_phase_sign(phase_sign)
        if phase_sign != 1:
            raise ValueError("SurfaceNormalCylindricalPlan currently supports phase_sign=+1")
        self.points = np.asarray(points, dtype=np.float64)
        self.vector_weights = np.asarray(vector_weights, dtype=np.complex128)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if self.vector_weights.shape != self.points.shape:
            raise ValueError("vector_weights must have shape (N, 3)")

        self.q, self.q_perp, self.q_z, self.wavelength = _resolve_q_geometry(
            q,
            wavelength,
            q_perp,
            q_z,
        )
        self.q_norm2 = self.q_perp * self.q_perp + self.q_z * self.q_z
        self.q_switch = float(q_switch)
        self.low_q_moments = low_q_moments
        self.phase_sign = int(phase_sign)
        self.circular_backend = circular_backend

        n_phi = int(n_phi if phi is None else np.asarray(phi).size)
        if n_phi <= 0:
            raise ValueError("n_phi must be positive")

        self.binned_components: tuple[BinnedStructure, BinnedStructure, BinnedStructure] = tuple(
            make_cylindrical_histogram(
                self.points,
                atom_weights=self.vector_weights[:, component],
                n_r=n_r,
                n_z=n_z,
                n_phi=n_phi,
                r_max=r_max,
                z_range=z_range,
                backend=hist_backend,
                hist_dtype=hist_dtype,
                angle_lut_size=angle_lut_size,
                angle_lut_mode=angle_lut_mode,
            )
            for component in range(3)
        )  # type: ignore[assignment]

        self.phi = self.binned_components[0].beta_centers if phi is None else np.asarray(phi, dtype=np.float64)
        if self.phi.shape != self.binned_components[0].beta_centers.shape or not np.allclose(
            self.phi,
            self.binned_components[0].beta_centers,
        ):
            raise ValueError("phi must match the regular histogram beta-center grid")

        self.component_plans: tuple[PreparedCakePlan, PreparedCakePlan, PreparedCakePlan] = tuple(
            PreparedCakePlan(
                binned,
                self.q,
                self.wavelength,
                phi=self.phi,
                circular_backend=circular_backend,
                complex_dtype=complex_dtype,
                q_block_size=q_block_size,
                q_perp=self.q_perp,
                q_z=self.q_z,
            )
            for binned in self.binned_components
        )  # type: ignore[assignment]

    @classmethod
    def from_sources(
        cls,
        sources: SurfaceNormalSources,
        **kwargs: Any,
    ) -> SurfaceNormalCylindricalPlan:
        kwargs.setdefault("low_q_moments", sources.moments)
        return cls(sources.points, sources.vector_weights, **kwargs)

    @classmethod
    def from_mesh_median_line(
        cls,
        mesh: Any,
        *,
        K: int = 4,
        delta_rho: float = 1.0,
        **kwargs: Any,
    ) -> SurfaceNormalCylindricalPlan:
        sources = median_line_sources(mesh, K=K, delta_rho=delta_rho)
        return cls.from_sources(sources, **kwargs)

    @classmethod
    def from_mesh_centroid(
        cls,
        mesh: Any,
        *,
        delta_rho: float = 1.0,
        **kwargs: Any,
    ) -> SurfaceNormalCylindricalPlan:
        sources = centroid_sources(mesh, delta_rho=delta_rho)
        return cls.from_sources(sources, **kwargs)

    @classmethod
    def ewald_ring(
        cls,
        sources: SurfaceNormalSources,
        q: ArrayLike,
        wavelength: float,
        **kwargs: Any,
    ) -> SurfaceNormalCylindricalPlan:
        return cls.from_sources(sources, q=q, wavelength=wavelength, **kwargs)

    @classmethod
    def flat_detector(
        cls,
        sources: SurfaceNormalSources,
        q: ArrayLike,
        **kwargs: Any,
    ) -> SurfaceNormalCylindricalPlan:
        q_arr = np.asarray(q, dtype=np.float64)
        return cls.from_sources(sources, q=q_arr, q_perp=q_arr, q_z=np.zeros_like(q_arr), **kwargs)

    def field_components(
        self,
        q_indices: np.ndarray | None = None,
        *,
        method: Literal["circular", "r-dependent"] = "circular",
        margin: int = 16,
        cutoff_bin_size: int = 16,
        analytic_kernel: bool = False,
        fused_analytic_kernel: bool = False,
        q_block_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if method == "circular":
            return tuple(
                plan.circular_fft(q_indices=q_indices, q_block_size=q_block_size)
                for plan in self.component_plans
            )  # type: ignore[return-value]
        if method == "r-dependent":
            return tuple(
                plan.circular_fft_r_dependent_bandlimit(
                    q_indices=q_indices,
                    margin=margin,
                    cutoff_bin_size=cutoff_bin_size,
                    analytic_kernel=analytic_kernel,
                    fused_analytic_kernel=fused_analytic_kernel,
                    q_block_size=q_block_size,
                )
                for plan in self.component_plans
            )  # type: ignore[return-value]
        raise ValueError("method must be 'circular' or 'r-dependent'")

    def _combine_components(
        self,
        components: tuple[np.ndarray, np.ndarray, np.ndarray],
        indices: np.ndarray,
        *,
        moment_order: int = 2,
    ) -> np.ndarray:
        fx, fy, fz = components
        q_perp = self.q_perp[indices]
        q_z = self.q_z[indices]
        q_norm2 = self.q_norm2[indices]
        numer = (
            q_perp[:, None] * np.cos(self.phi)[None, :] * fx
            + q_perp[:, None] * np.sin(self.phi)[None, :] * fy
            + q_z[:, None] * fz
        )
        out = np.empty_like(numer, dtype=np.complex128)
        low = q_norm2 < self.q_switch * self.q_switch
        if np.any(~low):
            out[~low] = numer[~low] / (1j * float(self.phase_sign) * q_norm2[~low, None])
        if np.any(low):
            if self.low_q_moments is None:
                raise ValueError("low_q_moments is required for low-q cylindrical targets")
            q_xyz = q_grid_from_cylindrical(q_perp[low], q_z[low], self.phi)
            low_amp = moment_expansion_amplitude(
                self.low_q_moments,
                q_xyz,
                phase_sign=self.phase_sign,
                order=moment_order,
            ).reshape(int(np.count_nonzero(low)), self.phi.size)
            out[low] = low_amp
        return out

    def amplitude(
        self,
        q_indices: np.ndarray | None = None,
        *,
        method: Literal["circular", "r-dependent"] = "circular",
        margin: int = 16,
        cutoff_bin_size: int = 16,
        analytic_kernel: bool = False,
        fused_analytic_kernel: bool = False,
        q_block_size: int | None = None,
        moment_order: int = 2,
    ) -> np.ndarray:
        indices = np.arange(self.q.size) if q_indices is None else np.asarray(q_indices, dtype=np.intp)
        components = self.field_components(
            q_indices=indices,
            method=method,
            margin=margin,
            cutoff_bin_size=cutoff_bin_size,
            analytic_kernel=analytic_kernel,
            fused_analytic_kernel=fused_analytic_kernel,
            q_block_size=q_block_size,
        )
        return self._combine_components(components, indices, moment_order=moment_order)

    def circular_fft(self, q_indices: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        return self.amplitude(q_indices=q_indices, method="circular", **kwargs)

    def circular_fft_r_dependent_bandlimit(
        self,
        q_indices: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.amplitude(q_indices=q_indices, method="r-dependent", **kwargs)
