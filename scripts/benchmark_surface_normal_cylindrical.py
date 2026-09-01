from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import (  # noqa: E402
    SurfaceNormalCylindricalPlan,
    amplitude_from_sources,
    centroid_sources,
    median_line_sources,
    mesh_face_normals,
    mesh_signed_volume,
    q_grid_from_cylindrical,
)
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import relative_l2  # noqa: E402
from waxs_cake.surface_normal import _prepare_cufinufft_runtime  # noqa: E402


def orient_faces_away_from(vertices: np.ndarray, faces: np.ndarray, center: np.ndarray) -> np.ndarray:
    faces = faces.copy()
    normals = mesh_face_normals(vertices, faces)
    centroids = vertices[faces].mean(axis=1)
    inward = np.einsum("ij,ij->i", normals, centroids - center[None, :]) < 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    if mesh_signed_volume(vertices, faces) < 0.0:
        faces = faces[:, [0, 2, 1]].copy()
    return faces


def sphere_mesh(radius: float, n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = [[0.0, 0.0, radius]]
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
    faces_arr = orient_faces_away_from(vertices_arr, np.asarray(faces, dtype=np.int64), np.zeros(3))
    return vertices_arr, faces_arr


def median_time(fn: Callable[[], np.ndarray], repeats: int) -> tuple[np.ndarray, float, list[float]]:
    values = None
    times = []
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        values = fn()
        times.append(time.perf_counter() - start)
    assert values is not None
    return values, float(np.median(times)), times


def cufinufft_hot_benchmark(
    points: np.ndarray,
    vector_weights: np.ndarray,
    q_xyz: np.ndarray,
    direct_flat: np.ndarray,
    *,
    eps: float,
    repeats: int,
) -> dict[str, object]:
    _prepare_cufinufft_runtime()
    import cupy as cp  # type: ignore
    import cufinufft  # type: ignore

    q_norm2 = np.einsum("ij,ij->i", q_xyz, q_xyz)
    active = q_norm2 > 1e-20
    q_active = q_xyz[active]
    q_norm2_active = q_norm2[active]

    max_abs_point = float(np.max(np.abs(points))) if points.size else 0.0
    scale = max(1.0, max_abs_point / (0.95 * np.pi))
    points_scaled = np.asarray(points / scale, dtype=np.float64)
    q_scaled = np.asarray(q_active * scale, dtype=np.float64)

    setup_start = time.perf_counter()
    x = cp.asarray(np.ascontiguousarray(points_scaled[:, 0], dtype=np.float64))
    y = cp.asarray(np.ascontiguousarray(points_scaled[:, 1], dtype=np.float64))
    z = cp.asarray(np.ascontiguousarray(points_scaled[:, 2], dtype=np.float64))
    s = cp.asarray(np.ascontiguousarray(q_scaled[:, 0], dtype=np.float64))
    t = cp.asarray(np.ascontiguousarray(q_scaled[:, 1], dtype=np.float64))
    u = cp.asarray(np.ascontiguousarray(q_scaled[:, 2], dtype=np.float64))
    coeff = cp.asarray(np.ascontiguousarray(vector_weights.T, dtype=np.complex128))
    out = cp.empty((3, q_active.shape[0]), dtype=cp.complex128)
    plan = cufinufft.Plan(3, 3, n_trans=3, eps=float(eps), isign=1, dtype="complex128")
    plan.setpts(x, y, z, s, t, u)
    cp.cuda.Stream.null.synchronize()
    setup_s = time.perf_counter() - setup_start

    execute_times = []
    transfer_combine_times = []
    last_amp = None
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        values_gpu = plan.execute(coeff, out=out)
        cp.cuda.Stream.null.synchronize()
        execute_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        values = cp.asnumpy(values_gpu).T
        numer = np.einsum("ij,ij->i", q_active, values)
        last_amp = numer / (1j * q_norm2_active)
        transfer_combine_times.append(time.perf_counter() - start)

    assert last_amp is not None
    return {
        "cufinufft_hot_plan_setup_s": setup_s,
        "cufinufft_hot_execute_s": float(np.median(execute_times)),
        "cufinufft_hot_execute_times": execute_times,
        "cufinufft_hot_transfer_combine_s": float(np.median(transfer_combine_times)),
        "cufinufft_hot_transfer_combine_times": transfer_combine_times,
        "cufinufft_hot_total_s": float(np.median(np.asarray(execute_times) + np.asarray(transfer_combine_times))),
        "cufinufft_hot_rel_l2_vs_direct_active": relative_l2(last_amp, direct_flat[active]),
        "cufinufft_hot_active_targets": int(q_active.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark surface-normal cylindrical and NUFFT routes.")
    parser.add_argument("--n-lat", type=int, default=28)
    parser.add_argument("--n-lon", type=int, default=56)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--source-mode", choices=["median-line", "centroid"], default="median-line")
    parser.add_argument("--median-k", type=int, default=4)
    parser.add_argument("--qmax", type=float, default=4.0)
    parser.add_argument("--nq", type=int, default=24)
    parser.add_argument("--nphi", type=int, default=256)
    parser.add_argument("--nr", type=int, default=72)
    parser.add_argument("--nz", type=int, default=72)
    parser.add_argument("--wavelength", type=float, default=0.1)
    parser.add_argument("--flat", action="store_true", help="Use qz=0 instead of Ewald-ring q geometry.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-finufft", action="store_true")
    parser.add_argument("--include-cufinufft", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mesh = sphere_mesh(args.radius, args.n_lat, args.n_lon)
    if args.source_mode == "median-line":
        sources = median_line_sources(mesh, K=args.median_k)
    else:
        sources = centroid_sources(mesh)
    q = np.linspace(0.0 if args.flat else 0.1, args.qmax, args.nq)

    if args.flat:
        plan = SurfaceNormalCylindricalPlan.flat_detector(
            sources,
            q,
            n_phi=args.nphi,
            n_r=args.nr,
            n_z=args.nz,
            circular_backend="auto",
        )
        q_perp = q
        q_z = np.zeros_like(q)
    else:
        plan = SurfaceNormalCylindricalPlan.ewald_ring(
            sources,
            q,
            args.wavelength,
            n_phi=args.nphi,
            n_r=args.nr,
            n_z=args.nz,
            circular_backend="auto",
        )
        q_perp, q_z = ewald_ring(q, args.wavelength)

    q_xyz = q_grid_from_cylindrical(q_perp, q_z, plan.phi)
    q_shape = (q.size, plan.phi.size)

    direct, direct_s, direct_times = median_time(
        lambda: amplitude_from_sources(sources, q_xyz, backend="direct").reshape(q_shape),
        args.repeats,
    )
    circular, circular_s, circular_times = median_time(plan.circular_fft, args.repeats)
    rdep, rdep_s, rdep_times = median_time(
        lambda: plan.circular_fft_r_dependent_bandlimit(
            margin=16,
            cutoff_bin_size=8,
            analytic_kernel=True,
        ),
        args.repeats,
    )

    result: dict[str, object] = {
        "mesh": {
            "vertices": int(mesh[0].shape[0]),
            "faces": int(mesh[1].shape[0]),
            "surface_sources": int(sources.points.shape[0]),
            "source_mode": args.source_mode,
        },
        "grid": {
            "geometry": "flat_qz0" if args.flat else "ewald_ring",
            "nq": int(q.size),
            "nphi": int(plan.phi.size),
            "targets": int(q.size * plan.phi.size),
        },
        "direct_s": direct_s,
        "direct_times": direct_times,
        "circular_s": circular_s,
        "circular_times": circular_times,
        "circular_rel_l2_vs_direct": relative_l2(circular, direct),
        "r_dependent_s": rdep_s,
        "r_dependent_times": rdep_times,
        "r_dependent_rel_l2_vs_circular": relative_l2(rdep, circular),
    }

    if args.include_finufft:
        finufft, finufft_s, finufft_times = median_time(
            lambda: amplitude_from_sources(sources, q_xyz, backend="finufft", eps=args.eps).reshape(q_shape),
            args.repeats,
        )
        result.update(
            {
                "finufft_s": finufft_s,
                "finufft_times": finufft_times,
                "finufft_rel_l2_vs_direct": relative_l2(finufft, direct),
            }
        )

    if args.include_cufinufft:
        cufinufft, cufinufft_s, cufinufft_times = median_time(
            lambda: amplitude_from_sources(sources, q_xyz, backend="cufinufft", eps=args.eps).reshape(q_shape),
            args.repeats,
        )
        result.update(
            {
                "cufinufft_s": cufinufft_s,
                "cufinufft_times": cufinufft_times,
                "cufinufft_rel_l2_vs_direct": relative_l2(cufinufft, direct),
            }
        )
        result.update(
            cufinufft_hot_benchmark(
                sources.points,
                sources.vector_weights,
                q_xyz,
                direct.ravel(),
                eps=args.eps,
                repeats=args.repeats,
            )
        )

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
