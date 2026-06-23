from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from benchmark_odt_ewald_cap_operator import (
    ROOT,
    QSamples,
    StructuredOdtPlan,
    build_structured_kernel,
    complex_dot,
    illumination_directions,
    make_cylindrical_object,
    random_residual,
    recommended_h_cutoff,
    relative_complex_error,
    relative_l2,
    resolve_structured_backend,
    structured_adjoint,
    structured_forward,
)


@dataclass(frozen=True)
class EllipseResamplingMap:
    local_indices: np.ndarray
    global_indices: np.ndarray
    weights: np.ndarray
    illumination_index: np.ndarray
    source_count_per_block: int
    source_size: int
    target_count: int
    source_beam_na: float
    max_target_beam_na: float
    radial_edge_clamped_fraction: float
    source_inside_plane_fraction: float


def parse_int_list(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item <= 0 for item in out):
        raise ValueError("expected positive comma-separated integers")
    return out


def parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item < 0.0 for item in out):
        raise ValueError("expected non-negative comma-separated floats")
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def median_time(func, *, repeats: int) -> tuple[Any, float, list[float]]:
    result = None
    times: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    if result is None:
        raise RuntimeError("timed function did not run")
    return result, float(median(times)), times


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cpp_odt_module(*, required: bool):
    try:
        from waxs_cake import _cpp_odt
    except ImportError:
        if required:
            raise
        return None
    return _cpp_odt


def rotation_from_z(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    norm = float(np.linalg.norm(target))
    if norm <= 0.0:
        raise ValueError("target direction must be non-zero")
    target = target / norm
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    cross = np.cross(z_axis, target)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.dot(z_axis, target))
    if sin_angle < 1e-14:
        if cos_angle > 0.0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])
    axis = cross / sin_angle
    kx = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + sin_angle * kx + (1.0 - cos_angle) * (kx @ kx)


def q_samples_from_vectors(q_vectors: np.ndarray, illumination_index: np.ndarray) -> QSamples:
    q_vectors = np.asarray(q_vectors, dtype=float)
    if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
        raise ValueError("q_vectors must have shape (n, 3)")
    illumination_index = np.asarray(illumination_index, dtype=np.int64)
    if illumination_index.shape != (q_vectors.shape[0],):
        raise ValueError("illumination_index shape does not match q_vectors")
    q_perp = np.hypot(q_vectors[:, 0], q_vectors[:, 1])
    phi = np.mod(np.arctan2(q_vectors[:, 1], q_vectors[:, 0]), 2.0 * np.pi)
    return QSamples(
        qx=np.ascontiguousarray(q_vectors[:, 0]),
        qy=np.ascontiguousarray(q_vectors[:, 1]),
        qz=np.ascontiguousarray(q_vectors[:, 2]),
        q_perp=np.ascontiguousarray(q_perp),
        phi=np.ascontiguousarray(phi),
        illumination_index=np.ascontiguousarray(illumination_index),
    )


def half_slope_from_na(na: float) -> float:
    if na <= 0.0 or na >= 1.0:
        raise ValueError("detector-na must be in (0, 1)")
    return float(na / math.sqrt(max(1.0 - na * na, 1e-300)))


def plane_detector_directions(*, half_slope: float, pixels: int) -> tuple[np.ndarray, np.ndarray]:
    if half_slope <= 0.0:
        raise ValueError("half_slope must be positive")
    if pixels <= 0:
        raise ValueError("pixels must be positive")
    axis = np.linspace(
        -half_slope + half_slope / float(pixels),
        half_slope - half_slope / float(pixels),
        pixels,
        dtype=float,
    )
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    vectors = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size, dtype=float)])
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    return vectors, axis


def plane_detector_q_samples(
    *,
    k: float,
    half_slope: float,
    pixels: int,
    illumination: np.ndarray,
) -> tuple[QSamples, np.ndarray, np.ndarray]:
    detector, slope_axis = plane_detector_directions(half_slope=half_slope, pixels=pixels)
    q_blocks: list[np.ndarray] = []
    illum_blocks: list[np.ndarray] = []
    for index, s_in in enumerate(illumination):
        q_blocks.append(float(k) * (detector - s_in[None, :]))
        illum_blocks.append(np.full(detector.shape[0], index, dtype=np.int64))
    return (
        q_samples_from_vectors(np.vstack(q_blocks), np.concatenate(illum_blocks)),
        detector,
        slope_axis,
    )


def required_source_beam_na(
    *,
    detector_lab: np.ndarray,
    illumination: np.ndarray,
    radial_samples: int,
    margin_fraction: float,
) -> tuple[float, float]:
    max_r = 0.0
    for s_in in illumination:
        rotation = rotation_from_z(s_in)
        detector_beam = detector_lab @ rotation
        max_r = max(max_r, float(np.max(np.hypot(detector_beam[:, 0], detector_beam[:, 1]))))
    step = max_r / float(radial_samples)
    source_na = min(0.999, max_r + 0.5 * step + margin_fraction * max_r)
    return source_na, max_r


def ellipse_contour_q_samples(
    *,
    k: float,
    source_beam_na: float,
    radial_samples: int,
    phi_samples: int,
    illumination: np.ndarray,
) -> tuple[QSamples, np.ndarray, np.ndarray]:
    radial = (
        np.arange(radial_samples, dtype=float) + 0.5
    ) * float(source_beam_na) / float(radial_samples)
    phi = np.linspace(0.0, 2.0 * np.pi, phi_samples, endpoint=False, dtype=float)
    rr, pp = np.meshgrid(radial, phi, indexing="ij")
    sx = rr.ravel() * np.cos(pp.ravel())
    sy = rr.ravel() * np.sin(pp.ravel())
    sz = np.sqrt(np.maximum(1.0 - sx * sx - sy * sy, 0.0))
    detector_beam = np.column_stack([sx, sy, sz])
    q_blocks: list[np.ndarray] = []
    illum_blocks: list[np.ndarray] = []
    for index, s_in in enumerate(illumination):
        rotation = rotation_from_z(s_in)
        detector_lab = detector_beam @ rotation.T
        q_blocks.append(float(k) * (detector_lab - s_in[None, :]))
        illum_blocks.append(np.full(detector_beam.shape[0], index, dtype=np.int64))
    return (
        q_samples_from_vectors(np.vstack(q_blocks), np.concatenate(illum_blocks)),
        radial,
        phi,
    )


def ellipse_source_inside_plane_fraction(
    *,
    source_beam_na: float,
    radial_samples: int,
    phi_samples: int,
    illumination: np.ndarray,
    half_slope: float,
) -> float:
    radial = (
        np.arange(radial_samples, dtype=float) + 0.5
    ) * float(source_beam_na) / float(radial_samples)
    phi = np.linspace(0.0, 2.0 * np.pi, phi_samples, endpoint=False, dtype=float)
    rr, pp = np.meshgrid(radial, phi, indexing="ij")
    sx = rr.ravel() * np.cos(pp.ravel())
    sy = rr.ravel() * np.sin(pp.ravel())
    sz = np.sqrt(np.maximum(1.0 - sx * sx - sy * sy, 0.0))
    detector_beam = np.column_stack([sx, sy, sz])
    inside = 0
    total = detector_beam.shape[0] * illumination.shape[0]
    for s_in in illumination:
        rotation = rotation_from_z(s_in)
        detector_lab = detector_beam @ rotation.T
        z_positive = detector_lab[:, 2] > 1e-12
        slope_x = detector_lab[:, 0] / np.maximum(detector_lab[:, 2], 1e-300)
        slope_y = detector_lab[:, 1] / np.maximum(detector_lab[:, 2], 1e-300)
        inside += int(
            np.count_nonzero(
                z_positive
                & (np.abs(slope_x) <= half_slope)
                & (np.abs(slope_y) <= half_slope)
            )
        )
    return float(inside / max(total, 1))


def build_ellipse_resampling_map(
    *,
    detector_lab: np.ndarray,
    illumination: np.ndarray,
    source_beam_na: float,
    radial_samples: int,
    phi_samples: int,
    half_slope: float,
) -> EllipseResamplingMap:
    target_per_illum = detector_lab.shape[0]
    total_targets = target_per_illum * illumination.shape[0]
    source_count = radial_samples * phi_samples
    local_indices = np.empty((total_targets, 4), dtype=np.int64)
    weights = np.empty(local_indices.shape, dtype=float)
    illum_index = np.empty(total_targets, dtype=np.int64)
    radial_step = float(source_beam_na) / float(radial_samples)
    phi_step = 2.0 * np.pi / float(phi_samples)
    edge_clamped = 0
    max_target_beam_na = 0.0
    row = 0
    for illum, s_in in enumerate(illumination):
        rotation = rotation_from_z(s_in)
        detector_beam = detector_lab @ rotation
        target_r = np.hypot(detector_beam[:, 0], detector_beam[:, 1])
        target_phi = np.mod(np.arctan2(detector_beam[:, 1], detector_beam[:, 0]), 2.0 * np.pi)
        max_target_beam_na = max(max_target_beam_na, float(np.max(target_r)))
        for r_value, phi_value in zip(target_r, target_phi, strict=True):
            raw_r = float(r_value) / radial_step - 0.5
            radial0_raw = math.floor(raw_r)
            radial_weight = raw_r - float(radial0_raw)
            if radial0_raw < 0:
                radial0 = radial1 = 0
                radial_weight = 0.0
                edge_clamped += 1
            elif radial0_raw >= radial_samples - 1:
                radial0 = radial1 = radial_samples - 1
                radial_weight = 0.0
                edge_clamped += 1
            else:
                radial0 = radial0_raw
                radial1 = radial0 + 1
            raw_phi = float(phi_value) / phi_step
            phi0_base = math.floor(raw_phi)
            phi_weight = raw_phi - float(phi0_base)
            phi0 = phi0_base % phi_samples
            phi1 = (phi0 + 1) % phi_samples
            local_indices[row] = [
                radial0 * phi_samples + phi0,
                radial1 * phi_samples + phi0,
                radial0 * phi_samples + phi1,
                radial1 * phi_samples + phi1,
            ]
            weights[row] = [
                (1.0 - radial_weight) * (1.0 - phi_weight),
                radial_weight * (1.0 - phi_weight),
                (1.0 - radial_weight) * phi_weight,
                radial_weight * phi_weight,
            ]
            illum_index[row] = illum
            row += 1
    global_indices = local_indices + illum_index[:, None] * source_count
    return EllipseResamplingMap(
        local_indices=np.ascontiguousarray(local_indices),
        global_indices=np.ascontiguousarray(global_indices),
        weights=np.ascontiguousarray(weights),
        illumination_index=np.ascontiguousarray(illum_index),
        source_count_per_block=int(source_count),
        source_size=int(source_count * illumination.shape[0]),
        target_count=int(total_targets),
        source_beam_na=float(source_beam_na),
        max_target_beam_na=float(max_target_beam_na),
        radial_edge_clamped_fraction=float(edge_clamped / max(total_targets, 1)),
        source_inside_plane_fraction=ellipse_source_inside_plane_fraction(
            source_beam_na=source_beam_na,
            radial_samples=radial_samples,
            phi_samples=phi_samples,
            illumination=illumination,
            half_slope=half_slope,
        ),
    )


def resolve_resampling_backend(requested: str) -> str:
    if requested not in {"auto", "numpy", "cpp"}:
        raise ValueError("resampling backend must be auto, numpy, or cpp")
    if requested == "numpy":
        return "numpy"
    module = _cpp_odt_module(required=requested == "cpp")
    if module is None or not hasattr(module, "resample4_interpolate"):
        if requested == "cpp":
            raise RuntimeError("waxs_cake._cpp_odt lacks resample4_interpolate")
        return "numpy"
    return "cpp"


def interpolate_from_ellipse(
    source: np.ndarray,
    mapping: EllipseResamplingMap,
    *,
    backend: str,
    cpp_threads: int,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.complex128)
    if backend == "cpp":
        cpp_odt = _cpp_odt_module(required=True)
        return cpp_odt.resample4_interpolate(
            np.ascontiguousarray(source),
            mapping.global_indices,
            mapping.weights,
            int(cpp_threads),
        )
    return np.sum(source[mapping.global_indices] * mapping.weights, axis=1)


def scatter_to_ellipse(
    residual: np.ndarray,
    mapping: EllipseResamplingMap,
    *,
    backend: str,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.shape != (mapping.target_count,):
        raise ValueError("residual shape does not match target count")
    if backend == "cpp":
        cpp_odt = _cpp_odt_module(required=True)
        return cpp_odt.resample4_scatter_adjoint(
            np.ascontiguousarray(residual),
            mapping.global_indices,
            mapping.weights,
            int(mapping.source_size),
        )
    source = np.zeros(mapping.source_size, dtype=np.complex128)
    np.add.at(
        source,
        mapping.global_indices.ravel(),
        (mapping.weights * residual[:, None]).ravel(),
    )
    return source


def subset_q_samples(q: QSamples, indices: np.ndarray) -> QSamples:
    indices = np.asarray(indices, dtype=np.int64)
    return QSamples(
        qx=np.ascontiguousarray(q.qx[indices]),
        qy=np.ascontiguousarray(q.qy[indices]),
        qz=np.ascontiguousarray(q.qz[indices]),
        q_perp=np.ascontiguousarray(q.q_perp[indices]),
        phi=np.ascontiguousarray(q.phi[indices]),
        illumination_index=np.ascontiguousarray(q.illumination_index[indices]),
    )


def compact_ellipse_sources(
    q: QSamples,
    mapping: EllipseResamplingMap,
) -> tuple[QSamples, EllipseResamplingMap, int]:
    active_slot = mapping.weights != 0.0
    active = np.unique(mapping.global_indices[active_slot])
    if active.size == 0:
        raise ValueError("ellipse resampling map has no nonzero interpolation weights")
    compact_indices = np.zeros_like(mapping.global_indices)
    compact_indices[active_slot] = np.searchsorted(
        active,
        mapping.global_indices[active_slot],
    )
    compact_mapping = EllipseResamplingMap(
        local_indices=mapping.local_indices,
        global_indices=np.ascontiguousarray(compact_indices.astype(np.int64, copy=False)),
        weights=mapping.weights,
        illumination_index=mapping.illumination_index,
        source_count_per_block=mapping.source_count_per_block,
        source_size=int(active.size),
        target_count=mapping.target_count,
        source_beam_na=mapping.source_beam_na,
        max_target_beam_na=mapping.max_target_beam_na,
        radial_edge_clamped_fraction=mapping.radial_edge_clamped_fraction,
        source_inside_plane_fraction=mapping.source_inside_plane_fraction,
    )
    return subset_q_samples(q, active), compact_mapping, int(q.count)


def kernel_mib(kernel: Any) -> float:
    return (kernel.radial.nbytes + kernel.axial.nbytes + kernel.angular.nbytes) / (
        1024.0 * 1024.0
    )


def benchmark_case(
    args: argparse.Namespace,
    *,
    illumination_na: float,
    oversample: int,
) -> dict[str, Any]:
    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    illumination = illumination_directions(
        mode="shifted",
        n_illum=args.n_illum,
        illumination_na=illumination_na,
    )
    half_slope = args.plane_half_slope
    if half_slope is None:
        half_slope = half_slope_from_na(args.detector_na)
    q_plane, detector_lab, _ = plane_detector_q_samples(
        k=args.k,
        half_slope=half_slope,
        pixels=args.plane_pixels,
        illumination=illumination,
    )
    radial_samples = args.contour_radial * oversample
    phi_samples = args.contour_phi * oversample
    source_beam_na, max_target_beam_na = required_source_beam_na(
        detector_lab=detector_lab,
        illumination=illumination,
        radial_samples=radial_samples,
        margin_fraction=args.source_na_margin_fraction,
    )
    q_ellipse_full, _, _ = ellipse_contour_q_samples(
        k=args.k,
        source_beam_na=source_beam_na,
        radial_samples=radial_samples,
        phi_samples=phi_samples,
        illumination=illumination,
    )
    mapping = build_ellipse_resampling_map(
        detector_lab=detector_lab,
        illumination=illumination,
        source_beam_na=source_beam_na,
        radial_samples=radial_samples,
        phi_samples=phi_samples,
        half_slope=half_slope,
    )
    q_ellipse = q_ellipse_full
    full_ellipse_q_samples = q_ellipse_full.count
    if args.compact_ellipse_source:
        q_ellipse, mapping, full_ellipse_q_samples = compact_ellipse_sources(
            q_ellipse_full,
            mapping,
        )
    residual = random_residual(q_plane, seed=args.seed + 7919)
    backend = resolve_structured_backend(args.structured_backend)
    resampling_backend = resolve_resampling_backend(args.resampling_backend)

    plane_h_cutoff = (
        recommended_h_cutoff(q_plane, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    ellipse_h_cutoff = (
        recommended_h_cutoff(q_ellipse, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    plane_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=plane_h_cutoff,
    )
    ellipse_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=ellipse_h_cutoff,
    )
    plane_kernel, plane_build_s, plane_build_times = median_time(
        lambda: build_structured_kernel(plane_plan, q_plane),
        repeats=args.build_repeats,
    )
    ellipse_kernel, ellipse_build_s, ellipse_build_times = median_time(
        lambda: build_structured_kernel(ellipse_plan, q_ellipse),
        repeats=args.build_repeats,
    )

    def plane_pair():
        forward = structured_forward(
            plane_plan,
            obj.coeff,
            q_plane,
            kernel=plane_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        adjoint = structured_adjoint(
            plane_plan,
            q_plane,
            residual,
            kernel=plane_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        return forward, adjoint

    def ellipse_pair():
        source_forward = structured_forward(
            ellipse_plan,
            obj.coeff,
            q_ellipse,
            kernel=ellipse_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        forward = interpolate_from_ellipse(
            source_forward,
            mapping,
            backend=resampling_backend,
            cpp_threads=args.cpp_threads,
        )
        source_residual = scatter_to_ellipse(
            residual,
            mapping,
            backend=resampling_backend,
        )
        adjoint = structured_adjoint(
            ellipse_plan,
            q_ellipse,
            source_residual,
            kernel=ellipse_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        return forward, adjoint

    (plane_forward, plane_adjoint), plane_hot_s, plane_hot_times = median_time(
        plane_pair,
        repeats=args.hot_repeats,
    )
    (ellipse_forward, ellipse_adjoint), ellipse_hot_s, ellipse_hot_times = median_time(
        ellipse_pair,
        repeats=args.hot_repeats,
    )
    ellipse_dot_error = relative_complex_error(
        complex_dot(ellipse_forward, residual),
        complex_dot(obj.coeff, ellipse_adjoint),
    )
    detector_na_max = float(
        np.max(np.hypot(detector_lab[:, 0], detector_lab[:, 1]))
    )
    row: dict[str, Any] = {
        "status": "ok",
        "n_beta": args.n_beta,
        "n_r": args.n_r,
        "n_z": args.n_z,
        "r_max": args.r_max,
        "z_max": args.z_max,
        "phantom": args.phantom,
        "k": args.k,
        "detector_na": args.detector_na,
        "detector_na_max_pixel": detector_na_max,
        "plane_half_slope": half_slope,
        "illumination_na": illumination_na,
        "n_illum": args.n_illum,
        "plane_pixels": args.plane_pixels,
        "contour_radial": args.contour_radial,
        "contour_phi": args.contour_phi,
        "oversample": oversample,
        "radial_samples": radial_samples,
        "phi_samples": phi_samples,
        "source_beam_na": source_beam_na,
        "max_target_beam_na": max_target_beam_na,
        "source_inside_plane_fraction": mapping.source_inside_plane_fraction,
        "radial_edge_clamped_fraction": mapping.radial_edge_clamped_fraction,
        "plane_q_samples": q_plane.count,
        "ellipse_q_samples": q_ellipse.count,
        "full_ellipse_q_samples": full_ellipse_q_samples,
        "ellipse_source_compact": bool(args.compact_ellipse_source),
        "ellipse_source_active_fraction": q_ellipse.count / max(full_ellipse_q_samples, 1),
        "plane_h_cutoff": plane_h_cutoff,
        "ellipse_h_cutoff": ellipse_h_cutoff,
        "plane_used_modes": plane_plan.used_modes,
        "ellipse_used_modes": ellipse_plan.used_modes,
        "structured_backend": backend,
        "resampling_backend": resampling_backend,
        "cpp_threads": args.cpp_threads,
        "plane_build_s": plane_build_s,
        "ellipse_build_s": ellipse_build_s,
        "plane_kernel_mib": kernel_mib(plane_kernel),
        "ellipse_kernel_mib": kernel_mib(ellipse_kernel),
        "plane_hot_pair_s": plane_hot_s,
        "ellipse_hot_pair_s": ellipse_hot_s,
        "plane_vs_ellipse_hot_speedup": speedup(plane_hot_s, ellipse_hot_s),
        "ellipse_forward_l2_vs_plane": relative_l2(ellipse_forward, plane_forward),
        "ellipse_adjoint_l2_vs_plane": relative_l2(ellipse_adjoint, plane_adjoint),
        "ellipse_adjoint_dot_error": ellipse_dot_error,
        "plane_build_times_s": " ".join(f"{item:.9g}" for item in plane_build_times),
        "ellipse_build_times_s": " ".join(f"{item:.9g}" for item in ellipse_build_times),
        "plane_hot_pair_times_s": " ".join(f"{item:.9g}" for item in plane_hot_times),
        "ellipse_hot_pair_times_s": " ".join(f"{item:.9g}" for item in ellipse_hot_times),
    }
    for iterations in args.iteration_counts:
        plane_amortized = (plane_build_s + float(iterations) * plane_hot_s) / float(
            iterations
        )
        ellipse_amortized = (
            ellipse_build_s + float(iterations) * ellipse_hot_s
        ) / float(iterations)
        row[f"plane_amortized_pair_s_n{iterations}"] = plane_amortized
        row[f"ellipse_amortized_pair_s_n{iterations}"] = ellipse_amortized
        row[f"plane_vs_ellipse_amortized_speedup_n{iterations}"] = speedup(
            plane_amortized,
            ellipse_amortized,
        )
    return row


def case_label(row: dict[str, Any]) -> str:
    return f"illum={row['illumination_na']:.3g} os={row['oversample']}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    report_iteration = payload["config"]["report_iteration"]
    lines = [
        "# ODT plane-detector ellipse-contour benchmark",
        "",
        "This benchmark compares a rectangular flat-detector q-list with an iso-q ellipse-contour resampling grid.",
        "",
        "- Plane reference: evaluate all rectangular detector pixels directly as `q = k(s_out - s_in)`.",
        "- Ellipse contour: express each plane pixel in the beam frame, interpolate from a polar grid in `(beam NA, phi)`, and evaluate those contour samples as q points.",
        "- Values above `1x` mean the ellipse-contour path is faster than direct plane-pixel evaluation.",
        "- The ellipse adjoint uses the transpose interpolation map, so its dot error checks internal operator consistency.",
        f"- Structured backend: `{rows[0]['structured_backend']}`; resampling backend: `{rows[0]['resampling_backend']}`; compact source: `{rows[0]['ellipse_source_compact']}`.",
        "",
        "## Results",
        "",
        "| case | plane q | ellipse q | active frac | inside plane | plane hot s | ellipse hot s | speedup | fwd err | adj err | dot err | N speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {pq} | {eq} | `{active}` | `{inside}` | `{ph}` | `{eh}` | `{speed}` | `{fe}` | `{ae}` | `{de}` | `{ns}` |".format(
                case=case_label(row),
                pq=row["plane_q_samples"],
                eq=row["ellipse_q_samples"],
                active=fmt(row["ellipse_source_active_fraction"], 4),
                inside=fmt(row["source_inside_plane_fraction"], 4),
                ph=fmt(row["plane_hot_pair_s"], 5),
                eh=fmt(row["ellipse_hot_pair_s"], 5),
                speed=fmt(row["plane_vs_ellipse_hot_speedup"], 4),
                fe=fmt(row["ellipse_forward_l2_vs_plane"], 4),
                ae=fmt(row["ellipse_adjoint_l2_vs_plane"], 4),
                de=fmt(row["ellipse_adjoint_dot_error"], 4),
                ns=fmt(row[f"plane_vs_ellipse_amortized_speedup_n{report_iteration}"], 4),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This tests detector-coordinate resampling, not a beam-frame object histogram.",
            "- With compact source enabled, unused full-ellipse grid points are removed before kernel build and contraction; interpolation weights are unchanged.",
            "- If ellipse error is small but speed is poor, the contour coordinate is physically useful but still needs a beam-frame or ellipse-aware contraction to become a fast solver.",
            "- `inside plane` below 1 means full iso-q ellipses extend outside the rectangular detector, so real measured data would contain partial arcs unless the detector is larger.",
            f"- `N speedup` uses N={report_iteration} repeated forward-adjoint pairs after one setup.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), dpi=180)
    ax_err, ax_speed = axes
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("illumination NA")
    ax_err.set_yscale("log")
    ax_err.set_ylabel("ellipse forward relative L2 vs plane")
    ax_speed.axhline(1.0, color="#444444", lw=1.0, ls="--", alpha=0.75)
    ax_speed.set_ylabel("plane / ellipse hot-pair speedup")
    oversamples = sorted({int(row["oversample"]) for row in rows})
    for oversample in oversamples:
        subset = [row for row in rows if int(row["oversample"]) == oversample]
        subset.sort(key=lambda row: row["illumination_na"])
        x = [row["illumination_na"] for row in subset]
        ax_err.plot(
            x,
            [row["ellipse_forward_l2_vs_plane"] for row in subset],
            marker="o",
            lw=1.7,
            label=f"os={oversample}",
        )
        ax_speed.plot(
            x,
            [row["plane_vs_ellipse_hot_speedup"] for row in subset],
            marker="o",
            lw=1.7,
            label=f"os={oversample}",
        )
    ax_err.set_title("A. Ellipse resampling error")
    ax_speed.set_title("B. Hot-pair speed")
    ax_err.legend(frameon=False, fontsize=8)
    ax_speed.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark iso-q ellipse-contour sampling for flat ODT detectors."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_plane_detector_ellipse")
    parser.add_argument("--illumination-na-values", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--oversample-values", default="1,2")
    parser.add_argument("--iteration-counts", default="1,2,4,8,16,32")
    parser.add_argument("--report-iteration", type=int, default=32)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--n-beta", type=int, default=192)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--plane-half-slope", type=float, default=None)
    parser.add_argument("--plane-pixels", type=int, default=32)
    parser.add_argument("--n-illum", type=int, default=9)
    parser.add_argument("--contour-radial", type=int, default=16)
    parser.add_argument("--contour-phi", type=int, default=64)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--source-na-margin-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--resampling-backend", choices=["auto", "numpy", "cpp"], default="auto")
    parser.add_argument("--no-compact-ellipse-source", dest="compact_ellipse_source", action="store_false")
    parser.set_defaults(compact_ellipse_source=True)
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--build-repeats", type=int, default=2)
    parser.add_argument("--hot-repeats", type=int, default=2)
    args = parser.parse_args()
    args.illumination_na_values = parse_float_list(args.illumination_na_values)
    args.oversample_values = parse_int_list(args.oversample_values)
    args.iteration_counts = parse_int_list(args.iteration_counts)
    if args.report_iteration not in args.iteration_counts:
        raise ValueError("report-iteration must be included in iteration-counts")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.plane_pixels <= 0:
        raise ValueError("plane-pixels must be positive")
    if args.contour_radial <= 0 or args.contour_phi <= 0:
        raise ValueError("contour-radial and contour-phi must be positive")
    if args.source_na_margin_fraction < 0.0:
        raise ValueError("source-na-margin-fraction must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    rows = [
        benchmark_case(args, illumination_na=illumination_na, oversample=oversample)
        for illumination_na in args.illumination_na_values
        for oversample in args.oversample_values
    ]
    payload = {
        "config": {
            **vars(args),
            "illumination_na_values": args.illumination_na_values,
            "oversample_values": args.oversample_values,
            "iteration_counts": args.iteration_counts,
        },
        "rows": rows,
    }
    output_prefix = ROOT / args.output_prefix
    write_json(output_prefix.with_suffix(".json"), payload)
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), payload)
    write_plot(
        output_prefix.with_name(output_prefix.name + ".png"),
        output_prefix.with_name(output_prefix.name + ".svg"),
        payload,
    )
    print(
        json.dumps(
            {
                "json": str(output_prefix.with_suffix(".json")),
                "csv": str(output_prefix.with_suffix(".csv")),
                "summary_md": str(output_prefix.with_name(output_prefix.name + "_summary.md")),
                "png": str(output_prefix.with_name(output_prefix.name + ".png")),
                "svg": str(output_prefix.with_name(output_prefix.name + ".svg")),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
