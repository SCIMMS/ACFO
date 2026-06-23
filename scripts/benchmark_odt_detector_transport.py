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
    build_shifted_axis_factorization,
    build_structured_kernel,
    complex_dot,
    detector_directions,
    ewald_cap_q_samples,
    illumination_directions,
    make_cylindrical_object,
    random_residual,
    recommended_h_cutoff,
    relative_complex_error,
    relative_l2,
    resolve_structured_backend,
    structured_adjoint,
    structured_adjoint_shifted_axis_fft_factored,
    structured_forward,
    structured_forward_shifted_axis_fft_factored,
)


@dataclass(frozen=True)
class DetectorTransportMap:
    local_indices: np.ndarray
    weights: np.ndarray
    illumination_index: np.ndarray
    source_count_per_block: int
    target_count_per_illumination: int
    source_detector_na: float
    max_target_beam_na: float
    outside_source_na_fraction: float
    radial_edge_clamped_fraction: float


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
    return (
        np.eye(3)
        + sin_angle * kx
        + (1.0 - cos_angle) * (kx @ kx)
    )


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


def moving_detector_q_samples(
    *,
    k: float,
    source_detector_na: float,
    source_cap_radial: int,
    source_cap_phi: int,
    illumination: np.ndarray,
) -> QSamples:
    source_detector = detector_directions(
        detector_na=source_detector_na,
        cap_radial=source_cap_radial,
        cap_phi=source_cap_phi,
    )
    q_blocks: list[np.ndarray] = []
    illum_blocks: list[np.ndarray] = []
    for illum_index, s_in in enumerate(illumination):
        rotation = rotation_from_z(s_in)
        detector_lab = source_detector @ rotation.T
        q_blocks.append(float(k) * (detector_lab - s_in[None, :]))
        illum_blocks.append(np.full(source_detector.shape[0], illum_index, dtype=np.int64))
    return q_samples_from_vectors(np.vstack(q_blocks), np.concatenate(illum_blocks))


def required_source_detector_na(
    *,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    illumination: np.ndarray,
    oversample_radial: int,
    margin_fraction: float,
) -> tuple[float, float]:
    fixed_detector = detector_directions(
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    max_r = float(detector_na)
    for s_in in illumination:
        rotation = rotation_from_z(s_in)
        detector_beam = fixed_detector @ rotation
        max_r = max(max_r, float(np.max(np.hypot(detector_beam[:, 0], detector_beam[:, 1]))))
    source_cap_radial = cap_radial * oversample_radial
    source_step = max_r / float(source_cap_radial)
    source_na = min(0.999, max_r + 0.5 * source_step + margin_fraction * max_r)
    return source_na, max_r


def build_detector_transport_map(
    *,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    source_detector_na: float,
    source_cap_radial: int,
    source_cap_phi: int,
    illumination: np.ndarray,
) -> DetectorTransportMap:
    fixed_detector = detector_directions(
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    source_count = int(source_cap_radial * source_cap_phi)
    target_count_per_illum = int(cap_radial * cap_phi)
    local_indices = np.empty((illumination.shape[0] * target_count_per_illum, 4), dtype=np.int64)
    weights = np.empty(local_indices.shape, dtype=float)
    illum_index = np.empty(local_indices.shape[0], dtype=np.int64)

    radial_step = float(source_detector_na) / float(source_cap_radial)
    phi_step = 2.0 * np.pi / float(source_cap_phi)
    outside = 0
    edge_clamped = 0
    max_target_r = 0.0
    row = 0
    for illum, s_in in enumerate(illumination):
        rotation = rotation_from_z(s_in)
        detector_beam = fixed_detector @ rotation
        target_r = np.hypot(detector_beam[:, 0], detector_beam[:, 1])
        target_phi = np.mod(np.arctan2(detector_beam[:, 1], detector_beam[:, 0]), 2.0 * np.pi)
        max_target_r = max(max_target_r, float(np.max(target_r)))
        for r_value, phi_value in zip(target_r, target_phi, strict=True):
            raw_r = float(r_value) / radial_step - 0.5
            radial0_raw = math.floor(raw_r)
            radial_weight = raw_r - float(radial0_raw)
            if r_value > source_detector_na:
                outside += 1
            if radial0_raw < 0:
                radial0 = radial1 = 0
                radial_weight = 0.0
                edge_clamped += 1
            elif radial0_raw >= source_cap_radial - 1:
                radial0 = radial1 = source_cap_radial - 1
                radial_weight = 0.0
                edge_clamped += 1
            else:
                radial0 = radial0_raw
                radial1 = radial0 + 1
            raw_phi = float(phi_value) / phi_step
            phi0_base = math.floor(raw_phi)
            phi_weight = raw_phi - float(phi0_base)
            phi0 = phi0_base % source_cap_phi
            phi1 = (phi0 + 1) % source_cap_phi
            local_indices[row] = [
                radial0 * source_cap_phi + phi0,
                radial1 * source_cap_phi + phi0,
                radial0 * source_cap_phi + phi1,
                radial1 * source_cap_phi + phi1,
            ]
            weights[row] = [
                (1.0 - radial_weight) * (1.0 - phi_weight),
                radial_weight * (1.0 - phi_weight),
                (1.0 - radial_weight) * phi_weight,
                radial_weight * phi_weight,
            ]
            illum_index[row] = illum
            row += 1

    total_targets = max(1, local_indices.shape[0])
    return DetectorTransportMap(
        local_indices=np.ascontiguousarray(local_indices),
        weights=np.ascontiguousarray(weights),
        illumination_index=np.ascontiguousarray(illum_index),
        source_count_per_block=source_count,
        target_count_per_illumination=target_count_per_illum,
        source_detector_na=float(source_detector_na),
        max_target_beam_na=float(max_target_r),
        outside_source_na_fraction=float(outside / total_targets),
        radial_edge_clamped_fraction=float(edge_clamped / total_targets),
    )


def transport_interpolate(
    source_values: np.ndarray,
    transport: DetectorTransportMap,
    *,
    per_illumination_source: bool,
) -> np.ndarray:
    source_values = np.asarray(source_values, dtype=np.complex128)
    if per_illumination_source:
        indices = transport.local_indices + (
            transport.illumination_index[:, None] * transport.source_count_per_block
        )
    else:
        indices = transport.local_indices
    return np.sum(source_values[indices] * transport.weights, axis=1)


def transport_scatter_adjoint(
    residual: np.ndarray,
    transport: DetectorTransportMap,
    *,
    source_blocks: int,
    per_illumination_source: bool,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.shape != (transport.local_indices.shape[0],):
        raise ValueError("residual shape does not match transport target count")
    source = np.zeros(source_blocks * transport.source_count_per_block, dtype=np.complex128)
    if per_illumination_source:
        indices = transport.local_indices + (
            transport.illumination_index[:, None] * transport.source_count_per_block
        )
    else:
        indices = transport.local_indices
    np.add.at(source, indices.ravel(), (transport.weights * residual[:, None]).ravel())
    return source


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
    q_fixed = ewald_cap_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        geometry="shifted",
        n_illum=args.n_illum,
        illumination_na=illumination_na,
    )
    q_base = ewald_cap_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        geometry="axis",
        n_illum=1,
        illumination_na=0.0,
    )
    source_cap_radial = args.cap_radial * oversample
    source_cap_phi = args.cap_phi * oversample
    source_detector_na, max_target_beam_na = required_source_detector_na(
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        illumination=illumination,
        oversample_radial=oversample,
        margin_fraction=args.source_na_margin_fraction,
    )
    q_axis_source = ewald_cap_q_samples(
        k=args.k,
        detector_na=source_detector_na,
        cap_radial=source_cap_radial,
        cap_phi=source_cap_phi,
        geometry="axis",
        n_illum=1,
        illumination_na=0.0,
    )
    q_moving_source = moving_detector_q_samples(
        k=args.k,
        source_detector_na=source_detector_na,
        source_cap_radial=source_cap_radial,
        source_cap_phi=source_cap_phi,
        illumination=illumination,
    )
    transport = build_detector_transport_map(
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        source_detector_na=source_detector_na,
        source_cap_radial=source_cap_radial,
        source_cap_phi=source_cap_phi,
        illumination=illumination,
    )

    backend = resolve_structured_backend(args.structured_backend)
    residual = random_residual(q_fixed, seed=args.seed + 7919)

    exact_h_cutoff = (
        recommended_h_cutoff(q_base, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    axis_h_cutoff = (
        recommended_h_cutoff(q_axis_source, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    moving_h_cutoff = (
        recommended_h_cutoff(q_moving_source, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    exact_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=exact_h_cutoff,
    )
    axis_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=axis_h_cutoff,
    )
    moving_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=moving_h_cutoff,
    )

    exact_factorization, exact_build_s, exact_build_times = median_time(
        lambda: build_shifted_axis_factorization(
            exact_plan,
            k=args.k,
            detector_na=args.detector_na,
            cap_radial=args.cap_radial,
            cap_phi=args.cap_phi,
            n_illum=args.n_illum,
            illumination_na=illumination_na,
        ),
        repeats=args.build_repeats,
    )
    axis_kernel, axis_build_s, axis_build_times = median_time(
        lambda: build_structured_kernel(axis_plan, q_axis_source),
        repeats=args.build_repeats,
    )
    moving_kernel, moving_build_s, moving_build_times = median_time(
        lambda: build_structured_kernel(moving_plan, q_moving_source),
        repeats=args.build_repeats,
    )

    def exact_pair():
        forward = structured_forward_shifted_axis_fft_factored(
            exact_plan,
            obj.coeff,
            exact_factorization,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        adjoint = structured_adjoint_shifted_axis_fft_factored(
            exact_plan,
            exact_factorization,
            residual,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        return forward, adjoint

    def axis_transport_pair():
        source_forward = structured_forward(
            axis_plan,
            obj.coeff,
            q_axis_source,
            kernel=axis_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        forward = transport_interpolate(
            source_forward,
            transport,
            per_illumination_source=False,
        )
        source_residual = transport_scatter_adjoint(
            residual,
            transport,
            source_blocks=1,
            per_illumination_source=False,
        )
        adjoint = structured_adjoint(
            axis_plan,
            q_axis_source,
            source_residual,
            kernel=axis_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        return forward, adjoint

    def moving_transport_pair():
        source_forward = structured_forward(
            moving_plan,
            obj.coeff,
            q_moving_source,
            kernel=moving_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        forward = transport_interpolate(
            source_forward,
            transport,
            per_illumination_source=True,
        )
        source_residual = transport_scatter_adjoint(
            residual,
            transport,
            source_blocks=args.n_illum,
            per_illumination_source=True,
        )
        adjoint = structured_adjoint(
            moving_plan,
            q_moving_source,
            source_residual,
            kernel=moving_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        return forward, adjoint

    (exact_forward, exact_adjoint), exact_hot_pair_s, exact_hot_times = median_time(
        exact_pair,
        repeats=args.hot_repeats,
    )
    (axis_forward, axis_adjoint), axis_hot_pair_s, axis_hot_times = median_time(
        axis_transport_pair,
        repeats=args.hot_repeats,
    )
    (moving_forward, moving_adjoint), moving_hot_pair_s, moving_hot_times = median_time(
        moving_transport_pair,
        repeats=args.hot_repeats,
    )

    axis_dot_error = relative_complex_error(
        complex_dot(axis_forward, residual),
        complex_dot(obj.coeff, axis_adjoint),
    )
    moving_dot_error = relative_complex_error(
        complex_dot(moving_forward, residual),
        complex_dot(obj.coeff, moving_adjoint),
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
        "illumination_na": illumination_na,
        "n_illum": args.n_illum,
        "cap_radial": args.cap_radial,
        "cap_phi": args.cap_phi,
        "oversample": oversample,
        "source_cap_radial": source_cap_radial,
        "source_cap_phi": source_cap_phi,
        "source_detector_na": source_detector_na,
        "max_target_beam_na": max_target_beam_na,
        "q_fixed_samples": q_fixed.count,
        "q_axis_source_samples": q_axis_source.count,
        "q_moving_source_samples": q_moving_source.count,
        "exact_h_cutoff": exact_h_cutoff,
        "axis_h_cutoff": axis_h_cutoff,
        "moving_h_cutoff": moving_h_cutoff,
        "exact_used_modes": exact_plan.used_modes,
        "axis_used_modes": axis_plan.used_modes,
        "moving_used_modes": moving_plan.used_modes,
        "structured_backend": backend,
        "cpp_threads": args.cpp_threads,
        "outside_source_na_fraction": transport.outside_source_na_fraction,
        "radial_edge_clamped_fraction": transport.radial_edge_clamped_fraction,
        "exact_build_s": exact_build_s,
        "axis_transport_build_s": axis_build_s,
        "moving_transport_build_s": moving_build_s,
        "axis_kernel_mib": kernel_mib(axis_kernel),
        "moving_kernel_mib": kernel_mib(moving_kernel),
        "exact_hot_pair_s": exact_hot_pair_s,
        "axis_transport_hot_pair_s": axis_hot_pair_s,
        "moving_transport_hot_pair_s": moving_hot_pair_s,
        "exact_vs_axis_transport_hot_speedup": speedup(exact_hot_pair_s, axis_hot_pair_s),
        "exact_vs_moving_transport_hot_speedup": speedup(
            exact_hot_pair_s,
            moving_hot_pair_s,
        ),
        "axis_transport_forward_l2_vs_exact": relative_l2(axis_forward, exact_forward),
        "axis_transport_adjoint_l2_vs_exact": relative_l2(axis_adjoint, exact_adjoint),
        "moving_transport_forward_l2_vs_exact": relative_l2(
            moving_forward,
            exact_forward,
        ),
        "moving_transport_adjoint_l2_vs_exact": relative_l2(
            moving_adjoint,
            exact_adjoint,
        ),
        "axis_transport_adjoint_dot_error": axis_dot_error,
        "moving_transport_adjoint_dot_error": moving_dot_error,
        "exact_build_times_s": " ".join(f"{item:.9g}" for item in exact_build_times),
        "axis_transport_build_times_s": " ".join(
            f"{item:.9g}" for item in axis_build_times
        ),
        "moving_transport_build_times_s": " ".join(
            f"{item:.9g}" for item in moving_build_times
        ),
        "exact_hot_pair_times_s": " ".join(f"{item:.9g}" for item in exact_hot_times),
        "axis_transport_hot_pair_times_s": " ".join(
            f"{item:.9g}" for item in axis_hot_times
        ),
        "moving_transport_hot_pair_times_s": " ".join(
            f"{item:.9g}" for item in moving_hot_times
        ),
    }
    for iterations in args.iteration_counts:
        exact_amortized = (exact_build_s + float(iterations) * exact_hot_pair_s) / float(
            iterations
        )
        axis_amortized = (
            axis_build_s + float(iterations) * axis_hot_pair_s
        ) / float(iterations)
        moving_amortized = (
            moving_build_s + float(iterations) * moving_hot_pair_s
        ) / float(iterations)
        row[f"exact_amortized_pair_s_n{iterations}"] = exact_amortized
        row[f"axis_transport_amortized_pair_s_n{iterations}"] = axis_amortized
        row[f"moving_transport_amortized_pair_s_n{iterations}"] = moving_amortized
        row[f"exact_vs_axis_transport_amortized_speedup_n{iterations}"] = speedup(
            exact_amortized,
            axis_amortized,
        )
        row[f"exact_vs_moving_transport_amortized_speedup_n{iterations}"] = speedup(
            exact_amortized,
            moving_amortized,
        )
    return row


def case_label(row: dict[str, Any]) -> str:
    return f"illum={row['illumination_na']:.3g} os={row['oversample']}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    report_iteration = payload["config"]["report_iteration"]
    lines = [
        "# ODT detector-transport benchmark",
        "",
        "This benchmark compares the exact fixed-detector shifted-cap operator with two transport-style approximations.",
        "",
        "- `axis transport`: rotate fixed detector directions into each beam frame, interpolate from one oversized axis-cap field, and ignore object rotation.",
        "- `moving transport`: evaluate the field on oversized detector caps that move with each beam, then interpolate those cap samples back to the fixed detector directions.",
        "- Both transport adjoints use the transpose interpolation map, so their own adjoint dot tests should remain small even when the approximation error is large.",
        "",
        "Values above `1x` in speedup columns mean the transport path is faster than the exact phase-ramp axis-FFT reference.",
        "",
        "## Results",
        "",
        "| case | source NA | axis q | moving q | exact hot s | axis hot s | axis speedup | axis fwd err | axis adj err | moving hot s | moving speedup | moving fwd err | moving adj err | clamp frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | `{sna}` | {aq} | {mq} | `{eh}` | `{ah}` | `{aspeed}` | `{afe}` | `{aae}` | `{mh}` | `{mspeed}` | `{mfe}` | `{mae}` | `{clamp}` |".format(
                case=case_label(row),
                sna=fmt(row["source_detector_na"], 4),
                aq=row["q_axis_source_samples"],
                mq=row["q_moving_source_samples"],
                eh=fmt(row["exact_hot_pair_s"], 5),
                ah=fmt(row["axis_transport_hot_pair_s"], 5),
                aspeed=fmt(row["exact_vs_axis_transport_hot_speedup"], 4),
                afe=fmt(row["axis_transport_forward_l2_vs_exact"], 4),
                aae=fmt(row["axis_transport_adjoint_l2_vs_exact"], 4),
                mh=fmt(row["moving_transport_hot_pair_s"], 5),
                mspeed=fmt(row["exact_vs_moving_transport_hot_speedup"], 4),
                mfe=fmt(row["moving_transport_forward_l2_vs_exact"], 4),
                mae=fmt(row["moving_transport_adjoint_l2_vs_exact"], 4),
                clamp=fmt(row["radial_edge_clamped_fraction"], 4),
            )
        )
    lines.extend(
        [
            "",
            "## Adjoint consistency",
            "",
            "| case | axis dot error | moving dot error |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {case_label(row)} | `{fmt(row['axis_transport_adjoint_dot_error'], 4)}` | `{fmt(row['moving_transport_adjoint_dot_error'], 4)}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `axis transport` is the cheap rotation-theorem-like approximation. It is only defensible when the object is effectively invariant under the beam-frame rotation or when screening tolerates the error.",
            "- `moving transport` isolates reciprocal detector interpolation error. It still evaluates the moved q cap, so it is an accuracy baseline for detector transport rather than a final fast solver.",
            f"- Amortized speedups are also written for N={report_iteration} forward-adjoint pairs in the JSON/CSV outputs.",
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
    ax_err.set_ylabel("forward relative L2 vs exact")
    ax_speed.axhline(1.0, color="#444444", lw=1.0, ls="--", alpha=0.75)
    ax_speed.set_ylabel("exact / transport hot-pair speedup")

    oversamples = sorted({int(row["oversample"]) for row in rows})
    for oversample in oversamples:
        subset = [row for row in rows if int(row["oversample"]) == oversample]
        subset.sort(key=lambda row: row["illumination_na"])
        x = [row["illumination_na"] for row in subset]
        ax_err.plot(
            x,
            [row["axis_transport_forward_l2_vs_exact"] for row in subset],
            marker="o",
            lw=1.6,
            label=f"axis os={oversample}",
        )
        ax_err.plot(
            x,
            [row["moving_transport_forward_l2_vs_exact"] for row in subset],
            marker="s",
            lw=1.3,
            ls="--",
            label=f"moving os={oversample}",
        )
        ax_speed.plot(
            x,
            [row["exact_vs_axis_transport_hot_speedup"] for row in subset],
            marker="o",
            lw=1.6,
            label=f"axis os={oversample}",
        )
        ax_speed.plot(
            x,
            [row["exact_vs_moving_transport_hot_speedup"] for row in subset],
            marker="s",
            lw=1.3,
            ls="--",
            label=f"moving os={oversample}",
        )
    ax_err.set_title("A. Transport error")
    ax_speed.set_title("B. Hot-pair speed")
    ax_err.legend(frameon=False, fontsize=8)
    ax_speed.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark detector-transport approximations for shifted ODT caps."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_detector_transport")
    parser.add_argument("--illumination-na-values", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--oversample-values", default="1,2,4")
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
    parser.add_argument("--n-illum", type=int, default=9)
    parser.add_argument("--cap-radial", type=int, default=8)
    parser.add_argument("--cap-phi", type=int, default=32)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--source-na-margin-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
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
