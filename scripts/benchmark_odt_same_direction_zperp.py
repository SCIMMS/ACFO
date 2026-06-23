from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy import special

from benchmark_odt_cone_axis_decomposition import default_l_cutoff, fmt
from benchmark_odt_cone_illumination import ROOT, parse_int_list, q_samples_from_vectors
from benchmark_odt_ewald_cap_operator import (
    QSamples,
    ShiftedAxisFactorization,
    StructuredOdtPlan,
    _axis_grid_adjoint_fft_compact,
    _axis_grid_forward_fft,
    _cpp_odt_module,
    build_shifted_axis_phases,
    build_structured_kernel,
    complex_dot,
    detector_directions,
    make_cylindrical_object,
    random_residual,
    recommended_h_cutoff,
    relative_complex_error,
    relative_l2,
    resolve_structured_backend,
    structured_adjoint_shifted_axis_fft_factored,
    structured_forward_shifted_axis_fft_factored,
)


@dataclass(frozen=True)
class SameDirectionDecomposition:
    base_q: QSamples
    flat_q: QSamples
    factorization: ShiftedAxisFactorization
    magnitudes: np.ndarray
    direction_phi: float
    l_values: np.ndarray
    transverse_by_mag: np.ndarray
    axial_phase_by_mag: np.ndarray
    source_slots: np.ndarray
    plan: StructuredOdtPlan


@dataclass(frozen=True)
class MagnitudeSvdCompression:
    u: np.ndarray
    weights: np.ndarray
    weights_conj: np.ndarray
    singular_values: np.ndarray


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


def speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


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


def same_direction_illumination(
    *,
    magnitudes: np.ndarray,
    direction_phi: float,
) -> np.ndarray:
    magnitudes = np.asarray(magnitudes, dtype=float)
    if magnitudes.ndim != 1 or magnitudes.size == 0:
        raise ValueError("magnitudes must be a non-empty vector")
    if np.any(magnitudes < 0.0) or np.any(magnitudes >= 1.0):
        raise ValueError("illumination magnitudes must be in [0, 1)")
    sx = magnitudes * math.cos(float(direction_phi))
    sy = magnitudes * math.sin(float(direction_phi))
    sz = np.sqrt(np.maximum(1.0 - magnitudes * magnitudes, 0.0))
    return np.column_stack([sx, sy, sz])


def same_direction_q_samples(
    *,
    k: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    illumination: np.ndarray,
) -> tuple[QSamples, QSamples]:
    detector = detector_directions(
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    base_q_vectors = k * (detector - np.array([[0.0, 0.0, 1.0]], dtype=float))
    base_q = q_samples_from_vectors(
        base_q_vectors,
        np.zeros(detector.shape[0], dtype=np.int64),
    )
    blocks = []
    illum_index = []
    for index, s_in in enumerate(illumination):
        blocks.append(k * (detector - s_in[None, :]))
        illum_index.append(np.full(detector.shape[0], index, dtype=np.int64))
    flat_q = q_samples_from_vectors(np.vstack(blocks), np.concatenate(illum_index))
    return flat_q, base_q


def build_same_direction_decomposition(
    plan: StructuredOdtPlan,
    *,
    k: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    magnitudes: np.ndarray,
    direction_phi: float,
    l_cutoff: int,
) -> SameDirectionDecomposition:
    illumination = same_direction_illumination(
        magnitudes=magnitudes,
        direction_phi=direction_phi,
    )
    flat_q, base_q = same_direction_q_samples(
        k=k,
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
        illumination=illumination,
    )
    l_values = np.arange(-int(l_cutoff), int(l_cutoff) + 1, dtype=np.int64)
    source_slots = np.mod(
        plan.h_values[:, None] + l_values[None, :],
        plan.n_beta,
    ).astype(np.int64)
    arg = float(k) * np.asarray(magnitudes, dtype=float)[:, None] * plan.r_axis[None, :]
    bessel = special.jv(l_values[None, :, None], arg[:, None, :])
    direction_phase = np.exp(-1j * float(direction_phi) * l_values)
    transverse = (
        ((-1j) ** l_values)[None, :, None]
        * direction_phase[None, :, None]
        * bessel
    )
    cos_alpha = np.sqrt(np.maximum(1.0 - np.asarray(magnitudes, dtype=float) ** 2, 0.0))
    axial_phase = np.exp(
        1j * float(k) * (1.0 - cos_alpha[:, None]) * plan.z_axis[None, :]
    )
    factorization = ShiftedAxisFactorization(
        base_q=base_q,
        illumination=illumination,
        phase=np.empty((0, 0, 0, 0), dtype=np.complex128),
        beta_twiddle=np.empty((0, 0), dtype=np.complex128),
        kernel=build_structured_kernel(plan, base_q),
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    return SameDirectionDecomposition(
        base_q=base_q,
        flat_q=flat_q,
        factorization=factorization,
        magnitudes=np.ascontiguousarray(magnitudes, dtype=float),
        direction_phi=float(direction_phi),
        l_values=np.ascontiguousarray(l_values),
        transverse_by_mag=np.ascontiguousarray(transverse),
        axial_phase_by_mag=np.ascontiguousarray(axial_phase),
        source_slots=np.ascontiguousarray(source_slots),
        plan=plan,
    )


def build_exact_factorization(
    plan: StructuredOdtPlan,
    decomp: SameDirectionDecomposition,
    *,
    k: float,
) -> ShiftedAxisFactorization:
    return ShiftedAxisFactorization(
        base_q=decomp.base_q,
        illumination=same_direction_illumination(
            magnitudes=decomp.magnitudes,
            direction_phi=decomp.direction_phi,
        ),
        phase=build_shifted_axis_phases(
            plan,
            k=k,
            illumination=same_direction_illumination(
                magnitudes=decomp.magnitudes,
                direction_phi=decomp.direction_phi,
            ),
        ),
        beta_twiddle=np.ascontiguousarray(
            np.exp(1j * plan.h_values[:, None] * plan.beta_axis[None, :])
        ),
        kernel=decomp.factorization.kernel,
        cap_radial=decomp.factorization.cap_radial,
        cap_phi=decomp.factorization.cap_phi,
    )


def _axis_slices(decomp: SameDirectionDecomposition) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cap_phi = decomp.factorization.cap_phi
    radial = decomp.factorization.kernel.radial[:, ::cap_phi, :]
    axial = decomp.factorization.kernel.axial[::cap_phi, :]
    mode_phase = decomp.factorization.kernel.angular[:, 0]
    h_slots = np.mod(decomp.plan.h_values, cap_phi).astype(np.int64)
    return (
        np.ascontiguousarray(radial),
        np.ascontiguousarray(axial),
        np.ascontiguousarray(mode_phase),
        np.ascontiguousarray(h_slots),
    )


def same_direction_forward(
    coeff: np.ndarray,
    decomp: SameDirectionDecomposition,
    *,
    backend: str,
    cpp_threads: int,
    grouped: bool,
) -> np.ndarray:
    plan = decomp.plan
    coeff_h_full = np.fft.ifft(coeff, axis=2) * float(plan.n_beta)
    radial, axial, mode_phase, h_slots = _axis_slices(decomp)
    cap_phi = decomp.factorization.cap_phi
    cpp_odt = _cpp_odt_module(required=backend == "cpp")
    if cpp_odt is not None and hasattr(cpp_odt, "same_direction_forward_fold"):
        if grouped:
            folded = cpp_odt.same_direction_forward_fold(
                np.ascontiguousarray(coeff_h_full),
                radial,
                axial,
                mode_phase,
                h_slots,
                decomp.transverse_by_mag,
                decomp.axial_phase_by_mag,
                decomp.source_slots,
                int(cap_phi),
                int(cpp_threads),
            )
            return np.fft.fft(folded, axis=2).reshape(decomp.flat_q.count)
        blocks = []
        for mag_index in range(decomp.magnitudes.size):
            folded = cpp_odt.same_direction_forward_fold(
                np.ascontiguousarray(coeff_h_full),
                radial,
                axial,
                mode_phase,
                h_slots,
                decomp.transverse_by_mag[mag_index : mag_index + 1],
                decomp.axial_phase_by_mag[mag_index : mag_index + 1],
                decomp.source_slots,
                int(cap_phi),
                int(cpp_threads),
            )
            blocks.append(np.fft.fft(folded, axis=2).reshape(decomp.base_q.count))
        return np.concatenate(blocks)
    if backend == "cpp":
        raise RuntimeError("waxs_cake._cpp_odt lacks same_direction_forward_fold")
    coeff_sources = coeff_h_full[:, :, decomp.source_slots]
    coeff_h_all = np.einsum(
        "rzhl,mlr,mz->mrzh",
        coeff_sources,
        decomp.transverse_by_mag,
        decomp.axial_phase_by_mag,
        optimize=True,
    )
    return _axis_grid_forward_fft(
        plan,
        np.ascontiguousarray(coeff_h_all),
        decomp.factorization,
        backend=backend,
        cpp_threads=cpp_threads,
    )


def same_direction_adjoint(
    residual: np.ndarray,
    decomp: SameDirectionDecomposition,
    *,
    backend: str,
    cpp_threads: int,
    grouped: bool,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.shape != (decomp.flat_q.count,):
        raise ValueError("residual shape does not match same-direction q stack")
    plan = decomp.plan
    radial, axial, mode_phase, h_slots = _axis_slices(decomp)
    cap_radial = decomp.factorization.cap_radial
    cap_phi = decomp.factorization.cap_phi
    residual_grid = residual.reshape(decomp.magnitudes.size, cap_radial, cap_phi)
    residual_modes = np.fft.ifft(residual_grid, axis=2) * float(cap_phi)
    cpp_odt = _cpp_odt_module(required=backend == "cpp")
    if cpp_odt is not None and hasattr(cpp_odt, "same_direction_adjoint_unfold_scatter"):
        if grouped:
            out_h = cpp_odt.same_direction_adjoint_unfold_scatter(
                np.ascontiguousarray(residual_modes),
                radial,
                axial,
                mode_phase,
                h_slots,
                decomp.transverse_by_mag,
                decomp.axial_phase_by_mag,
                decomp.source_slots,
                int(plan.n_beta),
                int(cpp_threads),
            )
            return np.fft.fft(out_h, axis=2)
        out_h = np.zeros(
            (plan.r_axis.size, plan.z_axis.size, plan.n_beta),
            dtype=np.complex128,
        )
        for mag_index in range(decomp.magnitudes.size):
            out_h += cpp_odt.same_direction_adjoint_unfold_scatter(
                np.ascontiguousarray(residual_modes[mag_index : mag_index + 1]),
                radial,
                axial,
                mode_phase,
                h_slots,
                decomp.transverse_by_mag[mag_index : mag_index + 1],
                decomp.axial_phase_by_mag[mag_index : mag_index + 1],
                decomp.source_slots,
                int(plan.n_beta),
                int(cpp_threads),
            )
        return np.fft.fft(out_h, axis=2)
    if backend == "cpp":
        raise RuntimeError("waxs_cake._cpp_odt lacks same_direction_adjoint_unfold_scatter")
    compact = _axis_grid_adjoint_fft_compact(
        plan,
        decomp.factorization,
        residual,
        backend=backend,
        cpp_threads=cpp_threads,
    )
    contributions = np.einsum(
        "mrzh,mlr,mz->rzhl",
        compact,
        np.conj(decomp.transverse_by_mag),
        np.conj(decomp.axial_phase_by_mag),
        optimize=True,
    )
    out_h = np.zeros((plan.r_axis.size, plan.z_axis.size, plan.n_beta), dtype=np.complex128)
    for h_index in range(decomp.source_slots.shape[0]):
        for l_index in range(decomp.source_slots.shape[1]):
            out_h[:, :, decomp.source_slots[h_index, l_index]] += contributions[
                :, :, h_index, l_index
            ]
    return np.fft.fft(out_h, axis=2)


def build_magnitude_svd(decomp: SameDirectionDecomposition) -> MagnitudeSvdCompression:
    product = np.einsum(
        "mlr,mz->mlrz",
        decomp.transverse_by_mag,
        decomp.axial_phase_by_mag,
        optimize=True,
    )
    matrix = product.reshape(decomp.magnitudes.size, -1)
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    weights = (singular_values[:, None] * vh).reshape(
        singular_values.size,
        decomp.l_values.size,
        decomp.plan.r_axis.size,
        decomp.plan.z_axis.size,
    )
    return MagnitudeSvdCompression(
        u=np.ascontiguousarray(u),
        weights=np.ascontiguousarray(weights),
        weights_conj=np.ascontiguousarray(np.conj(weights)),
        singular_values=np.ascontiguousarray(singular_values),
    )


def _rank_factorization(decomp: SameDirectionDecomposition, rank: int) -> ShiftedAxisFactorization:
    return ShiftedAxisFactorization(
        base_q=decomp.base_q,
        illumination=np.zeros((rank, 3), dtype=float),
        phase=np.empty((0, 0, 0, 0), dtype=np.complex128),
        beta_twiddle=np.empty((0, 0), dtype=np.complex128),
        kernel=decomp.factorization.kernel,
        cap_radial=decomp.factorization.cap_radial,
        cap_phi=decomp.factorization.cap_phi,
    )


def compressed_forward(
    coeff: np.ndarray,
    decomp: SameDirectionDecomposition,
    compression: MagnitudeSvdCompression,
    *,
    rank: int,
    backend: str,
    cpp_threads: int,
) -> np.ndarray:
    plan = decomp.plan
    coeff_h_full = np.fft.ifft(coeff, axis=2) * float(plan.n_beta)
    weights = compression.weights[:rank]
    backend = resolve_structured_backend(backend)
    cpp_odt = _cpp_odt_module(required=backend == "cpp")
    if backend == "cpp" and cpp_odt is not None and hasattr(cpp_odt, "svd_rank_forward_fold"):
        radial, axial, mode_phase, h_slots = _axis_slices(decomp)
        folded = cpp_odt.svd_rank_forward_fold(
            np.ascontiguousarray(coeff_h_full),
            radial,
            axial,
            mode_phase,
            h_slots,
            np.ascontiguousarray(weights),
            decomp.source_slots,
            int(decomp.factorization.cap_phi),
            int(cpp_threads),
        )
        rank_fields = np.fft.fft(folded, axis=2).reshape(rank, decomp.base_q.count)
        fields = compression.u[:, :rank] @ rank_fields
        return np.ascontiguousarray(fields.reshape(decomp.flat_q.count))
    if backend == "cpp":
        raise RuntimeError("waxs_cake._cpp_odt lacks svd_rank_forward_fold")
    coeff_sources = coeff_h_full[:, :, decomp.source_slots]
    coeff_h_rank = np.einsum(
        "rzhl,slrz->srzh",
        coeff_sources,
        weights,
        optimize=True,
    )
    rank_fields = _axis_grid_forward_fft(
        plan,
        np.ascontiguousarray(coeff_h_rank),
        _rank_factorization(decomp, rank),
        backend=backend,
        cpp_threads=cpp_threads,
    ).reshape(rank, decomp.base_q.count)
    fields = compression.u[:, :rank] @ rank_fields
    return np.ascontiguousarray(fields.reshape(decomp.flat_q.count))


def compressed_adjoint(
    residual: np.ndarray,
    decomp: SameDirectionDecomposition,
    compression: MagnitudeSvdCompression,
    *,
    rank: int,
    backend: str,
    cpp_threads: int,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    rank_residual = compression.u[:, :rank].conj().T @ residual.reshape(
        decomp.magnitudes.size,
        decomp.base_q.count,
    )
    backend = resolve_structured_backend(backend)
    cpp_odt = _cpp_odt_module(required=backend == "cpp")
    if backend == "cpp" and cpp_odt is not None and hasattr(cpp_odt, "svd_rank_adjoint_unfold_scatter"):
        cap_radial = decomp.factorization.cap_radial
        cap_phi = decomp.factorization.cap_phi
        residual_grid = rank_residual.reshape(rank, cap_radial, cap_phi)
        residual_modes = np.fft.ifft(residual_grid, axis=2) * float(cap_phi)
        radial, axial, mode_phase, h_slots = _axis_slices(decomp)
        out_h = cpp_odt.svd_rank_adjoint_unfold_scatter(
            np.ascontiguousarray(residual_modes),
            radial,
            axial,
            mode_phase,
            h_slots,
            np.ascontiguousarray(compression.weights_conj[:rank]),
            decomp.source_slots,
            int(decomp.plan.n_beta),
            int(cpp_threads),
            True,
        )
        return np.fft.fft(out_h, axis=2)
    if backend == "cpp":
        raise RuntimeError("waxs_cake._cpp_odt lacks svd_rank_adjoint_unfold_scatter")
    compact = _axis_grid_adjoint_fft_compact(
        decomp.plan,
        _rank_factorization(decomp, rank),
        np.ascontiguousarray(rank_residual.reshape(rank * decomp.base_q.count)),
        backend=backend,
        cpp_threads=cpp_threads,
    )
    contributions = np.einsum(
        "srzh,slrz->rzhl",
        compact,
        compression.weights_conj[:rank],
        optimize=True,
    )
    out_h = np.zeros(
        (decomp.plan.r_axis.size, decomp.plan.z_axis.size, decomp.plan.n_beta),
        dtype=np.complex128,
    )
    for h_index in range(decomp.source_slots.shape[0]):
        for l_index in range(decomp.source_slots.shape[1]):
            out_h[:, :, decomp.source_slots[h_index, l_index]] += contributions[
                :, :, h_index, l_index
            ]
    return np.fft.fft(out_h, axis=2)


def benchmark_case(args: argparse.Namespace, *, n_mag: int) -> list[dict[str, Any]]:
    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    magnitudes = np.linspace(args.min_illumination_na, args.max_illumination_na, n_mag)
    illumination = same_direction_illumination(
        magnitudes=magnitudes,
        direction_phi=args.direction_phi,
    )
    _, base_q = same_direction_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        illumination=illumination,
    )
    flat_h_cutoff = (
        args.h_cutoff
        if args.h_cutoff is not None
        else recommended_h_cutoff(
            same_direction_q_samples(
                k=args.k,
                detector_na=args.detector_na,
                cap_radial=args.cap_radial,
                cap_phi=args.cap_phi,
                illumination=illumination,
            )[0],
            args.r_max,
            args.n_beta,
            args.h_margin,
        )
    )
    axis_h_cutoff = (
        args.h_cutoff
        if args.h_cutoff is not None
        else recommended_h_cutoff(base_q, args.r_max, args.n_beta, args.h_margin)
    )
    axis_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=axis_h_cutoff,
    )
    l_cutoff = default_l_cutoff(
        k=args.k,
        illumination_na=args.max_illumination_na,
        r_max=args.r_max,
        margin=args.l_margin,
        n_beta=args.n_beta,
    )
    decomp, decomp_build_s, decomp_build_times = median_time(
        lambda: build_same_direction_decomposition(
            axis_plan,
            k=args.k,
            detector_na=args.detector_na,
            cap_radial=args.cap_radial,
            cap_phi=args.cap_phi,
            magnitudes=magnitudes,
            direction_phi=args.direction_phi,
            l_cutoff=l_cutoff,
        ),
        repeats=args.build_repeats,
    )
    exact_factorization, exact_build_s, exact_build_times = median_time(
        lambda: build_exact_factorization(axis_plan, decomp, k=args.k),
        repeats=args.build_repeats,
    )
    compression, svd_build_s, svd_build_times = median_time(
        lambda: build_magnitude_svd(decomp),
        repeats=args.build_repeats,
    )
    backend = resolve_structured_backend(args.structured_backend)
    residual = random_residual(decomp.flat_q, seed=args.seed + 1009 + n_mag)

    def phase_pair():
        forward = structured_forward_shifted_axis_fft_factored(
            axis_plan,
            obj.coeff,
            exact_factorization,
            backend=backend,
            cpp_threads=args.cpp_threads,
            phase_backend=args.phase_backend,
        )
        adjoint = structured_adjoint_shifted_axis_fft_factored(
            axis_plan,
            exact_factorization,
            residual,
            backend=backend,
            cpp_threads=args.cpp_threads,
            phase_backend=args.phase_backend,
        )
        return forward, adjoint

    def split_pair():
        forward = same_direction_forward(
            obj.coeff,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            grouped=False,
        )
        adjoint = same_direction_adjoint(
            residual,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            grouped=False,
        )
        return forward, adjoint

    def grouped_pair():
        forward = same_direction_forward(
            obj.coeff,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            grouped=True,
        )
        adjoint = same_direction_adjoint(
            residual,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            grouped=True,
        )
        return forward, adjoint

    (phase_forward, phase_adjoint), phase_s, phase_times = median_time(
        phase_pair,
        repeats=args.hot_repeats,
    )
    (split_forward, split_adjoint), split_s, split_times = median_time(
        split_pair,
        repeats=args.hot_repeats,
    )
    (grouped_forward, grouped_adjoint), grouped_s, grouped_times = median_time(
        grouped_pair,
        repeats=args.hot_repeats,
    )
    rows: list[dict[str, Any]] = []

    def base_row(method: str, pair_s: float, times: list[float], forward: np.ndarray, adjoint: np.ndarray, rank: int | None = None) -> dict[str, Any]:
        return {
            "status": "ok",
            "method": method,
            "rank": rank,
            "n_mag": n_mag,
            "n_beta": args.n_beta,
            "n_r": args.n_r,
            "n_z": args.n_z,
            "k": args.k,
            "detector_na": args.detector_na,
            "min_illumination_na": float(magnitudes.min()),
            "max_illumination_na": float(magnitudes.max()),
            "direction_phi": args.direction_phi,
            "cap_radial": args.cap_radial,
            "cap_phi": args.cap_phi,
            "flat_q_samples": decomp.flat_q.count,
            "axis_q_samples": decomp.base_q.count,
            "l_cutoff": l_cutoff,
            "l_modes": int(decomp.l_values.size),
            "flat_h_cutoff": flat_h_cutoff,
            "axis_h_cutoff": axis_h_cutoff,
            "axis_used_modes": axis_plan.used_modes,
            "structured_backend": backend,
            "phase_backend": args.phase_backend,
            "cpp_threads": args.cpp_threads,
            "decomp_build_s": decomp_build_s,
            "exact_phase_build_s": exact_build_s,
            "svd_build_s": svd_build_s,
            "exact_phase_mib": exact_factorization.phase.nbytes / (1024.0 * 1024.0),
            "decomp_factor_mib": (
                decomp.transverse_by_mag.nbytes + decomp.axial_phase_by_mag.nbytes
            )
            / (1024.0 * 1024.0),
            "pair_s": pair_s,
            "phase_pair_s": phase_s,
            "split_pair_s": split_s,
            "grouped_pair_s": grouped_s,
            "phase_over_method_speedup": speedup(phase_s, pair_s),
            "split_over_method_speedup": speedup(split_s, pair_s),
            "method_forward_l2_vs_phase": relative_l2(forward, phase_forward),
            "method_adjoint_l2_vs_phase": relative_l2(adjoint, phase_adjoint),
            "method_forward_l2_vs_grouped": relative_l2(forward, grouped_forward),
            "method_adjoint_l2_vs_grouped": relative_l2(adjoint, grouped_adjoint),
            "adjoint_dot_error": relative_complex_error(
                complex_dot(forward, residual),
                complex_dot(obj.coeff, adjoint),
            ),
            "hot_pair_times_s": " ".join(f"{item:.9g}" for item in times),
            "decomp_build_times_s": " ".join(f"{item:.9g}" for item in decomp_build_times),
            "exact_build_times_s": " ".join(f"{item:.9g}" for item in exact_build_times),
            "svd_build_times_s": " ".join(f"{item:.9g}" for item in svd_build_times),
        }

    rows.append(base_row("phase_ramp_axis_fft", phase_s, phase_times, phase_forward, phase_adjoint))
    rows.append(base_row("split_same_direction_z0", split_s, split_times, split_forward, split_adjoint))
    rows.append(base_row("grouped_same_direction_z0", grouped_s, grouped_times, grouped_forward, grouped_adjoint))

    max_rank = min(compression.singular_values.size, decomp.magnitudes.size)
    for rank in args.svd_rank_values:
        if rank <= 0 or rank > max_rank:
            continue

        def compressed_pair(rank: int = rank):
            forward = compressed_forward(
                obj.coeff,
                decomp,
                compression,
                rank=rank,
                backend=backend,
                cpp_threads=args.cpp_threads,
            )
            adjoint = compressed_adjoint(
                residual,
                decomp,
                compression,
                rank=rank,
                backend=backend,
                cpp_threads=args.cpp_threads,
            )
            return forward, adjoint

        (comp_forward, comp_adjoint), comp_s, comp_times = median_time(
            compressed_pair,
            repeats=args.hot_repeats,
        )
        row = base_row(
            f"svd_rank_{rank}",
            comp_s,
            comp_times,
            comp_forward,
            comp_adjoint,
            rank=rank,
        )
        energy = float(
            np.sum(compression.singular_values[:rank] ** 2)
            / max(np.sum(compression.singular_values**2), 1e-300)
        )
        row["svd_energy_fraction"] = energy
        rows.append(row)

    return rows


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# ODT same-direction z_perp benchmark",
        "",
        "This benchmark tests illumination vectors whose transverse component points in one fixed direction while its magnitude changes.",
        "",
        "The exact grouped path shares the same `z_perp` direction phase and evaluates all magnitudes inside one source-harmonic C++ kernel. The split path uses the same exact z0/zperp factors but calls the kernel one magnitude at a time.",
        "",
        "## Results",
        "",
        "| n_mag | method | rank | pair s | phase/method | split/method | fwd err vs phase | adj err vs phase | fwd err vs grouped | adj err vs grouped | dot err |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {nm} | {method} | {rank} | `{pair}` | `{ps}` | `{ss}` | `{fe}` | `{ae}` | `{feg}` | `{aeg}` | `{dot}` |".format(
                nm=row["n_mag"],
                method=row["method"],
                rank="" if row.get("rank") is None else row["rank"],
                pair=fmt(row["pair_s"], 5),
                ps=fmt(row.get("phase_over_method_speedup"), 4),
                ss=fmt(row.get("split_over_method_speedup"), 4),
                fe=fmt(row["method_forward_l2_vs_phase"], 4),
                ae=fmt(row["method_adjoint_l2_vs_phase"], 4),
                feg=fmt(row["method_forward_l2_vs_grouped"], 4),
                aeg=fmt(row["method_adjoint_l2_vs_grouped"], 4),
                dot=fmt(row["adjoint_dot_error"], 4),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `grouped_same_direction_z0` is the exact same-direction reuse test.",
            "- `split_same_direction_z0` is the same exact factorization without grouping magnitudes inside one C++ kernel.",
            "- `svd_rank_*` compresses the magnitude-dependent factor `J_l(k rho r) exp(i k(1-sqrt(1-rho^2)) z)` across the magnitude axis. It is approximate unless the retained rank equals `n_mag`.",
            "- With the rebuilt C++ extension, `svd_rank_*` uses fused rank-mode fold/scatter kernels; otherwise the script falls back to NumPy rank assembly plus axis-grid C++ contractions.",
            "- The current fused C++ SVD path caps automatic rank-kernel threading at 8 workers and reuses cached conjugated SVD weights in the adjoint path.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    methods = []
    for row in rows:
        method = row["method"]
        if method not in methods:
            methods.append(method)
    n_mag_values = sorted({int(row["n_mag"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=180)
    ax_time, ax_error = axes
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        subset.sort(key=lambda item: int(item["n_mag"]))
        ax_time.plot(
            [row["n_mag"] for row in subset],
            [row["pair_s"] for row in subset],
            marker="o",
            lw=1.5,
            label=method,
        )
        ax_error.plot(
            [row["n_mag"] for row in subset],
            [max(row["method_forward_l2_vs_grouped"], row["method_adjoint_l2_vs_grouped"]) for row in subset],
            marker="o",
            lw=1.2,
            label=method,
        )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("same-direction magnitudes")
        ax.set_xticks(n_mag_values)
    ax_time.set_ylabel("forward-adjoint pair time (s)")
    ax_time.set_title("A. Runtime")
    ax_error.set_yscale("log")
    ax_error.set_ylabel("max relative L2 vs grouped exact")
    ax_error.set_title("B. Approximation error")
    ax_time.legend(frameon=False, fontsize=7)
    ax_error.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark same-direction z_perp magnitude grouping for ODT."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_same_direction_zperp")
    parser.add_argument("--n-mag-values", default="8,16,32")
    parser.add_argument("--svd-rank-values", default="2,4,8,12,16")
    parser.add_argument("--min-illumination-na", type=float, default=0.02)
    parser.add_argument("--max-illumination-na", type=float, default=0.2)
    parser.add_argument("--direction-phi", type=float, default=0.0)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--n-beta", type=int, default=384)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--cap-radial", type=int, default=8)
    parser.add_argument("--cap-phi", type=int, default=32)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--l-margin", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--phase-backend", choices=["fft", "selected-dft"], default="fft")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--build-repeats", type=int, default=2)
    parser.add_argument("--hot-repeats", type=int, default=3)
    args = parser.parse_args()
    args.n_mag_values = parse_int_list(args.n_mag_values)
    args.svd_rank_values = parse_int_list(args.svd_rank_values)
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.min_illumination_na < 0.0 or args.max_illumination_na >= 1.0:
        raise ValueError("illumination NA values must be in [0, 1)")
    if args.min_illumination_na > args.max_illumination_na:
        raise ValueError("min-illumination-na must be <= max-illumination-na")
    return args


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for n_mag in args.n_mag_values:
        rows.extend(benchmark_case(args, n_mag=n_mag))
    payload = {
        "config": {
            **vars(args),
            "n_mag_values": args.n_mag_values,
            "svd_rank_values": args.svd_rank_values,
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
