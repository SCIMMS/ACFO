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

from benchmark_odt_cone_illumination import (
    ROOT,
    cone_illumination_directions,
    cone_q_samples,
    parse_float_list,
    parse_int_list,
)
from benchmark_odt_ewald_cap_operator import (
    QSamples,
    ShiftedAxisFactorization,
    StructuredOdtPlan,
    _axis_grid_adjoint_fft_compact,
    _axis_grid_forward_fft,
    build_shifted_axis_phases,
    build_structured_kernel,
    complex_dot,
    finufft_adjoint,
    finufft_forward,
    make_cylindrical_object,
    random_residual,
    recommended_h_cutoff,
    relative_complex_error,
    relative_l2,
    _cpp_odt_module,
    resolve_structured_backend,
    structured_adjoint,
    structured_adjoint_shifted_axis_fft_factored,
    structured_forward,
    structured_forward_shifted_axis_fft_factored,
)


@dataclass(frozen=True)
class ConeAxisDecomposition:
    base_q: QSamples
    flat_q: QSamples
    factorization: ShiftedAxisFactorization
    illumination_phi: np.ndarray
    l_values: np.ndarray
    transverse_coeff: np.ndarray
    psi_phase: np.ndarray
    axial_phase: np.ndarray
    source_slots: np.ndarray
    plan: StructuredOdtPlan
    active_l_offsets: np.ndarray | None = None
    active_l_indices: np.ndarray | None = None
    active_l_threshold: float | None = None


def build_radial_l_pruning(
    transverse_coeff: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if threshold <= 0.0:
        raise ValueError("adaptive l threshold must be positive")
    transverse_abs = np.abs(transverse_coeff)
    nl, nr = transverse_abs.shape
    offsets = np.empty(nr + 1, dtype=np.int64)
    indices: list[np.ndarray] = []
    offsets[0] = 0
    for r_index in range(nr):
        active = np.flatnonzero(transverse_abs[:, r_index] > threshold).astype(np.int64)
        if active.size == 0:
            active = np.array([int(np.argmax(transverse_abs[:, r_index]))], dtype=np.int64)
        indices.append(active)
        offsets[r_index + 1] = offsets[r_index] + active.size
    if indices:
        flat_indices = np.ascontiguousarray(np.concatenate(indices).astype(np.int64))
    else:
        flat_indices = np.empty(0, dtype=np.int64)
    if flat_indices.size > 0 and (flat_indices.min() < 0 or flat_indices.max() >= nl):
        raise ValueError("adaptive l pruning produced invalid l indices")
    return np.ascontiguousarray(offsets), flat_indices


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


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


def kernel_mib(kernel: Any) -> float:
    return (kernel.radial.nbytes + kernel.axial.nbytes + kernel.angular.nbytes) / (
        1024.0 * 1024.0
    )


def default_l_cutoff(*, k: float, illumination_na: float, r_max: float, margin: int, n_beta: int) -> int:
    estimate = int(math.ceil(float(k) * float(illumination_na) * float(r_max) + int(margin)))
    return max(0, min(n_beta // 2 - 1, estimate))


def build_exact_cone_factorization(
    plan: StructuredOdtPlan,
    *,
    k: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    illumination_na: float,
    n_illum: int,
) -> ShiftedAxisFactorization:
    illumination, _ = cone_illumination_directions(
        n_illum=n_illum,
        illumination_na=illumination_na,
    )
    _, base_q = cone_q_samples(
        k=k,
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
        illumination=illumination,
    )
    return ShiftedAxisFactorization(
        base_q=base_q,
        illumination=illumination,
        phase=build_shifted_axis_phases(plan, k=k, illumination=illumination),
        beta_twiddle=np.ascontiguousarray(
            np.exp(1j * plan.h_values[:, None] * plan.beta_axis[None, :])
        ),
        kernel=build_structured_kernel(plan, base_q),
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )


def build_cone_axis_decomposition(
    plan: StructuredOdtPlan,
    *,
    k: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    illumination_na: float,
    n_illum: int,
    l_cutoff: int,
    adaptive_l_threshold: float | None = None,
) -> ConeAxisDecomposition:
    illumination, illumination_phi = cone_illumination_directions(
        n_illum=n_illum,
        illumination_na=illumination_na,
    )
    flat_q, base_q = cone_q_samples(
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
    arg = float(k) * float(illumination_na) * plan.r_axis[None, :]
    transverse = ((-1j) ** l_values)[:, None] * special.jv(l_values[:, None], arg)
    active_l_offsets = None
    active_l_indices = None
    psi_phase = np.exp(-1j * illumination_phi[:, None] * l_values[None, :])
    if adaptive_l_threshold is not None and adaptive_l_threshold > 0.0:
        active_l_offsets, active_l_indices = build_radial_l_pruning(
            transverse,
            threshold=float(adaptive_l_threshold),
        )
    cos_alpha = math.sqrt(max(1.0 - float(illumination_na) ** 2, 0.0))
    axial_phase = np.exp(1j * float(k) * (1.0 - cos_alpha) * plan.z_axis)
    factorization = ShiftedAxisFactorization(
        base_q=base_q,
        illumination=illumination,
        phase=np.empty((0, 0, 0, 0), dtype=np.complex128),
        beta_twiddle=np.empty((0, 0), dtype=np.complex128),
        kernel=build_structured_kernel(plan, base_q),
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    return ConeAxisDecomposition(
        base_q=base_q,
        flat_q=flat_q,
        factorization=factorization,
        illumination_phi=np.ascontiguousarray(illumination_phi),
        l_values=np.ascontiguousarray(l_values),
        transverse_coeff=np.ascontiguousarray(transverse),
        psi_phase=np.ascontiguousarray(psi_phase),
        axial_phase=np.ascontiguousarray(axial_phase),
        source_slots=np.ascontiguousarray(source_slots),
        plan=plan,
        active_l_offsets=active_l_offsets,
        active_l_indices=active_l_indices,
        active_l_threshold=(
            None
            if adaptive_l_threshold is None or adaptive_l_threshold <= 0.0
            else float(adaptive_l_threshold)
        ),
    )


def decompose_coeff_h(
    coeff: np.ndarray,
    decomp: ConeAxisDecomposition,
    *,
    backend: str = "numpy",
    cpp_threads: int = 0,
) -> np.ndarray:
    plan = decomp.plan
    if coeff.shape != (plan.r_axis.size, plan.z_axis.size, plan.n_beta):
        raise ValueError("coefficient shape does not match plan")
    coeff_h_full = np.fft.ifft(coeff, axis=2) * float(plan.n_beta)
    cpp_odt = _cpp_odt_module(required=backend == "cpp")
    if cpp_odt is not None and hasattr(cpp_odt, "cone_axis_decompose_forward"):
        return cpp_odt.cone_axis_decompose_forward(
            np.ascontiguousarray(coeff_h_full),
            decomp.transverse_coeff,
            decomp.psi_phase,
            decomp.axial_phase,
            decomp.source_slots,
            int(cpp_threads),
        )
    if backend == "cpp":
        raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_decompose_forward")
    coeff_sources = coeff_h_full[:, :, decomp.source_slots]
    return np.ascontiguousarray(
        np.einsum(
            "rzhl,lr,il,z->irzh",
            coeff_sources,
            decomp.transverse_coeff,
            decomp.psi_phase,
            decomp.axial_phase,
            optimize=True,
        )
    )


def decomposed_forward(
    coeff: np.ndarray,
    decomp: ConeAxisDecomposition,
    *,
    backend: str,
    cpp_threads: int,
    forward_mode: str = "auto",
) -> np.ndarray:
    if forward_mode not in {"auto", "two-step", "fused"}:
        raise ValueError("forward_mode must be auto, two-step, or fused")
    use_fused = forward_mode in {"auto", "fused"}
    if use_fused:
        effective_backend = resolve_structured_backend(backend)
        cpp_odt = _cpp_odt_module(required=forward_mode == "fused")
        if (
            effective_backend == "cpp"
            and cpp_odt is not None
            and hasattr(cpp_odt, "cone_axis_forward_fold")
        ):
            plan = decomp.plan
            cap_phi = decomp.factorization.cap_phi
            cap_radial = decomp.factorization.cap_radial
            if decomp.factorization.base_q.count != cap_radial * cap_phi:
                raise ValueError("base q sample count does not match cap_radial * cap_phi")
            coeff_h_full = np.fft.ifft(coeff, axis=2) * float(plan.n_beta)
            radial = decomp.factorization.kernel.radial[:, ::cap_phi, :]
            axial = decomp.factorization.kernel.axial[::cap_phi, :]
            mode_phase = decomp.factorization.kernel.angular[:, 0]
            slots = np.mod(plan.h_values, cap_phi).astype(np.int64)
            if decomp.active_l_offsets is not None and decomp.active_l_indices is not None:
                if not hasattr(cpp_odt, "cone_axis_forward_fold_pruned"):
                    raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_forward_fold_pruned")
                folded = cpp_odt.cone_axis_forward_fold_pruned(
                    np.ascontiguousarray(coeff_h_full),
                    np.ascontiguousarray(radial),
                    np.ascontiguousarray(axial),
                    np.ascontiguousarray(mode_phase),
                    np.ascontiguousarray(slots),
                    decomp.transverse_coeff,
                    decomp.psi_phase,
                    decomp.axial_phase,
                    decomp.source_slots,
                    decomp.active_l_offsets,
                    decomp.active_l_indices,
                    int(cap_phi),
                    int(cpp_threads),
                )
            else:
                folded = cpp_odt.cone_axis_forward_fold(
                    np.ascontiguousarray(coeff_h_full),
                    np.ascontiguousarray(radial),
                    np.ascontiguousarray(axial),
                    np.ascontiguousarray(mode_phase),
                    np.ascontiguousarray(slots),
                    decomp.transverse_coeff,
                    decomp.psi_phase,
                    decomp.axial_phase,
                    decomp.source_slots,
                    int(cap_phi),
                    int(cpp_threads),
                )
            return np.fft.fft(folded, axis=2).reshape(
                decomp.illumination_phi.size * cap_radial * cap_phi
            )
        if forward_mode == "fused":
            raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_forward_fold")

    coeff_h_all = decompose_coeff_h(
        coeff,
        decomp,
        backend=backend,
        cpp_threads=cpp_threads,
    )
    return _axis_grid_forward_fft(
        decomp.plan,
        coeff_h_all,
        decomp.factorization,
        backend=backend,
        cpp_threads=cpp_threads,
    )


def decomposed_adjoint(
    residual: np.ndarray,
    decomp: ConeAxisDecomposition,
    *,
    backend: str,
    cpp_threads: int,
    adjoint_mode: str = "auto",
) -> np.ndarray:
    if adjoint_mode not in {"auto", "two-step", "fused"}:
        raise ValueError("adjoint_mode must be auto, two-step, or fused")
    residual = np.asarray(residual, dtype=np.complex128)
    expected = decomp.illumination_phi.size * decomp.base_q.count
    if residual.shape != (expected,):
        raise ValueError("residual shape does not match cone stack")
    use_fused = adjoint_mode in {"auto", "fused"}
    if use_fused:
        effective_backend = resolve_structured_backend(backend)
        cpp_odt = _cpp_odt_module(required=adjoint_mode == "fused")
        if (
            effective_backend == "cpp"
            and cpp_odt is not None
            and hasattr(cpp_odt, "cone_axis_adjoint_unfold_scatter")
        ):
            plan = decomp.plan
            cap_phi = decomp.factorization.cap_phi
            cap_radial = decomp.factorization.cap_radial
            if decomp.factorization.base_q.count != cap_radial * cap_phi:
                raise ValueError("base q sample count does not match cap_radial * cap_phi")
            residual_grid = residual.reshape(
                decomp.illumination_phi.size,
                cap_radial,
                cap_phi,
            )
            residual_modes = np.fft.ifft(residual_grid, axis=2) * float(cap_phi)
            radial = decomp.factorization.kernel.radial[:, ::cap_phi, :]
            axial = decomp.factorization.kernel.axial[::cap_phi, :]
            mode_phase = decomp.factorization.kernel.angular[:, 0]
            slots = np.mod(plan.h_values, cap_phi).astype(np.int64)
            if decomp.active_l_offsets is not None and decomp.active_l_indices is not None:
                if not hasattr(cpp_odt, "cone_axis_adjoint_unfold_scatter_pruned"):
                    raise RuntimeError(
                        "waxs_cake._cpp_odt lacks cone_axis_adjoint_unfold_scatter_pruned"
                    )
                out_h = cpp_odt.cone_axis_adjoint_unfold_scatter_pruned(
                    np.ascontiguousarray(residual_modes),
                    np.ascontiguousarray(radial),
                    np.ascontiguousarray(axial),
                    np.ascontiguousarray(mode_phase),
                    np.ascontiguousarray(slots),
                    decomp.transverse_coeff,
                    decomp.psi_phase,
                    decomp.axial_phase,
                    decomp.source_slots,
                    decomp.active_l_offsets,
                    decomp.active_l_indices,
                    int(plan.n_beta),
                    int(cpp_threads),
                )
            else:
                out_h = cpp_odt.cone_axis_adjoint_unfold_scatter(
                    np.ascontiguousarray(residual_modes),
                    np.ascontiguousarray(radial),
                    np.ascontiguousarray(axial),
                    np.ascontiguousarray(mode_phase),
                    np.ascontiguousarray(slots),
                    decomp.transverse_coeff,
                    decomp.psi_phase,
                    decomp.axial_phase,
                    decomp.source_slots,
                    int(plan.n_beta),
                    int(cpp_threads),
                )
            return np.fft.fft(out_h, axis=2)
        if adjoint_mode == "fused":
            raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_adjoint_unfold_scatter")

    compact = _axis_grid_adjoint_fft_compact(
        decomp.plan,
        decomp.factorization,
        residual,
        backend=backend,
        cpp_threads=cpp_threads,
    )
    plan = decomp.plan
    cpp_odt = _cpp_odt_module(required=backend == "cpp")
    if cpp_odt is not None and hasattr(cpp_odt, "cone_axis_decompose_adjoint"):
        out_h = cpp_odt.cone_axis_decompose_adjoint(
            np.ascontiguousarray(compact),
            decomp.transverse_coeff,
            decomp.psi_phase,
            decomp.axial_phase,
            decomp.source_slots,
            int(plan.n_beta),
            int(cpp_threads),
        )
        return np.fft.fft(out_h, axis=2)
    if backend == "cpp":
        raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_decompose_adjoint")
    out_h = np.zeros((plan.r_axis.size, plan.z_axis.size, plan.n_beta), dtype=np.complex128)
    contributions = np.einsum(
        "irzh,lr,il,z->rzhl",
        compact,
        np.conj(decomp.transverse_coeff),
        np.conj(decomp.psi_phase),
        np.conj(decomp.axial_phase),
        optimize=True,
    )
    for h_index in range(decomp.source_slots.shape[0]):
        for l_index in range(decomp.source_slots.shape[1]):
            out_h[:, :, decomp.source_slots[h_index, l_index]] += contributions[
                :, :, h_index, l_index
            ]
    return np.fft.fft(out_h, axis=2)


def benchmark_case(
    args: argparse.Namespace,
    *,
    n_illum: int,
    illumination_na: float,
    l_margin: int,
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
    illumination, _ = cone_illumination_directions(
        n_illum=n_illum,
        illumination_na=illumination_na,
    )
    flat_q, base_q = cone_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        illumination=illumination,
    )
    flat_h_cutoff = (
        recommended_h_cutoff(flat_q, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    axis_h_cutoff = (
        recommended_h_cutoff(base_q, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    flat_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=flat_h_cutoff,
    )
    axis_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=axis_h_cutoff,
    )
    l_cutoff = default_l_cutoff(
        k=args.k,
        illumination_na=illumination_na,
        r_max=args.r_max,
        margin=l_margin,
        n_beta=args.n_beta,
    )
    backend = resolve_structured_backend(args.structured_backend)
    residual = random_residual(flat_q, seed=args.seed + 7919 + n_illum + l_margin)

    flat_kernel, flat_build_s, flat_build_times = median_time(
        lambda: build_structured_kernel(flat_plan, flat_q),
        repeats=args.build_repeats,
    )
    exact_factorization, exact_build_s, exact_build_times = median_time(
        lambda: build_exact_cone_factorization(
            axis_plan,
            k=args.k,
            detector_na=args.detector_na,
            cap_radial=args.cap_radial,
            cap_phi=args.cap_phi,
            illumination_na=illumination_na,
            n_illum=n_illum,
        ),
        repeats=args.build_repeats,
    )
    decomp, decomp_build_s, decomp_build_times = median_time(
        lambda: build_cone_axis_decomposition(
            axis_plan,
            k=args.k,
            detector_na=args.detector_na,
            cap_radial=args.cap_radial,
            cap_phi=args.cap_phi,
            illumination_na=illumination_na,
            n_illum=n_illum,
            l_cutoff=l_cutoff,
            adaptive_l_threshold=args.cone_l_prune_threshold,
        ),
        repeats=args.build_repeats,
    )

    def flat_pair():
        forward = structured_forward(
            flat_plan,
            obj.coeff,
            flat_q,
            kernel=flat_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        adjoint = structured_adjoint(
            flat_plan,
            flat_q,
            residual,
            kernel=flat_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        return forward, adjoint

    def exact_pair():
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

    def decomp_pair():
        forward = decomposed_forward(
            obj.coeff,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            forward_mode=args.cone_forward_mode,
        )
        adjoint = decomposed_adjoint(
            residual,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            adjoint_mode=args.cone_adjoint_mode,
        )
        return forward, adjoint

    (flat_forward, flat_adjoint), flat_hot_s, flat_hot_times = median_time(
        flat_pair,
        repeats=args.hot_repeats,
    )
    (exact_forward, exact_adjoint), exact_hot_s, exact_hot_times = median_time(
        exact_pair,
        repeats=args.hot_repeats,
    )
    (decomp_forward, decomp_adjoint), decomp_hot_s, decomp_hot_times = median_time(
        decomp_pair,
        repeats=args.hot_repeats,
    )
    finufft_forward_value = None
    finufft_adjoint_value = None
    finufft_pair_s = None
    finufft_pair_times: list[float] = []
    finufft_skip_reason = None
    if not args.skip_finufft:
        try:
            def finufft_pair():
                forward = finufft_forward(obj, flat_q, eps=args.finufft_eps)
                adjoint = finufft_adjoint(obj, flat_q, residual, eps=args.finufft_eps)
                return forward, adjoint

            (finufft_forward_value, finufft_adjoint_value), finufft_pair_s, finufft_pair_times = (
                median_time(finufft_pair, repeats=args.hot_repeats)
            )
        except Exception as exc:  # pragma: no cover - optional dependency/runtime path
            finufft_skip_reason = str(exc)
    dot_error = relative_complex_error(
        complex_dot(decomp_forward, residual),
        complex_dot(obj.coeff, decomp_adjoint),
    )
    return {
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
        "n_illum": n_illum,
        "cap_radial": args.cap_radial,
        "cap_phi": args.cap_phi,
        "flat_q_samples": flat_q.count,
        "axis_q_samples": base_q.count,
        "l_margin": l_margin,
        "l_cutoff": l_cutoff,
        "l_modes": int(2 * l_cutoff + 1),
        "flat_h_cutoff": flat_h_cutoff,
        "axis_h_cutoff": axis_h_cutoff,
        "flat_used_modes": flat_plan.used_modes,
        "axis_used_modes": axis_plan.used_modes,
        "structured_backend": backend,
        "phase_backend": args.phase_backend,
        "cone_forward_mode": args.cone_forward_mode,
        "cone_adjoint_mode": args.cone_adjoint_mode,
        "cone_l_prune_threshold": args.cone_l_prune_threshold,
        "cone_l_active_total": (
            None if decomp.active_l_indices is None else int(decomp.active_l_indices.size)
        ),
        "cone_l_active_fraction": (
            None
            if decomp.active_l_indices is None
            else float(decomp.active_l_indices.size)
            / float(decomp.transverse_coeff.shape[0] * decomp.transverse_coeff.shape[1])
        ),
        "cone_l_active_mean": (
            None
            if decomp.active_l_indices is None
            else float(decomp.active_l_indices.size) / float(decomp.transverse_coeff.shape[1])
        ),
        "cpp_threads": args.cpp_threads,
        "flat_build_s": flat_build_s,
        "exact_build_s": exact_build_s,
        "decomp_build_s": decomp_build_s,
        "flat_kernel_mib": kernel_mib(flat_kernel),
        "axis_kernel_mib": kernel_mib(exact_factorization.kernel),
        "decomp_transverse_mib": decomp.transverse_coeff.nbytes / (1024.0 * 1024.0),
        "decomp_source_slots_mib": decomp.source_slots.nbytes / (1024.0 * 1024.0),
        "exact_phase_mib": exact_factorization.phase.nbytes / (1024.0 * 1024.0),
        "flat_hot_pair_s": flat_hot_s,
        "exact_hot_pair_s": exact_hot_s,
        "decomp_hot_pair_s": decomp_hot_s,
        "finufft_pair_s": finufft_pair_s,
        "flat_vs_exact_hot_speedup": speedup(flat_hot_s, exact_hot_s),
        "flat_vs_decomp_hot_speedup": speedup(flat_hot_s, decomp_hot_s),
        "finufft_vs_flat_hot_speedup": speedup(finufft_pair_s, flat_hot_s),
        "finufft_vs_exact_hot_speedup": speedup(finufft_pair_s, exact_hot_s),
        "finufft_vs_decomp_hot_speedup": speedup(finufft_pair_s, decomp_hot_s),
        "exact_vs_decomp_hot_speedup": speedup(exact_hot_s, decomp_hot_s),
        "exact_forward_l2_vs_flat": relative_l2(exact_forward, flat_forward),
        "exact_adjoint_l2_vs_flat": relative_l2(exact_adjoint, flat_adjoint),
        "decomp_forward_l2_vs_exact": relative_l2(decomp_forward, exact_forward),
        "decomp_adjoint_l2_vs_exact": relative_l2(decomp_adjoint, exact_adjoint),
        "decomp_forward_l2_vs_flat": relative_l2(decomp_forward, flat_forward),
        "decomp_adjoint_l2_vs_flat": relative_l2(decomp_adjoint, flat_adjoint),
        "finufft_forward_l2_vs_flat": (
            None if finufft_forward_value is None else relative_l2(finufft_forward_value, flat_forward)
        ),
        "finufft_adjoint_l2_vs_flat": (
            None if finufft_adjoint_value is None else relative_l2(finufft_adjoint_value, flat_adjoint)
        ),
        "finufft_forward_l2_vs_decomp": (
            None
            if finufft_forward_value is None
            else relative_l2(finufft_forward_value, decomp_forward)
        ),
        "finufft_adjoint_l2_vs_decomp": (
            None
            if finufft_adjoint_value is None
            else relative_l2(finufft_adjoint_value, decomp_adjoint)
        ),
        "finufft_eps": None if args.skip_finufft else args.finufft_eps,
        "finufft_skip_reason": finufft_skip_reason,
        "decomp_adjoint_dot_error": dot_error,
        "flat_build_times_s": " ".join(f"{item:.9g}" for item in flat_build_times),
        "exact_build_times_s": " ".join(f"{item:.9g}" for item in exact_build_times),
        "decomp_build_times_s": " ".join(f"{item:.9g}" for item in decomp_build_times),
        "flat_hot_pair_times_s": " ".join(f"{item:.9g}" for item in flat_hot_times),
        "exact_hot_pair_times_s": " ".join(f"{item:.9g}" for item in exact_hot_times),
        "decomp_hot_pair_times_s": " ".join(f"{item:.9g}" for item in decomp_hot_times),
        "finufft_pair_times_s": " ".join(f"{item:.9g}" for item in finufft_pair_times),
    }


def case_label(row: dict[str, Any]) -> str:
    return f"illum={row['n_illum']} l={row['l_cutoff']}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# ODT cone-axis z0/zperp decomposition benchmark",
        "",
        "This benchmark tests the exact illumination phase split",
        "",
        "`exp(i k((0,0,1)-s_in) dot r) = exp(i k(1-cos(alpha)) z) * exp(-i k sin(alpha) rho cos(beta-psi))`.",
        "",
        "The transverse factor is evaluated with a Jacobi-Anger/Bessel harmonic expansion, preserving the full sequential illumination-angle stack rather than truncating the source CSD rank.",
        "",
        "## Results",
        "",
        "| case | flat q | axis q | l modes | FINUFFT pair s | flat hot s | phase-ramp hot s | z0/zperp hot s | FINUFFT/z0 speedup | exact/z0 speedup | z0 fwd err | z0 adj err | FINUFFT fwd err | FINUFFT adj err |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {fq} | {aq} | {lm} | `{nu}` | `{fh}` | `{eh}` | `{dh}` | `{nds}` | `{eds}` | `{fe}` | `{ae}` | `{nfe}` | `{nae}` |".format(
                case=case_label(row),
                fq=row["flat_q_samples"],
                aq=row["axis_q_samples"],
                lm=row["l_modes"],
                nu=fmt(row.get("finufft_pair_s"), 5),
                fh=fmt(row["flat_hot_pair_s"], 5),
                eh=fmt(row["exact_hot_pair_s"], 5),
                dh=fmt(row["decomp_hot_pair_s"], 5),
                nds=fmt(row.get("finufft_vs_decomp_hot_speedup"), 4),
                eds=fmt(row["exact_vs_decomp_hot_speedup"], 4),
                fe=fmt(row["decomp_forward_l2_vs_exact"], 4),
                ae=fmt(row["decomp_adjoint_l2_vs_exact"], 4),
                nfe=fmt(row.get("finufft_forward_l2_vs_flat"), 4),
                nae=fmt(row.get("finufft_adjoint_l2_vs_flat"), 4),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a full-rank sequential-angle test: no CMD/source-rank truncation is used.",
            "- Accuracy is controlled by the transverse Bessel cutoff `l_cutoff ~= k * illumination_na * R + margin`.",
            "- FINUFFT is evaluated on the same flat shifted q-list as a type-3 forward plus adjoint pair; its reported time includes the standard Python `nufft3d3` calls.",
            "- The C++ source-harmonic path reuses the shared cone axial factor `exp(i k(1-cos(alpha)) z)` and avoids the Python `einsum`/scatter hot path.",
            "- In the current sweep, accurate `l_cutoff >= 15` cases are faster than the exact per-angle phase-ramp path while retaining numerical agreement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=180)
    ax_speed, ax_err = axes
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("transverse l cutoff")
    for n_illum in sorted({int(row["n_illum"]) for row in rows}):
        subset = [row for row in rows if int(row["n_illum"]) == n_illum]
        subset.sort(key=lambda row: row["l_cutoff"])
        x = [row["l_cutoff"] for row in subset]
        ax_speed.plot(
            x,
            [row["exact_vs_decomp_hot_speedup"] for row in subset],
            marker="o",
            lw=1.7,
            label=f"phase-ramp/z0 illum={n_illum}",
        )
        if any(row.get("finufft_vs_decomp_hot_speedup") is not None for row in subset):
            ax_speed.plot(
                x,
                [row.get("finufft_vs_decomp_hot_speedup") for row in subset],
                marker="^",
                lw=1.2,
                ls=":",
                label=f"FINUFFT/z0 illum={n_illum}",
            )
        ax_err.plot(
            x,
            [row["decomp_forward_l2_vs_exact"] for row in subset],
            marker="o",
            lw=1.7,
            label=f"fwd illum={n_illum}",
        )
        ax_err.plot(
            x,
            [row["decomp_adjoint_l2_vs_exact"] for row in subset],
            marker="s",
            lw=1.2,
            ls="--",
            label=f"adj illum={n_illum}",
        )
    ax_speed.axhline(1.0, color="#444444", lw=1.0, ls="--", alpha=0.75)
    ax_speed.set_ylabel("baseline / z0-zperp speedup")
    ax_speed.set_title("A. Full-rank cone speed")
    ax_err.set_yscale("log")
    ax_err.set_ylabel("relative L2 vs phase-ramp")
    ax_err.set_title("B. Decomposition accuracy")
    ax_speed.legend(frameon=False, fontsize=8)
    ax_err.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full-rank cone illumination using z0/zperp phase decomposition."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_cone_axis_decomposition")
    parser.add_argument("--n-illum-values", default="16,32")
    parser.add_argument("--illumination-na-values", default="0.2")
    parser.add_argument("--l-margin-values", default="4,8,12,18")
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--n-beta", type=int, default=192)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--cap-radial", type=int, default=8)
    parser.add_argument("--cap-phi", type=int, default=32)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--phase-backend", choices=["fft", "selected-dft"], default="fft")
    parser.add_argument("--cone-forward-mode", choices=["auto", "two-step", "fused"], default="auto")
    parser.add_argument("--cone-adjoint-mode", choices=["auto", "two-step", "fused"], default="auto")
    parser.add_argument("--cone-l-prune-threshold", type=float, default=0.0)
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--skip-finufft", action="store_true")
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--build-repeats", type=int, default=2)
    parser.add_argument("--hot-repeats", type=int, default=3)
    args = parser.parse_args()
    args.n_illum_values = parse_int_list(args.n_illum_values)
    args.illumination_na_values = parse_float_list(args.illumination_na_values)
    args.l_margin_values = parse_int_list(args.l_margin_values)
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.cone_l_prune_threshold < 0.0:
        raise ValueError("cone-l-prune-threshold must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    rows = [
        benchmark_case(
            args,
            n_illum=n_illum,
            illumination_na=illumination_na,
            l_margin=l_margin,
        )
        for illumination_na in args.illumination_na_values
        for n_illum in args.n_illum_values
        for l_margin in args.l_margin_values
    ]
    payload = {
        "config": {
            **vars(args),
            "n_illum_values": args.n_illum_values,
            "illumination_na_values": args.illumination_na_values,
            "l_margin_values": args.l_margin_values,
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
