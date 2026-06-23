import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_high_na_debye_wolf import (
    PositiveRhoDependentBasisCache,
    PreparedCachedPositiveRhoDependentHarmonicDebyeWolfPlan,
    PreparedFinufftDebyeWolfPlan,
    PreparedPositiveRhoDependentHarmonicDebyeWolfPlan,
    direct_debye_wolf,
    evaluate_direct_sequence,
    flatten_focal_axes,
    focal_axes,
    gauss_theta_grid,
    max_abs_over_ref,
    median_time,
    pupil_field,
    relative_l2,
    sweep_strengths,
)


FIELDNAMES = [
    "stage",
    "workload",
    "case",
    "variant",
    "mode_count",
    "ntheta",
    "nphi",
    "nrho",
    "npsi",
    "nz",
    "targets",
    "rho_max",
    "z_max",
    "na",
    "h_cutoff",
    "margin",
    "cutoff_bin_size",
    "groups",
    "mean_used_modes",
    "mode_rho_work",
    "build_s",
    "cache_build_s",
    "plan_build_s",
    "hot_s",
    "total_s",
    "amortized_total_s",
    "basis_mib",
    "coordinate_mib",
    "field_l2_vs_direct",
    "field_max_abs_vs_direct",
    "intensity_l2_vs_direct",
    "field_l2_vs_finufft",
    "field_max_abs_vs_finufft",
    "field_l2_vs_compact",
    "speedup_hot_vs_finufft",
    "speedup_total_vs_finufft",
    "speedup_total_vs_compact",
    "repeats",
    "finufft_eps",
    "notes",
]


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if any(item <= 0 for item in values):
        raise ValueError("integer list values must be positive")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDNAMES})


def write_payload(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"config": config, "rows": rows}, indent=2),
        encoding="utf-8",
    )


def speedup(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or candidate <= 0.0:
        return None
    return float(baseline / candidate)


def auto_nrho(rho_max: float, base_rho_max: float, base_nrho: int) -> int:
    if base_nrho <= 1:
        return max(1, base_nrho)
    drho = base_rho_max / float(base_nrho - 1)
    return max(2, int(round(rho_max / drho)) + 1)


def auto_h_cutoff(
    *,
    k: float,
    rho_max: float,
    sin_theta_max: float,
    margin: int,
    nphi: int,
) -> int:
    cutoff = int(np.ceil(k * rho_max * sin_theta_max + margin))
    return max(0, min(cutoff, nphi // 2))


def make_geometry(
    *,
    ntheta: int,
    nphi: int,
    nrho: int,
    npsi: int,
    nz: int,
    rho_max: float,
    z_max: float,
    na: float,
    n_medium: float,
    wavelength: float,
) -> dict[str, Any]:
    theta_max = float(np.arcsin(na / n_medium))
    theta, theta_weights = gauss_theta_grid(ntheta, theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False, dtype=float)
    rho_axis, psi_axis, z_axis = focal_axes(
        nrho=nrho,
        npsi=npsi,
        nz=nz,
        rho_max=rho_max,
        z_max=z_max,
    )
    rho_flat, psi_flat, z_flat = flatten_focal_axes(rho_axis, psi_axis, z_axis)
    return {
        "theta": theta,
        "theta_weights": theta_weights,
        "theta_max": theta_max,
        "phi": phi,
        "rho_axis": rho_axis,
        "psi_axis": psi_axis,
        "z_axis": z_axis,
        "rho_flat": rho_flat,
        "psi_flat": psi_flat,
        "z_flat": z_flat,
        "k": 2.0 * np.pi * n_medium / wavelength,
        "sin_theta_max": float(np.sin(theta_max)),
        "ntheta": ntheta,
        "nphi": nphi,
        "nrho": nrho,
        "npsi": npsi,
        "nz": nz,
        "rho_max": rho_max,
        "z_max": z_max,
        "na": na,
    }


def workload_pupil(
    case: str,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    theta_max: float,
    strength: float,
    vortex_charge: int,
    apodization: str,
) -> np.ndarray:
    if case == "vortex_high":
        return pupil_field(
            "vortex",
            theta,
            phi,
            theta_max=theta_max,
            strength=strength,
            vortex_charge=vortex_charge,
            apodization=apodization,
        )
    if case == "mixed_high":
        theta_2d = theta[:, None]
        phi_2d = phi[None, :]
        radial = np.sin(theta_2d) / max(np.sin(theta_max), np.finfo(float).eps)
        phase = strength * (
            0.35 * radial**2 * np.cos(2.0 * phi_2d)
            + 0.25 * radial**4 * np.sin(5.0 * phi_2d)
            + 0.20 * radial**5 * np.cos(11.0 * phi_2d)
            + 0.15 * radial**6 * np.sin(17.0 * phi_2d)
        )
        amplitude = 1.0 + 0.08 * radial * np.cos(7.0 * phi_2d)
        pupil = amplitude * np.exp(1j * phase)
        if apodization == "none":
            return pupil
        if apodization == "sqrt-cos":
            return pupil * np.sqrt(np.cos(theta_2d))
        raise ValueError(f"unknown apodization: {apodization}")
    return pupil_field(
        case,
        theta,
        phi,
        theta_max=theta_max,
        strength=strength,
        vortex_charge=vortex_charge,
        apodization=apodization,
    )


def make_pupils(
    geom: dict[str, Any],
    *,
    case: str,
    mode_count: int,
    strength: float,
    vortex_charge: int,
    apodization: str,
) -> list[np.ndarray]:
    strengths = sweep_strengths(strength, mode_count)
    pupils = []
    for idx, value in enumerate(strengths):
        charge = vortex_charge
        if case == "vortex_high":
            charge = vortex_charge + (idx % 3) - 1
        pupils.append(
            workload_pupil(
                case,
                geom["theta"],
                geom["phi"],
                theta_max=geom["theta_max"],
                strength=value,
                vortex_charge=charge,
                apodization=apodization,
            )
        )
    return pupils


def build_compact_plan(
    geom: dict[str, Any],
    *,
    h_cutoff: int,
    margin: int,
    cutoff_bin_size: int,
    cpp_threads: int,
) -> PreparedPositiveRhoDependentHarmonicDebyeWolfPlan:
    return PreparedPositiveRhoDependentHarmonicDebyeWolfPlan.build(
        geom["nphi"],
        geom["theta"],
        geom["theta_weights"],
        geom["rho_axis"],
        geom["psi_axis"],
        geom["z_axis"],
        k=geom["k"],
        h_cutoff=h_cutoff,
        margin=margin,
        cutoff_bin_size=cutoff_bin_size,
        sin_theta_max=geom["sin_theta_max"],
        backend="cpp",
        cpp_threads=cpp_threads,
        no_copy_coefficients=True,
    )


def build_basis_cache(
    geom: dict[str, Any],
    *,
    h_cutoff: int,
) -> PositiveRhoDependentBasisCache:
    return PositiveRhoDependentBasisCache.build(
        geom["nphi"],
        geom["theta"],
        geom["theta_weights"],
        geom["rho_axis"],
        geom["psi_axis"],
        geom["z_axis"],
        k=geom["k"],
        h_cutoff=h_cutoff,
        sin_theta_max=geom["sin_theta_max"],
    )


def build_cached_plan(
    cache: PositiveRhoDependentBasisCache,
    *,
    margin: int,
    cutoff_bin_size: int,
    cpp_threads: int,
) -> PreparedCachedPositiveRhoDependentHarmonicDebyeWolfPlan:
    return PreparedCachedPositiveRhoDependentHarmonicDebyeWolfPlan.build(
        cache,
        margin=margin,
        cutoff_bin_size=cutoff_bin_size,
        backend="cpp",
        cpp_threads=cpp_threads,
    )


def build_finufft_plan(geom: dict[str, Any]) -> PreparedFinufftDebyeWolfPlan:
    return PreparedFinufftDebyeWolfPlan.build(
        geom["theta"],
        geom["theta_weights"],
        geom["phi"],
        geom["rho_flat"],
        geom["psi_flat"],
        geom["z_flat"],
        k=geom["k"],
    )


def base_row(
    *,
    stage: str,
    workload: str,
    case: str,
    variant: str,
    mode_count: int,
    geom: dict[str, Any],
    h_cutoff: int,
    margin: int,
    cutoff_bin_size: int | str | None,
    repeats: int,
    finufft_eps: float,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "stage": stage,
        "workload": workload,
        "case": case,
        "variant": variant,
        "mode_count": mode_count,
        "ntheta": geom["ntheta"],
        "nphi": geom["nphi"],
        "nrho": geom["nrho"],
        "npsi": geom["npsi"],
        "nz": geom["nz"],
        "targets": int(geom["rho_flat"].size),
        "rho_max": geom["rho_max"],
        "z_max": geom["z_max"],
        "na": geom["na"],
        "h_cutoff": h_cutoff,
        "margin": margin,
        "cutoff_bin_size": cutoff_bin_size,
        "repeats": repeats,
        "finufft_eps": finufft_eps,
        "notes": notes,
    }


def add_error_metrics(
    row: dict[str, Any],
    result: np.ndarray,
    *,
    direct_ref: np.ndarray | None = None,
    finufft_ref: np.ndarray | None = None,
    compact_ref: np.ndarray | None = None,
) -> None:
    if direct_ref is not None:
        row["field_l2_vs_direct"] = relative_l2(result, direct_ref)
        row["field_max_abs_vs_direct"] = max_abs_over_ref(result, direct_ref)
        row["intensity_l2_vs_direct"] = relative_l2(
            np.abs(result) ** 2,
            np.abs(direct_ref) ** 2,
        )
    if finufft_ref is not None:
        row["field_l2_vs_finufft"] = relative_l2(result, finufft_ref)
        row["field_max_abs_vs_finufft"] = max_abs_over_ref(result, finufft_ref)
    if compact_ref is not None:
        row["field_l2_vs_compact"] = relative_l2(result, compact_ref)


def add_plan_metrics(row: dict[str, Any], plan: Any) -> None:
    row["groups"] = getattr(plan, "group_count", None)
    row["mean_used_modes"] = getattr(plan, "mean_used_modes", None)
    row["mode_rho_work"] = getattr(plan, "mode_rho_work", None)
    row["basis_mib"] = getattr(plan, "basis_mib", None)


def benchmark_correctness(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    geom = make_geometry(
        ntheta=24,
        nphi=96,
        nrho=5,
        npsi=16,
        nz=3,
        rho_max=0.75,
        z_max=0.5,
        na=args.na,
        n_medium=args.n_medium,
        wavelength=args.wavelength,
    )
    margin = 8
    h_cutoff = auto_h_cutoff(
        k=geom["k"],
        rho_max=geom["rho_max"],
        sin_theta_max=geom["sin_theta_max"],
        margin=margin,
        nphi=geom["nphi"],
    )
    cases = ["mixed", "vortex_high"]
    for case in cases:
        pupils = make_pupils(
            geom,
            case=case,
            mode_count=4,
            strength=args.strength,
            vortex_charge=18,
            apodization=args.apodization,
        )
        direct_ref, direct_s, _ = median_time(
            lambda: evaluate_direct_sequence(
                pupils,
                geom["theta"],
                geom["theta_weights"],
                geom["phi"],
                geom["rho_flat"],
                geom["psi_flat"],
                geom["z_flat"],
                k=geom["k"],
            ),
            args.repeats,
        )
        row = base_row(
            stage="correctness",
            workload="small_direct_reference",
            case=case,
            variant="direct_debye_wolf",
            mode_count=len(pupils),
            geom=geom,
            h_cutoff=h_cutoff,
            margin=margin,
            cutoff_bin_size=None,
            repeats=args.repeats,
            finufft_eps=args.finufft_eps,
        )
        row.update({"hot_s": direct_s, "total_s": direct_s})
        add_error_metrics(row, direct_ref, direct_ref=direct_ref)
        rows.append(row)

        finufft_plan, finufft_build_s, _ = median_time(
            lambda: build_finufft_plan(geom),
            args.repeats,
        )
        finufft_ref, finufft_hot_s, _ = median_time(
            lambda: finufft_plan.evaluate_many(pupils, eps=args.finufft_eps),
            args.repeats,
        )
        row = base_row(
            stage="correctness",
            workload="small_direct_reference",
            case=case,
            variant="finufft",
            mode_count=len(pupils),
            geom=geom,
            h_cutoff=h_cutoff,
            margin=margin,
            cutoff_bin_size=None,
            repeats=args.repeats,
            finufft_eps=args.finufft_eps,
        )
        row.update(
            {
                "build_s": finufft_build_s,
                "plan_build_s": finufft_build_s,
                "hot_s": finufft_hot_s,
                "total_s": finufft_build_s + finufft_hot_s,
                "coordinate_mib": finufft_plan.coordinate_mib,
            }
        )
        add_error_metrics(row, finufft_ref, direct_ref=direct_ref)
        rows.append(row)

        for cutoff_bin_size in [4, 8]:
            compact_plan, compact_build_s, _ = median_time(
                lambda: build_compact_plan(
                    geom,
                    h_cutoff=h_cutoff,
                    margin=margin,
                    cutoff_bin_size=cutoff_bin_size,
                    cpp_threads=args.cpp_threads,
                ),
                args.repeats,
            )
            compact_result, compact_hot_s, _ = median_time(
                lambda: compact_plan.evaluate_many(pupils),
                args.repeats,
            )
            row = base_row(
                stage="correctness",
                workload="small_direct_reference",
                case=case,
                variant="compact_positive_rho_nocopy",
                mode_count=len(pupils),
                geom=geom,
                h_cutoff=h_cutoff,
                margin=margin,
                cutoff_bin_size=cutoff_bin_size,
                repeats=args.repeats,
                finufft_eps=args.finufft_eps,
            )
            row.update(
                {
                    "build_s": compact_build_s,
                    "plan_build_s": compact_build_s,
                    "hot_s": compact_hot_s,
                    "total_s": compact_build_s + compact_hot_s,
                    "speedup_hot_vs_finufft": speedup(finufft_hot_s, compact_hot_s),
                    "speedup_total_vs_finufft": speedup(
                        finufft_build_s + finufft_hot_s,
                        compact_build_s + compact_hot_s,
                    ),
                }
            )
            add_plan_metrics(row, compact_plan)
            add_error_metrics(
                row,
                compact_result,
                direct_ref=direct_ref,
                finufft_ref=finufft_ref,
            )
            rows.append(row)

            cache, cache_build_s, _ = median_time(
                lambda: build_basis_cache(geom, h_cutoff=h_cutoff),
                args.repeats,
            )
            cached_plan, cached_plan_build_s, _ = median_time(
                lambda: build_cached_plan(
                    cache,
                    margin=margin,
                    cutoff_bin_size=cutoff_bin_size,
                    cpp_threads=args.cpp_threads,
                ),
                args.repeats,
            )
            cached_result, cached_hot_s, _ = median_time(
                lambda: cached_plan.evaluate_many(pupils),
                args.repeats,
            )
            row = base_row(
                stage="correctness",
                workload="small_direct_reference",
                case=case,
                variant="cached_positive_rho_nocopy",
                mode_count=len(pupils),
                geom=geom,
                h_cutoff=h_cutoff,
                margin=margin,
                cutoff_bin_size=cutoff_bin_size,
                repeats=args.repeats,
                finufft_eps=args.finufft_eps,
            )
            cached_total = cache_build_s + cached_plan_build_s + cached_hot_s
            row.update(
                {
                    "build_s": cache_build_s + cached_plan_build_s,
                    "cache_build_s": cache_build_s,
                    "plan_build_s": cached_plan_build_s,
                    "hot_s": cached_hot_s,
                    "total_s": cached_total,
                    "amortized_total_s": cached_plan_build_s + cached_hot_s,
                    "speedup_hot_vs_finufft": speedup(finufft_hot_s, cached_hot_s),
                    "speedup_total_vs_finufft": speedup(
                        finufft_build_s + finufft_hot_s,
                        cached_total,
                    ),
                    "speedup_total_vs_compact": speedup(
                        compact_build_s + compact_hot_s,
                        cached_total,
                    ),
                }
            )
            add_plan_metrics(row, cached_plan)
            add_error_metrics(
                row,
                cached_result,
                direct_ref=direct_ref,
                finufft_ref=finufft_ref,
                compact_ref=compact_result,
            )
            rows.append(row)
    return rows


def benchmark_geometry_workload(
    *,
    stage: str,
    workload: str,
    geom: dict[str, Any],
    case: str,
    mode_counts: list[int],
    margins: list[int],
    cutoff_bin_sizes: list[int],
    args: argparse.Namespace,
    vortex_charge: int = 18,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_margin = max(margins)
    h_cutoff = auto_h_cutoff(
        k=geom["k"],
        rho_max=geom["rho_max"],
        sin_theta_max=geom["sin_theta_max"],
        margin=max_margin,
        nphi=geom["nphi"],
    )
    finufft_plan, finufft_build_s, _ = median_time(
        lambda: build_finufft_plan(geom),
        args.repeats,
    )
    finufft_refs: dict[int, np.ndarray] = {}
    finufft_hot: dict[int, float] = {}
    for mode_count in mode_counts:
        pupils = make_pupils(
            geom,
            case=case,
            mode_count=mode_count,
            strength=args.strength,
            vortex_charge=vortex_charge,
            apodization=args.apodization,
        )
        result, hot_s, _ = median_time(
            lambda: finufft_plan.evaluate_many(pupils, eps=args.finufft_eps),
            args.repeats,
        )
        finufft_refs[mode_count] = result
        finufft_hot[mode_count] = hot_s
        row = base_row(
            stage=stage,
            workload=workload,
            case=case,
            variant="finufft",
            mode_count=mode_count,
            geom=geom,
            h_cutoff=h_cutoff,
            margin=max_margin,
            cutoff_bin_size=None,
            repeats=args.repeats,
            finufft_eps=args.finufft_eps,
        )
        row.update(
            {
                "build_s": finufft_build_s,
                "plan_build_s": finufft_build_s,
                "hot_s": hot_s,
                "total_s": finufft_build_s + hot_s,
                "coordinate_mib": finufft_plan.coordinate_mib,
            }
        )
        rows.append(row)

    cache, cache_build_s, _ = median_time(
        lambda: build_basis_cache(geom, h_cutoff=h_cutoff),
        args.repeats,
    )
    for margin in margins:
        for cutoff_bin_size in cutoff_bin_sizes:
            compact_plan, compact_build_s, _ = median_time(
                lambda: build_compact_plan(
                    geom,
                    h_cutoff=h_cutoff,
                    margin=margin,
                    cutoff_bin_size=cutoff_bin_size,
                    cpp_threads=args.cpp_threads,
                ),
                args.repeats,
            )
            cached_plan, cached_build_s, _ = median_time(
                lambda: build_cached_plan(
                    cache,
                    margin=margin,
                    cutoff_bin_size=cutoff_bin_size,
                    cpp_threads=args.cpp_threads,
                ),
                args.repeats,
            )
            for mode_count in mode_counts:
                pupils = make_pupils(
                    geom,
                    case=case,
                    mode_count=mode_count,
                    strength=args.strength,
                    vortex_charge=vortex_charge,
                    apodization=args.apodization,
                )
                compact_result, compact_hot_s, _ = median_time(
                    lambda: compact_plan.evaluate_many(pupils),
                    args.repeats,
                )
                compact_total = compact_build_s + compact_hot_s
                finufft_total = finufft_build_s + finufft_hot[mode_count]
                row = base_row(
                    stage=stage,
                    workload=workload,
                    case=case,
                    variant="compact_positive_rho_nocopy",
                    mode_count=mode_count,
                    geom=geom,
                    h_cutoff=h_cutoff,
                    margin=margin,
                    cutoff_bin_size=cutoff_bin_size,
                    repeats=args.repeats,
                    finufft_eps=args.finufft_eps,
                )
                row.update(
                    {
                        "build_s": compact_build_s,
                        "plan_build_s": compact_build_s,
                        "hot_s": compact_hot_s,
                        "total_s": compact_total,
                        "speedup_hot_vs_finufft": speedup(
                            finufft_hot[mode_count],
                            compact_hot_s,
                        ),
                        "speedup_total_vs_finufft": speedup(
                            finufft_total,
                            compact_total,
                        ),
                    }
                )
                add_plan_metrics(row, compact_plan)
                add_error_metrics(
                    row,
                    compact_result,
                    finufft_ref=finufft_refs[mode_count],
                )
                rows.append(row)

                cached_result, cached_hot_s, _ = median_time(
                    lambda: cached_plan.evaluate_many(pupils),
                    args.repeats,
                )
                cached_total = cache_build_s + cached_build_s + cached_hot_s
                row = base_row(
                    stage=stage,
                    workload=workload,
                    case=case,
                    variant="cached_positive_rho_nocopy",
                    mode_count=mode_count,
                    geom=geom,
                    h_cutoff=h_cutoff,
                    margin=margin,
                    cutoff_bin_size=cutoff_bin_size,
                    repeats=args.repeats,
                    finufft_eps=args.finufft_eps,
                )
                row.update(
                    {
                        "build_s": cache_build_s + cached_build_s,
                        "cache_build_s": cache_build_s,
                        "plan_build_s": cached_build_s,
                        "hot_s": cached_hot_s,
                        "total_s": cached_total,
                        "amortized_total_s": cached_build_s + cached_hot_s,
                        "speedup_hot_vs_finufft": speedup(
                            finufft_hot[mode_count],
                            cached_hot_s,
                        ),
                        "speedup_total_vs_finufft": speedup(
                            finufft_total,
                            cached_total,
                        ),
                        "speedup_total_vs_compact": speedup(
                            compact_total,
                            cached_total,
                        ),
                    }
                )
                add_plan_metrics(row, cached_plan)
                add_error_metrics(
                    row,
                    cached_result,
                    finufft_ref=finufft_refs[mode_count],
                    compact_ref=compact_result,
                )
                rows.append(row)
    return rows


def benchmark_representative(args: argparse.Namespace) -> list[dict[str, Any]]:
    rho_max = 3.0
    nrho = auto_nrho(rho_max, args.base_rho_max, args.base_nrho)
    geom = make_geometry(
        ntheta=48,
        nphi=128,
        nrho=nrho,
        npsi=96,
        nz=9,
        rho_max=rho_max,
        z_max=0.8,
        na=args.na,
        n_medium=args.n_medium,
        wavelength=args.wavelength,
    )
    return benchmark_geometry_workload(
        stage="representative",
        workload="rho3_grid_modes_bins",
        geom=geom,
        case="mixed",
        mode_counts=args.mode_counts,
        margins=[8],
        cutoff_bin_sizes=args.cutoff_bin_sizes,
        args=args,
    )


def benchmark_regime(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        {
            "workload": "unfavorable_small_one_shot",
            "case": "mixed",
            "rho_max": 0.75,
            "ntheta": 32,
            "nphi": 96,
            "npsi": 32,
            "nz": 3,
            "z_max": 0.5,
            "mode_counts": [1],
            "bins": [8],
            "margins": [8],
            "vortex_charge": 18,
        },
        {
            "workload": "unfavorable_high_azimuthal_mode",
            "case": "vortex_high",
            "rho_max": 1.5,
            "ntheta": 32,
            "nphi": 96,
            "npsi": 64,
            "nz": 5,
            "z_max": 0.6,
            "mode_counts": [1],
            "bins": [8],
            "margins": [8],
            "vortex_charge": 30,
        },
        {
            "workload": "favorable_many_modes",
            "case": "mixed",
            "rho_max": 3.0,
            "ntheta": 48,
            "nphi": 128,
            "npsi": 96,
            "nz": 9,
            "z_max": 0.8,
            "mode_counts": [64],
            "bins": [8],
            "margins": [8],
            "vortex_charge": 18,
        },
        {
            "workload": "favorable_large_volume",
            "case": "mixed",
            "rho_max": 6.0,
            "ntheta": 48,
            "nphi": 192,
            "npsi": 128,
            "nz": 9,
            "z_max": 0.8,
            "mode_counts": [16],
            "bins": [8],
            "margins": [8],
            "vortex_charge": 18,
        },
        {
            "workload": "pupil_complexity_clear",
            "case": "clear",
            "rho_max": 3.0,
            "ntheta": 48,
            "nphi": 128,
            "npsi": 96,
            "nz": 9,
            "z_max": 0.8,
            "mode_counts": [16],
            "bins": [8],
            "margins": [8],
            "vortex_charge": 18,
        },
        {
            "workload": "pupil_complexity_mixed_high",
            "case": "mixed_high",
            "rho_max": 3.0,
            "ntheta": 48,
            "nphi": 128,
            "npsi": 96,
            "nz": 9,
            "z_max": 0.8,
            "mode_counts": [16],
            "bins": [8],
            "margins": [8],
            "vortex_charge": 18,
        },
    ]
    for spec in specs:
        rho_max = float(spec["rho_max"])
        nrho = auto_nrho(rho_max, args.base_rho_max, args.base_nrho)
        geom = make_geometry(
            ntheta=int(spec["ntheta"]),
            nphi=int(spec["nphi"]),
            nrho=nrho,
            npsi=int(spec["npsi"]),
            nz=int(spec["nz"]),
            rho_max=rho_max,
            z_max=float(spec["z_max"]),
            na=args.na,
            n_medium=args.n_medium,
            wavelength=args.wavelength,
        )
        rows.extend(
            benchmark_geometry_workload(
                stage="regime",
                workload=str(spec["workload"]),
                geom=geom,
                case=str(spec["case"]),
                mode_counts=list(spec["mode_counts"]),
                margins=list(spec["margins"]),
                cutoff_bin_sizes=list(spec["bins"]),
                args=args,
                vortex_charge=int(spec["vortex_charge"]),
            )
        )
    rows.extend(benchmark_cache_sweep(args))
    return rows


def benchmark_cache_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rho_max = 3.0
    nrho = auto_nrho(rho_max, args.base_rho_max, args.base_nrho)
    geom = make_geometry(
        ntheta=48,
        nphi=128,
        nrho=nrho,
        npsi=96,
        nz=9,
        rho_max=rho_max,
        z_max=0.8,
        na=args.na,
        n_medium=args.n_medium,
        wavelength=args.wavelength,
    )
    settings = [(4, 4), (8, 4), (12, 4), (4, 8), (8, 8), (12, 8), (4, 16), (8, 16), (12, 16)]
    mode_count = 16
    pupils = make_pupils(
        geom,
        case="mixed",
        mode_count=mode_count,
        strength=args.strength,
        vortex_charge=18,
        apodization=args.apodization,
    )
    max_margin = max(margin for margin, _ in settings)
    h_cutoff = auto_h_cutoff(
        k=geom["k"],
        rho_max=geom["rho_max"],
        sin_theta_max=geom["sin_theta_max"],
        margin=max_margin,
        nphi=geom["nphi"],
    )
    cache, cache_build_s, _ = median_time(
        lambda: build_basis_cache(geom, h_cutoff=h_cutoff),
        args.repeats,
    )
    compact_total = 0.0
    cached_total = cache_build_s
    max_l2 = 0.0
    for margin, cutoff_bin_size in settings:
        compact_plan, compact_build_s, _ = median_time(
            lambda: build_compact_plan(
                geom,
                h_cutoff=h_cutoff,
                margin=margin,
                cutoff_bin_size=cutoff_bin_size,
                cpp_threads=args.cpp_threads,
            ),
            args.repeats,
        )
        compact_result, compact_hot_s, _ = median_time(
            lambda: compact_plan.evaluate_many(pupils),
            args.repeats,
        )
        cached_plan, cached_build_s, _ = median_time(
            lambda: build_cached_plan(
                cache,
                margin=margin,
                cutoff_bin_size=cutoff_bin_size,
                cpp_threads=args.cpp_threads,
            ),
            args.repeats,
        )
        cached_result, cached_hot_s, _ = median_time(
            lambda: cached_plan.evaluate_many(pupils),
            args.repeats,
        )
        compact_total += compact_build_s + compact_hot_s
        cached_total += cached_build_s + cached_hot_s
        max_l2 = max(max_l2, relative_l2(cached_result, compact_result))

    for k_settings in [3, 6, 9]:
        row = base_row(
            stage="regime",
            workload=f"cache_sweep_{k_settings}_settings",
            case="mixed",
            variant="cached_all_settings_summary",
            mode_count=mode_count,
            geom=geom,
            h_cutoff=h_cutoff,
            margin=max_margin,
            cutoff_bin_size="mixed",
            repeats=args.repeats,
            finufft_eps=args.finufft_eps,
            notes="summary uses measured full 9-setting totals scaled by setting count",
        )
        scaled_compact = compact_total * (k_settings / 9.0)
        scaled_cached = cache_build_s + (cached_total - cache_build_s) * (k_settings / 9.0)
        row.update(
            {
                "cache_build_s": cache_build_s,
                "total_s": scaled_cached,
                "amortized_total_s": scaled_cached - cache_build_s,
                "basis_mib": cache.basis_mib,
                "field_l2_vs_compact": max_l2,
                "speedup_total_vs_compact": speedup(scaled_compact, scaled_cached),
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run high-NA Debye-Wolf workload benchmarks by expected win/loss regime."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "correctness", "representative", "regime"],
        default="all",
    )
    parser.add_argument("--out-prefix", type=Path, default=Path("benchmark_results/high_na_workload_matrix"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode-counts", default="1,16,64")
    parser.add_argument("--cutoff-bin-sizes", default="4,8,16")
    parser.add_argument("--base-nrho", type=int, default=9)
    parser.add_argument("--base-rho-max", type=float, default=0.75)
    parser.add_argument("--wavelength", type=float, default=0.532)
    parser.add_argument("--na", type=float, default=0.8)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--apodization", choices=["none", "sqrt-cos"], default="none")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--finufft-eps", type=float, default=1e-9)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    if args.n_medium <= 0.0:
        raise ValueError("n-medium must be positive")
    if not (0.0 < args.na <= args.n_medium):
        raise ValueError("NA must satisfy 0 < NA <= n-medium")
    if args.finufft_eps <= 0.0:
        raise ValueError("finufft-eps must be positive")

    args.mode_counts = parse_int_list(args.mode_counts)
    args.cutoff_bin_sizes = parse_int_list(args.cutoff_bin_sizes)
    config = {
        "stage": args.stage,
        "repeats": args.repeats,
        "mode_counts": args.mode_counts,
        "cutoff_bin_sizes": args.cutoff_bin_sizes,
        "base_nrho": args.base_nrho,
        "base_rho_max": args.base_rho_max,
        "wavelength": args.wavelength,
        "na": args.na,
        "n_medium": args.n_medium,
        "strength": args.strength,
        "apodization": args.apodization,
        "cpp_threads": args.cpp_threads,
        "finufft_eps": args.finufft_eps,
    }

    outputs: list[tuple[str, list[dict[str, Any]]]] = []
    if args.stage in {"all", "correctness"}:
        outputs.append(("correctness", benchmark_correctness(args)))
    if args.stage in {"all", "representative"}:
        outputs.append(("representative", benchmark_representative(args)))
    if args.stage in {"all", "regime"}:
        outputs.append(("regime", benchmark_regime(args)))

    for name, rows in outputs:
        csv_path = args.out_prefix.with_name(f"{args.out_prefix.name}_{name}.csv")
        json_path = args.out_prefix.with_name(f"{args.out_prefix.name}_{name}.json")
        write_csv(csv_path, rows)
        write_payload(json_path, config, rows)
        print(f"{name}: rows={len(rows)} wrote {csv_path}")
        print(f"{name}: wrote {json_path}")


if __name__ == "__main__":
    main()
