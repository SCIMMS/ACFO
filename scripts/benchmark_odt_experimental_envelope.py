from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_odt_cone_axis_decomposition import benchmark_case as cone_axis_case
from benchmark_odt_cone_axis_decomposition import fmt
from benchmark_odt_cone_illumination import benchmark_case as cone_cmd_case
from benchmark_odt_same_direction_finufft import benchmark_case as same_direction_case
from benchmark_odt_shift_factorization import benchmark_case as shifted_case


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    family: str
    source_shape: str
    algorithms: tuple[str, ...]
    lambda_um: float
    medium_ri: float
    objective_na: float
    illumination_na: float
    cap_radial: int
    cap_phi: int
    n_illum: int
    n_mag: int
    cmd_ranks: tuple[int, ...]
    svd_ranks: tuple[int, ...]
    note: str
    requested_cap_phi: int | None = None


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_case_names(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("expected at least one case name")
    return names


def fft_friendly_length(n: int) -> int:
    n = int(n)
    if n <= 0:
        raise ValueError("FFT grid lengths must be positive")
    try:
        from scipy.fft import next_fast_len
    except Exception as exc:  # pragma: no cover - dependency is expected in benchmark envs.
        raise RuntimeError("scipy is required for --fft-friendly-grid") from exc
    return int(next_fast_len(n, real=False))


def is_fft_friendly_length(n: int) -> bool:
    return int(n) == fft_friendly_length(int(n))


def case_with_fft_metadata(case: ExperimentCase, *, fft_friendly_grid: bool) -> ExperimentCase:
    requested_cap_phi = int(case.cap_phi)
    cap_phi = fft_friendly_length(requested_cap_phi) if fft_friendly_grid else requested_cap_phi
    return replace(case, cap_phi=cap_phi, requested_cap_phi=requested_cap_phi)


def physical_to_solver(case: ExperimentCase) -> dict[str, float]:
    detector_sin = case.objective_na / case.medium_ri
    illumination_sin = case.illumination_na / case.medium_ri
    if detector_sin <= 0.0 or detector_sin >= 1.0:
        raise ValueError(f"{case.name}: objective_na / medium_ri must be in (0, 1)")
    if illumination_sin < 0.0 or illumination_sin >= 1.0:
        raise ValueError(f"{case.name}: illumination_na / medium_ri must be in [0, 1)")
    k_medium = 2.0 * math.pi * case.medium_ri / case.lambda_um
    q_perp_max_est = 2.0 * math.pi * (case.objective_na + case.illumination_na) / case.lambda_um
    return {
        "k": k_medium,
        "detector_sin": detector_sin,
        "illumination_sin": illumination_sin,
        "q_perp_max_est": q_perp_max_est,
    }


def default_cases(profile: str) -> dict[str, ExperimentCase]:
    if profile == "paper":
        high_cap = (32, 128)
        mid_cap = (24, 96)
        small_cap = (12, 48)
        n_illum_high = 64
        n_illum_mid = 32
        n_mag = 32
    else:
        high_cap = (16, 64)
        mid_cap = (12, 48)
        small_cap = (8, 32)
        n_illum_high = 32
        n_illum_mid = 16
        n_mag = 24

    return {
        "pc_annular_high_na": ExperimentCase(
            name="pc_annular_high_na",
            family="partially_coherent_high_na",
            source_shape="annular_outer_ring",
            algorithms=("cone_axis",),
            lambda_um=0.55,
            medium_ri=1.515,
            objective_na=1.40,
            illumination_na=1.33,
            cap_radial=high_cap[0],
            cap_phi=high_cap[1],
            n_illum=n_illum_high,
            n_mag=n_mag,
            cmd_ranks=(1, 5, 9),
            svd_ranks=(4, 8, 12, 16),
            note="High-NA annular/OFC-like outer source; use full-rank cone as the default sequential/incoherent-safe path.",
        ),
        "coherent_ring_mid_na": ExperimentCase(
            name="coherent_ring_mid_na",
            family="coherent_ring_scan",
            source_shape="single_radius_ring",
            algorithms=("cone_axis",),
            lambda_um=0.55,
            medium_ri=1.515,
            objective_na=1.20,
            illumination_na=0.78,
            cap_radial=mid_cap[0],
            cap_phi=mid_cap[1],
            n_illum=n_illum_mid,
            n_mag=n_mag,
            cmd_ranks=(1, 5),
            svd_ranks=(4, 8, 12),
            note="Sequential ring illumination at moderate source NA; full-rank cone preserves the angle stack.",
        ),
        "same_direction_high_na_scan": ExperimentCase(
            name="same_direction_high_na_scan",
            family="same_direction_magnitude_scan",
            source_shape="fixed_azimuth_varying_radius",
            algorithms=("same_direction",),
            lambda_um=0.55,
            medium_ri=1.515,
            objective_na=1.40,
            illumination_na=1.20,
            cap_radial=mid_cap[0],
            cap_phi=mid_cap[1],
            n_illum=n_illum_mid,
            n_mag=n_mag,
            cmd_ranks=(1,),
            svd_ranks=(4, 8, 12, 16, 24),
            note="Galvo/SLM line scan proxy; tests shared transverse direction with growing q.",
        ),
        "sparse_shifted_low_reuse": ExperimentCase(
            name="sparse_shifted_low_reuse",
            family="generic_shifted_sparse_angles",
            source_shape="few_shifted_angles",
            algorithms=("shifted_axis_fft",),
            lambda_um=0.55,
            medium_ri=1.33,
            objective_na=0.75,
            illumination_na=0.30,
            cap_radial=small_cap[0],
            cap_phi=small_cap[1],
            n_illum=9,
            n_mag=8,
            cmd_ranks=(1,),
            svd_ranks=(4, 8),
            note="Low-reuse shifted control; expected to be closer to FINUFFT.",
        ),
    }


def namespace_for_common(args: argparse.Namespace, case: ExperimentCase) -> argparse.Namespace:
    phys = physical_to_solver(case)
    return argparse.Namespace(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        k=phys["k"],
        detector_na=phys["detector_sin"],
        cap_radial=case.cap_radial,
        cap_phi=case.cap_phi,
        h_cutoff=None,
        h_margin=args.h_margin,
        l_margin=args.l_margin,
        seed=args.seed,
        structured_backend=args.structured_backend,
        phase_backend=args.phase_backend,
        cpp_threads=args.cpp_threads,
        build_repeats=args.build_repeats,
        hot_repeats=args.hot_repeats,
        cold_repeats=args.cold_repeats,
        finufft_repeats=args.finufft_repeats,
        finufft_eps=args.finufft_eps,
        skip_finufft=args.skip_finufft,
        cone_forward_mode=args.cone_forward_mode,
        cone_adjoint_mode=args.cone_adjoint_mode,
        cone_l_prune_threshold=args.cone_l_prune_threshold,
        iteration_counts=args.iteration_counts,
        report_iteration=args.report_iteration,
        n_illum=case.n_illum,
        min_illumination_na=max(args.min_same_direction_sin, phys["illumination_sin"] * 0.1),
        max_illumination_na=phys["illumination_sin"],
        direction_phi=0.0,
        svd_rank_values=list(case.svd_ranks),
    )


def common_metadata(case: ExperimentCase) -> dict[str, Any]:
    phys = physical_to_solver(case)
    requested_cap_phi = case.cap_phi if case.requested_cap_phi is None else case.requested_cap_phi
    return {
        "case": case.name,
        "family": case.family,
        "source_shape": case.source_shape,
        "case_note": case.note,
        "requested_cap_phi": requested_cap_phi,
        "cap_phi_fft_friendly": is_fft_friendly_length(case.cap_phi),
        "cap_phi_fft_grid_adjusted": int(requested_cap_phi) != int(case.cap_phi),
        "lambda_um": case.lambda_um,
        "medium_ri": case.medium_ri,
        "objective_na": case.objective_na,
        "illumination_na_physical": case.illumination_na,
        "detector_sin": phys["detector_sin"],
        "illumination_sin": phys["illumination_sin"],
        "k_medium_inv_um": phys["k"],
        "q_perp_max_est_inv_um": phys["q_perp_max_est"],
    }


def add_row(rows: list[dict[str, Any]], case: ExperimentCase, algorithm: str, row: dict[str, Any]) -> None:
    out = {**common_metadata(case), **row}
    out["algorithm"] = algorithm
    rows.append(out)


def normalize_same_direction(rows: list[dict[str, Any]], case: ExperimentCase, raw_rows: list[dict[str, Any]]) -> None:
    for row in raw_rows:
        add_row(
            rows,
            case,
            row["method"],
            {
                "comparison": "same_direction_vs_finufft",
                "timing_kind": "split",
                "rank": row.get("rank"),
                "n_mag": row.get("n_mag"),
                "n_illum": None,
                "cmd_rank": None,
                "cap_radial": row.get("cap_radial"),
                "cap_phi": row.get("cap_phi"),
                "flat_q_samples": row.get("flat_q_samples"),
                "axis_q_samples": row.get("axis_q_samples"),
                "unique_flat_q_perp_rounded12": row.get("unique_flat_q_perp_rounded12"),
                "flat_q_per_unique_q_perp": row.get("flat_q_per_unique_q_perp"),
                "forward_s": row.get("forward_s"),
                "adjoint_s": row.get("adjoint_s"),
                "pair_s": row.get("pair_s"),
                "finufft_forward_s": row.get("finufft_forward_s"),
                "finufft_adjoint_s": row.get("finufft_adjoint_s"),
                "finufft_pair_s": row.get("finufft_pair_s"),
                "speedup_forward_vs_finufft": row.get("finufft_over_method_forward_speedup"),
                "speedup_adjoint_vs_finufft": row.get("finufft_over_method_adjoint_speedup"),
                "speedup_pair_vs_finufft": row.get("finufft_over_method_speedup"),
                "speedup_pair_vs_flat": None,
                "forward_l2": row.get("method_forward_l2_vs_grouped"),
                "adjoint_l2": row.get("method_adjoint_l2_vs_grouped"),
                "dot_error": row.get("adjoint_dot_error"),
                "build_s": row.get("decomp_build_s"),
                "cache_mib": row.get("decomp_factor_mib"),
                "skip_reason": row.get("finufft_skip_reason"),
            },
        )


def normalize_cone_axis(rows: list[dict[str, Any]], case: ExperimentCase, row: dict[str, Any]) -> None:
    variants = [
        (
            "flat_shifted_structured",
            row.get("flat_hot_pair_s"),
            row.get("finufft_vs_flat_hot_speedup"),
            0.0,
            0.0,
            row.get("flat_build_s"),
            row.get("flat_kernel_mib"),
        ),
        (
            "phase_ramp_axis_fft",
            row.get("exact_hot_pair_s"),
            row.get("finufft_vs_exact_hot_speedup"),
            row.get("exact_forward_l2_vs_flat"),
            row.get("exact_adjoint_l2_vs_flat"),
            row.get("exact_build_s"),
            row.get("axis_kernel_mib"),
        ),
        (
            "cone_z0_zperp",
            row.get("decomp_hot_pair_s"),
            row.get("finufft_vs_decomp_hot_speedup"),
            row.get("decomp_forward_l2_vs_flat"),
            row.get("decomp_adjoint_l2_vs_flat"),
            row.get("decomp_build_s"),
            row.get("decomp_transverse_mib"),
        ),
        (
            "finufft_type3",
            row.get("finufft_pair_s"),
            1.0 if row.get("finufft_pair_s") is not None else None,
            row.get("finufft_forward_l2_vs_flat"),
            row.get("finufft_adjoint_l2_vs_flat"),
            None,
            None,
        ),
    ]
    for algorithm, pair_s, speedup, fwd_l2, adj_l2, build_s, cache_mib in variants:
        add_row(
            rows,
            case,
            algorithm,
            {
                "comparison": "cone_axis_vs_finufft",
                "timing_kind": "pair",
                "rank": None,
                "n_mag": None,
                "n_illum": row.get("n_illum"),
                "cmd_rank": None,
                "cap_radial": row.get("cap_radial"),
                "cap_phi": row.get("cap_phi"),
                "flat_q_samples": row.get("flat_q_samples"),
                "axis_q_samples": row.get("axis_q_samples"),
                "unique_flat_q_perp_rounded12": None,
                "flat_q_per_unique_q_perp": None,
                "forward_s": None,
                "adjoint_s": None,
                "pair_s": pair_s,
                "finufft_forward_s": None,
                "finufft_adjoint_s": None,
                "finufft_pair_s": row.get("finufft_pair_s"),
                "speedup_forward_vs_finufft": None,
                "speedup_adjoint_vs_finufft": None,
                "speedup_pair_vs_finufft": speedup,
                "speedup_pair_vs_flat": row.get("flat_vs_decomp_hot_speedup")
                if algorithm == "cone_z0_zperp"
                else row.get("flat_vs_exact_hot_speedup")
                if algorithm == "phase_ramp_axis_fft"
                else None,
                "forward_l2": fwd_l2,
                "adjoint_l2": adj_l2,
                "dot_error": row.get("decomp_adjoint_dot_error")
                if algorithm == "cone_z0_zperp"
                else None,
                "build_s": build_s,
                "cache_mib": cache_mib,
                "skip_reason": row.get("finufft_skip_reason"),
                "l_cutoff": row.get("l_cutoff"),
                "l_modes": row.get("l_modes"),
                "active_l_fraction": row.get("cone_l_active_fraction"),
            },
        )


def normalize_cone_cmd(rows: list[dict[str, Any]], case: ExperimentCase, row: dict[str, Any]) -> None:
    for algorithm, pair_s, speedup, fwd_l2, adj_l2 in [
        ("flat_cmd_modes", row.get("flat_hot_pair_s"), 1.0, 0.0, 0.0),
        (
            f"cone_cmd_rank_{row.get('cmd_rank')}",
            row.get("cone_hot_pair_s"),
            row.get("flat_vs_cone_hot_speedup"),
            row.get("cone_forward_l2_vs_flat"),
            row.get("cone_adjoint_l2_vs_flat"),
        ),
    ]:
        add_row(
            rows,
            case,
            algorithm,
            {
                "comparison": "cone_cmd_vs_flat_modes",
                "timing_kind": "pair",
                "rank": row.get("cmd_rank") if algorithm.startswith("cone_cmd") else None,
                "n_mag": None,
                "n_illum": row.get("n_illum"),
                "cmd_rank": row.get("cmd_rank"),
                "cap_radial": row.get("cap_radial"),
                "cap_phi": row.get("cap_phi"),
                "flat_q_samples": row.get("flat_q_samples"),
                "axis_q_samples": row.get("base_q_samples"),
                "unique_flat_q_perp_rounded12": row.get("unique_flat_q_perp_rounded12"),
                "flat_q_per_unique_q_perp": (
                    None
                    if not row.get("unique_flat_q_perp_rounded12")
                    else float(row.get("flat_q_samples")) / float(row.get("unique_flat_q_perp_rounded12"))
                ),
                "forward_s": None,
                "adjoint_s": None,
                "pair_s": pair_s,
                "finufft_forward_s": None,
                "finufft_adjoint_s": None,
                "finufft_pair_s": None,
                "speedup_forward_vs_finufft": None,
                "speedup_adjoint_vs_finufft": None,
                "speedup_pair_vs_finufft": None,
                "speedup_pair_vs_flat": speedup,
                "forward_l2": fwd_l2,
                "adjoint_l2": adj_l2,
                "dot_error": row.get("cone_adjoint_dot_error"),
                "build_s": row.get("cone_build_s"),
                "cache_mib": row.get("cone_cache_mib"),
                "skip_reason": None,
            },
        )


def normalize_shifted(rows: list[dict[str, Any]], case: ExperimentCase, row: dict[str, Any]) -> None:
    variants = [
        (
            "flat_shifted_structured",
            row.get("flat_hot_pair_s"),
            row.get("finufft_vs_flat_hot_pair_speedup"),
            0.0,
            0.0,
            row.get("flat_build_s"),
            row.get("flat_kernel_mib"),
        ),
        (
            "phase_ramp_axis_fft",
            row.get("fft_factored_hot_pair_s"),
            row.get("finufft_vs_fft_factored_hot_pair_speedup"),
            row.get("fft_factored_forward_l2_vs_flat"),
            row.get("fft_factored_adjoint_l2_vs_flat"),
            row.get("factored_build_s"),
            row.get("factored_cache_mib"),
        ),
        (
            "finufft_type3",
            row.get("finufft_pair_s"),
            1.0 if row.get("finufft_pair_s") is not None else None,
            row.get("factored_forward_l2_vs_finufft"),
            row.get("factored_adjoint_l2_vs_finufft"),
            None,
            None,
        ),
    ]
    for algorithm, pair_s, speedup, fwd_l2, adj_l2, build_s, cache_mib in variants:
        add_row(
            rows,
            case,
            algorithm,
            {
                "comparison": "shifted_axis_fft_vs_finufft",
                "timing_kind": "pair",
                "rank": None,
                "n_mag": None,
                "n_illum": row.get("n_illum"),
                "cmd_rank": None,
                "cap_radial": row.get("cap_radial"),
                "cap_phi": row.get("cap_phi"),
                "flat_q_samples": row.get("q_samples"),
                "axis_q_samples": row.get("base_q_samples"),
                "unique_flat_q_perp_rounded12": row.get("unique_flat_q_perp_rounded12"),
                "flat_q_per_unique_q_perp": (
                    None
                    if not row.get("unique_flat_q_perp_rounded12")
                    else float(row.get("q_samples")) / float(row.get("unique_flat_q_perp_rounded12"))
                ),
                "forward_s": None,
                "adjoint_s": None,
                "pair_s": pair_s,
                "finufft_forward_s": None,
                "finufft_adjoint_s": None,
                "finufft_pair_s": row.get("finufft_pair_s"),
                "speedup_forward_vs_finufft": None,
                "speedup_adjoint_vs_finufft": None,
                "speedup_pair_vs_finufft": speedup,
                "speedup_pair_vs_flat": row.get("flat_vs_fft_factored_hot_pair_speedup")
                if algorithm == "phase_ramp_axis_fft"
                else None,
                "forward_l2": fwd_l2,
                "adjoint_l2": adj_l2,
                "dot_error": None,
                "build_s": build_s,
                "cache_mib": cache_mib,
                "skip_reason": row.get("finufft_skip_reason"),
            },
        )


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


def best_rows(rows: list[dict[str, Any]], *, speed_key: str) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    for case in sorted({row["case"] for row in rows}):
        candidates = [
            row
            for row in rows
            if row["case"] == case
            and row.get(speed_key) is not None
            and row["algorithm"] != "finufft_type3"
            and not row["algorithm"].startswith("flat_")
        ]
        if candidates:
            best.append(max(candidates, key=lambda row: row[speed_key]))
    return best


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# ODT experimental envelope benchmark",
        "",
        "This benchmark maps experimentally motivated ODT sampling regimes to the structured algorithms currently implemented in the repository.",
        "",
        "Speedups greater than `1x` mean the structured method is faster than the stated baseline. Split forward/adjoint timing is currently available for the same-direction FINUFFT comparison; cone and generic shifted rows are pair timings.",
        "",
        "FFT-sensitive axes are the object beta grid (`n_beta`) and detector azimuth grid (`cap_phi`). With `--fft-friendly-grid`, unfriendly requested lengths are rounded upward and both requested/effective lengths are recorded in the JSON/CSV outputs.",
        "",
        "Cone CMD rows are optional diagnostics only. They are valid for a specified source CSD/CMD coherence model, but they are not used as the default baseline for standard sequential or fully incoherent ODT because low-rank CMD collapses source modes rather than preserving an independent incoherent angle stack.",
        "",
        "## Case Definitions",
        "",
        "| case | source | n_beta | cap_phi | FFT friendly | lambda um | objective NA | illumination NA | q_perp max est | algorithms | note |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    seen_cases: set[str] = set()
    for row in rows:
        if row["case"] in seen_cases:
            continue
        seen_cases.add(row["case"])
        case_rows = [item for item in rows if item["case"] == row["case"]]
        algorithms = ", ".join(sorted({item["algorithm"] for item in case_rows}))
        fft_status = (
            "yes"
            if row.get("n_beta_fft_friendly") and row.get("cap_phi_fft_friendly")
            else "no"
        )
        if row.get("fft_grid_adjusted"):
            fft_status += " (adjusted)"
        nb_grid = (
            f"{row.get('requested_n_beta')}->{row.get('n_beta')}"
            if row.get("requested_n_beta") != row.get("n_beta")
            else str(row.get("n_beta"))
        )
        cp_grid = (
            f"{row.get('requested_cap_phi')}->{row.get('cap_phi')}"
            if row.get("requested_cap_phi") != row.get("cap_phi")
            else str(row.get("cap_phi"))
        )
        lines.append(
            "| {case} | {source} | `{nb}` | `{cp}` | {fft} | `{lam}` | `{obj}` | `{ill}` | `{qmax}` | {algs} | {note} |".format(
                case=row["case"],
                source=row["source_shape"],
                nb=nb_grid,
                cp=cp_grid,
                fft=fft_status,
                lam=fmt(row["lambda_um"], 4),
                obj=fmt(row["objective_na"], 4),
                ill=fmt(row["illumination_na_physical"], 4),
                qmax=fmt(row["q_perp_max_est_inv_um"], 4),
                algs=algorithms,
                note=row["case_note"],
            )
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| case | comparison | algorithm | q | q/unique q_perp | rank | pair s | fwd s | adj s | speed vs FINUFFT | fwd vs FINUFFT | adj vs FINUFFT | speed vs flat | adj err |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {comp} | {alg} | {q} | `{rep}` | {rank} | `{pair}` | `{fwd}` | `{adj}` | `{sfin}` | `{sfwd}` | `{sadj}` | `{sflat}` | `{adjerr}` |".format(
                case=row["case"],
                comp=row["comparison"],
                alg=row["algorithm"],
                q="" if row.get("flat_q_samples") is None else row["flat_q_samples"],
                rep=fmt(row.get("flat_q_per_unique_q_perp"), 4),
                rank="" if row.get("rank") is None else row["rank"],
                pair=fmt(row.get("pair_s"), 5),
                fwd=fmt(row.get("forward_s"), 5),
                adj=fmt(row.get("adjoint_s"), 5),
                sfin=fmt(row.get("speedup_pair_vs_finufft"), 4),
                sfwd=fmt(row.get("speedup_forward_vs_finufft"), 4),
                sadj=fmt(row.get("speedup_adjoint_vs_finufft"), 4),
                sflat=fmt(row.get("speedup_pair_vs_flat"), 4),
                adjerr=fmt(row.get("adjoint_l2"), 4),
            )
        )
    lines.extend(["", "## Current Readout", ""])
    pair_best = best_rows(rows, speed_key="speedup_pair_vs_finufft")
    if pair_best:
        lines.append("- Best FINUFFT pair comparisons by case:")
        for row in pair_best:
            lines.append(
                "  - {case}: {alg}, `{speed}x` over FINUFFT, q={q}, adj err `{err}`.".format(
                    case=row["case"],
                    alg=row["algorithm"],
                    speed=fmt(row.get("speedup_pair_vs_finufft"), 4),
                    q=row.get("flat_q_samples"),
                    err=fmt(row.get("adjoint_l2"), 4),
                )
            )
    flat_best = best_rows(rows, speed_key="speedup_pair_vs_flat")
    if flat_best:
        lines.append("- Best flat-structured comparisons by case:")
        for row in flat_best:
            lines.append(
                "  - {case}: {alg}, `{speed}x` over flat structured path, q={q}, adj err `{err}`.".format(
                    case=row["case"],
                    alg=row["algorithm"],
                    speed=fmt(row.get("speedup_pair_vs_flat"), 4),
                    q=row.get("flat_q_samples"),
                    err=fmt(row.get("adjoint_l2"), 4),
                )
            )
    lines.extend(
        [
            "",
            "Interpretation rules:",
            "",
            "- Annular/ring cases test the same-q_perp repetition structure expected in high-NA partially coherent or ring illumination.",
            "- The default annular/ring ranking uses full-rank cone-axis paths. Low-rank cone CMD must be treated separately as a source-coherence-model diagnostic.",
            "- Same-direction cases test whether fixed azimuth with varying illumination magnitude can exploit shared direction phases.",
            "- Sparse shifted cases are controls where low reuse should make FINUFFT harder to beat.",
            "- A publication-grade sweep should expand the case grid around the cases where speedup and experimental relevance overlap.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    cases = list(dict.fromkeys(row["case"] for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180)
    for ax, key, title in [
        (axes[0], "speedup_pair_vs_finufft", "Pair speedup vs FINUFFT"),
        (axes[1], "speedup_pair_vs_flat", "Pair speedup vs flat structured"),
    ]:
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
        plotted = False
        algorithms = sorted({row["algorithm"] for row in rows if row.get(key) is not None})
        x = np.arange(len(cases))
        for algorithm in algorithms:
            values = []
            for case in cases:
                candidates = [
                    row
                    for row in rows
                    if row["case"] == case
                    and row["algorithm"] == algorithm
                    and row.get(key) is not None
                ]
                values.append(np.nan if not candidates else max(item[key] for item in candidates))
            if not np.all(np.isnan(values)):
                ax.plot(x, values, marker="o", label=algorithm)
                plotted = True
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=35, ha="right")
        ax.set_ylabel("speedup")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize=7)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experimentally motivated ODT envelope benchmarks.")
    parser.add_argument("--output-prefix", default="benchmark_results/odt_experimental_envelope_quick")
    parser.add_argument("--profile", choices=["quick", "paper"], default="quick")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--n-beta", type=int, default=192)
    parser.add_argument(
        "--fft-friendly-grid",
        action="store_true",
        help="Round n_beta and each case cap_phi upward to SciPy/pocketFFT-friendly complex FFT lengths.",
    )
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--l-margin", type=int, default=18)
    parser.add_argument("--min-same-direction-sin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--phase-backend", choices=["fft", "selected-dft"], default="fft")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--build-repeats", type=int, default=1)
    parser.add_argument("--hot-repeats", type=int, default=2)
    parser.add_argument("--cold-repeats", type=int, default=1)
    parser.add_argument("--finufft-repeats", type=int, default=2)
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--skip-finufft", action="store_true")
    parser.add_argument(
        "--include-cmd",
        action="store_true",
        help=(
            "Also run cone CMD source-mode diagnostics. These are not the default "
            "sequential/incoherent ODT baseline because low-rank CMD assumes a "
            "specific source CSD/coherence model."
        ),
    )
    parser.add_argument("--cone-forward-mode", choices=["auto", "two-step", "fused"], default="auto")
    parser.add_argument("--cone-adjoint-mode", choices=["auto", "two-step", "fused"], default="auto")
    parser.add_argument("--cone-l-prune-threshold", type=float, default=0.0)
    parser.add_argument("--iteration-counts", default="1,4,16,32")
    parser.add_argument("--report-iteration", type=int, default=32)
    args = parser.parse_args()
    args.iteration_counts = parse_int_list(args.iteration_counts)
    if args.report_iteration not in args.iteration_counts:
        args.iteration_counts.append(args.report_iteration)
        args.iteration_counts = sorted(set(args.iteration_counts))
    if args.profile == "paper" and args.n_beta == 192:
        args.n_beta = 384
    args.requested_n_beta = int(args.n_beta)
    if args.fft_friendly_grid:
        args.n_beta = fft_friendly_length(args.n_beta)
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.cone_l_prune_threshold < 0.0:
        raise ValueError("cone-l-prune-threshold must be non-negative")
    if args.finufft_eps <= 0.0:
        raise ValueError("finufft-eps must be positive")
    return args


def main() -> None:
    args = parse_args()
    cases_by_name = default_cases(args.profile)
    case_names = list(cases_by_name) if args.cases == "all" else parse_case_names(args.cases)
    missing = sorted(set(case_names) - set(cases_by_name))
    if missing:
        raise ValueError(f"unknown cases: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for case_name in case_names:
        case = case_with_fft_metadata(
            cases_by_name[case_name],
            fft_friendly_grid=args.fft_friendly_grid,
        )
        ns = namespace_for_common(args, case)
        phys = physical_to_solver(case)
        if "same_direction" in case.algorithms:
            normalize_same_direction(
                rows,
                case,
                same_direction_case(
                    ns,
                    n_mag=case.n_mag,
                    cap_radial=case.cap_radial,
                    cap_phi=case.cap_phi,
                ),
            )
        if "cone_axis" in case.algorithms:
            normalize_cone_axis(
                rows,
                case,
                cone_axis_case(
                    ns,
                    n_illum=case.n_illum,
                    illumination_na=phys["illumination_sin"],
                    l_margin=args.l_margin,
                ),
            )
        if "cone_cmd" in case.algorithms or (
            args.include_cmd
            and case.source_shape in {"annular_outer_ring", "single_radius_ring"}
        ):
            for rank in case.cmd_ranks:
                normalize_cone_cmd(
                    rows,
                    case,
                    cone_cmd_case(
                        ns,
                        n_illum=case.n_illum,
                        cmd_rank=rank,
                        illumination_na=phys["illumination_sin"],
                    ),
                )
        if "shifted_axis_fft" in case.algorithms:
            normalize_shifted(
                rows,
                case,
                shifted_case(
                    ns,
                    n_beta=args.n_beta,
                    illumination_na=phys["illumination_sin"],
                ),
            )

    for row in rows:
        row["requested_n_beta"] = args.requested_n_beta
        row["n_beta"] = args.n_beta
        row["n_beta_fft_friendly"] = is_fft_friendly_length(args.n_beta)
        row["fft_friendly_grid"] = args.fft_friendly_grid
        row["fft_grid_adjusted"] = (
            int(args.requested_n_beta) != int(args.n_beta)
            or bool(row.get("cap_phi_fft_grid_adjusted"))
        )
        if row.get("cap_phi") is not None:
            row["cap_phi_fft_friendly"] = is_fft_friendly_length(int(row["cap_phi"]))

    payload = {
        "args": {
            "profile": args.profile,
            "cases": case_names,
            "requested_n_beta": args.requested_n_beta,
            "n_beta": args.n_beta,
            "n_beta_fft_friendly": is_fft_friendly_length(args.n_beta),
            "fft_friendly_grid": args.fft_friendly_grid,
            "n_r": args.n_r,
            "n_z": args.n_z,
            "hot_repeats": args.hot_repeats,
            "build_repeats": args.build_repeats,
            "finufft_eps": args.finufft_eps,
            "skip_finufft": args.skip_finufft,
        },
        "rows": rows,
    }
    output_prefix = Path(args.output_prefix)
    write_json(output_prefix.with_suffix(".json"), payload)
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), payload)
    write_plot(output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg"), payload)
    print(
        json.dumps(
            {
                "json": str(output_prefix.with_suffix(".json")),
                "row_count": len(rows),
                "requested_n_beta": args.requested_n_beta,
                "n_beta": args.n_beta,
                "fft_friendly_grid": args.fft_friendly_grid,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
