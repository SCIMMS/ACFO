import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_high_na_debye_wolf import (
    PreparedFinufftDebyeWolfPlan,
    PreparedPositiveRhoDependentHarmonicDebyeWolfPlan,
    evaluate_direct_sequence,
    extra_pupil_h_abs,
    flatten_focal_axes,
    focal_axes,
    gauss_theta_grid,
    max_abs_over_ref,
    median_time,
    pupil_field,
    relative_l2,
    significant_pupil_h_abs,
    sweep_strengths,
)


FIELDNAMES = [
    "workload",
    "case",
    "variant",
    "pupil_spectrum",
    "mode_count",
    "charge",
    "ntheta",
    "nphi",
    "nrho",
    "npsi",
    "nz",
    "targets",
    "rho_max",
    "z_max",
    "na",
    "geometric_h_cutoff",
    "plan_max_cutoff",
    "spectrum_h_max",
    "spectrum_h_count",
    "required_h_count",
    "required_h_values",
    "cutoff_bin_size",
    "groups",
    "mean_used_modes",
    "mode_rho_work",
    "basis_mib",
    "build_s",
    "hot_s",
    "total_s",
    "field_l2_vs_direct",
    "field_max_abs_vs_direct",
    "field_l2_vs_finufft",
    "field_max_abs_vs_finufft",
    "speedup_total_vs_geometric",
    "speedup_hot_vs_finufft",
    "speedup_total_vs_finufft",
    "direct_reference",
    "repeats",
    "relative_threshold",
    "finufft_eps",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDNAMES})


def write_json(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")


def auto_nrho(rho_max: float, base_rho_max: float, base_nrho: int) -> int:
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


def make_pupils(
    geom: dict[str, Any],
    *,
    case: str,
    charge: int,
    mode_count: int,
    strength: float,
    apodization: str,
) -> list[np.ndarray]:
    pupils = []
    for index, value in enumerate(sweep_strengths(strength, mode_count)):
        local_charge = charge
        if case == "vortex" and mode_count > 1:
            local_charge = charge + (index % 3) - 1
        pupils.append(
            pupil_field(
                case,
                geom["theta"],
                geom["phi"],
                theta_max=geom["theta_max"],
                strength=value,
                vortex_charge=local_charge,
                apodization=apodization,
            )
        )
    return pupils


def build_plan(
    geom: dict[str, Any],
    *,
    h_cutoff: int,
    margin: int,
    cutoff_bin_size: int,
    required_h_abs: np.ndarray | None,
    pupil_spectrum: str,
    pupil_spectrum_pupils: list[np.ndarray] | None,
    pupil_spectrum_relative_threshold: float,
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
        required_h_abs=required_h_abs,
        pupil_spectrum=pupil_spectrum,
        pupil_spectrum_pupils=pupil_spectrum_pupils,
        pupil_spectrum_relative_threshold=pupil_spectrum_relative_threshold,
        sin_theta_max=geom["sin_theta_max"],
        backend="cpp",
        cpp_threads=cpp_threads,
        no_copy_coefficients=True,
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


def add_plan_metrics(row: dict[str, Any], plan: PreparedPositiveRhoDependentHarmonicDebyeWolfPlan) -> None:
    row["plan_max_cutoff"] = plan.max_cutoff
    row["groups"] = plan.group_count
    row["mean_used_modes"] = plan.mean_used_modes
    row["mode_rho_work"] = plan.mode_rho_work
    row["basis_mib"] = plan.basis_mib


def run_workload(
    spec: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    geom = make_geometry(
        ntheta=spec["ntheta"],
        nphi=spec["nphi"],
        nrho=spec["nrho"],
        npsi=spec["npsi"],
        nz=spec["nz"],
        rho_max=spec["rho_max"],
        z_max=spec["z_max"],
        na=args.na,
        n_medium=args.n_medium,
        wavelength=args.wavelength,
    )
    margin = spec["margin"]
    cutoff_bin_size = spec["cutoff_bin_size"]
    pupils = make_pupils(
        geom,
        case=spec["case"],
        charge=spec["charge"],
        mode_count=spec["mode_count"],
        strength=args.strength,
        apodization=args.apodization,
    )
    geometric_h_cutoff = auto_h_cutoff(
        k=geom["k"],
        rho_max=geom["rho_max"],
        sin_theta_max=geom["sin_theta_max"],
        margin=margin,
        nphi=geom["nphi"],
    )
    spectrum_h_abs = significant_pupil_h_abs(
        pupils,
        relative_threshold=args.relative_threshold,
    )
    spectrum_h_max = int(np.max(spectrum_h_abs))
    sparse_required = extra_pupil_h_abs(
        pupils,
        geometric_h_cutoff=geometric_h_cutoff,
        relative_threshold=args.relative_threshold,
    )
    dense_required = (
        np.arange(geometric_h_cutoff + 1, spectrum_h_max + 1, dtype=np.int64)
        if spectrum_h_max > geometric_h_cutoff
        else np.empty(0, dtype=np.int64)
    )
    variants = [
        ("geometric_only", None, "off"),
        ("adaptive_sparse", None, "adaptive"),
        ("adaptive_dense_prefix", dense_required, "off"),
    ]

    direct_reference = bool(spec.get("direct_reference", True))
    direct_ref = None
    direct_s = None
    if direct_reference:
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
    finufft_plan, finufft_build_s, _ = median_time(
        lambda: build_finufft_plan(geom),
        args.repeats,
    )
    finufft_ref, finufft_hot_s, _ = median_time(
        lambda: finufft_plan.evaluate_many(pupils, eps=args.finufft_eps),
        args.repeats,
    )

    rows: list[dict[str, Any]] = []
    geometric_total: float | None = None
    for variant, required_h_abs, pupil_spectrum in variants:
        plan, build_s, _ = median_time(
            lambda: build_plan(
                geom,
                h_cutoff=geometric_h_cutoff,
                margin=margin,
                cutoff_bin_size=cutoff_bin_size,
                required_h_abs=required_h_abs,
                pupil_spectrum=pupil_spectrum,
                pupil_spectrum_pupils=pupils if pupil_spectrum != "off" else None,
                pupil_spectrum_relative_threshold=args.relative_threshold,
                cpp_threads=args.cpp_threads,
            ),
            args.repeats,
        )
        result, hot_s, _ = median_time(lambda: plan.evaluate_many(pupils), args.repeats)
        total_s = build_s + hot_s
        if variant == "geometric_only":
            geometric_total = total_s
        required_values = [] if required_h_abs is None else [int(value) for value in required_h_abs]
        row = {
            "workload": spec["workload"],
            "case": spec["case"],
            "variant": variant,
            "pupil_spectrum": pupil_spectrum,
            "mode_count": spec["mode_count"],
            "charge": spec["charge"],
            "ntheta": geom["ntheta"],
            "nphi": geom["nphi"],
            "nrho": geom["nrho"],
            "npsi": geom["npsi"],
            "nz": geom["nz"],
            "targets": int(geom["rho_flat"].size),
            "rho_max": geom["rho_max"],
            "z_max": geom["z_max"],
            "na": geom["na"],
            "geometric_h_cutoff": geometric_h_cutoff,
            "spectrum_h_max": spectrum_h_max,
            "spectrum_h_count": int(spectrum_h_abs.size),
            "required_h_count": len(required_values),
            "required_h_values": " ".join(str(value) for value in required_values),
            "cutoff_bin_size": cutoff_bin_size,
            "build_s": build_s,
            "hot_s": hot_s,
            "total_s": total_s,
            "field_l2_vs_direct": None
            if direct_ref is None
            else relative_l2(result, direct_ref),
            "field_max_abs_vs_direct": None
            if direct_ref is None
            else max_abs_over_ref(result, direct_ref),
            "field_l2_vs_finufft": relative_l2(result, finufft_ref),
            "field_max_abs_vs_finufft": max_abs_over_ref(result, finufft_ref),
            "speedup_total_vs_geometric": None
            if geometric_total is None
            else geometric_total / total_s,
            "speedup_hot_vs_finufft": finufft_hot_s / hot_s if hot_s > 0.0 else None,
            "speedup_total_vs_finufft": (finufft_build_s + finufft_hot_s) / total_s
            if total_s > 0.0
            else None,
            "direct_reference": direct_reference,
            "repeats": args.repeats,
            "relative_threshold": args.relative_threshold,
            "finufft_eps": args.finufft_eps,
        }
        if variant == "adaptive_sparse":
            row["required_h_count"] = int(plan.required_h_abs.size)
            row["required_h_values"] = " ".join(
                str(int(value)) for value in plan.required_h_abs
            )
        add_plan_metrics(row, plan)
        rows.append(row)

    if direct_ref is not None:
        rows.append(
            {
                "workload": spec["workload"],
                "case": spec["case"],
                "variant": "direct_debye_wolf",
                "pupil_spectrum": "reference",
                "mode_count": spec["mode_count"],
                "charge": spec["charge"],
                "ntheta": geom["ntheta"],
                "nphi": geom["nphi"],
                "nrho": geom["nrho"],
                "npsi": geom["npsi"],
                "nz": geom["nz"],
                "targets": int(geom["rho_flat"].size),
                "rho_max": geom["rho_max"],
                "z_max": geom["z_max"],
                "na": geom["na"],
                "geometric_h_cutoff": geometric_h_cutoff,
                "spectrum_h_max": spectrum_h_max,
                "spectrum_h_count": int(spectrum_h_abs.size),
                "hot_s": direct_s,
                "total_s": direct_s,
                "field_l2_vs_direct": 0.0,
                "direct_reference": direct_reference,
                "repeats": args.repeats,
                "relative_threshold": args.relative_threshold,
                "finufft_eps": args.finufft_eps,
            }
        )
    rows.append(
        {
            "workload": spec["workload"],
            "case": spec["case"],
            "variant": "finufft",
            "pupil_spectrum": "reference",
            "mode_count": spec["mode_count"],
            "charge": spec["charge"],
            "ntheta": geom["ntheta"],
            "nphi": geom["nphi"],
            "nrho": geom["nrho"],
            "npsi": geom["npsi"],
            "nz": geom["nz"],
            "targets": int(geom["rho_flat"].size),
            "rho_max": geom["rho_max"],
            "z_max": geom["z_max"],
            "na": geom["na"],
            "geometric_h_cutoff": geometric_h_cutoff,
            "spectrum_h_max": spectrum_h_max,
            "spectrum_h_count": int(spectrum_h_abs.size),
            "build_s": finufft_build_s,
            "hot_s": finufft_hot_s,
            "total_s": finufft_build_s + finufft_hot_s,
            "field_l2_vs_direct": None
            if direct_ref is None
            else relative_l2(finufft_ref, direct_ref),
            "field_max_abs_vs_direct": None
            if direct_ref is None
            else max_abs_over_ref(finufft_ref, direct_ref),
            "direct_reference": direct_reference,
            "repeats": args.repeats,
            "relative_threshold": args.relative_threshold,
            "finufft_eps": args.finufft_eps,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pupil-spectrum adaptive harmonic support for high-NA Debye-Wolf plans."
    )
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/high_na_pupil_spectrum_adaptive.json"))
    parser.add_argument("--csv", type=Path, default=Path("benchmark_results/high_na_pupil_spectrum_adaptive.csv"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--relative-threshold", type=float, default=1e-6)
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
    if args.relative_threshold < 0.0:
        raise ValueError("relative-threshold must be non-negative")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    if args.n_medium <= 0.0:
        raise ValueError("n-medium must be positive")
    if not (0.0 < args.na <= args.n_medium):
        raise ValueError("NA must satisfy 0 < NA <= n-medium")

    specs = [
        {
            "workload": "benign_mixed_small",
            "case": "mixed",
            "charge": 1,
            "mode_count": 4,
            "ntheta": 24,
            "nphi": 96,
            "nrho": 5,
            "npsi": 16,
            "nz": 3,
            "rho_max": 0.75,
            "z_max": 0.5,
            "margin": 8,
            "cutoff_bin_size": 8,
            "direct_reference": True,
        },
        {
            "workload": "vortex_requires_extra_h_small",
            "case": "vortex",
            "charge": 18,
            "mode_count": 1,
            "ntheta": 24,
            "nphi": 96,
            "nrho": 5,
            "npsi": 16,
            "nz": 3,
            "rho_max": 0.75,
            "z_max": 0.5,
            "margin": 8,
            "cutoff_bin_size": 8,
            "direct_reference": True,
        },
        {
            "workload": "vortex_requires_extra_h_representative",
            "case": "vortex",
            "charge": 30,
            "mode_count": 1,
            "ntheta": 32,
            "nphi": 128,
            "nrho": 17,
            "npsi": 64,
            "nz": 5,
            "rho_max": 1.5,
            "z_max": 0.6,
            "margin": 8,
            "cutoff_bin_size": 8,
            "direct_reference": True,
        },
        {
            "workload": "benign_mixed_representative_modes16",
            "case": "mixed",
            "charge": 1,
            "mode_count": 16,
            "ntheta": 48,
            "nphi": 128,
            "nrho": 33,
            "npsi": 96,
            "nz": 9,
            "rho_max": 3.0,
            "z_max": 0.8,
            "margin": 8,
            "cutoff_bin_size": 8,
            "direct_reference": False,
        },
        {
            "workload": "vortex_extra_h_representative_modes16",
            "case": "vortex",
            "charge": 30,
            "mode_count": 16,
            "ntheta": 32,
            "nphi": 128,
            "nrho": 17,
            "npsi": 64,
            "nz": 5,
            "rho_max": 1.5,
            "z_max": 0.6,
            "margin": 8,
            "cutoff_bin_size": 8,
            "direct_reference": False,
        },
        {
            "workload": "vortex_extra_h_large_volume_modes16",
            "case": "vortex",
            "charge": 50,
            "mode_count": 16,
            "ntheta": 48,
            "nphi": 192,
            "nrho": 33,
            "npsi": 96,
            "nz": 9,
            "rho_max": 3.0,
            "z_max": 0.8,
            "margin": 8,
            "cutoff_bin_size": 8,
            "direct_reference": False,
        },
        {
            "workload": "benign_mixed_large_modes64",
            "case": "mixed",
            "charge": 1,
            "mode_count": 64,
            "ntheta": 48,
            "nphi": 128,
            "nrho": 33,
            "npsi": 96,
            "nz": 9,
            "rho_max": 3.0,
            "z_max": 0.8,
            "margin": 8,
            "cutoff_bin_size": 8,
            "direct_reference": False,
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.extend(run_workload(spec, args=args))

    config = {
        "repeats": args.repeats,
        "relative_threshold": args.relative_threshold,
        "wavelength": args.wavelength,
        "na": args.na,
        "n_medium": args.n_medium,
        "strength": args.strength,
        "apodization": args.apodization,
        "cpp_threads": args.cpp_threads,
        "finufft_eps": args.finufft_eps,
    }
    write_csv(args.csv, rows)
    write_json(args.out, config, rows)
    for row in rows:
        if row["variant"] not in {"geometric_only", "adaptive_sparse", "adaptive_dense_prefix"}:
            continue
        error_value = row["field_l2_vs_direct"]
        error_label = "direct"
        if error_value is None:
            error_value = row["field_l2_vs_finufft"]
            error_label = "finufft"
        print(
            f"{row['workload']:38s} {row['variant']:22s} "
            f"h_geo={row['geometric_h_cutoff']} h_spec={row['spectrum_h_max']} "
            f"work={row['mode_rho_work']} total={row['total_s']:.5f}s "
            f"l2_{error_label}={error_value:.3e}"
        )
    print(f"wrote {args.csv}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
