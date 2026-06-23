import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_high_na_debye_wolf import (
    PreparedFinufftDebyeWolfPlan,
    PreparedRhoDependentHarmonicDebyeWolfPlan,
    PreparedSeparableHarmonicDebyeWolfPlan,
    flatten_focal_axes,
    focal_axes,
    gauss_theta_grid,
    max_abs_over_ref,
    median_time,
    pupil_field,
    relative_l2,
    sweep_strengths,
)


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_cases(value: str) -> list[str]:
    cases = [part.strip() for part in value.split(",") if part.strip()]
    if not cases:
        raise ValueError("expected at least one case")
    return cases


def auto_nrho(rho_max: float, base_rho_max: float, base_nrho: int) -> int:
    if base_nrho <= 1:
        return base_nrho
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_pupil_sequence(
    *,
    case: str,
    theta: np.ndarray,
    phi: np.ndarray,
    theta_max: float,
    strength: float,
    vortex_charge: int,
    apodization: str,
    sweeps: int,
) -> list[np.ndarray]:
    return [
        pupil_field(
            case,
            theta,
            phi,
            theta_max=theta_max,
            strength=value,
            vortex_charge=vortex_charge,
            apodization=apodization,
        )
        for value in sweep_strengths(strength, sweeps)
    ]


def speedup(num: float | None, denom: float | None) -> float | None:
    if num is None or denom is None or denom <= 0.0:
        return None
    return num / denom


def build_base_row(
    *,
    case: str,
    rho_max: float,
    nrho: int,
    h_cutoff: int,
    margin: int,
    ntheta: int,
    nphi: int,
    npsi: int,
    nz: int,
    targets: int,
    na: float,
    n_medium: float,
    wavelength: float,
    z_max: float,
    strength: float,
    sweeps: int,
    apodization: str,
    finufft_build_s: float | None,
    finufft_single_s: float | None,
    finufft_hot_s: float | None,
    finufft_coordinate_mib: float | None,
) -> dict[str, object]:
    return {
        "case": case,
        "variant": "",
        "rho_max": rho_max,
        "nrho": nrho,
        "npsi": npsi,
        "nz": nz,
        "targets": targets,
        "ntheta": ntheta,
        "nphi": nphi,
        "na": na,
        "n_medium": n_medium,
        "wavelength": wavelength,
        "z_max": z_max,
        "strength": strength,
        "sweeps": sweeps,
        "apodization": apodization,
        "h_cutoff": h_cutoff,
        "rho_margin": margin,
        "cutoff_bin_size": None,
        "groups": None,
        "max_modes": None,
        "mean_modes": None,
        "mode_rho_work": None,
        "rho_cutoffs": "",
        "basis_mib": None,
        "build_s": None,
        "single_s": None,
        "hot_s": None,
        "hot_per_mask_s": None,
        "build_plus_hot_s": None,
        "basis_reduction_vs_global": None,
        "build_speedup_vs_global": None,
        "single_speedup_vs_global": None,
        "hot_speedup_vs_global": None,
        "build_plus_hot_speedup_vs_global": None,
        "hot_speedup_vs_finufft": None,
        "hot_speedup_vs_finufft_incl_build": None,
        "single_field_relative_l2_vs_finufft": None,
        "single_field_max_abs_over_finufft": None,
        "hot_field_relative_l2_vs_finufft": None,
        "single_field_relative_l2_vs_global": None,
        "hot_field_relative_l2_vs_global": None,
        "finufft_build_s": finufft_build_s,
        "finufft_single_s": finufft_single_s,
        "finufft_hot_s": finufft_hot_s,
        "finufft_hot_per_mask_s": None
        if finufft_hot_s is None
        else finufft_hot_s / float(sweeps),
        "finufft_coordinate_mib": finufft_coordinate_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark WAXS-style rho-dependent harmonic cutoffs for the "
            "scalar high-NA Debye-Wolf solver."
        )
    )
    parser.add_argument("--cases", default="mixed")
    parser.add_argument("--ntheta", type=int, default=48)
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--base-nrho", type=int, default=9)
    parser.add_argument("--base-rho-max", type=float, default=0.75)
    parser.add_argument("--rho-max-values", default="0.75,1.5")
    parser.add_argument("--npsi", type=int, default=48)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--z-max", type=float, default=0.8)
    parser.add_argument("--wavelength", type=float, default=0.532)
    parser.add_argument("--na", type=float, default=0.8)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--vortex-charge", type=int, default=1)
    parser.add_argument("--apodization", choices=["none", "sqrt-cos"], default="none")
    parser.add_argument("--rho-margin", type=int, default=8)
    parser.add_argument("--cutoff-bin-sizes", default="1,2,4,8")
    parser.add_argument(
        "--harmonic-backend",
        choices=["auto", "numpy", "cpp"],
        default="numpy",
    )
    parser.add_argument(
        "--rho-dependent-backend",
        choices=["auto", "numpy", "cpp"],
        default="auto",
    )
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--finufft-eps", type=float, default=1e-9)
    parser.add_argument("--skip-finufft", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/high_na_rho_dependent_cutoff.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results/high_na_rho_dependent_cutoff.csv"),
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.sweeps <= 0:
        raise ValueError("sweeps must be positive")
    if args.rho_margin < 0:
        raise ValueError("rho-margin must be non-negative")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    if args.n_medium <= 0.0:
        raise ValueError("n-medium must be positive")
    if not (0.0 < args.na <= args.n_medium):
        raise ValueError("NA must satisfy 0 < NA <= n-medium")

    cases = parse_cases(args.cases)
    rho_max_values = parse_float_list(args.rho_max_values)
    cutoff_bin_sizes = parse_int_list(args.cutoff_bin_sizes)
    if any(value <= 0 for value in cutoff_bin_sizes):
        raise ValueError("cutoff-bin-sizes must be positive")

    theta_max = float(np.arcsin(args.na / args.n_medium))
    sin_theta_max = float(np.sin(theta_max))
    theta, theta_weights = gauss_theta_grid(args.ntheta, theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, args.nphi, endpoint=False, dtype=float)
    k = 2.0 * np.pi * args.n_medium / args.wavelength

    rows: list[dict[str, object]] = []
    for case in cases:
        for rho_max in rho_max_values:
            nrho = auto_nrho(rho_max, args.base_rho_max, args.base_nrho)
            rho_axis, psi_axis, z_axis = focal_axes(
                nrho=nrho,
                npsi=args.npsi,
                nz=args.nz,
                rho_max=rho_max,
                z_max=args.z_max,
            )
            rho, psi, z = flatten_focal_axes(rho_axis, psi_axis, z_axis)
            h_cutoff = auto_h_cutoff(
                k=k,
                rho_max=rho_max,
                sin_theta_max=sin_theta_max,
                margin=args.rho_margin,
                nphi=args.nphi,
            )
            pupils = make_pupil_sequence(
                case=case,
                theta=theta,
                phi=phi,
                theta_max=theta_max,
                strength=args.strength,
                vortex_charge=args.vortex_charge,
                apodization=args.apodization,
                sweeps=args.sweeps,
            )
            pupil = pupils[len(pupils) // 2]

            if args.skip_finufft:
                finufft_plan = None
                finufft_single = None
                finufft_hot = None
                finufft_build_s = None
                finufft_single_s = None
                finufft_hot_s = None
                finufft_coordinate_mib = None
            else:
                finufft_plan, finufft_build_s, _ = median_time(
                    lambda: PreparedFinufftDebyeWolfPlan.build(
                        theta,
                        theta_weights,
                        phi,
                        rho,
                        psi,
                        z,
                        k=k,
                    ),
                    args.repeats,
                )
                finufft_coordinate_mib = finufft_plan.coordinate_mib
                finufft_single, finufft_single_s, _ = median_time(
                    lambda: finufft_plan.evaluate(pupil, eps=args.finufft_eps),
                    args.repeats,
                )
                finufft_hot, finufft_hot_s, _ = median_time(
                    lambda: finufft_plan.evaluate_many(pupils, eps=args.finufft_eps),
                    args.repeats,
                )

            base_row = build_base_row(
                case=case,
                rho_max=rho_max,
                nrho=nrho,
                h_cutoff=h_cutoff,
                margin=args.rho_margin,
                ntheta=args.ntheta,
                nphi=args.nphi,
                npsi=args.npsi,
                nz=args.nz,
                targets=int(rho.size),
                na=args.na,
                n_medium=args.n_medium,
                wavelength=args.wavelength,
                z_max=args.z_max,
                strength=args.strength,
                sweeps=args.sweeps,
                apodization=args.apodization,
                finufft_build_s=finufft_build_s,
                finufft_single_s=finufft_single_s,
                finufft_hot_s=finufft_hot_s,
                finufft_coordinate_mib=finufft_coordinate_mib,
            )

            global_plan, global_build_s, _ = median_time(
                lambda: PreparedSeparableHarmonicDebyeWolfPlan.build(
                    args.nphi,
                    theta,
                    theta_weights,
                    rho_axis,
                    psi_axis,
                    z_axis,
                    k=k,
                    h_cutoff=h_cutoff,
                    backend=args.harmonic_backend,
                    cpp_threads=args.cpp_threads,
                ),
                args.repeats,
            )
            global_single, global_single_s, _ = median_time(
                lambda: global_plan.evaluate(pupil),
                args.repeats,
            )
            global_hot, global_hot_s, _ = median_time(
                lambda: global_plan.evaluate_many(pupils),
                args.repeats,
            )

            global_row = dict(base_row)
            global_row.update(
                {
                    "variant": "global",
                    "cutoff_bin_size": 0,
                    "groups": 1,
                    "max_modes": global_plan.used_modes,
                    "mean_modes": float(global_plan.used_modes),
                    "mode_rho_work": int(global_plan.used_modes * nrho),
                    "rho_cutoffs": str(h_cutoff),
                    "basis_mib": global_plan.basis_mib,
                    "build_s": global_build_s,
                    "single_s": global_single_s,
                    "hot_s": global_hot_s,
                    "hot_per_mask_s": global_hot_s / float(args.sweeps),
                    "build_plus_hot_s": global_build_s + global_hot_s,
                    "basis_reduction_vs_global": 1.0,
                    "build_speedup_vs_global": 1.0,
                    "single_speedup_vs_global": 1.0,
                    "hot_speedup_vs_global": 1.0,
                    "build_plus_hot_speedup_vs_global": 1.0,
                    "hot_speedup_vs_finufft": speedup(finufft_hot_s, global_hot_s),
                    "hot_speedup_vs_finufft_incl_build": speedup(
                        finufft_hot_s,
                        global_build_s + global_hot_s,
                    ),
                    "single_field_relative_l2_vs_finufft": None
                    if finufft_single is None
                    else relative_l2(global_single, finufft_single),
                    "single_field_max_abs_over_finufft": None
                    if finufft_single is None
                    else max_abs_over_ref(global_single, finufft_single),
                    "hot_field_relative_l2_vs_finufft": None
                    if finufft_hot is None
                    else relative_l2(global_hot, finufft_hot),
                    "single_field_relative_l2_vs_global": 0.0,
                    "hot_field_relative_l2_vs_global": 0.0,
                }
            )
            rows.append(global_row)

            for cutoff_bin_size in cutoff_bin_sizes:
                rho_plan, rho_build_s, _ = median_time(
                    lambda: PreparedRhoDependentHarmonicDebyeWolfPlan.build(
                        args.nphi,
                        theta,
                        theta_weights,
                        rho_axis,
                        psi_axis,
                        z_axis,
                        k=k,
                        h_cutoff=h_cutoff,
                        margin=args.rho_margin,
                        cutoff_bin_size=cutoff_bin_size,
                        sin_theta_max=sin_theta_max,
                        backend=args.rho_dependent_backend,
                        cpp_threads=args.cpp_threads,
                    ),
                    args.repeats,
                )
                rho_single, rho_single_s, _ = median_time(
                    lambda: rho_plan.evaluate(pupil),
                    args.repeats,
                )
                rho_hot, rho_hot_s, _ = median_time(
                    lambda: rho_plan.evaluate_many(pupils),
                    args.repeats,
                )
                row = dict(base_row)
                row.update(
                    {
                        "variant": "rho_dependent",
                        "cutoff_bin_size": cutoff_bin_size,
                        "groups": rho_plan.group_count,
                        "max_modes": rho_plan.used_modes,
                        "mean_modes": rho_plan.mean_used_modes,
                        "mode_rho_work": rho_plan.mode_rho_work,
                        "rho_cutoffs": ";".join(
                            str(int(value)) for value in rho_plan.cutoffs
                        ),
                        "basis_mib": rho_plan.basis_mib,
                        "build_s": rho_build_s,
                        "single_s": rho_single_s,
                        "hot_s": rho_hot_s,
                        "hot_per_mask_s": rho_hot_s / float(args.sweeps),
                        "build_plus_hot_s": rho_build_s + rho_hot_s,
                        "basis_reduction_vs_global": speedup(
                            global_plan.basis_mib,
                            rho_plan.basis_mib,
                        ),
                        "build_speedup_vs_global": speedup(
                            global_build_s,
                            rho_build_s,
                        ),
                        "single_speedup_vs_global": speedup(
                            global_single_s,
                            rho_single_s,
                        ),
                        "hot_speedup_vs_global": speedup(global_hot_s, rho_hot_s),
                        "build_plus_hot_speedup_vs_global": speedup(
                            global_build_s + global_hot_s,
                            rho_build_s + rho_hot_s,
                        ),
                        "hot_speedup_vs_finufft": speedup(finufft_hot_s, rho_hot_s),
                        "hot_speedup_vs_finufft_incl_build": speedup(
                            finufft_hot_s,
                            rho_build_s + rho_hot_s,
                        ),
                        "single_field_relative_l2_vs_finufft": None
                        if finufft_single is None
                        else relative_l2(rho_single, finufft_single),
                        "single_field_max_abs_over_finufft": None
                        if finufft_single is None
                        else max_abs_over_ref(rho_single, finufft_single),
                        "hot_field_relative_l2_vs_finufft": None
                        if finufft_hot is None
                        else relative_l2(rho_hot, finufft_hot),
                        "single_field_relative_l2_vs_global": relative_l2(
                            rho_single,
                            global_single,
                        ),
                        "hot_field_relative_l2_vs_global": relative_l2(
                            rho_hot,
                            global_hot,
                        ),
                    }
                )
                rows.append(row)

    payload = {
        "config": {
            "cases": cases,
            "rho_max_values": rho_max_values,
            "cutoff_bin_sizes": cutoff_bin_sizes,
            "base_nrho": args.base_nrho,
            "base_rho_max": args.base_rho_max,
            "ntheta": args.ntheta,
            "nphi": args.nphi,
            "npsi": args.npsi,
            "nz": args.nz,
            "na": args.na,
            "n_medium": args.n_medium,
            "wavelength": args.wavelength,
            "rho_margin": args.rho_margin,
            "harmonic_backend": args.harmonic_backend,
            "rho_dependent_backend": args.rho_dependent_backend,
            "cpp_threads": args.cpp_threads,
            "sweeps": args.sweeps,
            "repeats": args.repeats,
            "finufft_eps": args.finufft_eps,
            "skip_finufft": args.skip_finufft,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.csv, rows)

    for row in rows:
        finufft_hot = row["hot_speedup_vs_finufft"]
        finufft_text = "NA" if finufft_hot is None else f"{finufft_hot:.2f}x"
        print(
            f"{row['case']:8s} rho={row['rho_max']:<4} "
            f"{row['variant']:13s} bin={row['cutoff_bin_size']} "
            f"groups={row['groups']} mean_modes={row['mean_modes']:.1f} "
            f"build={row['build_s']:.5f}s hot={row['hot_s']:.5f}s "
            f"basis={row['basis_mib']:.3f}MiB "
            f"vs_global_hot={row['hot_speedup_vs_global']:.2f}x "
            f"vs_finufft_hot={finufft_text} "
            f"l2_global={row['single_field_relative_l2_vs_global']}"
        )
    print(f"wrote {args.out}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
