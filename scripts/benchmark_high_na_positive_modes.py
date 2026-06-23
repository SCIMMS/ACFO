import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_high_na_debye_wolf import (
    PreparedFinufftDebyeWolfPlan,
    PreparedPositiveSeparableHarmonicDebyeWolfPlan,
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


def parse_cases(value: str) -> list[str]:
    cases = [part.strip() for part in value.split(",") if part.strip()]
    if not cases:
        raise ValueError("expected at least one case")
    return cases


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


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


def speedup(num: float | None, denom: float | None) -> float | None:
    if num is None or denom is None or denom <= 0.0:
        return None
    return num / denom


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark signed versus positive-only separable harmonic "
            "Debye-Wolf plans."
        )
    )
    parser.add_argument("--cases", default="mixed")
    parser.add_argument("--ntheta", type=int, default=48)
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--base-nrho", type=int, default=9)
    parser.add_argument("--base-rho-max", type=float, default=0.75)
    parser.add_argument("--rho-max-values", default="0.75,1.5,2.25")
    parser.add_argument("--npsi", type=int, default=48)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--z-max", type=float, default=0.8)
    parser.add_argument("--wavelength", type=float, default=0.532)
    parser.add_argument("--na", type=float, default=0.8)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--vortex-charge", type=int, default=1)
    parser.add_argument("--apodization", choices=["none", "sqrt-cos"], default="none")
    parser.add_argument("--h-margin", type=int, default=8)
    parser.add_argument(
        "--signed-backend",
        choices=["auto", "numpy", "cpp"],
        default="numpy",
    )
    parser.add_argument(
        "--positive-backend",
        choices=["auto", "numpy", "cpp"],
        default="auto",
    )
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--finufft-eps", type=float, default=1e-9)
    parser.add_argument("--skip-finufft", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/high_na_positive_modes.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results/high_na_positive_modes.csv"),
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.sweeps <= 0:
        raise ValueError("sweeps must be positive")
    if args.h_margin < 0:
        raise ValueError("h-margin must be non-negative")
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
                margin=args.h_margin,
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

            signed_plan, signed_build_s, _ = median_time(
                lambda: PreparedSeparableHarmonicDebyeWolfPlan.build(
                    args.nphi,
                    theta,
                    theta_weights,
                    rho_axis,
                    psi_axis,
                    z_axis,
                    k=k,
                    h_cutoff=h_cutoff,
                    backend=args.signed_backend,
                    cpp_threads=args.cpp_threads,
                ),
                args.repeats,
            )
            signed_single, signed_single_s, _ = median_time(
                lambda: signed_plan.evaluate(pupil),
                args.repeats,
            )
            signed_hot, signed_hot_s, _ = median_time(
                lambda: signed_plan.evaluate_many(pupils),
                args.repeats,
            )

            positive_plan, positive_build_s, _ = median_time(
                lambda: PreparedPositiveSeparableHarmonicDebyeWolfPlan.build(
                    args.nphi,
                    theta,
                    theta_weights,
                    rho_axis,
                    psi_axis,
                    z_axis,
                    k=k,
                    h_cutoff=h_cutoff,
                    backend=args.positive_backend,
                    cpp_threads=args.cpp_threads,
                ),
                args.repeats,
            )
            positive_single, positive_single_s, _ = median_time(
                lambda: positive_plan.evaluate(pupil),
                args.repeats,
            )
            positive_hot, positive_hot_s, _ = median_time(
                lambda: positive_plan.evaluate_many(pupils),
                args.repeats,
            )

            common = {
                "case": case,
                "rho_max": rho_max,
                "nrho": nrho,
                "npsi": args.npsi,
                "nz": args.nz,
                "targets": int(rho.size),
                "ntheta": args.ntheta,
                "nphi": args.nphi,
                "h_cutoff": h_cutoff,
                "h_margin": args.h_margin,
                "na": args.na,
                "n_medium": args.n_medium,
                "wavelength": args.wavelength,
                "z_max": args.z_max,
                "strength": args.strength,
                "sweeps": args.sweeps,
                "signed_backend": args.signed_backend,
                "positive_backend": args.positive_backend,
                "cpp_threads": args.cpp_threads,
                "finufft_build_s": finufft_build_s,
                "finufft_single_s": finufft_single_s,
                "finufft_hot_s": finufft_hot_s,
                "finufft_coordinate_mib": finufft_coordinate_mib,
            }

            for variant, plan, build_s, single_s, hot_s, single, hot in [
                (
                    "signed",
                    signed_plan,
                    signed_build_s,
                    signed_single_s,
                    signed_hot_s,
                    signed_single,
                    signed_hot,
                ),
                (
                    "positive",
                    positive_plan,
                    positive_build_s,
                    positive_single_s,
                    positive_hot_s,
                    positive_single,
                    positive_hot,
                ),
            ]:
                row = dict(common)
                row.update(
                    {
                        "variant": variant,
                        "signed_modes": signed_plan.used_modes,
                        "stored_abs_modes": None
                        if variant == "signed"
                        else positive_plan.stored_abs_modes,
                        "basis_mib": plan.basis_mib,
                        "build_s": build_s,
                        "single_s": single_s,
                        "hot_s": hot_s,
                        "hot_per_mask_s": hot_s / float(args.sweeps),
                        "build_plus_hot_s": build_s + hot_s,
                        "basis_reduction_vs_signed": speedup(
                            signed_plan.basis_mib,
                            plan.basis_mib,
                        ),
                        "build_speedup_vs_signed": speedup(signed_build_s, build_s),
                        "single_speedup_vs_signed": speedup(signed_single_s, single_s),
                        "hot_speedup_vs_signed": speedup(signed_hot_s, hot_s),
                        "build_plus_hot_speedup_vs_signed": speedup(
                            signed_build_s + signed_hot_s,
                            build_s + hot_s,
                        ),
                        "hot_speedup_vs_finufft": speedup(finufft_hot_s, hot_s),
                        "hot_speedup_vs_finufft_incl_build": speedup(
                            finufft_hot_s,
                            build_s + hot_s,
                        ),
                        "single_field_relative_l2_vs_signed": relative_l2(
                            single,
                            signed_single,
                        ),
                        "single_field_max_abs_over_signed": max_abs_over_ref(
                            single,
                            signed_single,
                        ),
                        "hot_field_relative_l2_vs_signed": relative_l2(
                            hot,
                            signed_hot,
                        ),
                        "single_field_relative_l2_vs_finufft": None
                        if finufft_single is None
                        else relative_l2(single, finufft_single),
                        "hot_field_relative_l2_vs_finufft": None
                        if finufft_hot is None
                        else relative_l2(hot, finufft_hot),
                    }
                )
                rows.append(row)

    payload = {
        "config": {
            "cases": cases,
            "rho_max_values": rho_max_values,
            "ntheta": args.ntheta,
            "nphi": args.nphi,
            "base_nrho": args.base_nrho,
            "base_rho_max": args.base_rho_max,
            "npsi": args.npsi,
            "nz": args.nz,
            "na": args.na,
            "n_medium": args.n_medium,
            "wavelength": args.wavelength,
            "h_margin": args.h_margin,
            "signed_backend": args.signed_backend,
            "positive_backend": args.positive_backend,
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
        finufft = row["hot_speedup_vs_finufft"]
        finufft_text = "NA" if finufft is None else f"{finufft:.2f}x"
        print(
            f"{row['case']:8s} rho={row['rho_max']:<4} {row['variant']:8s} "
            f"h={row['h_cutoff']} basis={row['basis_mib']:.3f}MiB "
            f"build={row['build_s']:.5f}s hot={row['hot_s']:.5f}s "
            f"vs_signed_hot={row['hot_speedup_vs_signed']:.2f}x "
            f"vs_finufft_hot={finufft_text} "
            f"l2_signed={row['single_field_relative_l2_vs_signed']}"
        )
    print(f"wrote {args.out}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
