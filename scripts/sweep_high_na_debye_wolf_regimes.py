from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import numpy as np

from benchmark_high_na_debye_wolf import (
    PreparedFinufftDebyeWolfPlan,
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


def ordered_azimuthal_pupil(
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    theta_max: float,
    strength: float,
    order: int,
    apodization: str,
) -> np.ndarray:
    if order <= 0:
        raise ValueError("order must be positive")
    theta_2d = theta[:, None]
    phi_2d = phi[None, :]
    radial = np.sin(theta_2d) / max(np.sin(theta_max), np.finfo(float).eps)
    phase = strength * radial**2 * np.cos(float(order) * phi_2d)
    pupil = np.exp(1j * phase)
    if apodization == "none":
        return pupil
    if apodization == "sqrt-cos":
        return pupil * np.sqrt(np.cos(theta_2d))
    raise ValueError(f"unknown apodization: {apodization}")


def make_pupil_sequence(
    case: str,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    theta_max: float,
    strength: float,
    sweeps: int,
    vortex_charge: int,
    apodization: str,
    phase_order: int,
) -> list[np.ndarray]:
    strengths = sweep_strengths(strength, sweeps)
    if case == "ordered":
        return [
            ordered_azimuthal_pupil(
                theta,
                phi,
                theta_max=theta_max,
                strength=value,
                order=phase_order,
                apodization=apodization,
            )
            for value in strengths
        ]
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
        for value in strengths
    ]


def auto_h_cutoff(
    *,
    k: float,
    rho_max: float,
    na: float,
    n_medium: float,
    margin: int,
    nphi: int,
) -> int:
    cutoff = int(np.ceil(k * rho_max * na / n_medium + margin))
    return max(0, min(cutoff, nphi // 2))


def auto_nrho(rho_max: float, base_rho_max: float, base_nrho: int) -> int:
    if base_nrho <= 1:
        return base_nrho
    drho = base_rho_max / float(base_nrho - 1)
    return max(2, int(round(rho_max / drho)) + 1)


def row_for_case(
    *,
    sweep_name: str,
    sweep_value: float | int | str,
    case: str,
    phase_order: int,
    ntheta: int,
    nphi: int,
    nrho: int,
    npsi: int,
    nz: int,
    rho_max: float,
    z_max: float,
    wavelength: float,
    na: float,
    n_medium: float,
    strength: float,
    sweeps: int,
    vortex_charge: int,
    apodization: str,
    h_cutoff: int,
    finufft_eps: float,
    repeats: int,
    harmonic_backend: str,
    cpp_threads: int,
) -> dict[str, object]:
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
    rho, psi, z = flatten_focal_axes(rho_axis, psi_axis, z_axis)
    k = 2.0 * np.pi * n_medium / wavelength

    pupils = make_pupil_sequence(
        case,
        theta,
        phi,
        theta_max=theta_max,
        strength=strength,
        sweeps=sweeps,
        vortex_charge=vortex_charge,
        apodization=apodization,
        phase_order=phase_order,
    )
    pupil = pupils[len(pupils) // 2]

    finufft_plan, finufft_build_s, finufft_build_times = median_time(
        lambda: PreparedFinufftDebyeWolfPlan.build(
            theta,
            theta_weights,
            phi,
            rho,
            psi,
            z,
            k=k,
        ),
        repeats,
    )
    finufft_single, finufft_single_s, finufft_single_times = median_time(
        lambda: finufft_plan.evaluate(pupil, eps=finufft_eps),
        repeats,
    )
    finufft_hot, finufft_hot_s, finufft_hot_times = median_time(
        lambda: finufft_plan.evaluate_many(pupils, eps=finufft_eps),
        repeats,
    )

    harmonic_plan, harmonic_build_s, harmonic_build_times = median_time(
        lambda: PreparedSeparableHarmonicDebyeWolfPlan.build(
            nphi,
            theta,
            theta_weights,
            rho_axis,
            psi_axis,
            z_axis,
            k=k,
            h_cutoff=h_cutoff,
            backend=harmonic_backend,
            cpp_threads=cpp_threads,
        ),
        repeats,
    )
    harmonic_single, harmonic_single_s, harmonic_single_times = median_time(
        lambda: harmonic_plan.evaluate(pupil),
        repeats,
    )
    harmonic_hot, harmonic_hot_s, harmonic_hot_times = median_time(
        lambda: harmonic_plan.evaluate_many(pupils),
        repeats,
    )

    intensity_harmonic = np.abs(harmonic_single) ** 2
    intensity_finufft = np.abs(finufft_single) ** 2
    hot_intensity_harmonic = np.abs(harmonic_hot) ** 2
    hot_intensity_finufft = np.abs(finufft_hot) ** 2

    return {
        "sweep": sweep_name,
        "sweep_value": sweep_value,
        "case": case,
        "phase_order": phase_order,
        "h_cutoff": h_cutoff,
        "used_modes": harmonic_plan.used_modes,
        "ntheta": ntheta,
        "nphi": nphi,
        "nrho": nrho,
        "npsi": npsi,
        "nz": nz,
        "targets": int(rho.size),
        "rho_max": rho_max,
        "z_max": z_max,
        "wavelength": wavelength,
        "na": na,
        "n_medium": n_medium,
        "strength": strength,
        "sweeps": sweeps,
        "finufft_eps": finufft_eps,
        "harmonic_backend": harmonic_backend,
        "cpp_threads": cpp_threads,
        "finufft_sources": finufft_plan.sources,
        "finufft_targets": finufft_plan.targets,
        "finufft_coordinate_mib": finufft_plan.coordinate_mib,
        "harmonic_basis_mib": harmonic_plan.basis_mib,
        "finufft_build_s": finufft_build_s,
        "harmonic_build_s": harmonic_build_s,
        "finufft_single_s": finufft_single_s,
        "harmonic_single_s": harmonic_single_s,
        "finufft_hot_s": finufft_hot_s,
        "harmonic_hot_s": harmonic_hot_s,
        "finufft_hot_per_mask_s": finufft_hot_s / float(sweeps),
        "harmonic_hot_per_mask_s": harmonic_hot_s / float(sweeps),
        "single_speedup_harmonic_vs_finufft": finufft_single_s / harmonic_single_s
        if harmonic_single_s > 0.0
        else None,
        "hot_speedup_harmonic_vs_finufft": finufft_hot_s / harmonic_hot_s
        if harmonic_hot_s > 0.0
        else None,
        "hot_speedup_including_harmonic_build": finufft_hot_s
        / (harmonic_build_s + harmonic_hot_s)
        if harmonic_build_s + harmonic_hot_s > 0.0
        else None,
        "field_relative_l2_vs_finufft": relative_l2(harmonic_single, finufft_single),
        "field_max_abs_over_finufft": max_abs_over_ref(harmonic_single, finufft_single),
        "intensity_relative_l2_vs_finufft": relative_l2(
            intensity_harmonic,
            intensity_finufft,
        ),
        "hot_field_relative_l2_vs_finufft": relative_l2(harmonic_hot, finufft_hot),
        "hot_intensity_relative_l2_vs_finufft": relative_l2(
            hot_intensity_harmonic,
            hot_intensity_finufft,
        ),
        "finufft_build_times_s": finufft_build_times,
        "harmonic_build_times_s": harmonic_build_times,
        "finufft_single_times_s": finufft_single_times,
        "harmonic_single_times_s": harmonic_single_times,
        "finufft_hot_times_s": finufft_hot_times,
        "harmonic_hot_times_s": harmonic_hot_times,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep scalar high-NA Debye-Wolf regimes against FINUFFT."
    )
    parser.add_argument("--ntheta", type=int, default=48)
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--base-nrho", type=int, default=9)
    parser.add_argument("--npsi", type=int, default=48)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--base-rho-max", type=float, default=0.75)
    parser.add_argument("--z-max", type=float, default=0.8)
    parser.add_argument("--wavelength", type=float, default=0.532)
    parser.add_argument("--na", type=float, default=0.8)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--vortex-charge", type=int, default=1)
    parser.add_argument("--apodization", choices=["none", "sqrt-cos"], default="none")
    parser.add_argument("--case", default="mixed")
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--h-margin", type=int, default=8)
    parser.add_argument("--finufft-eps", type=float, default=1e-9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--harmonic-backend",
        choices=["auto", "numpy", "cpp"],
        default="auto",
    )
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/high_na_debye_wolf_regime_sweep.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results/high_na_debye_wolf_regime_sweep.csv"),
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.sweeps <= 0:
        raise ValueError("sweeps must be positive")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if not (0.0 < args.na <= args.n_medium):
        raise ValueError("NA must satisfy 0 < NA <= n-medium")

    k_default = 2.0 * np.pi * args.n_medium / args.wavelength
    default_h = auto_h_cutoff(
        k=k_default,
        rho_max=args.base_rho_max,
        na=args.na,
        n_medium=args.n_medium,
        margin=args.h_margin,
        nphi=args.nphi,
    )

    rows: list[dict[str, object]] = []

    for rho_max in [0.25, 0.5, 0.75, 1.0, 1.5]:
        h_cutoff = auto_h_cutoff(
            k=k_default,
            rho_max=rho_max,
            na=args.na,
            n_medium=args.n_medium,
            margin=args.h_margin,
            nphi=args.nphi,
        )
        rows.append(
            row_for_case(
                sweep_name="rho_max_fixed_drho",
                sweep_value=rho_max,
                case=args.case,
                phase_order=1,
                ntheta=args.ntheta,
                nphi=args.nphi,
                nrho=auto_nrho(rho_max, args.base_rho_max, args.base_nrho),
                npsi=args.npsi,
                nz=args.nz,
                rho_max=rho_max,
                z_max=args.z_max,
                wavelength=args.wavelength,
                na=args.na,
                n_medium=args.n_medium,
                strength=args.strength,
                sweeps=args.sweeps,
                vortex_charge=args.vortex_charge,
                apodization=args.apodization,
                h_cutoff=h_cutoff,
                finufft_eps=args.finufft_eps,
                repeats=args.repeats,
                harmonic_backend=args.harmonic_backend,
                cpp_threads=args.cpp_threads,
            )
        )

    for na in [0.4, 0.6, 0.8, 0.95]:
        h_cutoff = auto_h_cutoff(
            k=k_default,
            rho_max=args.base_rho_max,
            na=na,
            n_medium=args.n_medium,
            margin=args.h_margin,
            nphi=args.nphi,
        )
        rows.append(
            row_for_case(
                sweep_name="na",
                sweep_value=na,
                case=args.case,
                phase_order=1,
                ntheta=args.ntheta,
                nphi=args.nphi,
                nrho=args.base_nrho,
                npsi=args.npsi,
                nz=args.nz,
                rho_max=args.base_rho_max,
                z_max=args.z_max,
                wavelength=args.wavelength,
                na=na,
                n_medium=args.n_medium,
                strength=args.strength,
                sweeps=args.sweeps,
                vortex_charge=args.vortex_charge,
                apodization=args.apodization,
                h_cutoff=h_cutoff,
                finufft_eps=args.finufft_eps,
                repeats=args.repeats,
                harmonic_backend=args.harmonic_backend,
                cpp_threads=args.cpp_threads,
            )
        )

    for sweep_count in [1, 2, 4, 8, 16, 32]:
        rows.append(
            row_for_case(
                sweep_name="mask_sweeps",
                sweep_value=sweep_count,
                case=args.case,
                phase_order=1,
                ntheta=args.ntheta,
                nphi=args.nphi,
                nrho=args.base_nrho,
                npsi=args.npsi,
                nz=args.nz,
                rho_max=args.base_rho_max,
                z_max=args.z_max,
                wavelength=args.wavelength,
                na=args.na,
                n_medium=args.n_medium,
                strength=args.strength,
                sweeps=sweep_count,
                vortex_charge=args.vortex_charge,
                apodization=args.apodization,
                h_cutoff=default_h,
                finufft_eps=args.finufft_eps,
                repeats=args.repeats,
                harmonic_backend=args.harmonic_backend,
                cpp_threads=args.cpp_threads,
            )
        )

    for order in [1, 2, 4, 8, 12, 16]:
        rows.append(
            row_for_case(
                sweep_name="pupil_phase_order_auto_h",
                sweep_value=order,
                case="ordered",
                phase_order=order,
                ntheta=args.ntheta,
                nphi=args.nphi,
                nrho=args.base_nrho,
                npsi=args.npsi,
                nz=args.nz,
                rho_max=args.base_rho_max,
                z_max=args.z_max,
                wavelength=args.wavelength,
                na=args.na,
                n_medium=args.n_medium,
                strength=args.strength,
                sweeps=args.sweeps,
                vortex_charge=args.vortex_charge,
                apodization=args.apodization,
                h_cutoff=default_h,
                finufft_eps=args.finufft_eps,
                repeats=args.repeats,
                harmonic_backend=args.harmonic_backend,
                cpp_threads=args.cpp_threads,
            )
        )
        rows.append(
            row_for_case(
                sweep_name="pupil_phase_order_full_h",
                sweep_value=order,
                case="ordered",
                phase_order=order,
                ntheta=args.ntheta,
                nphi=args.nphi,
                nrho=args.base_nrho,
                npsi=args.npsi,
                nz=args.nz,
                rho_max=args.base_rho_max,
                z_max=args.z_max,
                wavelength=args.wavelength,
                na=args.na,
                n_medium=args.n_medium,
                strength=args.strength,
                sweeps=args.sweeps,
                vortex_charge=args.vortex_charge,
                apodization=args.apodization,
                h_cutoff=args.nphi // 2,
                finufft_eps=args.finufft_eps,
                repeats=args.repeats,
                harmonic_backend=args.harmonic_backend,
                cpp_threads=args.cpp_threads,
            )
        )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {"config": config, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.csv, rows)

    print(f"wrote {args.out}")
    print(f"wrote {args.csv}")
    for row in rows:
        print(
            "{sweep:28s} value={sweep_value!s:>5s} h={h_cutoff:2d} "
            "targets={targets:5d} modes={used_modes:3d} "
            "err={field_relative_l2_vs_finufft:.2e} "
            "single={single_speedup_harmonic_vs_finufft:.2f}x "
            "hot={hot_speedup_harmonic_vs_finufft:.2f}x "
            "incl_build={hot_speedup_including_harmonic_build:.2f}x".format(**row)
        )


if __name__ == "__main__":
    main()
