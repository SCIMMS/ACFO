import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_high_na_debye_wolf import (
    PositiveRhoDependentBasisCache,
    PreparedCachedPositiveRhoDependentHarmonicDebyeWolfPlan,
    PreparedPositiveRhoDependentHarmonicDebyeWolfPlan,
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
        raise ValueError("expected at least one float")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("integer values must be positive")
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
        description="Benchmark reusable high-NA positive-rho basis build caches."
    )
    parser.add_argument("--cases", default="mixed")
    parser.add_argument("--ntheta", type=int, default=48)
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--base-nrho", type=int, default=9)
    parser.add_argument("--base-rho-max", type=float, default=0.75)
    parser.add_argument("--rho-max-values", default="1.5,2.25,3.0")
    parser.add_argument("--npsi", type=int, default=96)
    parser.add_argument("--nz", type=int, default=9)
    parser.add_argument("--z-max", type=float, default=0.8)
    parser.add_argument("--wavelength", type=float, default=0.532)
    parser.add_argument("--na", type=float, default=0.8)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--vortex-charge", type=int, default=1)
    parser.add_argument("--apodization", choices=["none", "sqrt-cos"], default="none")
    parser.add_argument("--rho-margin", type=int, default=8)
    parser.add_argument("--cutoff-bin-sizes", default="4,8,16")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/high_na_build_cache.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results/high_na_build_cache.csv"),
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

            cache, cache_build_s, _ = median_time(
                lambda: PositiveRhoDependentBasisCache.build(
                    args.nphi,
                    theta,
                    theta_weights,
                    rho_axis,
                    psi_axis,
                    z_axis,
                    k=k,
                    h_cutoff=h_cutoff,
                    sin_theta_max=sin_theta_max,
                ),
                args.repeats,
            )

            first_uncached_total = 0.0
            first_cached_total = cache_build_s
            max_single_l2 = 0.0
            for cutoff_bin_size in cutoff_bin_sizes:
                uncached_plan, uncached_build_s, _ = median_time(
                    lambda: PreparedPositiveRhoDependentHarmonicDebyeWolfPlan.build(
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
                        backend="cpp",
                        cpp_threads=args.cpp_threads,
                        no_copy_coefficients=True,
                    ),
                    args.repeats,
                )
                uncached_single, uncached_single_s, _ = median_time(
                    lambda: uncached_plan.evaluate(pupil),
                    args.repeats,
                )
                uncached_hot, uncached_hot_s, _ = median_time(
                    lambda: uncached_plan.evaluate_many(pupils),
                    args.repeats,
                )

                cached_plan, cached_plan_build_s, _ = median_time(
                    lambda: PreparedCachedPositiveRhoDependentHarmonicDebyeWolfPlan.build(
                        cache,
                        margin=args.rho_margin,
                        cutoff_bin_size=cutoff_bin_size,
                        backend="cpp",
                        cpp_threads=args.cpp_threads,
                    ),
                    args.repeats,
                )
                cached_single, cached_single_s, _ = median_time(
                    lambda: cached_plan.evaluate(pupil),
                    args.repeats,
                )
                cached_hot, cached_hot_s, _ = median_time(
                    lambda: cached_plan.evaluate_many(pupils),
                    args.repeats,
                )

                single_l2 = relative_l2(cached_single, uncached_single)
                hot_l2 = relative_l2(cached_hot, uncached_hot)
                max_single_l2 = max(max_single_l2, single_l2)
                first_uncached_total += uncached_build_s + uncached_hot_s
                first_cached_total += cached_plan_build_s + cached_hot_s
                common = {
                    "case": case,
                    "rho_max": rho_max,
                    "nrho": nrho,
                    "npsi": args.npsi,
                    "nz": args.nz,
                    "targets": int(nrho * args.npsi * args.nz),
                    "ntheta": args.ntheta,
                    "nphi": args.nphi,
                    "h_cutoff": h_cutoff,
                    "rho_margin": args.rho_margin,
                    "cutoff_bin_size": cutoff_bin_size,
                    "sweeps": args.sweeps,
                    "cache_build_s": cache_build_s,
                    "cache_basis_mib": cache.basis_mib,
                }
                rows.append(
                    {
                        **common,
                        "variant": "uncached_positive_rho_nocopy",
                        "groups": uncached_plan.group_count,
                        "basis_mib": uncached_plan.basis_mib,
                        "plan_build_s": uncached_build_s,
                        "single_s": uncached_single_s,
                        "hot_s": uncached_hot_s,
                        "first_total_s": uncached_build_s + uncached_hot_s,
                        "amortized_total_s": uncached_build_s + uncached_hot_s,
                        "build_speedup_vs_uncached": 1.0,
                        "hot_speedup_vs_uncached": 1.0,
                        "first_total_speedup_vs_uncached": 1.0,
                        "single_field_relative_l2_vs_uncached": 0.0,
                        "hot_field_relative_l2_vs_uncached": 0.0,
                        "single_field_max_abs_over_uncached": 0.0,
                    }
                )
                rows.append(
                    {
                        **common,
                        "variant": "cached_positive_rho_nocopy",
                        "groups": cached_plan.group_count,
                        "basis_mib": cached_plan.basis_mib,
                        "plan_build_s": cached_plan_build_s,
                        "single_s": cached_single_s,
                        "hot_s": cached_hot_s,
                        "first_total_s": cache_build_s
                        + cached_plan_build_s
                        + cached_hot_s,
                        "amortized_total_s": cached_plan_build_s + cached_hot_s,
                        "build_speedup_vs_uncached": speedup(
                            uncached_build_s,
                            cached_plan_build_s,
                        ),
                        "hot_speedup_vs_uncached": speedup(uncached_hot_s, cached_hot_s),
                        "first_total_speedup_vs_uncached": speedup(
                            uncached_build_s + uncached_hot_s,
                            cache_build_s + cached_plan_build_s + cached_hot_s,
                        ),
                        "single_field_relative_l2_vs_uncached": single_l2,
                        "hot_field_relative_l2_vs_uncached": hot_l2,
                        "single_field_max_abs_over_uncached": max_abs_over_ref(
                            cached_single,
                            uncached_single,
                        ),
                    }
                )

            rows.append(
                {
                    "case": case,
                    "rho_max": rho_max,
                    "nrho": nrho,
                    "npsi": args.npsi,
                    "nz": args.nz,
                    "targets": int(nrho * args.npsi * args.nz),
                    "ntheta": args.ntheta,
                    "nphi": args.nphi,
                    "h_cutoff": h_cutoff,
                    "rho_margin": args.rho_margin,
                    "cutoff_bin_size": "all",
                    "sweeps": args.sweeps,
                    "cache_build_s": cache_build_s,
                    "cache_basis_mib": cache.basis_mib,
                    "variant": "cached_all_bins_summary",
                    "groups": None,
                    "basis_mib": cache.basis_mib,
                    "plan_build_s": None,
                    "single_s": None,
                    "hot_s": None,
                    "first_total_s": first_cached_total,
                    "amortized_total_s": first_cached_total - cache_build_s,
                    "build_speedup_vs_uncached": None,
                    "hot_speedup_vs_uncached": None,
                    "first_total_speedup_vs_uncached": speedup(
                        first_uncached_total,
                        first_cached_total,
                    ),
                    "single_field_relative_l2_vs_uncached": max_single_l2,
                    "hot_field_relative_l2_vs_uncached": None,
                    "single_field_max_abs_over_uncached": None,
                }
            )

    payload = {
        "config": {
            "cases": cases,
            "rho_max_values": rho_max_values,
            "cutoff_bin_sizes": cutoff_bin_sizes,
            "ntheta": args.ntheta,
            "nphi": args.nphi,
            "base_nrho": args.base_nrho,
            "base_rho_max": args.base_rho_max,
            "npsi": args.npsi,
            "nz": args.nz,
            "na": args.na,
            "n_medium": args.n_medium,
            "wavelength": args.wavelength,
            "rho_margin": args.rho_margin,
            "cpp_threads": args.cpp_threads,
            "sweeps": args.sweeps,
            "repeats": args.repeats,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.csv, rows)

    for row in rows:
        if row["variant"] == "cached_all_bins_summary":
            print(
                f"{row['case']:8s} rho={row['rho_max']:<4} all_bins "
                f"first_total={row['first_total_s']:.5f}s "
                f"speedup={row['first_total_speedup_vs_uncached']:.2f}x "
                f"cache={row['cache_build_s']:.5f}s"
            )
            continue
        print(
            f"{row['case']:8s} rho={row['rho_max']:<4} "
            f"{row['variant']:28s} bin={row['cutoff_bin_size']} "
            f"build={row['plan_build_s']:.5f}s hot={row['hot_s']:.5f}s "
            f"first={row['first_total_s']:.5f}s "
            f"l2={row['single_field_relative_l2_vs_uncached']}"
        )
    print(f"wrote {args.out}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
