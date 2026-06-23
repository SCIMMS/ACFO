from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
(ROOT / ".matplotlib_cache").mkdir(exist_ok=True)

from benchmark_high_na_debye_wolf import (  # noqa: E402
    PreparedSeparableHarmonicDebyeWolfPlan,
    gauss_theta_grid,
)
from benchmark_high_na_pyfocus_vectorial_package import (  # noqa: E402
    CASES,
    make_pyfocus_parameters,
    pupil_jones_for_case,
    pyfocus_mask,
    scale_fit_relative_l2,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    richards_wolf_jones_matrix,
    separable_vectorial_evaluate,
)


def version_or_missing(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def timed(func, *, repeats: int, warmups: int = 0) -> tuple[Any, float, list[float]]:
    value = None
    for _ in range(max(0, warmups)):
        value = func()
    times: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed function did not run")
    return value, float(median(times)), times


def run_pyfocus_zstack_once(
    *,
    case_name: str,
    na: float,
    n_medium: float,
    h_mm: float,
    wavelength_vacuum_nm: float,
    x_range_nm: float,
    x_step_nm: float,
    z_step_nm: float,
    z_planes: int,
    divisions_theta: int,
    divisions_phi: int,
) -> dict[str, Any]:
    from PyFocus.model.focus_field_calculators.custom_mask import CustomMaskFocusFieldCalculator

    case = CASES[case_name]
    focus_parameters = make_pyfocus_parameters(
        case=case,
        na=na,
        n_medium=n_medium,
        h_mm=h_mm,
        wavelength_vacuum_nm=wavelength_vacuum_nm,
        x_range_nm=x_range_nm,
        x_step_nm=x_step_nm,
        z_step_nm=z_step_nm,
        divisions_theta=divisions_theta,
        divisions_phi=divisions_phi,
    )
    focus_parameters.z_range = float(2.0 * z_planes * z_step_nm)
    calculator = CustomMaskFocusFieldCalculator()
    mask = pyfocus_mask(case)
    ex_lens, ey_lens = calculator._generate_rotated_incident_field(mask, focus_parameters)
    z_axis = focus_parameters.z_steps * (np.arange(z_planes) - z_planes // 2)
    ex = np.zeros(
        (z_planes, focus_parameters.r_step_count, focus_parameters.r_step_count),
        dtype=np.complex128,
    )
    ey = np.zeros_like(ex)
    ez = np.zeros_like(ex)
    for iz, z_nm in enumerate(z_axis):
        focus_parameters.z = float(z_nm)
        ex[iz], ey[iz], ez[iz] = calculator._calculate_field_along_XY_plane(
            ex_lens,
            ey_lens,
            focus_parameters,
            verbose=False,
        )
    intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
    center = int(focus_parameters.r_step_count // 2)
    return {
        "field": np.stack([ex, ey, ez], axis=1),
        "intensity": intensity,
        "axis_profile": intensity[:, center, center],
        "grid_n": int(focus_parameters.r_step_count),
        "z_planes": int(z_planes),
        "z_axis_nm": np.asarray(z_axis, dtype=float),
        "theta_max": float(focus_parameters.alpha),
        "wavelength_medium_nm": float(focus_parameters.field_parameters.wavelength),
    }


def run_pyfocus_zstack(*, repeats: int, warmups: int, **kwargs: Any) -> tuple[dict[str, Any], float, list[float]]:
    return timed(lambda: run_pyfocus_zstack_once(**kwargs), repeats=repeats, warmups=warmups)


def run_local_separable_volume(
    *,
    case_name: str,
    theta_max: float,
    wavelength_medium_nm: float,
    x_range_nm: float,
    grid_n: int,
    z_axis_nm: np.ndarray,
    ntheta: int,
    nphi: int,
    repeats: int,
    backend: str,
    cpp_threads: int,
) -> dict[str, Any]:
    case = CASES[case_name]
    theta, theta_weights = gauss_theta_grid(ntheta, theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False, dtype=float)
    pupil_jones = pupil_jones_for_case(case, theta, phi)
    mixing = richards_wolf_jones_matrix(theta, phi, apodization="sqrt-cos")
    rho_max_nm = float(np.sqrt(2.0) * x_range_nm / 2.0)
    rho_axis = np.linspace(0.0, rho_max_nm, grid_n, dtype=float)
    psi_axis = np.linspace(0.0, 2.0 * np.pi, grid_n, endpoint=False, dtype=float)
    k = 2.0 * np.pi / wavelength_medium_nm
    h_cutoff = int(
        min(
            nphi // 2,
            max(0, np.ceil(k * rho_max_nm * np.sin(theta_max)) + abs(case.vortex_charge) + 4),
        )
    )
    build_start = time.perf_counter()
    plan = PreparedSeparableHarmonicDebyeWolfPlan.build(
        nphi,
        theta,
        theta_weights,
        rho_axis,
        psi_axis,
        z_axis_nm,
        k=k,
        h_cutoff=h_cutoff,
        backend=backend,
        cpp_threads=cpp_threads,
    )
    build_s = time.perf_counter() - build_start
    fields, hot_s, hot_times = timed(
        lambda: separable_vectorial_evaluate(plan, pupil_jones, mixing),
        repeats=repeats,
    )
    fields = fields.reshape(3, grid_n, grid_n, z_axis_nm.size)
    intensity = np.sum(np.abs(fields) ** 2, axis=0)
    axis_profile = np.mean(intensity[0, :, :], axis=0)
    return {
        "build_s": float(build_s),
        "hot_s": float(hot_s),
        "hot_times_s": hot_times,
        "one_shot_s": float(build_s + hot_s),
        "h_cutoff": int(h_cutoff),
        "used_modes": int(plan.used_modes),
        "basis_mib": float(plan.basis_mib),
        "target_count": int(grid_n * grid_n * z_axis_nm.size),
        "axis_profile": np.asarray(axis_profile, dtype=float),
    }


def write_summary(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# High-NA PyFocus z-stack repeated-mask sanity benchmark",
        "",
        "This benchmark exercises PyFocus/PyCustomFocus as a domain package on a small 3D z-stack. The local method is evaluated on an equal-size structured polar volume, so this is a repeated workload timing sanity check rather than a point-for-point Cartesian replacement claim.",
        "",
        "## Config",
        "",
        f"- PyCustomFocus: `{config['pycustomfocus_version']}`",
        f"- cases: `{', '.join(config['cases'])}`",
        f"- z planes: `{config['z_planes']}`",
        f"- x range / step: `{config['x_range_nm']}` / `{config['x_step_nm']}` nm",
        f"- z step: `{config['z_step_nm']}` nm",
        f"- quadrature: `{config['ntheta']} x {config['nphi']}`",
        "",
        "## Results",
        "",
        "| case | grid | z planes | PyFocus z-stack s | local build s | local hot s | local one-shot s | hot speedup | one-shot speedup | on-axis z-profile L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {grid} | {z} | {py} | {build} | {hot} | {one} | {hs}x | {os}x | {l2} |".format(
                case=row["case"],
                grid=row["grid_n"],
                z=row["z_planes"],
                py=fmt(row["pyfocus_zstack_s"]),
                build=fmt(row["ours_build_s"]),
                hot=fmt(row["ours_hot_s"]),
                one=fmt(row["ours_one_shot_s"]),
                hs=fmt(row["speedup_hot_vs_pyfocus"]),
                os=fmt(row["speedup_one_shot_vs_pyfocus"]),
                l2=fmt(row["on_axis_z_profile_l2_vs_pyfocus"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- PyFocus computes Cartesian XY slices for each z plane; the local timing is the structured polar-volume regime.",
            "- The on-axis z-profile L2 is a lightweight physical sanity check only. Full Cartesian-volume package replacement is not claimed here.",
            "- This is useful as a repeated package workload check because PyFocus does not batch multiple masks in the same way as the local harmonic plan.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if importlib.util.find_spec("PyFocus") is None:
        return vars(args).copy(), [{"status": "skipped", "skip_reason": "PyFocus is not installed"}]
    rows: list[dict[str, Any]] = []
    for case_name in args.cases:
        pyfocus, pyfocus_s, pyfocus_times = run_pyfocus_zstack(
            repeats=args.repeats,
            warmups=args.warmups,
            case_name=case_name,
            na=args.na,
            n_medium=args.n_medium,
            h_mm=args.h_mm,
            wavelength_vacuum_nm=args.wavelength_vacuum_nm,
            x_range_nm=args.x_range_nm,
            x_step_nm=args.x_step_nm,
            z_step_nm=args.z_step_nm,
            z_planes=args.z_planes,
            divisions_theta=args.ntheta,
            divisions_phi=args.nphi,
        )
        ours = run_local_separable_volume(
            case_name=case_name,
            theta_max=pyfocus["theta_max"],
            wavelength_medium_nm=pyfocus["wavelength_medium_nm"],
            x_range_nm=args.x_range_nm,
            grid_n=pyfocus["grid_n"],
            z_axis_nm=pyfocus["z_axis_nm"],
            ntheta=int(args.direct_ntheta or args.ntheta),
            nphi=int(args.direct_nphi or args.nphi),
            repeats=args.repeats,
            backend=args.separable_backend,
            cpp_threads=args.cpp_threads,
        )
        axis_l2, axis_scale = scale_fit_relative_l2(
            pyfocus["axis_profile"],
            ours["axis_profile"],
        )
        rows.append(
            {
                "status": "ok",
                "case": case_name,
                "package": "PyCustomFocus/PyFocus",
                "pycustomfocus_version": version_or_missing("PyCustomFocus"),
                "grid_n": int(pyfocus["grid_n"]),
                "z_planes": int(pyfocus["z_planes"]),
                "cartesian_targets": int(pyfocus["grid_n"] * pyfocus["grid_n"] * pyfocus["z_planes"]),
                "ours_structured_targets": int(ours["target_count"]),
                "pyfocus_zstack_s": float(pyfocus_s),
                "ours_build_s": float(ours["build_s"]),
                "ours_hot_s": float(ours["hot_s"]),
                "ours_one_shot_s": float(ours["one_shot_s"]),
                "speedup_hot_vs_pyfocus": float(pyfocus_s / ours["hot_s"]),
                "speedup_one_shot_vs_pyfocus": float(pyfocus_s / ours["one_shot_s"]),
                "on_axis_z_profile_l2_vs_pyfocus": float(axis_l2),
                "on_axis_z_profile_scale": float(axis_scale),
                "ours_h_cutoff": int(ours["h_cutoff"]),
                "ours_used_modes": int(ours["used_modes"]),
                "ours_basis_mib": float(ours["basis_mib"]),
                "pyfocus_times_s": " ".join(f"{value:.9g}" for value in pyfocus_times),
                "ours_hot_times_s": " ".join(f"{value:.9g}" for value in ours["hot_times_s"]),
            }
        )
    config = vars(args).copy()
    config.update(
        {
            "pycustomfocus_version": version_or_missing("PyCustomFocus"),
            "pydantic_version": version_or_missing("pydantic"),
            "diffractio_available": bool(importlib.util.find_spec("diffractio")),
        }
    )
    return config, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyFocus z-stack repeated workload sanity benchmark against the local structured polar-volume High-NA path."
    )
    parser.add_argument("--cases", nargs="+", default=["linear_x", "vortex_x"])
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_pyfocus_zstack_repeat")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--na", type=float, default=0.95)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--h-mm", type=float, default=3.0)
    parser.add_argument("--wavelength-vacuum-nm", type=float, default=532.0)
    parser.add_argument("--x-range-nm", type=float, default=1200.0)
    parser.add_argument("--x-step-nm", type=float, default=50.0)
    parser.add_argument("--z-step-nm", type=float, default=100.0)
    parser.add_argument("--z-planes", type=int, default=5)
    parser.add_argument("--ntheta", type=int, default=48)
    parser.add_argument("--nphi", type=int, default=96)
    parser.add_argument("--direct-ntheta", type=int, default=None)
    parser.add_argument("--direct-nphi", type=int, default=None)
    parser.add_argument("--separable-backend", choices=["auto", "numpy", "cpp"], default="auto")
    parser.add_argument("--cpp-threads", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, rows = run_case(args)
    output_prefix = ROOT / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_json(output_prefix.with_suffix(".json"), config, rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), config, rows)
    print(json.dumps({"config": config, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
