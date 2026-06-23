from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
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
    median_time,
    relative_l2,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    direct_vectorial_debye_wolf,
    richards_wolf_jones_matrix,
    separable_vectorial_evaluate,
)


@dataclass(frozen=True)
class PackageCase:
    name: str
    gamma_deg: float
    beta_deg: float
    vortex_charge: int = 0


CASES = {
    "linear_x": PackageCase("linear_x", gamma_deg=0.0, beta_deg=0.0),
    "right_circular": PackageCase("right_circular", gamma_deg=45.0, beta_deg=90.0),
    "vortex_x": PackageCase("vortex_x", gamma_deg=0.0, beta_deg=0.0, vortex_charge=3),
}


def version_or_missing(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def pyfocus_mask(case: PackageCase):
    charge = int(case.vortex_charge)

    def mask(rho: float, phi: float, w0: float, f: float, k: float) -> complex:
        del rho, w0, f, k
        if charge == 0:
            return 1.0 + 0.0j
        return complex(np.exp(1j * charge * phi))

    return mask


def pupil_jones_for_case(
    case: PackageCase,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    gamma = np.deg2rad(case.gamma_deg)
    beta = np.deg2rad(case.beta_deg)
    phase = np.ones((theta.size, 1), dtype=np.complex128) * np.exp(
        1j * int(case.vortex_charge) * phi
    )[None, :]
    pupil = np.zeros((2, theta.size, phi.size), dtype=np.complex128)
    pupil[0] = np.cos(gamma) * phase
    pupil[1] = np.sin(gamma) * np.exp(1j * beta) * phase
    return pupil


def make_pyfocus_parameters(
    *,
    case: PackageCase,
    na: float,
    n_medium: float,
    h_mm: float,
    wavelength_vacuum_nm: float,
    x_range_nm: float,
    x_step_nm: float,
    z_step_nm: float,
    divisions_theta: int,
    divisions_phi: int,
):
    from PyFocus.custom_dataclasses.custom_mask import CustomMaskParameters
    from PyFocus.custom_dataclasses.field_parameters import (
        FieldParameters,
        PolarizationParameters,
    )
    from PyFocus.model.focus_field_calculators.base import FocusFieldCalculator

    field_parameters = FieldParameters(
        w0=1.0,
        wavelength=float(wavelength_vacuum_nm),
        I_0=1.0,
        polarization=PolarizationParameters(
            gamma=float(case.gamma_deg),
            beta=float(case.beta_deg),
        ),
    )
    focus_parameters = FocusFieldCalculator.FocusFieldParameters(
        NA=float(na),
        n=float(n_medium),
        h=float(h_mm),
        x_steps=float(x_step_nm),
        z_steps=float(z_step_nm),
        x_range=float(x_range_nm),
        z_range=float(2.0 * z_step_nm),
        z=0.0,
        phip=0.0,
        field_parameters=field_parameters,
        custom_mask_parameters=CustomMaskParameters(
            divisions_theta=int(divisions_theta),
            divisions_phi=int(divisions_phi),
        ),
    )
    focus_parameters.transform_input_parameter_units()
    return focus_parameters


def run_pyfocus_xy_once(
    *,
    case: PackageCase,
    na: float,
    n_medium: float,
    h_mm: float,
    wavelength_vacuum_nm: float,
    x_range_nm: float,
    x_step_nm: float,
    z_step_nm: float,
    divisions_theta: int,
    divisions_phi: int,
) -> dict[str, Any]:
    from PyFocus.model.focus_field_calculators.custom_mask import (
        CustomMaskFocusFieldCalculator,
    )

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
    calculator = CustomMaskFocusFieldCalculator()
    mask = pyfocus_mask(case)
    start = time.perf_counter()
    ex_lens, ey_lens = calculator._generate_rotated_incident_field(
        mask,
        focus_parameters,
    )
    ex, ey, ez = calculator._calculate_field_along_XY_plane(
        ex_lens,
        ey_lens,
        focus_parameters,
        verbose=False,
    )
    elapsed_s = time.perf_counter() - start
    count = int(focus_parameters.r_step_count)
    x_values = np.linspace(-x_range_nm / 2.0, x_range_nm / 2.0, count)
    y_values = np.linspace(x_range_nm / 2.0, -x_range_nm / 2.0, count)
    return {
        "fields": np.stack([ex, ey, ez], axis=0),
        "elapsed_s": elapsed_s,
        "x_values": x_values,
        "y_values": y_values,
        "wavelength_medium_nm": float(focus_parameters.field_parameters.wavelength),
        "focal_length_nm": float(focus_parameters.f),
        "theta_max": float(focus_parameters.alpha),
        "grid_n": count,
    }


def run_pyfocus_xy(
    *,
    repeats: int,
    **kwargs: Any,
) -> dict[str, Any]:
    outputs = [run_pyfocus_xy_once(**kwargs) for _ in range(repeats)]
    best = outputs[int(np.argsort([out["elapsed_s"] for out in outputs])[len(outputs) // 2])]
    best = dict(best)
    best["elapsed_s"] = float(median(out["elapsed_s"] for out in outputs))
    return best


def cartesian_targets(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x_values, y_values)
    rho = np.sqrt(xx * xx + yy * yy).ravel()
    psi = np.arctan2(yy, xx).ravel()
    z = np.zeros_like(rho)
    return rho, psi, z


def intensity(fields: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(fields) ** 2, axis=0)


def scale_fit_relative_l2(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float).ravel()
    cand = np.asarray(candidate, dtype=float).ravel()
    denom = float(np.dot(cand, cand))
    if denom == 0.0:
        return float("inf"), 0.0
    scale = float(np.dot(ref, cand) / denom)
    err = np.linalg.norm(scale * cand - ref) / max(np.linalg.norm(ref), 1e-300)
    return float(err), scale


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def peak_position(values: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    j, i = np.unravel_index(int(np.argmax(values)), values.shape)
    return float(x_values[i]), float(y_values[j])


def peak_radius(values: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> float:
    peak_x, peak_y = peak_position(values, x_values, y_values)
    return float(np.hypot(peak_x, peak_y))


def evaluate_ours_cartesian_direct(
    *,
    case: PackageCase,
    theta_max: float,
    wavelength_medium_nm: float,
    x_values: np.ndarray,
    y_values: np.ndarray,
    ntheta: int,
    nphi: int,
    repeats: int,
) -> tuple[np.ndarray, float]:
    theta, theta_weights = gauss_theta_grid(ntheta, theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False, dtype=float)
    pupil_jones = pupil_jones_for_case(case, theta, phi)
    mixing = richards_wolf_jones_matrix(theta, phi, apodization="sqrt-cos")
    rho, psi, z = cartesian_targets(x_values, y_values)
    k = 2.0 * np.pi / wavelength_medium_nm
    fields, elapsed_s, _ = median_time(
        lambda: direct_vectorial_debye_wolf(
            pupil_jones,
            mixing,
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
    return fields.reshape(3, y_values.size, x_values.size), float(elapsed_s)


def evaluate_ours_separable_polar(
    *,
    case: PackageCase,
    theta_max: float,
    wavelength_medium_nm: float,
    grid_n: int,
    rho_max_nm: float,
    ntheta: int,
    nphi: int,
    repeats: int,
    backend: str,
    cpp_threads: int,
) -> dict[str, Any]:
    theta, theta_weights = gauss_theta_grid(ntheta, theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False, dtype=float)
    pupil_jones = pupil_jones_for_case(case, theta, phi)
    mixing = richards_wolf_jones_matrix(theta, phi, apodization="sqrt-cos")
    rho_axis = np.linspace(0.0, rho_max_nm, grid_n, dtype=float)
    psi_axis = np.linspace(0.0, 2.0 * np.pi, grid_n, endpoint=False, dtype=float)
    z_axis = np.array([0.0], dtype=float)
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
        z_axis,
        k=k,
        h_cutoff=h_cutoff,
        backend=backend,
        cpp_threads=cpp_threads,
    )
    build_s = time.perf_counter() - build_start
    fields, hot_s, _ = median_time(
        lambda: separable_vectorial_evaluate(plan, pupil_jones, mixing),
        repeats,
    )
    return {
        "fields": fields.reshape(3, grid_n, grid_n, 1)[..., 0],
        "build_s": float(build_s),
        "hot_s": float(hot_s),
        "one_shot_s": float(build_s + hot_s),
        "h_cutoff": h_cutoff,
        "used_modes": int(plan.used_modes),
        "basis_mib": float(plan.basis_mib),
        "target_count": int(grid_n * grid_n),
    }


def case_metrics(
    *,
    case: PackageCase,
    pyfocus: dict[str, Any],
    ours_direct: np.ndarray,
    ours_direct_s: float,
    ours_separable: dict[str, Any],
) -> dict[str, Any]:
    py_i = intensity(pyfocus["fields"])
    direct_i = intensity(ours_direct)
    direct_l2, direct_scale = scale_fit_relative_l2(py_i, direct_i)
    direct_corr = pearson_correlation(py_i, direct_i)
    py_peak_x, py_peak_y = peak_position(py_i, pyfocus["x_values"], pyfocus["y_values"])
    direct_peak_x, direct_peak_y = peak_position(
        direct_i,
        pyfocus["x_values"],
        pyfocus["y_values"],
    )
    py_peak_radius = peak_radius(py_i, pyfocus["x_values"], pyfocus["y_values"])
    direct_peak_radius = peak_radius(direct_i, pyfocus["x_values"], pyfocus["y_values"])
    center = py_i.shape[0] // 2
    py_center = float(py_i[center, center] / max(float(np.max(py_i)), 1e-300))
    direct_center = float(
        direct_i[center, center] / max(float(np.max(direct_i)), 1e-300)
    )

    separable_i = intensity(ours_separable["fields"])
    separable_on_axis = float(separable_i[0, 0] / max(float(np.max(separable_i)), 1e-300))

    return {
        "case": case.name,
        "gamma_deg": case.gamma_deg,
        "beta_deg": case.beta_deg,
        "vortex_charge": case.vortex_charge,
        "package": "PyCustomFocus/PyFocus",
        "pyfocus_s": float(pyfocus["elapsed_s"]),
        "ours_cartesian_direct_s": float(ours_direct_s),
        "ours_separable_build_s": float(ours_separable["build_s"]),
        "ours_separable_hot_s": float(ours_separable["hot_s"]),
        "ours_separable_one_shot_s": float(ours_separable["one_shot_s"]),
        "speedup_direct_vs_pyfocus": float(pyfocus["elapsed_s"] / ours_direct_s),
        "speedup_separable_hot_vs_pyfocus": float(
            pyfocus["elapsed_s"] / ours_separable["hot_s"]
        ),
        "speedup_separable_one_shot_vs_pyfocus": float(
            pyfocus["elapsed_s"] / ours_separable["one_shot_s"]
        ),
        "pyfocus_grid_n": int(pyfocus["grid_n"]),
        "cartesian_targets": int(py_i.size),
        "separable_polar_targets": int(ours_separable["target_count"]),
        "intensity_shape_l2_pyfocus_vs_ours_direct": direct_l2,
        "intensity_shape_scale_ours_to_pyfocus": direct_scale,
        "intensity_pearson_pyfocus_vs_ours_direct": direct_corr,
        "pyfocus_center_over_peak": py_center,
        "ours_direct_center_over_peak": direct_center,
        "ours_separable_polar_on_axis_over_peak": separable_on_axis,
        "peak_offset_nm": float(
            np.hypot(py_peak_x - direct_peak_x, py_peak_y - direct_peak_y)
        ),
        "peak_radius_offset_nm": float(abs(py_peak_radius - direct_peak_radius)),
        "pyfocus_peak_x_nm": py_peak_x,
        "pyfocus_peak_y_nm": py_peak_y,
        "pyfocus_peak_radius_nm": py_peak_radius,
        "ours_direct_peak_x_nm": direct_peak_x,
        "ours_direct_peak_y_nm": direct_peak_y,
        "ours_direct_peak_radius_nm": direct_peak_radius,
        "separable_h_cutoff": int(ours_separable["h_cutoff"]),
        "separable_used_modes": int(ours_separable["used_modes"]),
        "separable_basis_mib": float(ours_separable["basis_mib"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# High-NA PyFocus vectorial package baseline",
        "",
        "This run compares PyFocus/PyCustomFocus against the local vectorial "
        "Richards-Wolf implementation. PyFocus is evaluated on its native "
        "Cartesian XY plane; the local separable solver is timed on an equal-size "
        "structured polar XY grid, so speedups are a structured-grid regime "
        "comparison rather than a point-for-point Cartesian replacement.",
        "",
        "## Configuration",
        "",
        f"- PyCustomFocus version: `{config['pycustomfocus_version']}`",
        f"- diffractio availability: `{config['diffractio_available']}`",
        f"- NA: `{config['na']}`",
        f"- medium index: `{config['n_medium']}`",
        f"- vacuum wavelength: `{config['wavelength_vacuum_nm']}` nm",
        f"- quadrature: `{config['ntheta']} x {config['nphi']}`",
        f"- repeats: `{config['repeats']}`",
        "",
        "## Results",
        "",
        "| case | shape L2 vs PyFocus | corr | peak-radius offset nm | PyFocus s | direct speedup | separable hot speedup | separable one-shot speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {l2:.3e} | {corr:.6f} | {radius:.3g} | {py:.4f} | {sd:.2f}x | {sh:.2f}x | {so:.2f}x |".format(
                case=row["case"],
                l2=row["intensity_shape_l2_pyfocus_vs_ours_direct"],
                corr=row["intensity_pearson_pyfocus_vs_ours_direct"],
                radius=row["peak_radius_offset_nm"],
                py=row["pyfocus_s"],
                sd=row["speedup_direct_vs_pyfocus"],
                sh=row["speedup_separable_hot_vs_pyfocus"],
                so=row["speedup_separable_one_shot_vs_pyfocus"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The accuracy column uses scale-fit normalized intensity, not raw complex components. "
            "PyFocus and the local implementation use different component phase/sign conventions.",
            "- Peak-radius offset is a more stable focal-pattern check than the single-pixel peak "
            "offset for annular or vortex rows because nearly degenerate ring maxima can swap "
            "angular position under image-axis and azimuth conventions.",
            "- The PyFocus timing is a domain-package direct Cartesian XY calculation with plotting disabled.",
            "- The separable timing is the current algorithm's favorable structured polar grid. "
            "It is the right regime for repeated coherent modes or optimization loops when the "
            "focal volume can be represented in cylindrical coordinates.",
            "- diffractio is recorded as an installed/not-installed package here, but it is not yet "
            "used as a matched Richards-Wolf focal-field baseline in this script.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_plots(output_prefix: Path, rows: list[dict[str, Any]], images: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    for row in rows:
        case = row["case"]
        py_i, direct_i = images[case]
        err, scale = scale_fit_relative_l2(py_i, direct_i)
        del err
        vmax = float(np.max(py_i))
        diff = np.abs(py_i - scale * direct_i)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
        for ax, data, title in [
            (axes[0], py_i, "PyFocus"),
            (axes[1], scale * direct_i, "Ours direct"),
            (axes[2], diff, "Abs diff"),
        ]:
            image = ax.imshow(data, origin="upper", cmap="magma")
            if title != "Abs diff":
                image.set_clim(0.0, vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(axes[0].images[0], ax=axes[:2], shrink=0.75)
        fig.colorbar(axes[2].images[0], ax=axes[2], shrink=0.75)
        fig.suptitle(case)
        fig.savefig(output_prefix.with_name(f"{output_prefix.name}_{case}.png"), dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local vectorial High-NA solver against PyFocus.",
    )
    parser.add_argument("--cases", nargs="+", default=["linear_x", "right_circular", "vortex_x"])
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_pyfocus_vectorial_package")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--na", type=float, default=0.95)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--h-mm", type=float, default=3.0)
    parser.add_argument("--wavelength-vacuum-nm", type=float, default=532.0)
    parser.add_argument("--x-range-nm", type=float, default=1200.0)
    parser.add_argument("--x-step-nm", type=float, default=50.0)
    parser.add_argument("--z-step-nm", type=float, default=100.0)
    parser.add_argument("--ntheta", type=int, default=48)
    parser.add_argument("--nphi", type=int, default=96)
    parser.add_argument("--direct-ntheta", type=int, default=None)
    parser.add_argument("--direct-nphi", type=int, default=None)
    parser.add_argument("--separable-backend", choices=["auto", "numpy", "cpp"], default="auto")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repeats = max(1, int(args.repeats))
    direct_ntheta = int(args.direct_ntheta or args.ntheta)
    direct_nphi = int(args.direct_nphi or args.nphi)
    selected_cases = [CASES[name] for name in args.cases]

    rows: list[dict[str, Any]] = []
    plot_images: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in selected_cases:
        pyfocus = run_pyfocus_xy(
            repeats=repeats,
            case=case,
            na=args.na,
            n_medium=args.n_medium,
            h_mm=args.h_mm,
            wavelength_vacuum_nm=args.wavelength_vacuum_nm,
            x_range_nm=args.x_range_nm,
            x_step_nm=args.x_step_nm,
            z_step_nm=args.z_step_nm,
            divisions_theta=args.ntheta,
            divisions_phi=args.nphi,
        )
        ours_direct, ours_direct_s = evaluate_ours_cartesian_direct(
            case=case,
            theta_max=pyfocus["theta_max"],
            wavelength_medium_nm=pyfocus["wavelength_medium_nm"],
            x_values=pyfocus["x_values"],
            y_values=pyfocus["y_values"],
            ntheta=direct_ntheta,
            nphi=direct_nphi,
            repeats=repeats,
        )
        rho_max_nm = float(np.sqrt(2.0) * args.x_range_nm / 2.0)
        ours_separable = evaluate_ours_separable_polar(
            case=case,
            theta_max=pyfocus["theta_max"],
            wavelength_medium_nm=pyfocus["wavelength_medium_nm"],
            grid_n=pyfocus["grid_n"],
            rho_max_nm=rho_max_nm,
            ntheta=direct_ntheta,
            nphi=direct_nphi,
            repeats=repeats,
            backend=args.separable_backend,
            cpp_threads=args.cpp_threads,
        )
        rows.append(
            case_metrics(
                case=case,
                pyfocus=pyfocus,
                ours_direct=ours_direct,
                ours_direct_s=ours_direct_s,
                ours_separable=ours_separable,
            )
        )
        plot_images[case.name] = (intensity(pyfocus["fields"]), intensity(ours_direct))

    config = {
        "pycustomfocus_version": version_or_missing("PyCustomFocus"),
        "pydantic_version": version_or_missing("pydantic"),
        "tqdm_version": version_or_missing("tqdm"),
        "diffractio_available": bool(importlib.util.find_spec("diffractio")),
        "na": args.na,
        "n_medium": args.n_medium,
        "h_mm": args.h_mm,
        "wavelength_vacuum_nm": args.wavelength_vacuum_nm,
        "x_range_nm": args.x_range_nm,
        "x_step_nm": args.x_step_nm,
        "ntheta": args.ntheta,
        "nphi": args.nphi,
        "direct_ntheta": direct_ntheta,
        "direct_nphi": direct_nphi,
        "repeats": repeats,
        "separable_backend": args.separable_backend,
        "cpp_threads": args.cpp_threads,
    }
    output_prefix = ROOT / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_json(output_prefix.with_suffix(".json"), config, rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), config, rows)
    if args.make_plots:
        save_plots(output_prefix, rows, plot_images)

    print(json.dumps({"config": config, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
