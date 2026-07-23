from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import meep as mp
import numpy as np


def bilinear_sample(
    values: np.ndarray,
    kx_axis: np.ndarray,
    ky_axis: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
) -> np.ndarray:
    dx = float(kx_axis[1] - kx_axis[0])
    dy = float(ky_axis[1] - ky_axis[0])
    fx = (kx - kx_axis[0]) / dx
    fy = (ky - ky_axis[0]) / dy
    ix = np.floor(fx).astype(int)
    iy = np.floor(fy).astype(int)
    ix = np.clip(ix, 0, kx_axis.size - 2)
    iy = np.clip(iy, 0, ky_axis.size - 2)
    tx = fx - ix
    ty = fy - iy
    return (
        (1.0 - tx) * (1.0 - ty) * values[ix, iy]
        + tx * (1.0 - ty) * values[ix + 1, iy]
        + (1.0 - tx) * ty * values[ix, iy + 1]
        + tx * ty * values[ix + 1, iy + 1]
    )


def analyze_spectrum(
    ex: np.ndarray,
    ey: np.ndarray,
    *,
    resolution: float,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
) -> dict[str, object]:
    nx, ny = ex.shape
    window = np.hanning(nx)[:, None] * np.hanning(ny)[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(window * ex))) ** 2
    spectrum += np.abs(np.fft.fftshift(np.fft.fft2(window * ey))) ** 2
    kx_axis = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=1.0 / resolution))
    ky_axis = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny, d=1.0 / resolution))
    dk = float(max(kx_axis[1] - kx_axis[0], ky_axis[1] - ky_axis[0]))
    angles_deg = np.linspace(5.0, 75.0, 29)
    angles = np.deg2rad(angles_deg)
    k0 = 2.0 * np.pi * frequency
    correct_radius = k0 / np.sqrt(
        np.cos(angles) ** 2 / epsilon_parallel
        + np.sin(angles) ** 2 / epsilon_perpendicular
    )
    sphere_radius = np.full_like(correct_radius, k0 * np.sqrt(epsilon_perpendicular))
    peak_radius = np.empty_like(correct_radius)
    correct_ridge = np.empty_like(correct_radius)
    sphere_ridge = np.empty_like(correct_radius)
    for index, angle in enumerate(angles):
        radial = np.linspace(0.82 * correct_radius[index], 1.18 * correct_radius[index], 401)
        kx = radial * np.cos(angle)
        ky = radial * np.sin(angle)
        profile = bilinear_sample(spectrum, kx_axis, ky_axis, kx, ky)
        peak_radius[index] = radial[int(np.argmax(profile))]
        correct_ridge[index] = bilinear_sample(
            spectrum,
            kx_axis,
            ky_axis,
            np.array([correct_radius[index] * np.cos(angle)]),
            np.array([correct_radius[index] * np.sin(angle)]),
        )[0]
        sphere_ridge[index] = bilinear_sample(
            spectrum,
            kx_axis,
            ky_axis,
            np.array([sphere_radius[index] * np.cos(angle)]),
            np.array([sphere_radius[index] * np.sin(angle)]),
        )[0]
    correct_error = float(np.linalg.norm(peak_radius - correct_radius) / np.linalg.norm(correct_radius))
    sphere_error = float(np.linalg.norm(peak_radius - sphere_radius) / np.linalg.norm(correct_radius))
    ridge_ratio = float(np.sum(correct_ridge) / np.sum(sphere_ridge))
    return {
        "angles_deg": angles_deg.tolist(),
        "spectral_bin_width": dk,
        "correct_radius": correct_radius.tolist(),
        "sphere_radius": sphere_radius.tolist(),
        "measured_peak_radius": peak_radius.tolist(),
        "correct_relative_l2": correct_error,
        "forced_sphere_relative_l2": sphere_error,
        "forced_to_correct_error_ratio": sphere_error / correct_error if correct_error > 0.0 else float("inf"),
        "correct_to_sphere_ridge_energy_ratio": ridge_ratio,
        "maximum_correct_sphere_separation_bins": float(np.max(np.abs(correct_radius - sphere_radius)) / dk),
    }


def run_resolution(
    resolution: float,
    *,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
    cell_width: float,
    pml_width: float,
    until_after_sources: float,
) -> tuple[dict[str, object], float]:
    material = mp.Medium(
        epsilon_diag=mp.Vector3(epsilon_perpendicular, epsilon_parallel, epsilon_perpendicular)
    )
    source_time = mp.GaussianSource(frequency=frequency, fwidth=0.25, is_integrated=True)
    sources = [
        mp.Source(source_time, component=mp.Ex, center=mp.Vector3(), amplitude=1.0),
        mp.Source(source_time, component=mp.Ey, center=mp.Vector3(), amplitude=0.35j),
    ]
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_width, cell_width, 0),
        dimensions=2,
        resolution=resolution,
        default_material=material,
        sources=sources,
        boundary_layers=[mp.PML(pml_width)],
        force_complex_fields=True,
        progress_interval=30,
    )
    monitor_width = cell_width - 2.0 * pml_width - 1.0
    monitor = sim.add_dft_fields(
        [mp.Ex, mp.Ey],
        frequency,
        0.0,
        1,
        where=mp.Volume(center=mp.Vector3(), size=mp.Vector3(monitor_width, monitor_width, 0)),
    )
    started = time.perf_counter()
    sim.run(until_after_sources=until_after_sources)
    ex = np.squeeze(np.asarray(sim.get_dft_array(monitor, mp.Ex, 0)))
    ey = np.squeeze(np.asarray(sim.get_dft_array(monitor, mp.Ey, 0)))
    elapsed = time.perf_counter() - started
    if ex.ndim != 2 or ey.shape != ex.shape:
        raise RuntimeError(f"unexpected DFT array shapes: Ex={ex.shape}, Ey={ey.shape}")
    metrics = analyze_spectrum(
        ex,
        ey,
        resolution=resolution,
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
    )
    metrics["resolution"] = float(resolution)
    metrics["dft_shape"] = list(ex.shape)
    sim.reset_meep()
    return metrics, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduced PyMeep uniaxial dispersion-ridge probe.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_dispersion_reduced_probe.json"),
    )
    parser.add_argument("--resolutions", default="12,16,20")
    parser.add_argument("--cell-width", type=float, default=24.0)
    parser.add_argument("--until-after-sources", type=float, default=20.0)
    args = parser.parse_args()

    frequency = 1.0
    n_o_sh = 2.319393
    n_e_sh = 2.224439
    epsilon_perpendicular = n_o_sh**2
    epsilon_parallel = n_e_sh**2
    pml_width = 2.0
    rows: list[dict[str, object]] = []
    runtimes: list[float] = []
    resolutions = [float(value) for value in args.resolutions.split(",")]
    for index, resolution in enumerate(resolutions, start=1):
        print(f"[{index}/{len(resolutions)}] resolution={resolution:g}", flush=True)
        metrics, elapsed = run_resolution(
            resolution,
            frequency=frequency,
            epsilon_perpendicular=epsilon_perpendicular,
            epsilon_parallel=epsilon_parallel,
            cell_width=args.cell_width,
            pml_width=pml_width,
            until_after_sources=args.until_after_sources,
        )
        rows.append(metrics)
        runtimes.append(elapsed)
    finest = rows[int(np.argmax(resolutions))]
    if len(rows) >= 2:
        next_finest = rows[int(np.argsort(resolutions)[-2])]
        finest_peaks = np.asarray(finest["measured_peak_radius"])
        next_peaks = np.asarray(next_finest["measured_peak_radius"])
        grid_difference = float(np.linalg.norm(finest_peaks - next_peaks) / np.linalg.norm(finest_peaks))
    else:
        grid_difference = float("nan")
    gates = {
        "correct_ellipse_relative_l2_le_2pct": finest["correct_relative_l2"] <= 0.02,
        "forced_sphere_error_ratio_ge_5": finest["forced_to_correct_error_ratio"] >= 5.0,
        "correct_ridge_energy_ge_sphere": finest["correct_to_sphere_ridge_energy_ratio"] >= 1.0,
        "curvature_separation_ge_2_bins": finest["maximum_correct_sphere_separation_bins"] >= 2.0,
        "three_level_peak_grid_convergence_le_2pct": len(rows) >= 3 and grid_difference <= 0.02,
    }
    result = {
        "schema": "uniaxial-meep-dispersion-reduced-probe-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "2-D homogeneous uniaxial DFT-field spatial-spectrum stop-rule probe; not the 3-D publication full-wave gate",
        "material": {
            "name": "5 mol% MgO:LiNbO3 at 532 nm",
            "n_o": n_o_sh,
            "n_e": n_e_sh,
            "epsilon_perpendicular": epsilon_perpendicular,
            "epsilon_parallel": epsilon_parallel,
        },
        "cell": {"width": args.cell_width, "pml_width": pml_width},
        "resolutions": resolutions,
        "runtime_seconds": runtimes,
        "rows": rows,
        "finest_next_finest_peak_l2": grid_difference,
        "gates": gates,
        "passed": all(gates.values()),
        "environment": {
            "meep": mp.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
