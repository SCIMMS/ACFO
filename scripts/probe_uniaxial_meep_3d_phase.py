from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import meep as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import linbo3_3m_nonlinear_polarization  # noqa: E402


def fixed_source_grid(
    *, n: int, half_width: float, frequency: float, epsilon_perpendicular: float, epsilon_parallel: float
) -> np.ndarray:
    spacing = 2.0 * half_width / n
    axis = -half_width + (np.arange(n) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    ux = x / half_width
    uy = y / half_width
    uz = z / half_width
    mask = (ux / 0.82) ** 4 + (uy / 0.70) ** 4 + (uz / 0.90) ** 4 <= 1.0
    hologram = np.cos(2.3 * ux + 1.1 * uy - 0.7 * uz) + 0.55 * np.cos(
        -0.8 * ux + 2.0 * uy + 1.3 * uz + 0.4
    )
    domains = np.where(hologram >= 0.0, 1.0, -1.0)
    ray_angle = np.deg2rad(38.0)
    k0 = 2.0 * np.pi * frequency
    phase_slope = k0 * np.sqrt(
        epsilon_parallel * np.sin(ray_angle) ** 2
        + epsilon_perpendicular * np.cos(ray_angle) ** 2
    )
    carrier = np.exp(
        1j * phase_slope * (np.sin(ray_angle) * x + np.cos(ray_angle) * z)
    )
    return np.where(mask, domains * carrier, 0.0).astype(np.complex128)


def grid_amplitude(grid: np.ndarray, *, half_width: float):
    n = grid.shape[0]

    def amplitude(point: mp.Vector3) -> complex:
        scaled = np.array((point.x, point.y, point.z), dtype=np.float64)
        scaled = (scaled + half_width) * (n / (2.0 * half_width))
        index = np.floor(scaled).astype(int)
        if np.any(index < 0) or np.any(index >= n):
            return 0.0j
        return complex(grid[index[0], index[1], index[2]])

    return amplitude


def phase_slope(fields: np.ndarray, radii: np.ndarray, components: tuple[int, ...]) -> tuple[float, float]:
    selected = fields[:, components]
    cross = np.einsum("rc,rc->r", np.conjugate(selected[:-1]), selected[1:])
    weights = np.sqrt(
        np.sum(np.abs(selected[:-1]) ** 2, axis=1)
        * np.sum(np.abs(selected[1:]) ** 2, axis=1)
    )
    increments = np.angle(cross)
    dr = np.diff(radii)
    local = increments / dr
    valid = np.isfinite(local) & (weights > 1e-20)
    if not np.any(valid):
        return float("nan"), 0.0
    slope = float(np.sum(weights[valid] * local[valid]) / np.sum(weights[valid]))
    scatter = float(
        np.sqrt(
            np.sum(weights[valid] * (local[valid] - slope) ** 2)
            / np.sum(weights[valid])
        )
    )
    return abs(slope), scatter


def run_resolution(
    resolution: float,
    *,
    cell_width: float,
    pml_width: float,
    source_half_width: float,
    source_grid_n: int,
    until_after_sources: float,
    angles_deg: np.ndarray,
    radii: np.ndarray,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
) -> tuple[dict[str, object], float]:
    material = mp.Medium(
        epsilon_diag=mp.Vector3(
            epsilon_perpendicular,
            epsilon_perpendicular,
            epsilon_parallel,
        )
    )
    grid = fixed_source_grid(
        n=source_grid_n,
        half_width=source_half_width,
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
    )
    amplitude = grid_amplitude(grid, half_width=source_half_width)
    pump = np.array((0.25, 0.35, np.sqrt(1.0 - 0.25**2 - 0.35**2)), dtype=np.complex128)
    source_vector = linbo3_3m_nonlinear_polarization(pump)
    source_vector /= np.linalg.norm(source_vector)
    source_time = mp.GaussianSource(frequency=frequency, fwidth=0.30, is_integrated=True)
    components = (mp.Ex, mp.Ey, mp.Ez)
    sources = [
        mp.Source(
            source_time,
            component=component,
            center=mp.Vector3(),
            size=mp.Vector3(2.0 * source_half_width, 2.0 * source_half_width, 2.0 * source_half_width),
            amplitude=complex(source_vector[index]),
            amp_func=amplitude,
        )
        for index, component in enumerate(components)
    ]
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_width, cell_width, cell_width),
        resolution=resolution,
        default_material=material,
        sources=sources,
        boundary_layers=[mp.PML(pml_width)],
        force_complex_fields=True,
        progress_interval=30,
    )
    monitors: list[tuple[int, int, object]] = []
    for ia, angle_deg in enumerate(angles_deg):
        angle = np.deg2rad(angle_deg)
        for ir, radius in enumerate(radii):
            point = mp.Vector3(radius * np.sin(angle), 0.0, radius * np.cos(angle))
            monitor = sim.add_dft_fields(
                list(components),
                frequency,
                0.0,
                1,
                where=mp.Volume(center=point, size=mp.Vector3()),
            )
            monitors.append((ia, ir, monitor))
    started = time.perf_counter()
    sim.run(until_after_sources=until_after_sources)
    fields = np.empty((angles_deg.size, radii.size, 3), dtype=np.complex128)
    for ia, ir, monitor in monitors:
        for ic, component in enumerate(components):
            value = np.asarray(sim.get_dft_array(monitor, component, 0)).squeeze()
            fields[ia, ir, ic] = complex(value.item())
    elapsed = time.perf_counter() - started
    extraordinary = np.empty(angles_deg.size, dtype=np.float64)
    extraordinary_scatter = np.empty_like(extraordinary)
    ordinary = np.empty_like(extraordinary)
    ordinary_scatter = np.empty_like(extraordinary)
    for ia in range(angles_deg.size):
        extraordinary[ia], extraordinary_scatter[ia] = phase_slope(fields[ia], radii, (0, 2))
        ordinary[ia], ordinary_scatter[ia] = phase_slope(fields[ia], radii, (1,))
    angle = np.deg2rad(angles_deg)
    k0 = 2.0 * np.pi * frequency
    ellipse = k0 * np.sqrt(
        epsilon_parallel * np.sin(angle) ** 2
        + epsilon_perpendicular * np.cos(angle) ** 2
    )
    sphere = np.full_like(ellipse, k0 * np.sqrt(epsilon_perpendicular))
    ellipse_error = float(np.linalg.norm(extraordinary - ellipse) / np.linalg.norm(ellipse))
    sphere_error = float(np.linalg.norm(extraordinary - sphere) / np.linalg.norm(ellipse))
    ordinary_error = float(np.linalg.norm(ordinary - sphere) / np.linalg.norm(sphere))
    ordinary_gain = float(np.dot(ordinary, sphere) / np.dot(ordinary, ordinary))
    calibrated_extraordinary = ordinary_gain * extraordinary
    calibrated_ordinary = ordinary_gain * ordinary
    calibrated_ellipse_error = float(
        np.linalg.norm(calibrated_extraordinary - ellipse) / np.linalg.norm(ellipse)
    )
    calibrated_sphere_error = float(
        np.linalg.norm(calibrated_extraordinary - sphere) / np.linalg.norm(ellipse)
    )
    calibrated_ordinary_error = float(
        np.linalg.norm(calibrated_ordinary - sphere) / np.linalg.norm(sphere)
    )
    sim.reset_meep()
    return {
        "resolution": float(resolution),
        "angles_deg": angles_deg.tolist(),
        "radii": radii.tolist(),
        "extraordinary_measured_phase_slope": extraordinary.tolist(),
        "extraordinary_local_scatter": extraordinary_scatter.tolist(),
        "ordinary_measured_phase_slope": ordinary.tolist(),
        "ordinary_local_scatter": ordinary_scatter.tolist(),
        "ellipse_phase_slope": ellipse.tolist(),
        "sphere_phase_slope": sphere.tolist(),
        "extraordinary_ellipse_relative_l2": ellipse_error,
        "extraordinary_sphere_relative_l2": sphere_error,
        "sphere_to_ellipse_error_ratio": sphere_error / ellipse_error if ellipse_error > 0 else float("inf"),
        "ordinary_sphere_relative_l2": ordinary_error,
        "ordinary_complex_gain": ordinary_gain,
        "calibrated_extraordinary_phase_slope": calibrated_extraordinary.tolist(),
        "calibrated_extraordinary_ellipse_relative_l2": calibrated_ellipse_error,
        "calibrated_extraordinary_sphere_relative_l2": calibrated_sphere_error,
        "calibrated_sphere_to_ellipse_error_ratio": (
            calibrated_sphere_error / calibrated_ellipse_error
            if calibrated_ellipse_error > 0
            else float("inf")
        ),
        "calibrated_ordinary_sphere_relative_l2": calibrated_ordinary_error,
        "field_norm": float(np.linalg.norm(fields)),
        "source_vector": [[float(value.real), float(value.imag)] for value in source_vector],
    }, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="3-D PyMeep phase-slope uniaxial curvature probe.")
    parser.add_argument("--resolutions", default="16")
    parser.add_argument("--cell-width", type=float, default=2.6)
    parser.add_argument("--pml-width", type=float, default=0.35)
    parser.add_argument("--source-half-width", type=float, default=0.15)
    parser.add_argument("--until-after-sources", type=float, default=8.0)
    parser.add_argument("--radius-min", type=float, default=0.48)
    parser.add_argument("--radius-max", type=float, default=0.82)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_3d_phase_probe.json"),
    )
    args = parser.parse_args()
    frequency = 1.0
    n_o = 2.319393
    n_e = 2.224439
    epsilon_perpendicular = n_o**2
    epsilon_parallel = n_e**2
    angles_deg = np.linspace(10.0, 70.0, 13)
    interior_radius = args.cell_width / 2.0 - args.pml_width
    radius_max = min(args.radius_max, interior_radius - 0.08)
    if args.radius_min >= radius_max:
        raise ValueError("radius-min must be below radius-max and outside the PML")
    radii = np.linspace(args.radius_min, radius_max, 10)
    resolutions = [float(value) for value in args.resolutions.split(",")]
    rows: list[dict[str, object]] = []
    runtimes: list[float] = []
    for index, resolution in enumerate(resolutions, start=1):
        print(f"[{index}/{len(resolutions)}] resolution={resolution:g}", flush=True)
        row, runtime = run_resolution(
            resolution,
            cell_width=args.cell_width,
            pml_width=args.pml_width,
            source_half_width=args.source_half_width,
            source_grid_n=64,
            until_after_sources=args.until_after_sources,
            angles_deg=angles_deg,
            radii=radii,
            frequency=frequency,
            epsilon_perpendicular=epsilon_perpendicular,
            epsilon_parallel=epsilon_parallel,
        )
        rows.append(row)
        runtimes.append(runtime)
    finest = rows[int(np.argmax(resolutions))]
    if len(rows) >= 2:
        order = np.argsort(resolutions)
        next_finest = rows[int(order[-2])]
        fine = np.asarray(finest["calibrated_extraordinary_phase_slope"])
        coarse = np.asarray(next_finest["calibrated_extraordinary_phase_slope"])
        grid_l2 = float(np.linalg.norm(fine - coarse) / np.linalg.norm(fine))
    else:
        grid_l2 = float("nan")
    gates = {
        "calibrated_extraordinary_ellipse_l2_le_2pct": finest["calibrated_extraordinary_ellipse_relative_l2"] <= 0.02,
        "calibrated_forced_sphere_ratio_ge_5": finest["calibrated_sphere_to_ellipse_error_ratio"] >= 5.0,
        "calibrated_ordinary_sphere_l2_le_2pct": finest["calibrated_ordinary_sphere_relative_l2"] <= 0.02,
        "three_level_grid_l2_le_2pct": len(rows) >= 3 and grid_l2 <= 0.02,
    }
    result = {
        "schema": "uniaxial-meep-3d-phase-probe-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "3-D actual-uniaxial Maxwell phase-slope probe with fixed 64-cubed impressed source",
        "material": {
            "name": "5 mol% MgO:LiNbO3 at 532 nm",
            "n_o": n_o,
            "n_e": n_e,
            "epsilon_perpendicular": epsilon_perpendicular,
            "epsilon_parallel": epsilon_parallel,
        },
        "configuration": {
            "cell_width": args.cell_width,
            "pml_width": args.pml_width,
            "source_shape": [64, 64, 64],
            "source_half_width": args.source_half_width,
            "until_after_sources": args.until_after_sources,
            "angles_deg": angles_deg.tolist(),
            "radii": radii.tolist(),
        },
        "resolutions": resolutions,
        "runtime_seconds": runtimes,
        "rows": rows,
        "finest_next_finest_extraordinary_grid_l2": grid_l2,
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
