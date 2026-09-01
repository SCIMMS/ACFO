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
sys.path.insert(0, str(ROOT / "scripts"))

from probe_uniaxial_meep_3d_phase import fixed_source_grid, grid_amplitude  # noqa: E402
from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    PreparedAxisymmetricOperator,
    linbo3_3m_nonlinear_polarization,
    make_cylindrical_histogram,
    uniaxial_eigenpolarization,
)


def source_reference(
    angles_deg: np.ndarray,
    *,
    observation_azimuth: float,
    n: int,
    half_width: float,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
) -> dict[str, object]:
    grid = fixed_source_grid(
        n=n,
        half_width=half_width,
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
    )
    spacing = 2.0 * half_width / n
    axis = -half_width + (np.arange(n) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    active = grid != 0.0
    coords = np.column_stack((x[active], y[active], z[active]))
    weights = grid[active] * spacing**3
    binned = make_cylindrical_histogram(
        coords,
        atom_weights=weights,
        n_r=n,
        n_z=n,
        n_phi=n,
        r_max=np.sqrt(2.0) * half_width,
        z_range=(-half_width, half_width),
        hist_dtype=np.complex128,
        backend="numpy",
    )
    angle = np.deg2rad(angles_deg)
    k0 = 2.0 * np.pi * frequency
    denominator = np.sqrt(
        epsilon_parallel * np.sin(angle) ** 2
        + epsilon_perpendicular * np.cos(angle) ** 2
    )
    scale = k0 / denominator
    k_perp = scale * epsilon_parallel * np.sin(angle)
    k_z = scale * epsilon_perpendicular * np.cos(angle)
    sphere_k = k0 * np.sqrt(epsilon_perpendicular)
    sphere_perp = sphere_k * np.sin(angle)
    sphere_z = sphere_k * np.cos(angle)

    def evaluate(q_perp: np.ndarray, q_z: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        manifold = AxisymmetricManifold(
            angle,
            q_perp,
            -q_z,
            name=name,
            interpretation="dispersion-derived",
            frequency_units="inverse_length",
        )
        acfo_all = PreparedAxisymmetricOperator(
            binned, manifold, complex_dtype=np.complex128
        ).forward(binned.hist)
        target_beta = np.pi + observation_azimuth
        phi_index = int(
            np.argmin(
                np.abs(np.angle(np.exp(1j * (binned.beta_centers - target_beta))))
            )
        )
        acfo = acfo_all[:, phi_index]
        q_vectors = -np.column_stack(
            (
                q_perp * np.cos(observation_azimuth),
                q_perp * np.sin(observation_azimuth),
                q_z,
            )
        )
        cartesian = np.empty(angles_deg.size, dtype=np.complex128)
        for index, target in enumerate(q_vectors):
            cartesian[index] = weights @ np.exp(1j * (coords @ target))
        hist = np.asarray(binned.hist[0], dtype=np.complex128)
        ir, iz, ib = np.nonzero(hist)
        binned_weights = hist[ir, iz, ib]
        beta = binned.beta_centers[ib]
        positions = np.column_stack(
            (
                binned.r_centers[ir] * np.cos(beta),
                binned.r_centers[ir] * np.sin(beta),
                binned.z_centers[iz],
            )
        )
        direct_binned = np.empty_like(cartesian)
        for index, target in enumerate(q_vectors):
            direct_binned[index] = binned_weights @ np.exp(1j * (positions @ target))
        return acfo, direct_binned, cartesian

    ellipse_acfo, ellipse_direct, ellipse_cartesian = evaluate(
        k_perp, k_z, "extraordinary-ray-mapped-ellipse"
    )
    sphere_acfo, sphere_direct, sphere_cartesian = evaluate(
        sphere_perp, sphere_z, "forced-ordinary-sphere"
    )
    eigen = uniaxial_eigenpolarization(
        k_perp,
        k_z,
        np.array([observation_azimuth]),
        epsilon_parallel=epsilon_parallel,
        epsilon_perpendicular=epsilon_perpendicular,
        branch="extraordinary",
    )[:, 0, :]
    ordinary = np.column_stack(
        (
            np.full(angles_deg.size, -np.sin(observation_azimuth)),
            np.full(angles_deg.size, np.cos(observation_azimuth)),
            np.zeros(angles_deg.size),
        )
    )
    return {
        "ellipse_acfo": ellipse_acfo,
        "ellipse_direct_binned": ellipse_direct,
        "ellipse_cartesian": ellipse_cartesian,
        "sphere_acfo": sphere_acfo,
        "sphere_direct_binned": sphere_direct,
        "sphere_cartesian": sphere_cartesian,
        "extraordinary_eigenpolarization": eigen,
        "ordinary_eigenpolarization": ordinary,
        "active_voxels": int(coords.shape[0]),
        "nonzero_bins": int(np.count_nonzero(binned.hist)),
    }


def run_source(
    *,
    source_kind: str,
    source_vector: np.ndarray,
    resolution: float,
    angles_deg: np.ndarray,
    radii: np.ndarray,
    observation_azimuth: float,
    cell_width: float,
    pml_width: float,
    source_half_width: float,
    source_grid_n: int,
    until_after_sources: float,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
) -> tuple[np.ndarray, float]:
    material = mp.Medium(
        epsilon_diag=mp.Vector3(
            epsilon_perpendicular,
            epsilon_perpendicular,
            epsilon_parallel,
        )
    )
    source_vector = np.asarray(source_vector, dtype=np.complex128)
    if source_vector.shape != (3,) or not np.all(np.isfinite(source_vector)):
        raise ValueError("source_vector must be finite with shape (3,)")
    source_norm = float(np.linalg.norm(source_vector))
    if source_norm == 0.0:
        raise ValueError("source_vector must be nonzero")
    source_vector = source_vector / source_norm
    source_time = mp.GaussianSource(frequency=frequency, fwidth=0.30, is_integrated=True)
    components = (mp.Ex, mp.Ey, mp.Ez)
    if source_kind == "pattern":
        grid = fixed_source_grid(
            n=source_grid_n,
            half_width=source_half_width,
            frequency=frequency,
            epsilon_perpendicular=epsilon_perpendicular,
            epsilon_parallel=epsilon_parallel,
        )
        amplitude = grid_amplitude(grid, half_width=source_half_width)
        sources = [
            mp.Source(
                source_time,
                component=component,
                center=mp.Vector3(),
                size=mp.Vector3(
                    2.0 * source_half_width,
                    2.0 * source_half_width,
                    2.0 * source_half_width,
                ),
                amplitude=complex(source_vector[index]),
                amp_func=amplitude,
            )
            for index, component in enumerate(components)
        ]
    elif source_kind == "point":
        sources = [
            mp.Source(
                source_time,
                component=component,
                center=mp.Vector3(),
                amplitude=complex(source_vector[index]),
            )
            for index, component in enumerate(components)
        ]
    else:
        raise ValueError("source_kind must be pattern or point")
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_width, cell_width, cell_width),
        resolution=resolution,
        default_material=material,
        sources=sources,
        boundary_layers=[mp.PML(pml_width)],
        force_complex_fields=True,
        progress_interval=30,
    )
    components = (mp.Ex, mp.Ey, mp.Ez)
    monitors: list[tuple[int, int, object]] = []
    for ia, angle_deg in enumerate(angles_deg):
        angle = np.deg2rad(angle_deg)
        for ir, radius in enumerate(radii):
            point = mp.Vector3(
                radius * np.sin(angle) * np.cos(observation_azimuth),
                radius * np.sin(angle) * np.sin(observation_azimuth),
                radius * np.cos(angle),
            )
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
            fields[ia, ir, ic] = complex(
                np.asarray(sim.get_dft_array(monitor, component, 0)).squeeze().item()
            )
    elapsed = time.perf_counter() - started
    sim.reset_meep()
    return fields, elapsed


def projected_ratio(
    pattern: np.ndarray,
    point: np.ndarray,
    eigen: np.ndarray,
) -> tuple[np.ndarray, float]:
    pattern_projected = np.einsum("arc,ac->ar", pattern, eigen)
    point_projected = np.einsum("arc,ac->ar", point, eigen)
    valid = np.abs(point_projected) > 1e-12 * np.max(np.abs(point_projected))
    ratios = np.divide(
        pattern_projected,
        point_projected,
        out=np.full_like(pattern_projected, np.nan + 1j * np.nan),
        where=valid,
    )
    weights = np.where(valid, np.abs(point_projected) ** 2, 0.0)
    mean = np.sum(weights * np.nan_to_num(ratios), axis=1) / np.sum(weights, axis=1)
    scatter = float(
        np.linalg.norm(np.where(valid, ratios - mean[:, None], 0.0))
        / np.linalg.norm(np.where(valid, mean[:, None], 0.0))
    )
    return mean, scatter


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(model - reference) / np.linalg.norm(reference))


def intensity_ncc(left: np.ndarray, right: np.ndarray) -> float:
    x = np.abs(left) ** 2
    y = np.abs(right) ** 2
    x = x - np.mean(x)
    y = y - np.mean(y)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def main() -> None:
    parser = argparse.ArgumentParser(description="3-D PyMeep-to-ACFO complex amplitude double-ratio gate.")
    parser.add_argument("--resolutions", default="12")
    parser.add_argument("--cell-width", type=float, default=5.4)
    parser.add_argument("--pml-width", type=float, default=0.6)
    parser.add_argument("--source-half-width", type=float, default=0.8)
    parser.add_argument("--radius-min", type=float, default=1.25)
    parser.add_argument("--radius-max", type=float, default=1.9)
    parser.add_argument("--radius-count", type=int, default=5)
    parser.add_argument("--angle-min-deg", type=float, default=10.0)
    parser.add_argument("--angle-max-deg", type=float, default=70.0)
    parser.add_argument("--angle-count", type=int, default=31)
    parser.add_argument("--source-grid-n", type=int, default=64)
    parser.add_argument(
        "--source-vector",
        choices=["nonlinear", "x", "y", "z"],
        default="nonlinear",
    )
    parser.add_argument("--observation-azimuth-deg", type=float, default=-2.8125)
    parser.add_argument("--until-after-sources", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_3d_amplitude_gate.json"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional compressed NPZ with raw pattern/point fields and reference amplitudes.",
    )
    args = parser.parse_args()
    if args.radius_count < 2 or args.angle_count < 3 or args.source_grid_n < 4:
        raise ValueError("radius-count >=2, angle-count >=3, and source-grid-n >=4 are required")
    if args.radius_max <= args.radius_min or args.angle_max_deg <= args.angle_min_deg:
        raise ValueError("radius and angle maxima must exceed their minima")
    frequency = 1.0
    epsilon_perpendicular = 2.319393**2
    epsilon_parallel = 2.224439**2
    observation_azimuth = np.deg2rad(args.observation_azimuth_deg)
    if args.source_vector == "nonlinear":
        pump = np.array(
            (0.25, 0.35, np.sqrt(1.0 - 0.25**2 - 0.35**2)),
            dtype=np.complex128,
        )
        source_vector = linbo3_3m_nonlinear_polarization(pump)
    else:
        source_vector = np.zeros(3, dtype=np.complex128)
        source_vector[{"x": 0, "y": 1, "z": 2}[args.source_vector]] = 1.0
    source_vector /= np.linalg.norm(source_vector)
    angles_deg = np.linspace(args.angle_min_deg, args.angle_max_deg, args.angle_count)
    radii = np.linspace(args.radius_min, args.radius_max, args.radius_count)
    reference = source_reference(
        angles_deg,
        observation_azimuth=observation_azimuth,
        n=args.source_grid_n,
        half_width=args.source_half_width,
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
    )
    rows: list[dict[str, object]] = []
    raw_payload: dict[str, np.ndarray] = {
        "angles_deg": angles_deg,
        "radii": radii,
        "ellipse_acfo": np.asarray(reference["ellipse_acfo"]),
        "ellipse_direct_binned": np.asarray(reference["ellipse_direct_binned"]),
        "ellipse_cartesian": np.asarray(reference["ellipse_cartesian"]),
        "sphere_acfo": np.asarray(reference["sphere_acfo"]),
        "extraordinary_eigenpolarization": np.asarray(
            reference["extraordinary_eigenpolarization"]
        ),
        "ordinary_eigenpolarization": np.asarray(
            reference["ordinary_eigenpolarization"]
        ),
    }
    resolutions = [float(value) for value in args.resolutions.split(",")]
    for index, resolution in enumerate(resolutions, start=1):
        print(f"[{index}/{len(resolutions)}] resolution={resolution:g} pattern", flush=True)
        pattern, pattern_s = run_source(
            source_kind="pattern",
            source_vector=source_vector,
            resolution=resolution,
            angles_deg=angles_deg,
            radii=radii,
            observation_azimuth=observation_azimuth,
            cell_width=args.cell_width,
            pml_width=args.pml_width,
            source_half_width=args.source_half_width,
            source_grid_n=args.source_grid_n,
            until_after_sources=args.until_after_sources,
            frequency=frequency,
            epsilon_perpendicular=epsilon_perpendicular,
            epsilon_parallel=epsilon_parallel,
        )
        print(f"[{index}/{len(resolutions)}] resolution={resolution:g} point", flush=True)
        point, point_s = run_source(
            source_kind="point",
            source_vector=source_vector,
            resolution=resolution,
            angles_deg=angles_deg,
            radii=radii,
            observation_azimuth=observation_azimuth,
            cell_width=args.cell_width,
            pml_width=args.pml_width,
            source_half_width=args.source_half_width,
            source_grid_n=args.source_grid_n,
            until_after_sources=args.until_after_sources,
            frequency=frequency,
            epsilon_perpendicular=epsilon_perpendicular,
            epsilon_parallel=epsilon_parallel,
        )
        extra_ratio, extra_radial_scatter = projected_ratio(
            pattern, point, np.asarray(reference["extraordinary_eigenpolarization"])
        )
        ordinary_ratio, ordinary_radial_scatter = projected_ratio(
            pattern, point, np.asarray(reference["ordinary_eigenpolarization"])
        )
        raw_key = f"r{resolution:g}".replace(".", "p")
        raw_payload[f"pattern_{raw_key}"] = pattern
        raw_payload[f"point_{raw_key}"] = point
        sphere = np.asarray(reference["sphere_acfo"])
        ellipse = np.asarray(reference["ellipse_acfo"])
        gain = np.vdot(ordinary_ratio, sphere) / np.vdot(ordinary_ratio, ordinary_ratio)
        calibrated_extra = gain * extra_ratio
        calibrated_ordinary = gain * ordinary_ratio
        correct_error = relative_l2(calibrated_extra, ellipse)
        wrong_error = relative_l2(calibrated_extra, sphere)
        intensity_correct = np.abs(ellipse) ** 2
        intensity_measured = np.abs(calibrated_extra) ** 2
        intensity_wrong = np.abs(sphere) ** 2
        step = float(angles_deg[1] - angles_deg[0])
        peak_measured = float(angles_deg[int(np.argmax(intensity_measured))])
        peak_correct = float(angles_deg[int(np.argmax(intensity_correct))])
        peak_wrong = float(angles_deg[int(np.argmax(intensity_wrong))])
        row = {
            "resolution": resolution,
            "pattern_runtime_s": pattern_s,
            "point_runtime_s": point_s,
            "ordinary_complex_gain": [float(gain.real), float(gain.imag)],
            "extraordinary_ratio": [[float(v.real), float(v.imag)] for v in extra_ratio],
            "ordinary_ratio": [[float(v.real), float(v.imag)] for v in ordinary_ratio],
            "calibrated_extraordinary_ratio": [
                [float(v.real), float(v.imag)] for v in calibrated_extra
            ],
            "ordinary_calibration_l2": relative_l2(calibrated_ordinary, sphere),
            "extraordinary_complex_l2": correct_error,
            "forced_sphere_complex_l2": wrong_error,
            "forced_sphere_wrong_to_correct_ratio": wrong_error / correct_error,
            "intensity_ncc": intensity_ncc(calibrated_extra, ellipse),
            "peak_error_deg": abs(peak_measured - peak_correct),
            "forced_sphere_peak_shift_bins": abs(peak_wrong - peak_measured) / step,
            "forced_sphere_intensity_correlation_drop": 1.0
            - intensity_ncc(calibrated_extra, sphere),
            "extraordinary_radial_ratio_scatter": extra_radial_scatter,
            "ordinary_radial_ratio_scatter": ordinary_radial_scatter,
        }
        rows.append(row)
    finest = rows[int(np.argmax(resolutions))]
    if len(rows) >= 2:
        order = np.argsort(resolutions)
        fine = np.array([complex(*value) for value in rows[int(order[-1])]["calibrated_extraordinary_ratio"]])
        coarse = np.array([complex(*value) for value in rows[int(order[-2])]["calibrated_extraordinary_ratio"]])
        grid_l2 = relative_l2(fine, coarse)
    else:
        grid_l2 = float("nan")
    algorithm_error = relative_l2(
        np.asarray(reference["ellipse_acfo"]),
        np.asarray(reference["ellipse_direct_binned"]),
    )
    representation_error = relative_l2(
        np.asarray(reference["ellipse_acfo"]),
        np.asarray(reference["ellipse_cartesian"]),
    )
    gates = {
        "ordinary_calibration_l2_le_5pct": finest["ordinary_calibration_l2"] <= 0.05,
        "extraordinary_complex_l2_le_5pct": finest["extraordinary_complex_l2"] <= 0.05,
        "intensity_ncc_ge_0_98": finest["intensity_ncc"] >= 0.98,
        "peak_error_le_2deg": finest["peak_error_deg"] <= 2.0,
        "forced_sphere_ratio_ge_5": finest["forced_sphere_wrong_to_correct_ratio"] >= 5.0,
        "forced_sphere_observable": (
            finest["forced_sphere_peak_shift_bins"] >= 1.0
            or finest["forced_sphere_intensity_correlation_drop"] >= 0.10
        ),
        "algorithm_error_lt_1pct_physics_mismatch": algorithm_error < 0.01 * finest["extraordinary_complex_l2"],
        "three_level_grid_l2_le_2pct": len(rows) >= 3 and grid_l2 <= 0.02,
        "radial_ratio_scatter_le_5pct": finest["extraordinary_radial_ratio_scatter"] <= 0.05,
    }
    result = {
        "schema": "uniaxial-meep-3d-amplitude-double-ratio-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "3-D actual-uniaxial patterned-to-point source complex amplitude ratio compared with ACFO; homogeneous linear propagation only",
        "configuration": {
            "resolutions": resolutions,
            "cell_width": args.cell_width,
            "pml_width": args.pml_width,
            "source_shape": [args.source_grid_n] * 3,
            "source_half_width": args.source_half_width,
            "angles_deg": angles_deg.tolist(),
            "radii": radii.tolist(),
            "observation_azimuth": observation_azimuth,
            "observation_azimuth_deg": args.observation_azimuth_deg,
            "source_vector_mode": args.source_vector,
            "source_vector": [
                [float(value.real), float(value.imag)] for value in source_vector
            ],
            "until_after_sources": args.until_after_sources,
        },
        "reference": {
            "active_voxels": reference["active_voxels"],
            "nonzero_bins": reference["nonzero_bins"],
            "algorithm_complex_l2": algorithm_error,
            "representation_complex_l2": representation_error,
        },
        "rows": rows,
        "finest_next_finest_grid_l2": grid_l2,
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
    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.raw_output, **raw_payload)
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    if args.raw_output is not None:
        print(f"wrote {args.raw_output}", flush=True)


if __name__ == "__main__":
    main()
