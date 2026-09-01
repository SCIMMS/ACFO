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
from scipy.special import erf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_closed_surface_modal_reciprocity import (  # noqa: E402
    COMPONENTS,
    FaceMonitor,
    face_specs,
    monitor_fields,
    relative_l2,
)
from probe_uniaxial_closed_surface_reciprocity import (  # noqa: E402
    reciprocal_surface_amplitudes,
    uniaxial_modes,
)
from probe_uniaxial_distributed_source_modal import (  # noqa: E402
    branch_error,
    complex_pairs,
    load_point_calibration,
)
from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    PreparedAxisymmetricOperator,
    make_cylindrical_histogram,
)


SOURCE_HALF_WIDTH = 0.55
SOURCE_SIGMAS = np.array((0.13, 0.11, 0.15), dtype=np.float64)
SOURCE_CARRIER = np.array((3.5, -2.0, 2.8), dtype=np.float64)
MODE_PHI_DEGREES = (1.875, 61.875, 121.875)
REFERENCE_GRID_N = 96


def gaussian_interval(delta: np.ndarray, sigma: float, half_width: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    scale = np.sqrt(np.pi / 2.0) * sigma * np.exp(-0.5 * (sigma * delta) ** 2)
    denominator = np.sqrt(2.0) * sigma
    upper = (half_width - 1j * sigma**2 * delta) / denominator
    lower = (-half_width - 1j * sigma**2 * delta) / denominator
    return scale * (erf(upper) - erf(lower))


def tapered_gaussian_interval(
    delta: np.ndarray, sigma: float, half_width: float
) -> np.ndarray:
    taper_wave_number = np.pi / half_width
    return (
        0.5 * gaussian_interval(delta, sigma, half_width)
        + 0.25 * gaussian_interval(delta + taper_wave_number, sigma, half_width)
        + 0.25 * gaussian_interval(delta - taper_wave_number, sigma, half_width)
    )


def smooth_scalar_array(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    inside = np.all(np.abs(coordinates) <= SOURCE_HALF_WIDTH + 1e-12, axis=-1)
    scaled = coordinates / SOURCE_SIGMAS
    gaussian = np.exp(-0.5 * np.sum(scaled**2, axis=-1))
    taper = np.prod(
        np.cos(0.5 * np.pi * coordinates / SOURCE_HALF_WIDTH) ** 2,
        axis=-1,
    )
    carrier = np.exp(1j * (coordinates @ SOURCE_CARRIER))
    return np.where(inside, gaussian * taper * carrier, 0.0j)


def smooth_amp_func(point: mp.Vector3) -> complex:
    coordinate = np.array((point.x, point.y, point.z), dtype=np.float64)
    return complex(smooth_scalar_array(coordinate[None, :])[0])


def analytic_scalar_transform(wavevectors: np.ndarray) -> np.ndarray:
    delta = SOURCE_CARRIER[None, :] - np.asarray(wavevectors, dtype=np.float64)
    factors = [
        tapered_gaussian_interval(
            delta[:, axis], SOURCE_SIGMAS[axis], SOURCE_HALF_WIDTH
        )
        for axis in range(3)
    ]
    return factors[0] * factors[1] * factors[2]


def direct_scalar_transform(
    coordinates: np.ndarray, weights: np.ndarray, wavevectors: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [weights @ np.exp(-1j * (coordinates @ wavevector)) for wavevector in wavevectors]
    )


def acfo_scalar_transform(
    coordinates: np.ndarray,
    weights: np.ndarray,
    wavevectors: np.ndarray,
    labels: list[dict[str, float | str]],
    *,
    n_r: int,
    n_z: int,
    n_phi: int,
) -> tuple[np.ndarray, int]:
    binned = make_cylindrical_histogram(
        coordinates,
        atom_weights=weights,
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=np.sqrt(2.0) * SOURCE_HALF_WIDTH,
        z_range=(-SOURCE_HALF_WIDTH, SOURCE_HALF_WIDTH),
        hist_dtype=np.complex128,
        backend="numpy",
    )
    output = np.empty(wavevectors.shape[0], dtype=np.complex128)
    first_phi_deg = float(labels[0]["phi_deg"])
    for branch in ("ordinary", "extraordinary"):
        branch_indices = [
            index
            for index, label in enumerate(labels)
            if label["branch"] == branch
            and np.isclose(float(label["phi_deg"]), first_phi_deg)
        ]
        parameters_deg = np.asarray(
            [labels[index]["surface_parameter_deg"] for index in branch_indices]
        )
        branch_wavevectors = wavevectors[branch_indices]
        manifold = AxisymmetricManifold(
            np.deg2rad(parameters_deg),
            np.linalg.norm(branch_wavevectors[:, :2], axis=1),
            -branch_wavevectors[:, 2],
            name=f"smooth-source-{branch}",
            interpretation="dispersion-derived",
            frequency_units="inverse_length",
        )
        acfo_all = PreparedAxisymmetricOperator(
            binned,
            manifold,
            complex_dtype=np.complex128,
        ).forward(binned.hist)
        parameter_to_index = {
            float(parameter): index for index, parameter in enumerate(parameters_deg)
        }
        for index, label in enumerate(labels):
            if label["branch"] != branch:
                continue
            parameter_index = parameter_to_index[float(label["surface_parameter_deg"])]
            target_beta = np.deg2rad(float(label["phi_deg"])) + np.pi
            beta_index = int(
                np.argmin(
                    np.abs(
                        np.angle(
                            np.exp(1j * (binned.beta_centers - target_beta))
                        )
                    )
                )
            )
            output[index] = acfo_all[parameter_index, beta_index]
    return output, int(np.count_nonzero(binned.hist))


def fixed_continuous_reference(
    source_vector: np.ndarray,
    *,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
    grid_n: int,
) -> dict[str, object]:
    wavevectors, electric_modes, labels = uniaxial_modes(
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
        phi_degrees=MODE_PHI_DEGREES,
    )
    source_coupling = electric_modes @ source_vector
    analytic_scalar = analytic_scalar_transform(wavevectors)
    analytic_modal = analytic_scalar * source_coupling

    spacing = 2.0 * SOURCE_HALF_WIDTH / grid_n
    axis = -SOURCE_HALF_WIDTH + (np.arange(grid_n) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    coordinates = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    voxel_weights = smooth_scalar_array(coordinates) * spacing**3
    direct_voxel_scalar = direct_scalar_transform(
        coordinates, voxel_weights, wavevectors
    )
    acfo_voxel_scalar, nonzero_bins = acfo_scalar_transform(
        coordinates,
        voxel_weights,
        wavevectors,
        labels,
        n_r=grid_n,
        n_z=grid_n,
        n_phi=grid_n,
    )
    direct_voxel_modal = direct_voxel_scalar * source_coupling
    acfo_voxel_modal = acfo_voxel_scalar * source_coupling
    return {
        "wavevectors": wavevectors,
        "electric_modes": electric_modes,
        "labels": labels,
        "analytic_modal": analytic_modal,
        "direct_voxel_modal": direct_voxel_modal,
        "acfo_voxel_modal": acfo_voxel_modal,
        "voxel_direct_vs_analytic_relative_l2": relative_l2(
            direct_voxel_modal, analytic_modal
        ),
        "acfo_vs_voxel_direct_relative_l2": relative_l2(
            acfo_voxel_modal, direct_voxel_modal
        ),
        "acfo_vs_analytic_relative_l2": relative_l2(
            acfo_voxel_modal, analytic_modal
        ),
        "voxel_count": int(coordinates.shape[0]),
        "nonzero_bins": nonzero_bins,
        "voxel_integral": complex(np.sum(voxel_weights)),
    }


def yee_source_reference(
    sim: mp.Simulation,
    source_vector: np.ndarray,
    reference: dict[str, object],
    *,
    resolution: float,
) -> dict[str, object]:
    margin = 2.0 / resolution
    export_width = 2.0 * SOURCE_HALF_WIDTH + 2.0 * margin
    volume = mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(export_width, export_width, export_width),
    )
    x, y, z, quadrature_weights = sim.get_array_metadata(vol=volume)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    base_coordinates = np.column_stack(
        (x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1))
    )
    weights = np.asarray(quadrature_weights, dtype=np.float64).reshape(-1)
    currents = [
        np.asarray(sim.get_source(component, vol=volume)).reshape(-1)
        for component in (mp.Ex, mp.Ey, mp.Ez)
    ]
    wavevectors = np.asarray(reference["wavevectors"])
    electric_modes = np.asarray(reference["electric_modes"])
    labels = list(reference["labels"])
    direct_modal = np.zeros(wavevectors.shape[0], dtype=np.complex128)
    acfo_modal = np.zeros_like(direct_modal)
    active_counts: list[int] = []
    nonzero_bins_by_component: list[int] = []
    component_integrals: list[complex] = []
    sample_error_numerator = 0.0
    sample_error_denominator = 0.0
    grid_spacing = 1.0 / resolution
    for component_index, current in enumerate(currents):
        component_coordinates = base_coordinates.copy()
        for axis in range(3):
            if axis != component_index:
                component_coordinates[:, axis] -= 0.5 * grid_spacing
        active = np.abs(current) > 0.0
        coordinates = component_coordinates[active]
        component_weights = weights[active]
        component_current = current[active]
        expected_current = source_vector[component_index] * smooth_scalar_array(
            coordinates
        )
        sample_error_numerator += float(
            np.sum(component_weights * np.abs(component_current - expected_current) ** 2)
        )
        sample_error_denominator += float(
            np.sum(component_weights * np.abs(expected_current) ** 2)
        )
        scalar_direct = direct_scalar_transform(
            coordinates,
            component_current * component_weights,
            wavevectors,
        )
        direct_modal += electric_modes[:, component_index] * scalar_direct
        scalar_acfo, component_nonzero_bins = acfo_scalar_transform(
            coordinates,
            component_current * component_weights,
            wavevectors,
            labels,
            n_r=REFERENCE_GRID_N,
            n_z=REFERENCE_GRID_N,
            n_phi=REFERENCE_GRID_N,
        )
        acfo_modal += electric_modes[:, component_index] * scalar_acfo
        active_counts.append(int(np.count_nonzero(active)))
        nonzero_bins_by_component.append(component_nonzero_bins)
        component_integrals.append(
            complex(np.sum(component_weights * component_current))
        )
    spatial_sample_relative_l2 = float(
        np.sqrt(sample_error_numerator / sample_error_denominator)
    )
    scalar_integral = sum(
        source_vector[index] * value
        for index, value in enumerate(component_integrals)
    )
    return {
        "direct_modal": direct_modal,
        "acfo_modal": acfo_modal,
        "active_grid_points_by_component": active_counts,
        "nonzero_bins_by_component": nonzero_bins_by_component,
        "spatial_sample_relative_l2": spatial_sample_relative_l2,
        "component_integrals": component_integrals,
        "scalar_integral": scalar_integral,
        "acfo_vs_direct_relative_l2": relative_l2(acfo_modal, direct_modal),
        "direct_vs_analytic_relative_l2": relative_l2(
            direct_modal, np.asarray(reference["analytic_modal"])
        ),
    }


def run_resolution(
    resolution: float,
    source_vector: np.ndarray,
    reference: dict[str, object],
    calibration_gains: tuple[complex, complex],
    *,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
    cell_width: float,
    pml_width: float,
    monitor_half_widths: tuple[float, float],
    until_after_sources: float,
) -> tuple[dict[str, object], list[np.ndarray]]:
    source_time = mp.GaussianSource(
        frequency=frequency,
        fwidth=0.4,
        is_integrated=True,
    )
    sources = [
        mp.Source(
            source_time,
            component=component,
            center=mp.Vector3(),
            size=mp.Vector3(
                2.0 * SOURCE_HALF_WIDTH,
                2.0 * SOURCE_HALF_WIDTH,
                2.0 * SOURCE_HALF_WIDTH,
            ),
            amplitude=float(source_vector[index]),
            amp_func=smooth_amp_func,
        )
        for index, component in enumerate((mp.Ex, mp.Ey, mp.Ez))
    ]
    material = mp.Medium(
        epsilon_diag=mp.Vector3(
            epsilon_perpendicular,
            epsilon_perpendicular,
            epsilon_parallel,
        )
    )
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_width, cell_width, cell_width),
        resolution=resolution,
        default_material=material,
        sources=sources,
        boundary_layers=[mp.PML(pml_width)],
        force_complex_fields=True,
        progress_interval=30,
    )
    shells: list[list[FaceMonitor]] = []
    for half_width in monitor_half_widths:
        face_monitors: list[FaceMonitor] = []
        for center, size, normal, _ in face_specs(half_width):
            dft = sim.add_dft_fields(
                list(COMPONENTS),
                frequency,
                0.0,
                1,
                where=mp.Volume(center=center, size=size),
            )
            face_monitors.append(FaceMonitor(normal=normal, dft=dft))
        shells.append(face_monitors)

    sim.init_sim()
    yee_reference = yee_source_reference(
        sim,
        source_vector,
        reference,
        resolution=resolution,
    )
    started = time.perf_counter()
    sim.run(until_after_sources=until_after_sources)
    runtime = time.perf_counter() - started

    wavevectors = np.asarray(reference["wavevectors"])
    electric_modes = np.asarray(reference["electric_modes"])
    labels = list(reference["labels"])
    raw_amplitudes: list[np.ndarray] = []
    calibrated_amplitudes: list[np.ndarray] = []
    quadrature_points: list[int] = []
    for shell_index, face_monitors in enumerate(shells):
        face_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        point_count = 0
        for face in face_monitors:
            coordinates, weights, fields = monitor_fields(sim, face)
            face_data.append((coordinates, weights, fields, face.normal))
            point_count += weights.size
        raw = reciprocal_surface_amplitudes(
            face_data,
            wavevectors,
            electric_modes,
            frequency=frequency,
        )
        raw_amplitudes.append(raw)
        calibrated_amplitudes.append(calibration_gains[shell_index] * raw)
        quadrature_points.append(point_count)

    analytic_modal = np.asarray(reference["analytic_modal"])
    fixed_acfo_modal = np.asarray(reference["acfo_voxel_modal"])
    yee_direct_modal = np.asarray(yee_reference["direct_modal"])
    shell_rows: list[dict[str, object]] = []
    for shell_index, calibrated in enumerate(calibrated_amplitudes):
        shell_rows.append(
            {
                "point_calibration_gain": [
                    float(calibration_gains[shell_index].real),
                    float(calibration_gains[shell_index].imag),
                ],
                "fdtd_vs_analytic_relative_l2": relative_l2(
                    calibrated, analytic_modal
                ),
                "fdtd_vs_fixed_acfo_relative_l2": relative_l2(
                    calibrated, fixed_acfo_modal
                ),
                "fdtd_vs_yee_direct_relative_l2": relative_l2(
                    calibrated, yee_direct_modal
                ),
                "branch_fdtd_vs_analytic_relative_l2": branch_error(
                    calibrated, analytic_modal, labels
                ),
                "calibrated_norm": float(np.linalg.norm(calibrated)),
            }
        )
    row = {
        "resolution": float(resolution),
        "runtime_s": runtime,
        "meep_source_count": len(sources),
        "quadrature_points": quadrature_points,
        "yee_source": {
            "active_grid_points_by_component": yee_reference[
                "active_grid_points_by_component"
            ],
            "nonzero_bins_by_component": yee_reference[
                "nonzero_bins_by_component"
            ],
            "spatial_sample_relative_l2": yee_reference[
                "spatial_sample_relative_l2"
            ],
            "component_integrals": [
                [float(value.real), float(value.imag)]
                for value in yee_reference["component_integrals"]
            ],
            "scalar_integral": [
                float(yee_reference["scalar_integral"].real),
                float(yee_reference["scalar_integral"].imag),
            ],
            "direct_vs_analytic_relative_l2": yee_reference[
                "direct_vs_analytic_relative_l2"
            ],
            "acfo_vs_direct_relative_l2": yee_reference[
                "acfo_vs_direct_relative_l2"
            ],
        },
        "shells": shell_rows,
        "raw_inner_outer_relative_l2": relative_l2(
            raw_amplitudes[1], raw_amplitudes[0]
        ),
        "calibrated_inner_outer_relative_l2": relative_l2(
            calibrated_amplitudes[1], calibrated_amplitudes[0]
        ),
        "inner_calibrated_modal_amplitudes": complex_pairs(
            calibrated_amplitudes[0]
        ),
        "yee_direct_modal_amplitudes": complex_pairs(yee_direct_modal),
    }
    sim.reset_meep()
    return row, calibrated_amplitudes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind smooth compact voxel-source uniaxial FDTD-to-ACFO modal gate."
    )
    parser.add_argument("--resolutions", default="14,18,22")
    parser.add_argument("--cell-width", type=float, default=4.0)
    parser.add_argument("--pml-width", type=float, default=0.5)
    parser.add_argument("--monitor-half-widths", default="0.70,1.00")
    parser.add_argument("--until-after-sources", type=float, default=5.0)
    parser.add_argument("--reference-grid-n", type=int, default=REFERENCE_GRID_N)
    parser.add_argument(
        "--point-calibration",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/uniaxial_smooth_voxel_source_modal_probe.json"),
    )
    args = parser.parse_args()
    resolutions = tuple(float(value) for value in args.resolutions.split(","))
    monitor_half_widths = tuple(float(value) for value in args.monitor_half_widths.split(","))
    if len(monitor_half_widths) != 2 or not monitor_half_widths[0] < monitor_half_widths[1]:
        raise ValueError("monitor-half-widths must contain two increasing values")
    if SOURCE_HALF_WIDTH >= monitor_half_widths[0]:
        raise ValueError("source must lie strictly inside the inner monitor shell")
    if args.reference_grid_n != REFERENCE_GRID_N:
        raise ValueError(
            f"reference-grid-n must remain {REFERENCE_GRID_N} so the modal azimuths "
            "stay exactly aligned with cylindrical beta-bin centers"
        )
    frequency = 1.0
    epsilon_perpendicular = 1.5**2
    epsilon_parallel = 1.8**2
    source_vector = np.array((0.36, 0.48, 0.80), dtype=np.float64)
    source_vector /= np.linalg.norm(source_vector)
    reference = fixed_continuous_reference(
        source_vector,
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
        grid_n=args.reference_grid_n,
    )
    point_calibration_paths = tuple(
        args.point_calibration
        or [Path("benchmark_results/uniaxial_closed_surface_reciprocity_probe.json")]
    )
    calibration_gains, calibration_metadata = load_point_calibration(
        point_calibration_paths,
        resolutions,
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
        cell_width=args.cell_width,
        pml_width=args.pml_width,
        monitor_half_widths=monitor_half_widths,
    )
    rows: list[dict[str, object]] = []
    calibrated_by_resolution: dict[float, list[np.ndarray]] = {}
    for index, resolution in enumerate(resolutions, start=1):
        print(f"[{index}/{len(resolutions)}] resolution={resolution:g}", flush=True)
        row, calibrated = run_resolution(
            resolution,
            source_vector,
            reference,
            calibration_gains[resolution],
            frequency=frequency,
            epsilon_perpendicular=epsilon_perpendicular,
            epsilon_parallel=epsilon_parallel,
            cell_width=args.cell_width,
            pml_width=args.pml_width,
            monitor_half_widths=monitor_half_widths,
            until_after_sources=args.until_after_sources,
        )
        rows.append(row)
        calibrated_by_resolution[resolution] = calibrated
    finest_resolution = max(resolutions)
    finest = rows[resolutions.index(finest_resolution)]
    if len(resolutions) >= 2:
        next_finest_resolution = sorted(resolutions)[-2]
        grid_relative_l2 = relative_l2(
            calibrated_by_resolution[next_finest_resolution][0],
            calibrated_by_resolution[finest_resolution][0],
        )
    else:
        grid_relative_l2 = float("nan")
    gates = {
        "voxel_direct_vs_analytic_l2_le_1pct": reference[
            "voxel_direct_vs_analytic_relative_l2"
        ]
        <= 0.01,
        "fixed_acfo_vs_analytic_l2_le_2pct": reference[
            "acfo_vs_analytic_relative_l2"
        ]
        <= 0.02,
        "yee_source_vs_analytic_l2_le_2pct": finest["yee_source"][
            "direct_vs_analytic_relative_l2"
        ]
        <= 0.02,
        "yee_acfo_vs_yee_direct_l2_le_2pct": finest["yee_source"][
            "acfo_vs_direct_relative_l2"
        ]
        <= 0.02,
        "blind_fdtd_vs_yee_direct_l2_le_5pct": finest["shells"][0][
            "fdtd_vs_yee_direct_relative_l2"
        ]
        <= 0.05,
        "blind_fdtd_vs_fixed_acfo_l2_le_5pct": finest["shells"][0][
            "fdtd_vs_fixed_acfo_relative_l2"
        ]
        <= 0.05,
        "ordinary_l2_le_5pct": finest["shells"][0][
            "branch_fdtd_vs_analytic_relative_l2"
        ]["ordinary"]
        <= 0.05,
        "extraordinary_l2_le_5pct": finest["shells"][0][
            "branch_fdtd_vs_analytic_relative_l2"
        ]["extraordinary"]
        <= 0.05,
        "monitor_invariance_le_5pct": finest[
            "calibrated_inner_outer_relative_l2"
        ]
        <= 0.05,
        "finest_next_finest_grid_l2_le_5pct": len(resolutions) >= 2
        and grid_relative_l2 <= 0.05,
    }
    result = {
        "schema": "uniaxial-smooth-voxel-source-modal-probe-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "compact C1 tapered-Gaussian volume current in a homogeneous lossless uniaxial medium; analytic, high-resolution voxel, exact PyMeep Yee-source, ACFO, and closed-surface FDTD modal references",
        "normalization_contract": "no smooth-source fitting; point-source complex gains are frozen independently at the same resolution and shell",
        "configuration": {
            "resolutions": resolutions,
            "frequency": frequency,
            "epsilon_perpendicular": epsilon_perpendicular,
            "epsilon_parallel": epsilon_parallel,
            "cell_width": args.cell_width,
            "pml_width": args.pml_width,
            "monitor_half_widths": monitor_half_widths,
            "until_after_sources": args.until_after_sources,
            "source_half_width": SOURCE_HALF_WIDTH,
            "source_sigmas": SOURCE_SIGMAS.tolist(),
            "source_carrier": SOURCE_CARRIER.tolist(),
            "source_vector": source_vector.tolist(),
            "mode_phi_degrees": list(MODE_PHI_DEGREES),
            "reference_grid_n": args.reference_grid_n,
            "ordinary_mode_count": 27,
            "extraordinary_mode_count": 27,
        },
        "point_calibration": calibration_metadata,
        "fixed_reference": {
            "voxel_count": reference["voxel_count"],
            "nonzero_bins": reference["nonzero_bins"],
            "voxel_integral": [
                float(reference["voxel_integral"].real),
                float(reference["voxel_integral"].imag),
            ],
            "voxel_direct_vs_analytic_relative_l2": reference[
                "voxel_direct_vs_analytic_relative_l2"
            ],
            "acfo_vs_voxel_direct_relative_l2": reference[
                "acfo_vs_voxel_direct_relative_l2"
            ],
            "acfo_vs_analytic_relative_l2": reference[
                "acfo_vs_analytic_relative_l2"
            ],
            "analytic_modal_amplitudes": complex_pairs(
                np.asarray(reference["analytic_modal"])
            ),
            "acfo_modal_amplitudes": complex_pairs(
                np.asarray(reference["acfo_voxel_modal"])
            ),
            "sample_labels": reference["labels"],
        },
        "rows": rows,
        "finest_next_finest_grid_relative_l2": grid_relative_l2,
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
