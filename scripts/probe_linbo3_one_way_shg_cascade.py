from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import meep as mp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_uniaxial_smooth_voxel_source_modal as smooth  # noqa: E402
from probe_closed_surface_modal_reciprocity import (  # noqa: E402
    COMPONENTS,
    FaceMonitor,
    face_specs,
    monitor_fields,
    relative_l2,
)
from probe_linbo3_shg_modal_reciprocity import (  # noqa: E402
    D22_PM_PER_V,
    D31_PM_PER_V,
    D33_PM_PER_V,
    MODE_PHI_DEGREES,
    MODE_PARAMETER_DEGREES,
    PUMP_AZIMUTH_DEG,
    PUMP_FREQUENCY,
    PUMP_SURFACE_PARAMETER_DEG,
    PUMP_WAVELENGTH_UM,
    SH_FREQUENCY,
    SH_WAVELENGTH_UM,
    TEMPERATURE_C,
    physical_configuration,
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


INTERACTION_HALF_WIDTH = 0.55
ACFO_GRID_N = 96
PUMP_CELL_WIDTH = 6.0
PUMP_PML_WIDTH = 0.5
PUMP_SOURCE_Z = -1.5
PUMP_BEAM_SIGMA = 0.55
SH_CELL_WIDTH = 5.0
SH_PML_WIDTH = 0.5
MONITOR_HALF_WIDTHS = (1.40, 1.75)


def pairs(array: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in np.asarray(array)]


def relative_weighted_l2(
    model: np.ndarray, reference: np.ndarray, weights: np.ndarray
) -> float:
    model_array = np.asarray(model)
    reference_array = np.asarray(reference)
    weight_array = np.asarray(weights)
    numerator = np.sum(weight_array * np.abs(model_array - reference_array) ** 2)
    denominator = np.sum(weight_array * np.abs(reference_array) ** 2)
    return float(np.sqrt(numerator / denominator))


def pump_sheet_amplitude(
    pump_wavevector: np.ndarray,
    pump_indices: dict[str, float],
    *,
    beam_sigma: float,
):
    radial = pump_wavevector[:2] / np.linalg.norm(pump_wavevector[:2])
    group_tangent = (
        np.linalg.norm(pump_wavevector[:2]) / float(pump_indices["n_e_pump"]) ** 2
    ) / (
        float(pump_wavevector[2]) / float(pump_indices["n_o_pump"]) ** 2
    )
    beam_center = -abs(PUMP_SOURCE_Z) * group_tangent * radial

    def amplitude(point: mp.Vector3) -> complex:
        position = np.array((point.x, point.y), dtype=np.float64)
        offset = position - beam_center
        envelope = np.exp(-0.5 * np.sum(offset**2) / beam_sigma**2)
        phase = np.exp(
            1j
            * (
                pump_wavevector[0] * point.x
                + pump_wavevector[1] * point.y
            )
        )
        return complex(envelope * phase)

    return amplitude, beam_center, float(group_tangent)


def contract_3m_field(pump_fields: np.ndarray) -> np.ndarray:
    ex, ey, ez = np.asarray(pump_fields, dtype=np.complex128)
    output = np.empty_like(pump_fields, dtype=np.complex128)
    output[0] = D31_PM_PER_V * 2.0 * ex * ez - D22_PM_PER_V * 2.0 * ex * ey
    output[1] = (
        -D22_PM_PER_V * ex * ex
        + D22_PM_PER_V * ey * ey
        + D31_PM_PER_V * 2.0 * ey * ez
    )
    output[2] = (
        D31_PM_PER_V * ex * ex
        + D31_PM_PER_V * ey * ey
        + D33_PM_PER_V * ez * ez
    )
    return output


def c1_interaction_window(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    coordinates = np.stack((x_grid, y_grid, z_grid), axis=-1)
    inside = np.all(np.abs(coordinates) <= INTERACTION_HALF_WIDTH + 1e-12, axis=-1)
    taper = np.prod(
        np.cos(0.5 * np.pi * coordinates / INTERACTION_HALF_WIDTH) ** 2,
        axis=-1,
    )
    return np.where(inside, taper, 0.0)


def run_pump(
    resolution: float,
    physical: dict[str, object],
    *,
    until_after_sources: float,
    beam_sigma: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    indices = dict(physical["indices"])
    pump_wavevector = np.asarray(physical["pump_wavevector"], dtype=np.float64)
    pump_electric = np.asarray(physical["pump_electric"], dtype=np.float64)
    amplitude, beam_center, group_tangent = pump_sheet_amplitude(
        pump_wavevector, indices, beam_sigma=beam_sigma
    )
    source_time = mp.GaussianSource(
        frequency=PUMP_FREQUENCY,
        fwidth=0.30,
        is_integrated=True,
    )
    source_size = PUMP_CELL_WIDTH - 2.0 * PUMP_PML_WIDTH
    sources = [
        mp.Source(
            source_time,
            component=component,
            center=mp.Vector3(0.0, 0.0, PUMP_SOURCE_Z),
            size=mp.Vector3(source_size, source_size, 0.0),
            amplitude=float(pump_electric[index]),
            amp_func=amplitude,
        )
        for index, component in enumerate((mp.Ex, mp.Ey, mp.Ez))
        if abs(pump_electric[index]) > 1e-14
    ]
    material = mp.Medium(
        epsilon_diag=mp.Vector3(
            float(indices["n_o_pump"]) ** 2,
            float(indices["n_o_pump"]) ** 2,
            float(indices["n_e_pump"]) ** 2,
        )
    )
    interaction_volume = mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(
            2.0 * INTERACTION_HALF_WIDTH,
            2.0 * INTERACTION_HALF_WIDTH,
            2.0 * INTERACTION_HALF_WIDTH,
        ),
    )
    sim = mp.Simulation(
        cell_size=mp.Vector3(PUMP_CELL_WIDTH, PUMP_CELL_WIDTH, PUMP_CELL_WIDTH),
        resolution=resolution,
        default_material=material,
        sources=sources,
        boundary_layers=[mp.PML(PUMP_PML_WIDTH)],
        force_complex_fields=True,
        progress_interval=30,
    )
    pump_dft = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez],
        PUMP_FREQUENCY,
        0.0,
        1,
        where=interaction_volume,
    )
    started = time.perf_counter()
    sim.run(until_after_sources=until_after_sources)
    runtime = time.perf_counter() - started
    pump_fields = np.stack(
        [
            np.asarray(sim.get_dft_array(pump_dft, component, 0), dtype=np.complex128)
            for component in (mp.Ex, mp.Ey, mp.Ez)
        ],
        axis=0,
    )
    x, y, z, weights = (
        np.asarray(value) for value in sim.get_array_metadata(dft_cell=pump_dft)
    )
    sim.reset_meep()

    window = c1_interaction_window(x, y, z)
    nonlinear_polarization_unwindowed = contract_3m_field(pump_fields)
    nonlinear_polarization = nonlinear_polarization_unwindowed * window[None, ...]
    field_energy = np.sum(weights * np.sum(np.abs(pump_fields) ** 2, axis=0))
    projection = np.einsum("i,ixyz->xyz", pump_electric, pump_fields)
    projected_fields = pump_electric[:, None, None, None] * projection[None, ...]
    polarization_residual = relative_weighted_l2(
        pump_fields, projected_fields, weights[None, ...]
    )
    p2_energy = np.sum(
        weights * np.sum(np.abs(nonlinear_polarization) ** 2, axis=0)
    )
    if not np.isfinite(field_energy) or field_energy <= 0.0:
        raise RuntimeError("pump FDTD produced no finite field energy in the interaction volume")
    if not np.isfinite(p2_energy) or p2_energy <= 0.0:
        raise RuntimeError("3m contraction produced no finite nonlinear polarization")

    summary = {
        "runtime_s": runtime,
        "array_shape": list(pump_fields.shape[1:]),
        "field_weighted_l2_norm": float(np.sqrt(field_energy)),
        "nonlinear_polarization_weighted_l2_norm_pm_per_v_scale": float(
            np.sqrt(p2_energy)
        ),
        "analytic_extraordinary_polarization_residual_l2": polarization_residual,
        "beam_center_on_source_plane": beam_center.tolist(),
        "group_tangent": group_tangent,
        "source_plane_z": PUMP_SOURCE_Z,
        "beam_sigma": beam_sigma,
        "source_count": len(sources),
    }
    arrays = {
        "x": x,
        "y": y,
        "z": z,
        "weights": weights,
        "pump_fields": pump_fields,
        "interaction_window": window,
        "nonlinear_polarization_unwindowed": nonlinear_polarization_unwindowed,
        "nonlinear_polarization": nonlinear_polarization,
    }
    return summary, arrays


def modal_geometry(
    physical: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    indices = dict(physical["indices"])
    return uniaxial_modes(
        frequency=SH_FREQUENCY,
        epsilon_perpendicular=float(indices["n_o_sh"]) ** 2,
        epsilon_parallel=float(indices["n_e_sh"]) ** 2,
        phi_degrees=MODE_PHI_DEGREES,
        parameter_degrees=MODE_PARAMETER_DEGREES,
    )


def cell_center_modal_reference(
    arrays: dict[str, np.ndarray],
    wavevectors: np.ndarray,
    electric_modes: np.ndarray,
    labels: list[dict[str, float | str]],
) -> dict[str, object]:
    x, y, z = arrays["x"], arrays["y"], arrays["z"]
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    coordinates = np.column_stack(
        (x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1))
    )
    quadrature_weights = arrays["weights"].reshape(-1)
    nonlinear_polarization = arrays["nonlinear_polarization"]
    direct_modal = np.zeros(wavevectors.shape[0], dtype=np.complex128)
    acfo_modal = np.zeros_like(direct_modal)
    nonzero_bins: list[int] = []
    for component_index in range(3):
        component_weights = (
            nonlinear_polarization[component_index].reshape(-1) * quadrature_weights
        )
        scalar_direct = smooth.direct_scalar_transform(
            coordinates, component_weights, wavevectors
        )
        direct_modal += electric_modes[:, component_index] * scalar_direct
        scalar_acfo, count = smooth.acfo_scalar_transform(
            coordinates,
            component_weights,
            wavevectors,
            labels,
            n_r=ACFO_GRID_N,
            n_z=ACFO_GRID_N,
            n_phi=ACFO_GRID_N,
        )
        acfo_modal += electric_modes[:, component_index] * scalar_acfo
        nonzero_bins.append(count)
    return {
        "direct_modal": direct_modal,
        "acfo_modal": acfo_modal,
        "acfo_vs_direct_relative_l2": relative_l2(acfo_modal, direct_modal),
        "nonzero_bins_by_component": nonzero_bins,
        "point_count": int(coordinates.shape[0]),
    }


def exact_sh_source_reference(
    sim: mp.Simulation,
    wavevectors: np.ndarray,
    electric_modes: np.ndarray,
    labels: list[dict[str, float | str]],
    *,
    resolution: float,
) -> dict[str, object]:
    margin = 2.0 / resolution
    width = 2.0 * INTERACTION_HALF_WIDTH + 2.0 * margin
    volume = mp.Volume(center=mp.Vector3(), size=mp.Vector3(width, width, width))
    x, y, z, weights = sim.get_array_metadata(vol=volume)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    base_coordinates = np.column_stack(
        (x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1))
    )
    quadrature_weights = np.asarray(weights).reshape(-1)
    direct_modal = np.zeros(wavevectors.shape[0], dtype=np.complex128)
    acfo_modal = np.zeros_like(direct_modal)
    active_counts: list[int] = []
    nonzero_bins: list[int] = []
    component_integrals: list[complex] = []
    spacing = 1.0 / resolution
    for component_index, component in enumerate((mp.Ex, mp.Ey, mp.Ez)):
        current = np.asarray(sim.get_source(component, vol=volume)).reshape(-1)
        component_coordinates = base_coordinates.copy()
        for axis in range(3):
            if axis != component_index:
                component_coordinates[:, axis] -= 0.5 * spacing
        active = np.abs(current) > 0.0
        coordinates = component_coordinates[active]
        component_weights = quadrature_weights[active]
        weighted_current = current[active] * component_weights
        scalar_direct = smooth.direct_scalar_transform(
            coordinates, weighted_current, wavevectors
        )
        direct_modal += electric_modes[:, component_index] * scalar_direct
        scalar_acfo, count = smooth.acfo_scalar_transform(
            coordinates,
            weighted_current,
            wavevectors,
            labels,
            n_r=ACFO_GRID_N,
            n_z=ACFO_GRID_N,
            n_phi=ACFO_GRID_N,
        )
        acfo_modal += electric_modes[:, component_index] * scalar_acfo
        active_counts.append(int(np.count_nonzero(active)))
        nonzero_bins.append(count)
        component_integrals.append(complex(np.sum(weighted_current)))
    return {
        "direct_modal": direct_modal,
        "acfo_modal": acfo_modal,
        "acfo_vs_direct_relative_l2": relative_l2(acfo_modal, direct_modal),
        "active_grid_points_by_component": active_counts,
        "nonzero_bins_by_component": nonzero_bins,
        "component_integrals": component_integrals,
    }


def build_sh_simulation(
    resolution: float,
    physical: dict[str, object],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nonlinear_polarization: np.ndarray,
    *,
    add_surface_monitors: bool,
) -> tuple[mp.Simulation, list[list[FaceMonitor]]]:
    source_time = mp.GaussianSource(
        frequency=SH_FREQUENCY, fwidth=0.4, is_integrated=True
    )
    source_volume = mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(
            2.0 * INTERACTION_HALF_WIDTH,
            2.0 * INTERACTION_HALF_WIDTH,
            2.0 * INTERACTION_HALF_WIDTH,
        ),
    )
    interpolators = [
        RegularGridInterpolator(
            (x, y, z),
            np.asarray(nonlinear_polarization[index], dtype=np.complex128),
            bounds_error=False,
            fill_value=0.0j,
        )
        for index in range(3)
    ]

    def amplitude_function(interpolator: RegularGridInterpolator):
        def amplitude(point: mp.Vector3) -> complex:
            return complex(interpolator((point.x, point.y, point.z)).item())

        return amplitude

    sources = [
        mp.Source(
            source_time,
            component=component,
            volume=source_volume,
            amp_func=amplitude_function(interpolators[index]),
        )
        for index, component in enumerate((mp.Ex, mp.Ey, mp.Ez))
    ]
    indices = dict(physical["indices"])
    material = mp.Medium(
        epsilon_diag=mp.Vector3(
            float(indices["n_o_sh"]) ** 2,
            float(indices["n_o_sh"]) ** 2,
            float(indices["n_e_sh"]) ** 2,
        )
    )
    sim = mp.Simulation(
        cell_size=mp.Vector3(SH_CELL_WIDTH, SH_CELL_WIDTH, SH_CELL_WIDTH),
        resolution=resolution,
        default_material=material,
        sources=sources,
        boundary_layers=[mp.PML(SH_PML_WIDTH)],
        force_complex_fields=True,
        progress_interval=30,
    )
    shells: list[list[FaceMonitor]] = []
    if add_surface_monitors:
        for half_width in MONITOR_HALF_WIDTHS:
            monitors: list[FaceMonitor] = []
            for center, size, normal, _ in face_specs(half_width):
                dft = sim.add_dft_fields(
                    list(COMPONENTS),
                    SH_FREQUENCY,
                    0.0,
                    1,
                    where=mp.Volume(center=center, size=size),
                )
                monitors.append(FaceMonitor(normal=normal, dft=dft))
            shells.append(monitors)
    return sim, shells


def run_resolution(
    resolution: float,
    physical: dict[str, object],
    calibration_gains: tuple[complex, complex] | None,
    *,
    pump_until_after_sources: float,
    sh_until_after_sources: float,
    pump_beam_sigma: float,
    reference_only: bool,
    output: Path,
) -> tuple[dict[str, object], list[np.ndarray] | None]:
    pump_summary, arrays = run_pump(
        resolution,
        physical,
        until_after_sources=pump_until_after_sources,
        beam_sigma=pump_beam_sigma,
    )
    wavevectors, electric_modes, labels = modal_geometry(physical)
    cell_reference = cell_center_modal_reference(
        arrays, wavevectors, electric_modes, labels
    )
    sim, shells = build_sh_simulation(
        resolution,
        physical,
        arrays["x"],
        arrays["y"],
        arrays["z"],
        arrays["nonlinear_polarization"],
        add_surface_monitors=not reference_only,
    )
    sim.init_sim()
    exact_reference = exact_sh_source_reference(
        sim,
        wavevectors,
        electric_modes,
        labels,
        resolution=resolution,
    )
    exact_direct = np.asarray(exact_reference["direct_modal"])
    exact_acfo = np.asarray(exact_reference["acfo_modal"])
    cell_direct = np.asarray(cell_reference["direct_modal"])
    cell_acfo = np.asarray(cell_reference["acfo_modal"])

    field_output = output.with_name(
        f"{output.stem}_fields_r{resolution:g}.npz"
    )
    np.savez_compressed(
        field_output,
        x=arrays["x"],
        y=arrays["y"],
        z=arrays["z"],
        integration_weights=arrays["weights"],
        pump_fields=arrays["pump_fields"],
        interaction_window=arrays["interaction_window"],
        nonlinear_polarization_unwindowed=arrays[
            "nonlinear_polarization_unwindowed"
        ],
        nonlinear_polarization=arrays["nonlinear_polarization"],
        wavevectors=wavevectors,
        electric_modes=electric_modes,
        cell_center_direct_modal=cell_direct,
        exact_yee_direct_modal=exact_direct,
    )
    field_hash = hashlib.sha256(field_output.read_bytes()).hexdigest()

    row: dict[str, object] = {
        "resolution": float(resolution),
        "pump": pump_summary,
        "cell_center_source": {
            "point_count": cell_reference["point_count"],
            "nonzero_bins_by_component": cell_reference[
                "nonzero_bins_by_component"
            ],
            "acfo_vs_direct_relative_l2": cell_reference[
                "acfo_vs_direct_relative_l2"
            ],
        },
        "exact_yee_source": {
            "active_grid_points_by_component": exact_reference[
                "active_grid_points_by_component"
            ],
            "nonzero_bins_by_component": exact_reference[
                "nonzero_bins_by_component"
            ],
            "component_integrals": pairs(
                np.asarray(exact_reference["component_integrals"])
            ),
            "acfo_vs_direct_relative_l2": exact_reference[
                "acfo_vs_direct_relative_l2"
            ],
            "direct_vs_cell_center_direct_relative_l2": relative_l2(
                exact_direct, cell_direct
            ),
            "acfo_vs_cell_center_acfo_relative_l2": relative_l2(
                exact_acfo, cell_acfo
            ),
        },
        "field_artifact": {
            "path": str(field_output),
            "sha256": field_hash,
            "bytes": field_output.stat().st_size,
        },
        "sample_labels": labels,
    }
    if reference_only:
        sim.reset_meep()
        return row, None
    if calibration_gains is None:
        raise ValueError("full cascade requires frozen point calibration gains")

    started = time.perf_counter()
    sim.run(until_after_sources=sh_until_after_sources)
    sh_runtime = time.perf_counter() - started
    raw_amplitudes: list[np.ndarray] = []
    calibrated_amplitudes: list[np.ndarray] = []
    quadrature_points: list[int] = []
    for shell_index, monitors in enumerate(shells):
        face_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        point_count = 0
        for face in monitors:
            coordinates, weights, fields = monitor_fields(sim, face)
            face_data.append((coordinates, weights, fields, face.normal))
            point_count += weights.size
        raw = reciprocal_surface_amplitudes(
            face_data,
            wavevectors,
            electric_modes,
            frequency=SH_FREQUENCY,
        )
        raw_amplitudes.append(raw)
        calibrated_amplitudes.append(calibration_gains[shell_index] * raw)
        quadrature_points.append(point_count)
    shell_rows: list[dict[str, object]] = []
    for shell_index, calibrated in enumerate(calibrated_amplitudes):
        shell_rows.append(
            {
                "point_calibration_gain": [
                    float(calibration_gains[shell_index].real),
                    float(calibration_gains[shell_index].imag),
                ],
                "fdtd_vs_exact_yee_direct_relative_l2": relative_l2(
                    calibrated, exact_direct
                ),
                "fdtd_vs_exact_yee_acfo_relative_l2": relative_l2(
                    calibrated, exact_acfo
                ),
                "fdtd_vs_cell_center_direct_relative_l2": relative_l2(
                    calibrated, cell_direct
                ),
                "fdtd_vs_cell_center_acfo_relative_l2": relative_l2(
                    calibrated, cell_acfo
                ),
                "branch_fdtd_vs_exact_yee_direct_relative_l2": branch_error(
                    calibrated, exact_direct, labels
                ),
                "calibrated_norm": float(np.linalg.norm(calibrated)),
            }
        )
    row.update(
        {
            "sh_runtime_s": sh_runtime,
            "quadrature_points": quadrature_points,
            "shells": shell_rows,
            "calibrated_inner_outer_relative_l2": relative_l2(
                calibrated_amplitudes[1], calibrated_amplitudes[0]
            ),
            "inner_calibrated_modal_amplitudes": complex_pairs(
                calibrated_amplitudes[0]
            ),
            "exact_yee_direct_modal_amplitudes": complex_pairs(exact_direct),
        }
    )
    sim.reset_meep()
    return row, calibrated_amplitudes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-way 5 mol% MgO:LiNbO3 cascade: pump FDTD to centered Yee pump "
            "field to local 3m contraction to SH impressed-current FDTD."
        )
    )
    parser.add_argument("--resolutions", default="20,24")
    parser.add_argument(
        "--pump-surface-parameter-deg",
        type=float,
        default=PUMP_SURFACE_PARAMETER_DEG,
    )
    parser.add_argument("--pump-azimuth-deg", type=float, default=PUMP_AZIMUTH_DEG)
    parser.add_argument("--pump-beam-sigma", type=float, default=PUMP_BEAM_SIGMA)
    parser.add_argument("--pump-until-after-sources", type=float, default=6.0)
    parser.add_argument("--sh-until-after-sources", type=float, default=6.0)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--frozen-holdout",
        action="store_true",
        help=(
            "Use the original 38-degree source-vector point calibration and omit "
            "a within-holdout grid gate; all other accuracy gates remain frozen."
        ),
    )
    parser.add_argument(
        "--point-calibration", type=Path, action="append", default=None
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/linbo3_one_way_shg_cascade.json"),
    )
    args = parser.parse_args()
    resolutions = tuple(float(value) for value in args.resolutions.split(","))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.pump_beam_sigma <= 0.0 or not np.isfinite(args.pump_beam_sigma):
        raise ValueError("pump-beam-sigma must be finite and positive")
    physical = physical_configuration(
        pump_surface_parameter_deg=args.pump_surface_parameter_deg,
        pump_azimuth_deg=args.pump_azimuth_deg,
    )
    baseline_physical = physical_configuration()
    if args.frozen_holdout:
        if args.reference_only:
            raise ValueError("frozen-holdout requires the full cascade")
        if (
            np.isclose(args.pump_surface_parameter_deg, PUMP_SURFACE_PARAMETER_DEG)
            and np.isclose(args.pump_azimuth_deg, PUMP_AZIMUTH_DEG)
            and np.isclose(args.pump_beam_sigma, PUMP_BEAM_SIGMA)
        ):
            raise ValueError("frozen-holdout must change at least one pump parameter")
    smooth.SOURCE_HALF_WIDTH = INTERACTION_HALF_WIDTH
    smooth.MODE_PHI_DEGREES = MODE_PHI_DEGREES
    smooth.REFERENCE_GRID_N = ACFO_GRID_N

    calibration_metadata: dict[str, object] | None = None
    calibration_gains: dict[float, tuple[complex, complex]] = {}
    if not args.reference_only:
        indices = dict(physical["indices"])
        paths = tuple(
            args.point_calibration
            or [Path("benchmark_results/linbo3_point_modal_calibration.json")]
        )
        calibration_gains, calibration_metadata = load_point_calibration(
            paths,
            resolutions,
            frequency=SH_FREQUENCY,
            epsilon_perpendicular=float(indices["n_o_sh"]) ** 2,
            epsilon_parallel=float(indices["n_e_sh"]) ** 2,
            cell_width=SH_CELL_WIDTH,
            pml_width=SH_PML_WIDTH,
            monitor_half_widths=MONITOR_HALF_WIDTHS,
            source_vector=np.asarray(
                baseline_physical["source_vector"]
                if args.frozen_holdout
                else physical["source_vector"]
            ),
        )

    rows: list[dict[str, object]] = []
    calibrated_by_resolution: dict[float, list[np.ndarray]] = {}
    for index, resolution in enumerate(resolutions, start=1):
        print(
            f"[{index}/{len(resolutions)}] one-way cascade resolution={resolution:g}",
            flush=True,
        )
        row, calibrated = run_resolution(
            resolution,
            physical,
            calibration_gains.get(resolution),
            pump_until_after_sources=args.pump_until_after_sources,
            sh_until_after_sources=args.sh_until_after_sources,
            pump_beam_sigma=args.pump_beam_sigma,
            reference_only=args.reference_only,
            output=args.output,
        )
        rows.append(row)
        if calibrated is not None:
            calibrated_by_resolution[resolution] = calibrated

    finest_resolution = max(resolutions)
    finest = rows[resolutions.index(finest_resolution)]
    reference_gates = {
        "pump_field_finite_nonzero": finest["pump"]["field_weighted_l2_norm"] > 0.0,
        "nonlinear_polarization_finite_nonzero": finest["pump"][
            "nonlinear_polarization_weighted_l2_norm_pm_per_v_scale"
        ]
        > 0.0,
        "cell_center_acfo_vs_direct_l2_le_3pct": finest["cell_center_source"][
            "acfo_vs_direct_relative_l2"
        ]
        <= 0.03,
        "exact_yee_acfo_vs_direct_l2_le_3pct": finest["exact_yee_source"][
            "acfo_vs_direct_relative_l2"
        ]
        <= 0.03,
        "exact_yee_vs_cell_center_direct_l2_le_5pct": finest[
            "exact_yee_source"
        ]["direct_vs_cell_center_direct_relative_l2"]
        <= 0.05,
    }
    grid_relative_l2 = float("nan")
    gates = dict(reference_gates)
    if not args.reference_only:
        if len(resolutions) >= 2:
            next_finest_resolution = sorted(resolutions)[-2]
            grid_relative_l2 = relative_l2(
                calibrated_by_resolution[next_finest_resolution][0],
                calibrated_by_resolution[finest_resolution][0],
            )
        full_gates = {
                "blind_fdtd_vs_exact_yee_direct_l2_le_5pct": finest["shells"][0][
                    "fdtd_vs_exact_yee_direct_relative_l2"
                ]
                <= 0.05,
                "blind_fdtd_vs_exact_yee_acfo_l2_le_5pct": finest["shells"][0][
                    "fdtd_vs_exact_yee_acfo_relative_l2"
                ]
                <= 0.05,
                "ordinary_l2_le_5pct": finest["shells"][0][
                    "branch_fdtd_vs_exact_yee_direct_relative_l2"
                ]["ordinary"]
                <= 0.05,
                "extraordinary_l2_le_5pct": finest["shells"][0][
                    "branch_fdtd_vs_exact_yee_direct_relative_l2"
                ]["extraordinary"]
                <= 0.05,
                "monitor_invariance_le_5pct": finest[
                    "calibrated_inner_outer_relative_l2"
                ]
                <= 0.05,
            }
        if len(resolutions) >= 2:
            full_gates["finest_next_finest_grid_l2_le_5pct"] = (
                grid_relative_l2 <= 0.05
            )
        elif not args.frozen_holdout:
            full_gates["finest_next_finest_grid_l2_le_5pct"] = False
        gates.update(full_gates)

    indices = dict(physical["indices"])
    result = {
        "schema": "linbo3-one-way-shg-cascade-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "homogeneous lossless bulk 5 mol% MgO:LiNbO3 one-way cascade: "
            "1064-nm pump FDTD, centered Yee-cell pump-field export, local 3m "
            "tensor contraction inside a compact C1 interaction window, and "
            "532-nm impressed-current FDTD with branch-resolved surface projection"
        ),
        "claim_boundary": (
            "pump propagation is solved by Maxwell FDTD and feeds the SH source, "
            "but the cascade remains undepleted and one-way; it excludes pump "
            "back-action, coupled nonlinear time stepping, interfaces, periodic "
            "poling, loss, and absolute conversion efficiency"
        ),
        "normalization_contract": (
            "the pump current amplitude and common J=-i omega P prefactor set only "
            "a global SH scale; no cascade-source fitting; independently frozen "
            "point-source complex gain per SH resolution and monitor shell"
        ),
        "material": {
            "name": "5 mol% MgO-doped congruent LiNbO3",
            "temperature_c": TEMPERATURE_C,
            "pump_wavelength_um": PUMP_WAVELENGTH_UM,
            "sh_wavelength_um": SH_WAVELENGTH_UM,
            **indices,
            "d22_pm_per_v": D22_PM_PER_V,
            "d31_pm_per_v": D31_PM_PER_V,
            "d33_pm_per_v": D33_PM_PER_V,
        },
        "configuration": {
            "resolutions": resolutions,
            "pump_frequency": PUMP_FREQUENCY,
            "sh_frequency": SH_FREQUENCY,
            "pump_surface_parameter_deg": float(
                physical["pump_surface_parameter_deg"]
            ),
            "pump_azimuth_deg": float(physical["pump_azimuth_deg"]),
            "pump_cell_width": PUMP_CELL_WIDTH,
            "pump_pml_width": PUMP_PML_WIDTH,
            "pump_source_z": PUMP_SOURCE_Z,
            "pump_beam_sigma": args.pump_beam_sigma,
            "pump_until_after_sources": args.pump_until_after_sources,
            "interaction_half_width": INTERACTION_HALF_WIDTH,
            "interaction_half_width_um": INTERACTION_HALF_WIDTH * SH_WAVELENGTH_UM,
            "interaction_window": "product cos^2 C1 apodization",
            "sh_cell_width": SH_CELL_WIDTH,
            "sh_pml_width": SH_PML_WIDTH,
            "monitor_half_widths": MONITOR_HALF_WIDTHS,
            "sh_until_after_sources": args.sh_until_after_sources,
            "mode_phi_degrees": MODE_PHI_DEGREES,
            "mode_parameter_degrees": MODE_PARAMETER_DEGREES,
            "acfo_grid_n": ACFO_GRID_N,
            "pump_dft_contract": (
                "PyMeep default centered-voxel DFT fields; Ex/Ey/Ez are bilinearly "
                "interpolated to the same Yee-cell centers before 3m contraction"
            ),
            "sh_source_contract": (
                "three complex P2 component arrays retain the centered pump-DFT "
                "coordinate axes and are trilinearly evaluated by amp_func at each "
                "electric-current Yee point; exact injected staggered currents are "
                "re-exported with get_source"
            ),
        },
        "reference_only": args.reference_only,
        "frozen_holdout": args.frozen_holdout,
        "holdout_contract": (
            {
                "training_case_pump_surface_parameter_deg": PUMP_SURFACE_PARAMETER_DEG,
                "holdout_pump_surface_parameter_deg": args.pump_surface_parameter_deg,
                "training_case_pump_azimuth_deg": PUMP_AZIMUTH_DEG,
                "holdout_pump_azimuth_deg": args.pump_azimuth_deg,
                "training_case_pump_beam_sigma": PUMP_BEAM_SIGMA,
                "holdout_pump_beam_sigma": args.pump_beam_sigma,
                "frozen_calibration_source_vector": np.asarray(
                    baseline_physical["source_vector"]
                ).tolist(),
                "holdout_nonlinear_source_vector": np.asarray(
                    physical["source_vector"]
                ).tolist(),
                "no_holdout_refit": True,
                "grid_convergence_source": (
                    "baseline r20/r24 cascade; the holdout is a single preselected r24 case"
                ),
            }
            if args.frozen_holdout
            else None
        ),
        "point_calibration": calibration_metadata,
        "rows": rows,
        "finest_next_finest_grid_relative_l2": (
            grid_relative_l2 if len(resolutions) >= 2 else None
        ),
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
