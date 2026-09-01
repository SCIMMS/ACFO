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
from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    PreparedAxisymmetricOperator,
    make_cylindrical_histogram,
)


def complex_pairs(values: np.ndarray) -> list[list[float]]:
    flattened = np.asarray(values, dtype=np.complex128).reshape(-1)
    return [[float(value.real), float(value.imag)] for value in flattened]


def sparse_distributed_source() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    n_r = 32
    n_z = 32
    n_phi = 32
    r_max = 0.5
    z_min = -0.4
    z_max = 0.4
    dr = r_max / n_r
    dz = (z_max - z_min) / n_z
    dbeta = 2.0 * np.pi / n_phi
    coordinates: list[tuple[float, float, float]] = []
    weights: list[complex] = []
    for ir in (6, 13, 20):
        radius = (ir + 0.5) * dr
        for iz in (6, 15, 24):
            z = z_min + (iz + 0.5) * dz
            for ib in range(0, n_phi, 4):
                beta = (ib + 0.5) * dbeta
                envelope = np.exp(-0.5 * (radius / 0.28) ** 2 - 0.5 * (z / 0.24) ** 2)
                modulation = 1.0 + 0.22 * np.cos(3.0 * beta + 0.35)
                phase = np.exp(1j * (2.0 * beta + 5.0 * z + 1.7 * radius))
                coordinates.append(
                    (radius * np.cos(beta), radius * np.sin(beta), z)
                )
                weights.append(complex(envelope * modulation * phase))
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.complex128)
    weight_array /= np.linalg.norm(weight_array)
    metadata = {
        "point_count": int(coordinate_array.shape[0]),
        "cylindrical_grid": {
            "n_r": n_r,
            "n_z": n_z,
            "n_phi": n_phi,
            "r_max": r_max,
            "z_range": [z_min, z_max],
        },
        "radial_indices": [6, 13, 20],
        "axial_indices": [6, 15, 24],
        "azimuthal_indices": list(range(0, n_phi, 4)),
        "weight_l2_norm": float(np.linalg.norm(weight_array)),
        "max_radius": float(np.max(np.linalg.norm(coordinate_array[:, :2], axis=1))),
        "max_abs_z": float(np.max(np.abs(coordinate_array[:, 2]))),
    }
    return coordinate_array, weight_array, metadata


def build_modal_reference(
    coordinates: np.ndarray,
    weights: np.ndarray,
    source_vector: np.ndarray,
    *,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
) -> dict[str, object]:
    wavevectors, electric_modes, labels = uniaxial_modes(
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
        phi_degrees=(5.625, 61.875, 118.125),
    )
    source_coupling = electric_modes @ source_vector
    direct_scalar = np.asarray(
        [weights @ np.exp(-1j * (coordinates @ wavevector)) for wavevector in wavevectors]
    )
    direct_modal = direct_scalar * source_coupling

    binned = make_cylindrical_histogram(
        coordinates,
        atom_weights=weights,
        n_r=32,
        n_z=32,
        n_phi=32,
        r_max=0.5,
        z_range=(-0.4, 0.4),
        hist_dtype=np.complex128,
        backend="numpy",
    )
    acfo_scalar = np.empty_like(direct_scalar)
    first_phi_deg = float(labels[0]["phi_deg"])
    for branch in ("ordinary", "extraordinary"):
        branch_indices = [
            index
            for index, label in enumerate(labels)
            if label["branch"] == branch
            and np.isclose(float(label["phi_deg"]), first_phi_deg)
        ]
        parameters_deg = np.asarray(
            [labels[index]["surface_parameter_deg"] for index in branch_indices],
            dtype=np.float64,
        )
        branch_wavevectors = wavevectors[branch_indices]
        q_perpendicular = np.linalg.norm(branch_wavevectors[:, :2], axis=1)
        q_z = -branch_wavevectors[:, 2]
        manifold = AxisymmetricManifold(
            np.deg2rad(parameters_deg),
            q_perpendicular,
            q_z,
            name=f"distributed-source-{branch}",
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
            beta_distance = np.abs(
                np.angle(np.exp(1j * (binned.beta_centers - target_beta)))
            )
            beta_index = int(np.argmin(beta_distance))
            acfo_scalar[index] = acfo_all[parameter_index, beta_index]
    acfo_modal = acfo_scalar * source_coupling
    return {
        "wavevectors": wavevectors,
        "electric_modes": electric_modes,
        "labels": labels,
        "direct_scalar": direct_scalar,
        "direct_modal": direct_modal,
        "acfo_scalar": acfo_scalar,
        "acfo_modal": acfo_modal,
        "algorithm_relative_l2": relative_l2(acfo_modal, direct_modal),
        "nonzero_bins": int(np.count_nonzero(binned.hist)),
    }


def load_point_calibration(
    paths: tuple[Path, ...],
    resolutions: tuple[float, ...],
    *,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
    cell_width: float,
    pml_width: float,
    monitor_half_widths: tuple[float, float],
    source_vector: np.ndarray | None = None,
) -> tuple[dict[float, tuple[complex, complex]], dict[str, object]]:
    expected = {
        "frequency": frequency,
        "epsilon_perpendicular": epsilon_perpendicular,
        "epsilon_parallel": epsilon_parallel,
        "cell_width": cell_width,
        "pml_width": pml_width,
    }
    rows: dict[float, dict[str, object]] = {}
    source_files: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        configuration = payload["configuration"]
        for key, value in expected.items():
            if not np.isclose(float(configuration[key]), value):
                raise ValueError(
                    f"point calibration {path} {key} does not match distributed probe"
                )
        if not np.allclose(configuration["monitor_half_widths"], monitor_half_widths):
            raise ValueError(
                f"point calibration {path} monitor shells do not match distributed probe"
            )
        if source_vector is not None:
            if "source_vector" not in configuration or not np.allclose(
                configuration["source_vector"], source_vector
            ):
                raise ValueError(
                    f"point calibration {path} source vector does not match distributed probe"
                )
        for row in payload["rows"]:
            resolution = float(row["resolution"])
            if resolution in rows:
                raise ValueError(f"duplicate point calibration resolution {resolution:g}")
            rows[resolution] = row
        source_files.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "schema": payload["schema"],
                "generated_at_utc": payload["generated_at_utc"],
                "resolutions": [float(row["resolution"]) for row in payload["rows"]],
            }
        )
    gains: dict[float, tuple[complex, complex]] = {}
    for resolution in resolutions:
        if resolution not in rows:
            raise ValueError(f"point calibration has no resolution {resolution:g}")
        gains[resolution] = tuple(
            complex(*shell["gain"]) for shell in rows[resolution]["combined"]
        )
    metadata = {
        "source_files": source_files,
        "contract": "point-source combined ordinary+extraordinary gain is frozen before distributed-source evaluation",
    }
    return gains, metadata


def branch_error(
    model: np.ndarray,
    reference: np.ndarray,
    labels: list[dict[str, float | str]],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for branch in ("ordinary", "extraordinary"):
        mask = np.array([label["branch"] == branch for label in labels])
        output[branch] = relative_l2(model[mask], reference[mask])
    return output


def run_resolution(
    resolution: float,
    coordinates: np.ndarray,
    weights: np.ndarray,
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
    sources: list[mp.Source] = []
    for coordinate, spatial_weight in zip(coordinates, weights):
        center = mp.Vector3(*coordinate)
        for component_index, component in enumerate((mp.Ex, mp.Ey, mp.Ez)):
            amplitude = complex(spatial_weight * source_vector[component_index])
            sources.append(
                mp.Source(
                    source_time,
                    component=component,
                    center=center,
                    amplitude=amplitude,
                )
            )
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
            surface_coordinates, surface_weights, fields = monitor_fields(sim, face)
            face_data.append(
                (surface_coordinates, surface_weights, fields, face.normal)
            )
            point_count += surface_weights.size
        raw = reciprocal_surface_amplitudes(
            face_data,
            wavevectors,
            electric_modes,
            frequency=frequency,
        )
        raw_amplitudes.append(raw)
        calibrated_amplitudes.append(calibration_gains[shell_index] * raw)
        quadrature_points.append(point_count)

    direct_modal = np.asarray(reference["direct_modal"])
    acfo_modal = np.asarray(reference["acfo_modal"])
    shell_rows: list[dict[str, object]] = []
    for shell_index, calibrated in enumerate(calibrated_amplitudes):
        shell_rows.append(
            {
                "point_calibration_gain": [
                    float(calibration_gains[shell_index].real),
                    float(calibration_gains[shell_index].imag),
                ],
                "fdtd_vs_direct_relative_l2": relative_l2(calibrated, direct_modal),
                "fdtd_vs_acfo_relative_l2": relative_l2(calibrated, acfo_modal),
                "branch_fdtd_vs_direct_relative_l2": branch_error(
                    calibrated, direct_modal, labels
                ),
                "calibrated_norm": float(np.linalg.norm(calibrated)),
            }
        )
    row = {
        "resolution": float(resolution),
        "runtime_s": runtime,
        "meep_source_count": len(sources),
        "quadrature_points": quadrature_points,
        "shells": shell_rows,
        "raw_inner_outer_relative_l2": relative_l2(
            raw_amplitudes[1], raw_amplitudes[0]
        ),
        "calibrated_inner_outer_relative_l2": relative_l2(
            calibrated_amplitudes[1], calibrated_amplitudes[0]
        ),
        "inner_calibrated_modal_amplitudes": complex_pairs(calibrated_amplitudes[0]),
    }
    sim.reset_meep()
    return row, calibrated_amplitudes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind point-calibrated distributed-source uniaxial FDTD-to-ACFO modal gate."
    )
    parser.add_argument("--resolutions", default="14,18")
    parser.add_argument("--cell-width", type=float, default=4.0)
    parser.add_argument("--pml-width", type=float, default=0.5)
    parser.add_argument("--monitor-half-widths", default="0.70,1.00")
    parser.add_argument("--until-after-sources", type=float, default=5.0)
    parser.add_argument(
        "--point-calibration",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/uniaxial_distributed_source_modal_probe.json"),
    )
    args = parser.parse_args()
    resolutions = tuple(float(value) for value in args.resolutions.split(","))
    monitor_half_widths = tuple(float(value) for value in args.monitor_half_widths.split(","))
    if len(monitor_half_widths) != 2 or not monitor_half_widths[0] < monitor_half_widths[1]:
        raise ValueError("monitor-half-widths must contain two increasing values")
    frequency = 1.0
    epsilon_perpendicular = 1.5**2
    epsilon_parallel = 1.8**2
    source_vector = np.array((0.36, 0.48, 0.80), dtype=np.float64)
    source_vector /= np.linalg.norm(source_vector)
    coordinates, weights, source_metadata = sparse_distributed_source()
    reference = build_modal_reference(
        coordinates,
        weights,
        source_vector,
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
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
            coordinates,
            weights,
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
    direct_modal = np.asarray(reference["direct_modal"])
    acfo_modal = np.asarray(reference["acfo_modal"])
    algorithm_relative_l2 = float(reference["algorithm_relative_l2"])
    gates = {
        "acfo_vs_direct_l2_le_1e_10": algorithm_relative_l2 <= 1e-10,
        "blind_fdtd_vs_direct_l2_le_5pct": finest["shells"][0]["fdtd_vs_direct_relative_l2"] <= 0.05,
        "blind_fdtd_vs_acfo_l2_le_5pct": finest["shells"][0]["fdtd_vs_acfo_relative_l2"] <= 0.05,
        "ordinary_l2_le_5pct": finest["shells"][0]["branch_fdtd_vs_direct_relative_l2"]["ordinary"] <= 0.05,
        "extraordinary_l2_le_5pct": finest["shells"][0]["branch_fdtd_vs_direct_relative_l2"]["extraordinary"] <= 0.05,
        "monitor_invariance_le_5pct": finest["calibrated_inner_outer_relative_l2"] <= 0.05,
        "finest_next_finest_grid_l2_le_5pct": len(resolutions) >= 2 and grid_relative_l2 <= 0.05,
    }
    result = {
        "schema": "uniaxial-distributed-source-modal-probe-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "72-point sparse distributed electric-current source in a homogeneous lossless uniaxial medium; FDTD closed-surface modal amplitudes compared blindly with direct Cartesian and ACFO references",
        "normalization_contract": "no distributed-source fitting; complex gains are frozen from the independent point-source ordinary+extraordinary calibration at the same resolution and shell",
        "configuration": {
            "resolutions": resolutions,
            "frequency": frequency,
            "epsilon_perpendicular": epsilon_perpendicular,
            "epsilon_parallel": epsilon_parallel,
            "cell_width": args.cell_width,
            "pml_width": args.pml_width,
            "monitor_half_widths": monitor_half_widths,
            "until_after_sources": args.until_after_sources,
            "source_vector": source_vector.tolist(),
            "ordinary_mode_count": 27,
            "extraordinary_mode_count": 27,
        },
        "source": source_metadata,
        "point_calibration": calibration_metadata,
        "reference": {
            "nonzero_bins": reference["nonzero_bins"],
            "acfo_vs_direct_relative_l2": algorithm_relative_l2,
            "direct_modal_norm": float(np.linalg.norm(direct_modal)),
            "acfo_modal_norm": float(np.linalg.norm(acfo_modal)),
            "direct_modal_amplitudes": complex_pairs(direct_modal),
            "acfo_modal_amplitudes": complex_pairs(acfo_modal),
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
