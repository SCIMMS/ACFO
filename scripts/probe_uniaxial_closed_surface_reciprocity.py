from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import meep as mp
import numpy as np

from probe_closed_surface_modal_reciprocity import (
    COMPONENTS,
    FaceMonitor,
    calibrated_metrics,
    face_specs,
    monitor_fields,
    relative_l2,
)


def uniaxial_modes(
    *,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
    phi_degrees: tuple[float, ...] = (0.0, 60.0, 120.0),
    parameter_degrees: tuple[float, ...] = tuple(np.linspace(15.0, 75.0, 9)),
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    wavevectors: list[np.ndarray] = []
    electric_modes: list[np.ndarray] = []
    labels: list[dict[str, float | str]] = []
    k0 = 2.0 * np.pi * frequency
    for phi_deg in phi_degrees:
        phi = np.deg2rad(phi_deg)
        radial = np.array((np.cos(phi), np.sin(phi), 0.0), dtype=np.float64)
        ordinary_electric = np.array((-np.sin(phi), np.cos(phi), 0.0), dtype=np.float64)
        for parameter_deg in parameter_degrees:
            parameter = np.deg2rad(parameter_deg)

            ordinary_k = k0 * np.sqrt(epsilon_perpendicular) * (
                np.sin(parameter) * radial
                + np.cos(parameter) * np.array((0.0, 0.0, 1.0))
            )
            wavevectors.append(ordinary_k)
            electric_modes.append(ordinary_electric)
            labels.append(
                {
                    "branch": "ordinary",
                    "surface_parameter_deg": float(parameter_deg),
                    "phi_deg": phi_deg,
                }
            )

            k_perpendicular = k0 * np.sqrt(epsilon_parallel) * np.sin(parameter)
            k_z = k0 * np.sqrt(epsilon_perpendicular) * np.cos(parameter)
            extraordinary_k = k_perpendicular * radial + np.array((0.0, 0.0, k_z))
            extraordinary_electric = (
                (k_z / epsilon_perpendicular) * radial
                - np.array((0.0, 0.0, k_perpendicular / epsilon_parallel))
            )
            extraordinary_electric /= np.linalg.norm(extraordinary_electric)
            wavevectors.append(extraordinary_k)
            electric_modes.append(extraordinary_electric)
            labels.append(
                {
                    "branch": "extraordinary",
                    "surface_parameter_deg": float(parameter_deg),
                    "phi_deg": phi_deg,
                }
            )
    return np.asarray(wavevectors), np.asarray(electric_modes), labels


def reciprocal_surface_amplitudes(
    face_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    wavevectors: np.ndarray,
    electric_modes: np.ndarray,
    *,
    frequency: float,
) -> np.ndarray:
    angular_frequency = 2.0 * np.pi * frequency
    amplitudes = np.zeros(wavevectors.shape[0], dtype=np.complex128)
    for coordinates, weights, fields, normal in face_data:
        electric = fields[:, :3]
        magnetic = fields[:, 3:]
        for index, (wavevector, electric_mode) in enumerate(zip(wavevectors, electric_modes)):
            phase = np.exp(-1j * (coordinates @ wavevector))
            reciprocal_electric = phase[:, None] * electric_mode[None, :]
            reciprocal_magnetic = (
                -phase[:, None]
                * np.cross(wavevector, electric_mode)[None, :]
                / angular_frequency
            )
            integrand = np.einsum(
                "ij,j->i",
                np.cross(electric, reciprocal_magnetic)
                - np.cross(reciprocal_electric, magnetic),
                normal,
            )
            amplitudes[index] += np.sum(weights * integrand)
    return amplitudes


def branch_metrics(
    amplitudes: np.ndarray,
    reference: np.ndarray,
    labels: list[dict[str, float | str]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for branch in ("ordinary", "extraordinary"):
        mask = np.array([label["branch"] == branch for label in labels])
        metrics = calibrated_metrics(amplitudes[mask], reference[mask])
        output[branch] = {
            key: value for key, value in metrics.items() if key != "calibrated"
        }
    return output


def run_resolution(
    resolution: float,
    source_vector: np.ndarray,
    *,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
    cell_width: float,
    pml_width: float,
    monitor_half_widths: tuple[float, float],
    until_after_sources: float,
    mode_phi_degrees: tuple[float, ...],
    mode_parameter_degrees: tuple[float, ...],
) -> dict[str, object]:
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
            amplitude=float(source_vector[index]),
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

    started = time.perf_counter()
    sim.run(until_after_sources=until_after_sources)
    runtime = time.perf_counter() - started

    wavevectors, electric_modes, labels = uniaxial_modes(
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
        phi_degrees=mode_phi_degrees,
        parameter_degrees=mode_parameter_degrees,
    )
    reference = electric_modes @ source_vector
    amplitudes: list[np.ndarray] = []
    quadrature_points: list[int] = []
    for face_monitors in shells:
        face_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        point_count = 0
        for face in face_monitors:
            coordinates, weights, fields = monitor_fields(sim, face)
            face_data.append((coordinates, weights, fields, face.normal))
            point_count += weights.size
        amplitudes.append(
            reciprocal_surface_amplitudes(
                face_data,
                wavevectors,
                electric_modes,
                frequency=frequency,
            )
        )
        quadrature_points.append(point_count)

    combined_metrics = [calibrated_metrics(value, reference) for value in amplitudes]
    inner_gain = complex(*combined_metrics[0]["gain"])
    outer_gain = complex(*combined_metrics[1]["gain"])
    row = {
        "resolution": float(resolution),
        "runtime_s": runtime,
        "quadrature_points": quadrature_points,
        "combined": [
            {key: value for key, value in metrics.items() if key != "calibrated"}
            for metrics in combined_metrics
        ],
        "branches": [
            branch_metrics(value, reference, labels) for value in amplitudes
        ],
        "inner_outer_raw_relative_l2": relative_l2(amplitudes[1], amplitudes[0]),
        "inner_outer_shared_calibration_relative_l2": relative_l2(
            outer_gain * amplitudes[1], inner_gain * amplitudes[0]
        ),
        "reference_norm": float(np.linalg.norm(reference)),
        "surface_raw_norms": [float(np.linalg.norm(value)) for value in amplitudes],
        "sample_labels": labels,
    }
    sim.reset_meep()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small homogeneous-uniaxial closed-surface modal-reciprocity probe."
    )
    parser.add_argument("--resolutions", default="14,18")
    parser.add_argument("--cell-width", type=float, default=4.0)
    parser.add_argument("--pml-width", type=float, default=0.5)
    parser.add_argument("--monitor-half-widths", default="0.70,1.00")
    parser.add_argument("--until-after-sources", type=float, default=5.0)
    parser.add_argument("--n-ordinary", type=float, default=1.5)
    parser.add_argument("--n-extraordinary", type=float, default=1.8)
    parser.add_argument("--source-vector", default="0.36,0.48,0.80")
    parser.add_argument("--mode-phi-degrees", default="0,60,120")
    parser.add_argument(
        "--mode-parameter-degrees",
        default="15,22.5,30,37.5,45,52.5,60,67.5,75",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/uniaxial_closed_surface_reciprocity_probe.json"),
    )
    args = parser.parse_args()
    resolutions = tuple(float(value) for value in args.resolutions.split(","))
    monitor_half_widths = tuple(float(value) for value in args.monitor_half_widths.split(","))
    source_vector = np.asarray(
        tuple(float(value) for value in args.source_vector.split(",")),
        dtype=np.float64,
    )
    mode_phi_degrees = tuple(
        float(value) for value in args.mode_phi_degrees.split(",")
    )
    mode_parameter_degrees = tuple(
        float(value) for value in args.mode_parameter_degrees.split(",")
    )
    if source_vector.shape != (3,) or not np.all(np.isfinite(source_vector)):
        raise ValueError("source-vector must contain three finite real values")
    if np.linalg.norm(source_vector) == 0.0:
        raise ValueError("source-vector must be nonzero")
    if not mode_phi_degrees or not mode_parameter_degrees:
        raise ValueError("mode angle lists must be nonempty")
    if len(monitor_half_widths) != 2 or not monitor_half_widths[0] < monitor_half_widths[1]:
        raise ValueError("monitor-half-widths must contain two increasing values")
    if monitor_half_widths[1] >= args.cell_width / 2.0 - args.pml_width:
        raise ValueError("outer monitor must lie strictly inside the non-PML region")
    frequency = 1.0
    epsilon_perpendicular = args.n_ordinary**2
    epsilon_parallel = args.n_extraordinary**2
    rows: list[dict[str, object]] = []
    for index, resolution in enumerate(resolutions, start=1):
        print(f"[{index}/{len(resolutions)}] resolution={resolution:g}", flush=True)
        rows.append(
            run_resolution(
                resolution,
                source_vector,
                frequency=frequency,
                epsilon_perpendicular=epsilon_perpendicular,
                epsilon_parallel=epsilon_parallel,
                cell_width=args.cell_width,
                pml_width=args.pml_width,
                monitor_half_widths=monitor_half_widths,
                until_after_sources=args.until_after_sources,
                mode_phi_degrees=mode_phi_degrees,
                mode_parameter_degrees=mode_parameter_degrees,
            )
        )
    finest = rows[int(np.argmax(resolutions))]
    gates = {
        "combined_shape_l2_le_5pct": finest["combined"][0]["relative_l2"] <= 0.05,
        "ordinary_shape_l2_le_5pct": finest["branches"][0]["ordinary"]["relative_l2"] <= 0.05,
        "extraordinary_shape_l2_le_5pct": finest["branches"][0]["extraordinary"]["relative_l2"] <= 0.05,
        "shared_calibration_monitor_invariance_le_5pct": finest[
            "inner_outer_shared_calibration_relative_l2"
        ]
        <= 0.05,
    }
    result = {
        "schema": "uniaxial-closed-surface-modal-reciprocity-probe-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "homogeneous lossless uniaxial medium; point electric-current source; six-face reciprocity projection onto analytic ordinary and extraordinary plane-wave modes",
        "normalization_contract": "Euclidean-unit electric eigenvectors and one common complex least-squares gain across both branches per shell; no angle- or branch-dependent fitting for the combined gate",
        "configuration": {
            "resolutions": resolutions,
            "frequency": frequency,
            "n_ordinary": args.n_ordinary,
            "n_extraordinary": args.n_extraordinary,
            "epsilon_perpendicular": epsilon_perpendicular,
            "epsilon_parallel": epsilon_parallel,
            "source_vector": source_vector.tolist(),
            "cell_width": args.cell_width,
            "pml_width": args.pml_width,
            "monitor_half_widths": monitor_half_widths,
            "until_after_sources": args.until_after_sources,
            "mode_phi_degrees": mode_phi_degrees,
            "mode_parameter_degrees": mode_parameter_degrees,
            "ordinary_mode_count": len(mode_phi_degrees) * len(mode_parameter_degrees),
            "extraordinary_mode_count": len(mode_phi_degrees) * len(mode_parameter_degrees),
        },
        "rows": rows,
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
