from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import meep as mp
import numpy as np


COMPONENTS = (mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz)


@dataclass
class FaceMonitor:
    normal: np.ndarray
    dft: object


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(model - reference) / denominator)


def calibrated_metrics(model: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    denominator = np.vdot(model, model)
    gain = np.vdot(model, reference) / denominator
    calibrated = gain * model
    correlation = abs(np.vdot(model, reference)) / (
        np.linalg.norm(model) * np.linalg.norm(reference)
    )
    return {
        "gain": [float(gain.real), float(gain.imag)],
        "relative_l2": relative_l2(calibrated, reference),
        "complex_correlation": float(correlation),
        "calibrated": calibrated,
    }


def directions_and_polarizations() -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    directions: list[np.ndarray] = []
    polarizations: list[np.ndarray] = []
    labels: list[dict[str, float]] = []
    for phi_deg in (0.0, 60.0, 120.0):
        phi = np.deg2rad(phi_deg)
        for theta_deg in np.linspace(15.0, 75.0, 9):
            theta = np.deg2rad(theta_deg)
            direction = np.array(
                (
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ),
                dtype=np.float64,
            )
            e_theta = np.array(
                (
                    np.cos(theta) * np.cos(phi),
                    np.cos(theta) * np.sin(phi),
                    -np.sin(theta),
                ),
                dtype=np.float64,
            )
            e_phi = np.array((-np.sin(phi), np.cos(phi), 0.0), dtype=np.float64)
            directions.extend((direction, direction))
            polarizations.extend((e_theta, e_phi))
            labels.extend(
                (
                    {"theta_deg": float(theta_deg), "phi_deg": phi_deg, "polarization": "theta"},
                    {"theta_deg": float(theta_deg), "phi_deg": phi_deg, "polarization": "phi"},
                )
            )
    return np.asarray(directions), np.asarray(polarizations), labels


def face_specs(half_width: float) -> list[tuple[mp.Vector3, mp.Vector3, np.ndarray, float]]:
    side = 2.0 * half_width
    specs: list[tuple[mp.Vector3, mp.Vector3, np.ndarray, float]] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            center = [0.0, 0.0, 0.0]
            center[axis] = sign * half_width
            size = [side, side, side]
            size[axis] = 0.0
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = sign
            specs.append(
                (
                    mp.Vector3(*center),
                    mp.Vector3(*size),
                    normal,
                    sign,
                )
            )
    return specs


def monitor_fields(sim: mp.Simulation, face: FaceMonitor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z, weights = sim.get_array_metadata(dft_cell=face.dft)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    coordinates = np.column_stack(
        (x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1))
    )
    weights_flat = np.asarray(weights, dtype=np.float64).reshape(-1)
    fields = np.column_stack(
        [np.asarray(sim.get_dft_array(face.dft, component, 0)).reshape(-1) for component in COMPONENTS]
    )
    if coordinates.shape[0] != weights_flat.size or fields.shape[0] != weights_flat.size:
        raise RuntimeError("Meep DFT field and quadrature shapes do not match")
    return coordinates, weights_flat, fields


def reciprocal_surface_amplitudes(
    face_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    directions: np.ndarray,
    polarizations: np.ndarray,
    *,
    frequency: float,
    refractive_index: float,
) -> np.ndarray:
    wave_number = 2.0 * np.pi * frequency * refractive_index
    amplitudes = np.zeros(directions.shape[0], dtype=np.complex128)
    for coordinates, weights, fields, normal in face_data:
        electric = fields[:, :3]
        magnetic = fields[:, 3:]
        for index, (direction, polarization) in enumerate(zip(directions, polarizations)):
            phase = np.exp(-1j * wave_number * (coordinates @ direction))
            reciprocal_electric = phase[:, None] * polarization[None, :]
            reciprocal_magnetic = (
                -refractive_index
                * phase[:, None]
                * np.cross(direction, polarization)[None, :]
            )
            integrand = np.einsum(
                "ij,j->i",
                np.cross(electric, reciprocal_magnetic)
                - np.cross(reciprocal_electric, magnetic),
                normal,
            )
            amplitudes[index] += np.sum(weights * integrand)
    return amplitudes


def near2far_amplitudes(
    sim: mp.Simulation,
    near2far: object,
    directions: np.ndarray,
    polarizations: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    amplitudes = np.empty(directions.shape[0], dtype=np.complex128)
    cache: dict[tuple[float, float, float], np.ndarray] = {}
    for index, (direction, polarization) in enumerate(zip(directions, polarizations)):
        key = tuple(float(value) for value in direction)
        if key not in cache:
            point = mp.Vector3(*(radius * direction))
            cache[key] = np.asarray(sim.get_farfield(near2far, point), dtype=np.complex128)
        amplitudes[index] = np.dot(cache[key][:3], polarization)
    return amplitudes


def run_resolution(
    resolution: float,
    *,
    frequency: float,
    refractive_index: float,
    cell_width: float,
    pml_width: float,
    monitor_half_widths: tuple[float, float],
    until_after_sources: float,
    far_radius: float,
) -> dict[str, object]:
    source_vector = np.array((0.36, 0.48, 0.80), dtype=np.float64)
    source_vector /= np.linalg.norm(source_vector)
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
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_width, cell_width, cell_width),
        resolution=resolution,
        default_material=mp.Medium(index=refractive_index),
        sources=sources,
        boundary_layers=[mp.PML(pml_width)],
        force_complex_fields=True,
        progress_interval=30,
    )
    shells: list[list[FaceMonitor]] = []
    near2far_objects: list[object] = []
    for half_width in monitor_half_widths:
        face_monitors: list[FaceMonitor] = []
        near_regions: list[mp.Near2FarRegion] = []
        for center, size, normal, weight in face_specs(half_width):
            dft = sim.add_dft_fields(
                list(COMPONENTS),
                frequency,
                0.0,
                1,
                where=mp.Volume(center=center, size=size),
            )
            face_monitors.append(FaceMonitor(normal=normal, dft=dft))
            near_regions.append(mp.Near2FarRegion(center=center, size=size, weight=weight))
        shells.append(face_monitors)
        near2far_objects.append(sim.add_near2far(frequency, 0.0, 1, *near_regions))

    started = time.perf_counter()
    sim.run(until_after_sources=until_after_sources)
    runtime = time.perf_counter() - started

    directions, polarizations, labels = directions_and_polarizations()
    reference = polarizations @ source_vector
    surface_amplitudes: list[np.ndarray] = []
    near2far_values: list[np.ndarray] = []
    quadrature_points: list[int] = []
    quadrature_areas: list[float] = []
    for face_monitors, near2far in zip(shells, near2far_objects):
        face_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        point_count = 0
        area = 0.0
        for face in face_monitors:
            coordinates, weights, fields = monitor_fields(sim, face)
            face_data.append((coordinates, weights, fields, face.normal))
            point_count += weights.size
            area += float(np.sum(weights))
        surface_amplitudes.append(
            reciprocal_surface_amplitudes(
                face_data,
                directions,
                polarizations,
                frequency=frequency,
                refractive_index=refractive_index,
            )
        )
        near2far_values.append(
            near2far_amplitudes(
                sim,
                near2far,
                directions,
                polarizations,
                radius=far_radius,
            )
        )
        quadrature_points.append(point_count)
        quadrature_areas.append(area)

    surface_metrics = [calibrated_metrics(value, reference) for value in surface_amplitudes]
    near2far_metrics = [calibrated_metrics(value, reference) for value in near2far_values]
    surface_cross = relative_l2(surface_amplitudes[1], surface_amplitudes[0])
    near2far_cross = relative_l2(near2far_values[1], near2far_values[0])
    cross_method = relative_l2(
        np.asarray(surface_metrics[0]["calibrated"]),
        np.asarray(near2far_metrics[0]["calibrated"]),
    )
    row = {
        "resolution": float(resolution),
        "runtime_s": runtime,
        "quadrature_points": quadrature_points,
        "quadrature_areas": quadrature_areas,
        "expected_surface_areas": [24.0 * value**2 for value in monitor_half_widths],
        "surface": [
            {key: value for key, value in metrics.items() if key != "calibrated"}
            for metrics in surface_metrics
        ],
        "near2far": [
            {key: value for key, value in metrics.items() if key != "calibrated"}
            for metrics in near2far_metrics
        ],
        "surface_inner_outer_raw_relative_l2": surface_cross,
        "near2far_inner_outer_raw_relative_l2": near2far_cross,
        "inner_surface_near2far_calibrated_relative_l2": cross_method,
        "reference_norm": float(np.linalg.norm(reference)),
        "surface_raw_norms": [float(np.linalg.norm(value)) for value in surface_amplitudes],
        "near2far_raw_norms": [float(np.linalg.norm(value)) for value in near2far_values],
        "sample_labels": labels,
    }
    sim.reset_meep()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small isotropic closed-surface modal-reciprocity probe against Meep near2far."
    )
    parser.add_argument("--resolutions", default="10,14")
    parser.add_argument("--cell-width", type=float, default=4.0)
    parser.add_argument("--pml-width", type=float, default=0.5)
    parser.add_argument("--monitor-half-widths", default="0.70,1.00")
    parser.add_argument("--until-after-sources", type=float, default=5.0)
    parser.add_argument("--far-radius", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/closed_surface_modal_reciprocity_probe.json"),
    )
    args = parser.parse_args()
    resolutions = tuple(float(value) for value in args.resolutions.split(","))
    monitor_half_widths = tuple(float(value) for value in args.monitor_half_widths.split(","))
    if len(monitor_half_widths) != 2 or not monitor_half_widths[0] < monitor_half_widths[1]:
        raise ValueError("monitor-half-widths must contain two increasing values")
    if monitor_half_widths[1] >= args.cell_width / 2.0 - args.pml_width:
        raise ValueError("outer monitor must lie strictly inside the non-PML region")
    frequency = 1.0
    refractive_index = 1.4
    rows: list[dict[str, object]] = []
    for index, resolution in enumerate(resolutions, start=1):
        print(f"[{index}/{len(resolutions)}] resolution={resolution:g}", flush=True)
        rows.append(
            run_resolution(
                resolution,
                frequency=frequency,
                refractive_index=refractive_index,
                cell_width=args.cell_width,
                pml_width=args.pml_width,
                monitor_half_widths=monitor_half_widths,
                until_after_sources=args.until_after_sources,
                far_radius=args.far_radius,
            )
        )
    finest = rows[int(np.argmax(resolutions))]
    custom_projection_gates = {
        "surface_shape_l2_le_5pct": finest["surface"][0]["relative_l2"] <= 0.05,
        "surface_monitor_invariance_le_5pct": finest["surface_inner_outer_raw_relative_l2"] <= 0.05,
        "surface_near2far_cross_l2_le_5pct": finest["inner_surface_near2far_calibrated_relative_l2"] <= 0.05,
    }
    built_in_reference_diagnostics = {
        "near2far_shape_l2_le_5pct": finest["near2far"][0]["relative_l2"] <= 0.05,
        "near2far_monitor_invariance_le_5pct": finest["near2far_inner_outer_raw_relative_l2"] <= 0.05,
    }
    result = {
        "schema": "closed-surface-modal-reciprocity-probe-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "isotropic point electric-current source; six-face Lorentz-reciprocity projection against Meep near2far",
        "normalization_contract": "one complex least-squares gain per method and shell against the analytic dipole angular amplitude; no angle-dependent fitting",
        "configuration": {
            "resolutions": resolutions,
            "frequency": frequency,
            "refractive_index": refractive_index,
            "cell_width": args.cell_width,
            "pml_width": args.pml_width,
            "monitor_half_widths": monitor_half_widths,
            "until_after_sources": args.until_after_sources,
            "far_radius": args.far_radius,
            "direction_count": 27,
            "modal_amplitude_count": 54,
        },
        "rows": rows,
        "custom_projection_gates": custom_projection_gates,
        "built_in_reference_diagnostics": built_in_reference_diagnostics,
        "custom_projection_passed": all(custom_projection_gates.values()),
        "built_in_reference_diagnostics_passed": all(
            built_in_reference_diagnostics.values()
        ),
        "passed": all(custom_projection_gates.values()),
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
