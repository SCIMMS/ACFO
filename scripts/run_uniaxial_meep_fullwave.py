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

from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    PreparedAxisymmetricOperator,
    linbo3_3m_nonlinear_polarization,
    make_cylindrical_histogram,
)


PROFILES = {
    "smoke": {
        "resolutions": (8.0, 10.0, 12.0),
        "angles_deg": np.linspace(5.0, 85.0, 21),
        "until_after_sources": 8.0,
    },
    "production": {
        "resolutions": (14.0, 18.0, 22.0),
        "angles_deg": np.linspace(5.0, 85.0, 81),
        "until_after_sources": 12.0,
    },
}


def object_scalar(
    x: float,
    y: float,
    z: float,
    *,
    half_width: float,
    frequency: float,
    observation_azimuth: float,
) -> complex:
    ux = x / half_width
    uy = y / half_width
    uz = z / half_width
    inside = (ux / 0.82) ** 4 + (uy / 0.70) ** 4 + (uz / 0.90) ** 4 <= 1.0
    if not inside:
        return 0.0j
    hologram = np.cos(2.3 * ux + 1.1 * uy - 0.7 * uz) + 0.55 * np.cos(
        -0.8 * ux + 2.0 * uy + 1.3 * uz + 0.4
    )
    domain = 1.0 if hologram >= 0.0 else -1.0
    theta0 = np.deg2rad(38.0)
    wave_number = 2.0 * np.pi * frequency
    carrier = wave_number * (
        np.sin(theta0) * np.cos(observation_azimuth) * x
        + np.sin(theta0) * np.sin(observation_azimuth) * y
        + np.cos(theta0) * z
    )
    return complex(domain * np.exp(1j * carrier))


def reference_object(
    *, n: int, half_width: float, frequency: float, observation_azimuth: float
) -> tuple[np.ndarray, np.ndarray, float]:
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
    domain = np.where(hologram >= 0.0, 1.0, -1.0)
    theta0 = np.deg2rad(38.0)
    wave_number = 2.0 * np.pi * frequency
    carrier = np.exp(
        1j
        * wave_number
        * (
            np.sin(theta0) * np.cos(observation_azimuth) * x
            + np.sin(theta0) * np.sin(observation_azimuth) * y
            + np.cos(theta0) * z
        )
    )
    coords = np.column_stack((x[mask], y[mask], z[mask]))
    weights = domain[mask] * carrier[mask] * spacing**3
    return coords, weights.astype(np.complex128), spacing


def transverse_source(
    angles_deg: np.ndarray, source_vector: np.ndarray, *, observation_azimuth: float
) -> np.ndarray:
    theta = np.deg2rad(angles_deg)
    directions = np.column_stack(
        (
            np.sin(theta) * np.cos(observation_azimuth),
            np.sin(theta) * np.sin(observation_azimuth),
            np.cos(theta),
        )
    )
    return source_vector[None, :] - directions * (directions @ source_vector)[:, None]


def born_fields(
    angles_deg: np.ndarray,
    *,
    n: int,
    half_width: float,
    frequency: float,
    source_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    n_phi = n
    observation_azimuth = -np.pi / n_phi
    coords, weights, spacing = reference_object(
        n=n,
        half_width=half_width,
        frequency=frequency,
        observation_azimuth=observation_azimuth,
    )
    binned = make_cylindrical_histogram(
        coords,
        atom_weights=weights,
        n_r=n,
        n_z=n,
        n_phi=n_phi,
        r_max=np.sqrt(2.0) * half_width,
        z_range=(-half_width, half_width),
        hist_dtype=np.complex128,
        backend="numpy",
    )
    theta = np.deg2rad(angles_deg)
    wave_number = 2.0 * np.pi * frequency
    manifold = AxisymmetricManifold(
        theta,
        wave_number * np.sin(theta),
        -wave_number * np.cos(theta),
        name="vacuum-outgoing-sphere",
        interpretation="dispersion-derived",
        frequency_units="inverse_length",
    )
    acfo_all = PreparedAxisymmetricOperator(binned, manifold, complex_dtype=np.complex128).forward(binned.hist)
    phi_index = int(np.argmin(np.abs(np.angle(np.exp(1j * (binned.beta_centers - np.pi))))))
    acfo_scalar = acfo_all[:, phi_index]
    directions = np.column_stack(
        (
            np.sin(theta) * np.cos(observation_azimuth),
            np.sin(theta) * np.sin(observation_azimuth),
            np.cos(theta),
        )
    )
    q = -wave_number * directions
    background_scalar = np.empty(angles_deg.size, dtype=np.complex128)
    for index, target in enumerate(q):
        background_scalar[index] = weights @ np.exp(1j * (coords @ target))
    hist = np.asarray(binned.hist[0], dtype=np.complex128)
    ir, iz, ib = np.nonzero(hist)
    binned_weights = hist[ir, iz, ib]
    beta = binned.beta_centers[ib]
    binned_positions = np.column_stack(
        (
            binned.r_centers[ir] * np.cos(beta),
            binned.r_centers[ir] * np.sin(beta),
            binned.z_centers[iz],
        )
    )
    direct_scalar = np.empty(angles_deg.size, dtype=np.complex128)
    for index, target in enumerate(q):
        direct_scalar[index] = binned_weights @ np.exp(1j * (binned_positions @ target))
    polarization = transverse_source(
        angles_deg, source_vector, observation_azimuth=observation_azimuth
    )
    return (
        acfo_scalar[:, None] * polarization,
        direct_scalar[:, None] * polarization,
        background_scalar[:, None] * polarization,
        {
            "reference_shape": [n, n, n],
            "active_voxels": int(coords.shape[0]),
            "spacing": spacing,
            "nonzero_cylindrical_bins": int(np.count_nonzero(binned.hist)),
            "selected_phi_rad": float(binned.beta_centers[phi_index]),
            "observation_azimuth_rad": observation_azimuth,
        },
    )


def near2far_regions(half_size: float) -> list[mp.Near2FarRegion]:
    side = 2.0 * half_size
    return [
        mp.Near2FarRegion(center=mp.Vector3(-half_size, 0, 0), size=mp.Vector3(0, side, side), weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(+half_size, 0, 0), size=mp.Vector3(0, side, side), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -half_size, 0), size=mp.Vector3(side, 0, side), weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, +half_size, 0), size=mp.Vector3(side, 0, side), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, 0, -half_size), size=mp.Vector3(side, side, 0), weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, 0, +half_size), size=mp.Vector3(side, side, 0), weight=+1),
    ]


def run_meep_case(
    angles_deg: np.ndarray,
    *,
    resolution: float,
    contrast_scale: float,
    geometry_kind: str,
    half_width: float,
    frequency: float,
    source_vector: np.ndarray,
    until_after_sources: float,
    observation_azimuth: float,
) -> tuple[np.ndarray, float]:
    cell_width = 3.6
    pml_width = 0.45
    monitor_half_size = 1.15
    vacuum = mp.Medium()
    delta_perpendicular = 0.080
    delta_parallel = 0.060
    material = mp.Medium(
        epsilon_diag=mp.Vector3(
            1.0 + contrast_scale * delta_perpendicular,
            1.0 + contrast_scale * delta_perpendicular,
            1.0 + contrast_scale * delta_parallel,
        )
    )

    def material_at(point: mp.Vector3) -> mp.Medium:
        if geometry_kind == "background":
            return vacuum
        if geometry_kind == "correct":
            ux = point.x / half_width
            uy = point.y / half_width
            uz = point.z / half_width
            inside = (ux / 0.82) ** 4 + (uy / 0.70) ** 4 + (uz / 0.90) ** 4 <= 1.0
        elif geometry_kind == "forced_sphere":
            radius = 0.92
            inside = point.x * point.x + point.y * point.y + point.z * point.z <= radius * radius
        else:
            raise ValueError(f"unknown geometry_kind: {geometry_kind}")
        return material if inside else vacuum

    def amplitude(point: mp.Vector3) -> complex:
        return object_scalar(
            point.x,
            point.y,
            point.z,
            half_width=half_width,
            frequency=frequency,
            observation_azimuth=observation_azimuth,
        )

    source_time = mp.GaussianSource(frequency=frequency, fwidth=0.30, is_integrated=True)
    components = (mp.Ex, mp.Ey, mp.Ez)
    sources = [
        mp.Source(
            source_time,
            component=component,
            center=mp.Vector3(),
            size=mp.Vector3(2.0 * half_width, 2.0 * half_width, 2.0 * half_width),
            amplitude=complex(source_vector[index]),
            amp_func=amplitude,
        )
        for index, component in enumerate(components)
        if abs(source_vector[index]) > 1e-14
    ]
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_width, cell_width, cell_width),
        resolution=resolution,
        sources=sources,
        boundary_layers=[mp.PML(pml_width)],
        material_function=material_at,
        force_complex_fields=True,
        eps_averaging=True,
        progress_interval=30,
    )
    monitor = sim.add_near2far(frequency, 0.0, 1, *near2far_regions(monitor_half_size))
    started = time.perf_counter()
    sim.run(until_after_sources=until_after_sources)
    far_radius = 1000.0
    result = np.empty((angles_deg.size, 3), dtype=np.complex128)
    for index, angle_deg in enumerate(angles_deg):
        theta = np.deg2rad(angle_deg)
        far = sim.get_farfield(
            monitor,
            mp.Vector3(
                far_radius * np.sin(theta) * np.cos(observation_azimuth),
                far_radius * np.sin(theta) * np.sin(observation_azimuth),
                far_radius * np.cos(theta),
            ),
        )
        result[index] = np.asarray(far[:3], dtype=np.complex128)
    elapsed = time.perf_counter() - started
    sim.reset_meep()
    return result, elapsed


def save_partial(
    output: Path,
    *,
    angles_deg: np.ndarray,
    contrast_scales: np.ndarray,
    resolutions: np.ndarray,
    background_born_field: np.ndarray,
    background_fullwave_field: np.ndarray,
    acfo_field: np.ndarray,
    direct_born_field: np.ndarray,
    fullwave_field: np.ndarray,
    forced_sphere_field: np.ndarray,
    metadata: dict[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        angles_deg=angles_deg,
        contrast_scales=contrast_scales,
        resolutions=resolutions,
        background_born_field=background_born_field,
        background_fullwave_field=background_fullwave_field,
        acfo_field=acfo_field,
        direct_born_field=direct_born_field,
        fullwave_field=fullwave_field,
        forced_sphere_field=forced_sphere_field,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ACFO 3-D PyMeep weak-contrast full-wave gate.")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/uniaxial_meep_fullwave_smoke.npz"))
    parser.add_argument("--reference-n", type=int, default=64)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    angles_deg = np.asarray(profile["angles_deg"], dtype=np.float64)
    resolutions = np.asarray(profile["resolutions"], dtype=np.float64)
    contrast_scales = np.asarray((1.0, 0.5, 0.25, 0.125), dtype=np.float64)
    half_width = 0.55
    frequency = 1.0
    pump = np.array((0.25, 0.35, np.sqrt(1.0 - 0.25**2 - 0.35**2)), dtype=np.complex128)
    source_vector = linbo3_3m_nonlinear_polarization(pump)
    source_vector /= np.linalg.norm(source_vector)
    acfo_one, direct_one, background, reference_metadata = born_fields(
        angles_deg,
        n=args.reference_n,
        half_width=half_width,
        frequency=frequency,
        source_vector=source_vector,
    )
    observation_azimuth = float(reference_metadata["observation_azimuth_rad"])
    acfo = np.broadcast_to(acfo_one, (contrast_scales.size,) + acfo_one.shape).copy()
    direct = np.broadcast_to(direct_one, (contrast_scales.size,) + direct_one.shape).copy()
    shape_background = (resolutions.size, angles_deg.size, 3)
    shape_full = (resolutions.size, contrast_scales.size, angles_deg.size, 3)
    background_fullwave = np.full(shape_background, np.nan + 1j * np.nan, dtype=np.complex128)
    fullwave = np.full(shape_full, np.nan + 1j * np.nan, dtype=np.complex128)
    forced = np.full(shape_full, np.nan + 1j * np.nan, dtype=np.complex128)
    runtimes: list[dict[str, object]] = []
    generated = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, object] = {
        "schema": "uniaxial-meep-fullwave-v1",
        "generated_at_utc": generated,
        "profile": args.profile,
        "meep": mp.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "reference": reference_metadata,
        "source": {
            "type": "64-cubed off-axis binary two-carrier LiNbO3 chi2 impressed current",
            "normalized_vector": [[float(value.real), float(value.imag)] for value in source_vector],
            "frequency": frequency,
            "half_width": half_width,
        },
        "material": {
            "background_epsilon": 1.0,
            "delta_epsilon_perpendicular_at_scale_1": 0.080,
            "delta_epsilon_parallel_at_scale_1": 0.060,
            "scope": "controlled weak uniaxial perturbation retaining LiNbO3 source tensor; not bulk-index LiNbO3",
        },
        "cell": {"width": 3.6, "pml_width": 0.45, "near2far_half_size": 1.15},
        "runtimes": runtimes,
    }
    partial = args.output.with_name(args.output.stem + ".partial.npz")
    total_cases = resolutions.size * (1 + 2 * contrast_scales.size)
    case_number = 0
    for ir, resolution in enumerate(resolutions):
        case_number += 1
        print(f"[{case_number}/{total_cases}] resolution={resolution:g} background", flush=True)
        field, elapsed = run_meep_case(
            angles_deg,
            resolution=float(resolution),
            contrast_scale=0.0,
            geometry_kind="background",
            half_width=half_width,
            frequency=frequency,
            source_vector=source_vector,
            until_after_sources=float(profile["until_after_sources"]),
            observation_azimuth=observation_azimuth,
        )
        background_fullwave[ir] = field
        runtimes.append({"resolution": float(resolution), "contrast_scale": 0.0, "geometry": "background", "seconds": elapsed})
        for ic, contrast in enumerate(contrast_scales):
            for kind, target in (("correct", fullwave), ("forced_sphere", forced)):
                case_number += 1
                print(
                    f"[{case_number}/{total_cases}] resolution={resolution:g} contrast={contrast:g} {kind}",
                    flush=True,
                )
                field, elapsed = run_meep_case(
                    angles_deg,
                    resolution=float(resolution),
                    contrast_scale=float(contrast),
                    geometry_kind=kind,
                    half_width=half_width,
                    frequency=frequency,
                    source_vector=source_vector,
                    until_after_sources=float(profile["until_after_sources"]),
                    observation_azimuth=observation_azimuth,
                )
                target[ir, ic] = field
                runtimes.append(
                    {
                        "resolution": float(resolution),
                        "contrast_scale": float(contrast),
                        "geometry": kind,
                        "seconds": elapsed,
                    }
                )
                save_partial(
                    partial,
                    angles_deg=angles_deg,
                    contrast_scales=contrast_scales,
                    resolutions=resolutions,
                    background_born_field=background,
                    background_fullwave_field=background_fullwave,
                    acfo_field=acfo,
                    direct_born_field=direct,
                    fullwave_field=fullwave,
                    forced_sphere_field=forced,
                    metadata=metadata,
                )
    save_partial(
        args.output,
        angles_deg=angles_deg,
        contrast_scales=contrast_scales,
        resolutions=resolutions,
        background_born_field=background,
        background_fullwave_field=background_fullwave,
        acfo_field=acfo,
        direct_born_field=direct,
        fullwave_field=fullwave,
        forced_sphere_field=forced,
        metadata=metadata,
    )
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
