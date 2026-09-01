from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import meep as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_uniaxial_meep_3d_phase import fixed_source_grid, grid_amplitude  # noqa: E402
from waxs_cake import linbo3_3m_nonlinear_polarization  # noqa: E402


def source_vector(mode: str) -> np.ndarray:
    if mode == "nonlinear":
        pump = np.array(
            (0.25, 0.35, np.sqrt(1.0 - 0.25**2 - 0.35**2)),
            dtype=np.complex128,
        )
        vector = linbo3_3m_nonlinear_polarization(pump)
    else:
        vector = np.zeros(3, dtype=np.complex128)
        vector[{"x": 0, "y": 1, "z": 2}[mode]] = 1.0
    return vector / np.linalg.norm(vector)


def make_sources(
    *,
    kind: str,
    vector: np.ndarray,
    half_width: float,
    source_grid_n: int,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
) -> list[mp.Source]:
    source_time = mp.GaussianSource(
        frequency=frequency, fwidth=0.30, is_integrated=True
    )
    components = (mp.Ex, mp.Ey, mp.Ez)
    if kind == "pattern":
        grid = fixed_source_grid(
            n=source_grid_n,
            half_width=half_width,
            frequency=frequency,
            epsilon_perpendicular=epsilon_perpendicular,
            epsilon_parallel=epsilon_parallel,
        )
        amplitude = grid_amplitude(grid, half_width=half_width)
        return [
            mp.Source(
                source_time,
                component=component,
                center=mp.Vector3(),
                size=mp.Vector3(2 * half_width, 2 * half_width, 2 * half_width),
                amplitude=complex(vector[index]),
                amp_func=amplitude,
            )
            for index, component in enumerate(components)
        ]
    if kind == "point":
        return [
            mp.Source(
                source_time,
                component=component,
                center=mp.Vector3(),
                amplitude=complex(vector[index]),
            )
            for index, component in enumerate(components)
        ]
    raise ValueError("kind must be pattern or point")


def extract(
    *,
    kind: str,
    vector: np.ndarray,
    resolution: float,
    cell_width: float,
    pml_width: float,
    half_width: float,
    source_grid_n: int,
    frequency: float,
    epsilon_perpendicular: float,
    epsilon_parallel: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    material = mp.Medium(
        epsilon_diag=mp.Vector3(
            epsilon_perpendicular, epsilon_perpendicular, epsilon_parallel
        )
    )
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_width, cell_width, cell_width),
        resolution=resolution,
        default_material=material,
        sources=make_sources(
            kind=kind,
            vector=vector,
            half_width=half_width,
            source_grid_n=source_grid_n,
            frequency=frequency,
            epsilon_perpendicular=epsilon_perpendicular,
            epsilon_parallel=epsilon_parallel,
        ),
        boundary_layers=[mp.PML(pml_width)],
        force_complex_fields=True,
    )
    sim.init_sim()
    margin = 2.0 / resolution
    export_width = 2.0 * half_width + 2.0 * margin
    volume = mp.Volume(
        center=mp.Vector3(), size=mp.Vector3(export_width, export_width, export_width)
    )
    metadata = tuple(np.asarray(value) for value in sim.get_array_metadata(vol=volume))
    components = (mp.Ex, mp.Ey, mp.Ez)
    arrays = np.stack(
        [np.asarray(sim.get_source(component, vol=volume)) for component in components],
        axis=0,
    )
    sim.reset_meep()
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export exact PyMeep Yee-grid source arrays.")
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--cell-width", type=float, required=True)
    parser.add_argument("--pml-width", type=float, required=True)
    parser.add_argument("--source-half-width", type=float, required=True)
    parser.add_argument("--source-grid-n", type=int, default=64)
    parser.add_argument(
        "--source-vector", choices=["nonlinear", "x", "y", "z"], default="nonlinear"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frequency = 1.0
    epsilon_perpendicular = 2.319393**2
    epsilon_parallel = 2.224439**2
    vector = source_vector(args.source_vector)
    pattern, pattern_metadata = extract(
        kind="pattern",
        vector=vector,
        resolution=args.resolution,
        cell_width=args.cell_width,
        pml_width=args.pml_width,
        half_width=args.source_half_width,
        source_grid_n=args.source_grid_n,
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
    )
    point, point_metadata = extract(
        kind="point",
        vector=vector,
        resolution=args.resolution,
        cell_width=args.cell_width,
        pml_width=args.pml_width,
        half_width=args.source_half_width,
        source_grid_n=args.source_grid_n,
        frequency=frequency,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
    )
    for left, right in zip(pattern_metadata, point_metadata, strict=True):
        if not np.array_equal(left, right):
            raise RuntimeError("pattern and point source metadata do not match")
    x, y, z, weights = pattern_metadata
    payload = {
        "schema": "pymeep-yee-source-export-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": args.resolution,
        "cell_width": args.cell_width,
        "pml_width": args.pml_width,
        "source_half_width": args.source_half_width,
        "source_grid_n": args.source_grid_n,
        "source_vector_mode": args.source_vector,
        "source_vector": [[float(value.real), float(value.imag)] for value in vector],
        "array_shape": list(pattern.shape),
        "pattern_nonzero_by_component": [
            int(np.count_nonzero(component)) for component in pattern
        ],
        "point_nonzero_by_component": [int(np.count_nonzero(component)) for component in point],
        "meep": mp.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pattern_sources=pattern,
        point_sources=point,
        x=x,
        y=y,
        z=z,
        integration_weights=weights,
        metadata_json=np.array(json.dumps(payload)),
    )
    receipt = args.output.with_suffix(".json")
    receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.output} and {receipt}")


if __name__ == "__main__":
    main()
