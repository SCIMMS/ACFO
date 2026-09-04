from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm((candidate - reference).ravel())
        / max(np.linalg.norm(reference.ravel()), np.finfo(np.float64).tiny)
    )


def timed(function, warmup: int, repeats: int) -> tuple[np.ndarray, list[float]]:
    value = function()
    for _ in range(warmup):
        value = function()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        samples.append(time.perf_counter() - start)
    return np.asarray(value), samples


def direct_amplitude(
    coordinates_nm: np.ndarray,
    weights: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
) -> np.ndarray:
    phase = (
        coordinates_nm[:, 0, None] * qx.ravel()[None, :]
        + coordinates_nm[:, 1, None] * qy.ravel()[None, :]
        + coordinates_nm[:, 2, None] * qz.ravel()[None, :]
    )
    return (weights[:, None] * np.exp(1j * phase)).sum(axis=0).reshape(qx.shape)


def validate(
    structure: Path,
    output: Path,
    *,
    source_atoms: int,
    grid_sizes: tuple[int, ...],
    q_step_inverse_nm: float,
    wavelength_nm: float,
    eps: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    try:
        import finufft
    except ImportError as exc:
        raise RuntimeError("FINUFFT is required for the planar Type-1 boundary") from exc

    archive = np.load(structure, allow_pickle=False)
    coordinates = np.asarray(archive["coords"], dtype=np.float64)
    atomic_numbers = np.asarray(archive["atomic_numbers"], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("structure coordinates must have shape (natom, 3)")
    count = min(int(source_atoms), coordinates.shape[0])
    coordinates = coordinates[:count].copy()
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    weights = atomic_numbers[:count].astype(np.complex128)
    weights /= max(float(np.linalg.norm(weights)), np.finfo(np.float64).tiny)
    scaled_x = q_step_inverse_nm * coordinates[:, 0]
    scaled_y = q_step_inverse_nm * coordinates[:, 1]
    if np.max(np.abs(scaled_x)) >= np.pi or np.max(np.abs(scaled_y)) >= np.pi:
        raise ValueError("q step maps sources outside FINUFFT's principal interval")

    wavenumber = 2.0 * np.pi / wavelength_nm
    cases: list[dict[str, Any]] = []
    for grid_size in grid_sizes:
        if grid_size <= 0 or grid_size % 2:
            raise ValueError("grid sizes must be positive even integers")
        modes = np.arange(-grid_size // 2, grid_size // 2, dtype=np.float64)
        qx, qy = np.meshgrid(
            q_step_inverse_nm * modes,
            q_step_inverse_nm * modes,
            indexing="ij",
        )
        radial_squared = qx * qx + qy * qy
        if np.max(radial_squared) >= wavenumber * wavenumber:
            raise ValueError("requested reciprocal grid leaves the propagating Ewald cap")
        qz_exact = np.sqrt(wavenumber * wavenumber - radial_squared) - wavenumber
        qz_planar = np.zeros_like(qx)

        type1, type1_samples = timed(
            lambda: finufft.nufft2d1(
                scaled_x,
                scaled_y,
                weights,
                (grid_size, grid_size),
                eps=eps,
                isign=1,
                nthreads=1,
            ),
            warmup,
            repeats,
        )
        planar_direct, planar_direct_samples = timed(
            lambda: direct_amplitude(
                coordinates, weights, qx, qy, qz_planar
            ),
            0,
            max(1, min(3, repeats)),
        )
        exact_ewald, exact_samples = timed(
            lambda: direct_amplitude(coordinates, weights, qx, qy, qz_exact),
            0,
            max(1, min(3, repeats)),
        )
        type1_error = relative_l2(type1, planar_direct)
        geometry_error = relative_l2(planar_direct, exact_ewald)
        cases.append(
            {
                "grid_size": grid_size,
                "target_count": grid_size * grid_size,
                "maximum_transverse_q_inverse_nm": float(
                    np.sqrt(np.max(radial_squared))
                ),
                "maximum_exact_qz_magnitude_inverse_nm": float(
                    np.max(np.abs(qz_exact))
                ),
                "type1_vs_planar_direct_relative_l2": type1_error,
                "exact_ewald_vs_planar_geometry_relative_l2": geometry_error,
                "timing_diagnostic_seconds": {
                    "type1_samples": type1_samples,
                    "type1_median": float(np.median(type1_samples)),
                    "planar_direct_samples": planar_direct_samples,
                    "planar_direct_median": float(np.median(planar_direct_samples)),
                    "exact_ewald_direct_samples": exact_samples,
                    "exact_ewald_direct_median": float(np.median(exact_samples)),
                    "publication_speed_claim_eligible": False,
                },
            }
        )

    threshold = 5.0e-10
    passed = bool(
        cases
        and all(row["type1_vs_planar_direct_relative_l2"] <= threshold for row in cases)
    )
    result = {
        "schema": "waxs-planar-representation-dispatch-boundary-v1",
        "passed": passed,
        "scope": (
            "Representation-dispatch boundary only. Type-1 is tested on a uniform "
            "planar reciprocal grid; exact Ewald curvature is retained as a geometry "
            "difference and is not used as a Type-1 speed comparator for ACFO."
        ),
        "structure": str(structure),
        "source_atoms": count,
        "q_step_inverse_nm": q_step_inverse_nm,
        "wavelength_nm": wavelength_nm,
        "finufft_eps": eps,
        "finufft_version": getattr(finufft, "__version__", None),
        "thresholds": {
            "type1_vs_planar_direct_relative_l2_max": threshold,
            "geometry_error_has_no_optimized_pass_fail_threshold": True,
        },
        "dispatch_classification": {
            "uniform_planar_reciprocal_grid": "Type-1 eligible",
            "exact_ewald_curved_targets": "Type-3 or orbit-factorized ACFO eligible",
            "headline_speed_comparison_between_these_geometries": "forbidden",
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-atoms", type=int, default=512)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=(16, 24, 32))
    parser.add_argument("--q-step-inverse-nm", type=float, default=0.35)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--eps", type=float, default=1.0e-12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.source_atoms <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("source-atoms and repeats must be positive; warmup non-negative")
    result = validate(
        args.structure,
        args.output,
        source_atoms=args.source_atoms,
        grid_sizes=tuple(args.grid_sizes),
        q_step_inverse_nm=args.q_step_inverse_nm,
        wavelength_nm=args.wavelength_nm,
        eps=args.eps,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
