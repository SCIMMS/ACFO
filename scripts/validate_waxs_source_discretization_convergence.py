from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_protein_nanocrystal_finufft_fair import (  # noqa: E402
    detector_rectangle_mask,
    source_arrays,
)
from benchmark_protein_nanocrystal_sparse_memory import build_sparse_structure  # noqa: E402
from validate_public_waxs_crystals import choose_grid  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from validate_waxs_direct_finufft_triad import direct_active_amplitude  # noqa: E402
from waxs_cake import encode_elements  # noqa: E402
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def ring_mean(values: np.ndarray, q_indices: np.ndarray, nq: int) -> np.ndarray:
    values_i = intensity(values)
    return np.asarray([np.mean(values_i[q_indices == iq]) for iq in range(nq)])


def metrics(
    model: np.ndarray,
    reference: np.ndarray,
    q_indices: np.ndarray,
    nq: int,
) -> dict[str, float]:
    model_i = intensity(model)
    reference_i = intensity(reference)
    model_centered = model_i - np.mean(model_i)
    reference_centered = reference_i - np.mean(reference_i)
    ncc_denom = np.linalg.norm(model_centered) * np.linalg.norm(reference_centered)
    return {
        "complex_l2": relative_l2(model, reference),
        "intensity_l2": relative_l2(model_i, reference_i),
        "ring_intensity_l2": relative_l2(
            ring_mean(model, q_indices, nq),
            ring_mean(reference, q_indices, nq),
        ),
        "intensity_ncc": float(
            np.vdot(model_centered, reference_centered).real / ncc_denom
        ),
    }


def build_row(
    *,
    coords: np.ndarray,
    elements: np.ndarray,
    element_indices: np.ndarray,
    grid_template: dict,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    target_q_indices: np.ndarray,
    ff: np.ndarray,
    exact_atom: np.ndarray,
    nq: int,
    bin_width_nm: float,
    nphi_source: int,
    direct_target_chunk: int,
) -> dict:
    grid_args = argparse.Namespace(
        qmax=6.3,
        q_unit="inv_angstrom",
        bin_width_nm=bin_width_nm,
        harmonic_margin=32,
        nphi_detector=nphi_source,
    )
    grid = choose_grid(coords, grid_args)
    grid["n_phi"] = int(nphi_source)
    sparse, build_s = timed(
        lambda: build_sparse_structure(coords, elements, grid, index_backend="cpp")
    )
    sources, source_e, source_weights = source_arrays(
        "same_binned", coords, element_indices, sparse
    )
    binned, direct_s = timed(
        lambda: direct_active_amplitude(
            sources,
            source_e,
            source_weights,
            ff,
            qx,
            qy,
            qz,
            target_q_indices,
            target_chunk=direct_target_chunk,
        )
    )
    r_max = float(grid_template["r_max_nm"])
    return {
        "bin_width_nm": bin_width_nm,
        "nphi_source": int(nphi_source),
        "azimuth_arc_width_at_rmax_nm": 2.0 * np.pi * r_max / nphi_source,
        "active_bins": int(sparse.active_values.size),
        "sparse_storage_mib": sparse.sparse_storage_nbytes / 1024**2,
        "build_s": build_s,
        "direct_binned_s": direct_s,
        **metrics(binned, exact_atom, target_q_indices, nq),
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# WAXS high-q source-discretization convergence",
        "",
        "Detector target grid를 고정하고 cylindrical source representation만 바꾸어 exact-atom direct NDFT와 비교했다.",
        "",
        "## Azimuth-source convergence",
        "",
        "| source Nphi | arc at Rmax (nm) | complex L2 | intensity L2 | ring L2 | intensity NCC | storage MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["azimuth_sweep"]:
        lines.append(
            f"| {row['nphi_source']:,} | {row['azimuth_arc_width_at_rmax_nm']:.3e} "
            f"| {row['complex_l2']:.3e} | {row['intensity_l2']:.3e} "
            f"| {row['ring_intensity_l2']:.3e} | {row['intensity_ncc']:.6f} "
            f"| {row['sparse_storage_mib']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Radial/axial-bin convergence at finest source Nphi",
            "",
            "| bin width (nm) | complex L2 | intensity L2 | ring L2 | intensity NCC | storage MiB |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["bin_width_sweep"]:
        lines.append(
            f"| {row['bin_width_nm']:.7g} | {row['complex_l2']:.3e} "
            f"| {row['intensity_l2']:.3e} | {row['ring_intensity_l2']:.3e} "
            f"| {row['intensity_ncc']:.6f} | {row['sparse_storage_mib']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- finest exploratory intensity <=1%: **{'PASS' if result['finest_intensity_le_1pct'] else 'FAIL'}**",
            f"- finest exploratory ring intensity <=0.5%: **{'PASS' if result['finest_ring_intensity_le_0p5pct'] else 'FAIL'}**",
            "- 이 sweep은 operator가 아니라 atom-to-cylinder representation의 수렴성을 측정한다.",
            "- target 수를 고정했으므로 source Nphi 증가와 detector oversampling을 혼동하지 않는다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-q exact-atom vs sparse cylindrical source discretization convergence."
    )
    parser.add_argument(
        "structure",
        type=Path,
        nargs="?",
        default=Path("structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"),
    )
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--target-nphi", type=int, default=720)
    parser.add_argument("--direct-target-chunk", type=int, default=256)
    parser.add_argument("--detector-active-width-mm", type=float, default=155.1)
    parser.add_argument("--detector-active-height-mm", type=float, default=162.15)
    parser.add_argument("--detector-distance-mm", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/waxs_source_discretization_convergence.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/waxs_source_discretization_convergence.md"),
    )
    args = parser.parse_args()

    coords, elements, metadata = load_structure(args.structure)
    element_indices, element_order = encode_elements(elements)
    q_report = np.asarray([5.0, 5.65, 6.3], dtype=np.float64)
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    form_factors = build_form_factors(elements, q_solver, "xray_f0")
    ff = normalize_form_factors(element_order, q_solver, form_factors)
    phi = (np.arange(args.target_nphi) + 0.5) * (2.0 * np.pi / args.target_nphi)
    mask = detector_rectangle_mask(
        q_report,
        phi,
        wavelength_nm=args.wavelength_nm,
        active_width_mm=args.detector_active_width_mm,
        active_height_mm=args.detector_active_height_mm,
        distance_mm=args.detector_distance_mm,
    )
    target_q_indices = np.broadcast_to(np.arange(q_report.size)[:, None], mask.shape)[mask]
    q_perp, q_z_rows = ewald_ring(q_solver, args.wavelength_nm)
    qx = np.ascontiguousarray((q_perp[:, None] * np.cos(phi)[None, :])[mask])
    qy = np.ascontiguousarray((q_perp[:, None] * np.sin(phi)[None, :])[mask])
    qz = np.ascontiguousarray(np.broadcast_to(q_z_rows[:, None], mask.shape)[mask])
    exact_atom, exact_atom_s = timed(
        lambda: direct_active_amplitude(
            coords,
            element_indices,
            np.ones(coords.shape[0], dtype=np.float64),
            ff,
            qx,
            qy,
            qz,
            target_q_indices,
            target_chunk=args.direct_target_chunk,
        )
    )
    grid_template = choose_grid(
        coords,
        argparse.Namespace(
            qmax=6.3,
            q_unit="inv_angstrom",
            bin_width_nm=0.1,
            harmonic_margin=32,
            nphi_detector=128,
        ),
    )
    nphi_values = [750, 1500, 3000, 6000, 12000, 24000, 48000, 96000]
    fine_bin = 0.00078125
    azimuth_rows = [
        build_row(
            coords=coords,
            elements=elements,
            element_indices=element_indices,
            grid_template=grid_template,
            qx=qx,
            qy=qy,
            qz=qz,
            target_q_indices=target_q_indices,
            ff=ff,
            exact_atom=exact_atom,
            nq=q_report.size,
            bin_width_nm=fine_bin,
            nphi_source=nphi,
            direct_target_chunk=args.direct_target_chunk,
        )
        for nphi in nphi_values
    ]
    bin_widths = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, fine_bin]
    bin_rows = [
        build_row(
            coords=coords,
            elements=elements,
            element_indices=element_indices,
            grid_template=grid_template,
            qx=qx,
            qy=qy,
            qz=qz,
            target_q_indices=target_q_indices,
            ff=ff,
            exact_atom=exact_atom,
            nq=q_report.size,
            bin_width_nm=width,
            nphi_source=nphi_values[-1],
            direct_target_chunk=args.direct_target_chunk,
        )
        for width in bin_widths
    ]
    finest = bin_rows[-1]
    result = {
        "schema": "waxs-source-discretization-convergence-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "structure": metadata.get("structure_id", args.structure.stem),
        "structure_path": args.structure.as_posix(),
        "atom_count": int(coords.shape[0]),
        "q_inv_angstrom": q_report.tolist(),
        "fixed_target_nphi": args.target_nphi,
        "active_targets": int(mask.sum()),
        "exact_atom_direct_s": exact_atom_s,
        "contract": "fixed detector targets; direct exact atoms versus direct sparse cylindrical bin centers",
        "azimuth_sweep": azimuth_rows,
        "bin_width_sweep": bin_rows,
        "finest": finest,
        "finest_intensity_le_1pct": finest["intensity_l2"] <= 0.01,
        "finest_ring_intensity_le_0p5pct": finest["ring_intensity_l2"] <= 0.005,
        "passed": finest["intensity_l2"] <= 0.01,
        "claim_boundary": [
            "This is a source-representation convergence test, not an ACFO operator test.",
            "The detector target grid is held fixed while source azimuth and radial/axial bins change.",
            "The 1% and 0.5% thresholds are exploratory diagnostics, not preregistered publication gates.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(result, args.summary_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
