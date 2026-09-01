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

from benchmark_protein_nanocrystal_finufft_fair import detector_rectangle_mask  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from validate_waxs_direct_finufft_triad import direct_active_amplitude  # noqa: E402
from waxs_cake import (  # noqa: E402
    encode_elements,
    exact_coordinate_harmonic_amplitude_factorized,
    repeated_block_translations,
    translation_lattice_factor,
)
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def metrics(model: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {
        "complex_l2": relative_l2(model, reference),
        "intensity_l2": relative_l2(intensity(model), intensity(reference)),
    }


def write_markdown(result: dict, path: Path) -> None:
    unit = result["unit_harmonic"]
    lines = [
        "# Perfect protein-nanocrystal lattice-factorization control",
        "",
        "Exact-coordinate unit-cell harmonic amplitude와 finite translation lattice sum을 결합했다.",
        "",
        f"- unit exact harmonic vs direct NDFT complex L2: `{unit['complex_l2']:.3e}`",
        f"- unit harmonic seconds: `{unit['seconds']:.3f}`",
        "",
        "| supercell | atoms | cells | repetition residual | factorized full-target s | subset direct s | subset complex L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["supercells"]:
        lines.append(
            f"| {row['label']} | {row['atom_count']:,} | {row['cell_count']} "
            f"| {row['repetition_residual_nm']:.3e} | {row['factorized_full_target_seconds']:.3f} "
            f"| {row['direct_subset_seconds']:.3f} | {row['subset_complex_l2']:.3e} |"
        )
    defect = result["sparse_defect_control"]
    lines.extend(
        [
            "",
            "## Sparse positional-defect correction",
            "",
            f"- defect atoms: `{defect['defect_count']:,}` ({100*defect['defect_fraction']:.4f}%)",
            f"- displacement RMS: `{defect['realized_rms_nm']:.4f} nm`",
            f"- full-target delta correction: `{defect['correction_full_target_seconds']:.3f} s`",
            f"- corrected subset complex L2: `{defect['subset_complex_l2']:.3e}`",
            "",
        ]
    )
    lines.extend(
        [
            "",
            f"- exact periodic-control gates: **{'PASS' if result['passed'] else 'FAIL'}**",
            "- 이 경로는 perfect repeated crystal의 표준 구조인자×lattice-sum control이다.",
            "- disorder/defect가 있는 모든 원자구조를 자동으로 해결하지 않으며 ACFO novelty claim이 아니다.",
            "- sparse defects는 perfect background와 explicit delta-scatterer correction으로 확장할 수 있다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate exact unit-cell harmonic times finite translation lattice sum."
    )
    parser.add_argument(
        "--unit",
        type=Path,
        default=Path("structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"),
    )
    parser.add_argument(
        "--supercells",
        default=(
            "structures/processed/protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz,"
            "structures/processed/protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.npz"
        ),
    )
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--target-nphi", type=int, default=720)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--atom-chunk-size", type=int, default=256)
    parser.add_argument("--direct-subset-targets", type=int, default=64)
    parser.add_argument("--direct-target-chunk", type=int, default=8)
    parser.add_argument("--defect-fraction", type=float, default=0.001)
    parser.add_argument("--defect-rms-nm", type=float, default=0.02)
    parser.add_argument("--defect-seed", type=int, default=20260713)
    parser.add_argument("--detector-active-width-mm", type=float, default=155.1)
    parser.add_argument("--detector-active-height-mm", type=float, default=162.15)
    parser.add_argument("--detector-distance-mm", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/protein_nanocrystal_lattice_factorization.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/protein_nanocrystal_lattice_factorization.md"),
    )
    args = parser.parse_args()

    unit_coords, unit_elements, unit_metadata = load_structure(args.unit)
    unit_e, element_order = encode_elements(unit_elements)
    q_report = np.asarray([5.0, 5.65, 6.3], dtype=np.float64)
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    q_perp, q_z_rows = ewald_ring(q_solver, args.wavelength_nm)
    form_factors = build_form_factors(unit_elements, q_solver, "xray_f0")
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
    qx_full = q_perp[:, None] * np.cos(phi)[None, :]
    qy_full = q_perp[:, None] * np.sin(phi)[None, :]
    qz_full = np.broadcast_to(q_z_rows[:, None], qx_full.shape)
    qx_active = np.ascontiguousarray(qx_full[mask])
    qy_active = np.ascontiguousarray(qy_full[mask])
    qz_active = np.ascontiguousarray(qz_full[mask])

    (unit_harmonic, cutoffs), unit_harmonic_s = timed(
        lambda: exact_coordinate_harmonic_amplitude_factorized(
            unit_coords,
            q_perp,
            q_z_rows,
            phi,
            element_indices=unit_e,
            form_factors=ff,
            harmonic_margin=args.harmonic_margin,
        )
    )
    unit_direct, unit_direct_s = timed(
        lambda: direct_active_amplitude(
            unit_coords,
            unit_e,
            np.ones(unit_coords.shape[0]),
            ff,
            qx_active,
            qy_active,
            qz_active,
            target_q_indices,
            target_chunk=args.direct_target_chunk,
        )
    )
    unit_metrics = metrics(unit_harmonic[mask], unit_direct)
    unit_metrics.update(
        {
            "seconds": unit_harmonic_s,
            "direct_seconds": unit_direct_s,
            "maximum_harmonic": int(np.max(cutoffs)),
        }
    )

    active_count = int(mask.sum())
    subset_count = min(args.direct_subset_targets, active_count)
    subset = np.unique(
        np.linspace(0, active_count - 1, subset_count, dtype=np.int64)
    )
    rows = []
    defect_control = None
    for supercell_path_text in args.supercells.split(","):
        path = Path(supercell_path_text.strip())
        coords, elements, metadata = load_structure(path)
        if not np.array_equal(
            elements.reshape(-1, unit_elements.size),
            np.broadcast_to(unit_elements, (coords.shape[0] // unit_elements.size, unit_elements.size)),
        ):
            raise RuntimeError(f"{path} element blocks do not repeat the unit-cell ordering")
        translations, repetition_residual = repeated_block_translations(
            unit_coords, coords, atol=1e-9
        )
        lattice_active, lattice_s = timed(
            lambda: translation_lattice_factor(
                qx_active, qy_active, qz_active, translations
            )
        )
        factorized_active = unit_harmonic[mask] * lattice_active

        element_indices, _ = encode_elements(elements, element_order=element_order)
        direct_subset, direct_subset_s = timed(
            lambda: direct_active_amplitude(
                coords,
                element_indices,
                np.ones(coords.shape[0]),
                ff,
                qx_active[subset],
                qy_active[subset],
                qz_active[subset],
                target_q_indices[subset],
                target_chunk=args.direct_target_chunk,
            )
        )
        subset_metrics = metrics(factorized_active[subset], direct_subset)
        rows.append(
            {
                "label": "x".join(str(value) for value in metadata["supercell"]),
                "structure_path": path.as_posix(),
                "atom_count": int(coords.shape[0]),
                "cell_count": int(translations.shape[0]),
                "repetition_residual_nm": repetition_residual,
                "lattice_factor_seconds": lattice_s,
                "factorized_full_target_seconds": unit_harmonic_s + lattice_s,
                "direct_subset_targets": int(subset.size),
                "direct_subset_seconds": direct_subset_s,
                "subset_complex_l2": subset_metrics["complex_l2"],
                "subset_intensity_l2": subset_metrics["intensity_l2"],
            }
        )

        if metadata["supercell"] == [5, 5, 5]:
            rng = np.random.default_rng(args.defect_seed)
            defect_count = max(1, int(round(args.defect_fraction * coords.shape[0])))
            defect_indices = np.sort(
                rng.choice(coords.shape[0], size=defect_count, replace=False)
            )
            displacement = rng.normal(size=(defect_count, 3))
            displacement *= args.defect_rms_nm / np.sqrt(
                np.mean(np.sum(displacement**2, axis=1))
            )
            old_coords = np.ascontiguousarray(coords[defect_indices])
            new_coords = np.ascontiguousarray(old_coords + displacement)
            defect_e = np.ascontiguousarray(element_indices[defect_indices])
            correction_weights = np.ones(defect_count)
            (old_amp, new_amp), correction_s = timed(
                lambda: (
                    direct_active_amplitude(
                        old_coords,
                        defect_e,
                        correction_weights,
                        ff,
                        qx_active,
                        qy_active,
                        qz_active,
                        target_q_indices,
                        target_chunk=256,
                    ),
                    direct_active_amplitude(
                        new_coords,
                        defect_e,
                        correction_weights,
                        ff,
                        qx_active,
                        qy_active,
                        qz_active,
                        target_q_indices,
                        target_chunk=256,
                    ),
                )
            )
            corrected_active = factorized_active + new_amp - old_amp
            modified_coords = np.array(coords, copy=True)
            modified_coords[defect_indices] = new_coords
            direct_modified_subset, direct_modified_s = timed(
                lambda: direct_active_amplitude(
                    modified_coords,
                    element_indices,
                    np.ones(modified_coords.shape[0]),
                    ff,
                    qx_active[subset],
                    qy_active[subset],
                    qz_active[subset],
                    target_q_indices[subset],
                    target_chunk=args.direct_target_chunk,
                )
            )
            defect_metrics = metrics(
                corrected_active[subset], direct_modified_subset
            )
            defect_control = {
                "base_supercell": "5x5x5",
                "defect_model": "sparse independent positional displacement with exact old/new delta correction",
                "defect_count": defect_count,
                "defect_fraction": defect_count / coords.shape[0],
                "requested_rms_nm": args.defect_rms_nm,
                "realized_rms_nm": float(
                    np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
                ),
                "correction_full_target_seconds": correction_s,
                "factorized_plus_correction_seconds": (
                    unit_harmonic_s + lattice_s + correction_s
                ),
                "direct_modified_subset_targets": int(subset.size),
                "direct_modified_subset_seconds": direct_modified_s,
                "subset_complex_l2": defect_metrics["complex_l2"],
                "subset_intensity_l2": defect_metrics["intensity_l2"],
            }

    gates = {
        "unit_harmonic_complex_l2_le_1e_10": unit_metrics["complex_l2"] <= 1e-10,
        "all_repetition_residual_le_1e_9_nm": all(
            row["repetition_residual_nm"] <= 1e-9 for row in rows
        ),
        "all_supercell_subset_complex_l2_le_1e_9": all(
            row["subset_complex_l2"] <= 1e-9 for row in rows
        ),
        "sparse_defect_delta_subset_complex_l2_le_1e_9": (
            defect_control is not None
            and defect_control["subset_complex_l2"] <= 1e-9
        ),
    }
    result = {
        "schema": "protein-nanocrystal-lattice-factorization-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit_structure": unit_metadata.get("structure_id", args.unit.stem),
        "unit_atom_count": int(unit_coords.shape[0]),
        "q_inv_angstrom": q_report.tolist(),
        "target_nphi": args.target_nphi,
        "active_targets": active_count,
        "unit_harmonic": unit_metrics,
        "supercells": rows,
        "sparse_defect_control": defect_control,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "Perfect repeated protein nanocrystals admit an exact unit-cell structure-factor times finite lattice-sum control."
        ),
        "claim_boundary": [
            "This is the standard exact specialization for ordered repeated crystals, not an ACFO novelty claim.",
            "It is a validation/control path for perfect crystals and does not cover arbitrary atom-wise disorder.",
            "Sparse vacancy/substitution defects can be represented as explicit delta-scatterer corrections to the perfect background; dense disorder needs a separate contract.",
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
