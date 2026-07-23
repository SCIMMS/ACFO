from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_protein_nanocrystal_sparse_memory import build_sparse_structure  # noqa: E402
from validate_public_waxs_crystals import choose_grid  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import PreparedCakePlan  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep sparse ACFO margin and precision against a saved FINUFFT amplitude."
    )
    parser.add_argument("structure", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--margins", default="16,24,32,48,64")
    parser.add_argument("--complex-dtypes", default="complex64,complex128")
    parser.add_argument("--bin-width-nm", type=float, default=0.1)
    parser.add_argument("--q-unit", choices=["inv_angstrom", "inv_nm"], default="inv_angstrom")
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument("--cutoff-bin-size", type=int, default=16)
    parser.add_argument("--q-block-size", type=int, default=2)
    parser.add_argument("--profile-chunk-size", type=int, default=8)
    parser.add_argument("--wavelength-nm", type=float, default=0.1)
    parser.add_argument("--form-factor-model", default="xray_f0")
    parser.add_argument("--complex-gate", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/protein_nanocrystal_sparse_accuracy_sweep.json"),
    )
    args = parser.parse_args()

    with np.load(args.reference, allow_pickle=False) as payload:
        finufft_amp = np.asarray(payload["finufft"], dtype=np.complex128)
        q_report = np.asarray(payload["q_report"], dtype=np.float64)
        q_solver = np.asarray(payload["q_solver"], dtype=np.float64)
        phi = np.asarray(payload["phi"], dtype=np.float64)
        reference_metadata = json.loads(str(payload["metadata_json"]))

    coords, elements, metadata = load_structure(args.structure)
    grid_args = argparse.Namespace(
        qmax=float(q_report[-1]),
        q_unit=args.q_unit,
        bin_width_nm=args.bin_width_nm,
        harmonic_margin=args.harmonic_margin,
        nphi_detector=phi.size,
    )
    grid = choose_grid(coords, grid_args)
    if grid["n_phi"] != phi.size:
        raise ValueError("reference phi size does not match the reconstructed physical grid")
    form_factors = build_form_factors(elements, q_solver, args.form_factor_model)
    sparse = build_sparse_structure(coords, elements, grid, index_backend="cpp")
    finufft_i = intensity(finufft_amp)

    rows = []
    for dtype_name in [part.strip() for part in args.complex_dtypes.split(",") if part.strip()]:
        dtype = np.dtype(dtype_name)
        plan_start = time.perf_counter()
        plan = PreparedCakePlan(
            sparse,
            q_solver,
            args.wavelength_nm,
            form_factors=form_factors,
            circular_backend="cpp",
            complex_dtype=dtype,
            q_block_size=args.q_block_size,
        )
        plan_s = time.perf_counter() - plan_start
        for margin in parse_ints(args.margins):
            start = time.perf_counter()
            amplitude = plan.circular_fft_sparse_source_r_dependent(
                margin=margin,
                cutoff_bin_size=args.cutoff_bin_size,
                analytic_kernel=True,
                q_block_size=args.q_block_size,
                profile_chunk_size=args.profile_chunk_size,
            )
            solve_s = time.perf_counter() - start
            amp_i = intensity(amplitude)
            complex_error = relative_l2(amplitude, finufft_amp)
            rows.append(
                {
                    "complex_dtype": dtype.name,
                    "margin": margin,
                    "plan_s": plan_s,
                    "solve_s": solve_s,
                    "complex_l2": complex_error,
                    "intensity_l2": relative_l2(amp_i, finufft_i),
                    "ring_l2": relative_l2(
                        np.mean(amp_i, axis=1),
                        np.mean(finufft_i, axis=1),
                    ),
                    "complex_gate": args.complex_gate,
                    "complex_gate_pass": bool(complex_error <= args.complex_gate),
                    "all_finite": bool(np.all(np.isfinite(amplitude))),
                }
            )

    passing = [row for row in rows if row["complex_gate_pass"]]
    selected = min(passing, key=lambda row: row["solve_s"]) if passing else None
    result = {
        "structure_path": args.structure.as_posix(),
        "structure_id": metadata.get("structure_id", args.structure.stem),
        "reference_path": args.reference.as_posix(),
        "reference_finufft_eps": reference_metadata.get("finufft_eps"),
        "reference_finufft_threads": reference_metadata.get("finufft_threads"),
        "atoms": int(coords.shape[0]),
        "nq": int(q_solver.size),
        "n_phi": int(phi.size),
        "qmax": float(q_report[-1]),
        "bin_width_nm": args.bin_width_nm,
        "rows": rows,
        "selected_fastest_passing": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Protein nanocrystal sparse accuracy sweep",
        "",
        f"Reference: FINUFFT eps `{result['reference_finufft_eps']}`; atoms `{result['atoms']:,}`; grid `Nq={result['nq']}`, `Nphi={result['n_phi']}`.",
        "",
        "| dtype | margin | solve s | complex L2 | intensity L2 | ring L2 | gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['complex_dtype']} | {row['margin']} | {row['solve_s']:.3f} | {row['complex_l2']:.3e} | {row['intensity_l2']:.3e} | {row['ring_l2']:.3e} | {'PASS' if row['complex_gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Selected fastest passing configuration: `{selected}`" if selected else "No configuration passed the complex-amplitude gate.",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
