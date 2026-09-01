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

from validate_public_waxs_crystals import choose_grid  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import (  # noqa: E402
    PreparedCakePlan,
    cylindrical_flat_indices,
    direct_amplitude,
    encode_elements,
    make_cylindrical_histogram_indexed,
    make_sparse_cylindrical_histogram_from_flat_indices,
)
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate direct sparse protein bins against dense sparse-source and exact atoms."
    )
    parser.add_argument("structure", type=Path)
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=2.2)
    parser.add_argument("--q-unit", choices=["inv_angstrom", "inv_nm"], default="inv_angstrom")
    parser.add_argument("--nq", type=int, default=12)
    parser.add_argument("--wavelength-nm", type=float, default=0.1)
    parser.add_argument("--bin-width-nm", type=float, default=0.04)
    parser.add_argument("--nphi-min", type=int, default=1024)
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument("--r-dependent-margin", type=int, default=16)
    parser.add_argument("--cutoff-bin-size", type=int, default=16)
    parser.add_argument("--q-block-size", type=int, default=2)
    parser.add_argument("--profile-chunk-size", type=int, default=8)
    parser.add_argument("--form-factor-model", default="xray_f0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/protein_nanocrystal_sparse_accuracy.json"),
    )
    args = parser.parse_args()

    coords, elements, metadata = load_structure(args.structure)
    grid_args = argparse.Namespace(
        qmax=args.qmax,
        q_unit=args.q_unit,
        bin_width_nm=args.bin_width_nm,
        harmonic_margin=args.harmonic_margin,
        nphi_detector=args.nphi_min,
    )
    grid = choose_grid(coords, grid_args)
    q_report = np.linspace(args.qmin, args.qmax, args.nq)
    q_solver = np.asarray(
        [q_to_inv_nm(value, args.q_unit) for value in q_report],
        dtype=np.float64,
    )
    form_factors = build_form_factors(elements, q_solver, args.form_factor_model)
    element_indices, element_order = encode_elements(elements)
    flat = cylindrical_flat_indices(
        coords,
        element_indices=element_indices,
        n_elements=len(element_order),
        backend="cpp",
        n_r=grid["n_r"],
        n_z=grid["n_z"],
        n_phi=grid["n_phi"],
        r_max=grid["r_max_nm"],
        z_range=grid["z_range_nm"],
    )
    dense = make_cylindrical_histogram_indexed(
        coords,
        element_indices,
        n_elements=len(element_order),
        element_order=element_order,
        hist_dtype=np.float32,
        backend="cpp",
        n_r=grid["n_r"],
        n_z=grid["n_z"],
        n_phi=grid["n_phi"],
        r_max=grid["r_max_nm"],
        z_range=grid["z_range_nm"],
    )
    sparse = make_sparse_cylindrical_histogram_from_flat_indices(
        flat,
        n_elements=len(element_order),
        n_r=grid["n_r"],
        n_z=grid["n_z"],
        n_phi=grid["n_phi"],
        r_max=grid["r_max_nm"],
        z_range=grid["z_range_nm"],
        element_order=element_order,
        value_dtype=np.float32,
    )

    dense_plan = PreparedCakePlan(
        dense,
        q_solver,
        args.wavelength_nm,
        form_factors=form_factors,
        circular_backend="cpp",
        complex_dtype=np.complex64,
        q_block_size=args.q_block_size,
    )
    sparse_plan = PreparedCakePlan(
        sparse,
        q_solver,
        args.wavelength_nm,
        form_factors=form_factors,
        circular_backend="cpp",
        complex_dtype=np.complex64,
        q_block_size=args.q_block_size,
    )

    solve_options = dict(
        margin=args.r_dependent_margin,
        cutoff_bin_size=args.cutoff_bin_size,
        analytic_kernel=True,
        q_block_size=args.q_block_size,
        profile_chunk_size=args.profile_chunk_size,
    )
    dense_amp, dense_s = timed(
        lambda: dense_plan.circular_fft_sparse_source_r_dependent(**solve_options)
    )
    sparse_amp, sparse_s = timed(
        lambda: sparse_plan.circular_fft_sparse_source_r_dependent(**solve_options)
    )
    direct_amp, direct_s = timed(
        lambda: direct_amplitude(
            coords,
            q_solver,
            args.wavelength_nm,
            dense.beta_centers,
            elements=elements,
            form_factors=form_factors,
        )
    )

    dense_i = intensity(dense_amp)
    sparse_i = intensity(sparse_amp)
    direct_i = intensity(direct_amp)
    row = {
        "structure_path": args.structure.as_posix(),
        "structure_id": metadata.get("structure_id", args.structure.stem),
        "atoms": int(coords.shape[0]),
        "qmin": args.qmin,
        "qmax": args.qmax,
        "q_unit": args.q_unit,
        "nq": args.nq,
        "bin_width_nm": args.bin_width_nm,
        "n_r": grid["n_r"],
        "n_z": grid["n_z"],
        "n_phi": grid["n_phi"],
        "active_flat_bins": int(sparse.active_values.size),
        "sparse_structure_storage_mib": sparse.sparse_storage_nbytes / 1024**2,
        "dense_sparse_source_s": dense_s,
        "direct_sparse_source_s": sparse_s,
        "exact_atom_direct_s": direct_s,
        "sparse_vs_dense_complex_l2": relative_l2(sparse_amp, dense_amp),
        "sparse_vs_dense_intensity_l2": relative_l2(sparse_i, dense_i),
        "sparse_vs_dense_ring_l2": relative_l2(
            np.mean(sparse_i, axis=1),
            np.mean(dense_i, axis=1),
        ),
        "sparse_vs_exact_atom_complex_l2": relative_l2(sparse_amp, direct_amp),
        "sparse_vs_exact_atom_intensity_l2": relative_l2(sparse_i, direct_i),
        "sparse_vs_exact_atom_ring_l2": relative_l2(
            np.mean(sparse_i, axis=1),
            np.mean(direct_i, axis=1),
        ),
        "all_finite": bool(np.all(np.isfinite(sparse_amp))),
        "error_contract": {
            "sparse_vs_dense": "representation/operator identity on the same bins",
            "sparse_vs_exact_atom": "end-to-end cylindrical discretization plus operator error",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    md = [
        f"# Sparse protein-nanocrystal accuracy: {row['structure_id']}",
        "",
        f"- sparse vs dense complex L2: `{row['sparse_vs_dense_complex_l2']:.3e}`",
        f"- sparse vs dense intensity L2: `{row['sparse_vs_dense_intensity_l2']:.3e}`",
        f"- sparse vs exact atoms complex L2: `{row['sparse_vs_exact_atom_complex_l2']:.3e}`",
        f"- sparse vs exact atoms intensity L2: `{row['sparse_vs_exact_atom_intensity_l2']:.3e}`",
        f"- sparse vs exact atoms ring L2: `{row['sparse_vs_exact_atom_ring_l2']:.3e}`",
        "",
        "Sparse-vs-dense isolates the new representation. Sparse-vs-exact-atoms includes cylindrical binning/discretization error.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
