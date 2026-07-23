from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.validate_public_waxs_crystals import choose_grid
except ModuleNotFoundError:  # Direct script execution places scripts/ on sys.path.
    from validate_public_waxs_crystals import choose_grid

from waxs_cake import cylindrical_flat_indices


def parse_widths(value: str) -> list[float]:
    widths = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("bin-widths must contain positive comma-separated values")
    return widths


def estimate(path: Path, args: argparse.Namespace, bin_width_nm: float) -> dict:
    with np.load(path, allow_pickle=False) as data:
        coords = np.asarray(data["coords"], dtype=np.float64)
        elements = np.asarray(data["elements"])
        metadata = json.loads(str(data["metadata_json"]))
    grid_args = argparse.Namespace(
        qmax=args.qmax,
        q_unit=args.q_unit,
        bin_width_nm=bin_width_nm,
        harmonic_margin=args.harmonic_margin,
        nphi_detector=args.nphi_min,
    )
    grid = choose_grid(coords, grid_args)
    element_order, element_indices = np.unique(elements, return_inverse=True)
    n_elements = int(element_order.size)
    dense_values = n_elements * grid["n_r"] * grid["n_z"] * grid["n_phi"]
    flat_indices = cylindrical_flat_indices(
        coords,
        element_indices=element_indices,
        n_elements=n_elements,
        backend="numpy",
        n_r=grid["n_r"],
        n_z=grid["n_z"],
        n_phi=grid["n_phi"],
        r_max=grid["r_max_nm"],
        z_range=grid["z_range_nm"],
    )
    active_flat_bins = int(np.unique(flat_indices).size)
    active_rz_profiles = int(np.unique(flat_indices // grid["n_phi"]).size)
    total_rz_profiles = n_elements * grid["n_r"] * grid["n_z"]
    sparse_value_key_lower_bound_bytes = active_flat_bins * (4 + 8)
    return {
        "structure_path": path.as_posix(),
        "structure_id": metadata.get("structure_id", path.stem),
        "atoms": int(coords.shape[0]),
        "elements": n_elements,
        "qmax": args.qmax,
        "q_unit": args.q_unit,
        "bin_width_nm": bin_width_nm,
        "n_r": int(grid["n_r"]),
        "n_z": int(grid["n_z"]),
        "n_phi": int(grid["n_phi"]),
        "r_max_nm": float(grid["r_max_nm"]),
        "z_span_nm": float(grid["z_range_nm"][1] - grid["z_range_nm"][0]),
        "dense_hist_float32_gib": dense_values * 4 / 1024**3,
        "dense_hist_float64_gib": dense_values * 8 / 1024**3,
        "dense_float32_within_24_gib": dense_values * 4 <= 24 * 1024**3,
        "active_flat_bins": active_flat_bins,
        "total_flat_bins": dense_values,
        "active_flat_fraction": active_flat_bins / dense_values,
        "active_rz_profiles": active_rz_profiles,
        "total_rz_profiles": total_rz_profiles,
        "active_rz_fraction": active_rz_profiles / total_rz_profiles,
        "mean_atoms_per_active_flat_bin": coords.shape[0] / active_flat_bins,
        "sparse_float32_value_int64_key_lower_bound_mib": (
            sparse_value_key_lower_bound_bytes / 1024**2
        ),
    }


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# ACFO NCS W3a protein-nanocrystal grid estimate",
        "",
        "Dense histogram storage is a lower-level array estimate only; plan, FFT, temporary, and output memory are additional.",
        "Sparse lower bound counts one float32 value plus one int64 flat key per active bin only; offsets, metadata, sorting workspace, and solver state are additional.",
        "",
        "| structure | atoms | bin nm | grid (Nr x Nz x Nphi) | active flat | active % | active RZ % | sparse lower MiB | dense f32 GiB | f32 <=24 GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {structure_id} | {atoms} | {bin_width_nm:.2f} | {n_r}x{n_z}x{n_phi} | "
            "{active_flat_bins} | {active_flat_fraction:.4%} | {active_rz_fraction:.2%} | "
            "{sparse_float32_value_int64_key_lower_bound_mib:.2f} | "
            "{dense_hist_float32_gib:.2f} | {dense_float32_within_24_gib} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `Nphi` is selected from the physical azimuthal bandlimit, not capped at 2048.",
            "- The 24 GiB gate requires total process/GPU memory, not only this dense histogram estimate.",
            "- Sparse, R-dependent, fused, or streamed paths must be measured when the dense estimate leaves insufficient headroom.",
            "- The sparse lower bound is not a measured process RSS and must not be compared directly with a hot-solve RSS delta.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ACFO dense grid storage for protein-nanocrystal NPZ inputs.")
    parser.add_argument("structures", nargs="+", type=Path)
    parser.add_argument("--qmax", type=float, default=6.3)
    parser.add_argument("--q-unit", choices=["inv_angstrom", "inv_nm"], default="inv_angstrom")
    parser.add_argument("--bin-widths", default="0.10,0.08,0.04")
    parser.add_argument("--nphi-min", type=int, default=1024)
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/acfo_ncs_w3a_grid_resource_estimate.json"))
    args = parser.parse_args()

    rows = [
        estimate(path, args, width)
        for path in args.structures
        for width in parse_widths(args.bin_widths)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_markdown(rows, args.output.with_suffix(".md"))
    print(json.dumps(rows, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
