from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
from scipy.fft import next_fast_len

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from validate_public_waxs_structures import (  # noqa: E402
    ATOMIC_NUMBERS,
    build_form_factors,
    load_structure,
)
from waxs_cake import PreparedCakePlan, direct_amplitude, make_cylindrical_histogram  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402


def ceil_even(value: float) -> int:
    out = int(np.ceil(value))
    return out if out % 2 == 0 else out + 1


def next_fft_friendly_even(value: int) -> int:
    target = max(2, int(value))
    if target % 2:
        target += 1
    while True:
        candidate = int(next_fast_len(target, real=True))
        if candidate % 2 == 0:
            return candidate
        target = candidate + 1


def median_time(func, repeats: int):
    value = None
    times: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def choose_grid(coords: np.ndarray, args: argparse.Namespace) -> dict:
    qmax_inv_nm = q_to_inv_nm(args.qmax, args.q_unit)
    radius = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    r_max = float(radius.max(initial=0.0)) + args.bin_width_nm
    z_min = float(coords[:, 2].min(initial=0.0)) - args.bin_width_nm
    z_max = float(coords[:, 2].max(initial=0.0)) + args.bin_width_nm
    n_r = max(1, int(np.ceil(r_max / args.bin_width_nm)))
    n_z = max(1, int(np.ceil((z_max - z_min) / args.bin_width_nm)))
    n_phi_bandlimit = ceil_even(2.0 * qmax_inv_nm * r_max + 2.0 * args.harmonic_margin)
    n_phi = next_fft_friendly_even(max(args.nphi_detector, n_phi_bandlimit))
    return {
        "r_max_nm": r_max,
        "z_range_nm": (z_min, z_max),
        "n_r": n_r,
        "n_z": n_z,
        "n_phi": n_phi,
    }


def anisotropy_metric(values: np.ndarray) -> float:
    mean_phi = np.mean(values, axis=1)
    std_phi = np.std(values, axis=1)
    mask = mean_phi > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(std_phi[mask] / mean_phi[mask]))


def validate_one(path: Path, args: argparse.Namespace, model: str) -> dict:
    coords, elements, metadata = load_structure(path)
    if model == "atomic_number":
        missing = [
            element for element in sorted(set(elements.tolist())) if element not in ATOMIC_NUMBERS
        ]
        if missing:
            raise ValueError(f"missing atomic-number entries for {missing}")

    grid = choose_grid(coords, args)
    q_report = np.linspace(args.qmin, args.qmax, args.nq)
    q_solver = np.asarray([q_to_inv_nm(q, args.q_unit) for q in q_report], dtype=np.float64)
    phi = (np.arange(grid["n_phi"]) + 0.5) * (2.0 * np.pi / grid["n_phi"])
    form_factors = build_form_factors(elements, q_solver, model)

    direct_amp, direct_s, direct_times = median_time(
        lambda: direct_amplitude(
            coords,
            q_solver,
            args.wavelength_nm,
            phi,
            elements=elements,
            form_factors=form_factors,
        ),
        args.repeats,
    )

    def build_cake():
        binned = make_cylindrical_histogram(
            coords,
            elements=elements,
            n_r=grid["n_r"],
            n_z=grid["n_z"],
            n_phi=grid["n_phi"],
            r_max=grid["r_max_nm"],
            z_range=grid["z_range_nm"],
            backend=args.hist_backend,
            hist_dtype=np.dtype(args.hist_dtype),
            angle_lut_size=args.angle_lut_size,
            angle_lut_mode=args.angle_lut_mode,
        )
        plan = PreparedCakePlan(
            binned,
            q_solver,
            args.wavelength_nm,
            form_factors=form_factors,
            q_block_size=args.q_block_size,
            circular_backend=args.circular_backend,
            complex_dtype=np.dtype(args.complex_dtype),
        )
        if args.cake_backend == "circular":
            return plan.circular_fft()
        ahat = plan.circular_ahat_r_dependent_bandlimit(
            margin=args.r_dependent_margin,
            cutoff_bin_size=args.r_dependent_cutoff_bin_size,
            analytic_kernel=args.r_dependent_analytic_kernel,
            fused_analytic_kernel=args.r_dependent_fused_kernel,
            q_block_size=args.q_block_size,
        )
        return np.fft.ifft(ahat, axis=-1)

    cake_amp, cake_s, cake_times = median_time(build_cake, args.repeats)
    direct_i = intensity(direct_amp)
    cake_i = intensity(cake_amp)
    direct_ring = np.mean(direct_i, axis=1)
    cake_ring = np.mean(cake_i, axis=1)
    element_counts = {
        str(element): int(np.sum(elements == element))
        for element in sorted(set(elements.tolist()))
    }
    return {
        "structure_path": str(path.as_posix()),
        "structure_id": metadata.get("structure_id", path.stem),
        "source_id": metadata.get("source_id"),
        "source_database": metadata.get("source_database"),
        "formula": metadata.get("chemical_formula_sum"),
        "space_group_name": metadata.get("space_group_name"),
        "space_group_number": metadata.get("space_group_number"),
        "form_factor_model": model,
        "cake_backend": args.cake_backend,
        "atoms": int(coords.shape[0]),
        "element_counts": element_counts,
        "qmin": args.qmin,
        "qmax": args.qmax,
        "q_unit": args.q_unit,
        "nq": args.nq,
        "n_phi": int(grid["n_phi"]),
        "n_r": int(grid["n_r"]),
        "n_z": int(grid["n_z"]),
        "direct_cake_s": direct_s,
        "histogram_cake_s": cake_s,
        "direct_cake_times": direct_times,
        "histogram_cake_times": cake_times,
        "cake_intensity_rel_l2_vs_direct": relative_l2(cake_i, direct_i),
        "ring_rel_l2_vs_direct": relative_l2(cake_ring, direct_ring),
        "direct_anisotropy": anisotropy_metric(direct_i),
        "histogram_anisotropy": anisotropy_metric(cake_i),
        "speedup_histogram_vs_direct": direct_s / cake_s,
    }


def write_outputs(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    scalar_rows = [
        {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
        for row in rows
    ]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def write_summary(rows: list[dict], output: Path) -> None:
    lines = [
        "# Public WAXS crystal CIF validation",
        "",
        "Initial 2D cake-map validation using public COD CIF structures converted to finite supercell NPZ inputs.",
        "The comparison is direct 2D WAXS cake intensity versus the cylindrical-histogram cake path on the same fixed crystal orientation.",
        "",
        "| structure | atoms | model | n_phi | direct cake s | histogram cake s | speedup | 2D intensity L2 | ring L2 | direct anisotropy | histogram anisotropy |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {structure_id} | {atoms} | {form_factor_model} | {n_phi} | "
            "{direct_cake_s:.4f} | {histogram_cake_s:.4f} | "
            "{speedup_histogram_vs_direct:.2f}x | {cake_intensity_rel_l2_vs_direct:.3e} | "
            "{ring_rel_l2_vs_direct:.3e} | {direct_anisotropy:.3f} | "
            "{histogram_anisotropy:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `cake_intensity_rel_l2_vs_direct` is the primary 2D fixed-orientation crystal solver check.",
            "- `ring_rel_l2_vs_direct` verifies the 1D reduction after azimuthal averaging.",
            "- The anisotropy columns are not an error metric; they confirm that these CIF supercells exercise non-isotropic cake-map structure.",
            "- The `atomic_number` rows are a lightweight multi-element path check.",
            "- The `xray_f0` rows use q-dependent neutral-atom elastic X-ray f0 values from `periodictable`; anomalous dispersion, ionic state, solvent, and Debye-Waller effects are intentionally outside this first validation.",
            "",
            "Source artifacts:",
            "",
            f"- JSON: `{output.as_posix()}`",
            f"- CSV: `{output.with_suffix('.csv').as_posix()}`",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    print("structure\tmodel\tatoms\tn_phi\tdirect_s\thist_s\tcake_l2\tring_l2\tanisotropy")
    for row in rows:
        print(
            f"{row['structure_id']}\t{row['form_factor_model']}\t{row['atoms']}\t"
            f"{row['n_phi']}\t{row['direct_cake_s']:.4f}\t{row['histogram_cake_s']:.4f}\t"
            f"{row['cake_intensity_rel_l2_vs_direct']:.3e}\t{row['ring_rel_l2_vs_direct']:.3e}\t"
            f"{row['direct_anisotropy']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate public crystal CIF supercell NPZ files against direct 2D cake references."
    )
    parser.add_argument("structures", nargs="*", type=Path)
    parser.add_argument("--glob", default="structures/processed/crystal_*_cod*_*.npz")
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=2.2)
    parser.add_argument("--nq", type=int, default=24)
    parser.add_argument("--q-unit", choices=["inv_angstrom", "inv_nm"], default="inv_angstrom")
    parser.add_argument("--wavelength-nm", type=float, default=0.1)
    parser.add_argument("--bin-width-nm", type=float, default=0.08)
    parser.add_argument("--nphi-detector", type=int, default=180)
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument("--hist-backend", choices=["numpy", "cpp"], default="cpp")
    parser.add_argument("--hist-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--angle-lut-size", type=int, default=32)
    parser.add_argument("--angle-lut-mode", choices=["nearest", "cubic"], default="cubic")
    parser.add_argument("--circular-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--complex-dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--cake-backend", choices=["circular", "r-dependent"], default="circular")
    parser.add_argument("--r-dependent-margin", type=int, default=16)
    parser.add_argument("--r-dependent-cutoff-bin-size", type=int, default=16)
    parser.add_argument("--r-dependent-analytic-kernel", action="store_true")
    parser.add_argument("--r-dependent-fused-kernel", action="store_true")
    parser.add_argument("--form-factor-models", default="unit,atomic_number,xray_f0")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/public_waxs_crystal_validation.json"))
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    structure_paths = list(args.structures)
    if not structure_paths:
        structure_paths = sorted(Path().glob(args.glob))
    if not structure_paths:
        raise ValueError("no crystal structures found; run prepare_public_waxs_cif_structures.py first")
    models = [part.strip() for part in args.form_factor_models.split(",") if part.strip()]
    rows = []
    for path in structure_paths:
        for model in models:
            rows.append(validate_one(path, args, model))
    print_table(rows)
    write_outputs(rows, args.output)
    write_summary(rows, args.output)
    print(f"wrote {args.output}, {args.output.with_suffix('.csv')}, and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
