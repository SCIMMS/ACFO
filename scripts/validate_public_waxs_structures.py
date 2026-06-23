from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from scipy.fft import next_fast_len

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import PreparedCakePlan, direct_amplitude, make_cylindrical_histogram  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402


ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Ca": 20,
    "Fe": 26,
    "Zn": 30,
    "Br": 35,
    "I": 53,
}


@dataclass(frozen=True)
class StructureGrid:
    r_max_nm: float
    z_range_nm: tuple[float, float]
    n_r: int
    n_z: int
    n_phi: int
    qmax_inv_nm: float


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


def choose_grid_for_structure(
    coords: np.ndarray,
    *,
    bin_width_nm: float,
    qmax: float,
    q_unit: str,
    nphi_detector: int,
    harmonic_margin: int,
) -> StructureGrid:
    qmax_inv_nm = q_to_inv_nm(qmax, q_unit)
    radius = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    r_max = float(radius.max(initial=0.0)) + bin_width_nm
    z_min = float(coords[:, 2].min(initial=0.0)) - bin_width_nm
    z_max = float(coords[:, 2].max(initial=0.0)) + bin_width_nm
    if z_min >= z_max:
        z_min -= 0.5 * bin_width_nm
        z_max += 0.5 * bin_width_nm
    n_r = max(1, int(np.ceil(r_max / bin_width_nm)))
    n_z = max(1, int(np.ceil((z_max - z_min) / bin_width_nm)))
    n_phi_bandlimit = ceil_even(2.0 * qmax_inv_nm * r_max + 2.0 * harmonic_margin)
    n_phi = next_fft_friendly_even(max(int(nphi_detector), n_phi_bandlimit))
    return StructureGrid(
        r_max_nm=r_max,
        z_range_nm=(z_min, z_max),
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        qmax_inv_nm=qmax_inv_nm,
    )


def median_time(func, repeats: int):
    value = None
    times: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_structure(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=np.float64)
        elements = np.asarray(payload["elements"]).astype(str)
        metadata = json.loads(str(payload["metadata_json"]))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} coords must have shape (n_atoms, 3)")
    if elements.shape != (coords.shape[0],):
        raise ValueError(f"{path} elements must have one entry per atom")
    return coords, elements, metadata


def constant_form_factors(elements: np.ndarray, model: str) -> dict[str, float]:
    unique = sorted(set(str(element) for element in elements))
    if model == "unit":
        return {element: 1.0 for element in unique}
    if model == "atomic_number":
        out = {}
        for element in unique:
            if element not in ATOMIC_NUMBERS:
                raise ValueError(f"atomic-number model does not know element {element!r}")
            out[element] = float(ATOMIC_NUMBERS[element])
        return out
    raise ValueError("form-factor model must be unit or atomic_number")


def debye_weighted_pdist(
    coords: np.ndarray,
    elements: np.ndarray,
    q_solver: np.ndarray,
    form_factors: dict[str, float],
) -> np.ndarray:
    weights = np.asarray([form_factors[str(element)] for element in elements], dtype=np.float64)
    pair_i, pair_j = np.triu_indices(coords.shape[0], k=1)
    distances = np.linalg.norm(coords[pair_i] - coords[pair_j], axis=1)
    pair_weights = weights[pair_i] * weights[pair_j]
    self_term = float(np.sum(weights * weights))
    out = np.empty(q_solver.size, dtype=np.float64)
    for i, q_value in enumerate(q_solver):
        out[i] = self_term + 2.0 * np.sum(pair_weights * np.sinc((q_value * distances) / np.pi))
    return out


def direct_ring_curve(
    coords: np.ndarray,
    elements: np.ndarray,
    q_solver: np.ndarray,
    wavelength_nm: float,
    phi: np.ndarray,
    form_factors: dict[str, float],
) -> np.ndarray:
    amp = direct_amplitude(
        coords,
        q_solver,
        wavelength_nm,
        phi,
        elements=elements,
        form_factors=form_factors,
    )
    return np.mean(intensity(amp), axis=1)


def rotated_direct_ring_curve(
    coords: np.ndarray,
    elements: np.ndarray,
    q_solver: np.ndarray,
    wavelength_nm: float,
    phi: np.ndarray,
    form_factors: dict[str, float],
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    out = np.zeros(q_solver.size, dtype=np.float64)
    for _ in range(samples):
        rotation = random_rotation_matrix(rng)
        rotated = coords @ rotation.T
        out += direct_ring_curve(rotated, elements, q_solver, wavelength_nm, phi, form_factors)
    return out / float(samples)


def histogram_ring_curve(
    coords: np.ndarray,
    elements: np.ndarray,
    q_solver: np.ndarray,
    wavelength_nm: float,
    grid: StructureGrid,
    args: argparse.Namespace,
    form_factors: dict[str, float],
) -> np.ndarray:
    binned = make_cylindrical_histogram(
        coords,
        elements=elements,
        n_r=grid.n_r,
        n_z=grid.n_z,
        n_phi=grid.n_phi,
        r_max=grid.r_max_nm,
        z_range=grid.z_range_nm,
        backend=args.hist_backend,
        hist_dtype=np.dtype(args.hist_dtype),
        angle_lut_size=args.angle_lut_size,
        angle_lut_mode=args.angle_lut_mode,
    )
    plan = PreparedCakePlan(
        binned,
        q_solver,
        wavelength_nm,
        form_factors=form_factors,
        q_block_size=args.q_block_size,
        circular_backend=args.circular_backend,
        complex_dtype=np.dtype(args.complex_dtype),
    )
    if args.curve_backend == "r-dependent":
        return plan.ring_average_intensity_r_dependent_bandlimit(
            margin=args.r_dependent_margin,
            cutoff_bin_size=args.r_dependent_cutoff_bin_size,
        )
    return plan.ring_average_intensity_r_grouped()


def validate_one(path: Path, args: argparse.Namespace, model: str) -> dict:
    coords, elements, metadata = load_structure(path)
    grid = choose_grid_for_structure(
        coords,
        bin_width_nm=args.bin_width_nm,
        qmax=args.qmax,
        q_unit=args.q_unit,
        nphi_detector=args.nphi_detector,
        harmonic_margin=args.harmonic_margin,
    )
    q_report = np.linspace(args.qmin, args.qmax, args.nq)
    q_solver = np.asarray([q_to_inv_nm(q, args.q_unit) for q in q_report], dtype=np.float64)
    phi = (np.arange(grid.n_phi) + 0.5) * (2.0 * np.pi / grid.n_phi)
    form_factors = constant_form_factors(elements, model)

    debye, debye_s, debye_times = median_time(
        lambda: debye_weighted_pdist(coords, elements, q_solver, form_factors),
        args.repeats,
    )
    direct_curve, direct_s, direct_times = median_time(
        lambda: direct_ring_curve(coords, elements, q_solver, args.wavelength_nm, phi, form_factors),
        args.repeats,
    )
    if args.orientation_samples > 0:
        rotated_curve, rotated_s, rotated_times = median_time(
            lambda: rotated_direct_ring_curve(
                coords,
                elements,
                q_solver,
                args.wavelength_nm,
                phi,
                form_factors,
                samples=args.orientation_samples,
                seed=args.seed,
            ),
            args.repeats,
        )
    else:
        rotated_curve = None
        rotated_s = None
        rotated_times = []
    hist_curve, hist_s, hist_times = median_time(
        lambda: histogram_ring_curve(
            coords,
            elements,
            q_solver,
            args.wavelength_nm,
            grid,
            args,
            form_factors,
        ),
        args.repeats,
    )

    element_counts = {
        str(element): int(np.sum(elements == element))
        for element in sorted(set(elements.tolist()))
    }
    return {
        "structure_path": str(path.as_posix()),
        "structure_id": metadata.get("structure_id", path.stem),
        "source_id": metadata.get("source_id"),
        "form_factor_model": model,
        "atoms": int(coords.shape[0]),
        "element_counts": element_counts,
        "qmin": args.qmin,
        "qmax": args.qmax,
        "q_unit": args.q_unit,
        "nq": args.nq,
        "wavelength_nm": args.wavelength_nm,
        "bin_width_nm": args.bin_width_nm,
        "n_r": grid.n_r,
        "n_z": grid.n_z,
        "n_phi": grid.n_phi,
        "r_max_nm": grid.r_max_nm,
        "z_min_nm": grid.z_range_nm[0],
        "z_max_nm": grid.z_range_nm[1],
        "debye_s": debye_s,
        "direct_ring_s": direct_s,
        "rotated_ring_s": rotated_s,
        "histogram_curve_s": hist_s,
        "debye_times": debye_times,
        "direct_ring_times": direct_times,
        "rotated_ring_times": rotated_times,
        "histogram_curve_times": hist_times,
        "direct_ring_rel_l2_vs_debye": relative_l2(direct_curve, debye),
        "rotated_ring_rel_l2_vs_debye": None
        if rotated_curve is None
        else relative_l2(rotated_curve, debye),
        "histogram_rel_l2_vs_direct_ring": relative_l2(hist_curve, direct_curve),
        "histogram_rel_l2_vs_debye": relative_l2(hist_curve, debye),
        "speedup_histogram_vs_debye": debye_s / hist_s,
        "speedup_histogram_vs_direct_ring": direct_s / hist_s,
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
        "# Public WAXS structure validation",
        "",
        "Initial validation using public RCSB structures converted to the repository NPZ contract.",
        "The Debye reference, direct WAXS ring average, and histogram ring-average path use the same constant form-factor model per row.",
        "",
        "| structure | atoms | model | n_phi | Debye s | direct s | histogram s | hist/Debye speedup | direct vs Debye L2 | rotated vs Debye L2 | hist vs direct L2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {structure_id} | {atoms} | {form_factor_model} | {n_phi} | "
            "{debye_s:.4f} | {direct_ring_s:.4f} | {histogram_curve_s:.4f} | "
            "{speedup_histogram_vs_debye:.2f}x | {direct_ring_rel_l2_vs_debye:.3e} | "
            "{rotated} | {histogram_rel_l2_vs_direct_ring:.3e} |".format(
                rotated="-"
                if row["rotated_ring_rel_l2_vs_debye"] is None
                else f"{row['rotated_ring_rel_l2_vs_debye']:.3e}",
                **row,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `histogram_rel_l2_vs_direct_ring` is the primary solver correctness check for the current fixed molecular orientation.",
            "- `direct_ring_rel_l2_vs_debye` is expected to be nonzero for a single anisotropic protein orientation; it measures orientation-average mismatch, not a solver failure.",
            "- If `rotated_ring_rel_l2_vs_debye` is present, it is the direct-ring curve averaged over random molecular orientations.",
            "- The `atomic_number` rows are a lightweight multi-element path check. They are not a final q-dependent atomic form-factor WAXS model.",
            "",
            "Source artifacts:",
            "",
            f"- JSON: `{output.as_posix()}`",
            f"- CSV: `{output.with_suffix('.csv').as_posix()}`",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    print("structure\tmodel\tatoms\tn_phi\tdebye_s\tdirect_s\thist_s\thist_vs_direct_l2")
    for row in rows:
        print(
            f"{row['structure_id']}\t{row['form_factor_model']}\t{row['atoms']}\t"
            f"{row['n_phi']}\t{row['debye_s']:.4f}\t{row['direct_ring_s']:.4f}\t"
            f"{row['histogram_curve_s']:.4f}\t{row['histogram_rel_l2_vs_direct_ring']:.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate public WAXS structure NPZ files against Debye/direct-ring references."
    )
    parser.add_argument("structures", nargs="*", type=Path)
    parser.add_argument("--glob", default="structures/processed/protein_*_heavy_centered.npz")
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
    parser.add_argument("--curve-backend", choices=["r-grouped", "r-dependent"], default="r-dependent")
    parser.add_argument("--r-dependent-margin", type=int, default=16)
    parser.add_argument("--r-dependent-cutoff-bin-size", type=int, default=16)
    parser.add_argument("--form-factor-models", default="unit,atomic_number")
    parser.add_argument("--orientation-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/public_waxs_structure_validation.json"))
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    structure_paths = list(args.structures)
    if not structure_paths:
        structure_paths = sorted(Path().glob(args.glob))
    if not structure_paths:
        raise ValueError("no structures found; run prepare_public_waxs_structures.py first")

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
