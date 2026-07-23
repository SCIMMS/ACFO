from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_giwaxs_prepared_operator import (  # noqa: E402
    _source_stats,
    binned_direct_amplitude,
    build_prepared_giwaxs_miller_geometry,
    direct_atom_amplitude,
    execute_prepared_giwaxs_miller_geometry,
    make_giwaxs_detector,
    median_time,
    summarize_detector,
)
from prepare_public_waxs_cif_structures import (  # noqa: E402
    build_supercell,
    cell_matrix,
    expand_unit_cell,
    parse_cif,
)
from waxs_cake import make_cylindrical_histogram  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402


@dataclass(frozen=True)
class AtomicLayerCase:
    name: str
    coords_nm: np.ndarray
    elements: np.ndarray
    description: str


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.as_posix())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def infer_cod_identity(path: Path) -> tuple[str, str]:
    match = re.search(r"(?P<name>[A-Za-z0-9_-]+)_(?P<cod>\d{7})\.cif$", path.name)
    if match:
        return match.group("name").lower(), match.group("cod")
    return path.stem.lower(), "0000000"


def center_coordinates(coords: np.ndarray) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float64).copy()
    out -= out.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(out)


def load_cif_layer(
    cif_path: Path,
    *,
    supercell_xy: int,
    supercell_z: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cod_name, cod_id = infer_cod_identity(cif_path)
    structure = parse_cif(cif_path, cod_name=cod_name, cod_id=cod_id)
    elements_unit, frac_unit = expand_unit_cell(structure)
    cell = cell_matrix(structure.cell_lengths_angstrom, structure.cell_angles_deg)
    elements, coords_nm = build_supercell(
        elements_unit,
        frac_unit,
        cell,
        (int(supercell_xy), int(supercell_xy), int(supercell_z)),
    )
    metadata = {
        **structure.metadata,
        "cif_path": str(cif_path.as_posix()),
        "supercell": [int(supercell_xy), int(supercell_xy), int(supercell_z)],
        "unit_atoms_after_symmetry": int(elements_unit.size),
    }
    return center_coordinates(coords_nm), elements, metadata


def make_defect_layer(
    coords_nm: np.ndarray,
    elements: np.ndarray,
    *,
    defect_fraction: float,
    inplane_sigma_nm: float,
    z_sigma_nm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= defect_fraction < 1.0:
        raise ValueError("defect_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    keep = rng.random(coords_nm.shape[0]) >= defect_fraction
    if not np.any(keep):
        raise ValueError("defect_fraction removed all atoms")
    coords = np.asarray(coords_nm[keep], dtype=np.float64).copy()
    displacement = np.empty_like(coords)
    displacement[:, :2] = rng.normal(0.0, inplane_sigma_nm, size=(coords.shape[0], 2))
    displacement[:, 2] = rng.normal(0.0, z_sigma_nm, size=coords.shape[0])
    coords += displacement
    return center_coordinates(coords), np.asarray(elements[keep], dtype=elements.dtype)


def make_mismatch_bilayer(
    coords_nm: np.ndarray,
    elements: np.ndarray,
    *,
    twist_deg: float,
    strain_x: float,
    strain_y: float,
    z_gap_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    bottom = np.asarray(coords_nm, dtype=np.float64).copy()
    top = np.asarray(coords_nm, dtype=np.float64).copy()
    theta = math.radians(twist_deg)
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    top[:, :2] *= np.array([strain_x, strain_y], dtype=np.float64)
    top[:, :2] = top[:, :2] @ rot.T

    layer_span = float(np.ptp(coords_nm[:, 2]))
    separation = layer_span + float(z_gap_nm)
    bottom[:, 2] -= 0.5 * separation
    top[:, 2] += 0.5 * separation
    coords = np.vstack([bottom, top])
    elems = np.concatenate([elements, elements])
    return center_coordinates(coords), elems


def make_cases(args: argparse.Namespace) -> tuple[list[AtomicLayerCase], dict[str, Any]]:
    coords, elements, metadata = load_cif_layer(
        args.cif,
        supercell_xy=args.supercell_xy,
        supercell_z=args.supercell_z,
    )
    defect_coords, defect_elements = make_defect_layer(
        coords,
        elements,
        defect_fraction=args.defect_fraction,
        inplane_sigma_nm=args.defect_inplane_sigma_nm,
        z_sigma_nm=args.defect_z_sigma_nm,
        seed=args.seed,
    )
    mismatch_coords, mismatch_elements = make_mismatch_bilayer(
        coords,
        elements,
        twist_deg=args.mismatch_twist_deg,
        strain_x=args.mismatch_strain_x,
        strain_y=args.mismatch_strain_y,
        z_gap_nm=args.mismatch_z_gap_nm,
    )
    cases = [
        AtomicLayerCase(
            name="crystalline_layer",
            coords_nm=coords,
            elements=elements,
            description="CIF-derived high-crystallinity slab layer.",
        ),
        AtomicLayerCase(
            name="defect_displaced_layer",
            coords_nm=defect_coords,
            elements=defect_elements,
            description="Same slab with random vacancies and small positional disorder.",
        ),
        AtomicLayerCase(
            name="mismatch_bilayer",
            coords_nm=mismatch_coords,
            elements=mismatch_elements,
            description="Two CIF-derived slabs with strain, twist, and vertical mismatch.",
        ),
    ]
    return cases, metadata


def grid_for_case(coords_nm: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    radius = np.sqrt(coords_nm[:, 0] ** 2 + coords_nm[:, 1] ** 2)
    r_max = float(radius.max(initial=0.0)) + float(args.r_padding_nm)
    z_min = float(coords_nm[:, 2].min(initial=0.0)) - float(args.z_padding_nm)
    z_max = float(coords_nm[:, 2].max(initial=0.0)) + float(args.z_padding_nm)
    return {
        "r_max_nm": r_max,
        "z_range_nm": [z_min, z_max],
        "n_r": int(args.n_r),
        "n_z": int(args.n_z),
        "n_phi": int(args.n_phi),
    }


def save_case_npz(
    case: AtomicLayerCase,
    *,
    metadata: dict[str, Any],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case.name}.npz"
    case_metadata = {
        **metadata,
        "structure_id": case.name,
        "system_type": "cif_derived_layer",
        "description": case.description,
        "units": "nm",
        "n_atoms": int(case.coords_nm.shape[0]),
        "extent_nm": {
            "min": case.coords_nm.min(axis=0).tolist(),
            "max": case.coords_nm.max(axis=0).tolist(),
            "span": np.ptp(case.coords_nm, axis=0).tolist(),
        },
    }
    np.savez_compressed(
        path,
        coords=case.coords_nm,
        elements=case.elements,
        structure_id=np.asarray(case.name),
        metadata_json=np.asarray(json.dumps(case_metadata, separators=(",", ":"))),
    )
    return path


def run_case(
    case: AtomicLayerCase,
    *,
    case_index: int,
    detector,
    source_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    grid = grid_for_case(case.coords_nm, args)
    case_npz = save_case_npz(
        case,
        metadata=source_metadata,
        out_dir=args.case_npz_dir,
    )

    binned, hist_s, hist_times = median_time(
        lambda: make_cylindrical_histogram(
            case.coords_nm,
            elements=case.elements,
            n_r=grid["n_r"],
            n_z=grid["n_z"],
            n_phi=grid["n_phi"],
            r_max=grid["r_max_nm"],
            z_range=tuple(grid["z_range_nm"]),
            backend=args.hist_backend,
            hist_dtype=np.dtype(args.hist_dtype),
        ),
        args.repeats,
    )
    source_stats = _source_stats(binned)

    geometry, geometry_build_s, geometry_build_times = median_time(
        lambda: build_prepared_giwaxs_miller_geometry(
            binned,
            detector.qx,
            detector.qy,
            detector.qz,
            max_mode=args.max_mode,
            extra_order=args.miller_extra_order,
            enable_mode_pruning=not args.disable_mode_pruning,
            mode_pruning_margin=args.mode_pruning_margin,
            mode_pruning_bin_size=args.mode_pruning_bin_size,
            enable_qz_reduction=not args.disable_qz_reduction,
            precompute_kernel=False,
            complex_dtype=args.miller_complex_dtype,
        ),
        args.repeats,
    )

    sparse_amp, sparse_s, sparse_times = median_time(
        lambda: execute_prepared_giwaxs_miller_geometry(
            binned,
            geometry,
            detector.shape,
            source_backend="sparse",
        ),
        args.repeats,
    )

    dense_amp = None
    dense_s = None
    dense_times: list[float] = []
    if not args.skip_dense_source:
        dense_amp, dense_s, dense_times = median_time(
            lambda: execute_prepared_giwaxs_miller_geometry(
                binned,
                geometry,
                detector.shape,
                source_backend="dense",
            ),
            args.repeats,
        )

    binned_direct, binned_direct_s, binned_direct_times = median_time(
        lambda: binned_direct_amplitude(
            binned,
            detector.qx,
            detector.qy,
            detector.qz,
            target_chunk=args.target_chunk,
        ),
        max(1, min(args.repeats, 3)),
    )

    atom_direct = None
    atom_direct_s = None
    atom_direct_times: list[float] = []
    if case.coords_nm.shape[0] <= args.direct_atom_limit:
        atom_direct, atom_direct_s, atom_direct_times = median_time(
            lambda: direct_atom_amplitude(
                case.coords_nm,
                detector.qx,
                detector.qy,
                detector.qz,
                target_chunk=args.target_chunk,
            ),
            max(1, min(args.repeats, 3)),
        )

    sparse_i = intensity(sparse_amp)
    binned_i = intensity(binned_direct)
    atom_i = None if atom_direct is None else intensity(atom_direct)
    dense_i = None if dense_amp is None else intensity(dense_amp)

    row = {
        "case": case.name,
        "description": case.description,
        "case_npz": str(case_npz.as_posix()),
        "cif_path": str(args.cif.as_posix()),
        "atoms": int(case.coords_nm.shape[0]),
        "elements": sorted({str(e) for e in case.elements.tolist()}),
        "element_counts": {
            str(element): int(np.sum(case.elements == element))
            for element in sorted({str(e) for e in case.elements.tolist()})
        },
        "extent_nm": {
            "span": np.ptp(case.coords_nm, axis=0).tolist(),
            "r_max_atoms_nm": float(
                np.sqrt(case.coords_nm[:, 0] ** 2 + case.coords_nm[:, 1] ** 2).max(
                    initial=0.0
                )
            ),
        },
        **grid,
        "targets": int(detector.qx.size),
        "hist_s": hist_s,
        "miller_geometry_build_s": geometry_build_s,
        "sparse_hot_s": sparse_s,
        "dense_hot_s": dense_s,
        "binned_direct_s": binned_direct_s,
        "atom_direct_s": atom_direct_s,
        "sparse_speedup_vs_dense_source": None
        if dense_s is None or sparse_s == 0.0
        else float(dense_s / sparse_s),
        "sparse_speedup_vs_binned_direct": float(binned_direct_s / sparse_s)
        if sparse_s
        else None,
        "dense_speedup_vs_binned_direct": None
        if dense_s is None or dense_s == 0.0
        else float(binned_direct_s / dense_s),
        "sparse_rel_l2_vs_binned_direct": relative_l2(sparse_amp, binned_direct),
        "sparse_intensity_rel_l2_vs_binned_direct": relative_l2(sparse_i, binned_i),
        "dense_rel_l2_vs_binned_direct": None
        if dense_amp is None
        else relative_l2(dense_amp, binned_direct),
        "dense_intensity_rel_l2_vs_binned_direct": None
        if dense_i is None
        else relative_l2(dense_i, binned_i),
        "sparse_rel_l2_vs_dense_source": None
        if dense_amp is None
        else relative_l2(sparse_amp, dense_amp),
        "sparse_intensity_rel_l2_vs_dense_source": None
        if dense_i is None
        else relative_l2(sparse_i, dense_i),
        "binned_direct_rel_l2_vs_atom_direct": None
        if atom_direct is None
        else relative_l2(binned_direct, atom_direct),
        "binned_direct_intensity_rel_l2_vs_atom_direct": None
        if atom_i is None
        else relative_l2(binned_i, atom_i),
        "active_profile_count": source_stats["active_profile_count"],
        "active_profile_fraction": source_stats["active_profile_fraction"],
        "active_beta_count": source_stats["active_beta_count"],
        "active_beta_fraction": source_stats["active_beta_fraction"],
        "miller_requested_max_mode": int(geometry.requested_max_mode),
        "miller_max_mode": int(geometry.max_mode),
        "miller_qz_group_count": int(geometry.qz_group_count),
        "miller_cutoff_min": int(geometry.cutoff_min),
        "miller_cutoff_mean": float(geometry.cutoff_mean),
        "miller_mode_work_fraction": float(geometry.mode_work_fraction),
        "miller_complex_dtype": str(geometry.complex_dtype),
        "times": {
            "hist": hist_times,
            "miller_geometry_build": geometry_build_times,
            "sparse_hot": sparse_times,
            "dense_hot": dense_times,
            "binned_direct": binned_direct_times,
            "atom_direct": atom_direct_times,
        },
    }
    print(
        "{case}: atoms={atoms} active_profiles={ap:.2f}% active_beta={ab:.3f}% "
        "sparse={sparse:.4f}s dense={dense} binned={binned:.4f}s "
        "sparse_int_l2={l2:.3e}".format(
            case=case.name,
            atoms=case.coords_nm.shape[0],
            ap=100.0 * float(row["active_profile_fraction"]),
            ab=100.0 * float(row["active_beta_fraction"]),
            sparse=sparse_s,
            dense="n/a" if dense_s is None else f"{dense_s:.4f}s",
            binned=binned_direct_s,
            l2=float(row["sparse_intensity_rel_l2_vs_binned_direct"]),
        )
    )
    return row


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_sci(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3e}"


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# GIWAXS Real Atomic Layer Sparse-Contraction Benchmark",
        "",
        "This benchmark starts from a real CIF, expands it into finite slab coordinates,",
        "and evaluates a fixed kinematic GIWAXS detector map with the qz-reduced",
        "Miller recurrence path. The sparse source path contracts only populated",
        "cylindrical source profiles.",
        "",
        "## Source",
        "",
        f"- CIF: `{summary['source_metadata']['cif_path']}`",
        f"- Formula: `{summary['source_metadata'].get('chemical_formula_sum')}`",
        f"- Space group: `{summary['source_metadata'].get('space_group_name')}`",
        f"- Supercell: `{summary['source_metadata']['supercell']}`",
        "",
        "## Detector",
        "",
        "| field | value |",
        "|---|---:|",
    ]
    for key, value in summary["detector"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| case | atoms | grid | active profiles | active beta | sparse hot s | dense hot s | sparse/dense | binned direct s | sparse/direct | sparse intensity L2 | binned-vs-atom intensity L2 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        speed_dense = row.get("sparse_speedup_vs_dense_source")
        speed_direct = row.get("sparse_speedup_vs_binned_direct")
        grid = f"{row['n_r']}x{row['n_z']}x{row['n_phi']}"
        lines.append(
            "| {case} | {atoms} | `{grid}` | {active_profiles} | {active_beta} | "
            "`{sparse}` | `{dense}` | {speed_dense} | `{binned}` | {speed_direct} | "
            "{sparse_l2} | {atom_l2} |".format(
                case=row["case"],
                atoms=row["atoms"],
                grid=grid,
                active_profiles=_fmt_pct(row["active_profile_fraction"]),
                active_beta=_fmt_pct(row["active_beta_fraction"]),
                sparse=_fmt(row["sparse_hot_s"]),
                dense=_fmt(row["dense_hot_s"]),
                speed_dense="n/a" if speed_dense is None else f"{speed_dense:.2f}x",
                binned=_fmt(row["binned_direct_s"]),
                speed_direct="n/a" if speed_direct is None else f"{speed_direct:.2f}x",
                sparse_l2=_fmt_sci(row["sparse_intensity_rel_l2_vs_binned_direct"]),
                atom_l2=_fmt_sci(row["binned_direct_intensity_rel_l2_vs_atom_direct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `sparse intensity L2` compares the sparse Miller contraction against direct summation over populated cylindrical bins.",
            "- `binned-vs-atom intensity L2` is the finite-resolution source-binning error, not the contraction error.",
            "- The mismatch bilayer doubles the atom count and increases occupied source profiles, so it is the stress case for sparse contraction.",
            "- This is still a kinematic GIWAXS q-map benchmark; it does not include DWBA, refraction, footprint, or surface roughness physics.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{summary['config']['out']}`",
            f"- case NPZ directory: `{summary['config']['case_npz_dir']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse GIWAXS contraction on CIF-derived atomic layer coordinates."
    )
    parser.add_argument(
        "--cif",
        type=Path,
        default=ROOT / "structures" / "raw" / "cod" / "silicon_1526655.cif",
    )
    parser.add_argument("--supercell-xy", type=int, default=40)
    parser.add_argument("--supercell-z", type=int, default=1)
    parser.add_argument("--defect-fraction", type=float, default=0.05)
    parser.add_argument("--defect-inplane-sigma-nm", type=float, default=0.008)
    parser.add_argument("--defect-z-sigma-nm", type=float, default=0.015)
    parser.add_argument("--mismatch-twist-deg", type=float, default=5.0)
    parser.add_argument("--mismatch-strain-x", type=float, default=1.025)
    parser.add_argument("--mismatch-strain-y", type=float, default=0.985)
    parser.add_argument("--mismatch-z-gap-nm", type=float, default=0.34)
    parser.add_argument("--n-r", type=int, default=384)
    parser.add_argument("--n-z", type=int, default=256)
    parser.add_argument("--n-phi", type=int, default=512)
    parser.add_argument("--r-padding-nm", type=float, default=0.25)
    parser.add_argument("--z-padding-nm", type=float, default=1.0)
    parser.add_argument("--max-mode", type=int, default=255)
    parser.add_argument("--wavelength-nm", type=float, default=0.15406)
    parser.add_argument("--alpha-i-deg", type=float, default=0.2)
    parser.add_argument("--alpha-f-min-deg", type=float, default=0.3)
    parser.add_argument("--alpha-f-max-deg", type=float, default=18.0)
    parser.add_argument("--n-alpha-f", type=int, default=24)
    parser.add_argument("--two-theta-min-deg", type=float, default=-18.0)
    parser.add_argument("--two-theta-max-deg", type=float, default=18.0)
    parser.add_argument("--n-two-theta", type=int, default=48)
    parser.add_argument("--hist-backend", choices=["numpy", "cpp"], default="cpp")
    parser.add_argument("--hist-dtype", default="float32")
    parser.add_argument("--target-chunk", type=int, default=96)
    parser.add_argument("--miller-extra-order", type=int, default=64)
    parser.add_argument("--mode-pruning-margin", type=int, default=32)
    parser.add_argument("--mode-pruning-bin-size", type=int, default=1)
    parser.add_argument("--disable-mode-pruning", action="store_true")
    parser.add_argument("--disable-qz-reduction", action="store_true")
    parser.add_argument(
        "--miller-complex-dtype",
        choices=["complex64", "complex128"],
        default="complex128",
    )
    parser.add_argument("--skip-dense-source", action="store_true")
    parser.add_argument("--direct-atom-limit", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "giwaxs_real_atomic_layers_sparse.json",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "giwaxs_real_atomic_layers_sparse.md",
    )
    parser.add_argument(
        "--case-npz-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "giwaxs_real_atomic_layers_cases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = make_giwaxs_detector(
        wavelength_nm=args.wavelength_nm,
        alpha_i_deg=args.alpha_i_deg,
        alpha_f_min_deg=args.alpha_f_min_deg,
        alpha_f_max_deg=args.alpha_f_max_deg,
        n_alpha_f=args.n_alpha_f,
        two_theta_min_deg=args.two_theta_min_deg,
        two_theta_max_deg=args.two_theta_max_deg,
        n_two_theta=args.n_two_theta,
    )
    cases, source_metadata = make_cases(args)
    rows = [
        run_case(
            case,
            case_index=i,
            detector=detector,
            source_metadata=source_metadata,
            args=args,
        )
        for i, case in enumerate(cases)
    ]
    summary = {
        "config": _as_jsonable(vars(args)),
        "detector": summarize_detector(detector),
        "source_metadata": _as_jsonable(source_metadata),
        "rows": _as_jsonable(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary(args.summary_md, summary)
    print(args.out)
    print(args.summary_md)


if __name__ == "__main__":
    main()
