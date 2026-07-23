from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import urllib.request

import numpy as np

try:
    from scripts.prepare_public_waxs_cif_structures import cell_matrix
    from scripts.prepare_public_waxs_structures import ATOMIC_NUMBERS, WATER_RESNAMES, normalize_element
except ModuleNotFoundError:  # Direct script execution places scripts/ on sys.path.
    from prepare_public_waxs_cif_structures import cell_matrix
    from prepare_public_waxs_structures import ATOMIC_NUMBERS, WATER_RESNAMES, normalize_element


@dataclass(frozen=True)
class PdbCrystal:
    pdb_id: str
    raw_path: Path
    cell_lengths_angstrom: tuple[float, float, float]
    cell_angles_deg: tuple[float, float, float]
    space_group: str
    asym_coords_angstrom: np.ndarray
    asym_elements: np.ndarray
    asym_records: tuple[str, ...]
    asym_resnames: tuple[str, ...]
    symmetry_rotations: tuple[np.ndarray, ...]
    symmetry_translations_angstrom: tuple[np.ndarray, ...]


def parse_supercell(value: str) -> tuple[int, int, int]:
    normalized = value.lower().replace("x", ",")
    parts = [int(part.strip()) for part in normalized.split(",") if part.strip()]
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("supercell must be N or Nx,Ny,Nz with positive integers")
    return tuple(parts)  # type: ignore[return-value]


def parse_triplet(value: str, *, name: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain three comma-separated values")
    return tuple(parts)  # type: ignore[return-value]


def download_pdb(pdb_id: str, raw_dir: Path, *, refresh: bool) -> Path:
    pdb_id = pdb_id.upper()
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / f"{pdb_id}.pdb"
    if output.exists() and not refresh:
        return output
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    output.write_bytes(payload)
    return output


def parse_cryst1(line: str) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    try:
        lengths = (float(line[6:15]), float(line[15:24]), float(line[24:33]))
        angles = (float(line[33:40]), float(line[40:47]), float(line[47:54]))
    except ValueError as exc:
        raise ValueError(f"invalid CRYST1 record: {line!r}") from exc
    space_group = line[55:66].strip()
    return lengths, angles, space_group


SMTRY_RE = re.compile(
    r"^REMARK\s+290\s+SMTRY([123])\s+(\d+)\s+"
    r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
)


def parse_pdb_crystal(
    path: Path,
    *,
    pdb_id: str,
    include_hetatm: bool,
    include_waters: bool,
    include_hydrogen: bool,
) -> PdbCrystal:
    cell_lengths: tuple[float, float, float] | None = None
    cell_angles: tuple[float, float, float] | None = None
    space_group = ""
    coords: list[tuple[float, float, float]] = []
    elements: list[str] = []
    records: list[str] = []
    resnames: list[str] = []
    sym_rows: dict[int, dict[int, tuple[np.ndarray, float]]] = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CRYST1"):
            cell_lengths, cell_angles, space_group = parse_cryst1(line)
            continue

        sym_match = SMTRY_RE.match(line)
        if sym_match:
            row = int(sym_match.group(1)) - 1
            operation = int(sym_match.group(2))
            rotation_row = np.asarray([float(sym_match.group(i)) for i in (3, 4, 5)], dtype=np.float64)
            translation = float(sym_match.group(6))
            sym_rows.setdefault(operation, {})[row] = (rotation_row, translation)
            continue

        record = line[:6].strip()
        if record == "ATOM":
            pass
        elif record == "HETATM" and include_hetatm:
            pass
        else:
            continue

        alt_loc = line[16:17].strip()
        if alt_loc and alt_loc != "A":
            continue
        resname = line[17:20].strip().upper()
        if resname in WATER_RESNAMES and not include_waters:
            continue
        atom_name = line[12:16].strip()
        element = normalize_element(line[76:78] if len(line) >= 78 else "", atom_name)
        if element == "H" and not include_hydrogen:
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError as exc:
            raise ValueError(f"invalid atom coordinates: {line!r}") from exc
        coords.append(xyz)
        elements.append(element)
        records.append(record)
        resnames.append(resname)

    if cell_lengths is None or cell_angles is None:
        raise ValueError(f"{path} has no CRYST1 record")
    if not coords:
        raise ValueError(f"{path} has no selected atoms")

    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for operation in sorted(sym_rows):
        rows = sym_rows[operation]
        if set(rows) != {0, 1, 2}:
            raise ValueError(f"incomplete SMTRY operation {operation} in {path}")
        rotations.append(np.vstack([rows[index][0] for index in range(3)]))
        translations.append(np.asarray([rows[index][1] for index in range(3)], dtype=np.float64))
    if not rotations:
        rotations = [np.eye(3, dtype=np.float64)]
        translations = [np.zeros(3, dtype=np.float64)]

    return PdbCrystal(
        pdb_id=pdb_id.upper(),
        raw_path=path,
        cell_lengths_angstrom=cell_lengths,
        cell_angles_deg=cell_angles,
        space_group=space_group,
        asym_coords_angstrom=np.asarray(coords, dtype=np.float64),
        asym_elements=np.asarray(elements, dtype="U2"),
        asym_records=tuple(records),
        asym_resnames=tuple(resnames),
        symmetry_rotations=tuple(rotations),
        symmetry_translations_angstrom=tuple(translations),
    )


def expand_unit_cell(
    crystal: PdbCrystal,
    *,
    tolerance_fractional: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    cell = cell_matrix(crystal.cell_lengths_angstrom, crystal.cell_angles_deg)
    inverse_cell = np.linalg.inv(cell)
    scale = 1.0 / tolerance_fractional
    entries: dict[tuple[str, int, int, int], tuple[str, np.ndarray]] = {}
    for rotation, translation in zip(
        crystal.symmetry_rotations,
        crystal.symmetry_translations_angstrom,
        strict=True,
    ):
        transformed = crystal.asym_coords_angstrom @ rotation.T + translation
        fractional = transformed @ inverse_cell
        fractional -= np.floor(fractional + tolerance_fractional)
        fractional[np.isclose(fractional, 1.0, atol=tolerance_fractional)] = 0.0
        for element, frac in zip(crystal.asym_elements, fractional, strict=True):
            key = (str(element), *(int(round(float(value) * scale)) for value in frac))
            entries[key] = (str(element), frac)
    elements = np.asarray([value[0] for value in entries.values()], dtype="U2")
    fractional = np.asarray([value[1] for value in entries.values()], dtype=np.float64)
    order = np.lexsort((fractional[:, 2], fractional[:, 1], fractional[:, 0], elements.astype(str)))
    return elements[order], fractional[order]


def build_supercell(
    elements_unit: np.ndarray,
    fractional_unit: np.ndarray,
    cell_angstrom: np.ndarray,
    supercell: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    translations = np.asarray(
        [(ix, iy, iz) for ix in range(supercell[0]) for iy in range(supercell[1]) for iz in range(supercell[2])],
        dtype=np.float64,
    )
    fractional = (fractional_unit[None, :, :] + translations[:, None, :]).reshape(-1, 3)
    elements = np.tile(elements_unit, translations.shape[0])
    coords_angstrom = fractional @ cell_angstrom
    return elements, coords_angstrom


def euler_rotation_zyx(euler_deg: tuple[float, float, float]) -> np.ndarray:
    z_deg, y_deg, x_deg = euler_deg
    z, y, x = [math.radians(value) for value in (z_deg, y_deg, x_deg)]
    rz = np.asarray([[math.cos(z), -math.sin(z), 0.0], [math.sin(z), math.cos(z), 0.0], [0.0, 0.0, 1.0]])
    ry = np.asarray([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]])
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, math.cos(x), -math.sin(x)], [0.0, math.sin(x), math.cos(x)]])
    return rz @ ry @ rx


def apply_crop(
    elements: np.ndarray,
    coords_angstrom: np.ndarray,
    *,
    crop: str,
    crop_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    center = 0.5 * (coords_angstrom.min(axis=0) + coords_angstrom.max(axis=0))
    centered = coords_angstrom - center
    spans = np.ptp(coords_angstrom, axis=0)
    if crop == "none":
        mask = np.ones(coords_angstrom.shape[0], dtype=bool)
        radii = None
    elif crop == "sphere":
        radius = 0.5 * float(np.min(spans)) * crop_fraction
        mask = np.linalg.norm(centered, axis=1) <= radius
        radii = [radius, radius, radius]
    elif crop == "ellipsoid":
        radii_array = 0.5 * spans * crop_fraction
        mask = np.sum((centered / radii_array) ** 2, axis=1) <= 1.0
        radii = radii_array.tolist()
    else:
        raise ValueError(f"unsupported crop {crop!r}")
    if not np.any(mask):
        raise ValueError("crop removed every atom")
    selected = centered[mask]
    selected -= selected.mean(axis=0, keepdims=True)
    return elements[mask], selected, {
        "crop": crop,
        "crop_fraction": crop_fraction,
        "radii_angstrom": radii,
        "atoms_before_crop": int(coords_angstrom.shape[0]),
        "atoms_after_crop": int(np.count_nonzero(mask)),
    }


def build_output_name(
    pdb_id: str,
    supercell: tuple[int, int, int],
    crop: str,
    disorder_rms_angstrom: float,
) -> str:
    rms_label = f"{disorder_rms_angstrom:.2f}".replace(".", "p")
    return (
        f"protein_nanocrystal_{pdb_id.lower()}_"
        f"{supercell[0]}x{supercell[1]}x{supercell[2]}_{crop}_rms{rms_label}A"
    )


def save_npz(
    *,
    output: Path,
    crystal: PdbCrystal,
    elements_unit: np.ndarray,
    fractional_unit: np.ndarray,
    elements: np.ndarray,
    coords_angstrom: np.ndarray,
    supercell: tuple[int, int, int],
    crop_metadata: dict,
    euler_deg: tuple[float, float, float],
    disorder_rms_angstrom: float,
    seed: int,
    include_hetatm: bool,
    include_waters: bool,
    include_hydrogen: bool,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_numbers = np.asarray([ATOMIC_NUMBERS[str(element).upper()] for element in elements], dtype=np.int16)
    structure_id = output.stem
    metadata = {
        "schema": "protein-nanocrystal-waxs-v1",
        "structure_id": structure_id,
        "system_type": "finite_single_domain_protein_nanocrystal",
        "source_database": "RCSB PDB",
        "source_id": crystal.pdb_id,
        "source_url": f"https://www.rcsb.org/structure/{crystal.pdb_id}",
        "coordinate_url": f"https://files.rcsb.org/download/{crystal.pdb_id}.pdb",
        "raw_file": crystal.raw_path.as_posix(),
        "space_group": crystal.space_group,
        "cell_lengths_angstrom": list(crystal.cell_lengths_angstrom),
        "cell_angles_deg": list(crystal.cell_angles_deg),
        "n_symmetry_operations": len(crystal.symmetry_rotations),
        "n_asymmetric_atoms": int(crystal.asym_coords_angstrom.shape[0]),
        "n_unit_cell_atoms": int(elements_unit.size),
        "supercell": list(supercell),
        **crop_metadata,
        "euler_zyx_deg": list(euler_deg),
        "disorder_model": "independent isotropic Gaussian positional displacement",
        "disorder_rms_angstrom": disorder_rms_angstrom,
        "random_seed": seed,
        "filters": {
            "include_hetatm": include_hetatm,
            "include_waters": include_waters,
            "include_hydrogen": include_hydrogen,
            "altloc": "blank or A",
        },
        "units": "nm",
        "periodic": False,
        "single_crystallographic_domain": True,
        "n_atoms": int(coords_angstrom.shape[0]),
        "element_counts": {
            str(element): int(np.count_nonzero(elements == element))
            for element in sorted(set(elements.tolist()))
        },
        "extent_nm": {
            "span": (np.ptp(coords_angstrom, axis=0) * 0.1).round(8).tolist(),
        },
        "notes": (
            "Finite, fixed-orientation protein nanocrystal for ACFO NCS W3. "
            "This is a single crystallographic domain, not a single isolated protein molecule."
        ),
    }
    metadata_json = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    np.savez_compressed(
        output,
        coords=coords_angstrom * 0.1,
        elements=elements,
        atomic_numbers=atomic_numbers,
        cell_matrix_angstrom=cell_matrix(crystal.cell_lengths_angstrom, crystal.cell_angles_deg),
        unit_cell_fractional=fractional_unit,
        unit_cell_elements=elements_unit,
        structure_id=np.asarray(structure_id),
        metadata_json=np.asarray(metadata_json),
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a finite single-domain protein nanocrystal NPZ from PDB CRYST1/SMTRY records."
    )
    parser.add_argument("--pdb-id", default="1IEE")
    parser.add_argument("--supercell", default="3,3,3")
    parser.add_argument("--raw-dir", type=Path, default=Path("structures/raw/rcsb"))
    parser.add_argument("--output-dir", type=Path, default=Path("structures/processed"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--include-hetatm", action="store_true")
    parser.add_argument("--include-waters", action="store_true")
    parser.add_argument("--include-hydrogen", action="store_true")
    parser.add_argument("--crop", choices=["none", "sphere", "ellipsoid"], default="none")
    parser.add_argument("--crop-fraction", type=float, default=0.98)
    parser.add_argument("--euler-zyx-deg", default="17,31,43")
    parser.add_argument("--disorder-rms-angstrom", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    if not 0.0 < args.crop_fraction <= 1.0:
        raise ValueError("crop-fraction must be in (0, 1]")
    if args.disorder_rms_angstrom < 0.0:
        raise ValueError("disorder-rms-angstrom must be nonnegative")

    pdb_id = args.pdb_id.upper()
    supercell = parse_supercell(args.supercell)
    euler_deg = parse_triplet(args.euler_zyx_deg, name="euler-zyx-deg")
    raw_path = download_pdb(pdb_id, args.raw_dir, refresh=args.refresh)
    crystal = parse_pdb_crystal(
        raw_path,
        pdb_id=pdb_id,
        include_hetatm=args.include_hetatm,
        include_waters=args.include_waters,
        include_hydrogen=args.include_hydrogen,
    )
    elements_unit, fractional_unit = expand_unit_cell(crystal)
    cell = cell_matrix(crystal.cell_lengths_angstrom, crystal.cell_angles_deg)
    elements, coords_angstrom = build_supercell(elements_unit, fractional_unit, cell, supercell)
    elements, coords_angstrom, crop_metadata = apply_crop(
        elements,
        coords_angstrom,
        crop=args.crop,
        crop_fraction=args.crop_fraction,
    )

    rotation = euler_rotation_zyx(euler_deg)
    coords_angstrom = coords_angstrom @ rotation.T
    if args.disorder_rms_angstrom > 0.0:
        rng = np.random.default_rng(args.seed)
        coords_angstrom += rng.normal(
            0.0,
            args.disorder_rms_angstrom,
            size=coords_angstrom.shape,
        )
    coords_angstrom -= coords_angstrom.mean(axis=0, keepdims=True)

    output = args.output
    if output is None:
        output = args.output_dir / (
            build_output_name(pdb_id, supercell, args.crop, args.disorder_rms_angstrom) + ".npz"
        )
    metadata = save_npz(
        output=output,
        crystal=crystal,
        elements_unit=elements_unit,
        fractional_unit=fractional_unit,
        elements=elements,
        coords_angstrom=coords_angstrom,
        supercell=supercell,
        crop_metadata=crop_metadata,
        euler_deg=euler_deg,
        disorder_rms_angstrom=args.disorder_rms_angstrom,
        seed=args.seed,
        include_hetatm=args.include_hetatm,
        include_waters=args.include_waters,
        include_hydrogen=args.include_hydrogen,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
