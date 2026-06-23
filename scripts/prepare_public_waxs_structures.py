from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "TI": 22,
    "FE": 26,
    "CU": 29,
    "ZN": 30,
    "BR": 35,
    "AG": 47,
    "I": 53,
    "AU": 79,
    "PB": 82,
}

WATER_RESNAMES = {"HOH", "WAT", "DOD", "H2O"}


@dataclass(frozen=True)
class ParsedStructure:
    coords_nm: np.ndarray
    elements: np.ndarray
    records: tuple[str, ...]
    residues: tuple[str, ...]
    chains: tuple[str, ...]


def normalize_element(raw: str, atom_name: str) -> str:
    candidate = raw.strip().upper()
    if not candidate:
        letters = re.sub(r"[^A-Za-z]", "", atom_name).upper()
        if not letters:
            raise ValueError(f"cannot infer element from atom name {atom_name!r}")
        if len(letters) >= 2 and letters[:2] in ATOMIC_NUMBERS:
            candidate = letters[:2]
        else:
            candidate = letters[0]
    if len(candidate) > 1:
        two = candidate[:2]
        if two in ATOMIC_NUMBERS:
            candidate = two
        else:
            candidate = candidate[0]
    if candidate not in ATOMIC_NUMBERS:
        raise ValueError(f"unsupported element {candidate!r} from atom {atom_name!r}")
    return candidate[0] + candidate[1:].lower()


def parse_pdb(
    path: Path,
    *,
    include_hetatm: bool,
    include_waters: bool,
    include_hydrogen: bool,
) -> ParsedStructure:
    coords_angstrom: list[tuple[float, float, float]] = []
    elements: list[str] = []
    records: list[str] = []
    residues: list[str] = []
    chains: list[str] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError as exc:
            raise ValueError(f"failed to parse coordinates in {path}: {line!r}") from exc

        coords_angstrom.append((x, y, z))
        elements.append(element)
        records.append(record)
        residues.append(resname)
        chains.append(line[21:22].strip() or "_")

    if not coords_angstrom:
        raise ValueError(f"no atoms parsed from {path}")

    coords_nm = np.asarray(coords_angstrom, dtype=np.float64) * 0.1
    coords_nm -= coords_nm.mean(axis=0, keepdims=True)
    return ParsedStructure(
        coords_nm=coords_nm,
        elements=np.asarray(elements, dtype="U2"),
        records=tuple(records),
        residues=tuple(residues),
        chains=tuple(chains),
    )


def download_rcsb_pdb(pdb_id: str, raw_dir: Path, *, refresh: bool) -> Path:
    pdb_id = pdb_id.upper()
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{pdb_id}.pdb"
    if out_path.exists() and not refresh:
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    out_path.write_bytes(payload)
    return out_path


def write_xyz(path: Path, coords_nm: np.ndarray, elements: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(coords_nm.shape[0]), comment]
    coords_angstrom = coords_nm * 10.0
    for element, xyz in zip(elements, coords_angstrom, strict=True):
        lines.append(f"{element} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_structure(
    parsed: ParsedStructure,
    *,
    pdb_id: str,
    raw_path: Path,
    processed_dir: Path,
    metadata_dir: Path,
    write_helper_xyz: bool,
    include_hetatm: bool,
    include_waters: bool,
    include_hydrogen: bool,
) -> Path:
    structure_id = f"protein_{pdb_id.lower()}_heavy_centered"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    element_counts = {
        str(element): int(np.sum(parsed.elements == element))
        for element in sorted(set(parsed.elements.tolist()))
    }
    atomic_numbers = np.asarray(
        [ATOMIC_NUMBERS[str(element).upper()] for element in parsed.elements],
        dtype=np.int16,
    )
    extent = {
        "min_nm": parsed.coords_nm.min(axis=0).round(8).tolist(),
        "max_nm": parsed.coords_nm.max(axis=0).round(8).tolist(),
        "span_nm": np.ptp(parsed.coords_nm, axis=0).round(8).tolist(),
    }
    metadata = {
        "structure_id": structure_id,
        "system_type": "protein_crystal_structure",
        "source_database": "RCSB PDB",
        "source_id": pdb_id.upper(),
        "source_url": f"https://www.rcsb.org/structure/{pdb_id.upper()}",
        "coordinate_url": f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
        "raw_file": str(raw_path.as_posix()),
        "units": "nm",
        "periodic": False,
        "centered": "centroid",
        "filters": {
            "include_hetatm": include_hetatm,
            "include_waters": include_waters,
            "include_hydrogen": include_hydrogen,
            "altloc": "blank or A",
        },
        "n_atoms": int(parsed.coords_nm.shape[0]),
        "element_counts": element_counts,
        "chains": sorted(set(parsed.chains)),
        "extent": extent,
        "notes": "Initial public-structure WAXS validation input; hydrogens are absent unless provided by the source and include_hydrogen is set.",
    }
    metadata_json = json.dumps(metadata, separators=(",", ":"), sort_keys=True)

    out_npz = processed_dir / f"{structure_id}.npz"
    np.savez_compressed(
        out_npz,
        coords=parsed.coords_nm,
        elements=parsed.elements,
        atomic_numbers=atomic_numbers,
        structure_id=np.asarray(structure_id),
        metadata_json=np.asarray(metadata_json),
    )
    (metadata_dir / f"{structure_id}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if write_helper_xyz:
        write_xyz(
            processed_dir / f"{structure_id}.xyz",
            parsed.coords_nm,
            parsed.elements,
            f"{structure_id}; coordinates in Angstrom for visualization",
        )
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download small public RCSB structures and convert them to WAXS NPZ inputs."
    )
    parser.add_argument("--pdb-ids", default="1CRN,1UBQ")
    parser.add_argument("--raw-dir", type=Path, default=Path("structures/raw/rcsb"))
    parser.add_argument("--processed-dir", type=Path, default=Path("structures/processed"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("structures/metadata"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--include-hetatm", action="store_true")
    parser.add_argument("--include-waters", action="store_true")
    parser.add_argument("--include-hydrogen", action="store_true")
    parser.add_argument("--write-xyz", action="store_true")
    args = parser.parse_args()

    pdb_ids = [part.strip().upper() for part in args.pdb_ids.split(",") if part.strip()]
    if not pdb_ids:
        raise ValueError("at least one PDB ID is required")

    outputs: list[Path] = []
    for pdb_id in pdb_ids:
        raw_path = download_rcsb_pdb(pdb_id, args.raw_dir, refresh=args.refresh)
        parsed = parse_pdb(
            raw_path,
            include_hetatm=args.include_hetatm,
            include_waters=args.include_waters,
            include_hydrogen=args.include_hydrogen,
        )
        out_npz = save_structure(
            parsed,
            pdb_id=pdb_id,
            raw_path=raw_path,
            processed_dir=args.processed_dir,
            metadata_dir=args.metadata_dir,
            write_helper_xyz=args.write_xyz,
            include_hetatm=args.include_hetatm,
            include_waters=args.include_waters,
            include_hydrogen=args.include_hydrogen,
        )
        outputs.append(out_npz)
        print(f"{pdb_id}: atoms={parsed.coords_nm.shape[0]} -> {out_npz}")

    print("wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
