from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Ca": 20,
    "Ti": 22,
    "Fe": 26,
    "Cu": 29,
    "Zn": 30,
    "Br": 35,
    "Ag": 47,
    "I": 53,
    "Au": 79,
    "Pb": 82,
}


@dataclass(frozen=True)
class CifStructure:
    cell_lengths_angstrom: tuple[float, float, float]
    cell_angles_deg: tuple[float, float, float]
    asym_elements: tuple[str, ...]
    asym_frac: np.ndarray
    asym_multiplicities: tuple[int | None, ...]
    symops: tuple[str, ...]
    metadata: dict


def parse_supercell(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("supercell must be N or Nx,Ny,Nz with positive integers")
    return tuple(parts)  # type: ignore[return-value]


def parse_named_ids(value: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            name, cod_id = item.split("=", 1)
        else:
            cod_id = item
            name = f"cod_{cod_id}"
        cod_id = cod_id.strip()
        if not re.fullmatch(r"\d{7}", cod_id):
            raise ValueError(f"COD ID must be seven digits: {cod_id!r}")
        out.append((name.strip().lower(), cod_id))
    if not out:
        raise ValueError("at least one COD ID is required")
    return out


def download_cod_cif(name: str, cod_id: str, raw_dir: Path, *, refresh: bool) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{name}_{cod_id}.cif"
    if out_path.exists() and not refresh:
        return out_path
    url = f"https://www.crystallography.net/cod/{cod_id}.cif"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    out_path.write_bytes(payload)
    return out_path


def strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:i]
    return line


def tokenize(line: str) -> list[str]:
    lexer = shlex.shlex(strip_comment(line), posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def clean_number(value: str) -> float:
    text = value.strip().strip("'\"")
    if text in {".", "?"}:
        raise ValueError(f"missing CIF numeric value: {value!r}")
    text = re.sub(r"\([^)]*\)$", "", text)
    return float(text)


def normalize_element(raw: str, fallback: str = "") -> str:
    text = raw.strip().strip("'\"")
    if not text or text in {".", "?"}:
        text = fallback
    match = re.search(r"[A-Z][a-z]?", text)
    if not match:
        raise ValueError(f"cannot infer element from {raw!r}")
    element = match.group(0)
    if element not in ATOMIC_NUMBERS:
        upper = element.upper()
        for known in ATOMIC_NUMBERS:
            if known.upper() == upper:
                return known
        raise ValueError(f"unsupported element {element!r}")
    return element


def parse_cif_text(text: str) -> tuple[dict[str, str], list[tuple[list[str], list[list[str]]]]]:
    lines = text.splitlines()
    scalars: dict[str, str] = {}
    loops: list[tuple[list[str], list[list[str]]]] = []
    i = 0
    while i < len(lines):
        line = strip_comment(lines[i]).strip()
        if not line:
            i += 1
            continue
        lower = line.lower()
        if lower == "loop_":
            i += 1
            headers: list[str] = []
            while i < len(lines):
                header_line = strip_comment(lines[i]).strip()
                if header_line.startswith("_"):
                    headers.append(tokenize(header_line)[0])
                    i += 1
                    continue
                break
            rows: list[list[str]] = []
            pending: list[str] = []
            while i < len(lines):
                row_line = strip_comment(lines[i]).strip()
                row_lower = row_line.lower()
                if (
                    not row_line
                    or row_lower == "loop_"
                    or row_line.startswith("_")
                    or row_lower.startswith("data_")
                    or row_lower.startswith("save_")
                ):
                    if pending and len(pending) >= len(headers):
                        rows.append(pending[: len(headers)])
                    pending = []
                    if not row_line:
                        i += 1
                        continue
                    break
                pending.extend(tokenize(row_line))
                while headers and len(pending) >= len(headers):
                    rows.append(pending[: len(headers)])
                    pending = pending[len(headers) :]
                i += 1
            loops.append((headers, rows))
            continue
        if line.startswith("_"):
            parts = tokenize(line)
            if len(parts) >= 2:
                scalars[parts[0]] = parts[1]
            elif len(parts) == 1 and i + 1 < len(lines):
                scalars[parts[0]] = strip_comment(lines[i + 1]).strip()
                i += 1
        i += 1
    return scalars, loops


def find_loop(
    loops: list[tuple[list[str], list[list[str]]]],
    required: set[str],
) -> tuple[list[str], list[list[str]]] | None:
    lowered_required = {item.lower() for item in required}
    for headers, rows in loops:
        lowered = {header.lower() for header in headers}
        if lowered_required.issubset(lowered):
            return headers, rows
    return None


def parse_cif(path: Path, *, cod_name: str, cod_id: str) -> CifStructure:
    scalars, loops = parse_cif_text(path.read_text(encoding="utf-8", errors="replace"))
    cell_lengths = (
        clean_number(scalars["_cell_length_a"]),
        clean_number(scalars["_cell_length_b"]),
        clean_number(scalars["_cell_length_c"]),
    )
    cell_angles = (
        clean_number(scalars["_cell_angle_alpha"]),
        clean_number(scalars["_cell_angle_beta"]),
        clean_number(scalars["_cell_angle_gamma"]),
    )

    atom_loop = find_loop(
        loops,
        {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"},
    )
    if atom_loop is None:
        raise ValueError(f"{path} has no supported atom_site fractional loop")
    atom_headers, atom_rows = atom_loop
    atom_index = {header.lower(): i for i, header in enumerate(atom_headers)}
    ix = atom_index["_atom_site_fract_x"]
    iy = atom_index["_atom_site_fract_y"]
    iz = atom_index["_atom_site_fract_z"]
    itype = atom_index.get("_atom_site_type_symbol")
    ilabel = atom_index.get("_atom_site_label")
    imult = atom_index.get("_atom_site_symmetry_multiplicity")

    elements: list[str] = []
    frac: list[tuple[float, float, float]] = []
    multiplicities: list[int | None] = []
    for row in atom_rows:
        label = row[ilabel] if ilabel is not None else ""
        element = normalize_element(row[itype] if itype is not None else "", label)
        elements.append(element)
        frac.append((clean_number(row[ix]), clean_number(row[iy]), clean_number(row[iz])))
        multiplicities.append(None if imult is None else int(round(clean_number(row[imult]))))

    symops = ["x,y,z"]
    for key in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
        sym_loop = find_loop(loops, {key})
        if sym_loop is not None:
            headers, rows = sym_loop
            idx = {header.lower(): i for i, header in enumerate(headers)}[key]
            symops = [row[idx].strip().strip("'\"") for row in rows]
            break

    metadata = {
        "source_database": "Crystallography Open Database",
        "source_id": cod_id,
        "source_name": cod_name,
        "source_url": f"https://www.crystallography.net/cod/{cod_id}.cif",
        "chemical_formula_sum": scalars.get("_chemical_formula_sum"),
        "chemical_name_mineral": scalars.get("_chemical_name_mineral"),
        "chemical_name_common": scalars.get("_chemical_name_common"),
        "space_group_name": scalars.get("_space_group_name_H-M_alt")
        or scalars.get("_symmetry_space_group_name_H-M"),
        "space_group_number": scalars.get("_space_group_IT_number")
        or scalars.get("_symmetry_Int_Tables_number"),
        "cell_lengths_angstrom": list(cell_lengths),
        "cell_angles_deg": list(cell_angles),
        "n_asymmetric_atoms": len(elements),
        "n_symmetry_operations": len(symops),
    }
    return CifStructure(
        cell_lengths_angstrom=cell_lengths,
        cell_angles_deg=cell_angles,
        asym_elements=tuple(elements),
        asym_frac=np.asarray(frac, dtype=np.float64),
        asym_multiplicities=tuple(multiplicities),
        symops=tuple(symops),
        metadata=metadata,
    )


def eval_fraction(text: str) -> float:
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def eval_axis_expr(expr: str, xyz: np.ndarray) -> float:
    clean = expr.strip().strip("'\"").replace(" ", "").replace("-", "+-")
    if clean.startswith("+"):
        clean = clean[1:]
    total = 0.0
    for token in clean.split("+"):
        if not token:
            continue
        sign = -1.0 if token.startswith("-") else 1.0
        core = token[1:] if token.startswith("-") else token
        if core == "x":
            total += sign * xyz[0]
        elif core == "y":
            total += sign * xyz[1]
        elif core == "z":
            total += sign * xyz[2]
        else:
            total += sign * eval_fraction(core)
    return total


def apply_symop(symop: str, frac: np.ndarray) -> np.ndarray:
    parts = [part.strip() for part in symop.split(",")]
    if len(parts) != 3:
        raise ValueError(f"unsupported symmetry operation {symop!r}")
    return np.asarray([eval_axis_expr(part, frac) for part in parts], dtype=np.float64) % 1.0


def expand_unit_cell(structure: CifStructure, *, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    entries: dict[tuple[str, int, int, int], tuple[str, np.ndarray]] = {}
    scale = 1.0 / tol
    for element, frac, multiplicity in zip(
        structure.asym_elements,
        structure.asym_frac,
        structure.asym_multiplicities,
        strict=True,
    ):
        site_entries: list[tuple[tuple[str, int, int, int], tuple[str, np.ndarray]]] = []
        seen: set[tuple[str, int, int, int]] = set()
        for symop in structure.symops:
            pos = apply_symop(symop, frac)
            pos[np.isclose(pos, 1.0, atol=tol)] = 0.0
            key = (element, *(int(round(float(value) * scale)) for value in pos))
            if key not in seen:
                seen.add(key)
                site_entries.append((key, (element, pos)))
        if multiplicity is not None and len(site_entries) > multiplicity:
            site_entries = site_entries[:multiplicity]
        for key, value in site_entries:
            entries[key] = value
    elements = np.asarray([value[0] for value in entries.values()], dtype="U2")
    frac = np.asarray([value[1] for value in entries.values()], dtype=np.float64)
    order = np.lexsort((frac[:, 2], frac[:, 1], frac[:, 0], elements.astype(str)))
    return elements[order], frac[order]


def cell_matrix(
    cell_lengths: tuple[float, float, float],
    cell_angles: tuple[float, float, float],
) -> np.ndarray:
    a, b, c = cell_lengths
    alpha, beta, gamma = [math.radians(angle) for angle in cell_angles]
    va = np.array([a, 0.0, 0.0], dtype=np.float64)
    vb = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0], dtype=np.float64)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cz_sq = max(c * c - cx * cx - cy * cy, 0.0)
    vc = np.array([cx, cy, math.sqrt(cz_sq)], dtype=np.float64)
    return np.vstack([va, vb, vc])


def build_supercell(
    elements_unit: np.ndarray,
    frac_unit: np.ndarray,
    cell: np.ndarray,
    supercell: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    elems: list[str] = []
    frac_positions: list[np.ndarray] = []
    for ix in range(supercell[0]):
        for iy in range(supercell[1]):
            for iz in range(supercell[2]):
                shift = np.asarray([ix, iy, iz], dtype=np.float64)
                for element, frac in zip(elements_unit, frac_unit, strict=True):
                    elems.append(str(element))
                    frac_positions.append(frac + shift)
    frac_all = np.vstack(frac_positions)
    coords_nm = (frac_all @ cell) * 0.1
    coords_nm -= coords_nm.mean(axis=0, keepdims=True)
    return np.asarray(elems, dtype="U2"), coords_nm


def write_xyz(path: Path, coords_nm: np.ndarray, elements: np.ndarray, comment: str) -> None:
    lines = [str(coords_nm.shape[0]), comment]
    coords_angstrom = coords_nm * 10.0
    for element, xyz in zip(elements, coords_angstrom, strict=True):
        lines.append(f"{element} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_npz(
    *,
    name: str,
    cod_id: str,
    raw_path: Path,
    structure: CifStructure,
    elements_unit: np.ndarray,
    frac_unit: np.ndarray,
    elements: np.ndarray,
    coords_nm: np.ndarray,
    supercell: tuple[int, int, int],
    processed_dir: Path,
    metadata_dir: Path,
    write_helper_xyz: bool,
) -> Path:
    structure_id = f"crystal_{name}_cod{cod_id}_{supercell[0]}x{supercell[1]}x{supercell[2]}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    atomic_numbers = np.asarray([ATOMIC_NUMBERS[str(element)] for element in elements], dtype=np.int16)
    element_counts = {
        str(element): int(np.sum(elements == element))
        for element in sorted(set(elements.tolist()))
    }
    metadata = {
        **structure.metadata,
        "structure_id": structure_id,
        "system_type": "crystal_supercell",
        "raw_file": str(raw_path.as_posix()),
        "units": "nm",
        "periodic": False,
        "centered": "centroid",
        "supercell": list(supercell),
        "n_unit_cell_atoms_after_symmetry": int(elements_unit.size),
        "n_atoms": int(coords_nm.shape[0]),
        "element_counts": element_counts,
        "extent": {
            "min_nm": coords_nm.min(axis=0).round(8).tolist(),
            "max_nm": coords_nm.max(axis=0).round(8).tolist(),
            "span_nm": np.ptp(coords_nm, axis=0).round(8).tolist(),
        },
        "notes": "COD CIF converted with a lightweight parser for WAXS crystal cake-map validation.",
    }
    metadata_json = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    out_npz = processed_dir / f"{structure_id}.npz"
    np.savez_compressed(
        out_npz,
        coords=coords_nm,
        elements=elements,
        atomic_numbers=atomic_numbers,
        unit_cell_fractional=frac_unit,
        unit_cell_elements=elements_unit,
        cell_matrix_angstrom=cell_matrix(structure.cell_lengths_angstrom, structure.cell_angles_deg),
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
            coords_nm,
            elements,
            f"{structure_id}; coordinates in Angstrom for visualization",
        )
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public COD CIF crystal structures and convert them to WAXS NPZ supercells."
    )
    parser.add_argument(
        "--cod-ids",
        default="nacl=1000041,silicon=1526655,quartz=1011097",
        help="Comma-separated name=CODID pairs.",
    )
    parser.add_argument("--supercell", default="4,4,4")
    parser.add_argument("--raw-dir", type=Path, default=Path("structures/raw/cod"))
    parser.add_argument("--processed-dir", type=Path, default=Path("structures/processed"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("structures/metadata"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--write-xyz", action="store_true")
    args = parser.parse_args()

    supercell = parse_supercell(args.supercell)
    outputs: list[Path] = []
    for name, cod_id in parse_named_ids(args.cod_ids):
        raw_path = download_cod_cif(name, cod_id, args.raw_dir, refresh=args.refresh)
        structure = parse_cif(raw_path, cod_name=name, cod_id=cod_id)
        elements_unit, frac_unit = expand_unit_cell(structure)
        cell = cell_matrix(structure.cell_lengths_angstrom, structure.cell_angles_deg)
        elements, coords_nm = build_supercell(elements_unit, frac_unit, cell, supercell)
        out_npz = save_npz(
            name=name,
            cod_id=cod_id,
            raw_path=raw_path,
            structure=structure,
            elements_unit=elements_unit,
            frac_unit=frac_unit,
            elements=elements,
            coords_nm=coords_nm,
            supercell=supercell,
            processed_dir=args.processed_dir,
            metadata_dir=args.metadata_dir,
            write_helper_xyz=args.write_xyz,
        )
        outputs.append(out_npz)
        print(
            f"{name} COD {cod_id}: asym={len(structure.asym_elements)} "
            f"unit={elements_unit.size} supercell={coords_nm.shape[0]} -> {out_npz}"
        )

    print("wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
