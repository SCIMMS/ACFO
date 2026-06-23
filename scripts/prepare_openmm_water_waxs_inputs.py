from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=np.float64)
        elements = np.asarray(payload["elements"]).astype(str)
        box_vectors = (
            np.asarray(payload["box_vectors"], dtype=np.float64)
            if "box_vectors" in payload.files
            else None
        )
        metadata = (
            json.loads(str(payload["metadata_json"]))
            if "metadata_json" in payload.files
            else {}
        )
    return coords, elements, box_vectors, metadata


def parse_pdb_residue_groups(path: Path, n_atoms: int) -> list[list[int]]:
    groups: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    atom_index = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            key = (
                line[17:20].strip(),
                line[21].strip(),
                line[22:26].strip(),
                line[26].strip(),
                line[72:76].strip(),
            )
            groups[key].append(atom_index)
            atom_index += 1
    if atom_index != n_atoms:
        raise ValueError(f"{path} has {atom_index} PDB atoms but NPZ has {n_atoms} atoms")
    return list(groups.values())


def center_from_box(coords: np.ndarray, box_vectors: np.ndarray | None) -> np.ndarray:
    if box_vectors is None:
        return np.mean(coords, axis=0)
    return np.sum(box_vectors, axis=0) / 2.0


def select_residue_sphere(
    coords: np.ndarray,
    elements: np.ndarray,
    residue_groups: list[list[int]],
    center_nm: np.ndarray,
    radius_nm: float,
) -> np.ndarray:
    selected: list[int] = []
    for group in residue_groups:
        oxygen_indices = [idx for idx in group if elements[idx] == "O"]
        anchor = oxygen_indices[0] if oxygen_indices else group[0]
        if np.linalg.norm(coords[anchor] - center_nm) <= radius_nm:
            selected.extend(group)
    return np.asarray(selected, dtype=np.int64)


def write_npz(
    path: Path,
    coords: np.ndarray,
    elements: np.ndarray,
    box_vectors: np.ndarray | None,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coords": np.asarray(coords, dtype=np.float64),
        "elements": np.asarray(elements, dtype="<U4"),
        "metadata_json": np.asarray(json.dumps(metadata, separators=(",", ":"), sort_keys=True)),
    }
    if box_vectors is not None:
        payload["box_vectors"] = np.asarray(box_vectors, dtype=np.float64)
    np.savez_compressed(path, **payload)


def write_xyz(path: Path, coords: np.ndarray, elements: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{coords.shape[0]}\n")
        fh.write(f"{path.stem}\n")
        for element, (x, y, z) in zip(elements, coords, strict=True):
            fh.write(f"{element} {x * 10.0:.8f} {y * 10.0:.8f} {z * 10.0:.8f}\n")


def subset_variant(
    *,
    coords: np.ndarray,
    elements: np.ndarray,
    box_vectors: np.ndarray | None,
    metadata: dict,
    center_nm: np.ndarray,
    indices: np.ndarray,
    variant: str,
    output: Path,
    write_xyz_file: bool,
) -> dict:
    coords_sel = coords[indices] - center_nm
    elements_sel = elements[indices]
    if variant == "oonly":
        keep = elements_sel == "O"
        coords_sel = coords_sel[keep]
        elements_sel = elements_sel[keep]
    elif variant != "allatom":
        raise ValueError(f"unknown variant {variant!r}")

    out_metadata = dict(metadata)
    out_metadata.update(
        {
            "centered": True,
            "derived_variant": variant,
            "n_atoms": int(coords_sel.shape[0]),
            "source_kind": metadata.get("kind", "openmm_water_box"),
            "structure_id": output.stem,
        }
    )
    write_npz(output, coords_sel, elements_sel, box_vectors, out_metadata)
    if write_xyz_file:
        write_xyz(output.with_suffix(".xyz"), coords_sel, elements_sel)
    return {
        "path": str(output.as_posix()),
        "atoms": int(coords_sel.shape[0]),
        "variant": variant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an OpenMM TIP3P water-box NPZ/PDB pair into WAXS NPZ inputs."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("runs/water_tip3p_8nm/water_tip3p_8nm_final.npz"),
    )
    parser.add_argument(
        "--input-pdb",
        type=Path,
        default=Path("runs/water_tip3p_8nm/water_tip3p_8nm_final.pdb"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("structures/processed"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("structures/metadata"))
    parser.add_argument("--prefix", default="solvent_water_tip3p_8nm")
    parser.add_argument("--sphere-radius-nm", type=float, default=2.5)
    parser.add_argument("--write-xyz", action="store_true")
    args = parser.parse_args()

    coords, elements, box_vectors, metadata = load_npz(args.input_npz)
    center_nm = center_from_box(coords, box_vectors)
    residue_groups = parse_pdb_residue_groups(args.input_pdb, coords.shape[0])
    full_indices = np.arange(coords.shape[0], dtype=np.int64)
    sphere_indices = select_residue_sphere(
        coords,
        elements,
        residue_groups,
        center_nm,
        args.sphere_radius_nm,
    )

    rows: list[dict] = []
    for subset_name, indices in (
        ("full", full_indices),
        (f"sphere_r{args.sphere_radius_nm:g}nm".replace(".", "p"), sphere_indices),
    ):
        for variant in ("allatom", "oonly"):
            output = args.output_dir / f"{args.prefix}_{subset_name}_{variant}.npz"
            rows.append(
                subset_variant(
                    coords=coords,
                    elements=elements,
                    box_vectors=box_vectors,
                    metadata={
                        **metadata,
                        "input_npz": str(args.input_npz.as_posix()),
                        "input_pdb": str(args.input_pdb.as_posix()),
                        "subset": subset_name,
                        "sphere_radius_nm": args.sphere_radius_nm
                        if subset_name.startswith("sphere")
                        else None,
                    },
                    center_nm=center_nm,
                    indices=indices,
                    variant=variant,
                    output=output,
                    write_xyz_file=args.write_xyz,
                )
            )

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_npz": str(args.input_npz.as_posix()),
        "input_pdb": str(args.input_pdb.as_posix()),
        "source_atoms": int(coords.shape[0]),
        "source_elements": {
            str(element): int(np.sum(elements == element))
            for element in sorted(set(elements.tolist()))
        },
        "source_residues": len(residue_groups),
        "sphere_radius_nm": args.sphere_radius_nm,
        "outputs": rows,
    }
    summary_path = args.metadata_dir / f"{args.prefix}_derived_inputs.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
