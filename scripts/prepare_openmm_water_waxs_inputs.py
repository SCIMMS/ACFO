from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from collections.abc import Iterator
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


def read_dcd_record(fh, endian: str = "<") -> bytes | None:
    raw = fh.read(4)
    if raw == b"":
        return None
    if len(raw) != 4:
        raise ValueError("truncated DCD record marker")
    n_bytes = struct.unpack(f"{endian}i", raw)[0]
    payload = fh.read(n_bytes)
    if len(payload) != n_bytes:
        raise ValueError("truncated DCD record payload")
    end = struct.unpack(f"{endian}i", fh.read(4))[0]
    if end != n_bytes:
        raise ValueError(f"DCD record marker mismatch: {n_bytes} != {end}")
    return payload


def read_dcd_header(path: Path) -> dict:
    with path.open("rb") as fh:
        marker = fh.read(4)
        if len(marker) != 4:
            raise ValueError(f"{path} is too small to be a DCD file")
        little = struct.unpack("<i", marker)[0]
        big = struct.unpack(">i", marker)[0]
        if little == 84:
            endian = "<"
            n_header = little
        elif big == 84:
            endian = ">"
            n_header = big
        else:
            raise ValueError(f"{path} does not look like a DCD file")
        header = fh.read(n_header)
        end = struct.unpack(f"{endian}i", fh.read(4))[0]
        if end != n_header or header[:4] != b"CORD":
            raise ValueError(f"{path} has an invalid DCD header")
        ints = struct.unpack(f"{endian}20i", header[4:84])
        title = read_dcd_record(fh, endian)
        natoms_record = read_dcd_record(fh, endian)
        if title is None or natoms_record is None:
            raise ValueError(f"{path} is missing DCD title or atom-count records")
        natoms = struct.unpack(f"{endian}i", natoms_record)[0]
        return {
            "endian": endian,
            "frames": int(ints[0]),
            "first_step": int(ints[1]),
            "save_interval_steps": int(ints[2]),
            "last_step": int(ints[3]),
            "natoms": int(natoms),
            "data_offset": fh.tell(),
        }


def iter_dcd_frames(path: Path, *, frame_indices: set[int] | None = None) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    header = read_dcd_header(path)
    endian = header["endian"]
    natoms = int(header["natoms"])
    with path.open("rb") as fh:
        fh.seek(int(header["data_offset"]))
        frame_index = 0
        while True:
            first = read_dcd_record(fh, endian)
            if first is None:
                break
            if len(first) == 48:
                cell = np.asarray(struct.unpack(f"{endian}6d", first), dtype=np.float64)
                x_record = read_dcd_record(fh, endian)
            else:
                cell = None
                x_record = first
            y_record = read_dcd_record(fh, endian)
            z_record = read_dcd_record(fh, endian)
            if x_record is None or y_record is None or z_record is None:
                raise ValueError("truncated DCD coordinate records")

            if frame_indices is None or frame_index in frame_indices:
                x = np.frombuffer(x_record, dtype=f"{endian}f4", count=natoms)
                y = np.frombuffer(y_record, dtype=f"{endian}f4", count=natoms)
                z = np.frombuffer(z_record, dtype=f"{endian}f4", count=natoms)
                coords_nm = np.column_stack([x, y, z]).astype(np.float64) / 10.0
                if cell is None:
                    box_vectors = np.diag(np.ptp(coords_nm, axis=0))
                else:
                    # OpenMM DCD stores orthorhombic boxes as A, gamma, B, beta, alpha, C.
                    box_vectors = np.diag([cell[0], cell[2], cell[5]]) / 10.0
                yield frame_index, coords_nm, box_vectors
            frame_index += 1


def parse_frame_indices(spec: str | None, *, n_frames: int, max_frames: int) -> list[int]:
    if spec:
        out = [int(part.strip()) for part in spec.split(",") if part.strip()]
    elif max_frames >= n_frames:
        out = list(range(n_frames))
    else:
        out = np.linspace(0, n_frames - 1, max_frames, dtype=int).tolist()
    if not out:
        raise ValueError("no DCD frames selected")
    if min(out) < 0 or max(out) >= n_frames:
        raise ValueError(f"selected DCD frames must be in [0, {n_frames - 1}]")
    return sorted(dict.fromkeys(out))


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
    subset_name: str,
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
            "subset": subset_name,
        }
    )
    write_npz(output, coords_sel, elements_sel, box_vectors, out_metadata)
    if write_xyz_file:
        write_xyz(output.with_suffix(".xyz"), coords_sel, elements_sel)
    return {
        "path": str(output.as_posix()),
        "atoms": int(coords_sel.shape[0]),
        "subset": subset_name,
        "variant": variant,
    }


def derive_inputs_for_snapshot(
    *,
    coords: np.ndarray,
    elements: np.ndarray,
    box_vectors: np.ndarray | None,
    residue_groups: list[list[int]],
    metadata: dict,
    output_dir: Path,
    prefix: str,
    sphere_radius_nm: float,
    write_xyz_file: bool,
) -> list[dict]:
    center_nm = center_from_box(coords, box_vectors)
    full_indices = np.arange(coords.shape[0], dtype=np.int64)
    sphere_indices = select_residue_sphere(
        coords,
        elements,
        residue_groups,
        center_nm,
        sphere_radius_nm,
    )

    rows: list[dict] = []
    for subset_name, indices in (
        ("full", full_indices),
        (f"sphere_r{sphere_radius_nm:g}nm".replace(".", "p"), sphere_indices),
    ):
        for variant in ("allatom", "oonly"):
            output = output_dir / f"{prefix}_{subset_name}_{variant}.npz"
            rows.append(
                subset_variant(
                    coords=coords,
                    elements=elements,
                    box_vectors=box_vectors,
                    metadata={
                        **metadata,
                        "subset": subset_name,
                        "sphere_radius_nm": sphere_radius_nm
                        if subset_name.startswith("sphere")
                        else None,
                    },
                    center_nm=center_nm,
                    indices=indices,
                    subset_name=subset_name,
                    variant=variant,
                    output=output,
                    write_xyz_file=write_xyz_file,
                )
            )
    return rows


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
    parser.add_argument("--trajectory-dcd", type=Path)
    parser.add_argument("--frame-indices")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--write-xyz", action="store_true")
    args = parser.parse_args()

    coords, elements, box_vectors, metadata = load_npz(args.input_npz)
    residue_groups = parse_pdb_residue_groups(args.input_pdb, coords.shape[0])

    rows: list[dict] = []
    frame_indices: list[int] | None = None
    if args.trajectory_dcd is None:
        rows.extend(
            derive_inputs_for_snapshot(
                coords=coords,
                elements=elements,
                box_vectors=box_vectors,
                residue_groups=residue_groups,
                metadata={
                    **metadata,
                    "input_npz": str(args.input_npz.as_posix()),
                    "input_pdb": str(args.input_pdb.as_posix()),
                },
                output_dir=args.output_dir,
                prefix=args.prefix,
                sphere_radius_nm=args.sphere_radius_nm,
                write_xyz_file=args.write_xyz,
            )
        )
    else:
        dcd_header = read_dcd_header(args.trajectory_dcd)
        if int(dcd_header["natoms"]) != coords.shape[0]:
            raise ValueError(
                f"DCD has {dcd_header['natoms']} atoms but NPZ has {coords.shape[0]}"
            )
        frame_indices = parse_frame_indices(
            args.frame_indices,
            n_frames=int(dcd_header["frames"]),
            max_frames=args.max_frames,
        )
        for frame_index, frame_coords, frame_box_vectors in iter_dcd_frames(
            args.trajectory_dcd,
            frame_indices=set(frame_indices),
        ):
            frame_prefix = f"{args.prefix}_frame{frame_index:03d}"
            rows.extend(
                derive_inputs_for_snapshot(
                    coords=frame_coords,
                    elements=elements,
                    box_vectors=frame_box_vectors,
                    residue_groups=residue_groups,
                    metadata={
                        **metadata,
                        "dcd_frame_index": frame_index,
                        "dcd_header": dcd_header,
                        "input_dcd": str(args.trajectory_dcd.as_posix()),
                        "input_npz": str(args.input_npz.as_posix()),
                        "input_pdb": str(args.input_pdb.as_posix()),
                    },
                    output_dir=args.output_dir,
                    prefix=frame_prefix,
                    sphere_radius_nm=args.sphere_radius_nm,
                    write_xyz_file=args.write_xyz,
                )
            )

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_npz": str(args.input_npz.as_posix()),
        "input_pdb": str(args.input_pdb.as_posix()),
        "input_dcd": None
        if args.trajectory_dcd is None
        else str(args.trajectory_dcd.as_posix()),
        "frame_indices": frame_indices,
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
