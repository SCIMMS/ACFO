from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake.odt_measured_contract import (  # noqa: E402
    SCHEMA_VERSION,
    OdtMeasuredData,
    save_odt_measured_contract,
    validate_odt_measured_contract,
)


def centered_frequency_axis(n: int, pixel_size_um: float) -> np.ndarray:
    indices = np.arange(-n // 2, (n - 1) // 2 + 1, dtype=np.float64)
    if indices.size != n:
        indices = np.arange(-math.floor(n / 2), math.floor((n - 1) / 2) + 1, dtype=np.float64)
    return indices / (pixel_size_um * n)


def crop_center(data: np.ndarray, crop_size: int | None) -> np.ndarray:
    if crop_size is None or crop_size <= 0:
        return data
    if crop_size > data.shape[1] or crop_size > data.shape[2]:
        raise ValueError(f"crop_size {crop_size} exceeds data shape {data.shape}")
    y0 = (data.shape[1] - crop_size) // 2
    x0 = (data.shape[2] - crop_size) // 2
    return np.ascontiguousarray(data[:, y0 : y0 + crop_size, x0 : x0 + crop_size])


def load_aidt_raw(mat_path: Path, data_key: str, crop_size: int | None) -> np.ndarray:
    loaded = sio.loadmat(mat_path)
    if data_key not in loaded:
        keys = sorted(k for k in loaded if not k.startswith("__"))
        raise KeyError(f"{data_key!r} not found in {mat_path}; available keys: {keys}")
    raw = np.asarray(loaded[data_key], dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError(f"{data_key!r} must be a 3D MATLAB array, got {raw.shape}")
    data = np.moveaxis(raw, 2, 0)
    return crop_center(np.ascontiguousarray(data), crop_size)


def load_sorted_positions(path: Path, n_illum: int) -> np.ndarray:
    loaded = sio.loadmat(path)
    if "Sorted_Pos" not in loaded:
        raise KeyError(f"'Sorted_Pos' not found in {path}")
    sorted_pos = np.asarray(loaded["Sorted_Pos"], dtype=np.float64)
    if sorted_pos.shape[0] != 2:
        raise ValueError(f"Sorted_Pos must have shape (2, N), got {sorted_pos.shape}")
    if sorted_pos.shape[1] < n_illum:
        raise ValueError(f"Sorted_Pos has {sorted_pos.shape[1]} positions, but data has {n_illum} frames")
    if sorted_pos.shape[1] == n_illum:
        return np.ascontiguousarray(sorted_pos)
    step = sorted_pos.shape[1] // n_illum
    return np.ascontiguousarray(sorted_pos[:, ::step][:, :n_illum])


def design_source_na(sorted_pos: np.ndarray, objective_na: float) -> np.ndarray:
    return np.ascontiguousarray((sorted_pos.T * objective_na).astype(np.float64))


def illumination_dirs_from_source_na(source_na_xy: np.ndarray) -> np.ndarray:
    radial2 = np.sum(source_na_xy**2, axis=1)
    if np.any(radial2 >= 1.0):
        raise ValueError("source NA must be below 1.0 to form unit illumination directions")
    z = np.sqrt(1.0 - radial2)
    return np.ascontiguousarray(np.column_stack([source_na_xy[:, 0], source_na_xy[:, 1], z]))


def convert(args: argparse.Namespace) -> tuple[OdtMeasuredData, dict[str, Any]]:
    wavelength_um = float(args.wavelength_um)
    pixel_size_um = float(args.camera_pixel_um) / float(args.magnification)
    data = load_aidt_raw(args.raw_mat, args.data_key, args.crop_size)
    n_illum, height, width = data.shape
    sorted_pos = load_sorted_positions(args.sorted_pos_mat, n_illum)
    source_na_xy = design_source_na(sorted_pos, args.objective_na)
    illum_dirs = illumination_dirs_from_source_na(source_na_xy)
    frequency_x = centered_frequency_axis(width, pixel_size_um)
    frequency_y = centered_frequency_axis(height, pixel_size_um)
    fields: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": "annular_idt",
        "measurement_model": "coherent_intensity",
        "wavelength": wavelength_um,
        "k": float(2.0 * math.pi * args.medium_index / wavelength_um),
        "medium_index": float(args.medium_index),
        "units": "um",
        "illum_dirs": illum_dirs,
        "detector_origin": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "detector_u": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "detector_v": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "detector_pixel_size": np.array([pixel_size_um, pixel_size_um], dtype=np.float64),
        "detector_distance": 1.0,
        "q_layout": "annular_cartesian_stack",
        "objective_na": float(args.objective_na),
        "source_na_xy": source_na_xy,
        "frequency_x": frequency_x,
        "frequency_y": frequency_y,
        "coordinate_convention": "aidt_annular_led_cartesian_image_stack",
        "data": data,
        "source_name": "aIDT Diatom I raw intensity stack",
        "source_repo": "bu-cisl/IDT-using-Annular-Illumination",
        "source_data_url": "https://drive.google.com/drive/folders/1E8U1YWQ14bDbgwMJnaUZl1I6Lq3XoGWj",
        "source_paper": "High-speed in vitro intensity diffraction tomography",
        "source_arxiv": "1904.06004",
        "source_license_note": "Repository is BSD-3-Clause; raw data are linked from the repository README but do not carry a separate explicit data license in the checked files.",
        "source_raw_mat": str(args.raw_mat),
        "source_sorted_pos_mat": str(args.sorted_pos_mat),
    }
    source_summary = {
        "raw_mat": str(args.raw_mat),
        "sorted_pos_mat": str(args.sorted_pos_mat),
        "data_shape": tuple(int(v) for v in data.shape),
        "data_dtype": str(data.dtype),
        "data_min": float(np.min(data)),
        "data_max": float(np.max(data)),
        "data_mean": float(np.mean(data)),
        "data_std": float(np.std(data)),
        "wavelength_um": wavelength_um,
        "pixel_size_um": pixel_size_um,
        "objective_na": float(args.objective_na),
        "medium_index": float(args.medium_index),
        "source_na_min": float(np.min(np.linalg.norm(source_na_xy, axis=1))),
        "source_na_max": float(np.max(np.linalg.norm(source_na_xy, axis=1))),
        "frequency_x_min": float(np.min(frequency_x)),
        "frequency_x_max": float(np.max(frequency_x)),
        "frequency_y_min": float(np.min(frequency_y)),
        "frequency_y_max": float(np.max(frequency_y)),
    }
    return OdtMeasuredData(fields=fields, source_path=args.out), source_summary


def write_summary(path: Path, *, validation: dict[str, Any], source_summary: dict[str, Any]) -> None:
    report_summary = validation["summary"]
    lines = [
        "# Public aIDT annular-intensity contract conversion",
        "",
        "This file records conversion of the public aIDT Diatom I raw intensity stack into the local measured-data contract.",
        "",
        "## Source",
        "",
        "- repository: `bu-cisl/IDT-using-Annular-Illumination`",
        "- paper: `High-speed in vitro intensity diffraction tomography`",
        "- arXiv: `1904.06004`",
        "- raw data: public Google Drive folder linked from the repository README",
        "- data license note: the repository is BSD-3-Clause; a separate raw-data license was not found in the checked files",
        "",
        "## Converted Contract",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    for key in (
        "experiment_type",
        "measurement_model",
        "q_layout",
        "n_illum",
        "cap_radial",
        "cap_phi",
        "q_samples",
        "measurement_samples",
        "data_shape",
        "data_dtype",
        "objective_na",
        "source_na_min",
        "source_na_max",
        "has_mask",
    ):
        lines.append(f"| `{key}` | `{report_summary.get(key)}` |")
    lines.extend(
        [
            "",
            "## Source Data Readout",
            "",
            "| key | value |",
            "| --- | --- |",
        ]
    )
    for key, value in source_summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Geometry Fit",
            "",
            "This is a stronger public-data candidate than a conventional rotational sinogram for the current ODT acceleration story.",
            "It contains measured intensity frames under a fixed annular illumination design, so the acquisition geometry naturally exposes repeated ring/annular structure.",
            "It is still not a finished prepared-operator benchmark: the next step is to map the annular Cartesian image stack into the exact curved-Ewald operator and compare prepared GPU against cuFINUFFT on the same measured update.",
            "",
            "## Validation",
            "",
            f"- valid: `{validation['ok']}`",
            f"- errors: `{len(validation['errors'])}`",
            f"- warnings: `{len(validation['warnings'])}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert public aIDT annular raw intensity data to the local ODT measured-data contract.")
    base = ROOT / "benchmark_results" / "public_data_probe" / "aidt"
    parser.add_argument("--raw-mat", type=Path, default=base / "IRaw_Diatom_I.mat")
    parser.add_argument("--sorted-pos-mat", type=Path, default=base / "repo_files" / "Sorted_Pos.mat")
    parser.add_argument("--data-key", default="I_Raw")
    parser.add_argument("--wavelength-um", type=float, default=0.515)
    parser.add_argument("--objective-na", type=float, default=0.65)
    parser.add_argument("--magnification", type=float, default=40.0)
    parser.add_argument("--camera-pixel-um", type=float, default=6.5)
    parser.add_argument("--medium-index", type=float, default=1.47)
    parser.add_argument("--crop-size", type=int, default=0, help="Optional centered square crop; 0 keeps the full frame.")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark_results" / "aidt_diatom_public_contract.npz")
    parser.add_argument("--summary-md", type=Path, default=ROOT / "benchmark_results" / "aidt_diatom_public_contract_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crop_size = None if args.crop_size <= 0 else args.crop_size
    args.crop_size = crop_size
    contract, source_summary = convert(args)
    report = validate_odt_measured_contract(contract)
    report.raise_for_errors()
    save_odt_measured_contract(args.out, contract)
    write_summary(args.summary_md, validation=report.to_dict(), source_summary=source_summary)
    print(json.dumps({"out": str(args.out), "summary_md": str(args.summary_md), "validation": report.to_dict()}, indent=2, default=str))


if __name__ == "__main__":
    main()
