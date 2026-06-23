from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

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


def qpi_key_index(name: str) -> int:
    match = re.fullmatch(r"qpi_(\d+)", name)
    if match is None:
        raise ValueError(f"not a qpi key: {name}")
    return int(match.group(1))


def load_qpimage_series(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        keys = sorted((key for key in handle.keys() if key.startswith("qpi_")), key=qpi_key_index)
        if not keys:
            raise ValueError(f"no qpi_* groups found in {path}")
        first = handle[keys[0]]
        meta = {
            "wavelength_m": float(first.attrs["wavelength"]),
            "pixel_size_m": float(first.attrs["pixel size"]),
            "medium_index": float(first.attrs["medium index"]),
            "qpimage_version": str(first.attrs.get("qpimage version", "")),
            "frames": len(keys),
        }
        sample_shape = tuple(int(v) for v in first["amplitude/raw"].shape)
        data = np.empty((len(keys), *sample_shape), dtype=np.complex64)
        amp_min = math.inf
        amp_max = -math.inf
        phase_min = math.inf
        phase_max = -math.inf
        for out_index, key in enumerate(keys):
            group = handle[key]
            amplitude = np.asarray(group["amplitude/raw"], dtype=np.float32)
            phase = np.asarray(group["phase/raw"], dtype=np.float32)
            data[out_index] = amplitude * np.exp(1j * phase).astype(np.complex64)
            amp_min = min(amp_min, float(np.min(amplitude)))
            amp_max = max(amp_max, float(np.max(amplitude)))
            phase_min = min(phase_min, float(np.min(phase)))
            phase_max = max(phase_max, float(np.max(phase)))
        meta.update(
            {
                "amplitude_min": amp_min,
                "amplitude_max": amp_max,
                "phase_min": phase_min,
                "phase_max": phase_max,
            }
        )
    return data, meta


def rotation_illum_dirs(angles: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack(
            [np.sin(angles), np.zeros_like(angles), np.cos(angles)],
            axis=1,
        ).astype(np.float64)
    )


def convert(args: argparse.Namespace) -> tuple[OdtMeasuredData, dict[str, Any]]:
    data, meta = load_qpimage_series(args.series_h5)
    angles = np.loadtxt(args.angles_txt, dtype=np.float64)
    if angles.shape != (data.shape[0],):
        raise ValueError(f"angles shape {angles.shape} does not match data frames {data.shape[0]}")
    wavelength_um = meta["wavelength_m"] * 1e6
    pixel_size_um = meta["pixel_size_m"] * 1e6
    fields: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": "complex_odt",
        "measurement_model": "complex_field",
        "wavelength": float(wavelength_um),
        "k": float(2.0 * math.pi * meta["medium_index"] / wavelength_um),
        "medium_index": float(meta["medium_index"]),
        "units": "um",
        "illum_dirs": rotation_illum_dirs(angles),
        "detector_origin": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "detector_u": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "detector_v": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "detector_pixel_size": np.array([pixel_size_um, pixel_size_um], dtype=np.float64),
        "detector_distance": 1.0,
        "q_layout": "rotational_sinogram",
        "rotation_angles": np.ascontiguousarray(angles),
        "rotation_axis": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "coordinate_convention": "odtbrain_qlsi_rotational_sinogram_image_plane",
        "data": data,
        "mask": np.ones(data.shape, dtype=np.float32),
        "source_name": "ODTbrain HL60 QLSI example",
        "source_package": "ODTbrain 0.4.12 source distribution",
        "source_doi": "10.6084/m9.figshare.8055407.v1",
        "source_license": "CC0",
        "source_series_h5": str(args.series_h5),
        "source_angles_txt": str(args.angles_txt),
    }
    summary = {
        **meta,
        "wavelength_um": wavelength_um,
        "pixel_size_um": pixel_size_um,
        "data_shape": tuple(int(v) for v in data.shape),
        "data_dtype": str(data.dtype),
        "rotation_angle_min": float(np.min(angles)),
        "rotation_angle_max": float(np.max(angles)),
        "field_abs_mean": float(np.mean(np.abs(data))),
        "field_abs_max": float(np.max(np.abs(data))),
    }
    return OdtMeasuredData(fields=fields, source_path=args.out), summary


def write_summary(path: Path, *, contract: OdtMeasuredData, validation: dict[str, Any], source_summary: dict[str, Any]) -> None:
    lines = [
        "# Public ODTbrain HL60 contract conversion",
        "",
        "This file records conversion of the public ODTbrain HL60 QLSI example data into the local measured-data contract.",
        "",
        "## Source",
        "",
        "- source: `ODTbrain 0.4.12` source distribution, `examples/data/qlsi_3d_hl60-cell_A140.tar.lzma`",
        "- original public DOI: `10.6084/m9.figshare.8055407.v1`",
        "- license in bundled readme: `CC0`",
        "- sample: `HL60 S/4 cell`",
        "",
        "## Converted Contract",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    report_summary = validation["summary"]
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
            "This is real public ODT/QPI data, but it is a rotational complex-field sinogram rather than the fixed ring/cap FPDT/FS-ODT geometry used in the current acceleration benchmarks.",
            "It validates public measured-data ingestion. A speed benchmark still needs a rotational-sinogram adapter or a resampling step into a prepared curved-Ewald layout.",
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
    parser = argparse.ArgumentParser(description="Convert ODTbrain HL60 public QLSI example data to the local ODT measured-data contract.")
    base = ROOT / "benchmark_results" / "public_data_probe" / "odtbrain_hl60_extracted"
    parser.add_argument("--series-h5", type=Path, default=base / "series.h5")
    parser.add_argument("--angles-txt", type=Path, default=base / "angles.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark_results" / "odtbrain_hl60_public_contract.npz")
    parser.add_argument("--summary-md", type=Path, default=ROOT / "benchmark_results" / "odtbrain_hl60_public_contract_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract, source_summary = convert(args)
    report = validate_odt_measured_contract(contract)
    report.raise_for_errors()
    save_odt_measured_contract(args.out, contract)
    write_summary(args.summary_md, contract=contract, validation=report.to_dict(), source_summary=source_summary)
    print(json.dumps({"out": str(args.out), "summary_md": str(args.summary_md), "validation": report.to_dict()}, indent=2, default=str))


if __name__ == "__main__":
    main()
