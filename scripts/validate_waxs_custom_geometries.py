from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "benchmark_protein_nanocrystal_finufft_fair.py"
DEFAULT_STRUCTURE = (
    ROOT
    / "structures"
    / "processed"
    / "protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz"
)

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "close_wide_q",
        "label": "custom close chamber, wide-q partial arcs",
        "qmin": 0.05,
        "qmax": 8.0,
        "nq": 24,
        "wavelength_nm": 0.08,
        "active_width_mm": 240.0,
        "active_height_mm": 180.0,
        "distance_mm": 80.0,
    },
    {
        "name": "long_distance_narrow_q",
        "label": "custom long-distance chamber, narrow-q partial arcs",
        "qmin": 0.05,
        "qmax": 1.2,
        "nq": 24,
        "wavelength_nm": 0.10,
        "active_width_mm": 80.0,
        "active_height_mm": 60.0,
        "distance_mm": 250.0,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_path(case: dict[str, Any]) -> Path:
    return ROOT / "benchmark_results" / f"waxs_custom_geometry_{case['name']}_20260722.json"


def receipt_matches_case(receipt: dict[str, Any], case: dict[str, Any]) -> bool:
    return (
        receipt.get("schema") == "protein-nanocrystal-finufft-fair-v2"
        and receipt.get("source_mode") == "same_binned"
        and receipt.get("qmin") == case["qmin"]
        and receipt.get("qmax") == case["qmax"]
        and receipt.get("nq") == case["nq"]
        and receipt.get("wavelength_nm") == case["wavelength_nm"]
        and receipt.get("detector_active_width_mm") == case["active_width_mm"]
        and receipt.get("detector_active_height_mm") == case["active_height_mm"]
        and receipt.get("detector_distance_mm") == case["distance_mm"]
        and receipt.get("harmonic_margin") == 32
        and receipt.get("r_dependent_margin") == 32
        and receipt.get("finufft_q_block_size") == 2
    )


def build_command(
    case: dict[str, Any],
    *,
    structure: Path,
    finufft_threads: int,
) -> list[str]:
    return [
        sys.executable,
        str(BENCHMARK),
        str(structure),
        "--source-mode",
        "same_binned",
        "--qmin",
        str(case["qmin"]),
        "--qmax",
        str(case["qmax"]),
        "--q-unit",
        "inv_angstrom",
        "--nq",
        str(case["nq"]),
        "--wavelength-nm",
        str(case["wavelength_nm"]),
        "--bin-width-nm",
        "0.1",
        "--nphi-min",
        "256",
        "--harmonic-margin",
        "32",
        "--r-dependent-margin",
        "32",
        "--cutoff-bin-size",
        "16",
        "--acfo-q-block-size",
        "2",
        "--profile-chunk-size",
        "8",
        "--form-factor-model",
        "xray_f0",
        "--finufft-eps",
        "1e-6",
        "--finufft-threads",
        str(finufft_threads),
        "--detector-label",
        str(case["label"]),
        "--detector-active-width-mm",
        str(case["active_width_mm"]),
        "--detector-active-height-mm",
        str(case["active_height_mm"]),
        "--detector-distance-mm",
        str(case["distance_mm"]),
        "--finufft-q-block-size",
        "2",
        "--warmups",
        "0",
        "--repeats",
        "0",
        "--output",
        str(receipt_path(case)),
    ]


def summarize_case(case: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": case["name"],
        "label": case["label"],
        "qmin_inv_angstrom": receipt["qmin"],
        "qmax_inv_angstrom": receipt["qmax"],
        "nq": receipt["nq"],
        "nphi": receipt["n_phi"],
        "wavelength_nm": receipt["wavelength_nm"],
        "active_width_mm": receipt["detector_active_width_mm"],
        "active_height_mm": receipt["detector_active_height_mm"],
        "distance_mm": receipt["detector_distance_mm"],
        "full_targets": receipt["targets"],
        "active_targets": receipt["active_detector_targets"],
        "active_fraction": receipt["active_detector_fraction"],
        "outer_ring_active_fraction": receipt["active_fraction_at_qmax"],
        "complex_l2": receipt["complex_l2_acfo_vs_finufft"],
        "intensity_l2": receipt["intensity_l2_acfo_vs_finufft"],
        "ring_l2": receipt["ring_l2_acfo_vs_finufft"],
        "all_finite": receipt["all_finite"],
        "finufft_plan_mode": receipt["finufft_plan_mode"],
        "receipt_path": receipt_path(case).relative_to(ROOT).as_posix(),
        "receipt_sha256": sha256_file(receipt_path(case)),
    }


def build_gates(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "two_custom_geometries": len(rows) == 2,
        "all_finite": all(row["all_finite"] for row in rows),
        "complex_l2_le_2e_6": all(row["complex_l2"] <= 2e-6 for row in rows),
        "intensity_l2_le_5e_6": all(row["intensity_l2"] <= 5e-6 for row in rows),
        "ring_l2_le_5e_6": all(row["ring_l2"] <= 5e-6 for row in rows),
        "partial_active_masks": all(
            0.0 < row["active_fraction"] < 1.0
            and 0.0 < row["outer_ring_active_fraction"] < 1.0
            for row in rows
        ),
        "wide_vs_narrow_q_separation_ge_5x": (
            max(row["qmax_inv_angstrom"] for row in rows)
            / min(row["qmax_inv_angstrom"] for row in rows)
            >= 5.0
        ),
        "distance_separation_ge_3x": (
            max(row["distance_mm"] for row in rows)
            / min(row["distance_mm"] for row in rows)
            >= 3.0
        ),
        "accuracy_reference_policy_frozen": all(
            row["finufft_plan_mode"] == "memory_safe_q_blocked_setup_per_evaluation"
            for row in rows
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "qmin_inv_angstrom",
        "qmax_inv_angstrom",
        "nq",
        "nphi",
        "wavelength_nm",
        "active_width_mm",
        "active_height_mm",
        "distance_mm",
        "active_targets",
        "active_fraction",
        "outer_ring_active_fraction",
        "complex_l2",
        "intensity_l2",
        "ring_l2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# WAXS custom detector/chamber geometry correctness validation",
        "",
        "이 검증은 detector/chamber envelope가 달라져도 active detector node에서 같은 ACFO/FINUFFT forward amplitude를 얻는지 확인한다. 시간 측정값은 accuracy probe의 부산물이며 성능 주장에 사용하지 않는다.",
        "",
        "| geometry | q range (Å⁻¹) | distance (mm) | active targets | active fraction | outer-ring fraction | complex L2 | intensity L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['name']} | {row['qmin_inv_angstrom']:.2f}–{row['qmax_inv_angstrom']:.2f} "
            f"| {row['distance_mm']:.0f} | {row['active_targets']:,} "
            f"| {row['active_fraction']:.3f} | {row['outer_ring_active_fraction']:.3f} "
            f"| {row['complex_l2']:.3e} | {row['intensity_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "지원되는 주장은 tested rectangular active-region geometries에서의 forward correctness이다. 임의 detector 성능, end-to-end detector calibration, module gap/tilt/parallax, 또는 특정 XFEL chamber 전체의 성능을 일반화하지 않는다.",
            "",
            "FINUFFT는 이 accuracy probe에서 q-block마다 plan을 재구축한다. 따라서 여기의 실행 시간은 hot-reuse 성능 비교가 아니며, WAXS 성능 수치는 별도의 AB/BA receipt를 사용한다.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_waxs_custom_geometries.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--finufft-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "benchmark_results" / "waxs_custom_geometry_validation_20260722.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=ROOT / "benchmark_results" / "waxs_custom_geometry_validation_20260722.csv",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "waxs_custom_geometry_validation_20260722_ko.md",
    )
    args = parser.parse_args()
    if args.finufft_threads <= 0:
        raise ValueError("finufft-threads must be positive")
    if not args.structure.exists():
        raise FileNotFoundError(args.structure)

    rows: list[dict[str, Any]] = []
    for case in CASES:
        output = receipt_path(case)
        receipt = None
        if args.resume and output.exists():
            candidate = json.loads(output.read_text(encoding="utf-8"))
            if receipt_matches_case(candidate, case):
                receipt = candidate
                print(f"reuse {output.relative_to(ROOT)}", flush=True)
        if receipt is None:
            command = build_command(
                case,
                structure=args.structure,
                finufft_threads=args.finufft_threads,
            )
            print(f"run {case['name']}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            receipt = json.loads(output.read_text(encoding="utf-8"))
        if not receipt_matches_case(receipt, case):
            raise RuntimeError(f"receipt contract mismatch for {case['name']}")
        rows.append(summarize_case(case, receipt))

    gates = build_gates(rows)
    payload = {
        "schema": "waxs-custom-geometry-validation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "forward-amplitude correctness on two custom rectangular detector/chamber envelopes",
        "accuracy_only": True,
        "structure_path": args.structure.relative_to(ROOT).as_posix(),
        "structure_sha256": sha256_file(args.structure),
        "rows": rows,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": {
            "supported": "tested active-node geometry correctness across wide-q close and narrow-q long-distance envelopes",
            "not_supported": "universal detector performance, module gaps/tilts/parallax, calibration, or end-to-end XFEL acquisition",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "finufft_threads": args.finufft_threads,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(args.csv_output, rows)
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "gates": gates, "rows": rows}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
