from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def calibrated_row(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = dict(data["rows"][0])
    extraordinary = np.asarray(row["extraordinary_measured_phase_slope"], dtype=np.float64)
    ordinary = np.asarray(row["ordinary_measured_phase_slope"], dtype=np.float64)
    ellipse = np.asarray(row["ellipse_phase_slope"], dtype=np.float64)
    sphere = np.asarray(row["sphere_phase_slope"], dtype=np.float64)
    gain = float(np.dot(ordinary, sphere) / np.dot(ordinary, ordinary))
    calibrated_extraordinary = gain * extraordinary
    calibrated_ordinary = gain * ordinary
    ellipse_error = float(np.linalg.norm(calibrated_extraordinary - ellipse) / np.linalg.norm(ellipse))
    sphere_error = float(np.linalg.norm(calibrated_extraordinary - sphere) / np.linalg.norm(ellipse))
    ordinary_error = float(np.linalg.norm(calibrated_ordinary - sphere) / np.linalg.norm(sphere))
    row.update(
        {
            "ordinary_complex_gain": gain,
            "calibrated_extraordinary_phase_slope": calibrated_extraordinary.tolist(),
            "calibrated_extraordinary_ellipse_relative_l2": ellipse_error,
            "calibrated_extraordinary_sphere_relative_l2": sphere_error,
            "calibrated_sphere_to_ellipse_error_ratio": sphere_error / ellipse_error,
            "calibrated_ordinary_sphere_relative_l2": ordinary_error,
        }
    )
    return data, row


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine the 20/24/28 3-D PyMeep phase gate.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[
            Path("benchmark_results/uniaxial_meep_3d_phase_farfield_r20.json"),
            Path("benchmark_results/uniaxial_meep_3d_phase_farfield_r24.json"),
            Path("benchmark_results/uniaxial_meep_3d_phase_farfield_r28.json"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_3d_phase_gate_decision.json"),
    )
    parser.add_argument(
        "--time-sensitivity",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_3d_phase_time12_r20.json"),
    )
    parser.add_argument(
        "--boundary-sensitivity",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_3d_phase_boundary_r20.json"),
    )
    args = parser.parse_args()
    if len(args.inputs) != 3:
        raise ValueError("exactly three resolution inputs are required")
    loaded = [calibrated_row(path) for path in args.inputs]
    loaded.sort(key=lambda item: float(item[1]["resolution"]))
    data_rows = [item[0] for item in loaded]
    rows = [item[1] for item in loaded]
    finest = rows[-1]
    next_finest = rows[-2]
    fine = np.asarray(finest["calibrated_extraordinary_phase_slope"], dtype=np.float64)
    coarse = np.asarray(next_finest["calibrated_extraordinary_phase_slope"], dtype=np.float64)
    grid_l2 = float(np.linalg.norm(fine - coarse) / np.linalg.norm(fine))
    _, time_row = calibrated_row(args.time_sensitivity)
    _, boundary_row = calibrated_row(args.boundary_sensitivity)
    baseline = np.asarray(rows[0]["calibrated_extraordinary_phase_slope"], dtype=np.float64)
    time_field = np.asarray(time_row["calibrated_extraordinary_phase_slope"], dtype=np.float64)
    boundary_field = np.asarray(boundary_row["calibrated_extraordinary_phase_slope"], dtype=np.float64)
    time_l2 = float(np.linalg.norm(time_field - baseline) / np.linalg.norm(baseline))
    boundary_l2 = float(np.linalg.norm(boundary_field - baseline) / np.linalg.norm(baseline))
    gates = {
        "calibrated_extraordinary_ellipse_l2_le_2pct": finest["calibrated_extraordinary_ellipse_relative_l2"] <= 0.02,
        "calibrated_forced_sphere_ratio_ge_5": finest["calibrated_sphere_to_ellipse_error_ratio"] >= 5.0,
        "calibrated_ordinary_sphere_l2_le_2pct": finest["calibrated_ordinary_sphere_relative_l2"] <= 0.02,
        "three_level_grid_l2_le_2pct": grid_l2 <= 0.02,
        "time_sensitivity_l2_le_1pct": time_l2 <= 0.01,
        "boundary_sensitivity_l2_le_1pct": boundary_l2 <= 0.01,
    }
    result = {
        "schema": "uniaxial-meep-3d-phase-gate-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "3-D actual-uniaxial Maxwell phase-slope gate with fixed 64-cubed impressed source, including grid, time, and boundary sensitivities",
        "inputs": [str(path) for path in args.inputs]
        + [str(args.time_sensitivity), str(args.boundary_sensitivity)],
        "material": data_rows[-1]["material"],
        "configuration": data_rows[-1]["configuration"],
        "resolutions": [float(row["resolution"]) for row in rows],
        "runtime_seconds": [float(data["runtime_seconds"][0]) for data in data_rows],
        "rows": rows,
        "finest_next_finest_calibrated_grid_l2": grid_l2,
        "time_sensitivity": {
            "baseline_until_after_sources": data_rows[0]["configuration"]["until_after_sources"],
            "comparison_until_after_sources": 12.0,
            "calibrated_phase_l2": time_l2,
        },
        "boundary_sensitivity": {
            "baseline_cell_pml": [4.0, 0.5],
            "comparison_cell_pml": [4.8, 0.7],
            "calibrated_phase_l2": boundary_l2,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "environment": data_rows[-1]["environment"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                "# Uniaxial PyMeep 3-D phase-slope gate",
                "",
                f"- overall: **{'PASS' if result['passed'] else 'FAIL'}**",
                f"- resolutions: `{result['resolutions']}`",
                f"- calibrated ellipse L2: `{finest['calibrated_extraordinary_ellipse_relative_l2']:.3e}`",
                f"- calibrated forced-sphere ratio: `{finest['calibrated_sphere_to_ellipse_error_ratio']:.3f}`",
                f"- calibrated ordinary control L2: `{finest['calibrated_ordinary_sphere_relative_l2']:.3e}`",
                f"- 24-to-28 calibrated grid L2: `{grid_l2:.3e}`",
                f"- time sensitivity L2: `{time_l2:.3e}`",
                f"- boundary sensitivity L2: `{boundary_l2:.3e}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {md}")


if __name__ == "__main__":
    main()
