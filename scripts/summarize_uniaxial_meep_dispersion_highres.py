from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine the 24/32/40 PyMeep dispersion probes.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_dispersion_reduced_probe.json"),
    )
    parser.add_argument(
        "--resolution32",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_dispersion_resolution32_probe.json"),
    )
    parser.add_argument(
        "--resolution40",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_dispersion_resolution40_probe.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/uniaxial_meep_dispersion_highres_decision.json"),
    )
    args = parser.parse_args()

    base = load(args.base)
    r32 = load(args.resolution32)
    r40 = load(args.resolution40)
    row24 = next(row for row in base["rows"] if float(row["resolution"]) == 24.0)
    rows = [row24, r32["rows"][0], r40["rows"][0]]
    peaks32 = np.asarray(rows[1]["measured_peak_radius"], dtype=np.float64)
    peaks40 = np.asarray(rows[2]["measured_peak_radius"], dtype=np.float64)
    grid_difference = float(np.linalg.norm(peaks40 - peaks32) / np.linalg.norm(peaks40))
    finest = rows[-1]
    gates = {
        "correct_ellipse_relative_l2_le_2pct": finest["correct_relative_l2"] <= 0.02,
        "forced_sphere_error_ratio_ge_5": finest["forced_to_correct_error_ratio"] >= 5.0,
        "correct_ridge_energy_ge_sphere": finest["correct_to_sphere_ridge_energy_ratio"] >= 1.0,
        "curvature_separation_ge_2_bins": finest["maximum_correct_sphere_separation_bins"] >= 2.0,
        "three_level_peak_grid_convergence_le_2pct": grid_difference <= 0.02,
    }
    result = {
        "schema": "uniaxial-meep-dispersion-highres-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "2-D homogeneous uniaxial DFT-field spatial-spectrum stop-rule probe; not the 3-D publication full-wave gate",
        "inputs": [str(args.base), str(args.resolution32), str(args.resolution40)],
        "material": base["material"],
        "cell": base["cell"],
        "resolutions": [24.0, 32.0, 40.0],
        "runtime_seconds": [
            base["runtime_seconds"][2],
            r32["runtime_seconds"][0],
            r40["runtime_seconds"][0],
        ],
        "rows": rows,
        "finest_next_finest_peak_l2": grid_difference,
        "gates": gates,
        "passed": all(gates.values()),
        "environment": r40["environment"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                "# Uniaxial PyMeep dispersion high-resolution decision",
                "",
                f"- overall: **{'PASS' if result['passed'] else 'FAIL'}**",
                f"- resolutions: `{result['resolutions']}`",
                f"- correct ellipse L2: `{finest['correct_relative_l2']:.3e}`",
                f"- forced sphere error ratio: `{finest['forced_to_correct_error_ratio']:.3f}`",
                f"- correct/sphere ridge energy: `{finest['correct_to_sphere_ridge_energy_ratio']:.3f}`",
                f"- curvature separation: `{finest['maximum_correct_sphere_separation_bins']:.3f}` bins",
                f"- 32-to-40 grid peak L2: `{grid_difference:.3e}`",
                "",
                "This is a reduced 2-D stop-rule PASS, not the final 3-D publication full-wave gate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {md}")


if __name__ == "__main__":
    main()
