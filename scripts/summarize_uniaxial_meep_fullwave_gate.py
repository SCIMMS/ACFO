from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import summarize_fullwave_arrays  # noqa: E402


REQUIRED = (
    "angles_deg",
    "contrast_scales",
    "resolutions",
    "background_born_field",
    "background_fullwave_field",
    "acfo_field",
    "direct_born_field",
    "fullwave_field",
    "forced_sphere_field",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the independent Meep Maxwell/FDTD gate.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/uniaxial_meep_fullwave_gate_decision.json"))
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED if name not in archive]
        if missing:
            raise KeyError(f"missing NPZ arrays: {', '.join(missing)}")
        summary = summarize_fullwave_arrays(**{name: archive[name] for name in REQUIRED})
        metadata = json.loads(str(archive["metadata_json"].item())) if "metadata_json" in archive else {}
    result = {
        "schema": "uniaxial-meep-fullwave-gate-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "metadata": metadata,
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    nominal = max(result["per_contrast"], key=lambda row: row["contrast_scale"])
    weakest = min(result["per_contrast"], key=lambda row: row["contrast_scale"])
    md.write_text(
        "\n".join(
            [
                "# Uniaxial Meep full-wave gate decision",
                "",
                f"- overall: **{'PASS' if result['passed'] else 'FAIL'}**",
                f"- weakest-contrast complex L2: `{weakest['fullwave_vs_acfo_complex_l2']:.3e}`",
                f"- convergence slope: `{result['convergence']['fullwave_born_loglog_slope']:.3f}`",
                f"- nominal intensity NCC: `{nominal['intensity_ncc']:.6f}`",
                f"- nominal peak error: `{nominal['peak_angle_error_deg']:.4f} deg`",
                f"- forced-sphere residual ratio: `{nominal['forced_sphere_wrong_to_correct_residual_ratio']:.3f}`",
                f"- grid convergence L2: `{result['convergence']['nominal_next_to_finest_grid_complex_l2']:.3e}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {md}")


if __name__ == "__main__":
    main()
