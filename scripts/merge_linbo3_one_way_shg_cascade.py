from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def complex_array(values: list[list[float]]) -> np.ndarray:
    return np.asarray([complex(*value) for value in values], dtype=np.complex128)


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(model - reference) / np.linalg.norm(reference))


def load_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "linbo3-one-way-shg-cascade-v2":
        raise ValueError(f"{path} is not a v2 one-way cascade result")
    if payload.get("reference_only"):
        raise ValueError(f"{path} is reference-only")
    rows = payload.get("rows", [])
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one resolution row")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently completed one-way cascade resolutions.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/linbo3_one_way_shg_cascade_r20_r24.json"),
    )
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise ValueError("at least two inputs are required for a convergence result")
    payloads = [load_payload(path) for path in args.inputs]
    rows = [payload["rows"][0] for payload in payloads]
    order = np.argsort([float(row["resolution"]) for row in rows])
    input_paths = [args.inputs[int(index)] for index in order]
    rows = [rows[int(index)] for index in order]
    payloads = [payloads[int(index)] for index in order]
    resolutions = tuple(float(row["resolution"]) for row in rows)
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("input resolutions must be unique")

    reference_material = payloads[0]["material"]
    reference_configuration = dict(payloads[0]["configuration"])
    reference_configuration.pop("resolutions", None)
    for path, payload in zip(input_paths, payloads, strict=True):
        if payload["material"] != reference_material:
            raise ValueError(f"material mismatch in {path}")
        configuration = dict(payload["configuration"])
        configuration.pop("resolutions", None)
        if configuration != reference_configuration:
            raise ValueError(f"configuration mismatch in {path}")

    finest = rows[-1]
    next_finest = rows[-2]
    calibrated_grid_l2 = relative_l2(
        complex_array(next_finest["inner_calibrated_modal_amplitudes"]),
        complex_array(finest["inner_calibrated_modal_amplitudes"]),
    )
    exact_yee_grid_l2 = relative_l2(
        complex_array(next_finest["exact_yee_direct_modal_amplitudes"]),
        complex_array(finest["exact_yee_direct_modal_amplitudes"]),
    )
    next_fields = np.load(next_finest["field_artifact"]["path"])
    finest_fields = np.load(finest["field_artifact"]["path"])
    cell_center_source_grid_l2 = relative_l2(
        next_fields["cell_center_direct_modal"],
        finest_fields["cell_center_direct_modal"],
    )
    pump_norm_relative_change = abs(
        float(next_finest["pump"]["field_weighted_l2_norm"])
        - float(finest["pump"]["field_weighted_l2_norm"])
    ) / float(finest["pump"]["field_weighted_l2_norm"])
    p2_norm_relative_change = abs(
        float(next_finest["pump"][
            "nonlinear_polarization_weighted_l2_norm_pm_per_v_scale"
        ])
        - float(finest["pump"][
            "nonlinear_polarization_weighted_l2_norm_pm_per_v_scale"
        ])
    ) / float(
        finest["pump"]["nonlinear_polarization_weighted_l2_norm_pm_per_v_scale"]
    )

    gates = {
        "pump_field_finite_nonzero": finest["pump"]["field_weighted_l2_norm"] > 0.0,
        "nonlinear_polarization_finite_nonzero": finest["pump"][
            "nonlinear_polarization_weighted_l2_norm_pm_per_v_scale"
        ]
        > 0.0,
        "cell_center_acfo_vs_direct_l2_le_3pct": finest["cell_center_source"][
            "acfo_vs_direct_relative_l2"
        ]
        <= 0.03,
        "exact_yee_acfo_vs_direct_l2_le_3pct": finest["exact_yee_source"][
            "acfo_vs_direct_relative_l2"
        ]
        <= 0.03,
        "exact_yee_vs_cell_center_direct_l2_le_5pct": finest[
            "exact_yee_source"
        ]["direct_vs_cell_center_direct_relative_l2"]
        <= 0.05,
        "blind_fdtd_vs_exact_yee_direct_l2_le_5pct": finest["shells"][0][
            "fdtd_vs_exact_yee_direct_relative_l2"
        ]
        <= 0.05,
        "blind_fdtd_vs_exact_yee_acfo_l2_le_5pct": finest["shells"][0][
            "fdtd_vs_exact_yee_acfo_relative_l2"
        ]
        <= 0.05,
        "ordinary_l2_le_5pct": finest["shells"][0][
            "branch_fdtd_vs_exact_yee_direct_relative_l2"
        ]["ordinary"]
        <= 0.05,
        "extraordinary_l2_le_5pct": finest["shells"][0][
            "branch_fdtd_vs_exact_yee_direct_relative_l2"
        ]["extraordinary"]
        <= 0.05,
        "monitor_invariance_le_5pct": finest[
            "calibrated_inner_outer_relative_l2"
        ]
        <= 0.05,
        "finest_next_finest_grid_l2_le_5pct": calibrated_grid_l2 <= 0.05,
    }
    configuration = dict(payloads[-1]["configuration"])
    configuration["resolutions"] = resolutions
    provenance = [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "resolution": float(payload["rows"][0]["resolution"]),
        }
        for path, payload in zip(input_paths, payloads, strict=True)
    ]
    result = {
        **{
            key: value
            for key, value in payloads[-1].items()
            if key
            not in {
                "schema",
                "generated_at_utc",
                "configuration",
                "rows",
                "finest_next_finest_grid_relative_l2",
                "gates",
                "passed",
            }
        },
        "schema": "linbo3-one-way-shg-cascade-combined-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "input_provenance": provenance,
        "rows": rows,
        "convergence": {
            "next_finest_resolution": resolutions[-2],
            "finest_resolution": resolutions[-1],
            "calibrated_fdtd_modal_relative_l2": calibrated_grid_l2,
            "exact_yee_source_modal_relative_l2": exact_yee_grid_l2,
            "cell_center_p2_modal_relative_l2": cell_center_source_grid_l2,
            "pump_field_norm_relative_change": pump_norm_relative_change,
            "nonlinear_polarization_norm_relative_change": p2_norm_relative_change,
        },
        "finest_next_finest_grid_relative_l2": calibrated_grid_l2,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "convergence": result["convergence"], "gates": gates, "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
