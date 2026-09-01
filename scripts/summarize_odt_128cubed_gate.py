from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the ODT 128-cubed exact/adjoint/inverse gate.")
    parser.add_argument("full_dot", type=Path)
    parser.add_argument("independent_subset", type=Path)
    parser.add_argument("inverse_control", type=Path)
    parser.add_argument("beads_inverse", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/odt_128cubed_gate_decision.json"))
    args = parser.parse_args()

    full_dot = load(args.full_dot)
    full_dot_summary = full_dot.get("summary", full_dot)
    subset = load(args.independent_subset)
    inverse = load(args.inverse_control)
    beads = load(args.beads_inverse)
    scale_pass = bool(
        subset["problem"]["object_shape"] == [128, 128, 128]
        and subset["problem"]["illumination_count"] >= 61
        and subset["problem"]["detector_shape"] == [128, 128]
    )
    forward_pass = bool(subset["metrics"]["forward_complex_l2_vs_direct"] <= 1e-6)
    adjoint_subset_pass = bool(subset["metrics"]["adjoint_selected_object_l2_vs_direct"] <= 1e-9)
    dot_pass = bool(full_dot_summary["forward_adjoint_dot_error_complex128_accum"] <= 1e-9)
    inverse_pass = bool(inverse["final"]["object_nrmse"] <= 0.01)
    gates = {
        "scale_pass": scale_pass,
        "forward_complex_l2_pass": forward_pass,
        "adjoint_independent_subset_pass": adjoint_subset_pass,
        "full_forward_adjoint_dot_pass": dot_pass,
        "noiseless_in_range_inverse_pass": inverse_pass,
    }
    minimum_pass = all(gates.values())
    result = {
        "schema": "acfo-odt-128cubed-gate-v1",
        "sources": {
            "full_dot": args.full_dot.as_posix(),
            "independent_subset": args.independent_subset.as_posix(),
            "inverse_control": args.inverse_control.as_posix(),
            "beads_inverse": args.beads_inverse.as_posix(),
        },
        "problem": subset["problem"],
        "metrics": {
            "forward_complex_l2_vs_direct_subset": subset["metrics"]["forward_complex_l2_vs_direct"],
            "adjoint_selected_object_l2_vs_direct_subset": subset["metrics"]["adjoint_selected_object_l2_vs_direct"],
            "independent_subset_dot_error": subset["metrics"]["subset_dot_error"],
            "full_forward_adjoint_dot_error": full_dot_summary["forward_adjoint_dot_error_complex128_accum"],
            "in_range_inverse_nrmse": inverse["final"]["object_nrmse"],
            "in_range_inverse_data_residual": inverse["final"]["data_residual"],
            "in_range_inverse_converged_iteration": inverse["converged_iteration"],
            "beads_inverse_nrmse_after_100_cg": beads["final"]["object_nrmse"],
            "beads_data_residual_after_100_cg": beads["final"]["data_residual"],
            "complex128_low_memory_gpu_peak_mib": full_dot_summary["gpu_peak_allocated_mib"],
        },
        "gates": gates,
        "minimum_odt_128cubed_numerical_gate_pass": minimum_pass,
        "pass_scope": "in-range real-object numerical inverse control plus independent sparse-support direct subset",
        "limitations": [
            "The independent Cartesian exponent reference covers 2,048 active object bins and 2,048 detector nodes while the structured forward and adjoint execute at full 128-cubed scale.",
            "The <=1% inverse result uses a seeded real object generated in the adjoint range; it proves numerical inverse readiness on observable components.",
            "The unconstrained real beads phantom remains at 18.6% NRMSE after 100 CG iterations despite a 0.174% data residual, exposing missing-cone/null-space ambiguity.",
            "This gate does not prove physical-phantom recovery, 30 dB robustness, 256-cubed production performance, or independent full-wave accuracy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    m = result["metrics"]
    lines = [
        "# ODT 128-cubed numerical gate decision",
        "",
        f"Decision: `{'PASS' if minimum_pass else 'FAIL'}` within scope `{result['pass_scope']}`.",
        "",
        "| check | value | gate | result |",
        "|---|---:|---:|---|",
        f"| scale | 128^3, 61 illum, 128x128 detector | required | {'PASS' if scale_pass else 'FAIL'} |",
        f"| forward direct-subset L2 | {m['forward_complex_l2_vs_direct_subset']:.3e} | <=1e-6 | {'PASS' if forward_pass else 'FAIL'} |",
        f"| adjoint direct-subset L2 | {m['adjoint_selected_object_l2_vs_direct_subset']:.3e} | <=1e-9 | {'PASS' if adjoint_subset_pass else 'FAIL'} |",
        f"| full dot error | {m['full_forward_adjoint_dot_error']:.3e} | <=1e-9 | {'PASS' if dot_pass else 'FAIL'} |",
        f"| in-range inverse NRMSE | {m['in_range_inverse_nrmse']:.3%} | <=1% | {'PASS' if inverse_pass else 'FAIL'} |",
        "",
        f"Beads limitation: NRMSE `{m['beads_inverse_nrmse_after_100_cg']:.3%}` at data residual `{m['beads_data_residual_after_100_cg']:.3%}`.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
