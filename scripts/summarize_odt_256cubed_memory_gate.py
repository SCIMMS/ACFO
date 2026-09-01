from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the ODT 256-cubed memory-feasibility gate.")
    parser.add_argument(
        "--estimate",
        type=Path,
        default=Path("benchmark_results/odt_256cubed_memory_estimate.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/odt_256cubed_memory_gate_decision.json"),
    )
    args = parser.parse_args()

    estimate = json.loads(args.estimate.read_text(encoding="utf-8"))
    block_one = next(
        item for item in estimate["illumination_block_estimates"]
        if item["illumination_block"] == 1
    )
    basis = float(estimate["gpu_basis_mib"])
    current_forward = float(estimate["current_ring_forward_peak_lower_bound_including_basis_mib"])
    current_adjoint = float(estimate["current_ring_adjoint_peak_lower_bound_including_basis_mib"])
    blocked_forward = basis + float(block_one["ring"]["forward_peak_lower_bound_mib"])
    blocked_adjoint = basis + float(block_one["ring"]["adjoint_peak_lower_bound_mib"])
    gate_mib = float(estimate["gate_24gib"])

    result = {
        "schema": "odt-256cubed-memory-gate-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem": estimate["problem"],
        "dimension_derivation": estimate["dimension_derivation"],
        "metrics": {
            "gpu_basis_mib": basis,
            "unchunked_forward_live_array_lower_bound_mib": current_forward,
            "unchunked_adjoint_live_array_lower_bound_mib": current_adjoint,
            "illumination_block_1_forward_live_array_lower_bound_mib": blocked_forward,
            "illumination_block_1_adjoint_live_array_lower_bound_mib": blocked_adjoint,
            "native_prepared_adjoint_failed_allocation_shape": [73, 256, 256, 256],
            "native_prepared_adjoint_failed_allocation_dtype": "complex128",
            "native_prepared_adjoint_failed_allocation_mib": 73 * 256**3 * 16 / 1024**2,
            "available_gpu_name": "NVIDIA GeForce RTX 2070 SUPER",
            "available_gpu_total_mib": 8192,
            "available_gpu_free_mib_at_audit": 7519,
        },
        "gates": {
            "24gib_live_array_feasibility": {
                "threshold_mib": gate_mib,
                "pass": current_forward <= gate_mib and current_adjoint <= gate_mib,
                "scope": "analytic live-array lower bound; not a measured process peak",
            },
            "local_8gib_safe_execution": {
                "pass": False,
                "reason": "The unchunked lower bound leaves insufficient allocator/workspace headroom on an 8 GiB GPU.",
            },
            "current_native_context_build": {
                "pass": False,
                "reason": "CPU context preparation materializes an 18.25 GiB complex128 adjoint table before forward execution.",
            },
            "production_100_forward_adjoint": {
                "pass": False,
                "reason": "Not executed; illumination streaming and prepared-table-free construction are required first.",
            },
        },
        "decision": {
            "memory_feasibility_24gib": "PASS_LOWER_BOUND_ONLY",
            "step3_production_gate": "PENDING_BACKEND_STREAMING",
            "odt_manuscript_scope": "Keep ODT as a scoped numerical extension until a measured 256-cubed run passes memory and time-to-solution gates.",
        },
        "required_next_actions": [
            "Construct the GPU plan without the native complex128 prepared-adjoint table.",
            "Stream the illumination dimension; block size 1 has estimated forward/adjoint lower bounds near 2.73/2.61 GiB including basis.",
            "Measure actual CUDA peak memory, setup, first pair, cached pair, and 100 forward-adjoint pairs.",
            "Compare total time-to-solution against the tuned production baseline and require at least 3x for the minimum performance gate.",
        ],
        "limitations": [
            "The estimator sums known live arrays but does not include all CUDA library workspaces, allocator fragmentation, or transient einsum buffers.",
            "Illumination blocking does not reduce the approximately 2.32 GiB rzhl source/mixed tensor; L or RZ streaming may still be required.",
            "No 256-cubed numerical-accuracy, adjoint-dot, inverse, or timing result is claimed here.",
        ],
        "sources": [str(args.estimate)],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                "# ODT 256-cubed memory gate decision",
                "",
                f"- 24 GiB live-array feasibility: **{'PASS' if result['gates']['24gib_live_array_feasibility']['pass'] else 'FAIL'} (lower bound only)**",
                f"- unchunked forward / adjoint lower bound: `{current_forward:.1f}` / `{current_adjoint:.1f} MiB`",
                f"- illumination-block=1 forward / adjoint lower bound: `{blocked_forward:.1f}` / `{blocked_adjoint:.1f} MiB`",
                "- current native context build: **FAIL** (`18.25 GiB` complex128 prepared-adjoint allocation)",
                "- 100 forward-adjoint production gate: **PENDING**",
                "- decision: implement prepared-table-free construction and illumination streaming before a measured run",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {md}")


if __name__ == "__main__":
    main()
