from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    resolve_device,
    synchronize,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)
from benchmark_odt_selected_z_gpu import centered_z_indices  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
)
from benchmark_odt_virtual_polar_reconstruction import (  # noqa: E402
    parser as chained_parser,
    selected_data_lipschitz,
    selected_truth,
    solve_selected_fista,
)


GEOMETRIES = (
    ("rho256", 256, "uniform_rho", 2.0),
    ("rho192", 192, "uniform_rho", 2.0),
    ("theta192", 192, "uniform_theta", 2.0),
    ("outer192_p2", 192, "outer_power", 2.0),
    ("theta160", 160, "uniform_theta", 2.0),
    ("outer160_p2", 160, "outer_power", 2.0),
)


def parse_geometries(value: str) -> list[tuple[str, int, str, float]]:
    requested = {item.strip() for item in value.split(",") if item.strip()}
    known = {row[0]: row for row in GEOMETRIES}
    unknown = requested - set(known)
    if unknown:
        raise ValueError(f"unknown geometries: {sorted(unknown)}")
    return [row for row in GEOMETRIES if row[0] in requested]


def run_case(
    *,
    torch: Any,
    device: Any,
    base_args: argparse.Namespace,
    geometry: tuple[str, int, str, float],
    selected_counts: list[int],
    iterations: int,
    power_iterations: int,
    record_every: int,
) -> dict[str, Any]:
    label, cap_radial, sampling, outer_power = geometry
    args = argparse.Namespace(**vars(base_args))
    args.cap_radial = cap_radial
    args.detector_radial_sampling = sampling
    args.detector_radial_outer_power = outer_power
    args.detector_radial_min_fraction = 0.0
    args.detector_radial_max_fraction = 1.0

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    context_start = time.perf_counter()
    context = build_composite_context(args)
    context_s = time.perf_counter() - context_start
    plan_start = time.perf_counter()
    plan = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=True,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        forward_mode="illumination-reduced",
        adjoint_mode="illumination-reduced",
    )
    synchronize(torch, device)
    plan_s = time.perf_counter() - plan_start

    selected_rows: list[dict[str, Any]] = []
    for selected_n_z in selected_counts:
        z_indices = centered_z_indices(args.n_z, selected_n_z)
        z_index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
        truth = selected_truth(torch, plan, context, z_indices, dtype=args.dtype)
        with torch.inference_mode():
            data = plan.forward_selected_z(truth, z_index)
        synchronize(torch, device)
        lipschitz, power_s = selected_data_lipschitz(
            torch=torch,
            plan=plan,
            z_index=z_index,
            selected_n_z=selected_n_z,
            iterations=power_iterations,
            seed=args.seed + 2003 + selected_n_z + cap_radial,
        )
        solve, reconstruction, history = solve_selected_fista(
            torch=torch,
            plan=plan,
            z_index=z_index,
            data=data,
            truth=truth,
            label=label,
            data_lipschitz=lipschitz,
            iterations=iterations,
            record_every=record_every,
        )
        selected_rows.append(
            {
                "selected_n_z": selected_n_z,
                "selected_object_bins": int(truth.numel()),
                "q_samples": int(data.numel()),
                "data_lipschitz_estimate": float(lipschitz),
                "power_iteration_s": float(power_s),
                "solve": solve,
                "history": history,
            }
        )
        del z_index, truth, data, reconstruction
        gc.collect()
        torch.cuda.empty_cache()

    row = {
        "label": label,
        "cap_radial": int(cap_radial),
        "cap_phi": int(args.cap_phi),
        "radial_sampling": sampling,
        "radial_outer_power": float(outer_power),
        "samples_per_view": int(cap_radial * args.cap_phi),
        "total_views": int(args.ring_illum)
        + (0 if args.skip_axis_illumination else 1),
        "context_build_s": float(context_s),
        "plan_build_s": float(plan_s),
        "ring_h_cutoff": int(context.ring.axis_h_cutoff),
        "ring_h_modes": int(context.ring.plan.h_values.size),
        "ring_l_cutoff": int(context.ring.l_cutoff),
        "selected": selected_rows,
        "pytorch_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
    }
    del plan, context
    gc.collect()
    torch.cuda.empty_cache()
    return row


def add_baseline_comparisons(cases: list[dict[str, Any]]) -> None:
    baseline = next(row for row in cases if row["label"] == "rho256")
    baseline_by_n = {
        row["selected_n_z"]: row for row in baseline["selected"]
    }
    baseline_samples = float(baseline["samples_per_view"])
    for case in cases:
        case["sample_reduction_fraction_vs_rho256"] = 1.0 - float(
            case["samples_per_view"]
        ) / baseline_samples
        for selected in case["selected"]:
            reference = baseline_by_n[selected["selected_n_z"]]
            solve = selected["solve"]
            ref_solve = reference["solve"]
            selected["object_error_delta_vs_rho256"] = float(
                solve["object_rel_l2"] - ref_solve["object_rel_l2"]
            )
            selected["core_iteration_speedup_vs_rho256"] = float(
                ref_solve["core_iteration_timing"]["median_s"]
                / solve["core_iteration_timing"]["median_s"]
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    geometries = parse_geometries(args.geometries)
    if not any(row[0] == "rho256" for row in geometries):
        raise ValueError("rho256 baseline must be included")
    selected_counts = [int(value) for value in args.selected_slices.split(",")]
    cases = [
        run_case(
            torch=torch,
            device=device,
            base_args=args,
            geometry=geometry,
            selected_counts=selected_counts,
            iterations=args.iterations,
            power_iterations=args.power_iterations,
            record_every=args.record_every,
        )
        for geometry in geometries
    ]
    add_baseline_comparisons(cases)
    result = {
        "schema": "odt-radial-detector-geometry-sweep-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_name": device_name(torch, device),
        "dtype": args.dtype,
        "problem": {
            "full_object_shape": [args.n_r, args.n_z, args.n_beta],
            "selected_slices": selected_counts,
            "ring_illumination_count": int(args.ring_illum),
            "total_views": int(args.ring_illum)
            + (0 if args.skip_axis_illumination else 1),
            "cap_phi": int(args.cap_phi),
            "iterations": int(args.iterations),
            "power_iterations": int(args.power_iterations),
            "noise": "none",
        },
        "cases": cases,
        "claim_boundary": [
            "All cases use ideal geometry-matched polar data and the same selected-z nonnegative FISTA solver.",
            "This screening compares fixed iteration count, not matched convergence tolerance.",
            "The timings are processing-side GPU-resident iteration timings, not end-to-end microscope frame rates.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    p = chained_parser()
    p.description = "Sweep geometry-aware radial detector nodes for selected-z ODT."
    p.add_argument(
        "--geometries",
        default=",".join(row[0] for row in GEOMETRIES),
    )
    p.set_defaults(
        iterations=60,
        power_iterations=6,
        record_every=10,
        selected_slices="1,8",
        output=ROOT / "benchmark_results" / "odt_radial_geometry_sweep.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    compact = []
    for case in result["cases"]:
        compact.append(
            {
                "label": case["label"],
                "samples_per_view": case["samples_per_view"],
                "sample_reduction_fraction": case[
                    "sample_reduction_fraction_vs_rho256"
                ],
                "selected": [
                    {
                        "n": row["selected_n_z"],
                        "object_rel_l2": row["solve"]["object_rel_l2"],
                        "data_residual": row["solve"]["data_residual"],
                        "median_iter_s": row["solve"]["core_iteration_timing"][
                            "median_s"
                        ],
                        "speedup": row["core_iteration_speedup_vs_rho256"],
                        "object_error_delta": row[
                            "object_error_delta_vs_rho256"
                        ],
                    }
                    for row in case["selected"]
                ],
            }
        )
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
