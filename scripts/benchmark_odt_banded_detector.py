from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
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


@dataclass
class TorchBandedCompositePlan:
    plans: list[TorchCompositeOdtPlan]

    def __post_init__(self) -> None:
        if not self.plans:
            raise ValueError("at least one detector-band plan is required")
        first = self.plans[0]
        self.torch = first.torch
        self.device = first.device
        self.complex_dtype = first.complex_dtype
        self.real_dtype = first.real_dtype
        self.ring = first.ring
        if any(plan.device != self.device for plan in self.plans):
            raise ValueError("all detector bands must use the same device")

    @property
    def q_count(self) -> int:
        return int(sum(plan.q_count for plan in self.plans))

    @property
    def mode_q_count(self) -> int:
        return int(sum(plan.mode_q_count for plan in self.plans))

    @property
    def basis_mib(self) -> float:
        return float(sum(plan.basis_mib for plan in self.plans))

    def forward_selected_z(self, coeff: Any, z_indices: Any) -> Any:
        return self.torch.cat(
            [plan.forward_selected_z(coeff, z_indices) for plan in self.plans],
            dim=0,
        )

    def forward_selected_z_modes(self, coeff: Any, z_indices: Any) -> Any:
        return self.torch.cat(
            [
                plan.forward_selected_z_modes(coeff, z_indices)
                for plan in self.plans
            ],
            dim=0,
        )

    def adjoint_selected_z(self, residual: Any, z_indices: Any) -> Any:
        if residual.shape != (self.q_count,):
            raise ValueError("banded residual size mismatch")
        offset = 0
        gradient = None
        for plan in self.plans:
            part = residual[offset : offset + plan.q_count]
            local = plan.adjoint_selected_z(part, z_indices)
            gradient = local if gradient is None else gradient + local
            offset += plan.q_count
        return gradient

    def adjoint_selected_z_modes(self, residual: Any, z_indices: Any) -> Any:
        if residual.shape != (self.mode_q_count,):
            raise ValueError("banded mode residual size mismatch")
        offset = 0
        gradient = None
        for plan in self.plans:
            part = residual[offset : offset + plan.mode_q_count]
            local = plan.adjoint_selected_z_modes(part, z_indices)
            gradient = local if gradient is None else gradient + local
            offset += plan.mode_q_count
        return gradient


VARIANTS = {
    "rho256_phi256": [
        {
            "name": "full",
            "cap_radial": 256,
            "cap_phi": 256,
            "min_fraction": 0.0,
            "max_fraction": 1.0,
            "sampling": "uniform_rho",
            "outer_power": 2.0,
        }
    ],
    "outer160_phi128": [
        {
            "name": "full",
            "cap_radial": 160,
            "cap_phi": 128,
            "min_fraction": 0.0,
            "max_fraction": 1.0,
            "sampling": "outer_power",
            "outer_power": 2.0,
        }
    ],
    "banded_inner96_outer64": [
        {
            "name": "inner",
            "cap_radial": 96,
            "cap_phi": 128,
            "min_fraction": 0.0,
            "max_fraction": 0.75,
            "sampling": "outer_power",
            "outer_power": 2.0,
        },
        {
            "name": "outer",
            "cap_radial": 64,
            "cap_phi": 256,
            "min_fraction": 0.75,
            "max_fraction": 1.0,
            "sampling": "outer_power",
            "outer_power": 2.0,
        },
    ],
}


def torch_plan_options(base_args: argparse.Namespace) -> dict[str, Any]:
    """Resolve explicit optimized-plan options without changing legacy defaults."""
    return {
        "low_memory_adjoint": bool(
            getattr(base_args, "low_memory_adjoint", True)
        ),
        "radial_block_size": int(getattr(base_args, "radial_block_size", 0)),
        "illumination_block_size": int(
            getattr(base_args, "illumination_block_size", 0)
        ),
        "forward_mode": str(
            getattr(base_args, "forward_mode", "illumination-reduced")
        ),
        "adjoint_mode": str(
            getattr(base_args, "adjoint_mode", "illumination-reduced")
        ),
        "prune_axis_l0": bool(getattr(base_args, "prune_axis_l0", False)),
        "axial_lowrank_rank": int(
            getattr(base_args, "axial_lowrank_rank", 0)
        ),
        "ring_adaptive_l_packed_threshold": float(
            getattr(base_args, "ring_adaptive_l_packed_threshold", 0.0)
        ),
    }


def build_variant(
    *,
    torch: Any,
    device: Any,
    base_args: argparse.Namespace,
    label: str,
) -> tuple[TorchBandedCompositePlan, Any, list[dict[str, Any]], float, float]:
    specifications = VARIANTS[label]
    plan_options = torch_plan_options(base_args)
    contexts = []
    plans = []
    metadata = []
    context_total_s = 0.0
    plan_total_s = 0.0
    for specification in specifications:
        args = argparse.Namespace(**vars(base_args))
        args.cap_radial = specification["cap_radial"]
        args.cap_phi = specification["cap_phi"]
        args.detector_radial_sampling = specification["sampling"]
        args.detector_radial_outer_power = specification["outer_power"]
        args.detector_radial_min_fraction = specification["min_fraction"]
        args.detector_radial_max_fraction = specification["max_fraction"]
        start = time.perf_counter()
        context = build_composite_context(args)
        context_total_s += time.perf_counter() - start
        start = time.perf_counter()
        plan = TorchCompositeOdtPlan.from_context(
            context,
            torch=torch,
            device=device,
            dtype=args.dtype,
            **plan_options,
        )
        synchronize(torch, device)
        plan_total_s += time.perf_counter() - start
        if not plan.ring.slots_unique or (
            plan.axis is not None and not plan.axis.slots_unique
        ):
            raise RuntimeError(
                f"{label}/{specification['name']} aliases active h modes"
            )
        contexts.append(context)
        plans.append(plan)
        metadata.append(
            {
                **specification,
                "samples_per_view": int(
                    specification["cap_radial"] * specification["cap_phi"]
                ),
                "ring_h_cutoff": int(context.ring.axis_h_cutoff),
                "ring_h_modes": int(context.ring.plan.h_values.size),
                "q_samples": int(plan.q_count),
                "ring_axial_lowrank_rank": int(plan.ring.axial_lowrank_rank),
                "axis_l_modes": (
                    None if plan.axis is None else int(plan.axis.n_l)
                ),
                "ring_adaptive_l_active_fraction": float(
                    plan.ring.adaptive_l_active_fraction
                ),
            }
        )
    reference_coeff = contexts[0].ring.obj.coeff
    if any(
        not np.array_equal(context.ring.obj.coeff, reference_coeff)
        for context in contexts[1:]
    ):
        raise RuntimeError("detector bands did not build the same object")
    return (
        TorchBandedCompositePlan(plans),
        contexts[0],
        metadata,
        context_total_s,
        plan_total_s,
    )


def selected_dot_error(
    *, torch: Any, plan: Any, truth: Any, z_index: Any
) -> float:
    with torch.inference_mode():
        data = plan.forward_selected_z(truth, z_index)
        residual = data * (0.17 - 0.11j)
        gradient = plan.adjoint_selected_z(residual, z_index)
        lhs = torch.vdot(data.reshape(-1), residual.reshape(-1))
        rhs = torch.vdot(truth.reshape(-1), gradient.reshape(-1))
        denominator = torch.clamp(torch.abs(lhs) + torch.abs(rhs), min=1e-30)
        return float((torch.abs(lhs - rhs) / denominator).item())


def reference_rows(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        case["label"]: {
            row["selected_n_z"]: row for row in case["selected"]
        }
        for case in payload["cases"]
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    references = reference_rows(args.reference)
    selected_counts = [int(value) for value in args.selected_slices.split(",")]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")

    cases = []
    for label in variants:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        plan, context, bands, context_s, plan_s = build_variant(
            torch=torch, device=device, base_args=args, label=label
        )
        selected_rows = []
        for selected_n_z in selected_counts:
            z_indices = centered_z_indices(args.n_z, selected_n_z)
            z_index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
            truth = selected_truth(
                torch, plan, context, z_indices, dtype=args.dtype
            )
            with torch.inference_mode():
                data = plan.forward_selected_z(truth, z_index)
            synchronize(torch, device)
            dot_error = selected_dot_error(
                torch=torch, plan=plan, truth=truth, z_index=z_index
            )
            lipschitz, power_s = selected_data_lipschitz(
                torch=torch,
                plan=plan,
                z_index=z_index,
                selected_n_z=selected_n_z,
                iterations=args.power_iterations,
                seed=args.seed + 3001 + selected_n_z,
            )
            solve, reconstruction, history = solve_selected_fista(
                torch=torch,
                plan=plan,
                z_index=z_index,
                data=data,
                truth=truth,
                label=label,
                data_lipschitz=lipschitz,
                iterations=args.iterations,
                record_every=args.record_every,
            )
            baseline = references["rho256"][selected_n_z]["solve"]
            radial = references["outer160_p2"][selected_n_z]["solve"]
            selected_rows.append(
                {
                    "selected_n_z": selected_n_z,
                    "q_samples": int(data.numel()),
                    "forward_adjoint_dot_error": float(dot_error),
                    "power_iteration_s": float(power_s),
                    "solve": solve,
                    "history": history,
                    "object_error_delta_vs_rho256": float(
                        solve["object_rel_l2"] - baseline["object_rel_l2"]
                    ),
                    "object_error_delta_vs_outer160_phi256": float(
                        solve["object_rel_l2"] - radial["object_rel_l2"]
                    ),
                    "speedup_vs_rho256": float(
                        baseline["core_iteration_timing"]["median_s"]
                        / solve["core_iteration_timing"]["median_s"]
                    ),
                    "speedup_vs_outer160_phi256": float(
                        radial["core_iteration_timing"]["median_s"]
                        / solve["core_iteration_timing"]["median_s"]
                    ),
                }
            )
            del z_index, truth, data, reconstruction
            gc.collect()
            torch.cuda.empty_cache()
        samples_per_view = int(sum(row["samples_per_view"] for row in bands))
        cases.append(
            {
                "label": label,
                "bands": bands,
                "samples_per_view": samples_per_view,
                "sample_reduction_fraction_vs_rho256": float(
                    1.0 - samples_per_view / (256.0 * 256.0)
                ),
                "context_build_s": float(context_s),
                "plan_build_s": float(plan_s),
                "basis_mib": float(plan.basis_mib),
                "pytorch_peak_allocated_mib": float(
                    torch.cuda.max_memory_allocated(device) / 1024**2
                ),
                "selected": selected_rows,
            }
        )
        del plan, context
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "schema": "odt-banded-detector-selected-z-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_name": device_name(torch, device),
        "problem": {
            "full_object_shape": [args.n_r, args.n_z, args.n_beta],
            "selected_slices": selected_counts,
            "total_views": int(args.ring_illum)
            + (0 if args.skip_axis_illumination else 1),
            "iterations": int(args.iterations),
            "noise": "none",
        },
        "reference": str(args.reference),
        "cases": cases,
        "claim_boundary": [
            "Banded plans evaluate independent radial detector bands and sum their exact adjoints.",
            "The comparison uses ideal band-matched data and fixed iteration count.",
            "A multi-band plan repeats some object-side contractions, so sample reduction need not translate linearly to time reduction.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    p = chained_parser()
    p.description = "Evaluate single-grid and banded ACFO-friendly ODT detectors."
    p.add_argument(
        "--variants",
        default="outer160_phi128,banded_inner96_outer64",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_radial_geometry_shortlist_120.json",
    )
    p.set_defaults(
        iterations=120,
        power_iterations=6,
        record_every=10,
        selected_slices="1,8",
        output=ROOT / "benchmark_results" / "odt_banded_detector.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    compact = [
        {
            "label": case["label"],
            "samples_per_view": case["samples_per_view"],
            "sample_reduction": case["sample_reduction_fraction_vs_rho256"],
            "selected": [
                {
                    "n": row["selected_n_z"],
                    "dot_error": row["forward_adjoint_dot_error"],
                    "object_rel_l2": row["solve"]["object_rel_l2"],
                    "median_iter_s": row["solve"]["core_iteration_timing"][
                        "median_s"
                    ],
                    "speedup_vs_rho256": row["speedup_vs_rho256"],
                    "speedup_vs_outer160_phi256": row[
                        "speedup_vs_outer160_phi256"
                    ],
                }
                for row in case["selected"]
            ],
        }
        for case in result["cases"]
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
