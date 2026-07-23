from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    resolve_device,
    synchronize,
)
from benchmark_odt_banded_detector import (  # noqa: E402
    VARIANTS,
    TorchBandedCompositePlan,
    build_variant,
)
from benchmark_odt_selected_z_gpu import centered_z_indices  # noqa: E402
from benchmark_odt_virtual_polar_reconstruction import (  # noqa: E402
    parser as chained_parser,
    selected_truth,
    solve_selected_fista,
)


@dataclass
class HarmonicSelectedPlan:
    base: TorchBandedCompositePlan

    def __post_init__(self) -> None:
        self.torch = self.base.torch
        self.device = self.base.device
        self.complex_dtype = self.base.complex_dtype
        self.real_dtype = self.base.real_dtype
        self.ring = self.base.ring

    @property
    def q_count(self) -> int:
        return self.base.mode_q_count

    def forward_selected_z(self, coeff: Any, z_indices: Any) -> Any:
        return self.base.forward_selected_z_modes(coeff, z_indices)

    def adjoint_selected_z(self, residual: Any, z_indices: Any) -> Any:
        return self.base.adjoint_selected_z_modes(residual, z_indices)


def case_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {case["label"]: case for case in payload["cases"]}


def selected_map(case: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {row["selected_n_z"]: row for row in case["selected"]}


def relative_l2(torch: Any, candidate: Any, reference: Any) -> float:
    denominator = torch.clamp(torch.linalg.vector_norm(reference), min=1e-30)
    return float((torch.linalg.vector_norm(candidate - reference) / denominator).item())


def pixel_reference(
    *,
    label: str,
    selected_n_z: int,
    radial_cases: dict[str, dict[str, Any]],
    banded_cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if label == "rho256_phi256":
        return selected_map(radial_cases["rho256"])[selected_n_z]
    return selected_map(banded_cases[label])[selected_n_z]


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    radial_cases = case_map(args.radial_reference)
    banded_cases = case_map(args.banded_reference)
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
        pixel_plan, context, bands, context_s, plan_s = build_variant(
            torch=torch, device=device, base_args=args, label=label
        )
        mode_plan = HarmonicSelectedPlan(pixel_plan)
        selected_rows = []
        for selected_n_z in selected_counts:
            z_indices = centered_z_indices(args.n_z, selected_n_z)
            z_index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
            truth = selected_truth(
                torch, pixel_plan, context, z_indices, dtype=args.dtype
            )
            with torch.inference_mode():
                pixel_data = pixel_plan.forward_selected_z(truth, z_index)
                mode_data = mode_plan.forward_selected_z(truth, z_index)
                pixel_normal = pixel_plan.adjoint_selected_z(pixel_data, z_index)
                mode_normal = mode_plan.adjoint_selected_z(mode_data, z_index)
                residual = mode_data * (0.13 + 0.19j)
                mode_gradient = mode_plan.adjoint_selected_z(residual, z_index)
                lhs = torch.vdot(mode_data.reshape(-1), residual.reshape(-1))
                rhs = torch.vdot(truth.reshape(-1), mode_gradient.reshape(-1))
                dot_denominator = torch.clamp(
                    torch.abs(lhs) + torch.abs(rhs), min=1e-30
                )
                dot_error = float((torch.abs(lhs - rhs) / dot_denominator).item())
            synchronize(torch, device)
            reference = pixel_reference(
                label=label,
                selected_n_z=selected_n_z,
                radial_cases=radial_cases,
                banded_cases=banded_cases,
            )
            pixel_solve = reference["solve"]
            data_lipschitz = float(pixel_solve["lipschitz_with_margin"]) / 1.10
            solve, reconstruction, history = solve_selected_fista(
                torch=torch,
                plan=mode_plan,
                z_index=z_index,
                data=mode_data,
                truth=truth,
                label=f"{label}_modes",
                data_lipschitz=data_lipschitz,
                iterations=args.iterations,
                record_every=args.record_every,
            )
            pixel_median = float(
                pixel_solve["core_iteration_timing"]["median_s"]
            )
            mode_median = float(solve["core_iteration_timing"]["median_s"])
            selected_rows.append(
                {
                    "selected_n_z": selected_n_z,
                    "pixel_q_samples": int(pixel_data.numel()),
                    "mode_q_samples": int(mode_data.numel()),
                    "mode_data_reduction_fraction": float(
                        1.0 - mode_data.numel() / pixel_data.numel()
                    ),
                    "mode_data_reduction_fraction_vs_rho256_pixel": float(
                        1.0 - mode_data.numel() / (121.0 * 256.0 * 256.0)
                    ),
                    "pixel_vs_mode_data_norm_rel": float(
                        abs(
                            float(torch.linalg.vector_norm(pixel_data).item())
                            - float(torch.linalg.vector_norm(mode_data).item())
                        )
                        / max(float(torch.linalg.vector_norm(pixel_data).item()), 1e-30)
                    ),
                    "pixel_vs_mode_normal_rel_l2": relative_l2(
                        torch, mode_normal, pixel_normal
                    ),
                    "mode_forward_adjoint_dot_error": dot_error,
                    "solve": solve,
                    "history": history,
                    "object_error_delta_vs_pixel": float(
                        solve["object_rel_l2"] - pixel_solve["object_rel_l2"]
                    ),
                    "data_residual_delta_vs_pixel": float(
                        solve["data_residual"] - pixel_solve["data_residual"]
                    ),
                    "iteration_speedup_vs_pixel": float(
                        pixel_median / mode_median
                    ),
                }
            )
            del (
                z_index,
                truth,
                pixel_data,
                mode_data,
                pixel_normal,
                mode_normal,
                mode_gradient,
                reconstruction,
            )
            gc.collect()
            torch.cuda.empty_cache()
        cases.append(
            {
                "label": label,
                "bands": bands,
                "pixel_q_count": int(pixel_plan.q_count),
                "mode_q_count": int(pixel_plan.mode_q_count),
                "context_build_s": float(context_s),
                "plan_build_s": float(plan_s),
                "pytorch_peak_allocated_mib": float(
                    torch.cuda.max_memory_allocated(device) / 1024**2
                ),
                "selected": selected_rows,
            }
        )
        del mode_plan, pixel_plan, context
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "schema": "odt-active-detector-harmonic-reconstruction-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_name": device_name(torch, device),
        "problem": {
            "full_object_shape": [args.n_r, args.n_z, args.n_beta],
            "selected_slices": selected_counts,
            "total_views": int(args.ring_illum)
            + (0 if args.skip_axis_illumination else 1),
            "iterations": int(args.iterations),
            "noise": "none",
            "mode_scaling": "orthonormal azimuthal FFT",
        },
        "radial_reference": str(args.radial_reference),
        "banded_reference": str(args.banded_reference),
        "cases": cases,
        "claim_boundary": [
            "The mode-domain operator retains every active detector harmonic of the prepared ACFO model.",
            "Pixel and mode objectives are equivalent for unweighted complex L2 data fidelity under the recorded orthonormal scaling.",
            "Nonuniform detector noise weights or model-mismatch likelihoods require transforming the covariance/weighting as well.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    p = chained_parser()
    p.description = "Benchmark active detector-harmonic selected-z reconstruction."
    p.add_argument(
        "--variants",
        default="rho256_phi256,outer160_phi128,banded_inner96_outer64",
    )
    p.add_argument(
        "--radial-reference",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_radial_geometry_shortlist_120.json",
    )
    p.add_argument(
        "--banded-reference",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_banded_detector.json",
    )
    p.set_defaults(
        iterations=120,
        record_every=10,
        selected_slices="1,8",
        output=ROOT / "benchmark_results" / "odt_harmonic_detector.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    compact = [
        {
            "label": case["label"],
            "pixel_q_count": case["pixel_q_count"],
            "mode_q_count": case["mode_q_count"],
            "selected": [
                {
                    "n": row["selected_n_z"],
                    "normal_rel_l2": row["pixel_vs_mode_normal_rel_l2"],
                    "dot_error": row["mode_forward_adjoint_dot_error"],
                    "object_rel_l2": row["solve"]["object_rel_l2"],
                    "object_delta": row["object_error_delta_vs_pixel"],
                    "median_iter_s": row["solve"]["core_iteration_timing"][
                        "median_s"
                    ],
                    "speedup_vs_pixel": row["iteration_speedup_vs_pixel"],
                }
                for row in case["selected"]
            ],
        }
        for case in result["cases"]
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
