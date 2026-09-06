from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from benchmark_odt_banded_cartesian_reconstruction import (  # noqa: E402
    direct_cartesian_camera,
    hot_remap_timing,
    make_remap_plans,
    parser as chained_parser,
    remap_camera,
)
from benchmark_odt_banded_detector import (  # noqa: E402
    VARIANTS,
    TorchBandedCompositePlan,
    build_variant,
)
from benchmark_odt_selected_z_gpu import centered_z_indices  # noqa: E402
from benchmark_odt_virtual_polar_reconstruction import selected_truth  # noqa: E402


def norm(torch: Any, value: Any) -> Any:
    return torch.clamp(torch.linalg.vector_norm(value), min=1e-30)


def relative_l2(torch: Any, candidate: Any, reference: Any) -> float:
    return float((norm(torch, candidate - reference) / norm(torch, reference)).item())


def real_complex(torch: Any, value: Any) -> Any:
    real = torch.real(value)
    return torch.complex(real, torch.zeros_like(real))


def real_dot(torch: Any, left: Any, right: Any) -> Any:
    return torch.sum(torch.real(torch.conj(left) * right))


def timing_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median_s": float(np.median(array)),
        "mean_s": float(array.mean()),
        "p05_s": float(np.percentile(array, 5)),
        "p95_s": float(np.percentile(array, 95)),
        "min_s": float(array.min()),
        "max_s": float(array.max()),
    }


def timed_cuda(
    *,
    torch: Any,
    device: Any,
    callback: Callable[[], Any],
    warmups: int,
    repeats: int,
) -> tuple[dict[str, float | int], Any]:
    value = None
    with torch.inference_mode():
        for _ in range(warmups):
            value = callback()
            synchronize(torch, device)
        times: list[float] = []
        for _ in range(repeats):
            synchronize(torch, device)
            start = time.perf_counter()
            value = callback()
            synchronize(torch, device)
            times.append(time.perf_counter() - start)
    return timing_summary(times), value


def cone_pixel_to_modes(cone: Any, pixel_data: Any) -> Any:
    grid = pixel_data.reshape(cone.n_illum, cone.cap_radial, cone.cap_phi)
    folded = cone.torch.fft.ifft(grid, dim=2)
    selected = folded.index_select(2, cone.slots)
    return (selected * math.sqrt(float(cone.cap_phi))).reshape(-1)


def banded_pixel_to_modes(plan: TorchBandedCompositePlan, pixel_data: Any) -> Any:
    if pixel_data.shape != (plan.q_count,):
        raise ValueError("banded pixel data size mismatch")
    parts = []
    offset = 0
    for local_plan in plan.plans:
        local = pixel_data[offset : offset + local_plan.q_count]
        ring_count = local_plan.ring.q_count
        parts.append(cone_pixel_to_modes(local_plan.ring, local[:ring_count]))
        if local_plan.axis is not None:
            parts.append(cone_pixel_to_modes(local_plan.axis, local[ring_count:]))
        offset += local_plan.q_count
    return plan.torch.cat(parts, dim=0)


def random_complex_like(torch: Any, value: Any, seed: int) -> Any:
    torch.manual_seed(seed)
    real = torch.randn(value.shape, dtype=value.real.dtype, device=value.device)
    imag = torch.randn(value.shape, dtype=value.real.dtype, device=value.device)
    return torch.complex(real, imag)


def solve_real_cg_modes(
    *,
    torch: Any,
    plan: TorchBandedCompositePlan,
    z_index: Any,
    truth: Any,
    data_modes: Any,
    iterations: int,
) -> tuple[dict[str, Any], Any]:
    truth_norm = norm(torch, truth)
    data_norm = norm(torch, data_modes)
    synchronize(torch, plan.device)
    start = time.perf_counter()
    rhs = real_complex(torch, plan.adjoint_selected_z_modes(data_modes, z_index))
    synchronize(torch, plan.device)
    rhs_s = time.perf_counter() - start

    x = torch.zeros_like(truth)
    prediction = torch.zeros_like(data_modes)
    residual_normal = rhs.clone()
    direction = residual_normal.clone()
    rr = real_dot(torch, residual_normal, residual_normal)
    update_times: list[float] = []
    with torch.inference_mode():
        for _ in range(iterations):
            synchronize(torch, plan.device)
            start = time.perf_counter()
            projected = plan.forward_selected_z_modes(direction, z_index)
            normal_direction = real_complex(
                torch, plan.adjoint_selected_z_modes(projected, z_index)
            )
            denominator = torch.clamp(
                real_dot(torch, direction, normal_direction), min=1e-30
            )
            alpha = rr / denominator
            x = x + alpha * direction
            prediction = prediction + alpha * projected
            residual_new = residual_normal - alpha * normal_direction
            rr_new = real_dot(torch, residual_new, residual_new)
            beta = rr_new / torch.clamp(rr, min=1e-30)
            direction = residual_new + beta * direction
            residual_normal = residual_new
            rr = rr_new
            synchronize(torch, plan.device)
            update_times.append(time.perf_counter() - start)

    return (
        {
            "solver": "real-subspace CG on final packed banded mode operator",
            "iterations": int(iterations),
            "rhs_adjoint_s": float(rhs_s),
            "update_timing": timing_summary(update_times),
            "object_nrmse": float((norm(torch, x - truth) / truth_norm).item()),
            "data_residual": float(
                (norm(torch, prediction - data_modes) / data_norm).item()
            ),
        },
        x,
    )


def reference_args(args: argparse.Namespace) -> argparse.Namespace:
    result = argparse.Namespace(**vars(args))
    result.h_cutoff = int(args.reference_h_cutoff)
    result.prune_axis_l0 = False
    result.axial_lowrank_rank = 0
    result.ring_adaptive_l_packed_threshold = 0.0
    result.forward_mode = "auto"
    result.adjoint_mode = "auto"
    return result


def candidate_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "h_cutoff": int(args.h_cutoff),
        "prune_axis_l0": bool(args.prune_axis_l0),
        "axial_lowrank_rank": int(args.axial_lowrank_rank),
        "ring_adaptive_l_packed_threshold": float(
            args.ring_adaptive_l_packed_threshold
        ),
        "radial_block_size": int(args.radial_block_size),
        "illumination_block_size": int(args.illumination_block_size),
        "forward_mode": str(args.forward_mode),
        "adjoint_mode": str(args.adjoint_mode),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS:
        raise ValueError(f"unknown banded detector variant: {args.variant}")
    selected_counts = [int(value) for value in args.selected_slices.split(",")]
    if not selected_counts or any(value <= 0 or value > args.n_z for value in selected_counts):
        raise ValueError("selected-slices must be within [1, n-z]")

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    candidate_start = time.perf_counter()
    candidate_plan, candidate_context, candidate_bands, candidate_context_s, candidate_plan_s = (
        build_variant(
            torch=torch,
            device=device,
            base_args=args,
            label=args.variant,
        )
    )
    synchronize(torch, device)
    candidate_total_s = time.perf_counter() - candidate_start

    ref_args = reference_args(args)
    reference_start = time.perf_counter()
    reference_plan, reference_context, reference_bands, reference_context_s, reference_plan_s = (
        build_variant(
            torch=torch,
            device=device,
            base_args=ref_args,
            label=args.variant,
        )
    )
    synchronize(torch, device)
    reference_total_s = time.perf_counter() - reference_start
    if candidate_plan.q_count != reference_plan.q_count:
        raise RuntimeError("candidate and reference pixel q counts differ")
    if not np.array_equal(
        candidate_context.ring.obj.coeff, reference_context.ring.obj.coeff
    ):
        raise RuntimeError("candidate and reference object coefficients differ")

    remap_plans = make_remap_plans(
        torch=torch, device=device, args=args, variant=args.variant
    )
    cases: list[dict[str, Any]] = []
    for selected_n_z in selected_counts:
        z_indices = centered_z_indices(args.n_z, selected_n_z)
        z_index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
        truth = selected_truth(
            torch, candidate_plan, candidate_context, z_indices, dtype=args.dtype
        )
        stress_truth = random_complex_like(
            torch, truth, args.seed + 1000 + selected_n_z
        )
        with torch.inference_mode():
            reference_forward = reference_plan.forward_selected_z(truth, z_index)
            candidate_forward = candidate_plan.forward_selected_z(truth, z_index)
            reference_stress_forward = reference_plan.forward_selected_z(
                stress_truth, z_index
            )
            candidate_stress_forward = candidate_plan.forward_selected_z(
                stress_truth, z_index
            )
            stress_residual = random_complex_like(
                torch,
                reference_forward,
                args.seed + 2000 + selected_n_z,
            )
            reference_adjoint = reference_plan.adjoint_selected_z(
                reference_forward * (0.1 + 0.2j), z_index
            )
            candidate_adjoint = candidate_plan.adjoint_selected_z(
                reference_forward * (0.1 + 0.2j), z_index
            )
            reference_stress_adjoint = reference_plan.adjoint_selected_z(
                stress_residual, z_index
            )
            candidate_stress_adjoint = candidate_plan.adjoint_selected_z(
                stress_residual, z_index
            )
            ideal_modes = candidate_plan.forward_selected_z_modes(truth, z_index)
            pixel_derived_modes = banded_pixel_to_modes(
                candidate_plan, candidate_forward
            )

        operator_errors = {
            "physical_forward_rel_l2": relative_l2(
                torch, candidate_forward, reference_forward
            ),
            "physical_adjoint_rel_l2": relative_l2(
                torch, candidate_adjoint, reference_adjoint
            ),
            "stress_forward_rel_l2": relative_l2(
                torch, candidate_stress_forward, reference_stress_forward
            ),
            "stress_adjoint_rel_l2": relative_l2(
                torch, candidate_stress_adjoint, reference_stress_adjoint
            ),
        }
        operator_errors["worst_rel_l2"] = max(operator_errors.values())
        forward_mode_equivalence = relative_l2(
            torch, pixel_derived_modes, ideal_modes
        )

        camera, camera_metadata = direct_cartesian_camera(
            torch=torch,
            device=device,
            args=args,
            context=candidate_context,
            z_indices=z_indices,
        )
        remap_timing = hot_remap_timing(
            torch=torch,
            device=device,
            camera=camera,
            plans=remap_plans,
            warmups=args.remap_warmups,
            repeats=args.remap_repeats,
        )
        with torch.inference_mode():
            remapped_pixel = remap_camera(
                torch=torch, camera=camera, plans=remap_plans
            )
            remapped_modes = banded_pixel_to_modes(
                candidate_plan, remapped_pixel
            )
            remapped_pixel_adjoint = candidate_plan.adjoint_selected_z(
                remapped_pixel, z_index
            )
            remapped_mode_adjoint = candidate_plan.adjoint_selected_z_modes(
                remapped_modes, z_index
            )
        mode_reduction_timing, _ = timed_cuda(
            torch=torch,
            device=device,
            callback=lambda: banded_pixel_to_modes(candidate_plan, remapped_pixel),
            warmups=args.pair_warmups,
            repeats=args.pair_repeats,
        )
        pair_residual = remapped_modes * (0.1 + 0.2j)
        pair_timing, _ = timed_cuda(
            torch=torch,
            device=device,
            callback=lambda: (
                candidate_plan.forward_selected_z_modes(stress_truth, z_index),
                candidate_plan.adjoint_selected_z_modes(pair_residual, z_index),
            ),
            warmups=args.pair_warmups,
            repeats=args.pair_repeats,
        )

        def integrated_update() -> Any:
            current_pixel = remap_camera(
                torch=torch, camera=camera, plans=remap_plans
            )
            current_modes = banded_pixel_to_modes(candidate_plan, current_pixel)
            projected = candidate_plan.forward_selected_z_modes(
                stress_truth, z_index
            )
            return (
                current_modes,
                candidate_plan.adjoint_selected_z_modes(projected, z_index),
            )

        integrated_timing, _ = timed_cuda(
            torch=torch,
            device=device,
            callback=integrated_update,
            warmups=args.pair_warmups,
            repeats=args.pair_repeats,
        )

        ideal_solve, ideal_reconstruction = solve_real_cg_modes(
            torch=torch,
            plan=candidate_plan,
            z_index=z_index,
            truth=truth,
            data_modes=ideal_modes,
            iterations=args.iterations,
        )
        remap_solve, remap_reconstruction = solve_real_cg_modes(
            torch=torch,
            plan=candidate_plan,
            z_index=z_index,
            truth=truth,
            data_modes=remapped_modes,
            iterations=args.iterations,
        )
        reconstruction_difference = relative_l2(
            torch, remap_reconstruction, ideal_reconstruction
        )
        case = {
            "selected_n_z": int(selected_n_z),
            "object_shape": list(truth.shape),
            "pixel_q_samples": int(candidate_forward.numel()),
            "mode_q_samples": int(ideal_modes.numel()),
            "operator_errors_vs_h36": operator_errors,
            "candidate_pixel_to_direct_mode_rel_l2": forward_mode_equivalence,
            "cartesian_camera": camera_metadata,
            "remap_timing": remap_timing,
            "mode_reduction_timing": mode_reduction_timing,
            "final_packed_mode_pair_timing": pair_timing,
            "integrated_remap_mode_update_timing": integrated_timing,
            "integrated_updates_per_second": float(
                1.0 / integrated_timing["median_s"]
            ),
            "remapped_pixel_vs_candidate_ideal_rel_l2": relative_l2(
                torch, remapped_pixel, candidate_forward
            ),
            "remapped_pixel_vs_h36_reference_rel_l2": relative_l2(
                torch, remapped_pixel, reference_forward
            ),
            "remapped_mode_vs_candidate_ideal_rel_l2": relative_l2(
                torch, remapped_modes, ideal_modes
            ),
            "mode_vs_pixel_adjoint_rel_l2": relative_l2(
                torch, remapped_mode_adjoint, remapped_pixel_adjoint
            ),
            "ideal_reconstruction": ideal_solve,
            "remapped_reconstruction": remap_solve,
            "remap_vs_ideal_reconstruction_rel_l2": reconstruction_difference,
        }
        cases.append(case)
        print(
            f"z={selected_n_z}: operator={operator_errors['worst_rel_l2']:.3e}, "
            f"remap-recon={reconstruction_difference:.3e}, "
            f"integrated={1000.0 * integrated_timing['median_s']:.3f} ms "
            f"({1.0 / integrated_timing['median_s']:.2f} Hz)",
            flush=True,
        )
        del (
            z_index,
            truth,
            stress_truth,
            reference_forward,
            candidate_forward,
            reference_stress_forward,
            candidate_stress_forward,
            stress_residual,
            reference_adjoint,
            candidate_adjoint,
            reference_stress_adjoint,
            candidate_stress_adjoint,
            ideal_modes,
            pixel_derived_modes,
            camera,
            remapped_pixel,
            remapped_modes,
            remapped_pixel_adjoint,
            remapped_mode_adjoint,
            pair_residual,
            ideal_reconstruction,
            remap_reconstruction,
        )
        gc.collect()
        torch.cuda.empty_cache()

    gates = {
        "all_operator_errors_le_2e-6": all(
            case["operator_errors_vs_h36"]["worst_rel_l2"] <= 2e-6
            for case in cases
        ),
        "all_pixel_to_mode_equivalence_le_2e-6": all(
            case["candidate_pixel_to_direct_mode_rel_l2"] <= 2e-6
            for case in cases
        ),
        "all_mode_vs_pixel_adjoint_equivalence_le_2e-6": all(
            case["mode_vs_pixel_adjoint_rel_l2"] <= 2e-6 for case in cases
        ),
        "all_remap_reconstruction_difference_le_2e-3": all(
            case["remap_vs_ideal_reconstruction_rel_l2"] <= 2e-3
            for case in cases
        ),
        "all_integrated_timings_finite": all(
            np.isfinite(case["integrated_remap_mode_update_timing"]["median_s"])
            and case["integrated_remap_mode_update_timing"]["median_s"] > 0
            for case in cases
        ),
    }
    result = {
        "schema": "odt-banded-cartesian-final-packed-integration-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": device_name(torch, device),
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(torch.version, "cuda", None),
            "dtype": args.dtype,
        },
        "detector": {
            "variant": args.variant,
            "camera_shape_per_view": [args.camera_n_xy, args.camera_n_xy],
            "bands": candidate_bands,
            "views": int(args.ring_illum)
            + (0 if args.skip_axis_illumination else 1),
        },
        "candidate": {
            **candidate_contract(args),
            "context_build_s": float(candidate_context_s),
            "plan_build_s": float(candidate_plan_s),
            "total_build_s": float(candidate_total_s),
            "mode_q_count": int(candidate_plan.mode_q_count),
            "bands": candidate_bands,
        },
        "reference": {
            "h_cutoff": int(args.reference_h_cutoff),
            "axial_lowrank_rank": 0,
            "ring_adaptive_l_packed_threshold": 0.0,
            "context_build_s": float(reference_context_s),
            "plan_build_s": float(reference_plan_s),
            "total_build_s": float(reference_total_s),
            "bands": reference_bands,
        },
        "problem": {
            "full_object_shape": [args.n_r, args.n_z, args.n_beta],
            "selected_slices": selected_counts,
            "reconstruction_iterations": int(args.iterations),
            "noise": "none",
        },
        "cases": cases,
        "gpu_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "gates": gates,
        "passed": bool(all(gates.values())),
        "claim_boundary": [
            "This is the first single-run integration of Cartesian camera remap, angular-mode reduction and the final H28/rank16/adaptive-L packed operator.",
            "The H36 structured comparison isolates operator approximation; the remap-vs-ideal rows isolate detector interpolation.",
            "The integrated hot timing includes cached remap, mode reduction and one forward/adjoint normal-operator application; camera acquisition and hologram demodulation remain excluded.",
            "The default run is a z=1/z=8 probe. Full 256-z publication timing is a separate follow-up after this gate passes.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    p = chained_parser()
    p.description = (
        "Integrate Cartesian detector remap, angular-mode reduction and the final "
        "packed ODT operator against an H36 structured reference."
    )
    p.add_argument("--reference-h-cutoff", type=int, default=36)
    p.add_argument("--pair-warmups", type=int, default=5)
    p.add_argument("--pair-repeats", type=int, default=20)
    p.set_defaults(
        h_cutoff=28,
        prune_axis_l0=True,
        axial_lowrank_rank=16,
        ring_adaptive_l_packed_threshold=1e-6,
        forward_mode="auto",
        adjoint_mode="auto",
        low_memory_adjoint=True,
        radial_block_size=32,
        illumination_block_size=4,
        skip_native_prepared_adjoint=True,
        compact_axisymmetric_kernel=True,
        iterations=8,
        record_every=1,
        selected_slices="1,8",
        remap_warmups=5,
        remap_repeats=20,
        output=ROOT
        / "benchmark_results"
        / "odt_banded_cartesian_final_packed_probe.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    compact = {
        "passed": result["passed"],
        "gates": result["gates"],
        "gpu_peak_allocated_mib": result["gpu_peak_allocated_mib"],
        "cases": [
            {
                "selected_n_z": case["selected_n_z"],
                "operator_worst_rel_l2": case["operator_errors_vs_h36"][
                    "worst_rel_l2"
                ],
                "remap_reconstruction_rel_l2": case[
                    "remap_vs_ideal_reconstruction_rel_l2"
                ],
                "integrated_median_ms": 1000.0
                * case["integrated_remap_mode_update_timing"]["median_s"],
                "integrated_hz": case["integrated_updates_per_second"],
            }
            for case in result["cases"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
