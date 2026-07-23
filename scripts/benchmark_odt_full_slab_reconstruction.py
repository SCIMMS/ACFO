from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from benchmark_odt_realistic_geometry_reconstruction import build_composite_context
from benchmark_odt_selected_z_gpu import centered_z_indices
from benchmark_odt_torch_gpu_reconstruction import (
    TorchCompositeOdtPlan,
    device_name,
    import_torch,
    parser,
    resolve_device,
    synchronize,
    torch_dtypes,
)


def norm(torch: Any, value: Any) -> Any:
    return torch.clamp(torch.linalg.vector_norm(value), min=1e-30)


def real_complex(torch: Any, value: Any) -> Any:
    real = torch.real(value)
    return torch.complex(real, torch.zeros_like(real))


def real_dot(torch: Any, left: Any, right: Any) -> Any:
    return torch.sum(torch.real(torch.conj(left) * right))


def timing_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "total_s": float(array.sum()),
        "mean_s": float(array.mean()),
        "median_s": float(np.median(array)),
        "q1_s": float(np.quantile(array, 0.25)),
        "q3_s": float(np.quantile(array, 0.75)),
        "p05_s": float(np.percentile(array, 5)),
        "p95_s": float(np.percentile(array, 95)),
        "min_s": float(array.min()),
        "max_s": float(array.max()),
    }


def cone_pixel_to_modes(cone: Any, pixel_data: Any) -> Any:
    grid = pixel_data.reshape(cone.n_illum, cone.cap_radial, cone.cap_phi)
    folded = cone.torch.fft.ifft(grid, dim=2)
    selected = folded.index_select(2, cone.slots)
    return (selected * math.sqrt(float(cone.cap_phi))).reshape(-1)


def pixel_to_modes(plan: TorchCompositeOdtPlan, pixel_data: Any) -> Any:
    ring_count = plan.ring.q_count
    parts = [cone_pixel_to_modes(plan.ring, pixel_data[:ring_count])]
    if plan.axis is not None:
        parts.append(cone_pixel_to_modes(plan.axis, pixel_data[ring_count:]))
    return plan.torch.cat(parts, dim=0)


def relative_l2(torch: Any, candidate: Any, reference: Any) -> float:
    return float((norm(torch, candidate - reference) / norm(torch, reference)).item())


def threshold_crossings(
    history: list[dict[str, Any]],
    *,
    key: str,
    thresholds: list[float],
    rhs_s: float,
    preprocessing_s: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for threshold in thresholds:
        row = next((item for item in history if float(item[key]) <= threshold), None)
        result[f"{threshold:g}"] = (
            None
            if row is None
            else {
                "iteration": int(row["iteration"]),
                "value": float(row[key]),
                "iteration_core_cumulative_s": float(
                    row["iteration_core_cumulative_s"]
                ),
                "reconstruction_core_s_including_rhs": float(
                    rhs_s + row["iteration_core_cumulative_s"]
                ),
                "pixel_input_total_s": float(
                    preprocessing_s + rhs_s + row["iteration_core_cumulative_s"]
                ),
            }
        )
    return result


def solve_real_cg_modes(
    *,
    torch: Any,
    plan: TorchCompositeOdtPlan,
    z_index: Any,
    truth: Any,
    data_modes: Any,
    iterations: int,
    preprocessing_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth_norm = norm(torch, truth)
    data_norm = norm(torch, data_modes)
    synchronize(torch, plan.device)
    rhs_start = time.perf_counter()
    rhs = real_complex(
        torch, plan.adjoint_selected_z_modes(data_modes, z_index)
    )
    synchronize(torch, plan.device)
    rhs_s = time.perf_counter() - rhs_start
    rhs_norm = norm(torch, rhs)

    x = torch.zeros_like(truth)
    prediction = torch.zeros_like(data_modes)
    residual_normal = rhs.clone()
    direction = residual_normal.clone()
    rr = real_dot(torch, residual_normal, residual_normal)
    core_times: list[float] = []
    history: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for iteration in range(1, iterations + 1):
            synchronize(torch, plan.device)
            core_start = time.perf_counter()
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
            core_times.append(time.perf_counter() - core_start)

            object_nrmse = float((norm(torch, x - truth) / truth_norm).item())
            data_residual = float(
                (norm(torch, prediction - data_modes) / data_norm).item()
            )
            normal_residual = float(
                (torch.sqrt(torch.clamp(rr, min=0.0)) / rhs_norm).item()
            )
            synchronize(torch, plan.device)
            history.append(
                {
                    "iteration": int(iteration),
                    "object_nrmse": object_nrmse,
                    "data_residual": data_residual,
                    "normal_residual": normal_residual,
                    "alpha": float(alpha.item()),
                    "beta": float(beta.item()),
                    "iteration_core_s": float(core_times[-1]),
                    "iteration_core_cumulative_s": float(sum(core_times)),
                }
            )

    wall_s = time.perf_counter() - wall_start
    timing = timing_summary(core_times)
    result = {
        "solver": "conjugate gradient on real-subspace normal equations",
        "iterations": int(iterations),
        "rhs_adjoint_s": float(rhs_s),
        "iteration_core_timing": timing,
        "iteration_core_s": float(sum(core_times)),
        "reconstruction_core_s_including_rhs": float(rhs_s + sum(core_times)),
        "pixel_to_mode_preprocessing_s": float(preprocessing_s),
        "pixel_input_total_s": float(preprocessing_s + rhs_s + sum(core_times)),
        "wall_iterations_including_diagnostics_s": float(wall_s),
        "final": history[-1],
        "object_nrmse_crossings": threshold_crossings(
            history,
            key="object_nrmse",
            thresholds=[0.2, 0.1, 0.05, 0.02, 0.01],
            rhs_s=rhs_s,
            preprocessing_s=preprocessing_s,
        ),
        "data_residual_crossings": threshold_crossings(
            history,
            key="data_residual",
            thresholds=[0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
            rhs_s=rhs_s,
            preprocessing_s=preprocessing_s,
        ),
    }
    return result, history


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# ODT full and slab reconstruction timing",
        "",
        "Final H=28, axial-rank=16, adaptive-L=1e-6 operator를 사용한 GPU-resident iterative reconstruction 결과이다.",
        "",
        "## 공통 조건",
        "",
        f"- GPU: `{result['device']['name']}`",
        f"- full object: `{result['problem']['full_object_shape']}`",
        f"- illuminations: `{result['problem']['illumination_count']}`",
        f"- pixel q samples: `{result['problem']['pixel_q_samples']}`",
        f"- active mode samples: `{result['problem']['mode_q_samples']}`",
        f"- pixel-to-mode median: `{1000.0 * result['mode_preprocessing']['median_s']:.3f}` ms",
        f"- normal-operator equivalence rel-L2: `{result['mode_preprocessing']['normal_operator_rel_l2']:.3e}`",
        "",
        "## Reconstruction cases",
        "",
        "| z unknowns | object bins | median ms/iter | 100-iter core s | final object NRMSE | final data residual | time to object NRMSE <= 0.05 | time to data residual <= 0.01 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in result["cases"]:
        solver = case["reconstruction"]
        obj_cross = solver["object_nrmse_crossings"]["0.05"]
        data_cross = solver["data_residual_crossings"]["0.01"]
        obj_cross_text = (
            "-"
            if obj_cross is None
            else f"{obj_cross['reconstruction_core_s_including_rhs']:.3f} s"
        )
        data_cross_text = (
            "-"
            if data_cross is None
            else f"{data_cross['reconstruction_core_s_including_rhs']:.3f} s"
        )
        lines.append(
            f"| {case['selected_n_z']} | {case['object_bins']} | "
            f"{1000.0 * solver['iteration_core_timing']['median_s']:.3f} | "
            f"{solver['reconstruction_core_s_including_rhs']:.3f} | "
            f"{solver['final']['object_nrmse']:.4g} | "
            f"{solver['final']['data_residual']:.4g} | "
            f"{obj_cross_text} | {data_cross_text} |"
        )
    full_case = next(case for case in result["cases"] if case["selected_n_z"] == 256)
    full_solver = full_case["reconstruction"]
    full_data_1pct = full_solver["data_residual_crossings"]["0.01"]
    full_object_20pct = full_solver["object_nrmse_crossings"]["0.2"]
    ten_hz_slabs = [
        case
        for case in result["cases"]
        if case["selected_n_z"] < 256
        and case["reconstruction"]["iteration_core_timing"]["p95_s"] < 0.1
    ]
    largest_ten_hz_slab = max(case["selected_n_z"] for case in ten_hz_slabs)
    one_slice = next(case for case in result["cases"] if case["selected_n_z"] == 1)
    one_solver = one_slice["reconstruction"]
    one_object_5pct = one_solver["object_nrmse_crossings"]["0.05"]
    one_object_2pct = one_solver["object_nrmse_crossings"]["0.02"]
    one_object_1pct = one_solver["object_nrmse_crossings"]["0.01"]
    lines.extend(
        [
            "",
            "## Claim assessment",
            "",
            f"- Full 256-z: median `{1000.0 * full_solver['iteration_core_timing']['median_s']:.3f} ms/update` "
            f"(`{1.0 / full_solver['iteration_core_timing']['median_s']:.2f} Hz`), "
            f"RHS+100 updates `{full_solver['reconstruction_core_s_including_rhs']:.3f} s`.",
            f"- Full 256-z reaches data residual <= 1% at iteration `{full_data_1pct['iteration']}` / "
            f"`{full_data_1pct['reconstruction_core_s_including_rhs']:.3f} s`, but only reaches object NRMSE <= 20% at "
            f"iteration `{full_object_20pct['iteration']}` / `{full_object_20pct['reconstruction_core_s_including_rhs']:.3f} s`; "
            "it does not reach object NRMSE <= 10% in 100 iterations.",
            f"- Slab iterative-update claim: up to `{largest_ten_hz_slab}` centered z planes has p95 < 100 ms, "
            "so >=10 Hz repeated updates are supported under the stated known-support condition.",
            f"- One-slice completed-quality examples: object NRMSE <= 5% in `{one_object_5pct['reconstruction_core_s_including_rhs']:.3f} s`, "
            f"<=2% in `{one_object_2pct['reconstruction_core_s_including_rhs']:.3f} s`, and "
            f"<=1% in `{one_object_1pct['reconstruction_core_s_including_rhs']:.3f} s`.",
            "- Therefore the defensible wording is 10 Hz GPU-resident iterative updates for slabs up to 128 z planes, not 10 complete reconstructions per second.",
            "",
            "## Claim boundary",
            "",
            "- Full case는 256개 z plane 전체가 unknown이다.",
            "- Slab case는 표시된 centered z-support만 unknown이며, slab 밖의 contribution이 없거나 이미 제거되었다는 조건부 ROI reconstruction이다.",
            "- 모든 illumination과 모든 active detector harmonic을 사용한다. Pixel-to-mode 변환은 one-time preprocessing이며 normal operator를 보존한다.",
            "- Synthetic, noiseless, self-consistent inversion이다. 측정 noise, hologram demodulation, detector transfer는 포함하지 않는다.",
            "- Object NRMSE와 data residual을 함께 제시하며, missing-cone/conditioning 때문에 data consistency만으로 full object recovery를 주장하지 않는다.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args([])
    args.device = "cuda"
    args.dtype = "complex64"
    args.n_beta = 256
    args.n_r = 256
    args.n_z = 256
    args.ring_illum = 120
    args.cap_radial = 256
    args.cap_phi = 256
    args.h_cutoff = 28
    args.l_margin = 18
    args.cone_l_prune_threshold = 1e-12
    args.cpp_threads = 4
    args.skip_native_prepared_adjoint = True
    args.compact_axisymmetric_kernel = True
    args.radial_block_size = 32
    args.illumination_block_size = 4
    iterations = 100
    selected_counts = [256, 128, 64, 32, 16, 8, 4, 2, 1]

    torch = import_torch()
    device = resolve_device(torch, args.device)
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
        forward_mode="auto",
        adjoint_mode="auto",
        prune_axis_l0=True,
        axial_lowrank_rank=16,
        ring_adaptive_l_packed_threshold=1e-6,
    )
    synchronize(torch, device)
    plan_s = time.perf_counter() - plan_start
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)

    full_indices_np = np.arange(args.n_z, dtype=np.int64)
    full_indices = torch.as_tensor(full_indices_np, dtype=torch.long, device=device)
    full_truth_np = np.ascontiguousarray(
        np.real(context.ring.obj.coeff).astype(np_complex)
    )
    full_truth = torch.as_tensor(
        full_truth_np, dtype=plan.complex_dtype, device=device
    )
    with torch.inference_mode():
        full_pixel_data = plan.forward_selected_z(full_truth, full_indices)
        full_mode_data = plan.forward_selected_z_modes(full_truth, full_indices)
        preprocessed = pixel_to_modes(plan, full_pixel_data)
        preprocess_rel_l2 = relative_l2(torch, preprocessed, full_mode_data)
        pixel_normal = plan.adjoint_selected_z(full_pixel_data, full_indices)
        mode_normal = plan.adjoint_selected_z_modes(full_mode_data, full_indices)
        normal_rel_l2 = relative_l2(torch, mode_normal, pixel_normal)
        preprocess_times: list[float] = []
        for _ in range(5):
            pixel_to_modes(plan, full_pixel_data)
            synchronize(torch, device)
        for _ in range(30):
            synchronize(torch, device)
            start = time.perf_counter()
            pixel_to_modes(plan, full_pixel_data)
            synchronize(torch, device)
            preprocess_times.append(time.perf_counter() - start)
    preprocessing = timing_summary(preprocess_times)

    del preprocessed, pixel_normal, mode_normal
    cases: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for selected_n_z in selected_counts:
        indices_np = centered_z_indices(args.n_z, selected_n_z)
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
        truth_np = np.ascontiguousarray(
            np.real(context.ring.obj.coeff[:, indices_np, :]).astype(np_complex)
        )
        truth = torch.as_tensor(truth_np, dtype=plan.complex_dtype, device=device)
        with torch.inference_mode():
            data_modes = plan.forward_selected_z_modes(truth, indices)
            # One untimed solver-shaped warm-up for the current axial depth.
            warm_rhs = plan.adjoint_selected_z_modes(data_modes, indices)
            plan.forward_selected_z_modes(real_complex(torch, warm_rhs), indices)
            synchronize(torch, device)
            reconstruction, history = solve_real_cg_modes(
                torch=torch,
                plan=plan,
                z_index=indices,
                truth=truth,
                data_modes=data_modes,
                iterations=iterations,
                preprocessing_s=float(preprocessing["median_s"]),
            )
        case = {
            "selected_n_z": int(selected_n_z),
            "z_indices": indices_np.tolist(),
            "object_shape": list(truth.shape),
            "object_bins": int(truth.numel()),
            "data_mode_samples": int(data_modes.numel()),
            "all_illumination_views_used": True,
            "all_active_detector_modes_used": True,
            "known_z_support": bool(selected_n_z < args.n_z),
            "reconstruction": reconstruction,
        }
        cases.append(case)
        history_rows.extend(
            {"selected_n_z": selected_n_z, **row} for row in history
        )
        print(
            f"z={selected_n_z}: median={1000.0 * reconstruction['iteration_core_timing']['median_s']:.3f} ms, "
            f"core100={reconstruction['reconstruction_core_s_including_rhs']:.3f} s, "
            f"obj={reconstruction['final']['object_nrmse']:.4g}, "
            f"data={reconstruction['final']['data_residual']:.4g}",
            flush=True,
        )
        del indices, truth, data_modes, warm_rhs
        torch.cuda.empty_cache()

    result = {
        "schema": "odt-full-slab-reconstruction-claim-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": device_name(torch, device),
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(torch.version, "cuda", None),
        },
        "operator": {
            "h_cutoff": 28,
            "axial_lowrank_rank": 16,
            "axis_l_modes": plan.axis.n_l,
            "adaptive_l_threshold": 1e-6,
            "adaptive_l_active_fraction": plan.ring.adaptive_l_active_fraction,
            "dtype": args.dtype,
        },
        "problem": {
            "full_object_shape": [args.n_r, args.n_z, args.n_beta],
            "full_object_bins": int(np.prod([args.n_r, args.n_z, args.n_beta])),
            "illumination_count": int(args.ring_illum + 1),
            "pixel_detector_shape": [args.cap_radial, args.cap_phi],
            "pixel_q_samples": int(plan.q_count),
            "mode_q_samples": int(plan.mode_q_count),
            "mode_sample_fraction": float(plan.mode_q_count / plan.q_count),
            "phantom": args.phantom,
            "noise": "none",
            "object_constraint": "real-valued",
            "iterations": iterations,
        },
        "setup": {
            "context_build_s": context_s,
            "plan_build_s": plan_s,
        },
        "mode_preprocessing": {
            **preprocessing,
            "direct_mode_vs_preprocessed_rel_l2": preprocess_rel_l2,
            "normal_operator_rel_l2": normal_rel_l2,
            "one_time_per_repeated_geometry": True,
        },
        "cases": cases,
        "gpu_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        ),
        "claim_boundary": [
            "The full case treats all 256 z planes as unknown.",
            "Each slab case assumes that only the stated centered z support is unknown; outside-slab contributions are absent or already removed.",
            "All illumination views and all active detector harmonics are used.",
            "Pixel-to-mode conversion is a one-time exact normal-operator-preserving preprocessing step for repeated geometry.",
            "The inversion is synthetic, noiseless, and self-consistent; independent operator accuracy is reported separately.",
            "Object NRMSE and data residual are both reported because missing-cone conditioning limits what data consistency alone proves.",
        ],
    }
    out = ROOT / "benchmark_results" / "odt_full_slab_reconstruction_claim.json"
    csv_path = ROOT / "benchmark_results" / "odt_full_slab_reconstruction_claim_history.csv"
    md_path = ROOT / "benchmark_results" / "odt_full_slab_reconstruction_claim_ko.md"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    write_markdown(md_path, result)
    print(json.dumps({"output": str(out), "summary": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
