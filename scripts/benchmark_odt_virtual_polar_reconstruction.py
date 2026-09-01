from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
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
from benchmark_odt_cone_illumination import cone_illumination_directions  # noqa: E402
from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    import_cufinufft_modules,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)
from benchmark_odt_selected_z_gpu import (  # noqa: E402
    centered_z_indices,
    selected_object,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser as odt_parser,
    torch_dtypes,
)
from benchmark_odt_virtual_polar_detector import (  # noqa: E402
    CachedBilinearPolarRemap,
    cartesian_q_samples,
)


def norm(torch: Any, value: Any) -> Any:
    return torch.clamp(torch.linalg.vector_norm(value), min=1e-30)


def real_complex(torch: Any, value: Any) -> Any:
    real = torch.real(value)
    return torch.complex(real, torch.zeros_like(real))


def relative_l2(torch: Any, candidate: Any, reference: Any) -> float:
    return float((norm(torch, candidate - reference) / norm(torch, reference)).item())


def timing_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "total_s": float(array.sum()),
        "mean_s": float(array.mean()),
        "median_s": float(np.median(array)),
        "p05_s": float(np.percentile(array, 5)),
        "p95_s": float(np.percentile(array, 95)),
        "min_s": float(array.min()),
        "max_s": float(array.max()),
    }


def selected_truth(
    torch: Any,
    plan: TorchCompositeOdtPlan,
    context: Any,
    z_indices: np.ndarray,
    *,
    dtype: str,
) -> Any:
    _, _, np_complex, _ = torch_dtypes(torch, dtype)
    values = np.ascontiguousarray(
        np.real(context.ring.obj.coeff[:, z_indices, :]).astype(np_complex)
    )
    return torch.as_tensor(values, dtype=plan.complex_dtype, device=plan.device)


def direct_cartesian_remap(
    *,
    torch: Any,
    device: Any,
    args: argparse.Namespace,
    context: Any,
    z_indices: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """Generate direct Cartesian detector data in view blocks and remap to polar."""
    cp, cufinufft = import_cufinufft_modules()
    complex_dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    np_complex = np.complex64 if args.dtype == "complex64" else np.complex128
    np_real = np.float32 if args.dtype == "complex64" else np.float64
    cp_real = cp.float32 if args.dtype == "complex64" else cp.float64

    obj = selected_object(context.ring.obj, z_indices)
    coeff_np = np.ascontiguousarray(np.real(obj.coeff).astype(np_complex).ravel())
    x_src = cp.asarray(np.asarray(obj.x, dtype=np_real), dtype=cp_real)
    y_src = cp.asarray(np.asarray(obj.y, dtype=np_real), dtype=cp_real)
    z_src = cp.asarray(np.asarray(obj.z, dtype=np_real), dtype=cp_real)
    coeff = cp.asarray(coeff_np)

    remap = CachedBilinearPolarRemap(
        torch=torch,
        device=device,
        n_xy=args.camera_n_xy,
        n_radial=args.cap_radial,
        n_phi=args.cap_phi,
        complex_dtype=complex_dtype,
        radial_fraction=1.0,
        normalize_pupil_boundary=True,
    )
    ring_na = math.sin(math.radians(float(args.illumination_angle_deg)))
    ring_directions = cone_illumination_directions(
        n_illum=args.ring_illum,
        illumination_na=ring_na,
    )[0]

    output_parts: list[np.ndarray] = []
    block_rows: list[dict[str, Any]] = []
    active_reference: np.ndarray | None = None
    mask_reference: np.ndarray | None = None
    generation_start = time.perf_counter()

    direction_groups: list[tuple[str, np.ndarray]] = [("ring", ring_directions)]
    if not args.skip_axis_illumination:
        direction_groups.append(
            ("axis", np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64))
        )

    for group, directions in direction_groups:
        for view_start in range(0, directions.shape[0], args.reference_view_block):
            view_stop = min(view_start + args.reference_view_block, directions.shape[0])
            q, mask, active = cartesian_q_samples(
                k=args.k,
                detector_na=args.detector_na,
                n_xy=args.camera_n_xy,
                illumination=directions[view_start:view_stop],
            )
            if active_reference is None:
                active_reference = active
                mask_reference = mask
            elif not np.array_equal(active_reference, active) or not np.array_equal(
                mask_reference, mask
            ):
                raise RuntimeError("Cartesian detector mask changed between view blocks")

            block_start = time.perf_counter()
            qx = cp.asarray(np.asarray(q.qx, dtype=np_real), dtype=cp_real)
            qy = cp.asarray(np.asarray(q.qy, dtype=np_real), dtype=cp_real)
            qz = cp.asarray(np.asarray(q.qz, dtype=np_real), dtype=cp_real)
            values = cufinufft.nufft3d3(
                x_src,
                y_src,
                z_src,
                coeff,
                qx,
                qy,
                qz,
                eps=args.cufinufft_eps,
                isign=1,
            )
            cp.cuda.get_current_stream().synchronize()
            direct_s = time.perf_counter() - block_start
            values_np = np.asarray(values.get())

            view_count = int(view_stop - view_start)
            camera_np = np.zeros(
                (view_count, args.camera_n_xy * args.camera_n_xy), dtype=np_complex
            )
            camera_np[:, active] = values_np.reshape(view_count, active.size)
            camera = torch.as_tensor(
                camera_np.reshape(view_count, args.camera_n_xy, args.camera_n_xy),
                dtype=complex_dtype,
                device=device,
            )
            synchronize(torch, device)
            remap_start = time.perf_counter()
            polar = remap.gather_grid_sample(camera, batch_block=args.remap_view_block)
            synchronize(torch, device)
            remap_s = time.perf_counter() - remap_start
            output_parts.append(np.ascontiguousarray(polar.detach().cpu().numpy().ravel()))
            block_rows.append(
                {
                    "group": group,
                    "view_start": int(view_start),
                    "view_stop": int(view_stop),
                    "view_count": view_count,
                    "cartesian_q_count": int(q.count),
                    "direct_cufinufft_s": float(direct_s),
                    "remap_s": float(remap_s),
                }
            )
            del qx, qy, qz, values, values_np, camera_np, camera, polar, q
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

    if active_reference is None or mask_reference is None:
        raise RuntimeError("no Cartesian detector views were generated")
    output_np = np.concatenate(output_parts)
    output = torch.as_tensor(output_np, dtype=complex_dtype, device=device)
    synchronize(torch, device)
    metadata = {
        "generator": "direct cuFINUFFT type-3 forward, chunked by illumination",
        "generation_total_s": float(time.perf_counter() - generation_start),
        "reference_view_block": int(args.reference_view_block),
        "camera_n_xy": int(args.camera_n_xy),
        "active_pixels_per_view": int(active_reference.size),
        "active_pixel_fraction": float(active_reference.size / mask_reference.size),
        "polar_samples_per_view": int(args.cap_radial * args.cap_phi),
        "output_q_count": int(output.numel()),
        "direct_cufinufft_total_s": float(
            sum(row["direct_cufinufft_s"] for row in block_rows)
        ),
        "remap_total_s": float(sum(row["remap_s"] for row in block_rows)),
        "blocks": block_rows,
    }
    del x_src, y_src, z_src, coeff, output_np, output_parts, remap
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    return output, metadata


def selected_data_lipschitz(
    *,
    torch: Any,
    plan: TorchCompositeOdtPlan,
    z_index: Any,
    selected_n_z: int,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    real = torch.randn(
        (plan.ring.n_r, selected_n_z, plan.ring.n_beta),
        dtype=plan.real_dtype,
        device=plan.device,
    )
    value = torch.complex(real, torch.zeros_like(real))
    value = value / norm(torch, value)
    estimate = 0.0
    synchronize(torch, plan.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            normal = real_complex(
                torch,
                plan.adjoint_selected_z(
                    plan.forward_selected_z(value, z_index), z_index
                ),
            )
            estimate = float(
                torch.real(torch.vdot(value.reshape(-1), normal.reshape(-1))).item()
            )
            value = normal / norm(torch, normal)
    synchronize(torch, plan.device)
    return estimate, float(time.perf_counter() - start)


def solve_selected_fista(
    *,
    torch: Any,
    plan: TorchCompositeOdtPlan,
    z_index: Any,
    data: Any,
    truth: Any,
    label: str,
    data_lipschitz: float,
    iterations: int,
    record_every: int,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    lipschitz = 1.10 * float(data_lipschitz)
    step = 1.0 / lipschitz
    x = torch.zeros_like(truth)
    y = x.clone()
    momentum = 1.0
    data_norm = norm(torch, data)
    truth_norm = norm(torch, truth)
    history: list[dict[str, Any]] = [
        {
            "input": label,
            "iteration": 0,
            "data_residual": 1.0,
            "object_rel_l2": 1.0,
            "core_cumulative_s": 0.0,
        }
    ]
    core_times: list[float] = []
    restart_count = 0
    solve_start = time.perf_counter()
    with torch.inference_mode():
        for iteration in range(1, iterations + 1):
            synchronize(torch, plan.device)
            core_start = time.perf_counter()
            residual = plan.forward_selected_z(y, z_index) - data
            gradient = real_complex(
                torch, plan.adjoint_selected_z(residual, z_index)
            )
            x_new_real = torch.clamp(torch.real(y - step * gradient), min=0.0)
            x_new = torch.complex(x_new_real, torch.zeros_like(x_new_real))
            momentum_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
            y_new = x_new + ((momentum - 1.0) / momentum_new) * (x_new - x)
            restart = torch.real(
                torch.vdot(
                    (x_new - x).reshape(-1), (y_new - x_new).reshape(-1)
                )
            ) > 0
            if bool(restart.item()):
                momentum_new = 1.0
                y_new = x_new
                restart_count += 1
            x, y, momentum = x_new, y_new, momentum_new
            synchronize(torch, plan.device)
            core_times.append(float(time.perf_counter() - core_start))

            if iteration % record_every == 0 or iteration == iterations:
                prediction = plan.forward_selected_z(x, z_index)
                synchronize(torch, plan.device)
                history.append(
                    {
                        "input": label,
                        "iteration": int(iteration),
                        "data_residual": float(
                            (norm(torch, prediction - data) / data_norm).item()
                        ),
                        "object_rel_l2": float(
                            (norm(torch, x - truth) / truth_norm).item()
                        ),
                        "core_cumulative_s": float(sum(core_times)),
                    }
                )
    prediction = plan.forward_selected_z(x, z_index)
    synchronize(torch, plan.device)
    result = {
        "input": label,
        "iterations": int(iterations),
        "step": float(step),
        "lipschitz_with_margin": float(lipschitz),
        "adaptive_restart_count": int(restart_count),
        "object_rel_l2": float((norm(torch, x - truth) / truth_norm).item()),
        "data_residual": float((norm(torch, prediction - data) / data_norm).item()),
        "core_iteration_timing": timing_summary(core_times),
        "core_solve_s": float(sum(core_times)),
        "wall_solve_s_including_diagnostics": float(time.perf_counter() - solve_start),
    }
    return result, x, history


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
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

    selected_counts = [int(value) for value in args.selected_slices.split(",")]
    cases: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for selected_n_z in selected_counts:
        z_indices = centered_z_indices(args.n_z, selected_n_z)
        z_index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
        truth = selected_truth(
            torch, plan, context, z_indices, dtype=args.dtype
        )
        with torch.inference_mode():
            ideal_data = plan.forward_selected_z(truth, z_index)
        synchronize(torch, device)
        remapped_data, generation = direct_cartesian_remap(
            torch=torch,
            device=device,
            args=args,
            context=context,
            z_indices=z_indices,
        )
        if remapped_data.shape != ideal_data.shape:
            raise RuntimeError(
                f"remapped data shape {tuple(remapped_data.shape)} does not match "
                f"ideal polar data shape {tuple(ideal_data.shape)}"
            )
        data_difference = relative_l2(torch, remapped_data, ideal_data)
        lipschitz, power_s = selected_data_lipschitz(
            torch=torch,
            plan=plan,
            z_index=z_index,
            selected_n_z=selected_n_z,
            iterations=args.power_iterations,
            seed=args.seed + 1009 + selected_n_z,
        )
        ideal_result, ideal_reconstruction, ideal_history = solve_selected_fista(
            torch=torch,
            plan=plan,
            z_index=z_index,
            data=ideal_data,
            truth=truth,
            label="ideal_polar",
            data_lipschitz=lipschitz,
            iterations=args.iterations,
            record_every=args.record_every,
        )
        remap_result, remap_reconstruction, remap_history = solve_selected_fista(
            torch=torch,
            plan=plan,
            z_index=z_index,
            data=remapped_data,
            truth=truth,
            label="cartesian_remap",
            data_lipschitz=lipschitz,
            iterations=args.iterations,
            record_every=args.record_every,
        )
        reconstruction_difference = relative_l2(
            torch, remap_reconstruction, ideal_reconstruction
        )
        for row in ideal_history + remap_history:
            history_rows.append({"selected_n_z": selected_n_z, **row})
        cases.append(
            {
                "selected_n_z": selected_n_z,
                "z_indices": z_indices.tolist(),
                "selected_object_shape": list(truth.shape),
                "selected_object_bins": int(truth.numel()),
                "all_views_and_polar_samples_used": True,
                "q_samples": int(ideal_data.numel()),
                "cartesian_remap_rel_l2_vs_ideal_polar": float(data_difference),
                "data_lipschitz_estimate": float(lipschitz),
                "power_iteration_s": float(power_s),
                "reference_generation": generation,
                "ideal_polar": ideal_result,
                "cartesian_remap": remap_result,
                "remap_vs_ideal_reconstruction_rel_l2": float(
                    reconstruction_difference
                ),
            }
        )
        del (
            z_index,
            truth,
            ideal_data,
            remapped_data,
            ideal_reconstruction,
            remap_reconstruction,
        )
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "schema": "odt-virtual-polar-selected-z-reconstruction-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": device_name(torch, device),
        "dtype": args.dtype,
        "problem": {
            "full_object_shape": [args.n_r, args.n_z, args.n_beta],
            "ring_illumination_count": int(args.ring_illum),
            "axis_illumination_included": not args.skip_axis_illumination,
            "total_illumination_count": int(args.ring_illum)
            + (0 if args.skip_axis_illumination else 1),
            "cartesian_detector_shape": [args.camera_n_xy, args.camera_n_xy],
            "polar_detector_shape": [args.cap_radial, args.cap_phi],
            "noise": "none",
            "constraint": "real nonnegative",
            "solver": "selected-z adaptive-restart FISTA",
            "iterations": int(args.iterations),
            "power_iterations": int(args.power_iterations),
        },
        "timing_scope": {
            "context_build_s": float(context_s),
            "acfo_plan_build_s": float(plan_s),
            "reconstruction_core_excludes": [
                "geometry/context setup",
                "ACFO plan setup",
                "direct Cartesian reference generation",
                "Cartesian-to-polar remap",
                "diagnostic forward passes",
            ],
            "reference_generation_is_one_off_validation_work": True,
        },
        "claim_boundary": [
            "This is an actually iterated selected-z reconstruction using all illumination views and all detector samples.",
            "It is not a full 256x256x256 reconstruction: only the stated centered z support is unknown.",
            "Ideal polar and remapped Cartesian inputs use the same ACFO plan, initialization, step, constraint, and iteration count.",
            "No measurement noise or hologram demodulation is included in this first chained validation.",
        ],
        "cases": cases,
        "gpu_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "gpu_peak_scope": (
            "PyTorch allocator peak only; excludes transient CuPy/cuFINUFFT "
            "reference-generation allocations."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.history_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history_rows for key in row})
    with args.history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)
    return result


def parser() -> argparse.ArgumentParser:
    p = odt_parser()
    p.description = (
        "Chained Cartesian detector -> polar remap -> selected-z ACFO reconstruction."
    )
    p.add_argument("--camera-n-xy", type=int, default=320)
    p.add_argument("--selected-slices", default="1,8")
    p.add_argument("--power-iterations", type=int, default=8)
    p.add_argument("--record-every", type=int, default=5)
    p.add_argument("--reference-view-block", type=int, default=8)
    p.add_argument("--remap-view-block", type=int, default=0)
    p.add_argument("--cufinufft-eps", type=float, default=1e-6)
    p.add_argument(
        "--history-csv",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_selected_z_reconstruction_history.csv",
    )
    p.set_defaults(
        device="cuda",
        dtype="complex64",
        low_memory_adjoint=True,
        radial_block_size=16,
        illumination_block_size=4,
        forward_mode="illumination-reduced",
        adjoint_mode="illumination-reduced",
        skip_native_prepared_adjoint=True,
        compact_axisymmetric_kernel=True,
        real_object=True,
        n_beta=256,
        n_r=256,
        n_z=256,
        r_max=1.0,
        z_max=0.8,
        phantom="random_beads",
        seed=123,
        ring_illum=120,
        cap_radial=256,
        cap_phi=256,
        h_margin=20,
        l_margin=18,
        cpp_threads=4,
        iterations=60,
        output=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_selected_z_reconstruction.json",
        out=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_selected_z_reconstruction.json",
        summary_md=None,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_selected_z_reconstruction.json",
    )
    return p


def validate_args(args: argparse.Namespace) -> None:
    selected = [int(value) for value in args.selected_slices.split(",")]
    if not selected or any(value <= 0 or value > args.n_z for value in selected):
        raise ValueError("selected-slices must contain values in [1, n-z]")
    if args.iterations <= 0 or args.power_iterations <= 0 or args.record_every <= 0:
        raise ValueError("iteration counts must be positive")
    if args.reference_view_block <= 0:
        raise ValueError("reference-view-block must be positive")
    if args.camera_n_xy < 2:
        raise ValueError("camera-n-xy must be at least 2")


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    result = run(args)
    compact = {
        "schema": result["schema"],
        "device_name": result["device_name"],
        "problem": result["problem"],
        "cases": [
            {
                "selected_n_z": row["selected_n_z"],
                "remap_data_rel_l2": row[
                    "cartesian_remap_rel_l2_vs_ideal_polar"
                ],
                "ideal_object_rel_l2": row["ideal_polar"]["object_rel_l2"],
                "remap_object_rel_l2": row["cartesian_remap"]["object_rel_l2"],
                "reconstruction_rel_l2": row[
                    "remap_vs_ideal_reconstruction_rel_l2"
                ],
                "ideal_median_core_iter_s": row["ideal_polar"][
                    "core_iteration_timing"
                ]["median_s"],
                "remap_median_core_iter_s": row["cartesian_remap"][
                    "core_iteration_timing"
                ]["median_s"],
            }
            for row in result["cases"]
        ],
        "gpu_peak_allocated_mib": result["gpu_peak_allocated_mib"],
        "output": str(args.output),
        "history_csv": str(args.history_csv),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
