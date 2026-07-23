from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
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
from benchmark_odt_banded_cartesian_final_packed import (  # noqa: E402
    banded_pixel_to_modes,
    candidate_contract,
    relative_l2,
)
from benchmark_odt_banded_cartesian_reconstruction import (  # noqa: E402
    make_remap_plans,
    remap_camera,
)
from benchmark_odt_banded_detector import (  # noqa: E402
    TorchBandedCompositePlan,
    VARIANTS,
    build_variant,
)
from benchmark_odt_cone_illumination import (  # noqa: E402
    cone_illumination_directions,
)
from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    import_cufinufft_modules,
)
from benchmark_odt_gpu_warm_start_dynamic import (  # noqa: E402
    DynamicGrid,
    dynamic_coefficients,
    parse_int_list,
)
from benchmark_odt_selected_z_gpu import (  # noqa: E402
    centered_z_indices,
    selected_object,
)
from benchmark_odt_virtual_polar_detector import (  # noqa: E402
    cartesian_q_samples,
)


@dataclass
class UpdateResult:
    x: Any
    pred: Any
    row: dict[str, Any]


def build_selected_grid(context: Any, z_indices: np.ndarray) -> DynamicGrid:
    obj = context.ring.obj
    r_axis = np.asarray(obj.r_axis, dtype=np.float64)
    z_axis = np.asarray(obj.z_axis, dtype=np.float64)[z_indices]
    beta_axis = np.asarray(obj.beta_axis, dtype=np.float64)
    rr, zz, bb = np.meshgrid(r_axis, z_axis, beta_axis, indexing="ij")
    full_shape = (
        int(np.asarray(obj.r_axis).size),
        int(np.asarray(obj.z_axis).size),
        int(np.asarray(obj.beta_axis).size),
    )
    volume = np.asarray(obj.volume_weights, dtype=np.float64).reshape(full_shape)
    volume = np.ascontiguousarray(volume[:, z_indices, :])
    return DynamicGrid(
        x=np.ascontiguousarray(rr * np.cos(bb)),
        y=np.ascontiguousarray(rr * np.sin(bb)),
        z=np.ascontiguousarray(zz),
        r_axis=r_axis,
        z_axis=z_axis,
        beta_axis=beta_axis,
        volume=volume,
    )


def direct_cartesian_camera_from_coeff(
    *,
    torch: Any,
    device: Any,
    args: argparse.Namespace,
    context: Any,
    z_indices: np.ndarray,
    coeff_np: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    cp, cufinufft = import_cufinufft_modules()
    np_complex = np.complex64 if args.dtype == "complex64" else np.complex128
    np_real = np.float32 if args.dtype == "complex64" else np.float64
    cp_real = cp.float32 if args.dtype == "complex64" else cp.float64
    complex_dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    obj = selected_object(context.ring.obj, z_indices)
    if tuple(coeff_np.shape) != tuple(obj.coeff.shape):
        raise ValueError("dynamic coefficient shape does not match selected object")
    x_src = cp.asarray(np.asarray(obj.x, dtype=np_real), dtype=cp_real)
    y_src = cp.asarray(np.asarray(obj.y, dtype=np_real), dtype=cp_real)
    z_src = cp.asarray(np.asarray(obj.z, dtype=np_real), dtype=cp_real)
    coeff = cp.asarray(np.ascontiguousarray(coeff_np.astype(np_complex).ravel()))
    ring_na = math.sin(math.radians(float(args.illumination_angle_deg)))
    ring_directions = cone_illumination_directions(
        n_illum=args.ring_illum, illumination_na=ring_na
    )[0]
    groups = [("ring", ring_directions)]
    if not args.skip_axis_illumination:
        groups.append(("axis", np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)))
    views = int(args.ring_illum) + (0 if args.skip_axis_illumination else 1)
    camera_np = np.zeros(
        (views, args.camera_n_xy, args.camera_n_xy), dtype=np_complex
    )
    rows: list[dict[str, Any]] = []
    view_offset = 0
    active_reference = None
    total_start = time.perf_counter()
    for group, directions in groups:
        for start in range(0, directions.shape[0], args.reference_view_block):
            stop = min(start + args.reference_view_block, directions.shape[0])
            q, _mask, active = cartesian_q_samples(
                k=args.k,
                detector_na=args.detector_na,
                n_xy=args.camera_n_xy,
                illumination=directions[start:stop],
            )
            if active_reference is None:
                active_reference = active
            elif not np.array_equal(active_reference, active):
                raise RuntimeError("active Cartesian detector indices changed")
            execute_start = time.perf_counter()
            values = cufinufft.nufft3d3(
                x_src,
                y_src,
                z_src,
                coeff,
                cp.asarray(np.asarray(q.qx, dtype=np_real), dtype=cp_real),
                cp.asarray(np.asarray(q.qy, dtype=np_real), dtype=cp_real),
                cp.asarray(np.asarray(q.qz, dtype=np_real), dtype=cp_real),
                eps=args.cufinufft_eps,
                isign=1,
            )
            cp.cuda.runtime.deviceSynchronize()
            execute_s = time.perf_counter() - execute_start
            values_np = np.asarray(values.get()).reshape(stop - start, active.size)
            block = camera_np[
                view_offset + start : view_offset + stop
            ].reshape(stop - start, -1)
            block[:, active] = values_np
            rows.append(
                {
                    "group": group,
                    "view_start": int(start),
                    "view_stop": int(stop),
                    "execute_s": float(execute_s),
                }
            )
            del q, values, values_np
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        view_offset += directions.shape[0]
    camera = torch.as_tensor(camera_np, dtype=complex_dtype, device=device)
    synchronize(torch, device)
    metadata = {
        "generation_total_s": float(time.perf_counter() - total_start),
        "direct_cufinufft_execute_s": float(sum(row["execute_s"] for row in rows)),
        "active_pixels_per_view": int(active_reference.size),
        "blocks": rows,
    }
    del x_src, y_src, z_src, coeff, camera_np
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    return camera, metadata


def norm2(torch: Any, value: Any) -> Any:
    return torch.sum(torch.real(torch.conj(value) * value))


def run_updates(
    *,
    torch: Any,
    plan: TorchBandedCompositePlan,
    z_index: Any,
    data: Any,
    truth: Any,
    updates: int,
    x_init: Any | None,
    pred_init: Any | None,
    mode: str,
    frame: int,
    preprocessing_s: float,
) -> UpdateResult:
    x = torch.zeros_like(truth) if x_init is None else x_init.clone()
    pred = torch.zeros_like(data) if pred_init is None else pred_init.clone()
    data_norm = torch.clamp(torch.linalg.vector_norm(data), min=1e-30)
    truth_norm = torch.clamp(torch.linalg.vector_norm(truth), min=1e-30)
    update_times: list[float] = []
    alpha = torch.zeros((), dtype=truth.real.dtype, device=truth.device)
    with torch.inference_mode():
        for _ in range(updates):
            synchronize(torch, plan.device)
            start = time.perf_counter()
            residual = pred - data
            grad = plan.adjoint_selected_z_modes(residual, z_index)
            a_grad = plan.forward_selected_z_modes(grad, z_index)
            alpha = norm2(torch, grad) / torch.clamp(norm2(torch, a_grad), min=1e-30)
            x = x - alpha * grad
            pred = pred - alpha * a_grad
            synchronize(torch, plan.device)
            update_times.append(time.perf_counter() - start)
    update_s = float(sum(update_times))
    total_hot_s = float(preprocessing_s + update_s)
    row = {
        "mode": mode,
        "frame": int(frame),
        "updates": int(updates),
        "preprocessing_s": float(preprocessing_s),
        "update_s": update_s,
        "total_hot_s": total_hot_s,
        "hot_hz": float(1.0 / total_hot_s) if total_hot_s > 0.0 else None,
        "per_update_median_s": (
            None if not update_times else float(median(update_times))
        ),
        "object_rel_l2": float(
            (torch.linalg.vector_norm(x - truth) / truth_norm).item()
        ),
        "data_residual_rel_l2": float(
            (torch.linalg.vector_norm(pred - data) / data_norm).item()
        ),
        "alpha_last": float(alpha.item()),
    }
    return UpdateResult(x=x, pred=pred, row=row)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["mode"]), int(row["updates"])), []).append(row)
    output = []
    for (mode, updates), group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["frame"]))
        total = [float(row["total_hot_s"]) for row in group]
        errors = [float(row["object_rel_l2"]) for row in group]
        residuals = [float(row["data_residual_rel_l2"]) for row in group]
        output.append(
            {
                "mode": mode,
                "updates": updates,
                "frames": len(group),
                "median_total_hot_s": float(median(total)),
                "median_hot_hz": float(1.0 / median(total)),
                "mean_object_rel_l2": float(np.mean(errors)),
                "final_object_rel_l2": float(errors[-1]),
                "mean_data_residual_rel_l2": float(np.mean(residuals)),
                "final_data_residual_rel_l2": float(residuals[-1]),
            }
        )
    by_key = {(row["mode"], row["updates"]): row for row in output}
    reference = by_key.get(("reference", 20))
    for row in output:
        if row["mode"] != "warm_start":
            continue
        cold = by_key.get(("cold_start", row["updates"]))
        if cold is not None:
            row["mean_object_error_vs_cold_ratio"] = float(
                row["mean_object_rel_l2"] / cold["mean_object_rel_l2"]
            )
        if reference is not None:
            row["mean_object_error_vs_reference_ratio"] = float(
                row["mean_object_rel_l2"] / reference["mean_object_rel_l2"]
            )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS:
        raise ValueError(f"unknown variant {args.variant}")
    updates_per_frame = parse_int_list(args.updates_per_frame)
    if any(value <= 0 for value in updates_per_frame):
        raise ValueError("updates-per-frame must be positive")
    if args.frames < 3 or args.selected_n_z <= 0 or args.selected_n_z > args.n_z:
        raise ValueError("invalid frames or selected-n-z")
    if args.reference_iterations <= 0 or args.initial_iterations <= 0:
        raise ValueError("initial/reference iterations must be positive")

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    build_start = time.perf_counter()
    plan, context, bands, context_s, plan_s = build_variant(
        torch=torch, device=device, base_args=args, label=args.variant
    )
    remap_plans = make_remap_plans(
        torch=torch, device=device, args=args, variant=args.variant
    )
    synchronize(torch, device)
    total_build_s = time.perf_counter() - build_start
    z_indices = centered_z_indices(args.n_z, args.selected_n_z)
    z_index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
    grid = build_selected_grid(context, z_indices)

    truths: list[Any] = []
    data_modes: list[Any] = []
    frame_metadata: list[dict[str, Any]] = []
    for frame in range(args.frames):
        coeff_np = dynamic_coefficients(
            grid,
            frame=frame,
            frames=args.frames,
            object_scale=args.object_scale,
            motion_fraction=args.motion_fraction,
            phase_drift_rad=args.phase_drift_rad,
            np_complex=np.complex64 if args.dtype == "complex64" else np.complex128,
        )
        truth = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
        with torch.inference_mode():
            ideal_modes = plan.forward_selected_z_modes(truth, z_index)
        camera, camera_meta = direct_cartesian_camera_from_coeff(
            torch=torch,
            device=device,
            args=args,
            context=context,
            z_indices=z_indices,
            coeff_np=coeff_np,
        )
        synchronize(torch, device)
        preprocess_start = time.perf_counter()
        with torch.inference_mode():
            remapped_pixel = remap_camera(
                torch=torch, camera=camera, plans=remap_plans
            )
            modes = banded_pixel_to_modes(plan, remapped_pixel)
        synchronize(torch, device)
        preprocessing_s = time.perf_counter() - preprocess_start
        mode_error = relative_l2(torch, modes, ideal_modes)
        truths.append(truth)
        data_modes.append(modes)
        frame_metadata.append(
            {
                "frame": frame,
                "direct_camera": camera_meta,
                "preprocessing_s": float(preprocessing_s),
                "preprocessing_hz": float(1.0 / preprocessing_s),
                "remapped_modes_vs_candidate_ideal_rel_l2": float(mode_error),
            }
        )
        print(
            f"frame={frame}: direct={camera_meta['generation_total_s']:.3f} s, "
            f"preprocess={1000.0 * preprocessing_s:.3f} ms, "
            f"mode-error={mode_error:.3e}",
            flush=True,
        )
        del camera, remapped_pixel, ideal_modes, coeff_np
        gc.collect()
        torch.cuda.empty_cache()

    # Re-prime the final packed path after validation-only cuFINUFFT generation.
    with torch.inference_mode():
        prime = plan.forward_selected_z_modes(truths[0], z_index)
        plan.adjoint_selected_z_modes(prime, z_index)
    synchronize(torch, device)

    initial = run_updates(
        torch=torch,
        plan=plan,
        z_index=z_index,
        data=data_modes[0],
        truth=truths[0],
        updates=args.initial_iterations,
        x_init=None,
        pred_init=None,
        mode="initial",
        frame=0,
        preprocessing_s=frame_metadata[0]["preprocessing_s"],
    )
    rows: list[dict[str, Any]] = [initial.row]
    for updates in updates_per_frame:
        x = initial.x.clone()
        pred = initial.pred.clone()
        for frame in range(1, args.frames):
            result = run_updates(
                torch=torch,
                plan=plan,
                z_index=z_index,
                data=data_modes[frame],
                truth=truths[frame],
                updates=updates,
                x_init=x,
                pred_init=pred,
                mode="warm_start",
                frame=frame,
                preprocessing_s=frame_metadata[frame]["preprocessing_s"],
            )
            rows.append(result.row)
            x, pred = result.x, result.pred
    for updates in updates_per_frame:
        for frame in range(1, args.frames):
            result = run_updates(
                torch=torch,
                plan=plan,
                z_index=z_index,
                data=data_modes[frame],
                truth=truths[frame],
                updates=updates,
                x_init=None,
                pred_init=None,
                mode="cold_start",
                frame=frame,
                preprocessing_s=frame_metadata[frame]["preprocessing_s"],
            )
            rows.append(result.row)
    for frame in range(1, args.frames):
        result = run_updates(
            torch=torch,
            plan=plan,
            z_index=z_index,
            data=data_modes[frame],
            truth=truths[frame],
            updates=args.reference_iterations,
            x_init=None,
            pred_init=None,
            mode="reference",
            frame=frame,
            preprocessing_s=frame_metadata[frame]["preprocessing_s"],
        )
        rows.append(result.row)

    summary_rows = aggregate(rows)
    by_key = {(row["mode"], row["updates"]): row for row in summary_rows}
    warm1 = by_key[("warm_start", 1)]
    warm3 = by_key.get(("warm_start", 3), warm1)
    gates = {
        "all_frame_mode_errors_le_1e-2": bool(
            all(
                row["remapped_modes_vs_candidate_ideal_rel_l2"] <= 1e-2
                for row in frame_metadata
            )
        ),
        "warm_1_update_median_hot_at_least_10_hz": bool(
            warm1["median_hot_hz"] >= 10.0
        ),
        "warm_1_update_final_object_error_le_0_15": bool(
            warm1["final_object_rel_l2"] <= 0.15
        ),
        "warm_1_update_mean_error_better_than_cold": bool(
            warm1.get("mean_object_error_vs_cold_ratio", float("inf")) < 1.0
        ),
        "warm_3_update_mean_error_within_1_25x_reference": bool(
            warm3.get("mean_object_error_vs_reference_ratio", float("inf"))
            <= 1.25
        ),
    }
    result = {
        "schema": "odt-banded-cartesian-temporal-warm-start-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": device_name(torch, device),
            "torch": getattr(torch, "__version__", None),
            "dtype": args.dtype,
        },
        "detector": {
            "variant": args.variant,
            "camera_shape_per_view": [args.camera_n_xy, args.camera_n_xy],
            "bands": bands,
            "views": int(args.ring_illum)
            + (0 if args.skip_axis_illumination else 1),
        },
        "candidate": {
            **candidate_contract(args),
            "context_build_s": float(context_s),
            "plan_build_s": float(plan_s),
            "total_plan_and_remap_build_s": float(total_build_s),
        },
        "sequence": {
            "frames": int(args.frames),
            "selected_n_z": int(args.selected_n_z),
            "object_shape": list(truths[0].shape),
            "motion_fraction": float(args.motion_fraction),
            "phase_drift_rad": float(args.phase_drift_rad),
            "noise": "none",
            "updates_per_frame": updates_per_frame,
            "initial_iterations": int(args.initial_iterations),
            "reference_iterations": int(args.reference_iterations),
        },
        "frame_data": frame_metadata,
        "initial": initial.row,
        "summary_rows": summary_rows,
        "history": rows,
        "gpu_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "gates": gates,
        "passed": bool(all(gates.values())),
        "claim_boundary": [
            "Every frame begins with a physical Cartesian detector field generated independently by cuFINUFFT type-3, followed by the cached banded remap and angular-mode reduction used by the final packed operator.",
            "Direct camera generation is validation-only setup and is excluded; the reported frame hot time is measured GPU-resident remap/mode preprocessing plus the requested number of final packed gradient updates.",
            "The frozen sequence is a small-motion, noiseless synthetic tracking case. It demonstrates warm-start feasibility, not experimental robustness or a universal convergence guarantee.",
            "The initial frame is reconstructed from a cold start with the stated initial iteration count; subsequent warm rows reuse both x and A x rather than an oracle truth state.",
            "The 20-update cold reconstruction is a numerical reference on the same approximate/remapped operator, not an independent ground-truth inverse solution.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.summary_md:
        lines = [
            "# ODT integrated Cartesian temporal warm-start",
            "",
            f"- passed: `{result['passed']}`",
            f"- sequence: `{args.frames}` frames, `256 x {args.selected_n_z} x 256`",
            f"- motion fraction / phase drift: `{args.motion_fraction}` / `{args.phase_drift_rad}` rad",
            f"- GPU peak allocated: `{result['gpu_peak_allocated_mib']:.2f}` MiB",
            "",
            "| mode | updates | median hot ms | hot Hz | mean object L2 | final object L2 | mean data residual |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary_rows:
            lines.append(
                f"| {row['mode']} | {row['updates']} | "
                f"{1000.0 * row['median_total_hot_s']:.3f} | "
                f"{row['median_hot_hz']:.3f} | "
                f"{row['mean_object_rel_l2']:.4e} | "
                f"{row['final_object_rel_l2']:.4e} | "
                f"{row['mean_data_residual_rel_l2']:.4e} |"
            )
        lines.extend(["", "## Gates", ""])
        lines.extend(
            f"- {name}: `{'PASS' if value else 'FAIL'}`"
            for name, value in gates.items()
        )
        lines.extend(["", "## Claim boundary", ""])
        lines.extend(f"- {item}" for item in result["claim_boundary"])
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    from benchmark_odt_banded_cartesian_final_packed import parser as base_parser

    p = base_parser()
    p.description = (
        "Frozen physical Cartesian-camera temporal sequence through the final "
        "banded packed ODT operator."
    )
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--selected-n-z", type=int, default=8)
    p.add_argument("--updates-per-frame", default="1,2,3,5")
    p.add_argument("--initial-iterations", type=int, default=20)
    p.add_argument("--reference-iterations", type=int, default=20)
    p.add_argument("--motion-fraction", type=float, default=0.01)
    p.add_argument("--phase-drift-rad", type=float, default=0.02)
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
        camera_n_xy=320,
        reference_view_block=8,
        cufinufft_eps=1e-6,
        summary_md=ROOT
        / "benchmark_results"
        / "odt_banded_cartesian_temporal_warm_start_ko.md",
        output=ROOT
        / "benchmark_results"
        / "odt_banded_cartesian_temporal_warm_start.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    compact = {
        "passed": result["passed"],
        "gates": result["gates"],
        "summary_rows": result["summary_rows"],
        "gpu_peak_allocated_mib": result["gpu_peak_allocated_mib"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
