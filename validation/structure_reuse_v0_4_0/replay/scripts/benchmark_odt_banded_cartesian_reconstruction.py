from __future__ import annotations

import argparse
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
from benchmark_odt_banded_detector import (  # noqa: E402
    VARIANTS,
    build_variant,
)
from benchmark_odt_cone_illumination import cone_illumination_directions  # noqa: E402
from benchmark_odt_cufinufft_gpu_baseline import import_cufinufft_modules  # noqa: E402
from benchmark_odt_ewald_cap_operator import detector_radial_nodes  # noqa: E402
from benchmark_odt_selected_z_gpu import (  # noqa: E402
    centered_z_indices,
    selected_object,
)
from benchmark_odt_virtual_polar_detector import (  # noqa: E402
    CachedBilinearPolarRemap,
    cartesian_q_samples,
)
from benchmark_odt_virtual_polar_reconstruction import (  # noqa: E402
    parser as chained_parser,
    selected_truth,
    solve_selected_fista,
)


def norm(torch: Any, value: Any) -> Any:
    return torch.clamp(torch.linalg.vector_norm(value), min=1e-30)


def direct_cartesian_camera(
    *,
    torch: Any,
    device: Any,
    args: argparse.Namespace,
    context: Any,
    z_indices: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    cp, cufinufft = import_cufinufft_modules()
    np_complex = np.complex64 if args.dtype == "complex64" else np.complex128
    np_real = np.float32 if args.dtype == "complex64" else np.float64
    cp_real = cp.float32 if args.dtype == "complex64" else cp.float64
    complex_dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    obj = selected_object(context.ring.obj, z_indices)
    x_src = cp.asarray(np.asarray(obj.x, dtype=np_real), dtype=cp_real)
    y_src = cp.asarray(np.asarray(obj.y, dtype=np_real), dtype=cp_real)
    z_src = cp.asarray(np.asarray(obj.z, dtype=np_real), dtype=cp_real)
    coeff = cp.asarray(
        np.ascontiguousarray(np.real(obj.coeff).astype(np_complex).ravel())
    )
    ring_na = math.sin(math.radians(float(args.illumination_angle_deg)))
    ring_directions = cone_illumination_directions(
        n_illum=args.ring_illum, illumination_na=ring_na
    )[0]
    groups = [("ring", ring_directions)]
    if not args.skip_axis_illumination:
        groups.append(("axis", np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)))

    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    active_reference = None
    mask_reference = None
    start_total = time.perf_counter()
    for group, directions in groups:
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
            elif not np.array_equal(active_reference, active):
                raise RuntimeError("Cartesian active detector indices changed")
            start = time.perf_counter()
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
            cp.cuda.get_current_stream().synchronize()
            execute_s = time.perf_counter() - start
            values_np = np.asarray(values.get())
            view_count = view_stop - view_start
            block = np.zeros(
                (view_count, args.camera_n_xy * args.camera_n_xy), dtype=np_complex
            )
            block[:, active] = values_np.reshape(view_count, active.size)
            frames.append(block.reshape(view_count, args.camera_n_xy, args.camera_n_xy))
            rows.append(
                {
                    "group": group,
                    "view_start": int(view_start),
                    "view_stop": int(view_stop),
                    "direct_cufinufft_s": float(execute_s),
                }
            )
            del q, values, values_np, block
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
    camera_np = np.ascontiguousarray(np.concatenate(frames, axis=0))
    camera = torch.as_tensor(camera_np, dtype=complex_dtype, device=device)
    synchronize(torch, device)
    metadata = {
        "generation_total_s": float(time.perf_counter() - start_total),
        "direct_cufinufft_total_s": float(
            sum(row["direct_cufinufft_s"] for row in rows)
        ),
        "camera_shape": list(camera.shape),
        "active_pixels_per_view": int(active_reference.size),
        "camera_input_mib": float(camera.numel() * camera.element_size() / 1024**2),
        "blocks": rows,
    }
    del x_src, y_src, z_src, coeff, camera_np, frames
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    return camera, metadata


def make_remap_plans(
    *, torch: Any, device: Any, args: argparse.Namespace, variant: str
) -> list[CachedBilinearPolarRemap]:
    complex_dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    plans = []
    for band in VARIANTS[variant]:
        radial = detector_radial_nodes(
            detector_na=args.detector_na,
            cap_radial=band["cap_radial"],
            sampling=band["sampling"],
            outer_power=band["outer_power"],
            min_fraction=band["min_fraction"],
            max_fraction=band["max_fraction"],
        ) / float(args.detector_na)
        plans.append(
            CachedBilinearPolarRemap(
                torch=torch,
                device=device,
                n_xy=args.camera_n_xy,
                n_radial=band["cap_radial"],
                n_phi=band["cap_phi"],
                complex_dtype=complex_dtype,
                radial_nodes_fraction=radial,
                normalize_pupil_boundary=True,
            )
        )
    return plans


def remap_camera(
    *, torch: Any, camera: Any, plans: list[CachedBilinearPolarRemap]
) -> Any:
    return torch.cat(
        [plan.gather_grid_sample(camera).reshape(-1) for plan in plans], dim=0
    )


def hot_remap_timing(
    *, torch: Any, device: Any, camera: Any, plans: list[Any], warmups: int, repeats: int
) -> dict[str, Any]:
    value = None
    with torch.inference_mode():
        for _ in range(warmups):
            value = remap_camera(torch=torch, camera=camera, plans=plans)
            synchronize(torch, device)
        times = []
        for _ in range(repeats):
            synchronize(torch, device)
            start = time.perf_counter()
            value = remap_camera(torch=torch, camera=camera, plans=plans)
            synchronize(torch, device)
            times.append(time.perf_counter() - start)
    return {
        "count": int(len(times)),
        "median_s": float(median(times)),
        "mean_s": float(np.mean(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "output_q_count": int(value.numel()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    reference = json.loads(args.pixel_reference.read_text(encoding="utf-8"))
    reference_case = next(
        case for case in reference["cases"] if case["label"] == args.variant
    )
    reference_selected = {
        row["selected_n_z"]: row for row in reference_case["selected"]
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    plan, context, bands, context_s, plan_s = build_variant(
        torch=torch, device=device, base_args=args, label=args.variant
    )
    remap_plans = make_remap_plans(
        torch=torch, device=device, args=args, variant=args.variant
    )
    selected_counts = [int(value) for value in args.selected_slices.split(",")]
    cases = []
    for selected_n_z in selected_counts:
        z_indices = centered_z_indices(args.n_z, selected_n_z)
        z_index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
        truth = selected_truth(torch, plan, context, z_indices, dtype=args.dtype)
        with torch.inference_mode():
            ideal_data = plan.forward_selected_z(truth, z_index)
        camera, generation = direct_cartesian_camera(
            torch=torch,
            device=device,
            args=args,
            context=context,
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
            remapped_data = remap_camera(
                torch=torch, camera=camera, plans=remap_plans
            )
        synchronize(torch, device)
        if remapped_data.shape != ideal_data.shape:
            raise RuntimeError("banded remap output shape does not match ACFO data")
        data_rel_l2 = float(
            (norm(torch, remapped_data - ideal_data) / norm(torch, ideal_data)).item()
        )
        pixel_solve = reference_selected[selected_n_z]["solve"]
        data_lipschitz = float(pixel_solve["lipschitz_with_margin"]) / 1.10
        ideal_solve, ideal_reconstruction, ideal_history = solve_selected_fista(
            torch=torch,
            plan=plan,
            z_index=z_index,
            data=ideal_data,
            truth=truth,
            label="ideal_banded",
            data_lipschitz=data_lipschitz,
            iterations=args.iterations,
            record_every=args.record_every,
        )
        remap_solve, remap_reconstruction, remap_history = solve_selected_fista(
            torch=torch,
            plan=plan,
            z_index=z_index,
            data=remapped_data,
            truth=truth,
            label="cartesian_to_banded",
            data_lipschitz=data_lipschitz,
            iterations=args.iterations,
            record_every=args.record_every,
        )
        reconstruction_rel_l2 = float(
            (
                norm(torch, remap_reconstruction - ideal_reconstruction)
                / norm(torch, ideal_reconstruction)
            ).item()
        )
        cases.append(
            {
                "selected_n_z": selected_n_z,
                "q_samples": int(ideal_data.numel()),
                "cartesian_remap_rel_l2_vs_ideal": data_rel_l2,
                "remap_timing": remap_timing,
                "reference_generation": generation,
                "ideal": ideal_solve,
                "cartesian_remap": remap_solve,
                "remap_vs_ideal_reconstruction_rel_l2": reconstruction_rel_l2,
                "ideal_history": ideal_history,
                "remap_history": remap_history,
            }
        )
        del (
            z_index,
            truth,
            ideal_data,
            camera,
            remapped_data,
            ideal_reconstruction,
            remap_reconstruction,
        )
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "schema": "odt-banded-cartesian-selected-z-reconstruction-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_name": device_name(torch, device),
        "variant": args.variant,
        "bands": bands,
        "problem": {
            "full_object_shape": [args.n_r, args.n_z, args.n_beta],
            "selected_slices": selected_counts,
            "total_views": int(args.ring_illum)
            + (0 if args.skip_axis_illumination else 1),
            "cartesian_detector": [args.camera_n_xy, args.camera_n_xy],
            "iterations": int(args.iterations),
            "noise": "none",
        },
        "context_build_s": float(context_s),
        "plan_build_s": float(plan_s),
        "cases": cases,
        "pytorch_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "claim_boundary": [
            "The physical input remains a Cartesian complex camera field; banded polar samples are cached bilinear interpolants.",
            "The direct Cartesian reference is generated with chunked cuFINUFFT and is validation-only setup work.",
            "No acquisition transfer, hologram demodulation, or measurement noise is included.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    p = chained_parser()
    p.description = "Cartesian camera -> banded polar -> selected-z reconstruction."
    p.add_argument("--variant", default="banded_inner96_outer64")
    p.add_argument("--remap-warmups", type=int, default=5)
    p.add_argument("--remap-repeats", type=int, default=30)
    p.add_argument(
        "--pixel-reference",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_banded_detector.json",
    )
    p.set_defaults(
        iterations=120,
        record_every=10,
        selected_slices="1,8",
        camera_n_xy=320,
        reference_view_block=8,
        output=ROOT
        / "benchmark_results"
        / "odt_banded_cartesian_reconstruction.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.variant not in VARIANTS:
        raise ValueError(f"unknown variant {args.variant}")
    result = run(args)
    compact = {
        "variant": result["variant"],
        "device_name": result["device_name"],
        "cases": [
            {
                "n": row["selected_n_z"],
                "remap_data_rel_l2": row[
                    "cartesian_remap_rel_l2_vs_ideal"
                ],
                "hot_remap_ms": 1000.0 * row["remap_timing"]["median_s"],
                "ideal_object_rel_l2": row["ideal"]["object_rel_l2"],
                "remap_object_rel_l2": row["cartesian_remap"]["object_rel_l2"],
                "reconstruction_rel_l2": row[
                    "remap_vs_ideal_reconstruction_rel_l2"
                ],
                "remap_median_iter_ms": 1000.0
                * row["cartesian_remap"]["core_iteration_timing"]["median_s"],
            }
            for row in result["cases"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
