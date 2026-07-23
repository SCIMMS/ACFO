from __future__ import annotations

import argparse
import gc
import hashlib
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
from benchmark_odt_banded_cartesian_final_packed import (  # noqa: E402
    banded_pixel_to_modes,
    candidate_contract,
    random_complex_like,
    timed_cuda,
)
from benchmark_odt_banded_cartesian_reconstruction import (  # noqa: E402
    make_remap_plans,
    remap_camera,
)
from benchmark_odt_banded_detector import VARIANTS, build_variant  # noqa: E402
from benchmark_odt_selected_z_gpu import centered_z_indices  # noqa: E402
from benchmark_odt_virtual_polar_reconstruction import selected_truth  # noqa: E402


DEFAULT_PROBE = (
    ROOT / "benchmark_results" / "odt_banded_cartesian_final_packed_probe.json"
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_selected_counts(text: str, n_z: int) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0 or value > n_z for value in values):
        raise ValueError("selected-slices must contain values in [1, n-z]")
    if len(set(values)) != len(values):
        raise ValueError("selected-slices must not contain duplicates")
    return values


def deterministic_camera(torch: Any, args: argparse.Namespace, device: Any) -> Any:
    views = int(args.ring_illum) + (0 if args.skip_axis_illumination else 1)
    complex_dtype = (
        torch.complex64 if args.dtype == "complex64" else torch.complex128
    )
    real_dtype = torch.float32 if args.dtype == "complex64" else torch.float64
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.camera_seed))
    shape = (views, int(args.camera_n_xy), int(args.camera_n_xy))
    real = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
    imag = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
    return torch.complex(real, imag).to(dtype=complex_dtype)


def mib(torch: Any, value: Any) -> float:
    return float(value.numel() * value.element_size() / 1024**2)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS:
        raise ValueError(f"unknown banded detector variant: {args.variant}")
    selected_counts = parse_selected_counts(args.selected_slices, args.n_z)
    if args.timing_warmups < 0 or args.timing_repeats <= 0:
        raise ValueError("timing warmups/repeats must be non-negative/positive")

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
        torch=torch,
        device=device,
        base_args=args,
        label=args.variant,
    )
    remap_plans = make_remap_plans(
        torch=torch, device=device, args=args, variant=args.variant
    )
    camera = deterministic_camera(torch, args, device)
    synchronize(torch, device)
    total_build_s = time.perf_counter() - build_start
    allocation_after_build = int(torch.cuda.memory_allocated(device))
    peak_during_build = int(torch.cuda.max_memory_allocated(device))

    preprocess_timing, preprocessed_modes = timed_cuda(
        torch=torch,
        device=device,
        callback=lambda: banded_pixel_to_modes(
            plan,
            remap_camera(torch=torch, camera=camera, plans=remap_plans),
        ),
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    if preprocessed_modes.shape != (plan.mode_q_count,):
        raise RuntimeError("preprocessed mode vector has unexpected shape")

    cases: list[dict[str, Any]] = []
    for selected_n_z in selected_counts:
        z_indices = centered_z_indices(args.n_z, selected_n_z)
        z_index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
        truth = selected_truth(
            torch, plan, context, z_indices, dtype=args.dtype
        )
        direction = random_complex_like(
            torch, truth, args.seed + 5000 + selected_n_z
        )
        synchronize(torch, device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)

        data_adjoint_timing, data_rhs = timed_cuda(
            torch=torch,
            device=device,
            callback=lambda: plan.adjoint_selected_z_modes(
                preprocessed_modes, z_index
            ),
            warmups=args.timing_warmups,
            repeats=args.timing_repeats,
        )

        def normal_pair() -> Any:
            projected = plan.forward_selected_z_modes(direction, z_index)
            return plan.adjoint_selected_z_modes(projected, z_index)

        normal_pair_timing, normal_value = timed_cuda(
            torch=torch,
            device=device,
            callback=normal_pair,
            warmups=args.timing_warmups,
            repeats=args.timing_repeats,
        )

        def remap_mode_normal_pair() -> Any:
            current_pixel = remap_camera(
                torch=torch, camera=camera, plans=remap_plans
            )
            current_modes = banded_pixel_to_modes(plan, current_pixel)
            projected = plan.forward_selected_z_modes(direction, z_index)
            return current_modes, plan.adjoint_selected_z_modes(projected, z_index)

        steady_integrated_timing, steady_value = timed_cuda(
            torch=torch,
            device=device,
            callback=remap_mode_normal_pair,
            warmups=args.timing_warmups,
            repeats=args.timing_repeats,
        )

        def first_update() -> Any:
            current_pixel = remap_camera(
                torch=torch, camera=camera, plans=remap_plans
            )
            current_modes = banded_pixel_to_modes(plan, current_pixel)
            rhs = plan.adjoint_selected_z_modes(current_modes, z_index)
            projected = plan.forward_selected_z_modes(direction, z_index)
            normal = plan.adjoint_selected_z_modes(projected, z_index)
            return rhs, normal

        first_update_timing, first_value = timed_cuda(
            torch=torch,
            device=device,
            callback=first_update,
            warmups=args.timing_warmups,
            repeats=args.timing_repeats,
        )
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        case = {
            "selected_n_z": int(selected_n_z),
            "object_shape": list(direction.shape),
            "object_mib": mib(torch, direction),
            "data_adjoint_timing": data_adjoint_timing,
            "normal_pair_timing": normal_pair_timing,
            "normal_pairs_per_second": float(
                1.0 / normal_pair_timing["median_s"]
            ),
            "integrated_steady_remap_mode_normal_pair_timing": (
                steady_integrated_timing
            ),
            "integrated_steady_updates_per_second": float(
                1.0 / steady_integrated_timing["median_s"]
            ),
            "integrated_new_frame_first_update_timing": first_update_timing,
            "integrated_new_frame_first_updates_per_second": float(
                1.0 / first_update_timing["median_s"]
            ),
            "memory": {
                "baseline_allocated_mib": float(baseline_allocated / 1024**2),
                "peak_allocated_mib": float(peak_allocated / 1024**2),
                "incremental_peak_mib": float(
                    (peak_allocated - baseline_allocated) / 1024**2
                ),
            },
        }
        cases.append(case)
        print(
            f"z={selected_n_z}: normal={1000.0 * normal_pair_timing['median_s']:.3f} ms, "
            f"steady-integrated={1000.0 * steady_integrated_timing['median_s']:.3f} ms "
            f"({1.0 / steady_integrated_timing['median_s']:.2f} Hz), "
            f"first-update={1000.0 * first_update_timing['median_s']:.3f} ms "
            f"({1.0 / first_update_timing['median_s']:.2f} Hz), "
            f"peak={peak_allocated / 1024**2:.1f} MiB",
            flush=True,
        )
        del (
            z_index,
            truth,
            direction,
            data_rhs,
            normal_value,
            steady_value,
            first_value,
        )
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "schema": "odt-banded-cartesian-final-packed-full-timing-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": device_name(torch, device),
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(torch.version, "cuda", None),
            "dtype": args.dtype,
        },
        "detector": {
            "variant": args.variant,
            "camera_shape": list(camera.shape),
            "camera_input_mib": mib(torch, camera),
            "bands": bands,
            "pixel_q_count": int(plan.q_count),
            "mode_q_count": int(plan.mode_q_count),
        },
        "candidate": {
            **candidate_contract(args),
            "context_build_s": float(context_s),
            "plan_build_s": float(plan_s),
            "total_plan_remap_camera_build_s": float(total_build_s),
        },
        "problem": {
            "full_object_shape": [args.n_r, args.n_z, args.n_beta],
            "selected_slices": selected_counts,
            "camera_data": (
                "deterministic GPU-resident complex synthetic timing payload; "
                "data values do not affect linear hot-path timing"
            ),
        },
        "preprocessing_timing": {
            "cached_cartesian_remap_plus_mode_reduction": preprocess_timing,
            "preprocessed_mode_mib": mib(torch, preprocessed_modes),
        },
        "cases": cases,
        "memory": {
            "allocated_after_plan_remap_camera_build_mib": float(
                allocation_after_build / 1024**2
            ),
            "peak_during_plan_remap_camera_build_mib": float(
                peak_during_build / 1024**2
            ),
        },
        "accuracy_anchor": {
            "path": str(args.accuracy_probe.relative_to(ROOT))
            if args.accuracy_probe.is_relative_to(ROOT)
            else str(args.accuracy_probe),
            "sha256": sha256(args.accuracy_probe),
            "role": (
                "The separate frozen small probe supplies direct Cartesian-camera, "
                "H36 operator and reconstruction-difference validation."
            ),
        },
        "claim_boundary": [
            "This run is a timing and memory scale-up of a separately accuracy-validated operator path; it does not repeat the direct Cartesian cuFINUFFT reference at 64/128/256 z.",
            "All hot timings start with the complex camera tensor resident on the RTX 2070 SUPER GPU and exclude acquisition, host-to-device transfer and hologram demodulation.",
            "The steady integrated row includes cached Cartesian remap, angular-mode reduction and one forward/adjoint normal pair.",
            "The new-frame first-update row additionally includes the one-time data adjoint that forms the right-hand side for that frame.",
            "A normal pair is one iterative solver core application, not a converged reconstruction. Total reconstruction latency depends on the chosen iteration or warm-start update count.",
            "The synthetic camera values are used only because linear operator timing is data-independent; accuracy remains anchored to the physical direct-camera probe.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    from benchmark_odt_banded_cartesian_final_packed import parser as base_parser

    p = base_parser()
    p.description = (
        "Timing-only 64/128/256-z scale-up of the accuracy-gated Cartesian "
        "remap plus final packed ODT operator."
    )
    p.add_argument("--timing-warmups", type=int, default=5)
    p.add_argument("--timing-repeats", type=int, default=30)
    p.add_argument("--camera-seed", type=int, default=20260714)
    p.add_argument("--accuracy-probe", type=Path, default=DEFAULT_PROBE)
    p.set_defaults(
        selected_slices="64,128,256",
        output=ROOT
        / "benchmark_results"
        / "odt_banded_cartesian_final_packed_full_timing.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    compact = {
        "schema": result["schema"],
        "device": result["device"]["name"],
        "preprocessing_median_ms": 1000.0
        * result["preprocessing_timing"][
            "cached_cartesian_remap_plus_mode_reduction"
        ]["median_s"],
        "cases": [
            {
                "selected_n_z": case["selected_n_z"],
                "normal_pair_ms": 1000.0
                * case["normal_pair_timing"]["median_s"],
                "steady_integrated_ms": 1000.0
                * case[
                    "integrated_steady_remap_mode_normal_pair_timing"
                ]["median_s"],
                "steady_integrated_hz": case[
                    "integrated_steady_updates_per_second"
                ],
                "first_update_ms": 1000.0
                * case["integrated_new_frame_first_update_timing"]["median_s"],
                "first_update_hz": case[
                    "integrated_new_frame_first_updates_per_second"
                ],
                "peak_allocated_mib": case["memory"]["peak_allocated_mib"],
            }
            for case in result["cases"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
