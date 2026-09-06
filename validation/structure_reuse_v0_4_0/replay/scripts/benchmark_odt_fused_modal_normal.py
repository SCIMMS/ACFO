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
from benchmark_odt_banded_cartesian_final_packed import (  # noqa: E402
    random_complex_like,
)
from benchmark_odt_banded_cartesian_final_packed_full_timing import (  # noqa: E402
    parser as production_parser,
)
from benchmark_odt_banded_detector import (  # noqa: E402
    VARIANTS,
    TorchBandedCompositePlan,
    build_variant,
)
from benchmark_odt_selected_z_gpu import centered_z_indices  # noqa: E402
from benchmark_odt_virtual_polar_reconstruction import selected_truth  # noqa: E402


def relative_l2(torch: Any, candidate: Any, reference: Any) -> float:
    denominator = torch.clamp(torch.linalg.vector_norm(reference), min=1e-30)
    return float((torch.linalg.vector_norm(candidate - reference) / denominator).item())


def relative_scalar_error(torch: Any, candidate: Any, reference: Any) -> float:
    denominator = torch.clamp(torch.abs(reference), min=1e-30)
    return float((torch.abs(candidate - reference) / denominator).item())


def mib(value: Any) -> float:
    return float(value.numel() * value.element_size() / 1024**2)


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


def interleaved_cuda_timing(
    *,
    torch: Any,
    device: Any,
    baseline: Callable[[], Any],
    fused: Callable[[], Any],
    warmups: int,
    repeats: int,
) -> tuple[dict[str, float | int], dict[str, float | int], Any, Any]:
    """Alternate B/F then F/B to limit clock, cache, and order bias."""
    baseline_value = None
    fused_value = None
    with torch.inference_mode():
        for _ in range(warmups):
            baseline_value = baseline()
            fused_value = fused()
            synchronize(torch, device)
        baseline_times: list[float] = []
        fused_times: list[float] = []
        for repeat in range(repeats):
            callbacks = (
                (("baseline", baseline), ("fused", fused))
                if repeat % 2 == 0
                else (("fused", fused), ("baseline", baseline))
            )
            for label, callback in callbacks:
                synchronize(torch, device)
                start = time.perf_counter()
                value = callback()
                synchronize(torch, device)
                elapsed = time.perf_counter() - start
                if label == "baseline":
                    baseline_times.append(elapsed)
                    baseline_value = value
                else:
                    fused_times.append(elapsed)
                    fused_value = value
    return (
        timing_summary(baseline_times),
        timing_summary(fused_times),
        baseline_value,
        fused_value,
    )


def cone_reduced_h_l_u(cone: Any, coeff: Any, index: Any) -> Any:
    """Run the existing selected-z forward only through its h/l/u latent."""
    n_selected = int(index.numel())
    coeff_t = cone.as_coeff(coeff)
    expected = (cone.n_r, n_selected, cone.n_beta)
    if tuple(coeff_t.shape) != expected:
        raise ValueError(f"selected coefficient shape {tuple(coeff_t.shape)} != {expected}")

    coeff_h_full = cone.torch.fft.ifft(coeff_t, dim=2) * float(cone.n_beta)
    if cone.axial_lowrank_rank > 0:
        axial_lowrank_left = cone.axial_lowrank_left_z_rank.index_select(0, index)
        axial_phase = None
        axial_z_u = None
    else:
        axial_lowrank_left = None
        axial_phase = cone.axial_phase.index_select(0, index)
        axial_z_u = cone.axial_z_u.index_select(0, index)

    if cone.adaptive_l_packed_enabled:
        return cone._forward_packed_l_reduction(
            coeff_h_full,
            axial_phase=axial_phase,
            axial_z_u=axial_z_u,
            axial_lowrank_left=axial_lowrank_left,
        )

    r_block = min(cone.n_r, 16) if cone.radial_block_size <= 0 else cone.radial_block_size
    reduced = cone.torch.zeros(
        (cone.n_h, cone.n_l, cone.cap_radial),
        dtype=cone.complex_dtype,
        device=cone.device,
    )
    for r_start in range(0, cone.n_r, r_block):
        r_stop = min(r_start + r_block, cone.n_r)
        local_r = r_stop - r_start
        coeff_sources = coeff_h_full[r_start:r_stop].index_select(
            2, cone.source_slots_flat
        ).reshape(local_r, n_selected, cone.n_h, cone.n_l)
        source_matrix = coeff_sources.permute(0, 2, 3, 1).reshape(
            local_r * cone.n_h * cone.n_l, n_selected
        )
        if cone.axial_lowrank_rank > 0:
            projected = cone.torch.matmul(
                cone.torch.matmul(source_matrix, axial_lowrank_left),
                cone.axial_lowrank_right_rank_u,
            ).reshape(local_r, cone.n_h, cone.n_l, cone.cap_radial)
        else:
            source_matrix = source_matrix * axial_phase.reshape(1, n_selected)
            projected = cone.torch.matmul(
                source_matrix, axial_z_u
            ).reshape(local_r, cone.n_h, cone.n_l, cone.cap_radial)
        projected = projected * cone.transverse_r_l[
            r_start:r_stop
        ].reshape(local_r, 1, cone.n_l, 1)
        projected = projected * cone.radial[
            :, :, r_start:r_stop
        ].permute(2, 0, 1).reshape(local_r, cone.n_h, 1, cone.cap_radial)
        reduced.add_(projected.sum(dim=0))
    return reduced


def build_cone_gram(cone: Any, *, diagonal_tolerance: float) -> dict[str, Any]:
    """Cache the exact row-vector Gram used by forward then adjoint."""
    gram = cone.torch.matmul(cone.psi_phase_t, cone.psi_phase_conj)
    diagonal = cone.torch.diagonal(gram).contiguous()
    off_diagonal = gram - cone.torch.diag_embed(diagonal)
    offdiag_ratio = float(
        (
            cone.torch.linalg.vector_norm(off_diagonal)
            / cone.torch.clamp(cone.torch.linalg.vector_norm(gram), min=1e-30)
        ).item()
    )
    mode_power = (cone.mode_phase * cone.mode_phase_conj).contiguous()
    mode_power_deviation = float(
        cone.torch.max(cone.torch.abs(mode_power - 1)).item()
    )
    radial_imag_max = float(cone.torch.max(cone.torch.abs(cone.radial.imag)).item())
    return {
        "gram": gram,
        "diagonal": diagonal,
        "use_diagonal": bool(offdiag_ratio <= diagonal_tolerance),
        "offdiag_ratio": offdiag_ratio,
        "mode_power": mode_power,
        "mode_power_deviation": mode_power_deviation,
        "radial_imag_max": radial_imag_max,
        "storage_mib": mib(gram) + mib(diagonal) + mib(mode_power),
    }


def iter_cones(plan: TorchBandedCompositePlan) -> list[Any]:
    cones: list[Any] = []
    for composite in plan.plans:
        cones.append(composite.ring)
        if composite.axis is not None:
            cones.append(composite.axis)
    return cones


def build_gram_cache(
    plan: TorchBandedCompositePlan, *, diagonal_tolerance: float
) -> dict[int, dict[str, Any]]:
    return {
        id(cone): build_cone_gram(
            cone, diagonal_tolerance=diagonal_tolerance
        )
        for cone in iter_cones(plan)
    }


def cone_fused_normal(
    cone: Any,
    coeff: Any,
    z_indices: Any,
    cache: dict[int, dict[str, Any]],
    *,
    force_dense_gram: bool,
) -> Any:
    if not cone.slots_unique:
        raise RuntimeError("fused selected-mode normal requires unique detector slots")
    index = cone._selected_z_index(z_indices)
    reduced_h_l_u = cone_reduced_h_l_u(cone, coeff, index)
    gram_info = cache[id(cone)]
    weighted_h_l_u = reduced_h_l_u * gram_info["mode_power"].reshape(
        cone.n_h, 1, 1
    )
    if gram_info["use_diagonal"] and not force_dense_gram:
        mixed_h_l_u = weighted_h_l_u * gram_info["diagonal"].reshape(
            1, cone.n_l, 1
        )
        illumination_mixed = mixed_h_l_u.permute(2, 0, 1).reshape(
            cone.cap_radial * cone.n_h, cone.n_l
        )
    else:
        weighted_rows = weighted_h_l_u.permute(2, 0, 1).reshape(
            cone.cap_radial * cone.n_h, cone.n_l
        )
        illumination_mixed = cone.torch.matmul(
            weighted_rows, gram_info["gram"]
        )
    illumination_mixed = illumination_mixed * float(cone.cap_phi)
    return cone._adjoint_selected_z_from_illumination_mixed(
        illumination_mixed, index
    )


def composite_fused_normal(
    composite: Any,
    coeff: Any,
    z_indices: Any,
    cache: dict[int, dict[str, Any]],
    *,
    force_dense_gram: bool,
) -> Any:
    result = cone_fused_normal(
        composite.ring,
        coeff,
        z_indices,
        cache,
        force_dense_gram=force_dense_gram,
    )
    if composite.axis is not None:
        result = result + cone_fused_normal(
            composite.axis,
            coeff,
            z_indices,
            cache,
            force_dense_gram=force_dense_gram,
        )
    return result


def fused_normal(
    plan: TorchBandedCompositePlan,
    coeff: Any,
    z_indices: Any,
    cache: dict[int, dict[str, Any]],
    *,
    force_dense_gram: bool = False,
) -> Any:
    result = None
    for composite in plan.plans:
        local = composite_fused_normal(
            composite,
            coeff,
            z_indices,
            cache,
            force_dense_gram=force_dense_gram,
        )
        result = local if result is None else result + local
    return result


def baseline_normal(plan: Any, coeff: Any, z_indices: Any) -> Any:
    projected = plan.forward_selected_z_modes(coeff, z_indices)
    return plan.adjoint_selected_z_modes(projected, z_indices)


def memory_probe(
    *, torch: Any, device: Any, callback: Callable[[], Any]
) -> dict[str, float]:
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(torch, device)
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        value = callback()
        synchronize(torch, device)
    peak = int(torch.cuda.max_memory_allocated(device))
    output_mib = mib(value)
    del value
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "baseline_allocated_mib": float(baseline / 1024**2),
        "peak_allocated_mib": float(peak / 1024**2),
        "incremental_peak_mib": float((peak - baseline) / 1024**2),
        "output_mib": output_mib,
    }


def gram_rows(plan: Any, cache: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    cone_index = 0
    for band_index, composite in enumerate(plan.plans):
        for branch, cone in (("ring", composite.ring), ("axis", composite.axis)):
            if cone is None:
                continue
            info = cache[id(cone)]
            rows.append(
                {
                    "cone_index": cone_index,
                    "band_index": band_index,
                    "branch": branch,
                    "n_illum": cone.n_illum,
                    "n_h": cone.n_h,
                    "n_l": cone.n_l,
                    "cap_radial": cone.cap_radial,
                    "cap_phi": cone.cap_phi,
                    "offdiag_frobenius_ratio": info["offdiag_ratio"],
                    "diagonal_fast_path": info["use_diagonal"],
                    "mode_power_max_abs_deviation_from_one": info[
                        "mode_power_deviation"
                    ],
                    "radial_imag_max_abs": info["radial_imag_max"],
                    "gram_storage_mib": info["storage_mib"],
                }
            )
            cone_index += 1
    return rows


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ODT fused modal normal experiment",
        "",
        f"- overall accuracy: **{'PASS' if result['gates']['accuracy'] else 'FAIL'}**",
        f"- 1.5x full-volume speed gate: **{'PASS' if result['gates']['full_volume_speedup_1p5x'] else 'FAIL'}**",
        f"- decision: **{result['decision']}**",
        "",
        "## Configuration",
        "",
        f"- device: `{result['device']['name']}`",
        f"- detector: `{result['configuration']['variant']}`",
        f"- object: `{result['configuration']['n_r']} x {result['configuration']['n_z']} x {result['configuration']['n_beta']}`",
        f"- H / axial rank / adaptive-L: `{result['configuration']['h_cutoff']}` / `{result['configuration']['axial_lowrank_rank']}` / `{result['configuration']['ring_adaptive_l_packed_threshold']}`",
        f"- Gram setup: `{1000.0 * result['setup']['gram_build_s']:.3f} ms`, `{result['setup']['gram_storage_mib']:.3f} MiB`",
        "",
        "## Results",
        "",
        "| selected z | baseline ms | fused ms | speedup | rel-L2 | dense-Gram rel-L2 | baseline peak MiB | fused peak MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        lines.append(
            f"| {case['selected_n_z']} | "
            f"{1000.0 * case['baseline_timing']['median_s']:.3f} | "
            f"{1000.0 * case['fused_timing']['median_s']:.3f} | "
            f"{case['speedup']:.3f}x | "
            f"{case['accuracy']['auto_gram_vs_baseline_relative_l2']:.3e} | "
            f"{case['accuracy']['dense_gram_vs_baseline_relative_l2']:.3e} | "
            f"{case['memory']['baseline']['peak_allocated_mib']:.1f} | "
            f"{case['memory']['fused']['peak_allocated_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The fused path performs the same radial/axial forward reduction and adjoint scatter as the production operator, but replaces materialized illumination/detector-mode data and the two Psi contractions by one cached Psi^H Psi contraction. The diagonal fast path is used only when the measured Gram off-diagonal Frobenius ratio is below the declared tolerance.",
            "",
            "This is a hot normal-operator benchmark. It does not accelerate the one-time right-hand side A*Wy or individual forward predictions needed for data-space diagnostics.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS:
        raise ValueError(f"unknown detector variant: {args.variant}")
    if args.timing_warmups < 0 or args.timing_repeats <= 0:
        raise ValueError("timing warmups/repeats must be non-negative/positive")

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("this production-like experiment requires CUDA")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    plan, context, bands, context_s, plan_s = build_variant(
        torch=torch,
        device=device,
        base_args=args,
        label=args.variant,
    )
    synchronize(torch, device)
    gram_start = time.perf_counter()
    cache = build_gram_cache(
        plan, diagonal_tolerance=float(args.gram_diagonal_tolerance)
    )
    synchronize(torch, device)
    gram_s = time.perf_counter() - gram_start
    gram_metadata = gram_rows(plan, cache)

    cases: list[dict[str, Any]] = []
    selected_counts = [
        int(value) for value in args.selected_slices.split(",") if value.strip()
    ]
    for selected_n_z in selected_counts:
        z_indices = centered_z_indices(args.n_z, selected_n_z)
        z_index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
        truth = selected_truth(torch, plan, context, z_indices, dtype=args.dtype)
        direction = random_complex_like(
            torch, truth, int(args.seed) + 7000 + selected_n_z
        )
        probe = random_complex_like(
            torch, truth, int(args.seed) + 9000 + selected_n_z
        )

        with torch.inference_mode():
            baseline_value = baseline_normal(plan, direction, z_index)
            dense_value = fused_normal(
                plan,
                direction,
                z_index,
                cache,
                force_dense_gram=True,
            )
            fused_value = fused_normal(plan, direction, z_index, cache)
            projected = plan.forward_selected_z_modes(direction, z_index)
            normal_energy = torch.vdot(
                direction.reshape(-1), baseline_value.reshape(-1)
            )
            data_energy = torch.vdot(projected.reshape(-1), projected.reshape(-1))
            fused_probe = fused_normal(plan, probe, z_index, cache)
            lhs = torch.vdot(probe.reshape(-1), fused_value.reshape(-1))
            rhs = torch.vdot(fused_probe.reshape(-1), direction.reshape(-1))
            hermitian_denominator = torch.clamp(
                torch.abs(lhs) + torch.abs(rhs), min=1e-30
            )
            hermitian_error = float((torch.abs(lhs - rhs) / hermitian_denominator).item())

        accuracy = {
            "auto_gram_vs_baseline_relative_l2": relative_l2(
                torch, fused_value, baseline_value
            ),
            "dense_gram_vs_baseline_relative_l2": relative_l2(
                torch, dense_value, baseline_value
            ),
            "normal_energy_vs_forward_norm2_relative_error": relative_scalar_error(
                torch, normal_energy, data_energy
            ),
            "normal_energy_imag_over_abs": float(
                (torch.abs(normal_energy.imag) / torch.clamp(torch.abs(normal_energy), min=1e-30)).item()
            ),
            "fused_hermitian_bilinear_relative_error": hermitian_error,
            "normal_energy_real": float(normal_energy.real.item()),
            "normal_energy_nonnegative": bool(float(normal_energy.real.item()) >= 0.0),
        }
        del dense_value, fused_probe, projected

        baseline_memory = memory_probe(
            torch=torch,
            device=device,
            callback=lambda: baseline_normal(plan, direction, z_index),
        )
        fused_memory = memory_probe(
            torch=torch,
            device=device,
            callback=lambda: fused_normal(plan, direction, z_index, cache),
        )

        (
            baseline_timing,
            fused_timing,
            timed_baseline,
            timed_fused,
        ) = interleaved_cuda_timing(
            torch=torch,
            device=device,
            baseline=lambda: baseline_normal(plan, direction, z_index),
            fused=lambda: fused_normal(plan, direction, z_index, cache),
            warmups=args.timing_warmups,
            repeats=args.timing_repeats,
        )
        timed_relative_l2 = relative_l2(torch, timed_fused, timed_baseline)
        speedup = float(
            baseline_timing["median_s"] / fused_timing["median_s"]
        )
        saved_s = float(
            baseline_timing["median_s"] - fused_timing["median_s"]
        )
        amortization_iterations = (
            None if saved_s <= 0.0 else float(gram_s / saved_s)
        )
        case = {
            "selected_n_z": selected_n_z,
            "object_shape": list(direction.shape),
            "object_mib": mib(direction),
            "accuracy": {**accuracy, "timed_outputs_relative_l2": timed_relative_l2},
            "baseline_timing": baseline_timing,
            "fused_timing": fused_timing,
            "speedup": speedup,
            "gram_setup_amortization_iterations": amortization_iterations,
            "memory": {"baseline": baseline_memory, "fused": fused_memory},
        }
        cases.append(case)
        print(
            f"z={selected_n_z}: baseline={1000.0 * baseline_timing['median_s']:.3f} ms, "
            f"fused={1000.0 * fused_timing['median_s']:.3f} ms, "
            f"speedup={speedup:.3f}x, rel={accuracy['auto_gram_vs_baseline_relative_l2']:.3e}",
            flush=True,
        )
        del (
            z_index,
            truth,
            direction,
            probe,
            baseline_value,
            fused_value,
            timed_baseline,
            timed_fused,
        )
        gc.collect()
        torch.cuda.empty_cache()

    accuracy_gate = bool(
        max(
            case["accuracy"]["auto_gram_vs_baseline_relative_l2"]
            for case in cases
        )
        <= float(args.accuracy_tolerance)
        and max(
            case["accuracy"]["fused_hermitian_bilinear_relative_error"]
            for case in cases
        )
        <= float(args.hermitian_tolerance)
        and all(case["accuracy"]["normal_energy_nonnegative"] for case in cases)
    )
    full_case = next(
        (case for case in cases if case["selected_n_z"] == args.n_z), None
    )
    speed_gate = bool(full_case is not None and full_case["speedup"] >= 1.5)
    memory_gate = bool(
        all(
            case["memory"]["fused"]["incremental_peak_mib"]
            <= case["memory"]["baseline"]["incremental_peak_mib"]
            for case in cases
        )
    )
    decision = (
        "GO: continue fused modal normal toward solver integration"
        if accuracy_gate and speed_gate and memory_gate
        else "STOP as a standalone acceleration; retain as an optional backend/preconditioner study"
    )
    result = {
        "schema": "odt-fused-modal-normal-experiment-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": device_name(torch, device),
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(torch.version, "cuda", None),
            "dtype": args.dtype,
        },
        "configuration": {
            "variant": args.variant,
            "n_r": args.n_r,
            "n_z": args.n_z,
            "n_beta": args.n_beta,
            "h_cutoff": args.h_cutoff,
            "axial_lowrank_rank": args.axial_lowrank_rank,
            "ring_adaptive_l_packed_threshold": args.ring_adaptive_l_packed_threshold,
            "radial_block_size": args.radial_block_size,
            "illumination_block_size": args.illumination_block_size,
            "bands": bands,
            "timing_warmups": args.timing_warmups,
            "timing_repeats": args.timing_repeats,
            "gram_diagonal_tolerance": args.gram_diagonal_tolerance,
            "accuracy_tolerance": args.accuracy_tolerance,
            "hermitian_tolerance": args.hermitian_tolerance,
        },
        "setup": {
            "context_build_s": context_s,
            "plan_build_s": plan_s,
            "gram_build_s": gram_s,
            "gram_storage_mib": float(
                sum(item["gram_storage_mib"] for item in gram_metadata)
            ),
        },
        "gram": gram_metadata,
        "cases": cases,
        "gates": {
            "accuracy": accuracy_gate,
            "full_volume_speedup_1p5x": speed_gate,
            "memory_not_worse": memory_gate,
        },
        "decision": decision,
        "scope": (
            "Linear fixed-geometry W=I selected detector-mode normal only. "
            "The one-time data adjoint and data-space predictions are unchanged."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(render_markdown(result), encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    p = production_parser()
    p.description = (
        "Compare the production forward+adjoint normal pair with an exact "
        "illumination-Gram fused modal normal."
    )
    p.add_argument("--gram-diagonal-tolerance", type=float, default=2e-5)
    p.add_argument("--accuracy-tolerance", type=float, default=2e-6)
    p.add_argument("--hermitian-tolerance", type=float, default=2e-6)
    p.add_argument(
        "--summary",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_fused_modal_normal_experiment_20260805_ko.md",
    )
    p.set_defaults(
        selected_slices="64,128,256",
        timing_warmups=5,
        timing_repeats=20,
        output=ROOT
        / "benchmark_results"
        / "odt_fused_modal_normal_experiment_20260805.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "passed_accuracy": result["gates"]["accuracy"],
                "passed_speed_1p5x": result["gates"]["full_volume_speedup_1p5x"],
                "passed_memory": result["gates"]["memory_not_worse"],
                "decision": result["decision"],
                "output": str(args.output),
                "summary": str(args.summary),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
