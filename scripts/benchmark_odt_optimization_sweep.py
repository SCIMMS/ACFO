from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np
from scipy import special


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    CompositeContext,
    build_composite_context,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    TorchConeAxisOdtPlan,
    parser as odt_parser,
    resolve_device,
    synchronize,
    torch_dtypes,
)


@dataclass(frozen=True)
class AxialLowRankBasis:
    left_z_rank: Any
    right_rank_u: Any
    rank: int
    relative_frobenius_tail: float


@dataclass(frozen=True)
class AxialSvd:
    u: np.ndarray
    singular_values: np.ndarray
    vh: np.ndarray


def relative_l2(torch: Any, candidate: Any, reference: Any) -> float:
    denom = torch.clamp(torch.linalg.vector_norm(reference), min=1e-30)
    return float(
        (torch.linalg.vector_norm(candidate - reference) / denom)
        .detach()
        .cpu()
        .item()
    )


def timed_cuda(
    torch: Any,
    device: Any,
    func: Callable[[], Any],
    *,
    repeats: int,
    warmups: int,
) -> tuple[Any, float, list[float]]:
    value = None
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        for _ in range(max(0, int(warmups))):
            value = func()
            synchronize(torch, device)
        times: list[float] = []
        for _ in range(max(1, int(repeats))):
            synchronize(torch, device)
            start = time.perf_counter()
            value = func()
            synchronize(torch, device)
            times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed function did not execute")
    return value, float(median(times)), times


def trim_axis_context_to_l0(axis_context: Any) -> Any:
    """Return an axis-illumination context with the exactly active L=0 mode.

    For illumination NA=0, the incident transverse factor is J_L(0), which is
    exactly zero for every integer L except zero.  The current generic Torch
    plan carries the padded L margin anyway; this helper removes those exact
    zeros without changing the shared default implementation.
    """

    decomp = axis_context.decomp
    zero = np.flatnonzero(np.asarray(decomp.l_values) == 0)
    if zero.size != 1:
        raise ValueError("axis decomposition must contain exactly one L=0 mode")
    keep = zero.astype(np.intp)
    trimmed = replace(
        decomp,
        l_values=np.ascontiguousarray(decomp.l_values[keep]),
        transverse_coeff=np.ascontiguousarray(decomp.transverse_coeff[keep]),
        psi_phase=np.ascontiguousarray(decomp.psi_phase[:, keep]),
        source_slots=np.ascontiguousarray(decomp.source_slots[:, keep]),
        active_l_offsets=None,
        active_l_indices=None,
        active_l_threshold=None,
    )
    return replace(axis_context, decomp=trimmed, l_cutoff=0)


def make_torch_plan(
    context: CompositeContext,
    *,
    torch: Any,
    device: Any,
    dtype: str,
    radial_block_size: int,
    illumination_block_size: int,
    prune_axis_l0: bool,
) -> TorchCompositeOdtPlan:
    ring = TorchConeAxisOdtPlan.from_context(
        context.ring,
        torch=torch,
        device=device,
        dtype=dtype,
        low_memory_adjoint=True,
        radial_block_size=radial_block_size,
        illumination_block_size=illumination_block_size,
        forward_mode="auto",
        adjoint_mode="auto",
    )
    axis_context = context.axis
    if prune_axis_l0 and axis_context is not None:
        axis_context = trim_axis_context_to_l0(axis_context)
    axis = (
        None
        if axis_context is None
        else TorchConeAxisOdtPlan.from_context(
            axis_context,
            torch=torch,
            device=device,
            dtype=dtype,
            low_memory_adjoint=True,
            radial_block_size=radial_block_size,
            illumination_block_size=illumination_block_size,
            forward_mode="auto",
            adjoint_mode="auto",
        )
    )
    return TorchCompositeOdtPlan(
        torch=torch,
        device=device,
        complex_dtype=ring.complex_dtype,
        real_dtype=ring.real_dtype,
        ring=ring,
        axis=axis,
    )


def axial_svd(plan: TorchConeAxisOdtPlan) -> AxialSvd:
    axial = plan.axial_z_u.detach().cpu().numpy().astype(np.complex128, copy=False)
    phase = plan.axial_phase.detach().cpu().numpy().astype(np.complex128, copy=False)
    effective = np.ascontiguousarray(phase[:, None] * axial)
    u, singular_values, vh = np.linalg.svd(effective, full_matrices=False)
    return AxialSvd(
        u=np.ascontiguousarray(u),
        singular_values=np.ascontiguousarray(singular_values),
        vh=np.ascontiguousarray(vh),
    )


def lowrank_basis(
    plan: TorchConeAxisOdtPlan,
    svd: AxialSvd,
    *,
    rank: int,
) -> AxialLowRankBasis:
    rank = min(max(1, int(rank)), int(svd.singular_values.size))
    left = svd.u[:, :rank] * svd.singular_values[None, :rank]
    right = svd.vh[:rank]
    total = float(np.sum(svd.singular_values**2))
    tail = float(np.sum(svd.singular_values[rank:] ** 2))
    np_complex = np.complex64 if plan.complex_dtype == plan.torch.complex64 else np.complex128
    return AxialLowRankBasis(
        left_z_rank=plan.torch.as_tensor(
            np.ascontiguousarray(left.astype(np_complex, copy=False)),
            dtype=plan.complex_dtype,
            device=plan.device,
        ),
        right_rank_u=plan.torch.as_tensor(
            np.ascontiguousarray(right.astype(np_complex, copy=False)),
            dtype=plan.complex_dtype,
            device=plan.device,
        ),
        rank=rank,
        relative_frobenius_tail=math.sqrt(tail / max(total, np.finfo(float).tiny)),
    )


def cone_forward_lowrank(
    plan: TorchConeAxisOdtPlan,
    coeff: Any,
    basis: AxialLowRankBasis,
) -> Any:
    torch = plan.torch
    coeff_t = plan.as_coeff(coeff)
    if tuple(coeff_t.shape) != (plan.n_r, plan.n_z, plan.n_beta):
        raise ValueError("coefficient shape does not match low-rank ODT plan")
    coeff_h_full = torch.fft.ifft(coeff_t, dim=2) * float(plan.n_beta)
    r_block = min(plan.n_r, 16) if plan.radial_block_size <= 0 else plan.radial_block_size
    reduced_h_l_u = torch.zeros(
        (plan.n_h, plan.n_l, plan.cap_radial),
        dtype=plan.complex_dtype,
        device=plan.device,
    )
    for r_start in range(0, plan.n_r, r_block):
        r_stop = min(r_start + r_block, plan.n_r)
        local_r = r_stop - r_start
        coeff_sources = coeff_h_full[r_start:r_stop].index_select(
            2, plan.source_slots_flat
        ).reshape(local_r, plan.n_z, plan.n_h, plan.n_l)
        source_matrix = coeff_sources.permute(0, 2, 3, 1).reshape(
            local_r * plan.n_h * plan.n_l, plan.n_z
        )
        projected = torch.matmul(
            torch.matmul(source_matrix, basis.left_z_rank),
            basis.right_rank_u,
        ).reshape(local_r, plan.n_h, plan.n_l, plan.cap_radial)
        projected = projected * plan.transverse_r_l[r_start:r_stop].reshape(
            local_r, 1, plan.n_l, 1
        )
        projected = projected * plan.radial[:, :, r_start:r_stop].permute(
            2, 0, 1
        ).reshape(local_r, plan.n_h, 1, plan.cap_radial)
        reduced_h_l_u.add_(projected.sum(dim=0))

    inner = torch.matmul(
        reduced_h_l_u.permute(2, 0, 1).reshape(
            plan.cap_radial * plan.n_h, plan.n_l
        ),
        plan.psi_phase_t,
    ).reshape(plan.cap_radial, plan.n_h, plan.n_illum).permute(2, 0, 1)
    folded = torch.zeros(
        (plan.n_illum, plan.cap_radial, plan.cap_phi),
        dtype=plan.complex_dtype,
        device=plan.device,
    )
    source = inner * plan.mode_phase.reshape(1, 1, plan.n_h)
    if plan.slots_unique:
        folded.index_copy_(2, plan.slots, source)
    else:
        folded.index_add_(2, plan.slots, source)
    return torch.fft.fft(folded, dim=2).reshape(-1)


def cone_adjoint_lowrank(
    plan: TorchConeAxisOdtPlan,
    residual: Any,
    basis: AxialLowRankBasis,
) -> Any:
    torch = plan.torch
    residual_t = plan.as_coeff(residual)
    if residual_t.shape != (plan.q_count,):
        raise ValueError("residual size does not match low-rank ODT plan")
    residual_grid = residual_t.reshape(plan.n_illum, plan.cap_radial, plan.cap_phi)
    i_block = (
        plan.n_illum
        if plan.illumination_block_size <= 0
        else plan.illumination_block_size
    )
    illumination_mixed = torch.zeros(
        (plan.cap_radial * plan.n_h, plan.n_l),
        dtype=plan.complex_dtype,
        device=plan.device,
    )
    for i_start in range(0, plan.n_illum, i_block):
        i_stop = min(i_start + i_block, plan.n_illum)
        residual_modes = torch.fft.ifft(
            residual_grid[i_start:i_stop], dim=2
        ) * float(plan.cap_phi)
        selected = residual_modes.index_select(2, plan.slots)
        phi_sum = selected * plan.mode_phase_conj.reshape(1, 1, plan.n_h)
        illumination_mixed.add_(
            torch.matmul(
                phi_sum.permute(1, 2, 0).reshape(
                    plan.cap_radial * plan.n_h, i_stop - i_start
                ),
                plan.psi_phase_conj[i_start:i_stop],
            )
        )
    illumination_mixed_h_l_u = illumination_mixed.reshape(
        plan.cap_radial, plan.n_h, plan.n_l
    ).permute(1, 2, 0).contiguous()

    r_block = min(plan.n_r, 16) if plan.radial_block_size <= 0 else plan.radial_block_size
    out_h = torch.zeros(
        (plan.n_r, plan.n_z, plan.n_beta),
        dtype=plan.complex_dtype,
        device=plan.device,
    )
    right_h = basis.right_rank_u.conj().transpose(0, 1).contiguous()
    left_h = basis.left_z_rank.conj().transpose(0, 1).contiguous()
    for r_start in range(0, plan.n_r, r_block):
        r_stop = min(r_start + r_block, plan.n_r)
        local_r = r_stop - r_start
        radial_h_r_u = plan.radial[:, :, r_start:r_stop].permute(0, 2, 1)
        weighted = radial_h_r_u.unsqueeze(2) * illumination_mixed_h_l_u.unsqueeze(1)
        weighted_matrix = weighted.reshape(
            plan.n_h * local_r * plan.n_l, plan.cap_radial
        )
        contributions = torch.matmul(
            torch.matmul(weighted_matrix, right_h),
            left_h,
        ).reshape(plan.n_h, local_r, plan.n_l, plan.n_z).permute(1, 3, 0, 2)
        contributions = contributions * plan.transverse_conj_r_l[
            r_start:r_stop
        ].reshape(local_r, 1, 1, plan.n_l)
        out_h[r_start:r_stop].index_add_(
            2,
            plan.source_slots_flat,
            contributions.reshape(local_r, plan.n_z, plan.n_h * plan.n_l),
        )
    return torch.fft.fft(out_h, dim=2)


def composite_forward_lowrank(
    plan: TorchCompositeOdtPlan,
    coeff: Any,
    ring_basis: AxialLowRankBasis,
    axis_basis: AxialLowRankBasis | None,
) -> Any:
    parts = [cone_forward_lowrank(plan.ring, coeff, ring_basis)]
    if plan.axis is not None:
        if axis_basis is None:
            raise ValueError("axis basis is required for a composite axis plan")
        parts.append(cone_forward_lowrank(plan.axis, coeff, axis_basis))
    return plan.torch.cat(parts, dim=0)


def composite_adjoint_lowrank(
    plan: TorchCompositeOdtPlan,
    residual: Any,
    ring_basis: AxialLowRankBasis,
    axis_basis: AxialLowRankBasis | None,
) -> Any:
    ring_residual, axis_residual = plan.split_residual(residual)
    grad = cone_adjoint_lowrank(plan.ring, ring_residual, ring_basis)
    if plan.axis is not None and axis_residual is not None:
        if axis_basis is None:
            raise ValueError("axis basis is required for a composite axis plan")
        grad = grad + cone_adjoint_lowrank(plan.axis, axis_residual, axis_basis)
    return grad


def comma_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return items


def support_screen(args: argparse.Namespace) -> dict[str, Any]:
    n_r = int(args.n_r)
    r = (np.arange(n_r, dtype=float) + 0.5) * float(args.r_max) / float(n_r)
    illumination_na = math.sin(math.radians(float(args.illumination_angle_deg)))
    l_cutoff = int(math.ceil(args.k * illumination_na * args.r_max + args.l_margin))
    l_cutoff = min(l_cutoff, args.n_beta // 2 - 1)
    l_values = np.arange(-l_cutoff, l_cutoff + 1, dtype=int)
    transverse_abs = np.abs(
        special.jv(l_values[:, None], args.k * illumination_na * r[None, :])
    )
    l_rows = []
    for threshold in args.l_thresholds:
        counts = np.maximum(np.sum(transverse_abs > threshold, axis=0), 1)
        l_rows.append(
            {
                "threshold": float(threshold),
                "active_fraction": float(np.sum(counts) / counts.size / l_values.size),
                "active_count_min": int(np.min(counts)),
                "active_count_median": float(np.median(counts)),
                "active_count_p95": float(np.percentile(counts, 95)),
                "active_count_max": int(np.max(counts)),
            }
        )

    radial_fraction = (
        np.arange(args.cap_radial, dtype=float) + 0.5
    ) / float(args.cap_radial)
    q_perp = args.k * args.detector_na * radial_fraction
    radial_energy = []
    max_h = max(args.h_values + [args.reference_h_cutoff])
    for h in range(max_h + 1):
        radial = special.jv(h, q_perp[:, None] * r[None, :])
        radial_energy.append(float(np.sum(radial * radial)))
    total = radial_energy[0] + 2.0 * sum(radial_energy[1:])
    h_rows = []
    for cutoff in sorted(set(args.h_values + [args.reference_h_cutoff]), reverse=True):
        tail = 2.0 * sum(radial_energy[cutoff + 1 :])
        h_rows.append(
            {
                "h_cutoff": int(cutoff),
                "h_modes": int(2 * cutoff + 1),
                "radial_kernel_frobenius_tail": math.sqrt(
                    tail / max(total, np.finfo(float).tiny)
                ),
            }
        )
    return {
        "l_cutoff": l_cutoff,
        "l_modes": int(l_values.size),
        "adaptive_l": l_rows,
        "harmonic_cutoff": h_rows,
    }


def base_odt_args(args: argparse.Namespace, *, h_cutoff: int) -> argparse.Namespace:
    values = [
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--low-memory-adjoint",
        "--radial-block-size",
        str(args.radial_block_size),
        "--illumination-block-size",
        str(args.illumination_block_size),
        "--forward-mode",
        "auto",
        "--adjoint-mode",
        "auto",
        "--skip-native-prepared-adjoint",
        "--compact-axisymmetric-kernel",
        "--real-object",
        "--n-beta",
        str(args.n_beta),
        "--n-r",
        str(args.n_r),
        "--n-z",
        str(args.n_z),
        "--r-max",
        str(args.r_max),
        "--z-max",
        str(args.z_max),
        "--k",
        str(args.k),
        "--detector-na",
        str(args.detector_na),
        "--illumination-angle-deg",
        str(args.illumination_angle_deg),
        "--ring-illum",
        str(args.ring_illum),
        "--cap-radial",
        str(args.cap_radial),
        "--cap-phi",
        str(args.cap_phi),
        "--h-cutoff",
        str(h_cutoff),
        "--l-margin",
        str(args.l_margin),
        "--cone-l-prune-threshold",
        str(args.cone_l_prune_threshold),
        "--cpp-threads",
        str(args.cpp_threads),
        "--iterations",
        "0",
        "--repeats",
        "1",
        "--warmups",
        "0",
    ]
    return odt_parser().parse_args(values)


def release_cuda(torch: Any, *values: Any) -> None:
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("the production optimization sweep requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    screen = support_screen(args)

    reference_args = base_odt_args(args, h_cutoff=args.reference_h_cutoff)
    build_start = time.perf_counter()
    context = build_composite_context(reference_args)
    context_build_s = time.perf_counter() - build_start
    plan_build_start = time.perf_counter()
    reference_plan = make_torch_plan(
        context,
        torch=torch,
        device=device,
        dtype=args.dtype,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        prune_axis_l0=False,
    )
    reference_plan_build_s = time.perf_counter() - plan_build_start
    pruned_plan_build_start = time.perf_counter()
    pruned_plan = make_torch_plan(
        context,
        torch=torch,
        device=device,
        dtype=args.dtype,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        prune_axis_l0=True,
    )
    pruned_plan_build_s = time.perf_counter() - pruned_plan_build_start

    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(np.real(context.ring.obj.coeff).astype(np_complex))
    coeff = torch.as_tensor(coeff_np, dtype=reference_plan.complex_dtype, device=device)
    with torch.inference_mode():
        stress_generator = torch.Generator(device=device).manual_seed(args.stress_seed)
        stress_coeff = torch.complex(
            torch.randn(
                coeff.shape,
                generator=stress_generator,
                dtype=reference_plan.real_dtype,
                device=device,
            ),
            torch.randn(
                coeff.shape,
                generator=stress_generator,
                dtype=reference_plan.real_dtype,
                device=device,
            ),
        ) / math.sqrt(2.0)
        reference_forward = reference_plan.forward(coeff)
        residual = reference_forward * (0.1 + 0.2j)
        reference_adjoint = reference_plan.adjoint(residual)
        stress_residual = torch.complex(
            torch.randn(
                reference_forward.shape,
                generator=stress_generator,
                dtype=reference_plan.real_dtype,
                device=device,
            ),
            torch.randn(
                reference_forward.shape,
                generator=stress_generator,
                dtype=reference_plan.real_dtype,
                device=device,
            ),
        ) / math.sqrt(2.0)
        reference_stress_forward = reference_plan.forward(stress_coeff)
        reference_stress_adjoint = reference_plan.adjoint(stress_residual)
        _, reference_pair_s, reference_pair_times = timed_cuda(
            torch,
            device,
            lambda: reference_plan.forward(reference_plan.adjoint(residual)),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        pruned_forward = pruned_plan.forward(coeff)
        pruned_adjoint = pruned_plan.adjoint(residual)
        pruned_stress_forward = pruned_plan.forward(stress_coeff)
        pruned_stress_adjoint = pruned_plan.adjoint(stress_residual)
        _, pruned_pair_s, pruned_pair_times = timed_cuda(
            torch,
            device,
            lambda: pruned_plan.forward(pruned_plan.adjoint(residual)),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        axis_prune = {
            "axis_l_modes_reference": (
                None if reference_plan.axis is None else int(reference_plan.axis.n_l)
            ),
            "axis_l_modes_pruned": (
                None if pruned_plan.axis is None else int(pruned_plan.axis.n_l)
            ),
            "forward_rel_l2": relative_l2(torch, pruned_forward, reference_forward),
            "adjoint_rel_l2": relative_l2(torch, pruned_adjoint, reference_adjoint),
            "stress_forward_rel_l2": relative_l2(
                torch, pruned_stress_forward, reference_stress_forward
            ),
            "stress_adjoint_rel_l2": relative_l2(
                torch, pruned_stress_adjoint, reference_stress_adjoint
            ),
            "pair_median_s": pruned_pair_s,
            "pair_times_s": pruned_pair_times,
            "speedup_vs_reference": reference_pair_s / pruned_pair_s,
        }

        ring_svd = axial_svd(pruned_plan.ring)
        axis_svd = None if pruned_plan.axis is None else axial_svd(pruned_plan.axis)
        rank_rows = []
        passing_rank = None
        for rank in args.ranks:
            ring_basis = lowrank_basis(pruned_plan.ring, ring_svd, rank=rank)
            axis_basis = (
                None
                if pruned_plan.axis is None or axis_svd is None
                else lowrank_basis(pruned_plan.axis, axis_svd, rank=rank)
            )
            candidate_forward = composite_forward_lowrank(
                pruned_plan, coeff, ring_basis, axis_basis
            )
            candidate_adjoint = composite_adjoint_lowrank(
                pruned_plan, residual, ring_basis, axis_basis
            )
            candidate_stress_forward = composite_forward_lowrank(
                pruned_plan, stress_coeff, ring_basis, axis_basis
            )
            candidate_stress_adjoint = composite_adjoint_lowrank(
                pruned_plan, stress_residual, ring_basis, axis_basis
            )
            _, pair_s, pair_times = timed_cuda(
                torch,
                device,
                lambda rb=ring_basis, ab=axis_basis: composite_forward_lowrank(
                    pruned_plan,
                    composite_adjoint_lowrank(pruned_plan, residual, rb, ab),
                    rb,
                    ab,
                ),
                repeats=args.repeats,
                warmups=args.warmups,
            )
            forward_error = relative_l2(torch, candidate_forward, reference_forward)
            adjoint_error = relative_l2(torch, candidate_adjoint, reference_adjoint)
            stress_forward_error = relative_l2(
                torch, candidate_stress_forward, reference_stress_forward
            )
            stress_adjoint_error = relative_l2(
                torch, candidate_stress_adjoint, reference_stress_adjoint
            )
            dot_lhs = torch.vdot(candidate_forward.reshape(-1), residual.reshape(-1))
            dot_rhs = torch.vdot(coeff.reshape(-1), candidate_adjoint.reshape(-1))
            dot_error = float(
                (
                    torch.abs(dot_lhs - dot_rhs)
                    / torch.clamp(torch.abs(dot_lhs) + torch.abs(dot_rhs), min=1e-30)
                )
                .detach()
                .cpu()
                .item()
            )
            worst_error = max(
                forward_error,
                adjoint_error,
                stress_forward_error,
                stress_adjoint_error,
            )
            passed = worst_error <= args.operator_tolerance
            if passed and passing_rank is None:
                passing_rank = int(rank)
            rank_rows.append(
                {
                    "rank": int(rank),
                    "ring_axial_frobenius_tail": ring_basis.relative_frobenius_tail,
                    "axis_axial_frobenius_tail": (
                        None if axis_basis is None else axis_basis.relative_frobenius_tail
                    ),
                    "forward_rel_l2": forward_error,
                    "adjoint_rel_l2": adjoint_error,
                    "stress_forward_rel_l2": stress_forward_error,
                    "stress_adjoint_rel_l2": stress_adjoint_error,
                    "worst_rel_l2": worst_error,
                    "forward_adjoint_dot_error": dot_error,
                    "pair_median_s": pair_s,
                    "pair_times_s": pair_times,
                    "speedup_vs_reference": reference_pair_s / pair_s,
                    "passed": passed,
                }
            )

    if passing_rank is None:
        passing_rank = max(args.ranks)

    reference_forward_cpu = reference_forward.detach().cpu()
    residual_cpu = residual.detach().cpu()
    reference_adjoint_cpu = reference_adjoint.detach().cpu()
    stress_coeff_cpu = stress_coeff.detach().cpu()
    stress_residual_cpu = stress_residual.detach().cpu()
    reference_stress_forward_cpu = reference_stress_forward.detach().cpu()
    reference_stress_adjoint_cpu = reference_stress_adjoint.detach().cpu()
    coeff_cpu = coeff.detach().cpu()
    del (
        reference_forward,
        residual,
        reference_adjoint,
        coeff,
        pruned_forward,
        pruned_adjoint,
        stress_coeff,
        stress_residual,
        reference_stress_forward,
        reference_stress_adjoint,
        pruned_stress_forward,
        pruned_stress_adjoint,
        reference_plan,
        pruned_plan,
        context,
    )
    gc.collect()
    torch.cuda.empty_cache()

    h_rows = []
    for cutoff in args.h_values:
        case_args = base_odt_args(args, h_cutoff=cutoff)
        case_build_start = time.perf_counter()
        case_context = build_composite_context(case_args)
        context_s = time.perf_counter() - case_build_start
        plan_start = time.perf_counter()
        case_plan = make_torch_plan(
            case_context,
            torch=torch,
            device=device,
            dtype=args.dtype,
            radial_block_size=args.radial_block_size,
            illumination_block_size=args.illumination_block_size,
            prune_axis_l0=True,
        )
        plan_s = time.perf_counter() - plan_start
        coeff_case = coeff_cpu.to(device=device)
        residual_case = residual_cpu.to(device=device)
        forward_reference_case = reference_forward_cpu.to(device=device)
        adjoint_reference_case = reference_adjoint_cpu.to(device=device)
        stress_coeff_case = stress_coeff_cpu.to(device=device)
        stress_residual_case = stress_residual_cpu.to(device=device)
        stress_forward_reference_case = reference_stress_forward_cpu.to(device=device)
        stress_adjoint_reference_case = reference_stress_adjoint_cpu.to(device=device)
        with torch.inference_mode():
            forward_case = case_plan.forward(coeff_case)
            adjoint_case = case_plan.adjoint(residual_case)
            stress_forward_case = case_plan.forward(stress_coeff_case)
            stress_adjoint_case = case_plan.adjoint(stress_residual_case)
            _, pair_s, pair_times = timed_cuda(
                torch,
                device,
                lambda: case_plan.forward(case_plan.adjoint(residual_case)),
                repeats=args.repeats,
                warmups=args.warmups,
            )
            forward_error = relative_l2(torch, forward_case, forward_reference_case)
            adjoint_error = relative_l2(torch, adjoint_case, adjoint_reference_case)
            stress_forward_error = relative_l2(
                torch, stress_forward_case, stress_forward_reference_case
            )
            stress_adjoint_error = relative_l2(
                torch, stress_adjoint_case, stress_adjoint_reference_case
            )

            ring_svd_case = axial_svd(case_plan.ring)
            axis_svd_case = None if case_plan.axis is None else axial_svd(case_plan.axis)
            ring_basis = lowrank_basis(case_plan.ring, ring_svd_case, rank=passing_rank)
            axis_basis = (
                None
                if case_plan.axis is None or axis_svd_case is None
                else lowrank_basis(case_plan.axis, axis_svd_case, rank=passing_rank)
            )
            compound_forward = composite_forward_lowrank(
                case_plan, coeff_case, ring_basis, axis_basis
            )
            compound_adjoint = composite_adjoint_lowrank(
                case_plan, residual_case, ring_basis, axis_basis
            )
            compound_stress_forward = composite_forward_lowrank(
                case_plan, stress_coeff_case, ring_basis, axis_basis
            )
            compound_stress_adjoint = composite_adjoint_lowrank(
                case_plan, stress_residual_case, ring_basis, axis_basis
            )
            _, compound_pair_s, compound_pair_times = timed_cuda(
                torch,
                device,
                lambda: composite_forward_lowrank(
                    case_plan,
                    composite_adjoint_lowrank(
                        case_plan, residual_case, ring_basis, axis_basis
                    ),
                    ring_basis,
                    axis_basis,
                ),
                repeats=args.repeats,
                warmups=args.warmups,
            )
            compound_forward_error = relative_l2(
                torch, compound_forward, forward_reference_case
            )
            compound_adjoint_error = relative_l2(
                torch, compound_adjoint, adjoint_reference_case
            )
            compound_stress_forward_error = relative_l2(
                torch, compound_stress_forward, stress_forward_reference_case
            )
            compound_stress_adjoint_error = relative_l2(
                torch, compound_stress_adjoint, stress_adjoint_reference_case
            )
        exact_worst_error = max(
            forward_error,
            adjoint_error,
            stress_forward_error,
            stress_adjoint_error,
        )
        compound_worst_error = max(
            compound_forward_error,
            compound_adjoint_error,
            compound_stress_forward_error,
            compound_stress_adjoint_error,
        )
        h_rows.append(
            {
                "h_cutoff": int(cutoff),
                "h_modes": int(case_plan.ring.n_h),
                "context_build_s": context_s,
                "plan_build_s": plan_s,
                "forward_rel_l2": forward_error,
                "adjoint_rel_l2": adjoint_error,
                "stress_forward_rel_l2": stress_forward_error,
                "stress_adjoint_rel_l2": stress_adjoint_error,
                "worst_rel_l2": exact_worst_error,
                "pair_median_s": pair_s,
                "pair_times_s": pair_times,
                "speedup_vs_reference": reference_pair_s / pair_s,
                "passed": exact_worst_error <= args.operator_tolerance,
                "compound_rank": int(passing_rank),
                "compound_forward_rel_l2": compound_forward_error,
                "compound_adjoint_rel_l2": compound_adjoint_error,
                "compound_stress_forward_rel_l2": compound_stress_forward_error,
                "compound_stress_adjoint_rel_l2": compound_stress_adjoint_error,
                "compound_worst_rel_l2": compound_worst_error,
                "compound_pair_median_s": compound_pair_s,
                "compound_pair_times_s": compound_pair_times,
                "compound_speedup_vs_reference": reference_pair_s / compound_pair_s,
                "compound_passed": compound_worst_error <= args.operator_tolerance,
            }
        )
        del (
            case_plan,
            case_context,
            coeff_case,
            residual_case,
            forward_reference_case,
            adjoint_reference_case,
            stress_coeff_case,
            stress_residual_case,
            stress_forward_reference_case,
            stress_adjoint_reference_case,
            forward_case,
            adjoint_case,
            stress_forward_case,
            stress_adjoint_case,
            compound_forward,
            compound_adjoint,
            compound_stress_forward,
            compound_stress_adjoint,
            ring_basis,
            axis_basis,
        )
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "schema": "odt-promising-optimization-sweep-v1",
        "generated_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "config": vars(args),
        "problem": {
            "object_shape": [args.n_r, args.n_z, args.n_beta],
            "object_bins": int(args.n_r * args.n_z * args.n_beta),
            "ring_illum": int(args.ring_illum),
            "axis_included": True,
            "cap_radial": int(args.cap_radial),
            "cap_phi": int(args.cap_phi),
        },
        "reference": {
            "h_cutoff": int(args.reference_h_cutoff),
            "context_build_s": context_build_s,
            "plan_build_s": reference_plan_build_s,
            "pruned_plan_build_s": pruned_plan_build_s,
            "pair_median_s": reference_pair_s,
            "pair_times_s": reference_pair_times,
        },
        "support_screen": screen,
        "axis_l0_prune": axis_prune,
        "axial_lowrank": {
            "selected_passing_rank": int(passing_rank),
            "operator_tolerance": float(args.operator_tolerance),
            "rows": rank_rows,
        },
        "h_cutoff": h_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_markdown(args.summary_md, payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    reference = payload["reference"]
    axis = payload["axis_l0_prune"]
    ranks = payload["axial_lowrank"]["rows"]
    h_rows = payload["h_cutoff"]
    lines = [
        "# ODT 유망 최적화 sweep",
        "",
        f"- device: `{payload['device']}`",
        f"- problem: `{payload['problem']['object_shape']}`, "
        f"`{payload['problem']['ring_illum']}+1` illuminations, "
        f"`{payload['problem']['cap_radial']} x {payload['problem']['cap_phi']}` detector",
        f"- reference pair: `{reference['pair_median_s'] * 1e3:.3f} ms`",
        "",
        "## Axis L=0 exact pruning",
        "",
        "| reference L | pruned L | pair ms | speedup | forward rel-L2 | adjoint rel-L2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {axis['axis_l_modes_reference']} | {axis['axis_l_modes_pruned']} | "
        f"{axis['pair_median_s'] * 1e3:.3f} | {axis['speedup_vs_reference']:.3f}x | "
        f"{axis['forward_rel_l2']:.3e} | {axis['adjoint_rel_l2']:.3e} |",
        f"- random-complex stress rel-L2: forward `{axis['stress_forward_rel_l2']:.3e}`, "
        f"adjoint `{axis['stress_adjoint_rel_l2']:.3e}`",
        "",
        "## Axial low-rank sweep",
        "",
        "| rank | pair ms | speedup | physical max | stress max | worst | dot error | pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranks:
        lines.append(
            f"| {row['rank']} | {row['pair_median_s'] * 1e3:.3f} | "
            f"{row['speedup_vs_reference']:.3f}x | "
            f"{max(row['forward_rel_l2'], row['adjoint_rel_l2']):.3e} | "
            f"{max(row['stress_forward_rel_l2'], row['stress_adjoint_rel_l2']):.3e} | "
            f"{row['worst_rel_l2']:.3e} | {row['forward_adjoint_dot_error']:.3e} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## H cutoff and compound sweep",
            "",
            "| H | modes | exact pair ms | exact speedup | exact max error | "
            "compound rank | compound pair ms | compound speedup | compound max error | pass |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in h_rows:
        lines.append(
            f"| {row['h_cutoff']} | {row['h_modes']} | {row['pair_median_s'] * 1e3:.3f} | "
            f"{row['speedup_vs_reference']:.3f}x | "
            f"{row['worst_rel_l2']:.3e} | "
            f"{row['compound_rank']} | {row['compound_pair_median_s'] * 1e3:.3f} | "
            f"{row['compound_speedup_vs_reference']:.3f}x | "
            f"{row['compound_worst_rel_l2']:.3e} | "
            f"{'PASS' if row['compound_passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Axis L=0 pruning is algebraically exact for zero-NA axis illumination.",
            "- Low-rank and H-cutoff rows are controlled approximations and are judged against the full local operator.",
            "- This is a same-machine screening sweep. Publication timing still requires AB/BA repetition and an independent rerun.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep promising ODT GPU optimization candidates.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--n-beta", type=int, default=256)
    p.add_argument("--n-r", type=int, default=256)
    p.add_argument("--n-z", type=int, default=256)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=0.8)
    p.add_argument("--k", type=float, default=17.307319527958313)
    p.add_argument("--detector-na", type=float, default=0.9240924092409241)
    p.add_argument("--illumination-angle-deg", type=float, default=49.0)
    p.add_argument("--ring-illum", type=int, default=120)
    p.add_argument("--cap-radial", type=int, default=256)
    p.add_argument("--cap-phi", type=int, default=256)
    p.add_argument("--reference-h-cutoff", type=int, default=36)
    p.add_argument("--h-values", type=comma_ints, default=[32, 28, 26, 24])
    p.add_argument("--ranks", type=comma_ints, default=[8, 10, 12, 16, 20])
    p.add_argument("--l-margin", type=int, default=18)
    p.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    p.add_argument(
        "--l-thresholds",
        type=lambda value: [float(item) for item in value.split(",")],
        default=[1e-12, 1e-10, 1e-8, 1e-6],
    )
    p.add_argument("--operator-tolerance", type=float, default=2e-6)
    p.add_argument("--stress-seed", type=int, default=20260713)
    p.add_argument("--radial-block-size", type=int, default=32)
    p.add_argument("--illumination-block-size", type=int, default=4)
    p.add_argument("--cpp-threads", type=int, default=4)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_promising_optimization_sweep.json",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_promising_optimization_sweep_ko.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
