from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

ADJOINT_CPP_WORK_THRESHOLD = 150_000

from benchmark_high_na_debye_wolf import (
    PreparedFinufftDebyeWolfPlan,
    PreparedHarmonicDebyeWolfPlan,
    PreparedSeparableHarmonicDebyeWolfPlan,
    direct_debye_wolf,
    gauss_theta_grid,
    median_time,
    pupil_field,
    relative_l2,
    write_csv,
)


@dataclass(frozen=True)
class BackpropWorkload:
    name: str
    pupil_case: str
    ntheta: int
    nphi: int
    nrho: int
    npsi: int
    nz: int
    rho_max: float
    z_max: float
    theta_max: float
    k: float
    aberration_strength: float = 0.45
    vortex_charge: int = 12


@dataclass(frozen=True)
class ResidualSpectrumInfo:
    case: str
    detected_h_abs: np.ndarray
    alias_h_abs: np.ndarray
    h_cutoff: int

    @property
    def h_count(self) -> int:
        return int(self.detected_h_abs.size)

    @property
    def h_max(self) -> int:
        if self.detected_h_abs.size == 0:
            return 0
        return int(np.max(self.detected_h_abs))

    @property
    def alias_h_max(self) -> int:
        if self.alias_h_abs.size == 0:
            return 0
        return int(np.max(self.alias_h_abs))


@dataclass(frozen=True)
class PreparedSeparableAdjointPlan:
    """Reusable adjoint-side cache for a separable Debye-Wolf forward plan."""

    forward: PreparedSeparableHarmonicDebyeWolfPlan
    h_indices: np.ndarray
    radial_conj: np.ndarray
    defocus_conj: np.ndarray

    @classmethod
    def build(
        cls,
        forward: PreparedSeparableHarmonicDebyeWolfPlan,
    ) -> "PreparedSeparableAdjointPlan":
        npsi = int(forward.angular.shape[1])
        return cls(
            forward=forward,
            h_indices=np.mod(forward.h, npsi).astype(np.int64, copy=False),
            radial_conj=np.ascontiguousarray(np.conjugate(forward.radial)),
            defocus_conj=np.ascontiguousarray(np.conjugate(forward.defocus)),
        )


def focal_grid(
    *,
    nrho: int,
    npsi: int,
    nz: int,
    rho_max: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho_axis = np.linspace(0.0, rho_max, nrho, dtype=float)
    psi_axis = np.linspace(0.0, 2.0 * np.pi, npsi, endpoint=False, dtype=float)
    if nz == 1:
        z_axis = np.array([0.0], dtype=float)
    else:
        z_axis = np.linspace(-z_max, z_max, nz, dtype=float)
    rr, pp, zz = np.meshgrid(rho_axis, psi_axis, z_axis, indexing="ij")
    return rho_axis, psi_axis, z_axis, rr.ravel(), pp.ravel(), zz.ravel()


def make_residual(
    *,
    case: str,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    rng: np.random.Generator,
    order: int,
) -> np.ndarray:
    if order < 0:
        raise ValueError("residual order must be non-negative")

    shape = (rho_axis.size, psi_axis.size, z_axis.size)
    if case == "random":
        return (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ).astype(np.complex128).ravel()

    rr = rho_axis[:, None, None]
    pp = psi_axis[None, :, None]
    zz = z_axis[None, None, :]
    rho_scale = max(float(np.max(rho_axis)), np.finfo(float).eps)
    z_scale = max(float(np.max(np.abs(z_axis))), np.finfo(float).eps)
    envelope = np.exp(-0.4 * (rr / rho_scale) ** 2) * np.exp(
        -0.3 * (zz / z_scale) ** 2
    )

    if case == "low_order":
        residual = np.zeros(shape, dtype=np.complex128)
        for h in range(order + 1):
            amp = (0.7 ** h) * (
                rng.standard_normal() + 1j * rng.standard_normal()
            )
            rz_texture = envelope * (
                1.0
                + 0.08 * rng.standard_normal() * rr / rho_scale
                + 0.08 * rng.standard_normal() * zz / z_scale
            )
            residual += amp * rz_texture * np.exp(1j * h * pp)
            if h > 0:
                amp_neg = (0.7 ** h) * (
                    rng.standard_normal() + 1j * rng.standard_normal()
                )
                residual += amp_neg * rz_texture * np.exp(-1j * h * pp)
        return residual.ravel()

    if case == "annular_roi":
        ring = np.exp(-8.0 * ((rr / rho_scale) - 0.65) ** 2)
        axial = np.exp(-0.8 * (zz / z_scale) ** 2)
        angular = (
            1.0
            + 0.35 * np.exp(1j * 2.0 * pp)
            + 0.18 * np.exp(-1j * 3.0 * pp)
            + 0.08 * np.exp(1j * min(order, 6) * pp)
        )
        return (ring * axial * angular).astype(np.complex128).ravel()

    raise ValueError("residual case must be random, low_order, or annular_roi")


def significant_residual_h_values(
    residual: np.ndarray,
    *,
    nrho: int,
    npsi: int,
    nz: int,
    relative_threshold: float,
    absolute_threshold: float,
) -> np.ndarray:
    if relative_threshold < 0.0:
        raise ValueError("relative threshold must be non-negative")
    if absolute_threshold < 0.0:
        raise ValueError("absolute threshold must be non-negative")
    grid = np.asarray(residual, dtype=np.complex128).reshape(nrho, npsi, nz)
    coeff = np.fft.fft(grid, axis=1) / float(npsi)
    power = np.sqrt(np.sum(np.abs(coeff) ** 2, axis=(0, 2)))
    max_power = float(np.max(power)) if power.size else 0.0
    threshold = max(float(absolute_threshold), relative_threshold * max_power)
    if max_power == 0.0:
        return np.array([], dtype=np.int64)

    h_values = np.fft.fftfreq(npsi, d=1.0 / npsi).astype(int)
    significant = h_values[power >= threshold]
    return np.unique(significant.astype(np.int64, copy=False))


def expand_residual_h_aliases(
    h_values: np.ndarray,
    *,
    npsi: int,
    max_h_abs: int,
) -> np.ndarray:
    if max_h_abs < 0:
        raise ValueError("max_h_abs must be non-negative")
    expanded: set[int] = set()
    for h_value in np.asarray(h_values, dtype=np.int64):
        h0 = int(h_value)
        if npsi <= 0:
            raise ValueError("npsi must be positive")
        m_min = int(np.floor((-max_h_abs - h0) / npsi))
        m_max = int(np.ceil((max_h_abs - h0) / npsi))
        for m in range(m_min, m_max + 1):
            aliased = h0 + m * npsi
            if abs(aliased) <= max_h_abs:
                expanded.add(abs(int(aliased)))
    if not expanded:
        return np.array([], dtype=np.int64)
    return np.array(sorted(expanded), dtype=np.int64)


def residual_spectrum_info(
    residual: np.ndarray,
    *,
    case: str,
    nrho: int,
    npsi: int,
    nz: int,
    nphi: int,
    margin: int,
    relative_threshold: float,
    absolute_threshold: float,
    alias_limit: int | None = None,
) -> ResidualSpectrumInfo:
    if margin < 0:
        raise ValueError("residual h margin must be non-negative")
    signed = significant_residual_h_values(
        residual,
        nrho=nrho,
        npsi=npsi,
        nz=nz,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    detected = np.unique(np.abs(signed).astype(np.int64, copy=False))
    max_h_abs = nphi // 2 if alias_limit is None else min(nphi // 2, int(alias_limit))
    alias_h_abs = expand_residual_h_aliases(
        signed,
        npsi=npsi,
        max_h_abs=max_h_abs,
    )
    if alias_h_abs.size:
        cutoff = int(np.max(alias_h_abs)) + margin
    else:
        cutoff = 0
    cutoff = min(nphi // 2, max(0, cutoff))
    if alias_limit is not None:
        cutoff = min(cutoff, max_h_abs)
    return ResidualSpectrumInfo(
        case=case,
        detected_h_abs=np.ascontiguousarray(detected),
        alias_h_abs=np.ascontiguousarray(alias_h_abs),
        h_cutoff=cutoff,
    )


def direct_debye_wolf_adjoint(
    residual: np.ndarray,
    theta: np.ndarray,
    theta_weights: np.ndarray,
    phi: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
    *,
    k: float,
) -> np.ndarray:
    """Adjoint of direct_debye_wolf under numpy.vdot convention."""
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.shape != rho.shape:
        raise ValueError("residual shape must match flattened focal target arrays")

    dphi = 2.0 * np.pi / float(phi.size)
    adjoint = np.empty((theta.size, phi.size), dtype=np.complex128)
    phi_col = phi[:, None]
    psi_row = psi[None, :]
    rho_row = rho[None, :]

    for it, theta_i in enumerate(theta):
        sin_theta = float(np.sin(theta_i))
        cos_theta = float(np.cos(theta_i))
        phase = np.exp(
            1j
            * (
                k * sin_theta * rho_row * np.cos(phi_col - psi_row)
                + k * z[None, :] * cos_theta
            )
        )
        adjoint[it] = theta_weights[it] * dphi * np.sum(
            np.conjugate(phase) * residual[None, :],
            axis=1,
        )
    return adjoint


def harmonic_adjoint(
    plan: PreparedHarmonicDebyeWolfPlan,
    residual: np.ndarray,
) -> np.ndarray:
    """Adjoint of PreparedHarmonicDebyeWolfPlan.evaluate."""
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.ndim != 1 or residual.shape[0] != plan.basis.shape[2]:
        raise ValueError("residual must be a flat focal-target vector")

    coeff_adjoint = np.einsum(
        "thp,p->th",
        np.conjugate(plan.basis),
        residual,
        optimize=True,
    )
    full_coeff_adjoint = np.zeros((plan.basis.shape[0], plan.nphi), dtype=np.complex128)
    full_coeff_adjoint[:, plan.mask] = coeff_adjoint
    return np.fft.ifft(full_coeff_adjoint, axis=1)


def separable_harmonic_adjoint(
    plan: PreparedSeparableHarmonicDebyeWolfPlan | PreparedSeparableAdjointPlan,
    residual: np.ndarray,
) -> np.ndarray:
    """Adjoint of PreparedSeparableHarmonicDebyeWolfPlan.evaluate."""
    if isinstance(plan, PreparedSeparableHarmonicDebyeWolfPlan):
        adjoint_plan = PreparedSeparableAdjointPlan.build(plan)
    else:
        adjoint_plan = plan
    forward_plan = adjoint_plan.forward
    residual = np.asarray(residual, dtype=np.complex128)
    nrho = forward_plan.radial.shape[2]
    npsi = forward_plan.angular.shape[1]
    nz = forward_plan.defocus.shape[1]
    if residual.ndim != 1 or residual.size != nrho * npsi * nz:
        raise ValueError("residual must match the flattened separable focal grid")
    residual_grid = residual.reshape(nrho, npsi, nz)
    psi_contracted = np.ascontiguousarray(
        np.fft.fft(residual_grid, axis=1)[:, adjoint_plan.h_indices, :]
    )
    coeff_adjoint = None
    contraction_work = (
        int(forward_plan.radial.shape[0])
        * int(forward_plan.radial.shape[1])
        * nrho
        * nz
    )
    use_cpp = forward_plan.backend == "cpp" or (
        forward_plan.backend == "auto"
        and contraction_work <= ADJOINT_CPP_WORK_THRESHOLD
    )
    if use_cpp:
        try:
            from waxs_cake import _cpp_high_na

            coeff_adjoint = _cpp_high_na.separable_adjoint_contract(
                psi_contracted,
                adjoint_plan.radial_conj,
                adjoint_plan.defocus_conj,
                forward_plan.cpp_threads,
            )
        except (ImportError, AttributeError):
            if forward_plan.backend == "cpp":
                raise
    if coeff_adjoint is None:
        coeff_adjoint = np.einsum(
            "rhz,thr,tz->th",
            psi_contracted,
            adjoint_plan.radial_conj,
            adjoint_plan.defocus_conj,
            optimize=True,
        )
    full_coeff_adjoint = np.zeros(
        (forward_plan.radial.shape[0], forward_plan.nphi),
        dtype=np.complex128,
    )
    full_coeff_adjoint[:, forward_plan.mask] = coeff_adjoint
    return np.fft.ifft(full_coeff_adjoint, axis=1)


def finufft_adjoint(
    plan: PreparedFinufftDebyeWolfPlan,
    residual: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """Adjoint of PreparedFinufftDebyeWolfPlan.evaluate."""
    try:
        import finufft
    except ImportError as exc:
        raise RuntimeError("finufft is not installed") from exc

    residual = np.ascontiguousarray(np.asarray(residual, dtype=np.complex128))
    source_adjoint = finufft.nufft3d3(
        plan.target_s,
        plan.target_t,
        plan.target_u,
        residual,
        plan.source_x,
        plan.source_y,
        plan.source_z,
        eps=eps,
        isign=-1,
    )
    return (source_adjoint.reshape(plan.ntheta, plan.nphi) * plan.source_weight)


def complex_dot(a: np.ndarray, b: np.ndarray) -> complex:
    return complex(np.vdot(np.ravel(a), np.ravel(b)))


def relative_complex_error(a: complex, b: complex) -> float:
    denom = max(abs(a), abs(b), 1e-300)
    return float(abs(a - b) / denom)


def phase_gradient_from_pupil_gradient(
    pupil: np.ndarray,
    pupil_gradient: np.ndarray,
) -> np.ndarray:
    return np.imag(np.conjugate(pupil) * pupil_gradient)


def finite_difference_phase_check(
    *,
    plan: PreparedHarmonicDebyeWolfPlan,
    pupil: np.ndarray,
    target: np.ndarray,
    phase_gradient: np.ndarray,
    rng: np.random.Generator,
    step: float,
) -> dict[str, float]:
    direction = rng.standard_normal(pupil.shape)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0:
        raise RuntimeError("random direction has zero norm")
    direction = direction / direction_norm

    def loss(phase_step: float) -> float:
        trial = pupil * np.exp(1j * phase_step * direction)
        residual = plan.evaluate(trial) - target
        return 0.5 * float(np.vdot(residual, residual).real)

    plus = loss(step)
    minus = loss(-step)
    finite_difference = (plus - minus) / (2.0 * step)
    adjoint_prediction = float(np.sum(phase_gradient * direction))
    rel_error = abs(finite_difference - adjoint_prediction) / max(
        abs(finite_difference),
        abs(adjoint_prediction),
        1e-300,
    )
    return {
        "phase_fd_directional_derivative": float(finite_difference),
        "phase_adjoint_directional_derivative": float(adjoint_prediction),
        "phase_gradient_fd_relative_error": float(rel_error),
    }


def workloads(kind: str) -> list[BackpropWorkload]:
    base = [
        BackpropWorkload(
            name="small_mixed",
            pupil_case="mixed",
            ntheta=16,
            nphi=64,
            nrho=5,
            npsi=16,
            nz=3,
            rho_max=1.2,
            z_max=0.8,
            theta_max=1.0,
            k=8.0,
            aberration_strength=0.35,
            vortex_charge=8,
        ),
        BackpropWorkload(
            name="representative_vortex",
            pupil_case="vortex",
            ntheta=32,
            nphi=128,
            nrho=17,
            npsi=64,
            nz=5,
            rho_max=2.0,
            z_max=1.0,
            theta_max=1.0,
            k=10.0,
            aberration_strength=0.45,
            vortex_charge=18,
        ),
    ]
    large = BackpropWorkload(
        name="large_mixed",
        pupil_case="mixed",
        ntheta=48,
        nphi=128,
        nrho=33,
        npsi=96,
        nz=9,
        rho_max=3.0,
        z_max=1.2,
        theta_max=1.05,
        k=12.0,
        aberration_strength=0.5,
        vortex_charge=16,
    )
    if kind == "quick":
        return base[:1]
    if kind == "representative":
        return base
    if kind == "large":
        return [*base, large]
    raise ValueError("kind must be quick, representative, or large")


def result_row(
    *,
    workload: BackpropWorkload,
    variant: str,
    residual_info: ResidualSpectrumInfo,
    pupil: np.ndarray,
    h_cutoff: int | None,
    used_modes: int | None,
    forward_s: float | None,
    adjoint_s: float,
    forward: np.ndarray | None,
    direct_forward: np.ndarray,
    adjoint: np.ndarray,
    direct_adjoint: np.ndarray,
    dot_forward: np.ndarray,
    residual: np.ndarray,
    gradient_check: dict[str, float] | None = None,
) -> dict[str, Any]:
    left = complex_dot(dot_forward, residual)
    right = complex_dot(pupil, adjoint)
    dot_abs_error = float(abs(left - right))
    dot_scale = float(max(abs(left), abs(right)))
    row: dict[str, Any] = {
        "workload": workload.name,
        "variant": variant,
        "pupil_case": workload.pupil_case,
        "residual_case": residual_info.case,
        "residual_h_max": residual_info.h_max,
        "residual_alias_h_max": residual_info.alias_h_max,
        "residual_h_count": residual_info.h_count,
        "h_cutoff": h_cutoff,
        "used_modes": used_modes,
        "ntheta": workload.ntheta,
        "nphi": workload.nphi,
        "nrho": workload.nrho,
        "npsi": workload.npsi,
        "nz": workload.nz,
        "targets": workload.nrho * workload.npsi * workload.nz,
        "sources": workload.ntheta * workload.nphi,
        "forward_s": forward_s,
        "adjoint_s": adjoint_s,
        "adjoint_l2_vs_direct": relative_l2(adjoint, direct_adjoint),
        "dot_abs_error": dot_abs_error,
        "dot_scale": dot_scale,
        "dot_relative_error": relative_complex_error(left, right),
    }
    if forward is not None:
        row["forward_l2_vs_direct"] = relative_l2(forward, direct_forward)
    else:
        row["forward_l2_vs_direct"] = 0.0
    if gradient_check is not None:
        row.update(gradient_check)
    else:
        row.update(
            {
                "phase_fd_directional_derivative": None,
                "phase_adjoint_directional_derivative": None,
                "phase_gradient_fd_relative_error": None,
            }
        )
    return row


def harmonic_cutoff_for_workload(workload: BackpropWorkload, margin: int) -> int:
    if margin < 0:
        raise ValueError("margin must be non-negative")
    geometric = int(np.ceil(workload.k * workload.rho_max * np.sin(workload.theta_max))) + margin
    required = 0
    if workload.pupil_case == "vortex":
        required = abs(int(workload.vortex_charge))
    elif workload.pupil_case == "mixed":
        required = 3
    elif workload.pupil_case == "astigmatism":
        required = 2
    elif workload.pupil_case == "coma":
        required = 1
    return min(workload.nphi // 2, max(geometric, required))


def run_workload(
    workload: BackpropWorkload,
    *,
    repeats: int,
    finufft_eps: float,
    seed: int,
    finite_difference_step: float,
    h_margin: int,
    reference: str,
    variant_set: str,
    residual_case: str,
    residual_order: int,
    residual_h_margin: int,
    residual_relative_threshold: float,
    residual_absolute_threshold: float,
    include_residual_adaptive: bool,
    separable_backend: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    theta, theta_weights = gauss_theta_grid(workload.ntheta, workload.theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, workload.nphi, endpoint=False, dtype=float)
    rho_axis, psi_axis, z_axis, rho, psi, z = focal_grid(
        nrho=workload.nrho,
        npsi=workload.npsi,
        nz=workload.nz,
        rho_max=workload.rho_max,
        z_max=workload.z_max,
    )
    pupil = pupil_field(
        workload.pupil_case,
        theta,
        phi,
        theta_max=workload.theta_max,
        strength=workload.aberration_strength,
        vortex_charge=workload.vortex_charge,
        apodization="sqrt-cos",
    )
    residual = make_residual(
        case=residual_case,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        rng=rng,
        order=residual_order,
    )
    h_cutoff = harmonic_cutoff_for_workload(workload, h_margin)
    residual_info = residual_spectrum_info(
        residual,
        case=residual_case,
        nrho=workload.nrho,
        npsi=workload.npsi,
        nz=workload.nz,
        nphi=workload.nphi,
        margin=residual_h_margin,
        relative_threshold=residual_relative_threshold,
        absolute_threshold=residual_absolute_threshold,
        alias_limit=h_cutoff,
    )

    harmonic_full_plan = None
    harmonic_cutoff_plan = None
    if variant_set == "all":
        harmonic_full_plan = PreparedHarmonicDebyeWolfPlan.build(
            workload.nphi,
            theta,
            theta_weights,
            rho,
            psi,
            z,
            k=workload.k,
            h_cutoff=None,
        )
        harmonic_cutoff_plan = PreparedHarmonicDebyeWolfPlan.build(
            workload.nphi,
            theta,
            theta_weights,
            rho,
            psi,
            z,
            k=workload.k,
            h_cutoff=h_cutoff,
        )
    elif variant_set != "optimized":
        raise ValueError("variant_set must be all or optimized")
    separable_cutoff_plan = PreparedSeparableHarmonicDebyeWolfPlan.build(
        workload.nphi,
        theta,
        theta_weights,
        rho_axis,
        psi_axis,
        z_axis,
        k=workload.k,
        h_cutoff=h_cutoff,
        backend=separable_backend,
    )
    separable_cutoff_adjoint_plan = PreparedSeparableAdjointPlan.build(
        separable_cutoff_plan
    )
    separable_residual_adaptive_plan = None
    separable_residual_adaptive_adjoint_plan = None
    if include_residual_adaptive:
        separable_residual_adaptive_plan = PreparedSeparableHarmonicDebyeWolfPlan.build(
            workload.nphi,
            theta,
            theta_weights,
            rho_axis,
            psi_axis,
            z_axis,
            k=workload.k,
            h_cutoff=residual_info.h_cutoff,
            backend=separable_backend,
        )
        separable_residual_adaptive_adjoint_plan = PreparedSeparableAdjointPlan.build(
            separable_residual_adaptive_plan
        )
    finufft_plan = PreparedFinufftDebyeWolfPlan.build(
        theta,
        theta_weights,
        phi,
        rho,
        psi,
        z,
        k=workload.k,
    )

    harmonic_full_forward = None
    harmonic_full_forward_s = None
    harmonic_cutoff_forward = None
    harmonic_cutoff_forward_s = None
    if variant_set == "all":
        if harmonic_full_plan is None or harmonic_cutoff_plan is None:
            raise RuntimeError("flattened harmonic plans were not built")
        harmonic_full_forward, harmonic_full_forward_s, _ = median_time(
            lambda: harmonic_full_plan.evaluate(pupil),
            repeats,
        )
        harmonic_cutoff_forward, harmonic_cutoff_forward_s, _ = median_time(
            lambda: harmonic_cutoff_plan.evaluate(pupil),
            repeats,
        )
    separable_cutoff_forward, separable_cutoff_forward_s, _ = median_time(
        lambda: separable_cutoff_plan.evaluate(pupil),
        repeats,
    )
    finufft_forward, finufft_forward_s, _ = median_time(
        lambda: finufft_plan.evaluate(pupil, eps=finufft_eps),
        repeats,
    )

    harmonic_full_adjoint_value = None
    harmonic_full_adjoint_s = None
    harmonic_cutoff_adjoint_value = None
    harmonic_cutoff_adjoint_s = None
    if variant_set == "all":
        if harmonic_full_plan is None or harmonic_cutoff_plan is None:
            raise RuntimeError("flattened harmonic plans were not built")
        harmonic_full_adjoint_value, harmonic_full_adjoint_s, _ = median_time(
            lambda: harmonic_adjoint(harmonic_full_plan, residual),
            repeats,
        )
        harmonic_cutoff_adjoint_value, harmonic_cutoff_adjoint_s, _ = median_time(
            lambda: harmonic_adjoint(harmonic_cutoff_plan, residual),
            repeats,
        )
    separable_cutoff_adjoint_value, separable_cutoff_adjoint_s, _ = median_time(
        lambda: separable_harmonic_adjoint(separable_cutoff_adjoint_plan, residual),
        repeats,
    )
    separable_residual_adaptive_adjoint_value = None
    separable_residual_adaptive_adjoint_s = None
    if separable_residual_adaptive_plan is not None:
        (
            separable_residual_adaptive_adjoint_value,
            separable_residual_adaptive_adjoint_s,
            _,
        ) = median_time(
            lambda: separable_harmonic_adjoint(
                separable_residual_adaptive_adjoint_plan,
                residual,
            ),
            repeats,
        )
    finufft_adjoint_value, finufft_adjoint_s, _ = median_time(
        lambda: finufft_adjoint(finufft_plan, residual, eps=finufft_eps),
        repeats,
    )

    direct_forward = None
    direct_forward_s = None
    direct_adjoint = None
    direct_adjoint_s = None
    if reference == "direct":
        direct_forward, direct_forward_s, _ = median_time(
            lambda: direct_debye_wolf(
                pupil,
                theta,
                theta_weights,
                phi,
                rho,
                psi,
                z,
                k=workload.k,
            ),
            repeats,
        )
        direct_adjoint, direct_adjoint_s, _ = median_time(
            lambda: direct_debye_wolf_adjoint(
                residual,
                theta,
                theta_weights,
                phi,
                rho,
                psi,
                z,
                k=workload.k,
            ),
            repeats,
        )
    elif reference == "finufft":
        direct_forward = finufft_forward
        direct_forward_s = finufft_forward_s
        direct_adjoint = finufft_adjoint_value
        direct_adjoint_s = finufft_adjoint_s
    else:
        raise ValueError("reference must be direct or finufft")

    target = separable_cutoff_forward + 0.05 * residual
    loss_residual = separable_cutoff_forward - target
    pupil_gradient = separable_harmonic_adjoint(
        separable_cutoff_adjoint_plan,
        loss_residual,
    )
    phase_gradient = phase_gradient_from_pupil_gradient(pupil, pupil_gradient)
    gradient_check = finite_difference_phase_check(
        plan=separable_cutoff_plan,
        pupil=pupil,
        target=target,
        phase_gradient=phase_gradient,
        rng=rng,
        step=finite_difference_step,
    )
    adaptive_gradient_check = None
    if separable_residual_adaptive_plan is not None:
        adaptive_pupil_gradient = separable_harmonic_adjoint(
            separable_residual_adaptive_adjoint_plan,
            loss_residual,
        )
        adaptive_phase_gradient = phase_gradient_from_pupil_gradient(
            pupil,
            adaptive_pupil_gradient,
        )
        adaptive_gradient_check = finite_difference_phase_check(
            plan=separable_cutoff_plan,
            pupil=pupil,
            target=target,
            phase_gradient=adaptive_phase_gradient,
            rng=rng,
            step=finite_difference_step,
        )

    rows: list[dict[str, Any]] = []
    if reference == "direct":
        rows.append(
            result_row(
                workload=workload,
                variant="direct_adjoint",
                residual_info=residual_info,
                pupil=pupil,
                h_cutoff=None,
                used_modes=None,
                forward_s=direct_forward_s,
                adjoint_s=direct_adjoint_s,
                forward=None,
                direct_forward=direct_forward,
                adjoint=direct_adjoint,
                direct_adjoint=direct_adjoint,
                dot_forward=direct_forward,
                residual=residual,
            )
        )
    if variant_set == "all":
        if (
            harmonic_full_plan is None
            or harmonic_cutoff_plan is None
            or harmonic_full_forward is None
            or harmonic_cutoff_forward is None
            or harmonic_full_adjoint_value is None
            or harmonic_cutoff_adjoint_value is None
        ):
            raise RuntimeError("flattened harmonic variant results are missing")
        rows.extend(
            [
                result_row(
                    workload=workload,
                    variant="harmonic_full_adjoint",
                    residual_info=residual_info,
                    pupil=pupil,
                    h_cutoff=None,
                    used_modes=harmonic_full_plan.used_modes,
                    forward_s=harmonic_full_forward_s,
                    adjoint_s=harmonic_full_adjoint_s,
                    forward=harmonic_full_forward,
                    direct_forward=direct_forward,
                    adjoint=harmonic_full_adjoint_value,
                    direct_adjoint=direct_adjoint,
                    dot_forward=harmonic_full_forward,
                    residual=residual,
                ),
                result_row(
                    workload=workload,
                    variant="harmonic_cutoff_adjoint",
                    residual_info=residual_info,
                    pupil=pupil,
                    h_cutoff=h_cutoff,
                    used_modes=harmonic_cutoff_plan.used_modes,
                    forward_s=harmonic_cutoff_forward_s,
                    adjoint_s=harmonic_cutoff_adjoint_s,
                    forward=harmonic_cutoff_forward,
                    direct_forward=direct_forward,
                    adjoint=harmonic_cutoff_adjoint_value,
                    direct_adjoint=direct_adjoint,
                    dot_forward=harmonic_cutoff_forward,
                    residual=residual,
                ),
            ]
        )
    rows.extend(
        [
        result_row(
            workload=workload,
            variant="separable_cutoff_adjoint",
            residual_info=residual_info,
            pupil=pupil,
            h_cutoff=h_cutoff,
            used_modes=separable_cutoff_plan.used_modes,
            forward_s=separable_cutoff_forward_s,
            adjoint_s=separable_cutoff_adjoint_s,
            forward=separable_cutoff_forward,
            direct_forward=direct_forward,
            adjoint=separable_cutoff_adjoint_value,
            direct_adjoint=direct_adjoint,
            dot_forward=separable_cutoff_forward,
            residual=residual,
            gradient_check=gradient_check,
        ),
        ]
    )
    if separable_residual_adaptive_plan is not None:
        if (
            separable_residual_adaptive_adjoint_value is None
            or separable_residual_adaptive_adjoint_s is None
            or adaptive_gradient_check is None
        ):
            raise RuntimeError("residual-adaptive adjoint result is missing")
        rows.append(
            result_row(
                workload=workload,
                variant="separable_residual_adaptive_adjoint",
                residual_info=residual_info,
                pupil=pupil,
                h_cutoff=residual_info.h_cutoff,
                used_modes=separable_residual_adaptive_plan.used_modes,
                forward_s=separable_cutoff_forward_s,
                adjoint_s=separable_residual_adaptive_adjoint_s,
                forward=separable_cutoff_forward,
                direct_forward=direct_forward,
                adjoint=separable_residual_adaptive_adjoint_value,
                direct_adjoint=direct_adjoint,
                dot_forward=separable_cutoff_forward,
                residual=residual,
                gradient_check=adaptive_gradient_check,
            )
        )
    rows.extend(
        [
        result_row(
            workload=workload,
            variant="finufft_adjoint",
            residual_info=residual_info,
            pupil=pupil,
            h_cutoff=None,
            used_modes=None,
            forward_s=finufft_forward_s,
            adjoint_s=finufft_adjoint_s,
            forward=finufft_forward,
            direct_forward=direct_forward,
            adjoint=finufft_adjoint_value,
            direct_adjoint=direct_adjoint,
            dot_forward=finufft_forward,
            residual=residual,
        ),
        ]
    )
    return rows


def write_json(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"config": config, "rows": rows}, indent=2),
        encoding="utf-8",
    )


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def write_summary(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reference_label = str(config.get("reference", "direct"))
    if reference_label == "direct":
        reference_note = (
            "Direct adjoint is the correctness reference; FINUFFT type-3 adjoint "
            "is the generic optimized baseline."
        )
    else:
        reference_note = (
            "FINUFFT type-3 adjoint is the row accuracy reference; direct "
            "Debye-Wolf is omitted for this scaling-oriented run."
        )
    lines = [
        "# High-NA backpropagation benchmark summary",
        "",
        "This benchmark tests scalar Debye-Wolf adjoint/backpropagation paths.",
        reference_note,
        "",
        "## Config",
        "",
        f"- workload_set: `{config['workload_set']}`",
        f"- repeats: `{config['repeats']}`",
        f"- finufft_eps: `{config['finufft_eps']}`",
        f"- h_margin: `{config['h_margin']}`",
        f"- residual_case: `{config['residual_case']}`",
        f"- residual_order: `{config['residual_order']}`",
        f"- residual_h_margin: `{config['residual_h_margin']}`",
        f"- residual_relative_threshold: `{config['residual_relative_threshold']}`",
        f"- residual_adaptive: `{config['include_residual_adaptive']}`",
        f"- separable_backend: `{config['separable_backend']}`",
        f"- reference: `{config['reference']}`",
        f"- variant_set: `{config['variant_set']}`",
        f"- finite_difference_step: `{config['finite_difference_step']}`",
        "",
        "## Results",
        "",
        f"| workload | variant | residual h | alias h | h cutoff | modes | forward s | adjoint s | adjoint L2 vs {reference_label} | dot rel | dot abs | phase FD error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {variant} | {residual_h} | {alias_h} | {h_cutoff} | {used_modes} | {forward_s} | {adjoint_s} | {adj_l2} | {dot} | {dot_abs} | {fd} |".format(
                workload=row["workload"],
                variant=row["variant"],
                residual_h=row["residual_h_max"],
                alias_h=row["residual_alias_h_max"],
                h_cutoff="full" if row["h_cutoff"] is None else row["h_cutoff"],
                used_modes="n/a" if row["used_modes"] is None else row["used_modes"],
                forward_s=format_float(row["forward_s"]),
                adjoint_s=format_float(row["adjoint_s"]),
                adj_l2=format_float(row["adjoint_l2_vs_direct"]),
                dot=format_float(row["dot_relative_error"]),
                dot_abs=format_float(row["dot_abs_error"]),
                fd=format_float(row["phase_gradient_fd_relative_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `dot error` checks the operator adjoint identity under `numpy.vdot`.",
            "- `dot rel` can be unstable when the tested inner product is near zero; use `dot abs` in that case.",
            "- `phase FD error` is only reported for the separable cutoff adjoint and checks a real phase-only pupil gradient.",
            "- Exact timings are local to this workspace, build, machine, and benchmark snapshot.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark scalar High-NA Debye-Wolf adjoint/backpropagation paths."
    )
    parser.add_argument(
        "--workload-set",
        choices=["quick", "representative", "large"],
        default="representative",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument(
        "--h-margin",
        type=int,
        default=8,
        help="Global harmonic cutoff margin for the cutoff adjoint variant.",
    )
    parser.add_argument(
        "--reference",
        choices=["direct", "finufft"],
        default="direct",
        help="Use direct Debye-Wolf or FINUFFT as the row accuracy reference.",
    )
    parser.add_argument(
        "--variant-set",
        choices=["all", "optimized"],
        default="all",
        help="Use all variants or only separable cutoff plus FINUFFT.",
    )
    parser.add_argument(
        "--residual-case",
        choices=["random", "low_order", "annular_roi"],
        default="random",
        help="Residual pattern used for the adjoint/backprop benchmark.",
    )
    parser.add_argument(
        "--residual-order",
        type=int,
        default=6,
        help="Maximum azimuthal order for structured residual cases.",
    )
    parser.add_argument(
        "--residual-h-margin",
        type=int,
        default=2,
        help="Extra h margin added to the detected residual spectrum cutoff.",
    )
    parser.add_argument(
        "--residual-relative-threshold",
        type=float,
        default=1e-6,
        help="Relative threshold for residual psi-spectrum detection.",
    )
    parser.add_argument(
        "--residual-absolute-threshold",
        type=float,
        default=0.0,
        help="Absolute threshold for residual psi-spectrum detection.",
    )
    parser.add_argument(
        "--residual-adaptive",
        action="store_true",
        help="Add a residual-spectrum-adaptive separable adjoint variant.",
    )
    parser.add_argument(
        "--separable-backend",
        choices=["auto", "numpy", "cpp"],
        default="auto",
        help="Backend for separable forward/adjoint contractions.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results/high_na_backpropagation.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/high_na_backpropagation.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("benchmark_results/high_na_backpropagation_summary.md"),
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.finite_difference_step <= 0.0:
        raise ValueError("finite difference step must be positive")
    if args.h_margin < 0:
        raise ValueError("h margin must be non-negative")
    if args.residual_order < 0:
        raise ValueError("residual order must be non-negative")
    if args.residual_h_margin < 0:
        raise ValueError("residual h margin must be non-negative")
    if args.residual_relative_threshold < 0.0:
        raise ValueError("residual relative threshold must be non-negative")
    if args.residual_absolute_threshold < 0.0:
        raise ValueError("residual absolute threshold must be non-negative")

    config = {
        "workload_set": args.workload_set,
        "repeats": args.repeats,
        "finufft_eps": args.finufft_eps,
        "seed": args.seed,
        "finite_difference_step": args.finite_difference_step,
        "h_margin": args.h_margin,
        "residual_case": args.residual_case,
        "residual_order": args.residual_order,
        "residual_h_margin": args.residual_h_margin,
        "residual_relative_threshold": args.residual_relative_threshold,
        "residual_absolute_threshold": args.residual_absolute_threshold,
        "include_residual_adaptive": args.residual_adaptive,
        "separable_backend": args.separable_backend,
        "reference": args.reference,
        "variant_set": args.variant_set,
    }
    rows: list[dict[str, Any]] = []
    for index, workload in enumerate(workloads(args.workload_set)):
        rows.extend(
            run_workload(
                workload,
                repeats=args.repeats,
                finufft_eps=args.finufft_eps,
                seed=args.seed + index,
                finite_difference_step=args.finite_difference_step,
                h_margin=args.h_margin,
                reference=args.reference,
                variant_set=args.variant_set,
                residual_case=args.residual_case,
                residual_order=args.residual_order,
                residual_h_margin=args.residual_h_margin,
                residual_relative_threshold=args.residual_relative_threshold,
                residual_absolute_threshold=args.residual_absolute_threshold,
                include_residual_adaptive=args.residual_adaptive,
                separable_backend=args.separable_backend,
            )
        )
        print(f"{workload.name}: done")

    write_csv(args.csv, rows)
    write_json(args.out, config, rows)
    write_summary(args.summary, rows, config)
    print(f"wrote {args.csv}")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
