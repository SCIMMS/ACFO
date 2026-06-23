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

from benchmark_high_na_backpropagation import (
    PreparedSeparableAdjointPlan,
    ResidualSpectrumInfo,
    complex_dot,
    direct_debye_wolf_adjoint,
    expand_residual_h_aliases,
    finufft_adjoint,
    focal_grid,
    make_residual,
    relative_complex_error,
    separable_harmonic_adjoint,
)
from benchmark_high_na_debye_wolf import (
    PreparedFinufftDebyeWolfPlan,
    PreparedSeparableHarmonicDebyeWolfPlan,
    direct_debye_wolf,
    gauss_theta_grid,
    median_time,
    pupil_field,
    relative_l2,
    write_csv,
)


@dataclass(frozen=True)
class VectorialWorkload:
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
    vortex_charge: int = 10


def richards_wolf_jones_matrix(
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    apodization: str,
) -> np.ndarray:
    """Aplanatic Richards-Wolf map from transverse Jones pupil to Ex,Ey,Ez."""
    theta_2d = theta[:, None]
    phi_2d = phi[None, :]
    cos_theta = np.cos(theta_2d)
    sin_theta = np.sin(theta_2d)
    cos_phi = np.cos(phi_2d)
    sin_phi = np.sin(phi_2d)

    matrix = np.empty((3, 2, theta.size, phi.size), dtype=np.complex128)
    matrix[0, 0] = cos_theta * cos_phi**2 + sin_phi**2
    matrix[0, 1] = (cos_theta - 1.0) * cos_phi * sin_phi
    matrix[1, 0] = matrix[0, 1]
    matrix[1, 1] = cos_theta * sin_phi**2 + cos_phi**2
    matrix[2, 0] = -sin_theta * cos_phi
    matrix[2, 1] = -sin_theta * sin_phi

    if apodization == "none":
        return matrix
    if apodization == "sqrt-cos":
        return matrix * np.sqrt(np.maximum(cos_theta, 0.0))[None, None, :, :]
    raise ValueError("apodization must be none or sqrt-cos")


def vectorial_pupil_jones(
    case: str,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    theta_max: float,
    strength: float,
    vortex_charge: int,
) -> np.ndarray:
    """Build a two-component pupil Jones field before vectorial focusing."""
    scalar_case = "vortex" if "vortex" in case or "donut" in case else "mixed"
    base = pupil_field(
        scalar_case,
        theta,
        phi,
        theta_max=theta_max,
        strength=strength,
        vortex_charge=vortex_charge,
        apodization="none",
    )
    theta_2d = theta[:, None]
    phi_2d = phi[None, :]
    radial = np.sin(theta_2d) / max(np.sin(theta_max), np.finfo(float).eps)

    out = np.zeros((2, theta.size, phi.size), dtype=np.complex128)
    if case == "x_vortex":
        out[0] = base
    elif case == "mixed_jones":
        out[0] = base
        out[1] = (
            0.35
            * (1.0 + 0.1 * radial * np.sin(2.0 * phi_2d))
            * np.exp(1j * strength * radial**2 * np.sin(2.0 * phi_2d + 0.3))
        )
    elif case == "radial_donut":
        out[0] = base * np.cos(phi_2d)
        out[1] = base * np.sin(phi_2d)
    elif case == "azimuthal_donut":
        out[0] = -base * np.sin(phi_2d)
        out[1] = base * np.cos(phi_2d)
    else:
        raise ValueError(
            "pupil case must be x_vortex, mixed_jones, radial_donut, or azimuthal_donut"
        )
    return out


def mix_vectorial_pupil(
    pupil_jones: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    if pupil_jones.ndim != 3 or pupil_jones.shape[0] != 2:
        raise ValueError("pupil_jones must have shape (2, ntheta, nphi)")
    if mixing.shape[:2] != (3, 2) or mixing.shape[2:] != pupil_jones.shape[1:]:
        raise ValueError("mixing shape must be (3, 2, ntheta, nphi)")
    return np.einsum("cjtp,jtp->ctp", mixing, pupil_jones, optimize=True)


def unmix_vectorial_adjoint(
    effective_gradient: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    if effective_gradient.ndim != 3 or effective_gradient.shape[0] != 3:
        raise ValueError("effective_gradient must have shape (3, ntheta, nphi)")
    return np.einsum(
        "cjtp,ctp->jtp",
        np.conjugate(mixing),
        effective_gradient,
        optimize=True,
    )


def direct_vectorial_debye_wolf(
    pupil_jones: np.ndarray,
    mixing: np.ndarray,
    theta: np.ndarray,
    theta_weights: np.ndarray,
    phi: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
    *,
    k: float,
) -> np.ndarray:
    effective = mix_vectorial_pupil(pupil_jones, mixing)
    return np.stack(
        [
            direct_debye_wolf(
                effective[component],
                theta,
                theta_weights,
                phi,
                rho,
                psi,
                z,
                k=k,
            )
            for component in range(3)
        ],
        axis=0,
    )


def direct_vectorial_adjoint(
    residual: np.ndarray,
    mixing: np.ndarray,
    theta: np.ndarray,
    theta_weights: np.ndarray,
    phi: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
    *,
    k: float,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.ndim != 2 or residual.shape[0] != 3:
        raise ValueError("residual must have shape (3, targets)")
    effective_gradient = np.stack(
        [
            direct_debye_wolf_adjoint(
                residual[component],
                theta,
                theta_weights,
                phi,
                rho,
                psi,
                z,
                k=k,
            )
            for component in range(3)
        ],
        axis=0,
    )
    return unmix_vectorial_adjoint(effective_gradient, mixing)


def separable_vectorial_evaluate(
    plan: PreparedSeparableHarmonicDebyeWolfPlan,
    pupil_jones: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    effective = mix_vectorial_pupil(pupil_jones, mixing)
    return plan.evaluate_many([effective[component] for component in range(3)])


def separable_vectorial_adjoint(
    adjoint_plan: PreparedSeparableAdjointPlan,
    residual: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.ndim != 2 or residual.shape[0] != 3:
        raise ValueError("residual must have shape (3, targets)")
    effective_gradient = np.stack(
        [
            separable_harmonic_adjoint(adjoint_plan, residual[component])
            for component in range(3)
        ],
        axis=0,
    )
    return unmix_vectorial_adjoint(effective_gradient, mixing)


def finufft_vectorial_evaluate(
    plan: PreparedFinufftDebyeWolfPlan,
    pupil_jones: np.ndarray,
    mixing: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    effective = mix_vectorial_pupil(pupil_jones, mixing)
    return plan.evaluate_many([effective[component] for component in range(3)], eps=eps)


def finufft_vectorial_adjoint(
    plan: PreparedFinufftDebyeWolfPlan,
    residual: np.ndarray,
    mixing: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    effective_gradient = np.stack(
        [
            finufft_adjoint(plan, residual[component], eps=eps)
            for component in range(3)
        ],
        axis=0,
    )
    return unmix_vectorial_adjoint(effective_gradient, mixing)


def vectorial_residual(
    *,
    case: str,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    rng: np.random.Generator,
    order: int,
) -> np.ndarray:
    residuals = []
    for component in range(3):
        residual = make_residual(
            case=case,
            rho_axis=rho_axis,
            psi_axis=psi_axis,
            z_axis=z_axis,
            rng=rng,
            order=order + component % 2,
        )
        residuals.append((1.0 / (1.0 + component)) * residual)
    return np.stack(residuals, axis=0)


def vectorial_residual_spectrum_info(
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
    alias_limit: int | None,
) -> ResidualSpectrumInfo:
    if margin < 0:
        raise ValueError("residual h margin must be non-negative")
    grid = np.asarray(residual, dtype=np.complex128).reshape(3, nrho, npsi, nz)
    coeff = np.fft.fft(grid, axis=2) / float(npsi)
    power = np.sqrt(np.sum(np.abs(coeff) ** 2, axis=(0, 1, 3)))
    max_power = float(np.max(power)) if power.size else 0.0
    if max_power == 0.0:
        signed = np.empty(0, dtype=np.int64)
    else:
        threshold = max(float(absolute_threshold), float(relative_threshold) * max_power)
        h_values = np.fft.fftfreq(npsi, d=1.0 / npsi).astype(int)
        signed = h_values[power >= threshold].astype(np.int64, copy=False)
    detected = np.unique(np.abs(signed))
    max_h_abs = nphi // 2 if alias_limit is None else min(nphi // 2, int(alias_limit))
    alias_h_abs = expand_residual_h_aliases(
        signed,
        npsi=npsi,
        max_h_abs=max_h_abs,
    )
    cutoff = int(np.max(alias_h_abs)) + margin if alias_h_abs.size else 0
    cutoff = min(nphi // 2, max(0, cutoff))
    if alias_limit is not None:
        cutoff = min(cutoff, max_h_abs)
    return ResidualSpectrumInfo(
        case=case,
        detected_h_abs=np.ascontiguousarray(detected),
        alias_h_abs=np.ascontiguousarray(alias_h_abs),
        h_cutoff=cutoff,
    )


def vectorial_h_cutoff_for_workload(workload: VectorialWorkload, margin: int) -> int:
    if margin < 0:
        raise ValueError("margin must be non-negative")
    geometric = int(np.ceil(workload.k * workload.rho_max * np.sin(workload.theta_max)))
    if workload.pupil_case == "mixed_jones":
        required = 5
    else:
        required = abs(int(workload.vortex_charge)) + 2
        if workload.pupil_case in {"radial_donut", "azimuthal_donut"}:
            required += 1
    return min(workload.nphi // 2, max(geometric + margin, required))


def shared_phase_gradient(
    pupil_jones: np.ndarray,
    pupil_gradient: np.ndarray,
) -> np.ndarray:
    return np.sum(
        np.imag(np.conjugate(pupil_jones) * pupil_gradient),
        axis=0,
    )


def finite_difference_shared_phase_check(
    *,
    plan: PreparedSeparableHarmonicDebyeWolfPlan,
    mixing: np.ndarray,
    pupil_jones: np.ndarray,
    target: np.ndarray,
    phase_gradient: np.ndarray,
    rng: np.random.Generator,
    step: float,
) -> dict[str, float]:
    direction = rng.standard_normal(pupil_jones.shape[1:])
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise RuntimeError("random phase direction has zero norm")
    direction = direction / norm

    def loss(phase_step: float) -> float:
        trial = pupil_jones * np.exp(1j * phase_step * direction)[None, :, :]
        residual = separable_vectorial_evaluate(plan, trial, mixing) - target
        return 0.5 * float(np.vdot(residual, residual).real)

    plus = loss(step)
    minus = loss(-step)
    finite_difference = (plus - minus) / (2.0 * step)
    prediction = float(np.sum(phase_gradient * direction))
    rel_error = abs(finite_difference - prediction) / max(
        abs(finite_difference),
        abs(prediction),
        1e-300,
    )
    return {
        "phase_fd_directional_derivative": float(finite_difference),
        "phase_adjoint_directional_derivative": float(prediction),
        "phase_gradient_fd_relative_error": float(rel_error),
    }


def workloads(kind: str) -> list[VectorialWorkload]:
    base = [
        VectorialWorkload(
            name="small_vectorial",
            pupil_case="mixed_jones",
            ntheta=14,
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
        VectorialWorkload(
            name="representative_vectorial_vortex",
            pupil_case="radial_donut",
            ntheta=28,
            nphi=128,
            nrho=15,
            npsi=64,
            nz=5,
            rho_max=2.0,
            z_max=1.0,
            theta_max=1.0,
            k=10.0,
            aberration_strength=0.45,
            vortex_charge=14,
        ),
    ]
    large = VectorialWorkload(
        name="large_vectorial_mixed",
        pupil_case="mixed_jones",
        ntheta=42,
        nphi=128,
        nrho=27,
        npsi=80,
        nz=7,
        rho_max=2.8,
        z_max=1.2,
        theta_max=1.05,
        k=12.0,
        aberration_strength=0.5,
        vortex_charge=12,
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
    workload: VectorialWorkload,
    variant: str,
    residual_info: ResidualSpectrumInfo,
    pupil_jones: np.ndarray,
    h_cutoff: int | None,
    used_modes: int | None,
    forward_s: float | None,
    adjoint_s: float,
    forward: np.ndarray | None,
    reference_forward: np.ndarray,
    adjoint: np.ndarray,
    reference_adjoint: np.ndarray,
    dot_forward: np.ndarray,
    residual: np.ndarray,
    gradient_check: dict[str, float] | None = None,
) -> dict[str, Any]:
    left = complex_dot(dot_forward, residual)
    right = complex_dot(pupil_jones, adjoint)
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
        "field_components": 3,
        "pupil_components": 2,
        "forward_s": forward_s,
        "adjoint_s": adjoint_s,
        "forward_l2_vs_reference": 0.0
        if forward is None
        else relative_l2(forward, reference_forward),
        "adjoint_l2_vs_reference": relative_l2(adjoint, reference_adjoint),
        "dot_abs_error": float(abs(left - right)),
        "dot_scale": float(max(abs(left), abs(right))),
        "dot_relative_error": relative_complex_error(left, right),
    }
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


def run_workload(
    workload: VectorialWorkload,
    *,
    repeats: int,
    finufft_eps: float,
    seed: int,
    finite_difference_step: float,
    h_margin: int,
    reference: str,
    residual_case: str,
    residual_order: int,
    residual_h_margin: int,
    residual_relative_threshold: float,
    residual_absolute_threshold: float,
    include_residual_adaptive: bool,
    separable_backend: str,
    apodization: str,
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
    mixing = richards_wolf_jones_matrix(theta, phi, apodization=apodization)
    pupil_jones = vectorial_pupil_jones(
        workload.pupil_case,
        theta,
        phi,
        theta_max=workload.theta_max,
        strength=workload.aberration_strength,
        vortex_charge=workload.vortex_charge,
    )
    residual = vectorial_residual(
        case=residual_case,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        rng=rng,
        order=residual_order,
    )
    h_cutoff = vectorial_h_cutoff_for_workload(workload, h_margin)
    residual_info = vectorial_residual_spectrum_info(
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

    separable_plan = PreparedSeparableHarmonicDebyeWolfPlan.build(
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
    separable_adjoint_plan = PreparedSeparableAdjointPlan.build(separable_plan)
    adaptive_plan = None
    adaptive_adjoint_plan = None
    if include_residual_adaptive:
        adaptive_plan = PreparedSeparableHarmonicDebyeWolfPlan.build(
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
        adaptive_adjoint_plan = PreparedSeparableAdjointPlan.build(adaptive_plan)
    finufft_plan = PreparedFinufftDebyeWolfPlan.build(
        theta,
        theta_weights,
        phi,
        rho,
        psi,
        z,
        k=workload.k,
    )

    separable_forward, separable_forward_s, _ = median_time(
        lambda: separable_vectorial_evaluate(separable_plan, pupil_jones, mixing),
        repeats,
    )
    separable_adjoint_value, separable_adjoint_s, _ = median_time(
        lambda: separable_vectorial_adjoint(separable_adjoint_plan, residual, mixing),
        repeats,
    )
    adaptive_adjoint_value = None
    adaptive_adjoint_s = None
    if adaptive_adjoint_plan is not None:
        adaptive_adjoint_value, adaptive_adjoint_s, _ = median_time(
            lambda: separable_vectorial_adjoint(
                adaptive_adjoint_plan,
                residual,
                mixing,
            ),
            repeats,
        )
    finufft_forward, finufft_forward_s, _ = median_time(
        lambda: finufft_vectorial_evaluate(
            finufft_plan,
            pupil_jones,
            mixing,
            eps=finufft_eps,
        ),
        repeats,
    )
    finufft_adjoint_value, finufft_adjoint_s, _ = median_time(
        lambda: finufft_vectorial_adjoint(
            finufft_plan,
            residual,
            mixing,
            eps=finufft_eps,
        ),
        repeats,
    )

    direct_forward = None
    direct_forward_s = None
    direct_adjoint = None
    direct_adjoint_s = None
    if reference == "direct":
        direct_forward, direct_forward_s, _ = median_time(
            lambda: direct_vectorial_debye_wolf(
                pupil_jones,
                mixing,
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
            lambda: direct_vectorial_adjoint(
                residual,
                mixing,
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
        reference_forward = direct_forward
        reference_adjoint = direct_adjoint
    elif reference == "finufft":
        reference_forward = finufft_forward
        reference_adjoint = finufft_adjoint_value
    else:
        raise ValueError("reference must be direct or finufft")

    target = separable_forward + 0.05 * residual
    loss_residual = separable_forward - target
    pupil_gradient = separable_vectorial_adjoint(
        separable_adjoint_plan,
        loss_residual,
        mixing,
    )
    gradient_check = finite_difference_shared_phase_check(
        plan=separable_plan,
        mixing=mixing,
        pupil_jones=pupil_jones,
        target=target,
        phase_gradient=shared_phase_gradient(pupil_jones, pupil_gradient),
        rng=rng,
        step=finite_difference_step,
    )

    rows: list[dict[str, Any]] = []
    if reference == "direct":
        rows.append(
            result_row(
                workload=workload,
                variant="direct_vectorial_adjoint",
                residual_info=residual_info,
                pupil_jones=pupil_jones,
                h_cutoff=None,
                used_modes=None,
                forward_s=direct_forward_s,
                adjoint_s=direct_adjoint_s,
                forward=None,
                reference_forward=reference_forward,
                adjoint=direct_adjoint,
                reference_adjoint=reference_adjoint,
                dot_forward=direct_forward,
                residual=residual,
            )
        )
    rows.append(
        result_row(
            workload=workload,
            variant="separable_vectorial_adjoint",
            residual_info=residual_info,
            pupil_jones=pupil_jones,
            h_cutoff=h_cutoff,
            used_modes=separable_plan.used_modes,
            forward_s=separable_forward_s,
            adjoint_s=separable_adjoint_s,
            forward=separable_forward,
            reference_forward=reference_forward,
            adjoint=separable_adjoint_value,
            reference_adjoint=reference_adjoint,
            dot_forward=separable_forward,
            residual=residual,
            gradient_check=gradient_check,
        )
    )
    if adaptive_plan is not None:
        if adaptive_adjoint_value is None or adaptive_adjoint_s is None:
            raise RuntimeError("adaptive adjoint is missing")
        rows.append(
            result_row(
                workload=workload,
                variant="separable_residual_adaptive_vectorial_adjoint",
                residual_info=residual_info,
                pupil_jones=pupil_jones,
                h_cutoff=residual_info.h_cutoff,
                used_modes=adaptive_plan.used_modes,
                forward_s=separable_forward_s,
                adjoint_s=adaptive_adjoint_s,
                forward=separable_forward,
                reference_forward=reference_forward,
                adjoint=adaptive_adjoint_value,
                reference_adjoint=reference_adjoint,
                dot_forward=separable_forward,
                residual=residual,
            )
        )
    rows.append(
        result_row(
            workload=workload,
            variant="finufft_vectorial_adjoint",
            residual_info=residual_info,
            pupil_jones=pupil_jones,
            h_cutoff=None,
            used_modes=None,
            forward_s=finufft_forward_s,
            adjoint_s=finufft_adjoint_s,
            forward=finufft_forward,
            reference_forward=reference_forward,
            adjoint=finufft_adjoint_value,
            reference_adjoint=reference_adjoint,
            dot_forward=finufft_forward,
            residual=residual,
        )
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
    reference_label = str(config["reference"])
    lines = [
        "# High-NA vectorial backpropagation benchmark summary",
        "",
        "This benchmark tests vectorial Richards-Wolf/Debye-Wolf forward and adjoint paths.",
        "The vectorial model maps a two-component Jones pupil to Ex,Ey,Ez using an aplanatic focusing matrix.",
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
        f"- residual_adaptive: `{config['include_residual_adaptive']}`",
        f"- separable_backend: `{config['separable_backend']}`",
        f"- apodization: `{config['apodization']}`",
        f"- reference: `{config['reference']}`",
        "",
        "## Results",
        "",
        f"| workload | variant | residual h | alias h | h cutoff | modes | forward s | adjoint s | forward L2 vs {reference_label} | adjoint L2 vs {reference_label} | dot abs | phase FD error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {variant} | {residual_h} | {alias_h} | {h_cutoff} | {modes} | {forward_s} | {adjoint_s} | {forward_l2} | {adjoint_l2} | {dot_abs} | {fd} |".format(
                workload=row["workload"],
                variant=row["variant"],
                residual_h=row["residual_h_max"],
                alias_h=row["residual_alias_h_max"],
                h_cutoff="full" if row["h_cutoff"] is None else row["h_cutoff"],
                modes="n/a" if row["used_modes"] is None else row["used_modes"],
                forward_s=format_float(row["forward_s"]),
                adjoint_s=format_float(row["adjoint_s"]),
                forward_l2=format_float(row["forward_l2_vs_reference"]),
                adjoint_l2=format_float(row["adjoint_l2_vs_reference"]),
                dot_abs=format_float(row["dot_abs_error"]),
                fd=format_float(row["phase_gradient_fd_relative_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the first vectorial extension; it is not yet benchmarked against external optics packages.",
            "- `dot abs` checks the vectorial operator adjoint identity under `numpy.vdot`.",
            "- `phase FD error` checks a shared scalar pupil phase perturbation over both Jones components.",
            "- Exact timings are local to this workspace, build, machine, and benchmark snapshot.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark vectorial High-NA Richards-Wolf adjoint/backpropagation paths."
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
    parser.add_argument("--h-margin", type=int, default=8)
    parser.add_argument("--reference", choices=["direct", "finufft"], default="direct")
    parser.add_argument(
        "--residual-case",
        choices=["random", "low_order", "annular_roi"],
        default="low_order",
    )
    parser.add_argument("--residual-order", type=int, default=6)
    parser.add_argument("--residual-h-margin", type=int, default=2)
    parser.add_argument("--residual-relative-threshold", type=float, default=1e-6)
    parser.add_argument("--residual-absolute-threshold", type=float, default=0.0)
    parser.add_argument("--residual-adaptive", action="store_true")
    parser.add_argument(
        "--separable-backend",
        choices=["auto", "numpy", "cpp"],
        default="auto",
    )
    parser.add_argument(
        "--apodization",
        choices=["none", "sqrt-cos"],
        default="sqrt-cos",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results/high_na_vectorial_backpropagation.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/high_na_vectorial_backpropagation.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("benchmark_results/high_na_vectorial_backpropagation_summary.md"),
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
        "apodization": args.apodization,
        "reference": args.reference,
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
                residual_case=args.residual_case,
                residual_order=args.residual_order,
                residual_h_margin=args.residual_h_margin,
                residual_relative_threshold=args.residual_relative_threshold,
                residual_absolute_threshold=args.residual_absolute_threshold,
                include_residual_adaptive=args.residual_adaptive,
                separable_backend=args.separable_backend,
                apodization=args.apodization,
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
