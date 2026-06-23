from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    denom = float(np.linalg.norm(reference.ravel()))
    if denom == 0.0:
        return float(np.linalg.norm(candidate.ravel()))
    return float(np.linalg.norm((candidate - reference).ravel()) / denom)


def complex_dot(a: np.ndarray, b: np.ndarray) -> complex:
    return complex(np.vdot(np.ravel(a), np.ravel(b)))


def relative_complex_error(a: complex, b: complex) -> float:
    return float(abs(a - b) / max(abs(a), abs(b), 1e-300))


def median_time(func, *, repeats: int) -> tuple[Any, float, list[float]]:
    value = None
    times: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed function did not run")
    return value, float(median(times)), times


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


@dataclass(frozen=True)
class CylindricalObject:
    r_axis: np.ndarray
    z_axis: np.ndarray
    beta_axis: np.ndarray
    coeff: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    weights: np.ndarray
    volume_weights: np.ndarray

    @property
    def bins(self) -> int:
        return int(self.weights.size)


@dataclass(frozen=True)
class QSamples:
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    q_perp: np.ndarray
    phi: np.ndarray
    illumination_index: np.ndarray

    @property
    def count(self) -> int:
        return int(self.qx.size)


@dataclass(frozen=True)
class StructuredOdtPlan:
    r_axis: np.ndarray
    z_axis: np.ndarray
    beta_axis: np.ndarray
    h_values: np.ndarray
    h_indices: np.ndarray
    n_beta: int

    @classmethod
    def build(
        cls,
        *,
        r_axis: np.ndarray,
        z_axis: np.ndarray,
        beta_axis: np.ndarray,
        h_cutoff: int | None,
    ) -> "StructuredOdtPlan":
        n_beta = int(beta_axis.size)
        h_all = np.fft.fftfreq(n_beta, d=1.0 / n_beta).astype(int)
        if h_cutoff is None:
            mask = np.ones(n_beta, dtype=bool)
        else:
            mask = np.abs(h_all) <= int(h_cutoff)
        return cls(
            r_axis=np.asarray(r_axis, dtype=float),
            z_axis=np.asarray(z_axis, dtype=float),
            beta_axis=np.asarray(beta_axis, dtype=float),
            h_values=np.ascontiguousarray(h_all[mask]),
            h_indices=np.ascontiguousarray(np.nonzero(mask)[0].astype(np.int64)),
            n_beta=n_beta,
        )

    @property
    def used_modes(self) -> int:
        return int(self.h_values.size)


@dataclass(frozen=True)
class StructuredOdtKernel:
    radial: np.ndarray
    axial: np.ndarray
    angular: np.ndarray


@dataclass(frozen=True)
class ShiftedAxisFactorization:
    base_q: QSamples
    illumination: np.ndarray
    phase: np.ndarray
    beta_twiddle: np.ndarray
    kernel: StructuredOdtKernel
    cap_radial: int
    cap_phi: int


def build_structured_kernel(plan: StructuredOdtPlan, q: QSamples) -> StructuredOdtKernel:
    h_values = plan.h_values.astype(int, copy=False)
    abs_h = np.abs(h_values)
    unique_abs_h, inverse_abs_h = np.unique(abs_h, return_inverse=True)

    q_perp_key = np.round(q.q_perp, 12)
    _, unique_q_indices, inverse_q_perp = np.unique(
        q_perp_key,
        return_index=True,
        return_inverse=True,
    )
    q_perp_unique = q.q_perp[unique_q_indices]
    arg = q_perp_unique[None, :, None] * plan.r_axis[None, None, :]
    radial_by_abs = special.jv(unique_abs_h[:, None, None], arg)
    radial = np.ascontiguousarray(radial_by_abs[inverse_abs_h][:, inverse_q_perp, :])
    negative_odd = (h_values < 0) & ((abs_h % 2) == 1)
    radial[negative_odd] *= -1.0

    angular_by_abs = (
        np.exp(0.5j * np.pi * unique_abs_h[:, None])
        * np.exp(-1j * unique_abs_h[:, None] * q.phi[None, :])
    )
    angular = np.ascontiguousarray(angular_by_abs[inverse_abs_h])
    negative_h = h_values < 0
    angular[negative_h] = np.conj(angular[negative_h])
    axial = np.exp(1j * q.qz[:, None] * plan.z_axis[None, :])
    return StructuredOdtKernel(
        radial=np.ascontiguousarray(radial),
        axial=np.ascontiguousarray(axial),
        angular=np.ascontiguousarray(angular),
    )


def _forward_contract_numpy(coeff_h: np.ndarray, kernel: StructuredOdtKernel) -> np.ndarray:
    out = np.zeros(kernel.axial.shape[0], dtype=np.complex128)
    for local_index in range(coeff_h.shape[2]):
        rz_sum = np.einsum(
            "mr,mz,rz->m",
            kernel.radial[local_index],
            kernel.axial,
            coeff_h[:, :, local_index],
            optimize=True,
        )
        out += kernel.angular[local_index] * rz_sum
    return out


def _adjoint_contract_compact_numpy(
    residual: np.ndarray,
    kernel: StructuredOdtKernel,
) -> np.ndarray:
    radial = kernel.radial
    axial_conj = np.conj(kernel.axial)
    out = np.zeros((radial.shape[2], kernel.axial.shape[1], radial.shape[0]), dtype=np.complex128)
    for local_index in range(radial.shape[0]):
        out[:, :, local_index] = np.einsum(
            "m,mr,mz->rz",
            residual * np.conj(kernel.angular[local_index]),
            radial[local_index],
            axial_conj,
            optimize=True,
        )
    return out


def build_shifted_axis_phases(
    plan: StructuredOdtPlan,
    *,
    k: float,
    illumination: np.ndarray,
) -> np.ndarray:
    if k <= 0.0:
        raise ValueError("k must be positive")
    illumination = np.asarray(illumination, dtype=float)
    if illumination.ndim != 2 or illumination.shape[1] != 3:
        raise ValueError("illumination must have shape (n_illum, 3)")
    rr, zz, bb = np.meshgrid(plan.r_axis, plan.z_axis, plan.beta_axis, indexing="ij")
    x = rr * np.cos(bb)
    y = rr * np.sin(bb)
    phases = np.empty(
        (illumination.shape[0], plan.r_axis.size, plan.z_axis.size, plan.n_beta),
        dtype=np.complex128,
    )
    axis = np.array([0.0, 0.0, 1.0], dtype=float)
    for index, s_in in enumerate(illumination):
        delta = float(k) * (axis - s_in)
        phases[index] = np.exp(1j * (delta[0] * x + delta[1] * y + delta[2] * zz))
    return np.ascontiguousarray(phases)


def build_shifted_axis_factorization(
    plan: StructuredOdtPlan,
    *,
    k: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    n_illum: int,
    illumination_na: float,
) -> ShiftedAxisFactorization:
    base_q = ewald_cap_q_samples(
        k=k,
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
        geometry="axis",
        n_illum=1,
        illumination_na=0.0,
    )
    illumination = illumination_directions(
        mode="shifted",
        n_illum=n_illum,
        illumination_na=illumination_na,
    )
    return ShiftedAxisFactorization(
        base_q=base_q,
        illumination=illumination,
        phase=build_shifted_axis_phases(plan, k=k, illumination=illumination),
        beta_twiddle=np.ascontiguousarray(
            np.exp(1j * plan.h_values[:, None] * plan.beta_axis[None, :])
        ),
        kernel=build_structured_kernel(plan, base_q),
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )


def _cpp_odt_module(*, required: bool):
    try:
        from waxs_cake import _cpp_odt
    except ImportError:
        if required:
            raise
        return None
    return _cpp_odt


def resolve_structured_backend(requested: str) -> str:
    if requested not in {"auto", "numpy", "cpp"}:
        raise ValueError("structured backend must be auto, numpy, or cpp")
    if requested == "numpy":
        return "numpy"
    module = _cpp_odt_module(required=requested == "cpp")
    if module is None:
        return "numpy"
    return "cpp"


def make_cylindrical_object(
    *,
    n_r: int,
    n_z: int,
    n_beta: int,
    r_max: float,
    z_max: float,
    phantom: str,
    seed: int,
) -> CylindricalObject:
    if n_r <= 0 or n_z <= 0 or n_beta <= 0:
        raise ValueError("n_r, n_z, and n_beta must be positive")
    if r_max <= 0.0 or z_max <= 0.0:
        raise ValueError("r_max and z_max must be positive")
    dr = float(r_max) / float(n_r)
    dz = 2.0 * float(z_max) / float(n_z)
    dbeta = 2.0 * np.pi / float(n_beta)
    r_axis = (np.arange(n_r, dtype=float) + 0.5) * dr
    z_axis = -float(z_max) + (np.arange(n_z, dtype=float) + 0.5) * dz
    beta_axis = np.linspace(0.0, 2.0 * np.pi, n_beta, endpoint=False, dtype=float)
    rr, zz, bb = np.meshgrid(r_axis, z_axis, beta_axis, indexing="ij")
    x = rr * np.cos(bb)
    y = rr * np.sin(bb)

    values = np.zeros_like(x, dtype=np.complex128)
    if phantom == "beads":
        beads = [
            (-0.48 * r_max, 0.20 * r_max, -0.35 * z_max, 0.20 * r_max, 1.00 + 0.10j),
            (0.35 * r_max, -0.38 * r_max, 0.22 * z_max, 0.17 * r_max, 0.70 - 0.18j),
            (0.15 * r_max, 0.45 * r_max, 0.48 * z_max, 0.14 * r_max, 0.55 + 0.32j),
            (-0.10 * r_max, -0.10 * r_max, -0.05 * z_max, 0.28 * r_max, 0.35 + 0.05j),
        ]
    elif phantom == "random_beads":
        rng = np.random.default_rng(seed)
        beads = []
        for _ in range(9):
            radius = r_max * math.sqrt(float(rng.uniform(0.02, 0.72)))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            cx = radius * math.cos(angle)
            cy = radius * math.sin(angle)
            cz = float(rng.uniform(-0.75 * z_max, 0.75 * z_max))
            sigma = float(rng.uniform(0.08 * r_max, 0.22 * r_max))
            amp = complex(float(rng.uniform(0.25, 1.0)), float(rng.uniform(-0.35, 0.35)))
            beads.append((cx, cy, cz, sigma, amp))
    elif phantom == "shell":
        radius = np.sqrt(x**2 + y**2 + (0.65 * z_axis[None, :, None]) ** 2)
        values = np.exp(-0.5 * ((radius - 0.55 * r_max) / (0.09 * r_max)) ** 2)
        values = values.astype(np.complex128) * (1.0 + 0.15j * np.cos(2.0 * bb))
        beads = []
    else:
        raise ValueError("phantom must be beads, random_beads, or shell")

    for cx, cy, cz, sigma, amp in beads:
        dist2 = (x - cx) ** 2 + (y - cy) ** 2 + (zz - cz) ** 2
        values += amp * np.exp(-0.5 * dist2 / max(float(sigma) ** 2, 1e-300))

    volume = rr * dr * dbeta * dz
    coeff = values * volume
    return CylindricalObject(
        r_axis=r_axis,
        z_axis=z_axis,
        beta_axis=beta_axis,
        coeff=np.ascontiguousarray(coeff),
        x=np.ascontiguousarray(x.ravel()),
        y=np.ascontiguousarray(y.ravel()),
        z=np.ascontiguousarray(zz.ravel()),
        weights=np.ascontiguousarray(coeff.ravel()),
        volume_weights=np.ascontiguousarray(volume.ravel()),
    )


def detector_directions(*, detector_na: float, cap_radial: int, cap_phi: int) -> np.ndarray:
    if detector_na <= 0.0 or detector_na >= 1.0:
        raise ValueError("detector_na must be in (0, 1)")
    radial = (np.arange(cap_radial, dtype=float) + 0.5) * detector_na / float(cap_radial)
    phi = np.linspace(0.0, 2.0 * np.pi, cap_phi, endpoint=False, dtype=float)
    rr, pp = np.meshgrid(radial, phi, indexing="ij")
    sx = rr.ravel() * np.cos(pp.ravel())
    sy = rr.ravel() * np.sin(pp.ravel())
    sz = np.sqrt(np.maximum(1.0 - sx**2 - sy**2, 0.0))
    return np.column_stack([sx, sy, sz])


def illumination_directions(*, mode: str, n_illum: int, illumination_na: float) -> np.ndarray:
    if n_illum <= 0:
        raise ValueError("n_illum must be positive")
    if mode == "axis":
        return np.array([[0.0, 0.0, 1.0]], dtype=float)
    if mode != "shifted":
        raise ValueError("geometry mode must be axis or shifted")
    if illumination_na < 0.0 or illumination_na >= 1.0:
        raise ValueError("illumination_na must be in [0, 1)")
    out = [np.array([0.0, 0.0, 1.0], dtype=float)]
    if n_illum > 1:
        phi = np.linspace(0.0, 2.0 * np.pi, n_illum - 1, endpoint=False, dtype=float)
        sx = illumination_na * np.cos(phi)
        sy = illumination_na * np.sin(phi)
        sz = np.sqrt(np.maximum(1.0 - illumination_na**2, 0.0))
        out.extend(np.column_stack([sx, sy, np.full_like(sx, sz)]))
    return np.asarray(out[:n_illum], dtype=float)


def ewald_cap_q_samples(
    *,
    k: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    geometry: str,
    n_illum: int,
    illumination_na: float,
) -> QSamples:
    if k <= 0.0:
        raise ValueError("k must be positive")
    detector = detector_directions(
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    illum = illumination_directions(
        mode=geometry,
        n_illum=n_illum,
        illumination_na=illumination_na,
    )
    q_blocks = []
    illum_index = []
    for index, s_in in enumerate(illum):
        q_blocks.append(k * (detector - s_in[None, :]))
        illum_index.append(np.full(detector.shape[0], index, dtype=np.int64))
    q = np.vstack(q_blocks)
    illum_flat = np.concatenate(illum_index)
    q_perp = np.hypot(q[:, 0], q[:, 1])
    phi = np.mod(np.arctan2(q[:, 1], q[:, 0]), 2.0 * np.pi)
    return QSamples(
        qx=np.ascontiguousarray(q[:, 0]),
        qy=np.ascontiguousarray(q[:, 1]),
        qz=np.ascontiguousarray(q[:, 2]),
        q_perp=np.ascontiguousarray(q_perp),
        phi=np.ascontiguousarray(phi),
        illumination_index=np.ascontiguousarray(illum_flat),
    )


def direct_forward(
    obj: CylindricalObject,
    q: QSamples,
    *,
    chunk_q: int,
) -> np.ndarray:
    out = np.empty(q.count, dtype=np.complex128)
    xyz = (obj.x, obj.y, obj.z)
    for start in range(0, q.count, chunk_q):
        stop = min(start + chunk_q, q.count)
        phase = (
            q.qx[start:stop, None] * xyz[0][None, :]
            + q.qy[start:stop, None] * xyz[1][None, :]
            + q.qz[start:stop, None] * xyz[2][None, :]
        )
        out[start:stop] = np.exp(1j * phase) @ obj.weights
    return out


def direct_adjoint(
    obj: CylindricalObject,
    q: QSamples,
    residual: np.ndarray,
    *,
    chunk_q: int,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.shape != (q.count,):
        raise ValueError("residual shape must match q sample count")
    out = np.zeros(obj.bins, dtype=np.complex128)
    xyz = (obj.x, obj.y, obj.z)
    for start in range(0, q.count, chunk_q):
        stop = min(start + chunk_q, q.count)
        phase = (
            q.qx[start:stop, None] * xyz[0][None, :]
            + q.qy[start:stop, None] * xyz[1][None, :]
            + q.qz[start:stop, None] * xyz[2][None, :]
        )
        out += np.exp(-1j * phase).T @ residual[start:stop]
    return out.reshape(obj.coeff.shape)


def structured_forward(
    plan: StructuredOdtPlan,
    coeff: np.ndarray,
    q: QSamples,
    kernel: StructuredOdtKernel | None = None,
    *,
    backend: str = "auto",
    cpp_threads: int = 0,
) -> np.ndarray:
    if coeff.shape != (plan.r_axis.size, plan.z_axis.size, plan.n_beta):
        raise ValueError("coeff shape does not match structured plan")
    coeff_h_full = np.fft.ifft(coeff, axis=2) * float(plan.n_beta)
    if kernel is None:
        kernel = build_structured_kernel(plan, q)
    effective_backend = resolve_structured_backend(backend)
    coeff_h = np.ascontiguousarray(coeff_h_full[:, :, plan.h_indices])
    if effective_backend == "cpp":
        cpp_odt = _cpp_odt_module(required=True)
        return cpp_odt.forward_contract(
            coeff_h,
            kernel.radial,
            kernel.axial,
            kernel.angular,
            int(cpp_threads),
        )
    return _forward_contract_numpy(coeff_h, kernel)


def structured_forward_shifted_axis_factored(
    plan: StructuredOdtPlan,
    coeff: np.ndarray,
    factorization: ShiftedAxisFactorization,
    *,
    backend: str = "auto",
    cpp_threads: int = 0,
) -> np.ndarray:
    if coeff.shape != (plan.r_axis.size, plan.z_axis.size, plan.n_beta):
        raise ValueError("coeff shape does not match structured plan")
    if factorization.phase.shape[1:] != coeff.shape:
        raise ValueError("factorization phase shape does not match coefficient grid")
    effective_backend = resolve_structured_backend(backend)
    coeff_mod = factorization.phase * coeff[None, :, :, :]
    coeff_h_full = np.fft.ifft(coeff_mod, axis=3) * float(plan.n_beta)
    coeff_h_all = np.ascontiguousarray(coeff_h_full[:, :, :, plan.h_indices])
    out = np.empty(
        factorization.illumination.shape[0] * factorization.base_q.count,
        dtype=np.complex128,
    )
    if effective_backend == "cpp":
        cpp_odt = _cpp_odt_module(required=True)
        for index in range(factorization.illumination.shape[0]):
            start = index * factorization.base_q.count
            stop = start + factorization.base_q.count
            out[start:stop] = cpp_odt.forward_contract(
                np.ascontiguousarray(coeff_h_all[index]),
                factorization.kernel.radial,
                factorization.kernel.axial,
                factorization.kernel.angular,
                int(cpp_threads),
            )
        return out
    for index in range(factorization.illumination.shape[0]):
        start = index * factorization.base_q.count
        stop = start + factorization.base_q.count
        out[start:stop] = _forward_contract_numpy(coeff_h_all[index], factorization.kernel)
    return out


def _axis_grid_forward_fft(
    plan: StructuredOdtPlan,
    coeff_h_all: np.ndarray,
    factorization: ShiftedAxisFactorization,
    *,
    backend: str = "auto",
    cpp_threads: int = 0,
) -> np.ndarray:
    cap_phi = factorization.cap_phi
    cap_radial = factorization.cap_radial
    if factorization.base_q.count != cap_radial * cap_phi:
        raise ValueError("base q sample count does not match cap_radial * cap_phi")
    radial = factorization.kernel.radial[:, ::cap_phi, :]
    axial = factorization.kernel.axial[::cap_phi, :]
    mode_phase = factorization.kernel.angular[:, 0]
    slots = np.mod(plan.h_values, cap_phi).astype(np.int64)
    cpp_odt = _cpp_odt_module(required=backend == "cpp")
    if cpp_odt is not None and hasattr(cpp_odt, "axis_grid_forward_fold"):
        folded = cpp_odt.axis_grid_forward_fold(
            np.ascontiguousarray(coeff_h_all),
            np.ascontiguousarray(radial),
            np.ascontiguousarray(axial),
            np.ascontiguousarray(mode_phase),
            np.ascontiguousarray(slots),
            int(cap_phi),
            int(cpp_threads),
        )
        return np.fft.fft(folded, axis=2).reshape(
            coeff_h_all.shape[0] * cap_radial * cap_phi
        )
    if backend == "cpp":
        raise RuntimeError("waxs_cake._cpp_odt lacks axis_grid_forward_fold")
    inner = np.einsum(
        "hur,uz,irzh->iuh",
        radial,
        axial,
        coeff_h_all,
        optimize=True,
    )
    folded = np.zeros(
        (coeff_h_all.shape[0], cap_radial, cap_phi),
        dtype=np.complex128,
    )
    for local_index, slot in enumerate(slots):
        folded[:, :, slot] += inner[:, :, local_index] * mode_phase[local_index]
    return np.fft.fft(folded, axis=2).reshape(coeff_h_all.shape[0] * cap_radial * cap_phi)


def _axis_grid_adjoint_fft_compact(
    plan: StructuredOdtPlan,
    factorization: ShiftedAxisFactorization,
    residual: np.ndarray,
    *,
    backend: str = "auto",
    cpp_threads: int = 0,
) -> np.ndarray:
    cap_phi = factorization.cap_phi
    cap_radial = factorization.cap_radial
    n_illum = factorization.illumination.shape[0]
    if residual.shape != (n_illum * cap_radial * cap_phi,):
        raise ValueError("residual shape does not match factored axis grid")
    residual_grid = residual.reshape(n_illum, cap_radial, cap_phi)
    residual_modes = np.fft.ifft(residual_grid, axis=2) * float(cap_phi)
    radial = factorization.kernel.radial[:, ::cap_phi, :]
    axial = factorization.kernel.axial[::cap_phi, :]
    mode_phase = factorization.kernel.angular[:, 0]
    slots = np.mod(plan.h_values, cap_phi).astype(np.int64)
    cpp_odt = _cpp_odt_module(required=backend == "cpp")
    if cpp_odt is not None and hasattr(cpp_odt, "axis_grid_adjoint_unfold"):
        return cpp_odt.axis_grid_adjoint_unfold(
            np.ascontiguousarray(residual_modes),
            np.ascontiguousarray(radial),
            np.ascontiguousarray(axial),
            np.ascontiguousarray(mode_phase),
            np.ascontiguousarray(slots),
            int(cpp_threads),
        )
    if backend == "cpp":
        raise RuntimeError("waxs_cake._cpp_odt lacks axis_grid_adjoint_unfold")
    axial_conj = np.conj(axial)
    mode_phase_conj = np.conj(mode_phase)
    phi_sum = np.empty((n_illum, cap_radial, plan.used_modes), dtype=np.complex128)
    for local_index, slot in enumerate(slots):
        phi_sum[:, :, local_index] = residual_modes[:, :, slot] * mode_phase_conj[
            local_index
        ]
    return np.einsum(
        "iuh,hur,uz->irzh",
        phi_sum,
        radial,
        axial_conj,
        optimize=True,
    )


def structured_forward_shifted_axis_fft_factored(
    plan: StructuredOdtPlan,
    coeff: np.ndarray,
    factorization: ShiftedAxisFactorization,
    *,
    backend: str = "auto",
    cpp_threads: int = 0,
    phase_backend: str = "fft",
) -> np.ndarray:
    if coeff.shape != (plan.r_axis.size, plan.z_axis.size, plan.n_beta):
        raise ValueError("coeff shape does not match structured plan")
    if factorization.phase.shape[1:] != coeff.shape:
        raise ValueError("factorization phase shape does not match coefficient grid")
    if phase_backend not in {"fft", "selected-dft"}:
        raise ValueError("phase_backend must be fft or selected-dft")
    effective_backend = resolve_structured_backend(backend)
    cpp_odt = _cpp_odt_module(required=phase_backend == "selected-dft")
    if phase_backend == "selected-dft":
        if cpp_odt is None or not hasattr(cpp_odt, "phase_selected_dft"):
            raise RuntimeError("waxs_cake._cpp_odt lacks phase_selected_dft")
        coeff_h_all = cpp_odt.phase_selected_dft(
            np.ascontiguousarray(coeff),
            factorization.phase,
            factorization.beta_twiddle,
            int(cpp_threads),
        )
    else:
        coeff_mod = factorization.phase * coeff[None, :, :, :]
        coeff_h_full = np.fft.ifft(coeff_mod, axis=3) * float(plan.n_beta)
        coeff_h_all = np.ascontiguousarray(coeff_h_full[:, :, :, plan.h_indices])
    return _axis_grid_forward_fft(
        plan,
        coeff_h_all,
        factorization,
        backend=effective_backend,
        cpp_threads=cpp_threads,
    )


def structured_adjoint(
    plan: StructuredOdtPlan,
    q: QSamples,
    residual: np.ndarray,
    kernel: StructuredOdtKernel | None = None,
    *,
    backend: str = "auto",
    cpp_threads: int = 0,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.shape != (q.count,):
        raise ValueError("residual shape must match q sample count")
    if kernel is None:
        kernel = build_structured_kernel(plan, q)
    effective_backend = resolve_structured_backend(backend)
    coeff_adjoint_full = np.zeros(
        (plan.r_axis.size, plan.z_axis.size, plan.n_beta),
        dtype=np.complex128,
    )
    if effective_backend == "cpp":
        cpp_odt = _cpp_odt_module(required=True)
        coeff_adjoint_full[:, :, plan.h_indices] = cpp_odt.adjoint_contract_compact(
            np.ascontiguousarray(residual),
            kernel.radial,
            kernel.axial,
            kernel.angular,
            int(cpp_threads),
        )
        return np.fft.fft(coeff_adjoint_full, axis=2)
    coeff_adjoint_full[:, :, plan.h_indices] = _adjoint_contract_compact_numpy(
        residual,
        kernel,
    )
    return np.fft.fft(coeff_adjoint_full, axis=2)


def structured_adjoint_shifted_axis_factored(
    plan: StructuredOdtPlan,
    factorization: ShiftedAxisFactorization,
    residual: np.ndarray,
    *,
    backend: str = "auto",
    cpp_threads: int = 0,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    n_illum = factorization.illumination.shape[0]
    base_count = factorization.base_q.count
    if residual.shape != (n_illum * base_count,):
        raise ValueError("residual shape must match factored shifted q sample count")
    effective_backend = resolve_structured_backend(backend)
    coeff_adjoint_full = np.zeros(
        (n_illum, plan.r_axis.size, plan.z_axis.size, plan.n_beta),
        dtype=np.complex128,
    )
    if effective_backend == "cpp":
        cpp_odt = _cpp_odt_module(required=True)
        for index in range(n_illum):
            start = index * base_count
            stop = start + base_count
            compact = cpp_odt.adjoint_contract_compact(
                np.ascontiguousarray(residual[start:stop]),
                factorization.kernel.radial,
                factorization.kernel.axial,
                factorization.kernel.angular,
                int(cpp_threads),
            )
            coeff_adjoint_full[index][:, :, plan.h_indices] = compact
    else:
        for index in range(n_illum):
            start = index * base_count
            stop = start + base_count
            compact = _adjoint_contract_compact_numpy(
                residual[start:stop],
                factorization.kernel,
            )
            coeff_adjoint_full[index][:, :, plan.h_indices] = compact
    axis_adjoint = np.fft.fft(coeff_adjoint_full, axis=3)
    return np.sum(np.conj(factorization.phase) * axis_adjoint, axis=0)


def structured_adjoint_shifted_axis_fft_factored(
    plan: StructuredOdtPlan,
    factorization: ShiftedAxisFactorization,
    residual: np.ndarray,
    *,
    backend: str = "auto",
    cpp_threads: int = 0,
    phase_backend: str = "fft",
) -> np.ndarray:
    if phase_backend not in {"fft", "selected-dft"}:
        raise ValueError("phase_backend must be fft or selected-dft")
    residual = np.asarray(residual, dtype=np.complex128)
    compact = _axis_grid_adjoint_fft_compact(
        plan,
        factorization,
        residual,
        backend=backend,
        cpp_threads=cpp_threads,
    )
    effective_backend = resolve_structured_backend(backend)
    cpp_odt = _cpp_odt_module(required=phase_backend == "selected-dft")
    if phase_backend == "selected-dft":
        if cpp_odt is None or not hasattr(cpp_odt, "phase_selected_idft_adjoint"):
            raise RuntimeError("waxs_cake._cpp_odt lacks phase_selected_idft_adjoint")
        return cpp_odt.phase_selected_idft_adjoint(
            np.ascontiguousarray(compact),
            factorization.phase,
            factorization.beta_twiddle,
            int(cpp_threads),
        )
    coeff_adjoint_full = np.zeros(
        (
            factorization.illumination.shape[0],
            plan.r_axis.size,
            plan.z_axis.size,
            plan.n_beta,
        ),
        dtype=np.complex128,
    )
    coeff_adjoint_full[:, :, :, plan.h_indices] = compact
    axis_adjoint = np.fft.fft(coeff_adjoint_full, axis=3)
    return np.sum(np.conj(factorization.phase) * axis_adjoint, axis=0)


def finufft_forward(
    obj: CylindricalObject,
    q: QSamples,
    *,
    eps: float,
) -> np.ndarray:
    try:
        import finufft
    except ImportError as exc:
        raise RuntimeError("finufft is not installed") from exc
    return finufft.nufft3d3(
        obj.x,
        obj.y,
        obj.z,
        obj.weights.astype(np.complex128, copy=False),
        q.qx,
        q.qy,
        q.qz,
        eps=eps,
        isign=1,
    )


def finufft_adjoint(
    obj: CylindricalObject,
    q: QSamples,
    residual: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    try:
        import finufft
    except ImportError as exc:
        raise RuntimeError("finufft is not installed") from exc
    out = finufft.nufft3d3(
        q.qx,
        q.qy,
        q.qz,
        residual.astype(np.complex128, copy=False),
        obj.x,
        obj.y,
        obj.z,
        eps=eps,
        isign=-1,
    )
    return out.reshape(obj.coeff.shape)


def random_residual(q: QSamples, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=q.count)
    envelope = 1.0 + 0.25 * rng.standard_normal(q.count)
    illum_scale = 1.0 + 0.05 * q.illumination_index
    return (envelope * illum_scale * np.exp(1j * phase)).astype(np.complex128)


def recommended_h_cutoff(q: QSamples, r_max: float, n_beta: int, margin: int) -> int:
    estimate = int(math.ceil(float(np.max(q.q_perp)) * float(r_max) + int(margin)))
    return max(0, min(n_beta // 2, estimate))


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    q = ewald_cap_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        geometry=args.geometry,
        n_illum=args.n_illum,
        illumination_na=args.illumination_na,
    )
    h_cutoff = (
        recommended_h_cutoff(q, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=h_cutoff,
    )
    effective_structured_backend = resolve_structured_backend(args.structured_backend)
    kernel = None
    kernel_build_s = None
    if not args.no_structured_cache:
        start = time.perf_counter()
        kernel = build_structured_kernel(plan, q)
        kernel_build_s = time.perf_counter() - start
    residual = random_residual(q, seed=args.seed + 7919)

    direct_value, direct_s, direct_times = median_time(
        lambda: direct_forward(obj, q, chunk_q=args.chunk_q),
        repeats=args.repeats,
    )
    structured_value, structured_s, structured_times = median_time(
        lambda: structured_forward(
            plan,
            obj.coeff,
            q,
            kernel=kernel,
            backend=effective_structured_backend,
            cpp_threads=args.cpp_threads,
        ),
        repeats=args.repeats,
    )
    direct_adj_value, direct_adj_s, direct_adj_times = median_time(
        lambda: direct_adjoint(obj, q, residual, chunk_q=args.chunk_q),
        repeats=args.repeats,
    )
    structured_adj_value, structured_adj_s, structured_adj_times = median_time(
        lambda: structured_adjoint(
            plan,
            q,
            residual,
            kernel=kernel,
            backend=effective_structured_backend,
            cpp_threads=args.cpp_threads,
        ),
        repeats=args.repeats,
    )

    finufft_value = None
    finufft_s = None
    finufft_times: list[float] = []
    finufft_adj_value = None
    finufft_adj_s = None
    finufft_adj_times: list[float] = []
    finufft_error = None
    finufft_adj_error = None
    finufft_skip_reason = None
    if not args.skip_finufft:
        try:
            finufft_value, finufft_s, finufft_times = median_time(
                lambda: finufft_forward(obj, q, eps=args.finufft_eps),
                repeats=args.repeats,
            )
            finufft_adj_value, finufft_adj_s, finufft_adj_times = median_time(
                lambda: finufft_adjoint(obj, q, residual, eps=args.finufft_eps),
                repeats=args.repeats,
            )
            finufft_error = relative_l2(finufft_value, direct_value)
            finufft_adj_error = relative_l2(finufft_adj_value, direct_adj_value)
        except Exception as exc:  # pragma: no cover - optional dependency/runtime path
            finufft_skip_reason = str(exc)

    direct_dot_left = complex_dot(direct_value, residual)
    direct_dot_right = complex_dot(obj.coeff, direct_adj_value)
    structured_dot_left = complex_dot(structured_value, residual)
    structured_dot_right = complex_dot(obj.coeff, structured_adj_value)

    row: dict[str, Any] = {
        "status": "ok",
        "geometry": args.geometry,
        "phantom": args.phantom,
        "n_r": args.n_r,
        "n_z": args.n_z,
        "n_beta": args.n_beta,
        "r_max": args.r_max,
        "z_max": args.z_max,
        "k": args.k,
        "detector_na": args.detector_na,
        "illumination_na": args.illumination_na,
        "n_illum": int(np.max(q.illumination_index) + 1),
        "cap_radial": args.cap_radial,
        "cap_phi": args.cap_phi,
        "q_samples": q.count,
        "object_bins": obj.bins,
        "h_cutoff": h_cutoff,
        "used_modes": plan.used_modes,
        "structured_cache": not args.no_structured_cache,
        "structured_backend": effective_structured_backend,
        "requested_structured_backend": args.structured_backend,
        "cpp_threads": args.cpp_threads,
        "structured_kernel_build_s": kernel_build_s,
        "forward_l2_vs_direct": relative_l2(structured_value, direct_value),
        "adjoint_l2_vs_direct": relative_l2(structured_adj_value, direct_adj_value),
        "direct_adjoint_dot_error": relative_complex_error(direct_dot_left, direct_dot_right),
        "structured_adjoint_dot_error": relative_complex_error(
            structured_dot_left,
            structured_dot_right,
        ),
        "direct_forward_s": direct_s,
        "structured_forward_s": structured_s,
        "direct_adjoint_s": direct_adj_s,
        "structured_adjoint_s": structured_adj_s,
        "forward_speedup_direct_vs_structured": direct_s / structured_s,
        "adjoint_speedup_direct_vs_structured": direct_adj_s / structured_adj_s,
        "finufft_forward_s": finufft_s,
        "finufft_adjoint_s": finufft_adj_s,
        "finufft_forward_l2_vs_direct": finufft_error,
        "finufft_adjoint_l2_vs_direct": finufft_adj_error,
        "forward_speedup_finufft_vs_structured": None
        if finufft_s is None
        else finufft_s / structured_s,
        "adjoint_speedup_finufft_vs_structured": None
        if finufft_adj_s is None
        else finufft_adj_s / structured_adj_s,
        "finufft_skip_reason": finufft_skip_reason,
        "direct_forward_times_s": " ".join(f"{value:.9g}" for value in direct_times),
        "structured_forward_times_s": " ".join(f"{value:.9g}" for value in structured_times),
        "direct_adjoint_times_s": " ".join(f"{value:.9g}" for value in direct_adj_times),
        "structured_adjoint_times_s": " ".join(f"{value:.9g}" for value in structured_adj_times),
        "finufft_forward_times_s": " ".join(f"{value:.9g}" for value in finufft_times),
        "finufft_adjoint_times_s": " ".join(f"{value:.9g}" for value in finufft_adj_times),
        "note": (
            "operator acts on weighted cylindrical object coefficients; "
            "this isolates ODT Ewald-cap operator error from Cartesian-to-cylindrical binning error"
        ),
    }
    return {"config": vars(args).copy(), "rows": [row]}


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    row = payload["rows"][0]
    lines = [
        "# ODT Ewald-cap forward-adjoint operator benchmark",
        "",
        "This benchmark is the first third-application check for the curved-manifold Fourier factorization.",
        "It validates an operator-level weak-scattering ODT model: weighted cylindrical object coefficients are mapped to Ewald-cap Fourier samples, and cap residuals are mapped back by the adjoint.",
        "",
        "## Config",
        "",
        f"- geometry: `{row['geometry']}`",
        f"- phantom: `{row['phantom']}`",
        f"- object bins: `{row['object_bins']}` (`n_r={row['n_r']}`, `n_z={row['n_z']}`, `n_beta={row['n_beta']}`)",
        f"- q samples: `{row['q_samples']}` (`n_illum={row['n_illum']}`, `cap_radial={row['cap_radial']}`, `cap_phi={row['cap_phi']}`)",
        f"- h cutoff: `{row['h_cutoff']}`, used modes: `{row['used_modes']}`",
        f"- structured kernel cache: `{row['structured_cache']}`",
        f"- structured backend: `{row['structured_backend']}` (`cpp_threads={row['cpp_threads']}`)",
        "",
        "## Results",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| forward rel-L2 vs direct | `{fmt(row['forward_l2_vs_direct'], 6)}` |",
        f"| adjoint rel-L2 vs direct | `{fmt(row['adjoint_l2_vs_direct'], 6)}` |",
        f"| direct adjoint dot error | `{fmt(row['direct_adjoint_dot_error'], 6)}` |",
        f"| structured adjoint dot error | `{fmt(row['structured_adjoint_dot_error'], 6)}` |",
        f"| structured kernel build s | `{fmt(row['structured_kernel_build_s'], 6)}` |",
        f"| direct forward s | `{fmt(row['direct_forward_s'], 6)}` |",
        f"| structured forward s | `{fmt(row['structured_forward_s'], 6)}` |",
        f"| direct/structured forward speedup | `{fmt(row['forward_speedup_direct_vs_structured'], 4)}x` |",
        f"| direct adjoint s | `{fmt(row['direct_adjoint_s'], 6)}` |",
        f"| structured adjoint s | `{fmt(row['structured_adjoint_s'], 6)}` |",
        f"| direct/structured adjoint speedup | `{fmt(row['adjoint_speedup_direct_vs_structured'], 4)}x` |",
        f"| FINUFFT forward rel-L2 vs direct | `{fmt(row['finufft_forward_l2_vs_direct'], 6)}` |",
        f"| FINUFFT adjoint rel-L2 vs direct | `{fmt(row['finufft_adjoint_l2_vs_direct'], 6)}` |",
        f"| FINUFFT/structured forward speedup | `{fmt(row['forward_speedup_finufft_vs_structured'], 4)}x` |",
        f"| FINUFFT/structured adjoint speedup | `{fmt(row['adjoint_speedup_finufft_vs_structured'], 4)}x` |",
        "",
        "## Interpretation",
        "",
        "- This is not a full ODT reconstruction package and not intensity-only phase retrieval.",
        "- The pass/fail signal is complex forward accuracy and the adjoint inner-product identity.",
        "- The current operator is deliberately defined on weighted cylindrical object coefficients, so the test isolates curved-manifold operator accuracy from Cartesian binning error.",
        "- A shifted-cap run is the stronger evidence that the same factorization is not limited to axis-aligned WAXS-style rings.",
    ]
    if row.get("finufft_skip_reason"):
        lines.extend(["", f"FINUFFT skipped: `{row['finufft_skip_reason']}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an ODT Ewald-cap forward-adjoint operator."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_ewald_cap_operator")
    parser.add_argument("--geometry", choices=["axis", "shifted"], default="shifted")
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--n-beta", type=int, default=96)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--illumination-na", type=float, default=0.25)
    parser.add_argument("--n-illum", type=int, default=9)
    parser.add_argument("--cap-radial", type=int, default=8)
    parser.add_argument("--cap-phi", type=int, default=32)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=14)
    parser.add_argument("--chunk-q", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--skip-finufft", action="store_true")
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="auto")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument(
        "--no-structured-cache",
        action="store_true",
        help="Recompute Bessel/phase kernels inside each structured operator call.",
    )
    args = parser.parse_args()
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    result = run_case(args)
    output_prefix = ROOT / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), result["rows"])
    write_json(output_prefix.with_suffix(".json"), result)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), result)
    print(
        json.dumps(
            {
                "config": result["config"],
                "rows": result["rows"],
                "csv": str(output_prefix.with_suffix(".csv")),
                "json": str(output_prefix.with_suffix(".json")),
                "summary_md": str(output_prefix.with_name(output_prefix.name + "_summary.md")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
