from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import importlib.util
import json
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
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_backpropagation import (  # noqa: E402
    PreparedSeparableAdjointPlan,
    complex_dot,
    focal_grid,
    relative_complex_error,
)
from benchmark_high_na_debye_wolf import (  # noqa: E402
    PreparedSeparableHarmonicDebyeWolfPlan,
    gauss_theta_grid,
    relative_l2,
    write_csv,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    richards_wolf_jones_matrix,
    separable_vectorial_adjoint,
    separable_vectorial_evaluate,
    vectorial_h_cutoff_for_workload,
    vectorial_pupil_jones,
    vectorial_residual,
    workloads,
)


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def import_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


def torch_dtypes(torch: Any, dtype: str) -> tuple[Any, Any]:
    if dtype == "complex64":
        return torch.complex64, torch.float32
    if dtype == "complex128":
        return torch.complex128, torch.float64
    raise ValueError("dtype must be complex64 or complex128")


def sanitize_h_abs(required_h_abs: np.ndarray | list[int] | None, *, nphi: int) -> np.ndarray:
    if required_h_abs is None:
        return np.empty(0, dtype=np.int64)
    values = np.asarray(required_h_abs, dtype=np.int64).ravel()
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    n_half = int(nphi) // 2
    values = np.unique(np.abs(values))
    return np.ascontiguousarray(values[values <= n_half])


def significant_h_abs_from_stack(
    values: np.ndarray,
    *,
    relative_threshold: float,
    absolute_threshold: float,
) -> np.ndarray:
    """Return significant azimuthal |h| values for an arbitrary pupil stack."""
    if relative_threshold < 0.0:
        raise ValueError("relative_threshold must be non-negative")
    if absolute_threshold < 0.0:
        raise ValueError("absolute_threshold must be non-negative")
    stack = np.asarray(values)
    if stack.ndim < 2:
        raise ValueError("values must have at least theta and phi axes")
    nphi = int(stack.shape[-1])
    coeff = np.fft.fft(stack, axis=-1) / float(nphi)
    amplitudes = np.max(np.abs(coeff), axis=tuple(range(coeff.ndim - 1)))
    max_amplitude = float(np.max(amplitudes))
    if max_amplitude == 0.0:
        return np.array([0], dtype=np.int64)
    threshold = max(float(absolute_threshold), float(relative_threshold) * max_amplitude)
    h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
    h_abs = np.abs(h_values)
    return np.ascontiguousarray(np.unique(h_abs[amplitudes >= threshold]).astype(np.int64))


def resolve_required_h_abs(
    *,
    pupil_spectrum: str,
    spectrum_h_abs: np.ndarray,
    h_cutoff: int,
    nphi: int,
) -> np.ndarray:
    if pupil_spectrum == "off":
        return np.empty(0, dtype=np.int64)
    if pupil_spectrum not in {"adaptive", "dense-prefix"}:
        raise ValueError("pupil_spectrum must be 'off', 'adaptive', or 'dense-prefix'")
    spectrum_h_abs = sanitize_h_abs(spectrum_h_abs, nphi=nphi)
    extra = spectrum_h_abs[spectrum_h_abs > int(h_cutoff)]
    if pupil_spectrum == "adaptive" or extra.size == 0:
        return np.ascontiguousarray(extra)
    return np.ascontiguousarray(
        np.arange(int(h_cutoff) + 1, int(np.max(extra)) + 1, dtype=np.int64)
    )


def resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_torch(
    torch: Any,
    device: Any,
    func,
    *,
    repeats: int,
    warmups: int,
) -> tuple[Any, float, list[float]]:
    def run_once() -> Any:
        if hasattr(torch, "inference_mode"):
            with torch.inference_mode():
                return func()
        with torch.no_grad():
            return func()

    value = None
    for _ in range(max(0, warmups)):
        value = run_once()
        synchronize(torch, device)
    times: list[float] = []
    for _ in range(max(1, repeats)):
        gc.collect()
        synchronize(torch, device)
        start = time.perf_counter()
        value = run_once()
        synchronize(torch, device)
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed function did not run")
    return value, float(median(times)), times


def timed_numpy(func, repeats: int) -> tuple[Any, float, list[float]]:
    value = None
    times: list[float] = []
    for _ in range(max(1, repeats)):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed function did not run")
    return value, float(median(times)), times


@dataclass
class TorchSeparableHarmonicDebyeWolfPlan:
    h: np.ndarray
    mask: np.ndarray
    mask_indices: Any
    h_indices: Any
    nphi: int
    nrho: int
    npsi: int
    nz: int
    radial: Any
    angular: Any
    defocus: Any
    radial_defocus: Any | None
    radial_defocus_conj: Any | None
    radial_defocus_forward_bmm: Any | None
    radial_defocus_adjoint_bmm: Any | None
    basis_mode: str
    contract_mode: str
    geometric_h_cutoff: int
    required_h_abs: np.ndarray
    device: Any
    complex_dtype: Any
    real_dtype: Any
    torch: Any
    _adjoint_coeff_buffer: Any | None = None

    @classmethod
    def build(
        cls,
        *,
        torch: Any,
        nphi: int,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        rho_axis: np.ndarray,
        psi_axis: np.ndarray,
        z_axis: np.ndarray,
        k: float,
        h_cutoff: int | None,
        required_h_abs: np.ndarray | list[int] | None = None,
        device: Any,
        dtype: str,
        basis_mode: str = "separate",
        contract_mode: str = "einsum",
    ) -> "TorchSeparableHarmonicDebyeWolfPlan":
        if basis_mode not in {"separate", "fused"}:
            raise ValueError("basis_mode must be 'separate' or 'fused'")
        if contract_mode not in {"einsum", "matmul"}:
            raise ValueError("contract_mode must be 'einsum' or 'matmul'")
        if contract_mode == "matmul" and basis_mode != "fused":
            raise ValueError("matmul contract_mode currently requires basis_mode='fused'")
        complex_dtype, real_dtype = torch_dtypes(torch, dtype)
        h_values = np.fft.fftfreq(nphi, d=1.0 / nphi).astype(int)
        n_half = int(nphi) // 2
        if h_cutoff is None:
            geometric_h_cutoff = n_half
            mask = np.ones(h_values.shape, dtype=bool)
        else:
            geometric_h_cutoff = min(max(int(h_cutoff), 0), n_half)
            mask = np.abs(h_values) <= geometric_h_cutoff
        required_h_abs_array = sanitize_h_abs(required_h_abs, nphi=nphi)
        if required_h_abs_array.size:
            mask = mask | np.isin(np.abs(h_values), required_h_abs_array)
        h = h_values[mask]
        abs_h = np.abs(h)
        unique_abs_h, inverse_abs_h = np.unique(abs_h, return_inverse=True)
        i_pow = np.power(1j, abs_h)
        arg = k * np.sin(theta)[:, None, None] * rho_axis[None, None, :]
        radial_kernel = special.jv(unique_abs_h[None, :, None], arg)[:, inverse_abs_h, :]
        radial = (
            2.0
            * np.pi
            * theta_weights[:, None, None]
            * i_pow[None, :, None]
            * radial_kernel
        )
        angular = np.exp(1j * h[:, None] * psi_axis[None, :])
        defocus = np.exp(1j * k * np.cos(theta)[:, None] * z_axis[None, :])
        np_dtype = np.complex64 if dtype == "complex64" else np.complex128
        radial_t = torch.as_tensor(
            np.ascontiguousarray(radial.astype(np_dtype, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        angular_t = torch.as_tensor(
            np.ascontiguousarray(angular.astype(np_dtype, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        defocus_t = torch.as_tensor(
            np.ascontiguousarray(defocus.astype(np_dtype, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        radial_defocus_t = None
        radial_defocus_conj_t = None
        radial_defocus_forward_bmm_t = None
        radial_defocus_adjoint_bmm_t = None
        if basis_mode == "fused":
            radial_defocus = radial[:, :, :, None] * defocus[:, None, None, :]
            if contract_mode == "matmul":
                radial_defocus_forward_bmm = np.transpose(
                    radial_defocus,
                    (1, 0, 2, 3),
                ).reshape(h.size, theta.size, rho_axis.size * z_axis.size)
                radial_defocus_adjoint_bmm = np.transpose(
                    np.conjugate(radial_defocus),
                    (1, 2, 3, 0),
                ).reshape(h.size, rho_axis.size * z_axis.size, theta.size)
                radial_defocus_forward_bmm_t = torch.as_tensor(
                    np.ascontiguousarray(
                        radial_defocus_forward_bmm.astype(np_dtype, copy=False)
                    ),
                    dtype=complex_dtype,
                    device=device,
                ).contiguous()
                radial_defocus_adjoint_bmm_t = torch.as_tensor(
                    np.ascontiguousarray(
                        radial_defocus_adjoint_bmm.astype(np_dtype, copy=False)
                    ),
                    dtype=complex_dtype,
                    device=device,
                ).contiguous()
            else:
                radial_defocus_t = torch.as_tensor(
                    np.ascontiguousarray(radial_defocus.astype(np_dtype, copy=False)),
                    dtype=complex_dtype,
                    device=device,
                ).contiguous()
                radial_defocus_conj_t = radial_defocus_t.conj().contiguous()
        mask_indices = torch.as_tensor(
            np.ascontiguousarray(np.nonzero(mask)[0].astype(np.int64)),
            dtype=torch.long,
            device=device,
        )
        h_indices = torch.as_tensor(
            np.ascontiguousarray(np.mod(h, psi_axis.size).astype(np.int64)),
            dtype=torch.long,
            device=device,
        )
        return cls(
            h=h,
            mask=mask,
            mask_indices=mask_indices,
            h_indices=h_indices,
            nphi=int(nphi),
            nrho=int(rho_axis.size),
            npsi=int(psi_axis.size),
            nz=int(z_axis.size),
            radial=radial_t.contiguous(),
            angular=angular_t.contiguous(),
            defocus=defocus_t.contiguous(),
            radial_defocus=radial_defocus_t,
            radial_defocus_conj=radial_defocus_conj_t,
            radial_defocus_forward_bmm=radial_defocus_forward_bmm_t,
            radial_defocus_adjoint_bmm=radial_defocus_adjoint_bmm_t,
            basis_mode=basis_mode,
            contract_mode=contract_mode,
            geometric_h_cutoff=geometric_h_cutoff,
            required_h_abs=required_h_abs_array,
            device=device,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            torch=torch,
        )

    @property
    def used_modes(self) -> int:
        return int(self.h.size)

    @property
    def basis_mib(self) -> float:
        bytes_total = 0
        for tensor in (
            self.radial,
            self.angular,
            self.defocus,
            self.radial_defocus,
            self.radial_defocus_conj,
            self.radial_defocus_forward_bmm,
            self.radial_defocus_adjoint_bmm,
        ):
            if tensor is not None:
                bytes_total += int(tensor.nelement() * tensor.element_size())
        return float(bytes_total / (1024.0 * 1024.0))

    def as_tensor(self, value: np.ndarray | Any) -> Any:
        if self.torch.is_tensor(value):
            if value.device == self.device and value.dtype == self.complex_dtype:
                return value
            return value.to(device=self.device, dtype=self.complex_dtype)
        return self.torch.as_tensor(value, dtype=self.complex_dtype, device=self.device)

    def zeroed_adjoint_coeff_buffer(self, batch_size: int) -> Any:
        shape = (int(batch_size), int(self.radial.shape[0]), int(self.nphi))
        if self._adjoint_coeff_buffer is None or tuple(self._adjoint_coeff_buffer.shape) != shape:
            self._adjoint_coeff_buffer = self.torch.empty(
                shape,
                dtype=self.complex_dtype,
                device=self.device,
            )
        return self._adjoint_coeff_buffer.zero_()

    def evaluate_component_batch(self, pupils: np.ndarray | Any) -> Any:
        pupils_t = self.as_tensor(pupils)
        if pupils_t.ndim == 2:
            pupils_t = pupils_t.unsqueeze(0)
        if pupils_t.ndim != 3:
            raise ValueError("pupils must have shape (batch, ntheta, nphi)")
        if pupils_t.shape[2] != self.nphi:
            raise ValueError("pupil nphi does not match torch plan")
        coeff = self.torch.fft.fft(pupils_t, dim=2).index_select(2, self.mask_indices)
        coeff = coeff / float(self.nphi)
        if self.basis_mode == "fused":
            if self.contract_mode == "matmul":
                if self.radial_defocus_forward_bmm is None:
                    raise RuntimeError("fused matmul basis was not initialized")
                batch_size = int(coeff.shape[0])
                coeff_by_h = coeff.permute(2, 0, 1).contiguous()
                rho_z_by_h = self.torch.bmm(
                    coeff_by_h,
                    self.radial_defocus_forward_bmm,
                )
                rho_z_modes = rho_z_by_h.reshape(
                    self.used_modes,
                    batch_size,
                    self.nrho,
                    self.nz,
                ).permute(1, 2, 3, 0)
                out = self.torch.matmul(rho_z_modes, self.angular).permute(0, 1, 3, 2)
            else:
                if self.radial_defocus is None:
                    raise RuntimeError("fused basis was not initialized")
                rho_z_modes = self.torch.einsum(
                    "bth,thrz->brhz",
                    coeff,
                    self.radial_defocus,
                )
                out = self.torch.einsum("brhz,hp->brpz", rho_z_modes, self.angular)
        else:
            radial_sum = self.torch.einsum("bth,thr->brth", coeff, self.radial)
            angular_sum = self.torch.einsum("brth,hp->brtp", radial_sum, self.angular)
            out = self.torch.einsum("brtp,tz->brpz", angular_sum, self.defocus)
        return out.reshape(pupils_t.shape[0], -1)

    def adjoint_component_batch(self, residuals: np.ndarray | Any) -> Any:
        residuals_t = self.as_tensor(residuals)
        if residuals_t.ndim == 1:
            residuals_t = residuals_t.unsqueeze(0)
        if residuals_t.ndim == 2:
            if residuals_t.shape[1] != self.nrho * self.npsi * self.nz:
                raise ValueError("flat residual size does not match torch plan")
            residuals_t = residuals_t.reshape(
                residuals_t.shape[0],
                self.nrho,
                self.npsi,
                self.nz,
            )
        if residuals_t.ndim != 4:
            raise ValueError("residuals must have shape (batch, targets) or (batch, nrho, npsi, nz)")
        if residuals_t.shape[1:] != (self.nrho, self.npsi, self.nz):
            raise ValueError("residual grid shape does not match torch plan")

        psi_contracted = self.torch.fft.fft(residuals_t, dim=2).index_select(
            2,
            self.h_indices,
        )
        if self.basis_mode == "fused":
            if self.contract_mode == "matmul":
                if self.radial_defocus_adjoint_bmm is None:
                    raise RuntimeError("fused matmul basis was not initialized")
                batch_size = int(residuals_t.shape[0])
                psi_by_h = psi_contracted.permute(2, 0, 1, 3).reshape(
                    self.used_modes,
                    batch_size,
                    self.nrho * self.nz,
                )
                coeff_by_h = self.torch.bmm(
                    psi_by_h,
                    self.radial_defocus_adjoint_bmm,
                )
                coeff_adjoint = coeff_by_h.permute(1, 2, 0).contiguous()
            else:
                if self.radial_defocus_conj is None:
                    raise RuntimeError("fused basis was not initialized")
                coeff_adjoint = self.torch.einsum(
                    "brhz,thrz->bth",
                    psi_contracted,
                    self.radial_defocus_conj,
                )
        else:
            coeff_adjoint = self.torch.einsum(
                "brhz,thr,tz->bth",
                psi_contracted,
                self.radial.conj(),
                self.defocus.conj(),
            )
        full_coeff_adjoint = self.zeroed_adjoint_coeff_buffer(int(residuals_t.shape[0]))
        full_coeff_adjoint.index_copy_(2, self.mask_indices, coeff_adjoint)
        return self.torch.fft.ifft(full_coeff_adjoint, dim=2)

    def evaluate_vectorial_batch(
        self,
        pupil_jones_batch: np.ndarray | Any,
        mixing: np.ndarray | Any,
    ) -> Any:
        pupils_t = self.as_tensor(pupil_jones_batch)
        if pupils_t.ndim == 3:
            pupils_t = pupils_t.unsqueeze(0)
        if pupils_t.ndim != 4 or pupils_t.shape[1] != 2:
            raise ValueError("pupil_jones_batch must have shape (batch, 2, ntheta, nphi)")
        mixing_t = self.as_tensor(mixing)
        effective = self.torch.einsum("cjtp,bjtp->bctp", mixing_t, pupils_t)
        batch_size = int(effective.shape[0])
        component_fields = self.evaluate_component_batch(
            effective.reshape(batch_size * 3, effective.shape[-2], effective.shape[-1])
        )
        return component_fields.reshape(batch_size, 3, -1)

    def adjoint_vectorial_batch(
        self,
        residual_batch: np.ndarray | Any,
        mixing: np.ndarray | Any,
    ) -> Any:
        residuals_t = self.as_tensor(residual_batch)
        if residuals_t.ndim == 2:
            residuals_t = residuals_t.unsqueeze(0)
        if residuals_t.ndim != 3 or residuals_t.shape[1] != 3:
            raise ValueError("residual_batch must have shape (batch, 3, targets)")
        batch_size = int(residuals_t.shape[0])
        effective_gradient = self.adjoint_component_batch(
            residuals_t.reshape(batch_size * 3, residuals_t.shape[-1])
        ).reshape(batch_size, 3, self.radial.shape[0], self.nphi)
        mixing_t = self.as_tensor(mixing)
        return self.torch.einsum(
            "cjtp,bctp->bjtp",
            mixing_t.conj(),
            effective_gradient,
        )


def make_pupil_batch(
    base_pupil: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    theta_max: float,
    batch_size: int,
    phase_strength: float,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    radial = np.sin(theta)[:, None] / max(np.sin(theta_max), np.finfo(float).eps)
    phi_row = phi[None, :]
    out = np.empty((batch_size, *base_pupil.shape), dtype=np.complex128)
    for index in range(batch_size):
        m1 = 1 + (index % 5)
        m2 = 2 + ((3 * index) % 7)
        phase = phase_strength * (
            radial**2 * np.sin(m1 * phi_row + 0.37 * index)
            + 0.25 * radial * np.cos(m2 * phi_row - 0.19 * index)
        )
        amplitude = 1.0 + 0.04 * radial * np.cos((m1 + 1) * phi_row + 0.11 * index)
        out[index] = base_pupil * amplitude[None, :, :] * np.exp(1j * phase)[None, :, :]
    return out


def add_sparse_pupil_harmonic(
    pupil_batch: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    theta_max: float,
    harmonic: int,
    strength: float,
) -> np.ndarray:
    if harmonic == 0 or strength == 0.0:
        return pupil_batch
    radial = np.sin(theta)[:, None] / max(np.sin(theta_max), np.finfo(float).eps)
    envelope = radial**2
    carrier = np.exp(1j * int(harmonic) * phi[None, :])
    batch_phase = np.exp(0.173j * np.arange(pupil_batch.shape[0]))[:, None, None]
    out = np.array(pupil_batch, copy=True)
    out[:, 0] += float(strength) * batch_phase * envelope[None, :, :] * carrier[None, :, :]
    return out


def effective_pupil_stack(pupil_batch: np.ndarray, mixing: np.ndarray) -> np.ndarray:
    return np.einsum("cjtp,bjtp->bctp", mixing, pupil_batch, optimize=True)


def spectrum_source_stack(
    pupil_batch: np.ndarray,
    mixing: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    if source == "raw-jones":
        return pupil_batch
    if source == "effective":
        return effective_pupil_stack(pupil_batch, mixing)
    raise ValueError("pupil_spectrum_source must be 'raw-jones' or 'effective'")


def cpu_vectorial_batch(
    plan: PreparedSeparableHarmonicDebyeWolfPlan,
    pupil_batch: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            separable_vectorial_evaluate(plan, pupil_batch[index], mixing)
            for index in range(pupil_batch.shape[0])
        ],
        axis=0,
    )


def make_residual_batch(
    *,
    case: str,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    batch_size: int,
    seed: int,
    order: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            vectorial_residual(
                case=case,
                rho_axis=rho_axis,
                psi_axis=psi_axis,
                z_axis=z_axis,
                rng=rng,
                order=order + (index % 3),
            )
            for index in range(batch_size)
        ],
        axis=0,
    )


def cpu_vectorial_adjoint_batch(
    adjoint_plan: PreparedSeparableAdjointPlan,
    residual_batch: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            separable_vectorial_adjoint(adjoint_plan, residual_batch[index], mixing)
            for index in range(residual_batch.shape[0])
        ],
        axis=0,
    )


def to_numpy(torch: Any, device: Any, value: Any) -> np.ndarray:
    synchronize(torch, device)
    return value.detach().cpu().numpy()


def device_name(torch: Any, device: Any) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU"


def run_skipped(args: argparse.Namespace, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "status": "skipped",
            "skip_reason": reason,
            "workload_set": args.workload_set,
            "workload_index": args.workload_index,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "basis_mode": getattr(args, "basis_mode", "separate"),
            "contract_mode": getattr(args, "contract_mode", "einsum"),
            "torch_version": package_version("torch"),
        }
    ]


def run_gpu_workload(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch = import_torch()
    if torch is None:
        return run_skipped(args, "torch is not installed")

    try:
        device = resolve_device(torch, args.device)
    except RuntimeError as exc:
        return run_skipped(args, str(exc))

    workload = workloads(args.workload_set)[args.workload_index]
    theta, theta_weights = gauss_theta_grid(workload.ntheta, workload.theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, workload.nphi, endpoint=False, dtype=float)
    rho_axis, psi_axis, z_axis, _, _, _ = focal_grid(
        nrho=workload.nrho,
        npsi=workload.npsi,
        nz=workload.nz,
        rho_max=workload.rho_max,
        z_max=workload.z_max,
    )
    mixing = richards_wolf_jones_matrix(theta, phi, apodization=args.apodization)
    base_pupil = vectorial_pupil_jones(
        workload.pupil_case,
        theta,
        phi,
        theta_max=workload.theta_max,
        strength=workload.aberration_strength,
        vortex_charge=workload.vortex_charge,
    )
    pupil_batch = make_pupil_batch(
        base_pupil,
        theta,
        phi,
        theta_max=workload.theta_max,
        batch_size=args.batch_size,
        phase_strength=args.batch_phase_strength,
    )
    pupil_batch = add_sparse_pupil_harmonic(
        pupil_batch,
        theta,
        phi,
        theta_max=workload.theta_max,
        harmonic=args.extra_pupil_h,
        strength=args.extra_pupil_strength,
    )
    residual_batch = make_residual_batch(
        case=args.residual_case,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        batch_size=args.batch_size,
        seed=args.seed,
        order=args.residual_order,
    )
    h_cutoff = vectorial_h_cutoff_for_workload(workload, args.h_margin)
    spectrum_stack = spectrum_source_stack(
        pupil_batch,
        mixing,
        source=args.pupil_spectrum_source,
    )
    spectrum_h_abs = significant_h_abs_from_stack(
        spectrum_stack,
        relative_threshold=args.pupil_spectrum_relative_threshold,
        absolute_threshold=args.pupil_spectrum_absolute_threshold,
    )
    required_h_abs = resolve_required_h_abs(
        pupil_spectrum=args.pupil_spectrum,
        spectrum_h_abs=spectrum_h_abs,
        h_cutoff=h_cutoff,
        nphi=workload.nphi,
    )
    spectrum_h_max = int(np.max(spectrum_h_abs)) if spectrum_h_abs.size else 0
    extra_h_abs = np.ascontiguousarray(spectrum_h_abs[spectrum_h_abs > h_cutoff])
    reference_h_cutoff = max(h_cutoff, spectrum_h_max)

    cpu_plan = PreparedSeparableHarmonicDebyeWolfPlan.build(
        workload.nphi,
        theta,
        theta_weights,
        rho_axis,
        psi_axis,
        z_axis,
        k=workload.k,
        h_cutoff=reference_h_cutoff,
        backend=args.reference_backend,
    )
    cpu_adjoint_plan = PreparedSeparableAdjointPlan.build(cpu_plan)
    reference_count = min(args.reference_count, args.batch_size)
    cpu_forward_reference = cpu_vectorial_batch(
        cpu_plan,
        pupil_batch[:reference_count],
        mixing,
    )
    cpu_adjoint_reference = cpu_vectorial_adjoint_batch(
        cpu_adjoint_plan,
        residual_batch[:reference_count],
        mixing,
    )
    cpu_forward_batch_value = None
    cpu_forward_batch_s = None
    cpu_adjoint_batch_value = None
    cpu_adjoint_batch_s = None
    if not args.skip_cpu_batch_timing:
        cpu_forward_batch_value, cpu_forward_batch_s, _ = timed_numpy(
            lambda: cpu_vectorial_batch(cpu_plan, pupil_batch, mixing),
            args.repeats,
        )
        cpu_adjoint_batch_value, cpu_adjoint_batch_s, _ = timed_numpy(
            lambda: cpu_vectorial_adjoint_batch(
                cpu_adjoint_plan,
                residual_batch,
                mixing,
            ),
            args.repeats,
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    build_start = time.perf_counter()
    torch_plan = TorchSeparableHarmonicDebyeWolfPlan.build(
        torch=torch,
        nphi=workload.nphi,
        theta=theta,
        theta_weights=theta_weights,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        k=workload.k,
        h_cutoff=h_cutoff,
        required_h_abs=required_h_abs,
        device=device,
        dtype=args.dtype,
        basis_mode=args.basis_mode,
        contract_mode=args.contract_mode,
    )
    pupil_batch_t = torch_plan.as_tensor(pupil_batch)
    residual_batch_t = torch_plan.as_tensor(residual_batch)
    mixing_t = torch_plan.as_tensor(mixing)
    synchronize(torch, device)
    build_s = time.perf_counter() - build_start

    torch_forward_value, torch_forward_hot_s, torch_forward_times = timed_torch(
        torch,
        device,
        lambda: torch_plan.evaluate_vectorial_batch(pupil_batch_t, mixing_t),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    torch_adjoint_value, torch_adjoint_hot_s, torch_adjoint_times = timed_torch(
        torch,
        device,
        lambda: torch_plan.adjoint_vectorial_batch(residual_batch_t, mixing_t),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    torch_forward_reference = to_numpy(
        torch,
        device,
        torch_forward_value[:reference_count],
    )
    torch_adjoint_reference = to_numpy(
        torch,
        device,
        torch_adjoint_value[:reference_count],
    )
    field_l2 = relative_l2(torch_forward_reference, cpu_forward_reference)
    adjoint_l2 = relative_l2(torch_adjoint_reference, cpu_adjoint_reference)
    field_max_abs = float(
        np.max(np.abs(torch_forward_reference - cpu_forward_reference))
        / max(float(np.max(np.abs(cpu_forward_reference))), 1e-300)
    )
    adjoint_max_abs = float(
        np.max(np.abs(torch_adjoint_reference - cpu_adjoint_reference))
        / max(float(np.max(np.abs(cpu_adjoint_reference))), 1e-300)
    )
    forward_batch_l2 = None
    adjoint_batch_l2 = None
    torch_forward_numpy = to_numpy(torch, device, torch_forward_value)
    torch_adjoint_numpy = to_numpy(torch, device, torch_adjoint_value)
    if cpu_forward_batch_value is not None:
        forward_batch_l2 = relative_l2(torch_forward_numpy, cpu_forward_batch_value)
    if cpu_adjoint_batch_value is not None:
        adjoint_batch_l2 = relative_l2(torch_adjoint_numpy, cpu_adjoint_batch_value)

    dot_left = complex_dot(torch_forward_numpy, residual_batch)
    dot_right = complex_dot(pupil_batch, torch_adjoint_numpy)
    dot_abs = float(abs(dot_left - dot_right))
    dot_relative = relative_complex_error(dot_left, dot_right)

    peak_mib = None
    if device.type == "cuda":
        peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    targets = int(workload.nrho * workload.npsi * workload.nz)
    row: dict[str, Any] = {
        "status": "ok",
        "workload": workload.name,
        "workload_set": args.workload_set,
        "workload_index": args.workload_index,
        "pupil_case": workload.pupil_case,
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "ntheta": workload.ntheta,
        "nphi": workload.nphi,
        "nrho": workload.nrho,
        "npsi": workload.npsi,
        "nz": workload.nz,
        "targets_per_mask": targets,
        "vector_components": 3,
        "total_output_fields": int(args.batch_size * 3),
        "h_cutoff": h_cutoff,
        "reference_h_cutoff": int(reference_h_cutoff),
        "used_modes": int(torch_plan.used_modes),
        "geometric_h_cutoff": int(torch_plan.geometric_h_cutoff),
        "pupil_spectrum": args.pupil_spectrum,
        "pupil_spectrum_source": args.pupil_spectrum_source,
        "pupil_spectrum_relative_threshold": args.pupil_spectrum_relative_threshold,
        "pupil_spectrum_absolute_threshold": args.pupil_spectrum_absolute_threshold,
        "spectrum_h_max": int(spectrum_h_max),
        "spectrum_h_count": int(spectrum_h_abs.size),
        "extra_h_count": int(extra_h_abs.size),
        "extra_h_values": " ".join(str(int(value)) for value in extra_h_abs),
        "required_h_count": int(required_h_abs.size),
        "required_h_values": " ".join(str(int(value)) for value in required_h_abs),
        "extra_pupil_h": int(args.extra_pupil_h),
        "extra_pupil_strength": float(args.extra_pupil_strength),
        "basis_mode": torch_plan.basis_mode,
        "contract_mode": torch_plan.contract_mode,
        "basis_mib": float(torch_plan.basis_mib),
        "gpu_peak_allocated_mib": peak_mib,
        "build_s": float(build_s),
        "torch_forward_hot_s": float(torch_forward_hot_s),
        "torch_adjoint_hot_s": float(torch_adjoint_hot_s),
        "torch_forward_one_shot_s": float(build_s + torch_forward_hot_s),
        "torch_adjoint_one_shot_s": float(build_s + torch_adjoint_hot_s),
        "torch_forward_plus_adjoint_hot_s": float(
            torch_forward_hot_s + torch_adjoint_hot_s
        ),
        "cpu_forward_batch_s": None
        if cpu_forward_batch_s is None
        else float(cpu_forward_batch_s),
        "cpu_adjoint_batch_s": None
        if cpu_adjoint_batch_s is None
        else float(cpu_adjoint_batch_s),
        "cpu_forward_plus_adjoint_batch_s": None
        if cpu_forward_batch_s is None or cpu_adjoint_batch_s is None
        else float(cpu_forward_batch_s + cpu_adjoint_batch_s),
        "speedup_forward_hot_vs_cpu_batch": None
        if cpu_forward_batch_s is None
        else float(cpu_forward_batch_s / torch_forward_hot_s),
        "speedup_adjoint_hot_vs_cpu_batch": None
        if cpu_adjoint_batch_s is None
        else float(cpu_adjoint_batch_s / torch_adjoint_hot_s),
        "speedup_forward_plus_adjoint_hot_vs_cpu_batch": None
        if cpu_forward_batch_s is None or cpu_adjoint_batch_s is None
        else float(
            (cpu_forward_batch_s + cpu_adjoint_batch_s)
            / (torch_forward_hot_s + torch_adjoint_hot_s)
        ),
        "field_l2_vs_cpu_reference": float(field_l2),
        "field_max_abs_over_ref": field_max_abs,
        "adjoint_l2_vs_cpu_reference": float(adjoint_l2),
        "adjoint_max_abs_over_ref": adjoint_max_abs,
        "forward_batch_l2_vs_cpu_reference": forward_batch_l2,
        "adjoint_batch_l2_vs_cpu_reference": adjoint_batch_l2,
        "dot_abs_error": dot_abs,
        "dot_relative_error": dot_relative,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "torch_forward_times_s": " ".join(
            f"{value:.9g}" for value in torch_forward_times
        ),
        "torch_adjoint_times_s": " ".join(
            f"{value:.9g}" for value in torch_adjoint_times
        ),
        "reference_backend": args.reference_backend,
        "apodization": args.apodization,
        "batch_phase_strength": args.batch_phase_strength,
        "residual_case": args.residual_case,
        "residual_order": args.residual_order,
        "seed": args.seed,
    }
    return [row]


def write_json(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")


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


def write_summary(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# High-NA Torch GPU vectorial benchmark",
        "",
        "This benchmark adds an optional PyTorch backend for the vectorial separable "
        "Richards-Wolf forward and adjoint contractions. The Bessel/radial basis is "
        "still built on the CPU with SciPy and then transferred to the selected "
        "torch device; the hot path uses torch FFT and einsum contractions.",
        "",
        "## Config",
        "",
        f"- workload_set: `{config['workload_set']}`",
        f"- workload_index: `{config['workload_index']}`",
        f"- device request: `{config['device']}`",
        f"- dtype: `{config['dtype']}`",
        f"- batch_size: `{config['batch_size']}`",
        f"- repeats: `{config['repeats']}`",
        f"- pupil_spectrum: `{config['pupil_spectrum']}`",
        f"- pupil_spectrum_source: `{config['pupil_spectrum_source']}`",
        f"- extra_pupil_h: `{config['extra_pupil_h']}`",
        f"- extra_pupil_strength: `{config['extra_pupil_strength']}`",
        "",
        "## Results",
        "",
        "| status | workload | spectrum | h cutoff | spec h max | required h | modes | device | dtype | batch | fwd hot s | adj hot s | CPU fwd s | CPU adj s | fwd speedup | adj speedup | pair speedup | fwd L2 | adj L2 | dot rel | GPU peak MiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"| {row.get('status')} | {row.get('workload')} | {row.get('pupil_spectrum')} | "
                f"n/a | n/a | n/a | n/a | {row.get('device')} | {row.get('dtype')} | {row.get('batch_size')} | "
                "n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            lines.append("")
            lines.append(f"Skip reason: `{row.get('skip_reason')}`")
            continue
        lines.append(
            "| {status} | {workload} | {spectrum} | {h_cutoff} | {spectrum_h_max} | {required_h_count} | {used_modes} | {device_name} | {dtype} | {batch} | {fwd_hot} | {adj_hot} | {cpu_fwd} | {cpu_adj} | {fwd_speedup} | {adj_speedup} | {pair_speedup} | {fwd_l2} | {adj_l2} | {dot_rel} | {peak} |".format(
                status=row["status"],
                workload=row["workload"],
                spectrum=row["pupil_spectrum"],
                h_cutoff=row["h_cutoff"],
                spectrum_h_max=row["spectrum_h_max"],
                required_h_count=row["required_h_count"],
                used_modes=row["used_modes"],
                device_name=row["device_name"],
                dtype=row["dtype"],
                batch=row["batch_size"],
                fwd_hot=fmt(row["torch_forward_hot_s"]),
                adj_hot=fmt(row["torch_adjoint_hot_s"]),
                cpu_fwd=fmt(row["cpu_forward_batch_s"]),
                cpu_adj=fmt(row["cpu_adjoint_batch_s"]),
                fwd_speedup=fmt(row["speedup_forward_hot_vs_cpu_batch"]),
                adj_speedup=fmt(row["speedup_adjoint_hot_vs_cpu_batch"]),
                pair_speedup=fmt(row["speedup_forward_plus_adjoint_hot_vs_cpu_batch"]),
                fwd_l2=fmt(row["field_l2_vs_cpu_reference"]),
                adj_l2=fmt(row["adjoint_l2_vs_cpu_reference"]),
                dot_rel=fmt(row["dot_relative_error"]),
                peak=fmt(row["gpu_peak_allocated_mib"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Use CUDA rows for GPU performance claims. CPU torch rows are backend correctness checks.",
            "- `pupil_spectrum=adaptive` adds only significant out-of-band pupil harmonics; `dense-prefix` opens every harmonic up to the detected maximum.",
            "- The default spectrum source is the effective vectorial pupil after Richards-Wolf mixing, because the mixing matrix itself shifts azimuthal support.",
            "- The adjoint/backpropagation path uses the same tensor plan with conjugate contractions and the same FFT convention as the CPU reference.",
            "- Compare hot-loop and one-shot timings separately because plan build and device transfer are amortized in inverse-design loops.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an optional Torch/CUDA backend for vectorial High-NA separable propagation."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_torch_gpu_vectorial")
    parser.add_argument("--workload-set", choices=["quick", "representative", "large"], default="quick")
    parser.add_argument("--workload-index", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-phase-strength", type=float, default=0.25)
    parser.add_argument("--residual-case", choices=["random", "low_order", "annular_roi"], default="low_order")
    parser.add_argument("--residual-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-count", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--h-margin", type=int, default=6)
    parser.add_argument("--pupil-spectrum", choices=["off", "adaptive", "dense-prefix"], default="off")
    parser.add_argument("--pupil-spectrum-source", choices=["effective", "raw-jones"], default="effective")
    parser.add_argument("--pupil-spectrum-relative-threshold", type=float, default=1e-6)
    parser.add_argument("--pupil-spectrum-absolute-threshold", type=float, default=0.0)
    parser.add_argument("--extra-pupil-h", type=int, default=0)
    parser.add_argument("--extra-pupil-strength", type=float, default=0.0)
    parser.add_argument("--basis-mode", choices=["separate", "fused"], default="separate")
    parser.add_argument("--contract-mode", choices=["einsum", "matmul"], default="einsum")
    parser.add_argument("--reference-backend", choices=["auto", "numpy", "cpp"], default="auto")
    parser.add_argument("--apodization", choices=["sqrt-cos", "none"], default="sqrt-cos")
    parser.add_argument("--skip-cpu-batch-timing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = vars(args).copy()
    rows = run_gpu_workload(args)
    output_prefix = ROOT / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_json(output_prefix.with_suffix(".json"), config, rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), config, rows)
    print(json.dumps({"config": config, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
