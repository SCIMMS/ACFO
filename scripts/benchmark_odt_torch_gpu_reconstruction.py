from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    package_version,
    resolve_device,
    synchronize,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    CompositeContext,
    build_composite_context,
    composite_forward,
)


def torch_dtypes(torch: Any, dtype: str) -> tuple[Any, Any, Any, Any]:
    if dtype == "complex64":
        return torch.complex64, torch.float32, np.complex64, np.float32
    if dtype == "complex128":
        return torch.complex128, torch.float64, np.complex128, np.float64
    raise ValueError("dtype must be complex64 or complex128")


def to_numpy(torch: Any, device: Any, value: Any) -> np.ndarray:
    synchronize(torch, device)
    return value.detach().cpu().numpy()


def rel_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(np.ravel(reference)))
    if denom == 0.0:
        return float(np.linalg.norm(np.ravel(candidate)))
    return float(np.linalg.norm(np.ravel(candidate - reference)) / denom)


def timed_cuda(torch: Any, device: Any, func, *, repeats: int, warmups: int) -> tuple[Any, float, list[float]]:
    value = None
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        for _ in range(max(0, warmups)):
            value = func()
            synchronize(torch, device)
        times: list[float] = []
        for _ in range(max(1, repeats)):
            synchronize(torch, device)
            start = time.perf_counter()
            value = func()
            synchronize(torch, device)
            times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed function did not run")
    return value, float(median(times)), times


@dataclass
class TorchConeAxisOdtPlan:
    torch: Any
    device: Any
    complex_dtype: Any
    real_dtype: Any
    radial: Any
    axial: Any
    mode_phase: Any
    mode_phase_conj: Any
    slots: Any
    transverse: Any
    transverse_r_l: Any
    transverse_conj_r_l: Any
    psi_phase: Any
    psi_phase_t: Any
    psi_phase_conj: Any
    axial_phase: Any
    axial_phase_conj: Any
    axial_conj: Any
    axis_adjoint_kernel_h_u_rz: Any
    source_slots_flat: Any
    source_scatter_index: Any
    slots_unique: bool
    n_r: int
    n_z: int
    n_beta: int
    n_h: int
    n_l: int
    n_illum: int
    cap_radial: int
    cap_phi: int

    @classmethod
    def from_context(
        cls,
        ctx: Any,
        *,
        torch: Any,
        device: Any,
        dtype: str,
    ) -> "TorchConeAxisOdtPlan":
        complex_dtype, real_dtype, np_complex, np_real = torch_dtypes(torch, dtype)
        decomp = ctx.decomp
        cap_phi = int(decomp.factorization.cap_phi)
        radial = decomp.factorization.kernel.radial[:, ::cap_phi, :]
        axial = decomp.factorization.kernel.axial[::cap_phi, :]
        mode_phase = decomp.factorization.kernel.angular[:, 0]
        slots = np.mod(decomp.plan.h_values, cap_phi).astype(np.int64)
        slots_unique = bool(np.unique(slots).size == slots.size)
        n_r = int(decomp.plan.r_axis.size)
        n_z = int(decomp.plan.z_axis.size)
        n_beta = int(decomp.plan.n_beta)
        n_h = int(decomp.source_slots.shape[0])
        n_l = int(decomp.source_slots.shape[1])
        n_illum = int(decomp.illumination_phi.size)
        radial_t = torch.as_tensor(
            np.ascontiguousarray(radial.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        axial_t = torch.as_tensor(
            np.ascontiguousarray(axial.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        mode_phase_t = torch.as_tensor(
            np.ascontiguousarray(mode_phase.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        transverse_t = torch.as_tensor(
            np.ascontiguousarray(decomp.transverse_coeff.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        psi_phase_t = torch.as_tensor(
            np.ascontiguousarray(decomp.psi_phase.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        axial_phase_t = torch.as_tensor(
            np.ascontiguousarray(decomp.axial_phase.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        source_slots_flat_t = torch.as_tensor(
            np.ascontiguousarray(decomp.source_slots.reshape(-1).astype(np.int64)),
            dtype=torch.long,
            device=device,
        )
        axial_conj_t = axial_t.conj().contiguous()
        axis_adjoint_kernel_h_u_rz = (
            radial_t[:, :, :, None] * axial_conj_t[None, :, None, :]
        ).reshape(n_h, int(decomp.factorization.cap_radial), n_r * n_z).contiguous()
        return cls(
            torch=torch,
            device=device,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            radial=radial_t,
            axial=axial_t,
            mode_phase=mode_phase_t,
            mode_phase_conj=mode_phase_t.conj().contiguous(),
            slots=torch.as_tensor(np.ascontiguousarray(slots), dtype=torch.long, device=device),
            transverse=transverse_t,
            transverse_r_l=transverse_t.transpose(0, 1).contiguous(),
            transverse_conj_r_l=transverse_t.conj().transpose(0, 1).contiguous(),
            psi_phase=psi_phase_t,
            psi_phase_t=psi_phase_t.transpose(0, 1).contiguous(),
            psi_phase_conj=psi_phase_t.conj().contiguous(),
            axial_phase=axial_phase_t,
            axial_phase_conj=axial_phase_t.conj().contiguous(),
            axial_conj=axial_conj_t,
            axis_adjoint_kernel_h_u_rz=axis_adjoint_kernel_h_u_rz,
            source_slots_flat=source_slots_flat_t,
            source_scatter_index=source_slots_flat_t.reshape(1, 1, -1).expand(
                n_r,
                n_z,
                n_h * n_l,
            ),
            slots_unique=slots_unique,
            n_r=n_r,
            n_z=n_z,
            n_beta=n_beta,
            n_h=n_h,
            n_l=n_l,
            n_illum=n_illum,
            cap_radial=int(decomp.factorization.cap_radial),
            cap_phi=cap_phi,
        )

    @property
    def basis_mib(self) -> float:
        tensors = (
            self.radial,
            self.axial,
            self.mode_phase,
            self.mode_phase_conj,
            self.transverse,
            self.transverse_r_l,
            self.transverse_conj_r_l,
            self.psi_phase,
            self.psi_phase_t,
            self.psi_phase_conj,
            self.axial_phase,
            self.axial_phase_conj,
            self.axial_conj,
            self.axis_adjoint_kernel_h_u_rz,
            self.source_slots_flat,
            self.slots,
        )
        return float(
            sum(int(t.nelement() * t.element_size()) for t in tensors) / (1024.0 * 1024.0)
        )

    @property
    def q_count(self) -> int:
        return int(self.n_illum * self.cap_radial * self.cap_phi)

    def as_coeff(self, coeff: Any) -> Any:
        if self.torch.is_tensor(coeff):
            return coeff.to(device=self.device, dtype=self.complex_dtype)
        return self.torch.as_tensor(coeff, dtype=self.complex_dtype, device=self.device)

    def _decompose_forward(self, coeff_h_full: Any) -> Any:
        return self._decompose_forward_einsum(coeff_h_full)

    def _decompose_forward_matmul(self, coeff_h_full: Any) -> Any:
        coeff_sources = coeff_h_full.index_select(2, self.source_slots_flat)
        coeff_sources = coeff_sources.reshape(self.n_r, self.n_z, self.n_h, self.n_l)
        weighted = coeff_sources * self.transverse_r_l.reshape(self.n_r, 1, 1, self.n_l)
        weighted = weighted * self.axial_phase.reshape(1, self.n_z, 1, 1)
        mixed = self.torch.matmul(
            weighted.reshape(self.n_r * self.n_z * self.n_h, self.n_l),
            self.psi_phase_t,
        )
        return mixed.reshape(self.n_r, self.n_z, self.n_h, self.n_illum).permute(3, 0, 1, 2).contiguous()

    def _decompose_forward_einsum(self, coeff_h_full: Any) -> Any:
        coeff_sources = coeff_h_full.index_select(2, self.source_slots_flat)
        coeff_sources = coeff_sources.reshape(self.n_r, self.n_z, self.n_h, self.n_l)
        return self.torch.einsum(
            "rzhl,lr,il,z->irzh",
            coeff_sources,
            self.transverse,
            self.psi_phase,
            self.axial_phase,
        )

    def _axis_forward(self, coeff_h_all: Any) -> Any:
        inner = self.torch.einsum("hur,uz,irzh->iuh", self.radial, self.axial, coeff_h_all)
        folded = self.torch.zeros(
            (self.n_illum, self.cap_radial, self.cap_phi),
            dtype=self.complex_dtype,
            device=self.device,
        )
        src = inner * self.mode_phase.reshape(1, 1, self.n_h)
        if self.slots_unique:
            folded.index_copy_(2, self.slots, src)
        else:
            index = self.slots.reshape(1, 1, self.n_h).expand(
                self.n_illum,
                self.cap_radial,
                self.n_h,
            )
            folded.scatter_add_(2, index, src)
        return self.torch.fft.fft(folded, dim=2).reshape(-1)

    def forward(self, coeff: Any) -> Any:
        coeff_t = self.as_coeff(coeff)
        if tuple(coeff_t.shape) != (self.n_r, self.n_z, self.n_beta):
            raise ValueError("coefficient shape does not match torch ODT plan")
        coeff_h_full = self.torch.fft.ifft(coeff_t, dim=2) * float(self.n_beta)
        coeff_h_all = self._decompose_forward(coeff_h_full)
        return self._axis_forward(coeff_h_all)

    def _axis_adjoint_compact(self, residual: Any) -> Any:
        residual_t = self.as_coeff(residual)
        if residual_t.shape != (self.q_count,):
            raise ValueError("residual size does not match torch ODT plan")
        residual_grid = residual_t.reshape(self.n_illum, self.cap_radial, self.cap_phi)
        residual_modes = self.torch.fft.ifft(residual_grid, dim=2) * float(self.cap_phi)
        selected = residual_modes.index_select(2, self.slots)
        phi_sum = selected * self.mode_phase_conj.reshape(1, 1, self.n_h)
        compact = self.torch.bmm(
            phi_sum.permute(2, 0, 1).contiguous(),
            self.axis_adjoint_kernel_h_u_rz,
        )
        return compact.reshape(
            self.n_h,
            self.n_illum,
            self.n_r,
            self.n_z,
        ).permute(1, 2, 3, 0).contiguous()

    def _axis_adjoint_compact_einsum(self, residual: Any) -> Any:
        residual_t = self.as_coeff(residual)
        if residual_t.shape != (self.q_count,):
            raise ValueError("residual size does not match torch ODT plan")
        residual_grid = residual_t.reshape(self.n_illum, self.cap_radial, self.cap_phi)
        residual_modes = self.torch.fft.ifft(residual_grid, dim=2) * float(self.cap_phi)
        selected = residual_modes.index_select(2, self.slots)
        phi_sum = selected * self.mode_phase_conj.reshape(1, 1, self.n_h)
        return self.torch.einsum(
            "iuh,hur,uz->irzh",
            phi_sum,
            self.radial,
            self.axial_conj,
        )

    def _decompose_adjoint_einsum(self, compact: Any) -> Any:
        contributions = self.torch.einsum(
            "irzh,lr,il,z->rzhl",
            compact,
            self.transverse.conj(),
            self.psi_phase.conj(),
            self.axial_phase.conj(),
        )
        out_h = self.torch.zeros(
            (self.n_r, self.n_z, self.n_beta),
            dtype=self.complex_dtype,
            device=self.device,
        )
        out_h.scatter_add_(2, self.source_scatter_index, contributions.reshape(self.n_r, self.n_z, -1))
        return out_h

    def _decompose_adjoint(self, compact: Any) -> Any:
        mixed = self.torch.matmul(
            compact.permute(1, 2, 3, 0).reshape(-1, self.n_illum),
            self.psi_phase_conj,
        ).reshape(
            self.n_r,
            self.n_z,
            self.n_h,
            self.n_l,
        )
        contributions = mixed * self.transverse_conj_r_l.reshape(self.n_r, 1, 1, self.n_l)
        contributions = contributions * self.axial_phase_conj.reshape(1, self.n_z, 1, 1)
        out_h = self.torch.zeros(
            (self.n_r, self.n_z, self.n_beta),
            dtype=self.complex_dtype,
            device=self.device,
        )
        out_h.scatter_add_(2, self.source_scatter_index, contributions.reshape(self.n_r, self.n_z, -1))
        return out_h

    def adjoint(self, residual: Any) -> Any:
        compact = self._axis_adjoint_compact(residual)
        out_h = self._decompose_adjoint(compact)
        return self.torch.fft.fft(out_h, dim=2)


@dataclass
class TorchCompositeOdtPlan:
    torch: Any
    device: Any
    complex_dtype: Any
    real_dtype: Any
    ring: TorchConeAxisOdtPlan
    axis: TorchConeAxisOdtPlan | None

    @classmethod
    def from_context(
        cls,
        ctx: CompositeContext,
        *,
        torch: Any,
        device: Any,
        dtype: str,
    ) -> "TorchCompositeOdtPlan":
        complex_dtype, real_dtype, _, _ = torch_dtypes(torch, dtype)
        return cls(
            torch=torch,
            device=device,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            ring=TorchConeAxisOdtPlan.from_context(ctx.ring, torch=torch, device=device, dtype=dtype),
            axis=None
            if ctx.axis is None
            else TorchConeAxisOdtPlan.from_context(ctx.axis, torch=torch, device=device, dtype=dtype),
        )

    @property
    def q_count(self) -> int:
        return self.ring.q_count + (0 if self.axis is None else self.axis.q_count)

    @property
    def basis_mib(self) -> float:
        return self.ring.basis_mib + (0.0 if self.axis is None else self.axis.basis_mib)

    def forward(self, coeff: Any) -> Any:
        parts = [self.ring.forward(coeff)]
        if self.axis is not None:
            parts.append(self.axis.forward(coeff))
        return self.torch.cat(parts, dim=0)

    def forward_into(self, coeff: Any, out: Any) -> Any:
        if out.shape != (self.q_count,):
            raise ValueError("output buffer size does not match composite q-count")
        ring_count = self.ring.q_count
        out[:ring_count].copy_(self.ring.forward(coeff))
        if self.axis is not None:
            out[ring_count:].copy_(self.axis.forward(coeff))
        return out

    def split_residual(self, residual: Any) -> tuple[Any, Any | None]:
        ring_count = self.ring.q_count
        ring_residual = residual[:ring_count]
        axis_residual = None if self.axis is None else residual[ring_count:]
        return ring_residual, axis_residual

    def adjoint(self, residual: Any) -> Any:
        ring_residual, axis_residual = self.split_residual(residual)
        grad = self.ring.adjoint(ring_residual)
        if self.axis is not None and axis_residual is not None:
            grad = grad + self.axis.adjoint(axis_residual)
        return grad


def coeff_norm2(torch: Any, value: Any) -> Any:
    return torch.sum(torch.real(torch.conj(value) * value))


def run_gpu_iterations(
    plan: TorchCompositeOdtPlan,
    true_coeff: Any,
    data: Any,
    *,
    iterations: int,
) -> tuple[list[dict[str, Any]], Any]:
    torch = plan.torch
    device = plan.device
    x = torch.zeros_like(true_coeff)
    pred = torch.zeros_like(data)
    a_grad_buffer = torch.empty_like(data)
    data_norm = torch.clamp(torch.linalg.vector_norm(data), min=1e-30)
    true_norm = torch.clamp(torch.linalg.vector_norm(true_coeff), min=1e-30)
    rows: list[dict[str, Any]] = [
        {
            "method": "torch_gpu",
            "iteration": 0,
            "loss_rel": 1.0,
            "object_rel_l2": 1.0,
            "alpha": 0.0,
            "iter_s": 0.0,
            "adjoint_s": 0.0,
            "line_forward_s": 0.0,
            "cumulative_iter_s": 0.0,
        }
    ]
    cumulative = 0.0
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        for iteration in range(1, int(iterations) + 1):
            residual = pred - data
            synchronize(torch, device)
            iter_start = time.perf_counter()
            adj_start = time.perf_counter()
            grad = plan.adjoint(residual)
            synchronize(torch, device)
            adjoint_s = time.perf_counter() - adj_start
            fw_start = time.perf_counter()
            a_grad = plan.forward_into(grad, a_grad_buffer)
            synchronize(torch, device)
            forward_s = time.perf_counter() - fw_start
            alpha = coeff_norm2(torch, grad) / torch.clamp(coeff_norm2(torch, a_grad), min=1e-30)
            x = x - alpha * grad
            pred = pred - alpha * a_grad
            loss = torch.linalg.vector_norm(pred - data) / data_norm
            obj_err = torch.linalg.vector_norm(x - true_coeff) / true_norm
            synchronize(torch, device)
            elapsed = time.perf_counter() - iter_start
            cumulative += elapsed
            rows.append(
                {
                    "method": "torch_gpu",
                    "iteration": int(iteration),
                    "loss_rel": float(loss.detach().cpu().item()),
                    "object_rel_l2": float(obj_err.detach().cpu().item()),
                    "alpha": float(alpha.detach().cpu().item()),
                    "iter_s": float(elapsed),
                    "adjoint_s": float(adjoint_s),
                    "line_forward_s": float(forward_s),
                    "cumulative_iter_s": float(cumulative),
                }
            )
    return rows, x


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_summary_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    final = rows[-1]
    lines = [
        "# ODT Torch GPU reconstruction prototype",
        "",
        "This is a tensor-resident GPU prototype for the cone-axis ODT operator. It uses the same ring-plus-axis realistic geometry as the CPU benchmark, and evaluates the harmonic/L-mode contractions with PyTorch tensors. The adjoint decompose stage is reordered into a matmul-centered contraction to avoid the slow four-operand einsum path.",
        "",
        "## Configuration",
        "",
        f"- device: `{summary['device_name']}`",
        f"- dtype: `{summary['dtype']}`",
        f"- total q samples: `{summary['total_q_samples']}`",
        f"- cap: `{summary['cap_radial']} x {summary['cap_phi']}`",
        f"- total illuminations: `{summary['total_illumination_count']}`",
        f"- object bins: `{summary['object_bins']}`",
        f"- GPU basis memory: `{summary['gpu_basis_mib']:.3f} MiB`",
        f"- GPU peak allocated: `{summary['gpu_peak_allocated_mib']}` MiB",
        "",
        "## Hot Timings",
        "",
        "| path | median s |",
        "| --- | ---: |",
        f"| forward | {summary['gpu_forward_hot_s']:.6f} |",
        f"| adjoint | {summary['gpu_adjoint_hot_s']:.6f} |",
        f"| forward after adjoint pair | {summary['gpu_forward_adjoint_pair_hot_s']:.6f} |",
        f"| one-step update without diagnostics | {summary['gpu_one_step_update_hot_s']:.6f} |",
        f"| one iteration | {summary['gpu_median_iter_s']:.6f} |",
        "",
        "## Reconstruction",
        "",
        f"- iterations: `{int(final['iteration'])}`",
        f"- final loss rel: `{float(final['loss_rel']):.6g}`",
        f"- final object rel-L2: `{float(final['object_rel_l2']):.6g}`",
        f"- CPU/GPU forward rel-L2: `{summary['cpu_gpu_forward_rel_l2']}`",
        f"- CPU/GPU adjoint rel-L2: `{summary['cpu_gpu_adjoint_rel_l2']}`",
        "",
        "## Readout",
        "",
        "- This prototype is still PyTorch-level GPU code, but the dominant adjoint decompose contraction now uses a matmul-centered layout.",
        "- The next GPU step is a fused CUDA/Triton kernel for the remaining compact adjoint and update steps, plus a pruned tensor layout where the active L support is sparse.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    ctx = build_composite_context(args)
    plan = TorchCompositeOdtPlan.from_context(ctx, torch=torch, device=device, dtype=args.dtype)

    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    true_coeff_np = np.ascontiguousarray(ctx.ring.obj.coeff.astype(np_complex, copy=False))
    true_coeff_t = torch.as_tensor(true_coeff_np, dtype=plan.complex_dtype, device=device)

    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        data_t, gpu_forward_s, gpu_forward_times = timed_cuda(
            torch,
            device,
            lambda: plan.forward(true_coeff_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        residual_t = data_t * (0.1 + 0.2j)
        grad_t, gpu_adjoint_s, gpu_adjoint_times = timed_cuda(
            torch,
            device,
            lambda: plan.adjoint(residual_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        _, gpu_forward_adjoint_pair_s, gpu_forward_adjoint_pair_times = timed_cuda(
            torch,
            device,
            lambda: plan.forward(plan.adjoint(residual_t)),
            repeats=args.repeats,
            warmups=args.warmups,
        )

        x0 = torch.zeros_like(true_coeff_t)
        pred0 = torch.zeros_like(data_t)
        a_grad0 = torch.empty_like(data_t)

        def one_step_update() -> Any:
            residual = pred0 - data_t
            grad = plan.adjoint(residual)
            a_grad = plan.forward_into(grad, a_grad0)
            alpha = coeff_norm2(torch, grad) / torch.clamp(coeff_norm2(torch, a_grad), min=1e-30)
            return x0 - alpha * grad, pred0 - alpha * a_grad

        _, gpu_one_step_update_s, gpu_one_step_update_times = timed_cuda(
            torch,
            device,
            one_step_update,
            repeats=args.repeats,
            warmups=args.warmups,
        )

    cpu_gpu_forward_rel_l2 = None
    cpu_gpu_adjoint_rel_l2 = None
    if args.include_cpu_reference:
        cpu_data = composite_forward(ctx, true_coeff_np.astype(np.complex128), args, use_finufft=False)
        gpu_data_np = to_numpy(torch, device, data_t).astype(np.complex128, copy=False)
        cpu_gpu_forward_rel_l2 = rel_l2(gpu_data_np, cpu_data)
        cpu_residual = cpu_data * (0.1 + 0.2j)
        from benchmark_odt_realistic_geometry_reconstruction import composite_adjoint

        cpu_grad = composite_adjoint(ctx, cpu_residual, args, use_finufft=False)
        gpu_grad_np = to_numpy(torch, device, grad_t).astype(np.complex128, copy=False)
        cpu_gpu_adjoint_rel_l2 = rel_l2(gpu_grad_np, cpu_grad)

    rows, _ = run_gpu_iterations(plan, true_coeff_t, data_t, iterations=args.iterations)
    iter_times = [float(row["iter_s"]) for row in rows if int(row["iteration"]) > 0]
    adjoint_times = [float(row["adjoint_s"]) for row in rows if int(row["iteration"]) > 0]
    forward_times = [float(row["line_forward_s"]) for row in rows if int(row["iteration"]) > 0]
    peak_mib = None
    if device.type == "cuda":
        peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    ring_na = math.sin(math.radians(float(args.illumination_angle_deg)))
    summary = {
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": args.dtype,
        "ring_illum": int(args.ring_illum),
        "axis_illumination_included": not bool(args.skip_axis_illumination),
        "total_illumination_count": int(args.ring_illum)
        + (0 if args.skip_axis_illumination else 1),
        "illumination_angle_deg": float(args.illumination_angle_deg),
        "illumination_ring_na": float(ring_na),
        "detector_na": float(args.detector_na),
        "cap_radial": int(args.cap_radial),
        "cap_phi": int(args.cap_phi),
        "total_q_samples": int(plan.q_count),
        "object_bins": int(true_coeff_t.numel()),
        "gpu_basis_mib": plan.basis_mib,
        "gpu_peak_allocated_mib": peak_mib,
        "gpu_forward_hot_s": float(gpu_forward_s),
        "gpu_adjoint_hot_s": float(gpu_adjoint_s),
        "gpu_forward_adjoint_pair_hot_s": float(gpu_forward_adjoint_pair_s),
        "gpu_one_step_update_hot_s": float(gpu_one_step_update_s),
        "gpu_forward_times_s": gpu_forward_times,
        "gpu_adjoint_times_s": gpu_adjoint_times,
        "gpu_forward_adjoint_pair_times_s": gpu_forward_adjoint_pair_times,
        "gpu_one_step_update_times_s": gpu_one_step_update_times,
        "gpu_median_iter_s": float(median(iter_times)),
        "gpu_median_adjoint_s": float(median(adjoint_times)),
        "gpu_median_forward_s": float(median(forward_times)),
        "gpu_final_loss_rel": float(rows[-1]["loss_rel"]),
        "gpu_final_object_rel_l2": float(rows[-1]["object_rel_l2"]),
        "cpu_gpu_forward_rel_l2": cpu_gpu_forward_rel_l2,
        "cpu_gpu_adjoint_rel_l2": cpu_gpu_adjoint_rel_l2,
        "history_csv": str(args.csv),
    }

    write_csv(args.csv, rows)
    write_json(args.out, {"config": vars(args), "summary": summary, "history": rows})
    if args.summary_md:
        write_summary_markdown(args.summary_md, summary, rows)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Torch GPU prototype benchmark for realistic cone-axis ODT reconstruction."
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--n-beta", type=int, default=384)
    p.add_argument("--n-r", type=int, default=16)
    p.add_argument("--n-z", type=int, default=15)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=0.8)
    p.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="random_beads")
    p.add_argument("--object-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--k", type=float, default=17.307319527958313)
    p.add_argument("--detector-na", type=float, default=0.9240924092409241)
    p.add_argument("--illumination-angle-deg", type=float, default=49.0)
    p.add_argument("--ring-illum", type=int, default=100)
    p.add_argument("--skip-axis-illumination", action="store_true")
    p.add_argument("--cap-radial", type=int, default=128)
    p.add_argument("--cap-phi", type=int, default=512)
    p.add_argument("--h-cutoff", type=int, default=None)
    p.add_argument("--h-margin", type=int, default=20)
    p.add_argument("--l-margin", type=int, default=18)
    p.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    p.add_argument("--cpp-threads", type=int, default=16)
    p.add_argument(
        "--forward-execute-mode",
        choices=["prepared", "wrapper"],
        default="prepared",
    )
    p.add_argument(
        "--forward-kernel-mode",
        choices=["compact", "partitioned"],
        default="partitioned",
    )
    p.add_argument(
        "--native-prepared-plan-mode",
        choices=["auto", "direct", "gathered", "gathered-zmajor"],
        default="auto",
    )
    p.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--include-cpu-reference", action="store_true")
    p.add_argument("--finufft-eps", type=float, default=1e-12)
    p.add_argument("--finufft-q-batch-size", type=int, default=1_048_576)
    p.add_argument("--noise-rel", type=float, default=0.0)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_torch_gpu_reconstruction.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_torch_gpu_reconstruction_history.csv",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_torch_gpu_reconstruction_summary.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
