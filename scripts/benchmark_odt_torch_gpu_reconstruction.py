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
    axial_z_u: Any
    adjoint_axial_conj: Any
    axial_lowrank_left_z_rank: Any | None
    axial_lowrank_right_rank_u: Any | None
    axial_lowrank_right_h_u_rank: Any | None
    axial_lowrank_left_h_rank_z: Any | None
    axial_lowrank_rank: int
    axial_lowrank_relative_frobenius_tail: float
    adaptive_l_packed_threshold: float
    adaptive_l_active_fraction: float
    adaptive_l_offsets: tuple[int, ...] | None
    adaptive_l_r_indices: Any | None
    adaptive_l_indices: Any | None
    adaptive_l_transverse: Any | None
    adaptive_l_source_slots: Any | None
    axis_adjoint_kernel_h_u_rz: Any | None
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
    radial_block_size: int = 0
    illumination_block_size: int = 0
    forward_mode: str = "legacy"
    adjoint_mode: str = "legacy"

    @classmethod
    def from_context(
        cls,
        ctx: Any,
        *,
        torch: Any,
        device: Any,
        dtype: str,
        low_memory_adjoint: bool = False,
        radial_block_size: int = 0,
        illumination_block_size: int = 0,
        forward_mode: str = "legacy",
        adjoint_mode: str = "legacy",
        prune_exact_l0: bool = False,
        axial_lowrank_rank: int = 0,
        adaptive_l_packed_threshold: float = 0.0,
    ) -> "TorchConeAxisOdtPlan":
        if forward_mode not in {"legacy", "auto", "illumination-reduced"}:
            raise ValueError("forward_mode must be legacy, auto, or illumination-reduced")
        if adjoint_mode not in {"legacy", "auto", "illumination-reduced"}:
            raise ValueError("adjoint_mode must be legacy, auto, or illumination-reduced")
        if axial_lowrank_rank < 0:
            raise ValueError("axial_lowrank_rank must be non-negative")
        if adaptive_l_packed_threshold < 0.0:
            raise ValueError("adaptive_l_packed_threshold must be non-negative")
        complex_dtype, real_dtype, np_complex, np_real = torch_dtypes(torch, dtype)
        decomp = ctx.decomp
        cap_phi = int(decomp.factorization.cap_phi)
        kernel = decomp.factorization.kernel
        if kernel.radial.shape[1] == int(decomp.factorization.cap_radial):
            radial = kernel.radial
            axial = kernel.axial
        else:
            radial = kernel.radial[:, ::cap_phi, :]
            axial = kernel.axial[::cap_phi, :]
        mode_phase = kernel.angular[:, 0]
        slots = np.mod(decomp.plan.h_values, cap_phi).astype(np.int64)
        slots_unique = bool(np.unique(slots).size == slots.size)
        n_r = int(decomp.plan.r_axis.size)
        n_z = int(decomp.plan.z_axis.size)
        n_beta = int(decomp.plan.n_beta)
        transverse = decomp.transverse_coeff
        psi_phase = decomp.psi_phase
        source_slots = decomp.source_slots
        if prune_exact_l0:
            zero_l = np.flatnonzero(np.asarray(decomp.l_values) == 0)
            if zero_l.size != 1:
                raise ValueError("exact L=0 pruning requires exactly one L=0 mode")
            discarded = np.asarray(decomp.transverse_coeff).copy()
            discarded[zero_l] = 0
            if np.count_nonzero(discarded) != 0:
                raise ValueError(
                    "exact L=0 pruning is only valid when every discarded transverse coefficient is zero"
                )
            transverse = np.ascontiguousarray(decomp.transverse_coeff[zero_l])
            psi_phase = np.ascontiguousarray(decomp.psi_phase[:, zero_l])
            source_slots = np.ascontiguousarray(decomp.source_slots[:, zero_l])
        n_h = int(source_slots.shape[0])
        n_l = int(source_slots.shape[1])
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
            np.ascontiguousarray(transverse.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        psi_phase_t = torch.as_tensor(
            np.ascontiguousarray(psi_phase.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        axial_phase_t = torch.as_tensor(
            np.ascontiguousarray(decomp.axial_phase.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        source_slots_flat_t = torch.as_tensor(
            np.ascontiguousarray(source_slots.reshape(-1).astype(np.int64)),
            dtype=torch.long,
            device=device,
        )
        axial_conj_t = axial_t.conj().contiguous()
        axial_phase_conj_t = axial_phase_t.conj().contiguous()
        adjoint_axial_conj_t = (
            axial_conj_t * axial_phase_conj_t.reshape(1, n_z)
        ).contiguous()
        axial_lowrank_left_t = None
        axial_lowrank_right_t = None
        axial_lowrank_right_h_t = None
        axial_lowrank_left_h_t = None
        axial_lowrank_actual_rank = 0
        axial_lowrank_tail = 0.0
        if axial_lowrank_rank > 0:
            effective_axial = np.ascontiguousarray(
                decomp.axial_phase[:, None].astype(np.complex128, copy=False)
                * axial.T.astype(np.complex128, copy=False)
            )
            u_svd, singular_values, vh_svd = np.linalg.svd(
                effective_axial, full_matrices=False
            )
            axial_lowrank_actual_rank = min(
                int(axial_lowrank_rank), int(singular_values.size)
            )
            left = (
                u_svd[:, :axial_lowrank_actual_rank]
                * singular_values[None, :axial_lowrank_actual_rank]
            )
            right = vh_svd[:axial_lowrank_actual_rank]
            total = float(np.sum(singular_values**2))
            tail = float(np.sum(singular_values[axial_lowrank_actual_rank:] ** 2))
            axial_lowrank_tail = math.sqrt(
                tail / max(total, np.finfo(float).tiny)
            )
            axial_lowrank_left_t = torch.as_tensor(
                np.ascontiguousarray(left.astype(np_complex, copy=False)),
                dtype=complex_dtype,
                device=device,
            )
            axial_lowrank_right_t = torch.as_tensor(
                np.ascontiguousarray(right.astype(np_complex, copy=False)),
                dtype=complex_dtype,
                device=device,
            )
            axial_lowrank_right_h_t = (
                axial_lowrank_right_t.conj().transpose(0, 1).contiguous()
            )
            axial_lowrank_left_h_t = (
                axial_lowrank_left_t.conj().transpose(0, 1).contiguous()
            )
        adaptive_l_offsets = None
        adaptive_l_r_indices_t = None
        adaptive_l_indices_t = None
        adaptive_l_transverse_t = None
        adaptive_l_source_slots_t = None
        adaptive_l_active_fraction = 1.0
        if adaptive_l_packed_threshold > 0.0:
            transverse_abs = np.abs(transverse)
            offsets = np.empty(n_r + 1, dtype=np.int64)
            r_indices: list[np.ndarray] = []
            l_indices: list[np.ndarray] = []
            offsets[0] = 0
            for r_index in range(n_r):
                active = np.flatnonzero(
                    transverse_abs[:, r_index] > adaptive_l_packed_threshold
                ).astype(np.int64)
                if active.size == 0:
                    active = np.array(
                        [int(np.argmax(transverse_abs[:, r_index]))], dtype=np.int64
                    )
                r_indices.append(np.full(active.size, r_index, dtype=np.int64))
                l_indices.append(active)
                offsets[r_index + 1] = offsets[r_index] + active.size
            packed_r = np.ascontiguousarray(np.concatenate(r_indices))
            packed_l = np.ascontiguousarray(np.concatenate(l_indices))
            packed_transverse = np.ascontiguousarray(transverse[packed_l, packed_r])
            packed_source_slots = np.ascontiguousarray(
                source_slots[:, packed_l].T.astype(np.int64, copy=False)
            )
            adaptive_l_offsets = tuple(int(value) for value in offsets)
            adaptive_l_active_fraction = float(
                packed_l.size / max(n_r * n_l, 1)
            )
            adaptive_l_r_indices_t = torch.as_tensor(
                packed_r, dtype=torch.long, device=device
            )
            adaptive_l_indices_t = torch.as_tensor(
                packed_l, dtype=torch.long, device=device
            )
            adaptive_l_transverse_t = torch.as_tensor(
                packed_transverse.astype(np_complex, copy=False),
                dtype=complex_dtype,
                device=device,
            )
            adaptive_l_source_slots_t = torch.as_tensor(
                packed_source_slots, dtype=torch.long, device=device
            )
        axis_adjoint_kernel_h_u_rz = None
        if not low_memory_adjoint and axial_lowrank_actual_rank == 0:
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
            axial_phase_conj=axial_phase_conj_t,
            axial_conj=axial_conj_t,
            axial_z_u=axial_t.transpose(0, 1).contiguous(),
            adjoint_axial_conj=adjoint_axial_conj_t,
            axial_lowrank_left_z_rank=axial_lowrank_left_t,
            axial_lowrank_right_rank_u=axial_lowrank_right_t,
            axial_lowrank_right_h_u_rank=axial_lowrank_right_h_t,
            axial_lowrank_left_h_rank_z=axial_lowrank_left_h_t,
            axial_lowrank_rank=axial_lowrank_actual_rank,
            axial_lowrank_relative_frobenius_tail=axial_lowrank_tail,
            adaptive_l_packed_threshold=float(adaptive_l_packed_threshold),
            adaptive_l_active_fraction=adaptive_l_active_fraction,
            adaptive_l_offsets=adaptive_l_offsets,
            adaptive_l_r_indices=adaptive_l_r_indices_t,
            adaptive_l_indices=adaptive_l_indices_t,
            adaptive_l_transverse=adaptive_l_transverse_t,
            adaptive_l_source_slots=adaptive_l_source_slots_t,
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
            radial_block_size=int(radial_block_size),
            illumination_block_size=int(illumination_block_size),
            forward_mode=str(forward_mode),
            adjoint_mode=str(adjoint_mode),
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
            self.axial_z_u,
            self.adjoint_axial_conj,
            self.axial_lowrank_left_z_rank,
            self.axial_lowrank_right_rank_u,
            self.axial_lowrank_right_h_u_rank,
            self.axial_lowrank_left_h_rank_z,
            self.adaptive_l_r_indices,
            self.adaptive_l_indices,
            self.adaptive_l_transverse,
            self.adaptive_l_source_slots,
            self.axis_adjoint_kernel_h_u_rz,
            self.source_slots_flat,
            self.slots,
        )
        return float(
            sum(int(t.nelement() * t.element_size()) for t in tensors if t is not None)
            / (1024.0 * 1024.0)
        )

    @property
    def q_count(self) -> int:
        return int(self.n_illum * self.cap_radial * self.cap_phi)

    @property
    def mode_q_count(self) -> int:
        """Size of the orthonormal active detector-harmonic representation."""
        return int(self.n_illum * self.cap_radial * self.n_h)

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
        if self.resolved_forward_mode == "illumination-reduced":
            return self._forward_illumination_reduced_from_h(coeff_h_full)
        if self.radial_block_size > 0 or self.illumination_block_size > 0:
            return self._forward_streamed_from_h(coeff_h_full)
        coeff_h_all = self._decompose_forward(coeff_h_full)
        return self._axis_forward(coeff_h_all)

    def _selected_z_index(self, z_indices: Any) -> Any:
        if self.torch.is_tensor(z_indices):
            index = z_indices.to(device=self.device, dtype=self.torch.long)
        else:
            index_array = np.asarray(z_indices, dtype=np.int64)
            if index_array.ndim != 1 or index_array.size == 0:
                raise ValueError("z_indices must be a non-empty one-dimensional index vector")
            if np.any(index_array < 0) or np.any(index_array >= self.n_z):
                raise ValueError("z_indices contains an out-of-range index")
            if np.unique(index_array).size != index_array.size:
                raise ValueError("z_indices must not contain duplicates")
            index = self.torch.as_tensor(
                np.ascontiguousarray(index_array),
                dtype=self.torch.long,
                device=self.device,
            )
        if index.ndim != 1 or index.numel() == 0:
            raise ValueError("z_indices must be a non-empty one-dimensional index vector")
        return index

    @property
    def adaptive_l_packed_enabled(self) -> bool:
        return self.adaptive_l_offsets is not None

    def _forward_packed_l_reduction(
        self,
        coeff_h_full: Any,
        *,
        axial_phase: Any | None,
        axial_z_u: Any | None,
        axial_lowrank_left: Any | None,
    ) -> Any:
        n_axial = int(coeff_h_full.shape[1])
        if axial_lowrank_left is None:
            effective_axial = axial_phase.reshape(n_axial, 1) * axial_z_u
        else:
            effective_axial = None
        r_block = (
            min(self.n_r, 16)
            if self.radial_block_size <= 0
            else self.radial_block_size
        )
        reduced_h_l_u = self.torch.zeros(
            (self.n_h, self.n_l, self.cap_radial),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for r_start in range(0, self.n_r, r_block):
            r_stop = min(r_start + r_block, self.n_r)
            packed_start = self.adaptive_l_offsets[r_start]
            packed_stop = self.adaptive_l_offsets[r_stop]
            packed_r = self.adaptive_l_r_indices[packed_start:packed_stop]
            packed_l = self.adaptive_l_indices[packed_start:packed_stop]
            packed_transverse = self.adaptive_l_transverse[packed_start:packed_stop]
            packed_source_slots = self.adaptive_l_source_slots[
                packed_start:packed_stop
            ]
            pair_count = int(packed_stop - packed_start)
            local_r = int(r_stop - r_start)
            coeff_rows = coeff_h_full[r_start:r_stop].permute(0, 2, 1).reshape(
                local_r * self.n_beta, n_axial
            )
            source_rows = (
                (packed_r - int(r_start)).reshape(pair_count, 1) * self.n_beta
                + packed_source_slots
            )
            source_matrix = coeff_rows.index_select(
                0, source_rows.reshape(-1)
            )
            if axial_lowrank_left is not None:
                projected = self.torch.matmul(
                    self.torch.matmul(source_matrix, axial_lowrank_left),
                    self.axial_lowrank_right_rank_u,
                )
            else:
                projected = self.torch.matmul(source_matrix, effective_axial)
            projected = projected.reshape(pair_count, self.n_h, self.cap_radial)
            radial = self.radial.index_select(2, packed_r).permute(2, 0, 1)
            contributions = (
                projected * radial * packed_transverse.reshape(pair_count, 1, 1)
            )
            reduced_h_l_u.index_add_(
                1, packed_l, contributions.permute(1, 0, 2)
            )
        return reduced_h_l_u

    def _source_from_reduced_h_l_u(self, reduced_h_l_u: Any) -> Any:
        inner = self.torch.matmul(
            reduced_h_l_u.permute(2, 0, 1).reshape(
                self.cap_radial * self.n_h, self.n_l
            ),
            self.psi_phase_t,
        ).reshape(self.cap_radial, self.n_h, self.n_illum).permute(2, 0, 1)
        return inner * self.mode_phase.reshape(1, 1, self.n_h)

    def _forward_selected_z_source(self, coeff: Any, index: Any) -> Any:
        n_selected = int(index.numel())
        coeff_t = self.as_coeff(coeff)
        if tuple(coeff_t.shape) != (self.n_r, n_selected, self.n_beta):
            raise ValueError("selected coefficient shape does not match torch ODT plan")
        coeff_h_full = self.torch.fft.ifft(coeff_t, dim=2) * float(self.n_beta)
        if self.axial_lowrank_rank > 0:
            axial_lowrank_left = self.axial_lowrank_left_z_rank.index_select(
                0, index
            )
            axial_phase = None
            axial_z_u = None
        else:
            axial_lowrank_left = None
            axial_phase = self.axial_phase.index_select(0, index)
            axial_z_u = self.axial_z_u.index_select(0, index)

        if self.adaptive_l_packed_enabled:
            reduced_h_l_u = self._forward_packed_l_reduction(
                coeff_h_full,
                axial_phase=axial_phase,
                axial_z_u=axial_z_u,
                axial_lowrank_left=axial_lowrank_left,
            )
            return self._source_from_reduced_h_l_u(reduced_h_l_u)

        r_block = min(self.n_r, 16) if self.radial_block_size <= 0 else self.radial_block_size
        reduced_h_l_u = self.torch.zeros(
            (self.n_h, self.n_l, self.cap_radial),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for r_start in range(0, self.n_r, r_block):
            r_stop = min(r_start + r_block, self.n_r)
            local_r = r_stop - r_start
            coeff_sources = coeff_h_full[r_start:r_stop].index_select(
                2, self.source_slots_flat
            ).reshape(local_r, n_selected, self.n_h, self.n_l)
            source_matrix = coeff_sources.permute(0, 2, 3, 1).reshape(
                local_r * self.n_h * self.n_l, n_selected
            )
            if self.axial_lowrank_rank > 0:
                projected_r_h_l_u = self.torch.matmul(
                    self.torch.matmul(source_matrix, axial_lowrank_left),
                    self.axial_lowrank_right_rank_u,
                ).reshape(local_r, self.n_h, self.n_l, self.cap_radial)
            else:
                source_matrix = source_matrix * axial_phase.reshape(1, n_selected)
                projected_r_h_l_u = self.torch.matmul(
                    source_matrix,
                    axial_z_u,
                ).reshape(local_r, self.n_h, self.n_l, self.cap_radial)
            projected_r_h_l_u = projected_r_h_l_u * self.transverse_r_l[
                r_start:r_stop
            ].reshape(local_r, 1, self.n_l, 1)
            projected_r_h_l_u = projected_r_h_l_u * self.radial[
                :, :, r_start:r_stop
            ].permute(2, 0, 1).reshape(local_r, self.n_h, 1, self.cap_radial)
            reduced_h_l_u.add_(projected_r_h_l_u.sum(dim=0))

        return self._source_from_reduced_h_l_u(reduced_h_l_u)

    def forward_selected_z(self, coeff: Any, z_indices: Any) -> Any:
        """Forward from coefficients supported only on selected full-grid z planes."""
        index = self._selected_z_index(z_indices)
        source = self._forward_selected_z_source(coeff, index)
        folded = self.torch.zeros(
            (self.n_illum, self.cap_radial, self.cap_phi),
            dtype=self.complex_dtype,
            device=self.device,
        )
        if self.slots_unique:
            folded.index_copy_(2, self.slots, source)
        else:
            folded.index_add_(2, self.slots, source)
        return self.torch.fft.fft(folded, dim=2).reshape(-1)

    def forward_selected_z_modes(self, coeff: Any, z_indices: Any) -> Any:
        """Return active detector Fourier modes with orthonormal FFT scaling."""
        if not self.slots_unique:
            raise RuntimeError("active detector modes alias for this cap_phi")
        index = self._selected_z_index(z_indices)
        source = self._forward_selected_z_source(coeff, index)
        return (source * math.sqrt(float(self.cap_phi))).reshape(-1)

    def _forward_streamed_from_h(self, coeff_h_full: Any) -> Any:
        r_block = self.n_r if self.radial_block_size <= 0 else self.radial_block_size
        i_block = self.n_illum if self.illumination_block_size <= 0 else self.illumination_block_size
        folded = self.torch.zeros(
            (self.n_illum, self.cap_radial, self.cap_phi),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for r_start in range(0, self.n_r, r_block):
            r_stop = min(r_start + r_block, self.n_r)
            local_r = r_stop - r_start
            coeff_sources = coeff_h_full[r_start:r_stop].index_select(
                2, self.source_slots_flat
            ).reshape(local_r, self.n_z, self.n_h, self.n_l)
            for i_start in range(0, self.n_illum, i_block):
                i_stop = min(i_start + i_block, self.n_illum)
                compact = self.torch.einsum(
                    "rzhl,lr,il,z->irzh",
                    coeff_sources,
                    self.transverse[:, r_start:r_stop],
                    self.psi_phase[i_start:i_stop],
                    self.axial_phase,
                )
                inner = self.torch.einsum(
                    "hur,uz,irzh->iuh",
                    self.radial[:, :, r_start:r_stop],
                    self.axial,
                    compact,
                )
                src = inner * self.mode_phase.reshape(1, 1, self.n_h)
                folded[i_start:i_stop].index_add_(2, self.slots, src)
        return self.torch.fft.fft(folded, dim=2).reshape(-1)

    @property
    def resolved_forward_mode(self) -> str:
        return self._resolved_illumination_reduction_mode(self.forward_mode)

    def _forward_illumination_reduced_from_h(self, coeff_h_full: Any) -> Any:
        """Forward with object axes contracted before illumination synthesis."""
        if self.adaptive_l_packed_enabled:
            reduced_h_l_u = self._forward_packed_l_reduction(
                coeff_h_full,
                axial_phase=(
                    None if self.axial_lowrank_rank > 0 else self.axial_phase
                ),
                axial_z_u=(
                    None if self.axial_lowrank_rank > 0 else self.axial_z_u
                ),
                axial_lowrank_left=self.axial_lowrank_left_z_rank,
            )
            source = self._source_from_reduced_h_l_u(reduced_h_l_u)
            folded = self.torch.zeros(
                (self.n_illum, self.cap_radial, self.cap_phi),
                dtype=self.complex_dtype,
                device=self.device,
            )
            if self.slots_unique:
                folded.index_copy_(2, self.slots, source)
            else:
                folded.index_add_(2, self.slots, source)
            return self.torch.fft.fft(folded, dim=2).reshape(-1)
        r_block = (
            min(self.n_r, 16)
            if self.radial_block_size <= 0
            else self.radial_block_size
        )
        reduced_h_l_u = self.torch.zeros(
            (self.n_h, self.n_l, self.cap_radial),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for r_start in range(0, self.n_r, r_block):
            r_stop = min(r_start + r_block, self.n_r)
            local_r = r_stop - r_start
            coeff_sources = coeff_h_full[r_start:r_stop].index_select(
                2, self.source_slots_flat
            ).reshape(local_r, self.n_z, self.n_h, self.n_l)
            source_matrix = coeff_sources.permute(0, 2, 3, 1).reshape(
                local_r * self.n_h * self.n_l, self.n_z
            )
            if self.axial_lowrank_rank > 0:
                projected_r_h_l_u = self.torch.matmul(
                    self.torch.matmul(
                        source_matrix, self.axial_lowrank_left_z_rank
                    ),
                    self.axial_lowrank_right_rank_u,
                ).reshape(local_r, self.n_h, self.n_l, self.cap_radial)
            else:
                source_matrix = source_matrix * self.axial_phase.reshape(
                    1, self.n_z
                )
                projected_r_h_l_u = self.torch.matmul(
                    source_matrix,
                    self.axial_z_u,
                ).reshape(local_r, self.n_h, self.n_l, self.cap_radial)
            projected_r_h_l_u = projected_r_h_l_u * self.transverse_r_l[
                r_start:r_stop
            ].reshape(local_r, 1, self.n_l, 1)
            projected_r_h_l_u = projected_r_h_l_u * self.radial[
                :, :, r_start:r_stop
            ].permute(2, 0, 1).reshape(local_r, self.n_h, 1, self.cap_radial)
            reduced_h_l_u.add_(projected_r_h_l_u.sum(dim=0))

        inner = self.torch.matmul(
            reduced_h_l_u.permute(2, 0, 1).reshape(
                self.cap_radial * self.n_h, self.n_l
            ),
            self.psi_phase_t,
        ).reshape(self.cap_radial, self.n_h, self.n_illum).permute(2, 0, 1)
        folded = self.torch.zeros(
            (self.n_illum, self.cap_radial, self.cap_phi),
            dtype=self.complex_dtype,
            device=self.device,
        )
        source = inner * self.mode_phase.reshape(1, 1, self.n_h)
        if self.slots_unique:
            folded.index_copy_(2, self.slots, source)
        else:
            folded.index_add_(2, self.slots, source)
        return self.torch.fft.fft(folded, dim=2).reshape(-1)

    def _axis_adjoint_compact(self, residual: Any) -> Any:
        if self.axis_adjoint_kernel_h_u_rz is None:
            return self._axis_adjoint_compact_einsum(residual)
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
        if self.resolved_adjoint_mode == "illumination-reduced":
            return self._adjoint_illumination_reduced(residual)
        if self.radial_block_size > 0 or self.illumination_block_size > 0:
            return self._adjoint_streamed(residual)
        compact = self._axis_adjoint_compact(residual)
        out_h = self._decompose_adjoint(compact)
        return self.torch.fft.fft(out_h, dim=2)

    def _adjoint_packed_l_from_illumination_mixed(
        self,
        illumination_mixed_h_l_u: Any,
        *,
        n_axial: int,
        adjoint_axial_conj: Any | None,
        axial_lowrank_left_h: Any | None,
    ) -> Any:
        r_block = (
            min(self.n_r, 16)
            if self.radial_block_size <= 0
            else self.radial_block_size
        )
        out_h = self.torch.zeros(
            (self.n_r, n_axial, self.n_beta),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for r_start in range(0, self.n_r, r_block):
            r_stop = min(r_start + r_block, self.n_r)
            packed_start = self.adaptive_l_offsets[r_start]
            packed_stop = self.adaptive_l_offsets[r_stop]
            packed_r = self.adaptive_l_r_indices[packed_start:packed_stop]
            packed_l = self.adaptive_l_indices[packed_start:packed_stop]
            packed_transverse = self.adaptive_l_transverse[packed_start:packed_stop]
            packed_source_slots = self.adaptive_l_source_slots[
                packed_start:packed_stop
            ]
            pair_count = int(packed_stop - packed_start)
            local_r = int(r_stop - r_start)
            radial = self.radial.index_select(2, packed_r).permute(2, 0, 1)
            illumination = illumination_mixed_h_l_u.index_select(
                1, packed_l
            ).permute(1, 0, 2)
            weighted_matrix = (radial * illumination).reshape(
                pair_count * self.n_h, self.cap_radial
            )
            if axial_lowrank_left_h is not None:
                contributions = self.torch.matmul(
                    self.torch.matmul(
                        weighted_matrix, self.axial_lowrank_right_h_u_rank
                    ),
                    axial_lowrank_left_h,
                )
            else:
                contributions = self.torch.matmul(
                    weighted_matrix, adjoint_axial_conj
                )
            contributions = contributions.reshape(
                pair_count, self.n_h, n_axial
            ) * packed_transverse.conj().reshape(pair_count, 1, 1)
            source_rows = (
                (packed_r - int(r_start)).reshape(pair_count, 1) * self.n_beta
                + packed_source_slots
            )
            block_out = self.torch.zeros(
                (local_r * self.n_beta, n_axial),
                dtype=self.complex_dtype,
                device=self.device,
            )
            block_out.index_add_(
                0, source_rows.reshape(-1), contributions.reshape(-1, n_axial)
            )
            out_h[r_start:r_stop] = block_out.reshape(
                local_r, self.n_beta, n_axial
            ).permute(0, 2, 1)
        return self.torch.fft.fft(out_h, dim=2)

    def _adjoint_selected_z_from_illumination_mixed(
        self, illumination_mixed: Any, index: Any
    ) -> Any:
        n_selected = int(index.numel())
        illumination_mixed_h_l_u = illumination_mixed.reshape(
            self.cap_radial, self.n_h, self.n_l
        ).permute(1, 2, 0).contiguous()
        if self.axial_lowrank_rank > 0:
            selected_lowrank_left_h = self.axial_lowrank_left_h_rank_z.index_select(
                1, index
            )
            adjoint_axial_conj = None
        else:
            selected_lowrank_left_h = None
            adjoint_axial_conj = self.adjoint_axial_conj.index_select(1, index)

        if self.adaptive_l_packed_enabled:
            return self._adjoint_packed_l_from_illumination_mixed(
                illumination_mixed_h_l_u,
                n_axial=n_selected,
                adjoint_axial_conj=adjoint_axial_conj,
                axial_lowrank_left_h=selected_lowrank_left_h,
            )

        r_block = min(self.n_r, 16) if self.radial_block_size <= 0 else self.radial_block_size
        out_h = self.torch.zeros(
            (self.n_r, n_selected, self.n_beta),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for r_start in range(0, self.n_r, r_block):
            r_stop = min(r_start + r_block, self.n_r)
            local_r = r_stop - r_start
            radial_h_r_u = self.radial[:, :, r_start:r_stop].permute(0, 2, 1)
            weighted_h_r_l_u = (
                radial_h_r_u.unsqueeze(2) * illumination_mixed_h_l_u.unsqueeze(1)
            )
            weighted_matrix = weighted_h_r_l_u.reshape(
                self.n_h * local_r * self.n_l, self.cap_radial
            )
            if self.axial_lowrank_rank > 0:
                contributions = self.torch.matmul(
                    self.torch.matmul(
                        weighted_matrix, self.axial_lowrank_right_h_u_rank
                    ),
                    selected_lowrank_left_h,
                )
            else:
                contributions = self.torch.matmul(
                    weighted_matrix,
                    adjoint_axial_conj,
                )
            contributions = contributions.reshape(
                self.n_h,
                local_r,
                self.n_l,
                n_selected,
            ).permute(1, 3, 0, 2)
            contributions = contributions * self.transverse_conj_r_l[
                r_start:r_stop
            ].reshape(local_r, 1, 1, self.n_l)
            out_h[r_start:r_stop].index_add_(
                2,
                self.source_slots_flat,
                contributions.reshape(local_r, n_selected, self.n_h * self.n_l),
            )
        return self.torch.fft.fft(out_h, dim=2)

    def adjoint_selected_z(self, residual: Any, z_indices: Any) -> Any:
        """Evaluate selected z planes of the full-data illumination-reduced adjoint."""
        index = self._selected_z_index(z_indices)
        residual_t = self.as_coeff(residual)
        if residual_t.shape != (self.q_count,):
            raise ValueError("residual size does not match torch ODT plan")
        residual_grid = residual_t.reshape(self.n_illum, self.cap_radial, self.cap_phi)
        i_block = self.n_illum if self.illumination_block_size <= 0 else self.illumination_block_size
        illumination_mixed = self.torch.zeros(
            (self.cap_radial * self.n_h, self.n_l),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for i_start in range(0, self.n_illum, i_block):
            i_stop = min(i_start + i_block, self.n_illum)
            residual_modes = self.torch.fft.ifft(
                residual_grid[i_start:i_stop], dim=2
            ) * float(self.cap_phi)
            selected = residual_modes.index_select(2, self.slots)
            phi_sum = selected * self.mode_phase_conj.reshape(1, 1, self.n_h)
            illumination_mixed.add_(
                self.torch.matmul(
                    phi_sum.permute(1, 2, 0).reshape(
                        self.cap_radial * self.n_h, i_stop - i_start
                    ),
                    self.psi_phase_conj[i_start:i_stop],
                )
            )
        return self._adjoint_selected_z_from_illumination_mixed(
            illumination_mixed, index
        )

    def adjoint_selected_z_modes(self, residual: Any, z_indices: Any) -> Any:
        """Adjoint of forward_selected_z_modes under the complex Euclidean dot."""
        if not self.slots_unique:
            raise RuntimeError("active detector modes alias for this cap_phi")
        index = self._selected_z_index(z_indices)
        residual_t = self.as_coeff(residual)
        if residual_t.shape != (self.mode_q_count,):
            raise ValueError("mode residual size does not match torch ODT plan")
        residual_modes = residual_t.reshape(
            self.n_illum, self.cap_radial, self.n_h
        )
        i_block = self.n_illum if self.illumination_block_size <= 0 else self.illumination_block_size
        illumination_mixed = self.torch.zeros(
            (self.cap_radial * self.n_h, self.n_l),
            dtype=self.complex_dtype,
            device=self.device,
        )
        fft_scale = math.sqrt(float(self.cap_phi))
        for i_start in range(0, self.n_illum, i_block):
            i_stop = min(i_start + i_block, self.n_illum)
            phi_sum = (
                residual_modes[i_start:i_stop]
                * fft_scale
                * self.mode_phase_conj.reshape(1, 1, self.n_h)
            )
            illumination_mixed.add_(
                self.torch.matmul(
                    phi_sum.permute(1, 2, 0).reshape(
                        self.cap_radial * self.n_h, i_stop - i_start
                    ),
                    self.psi_phase_conj[i_start:i_stop],
                )
            )
        return self._adjoint_selected_z_from_illumination_mixed(
            illumination_mixed, index
        )

    @property
    def resolved_adjoint_mode(self) -> str:
        return self._resolved_illumination_reduction_mode(self.adjoint_mode)

    def _resolved_illumination_reduction_mode(self, requested_mode: str) -> str:
        if self.axial_lowrank_rank > 0 or self.adaptive_l_packed_enabled:
            return "illumination-reduced"
        if requested_mode != "auto":
            return requested_mode
        # The unblocked prepared path already uses a fast materialized kernel
        # and wins on the prior 92k-object/6.62M-q xlarge sentinel.  Reduction
        # is intended for the low-memory nested streaming path, where it removes
        # repeated illumination/radial launches.  Keep small illumination sets
        # conservative until the crossover sweep measures their boundary.
        streaming = self.radial_block_size > 0 or self.illumination_block_size > 0
        return (
            "illumination-reduced"
            if streaming and self.n_illum >= 8
            else "legacy"
        )

    def _adjoint_illumination_reduced(self, residual: Any) -> Any:
        """Adjoint with illumination summed before the radial/axial contraction.

        This is the GPU analogue of the earlier fused-contraction optimization:
        it removes the materialized (illumination, r, z, h) compact tensor and
        avoids repeating the expensive axis contraction for every illumination
        block.  The source-slot fold remains exact and uses a one-dimensional
        index_add instead of an expanded scatter index.
        """
        residual_t = self.as_coeff(residual)
        if residual_t.shape != (self.q_count,):
            raise ValueError("residual size does not match torch ODT plan")
        residual_grid = residual_t.reshape(self.n_illum, self.cap_radial, self.cap_phi)
        i_block = self.n_illum if self.illumination_block_size <= 0 else self.illumination_block_size
        illumination_mixed = self.torch.zeros(
            (self.cap_radial * self.n_h, self.n_l),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for i_start in range(0, self.n_illum, i_block):
            i_stop = min(i_start + i_block, self.n_illum)
            residual_modes = self.torch.fft.ifft(
                residual_grid[i_start:i_stop], dim=2
            ) * float(self.cap_phi)
            selected = residual_modes.index_select(2, self.slots)
            phi_sum = selected * self.mode_phase_conj.reshape(1, 1, self.n_h)
            illumination_mixed.add_(
                self.torch.matmul(
                    phi_sum.permute(1, 2, 0).reshape(
                        self.cap_radial * self.n_h, i_stop - i_start
                    ),
                    self.psi_phase_conj[i_start:i_stop],
                )
            )
        illumination_mixed_h_l_u = illumination_mixed.reshape(
            self.cap_radial, self.n_h, self.n_l
        ).permute(1, 2, 0).contiguous()

        if self.adaptive_l_packed_enabled:
            return self._adjoint_packed_l_from_illumination_mixed(
                illumination_mixed_h_l_u,
                n_axial=self.n_z,
                adjoint_axial_conj=(
                    None
                    if self.axial_lowrank_rank > 0
                    else self.adjoint_axial_conj
                ),
                axial_lowrank_left_h=self.axial_lowrank_left_h_rank_z,
            )

        # A small default radial tile prevents accidental multi-GiB
        # intermediates when auto mode is requested without an explicit block.
        r_block = (
            min(self.n_r, 16)
            if self.radial_block_size <= 0
            else self.radial_block_size
        )
        out_h = self.torch.zeros(
            (self.n_r, self.n_z, self.n_beta),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for r_start in range(0, self.n_r, r_block):
            r_stop = min(r_start + r_block, self.n_r)
            local_r = r_stop - r_start
            radial_h_r_u = self.radial[:, :, r_start:r_stop].permute(0, 2, 1)
            weighted_h_r_l_u = (
                radial_h_r_u.unsqueeze(2) * illumination_mixed_h_l_u.unsqueeze(1)
            )
            weighted_matrix = weighted_h_r_l_u.reshape(
                self.n_h * local_r * self.n_l, self.cap_radial
            )
            if self.axial_lowrank_rank > 0:
                contributions = self.torch.matmul(
                    self.torch.matmul(
                        weighted_matrix, self.axial_lowrank_right_h_u_rank
                    ),
                    self.axial_lowrank_left_h_rank_z,
                )
            else:
                contributions = self.torch.matmul(
                    weighted_matrix,
                    self.adjoint_axial_conj,
                )
            contributions = contributions.reshape(
                self.n_h,
                local_r,
                self.n_l,
                self.n_z,
            ).permute(1, 3, 0, 2)
            contributions = contributions * self.transverse_conj_r_l[
                r_start:r_stop
            ].reshape(local_r, 1, 1, self.n_l)
            out_h[r_start:r_stop].index_add_(
                2,
                self.source_slots_flat,
                contributions.reshape(local_r, self.n_z, self.n_h * self.n_l),
            )
        return self.torch.fft.fft(out_h, dim=2)

    def _adjoint_streamed(self, residual: Any) -> Any:
        residual_t = self.as_coeff(residual)
        if residual_t.shape != (self.q_count,):
            raise ValueError("residual size does not match torch ODT plan")
        r_block = self.n_r if self.radial_block_size <= 0 else self.radial_block_size
        i_block = self.n_illum if self.illumination_block_size <= 0 else self.illumination_block_size
        residual_grid = residual_t.reshape(self.n_illum, self.cap_radial, self.cap_phi)
        out_h = self.torch.zeros(
            (self.n_r, self.n_z, self.n_beta),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for i_start in range(0, self.n_illum, i_block):
            i_stop = min(i_start + i_block, self.n_illum)
            residual_modes = self.torch.fft.ifft(
                residual_grid[i_start:i_stop], dim=2
            ) * float(self.cap_phi)
            selected = residual_modes.index_select(2, self.slots)
            phi_sum = selected * self.mode_phase_conj.reshape(1, 1, self.n_h)
            for r_start in range(0, self.n_r, r_block):
                r_stop = min(r_start + r_block, self.n_r)
                local_r = r_stop - r_start
                compact = self.torch.einsum(
                    "iuh,hur,uz->irzh",
                    phi_sum,
                    self.radial[:, :, r_start:r_stop],
                    self.axial_conj,
                )
                contributions = self.torch.einsum(
                    "irzh,rl,il,z->rzhl",
                    compact,
                    self.transverse_conj_r_l[r_start:r_stop],
                    self.psi_phase_conj[i_start:i_stop],
                    self.axial_phase_conj,
                )
                scatter_index = self.source_slots_flat.reshape(1, 1, -1).expand(
                    local_r, self.n_z, self.n_h * self.n_l
                )
                out_h[r_start:r_stop].scatter_add_(
                    2,
                    scatter_index,
                    contributions.reshape(local_r, self.n_z, -1),
                )
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
        low_memory_adjoint: bool = False,
        radial_block_size: int = 0,
        illumination_block_size: int = 0,
        forward_mode: str = "legacy",
        adjoint_mode: str = "legacy",
        prune_axis_l0: bool = False,
        axial_lowrank_rank: int = 0,
        ring_adaptive_l_packed_threshold: float = 0.0,
    ) -> "TorchCompositeOdtPlan":
        complex_dtype, real_dtype, _, _ = torch_dtypes(torch, dtype)
        return cls(
            torch=torch,
            device=device,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            ring=TorchConeAxisOdtPlan.from_context(
                ctx.ring,
                torch=torch,
                device=device,
                dtype=dtype,
                low_memory_adjoint=low_memory_adjoint,
                radial_block_size=radial_block_size,
                illumination_block_size=illumination_block_size,
                forward_mode=forward_mode,
                adjoint_mode=adjoint_mode,
                prune_exact_l0=False,
                axial_lowrank_rank=axial_lowrank_rank,
                adaptive_l_packed_threshold=ring_adaptive_l_packed_threshold,
            ),
            axis=None
            if ctx.axis is None
            else TorchConeAxisOdtPlan.from_context(
                ctx.axis,
                torch=torch,
                device=device,
                dtype=dtype,
                low_memory_adjoint=low_memory_adjoint,
                radial_block_size=radial_block_size,
                illumination_block_size=illumination_block_size,
                forward_mode=forward_mode,
                adjoint_mode=adjoint_mode,
                prune_exact_l0=prune_axis_l0,
                axial_lowrank_rank=axial_lowrank_rank,
                adaptive_l_packed_threshold=0.0,
            ),
        )

    @property
    def q_count(self) -> int:
        return self.ring.q_count + (0 if self.axis is None else self.axis.q_count)

    @property
    def mode_q_count(self) -> int:
        return self.ring.mode_q_count + (
            0 if self.axis is None else self.axis.mode_q_count
        )

    @property
    def basis_mib(self) -> float:
        return self.ring.basis_mib + (0.0 if self.axis is None else self.axis.basis_mib)

    def forward(self, coeff: Any) -> Any:
        parts = [self.ring.forward(coeff)]
        if self.axis is not None:
            parts.append(self.axis.forward(coeff))
        return self.torch.cat(parts, dim=0)

    def forward_selected_z(self, coeff: Any, z_indices: Any) -> Any:
        parts = [self.ring.forward_selected_z(coeff, z_indices)]
        if self.axis is not None:
            parts.append(self.axis.forward_selected_z(coeff, z_indices))
        return self.torch.cat(parts, dim=0)

    def forward_selected_z_modes(self, coeff: Any, z_indices: Any) -> Any:
        parts = [self.ring.forward_selected_z_modes(coeff, z_indices)]
        if self.axis is not None:
            parts.append(self.axis.forward_selected_z_modes(coeff, z_indices))
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

    def adjoint_selected_z(self, residual: Any, z_indices: Any) -> Any:
        ring_residual, axis_residual = self.split_residual(residual)
        grad = self.ring.adjoint_selected_z(ring_residual, z_indices)
        if self.axis is not None and axis_residual is not None:
            grad = grad + self.axis.adjoint_selected_z(axis_residual, z_indices)
        return grad

    def adjoint_selected_z_modes(self, residual: Any, z_indices: Any) -> Any:
        ring_count = self.ring.mode_q_count
        ring_residual = residual[:ring_count]
        axis_residual = None if self.axis is None else residual[ring_count:]
        if residual.shape != (self.mode_q_count,):
            raise ValueError("mode residual size does not match composite mode q-count")
        grad = self.ring.adjoint_selected_z_modes(ring_residual, z_indices)
        if self.axis is not None and axis_residual is not None:
            grad = grad + self.axis.adjoint_selected_z_modes(
                axis_residual, z_indices
            )
        return grad


def coeff_norm2(torch: Any, value: Any) -> Any:
    return torch.sum(torch.real(torch.conj(value) * value))


def run_gpu_iterations(
    plan: TorchCompositeOdtPlan,
    true_coeff: Any,
    data: Any,
    *,
    iterations: int,
    real_object: bool = False,
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
            if real_object:
                grad = torch.complex(torch.real(grad), torch.zeros_like(torch.real(grad)))
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
        f"- axis L=0 exact pruning: `{summary['prune_axis_l0']}`",
        f"- axial low-rank requested/realized: `{summary['axial_lowrank_rank_requested']}` / "
        f"`{summary['ring_axial_lowrank_rank']}`",
        f"- ring/axis L modes: `{summary['ring_l_modes']}` / `{summary['axis_l_modes']}`",
        f"- ring adaptive-L threshold/active fraction: "
        f"`{summary['ring_adaptive_l_packed_threshold']}` / "
        f"`{summary['ring_adaptive_l_active_fraction']:.6f}`",
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
    plan = TorchCompositeOdtPlan.from_context(
        ctx,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=args.low_memory_adjoint,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        forward_mode=args.forward_mode,
        adjoint_mode=args.adjoint_mode,
        prune_axis_l0=args.prune_axis_l0,
        axial_lowrank_rank=args.axial_lowrank_rank,
        ring_adaptive_l_packed_threshold=args.ring_adaptive_l_packed_threshold,
    )

    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    true_coeff_np = np.ascontiguousarray(ctx.ring.obj.coeff.astype(np_complex, copy=False))
    if args.real_object:
        true_coeff_np = np.ascontiguousarray(np.real(true_coeff_np).astype(np_complex))
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
        dot_lhs = torch.vdot(data_t.reshape(-1), residual_t.reshape(-1))
        dot_rhs = torch.vdot(true_coeff_t.reshape(-1), grad_t.reshape(-1))
        dot_den = torch.abs(dot_lhs) + torch.abs(dot_rhs)
        forward_adjoint_dot_error = float(
            (torch.abs(dot_lhs - dot_rhs) / torch.clamp(dot_den, min=1e-30))
            .detach()
            .cpu()
            .item()
        )
        data_dot_np = to_numpy(torch, device, data_t).astype(np.complex128, copy=False)
        residual_dot_np = to_numpy(torch, device, residual_t).astype(np.complex128, copy=False)
        coeff_dot_np = to_numpy(torch, device, true_coeff_t).astype(np.complex128, copy=False)
        grad_dot_np = to_numpy(torch, device, grad_t).astype(np.complex128, copy=False)
        dot_lhs_128 = np.vdot(data_dot_np.reshape(-1), residual_dot_np.reshape(-1))
        dot_rhs_128 = np.vdot(coeff_dot_np.reshape(-1), grad_dot_np.reshape(-1))
        forward_adjoint_dot_error_complex128_accum = float(
            abs(dot_lhs_128 - dot_rhs_128)
            / max(abs(dot_lhs_128) + abs(dot_rhs_128), np.finfo(np.float64).tiny)
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

    rows, _ = run_gpu_iterations(
        plan,
        true_coeff_t,
        data_t,
        iterations=args.iterations,
        real_object=args.real_object,
    )
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
        "low_memory_adjoint": bool(args.low_memory_adjoint),
        "radial_block_size": int(args.radial_block_size),
        "illumination_block_size": int(args.illumination_block_size),
        "forward_mode": str(args.forward_mode),
        "adjoint_mode": str(args.adjoint_mode),
        "prune_axis_l0": bool(args.prune_axis_l0),
        "axial_lowrank_rank_requested": int(args.axial_lowrank_rank),
        "ring_axial_lowrank_rank": int(plan.ring.axial_lowrank_rank),
        "axis_axial_lowrank_rank": (
            None if plan.axis is None else int(plan.axis.axial_lowrank_rank)
        ),
        "ring_axial_lowrank_relative_frobenius_tail": float(
            plan.ring.axial_lowrank_relative_frobenius_tail
        ),
        "axis_axial_lowrank_relative_frobenius_tail": (
            None
            if plan.axis is None
            else float(plan.axis.axial_lowrank_relative_frobenius_tail)
        ),
        "ring_l_modes": int(plan.ring.n_l),
        "axis_l_modes": None if plan.axis is None else int(plan.axis.n_l),
        "ring_adaptive_l_packed_threshold": float(
            args.ring_adaptive_l_packed_threshold
        ),
        "ring_adaptive_l_active_fraction": float(
            plan.ring.adaptive_l_active_fraction
        ),
        "ring_forward_mode_resolved": plan.ring.resolved_forward_mode,
        "axis_forward_mode_resolved": (
            None if plan.axis is None else plan.axis.resolved_forward_mode
        ),
        "ring_adjoint_mode_resolved": plan.ring.resolved_adjoint_mode,
        "axis_adjoint_mode_resolved": (
            None if plan.axis is None else plan.axis.resolved_adjoint_mode
        ),
        "skip_native_prepared_adjoint": bool(args.skip_native_prepared_adjoint),
        "compact_axisymmetric_kernel": bool(args.compact_axisymmetric_kernel),
        "real_object": bool(args.real_object),
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
        "forward_adjoint_dot_error": forward_adjoint_dot_error,
        "forward_adjoint_dot_error_complex128_accum": forward_adjoint_dot_error_complex128_accum,
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
    p.add_argument("--low-memory-adjoint", action="store_true")
    p.add_argument("--radial-block-size", type=int, default=0)
    p.add_argument("--illumination-block-size", type=int, default=0)
    p.add_argument(
        "--prune-axis-l0",
        action="store_true",
        help="Remove algebraically zero nonzero-L modes from the zero-NA axis illumination only.",
    )
    p.add_argument(
        "--axial-lowrank-rank",
        type=int,
        default=0,
        help="Use an SVD rank for the effective axial operator; zero keeps the exact validated path.",
    )
    p.add_argument(
        "--ring-adaptive-l-packed-threshold",
        type=float,
        default=0.0,
        help="Pack ring (r,L) pairs whose transverse magnitude exceeds this threshold; zero keeps the dense L layout.",
    )
    p.add_argument(
        "--forward-mode",
        choices=["legacy", "auto", "illumination-reduced"],
        default="legacy",
        help="Keep the verified legacy path, auto-select for multi-angle streaming, or force illumination-last synthesis.",
    )
    p.add_argument(
        "--adjoint-mode",
        choices=["legacy", "auto", "illumination-reduced"],
        default="legacy",
        help="Keep the verified legacy path, auto-select for multi-angle streaming, or force illumination-first reduction.",
    )
    p.add_argument("--skip-native-prepared-adjoint", action="store_true")
    p.add_argument("--compact-axisymmetric-kernel", action="store_true")
    p.add_argument("--real-object", action="store_true")
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
    p.add_argument(
        "--detector-radial-sampling",
        choices=["uniform_rho", "uniform_theta", "outer_power"],
        default="uniform_rho",
    )
    p.add_argument("--detector-radial-outer-power", type=float, default=2.0)
    p.add_argument("--detector-radial-min-fraction", type=float, default=0.0)
    p.add_argument("--detector-radial-max-fraction", type=float, default=1.0)
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
