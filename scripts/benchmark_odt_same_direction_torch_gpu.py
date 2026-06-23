from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    package_version,
    resolve_device,
    synchronize,
)
from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    CuFinufftComposite,
    cufinufft_adjoint_block,
    cufinufft_forward_block,
    cupy_dtypes,
    import_cufinufft_modules,
    make_block,
    synchronize_cupy,
)
from benchmark_odt_cone_axis_decomposition import default_l_cutoff, fmt  # noqa: E402
from benchmark_odt_cone_illumination import parse_int_list  # noqa: E402
from benchmark_odt_ewald_cap_operator import (  # noqa: E402
    StructuredOdtPlan,
    make_cylindrical_object,
    random_residual,
    recommended_h_cutoff,
    relative_l2,
    resolve_structured_backend,
)
from benchmark_odt_same_direction_zperp import (  # noqa: E402
    SameDirectionDecomposition,
    build_magnitude_svd,
    build_same_direction_decomposition,
    compressed_adjoint,
    compressed_forward,
    same_direction_illumination,
    same_direction_adjoint,
    same_direction_forward,
    same_direction_q_samples,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    torch_dtypes,
    to_numpy,
)


def parse_case(text: str) -> tuple[str, int, float, int, int]:
    if "=" in text:
        label, payload = text.split("=", 1)
    else:
        label, payload = "", text
    parts = payload.lower().replace(" ", "").split(":")
    if len(parts) != 3 or "x" not in parts[2]:
        raise ValueError("case must look like label=n_mag:k:cap_radialxcap_phi")
    n_mag = int(parts[0])
    k = float(parts[1])
    cap_radial_text, cap_phi_text = parts[2].split("x", 1)
    cap_radial = int(cap_radial_text)
    cap_phi = int(cap_phi_text)
    if not label:
        label = f"nmag{n_mag}_k{fmt(k, 3)}_{cap_radial}x{cap_phi}"
    return label, n_mag, k, cap_radial, cap_phi


def median_time_cuda(
    torch: Any,
    device: Any,
    func,
    *,
    repeats: int,
    warmups: int,
) -> tuple[Any, float, list[float]]:
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
        raise RuntimeError("timed CUDA function did not run")
    return value, float(median(times)), times


def median_time_cpu(func, *, repeats: int) -> tuple[Any, float, list[float]]:
    value = None
    times: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed CPU function did not run")
    return value, float(median(times)), times


def torch_to_cupy(cp: Any, value: Any) -> Any:
    return cp.from_dlpack(value.detach().contiguous())


def cupy_to_torch(torch: Any, value: Any) -> Any:
    return torch.from_dlpack(value)


@dataclass
class TorchSameDirectionSvdPlan:
    torch: Any
    device: Any
    complex_dtype: Any
    radial: Any
    axial: Any
    mode_phase: Any
    mode_phase_conj: Any
    slots: Any
    weights: Any
    weights_conj: Any
    u: Any
    u_conj_t: Any
    axis_adjoint_kernel_h_u_rz: Any
    source_slots_flat: Any
    source_scatter_index: Any
    slots_unique: bool
    n_r: int
    n_z: int
    n_beta: int
    n_h: int
    n_l: int
    rank: int
    n_mag: int
    cap_radial: int
    cap_phi: int

    @classmethod
    def from_decomposition(
        cls,
        decomp: SameDirectionDecomposition,
        compression: Any,
        *,
        rank: int,
        torch: Any,
        device: Any,
        dtype: str,
    ) -> "TorchSameDirectionSvdPlan":
        complex_dtype, _, np_complex, _ = torch_dtypes(torch, dtype)
        cap_phi = int(decomp.factorization.cap_phi)
        radial = decomp.factorization.kernel.radial[:, ::cap_phi, :]
        axial = decomp.factorization.kernel.axial[::cap_phi, :]
        mode_phase = decomp.factorization.kernel.angular[:, 0]
        slots = np.mod(decomp.plan.h_values, cap_phi).astype(np.int64)
        source_slots = decomp.source_slots.reshape(-1).astype(np.int64)
        weights = np.ascontiguousarray(compression.weights[:rank].astype(np_complex, copy=False))
        u = np.ascontiguousarray(compression.u[:, :rank].astype(np_complex, copy=False))
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
        weights_t = torch.as_tensor(weights, dtype=complex_dtype, device=device)
        u_t = torch.as_tensor(u, dtype=complex_dtype, device=device)
        source_slots_t = torch.as_tensor(
            np.ascontiguousarray(source_slots),
            dtype=torch.long,
            device=device,
        )
        n_r = int(decomp.plan.r_axis.size)
        n_z = int(decomp.plan.z_axis.size)
        n_beta = int(decomp.plan.n_beta)
        n_h = int(decomp.plan.h_values.size)
        n_l = int(decomp.l_values.size)
        axial_conj_t = axial_t.conj().contiguous()
        axis_adjoint_kernel = (
            radial_t[:, :, :, None] * axial_conj_t[None, :, None, :]
        ).reshape(n_h, int(decomp.factorization.cap_radial), n_r * n_z).contiguous()
        return cls(
            torch=torch,
            device=device,
            complex_dtype=complex_dtype,
            radial=radial_t,
            axial=axial_t,
            mode_phase=mode_phase_t,
            mode_phase_conj=mode_phase_t.conj().contiguous(),
            slots=torch.as_tensor(np.ascontiguousarray(slots), dtype=torch.long, device=device),
            weights=weights_t,
            weights_conj=weights_t.conj().contiguous(),
            u=u_t,
            u_conj_t=u_t.conj().transpose(0, 1).contiguous(),
            axis_adjoint_kernel_h_u_rz=axis_adjoint_kernel,
            source_slots_flat=source_slots_t,
            source_scatter_index=source_slots_t.reshape(1, 1, -1).expand(n_r, n_z, n_h * n_l),
            slots_unique=bool(np.unique(slots).size == slots.size),
            n_r=n_r,
            n_z=n_z,
            n_beta=n_beta,
            n_h=n_h,
            n_l=n_l,
            rank=int(rank),
            n_mag=int(decomp.magnitudes.size),
            cap_radial=int(decomp.factorization.cap_radial),
            cap_phi=cap_phi,
        )

    @property
    def q_count(self) -> int:
        return self.n_mag * self.cap_radial * self.cap_phi

    @property
    def basis_mib(self) -> float:
        tensors = (
            self.radial,
            self.axial,
            self.mode_phase,
            self.mode_phase_conj,
            self.weights,
            self.weights_conj,
            self.u,
            self.u_conj_t,
            self.axis_adjoint_kernel_h_u_rz,
            self.source_slots_flat,
            self.slots,
        )
        return float(
            sum(int(t.nelement() * t.element_size()) for t in tensors) / (1024.0 * 1024.0)
        )

    def as_coeff(self, coeff: Any) -> Any:
        if self.torch.is_tensor(coeff):
            return coeff.to(device=self.device, dtype=self.complex_dtype)
        return self.torch.as_tensor(coeff, dtype=self.complex_dtype, device=self.device)

    def _axis_forward(self, coeff_h_rank: Any) -> Any:
        inner = self.torch.einsum("hur,uz,srzh->suh", self.radial, self.axial, coeff_h_rank)
        folded = self.torch.zeros(
            (self.rank, self.cap_radial, self.cap_phi),
            dtype=self.complex_dtype,
            device=self.device,
        )
        src = inner * self.mode_phase.reshape(1, 1, self.n_h)
        if self.slots_unique:
            folded.index_copy_(2, self.slots, src)
        else:
            index = self.slots.reshape(1, 1, self.n_h).expand(
                self.rank,
                self.cap_radial,
                self.n_h,
            )
            folded.scatter_add_(2, index, src)
        return self.torch.fft.fft(folded, dim=2).reshape(self.rank, -1)

    def forward(self, coeff: Any) -> Any:
        coeff_t = self.as_coeff(coeff)
        if tuple(coeff_t.shape) != (self.n_r, self.n_z, self.n_beta):
            raise ValueError("coefficient shape does not match same-direction plan")
        coeff_h_full = self.torch.fft.ifft(coeff_t, dim=2) * float(self.n_beta)
        coeff_sources = coeff_h_full.index_select(2, self.source_slots_flat)
        coeff_sources = coeff_sources.reshape(self.n_r, self.n_z, self.n_h, self.n_l)
        coeff_h_rank = self.torch.einsum("rzhl,slrz->srzh", coeff_sources, self.weights)
        rank_fields = self._axis_forward(coeff_h_rank)
        fields = self.u @ rank_fields
        return fields.reshape(-1)

    def _axis_adjoint_compact(self, rank_residual: Any) -> Any:
        residual_grid = rank_residual.reshape(self.rank, self.cap_radial, self.cap_phi)
        residual_modes = self.torch.fft.ifft(residual_grid, dim=2) * float(self.cap_phi)
        selected = residual_modes.index_select(2, self.slots)
        phi_sum = selected * self.mode_phase_conj.reshape(1, 1, self.n_h)
        compact = self.torch.bmm(
            phi_sum.permute(2, 0, 1).contiguous(),
            self.axis_adjoint_kernel_h_u_rz,
        )
        return compact.reshape(self.n_h, self.rank, self.n_r, self.n_z).permute(
            1,
            2,
            3,
            0,
        ).contiguous()

    def adjoint(self, residual: Any) -> Any:
        residual_t = self.as_coeff(residual)
        if residual_t.shape != (self.q_count,):
            raise ValueError("residual size does not match same-direction q stack")
        rank_residual = self.u_conj_t @ residual_t.reshape(self.n_mag, -1)
        compact = self._axis_adjoint_compact(rank_residual)
        contributions = self.torch.einsum("srzh,slrz->rzhl", compact, self.weights_conj)
        out_h = self.torch.zeros(
            (self.n_r, self.n_z, self.n_beta),
            dtype=self.complex_dtype,
            device=self.device,
        )
        out_h.scatter_add_(2, self.source_scatter_index, contributions.reshape(self.n_r, self.n_z, -1))
        return self.torch.fft.fft(out_h, dim=2)


def make_cufinufft_single_op(
    ctx: Any,
    *,
    dtype: str,
    plan_mode: str,
    eps: float,
) -> CuFinufftComposite:
    cp, cufinufft = import_cufinufft_modules()
    complex_dtype, _, _, np_real = cupy_dtypes(cp, dtype)
    block = make_block(
        cp,
        cufinufft,
        ctx,
        dtype=dtype,
        complex_dtype=complex_dtype,
        np_real=np_real,
        plan_mode=plan_mode,
        eps=eps,
    )
    return CuFinufftComposite(
        cp=cp,
        cufinufft=cufinufft,
        dtype=dtype,
        plan_mode=plan_mode,
        complex_dtype=complex_dtype,
        ring=block,
        axis=None,
    )


def cufinufft_forward_torch(torch: Any, op: CuFinufftComposite, coeff: Any, *, eps: float) -> Any:
    coeff_gpu = torch_to_cupy(op.cp, coeff.reshape(-1))
    out = cufinufft_forward_block(op, op.ring, coeff_gpu, eps=eps)
    synchronize_cupy(op.cp)
    return cupy_to_torch(torch, out)


def cufinufft_adjoint_torch(
    torch: Any,
    op: CuFinufftComposite,
    residual: Any,
    *,
    eps: float,
    coeff_shape: tuple[int, ...],
) -> Any:
    residual_gpu = torch_to_cupy(op.cp, residual.reshape(-1))
    out = cufinufft_adjoint_block(op, op.ring, residual_gpu, eps=eps)
    synchronize_cupy(op.cp)
    return cupy_to_torch(torch, out).reshape(coeff_shape)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# ODT same-direction GPU SVD benchmark",
        "",
        "This benchmark moves the focused/raster same-direction SVD operator to PyTorch GPU and compares it against a cuFINUFFT type-3 GPU Plan on the same flat shifted q-list.",
        "",
        "## Results",
        "",
        "| case | q samples | rank | method | fwd ms | adj ms | cuFINUFFT/fwd | cuFINUFFT/adj | fwd err vs grouped | adj err vs grouped | dot err |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {q} | {rank} | {method} | `{fwd}` | `{adj}` | `{sf}` | `{sa}` | `{ef}` | `{ea}` | `{dot}` |".format(
                case=row["case"],
                q=row["flat_q_samples"],
                rank=row["rank"] if row["rank"] is not None else "",
                method=row["method"],
                fwd=fmt(1000.0 * row["forward_s"], 4) if row.get("forward_s") is not None else "n/a",
                adj=fmt(1000.0 * row["adjoint_s"], 4) if row.get("adjoint_s") is not None else "n/a",
                sf=fmt(row.get("cufinufft_over_method_forward_speedup"), 4),
                sa=fmt(row.get("cufinufft_over_method_adjoint_speedup"), 4),
                ef=fmt(row.get("forward_l2_vs_grouped"), 4),
                ea=fmt(row.get("adjoint_l2_vs_grouped"), 4),
                dot=fmt(row.get("adjoint_dot_error"), 4),
            )
        )
    gpu_rows = [row for row in rows if row["method"] == "torch_gpu_svd"]
    if gpu_rows:
        lines.extend(["", "## Readout", ""])
        for row in gpu_rows:
            lines.append(
                "- `{case}` rank {rank}: forward `{fwd}`x and adjoint `{adj}`x faster than cuFINUFFT; grouped-reference errors are `{ef}` forward and `{ea}` adjoint.".format(
                    case=row["case"],
                    rank=row["rank"],
                    fwd=fmt(row.get("cufinufft_over_method_forward_speedup"), 4),
                    adj=fmt(row.get("cufinufft_over_method_adjoint_speedup"), 4),
                    ef=fmt(row.get("forward_l2_vs_grouped"), 4),
                    ea=fmt(row.get("adjoint_l2_vs_grouped"), 4),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `torch_gpu_svd` is the GPU version of the rank-compressed same-direction operator.",
            "- `cpu_cpp_grouped_exact` is the exact grouped same-direction reference used for error measurement.",
            "- cuFINUFFT is a GPU Plan baseline on the full shifted q cloud, so the speedup measures whether same-direction structure is worth exposing in acquisition design.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    gpu_rows = [row for row in rows if row["method"] == "torch_gpu_svd"]
    if not gpu_rows:
        return
    labels = [row["case"] for row in gpu_rows]
    fwd = [row["cufinufft_over_method_forward_speedup"] for row in gpu_rows]
    adj = [row["cufinufft_over_method_adjoint_speedup"] for row in gpu_rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width / 2, fwd, width=width, label="forward")
    ax.bar(x + width / 2, adj, width=width, label="adjoint")
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("cuFINUFFT time / torch GPU SVD time")
    ax.set_title("Same-direction GPU SVD speedup over cuFINUFFT")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_case(args: argparse.Namespace, *, label: str, n_mag: int, k: float, cap_radial: int, cap_phi: int) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("same-direction GPU benchmark requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    magnitudes = np.linspace(args.min_illumination_na, args.max_illumination_na, n_mag)
    _, base_q = same_direction_q_samples(
        k=k,
        detector_na=args.detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
        illumination=same_direction_illumination(
            magnitudes=magnitudes,
            direction_phi=args.direction_phi,
        ),
    )
    h_cutoff = (
        args.h_cutoff
        if args.h_cutoff is not None
        else recommended_h_cutoff(base_q, args.r_max, args.n_beta, args.h_margin)
    )
    plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=h_cutoff,
    )
    l_cutoff = default_l_cutoff(
        k=k,
        illumination_na=args.max_illumination_na,
        r_max=args.r_max,
        margin=args.l_margin,
        n_beta=args.n_beta,
    )
    decomp, decomp_build_s, _ = median_time_cpu(
        lambda: build_same_direction_decomposition(
            plan,
            k=k,
            detector_na=args.detector_na,
            cap_radial=cap_radial,
            cap_phi=cap_phi,
            magnitudes=magnitudes,
            direction_phi=args.direction_phi,
            l_cutoff=l_cutoff,
        ),
        repeats=args.build_repeats,
    )
    compression, svd_build_s, _ = median_time_cpu(lambda: build_magnitude_svd(decomp), repeats=args.build_repeats)
    if args.rank <= 0 or args.rank > compression.singular_values.size:
        raise ValueError(f"rank {args.rank} is outside available rank {compression.singular_values.size}")

    backend = resolve_structured_backend(args.structured_backend)
    residual = random_residual(decomp.flat_q, seed=args.seed + 9901 + n_mag + cap_radial + cap_phi)
    grouped_forward, grouped_forward_s, _ = median_time_cpu(
        lambda: same_direction_forward(
            obj.coeff,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            grouped=True,
        ),
        repeats=args.cpu_repeats,
    )
    grouped_adjoint, grouped_adjoint_s, _ = median_time_cpu(
        lambda: same_direction_adjoint(
            residual,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            grouped=True,
        ),
        repeats=args.cpu_repeats,
    )
    cpu_svd_forward, cpu_svd_forward_s, _ = median_time_cpu(
        lambda: compressed_forward(
            obj.coeff,
            decomp,
            compression,
            rank=args.rank,
            backend=backend,
            cpp_threads=args.cpp_threads,
        ),
        repeats=args.cpu_repeats,
    )
    cpu_svd_adjoint, cpu_svd_adjoint_s, _ = median_time_cpu(
        lambda: compressed_adjoint(
            residual,
            decomp,
            compression,
            rank=args.rank,
            backend=backend,
            cpp_threads=args.cpp_threads,
        ),
        repeats=args.cpu_repeats,
    )

    start = time.perf_counter()
    gpu_plan = TorchSameDirectionSvdPlan.from_decomposition(
        decomp,
        compression,
        rank=args.rank,
        torch=torch,
        device=device,
        dtype=args.dtype,
    )
    torch_setup_s = time.perf_counter() - start
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    true_coeff_np = np.ascontiguousarray(obj.coeff.astype(np_complex, copy=False))
    residual_np = np.ascontiguousarray(residual.astype(np_complex, copy=False))
    true_coeff_t = torch.as_tensor(true_coeff_np, dtype=gpu_plan.complex_dtype, device=device)
    residual_t = torch.as_tensor(residual_np, dtype=gpu_plan.complex_dtype, device=device)

    torch_forward, torch_forward_s, torch_forward_times = median_time_cuda(
        torch,
        device,
        lambda: gpu_plan.forward(true_coeff_t),
        repeats=args.gpu_repeats,
        warmups=args.gpu_warmups,
    )
    torch_adjoint, torch_adjoint_s, torch_adjoint_times = median_time_cuda(
        torch,
        device,
        lambda: gpu_plan.adjoint(residual_t),
        repeats=args.gpu_repeats,
        warmups=args.gpu_warmups,
    )
    torch_forward_np = to_numpy(torch, device, torch_forward)
    torch_adjoint_np = to_numpy(torch, device, torch_adjoint)

    cp, _ = import_cufinufft_modules()
    start = time.perf_counter()
    cu_op = make_cufinufft_single_op(
        SimpleNamespace(obj=obj, flat_q=decomp.flat_q),
        dtype=args.dtype,
        plan_mode=args.cufinufft_plan_mode,
        eps=args.cufinufft_eps,
    )
    cufinufft_setup_s = time.perf_counter() - start
    cu_forward, cu_forward_s, cu_forward_times = median_time_cuda(
        torch,
        device,
        lambda: cufinufft_forward_torch(torch, cu_op, true_coeff_t, eps=args.cufinufft_eps),
        repeats=args.cufinufft_repeats,
        warmups=args.gpu_warmups,
    )
    cu_adjoint, cu_adjoint_s, cu_adjoint_times = median_time_cuda(
        torch,
        device,
        lambda: cufinufft_adjoint_torch(
            torch,
            cu_op,
            residual_t,
            eps=args.cufinufft_eps,
            coeff_shape=true_coeff_np.shape,
        ),
        repeats=args.cufinufft_repeats,
        warmups=args.gpu_warmups,
    )
    cu_forward_np = to_numpy(torch, device, cu_forward)
    cu_adjoint_np = to_numpy(torch, device, cu_adjoint)

    def base_row(
        method: str,
        forward_s: float,
        adjoint_s: float,
        forward_value: np.ndarray,
        adjoint_value: np.ndarray,
        *,
        setup_s: float = 0.0,
        basis_mib: float | None = None,
        forward_times: list[float] | None = None,
        adjoint_times: list[float] | None = None,
    ) -> dict[str, Any]:
        pair_s = float(forward_s + adjoint_s)
        return {
            "case": label,
            "method": method,
            "rank": int(args.rank) if "svd" in method else None,
            "n_mag": int(n_mag),
            "k": float(k),
            "cap_radial": int(cap_radial),
            "cap_phi": int(cap_phi),
            "flat_q_samples": int(decomp.flat_q.count),
            "base_q_samples": int(decomp.base_q.count),
            "object_bins": int(obj.coeff.size),
            "n_beta": int(args.n_beta),
            "n_r": int(args.n_r),
            "n_z": int(args.n_z),
            "h_cutoff": int(h_cutoff),
            "l_cutoff": int(l_cutoff),
            "l_modes": int(decomp.l_values.size),
            "setup_s": float(setup_s),
            "basis_mib": basis_mib,
            "decomp_build_s": float(decomp_build_s),
            "svd_build_s": float(svd_build_s),
            "forward_s": float(forward_s),
            "adjoint_s": float(adjoint_s),
            "pair_s": pair_s,
            "cufinufft_forward_s": float(cu_forward_s),
            "cufinufft_adjoint_s": float(cu_adjoint_s),
            "cufinufft_pair_s": float(cu_forward_s + cu_adjoint_s),
            "cufinufft_over_method_forward_speedup": None
            if forward_s <= 0.0
            else float(cu_forward_s / forward_s),
            "cufinufft_over_method_adjoint_speedup": None
            if adjoint_s <= 0.0
            else float(cu_adjoint_s / adjoint_s),
            "cufinufft_over_method_pair_speedup": None
            if pair_s <= 0.0
            else float((cu_forward_s + cu_adjoint_s) / pair_s),
            "forward_l2_vs_grouped": relative_l2(forward_value, grouped_forward),
            "adjoint_l2_vs_grouped": relative_l2(adjoint_value, grouped_adjoint),
            "forward_l2_vs_cpu_svd": relative_l2(forward_value, cpu_svd_forward),
            "adjoint_l2_vs_cpu_svd": relative_l2(adjoint_value, cpu_svd_adjoint),
            "forward_l2_vs_cufinufft": relative_l2(forward_value, cu_forward_np),
            "adjoint_l2_vs_cufinufft": relative_l2(adjoint_value, cu_adjoint_np),
            "adjoint_dot_error": float(
                abs(
                    np.vdot(forward_value.reshape(-1), residual.reshape(-1))
                    - np.vdot(obj.coeff.reshape(-1), adjoint_value.reshape(-1))
                )
                / max(abs(np.vdot(forward_value.reshape(-1), residual.reshape(-1))), 1e-300)
            ),
            "forward_times_s": "" if forward_times is None else " ".join(f"{item:.9g}" for item in forward_times),
            "adjoint_times_s": "" if adjoint_times is None else " ".join(f"{item:.9g}" for item in adjoint_times),
        }

    rows = [
        base_row(
            "torch_gpu_svd",
            torch_forward_s,
            torch_adjoint_s,
            torch_forward_np,
            torch_adjoint_np,
            setup_s=torch_setup_s,
            basis_mib=gpu_plan.basis_mib,
            forward_times=torch_forward_times,
            adjoint_times=torch_adjoint_times,
        ),
        base_row(
            "cpu_cpp_svd",
            cpu_svd_forward_s,
            cpu_svd_adjoint_s,
            cpu_svd_forward,
            cpu_svd_adjoint,
        ),
        base_row(
            "cpu_cpp_grouped_exact",
            grouped_forward_s,
            grouped_adjoint_s,
            grouped_forward,
            grouped_adjoint,
        ),
        base_row(
            "cufinufft_gpu_plan",
            cu_forward_s,
            cu_adjoint_s,
            cu_forward_np,
            cu_adjoint_np,
            setup_s=cufinufft_setup_s,
            basis_mib=float(cp.get_default_memory_pool().total_bytes() / (1024.0 * 1024.0)),
            forward_times=cu_forward_times,
            adjoint_times=cu_adjoint_times,
        ),
    ]
    return {
        "case": {
            "label": label,
            "n_mag": n_mag,
            "k": k,
            "cap_radial": cap_radial,
            "cap_phi": cap_phi,
        },
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cupy_version": getattr(cp, "__version__", None),
        "cufinufft_version": getattr(cu_op.cufinufft, "__version__", None),
        "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark same-direction rank-SVD ODT operator on PyTorch GPU against cuFINUFFT GPU."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["nmag64=64:32:32x128", "qradial48=32:48:48x192"],
        help="Cases as label=n_mag:k:cap_radialxcap_phi.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--min-illumination-na", type=float, default=0.02)
    parser.add_argument("--max-illumination-na", type=float, default=0.2)
    parser.add_argument("--direction-phi", type=float, default=0.0)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--n-beta", type=int, default=384)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--l-margin", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--build-repeats", type=int, default=1)
    parser.add_argument("--cpu-repeats", type=int, default=1)
    parser.add_argument("--gpu-repeats", type=int, default=7)
    parser.add_argument("--gpu-warmups", type=int, default=2)
    parser.add_argument("--cufinufft-repeats", type=int, default=5)
    parser.add_argument("--cufinufft-eps", type=float, default=1e-6)
    parser.add_argument("--cufinufft-plan-mode", choices=["plan", "simple"], default="plan")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_same_direction_torch_gpu",
    )
    args = parser.parse_args()
    args.cases = [parse_case(value) for value in args.cases]
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.rank <= 0:
        raise ValueError("rank must be positive")
    if args.cufinufft_eps <= 0:
        raise ValueError("cufinufft-eps must be positive")
    return args


def main() -> None:
    args = parse_args()
    case_payloads = [
        build_case(
            args,
            label=label,
            n_mag=n_mag,
            k=k,
            cap_radial=cap_radial,
            cap_phi=cap_phi,
        )
        for label, n_mag, k, cap_radial, cap_phi in args.cases
    ]
    rows = [row for payload in case_payloads for row in payload["rows"]]
    prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    payload = {
        "config": {
            "cases": [
                {
                    "label": label,
                    "n_mag": n_mag,
                    "k": k,
                    "cap_radial": cap_radial,
                    "cap_phi": cap_phi,
                }
                for label, n_mag, k, cap_radial, cap_phi in args.cases
            ],
            "rank": args.rank,
            "dtype": args.dtype,
            "cufinufft_eps": args.cufinufft_eps,
            "cufinufft_plan_mode": args.cufinufft_plan_mode,
        },
        "environment": [
            {
                "case": payload["case"]["label"],
                "device_name": payload["device_name"],
                "torch_version": payload["torch_version"],
                "torch_cuda_version": payload["torch_cuda_version"],
                "cupy_version": payload["cupy_version"],
                "cufinufft_version": payload["cufinufft_version"],
                "torch_peak_allocated_mib": payload["torch_peak_allocated_mib"],
            }
            for payload in case_payloads
        ],
        "rows": rows,
    }
    write_json(prefix.with_suffix(".json"), payload)
    write_csv(prefix.with_suffix(".csv"), rows)
    write_summary(prefix.with_suffix(".md"), payload)
    write_plot(prefix.with_suffix(".png"), rows)
    print(json.dumps({"rows": len(rows), "json": str(prefix.with_suffix(".json"))}, indent=2))


if __name__ == "__main__":
    main()
