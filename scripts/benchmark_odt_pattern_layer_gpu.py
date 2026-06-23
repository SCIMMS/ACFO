from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
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
    cufinufft_adjoint_block,
    cufinufft_forward_block,
    import_cufinufft_modules,
    make_cufinufft_composite,
    synchronize_cupy,
)
from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    rel_l2,
    torch_dtypes,
    to_numpy,
)


def parse_case(text: str) -> tuple[str, int, int, int, int, int]:
    if "=" in text:
        label, payload = text.split("=", 1)
    else:
        label, payload = "", text
    parts = payload.lower().replace(" ", "").split(":")
    if len(parts) != 4 or "x" not in parts[3]:
        raise ValueError("case must look like label=n_illum:n_patterns:active:cap_radialxcap_phi")
    n_illum = int(parts[0])
    n_patterns = int(parts[1])
    active = int(parts[2])
    radial_text, phi_text = parts[3].split("x", 1)
    cap_radial = int(radial_text)
    cap_phi = int(phi_text)
    if not label:
        label = f"illum{n_illum}_pat{n_patterns}_{cap_radial}x{cap_phi}"
    return label, n_illum, n_patterns, active, cap_radial, cap_phi


def speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


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
        raise RuntimeError("timed CUDA function did not run")
    return value, float(median(times)), times


def torch_to_cupy(cp: Any, value: Any) -> Any:
    return cp.from_dlpack(value.detach().contiguous())


def cupy_to_torch(torch: Any, value: Any) -> Any:
    return torch.from_dlpack(value)


def cufinufft_forward_torch(torch: Any, cu_op: Any, coeff: Any, *, eps: float) -> Any:
    coeff_gpu = torch_to_cupy(cu_op.cp, coeff.reshape(-1))
    out = cufinufft_forward_block(cu_op, cu_op.ring, coeff_gpu, eps=eps)
    synchronize_cupy(cu_op.cp)
    return cupy_to_torch(torch, out)


def cufinufft_adjoint_torch(
    torch: Any,
    cu_op: Any,
    residual: Any,
    *,
    eps: float,
    coeff_shape: tuple[int, ...],
) -> Any:
    residual_gpu = torch_to_cupy(cu_op.cp, residual.reshape(-1))
    out = cufinufft_adjoint_block(cu_op, cu_op.ring, residual_gpu, eps=eps)
    synchronize_cupy(cu_op.cp)
    return cupy_to_torch(torch, out).reshape(coeff_shape)


def make_pattern_matrix_np(
    *,
    n_patterns: int,
    n_illum: int,
    active: int,
    seed: int,
    kind: str,
) -> np.ndarray:
    if active <= 0 or active > n_illum:
        raise ValueError("active pattern width must be in [1, n_illum]")
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_patterns, n_illum), dtype=np.complex128)
    for row in range(n_patterns):
        if kind == "rolling":
            start = (row * active) % n_illum
            indices = (start + np.arange(active)) % n_illum
            values = np.ones(active, dtype=np.complex128)
        else:
            indices = rng.choice(n_illum, size=active, replace=False)
            if kind == "binary":
                values = np.ones(active, dtype=np.complex128)
            elif kind == "phase":
                phases = rng.uniform(0.0, 2.0 * math.pi, size=active)
                values = np.exp(1j * phases)
            else:
                raise ValueError(f"unknown pattern kind {kind!r}")
        matrix[row, indices] = values / math.sqrt(float(active))
    row_norms = np.linalg.norm(matrix, axis=1)
    matrix = matrix / np.maximum(row_norms[:, None], 1e-300)
    return np.ascontiguousarray(matrix)


@dataclass
class TorchPatternedRingPlan:
    torch: Any
    device: Any
    ring: Any
    pattern: Any
    pattern_conj_t: Any
    incoherent_weights: Any
    n_patterns: int
    n_illum: int
    cap_radial: int
    cap_phi: int

    @classmethod
    def from_ring(
        cls,
        *,
        torch: Any,
        device: Any,
        ring: Any,
        pattern_np: np.ndarray,
        dtype: str,
    ) -> "TorchPatternedRingPlan":
        complex_dtype, real_dtype, np_complex, np_real = torch_dtypes(torch, dtype)
        pattern = torch.as_tensor(
            np.ascontiguousarray(pattern_np.astype(np_complex, copy=False)),
            dtype=complex_dtype,
            device=device,
        )
        weights_np = np.abs(pattern_np) ** 2
        weights_np = weights_np / np.maximum(weights_np.sum(axis=1, keepdims=True), 1e-300)
        weights = torch.as_tensor(
            np.ascontiguousarray(weights_np.astype(np_real, copy=False)),
            dtype=real_dtype,
            device=device,
        )
        return cls(
            torch=torch,
            device=device,
            ring=ring,
            pattern=pattern,
            pattern_conj_t=pattern.conj().transpose(0, 1).contiguous(),
            incoherent_weights=weights,
            n_patterns=int(pattern_np.shape[0]),
            n_illum=int(pattern_np.shape[1]),
            cap_radial=int(ring.cap_radial),
            cap_phi=int(ring.cap_phi),
        )

    @property
    def base_pixels(self) -> int:
        return self.cap_radial * self.cap_phi

    @property
    def q_count(self) -> int:
        return self.n_patterns * self.base_pixels

    @property
    def basis_mib(self) -> float:
        tensors = (self.pattern, self.pattern_conj_t, self.incoherent_weights)
        return self.ring.basis_mib + float(
            sum(int(t.nelement() * t.element_size()) for t in tensors) / (1024.0 * 1024.0)
        )

    def stack_forward(self, coeff: Any) -> Any:
        return self.ring.forward(coeff).reshape(self.n_illum, self.base_pixels)

    def mix_stack(self, stack: Any) -> Any:
        return self.pattern @ stack

    def backmix(self, residual: Any) -> Any:
        return self.pattern_conj_t @ residual.reshape(self.n_patterns, self.base_pixels)

    def forward(self, coeff: Any) -> Any:
        return self.mix_stack(self.stack_forward(coeff)).reshape(-1)

    def adjoint(self, residual: Any) -> Any:
        ring_residual = self.backmix(residual).reshape(-1)
        return self.ring.adjoint(ring_residual)

    def coherent_intensity_forward(self, coeff: Any) -> Any:
        field = self.mix_stack(self.stack_forward(coeff))
        return self.torch.real(field.conj() * field)

    def incoherent_intensity_forward(self, coeff: Any) -> Any:
        stack = self.stack_forward(coeff)
        intensity = self.torch.real(stack.conj() * stack)
        return self.incoherent_weights @ intensity


def relative_torch(torch: Any, candidate: Any, reference: Any) -> float:
    denom = torch.clamp(torch.linalg.vector_norm(reference.reshape(-1)), min=1e-30)
    return float((torch.linalg.vector_norm((candidate - reference).reshape(-1)) / denom).detach().cpu().item())


def adjoint_dot_error(torch: Any, coeff: Any, forward: Any, residual: Any, adjoint: Any) -> float:
    lhs = torch.sum(torch.conj(forward.reshape(-1)) * residual.reshape(-1))
    rhs = torch.sum(torch.conj(coeff.reshape(-1)) * adjoint.reshape(-1))
    denom = torch.clamp(torch.abs(lhs), min=1e-30)
    return float((torch.abs(lhs - rhs) / denom).detach().cpu().item())


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
        "# ODT FPDT/FS-ODT pattern-layer GPU proxy",
        "",
        "This benchmark adds a fixed illumination-pattern layer on top of the prepared ring ODT operator. It is a backend bridge for FPDT/FS-ODT-like repeated pattern stacks, not a full intensity-only ptychographic solver.",
        "",
        "## Results",
        "",
        "| case | stack q | pattern samples | active | complex fwd speedup | complex adj speedup | pair speedup | coherent intensity fwd speedup | incoherent intensity fwd speedup | fwd rel-L2 vs cuFINUFFT | adj rel-L2 vs cuFINUFFT | dot err |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {stack_q} | {pat_q} | {active} | `{sf}` | `{sa}` | `{sp}` | `{sci}` | `{sii}` | `{ef}` | `{ea}` | `{dot}` |".format(
                case=row["case"],
                stack_q=row["stack_q_samples"],
                pat_q=row["pattern_q_samples"],
                active=row["active_per_pattern"],
                sf=fmt(row["complex_forward_speedup"], 4),
                sa=fmt(row["complex_adjoint_speedup"], 4),
                sp=fmt(row["complex_pair_speedup"], 4),
                sci=fmt(row["coherent_intensity_forward_speedup"], 4),
                sii=fmt(row["incoherent_intensity_forward_speedup"], 4),
                ef=fmt(row["complex_forward_rel_l2_vs_cufinufft"], 4),
                ea=fmt(row["complex_adjoint_rel_l2_vs_cufinufft"], 4),
                dot=fmt(row["ours_complex_adjoint_dot_error"], 4),
            )
        )
    if rows:
        lines.extend(["", "## Setup", ""])
        for row in rows:
            lines.append(
                "- `{case}` setup: common geometry `{common}` s, ours backend `{ours}` s, cuFINUFFT backend `{cu}` s.".format(
                    case=row["case"],
                    common=fmt(row["common_geometry_build_s"], 4),
                    ours=fmt(row["ours_backend_setup_s"], 4),
                    cu=fmt(row["cufinufft_backend_setup_s"], 4),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The complex pattern layer is a linear operator: pattern mixing is applied after the prepared ring forward operator, and the conjugate pattern matrix is applied before the prepared adjoint.",
            "- The coherent and incoherent intensity rows are forward-only diagnostics. They show the cost of producing nonlinear intensity-style data, but they are not full FPDT/FS-ODT inverse solvers.",
            "- A speedup greater than 1 means the prepared structured GPU backend plus pattern layer is faster than cuFINUFFT GPU Plan plus the same pattern layer.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    labels = [row["case"] for row in rows]
    metrics = [
        ("complex_forward_speedup", "complex forward"),
        ("complex_adjoint_speedup", "complex adjoint"),
        ("coherent_intensity_forward_speedup", "coherent intensity"),
        ("incoherent_intensity_forward_speedup", "incoherent intensity"),
    ]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for offset, (key, label) in enumerate(metrics):
        values = [row[key] for row in rows]
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=label)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("cuFINUFFT pattern time / ours pattern time")
    ax.set_title("ODT fixed-pattern layer speedup over cuFINUFFT")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def benchmark_case(
    args: argparse.Namespace,
    *,
    label: str,
    n_illum: int,
    n_patterns: int,
    active: int,
    cap_radial: int,
    cap_phi: int,
) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("pattern-layer benchmark requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    case_args = argparse.Namespace(**vars(args))
    case_args.ring_illum = int(n_illum)
    case_args.cap_radial = int(cap_radial)
    case_args.cap_phi = int(cap_phi)
    case_args.skip_axis_illumination = True
    common_start = time.perf_counter()
    ctx = build_composite_context(case_args)
    common_geometry_build_s = time.perf_counter() - common_start
    start = time.perf_counter()
    composite = TorchCompositeOdtPlan.from_context(ctx, torch=torch, device=device, dtype=args.dtype)
    ring = composite.ring
    pattern_np = make_pattern_matrix_np(
        n_patterns=n_patterns,
        n_illum=n_illum,
        active=active,
        seed=args.seed + n_illum * 13 + n_patterns * 17 + active,
        kind=args.pattern_kind,
    )
    ours = TorchPatternedRingPlan.from_ring(
        torch=torch,
        device=device,
        ring=ring,
        pattern_np=pattern_np,
        dtype=args.dtype,
    )
    ours_backend_setup_s = time.perf_counter() - start

    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    true_coeff_np = np.ascontiguousarray((ctx.ring.obj.coeff * args.object_scale).astype(np_complex, copy=False))
    true_coeff_t = torch.as_tensor(true_coeff_np, dtype=ring.complex_dtype, device=device)
    rng = np.random.default_rng(args.seed + 5407)
    residual_np = (
        rng.standard_normal(ours.q_count) + 1j * rng.standard_normal(ours.q_count)
    ).astype(np_complex, copy=False)
    residual_t = torch.as_tensor(np.ascontiguousarray(residual_np), dtype=ring.complex_dtype, device=device)

    cp, _ = import_cufinufft_modules()
    start = time.perf_counter()
    cu_op = make_cufinufft_composite(
        ctx,
        dtype=args.dtype,
        plan_mode=args.cufinufft_plan_mode,
        eps=args.cufinufft_eps,
    )
    cufinufft_setup_s = time.perf_counter() - start

    def cu_stack_forward() -> Any:
        return cufinufft_forward_torch(torch, cu_op, true_coeff_t, eps=args.cufinufft_eps).reshape(
            n_illum,
            ours.base_pixels,
        )

    def cu_complex_forward() -> Any:
        return ours.mix_stack(cu_stack_forward()).reshape(-1)

    def cu_complex_adjoint() -> Any:
        ring_residual = ours.backmix(residual_t).reshape(-1)
        return cufinufft_adjoint_torch(
            torch,
            cu_op,
            ring_residual,
            eps=args.cufinufft_eps,
            coeff_shape=true_coeff_np.shape,
        )

    def cu_coherent_intensity_forward() -> Any:
        field = ours.mix_stack(cu_stack_forward())
        return torch.real(field.conj() * field)

    def cu_incoherent_intensity_forward() -> Any:
        stack = cu_stack_forward()
        intensity = torch.real(stack.conj() * stack)
        return ours.incoherent_weights @ intensity

    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        ours_forward, ours_forward_s, ours_forward_times = timed_cuda(
            torch,
            device,
            lambda: ours.forward(true_coeff_t),
            repeats=args.gpu_repeats,
            warmups=args.gpu_warmups,
        )
        ours_adjoint, ours_adjoint_s, ours_adjoint_times = timed_cuda(
            torch,
            device,
            lambda: ours.adjoint(residual_t),
            repeats=args.gpu_repeats,
            warmups=args.gpu_warmups,
        )
        ours_coherent_i, ours_coherent_i_s, _ = timed_cuda(
            torch,
            device,
            lambda: ours.coherent_intensity_forward(true_coeff_t),
            repeats=args.gpu_repeats,
            warmups=args.gpu_warmups,
        )
        ours_incoherent_i, ours_incoherent_i_s, _ = timed_cuda(
            torch,
            device,
            lambda: ours.incoherent_intensity_forward(true_coeff_t),
            repeats=args.gpu_repeats,
            warmups=args.gpu_warmups,
        )
        stack_for_mix = ours.stack_forward(true_coeff_t)
        _, pattern_mix_only_s, _ = timed_cuda(
            torch,
            device,
            lambda: ours.mix_stack(stack_for_mix),
            repeats=args.gpu_repeats,
            warmups=args.gpu_warmups,
        )
        _, pattern_backmix_only_s, _ = timed_cuda(
            torch,
            device,
            lambda: ours.backmix(residual_t),
            repeats=args.gpu_repeats,
            warmups=args.gpu_warmups,
        )
        cu_forward, cu_forward_s, cu_forward_times = timed_cuda(
            torch,
            device,
            cu_complex_forward,
            repeats=args.cufinufft_repeats,
            warmups=args.gpu_warmups,
        )
        cu_adjoint, cu_adjoint_s, cu_adjoint_times = timed_cuda(
            torch,
            device,
            cu_complex_adjoint,
            repeats=args.cufinufft_repeats,
            warmups=args.gpu_warmups,
        )
        cu_coherent_i, cu_coherent_i_s, _ = timed_cuda(
            torch,
            device,
            cu_coherent_intensity_forward,
            repeats=args.cufinufft_repeats,
            warmups=args.gpu_warmups,
        )
        cu_incoherent_i, cu_incoherent_i_s, _ = timed_cuda(
            torch,
            device,
            cu_incoherent_intensity_forward,
            repeats=args.cufinufft_repeats,
            warmups=args.gpu_warmups,
        )

    row = {
        "case": label,
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cupy_version": getattr(cp, "__version__", None),
        "cufinufft_version": getattr(cu_op.cufinufft, "__version__", None),
        "dtype": args.dtype,
        "pattern_kind": args.pattern_kind,
        "n_illum": int(n_illum),
        "n_patterns": int(n_patterns),
        "active_per_pattern": int(active),
        "cap_radial": int(cap_radial),
        "cap_phi": int(cap_phi),
        "base_pixels": int(ours.base_pixels),
        "stack_q_samples": int(n_illum * ours.base_pixels),
        "pattern_q_samples": int(ours.q_count),
        "object_bins": int(true_coeff_t.numel()),
        "common_geometry_build_s": float(common_geometry_build_s),
        "ours_backend_setup_s": float(ours_backend_setup_s),
        "cufinufft_backend_setup_s": float(cufinufft_setup_s),
        "ours_setup_s": float(common_geometry_build_s + ours_backend_setup_s),
        "cufinufft_setup_s": float(common_geometry_build_s + cufinufft_setup_s),
        "ours_basis_mib": float(ours.basis_mib),
        "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        "cupy_pool_total_mib": float(cp.get_default_memory_pool().total_bytes() / (1024.0 * 1024.0)),
        "ours_complex_forward_s": float(ours_forward_s),
        "cufinufft_complex_forward_s": float(cu_forward_s),
        "complex_forward_speedup": speedup(cu_forward_s, ours_forward_s),
        "ours_complex_adjoint_s": float(ours_adjoint_s),
        "cufinufft_complex_adjoint_s": float(cu_adjoint_s),
        "complex_adjoint_speedup": speedup(cu_adjoint_s, ours_adjoint_s),
        "ours_complex_pair_s": float(ours_forward_s + ours_adjoint_s),
        "cufinufft_complex_pair_s": float(cu_forward_s + cu_adjoint_s),
        "complex_pair_speedup": speedup(cu_forward_s + cu_adjoint_s, ours_forward_s + ours_adjoint_s),
        "ours_coherent_intensity_forward_s": float(ours_coherent_i_s),
        "cufinufft_coherent_intensity_forward_s": float(cu_coherent_i_s),
        "coherent_intensity_forward_speedup": speedup(cu_coherent_i_s, ours_coherent_i_s),
        "ours_incoherent_intensity_forward_s": float(ours_incoherent_i_s),
        "cufinufft_incoherent_intensity_forward_s": float(cu_incoherent_i_s),
        "incoherent_intensity_forward_speedup": speedup(cu_incoherent_i_s, ours_incoherent_i_s),
        "pattern_mix_only_s": float(pattern_mix_only_s),
        "pattern_backmix_only_s": float(pattern_backmix_only_s),
        "complex_forward_rel_l2_vs_cufinufft": relative_torch(torch, ours_forward, cu_forward),
        "complex_adjoint_rel_l2_vs_cufinufft": relative_torch(torch, ours_adjoint, cu_adjoint),
        "coherent_intensity_rel_l2_vs_cufinufft": relative_torch(torch, ours_coherent_i, cu_coherent_i),
        "incoherent_intensity_rel_l2_vs_cufinufft": relative_torch(torch, ours_incoherent_i, cu_incoherent_i),
        "ours_complex_adjoint_dot_error": adjoint_dot_error(
            torch,
            true_coeff_t,
            ours_forward,
            residual_t,
            ours_adjoint,
        ),
        "ours_forward_times_s": " ".join(f"{value:.9g}" for value in ours_forward_times),
        "ours_adjoint_times_s": " ".join(f"{value:.9g}" for value in ours_adjoint_times),
        "cufinufft_forward_times_s": " ".join(f"{value:.9g}" for value in cu_forward_times),
        "cufinufft_adjoint_times_s": " ".join(f"{value:.9g}" for value in cu_adjoint_times),
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed FPDT/FS-ODT-like illumination pattern layers on the ODT GPU backend."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["medium=64:16:8:32x128", "large=96:24:12:64x256"],
        help="Cases as label=n_illum:n_patterns:active:cap_radialxcap_phi.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--pattern-kind", choices=["phase", "binary", "rolling"], default="phase")
    parser.add_argument("--n-beta", type=int, default=256)
    parser.add_argument("--n-r", type=int, default=12)
    parser.add_argument("--n-z", type=int, default=11)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=0.8)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="random_beads")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--k", type=float, default=17.307319527958313)
    parser.add_argument("--detector-na", type=float, default=0.9240924092409241)
    parser.add_argument("--illumination-angle-deg", type=float, default=49.0)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=20)
    parser.add_argument("--l-margin", type=int, default=18)
    parser.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    parser.add_argument("--cpp-threads", type=int, default=16)
    parser.add_argument("--forward-execute-mode", choices=["prepared", "wrapper"], default="prepared")
    parser.add_argument("--forward-kernel-mode", choices=["compact", "partitioned"], default="partitioned")
    parser.add_argument(
        "--native-prepared-plan-mode",
        choices=["auto", "direct", "gathered", "gathered-zmajor"],
        default="auto",
    )
    parser.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--finufft-q-batch-size", type=int, default=1_048_576)
    parser.add_argument("--cufinufft-eps", type=float, default=1e-6)
    parser.add_argument("--cufinufft-plan-mode", choices=["plan", "simple"], default="plan")
    parser.add_argument("--gpu-repeats", type=int, default=5)
    parser.add_argument("--gpu-warmups", type=int, default=2)
    parser.add_argument("--cufinufft-repeats", type=int, default=3)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_pattern_layer_gpu",
    )
    args = parser.parse_args()
    args.cases = [parse_case(value) for value in args.cases]
    if args.gpu_repeats <= 0 or args.cufinufft_repeats <= 0:
        raise ValueError("repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = [
        benchmark_case(
            args,
            label=label,
            n_illum=n_illum,
            n_patterns=n_patterns,
            active=active,
            cap_radial=cap_radial,
            cap_phi=cap_phi,
        )
        for label, n_illum, n_patterns, active, cap_radial, cap_phi in args.cases
    ]
    prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    payload = {
        "config": {
            "cases": [
                {
                    "label": label,
                    "n_illum": n_illum,
                    "n_patterns": n_patterns,
                    "active": active,
                    "cap_radial": cap_radial,
                    "cap_phi": cap_phi,
                }
                for label, n_illum, n_patterns, active, cap_radial, cap_phi in args.cases
            ],
            "dtype": args.dtype,
            "pattern_kind": args.pattern_kind,
            "cufinufft_eps": args.cufinufft_eps,
            "cufinufft_plan_mode": args.cufinufft_plan_mode,
        },
        "rows": rows,
    }
    write_json(prefix.with_suffix(".json"), payload)
    write_csv(prefix.with_suffix(".csv"), rows)
    write_summary(prefix.with_suffix(".md"), payload)
    write_plot(prefix.with_suffix(".png"), rows)
    print(json.dumps({"rows": len(rows), "json": str(prefix.with_suffix(".json"))}, indent=2))


if __name__ == "__main__":
    main()
