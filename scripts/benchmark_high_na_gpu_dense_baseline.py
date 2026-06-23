from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_backpropagation import complex_dot, focal_grid, relative_complex_error  # noqa: E402
from benchmark_high_na_debye_wolf import gauss_theta_grid, relative_l2  # noqa: E402
from benchmark_high_na_torch_gpu import (  # noqa: E402
    TorchSeparableHarmonicDebyeWolfPlan,
    device_name,
    import_torch,
    make_pupil_batch,
    make_residual_batch,
    package_version,
    resolve_device,
    synchronize,
    timed_torch,
    to_numpy,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    richards_wolf_jones_matrix,
    vectorial_h_cutoff_for_workload,
    vectorial_pupil_jones,
    workloads,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")


def target_points_from_cylindrical(
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return rho * np.cos(psi), rho * np.sin(psi), z


def target_points_cartesian(
    *,
    nx: int,
    ny: int,
    nz: int,
    x_max: float,
    y_max: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.linspace(-x_max, x_max, nx, dtype=float)
    y_axis = np.linspace(-y_max, y_max, ny, dtype=float)
    if nz == 1:
        z_axis = np.array([0.0], dtype=float)
    else:
        z_axis = np.linspace(-z_max, z_max, nz, dtype=float)
    xx, yy, zz = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    return xx.ravel(), yy.ravel(), zz.ravel()


class TorchDenseDirectVectorialDebyeWolf:
    def __init__(
        self,
        *,
        torch: Any,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        phi: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        k: float,
        mixing: np.ndarray,
        device: Any,
        dtype: str,
        chunk_targets: int,
    ) -> None:
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.chunk_targets = int(chunk_targets)
        if self.chunk_targets <= 0:
            raise ValueError("chunk_targets must be positive")
        if dtype == "complex64":
            self.complex_dtype = torch.complex64
            self.real_dtype = torch.float32
            np_complex = np.complex64
            np_real = np.float32
        elif dtype == "complex128":
            self.complex_dtype = torch.complex128
            self.real_dtype = torch.float64
            np_complex = np.complex128
            np_real = np.float64
        else:
            raise ValueError("dtype must be complex64 or complex128")

        theta_2d = theta[:, None]
        phi_2d = phi[None, :]
        sx = np.sin(theta_2d) * np.cos(phi_2d)
        sy = np.sin(theta_2d) * np.sin(phi_2d)
        sz = np.cos(theta_2d) * np.ones_like(phi_2d)
        source_weight = np.broadcast_to(
            theta_weights[:, None] * (2.0 * np.pi / float(phi.size)),
            (theta.size, phi.size),
        )
        self.sx = torch.as_tensor(sx.ravel().astype(np_real), dtype=self.real_dtype, device=device)
        self.sy = torch.as_tensor(sy.ravel().astype(np_real), dtype=self.real_dtype, device=device)
        self.sz = torch.as_tensor(sz.ravel().astype(np_real), dtype=self.real_dtype, device=device)
        self.source_weight = torch.as_tensor(
            source_weight.ravel().astype(np_real),
            dtype=self.real_dtype,
            device=device,
        )
        self.x = torch.as_tensor(np.asarray(x, dtype=np_real), dtype=self.real_dtype, device=device)
        self.y = torch.as_tensor(np.asarray(y, dtype=np_real), dtype=self.real_dtype, device=device)
        self.z = torch.as_tensor(np.asarray(z, dtype=np_real), dtype=self.real_dtype, device=device)
        self.k = float(k)
        self.mixing = torch.as_tensor(
            np.ascontiguousarray(mixing.reshape(3, 2, -1).astype(np_complex, copy=False)),
            dtype=self.complex_dtype,
            device=device,
        )
        self.nsource = int(self.sx.numel())
        self.ntarget = int(self.x.numel())
        self.ntheta = int(theta.size)
        self.nphi = int(phi.size)

    @property
    def source_mib(self) -> float:
        tensors = (self.sx, self.sy, self.sz, self.source_weight, self.x, self.y, self.z, self.mixing)
        return float(
            sum(int(tensor.nelement() * tensor.element_size()) for tensor in tensors)
            / (1024.0 * 1024.0)
        )

    def _phase_chunk(self, start: int, stop: int) -> Any:
        phase_arg = self.k * (
            self.sx[:, None] * self.x[None, start:stop]
            + self.sy[:, None] * self.y[None, start:stop]
            + self.sz[:, None] * self.z[None, start:stop]
        )
        return self.torch.exp(1j * phase_arg).to(self.complex_dtype)

    def _effective_sources(self, pupil_batch: Any) -> Any:
        if not self.torch.is_tensor(pupil_batch):
            pupil_batch = self.torch.as_tensor(
                pupil_batch,
                dtype=self.complex_dtype,
                device=self.device,
            )
        else:
            pupil_batch = pupil_batch.to(device=self.device, dtype=self.complex_dtype)
        if pupil_batch.ndim == 3:
            pupil_batch = pupil_batch.unsqueeze(0)
        if pupil_batch.ndim != 4 or pupil_batch.shape[1] != 2:
            raise ValueError("pupil batch must have shape (batch, 2, ntheta, nphi)")
        pupil_flat = pupil_batch.reshape(pupil_batch.shape[0], 2, self.nsource)
        effective = self.torch.einsum("cjs,bjs->bcs", self.mixing, pupil_flat)
        return effective * self.source_weight[None, None, :]

    def forward(self, pupil_batch: Any) -> Any:
        weighted_effective = self._effective_sources(pupil_batch)
        out = self.torch.empty(
            (weighted_effective.shape[0], 3, self.ntarget),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for start in range(0, self.ntarget, self.chunk_targets):
            stop = min(self.ntarget, start + self.chunk_targets)
            phase = self._phase_chunk(start, stop)
            out[:, :, start:stop] = self.torch.einsum(
                "bcs,st->bct",
                weighted_effective,
                phase,
            )
        return out

    def adjoint(self, residual_batch: Any) -> Any:
        if not self.torch.is_tensor(residual_batch):
            residual_batch = self.torch.as_tensor(
                residual_batch,
                dtype=self.complex_dtype,
                device=self.device,
            )
        else:
            residual_batch = residual_batch.to(device=self.device, dtype=self.complex_dtype)
        if residual_batch.ndim == 2:
            residual_batch = residual_batch.unsqueeze(0)
        if residual_batch.ndim != 3 or residual_batch.shape[1] != 3:
            raise ValueError("residual batch must have shape (batch, 3, targets)")
        effective_grad = self.torch.zeros(
            (residual_batch.shape[0], 3, self.nsource),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for start in range(0, self.ntarget, self.chunk_targets):
            stop = min(self.ntarget, start + self.chunk_targets)
            phase = self._phase_chunk(start, stop)
            effective_grad += self.torch.einsum(
                "bct,st->bcs",
                residual_batch[:, :, start:stop],
                phase.conj(),
            )
        effective_grad = effective_grad * self.source_weight[None, None, :]
        pupil_grad = self.torch.einsum(
            "cjs,bcs->bjs",
            self.mixing.conj(),
            effective_grad,
        )
        return pupil_grad.reshape(residual_batch.shape[0], 2, self.ntheta, self.nphi)


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
        }
    ]


def run_case(args: argparse.Namespace) -> list[dict[str, Any]]:
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
    rho_axis, psi_axis, z_axis, rho, psi, z_cyl = focal_grid(
        nrho=workload.nrho,
        npsi=workload.npsi,
        nz=workload.nz,
        rho_max=workload.rho_max,
        z_max=workload.z_max,
    )
    if args.target_layout == "cylindrical":
        x, y, z = target_points_from_cylindrical(rho, psi, z_cyl)
        target_count = rho.size
    elif args.target_layout == "cartesian":
        x, y, z = target_points_cartesian(
            nx=args.cartesian_nx,
            ny=args.cartesian_ny,
            nz=args.cartesian_nz,
            x_max=workload.rho_max,
            y_max=workload.rho_max,
            z_max=workload.z_max,
        )
        target_count = x.size
    else:
        raise ValueError("target_layout must be cylindrical or cartesian")

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
    residual_batch = make_residual_batch(
        case=args.residual_case,
        rho_axis=rho_axis if args.target_layout == "cylindrical" else np.arange(target_count, dtype=float),
        psi_axis=psi_axis if args.target_layout == "cylindrical" else np.array([0.0]),
        z_axis=z_axis if args.target_layout == "cylindrical" else np.array([0.0]),
        batch_size=args.batch_size,
        seed=args.seed,
        order=args.residual_order,
    )
    if args.target_layout == "cartesian":
        rng = np.random.default_rng(args.seed)
        residual_batch = (
            rng.standard_normal((args.batch_size, 3, target_count))
            + 1j * rng.standard_normal((args.batch_size, 3, target_count))
        ).astype(np.complex128)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    dense = TorchDenseDirectVectorialDebyeWolf(
        torch=torch,
        theta=theta,
        theta_weights=theta_weights,
        phi=phi,
        x=x,
        y=y,
        z=z,
        k=workload.k,
        mixing=mixing,
        device=device,
        dtype=args.dtype,
        chunk_targets=args.chunk_targets,
    )
    pupil_t = torch.as_tensor(pupil_batch, dtype=dense.complex_dtype, device=device)
    residual_t = torch.as_tensor(residual_batch, dtype=dense.complex_dtype, device=device)
    dense_forward, dense_forward_s, dense_forward_times = timed_torch(
        torch,
        device,
        lambda: dense.forward(pupil_t),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    dense_adjoint, dense_adjoint_s, dense_adjoint_times = timed_torch(
        torch,
        device,
        lambda: dense.adjoint(residual_t),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    dense_forward_np = to_numpy(torch, device, dense_forward)
    dense_adjoint_np = to_numpy(torch, device, dense_adjoint)
    dense_dot_left = complex_dot(dense_forward_np, residual_batch)
    dense_dot_right = complex_dot(pupil_batch, dense_adjoint_np)

    row: dict[str, Any] = {
        "status": "ok",
        "variant": "torch_dense_direct",
        "workload": workload.name,
        "workload_set": args.workload_set,
        "workload_index": args.workload_index,
        "target_layout": args.target_layout,
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "ntheta": workload.ntheta,
        "nphi": workload.nphi,
        "targets_per_mask": int(target_count),
        "total_output_fields": int(args.batch_size * 3),
        "source_samples": int(workload.ntheta * workload.nphi),
        "chunk_targets": args.chunk_targets,
        "dense_source_mib": dense.source_mib,
        "dense_forward_hot_s": float(dense_forward_s),
        "dense_adjoint_hot_s": float(dense_adjoint_s),
        "dense_forward_plus_adjoint_hot_s": float(dense_forward_s + dense_adjoint_s),
        "dense_dot_abs_error": float(abs(dense_dot_left - dense_dot_right)),
        "dense_dot_relative_error": relative_complex_error(dense_dot_left, dense_dot_right),
        "dense_forward_times_s": " ".join(f"{value:.9g}" for value in dense_forward_times),
        "dense_adjoint_times_s": " ".join(f"{value:.9g}" for value in dense_adjoint_times),
        "gpu_peak_allocated_mib": None
        if device.type != "cuda"
        else float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
    }

    if args.target_layout == "cylindrical":
        h_cutoff = vectorial_h_cutoff_for_workload(workload, args.h_margin)
        sep_plan = TorchSeparableHarmonicDebyeWolfPlan.build(
            torch=torch,
            nphi=workload.nphi,
            theta=theta,
            theta_weights=theta_weights,
            rho_axis=rho_axis,
            psi_axis=psi_axis,
            z_axis=z_axis,
            k=workload.k,
            h_cutoff=h_cutoff,
            device=device,
            dtype=args.dtype,
            basis_mode=args.basis_mode,
            contract_mode=args.contract_mode,
        )
        sep_pupil_t = sep_plan.as_tensor(pupil_batch)
        sep_residual_t = sep_plan.as_tensor(residual_batch)
        mixing_t = sep_plan.as_tensor(mixing)
        sep_forward, sep_forward_s, sep_forward_times = timed_torch(
            torch,
            device,
            lambda: sep_plan.evaluate_vectorial_batch(sep_pupil_t, mixing_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        sep_adjoint, sep_adjoint_s, sep_adjoint_times = timed_torch(
            torch,
            device,
            lambda: sep_plan.adjoint_vectorial_batch(sep_residual_t, mixing_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        sep_forward_np = to_numpy(torch, device, sep_forward)
        sep_adjoint_np = to_numpy(torch, device, sep_adjoint)
        sep_dot_left = complex_dot(sep_forward_np, residual_batch)
        sep_dot_right = complex_dot(pupil_batch, sep_adjoint_np)
        row.update(
            {
                "h_cutoff": int(h_cutoff),
                "separable_used_modes": int(sep_plan.used_modes),
                "separable_basis_mode": sep_plan.basis_mode,
                "separable_contract_mode": sep_plan.contract_mode,
                "separable_basis_mib": float(sep_plan.basis_mib),
                "separable_forward_hot_s": float(sep_forward_s),
                "separable_adjoint_hot_s": float(sep_adjoint_s),
                "separable_forward_plus_adjoint_hot_s": float(sep_forward_s + sep_adjoint_s),
                "speedup_dense_vs_separable_forward": float(dense_forward_s / sep_forward_s),
                "speedup_dense_vs_separable_adjoint": float(dense_adjoint_s / sep_adjoint_s),
                "speedup_dense_vs_separable_pair": float(
                    (dense_forward_s + dense_adjoint_s) / (sep_forward_s + sep_adjoint_s)
                ),
                "forward_l2_dense_vs_separable": relative_l2(sep_forward_np, dense_forward_np),
                "adjoint_l2_dense_vs_separable": relative_l2(sep_adjoint_np, dense_adjoint_np),
                "separable_dot_abs_error": float(abs(sep_dot_left - sep_dot_right)),
                "separable_dot_relative_error": relative_complex_error(sep_dot_left, sep_dot_right),
                "separable_forward_times_s": " ".join(f"{value:.9g}" for value in sep_forward_times),
                "separable_adjoint_times_s": " ".join(f"{value:.9g}" for value in sep_adjoint_times),
            }
        )
    return [row]


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
        "# High-NA GPU dense baseline benchmark",
        "",
        "This benchmark compares the Torch separable vectorial High-NA path against a dense direct Torch Richards-Wolf quadrature on the same CUDA device.",
        "For `target_layout=cylindrical`, both methods evaluate the same structured target points, so accuracy and speed are directly comparable.",
        "For `target_layout=cartesian`, only the dense direct baseline is reported.",
        "",
        "## Config",
        "",
        f"- workload_set: `{config['workload_set']}`",
        f"- workload_index: `{config['workload_index']}`",
        f"- target_layout: `{config['target_layout']}`",
        f"- device: `{config['device']}`",
        f"- dtype: `{config['dtype']}`",
        f"- basis_mode: `{config.get('basis_mode', 'separate')}`",
        f"- contract_mode: `{config.get('contract_mode', 'einsum')}`",
        f"- batch_size: `{config['batch_size']}`",
        f"- repeats: `{config['repeats']}`",
        "",
        "## Results",
        "",
        "| workload | layout | dtype | batch | targets | dense fwd s | dense adj s | sep fwd s | sep adj s | fwd speedup | adj speedup | pair speedup | fwd L2 | adj L2 | dense dot rel |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {layout} | {dtype} | {batch} | {targets} | {dense_fwd} | {dense_adj} | {sep_fwd} | {sep_adj} | {fwd_speed} | {adj_speed} | {pair_speed} | {fwd_l2} | {adj_l2} | {dot_rel} |".format(
                workload=row.get("workload", "n/a"),
                layout=row.get("target_layout", "n/a"),
                dtype=row.get("dtype", "n/a"),
                batch=row.get("batch_size", "n/a"),
                targets=row.get("targets_per_mask", "n/a"),
                dense_fwd=fmt(row.get("dense_forward_hot_s")),
                dense_adj=fmt(row.get("dense_adjoint_hot_s")),
                sep_fwd=fmt(row.get("separable_forward_hot_s")),
                sep_adj=fmt(row.get("separable_adjoint_hot_s")),
                fwd_speed=fmt(row.get("speedup_dense_vs_separable_forward")),
                adj_speed=fmt(row.get("speedup_dense_vs_separable_adjoint")),
                pair_speed=fmt(row.get("speedup_dense_vs_separable_pair")),
                fwd_l2=fmt(row.get("forward_l2_dense_vs_separable")),
                adj_l2=fmt(row.get("adjoint_l2_dense_vs_separable")),
                dot_rel=fmt(row.get("dense_dot_relative_error")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The dense direct GPU path is a correctness and same-device baseline, not an optimized FFT-Debye implementation.",
            "- The cylindrical row is the direct test of the structured-grid advantage.",
            "- The Cartesian row records the cost of dense arbitrary-target quadrature and sets up the external PSF-generator comparison.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Torch separable High-NA propagation with dense direct GPU quadrature."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_gpu_dense_baseline")
    parser.add_argument("--workload-set", choices=["quick", "representative", "large"], default="representative")
    parser.add_argument("--workload-index", type=int, default=1)
    parser.add_argument("--target-layout", choices=["cylindrical", "cartesian"], default="cylindrical")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-phase-strength", type=float, default=0.25)
    parser.add_argument("--residual-case", choices=["random", "low_order", "annular_roi"], default="low_order")
    parser.add_argument("--residual-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--h-margin", type=int, default=6)
    parser.add_argument("--basis-mode", choices=["separate", "fused"], default="separate")
    parser.add_argument("--contract-mode", choices=["einsum", "matmul"], default="einsum")
    parser.add_argument("--apodization", choices=["sqrt-cos", "none"], default="sqrt-cos")
    parser.add_argument("--chunk-targets", type=int, default=1024)
    parser.add_argument("--cartesian-nx", type=int, default=32)
    parser.add_argument("--cartesian-ny", type=int, default=32)
    parser.add_argument("--cartesian-nz", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = vars(args).copy()
    rows = run_case(args)
    output_prefix = ROOT / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_json(output_prefix.with_suffix(".json"), config, rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), config, rows)
    print(json.dumps({"config": config, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
